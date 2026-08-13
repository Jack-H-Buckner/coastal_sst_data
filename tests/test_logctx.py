"""Which task a log line came from.

Serially, a line's context is the banner above it. On a thread pool that structure is gone:
eight workers interleave into one stderr stream and `[3/412] wrote a1_20230715.nc` no longer
says which AoI or product it belongs to -- and a failure landing mid-stream reads as another
product's.
"""

import logging
import threading

import pytest

from coastal_sst_data import logctx


@pytest.fixture
def records(monkeypatch):
    """Capture formatted output through a real handler, filter and formatter."""
    logged = []

    class Capture(logging.Handler):
        def emit(self, record):
            logged.append(self.format(record))

    handler = Capture()
    handler.addFilter(logctx.TaskLabel())
    handler.setFormatter(logging.Formatter("%(task)s%(message)s"))

    log = logging.getLogger("coastal_sst_data.test_logctx")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    yield log, logged
    log.removeHandler(handler)


def test_an_unlabelled_line_is_unchanged(records):
    """`--jobs 1` must print exactly what it always did, character for character."""
    log, logged = records
    log.info("[3/412] wrote a1_20230715.nc")
    assert logged == ["[3/412] wrote a1_20230715.nc"]


def test_a_labelled_line_names_its_task(records):
    log, logged = records
    with logctx.label("hobart/landsat"):
        log.info("[3/412] wrote a1_20230715.nc")
    assert logged == ["[hobart/landsat] [3/412] wrote a1_20230715.nc"]


def test_the_label_is_restored_not_cleared(records):
    """Pool threads are REUSED, so a label left behind would attribute the next task's
    output to the last one."""
    log, logged = records
    with logctx.label("outer"):
        with logctx.label("inner"):
            log.info("a")
        log.info("b")
    log.info("c")
    assert logged == ["[inner] a", "[outer] b", "c"]


def test_each_thread_carries_its_own_label(records):
    """The whole point: the stamp follows the thread, so every `log.info` in every stage
    picks it up without a single call site changing."""
    log, logged = records
    ready = threading.Barrier(3)

    def worker(name):
        with logctx.label(name):
            ready.wait(timeout=5)        # all three inside their labels at once
            log.info("working")

    threads = [threading.Thread(target=worker, args=(n,))
               for n in ("aoi_a/mur", "aoi_b/modis", "aoi_c/tides")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert sorted(logged) == ["[aoi_a/mur] working",
                              "[aoi_b/modis] working",
                              "[aoi_c/tides] working"]


def test_configure_labels_records_from_any_module_logger():
    """A logger in a dependency knows nothing about `task`; the filter must supply it for
    EVERY record or the formatter raises instead of logging."""
    logctx.configure()
    root = logging.getLogger()
    assert any(isinstance(f, logctx.TaskLabel)
               for h in root.handlers for f in h.filters)
    for handler in root.handlers:
        record = logging.LogRecord("some.third.party", logging.INFO, __file__, 1,
                                   "hello", None, None)
        for f in handler.filters:
            f.filter(record)
        handler.format(record)          # would raise if `task` were missing


def test_configure_is_idempotent():
    """Called from four entry points, and a `run` subcommand goes through more than one."""
    logctx.configure()
    logctx.configure()
    root = logging.getLogger()
    for handler in root.handlers:
        assert sum(isinstance(f, logctx.TaskLabel) for f in handler.filters) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
