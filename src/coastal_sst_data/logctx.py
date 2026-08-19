#!/usr/bin/env python3
"""
coastal_sst_data -- which task a log line came from.

Serially, a log line's context is the line above it: `=== AOI: hobart ===` scrolls past, and
everything after it is about Hobart until the next banner. That is the entire reason the
stages never had to name themselves in each line.

Run the same stages on a thread pool and that structure is gone. Eight workers interleave
their output into one stderr stream, the banners no longer bracket anything, and

    12:04:31 INFO   [3/412] wrote a1_20230715.nc
    12:04:31 INFO   [7/98] wrote a1_20230715.nc

says nothing about which AoI or which product either line belongs to. Worse than ugly: a
failure lands in the middle of another product's progress and reads as that product's.

So a worker STAMPS its task onto every record it emits, and the formatter prints it:

    with logctx.label("hobart/landsat"):
        module.acquire(...)

    12:04:31 INFO  [hobart/landsat]   [3/412] wrote a1_20230715.nc

The label is THREAD-LOCAL, which is what makes this work without touching a single logging
call in the stages -- the stamp follows whichever thread is running, and every `log.info` in
every module picks it up for free. A serial run sets no label and prints exactly what it
always did, so `--jobs 1` output is unchanged character for character.

`configure()` also replaces the four hand-copied `logging.basicConfig(...)` calls (cli,
pipeline, entry, auth), which had already drifted into being four places to edit a format
string that must match the filter's field.
"""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import os
import signal
import threading

# Thread-local rather than a contextvar: the unit of work here is a pool thread, and a
# ThreadPoolExecutor does not propagate a context into its workers, so a contextvar would
# arrive empty exactly where it is needed.
_local = threading.local()

# `%(task)s` is filled by TaskLabel for EVERY record, so a logger anywhere in the tree (or in
# a dependency) cannot raise a formatting error by not knowing about it.
FORMAT = "%(asctime)s %(levelname)s %(task)s%(message)s"
DATEFMT = "%H:%M:%S"


class TaskLabel(logging.Filter):
    """Stamps the calling thread's current label onto each record as `task`."""

    def filter(self, record: logging.LogRecord) -> bool:
        text = getattr(_local, "label", "")
        record.task = f"[{text}] " if text else ""
        return True


@contextlib.contextmanager
def label(text: str):
    """Label every log record this thread emits inside the block.

    Restores the previous value rather than clearing it, because pool threads are REUSED: a
    label left behind would silently attribute the next task's output to the last one.
    """
    previous = getattr(_local, "label", "")
    _local.label = text
    try:
        yield
    finally:
        _local.label = previous


def current() -> str:
    """This thread's label, or "" -- for a caller that wants to include it in a message."""
    return getattr(_local, "label", "")


def _install_stack_dumper() -> bool:
    """`kill -USR1 <pid>` -> every thread's Python stack on stderr, run CONTINUES.

    An acquisition run is hours of blocking reads on a thread pool, and when one wedges the
    log simply stops -- no traceback, no exit, nothing to distinguish a deadlock from a slow
    granule. Recovering the stack after the fact needs `py-spy`, which needs ptrace, which on
    a shared server needs root the user running the job does not have. A signal handler the
    process installs on ITSELF needs none of that.

    SIGUSR1 rather than SIGINT or SIGABRT because both of those end the run: SIGINT unwinds
    only the main thread (and `scheduler.run_graph`'s pool then joins the very worker that is
    stuck), and SIGABRT kills. Diagnosing a hang should not require giving up the hours of
    work already done.

    False if the platform has no SIGUSR1 (Windows) or we are not on the main thread, where
    `faulthandler.register` refuses -- `configure` is reachable from a library caller and
    from tests, and neither should fail over a debugging aid.
    """
    if not hasattr(signal, "SIGUSR1"):
        return False
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (ValueError, OSError, RuntimeError):
        return False
    return True


def configure(*, verbose: bool = False) -> None:
    """Set up stderr logging for a run. Idempotent, and safe to call after basicConfig.

    `basicConfig` is a no-op once the root logger has a handler, so this also fixes up an
    already-configured root -- which is what lets a library caller (or a test) that set up
    logging first still get task labels.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format=FORMAT, datefmt=DATEFMT)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in root.handlers:
        if not any(isinstance(f, TaskLabel) for f in handler.filters):
            handler.addFilter(TaskLabel())
        # A handler installed before us carries a format string with no `%(task)s` in it.
        fmt = getattr(handler.formatter, "_fmt", None)
        if fmt is None or "%(task)s" not in fmt:
            handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))

    # Said out loud, at the TOP of the run's output, because that is where it is looked for:
    # the person who needs it is reading a scrollback the morning after, deciding whether a
    # silent process is stuck. An instruction they have to go and find is one they will not.
    if _install_stack_dumper():
        logging.getLogger(__name__).info(
            "stack dump on demand: kill -USR1 %d (dumps all threads, run continues)",
            os.getpid())
