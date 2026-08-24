"""The dependency-graph runner.

Three properties have to hold before any of this is safe to point at real downloads: the
declared order is honoured, a per-service cap is never exceeded, and a failure does not
silently let its dependents run against output that was never written.
"""

import threading
import time

import pytest

from coastal_sst_data import scheduler
from coastal_sst_data.scheduler import Task


class _Peak:
    """Counts concurrent entries and remembers the high-water mark."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        return self

    def __exit__(self, *exc):
        with self.lock:
            self.now -= 1


def _recorder():
    order, lock = [], threading.Lock()

    def note(name):
        def run():
            with lock:
                order.append(name)
            return name
        return run
    return order, note


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #
def test_dependencies_run_first():
    order, note = _recorder()
    tasks = [
        Task(key=("c",), run=note("c"), deps=(("b",),)),
        Task(key=("b",), run=note("b"), deps=(("a",),)),
        Task(key=("a",), run=note("a")),
    ]
    out = scheduler.run_graph(tasks, jobs=4)
    assert order == ["a", "b", "c"]
    assert all(o.ok for o in out.values())


def test_independent_tasks_actually_overlap():
    """The whole point. If these serialised, the graph would be running but useless."""
    peak = _Peak()

    def slow():
        with peak:
            time.sleep(0.05)

    tasks = [Task(key=(f"t{i}",), run=slow) for i in range(4)]
    scheduler.run_graph(tasks, jobs=4)
    assert peak.peak > 1


def test_jobs_1_is_a_stable_topological_order():
    """The escape hatch has to be predictable, not merely correct."""
    order, note = _recorder()
    tasks = [
        Task(key=("x",), run=note("x")),
        Task(key=("y",), run=note("y"), deps=(("x",),)),
        Task(key=("z",), run=note("z")),
    ]
    scheduler.run_graph(tasks, jobs=1)
    assert order == ["x", "y", "z"]


def test_jobs_never_exceeded():
    peak = _Peak()

    def slow():
        with peak:
            time.sleep(0.02)

    scheduler.run_graph([Task(key=(i,), run=slow) for i in range(12)], jobs=3)
    assert peak.peak <= 3


# --------------------------------------------------------------------------- #
# Gates -- the per-service caps
# --------------------------------------------------------------------------- #
def test_a_gate_caps_its_service():
    """CMEMS hands out a dataset handle that is not safe to share, and the small NOAA
    endpoints sit behind a module-global requests.Session. One global `jobs` number cannot
    express either."""
    peak = _Peak()

    def slow():
        with peak:
            time.sleep(0.03)

    tasks = [Task(key=(f"cmems{i}",), run=slow, gates=("copernicus",)) for i in range(8)]
    scheduler.run_graph(tasks, jobs=8, gates={"copernicus": 1})
    assert peak.peak == 1


def test_gates_are_independent_of_each_other():
    slow_peaks = {"a": _Peak(), "b": _Peak()}

    def slow(which):
        def run():
            with slow_peaks[which]:
                time.sleep(0.03)
        return run

    tasks = ([Task(key=(f"a{i}",), run=slow("a"), gates=("ga",)) for i in range(4)]
             + [Task(key=(f"b{i}",), run=slow("b"), gates=("gb",)) for i in range(4)])
    scheduler.run_graph(tasks, jobs=8, gates={"ga": 1, "gb": 4})
    assert slow_peaks["a"].peak == 1
    assert slow_peaks["b"].peak > 1


def test_a_saturated_gate_does_not_starve_the_pool():
    """Gates are taken BEFORE submission for exactly this reason. If a worker blocked while
    holding a slot, eight queued CMEMS days behind a cap of one would occupy the whole pool
    and every other product would wait behind them."""
    started = []
    lock = threading.Lock()

    def capped():
        time.sleep(0.05)

    def free():
        with lock:
            started.append(time.monotonic())

    tasks = ([Task(key=(f"capped{i}",), run=capped, gates=("slow",)) for i in range(6)]
             + [Task(key=("free",), run=free)])
    t0 = time.monotonic()
    scheduler.run_graph(tasks, jobs=4, gates={"slow": 1})

    # The unrelated task must not have waited for the six serialised ones (~0.3s).
    assert started and (started[0] - t0) < 0.2


def test_an_undeclared_gate_is_unlimited():
    peak = _Peak()

    def slow():
        with peak:
            time.sleep(0.03)

    tasks = [Task(key=(i,), run=slow, gates=("nobody_declared_me",)) for i in range(4)]
    scheduler.run_graph(tasks, jobs=4)
    assert peak.peak > 1


def test_a_gate_is_released_even_when_the_task_fails():
    """Otherwise one failure permanently removes capacity from that service, and a long run
    grinds to a halt for a reason nothing reports."""
    def boom():
        raise RuntimeError("nope")

    tasks = ([Task(key=(f"bad{i}",), run=boom, gates=("g",)) for i in range(3)]
             + [Task(key=("good",), run=lambda: "ok", gates=("g",))])
    out = scheduler.run_graph(tasks, jobs=4, gates={"g": 1})
    assert out[("good",)].ok and out[("good",)].value == "ok"


# --------------------------------------------------------------------------- #
# Failure propagation
# --------------------------------------------------------------------------- #
def test_a_failure_is_recorded_not_raised():
    """One product dying must not abort the run -- that is today's behaviour and the report
    is built from it."""
    out = scheduler.run_graph([
        Task(key=("bad",), run=lambda: (_ for _ in ()).throw(ValueError("boom"))),
        Task(key=("good",), run=lambda: "fine"),
    ], jobs=2)
    assert isinstance(out[("bad",)].error, ValueError)
    assert out[("good",)].ok


def test_a_dependent_of_a_failure_is_skipped_and_never_runs():
    """MODIS reads Landsat's aligned files. If Landsat died, running MODIS against a
    directory that was never written produces a cube channel that looks like 'no scenes
    matched' rather than 'the stage it depends on failed'."""
    ran = []
    out = scheduler.run_graph([
        Task(key=("landsat",), run=lambda: (_ for _ in ()).throw(OSError("503"))),
        Task(key=("modis",), run=lambda: ran.append("modis"), deps=(("landsat",),)),
    ], jobs=2)

    assert ran == []
    assert out[("modis",)].skipped
    assert out[("modis",)].blocked_by == (("landsat",),)
    assert not out[("modis",)].ok


def test_skipping_is_transitive():
    ran = []
    out = scheduler.run_graph([
        Task(key=("a",), run=lambda: (_ for _ in ()).throw(OSError("x"))),
        Task(key=("b",), run=lambda: ran.append("b"), deps=(("a",),)),
        Task(key=("c",), run=lambda: ran.append("c"), deps=(("b",),)),
    ], jobs=2)
    assert ran == []
    assert out[("b",)].skipped and out[("c",)].skipped


def test_an_unrelated_branch_survives_a_failure():
    """Per-AoI chains are the reason this matters: Hobart's Landsat failing must not stop
    Tamar's MODIS, because Tamar's MODIS reads Tamar's files."""
    out = scheduler.run_graph([
        Task(key=("hobart", "landsat"), run=lambda: (_ for _ in ()).throw(OSError("x"))),
        Task(key=("hobart", "modis"), run=lambda: "h", deps=(("hobart", "landsat"),)),
        Task(key=("tamar", "landsat"), run=lambda: "t"),
        Task(key=("tamar", "modis"), run=lambda: "t2", deps=(("tamar", "landsat"),)),
    ], jobs=4)
    assert out[("hobart", "modis")].skipped
    assert out[("tamar", "modis")].ok and out[("tamar", "modis")].value == "t2"


# --------------------------------------------------------------------------- #
# Graph validation
# --------------------------------------------------------------------------- #
def test_a_duplicate_key_is_rejected():
    with pytest.raises(ValueError, match="duplicate task key"):
        scheduler.run_graph([Task(key=("a",), run=lambda: None),
                             Task(key=("a",), run=lambda: None)])


def test_a_dangling_dependency_is_rejected():
    """Silently treating an unknown dep as satisfied would run a task out of order."""
    with pytest.raises(ValueError, match="unknown task"):
        scheduler.run_graph([Task(key=("a",), run=lambda: None, deps=(("missing",),))])


def test_an_empty_graph_is_fine():
    assert scheduler.run_graph([], jobs=4) == {}


# --------------------------------------------------------------------------- #
# Log attribution
# --------------------------------------------------------------------------- #
def test_a_task_runs_under_its_label():
    """Eight workers interleave into one stderr stream; without this the progress lines
    do not say which AoI or product they belong to."""
    from coastal_sst_data import logctx
    seen = []
    scheduler.run_graph([
        Task(key=("a",), run=lambda: seen.append(logctx.current()), label="hobart/mur"),
    ], jobs=1)
    assert seen == ["hobart/mur"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --------------------------------------------------------------------------- #
# Liveness. A run's only signal that it is alive is its stages' own log lines, and those go
# quiet legitimately -- a MUR granule read is minutes inside one blocking call. So a stalled
# run and a slow one look identical from outside, and one that wedged overnight was
# indistinguishable from one still working until the next morning.
# --------------------------------------------------------------------------- #
def test_a_long_wait_reports_what_it_is_waiting_for(monkeypatch, caplog):
    monkeypatch.setattr(scheduler, "HEARTBEAT_S", 0.05)
    release = threading.Event()

    tasks = [Task(key=("acquire", "ecostress", "hobart"),
                  run=lambda: release.wait(timeout=10) or "done")]

    with caplog.at_level("INFO"):
        done = threading.Thread(target=lambda: (time.sleep(0.3), release.set()))
        done.start()
        outcomes = scheduler.run_graph(tasks, jobs=1)
        done.join(timeout=5)

    assert outcomes[("acquire", "ecostress", "hobart")].ok
    assert "still running" in caplog.text
    assert "acquire/ecostress/hobart" in caplog.text, (
        "the heartbeat has to NAME the task; 'something is slow' is what we already had")


def test_the_heartbeat_escalates_once_slow_stops_explaining_it(monkeypatch, caplog):
    """`WARNING` in a scrollback should mean "this is no longer plausibly slow work" -- the
    judgement a person reading the log at 7am is trying to make."""
    monkeypatch.setattr(scheduler, "HEARTBEAT_S", 0.05)
    monkeypatch.setattr(scheduler, "STALL_S", 0.0)      # everything is already a stall
    release = threading.Event()

    with caplog.at_level("INFO"):
        done = threading.Thread(target=lambda: (time.sleep(0.3), release.set()))
        done.start()
        scheduler.run_graph([Task(key=("k",), run=lambda: release.wait(timeout=10))], jobs=1)
        done.join(timeout=5)

    assert any(r.levelname == "WARNING" and "still running" in r.message
               for r in caplog.records)


def test_the_heartbeat_does_not_change_scheduling(monkeypatch):
    """It is a report, not a policy: no task is abandoned, retried or reordered because a
    heartbeat fired. A run that used to finish must still finish, with the same outcomes."""
    monkeypatch.setattr(scheduler, "HEARTBEAT_S", 0.01)
    order, lock = [], threading.Lock()

    def note(name, delay=0.0):
        def run():
            time.sleep(delay)
            with lock:
                order.append(name)
            return name
        return run

    tasks = [Task(key=("a",), run=note("a", 0.05)),
             Task(key=("b",), run=note("b"), deps=(("a",),))]
    outcomes = scheduler.run_graph(tasks, jobs=2)

    assert order == ["a", "b"]
    assert all(o.ok for o in outcomes.values())


def test_a_stalled_run_dumps_its_stacks_once(monkeypatch, caplog):
    """The hang this was written for ran 36 hours with the heartbeat ticking and nobody
    watching. The dump is what turns that scrollback into a diagnosis -- and it fires ONCE:
    at ~430 heartbeats x ~190 threads, repeating it would bury the log it is meant to explain.
    """
    monkeypatch.setattr(scheduler, "HEARTBEAT_S", 0.05)
    monkeypatch.setattr(scheduler, "STALL_S", 0.0)      # every heartbeat is a stall
    dumps = []
    monkeypatch.setattr(scheduler.logctx, "dump_all_stacks", lambda: dumps.append(1))
    release = threading.Event()

    with caplog.at_level("INFO"):
        done = threading.Thread(target=lambda: (time.sleep(0.4), release.set()))
        done.start()
        scheduler.run_graph([Task(key=("k",), run=lambda: release.wait(timeout=10))], jobs=1)
        done.join(timeout=5)

    assert dumps == [1], f"expected exactly one stack dump, got {len(dumps)}"
    assert any("dumping every thread's stack" in r.message for r in caplog.records)


def test_a_healthy_run_dumps_nothing(monkeypatch):
    """A run that is merely SLOW is not a run that is stuck, and must stay quiet."""
    monkeypatch.setattr(scheduler, "HEARTBEAT_S", 0.05)
    monkeypatch.setattr(scheduler, "STALL_S", 3600.0)   # nothing is a stall yet
    dumps = []
    monkeypatch.setattr(scheduler.logctx, "dump_all_stacks", lambda: dumps.append(1))

    scheduler.run_graph([Task(key=("k",), run=lambda: time.sleep(0.2))], jobs=1)
    assert dumps == []
