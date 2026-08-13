#!/usr/bin/env python3
"""
coastal_sst_data -- run a dependency graph of tasks concurrently, under per-backend caps.

The pipeline's ordering constraints are already declarative (`ProductSpec.depends_on`), and
every acquisition module already takes an AoI LIST -- so one call over N AoIs splits into N
calls over one AoI each without a single module changing. What was missing is something to
run the resulting tasks in dependency order, several at a time. That is this module, and it
is deliberately the ONLY place in the tree that knows what a thread pool is.

WHY THREADS AND NOT PROCESSES. `auth` keeps one credential record per backend in process
globals, guarded by a lock, with a 60-second storm guard that coalesces simultaneous refresh
requests onto a single re-login and a per-run refresh budget. Under threads, N workers share
one login, one budget and one rate-limit window -- one account footprint no matter how wide
the pool. Under processes each worker would get its own copy of all three, so eight workers
would become eight independent login storms against one account. The GIL is not the cost it
looks like here: the work is either waiting on a socket or inside GDAL/numpy/blosc, and both
release it.

TWO LIMITS, NOT ONE. A single `jobs` number is the wrong knob, because the backends tolerate
very different concurrency: Earthdata is happy with several granule reads at once, CMEMS
hands out a dataset handle that is not safe to share at all, and the small NOAA metadata
endpoints are behind a module-global `requests.Session`. So a task declares the GATES it
needs, and a gate is a semaphore with its own cap. `jobs` bounds the pool; the gates bound
each service.

GATES ARE TAKEN BEFORE SUBMISSION, NOT INSIDE THE WORKER, and that is load-bearing. A worker
that blocks waiting for a gate is a worker not doing anything else -- eight queued CMEMS days
behind a cap of one would occupy the whole pool and starve every other product. Acquiring in
the scheduling loop instead means a task that cannot get its gate simply is not started yet,
and the pool stays free for work that can run.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from . import logctx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    """One unit of work, and what has to be true before it may start."""

    key: tuple                       # unique; also how `deps` refer to it
    run: Callable[[], object]        # zero-argument, because everything else is bound already
    deps: tuple = ()                 # keys that must SUCCEED first
    gates: tuple[str, ...] = ()      # semaphore names, all of which must be free
    label: str = ""                  # stamped onto this task's log lines (see logctx)


@dataclass
class Outcome:
    """What became of one task. Three states, and the third is the one that matters.

    A task whose dependency failed is SKIPPED, not failed and not quietly absent: an AoI
    whose Landsat stage died must not have its MODIS coincidence filter run against a
    directory that was never written, and it must not look like MODIS simply found nothing.
    """

    key: tuple
    value: object = None
    error: BaseException | None = None
    blocked_by: tuple = ()

    @property
    def ok(self) -> bool:
        return self.error is None and not self.blocked_by

    @property
    def skipped(self) -> bool:
        return bool(self.blocked_by)


@dataclass
class _Gates:
    """Named concurrency caps. An unnamed gate is unlimited."""

    caps: dict = field(default_factory=dict)
    _sems: dict = field(default_factory=dict)

    def __post_init__(self):
        self._sems = {name: threading.BoundedSemaphore(max(1, int(cap)))
                      for name, cap in self.caps.items()}

    def take(self, names) -> bool:
        """All-or-nothing, non-blocking. True if every gate was acquired."""
        taken = []
        for name in sorted(set(names)):
            sem = self._sems.get(name)
            if sem is None:
                continue                       # no cap declared -> unlimited
            if not sem.acquire(blocking=False):
                for held in taken:             # partial acquisition helps nobody
                    self._sems[held].release()
                return False
            taken.append(name)
        return True

    def give(self, names) -> None:
        for name in sorted(set(names)):
            sem = self._sems.get(name)
            if sem is not None:
                sem.release()


def _validate(tasks: list[Task]) -> dict:
    """Index by key, rejecting duplicates and dangling dependencies."""
    by_key: dict = {}
    for t in tasks:
        if t.key in by_key:
            raise ValueError(f"duplicate task key {t.key!r}")
        by_key[t.key] = t
    for t in tasks:
        for d in t.deps:
            if d not in by_key:
                raise ValueError(f"task {t.key!r} depends on unknown task {d!r}")
    return by_key


def run_graph(tasks: list[Task], *, jobs: int = 1, gates: dict | None = None) -> dict:
    """Run `tasks` in dependency order, up to `jobs` at once. -> {key: Outcome}.

    Every task runs exactly once. A task runs only after every task in its `deps` has
    SUCCEEDED; if any dependency failed or was itself skipped, this one is skipped and says
    which key blocked it.

    Deterministic where it can be: ready tasks are started in the order they were given, so
    `jobs=1` walks the list in a stable topological order.
    """
    by_key = _validate(tasks)
    order = [t.key for t in tasks]
    gate_pool = _Gates(dict(gates or {}))

    outcomes: dict = {}
    pending = set(order)
    running: dict = {}                 # future -> key
    jobs = max(1, int(jobs))

    def _blockers(task: Task) -> tuple:
        return tuple(d for d in task.deps
                     if d in outcomes and not outcomes[d].ok)

    def _ready(task: Task) -> bool:
        return all(d in outcomes and outcomes[d].ok for d in task.deps)

    def _invoke(task: Task):
        with logctx.label(task.label):
            return task.run()

    with cf.ThreadPoolExecutor(max_workers=jobs,
                               thread_name_prefix="csd") as pool:
        try:
            while pending or running:
                # 1. Retire anything whose dependencies have already failed. Done before
                #    submitting, so a doomed subtree never occupies a slot or a gate.
                for key in [k for k in order if k in pending]:
                    blocked = _blockers(by_key[key])
                    if blocked:
                        outcomes[key] = Outcome(key, blocked_by=blocked)
                        pending.discard(key)
                        log.warning("  SKIPPED %s -- %s did not succeed",
                                    _name(key), ", ".join(_name(b) for b in blocked))

                # 2. Start whatever is ready, has a free slot, and can take its gates.
                for key in [k for k in order if k in pending]:
                    if len(running) >= jobs:
                        break
                    task = by_key[key]
                    if not _ready(task):
                        continue
                    if not gate_pool.take(task.gates):
                        continue          # service is at capacity; try again next round
                    pending.discard(key)
                    running[pool.submit(_invoke, task)] = key

                if not running:
                    if pending:
                        # Nothing running and nothing startable. With gates held only by
                        # running tasks this is unreachable, but a silent hang would be far
                        # worse than a loud error if it ever became reachable.
                        raise RuntimeError(
                            f"scheduler stalled with {len(pending)} task(s) unstartable: "
                            f"{sorted(_name(k) for k in pending)}")
                    break

                # 3. Wait for the first result, so the loop reacts as soon as anything frees
                #    a slot or unblocks a dependent -- rather than at a batch barrier.
                finished, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    key = running.pop(future)
                    gate_pool.give(by_key[key].gates)
                    try:
                        outcomes[key] = Outcome(key, value=future.result())
                    except BaseException as exc:          # noqa: BLE001 -- recorded, not lost
                        if isinstance(exc, KeyboardInterrupt):
                            raise
                        outcomes[key] = Outcome(key, error=exc)
                        log.error("  FAILED %s: %s", _name(key), exc)
        except KeyboardInterrupt:
            # Running futures cannot be cancelled -- Python cannot interrupt a thread mid-read
            # -- so the honest thing is to drop the queue, say what we are waiting for, and
            # let the in-flight reads finish. `store.atomic` guarantees none of them can leave
            # a half-written output behind.
            log.warning("Interrupted: cancelling %d queued task(s); waiting for %d in flight "
                        "(a download cannot be killed mid-read).", len(pending), len(running))
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    return outcomes


def _name(key) -> str:
    """A task key rendered for a human: ('acquire', 'mur', 'hobart') -> 'acquire/mur/hobart'."""
    return "/".join(str(p) for p in key) if isinstance(key, tuple) else str(key)
