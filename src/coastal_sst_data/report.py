#!/usr/bin/env python3
"""
coastal_sst_data -- run accounting: what a run ACTUALLY did.

Until now every acquisition stage ended with an unconditional `log.info("Done.")` and the
pipeline recorded `ok`. A run that downloaded 100 days and lost 40 of them to a flaky
network printed exactly the same two lines as a run that lost none, and the cube built
afterwards had a full 100-step time axis in which the 40 missing days were NaN slices --
indistinguishable from cloudy days. The failure was real, recorded nowhere, and invisible.

So every stage now keeps a tally and hands it back:

    rep = report.ProductReport("mur")
    ...
    rep.wrote(source="GHRSST MUR-JPL-L4-GLOB-v4.1")   # one output landed
    rep.skip()                                        # already complete on disk
    rep.fail("20230314", "HTTP 503")                  # attempted and lost
    rep.finish()
    return rep

and the pipeline renders them into one table at the end of the run. Two things that table
must never do:

  * REPORT A LOSSY RUN AS `ok`. `failed > 0` makes the product's outcome `degraded`, which
    is a different word in the summary and a different exit story for whoever is reading it.
  * HIDE WHICH SOURCE SERVED. `bathymetry ok` said nothing about the DEM having quietly
    fallen back from 3 m CUDEM to ~100 m GMRT. The tally counts sources, so the table says
    so out loud.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_FAILURES_KEPT = 20      # enough to diagnose; not enough to blow up a long run's memory


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


@dataclass
class ProductReport:
    """One product's accounting for one run."""

    product: str
    written: int = 0
    skipped: int = 0
    failed: int = 0
    # What the stage MEANT to produce, when it knows (e.g. len(days)). Left None by stages
    # whose item count is only discoverable from the remote catalogue (how many granules
    # exist is not something we can assert up front).
    expected: int | None = None
    sources: Counter = field(default_factory=Counter)
    failures: list[tuple[str, str]] = field(default_factory=list)
    dropped: int = 0                     # failures beyond MAX_FAILURES_KEPT
    note: str = ""
    _t0: float = field(default_factory=time.monotonic)
    elapsed_s: float = 0.0

    # ---- recording -------------------------------------------------------- #
    def wrote(self, source: str | None = None) -> None:
        self.written += 1
        if source:
            self.sources[str(source)] += 1

    def skip(self, n: int = 1) -> None:
        self.skipped += n

    def fail(self, item, reason) -> None:
        self.failed += 1
        if len(self.failures) < MAX_FAILURES_KEPT:
            self.failures.append((str(item), str(reason)))
        else:
            self.dropped += 1

    def expect(self, n: int) -> None:
        self.expected = (self.expected or 0) + n

    def finish(self) -> "ProductReport":
        self.elapsed_s = time.monotonic() - self._t0
        return self

    def merge(self, other: "ProductReport") -> "ProductReport":
        """Fold another report for the SAME product into this one.

        One product can now be served by more than one module in a single run: a project
        whose Pacific Northwest region uses the IOOS in-situ network and whose Mediterranean
        region uses another resolves `insitu` to two source modules, each handed its own
        AoIs. They are still ONE product to the reader, so their tallies combine into one
        row -- and the `sources` counter keeps them distinguishable, which is exactly the
        column that exists to say which source actually served what.
        """
        if other is None:
            return self
        self.written += other.written
        self.skipped += other.skipped
        self.failed += other.failed
        if other.expected is not None:
            self.expected = (self.expected or 0) + other.expected
        self.sources.update(other.sources)
        room = MAX_FAILURES_KEPT - len(self.failures)
        self.failures.extend(other.failures[:max(0, room)])
        self.dropped += other.dropped + max(0, len(other.failures) - max(0, room))
        self.note = "; ".join(n for n in (self.note, other.note) if n)
        self.elapsed_s += other.elapsed_s
        return self

    # ---- reading ---------------------------------------------------------- #
    @property
    def attempted(self) -> int:
        """Items we actually went to the network for. A skipped item was not attempted."""
        return self.written + self.failed

    @property
    def ok(self) -> bool:
        return self.failed == 0

    @property
    def success_rate(self) -> float:
        return 1.0 if not self.attempted else self.written / self.attempted

    @property
    def outcome(self) -> str:
        """`ok` only when nothing was lost. A lossy run must not read as a clean one."""
        if self.failed:
            return f"degraded ({self.failed} failed)"
        if not self.written and not self.skipped:
            return "no data"
        return "ok"

    def sources_str(self) -> str:
        """Never truncated. The whole point of this column is to tell a CUDEM run from a
        GMRT one and a reanalysis day from a forecast day; an elided source name defeats it."""
        return ", ".join(f"{s} x{n}" for s, n in self.sources.most_common())

    def log_summary(self) -> None:
        """Replaces the old unconditional `Done.`"""
        self.finish()
        bits = [f"{self.written} written"]
        if self.skipped:
            bits.append(f"{self.skipped} already complete")
        if self.failed:
            bits.append(f"{self.failed} FAILED")
        line = f"{self.product}: {', '.join(bits)} in {_hms(self.elapsed_s)}"
        if self.sources:
            line += f"  [{', '.join(f'{s}: {n}' for s, n in self.sources.most_common())}]"
        (log.warning if self.failed else log.info)(line)
        if self.failed:
            for item, why in self.failures:
                log.warning("    FAILED %s: %s", item, why)
            if self.dropped:
                log.warning("    ... and %d more", self.dropped)


@dataclass
class RunReport:
    """Every product's accounting, rendered as the report printed at the end of a run."""

    products: dict[str, ProductReport] = field(default_factory=dict)
    _t0: float = field(default_factory=time.monotonic)

    def add(self, key: str, rep: ProductReport | None, *, outcome: str | None = None) -> None:
        """Record a stage. `outcome` overrides (a stage that raised, or was skipped)."""
        if rep is None:
            rep = ProductReport(key)
            rep.note = outcome or ""
            rep.finish()
        if outcome:
            rep.note = outcome
        self.products[key] = rep

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    @property
    def any_failures(self) -> bool:
        return any(r.failed for r in self.products.values())

    def render(self) -> str:
        head = (f"{'product':<12} {'written':>7} {'skipped':>7} {'failed':>6} {'rate':>6}  "
                f"{'time':>7}  status")
        out = [f"Run report  ({_hms(self.elapsed_s)} total)", "", head, "-" * len(head)]
        for key, r in self.products.items():
            rate = "-" if not r.attempted else f"{100 * r.success_rate:.0f}%"
            status = r.note or r.outcome
            out.append(f"{key:<12} {r.written:>7} {r.skipped:>7} {r.failed:>6} {rate:>6}  "
                       f"{_hms(r.elapsed_s):>7}  {status}")

        # Sources on their own lines, never elided. This is what stops `bathymetry ok` from
        # concealing a 3 m CUDEM -> ~100 m GMRT downgrade, and it is where a CMEMS run that
        # served 300 reanalysis days and 65 forecast days finally says so.
        sourced = [(k, r) for k, r in self.products.items() if r.sources]
        if sourced:
            out += ["", "SOURCES -- what actually served this run (not what was configured):"]
            for key, r in sourced:
                for src, n in r.sources.most_common():
                    out.append(f"  {key:<12} {src} x{n}")

        failed = [(k, r) for k, r in self.products.items() if r.failed]
        if failed:
            out += ["", "FAILURES -- attempted and LOST. The cube will have gaps here, and a "
                        "gap looks exactly like a cloudy day:"]
            for key, r in failed:
                for item, why in r.failures:
                    out.append(f"  {key:<12} {item}: {why}")
                if r.dropped:
                    out.append(f"  {key:<12} ... and {r.dropped} more")
        return "\n".join(out)

    def log(self) -> None:
        for line in self.render().splitlines():
            (log.warning if self.any_failures else log.info)(line)
