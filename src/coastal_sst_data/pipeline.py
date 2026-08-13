#!/usr/bin/env python3
"""
coastal_sst_data -- pipeline orchestrator.

Loads a validated config, computes the shared per-AoI grid ONCE, then dispatches
each selected product to its acquisition module -- every module honours the same
``acquire(project, *, grids=None, aois=None, dry_run=False, overwrite=False)``
contract, so orchestration is a thin loop.

Three ordering facts matter and are handled here:
  * Products run in a fixed ORDER (statics first).
  * Landsat runs BEFORE MODIS, because MODIS coincidence reads the Landsat
    aligned files (match_landsat).
  * The thermal sensors run BEFORE met_overpass and MUR, both of which read the
    sensors' aligned dirs -- met_overpass to find the instants to snapshot, MUR
    to restrict its days to the ones a sensor flew (`overpass_sensors`).

Landsat and landcover are dispatched by their ``source`` selector; only the
free/anonymous sources are implemented today (Landsat `pc`, landcover `esa`).
Products without an implementation yet (aws/gee Landsat; gee landcover) are
skipped with a warning rather than failing the run.

Usage:
    python -m coastal_sst_data.pipeline --config config.yaml
    python -m coastal_sst_data.pipeline --config config.yaml --dry-run
    python -m coastal_sst_data.pipeline --config config.yaml --products mur landsat modis
    python -m coastal_sst_data.pipeline --config config.yaml --aoi tillamook_bay
"""

from __future__ import annotations

import argparse
import importlib
import logging

from . import auth, logctx, net, products, report, scheduler
from .config import (DataProduct, DEFAULT_SOURCE, Project, gate_caps, load_config, opt,
                     resolve_opts)
from .grid import AoiGrid, compute_aoi_grid
from .processes import datacube, preprocess as preprocess_stage

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Dispatch tables -- DERIVED from the product registry (coastal_sst_data.products)
#
# These used to be four hand-maintained dicts plus a hand-ordered list, and every one of
# them had to be edited to add a product. They are now views of the registry: a product
# declares its module (or its {source: module} map) once, in its ProductSpec.
#
# Modules are named as DOTTED STRINGS and imported LAZILY -- see the import-cycle note in
# products.py. A value may also be an already-imported module object, which is what lets a
# test register a stub source without touching the filesystem.
# --------------------------------------------------------------------------- #
PROCESS_MODULES: dict[DataProduct, object] = {
    s.product: s.module for s in products.REGISTRY if s.module
}

# product -> {source name: module}. A product listed here picks its module from its
# RESOLVED source, which is per-AoI (see _modules_for). A value of None is a recognised
# source with no implementation yet (Landsat via aws/gee, landcover via gee).
SOURCE_MODULES: dict[DataProduct, dict[str, object]] = {
    s.product: dict(s.sources) for s in products.REGISTRY if s.sources
}


def _resolve(target):
    """A dotted module path -> the imported module. None and module objects pass through."""
    if target is None:
        return None
    if isinstance(target, str):
        return importlib.import_module(target)
    return target                       # already a module (or a test double)


def process_order() -> list[DataProduct]:
    """Every product, in an order that honours its declared dependencies.

    This replaces a hand-ordered list whose real constraints lived only in a comment: Landsat
    must precede MODIS (MODIS's coincidence filter reads Landsat's aligned files), and the
    sensors must precede met (whose overpass snapshots are taken at times read from their
    directories) and MUR (whose `overpass_sensors` filter reads the same directories to decide
    which days to fetch). All are now `depends_on` edges on the specs, so a new product
    declares what it needs and is placed correctly -- rather than the author having to
    reason about a global ordering and get it right by hand.

    A stable topological sort: registry declaration order is preserved wherever the
    dependencies allow, so the result still reads as "statics first".
    """
    order: list[DataProduct] = []
    placed: set[DataProduct] = set()
    remaining = [s.product for s in products.REGISTRY]

    while remaining:
        ready = [p for p in remaining
                 if all(d in placed for d in products.spec(p).depends_on)]
        if not ready:
            # Only reachable if a spec declares a dependency cycle; products.py cannot catch
            # that (it validates each spec alone), so say so loudly rather than loop forever.
            raise RuntimeError(
                f"product dependency cycle among {[p.value for p in remaining]}")
        nxt = ready[0]                  # stable: first in declaration order among the ready
        order.append(nxt)
        placed.add(nxt)
        remaining.remove(nxt)
    return order


PROCESS_ORDER = process_order()


def _source_for(project: Project, product: DataProduct, aoi: str) -> str:
    """The `source` ONE AoI runs a source-selectable product with (region -> global)."""
    return opt(resolve_opts(project, aoi, product), "source",
               DEFAULT_SOURCE.get(product))


def _module_for(project: Project, product: DataProduct, aoi: str | None = None):
    """The module implementing a product FOR ONE AoI, or None if not implemented yet.

    `aoi=None` answers for the project as a whole (the first AoI's module) -- used by
    `validate`, which is summarising rather than dispatching.
    """
    spec = products.spec(product)
    if spec.is_stacked_data:
        # DISTINCT-DATA sources (bathymetry) are all served by ONE module that fans out over
        # the configured `sources` list internally, so there is no per-AoI source selector to
        # resolve here -- dispatch the shared module once and let it stack the sources.
        return _resolve(spec.one_module())
    sources = SOURCE_MODULES.get(product)
    if sources is None:
        return _resolve(PROCESS_MODULES.get(product))
    if aoi is None:
        aoi = project.all_areas[0].name
    return _resolve(sources.get(_source_for(project, product, aoi)))


def _modules_for(project: Project, product: DataProduct, aoi_names):
    """Group AoIs by the module that will serve them: [(module|None, [aoi, ...]), ...].

    The source SELECTOR is region-dependent, not project-wide, and this is where that
    finally bites: a project whose Pacific Northwest region uses the IOOS in-situ network
    and whose Mediterranean region uses another resolves `insitu` to TWO modules in one
    run. Dispatch used to read `project.products[product].source` -- one answer for the
    whole project -- so such a config was simply impossible to express, whatever the region
    block said.

    A `None` module means "no implementation for that source"; those AoIs are reported and
    skipped rather than silently dropped.
    """
    groups: dict = {}
    for name in aoi_names:
        groups.setdefault(_module_for(project, product, name), []).append(name)
    return list(groups.items())


def _preresolve(project: Project, ordered, grids) -> None:
    """Import every module this run will dispatch, on the calling thread.

    `_module_for` imports lazily, so without this the first import of a stage happens on
    whichever worker reaches it first -- concurrently with other first imports, and
    concurrently with code relying on their side effects (`processes.tides` patches
    `collections` and `numpy` at import time to make `pytides2` importable).

    Best-effort: a product whose module cannot be imported is left for dispatch to report in
    context, exactly as it does today. The point is to have done the importing, not to add a
    new failure mode.
    """
    for product in ordered:
        for name in grids:
            try:
                _module_for(project, product, name)
            except Exception as exc:      # noqa: BLE001 -- dispatch reports it properly
                log.debug("  pre-import of %s for %s deferred to dispatch (%s)",
                          product.value, name, exc)


def _acquire_parallel(project: Project, ordered, grids, run_aois, *, dry_run, overwrite,
                      jobs, outcomes, run_report) -> None:
    """Acquire every (product, AoI) concurrently, honouring the declared dependencies.

    THE UNIT OF WORK IS (product, AoI), and that falls out of two things the package already
    had: every module takes an AoI LIST, so one call over N AoIs splits into N calls over one
    AoI each with no module change at all; and `_modules_for` already groups AoIs by the
    module that will serve them, so the split is a loop over a grouping that exists.

    THE DEPENDENCY EDGES ARE PER-AoI, which is the part worth being precise about. The three
    edges in the registry all describe one product READING ANOTHER'S ALIGNED FILES -- MODIS's
    coincidence filter reads Landsat's, MUR's `overpass_sensors` and `met_overpass` read the
    sensors' -- and every one of those reads is scoped to an AoI (`overpass.days_with_scenes`
    and `match_landsat` both filter by AoI name). So `mur(hobart)` waits for
    `landsat(hobart)`, NOT for `landsat(tamar)`. Modelling it that way rather than as global
    per-product waves is what removes the barriers: Hobart can be on MUR while Tamar is still
    downloading Landsat.

    A product whose module raises for one AoI still runs for the others; the AoIs whose
    dependencies failed are SKIPPED and say so, rather than running against a directory that
    was never written.
    """
    caps = gate_caps(project)
    tasks: list[scheduler.Task] = []
    planned: dict[DataProduct, set[str]] = {}
    unimplemented: dict[DataProduct, list[str]] = {}

    def _work(module, aoi):
        return lambda: module.acquire(project, grids=grids, aois=[aoi],
                                      dry_run=dry_run, overwrite=overwrite)

    for product in ordered:
        spec = products.spec(product)
        for module, module_aois in _modules_for(project, product, run_aois):
            if module is None:
                srcs = sorted({_source_for(project, product, a) or "?" for a in module_aois})
                unimplemented.setdefault(product, []).append(
                    f"{', '.join(module_aois)} (source={'/'.join(srcs)})")
                continue
            for aoi in module_aois:
                planned.setdefault(product, set()).add(aoi)
                # Depend only on dependencies that are actually RUNNING for THIS AoI. A
                # dependency that is unselected, or that has no implementation here, produced
                # no task -- and waiting on a task that will never exist would deadlock.
                # `ordered` is topologically sorted, so `planned` already holds every
                # dependency by the time we reach the product that needs it.
                deps = tuple(("acquire", d.value, aoi) for d in spec.depends_on
                             if aoi in planned.get(d, ()))
                tasks.append(scheduler.Task(
                    key=("acquire", product.value, aoi),
                    run=_work(module, aoi),
                    deps=deps,
                    gates=(spec.gate,),
                    label=f"{aoi}/{product.value}"))

    for product, notes in unimplemented.items():
        log.warning("=== %s: no implementation for %s; those AoI(s) are SKIPPED ===",
                    product.value, "; ".join(notes))

    log.info("Running %d task(s) on %d worker(s); service caps: %s",
             len(tasks), jobs,
             ", ".join(f"{g}={caps.get(g, 'unlimited')}"
                       for g in sorted({t.gates[0] for t in tasks})) or "none")

    results = scheduler.run_graph(tasks, jobs=jobs, gates=caps)

    # Fold the per-AoI reports back into ONE row per product, which is what the reader wants:
    # `mur` is one product whether it ran once or once per AoI.
    for product in ordered:
        merged: report.ProductReport | None = None
        ran = False
        failures, skipped = [], []
        for aoi in sorted(planned.get(product, ())):
            got = results[("acquire", product.value, aoi)]
            if got.skipped:
                skipped.append(aoi)
                continue
            if got.error is not None:
                failures.append(f"{aoi}: {got.error}")
                continue
            ran = True
            if got.value is not None:
                merged = got.value if merged is None else merged.merge(got.value)

        notes = unimplemented.get(product, [])
        if failures:
            outcomes[product] = f"failed: {'; '.join(failures)}"
            run_report.add(product.value, merged, outcome=f"stage raised: {'; '.join(failures)}")
        elif skipped and not ran:
            why = f"dependency failed for {', '.join(skipped)}"
            outcomes[product] = f"skipped ({why})"
            run_report.add(product.value, None, outcome=why)
        elif not ran:
            outcomes[product] = "skipped (not implemented)"
            run_report.add(product.value, None,
                           outcome=f"not implemented: {'; '.join(notes)}")
        else:
            outcomes[product] = merged.outcome if merged is not None else "ok"
            extra = []
            if notes:
                extra.append(f"no implementation for {'; '.join(notes)}")
            if skipped:
                extra.append(f"dependency failed for {', '.join(skipped)}")
            if extra and merged is not None:
                merged.note = "; ".join(n for n in ([merged.note] + extra) if n)
            run_report.add(product.value, merged)


def compute_grids(project: Project) -> dict[str, AoiGrid]:
    """Shared grid for every AoI, skipping any that fail (antimeridian/pole).

    One pathological AoI shouldn't abort the whole run, so grid failures are
    logged and that AoI is dropped -- the same graceful-degradation posture the
    grid/bbox code already takes at the poles.
    """
    grids = {}
    for area in project.all_areas:
        try:
            grids[area.name] = compute_aoi_grid(area, project.grid)
        except Exception as exc:
            log.warning("AoI %r: grid computation failed (%s); skipping", area.name, exc)
    return grids


def run_pipeline(project: Project, *, aois=None, products=None, dry_run=False,
                 overwrite=False, verify_auth=None, assemble=False,
                 preprocess=False, jobs=None) -> dict:
    """Run selected products for a project. Returns {product: outcome} summary.

    Products default to everything selected in the config; `products` restricts
    to a subset (must themselves be selected). A product whose module isn't
    implemented is skipped; a product that raises is logged and the run continues.

    verify_auth: run the credential preflight (auth.verify) before any work.
    Defaults to True for a real run and False for `dry_run` (a quick preview
    doesn't need it). A failing preflight aborts before anything is downloaded.

    assemble: after acquisition, knit the aligned outputs into per-AoI datacubes
    (datacube.assemble). This is a terminal stage, not a product, so it always
    runs LAST and records its outcome under the "datacube" summary key.

    preprocess: after assembly, run the post-assembly preprocessing steps
    (preprocess.preprocess), which add their derived channels to each assembled
    cube. Runs AFTER assemble (so it sees a freshly-written cube) and is a
    no-op unless the config's `preprocess.enabled` is set. Records its outcome
    under the "preprocess" summary key.

    jobs: how many (product, AoI) acquisitions run at once; None takes `runtime.jobs` from
    the config (default 1). `jobs=1` runs the ORIGINAL serial loop verbatim -- not a
    one-worker emulation of the parallel path -- so the escape hatch is a different code
    path, not a differently-tuned one.
    """
    if verify_auth is None:
        verify_auth = not dry_run
    jobs = max(1, int(project.runtime.jobs if jobs is None else jobs))

    if aois:
        valid = {a.name for a in project.all_areas}
        missing = set(aois) - valid
        if missing:
            raise SystemExit(f"AOI(s) not in config: {sorted(missing)}")

    selected = set(project.products) if products is None else set(products)
    if products is not None:
        not_selected = selected - set(project.products)
        if not_selected:
            raise SystemExit("--products includes product(s) not selected in the config: "
                             f"{sorted(p.value for p in not_selected)}")

    grids = compute_grids(project)
    if not grids:
        raise SystemExit("no usable AoI grids; nothing to do.")

    ordered = [p for p in PROCESS_ORDER if p in selected]

    # Network policy, applied ONCE for the whole run rather than from inside each stage's
    # acquire(). Both are idempotent `setdefault`-style installs, so this is not a behaviour
    # change today -- it is the point at which "several stages racing to install the same
    # global" stops being a thing that can happen at all.
    net.setup_gdal_env()
    net.setup_requests_timeout()

    # Resolve every module NOW, on the main thread, before any work is dispatched. Dispatch
    # resolves dotted module paths lazily (`_resolve` -> importlib), and some of those imports
    # have process-global side effects -- `processes.tides` monkey-patches `collections` and
    # `numpy` at import time so `pytides2` will import at all. An import racing another
    # import, or racing the code that depends on its side effect, is a class of bug that is
    # very hard to reproduce and trivial to avoid: do them all up front, single-threaded.
    # A resolution FAILURE here is also a much better error than the same failure surfacing
    # halfway through a run, after an hour of downloads.
    _preresolve(project, ordered, grids)

    # Preflight: connect to every credentialed backend the run needs, up front.
    if verify_auth:
        log.info("Preflight: verifying credentials for %s ...", [p.value for p in ordered])
        try:
            auth.verify(project, products=ordered)
        except RuntimeError as exc:
            raise SystemExit(f"credential preflight failed:\n{exc}")

    log.info("Pipeline: %d AoI(s), %d product(s) in order: %s",
             len(grids), len(ordered), [p.value for p in ordered])

    outcomes: dict[DataProduct, str] = {}
    run_report = report.RunReport()

    # The AoIs this run touches (a valid --aoi subset, or everything with a usable grid).
    run_aois = [n for n in grids if not aois or n in set(aois)]

    if jobs > 1:
        _acquire_parallel(project, ordered, grids, run_aois, dry_run=dry_run,
                          overwrite=overwrite, jobs=jobs, outcomes=outcomes,
                          run_report=run_report)
        return _finish(project, grids, aois, dry_run=dry_run, overwrite=overwrite,
                       assemble=assemble, preprocess=preprocess,
                       outcomes=outcomes, run_report=run_report)

    for product in ordered:
        log.info("=== %s ===", product.value)
        merged: report.ProductReport | None = None
        ran = False                       # a module was dispatched (even if it returned None)
        unimplemented: list[str] = []

        # One product may resolve to SEVERAL modules across a project, because the source
        # selector is region-dependent (see _modules_for). Each module gets only the AoIs
        # that resolved to it; their reports fold into one row for the product.
        for module, module_aois in _modules_for(project, product, run_aois):
            if module is None:
                srcs = sorted({_source_for(project, product, a) or "?" for a in module_aois})
                unimplemented.append(f"{', '.join(module_aois)} (source={'/'.join(srcs)})")
                continue
            try:
                rep = module.acquire(project, grids=grids, aois=module_aois,
                                     dry_run=dry_run, overwrite=overwrite)
                ran = True
                if rep is not None:
                    merged = rep if merged is None else merged.merge(rep)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                log.error("=== %s FAILED: %s ===", product.value, exc)
                outcomes[product] = f"failed: {exc}"
                run_report.add(product.value, None, outcome=f"stage raised: {exc}")
                break
        else:
            if unimplemented:
                # Not silently dropped: an AoI whose source has no module produces NOTHING,
                # and a cube with a missing product looks exactly like one whose product
                # found no data.
                log.warning("=== %s: no implementation for %s; those AoI(s) are SKIPPED ===",
                            product.value, "; ".join(unimplemented))
            if not ran:
                outcomes[product] = "skipped (not implemented)"
                run_report.add(product.value, None,
                               outcome=f"not implemented: {'; '.join(unimplemented)}")
                continue
            # `ok` ONLY when nothing was lost. A stage that dropped 40 of 100 days used to
            # report `ok` here, which is the whole reason a lossy run was invisible.
            outcomes[product] = merged.outcome if merged is not None else "ok"
            if unimplemented and merged is not None:
                merged.note = "; ".join(
                    n for n in (merged.note, f"no implementation for {'; '.join(unimplemented)}")
                    if n)
            run_report.add(product.value, merged)

    return _finish(project, grids, aois, dry_run=dry_run, overwrite=overwrite,
                   assemble=assemble, preprocess=preprocess,
                   outcomes=outcomes, run_report=run_report)


def _finish(project: Project, grids, aois, *, dry_run, overwrite, assemble, preprocess,
            outcomes, run_report):
    """The terminal stages and the run report -- shared by the serial and parallel paths.

    Extracted rather than duplicated: these are the two stages whose failure handling and
    reporting a reader is most likely to compare against the acquisition loop's, and two
    copies would be two things to keep in step.
    """
    # (The DEM->MSL datum offset is no longer a separate stage: bathymetry resolves it INLINE
    # per DEM source and stamps it onto that source's output, so it always follows the DEM it
    # belongs to. The datacube assembler surfaces it as per-source cube attributes.)

    do_preprocess = preprocess and project.preprocess.enabled
    if (assemble or do_preprocess) and project.runtime.assemble_jobs > 1 and not dry_run:
        _terminal_parallel(project, grids, aois, overwrite=overwrite, assemble=assemble,
                           preprocess=do_preprocess, outcomes=outcomes,
                           run_report=run_report)
        return _report(run_report, outcomes)

    # Terminal stage: knit the aligned outputs into per-AoI datacubes.
    if assemble:
        log.info("=== datacube (assemble) ===")
        try:
            rep = datacube.assemble(project, grids=grids, aois=aois,
                                    dry_run=dry_run, overwrite=overwrite)
            outcomes["datacube"] = rep.outcome if rep is not None else "ok"
            run_report.add("datacube", rep)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("=== datacube FAILED: %s ===", exc)
            outcomes["datacube"] = f"failed: {exc}"
            run_report.add("datacube", None, outcome=f"stage raised: {exc}")

    # Terminal stage (after assembly): post-assembly preprocessing, which adds its derived
    # channels to the cube assemble just wrote. Opt-in via the config, and it runs AFTER
    # assemble so it always sees a freshly written cube in the same invocation.
    if preprocess and project.preprocess.enabled:
        log.info("=== preprocess ===")
        try:
            rep = preprocess_stage.preprocess(project, grids=grids, aois=aois,
                                              dry_run=dry_run, overwrite=overwrite)
            outcomes["preprocess"] = rep.outcome if rep is not None else "ok"
            run_report.add("preprocess", rep)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("=== preprocess FAILED: %s ===", exc)
            outcomes["preprocess"] = f"failed: {exc}"
            run_report.add("preprocess", None, outcome=f"stage raised: {exc}")
    elif preprocess and not project.preprocess.enabled:
        log.info("=== preprocess: skipped (set `preprocess.enabled: true` in the config) ===")

    return _report(run_report, outcomes)


def _report(run_report, outcomes) -> dict:
    """The run report: what was loaded, from which source, how long it took, and -- the part
    that did not exist before -- what was ATTEMPTED AND LOST."""
    log.info("")
    run_report.log()
    if run_report.any_failures:
        log.warning("")
        log.warning("Some items were attempted and lost; the cube will have gaps where they "
                    "should be. Re-run to retry them (completed outputs are skipped), or "
                    "`coastal-sst-data check --repair` first if a run died mid-write.")
    return outcomes


def _terminal_parallel(project: Project, grids, aois, *, overwrite, assemble, preprocess,
                       outcomes, run_report) -> None:
    """Assemble (and preprocess) several AoIs at once, on a DIVIDED memory budget.

    Two facts make this safe, and both are properties the tree already had:

      * Each AoI owns its own `<aoi>.zarr`, so concurrent AoIs never touch the same store.
      * `assemble(aoi)` and `preprocess(aoi)` DO touch the same store -- preprocess reads and
        rewrites the cube assemble just wrote -- so they are chained as a dependency edge
        rather than run as two phases. That is also the win: AoI A preprocesses while AoI B
        is still assembling, instead of every AoI waiting at a barrier between the stages.

    THE MEMORY BUDGET IS DIVIDED, and this is the part that turns a working run into an OOM
    if it is skipped. `datacube.budget_bytes` detects what the machine has and halves it, on
    the assumption that it is the only thing running -- true when one AoI assembles at a
    time, false the moment two do. Four AoIs each claiming half of a 200 GB machine is a
    400 GB working set. `resolve_block_days` then does the rest by itself: a smaller budget
    yields smaller (still chunk-aligned) blocks.
    """
    jobs = int(project.runtime.assemble_jobs)
    names = [a.name for a in project.all_areas
             if a.name in grids and (not aois or a.name in set(aois))]
    if not names:
        return

    budget, src = datacube.budget_bytes(datacube._build_eff(project))
    share_gb = (budget / jobs) / 1024**3
    log.info("=== terminal stages: %d AoI(s), %d at a time; memory budget %.1f GiB (%s) "
             "-> %.1f GiB each ===", len(names), jobs, budget / 1024**3, src, share_gb)

    tasks: list[scheduler.Task] = []
    for name in names:
        if assemble:
            tasks.append(scheduler.Task(
                key=("assemble", name), label=f"{name}/assemble",
                run=lambda n=name: datacube.assemble(
                    project, grids=grids, aois=[n], overwrite=overwrite,
                    memory_budget_gb=share_gb)))
        if preprocess:
            tasks.append(scheduler.Task(
                key=("preprocess", name), label=f"{name}/preprocess",
                # Reads and rewrites the very store `assemble` produced, so this edge is a
                # correctness requirement, not an optimisation.
                deps=(("assemble", name),) if assemble else (),
                run=lambda n=name: preprocess_stage.preprocess(
                    project, grids=grids, aois=[n], overwrite=overwrite,
                    memory_budget_gb=share_gb)))

    results = scheduler.run_graph(tasks, jobs=jobs)

    for stage, enabled in (("assemble", assemble), ("preprocess", preprocess)):
        if not enabled:
            continue
        merged, failures, skipped = None, [], []
        for name in names:
            got = results[(stage, name)]
            if got.skipped:
                skipped.append(name)
            elif got.error is not None:
                failures.append(f"{name}: {got.error}")
            elif got.value is not None:
                merged = got.value if merged is None else merged.merge(got.value)
        key = "datacube" if stage == "assemble" else "preprocess"
        if failures:
            outcomes[key] = f"failed: {'; '.join(failures)}"
            run_report.add(key, merged, outcome=f"stage raised: {'; '.join(failures)}")
        elif skipped:
            # Only reachable for preprocess, and only when the assembly it needed failed.
            why = f"assembly failed for {', '.join(skipped)}"
            outcomes[key] = f"skipped ({why})"
            run_report.add(key, merged, outcome=why)
        else:
            outcomes[key] = merged.outcome if merged is not None else "ok"
            run_report.add(key, merged)


def main():
    ap = argparse.ArgumentParser(description="coastal_sst_data pipeline orchestrator.")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", dest="aois", help="Process only these AoI name(s).")
    ap.add_argument("--products", nargs="+",
                    help="Run only these products (default: all selected in the config).")
    ap.add_argument("--dry-run", action="store_true", help="Search only; no download.")
    ap.add_argument("--overwrite", action="store_true", help="reprocess existing outputs")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the credential preflight (verify) before running")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify credentials for the selected products and exit")
    ap.add_argument("--assemble", action="store_true",
                    help="after acquisition, assemble the aligned outputs into per-AoI datacubes")
    ap.add_argument("--preprocess", action="store_true",
                    help="after assembly, run the post-assembly preprocessing steps into a "
                         "derived channels into the assembled cube (needs "
                         "`preprocess.enabled` in the config)")
    ap.add_argument("--jobs", type=int, default=None, metavar="N",
                    help="acquire N (product, AoI) pairs at once (default: runtime.jobs, "
                         "or 1); per-service caps still apply (runtime.gates)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logctx.configure(verbose=args.verbose)

    project = load_config(args.config)

    products = None
    if args.products:
        try:
            products = [DataProduct(p) for p in args.products]
        except ValueError:
            raise SystemExit(f"unknown --products value; choose from "
                             f"{[p.value for p in DataProduct]}")

    if args.verify_only:
        try:
            auth.verify(project, products=products)
            print("Credentials verified.")
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        return

    run_pipeline(project, aois=args.aois, products=products,
                 dry_run=args.dry_run, overwrite=args.overwrite,
                 verify_auth=None if not args.no_verify else False,
                 assemble=args.assemble, preprocess=args.preprocess, jobs=args.jobs)


if __name__ == "__main__":
    main()
