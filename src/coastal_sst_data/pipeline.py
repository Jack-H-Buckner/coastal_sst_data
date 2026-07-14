#!/usr/bin/env python3
"""
coastal_sst_data -- pipeline orchestrator.

Loads a validated config, computes the shared per-AoI grid ONCE, then dispatches
each selected product to its acquisition module -- every module honours the same
``acquire(project, *, grids=None, aois=None, dry_run=False, overwrite=False)``
contract, so orchestration is a thin loop.

Two ordering facts matter and are handled here:
  * Products run in a fixed ORDER (statics/backbone first).
  * Landsat runs BEFORE MODIS, because MODIS coincidence reads the Landsat
    aligned files (match_landsat).

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

from . import auth, products, report
from .config import (DataProduct, DEFAULT_SOURCE, Project, load_config, opt, resolve_opts)
from .grid import AoiGrid, compute_aoi_grid
from .processes import datacube, datum

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

    This replaces a hand-ordered list whose two real constraints lived only in a comment:
    Landsat must precede MODIS (MODIS's coincidence filter reads Landsat's aligned files),
    and the sensors must precede met (met's overpass snapshots are taken at times read from
    their directories). Both are now `depends_on` edges on the specs, so a new product
    declares what it needs and is placed correctly -- rather than the author having to
    reason about a global ordering and get it right by hand.

    A stable topological sort: registry declaration order is preserved wherever the
    dependencies allow, so the result still reads as "statics and backbone first".
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
                 overwrite=False, verify_auth=None, assemble=False) -> dict:
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
    """
    if verify_auth is None:
        verify_auth = not dry_run

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

    # Derived stage: resolve each AoI's DEM->MSL datum offset from the bathymetry file
    # that was just written (which DEM won is only known now -- bathymetry falls back
    # cudem->gmrt on coverage failure). Cheap, network-light, and idempotent; it must
    # run before the assembler, which reads its sidecar to build the water-level fields.
    if DataProduct.bathymetry in selected and project.datacube.water_level:
        log.info("=== datum (DEM->MSL offset) ===")
        try:
            rep = datum.resolve(project, grids=grids, aois=aois,
                                dry_run=dry_run, overwrite=overwrite)
            outcomes["datum"] = rep.outcome if rep is not None else "ok"
            run_report.add("datum", rep)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("=== datum FAILED: %s ===", exc)
            outcomes["datum"] = f"failed: {exc}"
            run_report.add("datum", None, outcome=f"stage raised: {exc}")

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

    # The run report: what was loaded, from which source, how long it took, and -- the part
    # that did not exist before -- what was ATTEMPTED AND LOST.
    log.info("")
    run_report.log()
    if run_report.any_failures:
        log.warning("")
        log.warning("Some items were attempted and lost; the cube will have gaps where they "
                    "should be. Re-run to retry them (completed outputs are skipped), or "
                    "`coastal-sst-data check --repair` first if a run died mid-write.")
    return outcomes


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
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

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
                 assemble=args.assemble)


if __name__ == "__main__":
    main()
