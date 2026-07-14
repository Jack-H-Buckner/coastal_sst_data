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
import logging

from . import auth
from .config import DataProduct, Project, load_config
from .grid import AoiGrid, compute_aoi_grid
from .processes import (
    bathymetry, cmems, datacube, datum, ecostress, landcover_esa, landsat_pc, met, modis,
    mur, tides,
)

log = logging.getLogger(__name__)

# product -> acquisition module. Landsat is special (source selector, below).
PROCESSES = {
    DataProduct.bathymetry: bathymetry,
    DataProduct.mur: mur,
    DataProduct.cmems: cmems,
    DataProduct.ecostress: ecostress,
    DataProduct.met: met,
    DataProduct.modis: modis,
    DataProduct.tides: tides,
}

# Landsat source -> module. Only Planetary Computer is implemented so far.
LANDSAT_SOURCES = {"pc": landsat_pc, "planetary_computer": landsat_pc}

# Landcover source -> module. Only ESA WorldCover (via PC) is implemented so far;
# the 'gee' source (JRC + NDWI water mask) is a future landcover_gee module.
LANDCOVER_SOURCES = {"esa": landcover_esa, "worldcover": landcover_esa}

# Execution order: statics + backbone first; Landsat BEFORE MODIS (coincidence);
# not-yet-implemented products last. Every DataProduct appears exactly once.
PROCESS_ORDER = [
    DataProduct.bathymetry,
    DataProduct.mur,
    DataProduct.cmems,
    DataProduct.ecostress,
    DataProduct.landsat,
    DataProduct.modis,
    DataProduct.met,
    DataProduct.tides,
    DataProduct.landcover,
]


def _module_for(project: Project, product: DataProduct):
    """The module implementing a product, or None if not implemented yet."""
    if product == DataProduct.landsat:
        source = getattr(project.products[product], "source", "pc")
        return LANDSAT_SOURCES.get(source)
    if product == DataProduct.landcover:
        source = getattr(project.products[product], "source", "esa")
        return LANDCOVER_SOURCES.get(source)
    return PROCESSES.get(product)


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
    for product in ordered:
        module = _module_for(project, product)
        if module is None:
            detail = ""
            if product == DataProduct.landsat:
                detail = f" (source={getattr(project.products[product], 'source', 'pc')})"
            log.warning("=== %s%s: not implemented yet, skipping ===", product.value, detail)
            outcomes[product] = "skipped (not implemented)"
            continue

        log.info("=== %s ===", product.value)
        try:
            module.acquire(project, grids=grids, aois=aois,
                           dry_run=dry_run, overwrite=overwrite)
            outcomes[product] = "ok"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("=== %s FAILED: %s ===", product.value, exc)
            outcomes[product] = f"failed: {exc}"

    # Derived stage: resolve each AoI's DEM->MSL datum offset from the bathymetry file
    # that was just written (which DEM won is only known now -- bathymetry falls back
    # cudem->gmrt on coverage failure). Cheap, network-light, and idempotent; it must
    # run before the assembler, which reads its sidecar to build the water-level fields.
    if DataProduct.bathymetry in selected and project.datacube.water_level:
        log.info("=== datum (DEM->MSL offset) ===")
        try:
            datum.resolve(project, grids=grids, aois=aois,
                          dry_run=dry_run, overwrite=overwrite)
            outcomes["datum"] = "ok"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("=== datum FAILED: %s ===", exc)
            outcomes["datum"] = f"failed: {exc}"

    # Terminal stage: knit the aligned outputs into per-AoI datacubes.
    if assemble:
        log.info("=== datacube (assemble) ===")
        try:
            datacube.assemble(project, grids=grids, aois=aois,
                              dry_run=dry_run, overwrite=overwrite)
            outcomes["datacube"] = "ok"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("=== datacube FAILED: %s ===", exc)
            outcomes["datacube"] = f"failed: {exc}"

    log.info("Pipeline done. Summary:")
    for product in ordered:
        log.info("  %-12s %s", product.value, outcomes[product])
    for stage in ("datum", "datacube"):
        if stage in outcomes:
            log.info("  %-12s %s", stage, outcomes[stage])
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
