#!/usr/bin/env python3
"""
coastal_sst_data -- command-line interface.

A single entry point over the pieces of the package:

    coastal-sst-data run      --config config.yaml [--dry-run] [--products ...]
    coastal-sst-data verify   --config config.yaml        # check credentials connect
    coastal-sst-data validate --config config.yaml        # load + summarize the config
    coastal-sst-data grids    --config config.yaml        # show each AoI's target grid
    coastal-sst-data assemble --config config.yaml        # knit aligned outputs -> datacubes

(Equivalent to `python -m coastal_sst_data.cli <command> ...`.) Each subcommand
just wires argparse to an existing function -- run_pipeline, auth.verify,
load_config, compute_grids, datacube.assemble -- so the CLI stays a thin shell.
"""

from __future__ import annotations

import argparse
import logging

from . import auth
from .config import DataProduct, load_config, required_backend
from .pipeline import _module_for, compute_grids, run_pipeline

log = logging.getLogger(__name__)


def _parse_products(values):
    """CLI product names -> [DataProduct], or None. Fails loudly on a typo."""
    if not values:
        return None
    try:
        return [DataProduct(v) for v in values]
    except ValueError:
        raise SystemExit(
            f"unknown product name; choose from {[p.value for p in DataProduct]}")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def _cmd_run(args):
    project = load_config(args.config)
    run_pipeline(
        project,
        aois=args.aois,
        products=_parse_products(args.products),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        verify_auth=False if args.no_verify else None,
        assemble=args.assemble,
    )


def _cmd_assemble(args):
    from .processes import datacube
    project = load_config(args.config)
    datacube.assemble(project, aois=args.aois, dry_run=args.dry_run,
                      overwrite=args.overwrite)


def _cmd_provenance(args):
    """Print an assembled cube's provenance: config, sources, access dates."""
    import json
    import xarray as xr
    from .processes import datacube

    project = load_config(args.config)
    out_dir = project.output_dir / project.datacube.output_subdir
    names = args.aois or [a.name for a in project.all_areas]
    for name in names:
        zpath = out_dir / f"{name}.zarr"
        if not zpath.exists():
            print(f"{name}: no cube at {zpath}")
            continue
        ds = xr.open_zarr(zpath)
        a = ds.attrs
        print(f"\n=== {name} ===")
        print(f"  built    : {a.get('created_at')}  (coastal_sst_data {a.get('package_version')})")
        print(f"  config   : {a.get('config_path') or '(from a dict)'}")
        print(f"  sha256   : {a.get('config_sha256')}")
        cur = project.config_sha256
        if a.get("config_sha256") and a["config_sha256"] != cur:
            print(f"  !! the config has CHANGED since this cube was built (now {cur[:12]}...)")
        prods = json.loads(a.get("provenance_products", "{}"))
        if prods:
            print("  sources:")
            for prod, r in sorted(prods.items()):
                flag = "" if r["basis"] == "stamped" else "   [date from file mtime, not recorded]"
                print(f"    {prod:12s} {', '.join(r['sources'])[:52]:54s} "
                      f"{(r['accessed_last'] or '')[:19]}  n={r['n_files']}{flag}")
        if args.fields:
            print("  fields:")
            for f, r in sorted(json.loads(a.get("provenance", "{}")).items()):
                print(f"    {f:24s} <- {', '.join(r['inputs']) or '(computed)'}")
        ds.close()


def _cmd_datum(args):
    from .processes import datum
    project = load_config(args.config)
    datum.resolve(project, aois=args.aois, dry_run=args.dry_run,
                  overwrite=args.overwrite)


def _cmd_verify(args):
    project = load_config(args.config)
    try:
        results = auth.verify(project, products=_parse_products(args.products))
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    if results:
        print("Credentials verified: " + ", ".join(f"{b}=ok" for b in results))
    else:
        print("No credentialed backends required (all selected products are public).")


def _cmd_validate(args):
    project = load_config(args.config)
    print(f"Config OK: {project.name}")
    print(f"  output_dir : {project.output_dir}")
    print(f"  time       : {project.time.start_date} -> {project.time.end_date}")
    print(f"  grid       : {project.grid.resolution_m:.0f} m, crs={project.grid.target_crs}")
    n_aoi = len(project.all_areas)
    print(f"  regions    : {len(project.regions)} ({n_aoi} AoI(s))")
    print("  products   :")
    for product, opts in project.products.items():
        backend = required_backend(product, opts) or "-"
        module = _module_for(project, product)
        status = "ready" if module is not None else "NOT IMPLEMENTED"
        source = getattr(opts, "source", None)
        src = f" source={source}" if source else ""
        print(f"    - {product.value:<11}{src:<16} auth={backend:<9} [{status}]")


def _cmd_grids(args):
    project = load_config(args.config)
    grids = compute_grids(project)
    wanted = set(args.aois) if args.aois else None
    for name, g in grids.items():
        if wanted and name not in wanted:
            continue
        w, s, e, n = g.search_bbox
        print(f"{name}: crs={g.target_crs} {g.width}x{g.height} @ {g.resolution_m:.0f} m "
              f"bbox=({w:.3f},{s:.3f},{e:.3f},{n:.3f})")

    if args.plot:
        from .plot import plot_project_aois
        try:
            paths = plot_project_aois(project, grids=grids, out_dir=args.plot_dir,
                                      show=args.show)
        except ImportError as exc:
            raise SystemExit(
                f"plotting needs matplotlib (optional): {exc}. Install it with "
                "`conda install matplotlib` (add `cartopy` for coastlines).")
        if paths:
            print("Wrote map(s):")
            for p in paths:
                print(f"  {p}")
        else:
            print("No AoI maps written (no drawable AoIs).")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="coastal-sst-data",
        description="Acquire and grid coastal SST + covariate data products.")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--config", required=True, help="Path to a project config YAML.")
        p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    p_run = sub.add_parser("run", help="Run the acquisition pipeline.")
    add_common(p_run)
    p_run.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_run.add_argument("--products", nargs="+", help="Only these products (default: all selected).")
    p_run.add_argument("--dry-run", action="store_true", help="Search only; no download.")
    p_run.add_argument("--overwrite", action="store_true", help="Reprocess existing outputs.")
    p_run.add_argument("--no-verify", action="store_true", help="Skip the credential preflight.")
    p_run.add_argument("--assemble", action="store_true",
                       help="After acquisition, assemble the aligned outputs into datacubes.")
    p_run.set_defaults(func=_cmd_run)

    p_verify = sub.add_parser("verify", help="Verify configured credentials connect.")
    add_common(p_verify)
    p_verify.add_argument("--products", nargs="+", help="Only check these products' backends.")
    p_verify.set_defaults(func=_cmd_verify)

    p_val = sub.add_parser("validate", help="Load + validate the config and print a summary.")
    add_common(p_val)
    p_val.set_defaults(func=_cmd_validate)

    p_grids = sub.add_parser("grids", help="Show the computed target grid for each AoI.")
    add_common(p_grids)
    p_grids.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_grids.add_argument("--plot", action="store_true",
                         help="Also save AoI maps: an all-regions overview + one per region.")
    p_grids.add_argument("--plot-dir", dest="plot_dir",
                         help="Directory for the maps (default: <output_dir>/figures).")
    p_grids.add_argument("--show", action="store_true",
                         help="Display the maps interactively (with --plot).")
    p_grids.set_defaults(func=_cmd_grids)

    p_asm = sub.add_parser("assemble",
                           help="Knit the aligned per-product outputs into per-AoI datacubes.")
    add_common(p_asm)
    p_asm.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_asm.add_argument("--overwrite", action="store_true", help="Rebuild existing .zarr cubes.")
    p_asm.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    p_asm.set_defaults(func=_cmd_assemble)

    p_dat = sub.add_parser(
        "datum",
        help="Resolve each AoI's DEM->MSL datum offset (needed by the water-level fields).")
    add_common(p_dat)
    p_dat.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_dat.add_argument("--overwrite", action="store_true",
                       help="Re-resolve offsets that are already on disk.")
    p_dat.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    p_dat.set_defaults(func=_cmd_datum)

    p_prov = sub.add_parser(
        "provenance",
        help="Print an assembled cube's provenance: the config that built it, each "
             "field's sources, and when they were accessed.")
    add_common(p_prov)
    p_prov.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_prov.add_argument("--fields", action="store_true",
                        help="Also list every field and the product(s) it came from.")
    p_prov.set_defaults(func=_cmd_provenance)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()
