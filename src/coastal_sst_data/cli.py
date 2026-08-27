#!/usr/bin/env python3
"""
coastal_sst_data -- command-line interface.

A single entry point over the pieces of the package:

    coastal-sst-data run      --config config.yaml [--dry-run] [--products ...]
    coastal-sst-data verify   --config config.yaml        # check credentials connect
    coastal-sst-data validate --config config.yaml        # load + summarize the config
    coastal-sst-data grids    --config config.yaml        # show each AoI's target grid
    coastal-sst-data assemble --config config.yaml        # knit aligned outputs -> datacubes
    coastal-sst-data preprocess --config config.yaml      # post-assembly derived channels
    coastal-sst-data extract  --config config.yaml        # point time series -> parquet/csv
    coastal-sst-data check    --config config.yaml        # find truncated/incomplete outputs

(Equivalent to `python -m coastal_sst_data.cli <command> ...`.) Each subcommand
just wires argparse to an existing function -- run_pipeline, auth.verify,
load_config, compute_grids, datacube.assemble, preprocess.preprocess -- so the CLI
stays a thin shell.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import auth, logctx, store
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
        preprocess=args.preprocess,
        jobs=args.jobs,
    )


def _cmd_assemble(args):
    from .processes import datacube
    project = load_config(args.config)
    datacube.assemble(project, aois=args.aois, dry_run=args.dry_run,
                      overwrite=args.overwrite,
                      memory_budget_gb=args.memory_budget_gb)


def _cmd_preprocess(args):
    from .processes import preprocess
    project = load_config(args.config)
    preprocess.preprocess(project, aois=args.aois, dry_run=args.dry_run,
                          overwrite=args.overwrite,
                          memory_budget_gb=args.memory_budget_gb)


def _cmd_extract(args):
    """Pull point time series out of the assembled cubes -> one long-format table.

    `processes.extract` is imported HERE, not at module scope: most projects never extract,
    and this keeps the stage (and its pandas/table machinery) off every other subcommand's
    import path.
    """
    from .processes import extract
    project = load_config(args.config)
    try:
        extract.extract(project, aois=args.aois, points_file=args.points, out=args.out,
                        fmt=args.fmt, dry_run=args.dry_run, overwrite=args.overwrite)
    except ImportError as exc:
        # store.write_table raises this when the parquet backend is missing. Say what to do
        # rather than quietly writing a different format than the one asked for.
        raise SystemExit(
            f"{exc}. Install it with `pip install 'coastal_sst_data[extract]'` "
            f"(or `conda install pyarrow`), or pass --format csv.")


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

    # --insitu is only ever useful as a layer ON a map, so it implies --plot rather than
    # silently doing nothing when someone forgets the other flag.
    if args.plot or args.insitu:
        from .plot import plot_project_aois
        try:
            paths = plot_project_aois(project, grids=grids, out_dir=args.plot_dir,
                                      show=args.show, insitu=args.insitu,
                                      insitu_halo_km=args.insitu_halo_km,
                                      insitu_pad=args.insitu_pad,
                                      insitu_max_labels=args.insitu_max_labels)
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
def _cmd_check(args):
    """Scan the output tree for files that are not finished outputs.

    Anything written before the writes became atomic may be truncated, and the skip guard
    has been treating those as done on every run. This reads each file's PAYLOAD -- not
    just its header, which survives a truncation intact -- and reports what it cannot read.
    With --repair it deletes them, which IS the repair: the next run finds no output there
    and re-fetches it, without a full --overwrite of everything that was already fine.

    --repair REFUSES to run while another job is writing this tree. Its verdicts are a
    snapshot: a file a concurrent run replaces during the scan can be condemned as bad and
    then deleted after it has become perfectly good, and its in-flight scratch is not junk at
    all. Neither is worth guessing at, so the answer is to wait for the other run to finish.
    """
    project = load_config(args.config)
    root = Path(project.output_dir)
    if not root.exists():
        raise SystemExit(f"output_dir does not exist: {root}")

    print(f"Scanning {root} ({'metadata only' if args.quick else 'deep read'}) ...")
    n, bad, leftovers, in_use = store.scan(root, aois=args.aois, deep=not args.quick)

    for product, f in bad:
        print(f"  BAD      {product:<11} {f.relative_to(root)}")
    for p in leftovers:
        print(f"  SCRATCH  {'':<11} {p.relative_to(root)}   (a run died mid-write)")
    for p in in_use:
        print(f"  IN USE   {'':<11} {p.relative_to(root)}   "
              f"({store.scratch_owner(p)} is still writing it)")

    print(f"\n{n} file(s) checked, {len(bad)} unreadable/incomplete, "
          f"{len(leftovers)} scratch leftover(s), {len(in_use)} in use.")
    if in_use and args.repair and not args.force:
        raise SystemExit(
            f"Refusing to repair: another run is writing this tree ({len(in_use)} file(s) in "
            f"use). Deleting its in-flight scratch is exactly the failure this pass exists "
            f"to prevent, and its verdicts on the rest are a moving target. Wait for that "
            f"run to finish, then re-run -- or pass --force if you KNOW nothing is running "
            f"(a reboot can leave scratch whose pid has since been reused).")
    if in_use and not args.repair:
        print("Another run is writing this tree; --repair will refuse until it finishes.")
    if not bad and not leftovers and not (in_use and args.force):
        print("Tree is clean." if not in_use else "Nothing to repair.")
        return
    if not args.repair:
        print("Re-run with --repair to delete these; the next run will then re-fetch them.")
        return
    removed = store.repair(bad, leftovers + (in_use if args.force else []), force=args.force)
    print(f"Removed {removed} path(s). Re-run the pipeline to re-fetch them.")


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
    p_run.add_argument("--jobs", type=int, default=None, metavar="N",
                       help="Acquire N (product, AoI) pairs at once (default: runtime.jobs "
                            "in the config, or 1). Per-service caps still apply -- see "
                            "runtime.gates. 1 runs the original serial path.")
    p_run.add_argument("--assemble", action="store_true",
                       help="After acquisition, assemble the aligned outputs into datacubes.")
    p_run.add_argument("--preprocess", action="store_true",
                       help="After assembly, run post-assembly preprocessing into a separate "
                            "derived channels into the assembled cube (needs "
                            "`preprocess.enabled` in the config).")
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
    p_grids.add_argument("--insitu", action="store_true",
                         help="Also save one in-situ context map per AoI, showing the "
                              "monitoring stations near it. Implies --plot. Queries ERDDAP "
                              "and the Copernicus index (metadata only, no data download).")
    p_grids.add_argument("--insitu-halo-km", dest="insitu_halo_km", type=float, default=None,
                         help="How far beyond each AoI to look for stations, in km "
                              "(default: 10). Raise it to see what a bigger move would catch.")
    p_grids.add_argument("--insitu-pad", dest="insitu_pad", type=float, default=2.5,
                         help="How much wider than the AoI to draw the in-situ maps "
                              "(linear factor, default: 2.5).")
    p_grids.add_argument("--insitu-max-labels", dest="insitu_max_labels", type=int, default=25,
                         help="Cap on labelled fixed stations per in-situ map (default: 25); "
                              "the full list is always written to the sidecar CSV.")
    p_grids.set_defaults(func=_cmd_grids)

    p_asm = sub.add_parser("assemble",
                           help="Knit the aligned per-product outputs into per-AoI datacubes.")
    add_common(p_asm)
    p_asm.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_asm.add_argument("--overwrite", action="store_true", help="Rebuild existing .zarr cubes.")
    p_asm.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    p_asm.add_argument("--memory-budget-gb", type=float, default=None,
                       dest="memory_budget_gb", metavar="GB",
                       help="Override the memory budget for this run (default: detected, halved).")
    p_asm.set_defaults(func=_cmd_assemble)

    p_pre = sub.add_parser(
        "preprocess",
        help="Post-assembly preprocessing: read each assembled cube and write a separate "
             "derived channels (waterline, gap-filled level-4, screened SST) into the "
             "assembled cube. Needs `preprocess.enabled`.")
    add_common(p_pre)
    p_pre.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_pre.add_argument("--overwrite", action="store_true",
                       help="Rebuild existing derived .zarr cubes.")
    p_pre.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    p_pre.add_argument("--memory-budget-gb", type=float, default=None,
                       dest="memory_budget_gb", metavar="GB",
                       help="Override the memory budget for this run (default: detected, halved).")
    p_pre.set_defaults(func=_cmd_preprocess)

    p_ext = sub.add_parser(
        "extract",
        help="Extract long-format time series from the assembled cubes at a CSV of lat/lon "
             "points. Optional: needs `extract.channels` in the config, and nothing else "
             "runs it.")
    add_common(p_ext)
    p_ext.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_ext.add_argument("--points", help="Points CSV (overrides `extract.points`).")
    p_ext.add_argument("--out", help="Output path (default: "
                                     "<output_dir>/<extract.output_subdir>/<stem>.<ext>).")
    # default=None on purpose: `default='parquet'` would silently override a config that
    # says `format: csv`, on every single invocation.
    p_ext.add_argument("--format", dest="fmt", choices=["parquet", "csv"], default=None,
                       help="Output format (default: `extract.format`, i.e. parquet).")
    p_ext.add_argument("--overwrite", action="store_true",
                       help="Replace an existing table.")
    p_ext.add_argument("--dry-run", action="store_true",
                       help="Report the point-to-AoI assignment, the channel plan and the "
                            "row count; write nothing.")
    p_ext.set_defaults(func=_cmd_extract)

    # The DEM->MSL datum offset is resolved INLINE by the bathymetry stage (per DEM source)
    # and stamped onto its output, so there is no separate `datum` subcommand -- re-run
    # `bathymetry --overwrite` to re-resolve it.

    p_prov = sub.add_parser(
        "provenance",
        help="Print an assembled cube's provenance: the config that built it, each "
             "field's sources, and when they were accessed.")
    add_common(p_prov)
    p_prov.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_prov.add_argument("--fields", action="store_true",
                        help="Also list every field and the product(s) it came from.")
    p_prov.set_defaults(func=_cmd_provenance)

    p_chk = sub.add_parser(
        "check",
        help="Scan the output tree for truncated or incomplete files and (with --repair) "
             "delete them so the next run re-fetches them.")
    add_common(p_chk)
    p_chk.add_argument("--aoi", nargs="+", dest="aois", help="Only these AoI name(s).")
    p_chk.add_argument("--quick", action="store_true",
                       help="Metadata only: check the layers, don't read the payload. "
                            "Much faster, but cannot see a truncated chunk.")
    p_chk.add_argument("--repair", action="store_true",
                       help="Delete the bad files (default: report only).")
    p_chk.add_argument("--force", action="store_true",
                       help="With --repair: delete scratch that still looks in use. Only "
                            "when you KNOW nothing is running -- a reboot can leave scratch "
                            "whose owning pid has since been reused, which nothing else "
                            "will ever clear.")
    p_chk.set_defaults(func=_cmd_check)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    logctx.configure(verbose=args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
