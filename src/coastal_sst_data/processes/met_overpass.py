#!/usr/bin/env python3
"""
coastal_sst_data -- meteorological OVERPASS documentation (met matched to sensor overpasses).

The forcing product (`met`) gives daily fields at a reference time. This product gives the
weather at the EXACT instant a thermal sensor flew -- which is a real information source, not
a reconstructable derivation: a model's value at 14:32 cannot be interpolated from a daily
sample (D13). It reuses `met`'s HRRR/ERA5 fetchers, but:

  * it DEPENDS ON the sensors -- it reads their aligned dirs to discover the overpass instants
    to snapshot against (Kind.OVERPASS_ALIGNED: timestamped rasters read at another product's
    times, with no clearest-scene pick and no SensorSpec);
  * the user names exactly which `(sensor, source)` COMBINATIONS to produce (D13/D15), so the
    cube emits `<sensor>_<var>_<source>` for those pairings only -- not the sensor x source
    cross-product.

    <output_dir>/MET_OVERPASS/<source>/aligned/<aoi>/<aoi>_<YYYYMMDDThhmmss>.nc

Distinct-data sources are STACKED (D10), same hrrr/era5 as forcing; a source with no data at
an overpass instant simply contributes NaN to its channel there (no fallback).

Usage (run AFTER the SST + met stages):
    python -m coastal_sst_data.processes.met_overpass --config config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, naming, net, overpass, provenance, report, store
from . import met

log = logging.getLogger(__name__)

SOURCE = "met_overpass"


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    """Acquire met snapshots at each combo's sensor overpass instants, one tree PER SOURCE.

    For each source, the overpass times are the UNION over the sensors paired with it in the
    `(sensor, source)` combinations -- one snapshot per instant, reused by every combo on that
    source. `only_source` narrows to one source.
    """
    grid_cfg = eff["grid"]
    mo_root, fmt, overwrite = eff["mo_root"], eff["fmt"], eff["overwrite"]
    to_celsius = grid_cfg.get("to_celsius", False)
    days = pd.date_range(eff["time"]["start_date"], eff["time"]["end_date"], freq="D")
    sensor_dirs = overpass.sensor_dirs(eff["root"])

    names = select_aois(grids, only_aoi)
    rep = report.ProductReport("met_overpass")

    for name in names:
        g = grids[name]
        ds_cfg = eff["ds"][name]
        # Group the paired sensors by source: one lazy fetch per source instant serves every
        # combo that reads it.
        by_source: dict[str, set] = {}
        for sensor, src in ds_cfg["combinations"]:
            by_source.setdefault(src, set()).add(sensor)

        for src in sorted(by_source):
            if only_source is not None and src != only_source:
                continue
            dirs = [d for pre in sorted(by_source[src]) if pre in sensor_dirs
                    for d in sensor_dirs[pre]]
            log.info("=== AOI: %s | source=%s | sensors=%s ===",
                     name, src, sorted(by_source[src]))
            aoi_out = mo_root / src / "aligned" / name

            for day in days:
                for op in overpass.times_for_day(dirs, name, day):
                    op_stem = naming.time_stem(name, op)
                    if store.done(aoi_out / f"{op_stem}.nc", store.REQUIRED_VARS["MET_OVERPASS"],
                                  shape=(g.height, g.width), overwrite=overwrite):
                        continue
                    if dry_run:
                        log.info("  [dry-run] would snapshot %s @ %s", src, naming.time_stamp(op))
                        continue
                    # `_fetch_one` RAISES on a failed read and returns None when there is
                    # genuinely nothing here (see its docstring). Tallying both as "no data"
                    # is what made a lost snapshot indistinguishable from an AoI the model
                    # does not cover.
                    try:
                        got = met._fetch_one(src, g, op, ds_cfg)
                    except Exception as exc:
                        log.warning("  overpass %s [%s] FAILED (%s)",
                                    naming.time_stamp(op), src, exc)
                        rep.fail(f"{name} {src} {naming.time_stamp(op)}", f"{src}: {exc}")
                        continue
                    if not got:
                        rep.fail(f"{name} {src} {naming.time_stamp(op)}", f"{src}: no data")
                        continue
                    ds = met.to_dataset(got, g, op, to_celsius)
                    ds.attrs.update(aoi_id=name, source=src, met_source=src,
                                    **provenance.stamp(eff))
                    log.info("  overpass %s [%s] -> %s", naming.time_stamp(op), src,
                             store.write_output(ds, aoi_out, op_stem, fmt))
                    rep.wrote(source=src)
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's overpass settings: the `(sensor, source)` combinations + the fetch knobs
    (shared with `met`). The combinations are region-overridable (which sensors/sources are
    worth pairing here); the variable set stays project-global."""
    combos = _opt(opts, "combinations", []) or []
    return {
        "combinations": [(str(a), str(b)) for a, b in combos],
        "model": _opt(opts, "model", "auto"),
        "variables": list(_opt(opts, "variables", met.DEFAULT_VARIABLES)),
        "regrid_radius_m": float(_opt(opts, "regrid_radius_m", met.DEFAULT_RADIUS_M)),
        "pad_deg": float(_opt(opts, "pad_deg", met.DEFAULT_PAD_DEG)),
        "fxx": int(_opt(opts, "fxx", met.DEFAULT_FXX)),
        "product": _opt(opts, "product", met.DEFAULT_PRODUCT),
        "era5_zarr": _opt(opts, "era5_zarr", met.ARCO_ERA5_URI),
    }


def _build_eff(project: Project) -> dict:
    opts = project.products.get(DataProduct.met_overpass)
    if opts is None:
        raise ValueError("met_overpass is not a selected product in this config")

    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)
    root = Path(project.output_dir)
    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.met_overpass))
               for a in project.all_areas},
        "grid": grid_cfg,
        "root": root,
        "mo_root": root / "MET_OVERPASS",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None) -> None:
    """Acquire overpass-aligned met for a validated Project. Entry point for pipeline.py.

    Runs AFTER the sensor stages (it reads their overpass times) and reuses met's fetchers.
    Every combo's source is validated against met's SOURCES, so a typo fails loudly.
    """
    # Reuses met's fetchers, so it needs met's network policy too (see met.acquire).
    net.setup_requests_timeout()
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    bad = sorted({s for c in eff["ds"].values() for _, s in c["combinations"]
                  if s not in met._SOURCES})
    if bad:
        raise ValueError(f"met_overpass source(s) not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(met._SOURCES)}.")
    return run(eff, grids, aois, dry_run, only_source=source)


def main():
    entry.process_main(
        acquire, "coastal_sst_data overpass-aligned met acquisition (per (sensor, source)).")


if __name__ == "__main__":
    main()
