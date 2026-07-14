#!/usr/bin/env python3
"""
OCEANSR — ECOSTRESS L2T LSTE (V3) acquisition.

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; ECOSTRESS-specific settings come from `sources.ecostress`.
For each AOI it searches ECO_L2T_LSTE by bbox + dates, then for every overpass
*streams a windowed read* of the SST + cloud/water/QC COGs (HTTP range requests
via earthaccess.open -- no full-tile download to disk), reprojects the AOI window
onto the AOI grid, clips to the AOI polygon, and writes one aligned file per
overpass into data/ECOSTRESS/aligned/. A later stage bins these to a daily cube.

Because only the AOI window is read and only the small aligned outputs are kept,
there is no multi-GB raw tile cache.

Usage (run from the OCEANSR project root):
    python src/acquire_ecostress.py --config configs/config.yaml
    python src/acquire_ecostress.py --config configs/config.yaml --dry-run
    python src/acquire_ecostress.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_ecostress.py --config configs/config.yaml --list-layers
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

import earthaccess
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, read_cog_window, select_aois
from .. import entry, naming, net, provenance, report, store

log = logging.getLogger(__name__)

# --- ECOSTRESS product constants -------------------------------------------- #
# These describe the ECO_L2T_LSTE product itself (not user choices), so they
# live here as defaults. Any of them can be overridden per-config via the
# ecostress product options block.
SOURCE = "ecostress"
SHORT_NAME = "ECO_L2T_LSTE"
DEFAULT_VERSION = "002"
# role -> COG asset suffix (..._<suffix>.tif). sst uses the LST band (LST ~= SST
# over water); cloud/water/quality are categorical masks.
LAYERS = {"sst": "LST", "lst": "LST", "cloud": "cloud", "water": "water", "quality": "QC"}
CATEGORICAL = ["cloud", "water", "quality"]
# The layers a granule CANNOT be written without: they are what `valid` is built from, and
# a granule lacking one reads downstream as a scene with nothing valid in it -- a silent
# total cloud-out rather than a visible failure. (`lst`/`quality` are optional extras.)
CORE_ROLES = ("sst", "water", "cloud")


def expected_vars(ds_cfg: dict) -> tuple[str, ...]:
    """The variables a COMPLETE aligned granule carries under this config's `layers`.

    Derived from config rather than hardcoded, because `layers` is overridable: demanding a
    layer the user never asked for would make every granule look incomplete and re-fetch on
    every run, forever.
    """
    roles = set(ds_cfg.get("layers", LAYERS))
    core = tuple(r for r in CORE_ROLES if r in roles)
    return core + (("valid",) if set(CORE_ROLES) <= roles else ())


# The per-AoI target CRS + grid now come from coastal_sst_data.grid (the single
# source of truth shared by every product), passed in as AoiGrid objects.


# --------------------------------------------------------------------------- #
# Earthdata search / download
# --------------------------------------------------------------------------- #
def login(strategy: str):
    log.info("Authenticating with Earthdata (strategy=%s)", strategy)
    earthaccess.login(strategy=strategy)


def search_granules(ds_cfg: dict, bbox, start: str, end: str):
    results = net.retry(
        lambda: earthaccess.search_data(
            short_name=ds_cfg["short_name"],
            version=ds_cfg["version"],
            temporal=(start, end),
            bounding_box=tuple(bbox),
        ),
        what=f"ECOSTRESS search {ds_cfg['short_name']}")
    log.info("  found %d granule(s)", len(results))
    return results


def granule_name(granule) -> str:
    """Clean granule id (strips the trailing _<LAYER>.tif), e.g.
    ECOv002_L2T_LSTE_25520_009_10UEU_20230105T123623_0710_01."""
    try:
        fn = granule.data_links()[0].split("/")[-1]
        return re.sub(r"_[A-Za-z0-9]+\.tif$", "", fn)
    except Exception:
        return "<granule>"


def filter_links_for_granule(granule, layers: dict) -> dict:
    """{role: url} for the COG assets we want from one granule."""
    links = granule.data_links()
    out, missing = {}, []
    for role, suffix in layers.items():
        match = next((u for u in links if u.endswith(f"_{suffix}.tif")), None)
        if match:
            out[role] = match
        else:
            missing.append(f"{role} (_{suffix}.tif)")
    if missing:
        log.warning("    %s: requested layer(s) not in granule: %s",
                    granule_name(granule), ", ".join(missing))
    return out


def parse_acq_time(filename: str) -> Optional[datetime]:
    """The overpass stamp in a granule id (..._20230715T210043_...), or None."""
    return naming.parse_time(filename)


# --------------------------------------------------------------------------- #
# Per-granule processing
# --------------------------------------------------------------------------- #
# ECOSTRESS COGs are read through the shared windowed reader (grid.read_cog_window). Its
# 110 km tiles want a slightly wider pad than the 300 m the optical products use -- ~6
# pixels of slack at the 70 m native posting -- so that is passed explicitly.
ECO_PAD_M = 600.0


def process_granule(role_to_file, ds_cfg, grid_cfg, g: AoiGrid, aoi_id,
                    acq_time) -> Optional[xr.Dataset]:
    """One granule's COG assets -> the aligned sst/water/cloud/valid Dataset, or None.

    `role_to_file` maps a role to anything rioxarray opens: the fsspec objects
    earthaccess.open() returns for the remote assets, or plain paths in a test.
    """
    categorical = set(ds_cfg.get("categorical", []))
    rs_cont = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    rs_cat = Resampling[grid_cfg.get("resampling_categorical", "nearest")]

    data_vars, failed = {}, []
    for role, fobj in role_to_file.items():
        resampling = rs_cat if ds_cfg["layers"][role] in categorical else rs_cont
        try:
            da = read_cog_window(fobj, g, resampling=resampling, pad_m=ECO_PAD_M)
            da = da.rio.clip([g.geom_proj], g.target_crs, drop=False)
        except Exception as exc:  # window outside this tile, transient COG read failure...
            log.warning("    layer %s failed to read (%s)", role, exc)
            failed.append(role)
            continue
        data_vars[role] = da

    if "sst" not in data_vars and "lst" not in data_vars:
        log.warning("    no SST/LST after processing; dropping granule")
        return None

    # A granule missing a CORE layer is degraded, not merely partial -- and a degraded
    # granule is worse than none at all. Without `water`/`cloud` the `valid` mask below is
    # never built, and the assembler reads a missing water layer as "claim nothing", so the
    # scene silently becomes an all-zero valid mask: a total cloud-out, indistinguishable
    # from a genuinely overcast day. Drop it and let a later run re-fetch it.
    #
    # This catches BOTH ways a layer goes missing: a read that failed (`failed`, e.g. a
    # transient COG error) and an asset the granule never published (dropped upstream by
    # filter_links_for_granule). Only layers this config actually ASKED for are required,
    # so a custom `layers` list cannot make every granule look degraded.
    missing = [r for r in CORE_ROLES if r in ds_cfg["layers"] and r not in data_vars]
    if missing:
        log.warning("    dropping granule: core layer(s) %s missing (%s); a granule with "
                    "no mask reads as a total cloud-out downstream",
                    ", ".join(missing),
                    "read failed" if failed else "not published by the granule")
        return None

    ds = xr.Dataset(data_vars)

    if grid_cfg.get("to_celsius", False):
        for v in ("sst", "lst"):
            if v in ds:
                ds[v] = ds[v] - 273.15
                ds[v].attrs["units"] = "degC"
    else:
        for v in ("sst", "lst"):
            if v in ds:
                ds[v].attrs["units"] = "K"

    if {"sst", "water", "cloud"} <= set(ds.data_vars):
        valid = np.isfinite(ds["sst"]) & (ds["water"] > 0) & ~(ds["cloud"] > 0)
        ds["valid"] = valid.astype("uint8")
        ds["valid"].attrs["long_name"] = "water & clear & finite SST"

    if acq_time is not None:
        ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    # Built from what we ACTUALLY searched, never a literal: this attr is what
    # provenance.source_of() reads and what every eco_* field in every cube is stamped
    # with, so a hardcoded string here is a lie the cube then repeats. (It used to say
    # "v003" while DEFAULT_VERSION -- and the search -- was "002".)
    ds.attrs.update(aoi_id=aoi_id,
                    source=f"ECOSTRESS {ds_cfg['short_name']} v{ds_cfg['version']}",
                    processing="reprojected+clipped to AOI grid")
    return ds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, list_layers):
    """Acquire ECOSTRESS onto the pre-computed per-AoI grids.

    `grids` (from coastal_sst_data.grid) is the single source of truth for the
    CRS + target grid, so ECOSTRESS lands on exactly the same grid as every
    other product for the AoI.
    """
    grid_cfg = eff["grid"]
    out_root = eff["out_dir"]
    fmt, overwrite = eff["fmt"], eff["overwrite"]

    login(eff["earthdata"]["auth_strategy"])

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("ecostress")

    for name in names:
        g = grids[name]
        # ECOSTRESS is a GLOBAL product, so it has no region-varying options -- but its
        # settings are still resolved per AoI, so every product answers to one contract.
        ds_cfg = eff["ds"][name]
        layers = dict(ds_cfg["layers"])
        log.info("=== AOI: %s (CRS=%s grid=%dx%d @ %.0fm) ===",
                 name, g.target_crs, g.width, g.height, g.resolution_m)

        granules = search_granules(ds_cfg, g.search_bbox, eff["time"]["start_date"],
                                   eff["time"]["end_date"])
        if not granules:
            continue

        if list_layers:
            links = granules[0].data_links()
            tails = sorted({u.rsplit("_", 1)[-1] for u in links if u.endswith(".tif")})
            log.info("  available COG suffixes: %s", tails)
            continue
        if dry_run:
            log.info("  [dry-run] would process %d granule(s)", len(granules))
            continue

        aoi_out = out_root / name

        for gi, granule in enumerate(granules, 1):
            role_to_url = filter_links_for_granule(granule, layers)
            if not role_to_url:
                continue
            t = parse_acq_time(granule_name(granule))
            tstr = naming.time_stamp(t) if t else f"g{gi}"
            stem = f"{name}_{tstr}"
            if store.done(aoi_out / f"{stem}.nc", expected_vars(ds_cfg),
                          shape=(g.height, g.width), overwrite=overwrite):
                log.info("  [%d/%d] %s already processed, skipping", gi, len(granules), tstr)
                rep.skip()
                continue

            log.info("  [%d/%d] streaming %d layer(s) for %s",
                     gi, len(granules), len(role_to_url), tstr)
            try:
                fobjs = net.retry(lambda: earthaccess.open(list(role_to_url.values())),
                                  what=f"ECOSTRESS open {tstr}")
            except Exception as exc:
                log.warning("    FAILED to open %s (%s)", tstr, exc)
                rep.fail(f"{name} {tstr}", exc)
                continue
            role_to_file = dict(zip(role_to_url.keys(), fobjs))

            ds = process_granule(role_to_file, ds_cfg, grid_cfg, g, name, t)
            if ds is None:
                # dropped as degraded (a mask layer was missing) -- a LOSS, not a no-op
                rep.fail(f"{name} {tstr}", "granule dropped (missing core layer)")
                continue
            ds.attrs.update(**provenance.stamp(eff))
            log.info("      wrote %s", store.write_output(ds, aoi_out, stem, fmt))
            rep.wrote(source=ds.attrs.get("source"))

    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's ECOSTRESS settings. ECOSTRESS is global -- nothing is region-overridable."""
    return {
        "short_name": _opt(opts, "short_name", SHORT_NAME),
        "version": str(_opt(opts, "version", DEFAULT_VERSION)),
        "layers": dict(_opt(opts, "layers", LAYERS)),
        "categorical": list(_opt(opts, "categorical", CATEGORICAL)),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes.

    ECOSTRESS product constants (short_name, layers, ...) come from module
    defaults; the config's ecostress product options may override any of them.
    Grid, time, auth and output_dir come from the shared project settings.
    """
    opts = project.products.get(DataProduct.ecostress)
    if opts is None:
        raise ValueError("ecostress is not a selected product in this config")
    if project.auth.earthdata is None:            # guaranteed by config validation
        raise ValueError("ecostress requires an auth.earthdata block")

    # Only the resampling / to_celsius knobs matter here; the CRS, resolution and
    # snapping are applied by coastal_sst_data.grid, not by this process.
    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.ecostress))
               for a in project.all_areas},
        "grid": grid_cfg,
        "out_dir": Path(project.output_dir) / "ECOSTRESS" / "aligned",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "earthdata": {"auth_strategy": project.auth.earthdata.auth_strategy},
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, list_layers=False) -> None:
    """Acquire ECOSTRESS for a validated Project. Entry point for pipeline.py.

    Parameters
    ----------
    project      validated config (coastal_sst_data.config.Project)
    grids        pre-computed {aoi_name: AoiGrid} from coastal_sst_data.grid;
                 if None, they are computed here. The pipeline computes them once
                 and passes the same dict to every product so all grids match.
    aois         restrict to these AoI name(s); default all
    dry_run      search only, no download/write
    overwrite    reprocess overpasses even if the aligned file exists
    list_layers  print available COG suffixes for the first granule and exit
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    net.setup_gdal_env()      # windowed COG reads: deadline + retries
    return run(eff, grids, aois, dry_run, list_layers)


def main():
    entry.process_main(
        acquire, "coastal_sst_data ECOSTRESS acquisition.",
        extra=[entry.Flag("--list-layers",
                          "Print available COG suffixes for the first granule and exit.")])


if __name__ == "__main__":
    main()