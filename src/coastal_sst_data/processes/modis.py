#!/usr/bin/env python3
"""
coastal_sst_data -- MODIS Terra L2P Sea-Surface Temperature acquisition.

Loads GHRSST MODIS Terra L2P skin SST (NASA OB.DAAC via earthaccess) and grids it
onto the shared AoiGrid. MODIS L2P is a SWATH product (2D curvilinear lat/lon),
so it is regridded with pyresample nearest-neighbour (`resample_nearest`) -- NOT
rioxarray reproject. Nearest is deliberate: it preserves the actual observed
MODIS values (block-constant on the fine grid) so a DOWNSTREAM calibration module
can match Landsat to MODIS faithfully. Calibration itself is out of scope here --
this module only LOADS and grids the data.

Coincidence with Landsat (configurable):
  * match_landsat: true (default) -- only load MODIS granules within
    `max_time_diff_minutes` of an already-acquired Landsat scene (read from
    <output_dir>/LANDSAT/aligned/<aoi>/). Requires Landsat to have run first.
  * match_landsat: false -- load the full MODIS time series over the date range.

Access backends (configurable via `access`):
  * download (default) -- earthaccess.download the full granule, crop in memory.
  * harmony  -- server-side AOI subsetting (NOT yet implemented; documented next step).

Output: one aligned NetCDF per granule at
    <output_dir>/MODIS/aligned/<aoi>/<aoi>_<YYYYMMDDTHHMMSS>.nc
Variables: sst (K|degC), valid (uint8), and (optional) footprint_id -- the MODIS
swath pixel index each grid cell was drawn from, for exact footprint-median
matchups downstream.

Usage:
    python -m coastal_sst_data.processes.modis --config config.yaml
    python -m coastal_sst_data.processes.modis --config config.yaml --dry-run
    python -m coastal_sst_data.processes.modis --config config.yaml --full-series
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import earthaccess
import rioxarray  # noqa: F401  (registers the .rio accessor)

from ..config import Project, DataProduct, load_config
from ..grid import AoiGrid, project_grids
from .. import provenance, store

log = logging.getLogger(__name__)

# --- MODIS L2P product constants (overridable via the modis product options) - #
SOURCE = "modis"
SHORT_NAME = "MODIS_T-JPL-L2P-v2019.0"       # MODIS Terra L2P skin SST (GHRSST)
DEFAULT_VARIABLE = "sea_surface_temperature"  # Kelvin (xarray auto-unscales)
QUALITY_VAR = "quality_level"                 # GHRSST: >=4 acceptable, 5 best
DEFAULT_QUALITY_MIN = 4
DEFAULT_RADIUS_M = 1500.0                      # pyresample search radius (~1 km px)
DEFAULT_MAX_TIME_DIFF_MIN = 360               # +/- 6 h Landsat<->MODIS overpass

_DT_RE = re.compile(r"(\d{8}T\d{6})")          # Landsat aligned filename stamp


# --------------------------------------------------------------------------- #
# Coincidence helpers
# --------------------------------------------------------------------------- #
def _landsat_times(landsat_dir: Path, aoi: str) -> list[datetime]:
    """Acquisition times of the Landsat aligned files already written for an AoI."""
    d = landsat_dir / aoi
    out = []
    if d.exists():
        for f in d.glob(f"{aoi}_*T*.nc"):
            m = _DT_RE.search(f.name)
            if m:
                out.append(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
    return out


def _granule_time(granule) -> datetime:
    t = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")


def _is_night(granule) -> bool:
    return "-N-" in granule["meta"].get("native-id", "")


# --------------------------------------------------------------------------- #
# Fetch backends: granule -> local NetCDF path. Swappable via `access`.
# --------------------------------------------------------------------------- #
def _fetch_download(granule, bbox, tmp_dir: Path) -> Path:
    """Download the full granule to disk (cropped in memory afterwards)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download([granule], local_path=str(tmp_dir))
    return Path(paths[0])


def _fetch_harmony(granule, bbox, tmp_dir: Path) -> Path:
    """Server-side AOI subset via Harmony (documented next step)."""
    raise NotImplementedError(
        "modis access='harmony' is not implemented yet; use access='download'.")


_ACCESS = {"download": _fetch_download, "harmony": _fetch_harmony}


# --------------------------------------------------------------------------- #
# Swath read + resample onto the shared grid
# --------------------------------------------------------------------------- #
def read_swath(path, variable, quality_min, to_celsius):
    """Read a MODIS L2P granule -> (sst, lat, lon) 2D swath arrays (NaN where bad)."""
    ds = xr.open_dataset(path, engine="netcdf4")
    sst = ds[variable].values.squeeze().astype("float32")     # Kelvin
    qual = ds[QUALITY_VAR].values.squeeze().astype("int16")
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()
    sst[(qual < quality_min) | ~np.isfinite(sst)] = np.nan
    if to_celsius:
        sst = sst - 273.15
    return sst, lat, lon


def resample_to_grid(sst, lat, lon, g: AoiGrid, radius_m, footprint=None):
    """Nearest-neighbour resample the swath onto the AoI grid (sst, and footprint)."""
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest

    lon = ((lon + 180.0) % 360.0) - 180.0                      # normalize longitudes
    swath = SwathDefinition(lons=lon, lats=lat)
    area = g.to_area_def()
    sst_g = resample_nearest(swath, sst, area, radius_of_influence=radius_m,
                             fill_value=np.nan)
    fp_g = None
    if footprint is not None:
        fp_g = resample_nearest(swath, footprint, area, radius_of_influence=radius_m,
                                fill_value=-1)
    return sst_g, fp_g


def _scene_dataset(sst_g, fp_g, g: AoiGrid, acq_time, aoi_id, to_celsius,
                   short_name=SHORT_NAME) -> xr.Dataset:
    xs, ys = g.xy_centers()
    data = {
        "sst": (("y", "x"), sst_g.astype("float32")),
        "valid": (("y", "x"), np.isfinite(sst_g).astype("uint8")),
    }
    if fp_g is not None:
        data["footprint_id"] = (("y", "x"), fp_g.astype("int32"))
    ds = xr.Dataset(data, coords={"y": ys, "x": xs})
    ds["sst"].attrs["units"] = "degC" if to_celsius else "K"
    ds["valid"].attrs["long_name"] = "finite, quality-good MODIS SST"
    if fp_g is not None:
        ds["footprint_id"].attrs["long_name"] = (
            "MODIS swath pixel index (for footprint-median matchups); -1 = none")
    ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    ds = ds.rio.write_crs(g.target_crs)
    # The short_name we actually SEARCHED, not the module default: configure Aqua and the
    # old constant still stamped every file "Terra", so the cube misnamed its own sensor.
    ds.attrs.update(aoi_id=aoi_id, source=f"GHRSST {short_name}",
                    processing="swath -> nearest resample onto AoI grid")
    return ds


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.Timestamp(ds["time"].values[0]).strftime("%Y%m%dT%H%M%S")
    stem = f"{aoi_id}_{t}"
    if fmt == "netcdf":
        return store.write_netcdf(ds, out_dir / f"{stem}.nc")
    if fmt == "geotiff":
        return store.write_rasters(ds, out_dir / stem,
                                   [(v, ds[v].isel(time=0)) for v in ds.data_vars])
    raise ValueError(f"Unknown output format: {fmt}")


# --------------------------------------------------------------------------- #
# Coincidence filter (day/time + Landsat matchup)
# --------------------------------------------------------------------------- #
def _select_granules(granules, ls_times, match_landsat, max_dt, daytime_only):
    """Filter granules to daytime and (optionally) within max_dt of a Landsat scene.

    Returns [(granule, acq_time), ...] sorted by time.
    """
    kept = []
    for gr in granules:
        if daytime_only and _is_night(gr):
            continue
        t = _granule_time(gr)
        if match_landsat:
            nearest = min((abs(t - lt) for lt in ls_times), default=None)
            if nearest is None or nearest > max_dt:
                continue
        kept.append((gr, t))
    return sorted(kept, key=lambda x: x[1])


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, tmp_dir, fmt, overwrite = eff["out_dir"], eff["tmp_dir"], eff["fmt"], eff["overwrite"]
    landsat_dir = eff["landsat_dir"]
    to_celsius = grid_cfg.get("to_celsius", False)
    variable, quality_min = ds_cfg["variable"], ds_cfg["quality_min"]
    radius, access = ds_cfg["regrid_radius_m"], ds_cfg["access"]
    match_landsat = ds_cfg["match_landsat"]
    max_dt = timedelta(minutes=ds_cfg["max_time_diff_minutes"])
    daytime_only, do_footprint = ds_cfg["daytime_only"], ds_cfg["footprint_id"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]
    fetch = _ACCESS[access]

    log.info("Authenticating with Earthdata (strategy=%s)", eff["earthdata"]["auth_strategy"])
    earthaccess.login(strategy=eff["earthdata"]["auth_strategy"])

    names = list(grids)
    if only_aoi:
        req = set(only_aoi)
        missing = req - set(names)
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        names = [n for n in names if n in req]

    for name in names:
        g = grids[name]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d) | match_landsat=%s access=%s ===",
                 name, g.target_crs, g.width, g.height, match_landsat, access)

        ls_times = _landsat_times(landsat_dir, name) if match_landsat else []
        if match_landsat and not ls_times:
            log.warning("  %s: match_landsat on but no Landsat aligned files in %s; "
                        "run Landsat first (or set match_landsat: false)", name, landsat_dir / name)
            continue

        granules = earthaccess.search_data(
            short_name=ds_cfg["short_name"], temporal=(start, end),
            bounding_box=tuple(g.search_bbox))
        kept = _select_granules(granules, ls_times, match_landsat, max_dt, daytime_only)
        log.info("  %d granule(s) over AOI -> %d after daytime/coincidence filter",
                 len(granules), len(kept))
        if not kept:
            continue
        if dry_run:
            log.info("  [dry-run] would process %d MODIS granule(s)", len(kept))
            continue

        aoi_out = out_root / name
        for gr, t in kept:
            tstr = t.strftime("%Y%m%dT%H%M%S")
            if store.done(aoi_out / f"{name}_{tstr}.nc", store.REQUIRED_VARS["MODIS"],
                          shape=(g.height, g.width), overwrite=overwrite):
                log.info("  %s already processed, skipping", tstr)
                continue
            try:
                path = fetch(gr, g.search_bbox, tmp_dir)
                sst, lat, lon = read_swath(path, variable, quality_min, to_celsius)
                fp = np.arange(sst.size, dtype="int32").reshape(sst.shape) if do_footprint else None
                sst_g, fp_g = resample_to_grid(sst, lat, lon, g, radius, fp)
                Path(path).unlink(missing_ok=True)
            except Exception as exc:
                log.warning("  skipping %s (%s)", tstr, exc)
                continue
            if not np.isfinite(sst_g).any():
                log.info("  %s: no valid MODIS pixels over AOI, skipping", tstr)
                continue
            ds = _scene_dataset(sst_g, fp_g, g, t, name, to_celsius,
                                short_name=ds_cfg["short_name"])
            ds.attrs.update(**provenance.stamp(eff))
            log.info("  wrote %s", write_output(ds, aoi_out, name, fmt))
    log.info("Done.")


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _opt(opts, name, default):
    """Read an optional override off a product-options bag (extra='allow')."""
    return getattr(opts, name, default) if opts is not None else default


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.modis)
    if opts is None:
        raise ValueError("modis is not a selected product in this config")
    if project.auth.earthdata is None:            # guaranteed by config validation
        raise ValueError("modis requires an auth.earthdata block")

    access = _opt(opts, "access", "download")
    if access not in _ACCESS:
        raise ValueError(f"modis access {access!r} not recognized; "
                         f"choose from {sorted(_ACCESS)}.")

    ds_cfg = {
        "short_name": _opt(opts, "short_name", SHORT_NAME),
        "variable": _opt(opts, "variable", DEFAULT_VARIABLE),
        "quality_min": int(_opt(opts, "quality_min", DEFAULT_QUALITY_MIN)),
        "regrid_radius_m": float(_opt(opts, "regrid_radius_m", DEFAULT_RADIUS_M)),
        "access": access,
        "match_landsat": bool(_opt(opts, "match_landsat", True)),
        "max_time_diff_minutes": float(_opt(opts, "max_time_diff_minutes", DEFAULT_MAX_TIME_DIFF_MIN)),
        "daytime_only": bool(_opt(opts, "daytime_only", True)),
        "footprint_id": bool(_opt(opts, "footprint_id", True)),
    }
    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    root = Path(project.output_dir)
    return {
        "config_sha256": project.config_sha256,
        "ds": ds_cfg,
        "grid": grid_cfg,
        "out_dir": root / "MODIS" / "aligned",
        "landsat_dir": root / "LANDSAT" / "aligned",   # coincidence source
        "tmp_dir": root / "MODIS" / "_tmp",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "earthdata": {"auth_strategy": project.auth.earthdata.auth_strategy},
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, full_series=False) -> None:
    """Acquire MODIS for a validated Project. Entry point for pipeline.py.

    full_series=True forces match_landsat off (load the whole time series). MODIS
    with match_landsat should run AFTER Landsat so its aligned files exist.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if full_series:
        eff["ds"]["match_landsat"] = False
    if grids is None:
        grids = project_grids(project)
    run(eff, grids, aois, dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="coastal_sst_data MODIS Terra L2P SST acquisition.")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Process only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess granules even if the aligned file exists")
    ap.add_argument("--full-series", action="store_true",
                    help="ignore Landsat coincidence; load the full MODIS time series")
    ap.add_argument("--dry-run", action="store_true", help="Search only; no download.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    project = load_config(args.config)
    acquire(project, aois=args.aoi, dry_run=args.dry_run,
            overwrite=args.overwrite, full_series=args.full_series)


if __name__ == "__main__":
    main()
