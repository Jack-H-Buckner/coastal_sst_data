#!/usr/bin/env python3
"""
coastal_sst_data -- meteorological FORCING acquisition (HRRR + ERA5, STACKED per source).

Reads the validated project config (coastal_sst_data.config.Project) and the shared per-AoI
grids (coastal_sst_data.grid.AoiGrid), exactly like the SST stages. For each AoI it pulls
2 m air temperature, 10 m wind (u/v + speed), downward shortwave and total cloud cover,
regrids onto the AoI grid, and writes, PER SOURCE:

  * a REFERENCE-time file per day: <output_dir>/MET/<source>/aligned/<aoi>/<aoi>_ref_<YYYYMMDD>.nc
  * a daily-mean file per day:     <output_dir>/MET/<source>/aligned/<aoi>/<aoi>_<YYYYMMDD>.nc

The REFERENCE file is one snapshot per day at a fixed time of day -- by default 10:30 LOCAL
SOLAR time, Landsat's overpass. It is the cube's default met channel: a daily mean smears the
diurnal cycle, which is the wrong forcing to hand a model of a sensor that flew at one instant.
The basis is solar rather than UTC because a fixed UTC hour is a different time of day in every
AoI, so cross-AoI comparisons would not be like-for-like. Set `reference_time: null` to skip.

This is FORCING only (D14) -- NO sensor dependency. Met matched to each thermal sensor's
overpass instant is the SEPARATE `met_overpass` product (processes.met_overpass), which reuses
the HRRR/ERA5 fetchers here.

DISTINCT-DATA sources, STACKED one channel per source (D10) -- NOT a chain/fallback: the cube
emits `<var>_<source>` (`airtemp_hrrr`, `airtemp_era5`). A source with no data here (HRRR is
North America only) simply contributes NaN to ITS channel; the two are never conflated (they
differ in resolution, and `swrad` is instantaneous under HRRR but an hourly mean in ERA5).

  * hrrr -- NOAA HRRR 3 km surface fields via Herbie. CONUS uses the 'hrrr' domain; AoIs above
    ~50N use the Alaska domain 'hrrrak' (auto by latitude). Curvilinear -> pyresample nearest.
  * era5 -- ECMWF ERA5 0.25 deg hourly reanalysis, streamed from Google's public ARCO-ERA5 Zarr
    on GCS (no auth). Global. Regular lat/lon grid -> rioxarray bilinear reproject.

Both are unit-harmonized (airtemp K, swrad W m-2, cloud_cover %); each file records its source.

Usage:
    python -m coastal_sst_data.processes.met --config config.yaml
    python -m coastal_sst_data.processes.met --config config.yaml --aoi hood_canal --dry-run
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, naming, provenance, report, store

log = logging.getLogger(__name__)

SOURCE = "met"

# --- met product defaults (overridable via the met product options block) ---- #
DEFAULT_SOURCES = ["hrrr", "era5"]        # STACKED (D10): hrrr in N. America, era5 global
DEFAULT_VARIABLES = ["airtemp", "wind", "swrad", "cloud"]
DEFAULT_MEAN_HOURS = [0, 6, 12, 18]       # UTC hours averaged for the daily field ([] = skip)
# The daily REFERENCE time: one snapshot per day, the same time of day everywhere, so a
# cross-AoI comparison is like-for-like. The default is Landsat's ~10:30 local overpass,
# which is why the basis is LOCAL SOLAR time (converted per AoI from its longitude) --
# a fixed UTC hour would be a different time of day in every AoI, which is exactly what
# a "reference time of day" must not be.
DEFAULT_REFERENCE_TIME = "10:30"          # HH:MM; None/"" to skip the reference snapshot
DEFAULT_REFERENCE_BASIS = "solar"         # solar (local, per-AoI longitude) | utc
DEFAULT_RADIUS_M = 6000.0                 # pyresample search radius (HRRR ~3 km px)
DEFAULT_PAD_DEG = 0.25                    # ERA5 window pad (>= one 0.25 deg cell)
DEFAULT_FXX = 0                           # HRRR forecast hour (0 = analysis)
DEFAULT_PRODUCT = "sfc"                   # HRRR 2D surface fields
# Google Analysis-Ready Cloud-Optimized ERA5 (public, no auth).
ARCO_ERA5_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Common output var name -> units (after harmonization). Shared by both sources.
OUT_UNITS = {"airtemp": "K", "swrad": "W m-2", "cloud_cover": "%", "wind_speed": "m s-1"}

# HRRR: config var key -> (Herbie search regex, {cfgrib var name: output name}).
HRRR_VAR_SEARCH = {
    "airtemp": (r"TMP:2 m above ground", {"t2m": "airtemp"}),
    "wind":    (r":(U|V)GRD:10 m above ground", {"u10": "wind_u", "v10": "wind_v"}),
    "swrad":   (r"DSWRF:surface", {"dswrf": "swrad"}),
    "cloud":   (r"TCDC:entire atmosphere", {"tcc": "cloud_cover"}),
}

# ERA5 (ARCO): config var key -> {ARCO variable name: output name}.
ERA5_VARS = {
    "airtemp": {"2m_temperature": "airtemp"},
    "wind":    {"10m_u_component_of_wind": "wind_u", "10m_v_component_of_wind": "wind_v"},
    "swrad":   {"surface_solar_radiation_downwards": "swrad"},
    "cloud":   {"total_cloud_cover": "cloud_cover"},
}

# Coarse North America bounds (S, N, W, E) in EPSG:4326. HRRR (CONUS + Alaska)
# covers North America only; an AoI outside this box (e.g. Europe) is skipped so
# it falls through to ERA5 without a wasted Herbie fetch. Within it, the CONUS
# 'hrrr' vs Alaska 'hrrrak' domain is chosen by latitude (>= 50N -> Alaska).
_HRRR_NA_BOX = (18.0, 72.0, -179.0, -52.0)


# --------------------------------------------------------------------------- #
# HRRR source (curvilinear grid -> pyresample nearest)
# --------------------------------------------------------------------------- #
def _hrrr_model_for_grid(g: AoiGrid, configured: str) -> str | None:
    """HRRR domain covering this AoI: 'hrrr', 'hrrrak', or None (uncovered)."""
    if configured and configured != "auto":
        return configured
    w, s, e, n = g.search_bbox
    lat, lon = (s + n) / 2.0, (w + e) / 2.0
    bs, bn, bw, be = _HRRR_NA_BOX
    if not (bs <= lat <= bn and bw <= lon <= be):
        return None                                  # off-continent -> ERA5
    return "hrrrak" if lat >= 50.0 else "hrrr"


def _snap_to_cycle(dt: datetime, model: str) -> datetime:
    """Round to the nearest available analysis cycle (1h hrrr, 3h hrrrak)."""
    step = 3 if model == "hrrrak" else 1
    h = int(round(dt.hour / step) * step)
    base = dt.replace(minute=0, second=0, microsecond=0, hour=0)
    return base + timedelta(hours=min(h, 24 - step))


def _hrrr_fetch_cycle(model, dt, fxx, product, var_keys):
    """Fetch one HRRR cycle -> ({outname: 2D array}, lon2d, lat2d) or None."""
    from herbie import Herbie
    H = Herbie(dt.strftime("%Y-%m-%d %H:00"), model=model, product=product, fxx=fxx)
    fields, lon2d, lat2d = {}, None, None
    for key in var_keys:
        search, rename = HRRR_VAR_SEARCH[key]
        ds = H.xarray(search, remove_grib=True)
        if isinstance(ds, list):
            ds = xr.merge(ds, compat="override")
        matched = False
        for src, dst in rename.items():
            if src in ds:
                fields[dst] = np.asarray(ds[src].values)
                matched = True
        # single-field searches (e.g. TCDC) can come back under a different
        # cfgrib shortName -> fall back to the lone data variable.
        if not matched and len(rename) == 1:
            dvs = list(ds.data_vars)
            if len(dvs) == 1:
                fields[next(iter(rename.values()))] = np.asarray(ds[dvs[0]].values)
        if lon2d is None and "longitude" in ds.coords:
            lon2d = np.asarray(ds.longitude.values)
            lat2d = np.asarray(ds.latitude.values)
    if not fields or lon2d is None:
        return None
    lon2d = ((lon2d + 180.0) % 360.0) - 180.0        # 0..360 -> -180..180
    return fields, lon2d, lat2d


def _regrid_nearest(fields, lon2d, lat2d, g: AoiGrid, radius_m) -> dict:
    """Nearest-neighbour resample curvilinear fields onto the AoI grid."""
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest
    swath = SwathDefinition(lons=lon2d, lats=lat2d)
    area = g.to_area_def()
    return {name: resample_nearest(swath, arr, area, radius_of_influence=radius_m,
                                   fill_value=np.nan)
            for name, arr in fields.items()}


def _fetch_hrrr(g: AoiGrid, dt: datetime, cfg: dict) -> dict | None:
    """Fetch + regrid HRRR for one timestamp -> {outname: 2D array} or None."""
    model = _hrrr_model_for_grid(g, cfg.get("model", "auto"))
    if model is None:
        return None                                  # AoI outside HRRR coverage
    cyc = _snap_to_cycle(dt, model)
    got = _hrrr_fetch_cycle(model, cyc, cfg["fxx"], cfg["product"], cfg["variables"])
    if got is None:
        return None
    fields, lon2d, lat2d = got
    return _regrid_nearest(fields, lon2d, lat2d, g, cfg["regrid_radius_m"])


# --------------------------------------------------------------------------- #
# ERA5 source (ARCO Zarr on GCS -> rioxarray bilinear reproject)
# --------------------------------------------------------------------------- #
_ERA5_CACHE: dict[str, xr.Dataset] = {}


def _era5_store(uri: str) -> xr.Dataset:
    """Lazily open (and cache) the ARCO-ERA5 Zarr store; metadata only."""
    if uri not in _ERA5_CACHE:
        _ERA5_CACHE[uri] = xr.open_zarr(uri, chunks="auto",
                                        storage_options={"token": "anon"})
    return _ERA5_CACHE[uri]


def _era5_normalize(fields: dict) -> dict:
    """Harmonize ERA5 fields to the common convention (in place, returns dict).

    ERA5 differs from HRRR in two channels: shortwave is an hourly accumulation
    in J m-2 (-> W m-2 by /3600) and cloud cover is a 0-1 fraction (-> % by x100).
    2 m temperature and wind are already in the common units.
    """
    if "swrad" in fields:
        fields["swrad"] = fields["swrad"] / 3600.0           # J m-2 (1h) -> W m-2
    if "cloud_cover" in fields:
        fields["cloud_cover"] = fields["cloud_cover"] * 100.0  # fraction -> %
    return fields


def _fetch_era5(g: AoiGrid, dt: datetime, cfg: dict) -> dict | None:
    """Fetch + regrid ERA5 for one timestamp -> {outname: 2D array} or None."""
    from rasterio.enums import Resampling

    var_map: dict[str, str] = {}
    for key in cfg["variables"]:
        var_map.update(ERA5_VARS.get(key, {}))
    if not var_map:
        return None

    ds = _era5_store(cfg["era5_zarr"])
    t = pd.Timestamp(dt).round("1h")
    src_names = list(var_map)
    try:
        snap = ds[src_names].sel(time=t, method="nearest")
    except (KeyError, ValueError):
        return None

    # ARCO longitude is 0..360 ascending, latitude 90..-90 descending. Subset the
    # AoI window in that frame, then relabel longitude to -180..180 for reproject.
    w, s, e, n = g.search_bbox
    pad = cfg["pad_deg"]
    lon0, lon1 = (w - pad) % 360.0, (e + pad) % 360.0
    sub = snap.sel(latitude=slice(n + pad, s - pad),
                   longitude=slice(lon0, lon1)).load()
    if sub.longitude.size == 0 or sub.latitude.size == 0:
        return None

    sub = sub.assign_coords(longitude=(((sub.longitude + 180.0) % 360.0) - 180.0))
    sub = sub.sortby("longitude").rename({"longitude": "x", "latitude": "y"})
    sub = sub.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    out = sub.rio.reproject(dst_crs=g.target_crs, shape=g.shape, transform=g.transform,
                            resampling=Resampling.bilinear, nodata=np.nan)
    fields = {dst: np.asarray(out[src].values, dtype="float32")
              for src, dst in var_map.items()}
    return _era5_normalize(fields)


# --------------------------------------------------------------------------- #
# Source chain
# --------------------------------------------------------------------------- #
_SOURCES = {"hrrr": _fetch_hrrr, "era5": _fetch_era5}


def _fetch_one(src: str, g: AoiGrid, dt: datetime, cfg: dict) -> dict | None:
    """Fetch + regrid ONE source for one timestamp -> {outname: 2D array} or None.

    No chain / no fallback (D10): sources are STACKED as separate channels, so a source with
    no data here (HRRR off-continent, no ERA5 cell) simply contributes NaN to ITS channel --
    it is never silently substituted by another (HRRR and ERA5 are not interchangeable:
    different resolution, and `swrad` is instantaneous under HRRR but an hourly mean in ERA5).
    """
    try:
        return _SOURCES[src](g, dt, cfg)
    except Exception as exc:
        log.warning("    met: %s failed @ %s (%s)", src, dt, exc)
        return None


# --------------------------------------------------------------------------- #
# Dataset assembly + overpass discovery + IO
# --------------------------------------------------------------------------- #
def to_dataset(grids: dict, g: AoiGrid, t, to_celsius: bool) -> xr.Dataset:
    """Grids {name: 2D array} on the AoI grid -> Dataset (+ wind_speed, units)."""
    xs, ys = g.xy_centers()
    dv = {k: (("y", "x"), np.asarray(v, dtype="float32")) for k, v in grids.items()}
    ds = xr.Dataset(dv, coords={"y": ys, "x": xs})
    if "wind_u" in ds and "wind_v" in ds:
        ds["wind_speed"] = np.sqrt(ds["wind_u"] ** 2 + ds["wind_v"] ** 2)
        ds["wind_speed"].attrs["units"] = OUT_UNITS["wind_speed"]
    if "airtemp" in ds:
        if to_celsius:
            ds["airtemp"] = ds["airtemp"] - 273.15
            ds["airtemp"].attrs["units"] = "degC"
        else:
            ds["airtemp"].attrs["units"] = OUT_UNITS["airtemp"]
    if "swrad" in ds:
        ds["swrad"].attrs["units"] = OUT_UNITS["swrad"]
    if "cloud_cover" in ds:
        ds["cloud_cover"].attrs["units"] = OUT_UNITS["cloud_cover"]
    ds = ds.expand_dims(time=[pd.Timestamp(t)])
    ds = ds.rio.write_crs(g.target_crs)
    return ds


def parse_hhmm(value) -> float | None:
    """'10:30' -> 10.5 (hours). None/'' -> None (no reference snapshot)."""
    if value in (None, "", False):
        return None
    s = str(value).strip()
    if ":" in s:
        hh, mm = s.split(":", 1)
        return int(hh) + int(mm) / 60.0
    return float(s)


def reference_time_utc(day, lon: float, ref_hours: float, basis: str) -> pd.Timestamp:
    """The UTC instant of the reference time-of-day for one AoI on one day.

    With basis='solar' the reference is LOCAL SOLAR time, so 10:30 means 10:30 by the
    sun wherever the AoI is: UTC = local - lon/15 (each 15 deg of longitude is an hour).
    Rounded to the hour, because both HRRR and ERA5 are hourly; the day rolls over on
    its own where the conversion crosses midnight.
    """
    base = pd.Timestamp(day).normalize()
    hours = ref_hours if basis == "utc" else ref_hours - lon / 15.0
    return (base + pd.Timedelta(hours=hours)).round("1h")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    """Acquire met FORCING onto the pre-computed per-AoI grids, one tree PER SOURCE.

    Each configured source is written independently to `MET/<source>/aligned/<aoi>` (the
    reference-time snapshot and, optionally, the daily mean); the cube emits `<var>_<source>`.
    No chain / no fallback (D10). `only_source` narrows the run to one source. Overpass
    documentation is a SEPARATE product now (`met_overpass`), not written here.
    """
    grid_cfg = eff["grid"]
    met_root, fmt, overwrite = eff["met_root"], eff["fmt"], eff["overwrite"]
    to_celsius = grid_cfg.get("to_celsius", False)
    start = pd.Timestamp(eff["time"]["start_date"])
    end = pd.Timestamp(eff["time"]["end_date"])
    days = pd.date_range(start, end, freq="D")

    names = select_aois(grids, only_aoi)
    rep = report.ProductReport("met")

    for name in names:
        g = grids[name]
        ds_cfg = eff["ds"][name]
        mean_hours = ds_cfg["daily_mean_hours"]
        ref_time = ds_cfg["reference_time"]
        ref_basis = ds_cfg["reference_basis"]
        ref_hours = parse_hhmm(ref_time)
        lon = 0.5 * (g.search_bbox[0] + g.search_bbox[2])
        sources = [s for s in ds_cfg["sources"] if only_source is None or s == only_source]

        for src in sources:
            log.info("=== AOI: %s (grid=%dx%d) | source=%s ===",
                     name, g.width, g.height, src)
            if dry_run:
                log.info("  [dry-run] %d day(s); reference=%s (%s) + daily hours=%s",
                         len(days), ref_time or "off", ref_basis, mean_hours or "off"); continue
            aoi_out = met_root / src / "aligned" / name

            for day in days:
                dstr = naming.day_stamp(day)
                ref_stem = naming.day_stem(name, day, prefix="ref_")
                mean_stem = naming.day_stem(name, day)

                # ---- reference-time snapshot (the cube's default met channel) ----
                if ref_hours is not None and not store.done(
                        aoi_out / f"{ref_stem}.nc", store.REQUIRED_VARS["MET"],
                        shape=(g.height, g.width), overwrite=overwrite):
                    rt = reference_time_utc(day, lon, ref_hours, ref_basis)
                    got = _fetch_one(src, g, rt.to_pydatetime(), ds_cfg)
                    if got:
                        ds = to_dataset(got, g, rt, to_celsius)
                        ds.attrs.update(aoi_id=name, source=src, met_source=src,
                                        reference_time=str(ref_time), reference_basis=ref_basis,
                                        reference_time_utc=rt.isoformat(),
                                        **provenance.stamp(eff))
                        log.info("  %s reference [%s] (%s %s -> %s UTC) -> %s", dstr, src,
                                 ref_time, ref_basis, rt.strftime("%H:%M"),
                                 store.write_output(ds, aoi_out, ref_stem, fmt))
                        rep.wrote(source=src)
                    else:
                        rep.fail(f"{name} {src} ref {dstr}", f"{src}: no data")

                # ---- daily mean over mean_hours (skipped when daily_mean_hours: []) ----
                if mean_hours and not store.done(
                        aoi_out / f"{mean_stem}.nc", store.REQUIRED_VARS["MET"],
                        shape=(g.height, g.width), overwrite=overwrite):
                    stack, used_hours = {}, []
                    for hh in mean_hours:
                        dt = day.to_pydatetime().replace(hour=int(hh))
                        got = _fetch_one(src, g, dt, ds_cfg)
                        if not got:
                            continue
                        used_hours.append(int(hh))
                        for k, v in got.items():
                            stack.setdefault(k, []).append(v)
                    if stack:
                        mean_grids = {k: np.nanmean(np.stack(v), axis=0) for k, v in stack.items()}
                        ds = to_dataset(mean_grids, g, day, to_celsius)
                        # The hours that ACTUALLY contributed, not the ones we asked for -- a
                        # "daily mean" built from 1 of 4 hours must not claim all 4.
                        if len(used_hours) < len(mean_hours):
                            missing = [int(h) for h in mean_hours if int(h) not in used_hours]
                            log.warning("  %s [%s]: daily mean built from %d of %d hours "
                                        "(no data at %s UTC); NOT a full-day mean", dstr, src,
                                        len(used_hours), len(mean_hours),
                                        ", ".join(f"{h:02d}h" for h in missing))
                        ds.attrs.update(aoi_id=name, source=src, met_source=src,
                                        daily_mean_hours=str(used_hours),
                                        daily_mean_hours_requested=str([int(h) for h in mean_hours]),
                                        **provenance.stamp(eff))
                        log.info("  %s daily [%s, %dh] -> %s", dstr, src, len(used_hours),
                                 store.write_output(ds, aoi_out, mean_stem, fmt))
                        rep.wrote(source=src)
                    else:
                        rep.fail(f"{name} {src} daily {dstr}", f"{src}: no data at any hour")
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's met FORCING settings: which SOURCES to STACK (region-overridable) + the fetch
    knobs. HRRR is North America only, so an AoI outside it stacks only ERA5. The variable set,
    reference time/basis and daily-mean hours stay project-global (they decide the cube's met
    channels and what they mean)."""
    srcs = _opt(opts, "sources", None)
    if srcs is None:
        srcs = list(DEFAULT_SOURCES)
    elif isinstance(srcs, str):
        srcs = [srcs]
    return {
        "sources": [str(s) for s in srcs],
        "model": _opt(opts, "model", "auto"),
        "variables": list(_opt(opts, "variables", DEFAULT_VARIABLES)),
        "daily_mean_hours": list(_opt(opts, "daily_mean_hours", DEFAULT_MEAN_HOURS)),
        "reference_time": _opt(opts, "reference_time", DEFAULT_REFERENCE_TIME),
        "reference_basis": _opt(opts, "reference_basis", DEFAULT_REFERENCE_BASIS),
        "regrid_radius_m": float(_opt(opts, "regrid_radius_m", DEFAULT_RADIUS_M)),
        "pad_deg": float(_opt(opts, "pad_deg", DEFAULT_PAD_DEG)),
        "fxx": int(_opt(opts, "fxx", DEFAULT_FXX)),
        "product": _opt(opts, "product", DEFAULT_PRODUCT),
        "era5_zarr": _opt(opts, "era5_zarr", ARCO_ERA5_URI),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.met)
    if opts is None:
        raise ValueError("met is not a selected product in this config")

    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)          # GridSpec has no such field yet

    root = Path(project.output_dir)
    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.met))
               for a in project.all_areas},
        "grid": grid_cfg,
        "met_root": root / "MET",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None) -> None:
    """Acquire met FORCING for a validated Project. Entry point for pipeline.py.

    Every configured source is acquired and STACKED into its own `MET/<source>/` tree;
    `source` (from the pipeline, or unset on a direct CLI run) narrows to one. No sensor
    dependency any more -- overpass documentation is the separate `met_overpass` product.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    bad = sorted(f"{n}:{s}" for n, c in eff["ds"].items() for s in c["sources"]
                 if s not in _SOURCES)
    if bad:
        raise ValueError(f"met source not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(_SOURCES)}.")
    return run(eff, grids, aois, dry_run, only_source=source)


def main():
    entry.process_main(
        acquire, "coastal_sst_data meteorological forcing acquisition (HRRR/ERA5).")


if __name__ == "__main__":
    main()
