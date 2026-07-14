#!/usr/bin/env python3
"""
coastal_sst_data -- meteorological forcing acquisition (HRRR, ERA5 fallback).

Reads the validated project config (coastal_sst_data.config.Project) and the
shared per-AoI grids (coastal_sst_data.grid.AoiGrid), exactly like the SST
stages. For each AoI it pulls 2 m air temperature, 10 m wind (u/v + speed),
downward shortwave and total cloud cover, regrids onto the AoI grid, and writes:

  * a REFERENCE-time file per day:       <output_dir>/MET/aligned/<aoi>/<aoi>_ref_<YYYYMMDD>.nc
  * a daily-mean file per day:           <output_dir>/MET/aligned/<aoi>/<aoi>_<YYYYMMDD>.nc
  * an instantaneous file per overpass:  <output_dir>/MET/aligned/<aoi>/<aoi>_<YYYYMMDDThhmmss>.nc

The REFERENCE file is one snapshot per day at a fixed time of day -- by default 10:30
LOCAL SOLAR time, Landsat's overpass. It is the datacube's default met channel: a daily
mean smears the diurnal cycle (dawn and mid-afternoon forcing averaged together), which
is the wrong forcing to hand a model of a sensor that flew at one instant. The basis is
solar rather than UTC because a fixed UTC hour is a different time of day in every AoI,
so cross-AoI comparisons would not be like-for-like. Set `reference_time: null` to skip.

Overpass times are discovered from the ECOSTRESS/Landsat aligned dirs so the
forcing can be matched to each thermal scene (for the skin->bulk correction); the
datacube reads those snapshots into per-sensor channels (`eco_airtemp`, ...).

Two sources, tried in order (the "source chain"):
  * hrrr -- NOAA HRRR 3 km surface fields via Herbie. CONUS uses the 'hrrr'
    domain; AoIs above ~50N use the Alaska domain 'hrrrak' (auto by latitude).
    Curvilinear grid -> pyresample nearest-neighbour resample.
  * era5 -- ECMWF ERA5 0.25 deg hourly reanalysis, streamed from Google's public
    ARCO-ERA5 Zarr on GCS (no auth). Global, so it fills any AoI/date/cycle HRRR
    cannot cover. Regular lat/lon grid -> rioxarray bilinear reproject.

With `source: auto` (default) the chain is [hrrr, era5]: ERA5 transparently
backfills wherever HRRR misses. Both sources are unit-harmonized to a single
convention (airtemp K, swrad W m-2, cloud_cover %) so the written files are
source-agnostic; each file records its provenance in `ds.attrs["source"]`.

Forcing is gap-free, so there is no mask -- these are complete driver channels.

Usage (run AFTER the SST stages so overpass times exist):
    python -m coastal_sst_data.processes.met --config config.yaml
    python -m coastal_sst_data.processes.met --config config.yaml --aoi hood_canal
    python -m coastal_sst_data.processes.met --config config.yaml --dry-run
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

import rioxarray  # noqa: F401  (registers the .rio accessor)

from ..config import Project, DataProduct, load_config
from ..grid import AoiGrid, project_grids
from .. import provenance, report, store

log = logging.getLogger(__name__)

SOURCE = "met"
_DT_RE = re.compile(r"(\d{8}T\d{6})")     # aligned-scene filename stamp

# --- met product defaults (overridable via the met product options block) ---- #
DEFAULT_SOURCE = "auto"                   # auto (hrrr->era5) | hrrr | era5
DEFAULT_FALLBACK = "era5"                 # era5 | none
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


def _resolve_chain(source: str, fallback: str) -> list[str]:
    """Ordered source chain from the `source`/`fallback` selectors.

    auto  -> [hrrr]  (then + fallback)   hrrr -> [hrrr]   era5 -> [era5]
    A non-'none' fallback is appended if not already the primary, so the default
    (source=auto, fallback=era5) yields [hrrr, era5].
    """
    source = str(source or "auto").lower()
    if source == "auto":
        chain = ["hrrr"]
    elif source in _SOURCES:
        chain = [source]
    else:
        raise ValueError(f"met source {source!r} not recognized; choose from "
                         f"{['auto'] + sorted(_SOURCES)}.")
    fb = str(fallback or "none").lower()
    if fb != "none":
        if fb not in _SOURCES:
            raise ValueError(f"met fallback {fb!r} not recognized; choose from "
                             f"{['none'] + sorted(_SOURCES)}.")
        if fb not in chain:
            chain.append(fb)
    return chain


def _fetch_at(chain, g: AoiGrid, dt: datetime, cfg: dict, tally: dict | None = None):
    """Walk the source chain; return (grids, source_name) of the first hit.

    EVERY fall-through is logged. This was the quietest failure in the package: the three
    clean `None` returns on the HRRR path (AoI outside the HRRR box, no fields returned, no
    cycle available) said nothing at all, and the success lines never named the source that
    won -- so a run whose entire met forcing silently came from 28 km ERA5 instead of 3 km
    HRRR produced a log in which the string "era5" never appeared. The two are not
    interchangeable: different resolution, different regridding, and `swrad` is an
    INSTANTANEOUS flux under HRRR but an hourly MEAN under ERA5.

    `tally` accumulates {source: n} so the run can report what actually served it.
    """
    for i, name in enumerate(chain):
        try:
            grids = _SOURCES[name](g, dt, cfg)
        except Exception as exc:
            log.warning("    met: %s failed @ %s (%s)", name, dt, exc)
            grids = None
        if grids:
            if i:                      # a later link won -> we fell back, silently until now
                log.info("    met: FELL BACK to %s @ %s (%s had no data)",
                         name, dt, " -> ".join(chain[:i]))
            if tally is not None:
                tally[name] = tally.get(name, 0) + 1
            return grids, name
        log.info("    met: %s has no data @ %s; trying the next source in the chain",
                 name, dt)
    log.warning("    met: NO source in %s returned data @ %s", " -> ".join(chain), dt)
    if tally is not None:
        tally["<none>"] = tally.get("<none>", 0) + 1
    return None, None


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


def overpass_times_for_day(overpass_dirs, aoi_id: str, day) -> list[datetime]:
    """Datetimes (UTC, naive) of thermal scenes for this AoI on `day`."""
    daystr = day.strftime("%Y%m%d")
    times = []
    for d in overpass_dirs:
        adir = Path(d) / aoi_id
        if not adir.exists():
            continue
        for f in adir.glob(f"{aoi_id}_{daystr}T*.nc"):
            m = _DT_RE.search(f.name)
            if m:
                times.append(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
    return sorted(set(times))


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str, fmt: str, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "netcdf":
        return store.write_netcdf(ds, out_dir / f"{stem}.nc")
    if fmt == "geotiff":
        return store.write_rasters(ds, out_dir / stem,
                                   [(v, ds[v].isel(time=0)) for v in ds.data_vars])
    raise ValueError(f"Unknown output format: {fmt}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Acquire met forcing onto the pre-computed per-AoI grids."""
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    to_celsius = grid_cfg.get("to_celsius", False)
    chain = ds_cfg["chain"]
    mean_hours = ds_cfg["daily_mean_hours"]
    ref_time = ds_cfg["reference_time"]
    ref_basis = ds_cfg["reference_basis"]
    ref_hours = parse_hhmm(ref_time)
    overpass_dirs = eff["overpass_dirs"]
    start = pd.Timestamp(eff["time"]["start_date"])
    end = pd.Timestamp(eff["time"]["end_date"])
    days = pd.date_range(start, end, freq="D")

    names = list(grids)
    if only_aoi:
        req = set(only_aoi)
        missing = req - set(names)
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        names = [n for n in names if n in req]

    rep = report.ProductReport("met")
    tally: dict[str, int] = {}     # {source: n fetches} -- folded into `rep` at the end

    for name in names:
        g = grids[name]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d) | chain=%s ===",
                 name, g.target_crs, g.width, g.height, "->".join(chain))
        if dry_run:
            log.info("  [dry-run] %d day(s); reference=%s (%s) + daily hours=%s "
                     "+ overpass snapshots", len(days), ref_time or "off", ref_basis,
                     mean_hours or "off")
            continue
        aoi_out = out_root / name

        lon = 0.5 * (g.search_bbox[0] + g.search_bbox[2])

        for day in days:
            dstr = day.strftime("%Y%m%d")

            # ---- reference-time snapshot (the cube's default met channel) ----
            if ref_hours is not None and not store.done(
                    aoi_out / f"{name}_ref_{dstr}.nc", store.REQUIRED_VARS["MET"],
                    shape=(g.height, g.width), overwrite=overwrite):
                rt = reference_time_utc(day, lon, ref_hours, ref_basis)
                got, src = _fetch_at(chain, g, rt.to_pydatetime(), ds_cfg, tally)
                if got:
                    ds = to_dataset(got, g, rt, to_celsius)
                    ds.attrs.update(aoi_id=name, source=f"{src} @reference",
                                    reference_time=str(ref_time), reference_basis=ref_basis,
                                    reference_time_utc=rt.isoformat(),
                                    **provenance.stamp(eff))
                    log.info("  %s reference [%s] (%s %s -> %s UTC) -> %s", dstr, src,
                             ref_time, ref_basis, rt.strftime("%H:%M"),
                             write_output(ds, aoi_out, name, fmt, f"{name}_ref_{dstr}"))
                    rep.wrote(source=src)
                else:
                    rep.fail(f"{name} ref {dstr}", "no source in the chain had data")

            # ---- daily mean over mean_hours (skipped when daily_mean_hours: []) ----
            if mean_hours and not store.done(
                    aoi_out / f"{name}_{dstr}.nc", store.REQUIRED_VARS["MET"],
                    shape=(g.height, g.width), overwrite=overwrite):
                stack, srcs, used_hours = {}, set(), []
                for hh in mean_hours:
                    dt = day.to_pydatetime().replace(hour=int(hh))
                    got, src = _fetch_at(chain, g, dt, ds_cfg, tally)
                    if not got:
                        continue
                    used_hours.append(int(hh))
                    srcs.add(src)
                    for k, v in got.items():
                        stack.setdefault(k, []).append(v)
                if stack:
                    mean_grids = {k: np.nanmean(np.stack(v), axis=0) for k, v in stack.items()}
                    ds = to_dataset(mean_grids, g, day, to_celsius)
                    # The hours that ACTUALLY contributed, not the ones we asked for. The
                    # attr used to echo `mean_hours` regardless, so a "daily mean" built
                    # from 1 of 4 hours claimed all 4 -- a mean over a quarter of the
                    # diurnal cycle, indistinguishable from the real thing.
                    if len(used_hours) < len(mean_hours):
                        missing = [int(h) for h in mean_hours if int(h) not in used_hours]
                        log.warning("  %s: daily mean built from %d of %d hours "
                                    "(no data at %s UTC); it is NOT a full-day mean",
                                    dstr, len(used_hours), len(mean_hours),
                                    ", ".join(f"{h:02d}h" for h in missing))
                    ds.attrs.update(aoi_id=name, source=f"{'+'.join(sorted(srcs))} daily mean",
                                    daily_mean_hours=str(used_hours),
                                    daily_mean_hours_requested=str([int(h) for h in mean_hours]),
                                    **provenance.stamp(eff))
                    log.info("  %s daily [%s, %dh] -> %s", dstr, "+".join(sorted(srcs)),
                             len(used_hours),
                             write_output(ds, aoi_out, name, fmt, f"{name}_{dstr}"))
                    rep.wrote(source="+".join(sorted(srcs)) + " daily mean")
                else:
                    rep.fail(f"{name} daily {dstr}", "no source had data at any hour")

            # ---- overpass snapshots ----
            for op in overpass_times_for_day(overpass_dirs, name, day):
                tstr = op.strftime("%Y%m%dT%H%M%S")
                if store.done(aoi_out / f"{name}_{tstr}.nc", store.REQUIRED_VARS["MET"],
                              shape=(g.height, g.width), overwrite=overwrite):
                    continue
                got, src = _fetch_at(chain, g, op, ds_cfg, tally)
                if not got:
                    rep.fail(f"{name} overpass {tstr}", "no source in the chain had data")
                    continue
                ds = to_dataset(got, g, op, to_celsius)
                ds.attrs.update(aoi_id=name, source=f"{src} @overpass", **provenance.stamp(eff))
                log.info("  overpass %s [%s] -> %s", tstr, src,
                         write_output(ds, aoi_out, name, fmt, f"{name}_{tstr}"))
                rep.wrote(source=src)

    # Which source actually SERVED this run -- not which was configured. The per-file
    # `source` attr is the truth, but nobody reads 3,000 attrs, so the run says it once.
    fell_back = [k for k in tally if k not in ("<none>", chain[0])]
    if fell_back:
        n = sum(tally[k] for k in fell_back)
        rep.note = (f"FELL BACK from {chain[0]} for {n} fetch(es) -> "
                    f"{', '.join(sorted(fell_back))} (different resolution; swrad means "
                    "something different)")
        log.warning("met: %s", rep.note)
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _opt(opts, name, default):
    """Read an optional override off a product-options bag (extra='allow')."""
    return getattr(opts, name, default) if opts is not None else default


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.met)
    if opts is None:
        raise ValueError("met is not a selected product in this config")

    chain = _resolve_chain(_opt(opts, "source", DEFAULT_SOURCE),
                           _opt(opts, "fallback", DEFAULT_FALLBACK))
    overpass_from = list(_opt(opts, "overpass_from", []))

    ds_cfg = {
        "chain": chain,
        "variables": list(_opt(opts, "variables", DEFAULT_VARIABLES)),
        "daily_mean_hours": list(_opt(opts, "daily_mean_hours", DEFAULT_MEAN_HOURS)),
        "reference_time": _opt(opts, "reference_time", DEFAULT_REFERENCE_TIME),
        "reference_basis": _opt(opts, "reference_basis", DEFAULT_REFERENCE_BASIS),
        "regrid_radius_m": float(_opt(opts, "regrid_radius_m", DEFAULT_RADIUS_M)),
        "pad_deg": float(_opt(opts, "pad_deg", DEFAULT_PAD_DEG)),
        "fxx": int(_opt(opts, "fxx", DEFAULT_FXX)),
        "product": _opt(opts, "product", DEFAULT_PRODUCT),
        "model": _opt(opts, "model", "auto"),
        "era5_zarr": _opt(opts, "era5_zarr", ARCO_ERA5_URI),
    }
    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)          # GridSpec has no such field yet

    root = Path(project.output_dir)
    return {
        "config_sha256": project.config_sha256,
        "ds": ds_cfg,
        "grid": grid_cfg,
        "out_dir": root / "MET" / "aligned",
        # thermal-scene dirs to snapshot forcing at (e.g. ECOSTRESS, LANDSAT).
        "overpass_dirs": [root / s.upper() / "aligned" for s in overpass_from],
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire met forcing for a validated Project. Entry point for pipeline.py.

    Parameters
    ----------
    project      validated config (coastal_sst_data.config.Project)
    grids        pre-computed {aoi_name: AoiGrid}; if None, computed here so all
                 products share one grid computation.
    aois         restrict to these AoI name(s); default all
    dry_run      report only, no fetch/write
    overwrite    reprocess days even if the aligned file exists

    Runs AFTER the SST stages when `overpass_from` is set, so their aligned files
    exist to snapshot forcing against.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="coastal_sst_data meteorological forcing acquisition (HRRR/ERA5).")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Process only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess days even if the aligned file exists")
    ap.add_argument("--dry-run", action="store_true", help="Report only; no fetch.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    project = load_config(args.config)
    acquire(project, aois=args.aoi, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
