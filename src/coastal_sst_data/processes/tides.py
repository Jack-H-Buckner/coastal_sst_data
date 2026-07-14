#!/usr/bin/env python3
"""
coastal_sst_data -- tide-height forcing (NOAA CO-OPS + global model backup).

Reads the validated project config (coastal_sst_data.config.Project) and the
shared per-AoI grids (coastal_sst_data.grid.AoiGrid). Tide is ~spatially uniform
over a small AoI, so this produces one 1D height series per AoI from one of two
interchangeable SOURCES (a source registry + fallback, like bathymetry):

  * coops (default) -- NOAA CO-OPS. Find the nearest CO-OPS water-level station
    (from the AoI grid centroid), fetch its published HARMONIC CONSTITUENTS
    (harcon.json, one tiny/fast metadata request), and synthesize the series
    LOCALLY with pytides2. Public, no auth, any date range -- but CO-OPS gauges
    only exist in U.S. waters, so it has no coverage elsewhere.
  * eo_tides -- a GLOBAL ocean tide model (EOT20 by default) sampled at the AoI
    centroid via the eo-tides package (pyTMD under the hood). Works anywhere, so
    it is the natural BACKUP where CO-OPS has no nearby gauge.

By default coops is the source with eo_tides as the fallback: if the nearest
CO-OPS gauge is farther than `fallback_distance_km`, or the CO-OPS fetch fails,
the AoI is served from the global model instead. Either can also be selected
explicitly (project `default_source`, or per-region `sources.tides.source`).

    <output_dir>/TIDE/aligned/<aoi>/<aoi>_tides.nc   (dims: time; var: tide [m], rel. MSL)

Unlike the gridded products this is a 1D time series per Aoi: the datacube
assembler broadcasts it across the AoI grid and samples it at the daily /
overpass times. Needs `requests` + `pytides2` (coops) and `eo-tides` (eo_tides;
plus a downloaded tide-model directory -- see the eo-tides "Setting up tide
models" docs, pointed at via `model_directory` or $EO_TIDES_TIDE_MODELS).

Usage:
    python -m coastal_sst_data.processes.tides --config config.yaml
    python -m coastal_sst_data.processes.tides --config config.yaml --aoi hood_canal --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from ..config import Project, DataProduct, load_config
from ..grid import AoiGrid, project_grids

log = logging.getLogger(__name__)

SOURCE = "tides"

# --- tides product defaults (overridable via the tides product options block) - #
DEFAULT_INTERVAL = "h"        # prediction step; "h" = hourly
DEFAULT_WARN_KM = 75.0        # warn if the nearest gauge is farther than this
DEFAULT_SOURCE = "coops"      # primary tide source
DEFAULT_FALLBACK = "eo_tides"  # global backup when coops has no coverage
DEFAULT_FALLBACK_KM = 150.0   # nearest coops gauge farther than this -> backup
DEFAULT_MODEL = "EOT20"       # eo-tides global model (see eo-tides docs)


def _patch_legacy_compat():
    """pytides2 0.0.5 predates Py3.10 / NumPy>=1.24. Restore the aliases it uses
    (collections.Iterable etc., np.float etc.) so it imports cleanly."""
    import collections
    import collections.abc as _abc
    for _n in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Callable", "Hashable"):
        if not hasattr(collections, _n):
            setattr(collections, _n, getattr(_abc, _n))
    # Check np.__dict__ rather than hasattr: accessing a removed alias like
    # np.object goes through NumPy's __getattr__, which itself emits the
    # "np.object will be defined as..." FutureWarning we're trying to avoid.
    for _n, _t in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if _n not in np.__dict__:
            setattr(np, _n, _t)


_patch_legacy_compat()
STATIONS_MD = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
HARCON_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}/harcon.json"


# --------------------------------------------------------------------------- #
# Geometry + station selection
# --------------------------------------------------------------------------- #
def grid_centroid_lonlat(g: AoiGrid) -> tuple[float, float]:
    """(lon, lat) of the AoI grid's search bbox center (for gauge selection)."""
    w, s, e, n = g.search_bbox
    return (w + e) / 2.0, (s + n) / 2.0


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_stations() -> list:
    js = _get_json(STATIONS_MD, {"type": "waterlevels"})
    out = []
    for s in js.get("stations", []):
        try:
            out.append({"id": s["id"], "name": s["name"],
                        "lat": float(s["lat"]), "lon": float(s["lng"])})
        except (KeyError, ValueError, TypeError):
            continue
    return out


def nearest_station(lon, lat, stations):
    best, best_d = None, float("inf")
    for s in stations:
        d = haversine_km(lon, lat, s["lon"], s["lat"])
        if d < best_d:
            best, best_d = s, d
    return best, best_d


# --------------------------------------------------------------------------- #
# CO-OPS metadata fetch + local harmonic prediction
# --------------------------------------------------------------------------- #
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "coastal_sst_data/1.0 (research; coastal SST pipeline)",
    "Accept": "application/json",
})


def _get_json(url, params=None, retries=5):
    """GET with a real UA + exponential backoff (metadata endpoints are fast)."""
    last = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params or {}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            wait = min(30, 5 * (2 ** attempt))  # 5,10,20,30,30
            log.warning("    request failed (%s); retry %d/%d in %ds",
                        str(exc)[:70], attempt + 1, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"CO-OPS request failed after {retries} attempts: {last}")


def fetch_harcon(station_id) -> list:
    """Published harmonic constituents for a station (small, fast metadata call)."""
    js = _get_json(HARCON_URL.format(sid=station_id), {"units": "metric"})
    cons = js.get("HarmonicConstituents") or []
    if not cons:
        raise RuntimeError("no harmonic constituents returned")
    return cons


def predict_series(harcon, start, end, interval="h") -> pd.Series:
    """Compute the tide series locally from constituents (pytides2). Heights in m,
    relative to mean sea level (Z0 = 0). Nodal corrections handled by pytides2."""
    try:
        try:
            from pytides2.tide import Tide
            from pytides2.constituent import noaa
        except ImportError:  # original package name fallback
            from pytides.tide import Tide
            from pytides.constituent import noaa
    except ImportError as exc:
        raise RuntimeError(
            "pytides2 is not installed. Install it (deps come from conda) with:\n"
            "  mamba install -c conda-forge numpy scipy\n"
            "  pip install --no-build-isolation --no-deps pytides2"
        ) from exc

    # Plain Python lists, not numpy arrays: pytides2 does `None in [amps, phases]`
    # which raises "ambiguous truth value" if these are arrays.
    amps = [0.0] * len(noaa)
    phases = [0.0] * len(noaa)
    used = 0
    for hc in harcon:
        i = int(hc.get("number", 0)) - 1   # harcon 'number' indexes the NOAA order
        if 0 <= i < len(noaa):
            amps[i] = float(hc["amplitude"])
            phases[i] = float(hc["phase_GMT"])
            used += 1
    if used == 0:
        raise RuntimeError("constituents did not map to the NOAA set")

    tide = Tide(constituents=list(noaa), amplitudes=amps, phases=phases)
    freq = {"h": "h", "hourly": "h"}.get(str(interval), str(interval))
    times = pd.date_range(pd.Timestamp(start),
                          pd.Timestamp(end) + pd.Timedelta(days=1),
                          freq=freq, inclusive="left")
    heights = tide.at(list(times.to_pydatetime()))
    return pd.Series(np.asarray(heights, dtype="float32"), index=times, name="tide")


def predict_global(lon, lat, start, end, interval="h", model=DEFAULT_MODEL,
                   directory=None) -> pd.Series:
    """Sample a global ocean-tide model at (lon, lat) via eo-tides (pyTMD).

    Returns the same shape as `predict_series`: a time-indexed height Series in
    metres, relative to mean sea level (the model's tidal signal has zero mean).
    `directory` points at the downloaded tide-model files; if None, eo-tides
    reads the $EO_TIDES_TIDE_MODELS environment variable.
    """
    try:
        from eo_tides.model import model_tides
    except ImportError as exc:
        raise RuntimeError(
            "eo-tides is not installed. Install it with:\n"
            "  pip install eo-tides\n"
            "and download a tide-model directory (see the eo-tides 'Setting up "
            "tide models' docs), pointed at via the `model_directory` option or "
            "the EO_TIDES_TIDE_MODELS environment variable."
        ) from exc

    freq = {"h": "h", "hourly": "h"}.get(str(interval), str(interval))
    times = pd.date_range(pd.Timestamp(start),
                          pd.Timestamp(end) + pd.Timedelta(days=1),
                          freq=freq, inclusive="left")
    df = model_tides(x=float(lon), y=float(lat), time=times, model=model,
                     directory=directory, output_units="m")
    # Long format is multi-indexed by (time, x, y); reduce to a time-only Series.
    heights = df["tide_height"]
    if isinstance(heights.index, pd.MultiIndex):
        drop = [n for n in ("x", "y") if n in heights.index.names]
        heights = heights.droplevel(drop)
    s = pd.Series(np.asarray(heights.to_numpy(), dtype="float32"),
                  index=pd.DatetimeIndex(heights.index), name="tide")
    return s[~s.index.duplicated()].reindex(times)


def write_output(ds, out_dir, aoi_id, fmt) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt != "netcdf":
        # Tide is 1D; geotiff doesn't apply -> always NetCDF.
        log.info("  (tide is a 1D series; writing NetCDF regardless of output_format)")
    path = out_dir / f"{aoi_id}_tides.nc"
    ds.to_netcdf(path, encoding={"tide": {"zlib": True, "complevel": 4}})
    return path


# --------------------------------------------------------------------------- #
# Source registry. Each source: (lon, lat, start, end, ds_cfg, station) ->
# (height Series [m, rel. MSL], attrs dict). `station` is the resolved CO-OPS
# gauge (coops only; ignored by the global model). Extensible: fes2022, tpxo, ...
# --------------------------------------------------------------------------- #
def _source_coops(lon, lat, start, end, ds_cfg, station):
    """NOAA CO-OPS harmonic constituents -> local pytides2 synthesis."""
    if station is None:
        raise RuntimeError("no CO-OPS station resolved for this AoI")
    harcon = fetch_harcon(station["id"])
    s = predict_series(harcon, start, end, ds_cfg["interval"])
    attrs = {"station_id": station["id"], "station_name": station["name"],
             "station_lat": station["lat"], "station_lon": station["lon"],
             "distance_km": round(station.get("distance_km", 0.0), 2),
             "n_constituents": len(harcon),
             "method": "harmonic synthesis (pytides2)", "datum": "MSL"}
    return s, attrs


def _source_eo_tides(lon, lat, start, end, ds_cfg, station):
    """Global ocean-tide model (EOT20 etc.) sampled at the AoI centroid."""
    model = ds_cfg["model"]
    s = predict_global(lon, lat, start, end, ds_cfg["interval"], model,
                       ds_cfg["model_directory"])
    attrs = {"tide_model": model, "model_lon": round(float(lon), 4),
             "model_lat": round(float(lat), 4),
             "method": f"global tide model ({model}) via eo-tides", "datum": "MSL"}
    return s, attrs


SOURCES = {"coops": _source_coops, "eo_tides": _source_eo_tides}


def _predict_with_fallback(source, fallback, lon, lat, start, end, ds_cfg, station, name):
    """Run the resolved source; on error, try `fallback` if set/different.

    Returns (Series, attrs, used_source) or None if both failed.
    """
    try:
        s, attrs = SOURCES[source](lon, lat, start, end, ds_cfg, station)
        return s, attrs, source
    except Exception as exc:
        log.warning("  %s: %s tide source failed (%s)", name, source, exc)
    if fallback and fallback != source and fallback in SOURCES:
        log.info("  %s: falling back to %s", name, fallback)
        try:
            s, attrs = SOURCES[fallback](lon, lat, start, end, ds_cfg, station)
            return s, attrs, fallback
        except Exception as exc:
            log.warning("  %s: %s fallback failed (%s)", name, fallback, exc)
    return None


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], aoi_sources: dict[str, str],
        only_aoi, dry_run):
    """Acquire a 1D tide series per AoI from its resolved source (+ fallback)."""
    ds_cfg = eff["ds"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    station_overrides = ds_cfg["stations"]
    warn_km, fallback = ds_cfg["warn_distance_km"], ds_cfg["fallback"]
    fallback_km = ds_cfg["fallback_distance_km"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    names = list(grids)
    if only_aoi:
        req = set(only_aoi)
        missing = req - set(names)
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        names = [n for n in names if n in req]

    # The CO-OPS station list is only needed if some AoI uses coops (as its
    # source or as the fallback); load it lazily on first use.
    coops_in_play = fallback == "coops" or any(aoi_sources[n] == "coops" for n in names)
    stations = None

    for name in names:
        g = grids[name]
        lon, lat = grid_centroid_lonlat(g)
        source = aoi_sources[name]

        # Resolve a CO-OPS gauge if coops could serve this AoI (source or fallback).
        station = None
        if coops_in_play and (source == "coops" or fallback == "coops"):
            if station_overrides.get(name):
                station = {"id": str(station_overrides[name]), "name": "(config override)",
                           "lat": lat, "lon": lon, "distance_km": 0.0}
            else:
                if stations is None:
                    log.info("Fetching CO-OPS water-level station list...")
                    stations = fetch_stations()
                    log.info("  %d stations", len(stations))
                st, dist = nearest_station(lon, lat, stations)
                station = {**st, "distance_km": dist}

        # Distance pre-empt: if the nearest coops gauge is too far and a global
        # backup is available, serve this AoI from the backup instead.
        effective = source
        if (source == "coops" and station is not None
                and station["name"] != "(config override)"
                and station["distance_km"] > fallback_km
                and fallback and fallback != "coops" and fallback in SOURCES):
            log.info("=== AOI: %s | nearest gauge %s '%s' is %.0f km (> %.0f km) -> %s ===",
                     name, station["id"], station["name"], station["distance_km"],
                     fallback_km, fallback)
            effective = fallback
        elif source == "coops" and station is not None:
            log.info("=== AOI: %s | station %s '%s' (%.1f km) ===",
                     name, station["id"], station["name"], station["distance_km"])
            if station["distance_km"] > warn_km:
                log.warning("    nearest gauge is %.0f km away", station["distance_km"])
        else:
            log.info("=== AOI: %s | source=%s ===", name, effective)

        out_path = out_root / name / f"{name}_tides.nc"
        if not overwrite and out_path.exists():
            log.info("  already processed, skipping")
            continue
        if dry_run:
            log.info("  [dry-run] would build tides (%s) for %s (%s..%s @ %s)",
                     effective, name, start, end, ds_cfg["interval"])
            continue

        res = _predict_with_fallback(effective, fallback, lon, lat, start, end,
                                     ds_cfg, station, name)
        if res is None:
            log.warning("  skipping %s (no tide series from %s or fallback)", name, effective)
            continue
        s, attrs, used = res

        da = xr.DataArray(s.values.astype("float32"),
                          coords={"time": s.index.values}, dims="time", name="tide")
        da.attrs.update(units="m", long_name="tide height (harmonic prediction, rel. MSL)")
        ds = da.to_dataset()
        ds.attrs.update(aoi_id=name, source=used, **attrs)
        log.info("  wrote %s (%d steps) [%s]",
                 write_output(ds, out_root / name, name, fmt), ds.sizes["time"], used)
    log.info("Done.")


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _opt(opts, name, default):
    """Read an optional override off a product-options bag (extra='allow')."""
    return getattr(opts, name, default) if opts is not None else default


def _resolve_source(project: Project, aoi_name: str, default: str) -> str:
    """The tide source for one AoI: its region's `sources.tides.source`, else the
    project default. The two-level lookup mirrors bathymetry's dem_source."""
    opts = project.region_of(aoi_name).sources.get(DataProduct.tides)
    src = getattr(opts, "source", None) if opts is not None else None
    return src or default


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.tides)
    if opts is None:
        raise ValueError("tides is not a selected product in this config")

    # A blank / "none" fallback disables the backup entirely.
    fallback = _opt(opts, "fallback", DEFAULT_FALLBACK)
    if fallback in (None, "none", ""):
        fallback = None

    ds_cfg = {
        "interval": _opt(opts, "interval", DEFAULT_INTERVAL),
        # per-AoI gauge overrides: {aoi_name: CO-OPS station id}. Tide gauges are
        # per-location, so this lives with the product rather than on the AoI.
        "stations": dict(_opt(opts, "stations", {}) or {}),
        "warn_distance_km": float(_opt(opts, "warn_distance_km", DEFAULT_WARN_KM)),
        # Source selection + global backup.
        "default_source": _opt(opts, "default_source", DEFAULT_SOURCE),
        "fallback": fallback,
        "fallback_distance_km": float(_opt(opts, "fallback_distance_km", DEFAULT_FALLBACK_KM)),
        # eo_tides (global model) options.
        "model": _opt(opts, "model", DEFAULT_MODEL),
        "model_directory": _opt(opts, "model_directory", None),
    }

    root = Path(project.output_dir)
    return {
        "ds": ds_cfg,
        "out_dir": root / "TIDE" / "aligned",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire tide forcing for a validated Project. Entry point for pipeline.py.

    Parameters
    ----------
    project      validated config (coastal_sst_data.config.Project)
    grids        pre-computed {aoi_name: AoiGrid}; if None, computed here so all
                 products share one grid computation. Only the AoI centroid is
                 used (tide is a per-AoI 1D series, not regridded).
    aois         restrict to these AoI name(s); default all
    dry_run      resolve the source only, no fetch/predict/write
    overwrite    reprocess AoIs even if the aligned file exists

    The source is resolved PER AoI (region override -> project default), then
    validated against the SOURCES registry, so a typo'd source fails loudly.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    aoi_sources = {name: _resolve_source(project, name, eff["ds"]["default_source"])
                   for name in grids}
    bad = sorted(f"{n}:{s}" for n, s in aoi_sources.items() if s not in SOURCES)
    fb = eff["ds"]["fallback"]
    if fb is not None and fb not in SOURCES:
        bad.append(f"fallback:{fb}")
    if bad:
        raise ValueError(f"tides source not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(SOURCES)}.")
    run(eff, grids, aoi_sources, aois, dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="coastal_sst_data tide-height acquisition (NOAA CO-OPS).")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Process only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess AoIs even if the aligned file exists")
    ap.add_argument("--dry-run", action="store_true", help="Resolve gauges only; no fetch.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    project = load_config(args.config)
    acquire(project, aois=args.aoi, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
