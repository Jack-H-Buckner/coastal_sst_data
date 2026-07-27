#!/usr/bin/env python3
"""
coastal_sst_data -- tide-height forcing (NOAA CO-OPS + global model, STACKED).

Reads the validated project config (coastal_sst_data.config.Project) and the
shared per-AoI grids (coastal_sst_data.grid.AoiGrid). Tide is ~spatially uniform
over a small AoI, so this produces one 1D height series per AoI per SOURCE. The
sources are DISTINCT DATA, STACKED one channel per source (D10) -- not a primary +
fallback -- so the user lists what they need and there is no substitution:

  * coops -- NOAA CO-OPS. Find the nearest CO-OPS water-level station (from the AoI
    grid centroid), fetch its published HARMONIC CONSTITUENTS (harcon.json, one
    tiny/fast metadata request), and synthesize the series LOCALLY with pytides2.
    Public, no auth, any date range -- but CO-OPS gauges only exist in U.S. waters,
    so where the nearest gauge is farther than `max_distance_km` it contributes NO
    channel here (the user stacks the global model).
  * eo_tides -- a GLOBAL ocean tide model (EOT20 by default) sampled at the AoI
    centroid via the eo-tides package (pyTMD under the hood). Works anywhere.

By default both are stacked (`sources: [coops, eo_tides]`); a region overrides the
list (e.g. `sources: [eo_tides]` outside U.S. waters). The cube emits `tide_<source>`
/ `tide_range_<source>` per source.

    <output_dir>/TIDE/<source>/aligned/<aoi>/<aoi>_tides.nc  (dims: time; var: tide [m], rel. MSL)

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

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, provenance, report, store

log = logging.getLogger(__name__)

SOURCE = "tides"

# --- tides product defaults (overridable via the tides product options block) - #
DEFAULT_INTERVAL = "h"        # prediction step; "h" = hourly
DEFAULT_WARN_KM = 75.0        # warn if the nearest gauge is farther than this
DEFAULT_SOURCES = ["coops", "eo_tides"]   # STACKED (D10): coops in U.S. waters, model global
DEFAULT_MAX_KM = 150.0        # nearest coops gauge farther than this -> no coops channel here
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
    """Tide is a 1D series, not a raster: it does NOT go through store.write_output.

    A GeoTIFF has no meaning for a time series with no y/x, so `output_format: geotiff`
    is deliberately ignored here rather than raising -- a project that asks for GeoTIFF
    rasters should still get its tide series.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt != "netcdf":
        log.info("  (tide is a 1D series; writing NetCDF regardless of output_format)")
    return store.write_netcdf(ds, out_dir / f"{aoi_id}_tides.nc",
                              encoding={"tide": {"zlib": True, "complevel": 4}})


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


def _resolve_coops_station(name, lon, lat, ds_cfg, stations_cache):
    """The CO-OPS gauge serving this AoI (config override, else nearest), or None if the
    nearest is farther than `max_distance_km` -- in which case coops has no coverage here and
    contributes no channel (the user stacks the global model). Returns (station|None, stations_cache)."""
    if ds_cfg["stations"].get(name):
        return {"id": str(ds_cfg["stations"][name]), "name": "(config override)",
                "lat": lat, "lon": lon, "distance_km": 0.0}, stations_cache
    if stations_cache is None:
        log.info("Fetching CO-OPS water-level station list...")
        stations_cache = fetch_stations()
        log.info("  %d stations", len(stations_cache))
    st, dist = nearest_station(lon, lat, stations_cache)
    if dist > ds_cfg["max_distance_km"]:
        log.warning("  %s: nearest CO-OPS gauge %s '%s' is %.0f km (> %.0f km); no tide_coops "
                    "channel here (stack the global model)", name, st["id"], st["name"], dist,
                    ds_cfg["max_distance_km"])
        return None, stations_cache
    if dist > ds_cfg["warn_distance_km"]:
        log.warning("  %s: nearest gauge is %.0f km away", name, dist)
    return {**st, "distance_km": dist}, stations_cache


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    """Acquire a 1D tide series per (AoI, SOURCE), each in its own `<source>/` tree.

    No fallback (D10): each configured source is produced independently. CO-OPS with no gauge
    within `max_distance_km` contributes no channel; the global model always does. `only_source`
    narrows the run to one source.
    """
    tide_root, fmt, overwrite = eff["tide_root"], eff["fmt"], eff["overwrite"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    names = select_aois(grids, only_aoi)
    stations = None                       # CO-OPS station list, fetched lazily on first coops use
    rep = report.ProductReport("tides")

    for name in names:
        g = grids[name]
        lon, lat = grid_centroid_lonlat(g)
        ds_cfg = eff["ds"][name]
        sources = [s for s in ds_cfg["sources"] if only_source is None or s == only_source]

        for src in sources:
            out_path = tide_root / src / "aligned" / name / f"{name}_tides.nc"
            if store.done(out_path, store.REQUIRED_VARS["TIDE"],
                          covers=(start, end), overwrite=overwrite):
                log.info("  %s [%s]: already processed, skipping", name, src); rep.skip(); continue

            station = None
            if src == "coops":
                station, stations = _resolve_coops_station(name, lon, lat, ds_cfg, stations)
                if station is None:                  # no gauge in range -> no coops channel here
                    rep.fail(f"{name} coops", "no CO-OPS gauge in range"); continue
                log.info("=== AOI: %s | source=coops | station %s '%s' (%.1f km) ===",
                         name, station["id"], station["name"], station["distance_km"])
            else:
                log.info("=== AOI: %s | source=%s ===", name, src)
            if dry_run:
                log.info("  [dry-run] would build tides (%s) for %s (%s..%s @ %s)",
                         src, name, start, end, ds_cfg["interval"]); continue

            try:
                s, attrs = SOURCES[src](lon, lat, start, end, ds_cfg, station)
            except Exception as exc:
                log.warning("  %s: %s tide source failed (%s); no channel from it", name, src, exc)
                rep.fail(f"{name} {src}", f"{src} failed: {exc}"); continue

            da = xr.DataArray(s.values.astype("float32"),
                              coords={"time": s.index.values}, dims="time", name="tide")
            da.attrs.update(units="m", long_name="tide height (harmonic prediction, rel. MSL)")
            ds = da.to_dataset()
            ds.attrs.update(aoi_id=name, source=src, tide_source=src, **attrs,
                            **provenance.requested_range(start, end),
                            **provenance.stamp(eff))
            log.info("  wrote %s (%d steps) [%s]",
                     write_output(ds, tide_root / src / "aligned" / name, name, fmt),
                     ds.sizes["time"], src)
            rep.wrote(source=src)
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's tide settings: which SOURCES to STACK (region-overridable), plus the model /
    gauge knobs. CO-OPS gauges exist only in U.S. waters, so elsewhere the AoI stacks the
    global model; which model is downloaded (and where) is a property of the machine/region.

    The distance thresholds and the prediction interval stay project-global -- they decide how
    a series is BUILT / whether coops has coverage, not which sources exist.
    """
    srcs = _opt(opts, "sources", None)
    if srcs is None:
        srcs = list(DEFAULT_SOURCES)
    elif isinstance(srcs, str):
        srcs = [srcs]
    return {
        "sources": [str(s) for s in srcs],
        "interval": _opt(opts, "interval", DEFAULT_INTERVAL),
        # per-AoI gauge overrides: {aoi_name: CO-OPS station id}. Tide gauges are per-location,
        # so this lives with the product rather than on the AoI.
        "stations": dict(_opt(opts, "stations", {}) or {}),
        "warn_distance_km": float(_opt(opts, "warn_distance_km", DEFAULT_WARN_KM)),
        "max_distance_km": float(_opt(opts, "max_distance_km", DEFAULT_MAX_KM)),
        # eo_tides (global model) options.
        "model": _opt(opts, "model", DEFAULT_MODEL),
        "model_directory": _opt(opts, "model_directory", None),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.tides)
    if opts is None:
        raise ValueError("tides is not a selected product in this config")

    root = Path(project.output_dir)
    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.tides))
               for a in project.all_areas},
        "tide_root": root / "TIDE",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None) -> None:
    """Acquire tide forcing for a validated Project. Entry point for pipeline.py.

    Every configured source is acquired and STACKED into its own `TIDE/<source>/` tree;
    `source` (from the pipeline, or unset on a direct CLI run) narrows to one. The sources are
    resolved PER AoI (region override -> project default) and validated against the SOURCES
    registry, so a typo fails loudly -- before a single gauge is queried.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    bad = sorted(f"{n}:{s}" for n, c in eff["ds"].items() for s in c["sources"]
                 if s not in SOURCES)
    if bad:
        raise ValueError(f"tides source not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(SOURCES)}.")
    return run(eff, grids, aois, dry_run, only_source=source)


def main():
    entry.process_main(
        acquire, "coastal_sst_data tide-height acquisition (NOAA CO-OPS + global model).")


if __name__ == "__main__":
    main()
