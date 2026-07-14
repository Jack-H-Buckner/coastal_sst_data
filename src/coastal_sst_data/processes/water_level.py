#!/usr/bin/env python3
"""
coastal_sst_data -- water level (derived; runs inside the datacube assembler).

Combines the STATIC bathymetry DEM (elevation, m, negative below sea level) with
the 1D TIDE series (m, relative to MSL) to describe where the waterline actually
was when each thermal scene was taken. Two fields per sensor, both on the cube's
daily axis, evaluated at that sensor's OVERPASS time (the hourly tide series is
linearly interpolated to the scene hour):

  <sensor>_water_elev  (time,y,x) float32 -- elevation relative to the tide-adjusted
      waterline: 0 AT the waterline, positive above it (exposed, height above
      water), negative below it (submerged; the magnitude is the water depth).
  <sensor>_water_class (time,y,x) uint8   -- SUBMERGED (0) / EXPOSED (1), with
      UNKNOWN (255) where the DEM or the overpass tide is missing.

This derives no new data -- it is a pure function of two products the pipeline
already acquired -- so it lives in the assembler rather than being an acquisition
stage of its own. It needs both `bathymetry` and `tides` in the config; with
either missing the fields are all-UNKNOWN/NaN.

Two caveats worth knowing:

  * DATUM. Tides are relative to MSL; a DEM is not necessarily (CUDEM is NAVD88,
    a geodetic datum, and MSL is a tidal one -- the gap between the two surfaces
    is LOCAL and on the U.S. Pacific coast is of order a metre, i.e. comparable
    to the whole intertidal range). `datum_offset_m` is the elevation of the MSL
    surface IN THE DEM'S DATUM (equivalently: how far MSL sits above the DEM's
    zero), in metres; it is subtracted from the DEM to put the ground on MSL
    before the tide is applied. So for a NAVD88 DEM in Puget Sound, where MSL is
    ~1.3 m above NAVD88, datum_offset_m = +1.3.
    It is RESOLVED automatically by the `datum` stage (processes.datum), which
    reads the DEM that actually ran and looks the offset up from NOAA VDatum,
    cross-checked against the nearest CO-OPS gauge, writing it to a sidecar this
    module reads back. The only user knob is an optional per-region override,
    `regions[].sources.bathymetry.datum_offset_m`, for local knowledge.
  * The classification is purely GEOMETRIC (is this cell's ground below the
    waterline?). It does not consult land-cover, so diked or reclaimed ground
    lying below the waterline reads as SUBMERGED. That is deliberate: land-cover
    routinely calls tidal flats non-water, and letting it override here would
    erase exactly the intertidal signal these fields exist to capture. Downstream,
    intersect with the cube's `landmask` / `landcover_water` if you need a
    hydrologically-connected water mask.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Project
from . import datum

log = logging.getLogger(__name__)

# `<sensor>_water_class` codes.
SUBMERGED = 0     # ground lies below the tide-adjusted waterline
EXPOSED = 1       # ground lies above it (dry land or an exposed tidal flat)
UNKNOWN = 255     # no DEM here, or no overpass on this day (so no tide time)

DEFAULT_DATUM_OFFSET_M = 0.0


def resolve_datum_offset(project: Project, aoi_name: str,
                         *, bathy_attrs: dict | None = None) -> tuple[float, dict]:
    """(offset_m, provenance) for one AoI. Provenance is stamped onto the cube.

    Precedence:
      1. The `datum` stage's sidecar (which has already honoured any region override
         and validated it against the DEM that actually ran) -- unless it is STALE.
      2. A region override, if the stage never ran.
      3. 0.0, flagged `unresolved_assumed_zero`.

    STALENESS is the top silent-wrong-answer risk: bathymetry rewrites <aoi>.nc in
    place, so a DEM re-run that falls back cudem->gmrt leaves an old NAVD88 offset
    sitting beside an MSL DEM -- applying it would bias every pixel by ~1.3 m while
    looking perfectly healthy. The sidecar records the DEM's `source` attr verbatim;
    if it no longer matches the file we just read, the sidecar is REFUSED.

    Case 3 keeps the cube complete rather than blocking, but stamps a status the
    downstream user can see -- a warning in a log scrolls away; an attr does not.
    """
    root = Path(project.output_dir)
    rec = datum.load_sidecar(root, aoi_name)
    fingerprint = str((bathy_attrs or {}).get("source", ""))

    if rec is not None and fingerprint:
        stamped = rec.get("dem_fingerprint")
        if stamped and stamped != fingerprint:
            log.error(
                "%s: the datum sidecar was resolved for a different DEM (%r, now %r); "
                "refusing it. Re-run `coastal-sst-data datum` to resolve the offset for "
                "the DEM that is actually on disk.", aoi_name, stamped, fingerprint)
            rec = None

    if rec is not None:
        return float(rec.get("datum_offset_m", 0.0)), {
            "datum_offset_m": float(rec.get("datum_offset_m", 0.0)),
            "datum_method": str(rec.get("method", "unknown")),
            "datum_status": str(rec.get("status", "unknown")),
            "datum_uncertainty_m": rec.get("uncertainty_m"),
            "dem_source": str(rec.get("dem_source", "unknown")),
            "dem_vertical_datum": str(rec.get("dem_vertical_datum", "unknown")),
            "tidal_datum_epoch": str(rec.get("tidal_datum_epoch", "")),
        }

    override = datum.resolve_override(project, aoi_name)
    if override is not None:
        return float(override), {"datum_offset_m": float(override),
                                 "datum_method": "config_override", "datum_status": "ok"}

    log.error(
        "%s: no datum offset resolved -- assuming 0.0 m, i.e. the DEM is MSL-referenced. "
        "If it is a NAVD88 DEM (CUDEM) the water level will be biased by ~1 m on the "
        "Pacific coast. Run `coastal-sst-data datum --config <cfg>` to resolve it.",
        aoi_name)
    return DEFAULT_DATUM_OFFSET_M, {"datum_offset_m": DEFAULT_DATUM_OFFSET_M,
                                    "datum_method": "none",
                                    "datum_status": "unresolved_assumed_zero"}


def load_tide_series(d, aoi_id) -> pd.Series | None:
    """The hourly tide series for one AoI (m, rel. MSL), or None if absent.

    The assembler's `load_tide_daily` reduces this same file to daily statistics;
    here the full-resolution series is kept, because an overpass tide has to be
    interpolated to the scene's hour.
    """
    f = d / f"{aoi_id}_tides.nc"
    if not f.exists():
        return None
    with xr.open_dataset(f) as ds:
        if "tide" not in ds or "time" not in ds["tide"].dims:
            return None
        s = ds["tide"].to_series().astype("float64")
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)   # tide files are UTC
    s = s[~s.index.duplicated()].sort_index().dropna()
    return s if not s.empty else None


def tide_at_overpass(series: pd.Series | None, days, hours) -> np.ndarray:
    """Tide height at each day's overpass time -> (T,) float32, NaN where absent.

    `hours` is the assembler's per-sensor `<sensor>_hour` (fractional UTC hour of
    the day's chosen scene, NaN on days with no scene). The hourly series is
    linearly interpolated; a time outside its span yields NaN rather than being
    clamped to an endpoint, so a scene beyond the tide record is not silently
    given the last known height.
    """
    hours = np.asarray(hours, dtype="float64")
    out = np.full(len(days), np.nan, dtype="float32")
    have = np.isfinite(hours)
    if series is None or not have.any():
        return out

    t = pd.DatetimeIndex(days)[have] + pd.to_timedelta(hours[have], unit="h")
    xp = series.index.values.astype("datetime64[ns]").astype("int64")
    fp = series.to_numpy(dtype="float64")
    x = t.values.astype("datetime64[ns]").astype("int64")
    out[have] = np.interp(x, xp, fp, left=np.nan, right=np.nan).astype("float32")
    return out


def water_level_fields(elev: np.ndarray, tide: np.ndarray, *,
                       datum_offset_m: float = DEFAULT_DATUM_OFFSET_M):
    """Water-level fields for one sensor: (water_elev (T,H,W), water_class (T,H,W)).

    `elev` is the static DEM (H,W) in its own datum; `datum_offset_m` is where MSL
    sits in that datum, so subtracting it puts the ground on MSL, where the tide
    (T,) in m rel. MSL applies (NaN on days with no overpass). `water_elev`
    re-references the ground to the
    tide-adjusted waterline (0 there, + above, - below); `water_class` is the sign
    of that as SUBMERGED / EXPOSED, and UNKNOWN wherever either input is NaN.
    """
    elev_msl = elev.astype("float32") - np.float32(datum_offset_m)
    tide = np.asarray(tide, dtype="float32")

    water_elev = (elev_msl[None, :, :] - tide[:, None, None]).astype("float32")
    known = np.isfinite(water_elev)
    water_class = np.full(water_elev.shape, UNKNOWN, dtype="uint8")
    water_class[known] = np.where(water_elev[known] > 0, EXPOSED, SUBMERGED)
    return water_elev, water_class
