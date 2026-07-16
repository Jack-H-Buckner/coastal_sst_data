#!/usr/bin/env python3
"""
coastal_sst_data -- DEM vertical-datum offset RESOLVER (a library, used by bathymetry).

Referencing a DEM's `elevation` to MSL needs the DEM and the tide on ONE vertical datum:

    elev_msl = elev_dem - datum_offset_m        # datum_offset_m = MSL in the DEM's datum

`resolve_aoi(grid, elevation, dem_source, ...)` computes that offset per (AoI, DEM source).
It is called INLINE by the bathymetry module as each DEM source is acquired, so the offset
travels with that source's output (stamped as its `datum_offset_m` / `datum_status` attrs)
rather than in a separate stage/sidecar. Adding a bathymetry source therefore means adding
its `DEM_DATUM` entry here -- part of the contract for a new DEM.

The right value is a property of (the DEM that actually ran, where the AoI is) -- not of
the project -- so it cannot be a project-wide constant:

  * CUDEM is referenced to NAVD88, a GEODETIC datum. MSL is a TIDAL one (the 19-year
    mean of observed water level). The gap between the two surfaces is LOCAL: ~1.3 m
    on the U.S. Pacific coast, ~0.1-0.3 m on the Gulf coast. At 1.3 m it is comparable
    to the whole intertidal range, so getting it wrong misclassifies most of a flat.
    -> resolved from NOAA VDatum (NAVD88 -> LMSL), cross-checked against the nearest
       CO-OPS gauge's published datums.
  * GMRT is already ~sea-level referenced -> the offset is 0.0, and no network call is
    made. (Residual geoid-vs-LMSL difference is well under GMRT's own vertical error
    at 100 m in an estuary.)

Because each DEM source is now acquired into its OWN tree and the offset is resolved for it
right there, the offset always follows the DEM it belongs to -- there is no separate stage
reading a possibly-stale winner back off disk, and no cudem->gmrt fallback to track.

Both tide sources are zero-mean (CO-OPS harmonic constituents carry no Z0 term; EOT20
likewise), so the TIDE side of the equation needs no correction -- only the DEM side.
Both are on the 1983-2001 NTDE, so a ~5-15 cm residual against contemporaneous MSL
remains; it is recorded, not corrected.

The result is a small record; bathymetry stamps `datum_offset_m` / `datum_status` /
`datum_method` onto each DEM source's output, and the datacube assembler surfaces them as
per-source cube attributes. To re-resolve, re-run bathymetry with `--overwrite` (there is no
separate `datum` subcommand any more).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import numpy as np
import requests
import xarray as xr

from ..config import DataProduct, Project, opt, resolve_opts
from ..grid import AoiGrid
from . import tides

log = logging.getLogger(__name__)

SOURCE = "datum"

VDATUM_URL = "https://vdatum.noaa.gov/vdatumweb/api/convert"
COOPS_DATUMS_URL = ("https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/"
                    "stations/{sid}/datums.json")

# --- the DEM side --------------------------------------------------------------- #
# Which vertical datum each DEM source is referenced to. NAVD88 is the only one that
# needs resolving; MSL_APPROX means "already sea-level referenced" -> offset 0.
DEM_DATUM = {"cudem": "NAVD88", "gmrt": "MSL_APPROX"}

# bathymetry records the DEM that actually ran only in a prose `source` attr. Match it
# strictly: an unrecognized string RAISES rather than being guessed at, because
# guessing wrong here silently biases every water-level pixel by ~1.3 m.
_DEM_PATTERNS = ((re.compile(r"^NCEI CUDEM", re.I), "cudem"),
                 (re.compile(r"^GMRT", re.I), "gmrt"))

# --- VDatum ---------------------------------------------------------------------- #
# Out-of-coverage comes back as HTTP 200 with t_z = -999999 (NOT an error), so a naive
# negation would yield an offset of +999999 m -- a cube where every pixel reads EXPOSED
# with no NaNs at all, i.e. wrong AND healthy-looking. Reject on magnitude.
VDATUM_SENTINEL_ABS = 1e4
# Nothing on Earth has an MSL-to-NAVD88 separation this large; a value past it means we
# have misread the API, not found an unusual coast.
MAX_PLAUSIBLE_OFFSET_M = 5.0

# Sampling: NOT the AoI centroid (VDatum coverage is patchy at point scale -- the Padilla
# Bay centroid returns the sentinel while points 3-6 km away resolve fine), and NOT the
# deepest cells (those are open ocean, tens of km from the estuary and possibly in a
# different tidal zone). Sample the WATERLINE BAND, spread across the AoI.
SAMPLE_BAND_M = (-5.0, 2.0)      # DEM elevation range to draw sample points from
SAMPLE_BLOCKS = 5                # NxN blocks over the AoI; one point per non-empty block
MIN_SAMPLES = 5                  # need this many non-sentinel returns to trust the median
MAX_QUERIES = 25                 # hard cap on VDatum requests per AoI

SPREAD_WARN_M = 0.15             # samples disagree this much -> warn
SPREAD_FAIL_M = 0.5              # ...this much -> the AoI straddles tidal zones; refuse
                                 # a scalar rather than median two zones into one number
                                 # that is wrong everywhere and looks fine.

# --- CO-OPS cross-check ---------------------------------------------------------- #
# A datum's gradient is real over tens of km, so these are deliberately much tighter
# than the tides module's 75/150 km gates (fine for tidal phase, far too loose here).
COOPS_WARN_KM = 10.0
COOPS_MAX_KM = 30.0
XCHECK_WARN_M = 0.15             # VDatum vs CO-OPS disagreement -> loud warning
OVERRIDE_WARN_M = 0.25           # config override vs resolved -> warn
# A >0.5 m override on an already-MSL DEM is provably a NAVD88 number on an MSL DEM.
OVERRIDE_MSL_SANITY_M = 0.5

TIDE_DATUM = "MSL"
TIDAL_DATUM_EPOCH = "1983-2001"  # NTDE that both VDatum LMSL and CO-OPS MSL are on


# --------------------------------------------------------------------------- #
# The single network seam. Everything else in this module is pure.
# --------------------------------------------------------------------------- #
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "coastal_sst_data/1.0 (research; coastal SST pipeline)",
    "Accept": "application/json",
})


def _http_json(url: str, params: dict, *, retries: int = 3, timeout: int = 60) -> dict:
    """GET JSON. Retries transient failures only; 4xx bodies are RETURNED, not raised.

    VDatum reports a wrong region as a 412 whose *body* names the horizontal frame we
    should have used -- that message is the fix, so it must reach the caller. Retrying
    it (as tides._get_json would, 5x with backoff) would burn ~95 s to no purpose.
    """
    last = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            else:
                return r.json()                      # 2xx and 4xx alike: the body is data
        except Exception as exc:                     # network error / bad JSON
            last = exc
        if attempt < retries - 1:
            import time
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request to {url} failed after {retries} attempts: {last}")


# --------------------------------------------------------------------------- #
# Which DEM actually ran
# --------------------------------------------------------------------------- #
def dem_source_of(bathy_ds: xr.Dataset) -> str:
    """The DEM source that actually produced this file, from its `source` attr.

    bathymetry records the winner only as prose ('NCEI CUDEM 1/9" (98% cover...)' /
    'GMRT (topo, max)'). Parse it strictly: an unknown form raises, because a wrong
    guess here biases every water-level pixel rather than failing visibly.
    """
    s = str(bathy_ds.attrs.get("source", "")).strip()
    for pat, name in _DEM_PATTERNS:
        if pat.search(s):
            return name
    raise ValueError(
        f"cannot tell which DEM produced this bathymetry file from its source attr "
        f"{s!r}; expected it to start with 'NCEI CUDEM' or 'GMRT'. The datum offset "
        f"must follow the DEM that actually ran, so this is not guessed at."
    )


# --------------------------------------------------------------------------- #
# Sampling (pure)
# --------------------------------------------------------------------------- #
def sample_points(elev: np.ndarray, g: AoiGrid, *, band=SAMPLE_BAND_M,
                  blocks: int = SAMPLE_BLOCKS, cap: int = MAX_QUERIES):
    """Lon/lat points to query, drawn from the DEM's waterline band, spread over the AoI.

    One point per non-empty block of a `blocks` x `blocks` partition, so the samples are
    spatially stratified rather than clustered in one channel. Within a block we take the
    cell whose elevation is closest to 0 -- the waterline itself, which is both the zone
    the offset is applied to and where VDatum's tidal grids are best defined.

    If no cell falls in the band (e.g. an entirely deep-water AoI) the band is dropped and
    the closest-to-zero finite cell per block is used instead, so we always return points
    when the DEM has any data at all.
    """
    H, W = elev.shape
    finite = np.isfinite(elev)
    in_band = finite & (elev >= band[0]) & (elev <= band[1])
    mask = in_band if in_band.any() else finite
    if not mask.any():
        return []

    lons, lats = g.lonlat_centers()
    ys = np.linspace(0, H, blocks + 1).astype(int)
    xs = np.linspace(0, W, blocks + 1).astype(int)

    pts = []
    for i in range(blocks):
        for j in range(blocks):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            sub = mask[y0:y1, x0:x1]
            if not sub.any():
                continue
            # closest to the waterline within this block
            d = np.where(sub, np.abs(elev[y0:y1, x0:x1]), np.inf)
            r, c = np.unravel_index(int(np.argmin(d)), d.shape)
            pts.append((float(lons[y0 + r, x0 + c]), float(lats[y0 + r, x0 + c])))
    return pts[:cap]


# --------------------------------------------------------------------------- #
# VDatum
# --------------------------------------------------------------------------- #
def _vdatum_regions(lon: float) -> list[str]:
    """Candidate VDatum regions. CUDEM is CONUS-only, so AK/HI/PRVI are unreachable.

    No lat/lon->region bbox table: VDatum's region extents are not published as clean
    boxes and shift between releases, so a hardcoded table would mis-route silently. A
    wrong region 412s loudly instead, and the message names the frame to retry with.
    """
    return ["westcoast", "contiguous"] if lon < -115.0 else ["contiguous"]


def _required_frame(message: str) -> str | None:
    """The horizontal frame a 412 tells us to use ('...should be IGS14 for Tidal')."""
    m = re.search(r"Horizontal Frame should be (\S+)", str(message), re.I)
    return m.group(1) if m else None


def vdatum_point(lon: float, lat: float, region: str, frame: str | None = None) -> dict:
    """One NAVD88->LMSL conversion. Returns {offset, uncertainty, frame} / {error|none}.

    `offset` is None when the point is out of coverage (the -999999 sentinel), which is a
    POINT failure -- try another point, same region. An `error` is a REGION failure --
    try another region. Conflating the two is how a west-coast point ends up silently
    answered by the contiguous tidal grid.
    """
    params = {"s_x": f"{lon:.6f}", "s_y": f"{lat:.6f}", "s_z": "0",
              "region": region, "s_v_frame": "NAVD88", "t_v_frame": "LMSL",
              "s_v_unit": "m", "t_v_unit": "m"}
    if frame:
        # Only the TARGET frame: setting s_h_frame as well makes VDatum error out.
        params["t_h_frame"] = frame
    js = _http_json(VDATUM_URL, params)

    if js.get("errorCode") is not None:
        return {"error": str(js.get("message", "")), "frame": frame}
    try:
        t_z = float(js["t_z"])
    except (KeyError, TypeError, ValueError):
        return {"error": f"no t_z in response: {js!r}", "frame": frame}
    if abs(t_z) > VDATUM_SENTINEL_ABS:              # -999999 -> out of coverage
        return {"offset": None, "uncertainty": None, "frame": frame}

    unc = js.get("uncertainty")
    try:
        unc = float(unc)
    except (TypeError, ValueError):                  # "" on some responses
        unc = None
    # offset = elevation of MSL in the DEM's datum = -(LMSL height of the NAVD88 zero)
    return {"offset": -t_z, "uncertainty": unc, "frame": frame}


def vdatum_offset(points, lon_hint: float) -> dict | None:
    """Median NAVD88->MSL offset over sample points. None if it cannot be resolved.

    Region is PINNED to the first one that answers for this AoI and never switched
    afterwards, so a later out-of-coverage point cannot silently re-route us into a
    different tidal grid.
    """
    if not points:
        return None
    offsets, uncs = [], []
    region_used = frame_used = None

    for region in _vdatum_regions(lon_hint):
        frame = None
        offsets, uncs = [], []
        for (lon, lat) in points:
            res = vdatum_point(lon, lat, region, frame)
            if "error" in res:
                want = _required_frame(res["error"])
                if want and want != frame:           # self-repair: the 412 names the frame
                    frame = want
                    res = vdatum_point(lon, lat, region, frame)
                if "error" in res:
                    log.debug("  vdatum %s: %s", region, res["error"])
                    break                            # region failure -> next region
            if res.get("offset") is None:
                continue                             # point failure -> next point
            offsets.append(float(res["offset"]))
            if res.get("uncertainty") is not None:
                uncs.append(float(res["uncertainty"]))
            region_used, frame_used = region, res.get("frame")
        if offsets:
            break                                    # region answered -> pin it

    if len(offsets) < MIN_SAMPLES:
        log.warning("  VDatum returned only %d/%d usable point(s) (need %d)",
                    len(offsets), len(points), MIN_SAMPLES)
        if not offsets:
            return None

    arr = np.asarray(offsets, dtype="float64")
    offset = float(np.median(arr))
    spread = float(arr.max() - arr.min())
    if abs(offset) > MAX_PLAUSIBLE_OFFSET_M:
        log.error("  VDatum offset %.2f m is implausible (>%.0f m); refusing it",
                  offset, MAX_PLAUSIBLE_OFFSET_M)
        return None
    return {
        "offset_m": offset,
        "spread_m": spread,
        "n_samples": len(offsets),
        "n_points": len(points),
        "uncertainty_m": float(np.median(uncs)) if uncs else None,
        "vdatum_region": region_used,
        "vdatum_frame": frame_used,
    }


# --------------------------------------------------------------------------- #
# CO-OPS (cross-check, and fallback when VDatum has no coverage)
# --------------------------------------------------------------------------- #
def coops_offset(lon: float, lat: float, stations=None) -> dict | None:
    """MSL - NAVD88 from the nearest CO-OPS gauge that publishes both. None if none does.

    Walks outward from the AoI: not every gauge publishes NAVD88, so the nearest one may
    be unusable. Hard-stops at COOPS_MAX_KM -- a gauge farther than that says little about
    this AoI's datum, and a plausible-but-wrong number is worse than none.
    """
    if stations is None:
        stations = tides.fetch_stations()
    ranked = sorted(
        ((tides.haversine_km(lon, lat, s["lon"], s["lat"]), s) for s in stations),
        key=lambda t: t[0])

    for dist, st in ranked:
        if dist > COOPS_MAX_KM:
            return None
        js = _http_json(COOPS_DATUMS_URL.format(sid=st["id"]), {"units": "metric"})
        units = str(js.get("units", "")).lower()
        if units and not units.startswith("meter"):          # never trust feet as metres
            log.debug("  coops %s: units=%r, skipping", st["id"], units)
            continue
        vals = {d.get("name"): d.get("value") for d in js.get("datums", []) or []}
        if vals.get("MSL") is None or vals.get("NAVD88") is None:
            continue                                          # gauge has no NAVD88 tie
        offset = float(vals["MSL"]) - float(vals["NAVD88"])
        if abs(offset) > MAX_PLAUSIBLE_OFFSET_M:
            continue
        if dist > COOPS_WARN_KM:
            log.warning("  nearest NAVD88 gauge (%s) is %.0f km away; datum offset is "
                        "less certain here", st["id"], dist)
        return {"offset_m": offset, "station_id": st["id"], "station_name": st.get("name"),
                "distance_km": round(float(dist), 2), "epoch": js.get("epoch")}
    return None


# --------------------------------------------------------------------------- #
# Resolve one AoI
# --------------------------------------------------------------------------- #
def resolve_aoi(g: AoiGrid, elev: np.ndarray, dem_source: str,
                *, override: float | None = None, stations=None) -> dict:
    """The full resolution chain for one AoI -> the sidecar record.

    Chain: GMRT is 0 by construction (no network). For a NAVD88 DEM, VDatum over the
    stratified waterline samples, cross-checked against the nearest CO-OPS gauge; if
    VDatum has no coverage, CO-OPS alone. A region override wins, but is validated
    against the DEM that actually ran. Unresolved -> 0.0, flagged `unresolved_assumed_zero`.
    """
    datum = DEM_DATUM.get(dem_source, "UNKNOWN")
    rec = {
        "aoi_id": g.name,
        "dem_source": dem_source,
        "dem_vertical_datum": datum,
        "tide_datum": TIDE_DATUM,
        "tidal_datum_epoch": TIDAL_DATUM_EPOCH,
        "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # --- the DEM is already sea-level referenced: 0, no network -------------------- #
    if datum == "MSL_APPROX":
        rec.update(datum_offset_m=0.0, method="gmrt_msl_assumed", status="ok",
                   note="GMRT is already ~sea-level referenced; no offset needed")
        return _apply_override(rec, override, g.name)

    if datum != "NAVD88":
        rec.update(datum_offset_m=0.0, method="unknown_dem_datum",
                   status="unresolved_assumed_zero")
        log.error("  %s: DEM source %r has no known vertical datum; assuming 0.0 m "
                  "-- water level will be biased if it is not MSL-referenced",
                  g.name, dem_source)
        return _apply_override(rec, override, g.name)

    # --- NAVD88 DEM: resolve it --------------------------------------------------- #
    lon_hint = 0.5 * (g.search_bbox[0] + g.search_bbox[2])
    lat_hint = 0.5 * (g.search_bbox[1] + g.search_bbox[3])

    vd = vdatum_offset(sample_points(elev, g), lon_hint)
    try:
        co = coops_offset(lon_hint, lat_hint, stations)
    except Exception as exc:                          # the cross-check must never be fatal
        log.warning("  %s: CO-OPS datum lookup failed (%s)", g.name, exc)
        co = None

    if vd is not None:
        rec.update({k: v for k, v in vd.items() if k != "offset_m"})
        rec.update(datum_offset_m=round(vd["offset_m"], 3),
                   method="vdatum_navd88_to_lmsl", status="ok")
        if vd["spread_m"] > SPREAD_FAIL_M:
            log.error("  %s: VDatum offsets span %.2f m across the AoI (>%.2f m) -- it "
                      "straddles tidal zones; one scalar offset would be wrong everywhere. "
                      "Refusing to resolve; set sources.bathymetry.datum_offset_m to assert one.",
                      g.name, vd["spread_m"], SPREAD_FAIL_M)
            rec.update(datum_offset_m=0.0, status="unresolved_assumed_zero",
                       method="vdatum_spread_too_large")
            return _apply_override(rec, override, g.name)
        if vd["spread_m"] > SPREAD_WARN_M:
            log.warning("  %s: VDatum offsets vary by %.2f m across the AoI",
                        g.name, vd["spread_m"])

        if co is not None:
            rec.update(coops_offset_m=round(co["offset_m"], 3),
                       coops_station_id=co["station_id"],
                       coops_distance_km=co["distance_km"])
            delta = abs(vd["offset_m"] - co["offset_m"])
            rec["crosscheck_delta_m"] = round(delta, 3)
            if delta > XCHECK_WARN_M:
                log.warning("  %s: VDatum (%.3f m) and CO-OPS gauge %s (%.3f m) disagree "
                            "by %.3f m -- check the datum before trusting water level",
                            g.name, vd["offset_m"], co["station_id"], co["offset_m"], delta)
            else:
                log.info("  %s: datum offset %.3f m (VDatum), confirmed by gauge %s (%.3f m)",
                         g.name, vd["offset_m"], co["station_id"], co["offset_m"])
        else:
            log.info("  %s: datum offset %.3f m (VDatum, no gauge to cross-check)",
                     g.name, vd["offset_m"])
        return _apply_override(rec, override, g.name)

    # --- VDatum had no coverage: the gauge is the fallback ------------------------ #
    if co is not None:
        log.warning("  %s: VDatum has no coverage here; using CO-OPS gauge %s (%.0f km) "
                    "-> %.3f m", g.name, co["station_id"], co["distance_km"], co["offset_m"])
        rec.update(datum_offset_m=round(co["offset_m"], 3),
                   method="coops_msl_minus_navd88", status="ok",
                   coops_offset_m=round(co["offset_m"], 3),
                   coops_station_id=co["station_id"],
                   coops_distance_km=co["distance_km"])
        return _apply_override(rec, override, g.name)

    # --- nothing worked ----------------------------------------------------------- #
    rec.update(datum_offset_m=0.0, method="none", status="unresolved_assumed_zero")
    log.error("  %s: COULD NOT RESOLVE the NAVD88->MSL offset (VDatum has no coverage "
              "and no NAVD88 gauge within %.0f km). Assuming 0.0 m -- the water level "
              "will be biased by roughly a metre on the Pacific coast. Set "
              "regions[].sources.bathymetry.datum_offset_m to fix this.",
              g.name, COOPS_MAX_KM)
    return _apply_override(rec, override, g.name)


def _apply_override(rec: dict, override: float | None, aoi: str) -> dict:
    """Honour a region-level override, but validate it against the DEM that ran.

    The override is a user's local assertion and normally wins. The one case where it
    does not: a >0.5 m offset on an already-MSL DEM is provably a NAVD88 number applied
    to the wrong DEM (bathymetry fell back cudem->gmrt underneath it) -- that is the
    silent-1.3 m footgun this whole stage exists to remove, so the resolved value wins.
    """
    if override is None:
        return rec

    resolved = rec.get("datum_offset_m")
    rec["resolved_offset_m"] = resolved
    rec["config_offset_m"] = float(override)

    if (rec["dem_vertical_datum"] == "MSL_APPROX"
            and abs(float(override)) > OVERRIDE_MSL_SANITY_M):
        log.error("  %s: config datum_offset_m=%.3f m looks like a NAVD88 offset, but the "
                  "DEM that actually ran is %s (already ~MSL). Ignoring the override and "
                  "using %.3f m. Remove it, or set dem_source so CUDEM is used here.",
                  aoi, float(override), rec["dem_source"], resolved)
        rec["status"] = "override_rejected_msl_dem"
        return rec

    if resolved is not None and abs(float(override) - float(resolved)) > OVERRIDE_WARN_M:
        log.warning("  %s: config datum_offset_m=%.3f m differs from the resolved %.3f m "
                    "by %.3f m; using the config value.",
                    aoi, float(override), resolved, abs(float(override) - float(resolved)))
    rec.update(datum_offset_m=float(override), method="config_override", status="ok")
    return rec


def resolve_override(project: Project, aoi_name: str) -> float | None:
    """The optional region-level `sources.bathymetry.datum_offset_m`, or None.

    Region-level ONLY (it is in config.REGION_ONLY_OPTIONS, so it has no project-level
    counterpart): the offset follows the DEM, and which DEM wins is decided per AoI at
    runtime, so a project-wide constant can never be right across a study area.
    """
    val = opt(resolve_opts(project, aoi_name, DataProduct.bathymetry), "datum_offset_m")
    return None if val is None else float(val)
