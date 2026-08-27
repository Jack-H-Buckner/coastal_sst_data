#!/usr/bin/env python3
"""
coastal_sst_data -- in-situ observations from IOOS (ERDDAP).

The cube's only GROUND TRUTH. Every other channel is modelled (met, CMEMS, tides) or
remotely sensed (ECOSTRESS, Landsat, MODIS, MUR); this is what a thermometer in the water
actually read. The datacube writes each station's value into the grid cell the station
sits in, and -- the point of the exercise -- at the INSTANT each satellite flew, so a
scene can be validated against a buoy pixel-for-pixel and minute-for-minute.

  * Where it comes from: the IOOS Sensors ERDDAP (https://erddap.sensors.ioos.us/erddap),
    one server aggregating NDBC, NOAA CO-OPS, CDIP and the IOOS regional associations --
    so most of North America is one query. PUBLIC: no credentials.
  * What it measures: water temperature (`sea_water_temperature`, falling back per station
    to `sea_surface_temperature`), quality-flagged with QARTOD.

Two traps in this API, both of which produce a silently EMPTY in-situ channel if ignored,
and both of which are guarded here:

  1. Asking a station for a variable it does not have is an HTTP 400 -- not an empty
     result. So each station's real variable list is read (/info/<id>) BEFORE querying it.
  2. A station can ADVERTISE a variable and never report it. NDBC 46120 exposes both
     temperature names and returns all-NaN, QARTOD flag 2, for a whole month (it is a wave
     buoy with no thermometer). So a station whose fetched series has ZERO finite values is
     dropped AND LOGGED BY NAME -- an empty channel must be loud, not silent.

QARTOD flags: 1 pass, 2 not-evaluated, 3 suspect, 4 fail, 9 missing. The default keeps
{1, 2}: flag 2 is what stations that do not run QARTOD emit, and demanding flag 1 would
throw away much of the network.

This module is ONE SOURCE behind the shared in-situ loop: it discovers and fetches, and
`insitu_acquire` owns the union time axis, the skip guard, the write and the report, so
this and the user's own CSVs STACK into one cube. Its output lands at

    <output_dir>/INSITU/ioos/aligned/<aoi>/<aoi>_insitu.nc   dims (station, time)

The NATIVE sampling interval is kept (6 min for CO-OPS): the assembler needs the
sub-hourly series to match a satellite overpass instant, exactly as `tides` writes a
series that `water_level` later samples.

Usage:
    python -m coastal_sst_data.processes.insitu_ioos --config config.yaml
    python -m coastal_sst_data.processes.insitu_ioos --config config.yaml --aoi hood_canal
"""

from __future__ import annotations

import io
import logging
import time
from urllib.parse import quote

import pandas as pd
import requests

from .. import entry

log = logging.getLogger(__name__)

SOURCE = "ioos"
ERDDAP = "https://erddap.sensors.ioos.us/erddap"

# Preference order: the first name a station actually exposes wins.
DEFAULT_VARIABLES = ["sea_water_temperature", "sea_surface_temperature"]
DEFAULT_QC_FLAGS = [1, 2]          # QARTOD pass + not-evaluated
DEFAULT_MAX_SENSOR_DEPTH_M = 5.0   # ignore deep sensors on profiling moorings
DEFAULT_PAD_DEG = 0.0              # extra search padding around the AoI bbox

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "coastal_sst_data/1.0 (research; coastal SST pipeline)",
    "Accept": "application/json",
})


# --------------------------------------------------------------------------- #
# The network seam. Everything else in this module is pure.
# --------------------------------------------------------------------------- #
def _get(url: str, params: dict | None = None, *, retries: int = 3, timeout: int = 120) -> str:
    """GET, returning the raw body. 4xx bodies are RETURNED (ERDDAP explains itself in
    them); only 5xx/timeouts are retried.

    `params` is for ordinary key=value queries. tabledap's projection syntax
    (`?var1,var2&time>=…`) is not key=value at all, so those callers pre-build the query
    string into `url` instead.
    """
    last = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            else:
                return r.text
        except Exception as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ERDDAP request failed after {retries} attempts: {last}")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_stations(bbox, start, end, pad=DEFAULT_PAD_DEG, searchfor="sea_water_temperature"):
    """Station dataset ids overlapping the AoI window -> [{id, title}].

    ERDDAP's advanced search covers every provider on the server at once (NDBC, CO-OPS,
    CDIP, the regional associations), so one request finds them all.
    """
    import json

    w, s, e, n = bbox
    body = _get(f"{ERDDAP}/search/advanced.json", {
        "searchFor": searchfor, "protocol": "tabledap",
        "minLon": w - pad, "maxLon": e + pad, "minLat": s - pad, "maxLat": n + pad,
        "minTime": f"{start}T00:00:00Z", "maxTime": f"{end}T23:59:59Z",
        "page": 1, "itemsPerPage": 1000,
    })
    if body.lstrip().startswith("Error") or '"table"' not in body:
        log.info("  no stations found in the AoI window")
        return []
    tbl = json.loads(body)["table"]
    cols = tbl["columnNames"]
    i_id, i_title = cols.index("Dataset ID"), cols.index("Title")
    return [{"id": r[i_id], "title": r[i_title]} for r in tbl["rows"]]


def station_metadata(bbox, pad=DEFAULT_PAD_DEG) -> dict[str, dict]:
    """Every dataset overlapping the box -> {id: {lat, lon, start, end, mobile, title, bbox}}.

    ERDDAP's `allDatasets` table is the server's own catalogue of itself, so ONE request
    returns position, record span and platform class for the whole box -- the three things
    `find_stations` does not tell you. Without it a station's position is only known after its
    data is fetched, which is the cost this whole module exists to avoid.

    INTERSECTION, NOT CONTAINMENT, in the constraints -- `maxLongitude >= W`, not
    `minLongitude >= W`. This is the same trap `insitu_cmems.select_files` documents: the
    containment form is the one that reads naturally and it silently drops every trajectory,
    because a glider's bounds span degrees and never fall inside a coastal box. Measured on
    the live server: a Puget Sound box returns 205 TimeSeries and 7 TrajectoryProfile with
    intersection, and zero trajectories with containment.

    `cdm_data_type` is the platform class, declared by the provider: `TimeSeries` /
    `TimeSeriesProfile` are moorings and tide gauges, `Trajectory` / `TrajectoryProfile` are
    gliders and ship tracks.
    """
    import json

    w, s, e, n = bbox
    w, s, e, n = w - pad, s - pad, e + pad, n + pad
    cols = ("datasetID,cdm_data_type,title,minLongitude,maxLongitude,"
            "minLatitude,maxLatitude,minTime,maxTime")
    # tabledap's projection+constraint syntax is not key=value, so it is built into the URL
    # rather than handed to requests as params -- exactly as `fetch_station` does.
    query = (f"{quote(cols)}"
             f"&maxLongitude%3E={w}&minLongitude%3C={e}"
             f"&maxLatitude%3E={s}&minLatitude%3C={n}")
    body = _get(f"{ERDDAP}/tabledap/allDatasets.json?{query}")
    if body.lstrip().startswith("Error") or '"table"' not in body:
        log.info("  allDatasets returned no rows for this box")
        return {}

    tbl = json.loads(body)["table"]
    i = {name: k for k, name in enumerate(tbl["columnNames"])}
    out: dict[str, dict] = {}
    for r in tbl["rows"]:
        kind = str(r[i["cdm_data_type"]] or "")
        lo_lon, hi_lon = r[i["minLongitude"]], r[i["maxLongitude"]]
        lo_lat, hi_lat = r[i["minLatitude"]], r[i["maxLatitude"]]
        if lo_lon is None or lo_lat is None:
            continue                                  # nothing placeable in this row
        mobile = kind.startswith("Trajectory")
        out[str(r[i["datasetID"]])] = {
            "title": str(r[i["title"]] or r[i["datasetID"]]),
            "lat": (float(lo_lat) + float(hi_lat)) / 2.0,
            "lon": (float(lo_lon) + float(hi_lon)) / 2.0,
            "start": r[i["minTime"]], "end": r[i["maxTime"]],
            "mobile": mobile,
            # Only worth carrying when it has real extent: a mooring's bounds are a point.
            "bbox": ((float(lo_lon), float(lo_lat), float(hi_lon), float(hi_lat))
                     if mobile else None),
        }
    return out


def stations(bbox, start, end, cfg: dict) -> list:
    """The discovery seam: every IOOS temperature platform in `bbox` -> [Station, ...].

    Two requests, and each answers half the question. `find_stations` is the only query that
    knows which datasets carry a TEMPERATURE variable -- so using it means the map shows what
    `fetch_aoi` would actually acquire, not every barometer on the coast. `station_metadata`
    then says where each one is and when it ran.

    `bbox` arrives ALREADY GROWN by the caller's halo, so no padding is applied here; see
    `insitu_stations.discover`.
    """
    from .insitu_stations import Station

    variables = cfg.get("variables") or DEFAULT_VARIABLES
    found = find_stations(bbox, start, end, 0.0, searchfor=variables[0])
    allow, deny = set(cfg.get("stations") or []), set(cfg.get("exclude_stations") or [])
    if allow:
        found = [s for s in found if s["id"] in allow]
    found = [s for s in found if s["id"] not in deny]
    if not found:
        return []

    meta = station_metadata(bbox)
    out, unplaced = [], []
    for s in found:
        m = meta.get(s["id"])
        if m is None:
            # The search found it and the catalogue did not describe it. Rare, and never
            # silent: an in-situ platform that vanishes without being named is the failure
            # mode this module refuses to have.
            unplaced.append(s["id"])
            continue
        out.append(Station(id=s["id"], title=s["title"] or m["title"], source=SOURCE,
                           lat=m["lat"], lon=m["lon"], start=m["start"], end=m["end"],
                           mobile=m["mobile"], bbox=m["bbox"]))
    if unplaced:
        log.warning("  %d IOOS station(s) have no position in allDatasets and are missing "
                    "from the map: %s", len(unplaced), ", ".join(unplaced[:8]))
    return out


def station_variables(station_id: str) -> set[str]:
    """The variables a station ACTUALLY exposes.

    Read before querying: asking for a variable a station lacks is an HTTP 400, so this
    is what keeps a mixed set of providers from failing the whole AoI.
    """
    import json

    body = _get(f"{ERDDAP}/info/{station_id}/index.json", {})
    if '"table"' not in body:
        return set()
    return {r[1] for r in json.loads(body)["table"]["rows"] if r[0] == "variable"}


def pick_variable(available: set[str], preferred) -> str | None:
    """The first preferred variable this station has (naming is not uniform across
    providers -- some expose only sea_surface_temperature)."""
    for v in preferred:
        if v in available:
            return v
    return None


# --------------------------------------------------------------------------- #
# Fetch one station
# --------------------------------------------------------------------------- #
def fetch_station(station_id: str, var: str, start, end, qc_flags, max_depth_m,
                  available: set[str] | None = None) -> pd.DataFrame | None:
    """One station's QC-passing series -> DataFrame[time, lat, lon, value] or None.

    None means 'no usable data': either the request failed, or -- the trap -- the station
    advertises the variable but never reports it. The caller logs the drop by name.

    Only columns the station actually has are requested: `z` and the QC flag are absent on
    some providers, and asking for either would be an HTTP 400 that kills the station.
    """
    available = available if available is not None else {"z", f"{var}_qc_agg"}
    cols = ["time", "latitude", "longitude"]
    if "z" in available:
        cols.append("z")
    cols.append(var)
    if f"{var}_qc_agg" in available:
        cols.append(f"{var}_qc_agg")
    # tabledap's query is a projection + constraints, not key=value pairs, so it is built
    # here rather than handed to requests as params.
    query = (quote(",".join(cols)) +
             f"&time%3E={start}T00:00:00Z&time%3C={end}T23:59:59Z")
    body = _get(f"{ERDDAP}/tabledap/{station_id}.csvp?{query}")
    if body.lstrip().startswith("Error"):
        log.debug("  %s: %s", station_id, body.splitlines()[:2])
        return None

    df = pd.read_csv(io.StringIO(body))
    if df.empty:
        return None
    # csvp gives 'name (units)' headers; strip the units back off.
    df.columns = [c.split(" (")[0] for c in df.columns]
    if var not in df:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.rename(columns={var: "value", f"{var}_qc_agg": "qc"})
    if "qc" in df:
        df = df[df["qc"].isin(list(qc_flags))]
    if "z" in df and max_depth_m is not None:
        # z is the sensor depth; a profiling mooring's deep sensors are not surface truth.
        df = df[~(df["z"].abs() > float(max_depth_m))]

    df = df.dropna(subset=["time", "value"])
    if df.empty:                       # advertised the variable, reported nothing
        return None
    return df[["time", "latitude", "longitude", "value"] +
              (["qc"] if "qc" in df else [])].sort_values("time")


# --------------------------------------------------------------------------- #
# The source seam: one AoI's stations -> records
# --------------------------------------------------------------------------- #
def fetch_aoi(g, start, end, cfg: dict, dry_run: bool = False) -> list[dict]:
    """Every usable IOOS station overlapping this AoI, as acquisition records.

    The seam `insitu_acquire` calls (see its docstring for the record contract). Everything
    downstream of this -- the union time axis, the write, the report -- is shared with every
    other network, so this function's whole job is: discover, filter, fetch, and SAY WHAT IT
    DROPPED.
    """
    variables = cfg["variables"]
    stations = find_stations(g.search_bbox, start, end, cfg["pad_deg"],
                             searchfor=variables[0])
    allow, deny = set(cfg["stations"]), set(cfg["exclude_stations"])
    if allow:
        stations = [s for s in stations if s["id"] in allow]
    stations = [s for s in stations if s["id"] not in deny]
    log.info("  %d candidate station(s)", len(stations))
    if dry_run:
        for s in stations:
            log.info("  [dry-run] %s | %s", s["id"], s["title"])
        return []

    records, empty = [], []
    for s in stations:
        available = station_variables(s["id"])
        var = pick_variable(available, variables)
        if var is None:                       # would be an HTTP 400 -- never asked
            empty.append(f"{s['id']} (no temperature variable)")
            continue
        df = fetch_station(s["id"], var, start, end, cfg["qc_flags"],
                           cfg["max_sensor_depth_m"], available)
        if df is None or df.empty:            # advertised it, never reported it
            empty.append(f"{s['id']} (no QC-passing values)")
            continue
        records.append({"id": s["id"], "title": s["title"], "var": var, "df": df,
                        "lat": float(df["latitude"].median()),
                        "lon": float(df["longitude"].median())})
        log.info("  %-28s %-24s %5d obs  %.1f-%.1f degC", s["id"], var, len(df),
                 df["value"].min(), df["value"].max())

    # A station that reports nothing must be VISIBLE: an empty in-situ channel that looks
    # like "no buoys here" is the failure mode this product cannot afford.
    for e in empty:
        log.warning("  dropped %s", e)
    return records


def main():
    """IOOS is acquired through the shared loop; run that, narrowed to this source."""
    from . import insitu_acquire

    entry.process_main(
        lambda project, **kw: insitu_acquire.acquire(project, source=SOURCE, **kw),
        "coastal_sst_data IOOS in-situ acquisition.")


if __name__ == "__main__":
    main()
