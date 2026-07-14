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

    <output_dir>/INSITU/aligned/<aoi>/<aoi>_insitu.nc   dims (station, time)

The NATIVE sampling interval is kept (6 min for CO-OPS): the assembler needs the
sub-hourly series to match a satellite overpass instant, exactly as `tides` writes a
series that `water_level` later samples.

Usage:
    python -m coastal_sst_data.processes.insitu_ioos --config config.yaml
    python -m coastal_sst_data.processes.insitu_ioos --config config.yaml --aoi hood_canal
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from ..config import DataProduct, Project, load_config
from ..grid import AoiGrid, project_grids
from .. import provenance, store

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
    from urllib.parse import quote

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
# Assemble one AoI -> (station, time)
# --------------------------------------------------------------------------- #
def build_dataset(records: list[dict]) -> xr.Dataset:
    """Per-station series -> one (station, time) Dataset on the union time axis."""
    times = pd.DatetimeIndex(sorted({t for r in records for t in r["df"]["time"]}))
    S, T = len(records), len(times)

    sst = np.full((S, T), np.nan, dtype="float32")
    qc = np.zeros((S, T), dtype="uint8")
    for i, r in enumerate(records):
        df = r["df"].drop_duplicates(subset="time").set_index("time").reindex(times)
        sst[i] = df["value"].to_numpy(dtype="float32")
        if "qc" in df:
            qc[i] = np.nan_to_num(df["qc"].to_numpy(dtype="float32"), nan=0).astype("uint8")

    ds = xr.Dataset(
        {"sst": (("station", "time"), sst), "qc": (("station", "time"), qc)},
        coords={
            "station": np.arange(S, dtype="int32"),
            "time": times,
            "station_id": ("station", [r["id"] for r in records]),
            "station_name": ("station", [r["title"] for r in records]),
            "lat": ("station", np.array([r["lat"] for r in records], "float64")),
            "lon": ("station", np.array([r["lon"] for r in records], "float64")),
            "variable": ("station", [r["var"] for r in records]),
        },
    )
    ds["sst"].attrs.update(units="degC", long_name="in-situ water temperature")
    ds["qc"].attrs.update(long_name="QARTOD aggregate flag",
                          flag_values="1 pass, 2 not_evaluated, 3 suspect, 4 fail, 9 missing")
    return ds


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return store.write_netcdf(ds, out_dir / f"{aoi_id}_insitu.nc",
                              encoding={"sst": {"zlib": True, "complevel": 4},
                                        "qc": {"zlib": True, "complevel": 4}})


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    ds_cfg = eff["ds"]
    out_root, overwrite = eff["out_dir"], eff["overwrite"]
    variables = ds_cfg["variables"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    names = list(grids)
    if only_aoi:
        missing = set(only_aoi) - set(names)
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        names = [n for n in names if n in only_aoi]

    for name in names:
        g = grids[name]
        out_path = out_root / name / f"{name}_insitu.nc"
        if store.done(out_path, store.REQUIRED_VARS["INSITU"], overwrite=overwrite):
            log.info("=== %s: %s exists, skipping ===", name, out_path.name)
            continue

        log.info("=== AOI: %s ===", name)
        stations = find_stations(g.search_bbox, start, end, ds_cfg["pad_deg"],
                                 searchfor=variables[0])
        allow, deny = set(ds_cfg["stations"]), set(ds_cfg["exclude_stations"])
        if allow:
            stations = [s for s in stations if s["id"] in allow]
        stations = [s for s in stations if s["id"] not in deny]
        log.info("  %d candidate station(s)", len(stations))
        if dry_run:
            for s in stations:
                log.info("  [dry-run] %s | %s", s["id"], s["title"])
            continue

        records, empty = [], []
        for s in stations:
            available = station_variables(s["id"])
            var = pick_variable(available, variables)
            if var is None:                       # would be an HTTP 400 -- never asked
                empty.append(f"{s['id']} (no temperature variable)")
                continue
            df = fetch_station(s["id"], var, start, end, ds_cfg["qc_flags"],
                               ds_cfg["max_sensor_depth_m"], available)
            if df is None or df.empty:            # advertised it, never reported it
                empty.append(f"{s['id']} (no QC-passing values)")
                continue
            records.append({"id": s["id"], "title": s["title"], "var": var, "df": df,
                            "lat": float(df["latitude"].median()),
                            "lon": float(df["longitude"].median())})
            log.info("  %-28s %-24s %5d obs  %.1f-%.1f degC", s["id"], var, len(df),
                     df["value"].min(), df["value"].max())

        # A station that reports nothing must be VISIBLE: an empty in-situ channel that
        # looks like "no buoys here" is the failure mode this product cannot afford.
        for e in empty:
            log.warning("  dropped %s", e)
        if not records:
            log.warning("  %s: no usable in-situ stations; nothing written", name)
            continue

        ds = build_dataset(records)
        ds.attrs.update(aoi_id=name, source="IOOS Sensors ERDDAP", erddap=ERDDAP,
                        qc_flags=str(ds_cfg["qc_flags"]), **provenance.stamp(eff))
        log.info("  wrote %s  (%d station(s), %d timestep(s))",
                 write_output(ds, out_root / name, name).name,
                 ds.sizes["station"], ds.sizes["time"])
    log.info("Done.")


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _opt(opts, name, default):
    return getattr(opts, name, default) if opts is not None else default


def _build_eff(project: Project) -> dict:
    opts = project.products.get(DataProduct.insitu)
    if opts is None:
        raise ValueError("insitu is not a selected product in this config")

    ds_cfg = {
        "variables": list(_opt(opts, "variables", DEFAULT_VARIABLES)),
        "qc_flags": [int(f) for f in _opt(opts, "qc_flags", DEFAULT_QC_FLAGS)],
        "max_sensor_depth_m": _opt(opts, "max_sensor_depth_m", DEFAULT_MAX_SENSOR_DEPTH_M),
        "pad_deg": float(_opt(opts, "pad_deg", DEFAULT_PAD_DEG)),
        "stations": list(_opt(opts, "stations", []) or []),
        "exclude_stations": list(_opt(opts, "exclude_stations", []) or []),
    }
    return {
        "config_sha256": project.config_sha256,
        "ds": ds_cfg,
        "out_dir": Path(project.output_dir) / "INSITU" / "aligned",
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire IOOS in-situ observations. Entry point for pipeline.py."""
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    run(eff, grids, aois, dry_run)


def main():
    ap = argparse.ArgumentParser(description="coastal_sst_data IOOS in-situ acquisition.")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Process only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true", help="refetch existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="List stations only.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    project = load_config(args.config)
    acquire(project, aois=args.aoi, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
