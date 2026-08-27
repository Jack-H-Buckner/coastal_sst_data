#!/usr/bin/env python3
"""
coastal_sst_data -- in-situ observations the USER provides, as long-format CSVs.

The other in-situ source (insitu_ioos) discovers public buoys. This one reads the
thermometers you deployed yourself: a mooring string, a hand logger, an intertidal array --
data that exists in no public network and that is, very often, the only ground truth in the
AoI you actually care about. It STACKS with IOOS rather than replacing it (see
insitu_acquire): a cube can carry both.

LONG FORMAT: one row per observation.

    station_id,time,latitude,longitude,value,z
    mooring_a,2026-06-01T18:00:00Z,45.52,-123.92,11.8,1.0
    mooring_a,2026-06-01T18:10:00Z,45.52,-123.92,11.9,1.0
    mooring_b,2026-06-01T18:00:00Z,45.48,-123.90,12.4,0.5

ROWS ARE GROUPED BY `station_id`, so ONE FILE MAY HOLD ANY NUMBER OF STATIONS -- each
becomes its own platform, with its own position, and lands in its own pixel. That is the
normal case, not an edge case: an export from a logger fleet is one file.

Column NAMES are configurable (`columns:`), because asking a user to rewrite files they
already have is a good way to have the feature go unused. Only the four required fields
must be present under SOME name.

TIME imposes no format -- pandas infers it, so ISO-8601 ("2026-06-01T18:00:00Z") and
"2026-06-01 11:00:00" both read. A stamp CARRYING AN OFFSET is honoured and `time_zone` is
moot for it; a NAIVE stamp means `time_zone` (default UTC). That last assumption is the one
most likely to be wrong in a user's file -- a logger writes local time and says so nowhere --
so it is configurable rather than hard-coded. Times are stored naive UTC, matching
insitu_ioos, and a row whose stamp will not parse is counted and dropped, never guessed at.

DEPTH: `z` is the sensor's depth in metres, kept when it is within `max_sensor_depth_m`
(default 5) -- a profiling mooring's deep sensors are not surface truth. The comparison is on
`abs(z)`, so it does not matter which sign convention the file uses, and a row with no `z`
survives (an unstated depth is not evidence of a deep one). But `z` only FILTERS: it is not
carried into the cube, whose in-situ model has no depth dimension. So the filter must leave
ONE sensor per instant -- if several depths survive at the same timestamp, one row is kept
arbitrarily and the rest are dropped, and `_reject_merged_stations` below cannot catch it
because a mooring's depths all share one position.

Two things this module is deliberately fussy about, because both fail silently otherwise:

  * A `units` mistake is invisible. 54 degF read as degC is a plausible sea temperature, and
    nothing downstream would ever flag it. So conversions are explicit and the result is
    range-checked, loudly.
  * Zero matching files is a FAILURE, not an empty result. An in-situ channel that is empty
    because a path was mistyped looks exactly like one that is empty because there are no
    thermometers here.

A platform that MOVES between observations is caught downstream by insitu_acquire's drift
guard -- most often because a multi-station file's `station_id` column was never mapped.
"""

from __future__ import annotations

import glob as _glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import insitu

log = logging.getLogger(__name__)

SOURCE = "csv"

# Canonical field -> the column name assumed when `columns:` does not remap it.
DEFAULT_COLUMNS = {
    "station_id": "station_id",
    "station_name": "station_name",
    "time": "time",
    "latitude": "latitude",
    "longitude": "longitude",
    "value": "value",
    "z": "z",
    "qc": "qc",
}
REQUIRED_FIELDS = ("time", "latitude", "longitude", "value")

# Converters onto degC, keyed by the spellings people actually write.
_UNITS = {
    "degc": lambda v: v, "c": lambda v: v, "celsius": lambda v: v,
    "degree_celsius": lambda v: v,
    "degf": lambda v: (v - 32.0) * 5.0 / 9.0, "f": lambda v: (v - 32.0) * 5.0 / 9.0,
    "fahrenheit": lambda v: (v - 32.0) * 5.0 / 9.0,
    "k": lambda v: v - 273.15, "kelvin": lambda v: v - 273.15,
}
DEFAULT_UNITS = "degC"

# Not a QC bar -- a units tripwire. Sea-surface temperature outside this is possible in
# principle and overwhelmingly means degF or K read as degC in practice.
SANE_MIN_C, SANE_MAX_C = -5.0, 45.0

DEFAULT_QC_FLAG = 2        # QARTOD "not evaluated" -- what a file with no qc column asserts


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def resolve_paths(path) -> list[Path]:
    """`path` -> the CSV files it names: one file, every `*.csv` in a directory, or a glob.

    Returns them sorted, so a run is reproducible and a multi-file fleet loads in a stable
    order. An empty result is the CALLER's error to report -- see `fetch_aoi`.
    """
    p = Path(str(path)).expanduser()
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    if p.exists():
        return [p]
    return sorted(Path(m) for m in _glob.glob(str(p)))       # a glob pattern, or nothing


# --------------------------------------------------------------------------- #
# One file -> canonical columns
# --------------------------------------------------------------------------- #
def column_map(columns: dict | None) -> dict[str, str]:
    """Canonical field -> the column name to read it from."""
    out = dict(DEFAULT_COLUMNS)
    for field, name in (columns or {}).items():
        if field not in DEFAULT_COLUMNS:
            raise ValueError(
                f"insitu.columns: {field!r} is not a field this loader reads. "
                f"Choose from {', '.join(sorted(DEFAULT_COLUMNS))}.")
        out[field] = str(name)
    return out


def _parse_times(values: pd.Series, tz) -> pd.Series:
    """Timestamps -> naive UTC, matching what insitu_ioos stores.

    A stamp carrying an offset is honoured. A NAIVE stamp means `tz` (default UTC) -- which
    is the assumption most likely to be wrong in a user's file, so it is configurable rather
    than hard-coded.
    """
    name = str(tz or "UTC")
    if name.upper() in ("UTC", "GMT", "Z"):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    else:
        naive = pd.to_datetime(values, errors="coerce")
        if getattr(naive.dtype, "tz", None) is None:
            parsed = naive.dt.tz_localize(name, ambiguous="NaT", nonexistent="NaT") \
                          .dt.tz_convert("UTC")
        else:                                    # already carried offsets; tz is moot
            parsed = naive.dt.tz_convert("UTC")
    return parsed.dt.tz_localize(None)


def read_file(path: Path, cfg: dict) -> pd.DataFrame:
    """One CSV -> DataFrame[station_id, station_name, time, latitude, longitude, value, qc, z].

    Everything that can go wrong with a user's file is turned into an exception naming the
    FILE, or a warning with a COUNT. Nothing is dropped quietly.
    """
    cols = column_map(cfg.get("columns"))
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [f for f in REQUIRED_FIELDS if cols[f] not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing required column(s) "
            f"{', '.join(f'{f} (expected {cols[f]!r})' for f in missing)}. "
            f"Columns present: {', '.join(map(str, df.columns))}. "
            "Use `insitu.columns` to map your names onto the expected ones.")

    out = pd.DataFrame({
        "time": _parse_times(df[cols["time"]], cfg.get("time_zone")),
        "latitude": pd.to_numeric(df[cols["latitude"]], errors="coerce"),
        "longitude": pd.to_numeric(df[cols["longitude"]], errors="coerce"),
        "value": pd.to_numeric(df[cols["value"]], errors="coerce"),
    })

    # Station identity. With no id column the whole file is ONE platform -- which is right
    # for a single-logger export and disastrous for a multi-station one, so the drift guard
    # downstream names this as a likely cause when a "platform" turns out to span km.
    if cols["station_id"] in df.columns:
        out["station_id"] = df[cols["station_id"]].astype(str).str.strip()
    else:
        out["station_id"] = str(cfg.get("default_station_id") or path.stem)
    out["station_name"] = (df[cols["station_name"]].astype(str)
                           if cols["station_name"] in df.columns else out["station_id"])

    if cols["z"] in df.columns:
        out["z"] = pd.to_numeric(df[cols["z"]], errors="coerce")
    if cols["qc"] in df.columns:
        out["qc"] = pd.to_numeric(df[cols["qc"]], errors="coerce")

    # --- units ------------------------------------------------------------- #
    units = str(cfg.get("units") or DEFAULT_UNITS)
    convert = _UNITS.get(units.strip().lower())
    if convert is None:
        raise ValueError(f"{path.name}: insitu.units {units!r} is not recognised; "
                         f"choose from degC, degF, K.")
    out["value"] = convert(out["value"])

    # --- report what was dropped or looks wrong ---------------------------- #
    bad_time = int(out["time"].isna().sum())
    if bad_time:
        log.warning("  %s: %d row(s) have an unparseable time; dropped", path.name, bad_time)
    out = out.dropna(subset=["time", "latitude", "longitude", "value"])

    finite = out["value"].to_numpy(dtype="float64")
    odd = int(np.count_nonzero((finite < SANE_MIN_C) | (finite > SANE_MAX_C)))
    if odd:
        log.warning("  %s: %d value(s) fall outside %g..%g degC after reading them as %s -- "
                    "check `insitu.units`", path.name, odd, SANE_MIN_C, SANE_MAX_C, units)
    return out


def _reject_merged_stations(sid: str, grp: pd.DataFrame) -> None:
    """Fail if one 'platform' holds the same instant at two different POSITIONS.

    That is not a platform, it is several stations merged into one -- almost always because a
    multi-station file's `station_id` column was never mapped. The drift guard downstream
    cannot catch this case: synchronised loggers share a time base, so de-duplicating by time
    would throw the other stations' rows away BEFORE anything measured how far apart they
    were, leaving one station's data wearing the whole file's name. Silent, and wrong rather
    than absent.
    """
    dup = grp[grp.duplicated("time", keep=False)]
    if dup.empty:
        return
    spread = dup.groupby("time")[["latitude", "longitude"]].nunique()
    if not (spread > 1).any().any():
        return                                   # repeated rows at one place: harmless
    n = int((spread > 1).any(axis=1).sum())
    raise ValueError(
        f"platform {sid!r} reports {n} instant(s) from more than one position -- these are "
        "SEVERAL STATIONS merged into one. Map the `station_id` column via `insitu.columns` "
        "so each station keeps its own identity and pixel.")


# --------------------------------------------------------------------------- #
# The discovery seam: what is in the user's files, near this AoI
# --------------------------------------------------------------------------- #
def stations(bbox, start, end, cfg: dict) -> list:
    """The discovery seam: every platform in the user's files inside `bbox` -> [Station, ...].

    The one EXACT source of the three -- the files are local, so this reads real positions and
    real timestamps rather than a catalogue's claim about them.

    Deliberately NOT `insitu.station_pixels(...)["inside"]`, which is how `fetch_aoi` decides
    what belongs to an AoI. That test asks "does this land on the grid", and the whole point of
    the map is to show the platforms that DO NOT -- the ones a small move of the box would
    capture. A plain bbox test is the right one here.

    `mobile` is measured drift against the same threshold the two products partition on
    (`insitu_acquire.platform_drift_m` > one grid cell by default), so a platform drawn small
    here is exactly the one `insitu_mobile` would claim.
    """
    from . import insitu_acquire
    from .insitu_stations import Station

    if not cfg.get("path"):
        return []                            # csv configured but given no path: nothing to show
    files = resolve_paths(cfg["path"])
    if not files:
        raise ValueError(f"insitu.path {cfg['path']!r} matched no CSV files.")

    df = pd.concat([read_file(f, cfg) for f in files], ignore_index=True)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)

    w, s, e, n = bbox
    max_drift = cfg.get("max_position_drift_m")
    if max_drift is None:
        max_drift = cfg.get("resolution_m") or 100.0
    allow, deny = set(cfg.get("stations") or []), set(cfg.get("exclude_stations") or [])

    out = []
    for sid, grp in df.groupby("station_id", sort=True):
        if (allow and str(sid) not in allow) or str(sid) in deny:
            continue
        grp = grp.dropna(subset=["time", "latitude", "longitude"])
        if grp.empty:
            continue
        lat, lon = float(grp["latitude"].median()), float(grp["longitude"].median())
        if not (s <= lat <= n and w <= lon <= e):
            continue
        # The record must OVERLAP the project window -- but the label carries the platform's
        # whole record, which is the more useful fact when deciding on a window.
        first, last = grp["time"].min(), grp["time"].max()
        if last < lo or first >= hi:
            continue
        names = grp["station_name"].dropna().unique()
        out.append(Station(
            id=str(sid), title=str(names[0]) if len(names) else str(sid), source=SOURCE,
            lat=lat, lon=lon,
            start=first.strftime("%Y-%m-%d"), end=last.strftime("%Y-%m-%d"),
            mobile=insitu_acquire.platform_drift_m(grp) > float(max_drift)))
    return out


# --------------------------------------------------------------------------- #
# The source seam: one AoI's platforms -> records
# --------------------------------------------------------------------------- #
def fetch_aoi(g, start, end, cfg: dict, dry_run: bool = False) -> list[dict]:
    """Every platform in the user's files that belongs to this AoI, as acquisition records.

    The same seam `insitu_ioos.fetch_aoi` implements; `insitu_acquire` owns everything after
    it. One file can serve every AoI in the project -- each AoI keeps the platforms that fall
    inside its own grid.
    """
    if not cfg.get("path"):
        raise ValueError(
            "the `csv` in-situ source needs `insitu.path` -- the file, directory or glob "
            "holding your observations.")
    files = resolve_paths(cfg["path"])
    if not files:
        # An empty in-situ channel because a path was mistyped looks exactly like one because
        # there are no thermometers here. Never let those be the same outcome.
        raise ValueError(f"insitu.path {cfg['path']!r} matched no CSV files.")
    log.info("  %d file(s) from %s", len(files), cfg["path"])

    df = pd.concat([read_file(f, cfg) for f in files], ignore_index=True)

    # The project window, inclusive of the end DAY (the config's dates are days, not instants).
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    df = df[(df["time"] >= lo) & (df["time"] < hi)]

    # z (sensor depth): a profiling mooring's deep sensors are not surface truth.
    max_depth = cfg.get("max_sensor_depth_m")
    if "z" in df and max_depth is not None:
        df = df[~(df["z"].abs() > float(max_depth))]

    # qc: filter only when the user says what passing means. With no qc column at all the
    # rows assert "not evaluated" (QARTOD 2), which the default qc_flags accepts.
    passing = cfg.get("qc_pass_values")
    if "qc" in df and passing:
        df = df[df["qc"].isin([float(v) for v in passing])]
    if "qc" not in df:
        df = df.assign(qc=float(DEFAULT_QC_FLAG))

    allow, deny = set(cfg.get("stations") or []), set(cfg.get("exclude_stations") or [])
    records, outside = [], []
    for sid, grp in df.groupby("station_id", sort=True):
        if (allow and sid not in allow) or sid in deny:
            continue
        grp = grp.sort_values("time")
        _reject_merged_stations(sid, grp)
        grp = grp.drop_duplicates(subset="time")
        if grp.empty:
            continue
        lat, lon = float(grp["latitude"].median()), float(grp["longitude"].median())
        # Placed at the position it will actually occupy in the cube, so a platform kept here
        # cannot be silently dropped by the assembler for being off-grid.
        [place] = insitu.station_pixels([lon], [lat], g)
        if not place["inside"]:
            outside.append(sid)
            continue
        names = grp["station_name"].dropna().unique()
        if len(names) > 1:
            log.warning("  platform %s has %d different station_name values; using %r "
                        "(the id is the identity, the name is a label)",
                        sid, len(names), names[0])
        records.append({
            "id": str(sid), "title": str(names[0]) if len(names) else str(sid),
            "var": "value", "lat": lat, "lon": lon,
            "df": grp[["time", "latitude", "longitude", "value", "qc"]].reset_index(drop=True),
        })
        log.info("  %-28s %5d obs  %.1f-%.1f degC", sid, len(grp),
                 grp["value"].min(), grp["value"].max())

    if outside:
        log.info("  %d platform(s) fall outside this AoI: %s",
                 len(outside), ", ".join(map(str, outside[:8])))
    if dry_run:
        for r in records:
            log.info("  [dry-run] %s | %s | %.4f, %.4f | %d obs | %s .. %s",
                     r["id"], r["title"], r["lat"], r["lon"], len(r["df"]),
                     r["df"]["time"].iloc[0], r["df"]["time"].iloc[-1])
        return []
    return records
