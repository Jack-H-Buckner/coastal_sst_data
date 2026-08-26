#!/usr/bin/env python3
"""
coastal_sst_data -- in-situ observations from the Copernicus Marine In-Situ TAC.

The GLOBAL counterpart to `insitu_ioos`. IOOS aggregates North America and stops there, so
until now an AoI outside US waters had no discoverable ground truth at all -- only whatever the
user could supply as a CSV. The In-Situ TAC (https://marineinsitu.eu/) is seven production units
harmonising Argo, OceanSITES, GOSUD, the GTS and the EuroGOOS national coastal networks into one
NetCDF format with one QC scale, so one query reaches any coast on Earth.

  * Where it comes from: the `INSITU_*_PHYBGCWAV_DISCRETE_MYNRT_*` products, `history` part,
    via the Copernicus Marine Data Store. CREDENTIALED (free account); the same `copernicus`
    backend `cmems.py` already uses, so `~/.netrc` covers both.
  * What it measures: `TEMP` (sea water temperature, degC) with its `TEMP_QC` companion, from
    FIXED platforms only -- moorings (`MO`) and tide gauges (`TG`).

WHY THE INDEX FILES AND NOT `subset()`. The toolbox's tidy-DataFrame path (`read_dataframe`)
only works where a sparse ARCO cube has been published, which for the global product is the
`latest` and `monthly` parts -- and `monthly` BEGINS 2020-01-01, raising
`CoordinatesOutOfDatasetBounds` for anything earlier. The `history` part, which holds the
multi-decade archive, publishes `original-files` only. Measured over one AoI: `history` reaches
back to 1987-05-06 where `monthly` starts in 2020. A pipeline whose satellite record starts in
1984 cannot use the 2020 door, so this module takes the index-catalog route instead:

    index_<part>.txt  ->  rows intersecting the AoI box + window + carrying TEMP
                      ->  copernicusmarine.get(file_list=...)
                      ->  per-platform NetCDF -> the shared record contract

FIXED PLATFORMS ONLY, and this is the fact that most shapes a user's expectations. Under 5% of
the archive's platforms are moorings or tide gauges; drifters and profiling floats alone are
~70%. Over one AoI the index listed 409 files carrying TEMP of which exactly ONE was fixed --
the rest were ship tracks, XBT drops, gliders and bottle casts. Mobile platforms are counted and
reported but never downloaded, because `insitu_acquire.split_moving_platforms` would drop them
again downstream: the cube's in-situ model is one position per station for the whole window, and
a track collapsed to its median position produces confidently WRONG matchups. An AoI can
therefore legitimately return nothing, and that is a fact about the ocean rather than a failure.

This module is ONE SOURCE behind the shared in-situ loop: it discovers and fetches, and
`insitu_acquire` owns the union time axis, the skip guard, the write and the report, so this,
IOOS and the user's own CSVs STACK into one station table. Its output lands at

    <output_dir>/INSITU/marineinsitu/aligned/<aoi>/<aoi>_insitu.nc   dims (station, time)

Usage:
    python -m coastal_sst_data.processes.insitu_cmems --config config.yaml
    python -m coastal_sst_data.processes.insitu_cmems --config config.yaml --aoi hood_canal
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import auth, entry, net, store

log = logging.getLogger(__name__)

SOURCE = "marineinsitu"

# The global product. Every coast on Earth is in here; the regional products below merely add
# the national networks that never reach the GTS.
DEFAULT_DATASET_ID = "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"

# Region shorthands, so a config can say `dataset_id: nws` instead of pasting an opaque id.
# A regional product is worth naming where one exists: its production unit ingests coastal
# networks the global stream never sees (measured: 10-11 fixed platforms in a North Sea or
# Galician box against 1-4 in a comparable US one).
REGIONAL_DATASETS = {
    "glo": DEFAULT_DATASET_ID,
    "arc": "cmems_obs-ins_arc_phybgcwav_mynrt_na_irr",
    "bal": "cmems_obs-ins_bal_phybgcwav_mynrt_na_irr",
    "blk": "cmems_obs-ins_blk_phybgcwav_mynrt_na_irr",
    "ibi": "cmems_obs-ins_ibi_phybgcwav_mynrt_na_irr",
    "med": "cmems_obs-ins_med_phybgcwav_mynrt_na_irr",
    "nws": "cmems_obs-ins_nws_phybgcwav_mynrt_na_irr",
}

# `history` is the only part with the full archive -- see the module docstring.
DEFAULT_PART = "history"

# Copernicus data-type bigrams for platforms that DO NOT MOVE. `MO` is fixed buoys/moorings,
# `TG` tide gauges. Everything else in the vocabulary (DB drifters, PF floats, GL gliders, TS
# ship thermosalinographs, XB XBT drops, CT vessel CTDs, BO bottles, SD saildrones, SM
# animal-borne, FB ferryboxes...) is mobile or a one-off cast.
DEFAULT_PLATFORM_TYPES = ["MO", "TG"]

# The one temperature variable in the In-Situ format. Not a config knob: unlike IOOS, where
# providers disagree about sea_water_temperature vs sea_surface_temperature, this format is
# harmonised across all seven production units.
TEMP_VAR = "TEMP"

# The index catalogs carry five comment lines, then a SIXTH line that is the real CSV header
# with a "# " glued to the front of the first column name.
_INDEX_HEADER_ROW = 5

# The parsed catalog, per (dataset, part). `fetch_aoi` runs once per AoI and this answer is
# AoI-independent; the index is ~28 MB of CSV, so a ten-AoI project would otherwise fetch and
# parse 280 MB of identical bytes.
#
# Held HERE rather than inside the network seams deliberately: the seams are what the tests
# replace, so a cache living inside one would be replaced along with it and never exercised.
_CATALOG_CACHE: dict[tuple[str, str], "pd.DataFrame"] = {}

# How old the on-disk index may be before it is refetched. The catalog is rebuilt about daily
# and a platform's whole-life record does not appear or vanish within one, so a day is ample --
# and the stale-fallback path below means an outage costs freshness, not the run.
_INDEX_TTL_S = 24 * 3600


# --------------------------------------------------------------------------- #
# The network seams. Everything else in this module is pure, which is what lets
# the tests run offline -- they replace these three wholesale (as test_cmems
# replaces `cmems.open_window`), not a single low-level `_get`.
# --------------------------------------------------------------------------- #
def _catalogue_root(dataset_id: str, part: str) -> str:
    """The product-root URL holding `index_<part>.txt`, from the Data Store catalogue.

    DERIVED, NEVER HARD-CODED. The native files live in numbered buckets and the number differs
    per product -- `mdl-native-01` for the global product, `mdl-native-03` for IBI/MED/NWS -- so
    a literal path works for whichever product it was copied from and 404s for the rest. The
    catalogue's `original-files` service URI points at `<root>/<part>`; the index sits one level
    above it.

    Lazy import, so the module loads without the toolbox installed.
    """
    import copernicusmarine

    def _describe():
        return copernicusmarine.describe(dataset_id=dataset_id, disable_progress_bar=True)

    cat = net.retry(_describe, what=f"CMEMS in-situ describe {dataset_id}",
                    refresh=auth.refresher("copernicus"))
    for product in cat.products:
        for dset in product.datasets:
            if dset.dataset_id != dataset_id:
                continue
            for version in dset.versions:
                for pt in version.parts:
                    if pt.name != part:
                        continue
                    for svc in pt.services:
                        name = getattr(svc.service_name, "value", svc.service_name)
                        if name == "original-files":
                            return str(svc.uri).rsplit("/", 1)[0]
    raise RuntimeError(
        f"{dataset_id!r} part {part!r} publishes no `original-files` service, so its index "
        "catalog cannot be located. Check the dataset id, or name a different `dataset_part`.")


def _fetch_index(url: str) -> bytes:
    """The raw index catalog. ~28 MB for the global product."""
    import requests

    def _get():
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        return r.content

    return net.retry(_get, what=f"CMEMS in-situ index {url.rsplit('/', 1)[-1]}")


def _download(dataset_id: str, part: str, file_names, out_dir: Path) -> None:
    """Fetch the selected per-platform files into `out_dir`, skipping any already there.

    `copernicusmarine.get` takes the file list through a FILE, not an argument, so the list is
    written beside the cache. `no_directories=True` flattens the product's `<part>/<TYPE>/`
    nesting -- platform codes are already unique within a product, and the flat layout is what
    lets `read_platform_file` be handed a plain path.
    """
    import copernicusmarine

    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / "_file_list.txt"
    listing.write_text("\n".join(file_names) + "\n")

    def _get():
        return copernicusmarine.get(
            dataset_id=dataset_id, dataset_part=part, file_list=str(listing),
            output_directory=str(out_dir), no_directories=True, skip_existing=True,
            disable_progress_bar=True)

    net.retry(_get, what=f"CMEMS in-situ get {len(file_names)} file(s)",
              refresh=auth.refresher("copernicus"))


def _read_arco(dataset_id: str, part: str, bbox, start, end, max_depth_m):
    """A bbox+time subset of the SPARSE ARCO service, as a tidy per-observation DataFrame.

    THE FOURTH SEAM, and it exists because the index route the fixed platforms use is unusable
    for tracks. There, a file is a platform's WHOLE LIFE and its catalogued bounds are the
    bounding box of that whole life -- so a drifter that crossed the AoI once carries a bbox
    spanning an ocean, and selecting on it means downloading the ocean. Measured over one AoI:
    348 MB of whole-life files yielded FOUR observations inside it, 0.000% of what was parsed,
    and the full set would have been ~4.5 GB.

    The sparse ARCO service subsets by bounding box SERVER-SIDE, which is exactly the shape of
    the question a track asks. The same AoI returned 16,088 in-box observations, 100% of what
    came back, in a few MB.

    The cost is temporal: only `latest` and `monthly` are published as sparse cubes, and
    `monthly` BEGINS 2020-01-01 -- see `_clamp_window`, which is not optional.

    Returns the toolbox's tidy columns: variable, platform_id, platform_type, time, longitude,
    latitude, depth, value, value_qc, institution, ...
    """
    import copernicusmarine

    w, s, e, n = bbox

    def _read():
        return copernicusmarine.read_dataframe(
            dataset_id=dataset_id, dataset_part=part, variables=[TEMP_VAR],
            minimum_longitude=w, maximum_longitude=e,
            minimum_latitude=s, maximum_latitude=n,
            minimum_depth=0.0, maximum_depth=float(max_depth_m),
            start_datetime=str(start), end_datetime=str(end),
            disable_progress_bar=True)

    return net.retry(_read, what=f"CMEMS in-situ ARCO {dataset_id} {start}..{end}",
                     refresh=auth.refresher("copernicus"))


# --------------------------------------------------------------------------- #
# The index catalog -- pure
# --------------------------------------------------------------------------- #
def parse_index(raw: bytes) -> pd.DataFrame:
    """The index catalog bytes -> a DataFrame with usable column names and real timestamps.

    The header line is `# product_id,file_name,geospatial_lat_min,...` -- a comment marker glued
    to the first column name -- so pandas reads that first column as `"# product_id"` unless the
    prefix is stripped afterwards.
    """
    idx = pd.read_csv(io.BytesIO(raw), skiprows=_INDEX_HEADER_ROW, header=0, low_memory=False)
    idx.columns = [c.lstrip("# ").strip() for c in idx.columns]
    for col in ("time_coverage_start", "time_coverage_end"):
        idx[col] = pd.to_datetime(idx[col], errors="coerce", utc=True)
    return idx


def platform_type(file_name: str) -> str:
    """The data-type bigram of an In-Situ filename: `<PU>_<FILETYPE>_<DATATYPE>_<code>.nc`.

    `GL_TS_MO_46211.nc` -> `MO`. Positional, deliberately: the vocabularies COLLIDE across
    fields -- `MO` is Mediterranean in position 1 and mooring in position 3, `BO` is Baltic
    or bottle -- so a substring search would misclassify every Mediterranean file as a mooring.
    """
    stem = file_name.rsplit("/", 1)[-1]
    bits = stem.split("_")
    return bits[2] if len(bits) > 3 else ""


def index_bytes(dataset_id: str, part: str, cache_dir: Path) -> bytes:
    """The index catalog, from disk when it is fresh, from the network when it is not.

    ON DISK as well as in memory, for two reasons that are really one. It is ~28 MB fetched in
    a single unresumed GET, and it is fetched ONCE PER RUN for every AoI -- so a blip does not
    cost one AoI, it costs the whole run. That is not hypothetical: this was written after
    `Connection broken: IncompleteRead(12243941 bytes read, 16141403 more expected)` exhausted
    all four `net.retry` attempts and failed every AoI, on an endpoint that served the same
    file cleanly in 2.8 s three times a minute later.

    So: a copy less than `_INDEX_TTL_S` old is used as-is (the catalog is rebuilt about daily,
    and an AoI's platforms do not appear and vanish within one), and when a refresh FAILS but a
    stale copy exists, THE STALE COPY IS USED with a warning. A day-old catalog is a far better
    answer than no data at all, and the failure is still said out loud.
    """
    import time

    path = cache_dir / f"index_{part}.txt"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < _INDEX_TTL_S
    if fresh:
        return path.read_bytes()

    root = _catalogue_root(dataset_id, part)
    try:
        raw = _fetch_index(f"{root}/index_{part}.txt")
    except Exception as exc:
        if not path.exists():
            raise
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        log.warning("  index refresh failed (%s); using the cached copy from %.1f h ago. "
                    "Platforms added since then will be missing.", exc, age_h)
        return path.read_bytes()

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")             # never leave a truncated index looking whole
    tmp.write_bytes(raw)
    tmp.replace(path)
    return raw


def catalog(dataset_id: str, part: str, cache_dir: Path) -> pd.DataFrame:
    """The parsed index catalog for one product part, parsed at most once per process.

    Every AoI in a run asks the same question and gets the same answer, so the parse -- 28 MB of
    CSV -- is memoised alongside the bytes. Held HERE rather than inside a network seam because
    the seams are what the tests replace; a cache inside one would never be exercised.
    """
    key = (dataset_id, part)
    if key not in _CATALOG_CACHE:
        _CATALOG_CACHE[key] = parse_index(index_bytes(dataset_id, part, cache_dir))
    return _CATALOG_CACHE[key]


def _all_types(idx: pd.DataFrame):
    """Every data-type bigram present in the catalog -- the "no type filter" argument.

    `select_files` takes an explicit allow-list rather than an optional one, so there is a single
    filtering path; asking it for everything is how `fetch_aoi` counts what it then passes over.
    """
    return sorted({platform_type(f) for f in idx["file_name"]})


def select_files(idx: pd.DataFrame, bbox, start, end, platform_types, pad_deg: float = 0.0):
    """The index rows whose file could hold TEMP for this AoI window -> DataFrame.

    INTERSECTION, NOT CONTAINMENT, and this is the one thing in this module most likely to be
    "corrected" into a bug. Copernicus's own published example filters with

        lon_min >= box_lon_min and lon_max <= box_lon_max

    which keeps only files whose bounds fall ENTIRELY INSIDE the box. A mooring's bounds are a
    point and survive that test by luck; anything with real extent -- and, more to the point,
    any platform whose recorded bounds are merely generous -- is silently discarded. Two
    intervals overlap when each starts before the other ends, which is what is used here.

    Rows are also required to carry TEMP. `parameters` is a SPACE-separated list inside a
    COMMA-separated file, so it is split rather than substring-searched: `"TEMPX"` must not
    match, and neither must the `TEMP` inside `ATEMP` (air temperature).
    """
    w, s, e, n = bbox
    w, s, e, n = w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)   # `end` is day-inclusive

    has_temp = idx["parameters"].astype(str).str.split().apply(lambda p: TEMP_VAR in p)
    overlaps = (
        (idx["geospatial_lon_min"] <= e) & (idx["geospatial_lon_max"] >= w) &
        (idx["geospatial_lat_min"] <= n) & (idx["geospatial_lat_max"] >= s) &
        (idx["time_coverage_start"] <= t1) & (idx["time_coverage_end"] >= t0)
    )
    sel = idx[overlaps & has_temp].copy()
    sel["platform_type"] = sel["file_name"].map(platform_type)
    return sel[sel["platform_type"].isin(list(platform_types))]


# --------------------------------------------------------------------------- #
# One platform file -> the record contract
# --------------------------------------------------------------------------- #
def read_platform_file(path: Path, qc_flags, max_depth_m) -> pd.DataFrame | None:
    """One In-Situ NetCDF -> DataFrame[time, latitude, longitude, value, qc], or None.

    None means "nothing usable here", which is a normal outcome: the index says a file's window
    and box overlap the AoI's and that it carries TEMP somewhere, not that any QC-passing
    near-surface reading survives inside them.

    Three format details, each of which silently empties the series if missed:

      * `TEMP_QC` is stored as int8 with a -127 fill, and xarray DECODES IT TO FLOAT32. An
        `isin([1, 2])` against Python ints still works, but only because 1.0 == 1; comparing
        against a numpy int dtype, or testing `.astype(int)` on a column holding NaN, does not.
        The flags are cast through float here so the intent is explicit.
      * The depth VARIABLE is `DEPH`, while the DIMENSION is `DEPTH`. Profiling platforms carry
        `PRES` (dbar) instead, which for the top few metres is within a few percent of depth in
        metres -- close enough for a `<= 5 m` gate, and better than dropping the platform.
      * `LATITUDE`/`LONGITUDE` are SCALAR for a fixed platform and `(TIME,)` for a mobile one.
        `to_dataframe()` broadcasts either into a per-row column, which is what the record
        contract wants and what lets `insitu_acquire.platform_drift_m` measure a real drift
        rather than assume one.

    A mooring reports several depths; the SHALLOWEST surviving level wins per timestamp, because
    that is the one comparable to a satellite's surface retrieval.
    """
    # `store.open_netcdf`, never a bare `xr.open_dataset`: the gate has to span open->use->close
    # because xarray reads lazily, and one unguarded reader anywhere puts two threads back
    # inside xarray's lock layer. Everything needed afterwards is materialised inside the block.
    with store.open_netcdf(path) as ds:
        if TEMP_VAR not in ds:
            return None
        depth_var = "DEPH" if "DEPH" in ds else ("PRES" if "PRES" in ds else None)
        want = [TEMP_VAR] + [v for v in (f"{TEMP_VAR}_QC", depth_var) if v and v in ds]
        # POSITION HAS TO BE ASKED FOR when it is a data variable. In the files this was built
        # against LATITUDE/LONGITUDE are COORDINATES, so `ds[want]` carries them along for free
        # -- but the format permits either, and a file that stores them as variables would come
        # back with no position at all and be dropped as unplaceable, silently, with the index
        # still insisting the platform is there. Naming them costs nothing when they are coords.
        want += [v for v in ("LATITUDE", "LONGITUDE") if v in ds.data_vars]
        df = ds[want].to_dataframe().reset_index()
        attrs = dict(ds.attrs)

    ren = {TEMP_VAR: "value", f"{TEMP_VAR}_QC": "qc", "TIME": "time",
           "LATITUDE": "latitude", "LONGITUDE": "longitude"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df})
    if "time" not in df or "value" not in df:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["time", "value"])
    if "qc" in df and qc_flags is not None:
        df = df[df["qc"].astype("float64").isin([float(f) for f in qc_flags])]

    if depth_var and depth_var in df.columns and max_depth_m is not None:
        df = df[df[depth_var].abs() <= float(max_depth_m)]
        # Shallowest level per timestamp. Sorting first makes `first()` deterministic; a
        # groupby-min on the value would mix depths within one series.
        df = df.sort_values([depth_var]).groupby("time", as_index=False).first()

    if df.empty:
        return None
    for col in ("latitude", "longitude"):
        if col not in df:                       # a file with no position is not placeable
            return None
    keep = ["time", "latitude", "longitude", "value"] + (["qc"] if "qc" in df else [])
    out = df[keep].sort_values("time").reset_index(drop=True)
    out.attrs = attrs
    return out


def within(lat: float, lon: float, bbox, pad_deg: float = 0.0) -> bool:
    """Does a platform's OWN position fall in the box we searched?

    A second, independent check on something the index already claimed -- and it has to be,
    because the index's `geospatial_*` bounds are sometimes wrong by continents. Measured on the
    real catalog: `GL_TS_MO_31261` advertises lat -31.5..48.7, lon -123.4..-34.6 -- a box
    spanning half the planet -- while the file itself sits at (-8.16, -34.56), off Brazil. It
    intersects a Puget Sound AoI on paper and was duly returned by it, arriving in the station
    table as a plausible-looking mooring reporting 0.0-38.9 degC.

    Reverting the selection to CONTAINMENT would also have excluded it, but at the cost of every
    genuinely wide-bounded platform (see `select_files`) -- so the fix belongs here, after the
    file is open and its real position is known, not in the index filter.
    """
    w, s, e, n = bbox
    return (s - pad_deg) <= lat <= (n + pad_deg) and (w - pad_deg) <= lon <= (e + pad_deg)


def _title(attrs: dict, code: str) -> str:
    """A human label for the station table: the platform's name, else its code."""
    for key in ("platform_name", "platform_code", "site_code"):
        val = str(attrs.get(key) or "").strip()
        if val:
            return val
    return code


# --------------------------------------------------------------------------- #
# The source seam: one AoI's platforms -> records
# --------------------------------------------------------------------------- #
def fetch_aoi(g, start, end, cfg: dict, dry_run: bool = False) -> list[dict]:
    """Every usable fixed In-Situ platform overlapping this AoI, as acquisition records.

    The seam `insitu_acquire` calls (see its docstring for the record contract). Discover from
    the index, filter to fixed platforms BEFORE downloading anything, fetch, and SAY WHAT IT
    DROPPED -- an empty in-situ channel that reads as "no buoys here" is the failure mode this
    product cannot afford.
    """
    dataset_id = REGIONAL_DATASETS.get(str(cfg["dataset_id"]).lower(), cfg["dataset_id"])
    part = cfg["dataset_part"]
    types = list(cfg["platform_types"])

    idx = catalog(dataset_id, part, Path(cfg["cache_dir"]))

    # Everything with TEMP, before the platform-type gate, so the log can say what was passed
    # over rather than only what was kept.
    all_temp = select_files(idx, g.search_bbox, start, end,
                            platform_types=_all_types(idx), pad_deg=cfg["pad_deg"])
    sel = all_temp[all_temp["platform_type"].isin(types)]
    skipped = len(all_temp) - len(sel)

    allow, deny = set(cfg["stations"]), set(cfg["exclude_stations"])
    codes = sel["file_name"].map(lambda f: f.rsplit("/", 1)[-1].split("_")[-1].removesuffix(".nc"))
    sel = sel.assign(code=codes)
    if allow:
        sel = sel[sel["code"].isin(allow)]
    sel = sel[~sel["code"].isin(deny)]

    log.info("  %d fixed platform(s) [%s]; %d mobile/one-off file(s) with TEMP not fetched",
             len(sel), "/".join(types), skipped)
    if sel.empty:
        # Not an error. Under 5% of this archive's platforms are fixed, so a coastal AoI with
        # only ship tracks and drifters over it is an ordinary outcome -- said plainly, because
        # the alternative reading ("the download failed") sends people to the wrong file.
        log.warning("  no FIXED platforms (moorings/tide gauges) in this AoI window; "
                    "%d mobile file(s) were available but cannot be placed by a "
                    "fixed-station cube", skipped)
        return []
    if dry_run:
        for _, row in sel.iterrows():
            log.info("  [dry-run] %-22s %s  %s -> %s", row["code"], row["platform_type"],
                     str(row["time_coverage_start"])[:10], str(row["time_coverage_end"])[:10])
        return []

    cache = Path(cfg["cache_dir"])
    _download(dataset_id, part, list(sel["file_name"]), cache)

    records, empty = [], []
    for _, row in sel.iterrows():
        name = row["file_name"].rsplit("/", 1)[-1]
        path = cache / name
        if not path.exists():
            empty.append(f"{row['code']} (file not delivered)")
            continue
        try:
            df = read_platform_file(path, cfg["qc_flags"], cfg["max_sensor_depth_m"])
        except Exception as exc:                # one unreadable file must not kill the AoI
            empty.append(f"{row['code']} (unreadable: {exc})")
            continue
        if df is None or df.empty:
            empty.append(f"{row['code']} (no QC-passing near-surface values)")
            continue
        lat, lon = float(np.nanmedian(df["latitude"])), float(np.nanmedian(df["longitude"]))
        if not within(lat, lon, g.search_bbox, cfg["pad_deg"]):
            # The index said this platform reached the AoI and its own position says otherwise.
            # Trust the file. See `within` for the real case this was written against.
            empty.append(f"{row['code']} (index bounds claim this AoI, but the file sits at "
                         f"{lat:.3f},{lon:.3f})")
            continue
        records.append({
            "id": str(row["code"]), "title": _title(df.attrs, str(row["code"])),
            "var": TEMP_VAR, "df": df[["time", "latitude", "longitude", "value"] +
                                      (["qc"] if "qc" in df else [])],
            "lat": lat, "lon": lon,
        })
        log.info("  %-22s %-6s %7d obs  %.1f-%.1f degC  %s -> %s", row["code"],
                 row["platform_type"], len(df), df["value"].min(), df["value"].max(),
                 str(df["time"].min())[:10], str(df["time"].max())[:10])

    for e in empty:
        log.warning("  dropped %s", e)
    return records


# The first day the sparse ARCO cube covers. Not discovered at runtime: the toolbox reports a
# window out of bounds by RAISING, so there is nothing to read the extent from without first
# making a request that fails. Verified by bisection -- every range ending before this date
# raises `CoordinatesOutOfDatasetBounds`, and 2020-01-02 is the earliest observation returned.
ARCO_FIRST_DAY = "2020-01-01"

# Which part to read tracks from. `monthly` is the full sparse record; `latest` is a rolling
# ~30 days. `history` is NOT an option here -- it publishes original files only, which is the
# whole reason `_read_arco` exists.
DEFAULT_MOBILE_PART = "monthly"


def _clamp_window(start: str, end: str):
    """The requested window, narrowed to what the sparse cube actually covers, or None.

    NOT a convenience. A range entirely before `ARCO_FIRST_DAY` does not come back empty, it
    raises `CoordinatesOutOfDatasetBounds` -- so a project whose window starts in 1984 (which
    is the ordinary case here, the satellite record reaches back that far) would lose the AoI
    outright rather than acquiring the part of it that exists.
    """
    lo = pd.Timestamp(ARCO_FIRST_DAY)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e < lo:
        return None
    return (max(s, lo).date().isoformat(), e.date().isoformat())


def fetch_aoi_mobile(g, start, end, cfg: dict, dry_run: bool = False) -> list[dict]:
    """Every MOVING platform crossing this AoI, as acquisition records.

    The `insitu_mobile` counterpart of `fetch_aoi`, and a different route entirely -- see
    `_read_arco` for why the fixed product's index route cannot serve tracks.

    Platform selection is by EXCLUSION: everything that is not a fixed class. Taking the
    complement rather than listing the mobile codes means a class the archive adds later (or
    one of the several rare ones -- saildrones, animal-borne sensors, ferryboxes) is picked up
    without this list having to be maintained in lockstep with the fixed product's.
    """
    dataset_id = REGIONAL_DATASETS.get(str(cfg["dataset_id"]).lower(), cfg["dataset_id"])
    part = cfg.get("dataset_part") or DEFAULT_MOBILE_PART
    if part == DEFAULT_PART:            # `history` is the fixed product's part, not usable here
        part = DEFAULT_MOBILE_PART

    window = _clamp_window(start, end)
    if window is None:
        log.warning("  the sparse in-situ cube begins %s, and this window ends before it; "
                    "no moving platforms can be acquired for it", ARCO_FIRST_DAY)
        return []
    a, b = window
    if a != str(start):
        log.info("  window clamped to %s..%s (the sparse cube begins %s; the fixed `insitu` "
                 "product reaches further back)", a, b, ARCO_FIRST_DAY)

    df = _read_arco(dataset_id, part, g.search_bbox, a, b, cfg["max_sensor_depth_m"])
    if df is None or len(df) == 0:
        log.info("  no in-situ observations of any kind in this AoI window")
        return []

    keep_types = set(cfg.get("platform_types") or [])
    fixed_types = set(DEFAULT_PLATFORM_TYPES)
    types = df["platform_type"].astype(str)
    sel = df[types.isin(keep_types)] if keep_types else df[~types.isin(fixed_types)]
    sel = sel[sel["value_qc"].astype("float64").isin([float(f) for f in cfg["qc_flags"]])]
    sel = sel.dropna(subset=["value", "latitude", "longitude", "time"])
    if sel.empty:
        log.info("  %d observation(s) in the AoI, none from a moving platform passing QC",
                 len(df))
        return []

    allow, deny = set(cfg["stations"]), set(cfg["exclude_stations"])
    if allow:
        sel = sel[sel["platform_id"].astype(str).isin(allow)]
    sel = sel[~sel["platform_id"].astype(str).isin(deny)]

    n_plat = sel["platform_id"].nunique()
    log.info("  %d moving platform(s), %d observation(s) in the AoI", n_plat, len(sel))
    if dry_run:
        for pid, grp in sel.groupby("platform_id"):
            log.info("  [dry-run] %-16s %-4s %6d obs  %s -> %s", str(pid),
                     str(grp["platform_type"].iloc[0]), len(grp),
                     str(grp["time"].min())[:10], str(grp["time"].max())[:10])
        return []

    records = []
    for (pid, ptype), grp in sel.groupby(["platform_id", "platform_type"]):
        obs = pd.DataFrame({
            "time": pd.to_datetime(grp["time"], utc=True, errors="coerce").dt.tz_localize(None),
            "latitude": grp["latitude"].to_numpy(dtype="float64"),
            "longitude": grp["longitude"].to_numpy(dtype="float64"),
            "value": grp["value"].to_numpy(dtype="float64"),
            "qc": grp["value_qc"].to_numpy(dtype="float64"),
        }).dropna(subset=["time"]).sort_values("time")
        if obs.empty:
            continue
        records.append({
            "id": str(pid), "title": str(pid), "var": TEMP_VAR,
            "platform_type": str(ptype), "df": obs,
            "lat": float(np.nanmedian(obs["latitude"])),
            "lon": float(np.nanmedian(obs["longitude"])),
        })
        log.info("  %-16s %-4s %6d obs  %.1f-%.1f degC  %s -> %s", str(pid), str(ptype),
                 len(obs), obs["value"].min(), obs["value"].max(),
                 str(obs["time"].min())[:10], str(obs["time"].max())[:10])
    return records


def main():
    """Copernicus in-situ is acquired through the shared loop; run that, narrowed to it."""
    from . import insitu_acquire

    entry.process_main(
        lambda project, **kw: insitu_acquire.acquire(project, source=SOURCE, **kw),
        "coastal_sst_data Copernicus Marine In-Situ acquisition.")


if __name__ == "__main__":
    main()
