#!/usr/bin/env python3
"""
coastal_sst_data -- datacube assembler (final stage).

Knits the per-AoI aligned outputs of every acquisition stage (MUR, ECOSTRESS,
Landsat, MODIS, met, bathymetry, tide, land-cover) into ONE analysis-ready,
chunked+compressed Zarr cube per AoI on a common DAILY time axis:

    <output_dir>/<datacube.output_subdir>/<aoi>.zarr

Unlike the acquisition modules this reads no network -- it only reads files the
other stages already wrote, so run it LAST. It drives off the validated Project
and the SHARED per-AoI grid (coastal_sst_data.grid), so the cube lands on exactly
the grid every product was regridded onto (no grid is re-derived here).

Design (locked with the maintainer):
  * Zarr per AoI, chunked in (time, y, x), lossless float32 + Blosc(zstd) codec.
  * SST kept SEPARATE per sensor (mur/eco/lst/modis) so the model's learned
    per-source offsets survive; each high-res sensor carries its own valid mask
    and overpass hour.
  * Multiple scenes of one sensor on a day -> keep the CLEAREST (most valid px).
  * The cube ships RAW ingredients on a common grid + daily axis; masking, water-filling,
    station snapping, and multi-input derivations are DOWNSTREAM modelling determinations.
    So MUR/CMEMS ship observed values with honest NaN gaps (no NN fill), land-cover water
    ships raw (no opinionated land mask), and bathymetry ships `elevation` + `depth` for a
    downstream water-level computation rather than a pre-derived water_level channel.

  * Met FORCING (per source) is taken at a REFERENCE time of day (default 10:30 local solar --
    Landsat's overpass), not a daily mean, which would smear the diurnal cycle. Met matched to
    each sensor's OWN overpass is the separate `met_overpass` product, emitted only for the
    user's `(sensor, source)` combinations so two sensors hours apart don't share one value.

  * CMEMS gives the offshore water column at the requested depths (its ~9 km land mask can
    swallow a whole estuary, so expect NaN gaps a downstream process may choose to fill).

  * IN-SITU is the cube's only ground truth: each station's value is written into the
    grid cell it sits in, at the INSTANT each satellite flew, so a scene can be validated
    against a buoy pixel-for-pixel and minute-for-minute (see processes.insitu).

  * DISTINCT-DATA products SHIP ONE CHANNEL PER SOURCE (D10): bathymetry (`depth_<dem>`), CMEMS
    (`cmems_<var>_<tag>`), met forcing (`airtemp_<src>`), tides (`tide_<src>`). Each DEM's
    DEM->MSL datum offset ships as attributes on its own `elevation_<dem>` channel.

Channel layout in each <aoi>.zarr. `<s>` ranges over the per-overpass thermal SENSORS (every
product declaring a SensorSpec, today {eco, lst, modis}); `<src>` over a product's stacked
sources; `(<s>,<src>)` over the user's overpass COMBINATIONS. A STACKED-DATA sensor (ECOSTRESS)
forks its data channels per collection version tag `<ver>` (`eco_sst_v002`, `eco_sst_v003`),
but the matchup channels (met_overpass, tide_overpass, in-situ) key off ONE overpass identity
per sensor, so they stay unqualified (`eco_insitu_sst`, not per-version) -- see D5.

  3D (time,y,x): mur_sst,
                 <s>_sst, <s>_valid, <s>_cloud (where the sensor publishes a cloud layer),
                 <s>_footprint_id (where the sensor is gridded from a COARSER swath and its
                   granules carried the layer -- today MODIS only; ids group the grid cells
                   sharing one native observation, WITHIN a day, and restart every scene),
                   -- for a stacked-data sensor: <s>_sst_<ver>, <s>_valid_<ver>, <s>_cloud_<ver>,
                 <metvar>_<src>  (forcing: airtemp/wind_u/wind_v/wind_speed/swrad/cloud_cover),
                 <s>_<metvar>_<src>  (met_overpass, per combo),
                 cmems_<var>_<depth>m_<tag>,
                 insitu_sst, insitu_n, <s>_insitu_sst, <s>_insitu_dt_min
  2D (y,x) static: elevation_<dem>, depth_<dem>, depth_p25_<dem>, depth_p75_<dem>,
                   landcover_water, insitu_station (index into the insitu_stations attr)
  1D (time): tide_<src>, tide_range_<src>, <s>_tide_<src> (tide_overpass, per combo),
             <s>_hour (or <s>_hour_<ver> for a stacked-data sensor), doy_sin, doy_cos

Usage:
    python -m coastal_sst_data.processes.datacube --config config.yaml
    python -m coastal_sst_data.processes.datacube --config config.yaml --aoi hood_canal
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from ..config import CompressionSpec, DataProduct, Project, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, naming, products, provenance, report, store
from . import insitu, met as met_mod, water_level

log = logging.getLogger(__name__)

# product -> the ALLCAPS "<DIR>/aligned/<aoi>/" folder its acquisition stage wrote.
# DERIVED from the product registry, and keyed by the PRODUCT's own name throughout -- so
# the tide product is `tides` here, in provenance, and in the coverage report, even though
# it writes to TIDE/. Two alias tables used to exist solely because this dict said `tide`
# and the others said `tides`.
PRODUCT_DIRS = products.product_dirs()


# --------------------------------------------------------------------------- #
# Loaders (each returns arrays on the daily axis / shared AoI grid)
# --------------------------------------------------------------------------- #
def _empty3d(days, H, W):
    return np.full((len(days), H, W), np.nan, dtype="float32")


def _cached(cache, key, build):
    """`build()`, memoised in `cache` under `key`. Uncached (a plain call) when `cache` is None.

    Every loader here discovers its inputs by globbing its own directory. That is paid once per
    AoI today, but the assembler walks the time axis in BLOCKS for a cube too large to hold at
    once (see `run`), and a per-block rescan of a directory holding thousands of scenes -- times
    the number of products, times the number of blocks -- would trade the memory problem for a
    wall-clock one. One dict per AoI, threaded through the loaders, keeps it at once.

    `cache=None` is the default everywhere, so a contributor written against the documented
    loader signatures (which take no cache) keeps working, just uncached.
    """
    if cache is None:
        return build()
    if key not in cache:
        cache[key] = build()
    return cache[key]


def scene_index(d: Path, aoi_id, *, cache=None) -> dict[str, list]:
    """{day_stamp: [(datetime, path), ...]} for a per-overpass sensor's aligned directory.

    Built from FILENAMES alone -- nothing is opened -- so `load_clearest_overpass` can work
    through one day's granules at a time instead of globbing the whole directory and holding
    every day's chosen scene until the end.

    Each day's granules are sorted by (time, name). That also settles a tie in the
    clearest-scene contest deterministically (the earliest scene wins) where it used to fall
    to whatever order the filesystem happened to hand back.
    """
    def build() -> dict[str, list]:
        out: dict[str, list] = {}
        if not d.exists():
            return out
        for f in d.glob(f"{aoi_id}_*T*.nc"):
            dt = naming.parse_time(f.name)
            if dt is None:
                continue
            out.setdefault(naming.day_stamp(dt), []).append((dt, f))
        for granules in out.values():
            granules.sort(key=lambda g: (g[0], g[1].name))
        return out

    return _cached(cache, ("scenes", str(d)), build)


def load_daily_sensor(d: Path, aoi_id, days, H, W, var, *, prefix=""):
    """MUR/met style: one file per day (<aoi>_<prefix><YYYYMMDD>.nc) -> (T,H,W).

    `prefix` selects a variant written into the same directory -- met writes both a
    daily mean (`<aoi>_20230715.nc`) and a reference-time snapshot
    (`<aoi>_ref_20230715.nc`). The name is matched WHOLE rather than by suffix, so the
    two cannot be confused for each other.
    """
    out = _empty3d(days, H, W)
    if not d.exists():
        return out
    pat = naming.day_pattern(aoi_id, prefix)
    didx = {naming.day_stamp(dd): i for i, dd in enumerate(days)}
    for f in d.glob(f"{aoi_id}_{prefix}*.nc"):
        m = pat.match(f.name)
        if not m or m.group(1) not in didx:
            continue
        ds = xr.open_dataset(f)
        if var in ds:
            arr = ds[var].isel(time=0).values if "time" in ds[var].dims else ds[var].values
            if arr.shape == (H, W):
                out[didx[m.group(1)]] = arr
        ds.close()
    return out


def load_at_times(d: Path, aoi_id, times, H, W, var):
    """Per-overpass file (<aoi>_YYYYMMDDThhmmss.nc) at each day's chosen scene time.

    `times` is one datetime per day (None where that sensor had no scene), so the value
    read is the forcing at the EXACT scene the cube kept -- not merely something from
    that day. Days with no scene stay NaN.
    """
    out = _empty3d(times, H, W)
    if not d.exists():
        return out
    for i, t in enumerate(times):
        if t is None:
            continue
        f = d / f"{naming.time_stem(aoi_id, t)}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        if var in ds:
            arr = ds[var].isel(time=0).values if "time" in ds[var].dims else ds[var].values
            if arr.shape == (H, W):
                out[i] = arr
        ds.close()
    return out


def met_prefix(d: Path, aoi_id, days, want: str) -> tuple[str, str]:
    """Which met variant to feed the cube's met channels: ('ref_'|'', label).

    Honours `datacube.met_time`, but falls back to the other variant if the one asked
    for was never written (e.g. an older MET tree with no reference snapshots) rather
    than silently emitting an all-NaN forcing channel.
    """
    def has(prefix):
        return any((d / f"{naming.day_stem(aoi_id, dd, prefix)}.nc").exists()
                   for dd in days)

    want_prefix = "ref_" if want == "reference" else ""
    if has(want_prefix):
        return want_prefix, ("reference" if want_prefix else "daily_mean")
    other = "" if want_prefix else "ref_"
    if has(other):
        label = "reference" if other else "daily_mean"
        log.warning("  no %s met files; falling back to the %s", want, label)
        return other, label
    return want_prefix, ("reference" if want_prefix else "daily_mean")


def load_clearest_overpass(d: Path, aoi_id, days, H, W, *, water_is_land=False,
                           use_cloud=True, qc_levels=None, trust_valid=False,
                           read_footprint=False, cache=None):
    """Per-overpass sensors (ECOSTRESS/Landsat/MODIS): keep the clearest scene/day.

    Validity per sensor:
      * trust_valid=True (MODIS -- already quality-filtered): use the file's
        `valid` layer directly; the sensor has no water/cloud layer.
      * else recompute finite(sst) & water [& clear] [& QC]:
          - water: the sensor water layer with per-sensor polarity (water_is_land).
          - use_cloud gates on the binary cloud layer (Landsat: reliable).
          - qc_levels (e.g. {0,1}) gates on QC mandatory-QA bits 0-1 instead of
            cloud (ECOSTRESS: cloud over-masks cold water, so gate on QC).
    Returns (sst, cloud, valid, hour, times, footprint).
    `times` is the CHOSEN scene's datetime per day (None where the sensor had no scene), so
    the tide and the met snapshot can be matched to that exact scene rather than to the day.

    ONE DAY AT A TIME. The scenes are walked per day (via `scene_index`) and the day's winner
    is written into the output before the next day is read, so the only scene arrays alive at
    once are the contender and the incumbent. Accumulating every day's winner first and
    copying afterwards -- which is what this did -- held a WHOLE WINDOW of scenes alongside
    the outputs that were already allocated, roughly doubling the peak for no benefit.

    `footprint` is the native sensor-pixel index per grid cell, read only when
    `read_footprint` and returned as None when NO granule on disk carried the layer -- it is
    optional at acquisition time, so its absence is a fact about the files, not a failure.
    Collapsing "we saw one" into the value is what lets the caller emit the channel or not
    with a plain `is not None`, rather than shipping an all -1 array that reads as data.
    It rides the SAME clearest-scene choice as the SST: read outside that selection it would
    pair one granule's pixel indices with another granule's temperatures. The array is
    allocated only once a scene actually yields one: it is int32 (T,H,W), the same size as an
    SST channel, and eagerly filling one with -1 for a sensor that never carried the layer
    cost a large AoI many GB to build something the caller then threw away.
    """
    sst, cloud = _empty3d(days, H, W), _empty3d(days, H, W)
    valid = np.zeros((len(days), H, W), dtype="uint8")
    hour = np.full(len(days), np.nan, dtype="float32")
    times: list = [None] * len(days)
    footprint = None
    saw_footprint = False
    qset = list(qc_levels) if qc_levels is not None else None
    didx = {naming.day_stamp(dd): i for i, dd in enumerate(days)}
    missing_fp = 0
    scene_days = 0
    for day, granules in scene_index(d, aoi_id, cache=cache).items():
        i = didx.get(day)
        if i is None:
            continue
        best = None  # (valid_count, sst, cloud, valid, footprint|None, datetime)
        for dt, f in granules:
            ds = xr.open_dataset(f)
            if "time" in ds.dims:
                ds = ds.isel(time=0)
            if "sst" not in ds or ds["sst"].shape != (H, W):
                ds.close(); continue
            s = ds["sst"].values.astype("float32")
            c = (ds["cloud"].values.astype("float32")
                 if "cloud" in ds and ds["cloud"].shape == (H, W) else np.zeros((H, W), "float32"))
            fp = None
            if read_footprint and "footprint_id" in ds and ds["footprint_id"].shape == (H, W):
                fp = ds["footprint_id"].values.astype("int32")
                saw_footprint = True
            if trust_valid and "valid" in ds and ds["valid"].shape == (H, W):
                v = (ds["valid"].values > 0) & np.isfinite(s)
            else:
                if "water" in ds and ds["water"].shape == (H, W):
                    w = ds["water"].values.astype("float32")
                    wp = np.isfinite(w) & ((w < 0.5) if water_is_land else (w > 0.5))
                else:
                    wp = np.zeros((H, W), dtype=bool)      # no water layer -> claim NOTHING
                q = (ds["quality"].values if "quality" in ds and ds["quality"].shape == (H, W)
                     else None)
                v = np.isfinite(s) & wp
                if use_cloud:
                    v &= ~(np.nan_to_num(c, nan=1.0) > 0)
                if qset is not None and q is not None:
                    mqa = np.full((H, W), -1, dtype="int64")
                    fin = np.isfinite(q)
                    mqa[fin] = q[fin].astype("int64") & 0b11   # mandatory-QA bits 0-1
                    v &= np.isin(mqa, qset)
            ds.close()

            vc = int(v.sum())
            if best is None or vc > best[0]:
                best = (vc, s, c, v, fp, dt)
        if best is None:
            continue                                       # every granule failed the shape check
        _, s, c, v, fp, dt = best
        sst[i] = s
        cloud[i] = np.nan_to_num(c, nan=0.0)
        valid[i] = v.astype("uint8")
        hour[i] = dt.hour + dt.minute / 60.0
        times[i] = dt
        scene_days += 1
        if fp is None:
            missing_fp += 1
        else:
            if footprint is None:
                # -1 (no native pixel), never NaN: this is an INDEX, and _empty3d is float.
                footprint = np.full((len(days), H, W), -1, dtype="int32")
            footprint[i] = fp
    if saw_footprint and footprint is None:
        # Granules carried the layer but no CHOSEN scene did -- an all -1 channel, which the
        # warning below explains. Kept distinct from "no granule ever had one" (-> None, no
        # channel at all), because those say different things about the tree.
        footprint = np.full((len(days), H, W), -1, dtype="int32")
    if saw_footprint and missing_fp:
        # A partly backfilled directory: some granules predate the layer, or were acquired
        # with `footprint_id: false`. Those days stay -1, which is indistinguishable from
        # "off-swath" unless we say so here. Re-acquire with --overwrite to fill them.
        log.warning("  %s: %d of %d scene-days have no footprint_id layer; those days are "
                    "all -1 in the footprint channel", d.name, missing_fp, scene_days)
    return sst, cloud, valid, hour, times, footprint


def load_tide_daily(d: Path, aoi_id, days):
    """Tide 1D series -> (daily_mean, daily_range) on the daily axis."""
    mean = np.full(len(days), np.nan, "float32")
    rng = np.full(len(days), np.nan, "float32")
    f = d / f"{aoi_id}_tides.nc"
    if not f.exists():
        return mean, rng
    ds = xr.open_dataset(f)
    t = ds["tide"]
    dm = t.resample(time="1D").mean()
    dr = t.resample(time="1D").max() - t.resample(time="1D").min()
    lut_m = dict(zip(dm["time"].dt.strftime(naming.DAY_FMT).values, dm.values))
    lut_r = dict(zip(dr["time"].dt.strftime(naming.DAY_FMT).values, dr.values))
    for i, dd in enumerate(days):
        k = naming.day_stamp(dd)
        if k in lut_m:
            mean[i] = lut_m[k]
            rng[i] = lut_r[k]
    ds.close()
    return mean, rng


def load_bathy(d: Path, aoi_id, H, W):
    """Static bathymetry: (elevation, depth, depth_p25, depth_p75), NaN where absent.

    An ABSENT DEM must produce NaN, never zeros. The obvious derivation --
    `np.where(elev < 0, -elev, 0.0)` -- looks right and is catastrophically wrong when the
    file is missing: `elev` is then all-NaN, `np.nan < 0` is False, so every cell takes the
    0.0 branch and the cube ships a flawless, NaN-free "everything is exactly at sea level"
    bathymetry. That is fabricated data wearing the costume of real data, and downstream
    reconstructs water level from `elevation` + `depth`. Depth is therefore derived only
    where the elevation is actually KNOWN.
    """
    elev = np.full((H, W), np.nan, "float32")
    depth = dp25 = dp75 = None
    f = d / f"{aoi_id}.nc"
    if not f.exists():
        log.warning("  %s: no bathymetry file (%s); elevation/depth/depth_p25/depth_p75 "
                    "will be NaN", aoi_id, f.name)
    else:
        ds = xr.open_dataset(f)

        def g(name):
            return (ds[name].values.astype("float32")
                    if name in ds and ds[name].shape == (H, W) else None)
        if g("elevation") is not None:
            elev = g("elevation")
        else:
            log.warning("  %s: bathymetry file has no usable `elevation` on this grid; "
                        "depth fields will be NaN", aoi_id)
        depth, dp25, dp75 = g("depth"), g("depth_p25"), g("depth_p75")
        ds.close()

    known = np.isfinite(elev)
    if depth is None:      # derive mean depth from elevation -- ONLY where it is known
        depth = np.where(known, np.where(elev < 0, -elev, 0.0), np.nan).astype("float32")
    if dp25 is None:
        dp25 = depth.copy()
    if dp75 is None:
        dp75 = depth.copy()
    return elev, depth, dp25, dp75


def load_insitu(base: Path, aoi_id) -> list[tuple[str, xr.Dataset]]:
    """Every in-situ source's station table for this AoI -> [(source, Dataset), ...].

    In-situ STACKS (D10): a public network and the user's own thermometers are not two pipes
    to the same observations, they are different PLATFORMS, so each source writes its own
    `INSITU/<source>/aligned/<aoi>/` tree and they are read together here.

    Only a directory that actually HOLDS this AoI's table counts as a source. That check is
    not defensive tidiness: `_contribute_stacked_sensor` took EVERY subdirectory as a version
    tag, and one stray leftover directory was enough to produce silent all-sentinel channels
    across a whole project (docs/bug-empty-version-tag-channels.md). Every skip is logged, so
    a mis-named tree is visible rather than absorbed.
    """
    out: list[tuple[str, xr.Dataset]] = []
    if not base.exists():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if d.name == "aligned":
            continue                       # the pre-stacking flat layout, handled below
        f = d / "aligned" / aoi_id / f"{aoi_id}_insitu.nc"
        if not f.exists():
            log.warning("  INSITU/%s holds no %s_insitu.nc; not read as an in-situ source",
                        d.name, aoi_id)
            continue
        out.append((d.name, xr.open_dataset(f)))

    # The pre-0.2 flat layout. Read it rather than silently dropping a project's only ground
    # truth -- but say so, because nothing writes there any more. IOOS was the sole source
    # that ever wrote it, so that is the tag it takes.
    legacy = base / "aligned" / aoi_id / f"{aoi_id}_insitu.nc"
    if legacy.exists():
        if any(tag == "ioos" for tag, _ in out):
            log.warning("  ignoring the legacy flat in-situ layout %s; INSITU/ioos/ supersedes it",
                        legacy.parent)
        else:
            log.warning("  reading in-situ from the LEGACY flat layout %s; move it to "
                        "INSITU/ioos/aligned/ -- it is no longer written there", legacy.parent)
            out.append(("ioos", xr.open_dataset(legacy)))
    return out


def build_insitu(sources: list[tuple[str, xr.Dataset]], g: AoiGrid, days, targets: dict,
                 max_dt_min):
    """In-situ channels: the station's value at each target time, in the station's pixel.

    `targets` maps a channel prefix to one target datetime per day -- 'insitu' (the daily
    reference time) and one per sensor ('eco', 'lst', 'modis') at that sensor's chosen
    overpass. Each becomes a sparse (T,H,W) channel: NaN everywhere except the cells where
    stations sit. A satellite pixel and a buoy pixel then line up exactly, at the same
    instant, which is the whole point of carrying in-situ at all.

    A station goes into the cell it FALLS in -- no snapping to a water mask (that was the
    last water-mask consumer; masking is a downstream determination now, per Goal 3). A
    station in a land cell stays there.

    Every SOURCE's stations are merged into ONE channel set, because stations are ROWS, not
    channels: they occupy disjoint pixels anyway, and splitting them per source would multiply
    the whole `<sensor>_insitu_*` family while making `insitu_sst` stop meaning "ground truth".
    Which source a platform came from is recorded per station in the table instead. The sources
    are read as a LIST rather than concatenated: an outer join across two unrelated time axes
    (6-minute CO-OPS against a 1-minute logger) would build a dense (station, time) block an
    order of magnitude larger than either, for nothing.

    Returns (channels, station_table, station_map).
    """
    H, W = g.height, g.width

    chans = {k: _empty3d(days, H, W) for k in targets}
    dts = {k: _empty3d(days, H, W) for k in targets if k != "insitu"}
    counts = np.zeros((len(days), H, W), dtype="float32")   # stations sharing a cell
    station_map = np.zeros((H, W), dtype="uint16")          # 0 = no station
    table = []
    # Sums + contributor counts, so N co-located stations give their true MEAN. A running
    # `(prev + v) / 2` is not one for N > 2 -- it yields a/4 + b/4 + c/2.
    #
    # Accumulated PER STATION CELL -- {(row, col): {sums/dt_sums/hits: {key: (T,) array}}} --
    # not as dense (T,H,W) grids. The values exist in a handful of pixels, so the dense form
    # spent three arrays per target (two of them float64) on a grid that is NaN almost
    # everywhere: 104 bytes per cell per day, of which 72 were these accumulators. On a large
    # AoI that is hundreds of GB to hold a few buoys. Only the EMITTED channels are dense.
    cells: dict[tuple[int, int], dict] = {}
    seen_ids: dict[str, str] = {}                           # station id -> its source

    for src, ids in sources:
        lons = np.asarray(ids["lon"].values, dtype="float64")
        lats = np.asarray(ids["lat"].values, dtype="float64")
        placed = insitu.station_pixels(lons, lats, g)         # water=None: no snapping

        times = pd.DatetimeIndex(ids["time"].values)
        sst = ids["sst"].values                     # (station, time)

        for s, place in enumerate(placed):
            sid = str(ids["station_id"].values[s])
            if not place["inside"]:
                log.warning("  in-situ station %s [%s] falls outside the AoI grid; dropped",
                            sid, src)
                continue
            r, c = place["row"], place["col"]

            # Two networks may name a platform the same thing; the table's ids must stay
            # unique or a pixel's `insitu_station` index would point at either of them.
            if seen_ids.get(sid, src) != src:
                log.warning("  in-situ station id %r is used by both %s and %s; recording "
                            "the latter as %s", sid, seen_ids[sid], src, f"{sid}@{src}")
                sid = f"{sid}@{src}"
            seen_ids.setdefault(sid, src)

            table.append({"index": len(table) + 1, "id": sid,
                          "name": str(ids["station_name"].values[s]),
                          "source": src,
                          "lat": float(lats[s]), "lon": float(lons[s]),
                          "row": r, "col": c})
            station_map[r, c] = len(table)

            acc = cells.setdefault((r, c), {
                "sums": {k: np.zeros(len(days), dtype="float64") for k in targets},
                "dt_sums": {k: np.zeros(len(days), dtype="float64") for k in dts},
                "hits": {k: np.zeros(len(days), dtype="int32") for k in targets},
            })
            for i in range(len(days)):
                for key, tgt in targets.items():
                    v, dt = insitu.value_at(times, sst[s], tgt[i], max_dt_min)
                    if not np.isfinite(v):
                        continue
                    acc["sums"][key][i] += v
                    acc["hits"][key][i] += 1
                    if key in acc["dt_sums"]:
                        acc["dt_sums"][key][i] += dt
                counts[i, r, c] += 1

    # Densify, one station cell at a time. The dtypes going into `np.divide` are the ones the
    # dense form used (float64 sums / int32 hits -> float32 out, casting="unsafe"): the ufunc
    # picks its loop from them, so widening `hits` here could move a value in the last bit.
    # Everything not written stays NaN from `_empty3d`, which is what a cell with no station
    # -- or a day with no observation in tolerance -- means.
    for (r, c), acc in cells.items():
        for key in targets:
            hit = acc["hits"][key]
            got = hit > 0
            col = np.full(len(days), np.nan, dtype="float32")
            np.divide(acc["sums"][key], hit, out=col, where=got, casting="unsafe")
            chans[key][:, r, c] = col
            if key in dts:
                dcol = np.full(len(days), np.nan, dtype="float32")
                np.divide(acc["dt_sums"][key], hit, out=dcol, where=got, casting="unsafe")
                dts[key][:, r, c] = dcol

    out = {}
    for key in targets:
        name = "insitu_sst" if key == "insitu" else f"{key}_insitu_sst"
        out[name] = (("time", "y", "x"), chans[key])
        if key in dts:
            out[f"{key}_insitu_dt_min"] = (("time", "y", "x"), dts[key])
    out["insitu_n"] = (("time", "y", "x"), counts)
    return out, table, station_map


def cmems_channels(d: Path, aoi_id) -> list[str]:
    """The CMEMS variables actually on disk (thetao_0m, thetao_10m, zos, ...).

    Discovered from the files rather than re-derived from config, so the cube carries
    exactly the variables x depths that were acquired -- no second list to keep in sync.
    """
    if not d.exists():
        return []
    for f in sorted(d.glob(f"{aoi_id}_*.nc")):
        with xr.open_dataset(f) as ds:
            return [v for v in ds.data_vars if v != "valid"]
    return []


def load_bathy_attrs(d: Path, aoi_id) -> dict:
    """The bathymetry file's global attrs (its `source` is the DEM fingerprint)."""
    f = d / f"{aoi_id}.nc"
    if not f.exists():
        return {}
    with xr.open_dataset(f) as ds:
        return dict(ds.attrs)


def load_landcover(d: Path, aoi_id, H, W):
    """Static land-cover water mask -> float (1=water, 0=land, NaN=unknown/absent)."""
    water = np.full((H, W), np.nan, "float32")
    f = d / f"{aoi_id}.nc"
    if f.exists():
        ds = xr.open_dataset(f)
        if "water" in ds and ds["water"].shape == (H, W):
            water = ds["water"].values.astype("float32")
        ds.close()
    return water


# --------------------------------------------------------------------------- #
# Assemble one AoI onto its shared grid
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The contributor protocol
#
# Every product -- sensor and non-sensor alike -- contributes to the cube through ONE uniform
# `(ctx) -> channels` mechanism, and the run order is DERIVED (topological sort) from each
# contributor's declared slot reads/writes rather than a hand-kept sequence. This replaces the
# old asymmetry where the sensor family was registry-driven but every other product was a
# hand-written block in `assemble_aoi` -- an omission the loud invariant (`_check_contributors`)
# now makes impossible. See docs/DEVELOPMENT.md.
# --------------------------------------------------------------------------- #

# Shared-intermediate slot names, as module constants so a typo is a NameError, not a silent
# miss. After the S1 raw-output simplification the surface is just these two:
SLOT_SENSOR_TIMES = "sensor_times"   # {prefix: [datetime|None per day]}  (sensors -> met_overpass, insitu)
SLOT_REF_UTC = "ref_utc"             # [datetime|None per day]  met reference time (met -> insitu)

T3 = ("time", "y", "x")


@dataclass
class AssemblyContext:
    """The shared state a contributor reads and writes while one AoI's cube is assembled.

    A contributor mutates `channels` (name -> (dims, array), merged into the Dataset),
    `slots` (shared intermediates keyed by SLOT_*), `global_attrs` (ds.attrs), and
    `var_attrs` (per-channel attrs). It never touches the Dataset object directly -- the
    orchestrator builds that once every contributor has run.
    """
    g: AoiGrid
    eff: dict
    days: Any
    aid: str
    H: int
    W: int
    slots: dict[str, Any]
    channels: dict[str, tuple]
    global_attrs: dict[str, Any]
    var_attrs: dict[str, dict]

    def adir(self, product: str, source: str | None = None) -> Path:
        """The `<DIR>[/<source>]/aligned/<aoi>` folder a product's acquisition stage wrote.

        A DISTINCT-DATA (stacked) product nests each source under its own `<source>` level
        (bathymetry: `BATHYMETRY/cudem/aligned/<aoi>`); every other product is flat
        (`<DIR>/aligned/<aoi>`), i.e. pass `source=None`.
        """
        return (self.eff["aligned_root"]
                / products.aligned_rel(PRODUCT_DIRS[product], source) / self.aid)

    def emit(self, name: str, dims, arr, **attrs) -> None:
        """Add a channel (and, optionally, merge in its per-variable attrs)."""
        self.channels[name] = (dims, arr)
        if attrs:
            self.var_attrs.setdefault(name, {}).update(attrs)


@dataclass(frozen=True)
class Contributor:
    """One product's (or derived field's) contribution to the cube.

    `reads` / `writes` are SLOT_* names; the assembler topologically sorts the registry so
    every slot is written before it is read (see `_topo_order`). `key` is the product's
    registry name (`s.product.value`), or `"derived:<name>"` / a plain label for a pure
    computation; the loud-omission invariant asserts every non-sensor product has one.
    """
    key: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    fn: Callable[[AssemblyContext], None]


# --------------------------------------------------------------------------- #
# Contributors (one per product; two derived). Each reads/writes only what it declares.
# --------------------------------------------------------------------------- #
_DATUM_ATTRS = ("datum_offset_m", "datum_status", "datum_method", "dem_vertical_datum")


def _contribute_bathymetry(ctx: AssemblyContext) -> None:
    """DISTINCT-DATA per source (D5): one channel set PER stacked DEM source, discovered from
    the per-source directories on disk (`BATHYMETRY/<src>/aligned/<aoi>`). Each source's
    DEM->MSL datum offset -- resolved and stamped by the bathymetry module when it acquired
    that DEM -- travels as attributes on its `elevation_<src>` channel, so a downstream user
    references THAT DEM's elevation to MSL with its OWN offset (CUDEM/NAVD88 != GMRT/MSL)."""
    base = ctx.eff["aligned_root"] / PRODUCT_DIRS["bathymetry"]
    if not base.exists():
        return
    for src in sorted(d.name for d in base.iterdir() if d.is_dir()):
        adir = ctx.adir("bathymetry", src)
        if not (adir / f"{ctx.aid}.nc").exists():
            continue                              # e.g. the _tmp scratch dir, or a lost AoI
        elev, depth, depth_p25, depth_p75 = load_bathy(adir, ctx.aid, ctx.H, ctx.W)
        attrs = load_bathy_attrs(adir, ctx.aid)
        datum = {k: attrs[k] for k in _DATUM_ATTRS if k in attrs}
        ctx.emit(f"elevation_{src}", ("y", "x"), elev, units="m",
                 long_name=f"{src} DEM elevation in its native vertical datum (+ up); add "
                           "datum_offset_m to reference it to MSL", **datum)
        ctx.emit(f"depth_{src}", ("y", "x"), depth)
        ctx.emit(f"depth_p25_{src}", ("y", "x"), depth_p25)
        ctx.emit(f"depth_p75_{src}", ("y", "x"), depth_p75)


def _load_sensor(ctx: AssemblyContext, sp, adir):
    """Read one sensor scene-stream from one aligned dir, applying the sensor's validity
    rules. Factored out so a flat sensor and one per-version tree share exactly one path."""
    return load_clearest_overpass(
        adir, ctx.aid, ctx.days, ctx.H, ctx.W,
        water_is_land=sp.water_is_land, use_cloud=sp.use_cloud,
        qc_levels=list(sp.qc_levels) if sp.qc_levels is not None else None,
        trust_valid=sp.trust_valid, read_footprint=sp.has_footprint)


# The footprint index is a per-day identity, and saying so on the variable is the only place a
# user reading the cube will see it: ids restart at 0 in every granule, so grouping across the
# time axis merges unrelated native pixels into one bucket -- silently, and wrongly.
_FOOTPRINT_ATTRS = dict(
    long_name="native sensor pixel index this grid cell was resampled from; -1 = none",
    comment="ids identify a pixel WITHIN one day's chosen scene only -- they restart at 0 "
            "for every scene and are NOT comparable across days")


def _contribute_flat_sensor(ctx: AssemblyContext, s, sensor_times: dict) -> None:
    """A single-collection sensor (Landsat, MODIS): one flat `<DIR>/aligned/<aoi>` tree, one
    channel-set under the bare prefix, one entry in `sensor_times`."""
    sp = s.sensor
    sst, cloud, valid, hour, times, footprint = _load_sensor(
        ctx, sp, ctx.adir(s.product.value))
    pre = sp.prefix
    ctx.emit(f"{pre}_sst", T3, sst)
    if sp.has_cloud:          # MODIS arrives pre-filtered with no cloud layer; an all-zero
        ctx.emit(f"{pre}_cloud", T3, cloud)   # channel would falsely read "never cloudy"
    ctx.emit(f"{pre}_valid", T3, valid)
    ctx.emit(f"{pre}_hour", ("time",), hour)
    # Only when a granule actually carried the layer (it is optional at acquisition): an
    # all -1 channel would read as "nothing was ever in swath" rather than "never recorded".
    if footprint is not None:
        ctx.emit(f"{pre}_footprint_id", T3, footprint, **_FOOTPRINT_ATTRS)
    sensor_times[pre] = times


def _contribute_stacked_sensor(ctx: AssemblyContext, s, sensor_times: dict) -> None:
    """A STACKED-DATA sensor (ECOSTRESS collection versions, D10): one channel-set PER source
    tag, discovered from the per-source dirs on disk (`ECOSTRESS/<tag>/aligned/<aoi>`), emitted
    tag-last (`eco_sst_v002`, `eco_sst_v003`) like every other per-source channel.

    Downstream matchups (met_overpass, tide_overpass, in-situ) key off ONE overpass identity
    per sensor, so a single `sensor_times[prefix]` is built by merging the per-version chosen
    scenes, preferring the config-listed version order per day (D5) -- the versions describe the
    same physical overpasses, so a per-day preference is honest, and the raw DATA channels stay
    fallback-free (one per version).
    """
    sp = s.sensor
    base = ctx.eff["aligned_root"] / PRODUCT_DIRS[s.product.value]
    if not base.exists():
        sensor_times[sp.prefix] = [None] * len(ctx.days)
        return
    # Preference order: config-listed versions first (in that order), then any other tag found
    # on disk (sorted), so a version acquired but dropped from the config still contributes a
    # channel and merges last rather than vanishing.
    pref = ctx.eff["sensor_version_pref"].get(s.product.value, [])
    on_disk = {d.name for d in base.iterdir() if d.is_dir()}
    ordered = [t for t in pref if t in on_disk] + sorted(on_disk - set(pref))

    per_tag_times: dict[str, list] = {}
    for tag in ordered:
        sst, cloud, valid, hour, times, footprint = _load_sensor(
            ctx, sp, ctx.adir(s.product.value, tag))
        pre = sp.prefix
        ctx.emit(f"{pre}_sst_{tag}", T3, sst)
        if sp.has_cloud:
            ctx.emit(f"{pre}_cloud_{tag}", T3, cloud)
        ctx.emit(f"{pre}_valid_{tag}", T3, valid)
        ctx.emit(f"{pre}_hour_{tag}", ("time",), hour)
        if footprint is not None:
            ctx.emit(f"{pre}_footprint_id_{tag}", T3, footprint, **_FOOTPRINT_ATTRS)
        per_tag_times[tag] = times

    # Single overpass identity: first-listed version wins each day, later versions fill gaps.
    merged: list = [None] * len(ctx.days)
    for tag in ordered:
        for i, t in enumerate(per_tag_times[tag]):
            if merged[i] is None and t is not None:
                merged[i] = t
    sensor_times[sp.prefix] = merged


def _contribute_sensors(ctx: AssemblyContext) -> None:
    """THE PER-OVERPASS THERMAL SENSORS -- one COLLECTIVE contributor over the registry.

    Each sensor's validity rules (inverted water polarity, gate-on-QC-not-cloud, trust-the-
    valid-layer) live on its ProductSpec.sensor, so a fourth sensor gets its full channel set
    -- `_sst`, `_cloud`, `_valid`, `_hour`, plus overpass-met and in-situ matchups downstream
    -- from its spec alone. A STACKED-DATA sensor (ECOSTRESS) forks its DATA channels per
    version (`eco_sst_v002/_v003`) while keeping ONE overpass identity for the matchups. Writes
    each sensor's chosen scene time into `sensor_times`.
    """
    sensor_times: dict[str, list] = {}
    for s in products.sensors():
        if s.is_stacked_data:
            _contribute_stacked_sensor(ctx, s, sensor_times)
        else:
            _contribute_flat_sensor(ctx, s, sensor_times)
    ctx.slots[SLOT_SENSOR_TIMES] = sensor_times


def _contribute_mur(ctx: AssemblyContext) -> None:
    # MUR ships its OBSERVED values with honest NaN gaps -- no NN water fill (S1). Filling is
    # a downstream determination the model makes per-process from the raw channels.
    ctx.emit("mur_sst", T3, load_daily_sensor(ctx.adir("mur"), ctx.aid, ctx.days, ctx.H, ctx.W, "sst"))


_MET_FORCING_VARS = ("airtemp", "wind_u", "wind_v", "wind_speed", "swrad", "cloud_cover")


def _contribute_met(ctx: AssemblyContext) -> None:
    """Met FORCING per source (D10): one snapshot per day at the REFERENCE time of day (default
    10:30 local solar) rather than a daily mean, emitted as `<var>_<source>` per stacked source
    (`MET/<src>/aligned`). Also writes `ref_utc` -- the daily reference instant the in-situ
    channel samples against (source-independent, so always written).
    """
    lon_c = 0.5 * (ctx.g.search_bbox[0] + ctx.g.search_bbox[2])
    ctx.slots[SLOT_REF_UTC] = [
        met_mod.reference_time_utc(dd, lon_c, ctx.eff["ref_hours"], ctx.eff["ref_basis"])
        if ctx.eff["ref_hours"] is not None else None for dd in ctx.days]

    base = ctx.eff["aligned_root"] / PRODUCT_DIRS["met"]
    label = "reference" if ctx.eff["met_time"] == "reference" else "daily_mean"
    if base.exists():
        for src in sorted(dd.name for dd in base.iterdir() if dd.is_dir()):
            d = ctx.adir("met", src)
            mprefix, mlabel = met_prefix(d, ctx.aid, ctx.days, ctx.eff["met_time"])
            label = mlabel
            for var in _MET_FORCING_VARS:
                ctx.emit(f"{var}_{src}", T3,
                         load_daily_sensor(d, ctx.aid, ctx.days, ctx.H, ctx.W, var, prefix=mprefix))
    ctx.global_attrs["met_time"] = label


def _overpass_met_vars(d: Path, aoi_id) -> list[str]:
    """The output variables in a met_overpass source's snapshot files (discovered from disk,
    like CMEMS -- so the channel set is exactly what was acquired)."""
    if not d.exists():
        return []
    for f in sorted(d.glob(f"{aoi_id}_*T*.nc")):
        with xr.open_dataset(f) as ds:
            return list(ds.data_vars)
    return []


def _contribute_met_overpass(ctx: AssemblyContext) -> None:
    """Met at each sensor's OWN overpass, for the user's `(sensor, source)` COMBINATIONS only
    (D13). Emits `<sensor>_<var>_<source>` -- the source's snapshot read at that sensor's chosen
    scene time (`sensor_times`), so a pre-dawn ECOSTRESS scene and a mid-morning Landsat scene
    on the same day do not share one value. Reads `sensor_times`.
    """
    sensor_times = ctx.slots[SLOT_SENSOR_TIMES]
    for sensor, src in ctx.eff["met_overpass_combos"]:
        tt = sensor_times.get(sensor)
        if tt is None:                    # a combo naming an unloaded sensor -> nothing to align
            continue
        d = ctx.adir("met_overpass", src)
        for var in _overpass_met_vars(d, ctx.aid):
            ctx.emit(f"{sensor}_{var}_{src}", T3,
                     load_at_times(d, ctx.aid, tt, ctx.H, ctx.W, var),
                     long_name=f"{var} at the {sensor} overpass ({src})")


def _contribute_tides(ctx: AssemblyContext) -> None:
    # DISTINCT-DATA per source (D10): one daily tide channel set per stacked source
    # (`tide_coops`, `tide_eo_tides`), discovered from `TIDE/<src>/aligned/<aoi>`. A source
    # with no series here (e.g. no CO-OPS gauge nearby) simply has no channel.
    base = ctx.eff["aligned_root"] / PRODUCT_DIRS["tides"]
    if not base.exists():
        return
    for src in sorted(d.name for d in base.iterdir() if d.is_dir()):
        d = ctx.adir("tides", src)
        if not (d / f"{ctx.aid}_tides.nc").exists():
            continue
        tide, tide_range = load_tide_daily(d, ctx.aid, ctx.days)
        ctx.emit(f"tide_{src}", ("time",), tide)
        ctx.emit(f"tide_range_{src}", ("time",), tide_range)


def _contribute_tide_overpass(ctx: AssemblyContext) -> None:
    """Tide at each sensor's OWN overpass, for the user's `(sensor, tide_source)` COMBINATIONS
    (D17): the per-source hourly tide series interpolated to the sensor's chosen scene hour.
    A DERIVED contributor -- the tide series is smooth and already on disk, so this is an
    interpolation, not a re-acquisition (unlike met_overpass). Emits `<sensor>_tide_<src>`.
    Reads `sensor_times`.
    """
    combos = ctx.eff["tide_overpass_combos"]
    if not combos:
        return
    sensor_times = ctx.slots[SLOT_SENSOR_TIMES]
    for sensor, src in combos:
        tt = sensor_times.get(sensor)
        if tt is None:                    # a combo naming an unloaded sensor -> nothing to align
            continue
        series = water_level.load_tide_series(ctx.adir("tides", src), ctx.aid)
        hours = [t.hour + t.minute / 60.0 if t is not None else np.nan for t in tt]
        th = water_level.tide_at_overpass(series, ctx.days, hours)
        ctx.emit(f"{sensor}_tide_{src}", ("time",), th,
                 units="m", long_name=f"tide at the {sensor} overpass ({src}), rel. MSL")


def _contribute_cmems(ctx: AssemblyContext) -> None:
    # DISTINCT-DATA per source TAG (D10): one channel set per stacked CMEMS source, discovered
    # from the per-source dirs (`CMEMS/<tag>/aligned/<aoi>`). The offshore water column ships
    # with honest NaN gaps (its ~9 km land mask can swallow an estuary; downstream fills). The
    # variables/depths within a source are discovered from its files.
    base = ctx.eff["aligned_root"] / PRODUCT_DIRS["cmems"]
    if not base.exists():
        return
    for src in sorted(d.name for d in base.iterdir() if d.is_dir()):
        d = ctx.adir("cmems", src)
        for var in cmems_channels(d, ctx.aid):
            ctx.emit(f"cmems_{var}_{src}", T3,
                     load_daily_sensor(d, ctx.aid, ctx.days, ctx.H, ctx.W, var))


def _contribute_landcover(ctx: AssemblyContext) -> None:
    # Land-cover water, shipped RAW as a loss-filter channel (1=water, 0=land; unknown ->
    # water, a no-op filter). Land-masking is a downstream MODELLING determination now (S1).
    lc_raw = load_landcover(ctx.adir("landcover"), ctx.aid, ctx.H, ctx.W)
    lc_known = np.isfinite(lc_raw)
    ctx.emit("landcover_water", ("y", "x"), np.where(lc_known, lc_raw > 0.5, True).astype("uint8"))


def _contribute_insitu(ctx: AssemblyContext) -> None:
    """In-situ: the cube's only ground truth. The value goes in the cell the station sits in,
    sampled at the SAME INSTANT each satellite flew -- so a scene can be validated against a
    buoy pixel-for-pixel and minute-for-minute -- plus one at the daily reference time,
    contemporaneous with the met channels. Reads `sensor_times` and `ref_utc`.
    """
    if not ctx.eff["insitu"]:
        return
    sources = load_insitu(ctx.eff["aligned_root"] / PRODUCT_DIRS["insitu"], ctx.aid)
    if not sources:
        return
    targets = {"insitu": ctx.slots[SLOT_REF_UTC], **ctx.slots[SLOT_SENSOR_TIMES]}
    insitu_vars, station_table, station_map = build_insitu(
        sources, ctx.g, ctx.days, targets, ctx.eff["insitu_max_dt_min"])
    for _, ids in sources:
        ids.close()
    log.info("  in-situ: %d station(s) placed from %d source(s): %s",
             len(station_table), len(sources), ", ".join(s for s, _ in sources))
    for name, (dims, arr) in insitu_vars.items():
        ctx.emit(name, dims, arr)
    if station_map is not None:
        ctx.emit("insitu_station", ("y", "x"), station_map)
    if station_table:
        # The station map is an index INTO this table, so the table must travel with the
        # cube -- a pixel that says "station 3" is useless without it.
        ctx.global_attrs["insitu_stations"] = json.dumps(station_table)
        ctx.var_attrs.setdefault("insitu_station", {})["long_name"] = (
            "index into the insitu_stations attr (0 = none)")


def _contribute_doy(ctx: AssemblyContext) -> None:
    doy = ctx.days.dayofyear.values.astype("float32")
    ctx.emit("doy_sin", ("time",), np.sin(2 * np.pi * doy / 365.25).astype("float32"))
    ctx.emit("doy_cos", ("time",), np.cos(2 * np.pi * doy / 365.25).astype("float32"))


# The registry of contributors. Declaration order is the deterministic tie-break for the
# topo-sort (see `_topo_order`); it is NOT the run order, which is derived from reads/writes.
CONTRIBUTORS: tuple[Contributor, ...] = (
    Contributor("bathymetry", (), (), _contribute_bathymetry),
    Contributor("sensors", (), (SLOT_SENSOR_TIMES,), _contribute_sensors),
    Contributor("mur", (), (), _contribute_mur),
    Contributor("met", (), (SLOT_REF_UTC,), _contribute_met),
    Contributor("met_overpass", (SLOT_SENSOR_TIMES,), (), _contribute_met_overpass),
    Contributor("tides", (), (), _contribute_tides),
    Contributor("tide_overpass", (SLOT_SENSOR_TIMES,), (), _contribute_tide_overpass),
    Contributor("cmems", (), (), _contribute_cmems),
    Contributor("landcover", (), (), _contribute_landcover),
    Contributor("insitu", (SLOT_SENSOR_TIMES, SLOT_REF_UTC), (), _contribute_insitu),
    Contributor("derived:doy", (), (), _contribute_doy),
)


def _topo_order(contributors: tuple[Contributor, ...]) -> list[Contributor]:
    """Order contributors so every slot is written before it is read.

    Edges run writer(slot) -> reader(slot); ties are broken by declaration order for
    determinism (mirrors pipeline.process_order). `_check_contributors` guarantees every
    `reads` slot has a producer, so the sort cannot starve on a missing slot -- a cycle
    (which `required_vars`/slot design should preclude) raises rather than dropping work.
    """
    import heapq
    idx = {c.key: i for i, c in enumerate(contributors)}
    by_key = {c.key: c for c in contributors}
    writers: dict[str, list[str]] = {}
    for c in contributors:
        for w in c.writes:
            writers.setdefault(w, []).append(c.key)
    adj: dict[str, set] = {c.key: set() for c in contributors}
    indeg = {c.key: 0 for c in contributors}
    for c in contributors:
        for r in c.reads:
            for wk in writers.get(r, ()):
                if c.key not in adj[wk]:
                    adj[wk].add(c.key)
                    indeg[c.key] += 1
    ready = [idx[c.key] for c in contributors if indeg[c.key] == 0]
    heapq.heapify(ready)
    out: list[Contributor] = []
    while ready:
        c = contributors[heapq.heappop(ready)]
        out.append(c)
        for nk in adj[c.key]:
            indeg[nk] -= 1
            if indeg[nk] == 0:
                heapq.heappush(ready, idx[nk])
    if len(out) != len(contributors):
        stuck = sorted(k for k in indeg if by_key[k] not in out)
        raise RuntimeError(f"contributor graph has a cycle among slots; unresolved: {stuck}")
    return out


def _check_contributors() -> None:
    """Fail LOUDLY at import if a non-sensor product has cube presence but no contributor.

    This closes the silent-omission trap: a new covariate that acquires to disk from its
    ProductSpec but is never wired into the cube used to vanish without a trace. Now it raises
    here until a contributor is registered (or the product sets `cube_opt_out=True`). Also
    asserts every declared `reads` slot has a producer, so `_topo_order` cannot starve.
    """
    keys = {c.key for c in CONTRIBUTORS}
    for s in products.REGISTRY:
        if s.sensor is not None:        # covered collectively by the `sensors` contributor
            continue
        if s.cube_opt_out:              # a product that deliberately has no cube presence
            continue
        if s.product.value not in keys:
            raise RuntimeError(
                f"{s.product.value}: no cube contributor registered. Add one to "
                "datacube.CONTRIBUTORS, or set cube_opt_out=True on its ProductSpec.")
    produced = {w for c in CONTRIBUTORS for w in c.writes}
    for c in CONTRIBUTORS:
        missing = set(c.reads) - produced
        if missing:
            raise RuntimeError(
                f"contributor {c.key!r} reads unproduced slot(s) {sorted(missing)}.")


_check_contributors()


def assemble_aoi(g: AoiGrid, eff: dict, days) -> xr.Dataset:
    """Build the analysis-ready Dataset for one AoI from its aligned files.

    UNIFORM CONTRIBUTOR PROTOCOL: every product -- sensor and non-sensor alike -- contributes
    through one `(ctx) -> channels` mechanism (`CONTRIBUTORS`), run in an order derived by
    topological sort from each contributor's declared slot reads/writes. A new non-sensor
    covariate needs a `ProductSpec`, a module, and a registered `Contributor`; forgetting the
    last is a hard error at import (`_check_contributors`), not a silent omission. See
    docs/DEVELOPMENT.md.
    """
    ctx = AssemblyContext(
        g=g, eff=eff, days=days, aid=g.name, H=g.height, W=g.width,
        slots={}, channels={}, global_attrs={}, var_attrs={})
    for c in _topo_order(CONTRIBUTORS):
        c.fn(ctx)

    xs, ys = g.xy_centers()
    ds = xr.Dataset(ctx.channels, coords={"time": days, "y": ys, "x": xs})
    ds.attrs.update(aoi_id=ctx.aid, crs=g.target_crs)
    ds.attrs.update(ctx.global_attrs)
    for name, attrs in ctx.var_attrs.items():
        ds[name].attrs.update(attrs)

    # COVERAGE. The time axis is built from the CONFIG (start..end), not from the data, and
    # every loader defaults a missing day to NaN. So a run that lost 40 of 100 days to a
    # flaky network still yields a 100-step cube whose gaps are indistinguishable from cloudy
    # days. Count what actually landed, stamp it on the cube, and say so when a product is
    # thin. `prod` (which products wrote files) also feeds provenance, so it is collected once.
    prod = provenance.collect(eff["aligned_root"], ctx.aid, PRODUCT_DIRS)
    cov = coverage(ds, days, present=set(prod),
                   sparse=eff.get("sparse_daily", {}).get(ctx.aid))
    ds.attrs["coverage"] = json.dumps(cov, sort_keys=True)
    for product, c in sorted(cov.items()):
        # A product the config deliberately fetched on only some days is thin ON PURPOSE.
        if c["fraction"] < COVERAGE_WARN and not c.get("sparse"):
            log.warning("  %s: %s covers only %d of %d day(s) (%.0f%%) -- the rest are NaN "
                        "slices, which look exactly like cloudy days. Check the run report "
                        "for what failed.",
                        ctx.aid, product, c["days_with_data"], c["days_expected"],
                        100 * c["fraction"])

    # (The DEM->MSL datum offset ships PER SOURCE, as attributes on each `elevation_<src>`
    # channel -- resolved and stamped by the bathymetry module when it acquired that DEM, and
    # surfaced by `_contribute_bathymetry`. CUDEM/NAVD88 and GMRT/MSL need different offsets,
    # so a single global attr would be wrong for one of them.)

    # PROVENANCE: the config that built this cube, and for every field the source(s) it came
    # from and when they were accessed. Zarr attrs must be JSON-serialisable, so the structured
    # parts are JSON strings.
    rec = provenance.build(eff["project"], list(ds.data_vars), prod)
    ds.attrs.update(
        created_at=rec["created_at"], package_version=rec["package_version"],
        code_version=rec["code_version"],
        config_sha256=rec["config_sha256"] or "", config_path=rec["config_path"] or "",
        config_yaml=rec["config_yaml"] or "",
        provenance=json.dumps(rec["fields"], sort_keys=True),
        provenance_products=json.dumps(rec["products"], sort_keys=True))
    guessed = [p for p, r in prod.items() if r["basis"] == provenance.FILE_MTIME]
    if guessed:
        log.warning("  %s: access dates for %s came from FILE MTIMES, not recorded stamps "
                    "(acquired before provenance existed, or the tree was copied)",
                    ctx.aid, ", ".join(sorted(guessed)))
    return ds


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
COVERAGE_WARN = 0.95      # below this fraction of days, a daily product is reported thin

# Products that SHOULD have a value every day, and the channel that proves it. DERIVED from
# the registry (products.ProductSpec.coverage_channel).
#
# Only daily products can be judged this way: ECOSTRESS/Landsat/MODIS are OVERPASS sensors,
# so a day with no scene is normal and not a defect -- warning on those would train the user
# to ignore the warning. They declare no coverage_channel, so they are not listed here.
# `cmems` is matched by prefix below, because its channels are config-dependent and so
# cannot be named up front.
DAILY_CHANNELS = {s.product.value: s.coverage_channel
                  for s in products.REGISTRY if s.coverage_channel}


def coverage(ds: xr.Dataset, days, present=None, sparse=None) -> dict:
    """{product: {days_with_data, days_expected, fraction}} for the daily products.

    A day "has data" if its slice holds at least one finite value. This is the check that
    makes a network-shaped hole visible: the cube's time axis is always len(days) long, so
    the ONLY evidence a day was lost is that its slice is entirely NaN.

    `present` is the set of products that actually wrote files for this AoI. A product that
    was never run is ABSENT, not thin -- reporting it at 0% would bury the products that
    really are thin under noise about products nobody asked for.

    `sparse` is the set the CONFIG asked to be partial -- MUR restricted to overpass days.
    Their coverage is still measured and still reported (the number is the honest answer, and
    a reader of the cube should see it), but flagged `sparse: true` so the caller does not
    warn about a gap the user chose. A warning that fires on every run of a correct config is
    how a real one gets ignored.
    """
    out = {}
    # Coverage-channel PREFIXES. A per-source product (D5) has no single channel any more, so
    # judge it present on a day if ANY of its per-source channels is finite (S4.7): met's
    # `airtemp` prefix matches `airtemp_hrrr`/`airtemp_era5`, tides' `tide` matches
    # `tide_<src>`/`tide_range_<src>`, mur's `mur_sst` is still the bare channel. `cmems` is
    # discovered (its channels are config-dependent). A stacked source with NO regional coverage
    # is all-NaN by design, so "any source finite" is the honest presence test.
    prefixes = dict(DAILY_CHANNELS)
    prefixes["cmems"] = "cmems"

    for product, c in prefixes.items():
        # `present` is keyed by the product's own name, and so is `prefixes` -- one registry,
        # one name. The alias table that used to reconcile `tide` with `tides` is gone.
        if present is not None and product not in present:
            continue
        chans = [v for v in ds.data_vars
                 if (v == c or v.startswith(c + "_")) and "time" in ds[v].dims]
        if not chans:
            continue
        has = np.zeros(len(days), dtype=bool)
        for v in chans:
            finite = np.isfinite(ds[v].values)
            axes = tuple(range(1, finite.ndim))       # everything but time
            has |= (finite.any(axis=axes) if axes else finite)
        n = int(has.sum())
        out[product] = {"days_with_data": n, "days_expected": len(days),
                        "fraction": (n / len(days)) if len(days) else 0.0}
        if sparse and product in sparse:
            out[product]["sparse"] = True
    return out


# --------------------------------------------------------------------------- #
# Compression / encoding
# --------------------------------------------------------------------------- #
_SHUFFLE = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def _blosc_codec(cname: str, clevel: int, shuffle: str):
    """A Blosc codec for the installed Zarr (v3 BloscCodec, else numcodecs Blosc)."""
    import zarr
    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.codecs import BloscCodec
        return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle), "compressors"
    from numcodecs import Blosc
    return Blosc(cname=cname, clevel=clevel, shuffle=_SHUFFLE[shuffle]), "compressor"


def build_encoding(ds: xr.Dataset, compression: CompressionSpec, chunks: dict) -> dict:
    """Per-variable chunk + Blosc(zstd) encoding.

    Lossless: dtypes are untouched. Integer masks get BITSHUFFLE (long runs of a
    constant class compress hard); floats get the configured shuffle. The result
    keys ('compressors' for Zarr v3, 'compressor' for v2) match the installed Zarr.
    """
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        ch = tuple(min(chunks.get(d, ds.sizes[d]), ds.sizes[d]) for d in dims)
        shuffle = "bitshuffle" if ds[v].dtype == np.uint8 else compression.shuffle
        codec, key = _blosc_codec(compression.codec, compression.level, shuffle)
        e = {key: (codec,) if key == "compressors" else codec}
        if ch:
            e["chunks"] = ch
        enc[v] = e
    return enc


# --------------------------------------------------------------------------- #
# Write (atomic, NFS-safe)
# --------------------------------------------------------------------------- #
def write_zarr(ds: xr.Dataset, zpath: Path, encoding: dict):
    """Write a Zarr cube to `zpath`, which must not exist. NOT atomic -- see `write_zarr_safe`.

    Exposed on its own for the one caller that has to keep its SOURCE store open across the
    write and so cannot let the swap happen inside `write_zarr_safe`: `preprocess.run`, which
    rewrites the cube it is reading (it drives `store.atomic` itself, writing here and closing
    the source before the block ends).
    """
    with warnings.catch_warnings():
        # Zarr v3 warns that consolidated metadata isn't in the v3 spec; xarray
        # still writes+reads it and it speeds opening a many-variable cube.
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        ds.to_zarr(zpath, mode="w-", consolidated=True, encoding=encoding)


def write_zarr_safe(ds: xr.Dataset, zpath: Path, encoding: dict):
    """Write a Zarr cube ATOMICALLY: the final path only ever holds a COMPLETE cube.

    `store.atomic` builds the cube in a scratch dir and swaps it in only once `to_zarr`
    has RETURNED, so a run killed mid-write (dropped connection, Ctrl-C, OOM) leaves the
    previous cube intact rather than parking a truncated one where `run()`'s existence
    check would take it for finished. The swap also moves any existing cube aside by
    rename rather than rmtree-ing it in place, which is what makes overwriting safe on
    NFS when a reader still holds a chunk open.
    """
    with store.atomic(Path(zpath)) as tmp:
        write_zarr(ds, tmp, encoding)


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _combos_from(opts) -> list[tuple[str, str]]:
    raw = (getattr(opts, "model_extra", None) or {}).get("combinations", []) or [] if opts else []
    return [(str(a), str(b)) for a, b in raw]


def _met_overpass_combos(project: Project) -> list[tuple[str, str]]:
    """The (sensor, source) overpass-met combinations the cube emits, from the met_overpass
    product's config (D14/D15). Empty when met_overpass is not selected."""
    return _combos_from(project.products.get(DataProduct.met_overpass))


def _tide_overpass_combos(project: Project) -> list[tuple[str, str]]:
    """The (sensor, tide_source) overpass-tide combinations (D17), from the tides product's
    `overpass_combinations` option -- tide_overpass is a derived contributor, not a product,
    so its combos live with the tide data. Empty when tides is not selected."""
    opts = project.products.get(DataProduct.tides)
    if opts is None:
        return []
    raw = (getattr(opts, "model_extra", None) or {}).get("overpass_combinations", []) or []
    return [(str(a), str(b)) for a, b in raw]


def _sensor_version_pref(project: Project) -> dict[str, list[str]]:
    """Config-declared version order for each STACKED-DATA sensor, so the single overpass
    identity prefers the first-listed version per day (D5). Only products whose config sets
    the stacked-source key are listed; the rest fall back to sorted on-disk order in the
    contributor. Keyed by the product's own name (matches PRODUCT_DIRS)."""
    out: dict[str, list[str]] = {}
    for s in products.sensors():
        if not s.is_stacked_data:
            continue
        opts = project.products.get(s.product)
        raw = (getattr(opts, "model_extra", None) or {}).get(s.sources_option) if opts else None
        if raw is None:
            continue
        out[s.product.value] = [raw] if isinstance(raw, str) else [str(x) for x in raw]
    return out


def _sparse_daily(project: Project, aoi: str) -> set[str]:
    """Daily products this AoI's config deliberately fetched on only SOME days.

    Today that is MUR with `overpass_sensors` set: its days are restricted to the ones a
    thermal sensor flew, so the resulting cube gaps are the config working, not a lost run.
    Resolved per AoI because the option is region-overridable.
    """
    out: set[str] = set()
    if DataProduct.mur in project.products and _opt(
            resolve_opts(project, aoi, DataProduct.mur), "overpass_sensors", None):
        out.add("mur")
    return out


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    dc = project.datacube
    root = Path(project.output_dir)
    return {
        "aligned_root": root,                         # per-product <DIR>/aligned/<aoi>
        # {aoi: products whose thin coverage is intended} -- see _sparse_daily.
        "sparse_daily": {a.name: _sparse_daily(project, a.name) for a in project.all_areas},
        # Version preference order for stacked-DATA sensors (ECOSTRESS), for the D5 merge.
        "sensor_version_pref": _sensor_version_pref(project),
        "out_dir": root / dc.output_subdir,
        "chunks": dict(dc.chunks),
        "met_time": str(dc.met_time),
        # The (sensor, source) overpass-met combos come from the met_overpass PRODUCT now
        # (D14/D15), not datacube.overpass_met. The cube emits <sensor>_<var>_<src> for these.
        "met_overpass_combos": _met_overpass_combos(project),
        # ...and the (sensor, tide_source) overpass-tide combos from the tides product (D17).
        "tide_overpass_combos": _tide_overpass_combos(project),
        "insitu": bool(dc.insitu),
        "insitu_max_dt_min": float(dc.insitu_max_dt_min),
        # The in-situ reference-time channel is sampled at the same instant as met.
        "ref_hours": met_mod.parse_hhmm(
            getattr(project.products.get(DataProduct.met), "reference_time",
                    met_mod.DEFAULT_REFERENCE_TIME)
            if DataProduct.met in project.products else met_mod.DEFAULT_REFERENCE_TIME),
        "ref_basis": (getattr(project.products.get(DataProduct.met), "reference_basis",
                              met_mod.DEFAULT_REFERENCE_BASIS)
                      if DataProduct.met in project.products
                      else met_mod.DEFAULT_REFERENCE_BASIS),
        # Needed to resolve the datum offset per AoI (region override + sidecar lookup).
        "project": project,
        "compression": dc.compression,
        "overwrite": bool(dc.overwrite),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Assemble one Zarr cube per AoI from the pre-computed shared grids."""
    out_dir = eff["out_dir"]
    overwrite = eff["overwrite"]
    days = pd.date_range(eff["time"]["start_date"], eff["time"]["end_date"], freq="D")

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("datacube")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        g = grids[name]
        zpath = out_dir / f"{name}.zarr"
        if zpath.exists() and not overwrite:
            log.info("=== %s: %s exists, skipping (use overwrite) ===", name, zpath.name)
            rep.skip()
            continue
        store.sweep_scratch(zpath)      # clear scratch from a run that died mid-write
        if dry_run:
            log.info("=== %s: [dry-run] would assemble %d day(s) -> %s ===",
                     name, len(days), zpath.name)
            continue

        log.info("=== assembling %s (%d days, grid=%dx%d) ===", name, len(days), g.width, g.height)
        ds = assemble_aoi(g, eff, days)
        write_zarr_safe(ds, zpath, build_encoding(ds, eff["compression"], eff["chunks"]))

        # `t=%d` was always len(days) -- it said nothing about how much of the cube is real.
        # Report the coverage the cube actually has, so a thin product is visible here.
        cov = json.loads(ds.attrs.get("coverage", "{}"))
        cov_str = ", ".join(f"{p} {100 * c['fraction']:.0f}%" for p, c in sorted(cov.items()))
        log.info("  wrote %s  vars=%d shape=(t=%d,y=%d,x=%d)  coverage: %s", zpath.name,
                 len(ds.data_vars), ds.sizes["time"], ds.sizes["y"], ds.sizes["x"],
                 cov_str or "n/a")
        rep.wrote()
        thin = [p for p, c in cov.items() if c["fraction"] < COVERAGE_WARN]
        if thin:
            rep.note = f"thin coverage: {', '.join(sorted(thin))} (see the cube's `coverage` attr)"

    rep.log_summary()
    return rep


def assemble(project: Project, *, grids=None, aois=None, dry_run=False,
             overwrite=False) -> None:
    """Assemble datacubes for a validated Project. Terminal pipeline stage.

    Same signature as every product's acquire(); reads only the aligned files the
    acquisition stages wrote, so it must run AFTER them.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    # The terminal stage's entry point is `assemble`, not `acquire`, but it takes the same
    # (project, aois, dry_run, overwrite) arguments -- so it rides the shared parser too.
    entry.process_main(assemble, "coastal_sst_data datacube assembler.")


if __name__ == "__main__":
    main()
