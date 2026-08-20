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
  * BLOCKED ASSEMBLY. A cube costs `channels x days x height x width x 4` bytes to build, which
    grows with the AoI AND with the window -- a multi-year run on a large grid wants hundreds
    of GB and is simply killed. So the time axis is assembled and written a BLOCK of days at a
    time (`datacube.block_days`, sized per AoI from a memory budget), which bounds peak memory
    by the block rather than the window. An AoI that fits is still built in one pass, by the
    same code as before. Two things make this safe, and both are enforced rather than trusted:
    a contributor must answer whole-cube questions from `ctx.all_days` (not its block), and
    every block must emit the SAME channel set -- see AssemblyContext and `_check_channel_set`.
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

import contextlib
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
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


def day_index(d: Path, aoi_id, prefix="", *, cache=None) -> dict[str, Path]:
    """{day_stamp: path} for a one-file-per-day product's aligned directory.

    Names only -- nothing is opened. `prefix` selects a variant written into the same
    directory (see `load_daily_sensor`), and is part of the cache key: met's daily mean and
    its reference snapshot live side by side and must not be conflated.
    """
    def build() -> dict[str, Path]:
        out: dict[str, Path] = {}
        if not d.exists():
            return out
        pat = naming.day_pattern(aoi_id, prefix)
        for f in d.glob(f"{aoi_id}_{prefix}*.nc"):
            m = pat.match(f.name)
            if m:
                out[m.group(1)] = f
        return out

    return _cached(cache, ("dayfiles", str(d), prefix), build)


def load_daily_sensor(d: Path, aoi_id, days, H, W, var, *, prefix="", cache=None):
    """MUR/met style: one file per day (<aoi>_<prefix><YYYYMMDD>.nc) -> (T,H,W).

    `prefix` selects a variant written into the same directory -- met writes both a
    daily mean (`<aoi>_20230715.nc`) and a reference-time snapshot
    (`<aoi>_ref_20230715.nc`). The name is matched WHOLE rather than by suffix, so the
    two cannot be confused for each other.
    """
    out = _empty3d(days, H, W)
    files = day_index(d, aoi_id, prefix, cache=cache)
    if not files:
        return out
    for i, dd in enumerate(days):
        f = files.get(naming.day_stamp(dd))
        if f is None:
            continue
        ds = xr.open_dataset(f)
        if var in ds:
            arr = ds[var].isel(time=0).values if "time" in ds[var].dims else ds[var].values
            if arr.shape == (H, W):
                out[i] = arr
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


def met_prefix(d: Path, aoi_id, days, want: str, *, cache=None) -> tuple[str, str]:
    """Which met variant to feed the cube's met channels: ('ref_'|'', label).

    Honours `datacube.met_time`, but falls back to the other variant if the one asked
    for was never written (e.g. an older MET tree with no reference snapshots) rather
    than silently emitting an all-NaN forcing channel.

    WHOLE-WINDOW DECISION. `days` must be the cube's full time axis (`ctx.all_days`), never
    one block of it. The answer is a property of the TREE, not of a slice: asked about a
    block that happens to hold only daily means, this would answer "daily_mean" for that
    block and "reference" for another, silently splicing two different times of day into one
    forcing channel -- with a `met_time` attr describing whichever block wrote last.
    """
    def has(prefix):
        return any((d / f"{naming.day_stem(aoi_id, dd, prefix)}.nc").exists()
                   for dd in days)

    def build():
        want_prefix = "ref_" if want == "reference" else ""
        if has(want_prefix):
            return want_prefix, ("reference" if want_prefix else "daily_mean")
        other = "" if want_prefix else "ref_"
        if has(other):
            label = "reference" if other else "daily_mean"
            log.warning("  no %s met files; falling back to the %s", want, label)
            return other, label
        return want_prefix, ("reference" if want_prefix else "daily_mean")

    return _cached(cache, ("met_prefix", str(d), want), build)


def footprint_available(d: Path, aoi_id, days, H, W, *, cache=None) -> bool:
    """Did ANY granule in the WINDOW carry a usable `footprint_id`?

    Exactly the rule `load_clearest_overpass` applies while it reads, hoisted to a decision
    about the whole window -- because under blocked assembly it can no longer be made from
    the granules one block happens to hold. `<sensor>_footprint_id` is the only channel whose
    existence depends on file CONTENTS rather than on which files exist, so it is the only
    one that can differ between blocks; and a cube whose variables disagree on their time
    length does not fail to write, it fails to READ:

        ValueError: conflicting sizes for dimension 'time'

    Early-exits on the first granule that carries the layer, so the common case is one open.
    """
    def build() -> bool:
        want = {naming.day_stamp(dd) for dd in days}
        for day, granules in scene_index(d, aoi_id, cache=cache).items():
            if day not in want:
                continue
            for _dt, f in granules:
                with xr.open_dataset(f) as ds:
                    if "time" in ds.dims:
                        ds = ds.isel(time=0)
                    # Same guards, same order as the loader: a granule whose sst is off-grid
                    # is skipped there before its footprint is ever looked at.
                    if "sst" not in ds or ds["sst"].shape != (H, W):
                        continue
                    if "footprint_id" in ds and ds["footprint_id"].shape == (H, W):
                        return True
        return False

    return _cached(cache, ("footprint", str(d)), build)


def _read_granule(f: Path, H, W, *, water_is_land, use_cloud, qset, trust_valid, read_fp):
    """One granule -> (sst, cloud, valid, footprint|None), or None if it is not on this grid.

    The validity rules `load_clearest_overpass` documents, applied to a single file. Split out
    because that function now has two ways to combine a day's granules (keep the clearest, or
    mosaic them) and only one way to READ one -- the guards and the mask algebra are identical
    either way, and duplicating them is how the two regimes would drift apart.
    """
    with xr.open_dataset(f) as raw:
        ds = raw.isel(time=0) if "time" in raw.dims else raw
        if "sst" not in ds or ds["sst"].shape != (H, W):
            return None
        s = ds["sst"].values.astype("float32")
        c = (ds["cloud"].values.astype("float32")
             if "cloud" in ds and ds["cloud"].shape == (H, W) else np.zeros((H, W), "float32"))
        fp = None
        if read_fp and "footprint_id" in ds and ds["footprint_id"].shape == (H, W):
            fp = ds["footprint_id"].values.astype("int32")
        if trust_valid and "valid" in ds and ds["valid"].shape == (H, W):
            v = (ds["valid"].values > 0) & np.isfinite(s)
        else:
            if "water" in ds and ds["water"].shape == (H, W):
                w = ds["water"].values.astype("float32")
                wp = np.isfinite(w) & products.water_cells(w, water_is_land=water_is_land)
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
    return s, c, v, fp


def multi_granule_days(d: Path, aoi_id, days, *, cache=None) -> tuple[int, int]:
    """(scene-days holding MORE than one granule, scene-days at all) over the window.

    FILENAMES only -- `scene_index` is already built and cached, so this opens nothing.

    WHOLE-WINDOW, like `footprint_available`, and for a related reason: the caller logs these
    counts, and a count taken from one BLOCK's days differs between blocks. `_LogOnce` keys on
    the record's args, so a per-block count produces a different key each time and the summary
    is emitted once per block -- the very thing that filter exists to stop. Pass `ctx.all_days`.
    """
    def build() -> tuple[int, int]:
        want = {naming.day_stamp(dd) for dd in days}
        seen = [len(g) for day, g in scene_index(d, aoi_id, cache=cache).items() if day in want]
        return sum(1 for n in seen if n > 1), len(seen)

    return _cached(cache, ("multiday", str(d)), build)


def load_clearest_overpass(d: Path, aoi_id, days, H, W, *, water_is_land=False,
                           use_cloud=True, qc_levels=None, trust_valid=False,
                           read_footprint=False, footprint_present=None, mosaic=False,
                           cache=None):
    """Per-overpass sensors (ECOSTRESS/Landsat/MODIS): one scene-set per day, clearest first.

    Two regimes for a day holding MORE than one granule, chosen per sensor by
    `SensorSpec.mosaic_same_day`:
      * mosaic=False -- keep the clearest granule (most valid pixels) and DISCARD the rest.
      * mosaic=True  -- the clearest is the BASE, and lower-ranked granules fill ONLY the
        cells it left invalid. Two non-overlapping Landsat path/rows over one AoI used to
        leave half of it NaN under the first rule; the day's coverage is now their union.
    Either way a day with ONE granule takes the same path it always did.

    Validity per sensor:
      * trust_valid=True (MODIS -- already quality-filtered): use the file's
        `valid` layer directly; the sensor has no water/cloud layer.
      * else recompute finite(sst) & water [& clear] [& QC]:
          - water: the sensor water layer with per-sensor polarity (water_is_land).
          - use_cloud gates on the binary cloud layer (Landsat: reliable).
          - qc_levels (e.g. {0,1}) gates on QC mandatory-QA bits 0-1 instead of
            cloud (ECOSTRESS: cloud over-masks cold water, so gate on QC).
    Returns (sst, cloud, valid, hour, times, footprint).
    `times` is the BASE scene's datetime per day (None where the sensor had no scene), so
    the tide and the met snapshot can be matched to that exact scene rather than to the day.
    MOSAICKING DOES NOT CHANGE IT: a merged day still reports the base granule's instant, and
    `hour` with it, so every downstream matchup keys off one overpass identity exactly as
    before. The cost is that a merged slice can hold pixels acquired at a time the cube does
    not report, and its overpass forcing describes the base granule alone -- said out loud on
    the channels themselves via `_MOSAIC_ATTRS`.

    Two consequences of filling only what the base left INVALID, both deliberate:
      * a lower-ranked granule wins a cell where the base is finite but cloudy/QC-rejected,
        so `<prefix>_sst` is no longer one granule's raw scene there; and
      * a cell NO granule validly observed keeps the base's raw value -- possibly NaN -- even
        where a lower-ranked granule held a masked reading. Filling it would put unvalidated
        pixels into the sst channel, which is a stronger claim than the files make.

    Ties go to the earliest granule, in both regimes: the contest is a strict `>` over
    `scene_index`'s (time, name) ordering. That is load-bearing under mosaicking, because the
    winner's datetime IS the day's reported overpass.

    ONE DAY AT A TIME. The scenes are walked per day (via `scene_index`) and the day's result
    is written into the output before the next day is read, so the only scene arrays alive at
    once are the contender and the incumbent. Accumulating every day's winner first and
    copying afterwards -- which is what this did -- held a WHOLE WINDOW of scenes alongside
    the outputs that were already allocated, roughly doubling the peak for no benefit.
    Mosaicking keeps that bound: it needs no ranking pass and no second read, because `score`
    (below) records what each cell already came from, which is enough to apply a ranking that
    has not finished arriving. It costs ONE (H, W) int32, not one per granule.

    BLOCK-INVARIANT. The merge reads only `scene_index[day]` -- built from FILENAMES, cached
    per directory, identical in every block -- and that day's granule contents. A day lives in
    exactly one block, so it merges once per run, the same way, whatever the block boundaries.
    Nothing here consults the other days in `days`.

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
    if mosaic and read_footprint:
        raise ValueError(
            "mosaic and read_footprint are mutually exclusive: footprint ids restart at 0 in "
            "every granule, so a merged day would mix two granules' unrelated native-pixel "
            "indices in one channel (see SensorSpec.mosaic_same_day).")
    sst, cloud = _empty3d(days, H, W), _empty3d(days, H, W)
    valid = np.zeros((len(days), H, W), dtype="uint8")
    hour = np.full(len(days), np.nan, dtype="float32")
    times: list = [None] * len(days)
    footprint = None
    # Per-cell provenance for the mosaic: the valid-pixel count of the granule each cell's
    # values came from, or -1 for a cell no granule has validly claimed yet. This is what lets
    # a ranked fill run in ONE arrival-ordered pass -- see the merge below. Allocated lazily,
    # on the first day that actually has granules to merge, and reused across days; a sensor
    # with no multi-granule day never pays for it. `np.empty(...).fill()` rather than
    # `np.full`, because a test spies on `np.full` to prove no stray int32 is allocated.
    score = None
    # `footprint_present` is `footprint_available`'s whole-window answer, so that every time
    # block emits the same channel set (see that function). None keeps the self-contained
    # behaviour -- decide from the granules actually read -- for a caller holding one window.
    saw_footprint = bool(footprint_present)
    read_fp = read_footprint and footprint_present is not False
    qset = list(qc_levels) if qc_levels is not None else None
    didx = {naming.day_stamp(dd): i for i, dd in enumerate(days)}
    missing_fp = 0
    scene_days = 0
    # Granules that HAVE observations but whose validity mask came out entirely empty. That is
    # never a real scene: it means the water/cloud/QC gate is rejecting everything, which is
    # what an inverted polarity or a mis-assigned mask layer looks like from here. It is also
    # invisible in the output -- an all-zero `valid` makes every granule tie at 0, so the first
    # one becomes the base, paints its raw (mostly NaN) scene over the whole grid, and no other
    # granule can ever outrank it. The day's mosaic silently collapses onto ONE granule, and a
    # tiled sensor ends up with a single tile's footprint in a corner of the AoI.
    blank_valid = 0
    granules_read = 0
    for day, granules in scene_index(d, aoi_id, cache=cache).items():
        i = didx.get(day)
        if i is None:
            continue
        # A day with one granule cannot be mosaicked, so it keeps the contest path exactly --
        # which is what makes this change a no-op for every tree that has no merged day.
        mosaic_day = mosaic and len(granules) > 1
        if mosaic_day:
            if score is None:
                score = np.empty((H, W), dtype="int32")
            score.fill(-1)
        best = None  # (valid_count, sst, cloud, valid, footprint|None, datetime)
        for dt, f in granules:
            got = _read_granule(f, H, W, water_is_land=water_is_land, use_cloud=use_cloud,
                                qset=qset, trust_valid=trust_valid, read_fp=read_fp)
            if got is None:
                continue
            s, c, v, fp = got
            if fp is not None:
                saw_footprint = True

            vc = int(v.sum())
            granules_read += 1
            if vc == 0 and np.isfinite(s).any():
                blank_valid += 1
            new_base = best is None or vc > best[0]
            if mosaic_day:
                # THE MERGE, in one pass. A new base claims every cell it validly observes plus
                # every cell still unclaimed (score < 0) -- NOT every cell, or it would wipe the
                # valid fill an earlier, lower-ranked granule contributed where this one is
                # blind. A non-base granule may only claim cells whose current source ranked
                # BELOW it. Together those give exactly the ranked fill: each cell ends with
                # the highest-count granule that validly saw it, and a cell none of them saw
                # ends with the last new base -- which, since `new_base` fires only on running
                # maxima, is the clearest granule of the day.
                take = (v | (score < 0)) if new_base else (v & (score < vc))
                # sst/cloud/valid are written under ONE mask from ONE granule, so a filled
                # cell's temperature, cloud flag and validity all describe the same overpass.
                np.copyto(sst[i], s, where=take)
                np.copyto(cloud[i], np.nan_to_num(c, nan=0.0), where=take)
                np.copyto(valid[i], v, where=take)
                np.copyto(score, np.int32(vc), where=take & v)   # unclaimed cells stay at -1
            if new_base:
                best = (vc, s, c, v, fp, dt)
        if best is None:
            continue                                       # every granule failed the shape check
        _, s, c, v, fp, dt = best
        if not mosaic_day:
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
    if blank_valid:
        log.warning(
            "  %s: %d of %d granule(s) have finite SST but NO valid pixels -- the "
            "water/cloud/QC gate is rejecting everything they observed. %s Check that the "
            "aligned files carry the mask layers this sensor's SensorSpec expects "
            "(water_is_land/use_cloud/qc_levels).", d.name, blank_valid, granules_read,
            "Each day's mosaic therefore keeps only its FIRST granule, so a tiled sensor "
            "shows one tile's footprint." if mosaic else
            "Those scenes contribute nothing to the cube.")
    if saw_footprint and missing_fp:
        # A partly backfilled directory: some granules predate the layer, or were acquired
        # with `footprint_id: false`. Those days stay -1, which is indistinguishable from
        # "off-swath" unless we say so here. Re-acquire with --overwrite to fill them.
        log.warning("  %s: %d of %d scene-days have no footprint_id layer; those days are "
                    "all -1 in the footprint channel", d.name, missing_fp, scene_days)
    return sst, cloud, valid, hour, times, footprint


def tide_daily_lut(d: Path, aoi_id, *, cache=None) -> tuple[dict, dict]:
    """({day: daily mean}, {day: daily range}) from a source's whole tide series.

    A tide source writes ONE file spanning the entire window, so this resamples the whole
    multi-year series however few days are being asked about -- which is why the result is
    cached per directory rather than recomputed per block.
    """
    def build() -> tuple[dict, dict]:
        f = d / f"{aoi_id}_tides.nc"
        if not f.exists():
            return {}, {}
        with xr.open_dataset(f) as ds:
            t = ds["tide"]
            dm = t.resample(time="1D").mean()
            dr = t.resample(time="1D").max() - t.resample(time="1D").min()
            return (dict(zip(dm["time"].dt.strftime(naming.DAY_FMT).values, dm.values)),
                    dict(zip(dr["time"].dt.strftime(naming.DAY_FMT).values, dr.values)))

    return _cached(cache, ("tide_daily", str(d)), build)


def load_tide_daily(d: Path, aoi_id, days, *, cache=None):
    """Tide 1D series -> (daily_mean, daily_range) on the daily axis."""
    mean = np.full(len(days), np.nan, "float32")
    rng = np.full(len(days), np.nan, "float32")
    lut_m, lut_r = tide_daily_lut(d, aoi_id, cache=cache)
    for i, dd in enumerate(days):
        k = naming.day_stamp(dd)
        if k in lut_m:
            mean[i] = lut_m[k]
            rng[i] = lut_r[k]
    return mean, rng


def tide_series(d: Path, aoi_id, *, cache=None):
    """The raw sub-daily tide series for a source, cached per directory.

    Wraps `water_level.load_tide_series` here rather than caching inside that module, which
    stays a pure function of its arguments.
    """
    return _cached(cache, ("tide_series", str(d)),
                   lambda: water_level.load_tide_series(d, aoi_id))


def load_bathy(d: Path, aoi_id, H, W, *, cache=None):
    """Static bathymetry: (elevation, depth, depth_p25, depth_p75), NaN where absent.

    An ABSENT DEM must produce NaN, never zeros. The obvious derivation --
    `np.where(elev < 0, -elev, 0.0)` -- looks right and is catastrophically wrong when the
    file is missing: `elev` is then all-NaN, `np.nan < 0` is False, so every cell takes the
    0.0 branch and the cube ships a flawless, NaN-free "everything is exactly at sea level"
    bathymetry. That is fabricated data wearing the costume of real data, and downstream
    reconstructs water level from `elevation` + `depth`. Depth is therefore derived only
    where the elevation is actually KNOWN.

    Static, so it is cached per directory: it does not vary with the days being assembled.
    """
    def build():
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
        if depth is None:  # derive mean depth from elevation -- ONLY where it is known
            depth = np.where(known, np.where(elev < 0, -elev, 0.0), np.nan).astype("float32")
        if dp25 is None:
            dp25 = depth.copy()
        if dp75 is None:
            dp75 = depth.copy()
        return elev, depth, dp25, dp75

    return _cached(cache, ("bathy", str(d)), build)


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


def insitu_sources(base: Path, aoi_id, *, cache=None) -> list[tuple[str, xr.Dataset]]:
    """`load_insitu`, cached per tree.

    An in-situ source writes ONE file spanning the whole window, so re-opening it per time
    block would re-parse a multi-year station table for every block. When cached, the
    Datasets are owned by the CACHE -- `close_cache` closes them once the AoI is done, and
    the caller must not close them itself.
    """
    return _cached(cache, ("insitu", str(base)), lambda: load_insitu(base, aoi_id))


def close_cache(cache) -> None:
    """Release anything an assembly cache holds open, and empty it.

    Only the in-situ tables are open handles; everything else the cache keeps is plain data.
    Called once per AoI, after the last time block -- not per block, which is the whole point
    of caching them.
    """
    if not cache:
        return
    for key, val in cache.items():
        if isinstance(key, tuple) and key and key[0] == "insitu":
            for _src, ds in val:
                ds.close()
    cache.clear()


def build_insitu(sources: list[tuple[str, xr.Dataset]], g: AoiGrid, days, targets: dict,
                 max_dt_min, *, cache=None):
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
        # water=None: no snapping. Cached per source -- the placement is a property of the
        # stations and the grid, and building it spins up a pyproj Transformer each time.
        placed = _cached(cache, ("insitu_pixels", src),
                         lambda: insitu.station_pixels(lons, lats, g))

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


def cmems_channels(d: Path, aoi_id, *, cache=None) -> list[str]:
    """The CMEMS variables actually on disk (thetao_0m, thetao_10m, zos, ...).

    Discovered from the files rather than re-derived from config, so the cube carries
    exactly the variables x depths that were acquired -- no second list to keep in sync.
    """
    def build() -> list[str]:
        if not d.exists():
            return []
        for f in sorted(d.glob(f"{aoi_id}_*.nc")):
            with xr.open_dataset(f) as ds:
                return [v for v in ds.data_vars if v != "valid"]
        return []

    return _cached(cache, ("cmems_vars", str(d)), build)


def load_bathy_attrs(d: Path, aoi_id, *, cache=None) -> dict:
    """The bathymetry file's global attrs (its `source` is the DEM fingerprint)."""
    def build() -> dict:
        f = d / f"{aoi_id}.nc"
        if not f.exists():
            return {}
        with xr.open_dataset(f) as ds:
            return dict(ds.attrs)

    return _cached(cache, ("bathy_attrs", str(d)), build)


def load_landcover(d: Path, aoi_id, H, W, *, cache=None):
    """Static land-cover water mask -> float (1=water, 0=land, NaN=unknown/absent)."""
    def build():
        water = np.full((H, W), np.nan, "float32")
        f = d / f"{aoi_id}.nc"
        if f.exists():
            ds = xr.open_dataset(f)
            if "water" in ds and ds["water"].shape == (H, W):
                water = ds["water"].values.astype("float32")
            ds.close()
        return water

    return _cached(cache, ("landcover", str(d)), build)


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

    `days` may be ONE TIME BLOCK of the cube rather than the whole axis (see `run`), so a
    contributor gets two views of time and must pick deliberately:

      * `days`     -- the block being built. Emit arrays on THIS axis.
      * `all_days` -- the cube's full axis. Use it for any question whose answer is a
                      property of the whole cube rather than of this block; `met_prefix`
                      (which met variant feeds the cube) is the worked example.

    A contributor's CHANNEL SET must depend only on the files on disk, never on `days` --
    blocks that emit different channels build a cube whose variables disagree on their time
    length, which writes cleanly and then cannot be opened (`_check_channel_set` catches it).

    `cache` is shared across every block of one AoI; pass it to the loaders that take it so
    a directory is scanned once per AoI rather than once per block.
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
    all_days: Any = None
    cache: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.all_days is None:
            self.all_days = self.days

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
        elev, depth, depth_p25, depth_p75 = load_bathy(adir, ctx.aid, ctx.H, ctx.W,
                                                        cache=ctx.cache)
        attrs = load_bathy_attrs(adir, ctx.aid, cache=ctx.cache)
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
    # `ctx.days` is empty only in the CENSUS pass, which runs the real contributors over a
    # zero-length axis purely to learn the channel set. `_quiet` covers it, but only while the
    # module logger defers to the package one -- set a level on the module logger (as a test
    # reasonably might) and the census starts narrating. Every other loader here is silent then
    # by accident, because no day matches and the message sits behind a per-day counter; this
    # one reads the whole window, so it has to say so itself.
    if sp.mosaic_same_day and len(ctx.days):
        # Asked over the WHOLE cube axis, like footprint_available and for a related reason:
        # a count taken from THIS block's days differs between blocks, which makes each block's
        # message a different log record and defeats `_LogOnce` -- one identical summary per
        # block is exactly what that filter exists to prevent.
        n_multi, n_days = multi_granule_days(adir, ctx.aid, ctx.all_days, cache=ctx.cache)
        if n_multi:
            # The tree path, not adir.name: that is the AoI id for BOTH ECOSTRESS/v002/... and
            # /v003/..., so identical args would make _LogOnce swallow the second version.
            log.info("  %s: %d of %d scene-days have more than one granule; those days are "
                     "MOSAICKED -- the clearest granule is the base and the rest fill only the "
                     "cells it left invalid, while %s_hour stays the base granule's time",
                     adir.relative_to(ctx.eff["aligned_root"]), n_multi, n_days, sp.prefix)
    return load_clearest_overpass(
        adir, ctx.aid, ctx.days, ctx.H, ctx.W,
        water_is_land=sp.water_is_land, use_cloud=sp.use_cloud,
        qc_levels=list(sp.qc_levels) if sp.qc_levels is not None else None,
        trust_valid=sp.trust_valid, read_footprint=sp.has_footprint,
        # Decided over the WHOLE cube axis, not this block's days -- see footprint_available.
        footprint_present=(footprint_available(adir, ctx.aid, ctx.all_days, ctx.H, ctx.W,
                                               cache=ctx.cache)
                           if sp.has_footprint else False),
        mosaic=sp.mosaic_same_day, cache=ctx.cache)


# The footprint index is a per-day identity, and saying so on the variable is the only place a
# user reading the cube will see it: ids restart at 0 in every granule, so grouping across the
# time axis merges unrelated native pixels into one bucket -- silently, and wrongly.
_FOOTPRINT_ATTRS = dict(
    long_name="native sensor pixel index this grid cell was resampled from; -1 = none",
    comment="ids identify a pixel WITHIN one day's chosen scene only -- they restart at 0 "
            "for every scene and are NOT comparable across days")


# Said on the channels themselves, because it is the only place a user reading the cube would
# ever learn it: a mosaicked slice can hold pixels from more than one overpass, while
# `<prefix>_hour` -- and every forcing matched to it -- reports only the base granule's.
_MOSAIC_ATTRS = dict(
    mosaic="days with more than one granule are merged: the granule with the most valid "
           "pixels is the base, and the rest fill ONLY the cells it left invalid",
    mosaic_time="the day's `_hour`, and every overpass-matched forcing (met, tide, in-situ), "
                "describe the BASE granule alone -- filled cells may come from another "
                "overpass of the same day")


def _contribute_flat_sensor(ctx: AssemblyContext, s, sensor_times: dict) -> None:
    """A single-collection sensor (Landsat, MODIS): one flat `<DIR>/aligned/<aoi>` tree, one
    channel-set under the bare prefix, one entry in `sensor_times`."""
    sp = s.sensor
    sst, cloud, valid, hour, times, footprint = _load_sensor(
        ctx, sp, ctx.adir(s.product.value))
    pre = sp.prefix
    mos = _MOSAIC_ATTRS if sp.mosaic_same_day else {}
    ctx.emit(f"{pre}_sst", T3, sst, **mos)
    if sp.has_cloud:          # MODIS arrives pre-filtered with no cloud layer; an all-zero
        ctx.emit(f"{pre}_cloud", T3, cloud, **mos)   # channel would falsely read "never cloudy"
    ctx.emit(f"{pre}_valid", T3, valid, **mos)
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
    # A SOURCE TAG IS A DIRECTORY THAT ACTUALLY HOLDS GRANULES, not merely a subdirectory.
    # Every subdir used to become a tag, so a flat pre-stacking leftover (`MODIS/aligned`,
    # `ECOSTRESS/aligned`) or a scratch dir (`MODIS/_tmp`) was loaded as a version: the loader
    # then resolved its granules at `<DIR>/<tag>/aligned/<aoi>`, which does not exist, and
    # `_load_sensor` returned empty arrays that were emitted anyway. The result was a COMPLETE
    # but entirely-NaN channel set (`eco_sst_aligned`, `eco_valid_aligned`, ...) which every
    # prefix-fanning preprocess step then forked again -- and an all-NaN `_georef_corrected`
    # channel reads as "the correction ran and moved nothing", not as "this tag is a phantom".
    # See docs/bug-empty-version-tag-channels.md, Defect A.
    #
    # The test is STRUCTURAL -- does `<DIR>/<tag>/aligned/` exist -- rather than "does this
    # AoI have granules under it". A real tag whose coverage misses ONE AoI must still emit
    # that AoI's (empty) channel set, or two AoIs in the same project would end up with
    # different channel sets and stop being comparable; a phantom has no `aligned` level at
    # all, which is exactly what distinguishes it.
    all_dirs = {d.name for d in base.iterdir() if d.is_dir()}
    on_disk = {t for t in all_dirs if ctx.adir(s.product.value, t).parent.is_dir()}
    # Silently ignoring a directory the user put on disk is its own failure mode.
    for t in sorted(all_dirs - on_disk):
        log.info("  %s: ignoring %s/%s -- no %s level, so it is not a source tag",
                 sp.prefix, PRODUCT_DIRS[s.product.value], t,
                 ctx.adir(s.product.value, t).parent.name)
    ordered = [t for t in pref if t in on_disk] + sorted(on_disk - set(pref))

    per_tag_times: dict[str, list] = {}
    for tag in ordered:
        sst, cloud, valid, hour, times, footprint = _load_sensor(
            ctx, sp, ctx.adir(s.product.value, tag))
        pre = sp.prefix
        mos = _MOSAIC_ATTRS if sp.mosaic_same_day else {}
        ctx.emit(f"{pre}_sst_{tag}", T3, sst, **mos)
        if sp.has_cloud:
            ctx.emit(f"{pre}_cloud_{tag}", T3, cloud, **mos)
        ctx.emit(f"{pre}_valid_{tag}", T3, valid, **mos)
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
    ctx.emit("mur_sst", T3, load_daily_sensor(ctx.adir("mur"), ctx.aid, ctx.days, ctx.H, ctx.W,
                                              "sst", cache=ctx.cache))


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
            # ctx.all_days: which variant feeds the cube is a fact about the TREE, and a
            # per-block answer would splice two times of day into one channel.
            mprefix, mlabel = met_prefix(d, ctx.aid, ctx.all_days, ctx.eff["met_time"],
                                         cache=ctx.cache)
            label = mlabel
            for var in _MET_FORCING_VARS:
                ctx.emit(f"{var}_{src}", T3,
                         load_daily_sensor(d, ctx.aid, ctx.days, ctx.H, ctx.W, var,
                                           prefix=mprefix, cache=ctx.cache))
    ctx.global_attrs["met_time"] = label


def _overpass_met_vars(d: Path, aoi_id, *, cache=None) -> list[str]:
    """The output variables in a met_overpass source's snapshot files (discovered from disk,
    like CMEMS -- so the channel set is exactly what was acquired)."""
    def build() -> list[str]:
        if not d.exists():
            return []
        for f in sorted(d.glob(f"{aoi_id}_*T*.nc")):
            with xr.open_dataset(f) as ds:
                return list(ds.data_vars)
        return []

    return _cached(cache, ("overpass_vars", str(d)), build)


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
        for var in _overpass_met_vars(d, ctx.aid, cache=ctx.cache):
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
        tide, tide_range = load_tide_daily(d, ctx.aid, ctx.days, cache=ctx.cache)
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
        series = tide_series(ctx.adir("tides", src), ctx.aid, cache=ctx.cache)
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
        for var in cmems_channels(d, ctx.aid, cache=ctx.cache):
            ctx.emit(f"cmems_{var}_{src}", T3,
                     load_daily_sensor(d, ctx.aid, ctx.days, ctx.H, ctx.W, var,
                                       cache=ctx.cache))


def _contribute_landcover(ctx: AssemblyContext) -> None:
    # Land-cover water, shipped RAW as a loss-filter channel (1=water, 0=land; unknown ->
    # water, a no-op filter). Land-masking is a downstream MODELLING determination now (S1).
    lc_raw = load_landcover(ctx.adir("landcover"), ctx.aid, ctx.H, ctx.W, cache=ctx.cache)
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
    sources = insitu_sources(ctx.eff["aligned_root"] / PRODUCT_DIRS["insitu"], ctx.aid,
                             cache=ctx.cache)
    if not sources:
        return
    targets = {"insitu": ctx.slots[SLOT_REF_UTC], **ctx.slots[SLOT_SENSOR_TIMES]}
    insitu_vars, station_table, station_map = build_insitu(
        sources, ctx.g, ctx.days, targets, ctx.eff["insitu_max_dt_min"], cache=ctx.cache)
    # The Datasets belong to the cache now; `close_cache` releases them once the AoI is done.
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


def assemble_block(g: AoiGrid, eff: dict, days, *, all_days=None, cache=None) -> xr.Dataset:
    """The cube's channels for ONE span of days -- no coverage, no provenance.

    UNIFORM CONTRIBUTOR PROTOCOL: every product -- sensor and non-sensor alike -- contributes
    through one `(ctx) -> channels` mechanism (`CONTRIBUTORS`), run in an order derived by
    topological sort from each contributor's declared slot reads/writes. A new non-sensor
    covariate needs a `ProductSpec`, a module, and a registered `Contributor`; forgetting the
    last is a hard error at import (`_check_contributors`), not a silent omission. See
    docs/DEVELOPMENT.md.

    `days` is a BLOCK of `all_days` when the cube is too large to build at once; the two are
    the same object for a whole-window call. `cache` is shared across an AoI's blocks and
    belongs to the CALLER, who must `close_cache` it -- this function does not, because the
    next block still needs what is in it.

    The attrs set here are the ones contributors produce. The whole-cube attrs (coverage,
    provenance) are `finalize_attrs`'s job: they describe the finished cube, so a block cannot
    know them, and appending to a Zarr store overwrites the group attrs anyway.
    """
    ctx = AssemblyContext(
        g=g, eff=eff, days=days, aid=g.name, H=g.height, W=g.width,
        slots={}, channels={}, global_attrs={}, var_attrs={},
        all_days=all_days if all_days is not None else days,
        cache=cache if cache is not None else {})
    for c in _topo_order(CONTRIBUTORS):
        c.fn(ctx)

    xs, ys = g.xy_centers()
    ds = xr.Dataset(ctx.channels, coords={"time": days, "y": ys, "x": xs})
    ds.attrs.update(aoi_id=ctx.aid, crs=g.target_crs)
    ds.attrs.update(ctx.global_attrs)
    for name, attrs in ctx.var_attrs.items():
        ds[name].attrs.update(attrs)
    return ds


def finalize_attrs(eff: dict, aid: str, fields, days, cov: dict, prod: dict) -> dict:
    """The whole-cube attrs: coverage (already tallied) plus provenance.

    Split out of `assemble_aoi` because both are properties of the FINISHED cube -- a blocked
    assembly can only compute them once every block has landed.

    `prod` is `provenance.collect`'s result, passed in rather than gathered here: it feeds
    BOTH the coverage tally (which products wrote files at all) and provenance, and it opens
    every aligned file in the tree, so it is collected exactly once per AoI.
    """
    # (The DEM->MSL datum offset ships PER SOURCE, as attributes on each `elevation_<src>`
    # channel -- resolved and stamped by the bathymetry module when it acquired that DEM, and
    # surfaced by `_contribute_bathymetry`. CUDEM/NAVD88 and GMRT/MSL need different offsets,
    # so a single global attr would be wrong for one of them.)
    for product, c in sorted(cov.items()):
        # A product the config deliberately fetched on only some days is thin ON PURPOSE.
        if c["fraction"] < COVERAGE_WARN and not c.get("sparse"):
            log.warning("  %s: %s covers only %d of %d day(s) (%.0f%%) -- the rest are NaN "
                        "slices, which look exactly like cloudy days. Check the run report "
                        "for what failed.",
                        aid, product, c["days_with_data"], c["days_expected"],
                        100 * c["fraction"])

    # PROVENANCE: the config that built this cube, and for every field the source(s) it came
    # from and when they were accessed. Zarr attrs must be JSON-serialisable, so the structured
    # parts are JSON strings.
    rec = provenance.build(eff["project"], list(fields), prod)
    guessed = [p for p, r in prod.items() if r["basis"] == provenance.FILE_MTIME]
    if guessed:
        log.warning("  %s: access dates for %s came from FILE MTIMES, not recorded stamps "
                    "(acquired before provenance existed, or the tree was copied)",
                    aid, ", ".join(sorted(guessed)))
    return dict(
        coverage=json.dumps(cov, sort_keys=True),
        created_at=rec["created_at"], package_version=rec["package_version"],
        code_version=rec["code_version"],
        config_sha256=rec["config_sha256"] or "", config_path=rec["config_path"] or "",
        config_yaml=rec["config_yaml"] or "",
        provenance=json.dumps(rec["fields"], sort_keys=True),
        provenance_products=json.dumps(rec["products"], sort_keys=True))


def finish_cube(ds: xr.Dataset, g: AoiGrid, eff: dict, days, cache: dict, hits=None,
                grid_hits=None) -> dict:
    """Release the AoI's cache, then stamp the whole-cube attrs on `ds`. Returns the coverage.

    Closing the cache FIRST is not tidiness: `provenance.collect` re-opens every aligned file
    in the tree, and doing that while the cached in-situ tables are still open segfaults the
    netCDF library outright.

    `hits` / `grid_hits` are pre-tallied coverage counts (days with data, and summed grid
    share), for a caller that assembled the cube in blocks and no longer has it in hand.
    Without them the counts are taken from `ds`.
    """
    close_cache(cache)

    # COVERAGE. The time axis is built from the CONFIG (start..end), not from the data, and
    # every loader defaults a missing day to NaN. So a run that lost 40 of 100 days to a
    # flaky network still yields a 100-step cube whose gaps are indistinguishable from cloudy
    # days. Count what actually landed, stamp it on the cube, and say so when a product is
    # thin. `prod` (which products wrote files at all) feeds both coverage and provenance.
    prod = provenance.collect(eff["aligned_root"], g.name, PRODUCT_DIRS)
    cov = coverage_from_hits(coverage_hits(ds) if hits is None else hits, len(days),
                             present=set(prod),
                             sparse=eff.get("sparse_daily", {}).get(g.name),
                             grid_hits=(coverage_grid_hits(ds) if grid_hits is None
                                        else grid_hits))
    ds.attrs.update(finalize_attrs(eff, g.name, list(ds.data_vars), days, cov, prod))
    return cov


def assemble_aoi(g: AoiGrid, eff: dict, days) -> xr.Dataset:
    """Build the complete analysis-ready Dataset for one AoI, in memory, in one pass.

    The whole cube at once: fine for an AoI that fits in memory, and the path `run` still
    takes when it does. `run` blocks the time axis instead when it does not.
    """
    cache: dict = {}
    try:
        ds = assemble_block(g, eff, days, cache=cache)
        finish_cube(ds, g, eff, days, cache)
    finally:
        close_cache(cache)
    return ds


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
COVERAGE_WARN = 0.95      # below this fraction of days, a daily product is reported thin
# Below this share of the GRID -- on the days it does have data -- a product that is
# otherwise well covered is worth a word. Deliberately low: a polar-orbiting swath or a
# scene footprint clips a large AoI as a matter of course, and a threshold that fires on
# every correct run is how the one real case gets ignored. The failure this exists to catch
# put a ninth of the AoI on every single day and reported 100%.
GRID_COVERAGE_WARN = 0.25

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
    return coverage_from_hits(coverage_hits(ds), len(days), present=present, sparse=sparse,
                              grid_hits=coverage_grid_hits(ds))


def coverage_hits(ds: xr.Dataset) -> dict[str, int]:
    """{product: days in THIS dataset that hold data}. The per-block half of `coverage`.

    Blocks are disjoint spans of the time axis, so a cube's totals are the sum of its blocks'
    -- which is what lets a cube too large to hold at once still report honest coverage.

    Deliberately unfiltered: which products EXIST comes from `provenance.collect`, which
    re-opens every aligned file and so cannot run until the assembly cache has been released.
    Tally everything here and drop the absent products in `coverage_from_hits`.
    """
    out: dict[str, int] = {}
    n_time = ds.sizes.get("time", 0)
    for product, chans in _coverage_channels(ds).items():
        has = np.zeros(n_time, dtype=bool)
        for v in chans:
            finite = np.isfinite(ds[v].values)
            axes = tuple(range(1, finite.ndim))       # everything but time
            has |= (finite.any(axis=axes) if axes else finite)
        out[product] = int(has.sum())
    return out


def _coverage_channels(ds: xr.Dataset) -> dict[str, list[str]]:
    """{product: its coverage channels in `ds`}.

    Coverage-channel PREFIXES. A per-source product (D5) has no single channel any more, so
    judge it present on a day if ANY of its per-source channels is finite (S4.7): met's
    `airtemp` prefix matches `airtemp_hrrr`/`airtemp_era5`, tides' `tide` matches
    `tide_<src>`/`tide_range_<src>`, mur's `mur_sst` is still the bare channel. `cmems` is
    discovered (its channels are config-dependent). A stacked source with NO regional coverage
    is all-NaN by design, so "any source finite" is the honest presence test.
    """
    prefixes = dict(DAILY_CHANNELS)
    prefixes["cmems"] = "cmems"
    out = {}
    for product, c in prefixes.items():
        chans = [v for v in ds.data_vars
                 if (v == c or v.startswith(c + "_")) and "time" in ds[v].dims]
        if chans:
            out[product] = chans
    return out


def coverage_grid_hits(ds: xr.Dataset) -> dict[str, float]:
    """{product: summed share of the GRID it fills, over the days it has data}.

    The companion to `coverage_hits`, and the number that was missing. Coverage as it stood
    is purely TEMPORAL -- a day counts as covered if its slice holds AT LEAST ONE finite
    value -- so a product delivering one corner of the AoI every single day reported 100%.
    That is exactly how a tiled sensor whose mosaic had collapsed onto a single tile passed
    for healthy through a whole release.

    Summed rather than averaged, for the same reason `coverage_hits` counts rather than
    divides: blocks are disjoint spans of the time axis, so a cube's total is the sum of its
    blocks' and the division happens once, at the end, in `coverage_from_hits`.

    Only channels with a grid are counted; a 1-D `(time,)` channel (a tide, an overpass hour)
    has no spatial extent to be a fraction of, and a product with only those is absent here.
    """
    out: dict[str, float] = {}
    for product, chans in _coverage_channels(ds).items():
        grids = [v for v in chans if {"y", "x"} <= set(ds[v].dims)]
        if not grids:
            continue
        covered = None
        for v in grids:
            finite = np.isfinite(ds[v].values)
            covered = finite if covered is None else (covered | finite)
        axes = tuple(range(1, covered.ndim))          # everything but time
        per_day = covered.mean(axis=axes)             # share of the grid, per day
        out[product] = float(per_day[per_day > 0].sum())   # days with data only
    return out


def coverage_from_hits(hits: dict[str, int], n_days: int, present=None, sparse=None,
                       grid_hits=None) -> dict:
    """Per-block tallies -> the cube's coverage report.

    `n_days` is the FULL axis however the tallies were gathered: coverage exists to say how
    much of the configured window actually landed, so a per-block denominator would report
    100% on every block of a cube that is mostly holes.
    """
    out = {}
    for product, n in hits.items():
        # `present` is keyed by the product's own name, and so are the tallies -- one
        # registry, one name. The alias table that reconciled `tide` with `tides` is gone.
        if present is not None and product not in present:
            continue
        out[product] = {"days_with_data": n, "days_expected": n_days,
                        "fraction": (n / n_days) if n_days else 0.0}
        # How much of the GRID a covered day actually holds. Optional, because a product made
        # only of 1-D channels has no grid to be a fraction of -- and because a caller with
        # pre-tallied day counts and no grid tally must still get the report it always got.
        if grid_hits is not None and product in grid_hits and n:
            out[product]["grid_fraction"] = grid_hits[product] / n
        if sparse and product in sparse:
            out[product]["sparse"] = True
    return out


# --------------------------------------------------------------------------- #
# Sizing the time block
# --------------------------------------------------------------------------- #
# Peak memory is more than the channels the cube keeps: a loader holds a scene or two beyond
# its output, Blosc needs somewhere to compress into, and the interpreter is not free. The
# channel arithmetic is the part that can be computed, so it is doubled to stand in for the
# rest -- with detection already taking half of what it finds, that is a ~4x margin.
_TRANSIENT_FACTOR = 2.0
_BUDGET_HEADROOM = 512 * 1024**2
_MIN_BUDGET = 4 * 1024**3
# Halving, so every value divides the one above it: a block that is a whole number of chunks
# stays a whole number of chunks after stepping down.
_TIME_CHUNK_LADDER = (64, 32, 16, 8, 4, 2, 1)


@contextlib.contextmanager
def _quiet(logger):
    """Silence a logger for the duration. Used for the census pass, which runs the real
    contributors and would otherwise duplicate every file-presence warning of the real run."""
    prev = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(prev)


def channel_census(g: AoiGrid, eff: dict, days, *, cache=None) -> dict:
    """{name: (dims, dtype)} for the cube `assemble_aoi(g, eff, days)` would build.

    Runs the REAL contributors over a zero-length day index. Every channel is therefore named
    by the code that emits it -- there is no second list to drift out of sync -- and nothing
    is materialised: each array is (0, H, W). The loaders skip every file before opening it,
    because no day matches.

    Used to predict what a day of cube costs before committing to a block size, and as the
    frozen channel set each block is checked against (`_check_channel_set`).

    It shares the blocks' `cache`, so the scan it does -- directories, tide tables, station
    placement, which met variant, whether a footprint layer exists -- is the scan they would
    have done anyway.
    """
    with _quiet(logging.getLogger(__package__.split(".")[0])):
        ctx = AssemblyContext(
            g=g, eff=eff, days=days[:0], aid=g.name, H=g.height, W=g.width,
            slots={}, channels={}, global_attrs={}, var_attrs={},
            all_days=days, cache=cache if cache is not None else {})
        for c in _topo_order(CONTRIBUTORS):
            c.fn(ctx)
    return {name: (dims, np.asarray(arr).dtype) for name, (dims, arr) in ctx.channels.items()}


def bytes_per_day(census: dict, H: int, W: int) -> int:
    """Resident bytes one day of cube costs, from a channel census.

    Only the channels that GROW with the time axis count. The static (y,x) rasters are paid
    once however the axis is split, and the 1-D per-day channels are noise beside a raster,
    but they are counted for completeness.
    """
    n = 0
    for dims, dtype in census.values():
        if dims == T3:
            n += H * W * dtype.itemsize
        elif dims == ("time",):
            n += dtype.itemsize
    return n


def _detected_budget_bytes() -> tuple[int, str]:
    """(bytes, where it came from) -- what this machine will actually let the assembler have.

    Order matters. Physical RAM is the LAST resort because on a scheduled node it is the
    wrong number: the job is killed at its cgroup or scheduler limit, which can be a small
    fraction of the hardware. Reading that limit first is what makes the default safe in the
    place the unbounded assembler died.
    """
    env = os.environ.get("COASTAL_SST_DATA_MEM_GB")
    if env:
        try:
            return int(float(env) * 1024**3), "$COASTAL_SST_DATA_MEM_GB"
        except ValueError:
            log.warning("  COASTAL_SST_DATA_MEM_GB=%r is not a number; ignoring", env)
    slurm = os.environ.get("SLURM_MEM_PER_NODE")            # megabytes
    if slurm:
        try:
            return int(float(slurm) * 1024**2), "$SLURM_MEM_PER_NODE"
        except ValueError:
            pass
    for p, label in ((Path("/sys/fs/cgroup/memory.max"), "cgroup v2"),
                     (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), "cgroup v1")):
        try:
            raw = p.read_text().strip()
        except OSError:
            continue
        if raw and raw != "max":
            try:
                v = int(raw)
            except ValueError:
                continue
            # cgroup v1 reports a sentinel near 2**63 when no limit is set.
            if 0 < v < 2**62:
                return v, label
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "physical RAM"
    except (ValueError, OSError, AttributeError):
        return _MIN_BUDGET, "fallback"


def budget_bytes(eff: dict) -> tuple[int, str]:
    """The memory the assembler may use, and a phrase describing where the number came from.

    A configured budget is taken as given -- the user said what the job has. A DETECTED one is
    halved: the assembler is not the only thing on the machine, and being wrong here costs the
    whole AoI.
    """
    gb = eff.get("memory_budget_gb")
    if gb:
        return int(float(gb) * 1024**3), "datacube.memory_budget_gb"
    raw, src = _detected_budget_bytes()
    return max(int(raw * 0.5), _MIN_BUDGET), f"half of {src}"


def resolve_block_days(eff: dict, per_day: int, n_days: int,
                       *, transient: float = _TRANSIENT_FACTOR) -> tuple[int, int]:
    """(block_days, time_chunk) for one AoI.

    `transient` scales the channel arithmetic to stand in for what a stage holds BESIDE its
    channels. The default is calibrated on the assembler, whose transients are a scene or two;
    a stage with heavier ones (preprocess promotes its baseline to float64 and copies arrays to
    fold drops into) passes a larger factor.

    Every block boundary lands on a chunk boundary. An append that starts mid-chunk makes Zarr
    read, decompress, merge, recompress and rewrite every chunk it touches -- on a large grid
    that is tens of GB of pointless work per boundary -- so the block is a whole number of
    chunks, and when the budget cannot afford even one chunk's worth of days it is the CHUNK
    that gives way, not the alignment.
    """
    tc = int(eff.get("chunks", {}).get("time", n_days) or n_days)
    tc = max(1, min(tc, n_days))
    forced = eff.get("block_days", "auto")
    if forced != "auto" and forced is not None:
        block = max(1, min(int(forced), n_days))
        return block, min(tc, block)

    budget, _src = budget_bytes(eff)
    max_days = max(1, int((budget - _BUDGET_HEADROOM) // (per_day * transient))
                   if per_day > 0 else n_days)
    if max_days >= n_days:                       # the whole cube fits: one pass, as before
        return n_days, tc
    if max_days >= tc:
        return (max_days // tc) * tc, tc
    tc_eff = next((c for c in _TIME_CHUNK_LADDER if c <= max_days and c <= tc), 1)
    return tc_eff, tc_eff


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


def build_encoding(ds: xr.Dataset, compression: CompressionSpec, chunks: dict,
                   *, sizes=None) -> dict:
    """Per-variable chunk + Blosc(zstd) encoding.

    Lossless: dtypes are untouched. Integer masks get BITSHUFFLE (long runs of a
    constant class compress hard); floats get the configured shuffle. The result
    keys ('compressors' for Zarr v3, 'compressor' for v2) match the installed Zarr.

    `sizes` overrides the dimension lengths the chunk clamp is measured against. A chunk is
    capped at its axis so a short cube is not given chunks longer than itself -- but when the
    dataset in hand is one BLOCK of a larger cube, its own `time` length is the block, not the
    axis, and the store would silently be chunked in blocks. Pass the finished cube's sizes.
    """
    dim_sizes = {**ds.sizes, **(sizes or {})}
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        ch = tuple(min(chunks.get(d, dim_sizes[d]), dim_sizes[d]) for d in dims)
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
def write_zarr(ds: xr.Dataset, zpath: Path, encoding: dict, *, consolidated: bool = True):
    """Write a Zarr cube to `zpath`, which must not exist. NOT atomic -- see `write_zarr_safe`.

    Exposed on its own for the one caller that has to keep its SOURCE store open across the
    write and so cannot let the swap happen inside `write_zarr_safe`: `preprocess.run`, which
    rewrites the cube it is reading (it drives `store.atomic` itself, writing here and closing
    the source before the block ends).

    `consolidated=False` is for the first of several writes: xarray re-consolidates the whole
    group on every call, so a blocked write consolidates once at the end (`finalize_cube`)
    instead of once per block.
    """
    with warnings.catch_warnings():
        # Zarr v3 warns that consolidated metadata isn't in the v3 spec; xarray
        # still writes+reads it and it speeds opening a many-variable cube.
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        ds.to_zarr(zpath, mode="w-", consolidated=consolidated, encoding=encoding)


def append_zarr(ds: xr.Dataset, zpath: Path, *, append_dim: str = "time"):
    """Extend an existing Zarr cube along `append_dim` with one more block.

    NO encoding argument, deliberately: xarray raises for a variable that is already in the
    store ("already exists, but encoding was provided"). Chunking and codecs are settled once,
    by the first write, and every later block inherits them.

    `mode="a-"` -- not `"a"` -- so a variable WITHOUT the append dim is left alone rather than
    rewritten. The static (y,x) channels (bathymetry, land cover, the station map) are the
    same array in every block; with `"a"` each block would rewrite all of them.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        ds.to_zarr(zpath, mode="a-", append_dim=append_dim, consolidated=False)


def finalize_cube(zpath: Path, attrs: dict):
    """Stamp the cube's global attrs and consolidate its metadata -- once, after the last block.

    Appending REPLACES a Zarr group's attrs (zarr's `Attributes.put` clears before it writes),
    so after the final block the store carries only that block's attrs. Coverage, provenance,
    the config that built the cube and the in-situ station table would all be silently gone --
    silently, because the cube still opens and still holds every value. Hence a merge here
    rather than trusting the last write.
    """
    import zarr
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        g = zarr.open_group(str(zpath), mode="r+")
        g.attrs.put({**dict(g.attrs), **attrs})
        zarr.consolidate_metadata(g.store)


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
# Blocked assembly
# --------------------------------------------------------------------------- #
class _LogOnce(logging.Filter):
    """Drop a log record this AoI has already emitted.

    Every block re-runs every contributor over the same tree, so a message about the TREE --
    no bathymetry file, a station outside the grid, a partly backfilled footprint layer -- is
    true once and repeated once per block. Seventy identical warnings do not inform anyone;
    they bury the one message that is about a specific block.

    Attach to the logger that EMITS the records (this module's), not to the package logger: a
    Logger's filters run only for records logged on it directly, and are skipped entirely for
    records propagating up from a child. (Levels are the opposite -- a child with no level of
    its own defers to its parent -- which is why `_quiet` can work the other way round.)
    """
    def __init__(self):
        super().__init__()
        self.seen: set = set()

    def filter(self, record) -> bool:
        key = (record.levelno, record.msg, repr(record.args))
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


def _check_channel_set(got: dict, census: dict, block_i: int, days) -> None:
    """Every block must emit the SAME channels, with the same dims and dtypes.

    A contributor whose channel set depends on WHICH days it was handed corrupts a blocked
    cube in a way that does not surface until someone reads it: a variable that misses one
    append is permanently shorter than the time axis, and

        ValueError: conflicting sizes for dimension 'time'

    is raised by `open_zarr`, long after the run that caused it. Fail here instead, naming the
    channel and the block. See `footprint_available` for the one channel this really applies
    to, and AssemblyContext for the rule contributors are held to.
    """
    problems = [f"{name}: census {census.get(name)}, block {got.get(name)}"
                for name in sorted(set(census) | set(got)) if census.get(name) != got.get(name)]
    if problems:
        raise RuntimeError(
            f"block {block_i} ({days[0].date()}..{days[-1].date()}) does not emit the cube's "
            f"channel set; a contributor is deciding what to emit from the days it was given, "
            f"which cannot work when the cube is assembled in blocks:\n  "
            + "\n  ".join(problems))


def _merge_block_attrs(acc: dict, new: dict, block_i: int) -> None:
    """Fold one block's global attrs into the cube's, refusing to let a value change.

    A key whose value differs between blocks means a contributor answered a whole-cube
    question from one block's days -- `met_time` flipping between "reference" and
    "daily_mean" is the concrete case -- and the cube would then describe itself by whichever
    block wrote last while holding a mixture.
    """
    for k, v in new.items():
        if k in acc and acc[k] != v:
            raise RuntimeError(
                f"block {block_i} reports {k}={v!r} but an earlier block reported "
                f"{acc[k]!r}. This attr describes the whole cube, so a contributor computed "
                f"it from one block's days; it must use ctx.all_days.")
        acc[k] = v


def _assemble_blocked(g: AoiGrid, eff: dict, days, zpath: Path, *, block_days: int,
                      time_chunk: int, census: dict, cache: dict) -> dict:
    """Assemble and write one AoI a block of days at a time. Returns its coverage report.

    The whole run happens inside ONE `store.atomic`, so the cube at `zpath` is either the
    previous one or the finished new one -- never a half-appended store. That matters more
    here than for a single write: this loop can run for hours.
    """
    hits: dict[str, int] = {}
    grid_hits: dict[str, float] = {}
    attrs: dict = {}
    blocks = [days[i:i + block_days] for i in range(0, len(days), block_days)]
    quiet = _LogOnce()
    log.addFilter(quiet)
    try:
        with store.atomic(Path(zpath)) as tmp:
            for i, blk in enumerate(blocks):
                ds = assemble_block(g, eff, blk, all_days=days, cache=cache)
                _check_channel_set({k: (ds[k].dims, ds[k].dtype) for k in ds.data_vars},
                                   census, i, blk)
                _merge_block_attrs(attrs, dict(ds.attrs), i)
                for product, n in coverage_hits(ds).items():
                    hits[product] = hits.get(product, 0) + n
                for product, frac in coverage_grid_hits(ds).items():
                    grid_hits[product] = grid_hits.get(product, 0.0) + frac
                if i == 0:
                    # The encoding is settled here, against the FINISHED cube's shape -- not
                    # this block's, which would chunk the time axis in blocks.
                    write_zarr(ds, tmp, build_encoding(
                        ds, eff["compression"], {**eff["chunks"], "time": time_chunk},
                        sizes={"time": len(days)}), consolidated=False)
                else:
                    append_zarr(ds, tmp)
                log.info("    block %d/%d: %s..%s", i + 1, len(blocks),
                         blk[0].date(), blk[-1].date())
                if i < len(blocks) - 1:
                    del ds                                    # before the next block is built
            # `ds` is the LAST block, kept only so `finish_cube` can name the cube's fields
            # and hang the finished attrs somewhere; the values are already on disk.
            cov = finish_cube(ds, g, eff, days, cache, hits=hits, grid_hits=grid_hits)
            attrs.update(ds.attrs)      # + the whole-cube attrs finish_cube just stamped on
            finalize_cube(tmp, attrs)
    finally:
        log.removeFilter(quiet)
    return cov


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
        # Time-blocking: "auto" is resolved PER AOI in `run` (it needs that AoI's grid).
        "block_days": dc.block_days,
        "memory_budget_gb": dc.memory_budget_gb,
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

        # How much cube is one day? Ask the contributors, over a zero-length axis, before
        # committing to anything -- and keep the scan they do for the blocks that follow.
        cache: dict = {}
        try:
            census = channel_census(g, eff, days, cache=cache)
            per_day = bytes_per_day(census, g.height, g.width)
            block_days, time_chunk = resolve_block_days(eff, per_day, len(days))
            budget, src = budget_bytes(eff)
            n3 = sum(1 for dims, _ in census.values() if dims == T3)
            log.info("  %d channel(s), %d of them (t,y,x): %.0f MB/day; budget %.1f GiB (%s) "
                     "-> %d block(s) of %d day(s), time chunk %d, peak ~%.1f GiB",
                     len(census), n3, per_day / 1e6, budget / 1024**3, src,
                     -(-len(days) // block_days), block_days, time_chunk,
                     block_days * per_day * _TRANSIENT_FACTOR / 1024**3)
            if time_chunk < int(eff.get("chunks", {}).get("time", time_chunk) or time_chunk):
                log.warning("  %s: the memory budget fits only %d day(s) per block, fewer than "
                            "the configured time chunk; reducing the cube's time chunk to %d so "
                            "every block boundary stays chunk-aligned. Raise "
                            "datacube.memory_budget_gb, or lower datacube.chunks.time, to "
                            "choose this yourself.", name, block_days, time_chunk)

            if block_days >= len(days):
                # The cube fits: build it whole and write it once, exactly as before. Built
                # through assemble_block rather than assemble_aoi so it inherits the scan the
                # census already paid for, instead of repeating it.
                ds = assemble_block(g, eff, days, all_days=days, cache=cache)
                cov = finish_cube(ds, g, eff, days, cache)
                write_zarr_safe(ds, zpath,
                                build_encoding(ds, eff["compression"], eff["chunks"]))
                nvars, shape = len(ds.data_vars), (ds.sizes["time"], ds.sizes["y"], ds.sizes["x"])
                del ds
            else:
                cov = _assemble_blocked(g, eff, days, zpath, block_days=block_days,
                                        time_chunk=time_chunk, census=census, cache=cache)
                nvars, shape = len(census), (len(days), g.height, g.width)
        finally:
            close_cache(cache)

        # `t=%d` was always len(days) -- it said nothing about how much of the cube is real.
        # Report the coverage the cube actually has, so a thin product is visible here.
        # Days AND grid. Days alone said a product was complete when every one of its days
        # held a single corner of the AoI -- which is what a tiled sensor looks like when its
        # mosaic has collapsed onto one tile, and it read as 100% for a whole release.
        cov_str = ", ".join(
            f"{p} {100 * c['fraction']:.0f}% of days"
            + (f" ({100 * c['grid_fraction']:.0f}% of grid)" if "grid_fraction" in c else "")
            for p, c in sorted(cov.items()))
        log.info("  wrote %s  vars=%d shape=(t=%d,y=%d,x=%d)  coverage: %s", zpath.name,
                 nvars, *shape, cov_str or "n/a")
        rep.wrote()
        thin = [p for p, c in cov.items() if c["fraction"] < COVERAGE_WARN]
        if thin:
            rep.note = f"thin coverage: {', '.join(sorted(thin))} (see the cube's `coverage` attr)"
        # A product whose DAYS are well covered but whose GRID is not: every day arrived and
        # every day is mostly empty. Sensor swaths legitimately clip a large AoI, so this
        # warns rather than failing -- but it is the only place a corner-shaped cube announces
        # itself, and silence here is what let one ship.
        patchy = [p for p, c in cov.items()
                  if c["fraction"] >= COVERAGE_WARN
                  and c.get("grid_fraction", 1.0) < GRID_COVERAGE_WARN]
        if patchy:
            log.warning("  %s: covered on most days but filling under %.0f%% of the grid on "
                        "those days (%s). A sensor can legitimately clip a large AoI; if it "
                        "should not, check that every granule is contributing -- a mask that "
                        "rejects everything collapses a mosaicked day onto one granule.",
                        zpath.name, 100 * GRID_COVERAGE_WARN,
                        ", ".join(f"{p} {100 * cov[p]['grid_fraction']:.0f}%"
                                  for p in sorted(patchy)))

    rep.log_summary()
    return rep


def assemble(project: Project, *, grids=None, aois=None, dry_run=False,
             overwrite=False, memory_budget_gb=None) -> None:
    """Assemble datacubes for a validated Project. Terminal pipeline stage.

    Same signature as every product's acquire(); reads only the aligned files the
    acquisition stages wrote, so it must run AFTER them.

    `memory_budget_gb` overrides the config's (and the detection chain's) answer for THIS
    call. It exists for one caller: an orchestrator assembling several AoIs at once, which
    must DIVIDE the budget between them. `budget_bytes` otherwise assumes it owns the
    machine, so N concurrent AoIs would each claim the same allowance and their sum would be
    N times what is actually there -- and this stage is precisely the one that gets
    OOM-killed when that arithmetic is wrong.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if memory_budget_gb is not None:
        eff["memory_budget_gb"] = float(memory_budget_gb)
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    # The terminal stage's entry point is `assemble`, not `acquire`, but it takes the same
    # (project, aois, dry_run, overwrite) arguments -- so it rides the shared parser too.
    entry.process_main(assemble, "coastal_sst_data datacube assembler.")


if __name__ == "__main__":
    main()
