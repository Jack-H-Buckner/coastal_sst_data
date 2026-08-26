#!/usr/bin/env python3
"""
coastal_sst_data -- in-situ observations: the shared, source-agnostic half.

The acquisition side is per-network (processes.insitu_ioos and, later, others behind the
same `source` selector). THIS module is what every network's output feeds into: the pure
functions the datacube assembler uses to put a point observation into a grid cell, and to
match it in time to a satellite overpass. No network, no config -- so it is trivially
testable and shared by every source.

Two problems to get right:

  * WHERE. A station is a point; a grid cell is 100 m. The value goes in the cell the
    station sits in -- but a mooring near shore, on a coarse water mask, can land in a
    cell the cube calls LAND, where it would be masked out of every downstream loss. So a
    station in a land cell is SNAPPED to the nearest water pixel, and the snap distance is
    recorded: a long snap means the station is not where we think it is, and that has to
    be visible rather than silently absorbed.

  * WHEN. Buoys report every 6-60 min; the cube's axis is daily. A daily mean would throw
    away the thing that makes in-situ valuable -- that it can be matched to the INSTANT a
    satellite flew. So a value is picked at a target time (an overpass, or the daily
    reference time) as the NEAREST observation within a tolerance, and the signed offset
    is kept alongside it. Beyond the tolerance the answer is NaN, never a stale value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# A snap longer than this means the station's coordinates and our water mask disagree
# badly enough that the pixel is probably wrong -- say so.
SNAP_WARN_M = 500.0

DEFAULT_MAX_DT_MIN = 60.0     # matchup tolerance


@dataclass(frozen=True)
class InsituTable:
    """One source's station table, IN MEMORY. Mirrors the on-disk schema that
    `insitu_acquire.build_dataset` writes: dims (station, time).

    WHY THIS IS NOT AN `xr.Dataset`. It used to be one, handed back still open and read
    lazily for the whole of an AoI's assembly. That put a live netCDF handle outside
    `store.open_netcdf` -- the one place in the package that was, after the gate went in --
    on a thread pool, which is the precondition for the upstream deadlock the gate exists to
    remove (see store.NETCDF_LOCK). It also kept alive a documented segfault: `provenance`
    re-opening every aligned file while these were open took netCDF down outright, and
    nothing but call ordering prevented it.

    Plain arrays make that unrepresentable. The file is read inside one gated block and shut;
    what comes back is data, owns nothing, and cannot be closed at the wrong moment.

    `qc` is deliberately absent. `insitu_acquire.build_dataset` writes it, but nothing in this
    package reads it -- carrying it would double the table for no consumer. Add it here if a
    consumer ever appears; its absence is a decision, not an oversight.
    """

    lon: np.ndarray            # (station,) float64
    lat: np.ndarray            # (station,) float64
    ids: np.ndarray            # (station,) station identifiers
    names: np.ndarray          # (station,) human-readable station names
    times: pd.DatetimeIndex    # (time,) the union axis every station is reindexed onto
    sst: np.ndarray            # (station, time) float32

    @classmethod
    def from_dataset(cls, ds) -> InsituTable:
        """Materialise every array from an open Dataset -> a table that owns no handle.

        CALL THIS INSIDE THE `store.open_netcdf` BLOCK. Every `.values` here is what forces
        the read; doing it after the block would be the unguarded lazy read this type exists
        to prevent -- and, once the file is shut, an error rather than a subtle one.
        """
        return cls(lon=np.asarray(ds["lon"].values, dtype="float64"),
                   lat=np.asarray(ds["lat"].values, dtype="float64"),
                   ids=np.asarray(ds["station_id"].values),
                   names=np.asarray(ds["station_name"].values),
                   times=pd.DatetimeIndex(ds["time"].values),
                   sst=np.asarray(ds["sst"].values))

    @property
    def n_stations(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class TrackTable:
    """One source's MOVING-platform observations, IN MEMORY. Mirrors the on-disk schema that
    `insitu_mobile.build_track_dataset` writes: dims `(obs,)`.

    A FLAT LIST, not `InsituTable`'s `(station, time)` rectangle, because a track has no
    rectangle: every observation carries its own position, and platforms whose sampling
    schedules have nothing in common would make a union time axis the SUM of their lengths
    with a block that is almost entirely NaN.

    Same handle-ownership rule as `InsituTable`: built inside the `store.open_netcdf` block,
    owns nothing, cannot be closed at the wrong moment.
    """

    times: pd.DatetimeIndex    # (obs,)
    lon: np.ndarray            # (obs,) float64
    lat: np.ndarray            # (obs,) float64
    sst: np.ndarray            # (obs,) float32
    platform: np.ndarray       # (obs,) the platform each observation came from

    @classmethod
    def from_dataset(cls, ds) -> TrackTable:
        """Materialise every array from an open Dataset. CALL THIS INSIDE THE GATED BLOCK."""
        return cls(times=pd.DatetimeIndex(ds["time"].values),
                   lon=np.asarray(ds["lon"].values, dtype="float64"),
                   lat=np.asarray(ds["lat"].values, dtype="float64"),
                   sst=np.asarray(ds["sst"].values, dtype="float32"),
                   platform=np.asarray(ds["platform_id"].values))

    @property
    def n_obs(self) -> int:
        return len(self.times)


def observation_pixels(lons, lats, g):
    """Positions -> (rows, cols, inside) arrays, one entry per position. Vectorized, no snapping.

    The placement half of `station_pixels`, split out because a MOVING platform needs a pixel
    per OBSERVATION rather than one per station, and a track can be 10^5-10^6 rows -- far too
    many for the per-point Python loop and list-of-dicts that `station_pixels` returns.

    One `Transformer` and one affine inversion for every position, however many there are; the
    arithmetic was always vectorized (only the return shape was not). Positions outside the grid
    come back with `inside=False` and their row/col clamped to 0 -- the caller must mask on
    `inside` rather than trusting the index, which is exactly what `station_pixels` does.

    NO SNAPPING, deliberately, and that is not merely an omission to keep this pure: snapping
    exists to move a mooring's NOMINAL coordinates onto water when a coarse mask disagrees with
    them. A glider's position is MEASURED. Moving it would be inventing a track.
    """
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", g.target_crs, always_xy=True)
    lon = np.asarray(lons, dtype="float64").ravel()
    lat = np.asarray(lats, dtype="float64").ravel()
    # Plain lists, not 0-d/1-element arrays: pyproj takes its scalar path on those and
    # emits a NumPy deprecation that becomes an error in a later release.
    xs, ys = fwd.transform(lon.tolist(), lat.tolist())

    # Invert the affine: the grid's origin is its top-left corner, y descending.
    x0, y0, res = g.transform.c, g.transform.f, g.transform.a
    cols = np.floor((np.asarray(xs, dtype="float64") - x0) / res)
    rows = np.floor((y0 - np.asarray(ys, dtype="float64")) / res)

    # A non-finite position is not "outside the grid", it is no position at all -- but it would
    # sail through the bounds test below as NaN, and `astype(int)` on NaN is a platform-dependent
    # garbage index. Excluded explicitly.
    ok = np.isfinite(rows) & np.isfinite(cols)
    rows = np.where(ok, rows, 0).astype("int64")
    cols = np.where(ok, cols, 0).astype("int64")
    inside = ok & (rows >= 0) & (rows < g.height) & (cols >= 0) & (cols < g.width)
    return np.where(inside, rows, 0), np.where(inside, cols, 0), inside


def station_pixels(lons, lats, g, water: np.ndarray | None = None):
    """Map stations to grid cells, snapping a land cell to the nearest water pixel.

    Returns a list of dicts: {row, col, snap_m, inside}. `inside` is False for a station
    that falls outside the AoI grid entirely (it is dropped by the caller).

    `water` is the cube's (H,W) boolean water mask. When it is None (no land-cover and no
    bathymetry), no snapping is attempted -- with nothing known to be water, snapping
    would be guesswork.

    A thin wrapper over `observation_pixels` since tracks arrived: the placement arithmetic is
    identical, and only the snapping and the dict-per-station return shape are this function's
    own. `points.assign_aois` and `extract` share this exact affine so a lat/lon cannot land in
    one pixel here and another there -- which is why the split had to leave this signature and
    its behaviour untouched.
    """
    rows_a, cols_a, inside_a = observation_pixels(lons, lats, g)
    rows, cols = rows_a.tolist(), cols_a.tolist()
    res = g.transform.a

    idx = None
    if water is not None and water.any():
        from scipy.ndimage import distance_transform_edt
        # Nearest water pixel for every cell, computed once for all stations.
        dist, idx = distance_transform_edt(~water, return_distances=True,
                                           return_indices=True)

    out = []
    for r, c, ins in zip(rows, cols, inside_a.tolist()):
        if not ins:
            out.append({"row": None, "col": None, "snap_m": None, "inside": False})
            continue
        snap = 0.0
        if idx is not None and not water[r, c]:
            rr, cc = int(idx[0][r, c]), int(idx[1][r, c])
            snap = float(np.hypot(rr - r, cc - c) * res)
            r, c = rr, cc
        out.append({"row": int(r), "col": int(c), "snap_m": snap, "inside": True})
    return out


def nearest_index(times: pd.DatetimeIndex, values: np.ndarray, target,
                  max_dt_min: float = DEFAULT_MAX_DT_MIN):
    """The observation NEAREST `target`, as (index, signed_dt_minutes), or (None, NaN).

    THE INDEX, not just the value, and it is an index into the ORIGINAL `times`/`values` -- not
    into the finite-only subset. That distinction is the whole reason this function exists: a
    moving platform needs to look up the winning observation's POSITION, and a position array
    is indexed by the original axis. `value_at` used to compact by `finite` first and return
    only the value, so its internal `k` was an index into the compacted array; reusing it for a
    track would have placed observations at the wrong coordinates -- silently, and plausibly.

    Only finite observations are candidates -- the nearest-in-time record is useless if it is a
    gap. Beyond `max_dt_min` the answer is (None, NaN): a buoy reading two hours away from an
    overpass is not a matchup, and pretending otherwise is how a validation set quietly acquires
    a bias.

    `dt` is signed: positive means the observation came AFTER the target.
    """
    if target is None or len(times) == 0:
        return None, np.nan
    finite = np.isfinite(np.asarray(values))
    if not finite.any():
        return None, np.nan

    # `where` maps a position in the compacted array back to the original axis.
    where = np.flatnonzero(finite)
    t = pd.DatetimeIndex(times)[finite]
    dt_min = (t - pd.Timestamp(target)).total_seconds().to_numpy() / 60.0
    j = int(np.argmin(np.abs(dt_min)))
    if abs(dt_min[j]) > float(max_dt_min):
        return None, np.nan
    return int(where[j]), float(dt_min[j])


def value_at(times: pd.DatetimeIndex, values: np.ndarray, target,
             max_dt_min: float = DEFAULT_MAX_DT_MIN):
    """The observation NEAREST `target`, as (value, signed_dt_minutes), or (NaN, NaN).

    A wrapper over `nearest_index` for the fixed-station callers, which need the value and
    never the position -- their position is a property of the station, not of the observation.
    """
    k, dt = nearest_index(times, values, target, max_dt_min)
    if k is None:
        return np.nan, np.nan
    return float(np.asarray(values)[k]), dt
