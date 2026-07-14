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

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# A snap longer than this means the station's coordinates and our water mask disagree
# badly enough that the pixel is probably wrong -- say so.
SNAP_WARN_M = 500.0

DEFAULT_MAX_DT_MIN = 60.0     # matchup tolerance


def station_pixels(lons, lats, g, water: np.ndarray | None = None):
    """Map stations to grid cells, snapping a land cell to the nearest water pixel.

    Returns a list of dicts: {row, col, snap_m, inside}. `inside` is False for a station
    that falls outside the AoI grid entirely (it is dropped by the caller).

    `water` is the cube's (H,W) boolean water mask. When it is None (no land-cover and no
    bathymetry), no snapping is attempted -- with nothing known to be water, snapping
    would be guesswork.
    """
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", g.target_crs, always_xy=True)
    # Plain lists, not 0-d/1-element arrays: pyproj takes its scalar path on those and
    # emits a NumPy deprecation that becomes an error in a later release.
    xs, ys = fwd.transform([float(v) for v in lons], [float(v) for v in lats])

    # Invert the affine: the grid's origin is its top-left corner, y descending.
    x0, y0, res = g.transform.c, g.transform.f, g.transform.a
    cols = np.floor((np.asarray(xs, dtype="float64") - x0) / res).astype(int)
    rows = np.floor((y0 - np.asarray(ys, dtype="float64")) / res).astype(int)

    idx = None
    if water is not None and water.any():
        from scipy.ndimage import distance_transform_edt
        # Nearest water pixel for every cell, computed once for all stations.
        dist, idx = distance_transform_edt(~water, return_distances=True,
                                           return_indices=True)

    out = []
    for r, c in zip(rows, cols):
        if not (0 <= r < g.height and 0 <= c < g.width):
            out.append({"row": None, "col": None, "snap_m": None, "inside": False})
            continue
        snap = 0.0
        if idx is not None and not water[r, c]:
            rr, cc = int(idx[0][r, c]), int(idx[1][r, c])
            snap = float(np.hypot(rr - r, cc - c) * res)
            r, c = rr, cc
        out.append({"row": int(r), "col": int(c), "snap_m": snap, "inside": True})
    return out


def value_at(times: pd.DatetimeIndex, values: np.ndarray, target,
             max_dt_min: float = DEFAULT_MAX_DT_MIN):
    """The observation NEAREST `target`, as (value, signed_dt_minutes).

    Only finite observations are candidates -- the nearest-in-time record is useless if it
    is a gap. Beyond `max_dt_min` the answer is (NaN, NaN): a buoy reading two hours away
    from an overpass is not a matchup, and pretending otherwise is how a validation set
    quietly acquires a bias.

    `dt` is signed: positive means the observation came AFTER the target.
    """
    if target is None or len(times) == 0:
        return np.nan, np.nan
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, np.nan

    t = pd.DatetimeIndex(times)[finite]
    v = np.asarray(values)[finite]
    dt_min = (t - pd.Timestamp(target)).total_seconds().to_numpy() / 60.0
    k = int(np.argmin(np.abs(dt_min)))
    if abs(dt_min[k]) > float(max_dt_min):
        return np.nan, np.nan
    return float(v[k]), float(dt_min[k])
