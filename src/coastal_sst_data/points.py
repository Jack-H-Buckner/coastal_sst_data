#!/usr/bin/env python3
"""
coastal_sst_data -- user-supplied extraction points: read them, and place them on a grid.

The pure half of the `extract` stage, split out for the same reason `processes.insitu` is
split from `processes.insitu_ioos`: no network, no zarr, no config objects, so every
geometric decision that can silently produce a WRONG NUMBER is testable on its own.

Two problems, and both fail quietly if you get them wrong:

  * WHAT THE FILE SAYS. A points CSV is written by a person, so its columns are called
    whatever they are called -- `lat`/`latitude`/`y`, `id`/`station`/`site`. Accepting the
    variants is necessary; accepting them CARELESSLY is worse than rejecting them. `x`/`y`
    is the trap: a file of PROJECTED coordinates uses exactly those names, and 512345.0
    read as a longitude does not raise anything -- it just lands in the Pacific, matches no
    AoI, and produces an empty table that looks like a configuration mistake. So the aliases
    are offered, and then every coordinate is range-checked against WGS84, and an ambiguous
    file (both `lat` and `latitude` present) is refused rather than guessed at.

  * WHICH PIXEL, AND WHOSE. A point is placed with `insitu.station_pixels` -- deliberately
    the SAME affine inversion the cube's own `insitu_station` channel uses, so the two can
    never disagree about which cell a lat/lon is in -- but with `water=None`, i.e. WITHOUT
    the snap-to-water step. A buoy is snapped off a land pixel because that is a mask
    artefact; an extraction point is where the user SAID it is, and moving it 300 m would
    change the answer with nothing in the output recording that it happened.

    A point can also fall inside several AoIs (overlapping regions are legal), or none. It
    gets exactly ONE AoI -- nearest grid centre, then name, so the answer never depends on
    dict ordering -- and a point in no AoI is dropped LOUDLY, because an empty output file
    and a mistyped coordinate column look identical from the outside.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Canonical field -> the spellings people actually write.
#
# `x`/`y` are accepted last and are the dangerous pair (see the module docstring); the WGS84
# range check in `read_points` is what makes offering them safe at all.
ALIASES: dict[str, tuple[str, ...]] = {
    "point_id": ("point_id", "id", "station_id", "station", "name", "site"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}

# Columns every downstream stage can rely on.
OUT_COLUMNS = ("point_id", "lat", "lon")


def _resolve_column(df: pd.DataFrame, field: str, override: str | None) -> str | None:
    """Which column of `df` holds `field` -- or None if it has none.

    An explicit override wins and must exist. Otherwise every alias is matched
    case-insensitively and AMBIGUITY IS AN ERROR: a file carrying both `lat` and `latitude`
    gives us no way to know which one is real, and picking by priority would silently
    extract at the wrong coordinates for anyone whose second column was the live one.
    """
    lower = {str(c).lower(): c for c in df.columns}
    if override is not None:
        if override not in df.columns:
            raise ValueError(
                f"extract.columns maps {field!r} to column {override!r}, which the points "
                f"file does not have. It has: {', '.join(map(str, df.columns))}")
        return override
    hits = [lower[a] for a in ALIASES[field] if a in lower]
    if len(hits) > 1:
        raise ValueError(
            f"the points file has more than one column that could be {field!r}: "
            f"{', '.join(map(str, hits))}. Name the right one explicitly with "
            f"`extract.columns.{field}`.")
    return hits[0] if hits else None


def read_points(path, columns: dict[str, str] | None = None) -> pd.DataFrame:
    """A points CSV -> DataFrame[point_id, lat, lon]. Loud about anything ambiguous.

    Extra columns in the input are dropped, so the extracted table's schema is fixed no
    matter how wide the points file is; join them back on `point_id`.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"points file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError(f"{path.name}: the points file has no rows.")

    over = dict(columns or {})
    unknown = set(over) - set(ALIASES)
    if unknown:
        raise ValueError(
            f"extract.columns has unknown field(s) {sorted(unknown)}; "
            f"choose from {', '.join(ALIASES)}.")

    cols = {f: _resolve_column(df, f, over.get(f)) for f in ALIASES}
    missing = [f for f in ("lat", "lon") if cols[f] is None]
    if missing:
        raise ValueError(
            f"{path.name}: no column found for {', '.join(missing)}. It has: "
            f"{', '.join(map(str, df.columns))}. Accepted names: "
            + "; ".join(f"{f} = {'/'.join(ALIASES[f])}" for f in missing)
            + ". Or name them with `extract.columns`.")

    out = pd.DataFrame({
        "lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
    })
    if cols["point_id"] is not None:
        out.insert(0, "point_id", df[cols["point_id"]].astype(str).str.strip())
    else:
        # Row numbers as ids cannot be joined back to anything the user recognises, so this
        # is a warning, not a convenience.
        out.insert(0, "point_id", [f"{path.stem}_{i:04d}" for i in range(1, len(df) + 1)])
        log.warning("  %s has no id column; ids were generated (%s_0001, ...). Add one of "
                    "%s so the output can be joined back to your sites.",
                    path.name, path.stem, "/".join(ALIASES["point_id"]))

    bad_num = out["lat"].isna() | out["lon"].isna()
    if bad_num.any():
        log.warning("  %s: dropped %d row(s) with a non-numeric or empty coordinate.",
                    path.name, int(bad_num.sum()))
        out = out[~bad_num]
    if out.empty:
        raise ValueError(f"{path.name}: no row has a usable lat/lon.")

    # The units tripwire. Deliberately NOT auto-swapped: a file we silently transposed is a
    # file whose coordinates nobody ever checks again.
    bad_lat = ~out["lat"].between(-90, 90)
    bad_lon = ~out["lon"].between(-180, 180)
    if bad_lat.any() or bad_lon.any():
        raise ValueError(
            f"{path.name}: {int(bad_lat.sum())} latitude(s) and {int(bad_lon.sum())} "
            f"longitude(s) fall outside WGS84 range (e.g. lat={out['lat'].iloc[0]:g}, "
            f"lon={out['lon'].iloc[0]:g}). Two things look like this: the lat/lon columns "
            f"are swapped, or the file holds PROJECTED coordinates in metres (a file with "
            f"`x`/`y` columns usually does). This package extracts by lat/lon only -- "
            f"name the right columns with `extract.columns`.")

    dup = out["point_id"].duplicated()
    if dup.any():
        raise ValueError(
            f"{path.name}: duplicate point id(s) {sorted(set(out['point_id'][dup]))[:5]}. "
            f"Ids are the extracted table's primary key -- duplicates fan out on any join.")
    co = out.duplicated(subset=["lat", "lon"])
    if co.any():
        # Legal (co-located sensors), but it means N identical row-sets in the output.
        log.warning("  %s: %d point(s) share coordinates with another point; they will "
                    "produce identical values under different ids.", path.name, int(co.sum()))

    return out.reset_index(drop=True)[list(OUT_COLUMNS)]


def grid_center(g) -> tuple[float, float]:
    """The grid's centre in its own projected CRS (metres)."""
    t = g.transform
    return t.c + 0.5 * g.width * t.a, t.f - 0.5 * g.height * t.a


def assign_aois(pts: pd.DataFrame, grids: dict) -> pd.DataFrame:
    """Give each point the ONE AoI whose grid contains it -> + [aoi, row, col, px, py].

    Containment is tested against the GRID, i.e. what the cube actually has pixels for --
    which is up to one resolution unit larger than the configured AoI bbox, because
    `compute_aoi_grid` snaps the origin outward. That is the right question to ask: a point
    the cube covers is a point we can extract.

    Several AoIs may contain one point. The tie-break is the nearest grid CENTRE, then the
    AoI name, so two runs of the same config never disagree. (Candidates can sit in
    different UTM zones; both distances are metres and the comparison only ever breaks a tie
    inside an overlap, so this is deliberately not a geodesic calculation.)

    `px`/`py` are the point's exact projected position in the winning AoI's CRS -- carried
    forward because the extraction disc is centred on the POINT, not on its pixel's centre,
    and recomputing it per channel would mean a pyproj Transformer per channel per point.
    """
    from pyproj import Transformer

    from .processes.insitu import station_pixels

    if not grids:
        raise ValueError("no AoI grids to assign points to.")

    lons = pts["lon"].tolist()      # plain lists: see station_pixels' pyproj note
    lats = pts["lat"].tolist()

    # One Transformer and one placement pass per AoI, not per point.
    placed: dict[str, list] = {}
    proj: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, g in grids.items():
        # water=None: NO SNAPPING. See the module docstring -- an extraction point is where
        # the user said it is.
        placed[name] = station_pixels(lons, lats, g, water=None)
        fwd = Transformer.from_crs("EPSG:4326", g.target_crs, always_xy=True)
        xs, ys = fwd.transform(lons, lats)
        proj[name] = (np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64"))

    rows, dropped = [], []
    for i, rec in enumerate(pts.itertuples(index=False)):
        cands = []
        for name in sorted(grids):                      # sorted: deterministic tie-break
            p = placed[name][i]
            if not p["inside"]:
                continue
            px, py = proj[name][0][i], proj[name][1][i]
            cx, cy = grid_center(grids[name])
            cands.append((float(np.hypot(px - cx, py - cy)), name, p, px, py))
        if not cands:
            dropped.append(rec.point_id)
            continue
        _, name, p, px, py = min(cands, key=lambda c: (c[0], c[1]))
        rows.append({"point_id": rec.point_id, "lat": rec.lat, "lon": rec.lon,
                     "aoi": name, "row": p["row"], "col": p["col"], "px": px, "py": py})

    if dropped:
        shown = ", ".join(map(str, dropped[:8])) + (" ..." if len(dropped) > 8 else "")
        log.warning("  %d of %d point(s) fall outside every AoI grid and were dropped: %s",
                    len(dropped), len(pts), shown)

    return pd.DataFrame(rows, columns=["point_id", "lat", "lon", "aoi",
                                       "row", "col", "px", "py"])


def flag_edge_points(assigned: pd.DataFrame, grids: dict, max_radius_m: float) -> None:
    """Warn for points whose neighbourhood will be CLIPPED by the grid edge.

    A clipped window still returns a mean -- of a half-disc -- and nothing in the value says
    so. This warns once per point at assignment time rather than once per time step, and it
    is why `count` is in the stat vocabulary: it is the only thing that makes the shrinkage
    visible in the output itself.
    """
    if max_radius_m <= 0 or assigned.empty:
        return
    for rec in assigned.itertuples(index=False):
        g = grids[rec.aoi]
        free_px = min(rec.row, rec.col, g.height - 1 - rec.row, g.width - 1 - rec.col)
        free_m = free_px * g.resolution_m
        if free_m < max_radius_m:
            log.warning("  point %s is %.0f m from the edge of AoI %s, closer than its "
                        "%.0f m neighbourhood: its window is clipped and the statistic is "
                        "over a partial disc (check the `count` stat).",
                        rec.point_id, free_m, rec.aoi, max_radius_m)
