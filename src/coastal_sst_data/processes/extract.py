#!/usr/bin/env python3
"""
coastal_sst_data -- point extraction: assembled cubes -> one long-format table.

The transpose of everything else in this package. The pipeline spends its effort turning
scattered granules into a dense `(time, y, x)` cube; a downstream model usually wants the
opposite -- for a handful of sites, the time series of every channel, as rows. This stage is
that transpose, and nothing else: it READS the cubes and writes one table. It never touches
a `.zarr`, so it cannot corrupt anything upstream and is always safe to re-run.

It is also OPTIONAL in a load-bearing way. Most projects never call it, so it costs them
nothing: the config block defaults, the module is imported inside the CLI handler rather
than at package import, pyarrow is an extra behind a lazy import, and no pipeline stage
invokes it. Deleting `extract:` from a config changes nothing about a run.

What has to be right, because all four fail SILENTLY -- with finite, plausible numbers:

  * WHICH PIXEL. The affine inversion is `insitu.station_pixels`, not a local copy, so this
    stage and the cube's own `insitu_station` channel cannot disagree about where a lat/lon
    lands. See `points.assign_aois`.

  * WHICH GRID. The points are placed on the grid the CONFIG computes; the values come out
    of a cube some earlier run WROTE. If the config's resolution or CRS changed in between,
    every read is from the wrong cell of a perfectly valid cube. `grid_from_cube` re-derives
    the grid from the store's own coordinates and refuses to proceed if the two disagree.

  * HOW BIG THE NEIGHBOURHOOD IS. `radius_m` is metres in the cube's projected CRS, measured
    from the point's exact position to each pixel CENTRE, and the region is a disc -- a box
    of the same nominal size reaches radius*sqrt(2) into its corners. The pixel containing
    the point is always included, because a radius smaller than the posting (a 50 m radius
    on the default 100 m grid) otherwise selects nothing and returns a column of NaN that
    reads exactly like a channel that was cloudy for the entire record.

  * WHAT THE STATISTIC MEANS WHEN DATA IS MISSING. `mean` and `nanmean` are both offered and
    are different questions: over a cloudy neighbourhood the first says NaN and the second
    averages what was seen. `nearest` is the containing pixel's value and is NEVER the
    nearest FINITE value -- substituting a neighbour is how a validation set quietly
    acquires a warm bias (`insitu.value_at` makes the same call in the time dimension).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_origin

from .. import entry, points, report, store
from ..config import PERCENTILE_RE, WATER_MASK_CHANNEL, Project
from ..grid import AoiGrid, project_grids, select_aois

log = logging.getLogger(__name__)

# The output schema, fixed. Downstream code pivots on these names.
COLUMNS = ["point_id", "lat", "lon", "aoi", "time", "variable", "stat", "radius_m", "value"]
# ...and its primary key. Asserted after assembly: a duplicate here means a point was placed
# in two AoIs or a channel was configured twice, both of which fan out on any join.
KEY = ["point_id", "aoi", "time", "variable", "stat"]

T3 = ("time", "y", "x")
T2 = ("y", "x")
T1 = ("time",)

# Above this, a CSV is worth warning about (the parquet is roughly a tenth the size).
CSV_ROW_WARN = 2_000_000

# Slack, in metres, on the disc's edge. A round radius on a round posting puts pixels at
# EXACTLY the radius -- a 300 m disc on a 100 m grid has four of them -- and their distances
# come out as 300.0000000009 and 299.9999999991 because the point's position has been through
# a lon/lat round-trip. A bare `<=` then admits one and rejects its mirror image, so the
# neighbourhood is lopsided by a nanometre of float noise, systematically, and `count` reads
# 27 where the geometry says 29. A micrometre is far below any real positional accuracy and
# makes the edge deterministic and symmetric.
EDGE_TOL_M = 1e-6


# --------------------------------------------------------------------------- #
# The grid the cube was actually written on
# --------------------------------------------------------------------------- #
def grid_from_cube(ds: xr.Dataset, g: AoiGrid) -> AoiGrid:
    """The grid the CUBE holds, cross-checked against the one this config computes.

    Indexing a cube with a grid it was not written on is the quietest wrong answer available
    here: every value comes back finite, in range, and from the wrong place. So the read uses
    the store's OWN coordinates, and any divergence from the config is refused rather than
    reconciled -- there is no way to tell, from inside, which of the two the user meant.
    """
    crs = ds.attrs.get("crs")
    if not crs:
        # An older cube, or one written by something else. The config's CRS is the only
        # candidate, but assuming it silently would make a wrong answer unattributable.
        log.warning("  %s: the cube carries no `crs` attr; assuming the config's %s. "
                    "Rebuild it (`assemble --overwrite`) to stamp one.",
                    g.name, g.target_crs)
        crs = g.target_crs

    ys = np.asarray(ds["y"].values, dtype="float64")
    xs = np.asarray(ds["x"].values, dtype="float64")
    if len(ys) < 2 or len(xs) < 2:
        raise ValueError(f"{g.name}: the cube's y/x axes are too short to describe a grid.")
    if ys[1] >= ys[0]:
        raise ValueError(
            f"{g.name}: the cube's `y` coordinate ASCENDS. Every grid this package writes "
            f"has a top-left origin with y running down (grid.AoiGrid.xy_centers), and the "
            f"row arithmetic here assumes it -- an ascending axis would flip every window "
            f"north for south without raising anything.")

    res = float(xs[1] - xs[0])
    cube_g = replace(g, target_crs=str(crs), resolution_m=res,
                     width=len(xs), height=len(ys),
                     transform=from_origin(xs[0] - res / 2, ys[0] + res / 2, res, res))
    if (cube_g.target_crs != g.target_crs or cube_g.shape != g.shape
            or abs(cube_g.resolution_m - g.resolution_m) > 1e-6):
        raise ValueError(
            f"{g.name}: the assembled cube ({cube_g.target_crs}, {cube_g.width}x"
            f"{cube_g.height} @ {cube_g.resolution_m:g} m) does not match the grid this "
            f"config computes ({g.target_crs}, {g.width}x{g.height} @ "
            f"{g.resolution_m:g} m). The config's grid changed after the cube was built; "
            f"re-run `assemble --aoi {g.name} --overwrite`.")
    return cube_g


# --------------------------------------------------------------------------- #
# What each requested channel is, and whether it can be extracted at all
# --------------------------------------------------------------------------- #
def plan_channels(ds: xr.Dataset, channels: dict) -> list[tuple[str, object, tuple]]:
    """Validate the requested channels against the open cube -> [(name, spec, dims)].

    A listed channel the cube does not have is a HARD ERROR. Skipping it would put a
    silently-absent variable into a modelling table, which reads identically to a variable
    that was present and genuinely all-NaN -- and no downstream check can tell them apart.
    """
    missing = [c for c in channels if c not in ds.data_vars]
    if missing:
        problems = []
        for c in missing:
            near = get_close_matches(c, list(ds.data_vars), n=3, cutoff=0.6)
            problems.append(f"{c!r}" + (f" (did you mean {', '.join(near)}?)" if near else ""))
        raise ValueError(
            f"extract.channels names {len(missing)} channel(s) this cube does not have: "
            + "; ".join(problems)
            + f". The cube holds: {', '.join(sorted(ds.data_vars))}")

    plan, overspecified = [], []
    for name, spec in channels.items():
        dims = tuple(ds[name].dims)
        if dims not in (T3, T2, T1):
            raise ValueError(
                f"channel {name!r} has dims {dims}, which this stage cannot place at a "
                f"point. It handles {T3} (per-day raster), {T2} (static raster) and "
                f"{T1} (AoI-wide series).")
        if dims == T1:
            given = _spatial_options(spec)
            if given:
                # Collected, not raised: a cube carries a DOZEN 1-D channels (every
                # <sensor>_hour, every tide_*, doy_sin/cos, the georef diagnostics), so a
                # config that applies one uniform block to everything trips all of them.
                # Raising on the first would mean a dozen edit-and-re-run cycles, each one
                # paying for the whole point-assignment pass before it failed again.
                overspecified.append((name, given))
                continue
        plan.append((name, spec, dims))

    if overspecified:
        raise ValueError(_one_d_message(overspecified))
    return plan


def _spatial_options(spec) -> list[str]:
    """The options on `spec` that only mean something for a channel with a grid.

    Echoed back with their VALUES in the error below: in a config with dozens of channels,
    "it has a radius" is not enough to find the line to edit.
    """
    given = []
    if spec.radius_m > 0:
        given.append(f"radius_m: {spec.radius_m:g}")
    if set(spec.stat) != {"nearest"}:
        # Bracketed when it is a list, so a multi-stat value cannot be misread as two
        # separate options in the comma-joined line below.
        stat = spec.stat[0] if len(spec.stat) == 1 else "[" + ", ".join(spec.stat) + "]"
        given.append(f"stat: {stat}")
    if spec.mask:
        given.append(f"mask: {spec.mask}")
    return given


def _one_d_message(overspecified: list[tuple[str, list[str]]]) -> str:
    """The error for 1-D channels given spatial options.

    These channels ARE extractable, and the message's whole job is to say so BEFORE it says
    anything else. The previous wording led with the failure and put the one-word fix last,
    and was read -- reasonably -- as "extraction does not support overpass times at all".
    """
    width = max(len(n) for n, _ in overspecified)
    listing = "\n".join(f"    {n:<{width}}  has {', '.join(g)}" for n, g in overspecified)
    yaml = "\n".join(f"      {n}:" for n, _ in overspecified)
    n = len(overspecified)
    return (
        f"{n} channel(s) in extract.channels are 1-D {T1}: one value per day for the WHOLE "
        f"AoI (an overpass time, a tide height, a day-of-year term), so there is no "
        f"neighbourhood for a radius or a statistic to reduce over --\n\n"
        f"{listing}\n\n"
        f"These channels ARE extracted. Write them with no options:\n\n"
        f"    channels:\n{yaml}\n\n"
        f"Each then ships one row per date, with stat=nearest and radius_m=0, carrying that "
        f"day's AoI-wide value for every point in the AoI. (Writing `stat: nearest` "
        f"explicitly is also accepted.)")


def resolve_mask(ds: xr.Dataset, spec, cache: dict | None = None) -> np.ndarray | None:
    """The (H,W) boolean mask a channel's neighbourhood is restricted to, or None.

    `water` is an alias for the cube's water mask; anything else names a channel. A missing
    one is an error and never a silent fall-back to the unmasked window -- a coastal mean
    with land in it is exactly the number the mask was configured to prevent.

    `cache` holds the resolved arrays for one AoI: several channels usually share `water`,
    and a mask is a whole grid (tens of MB on a real AoI) that would otherwise be read once
    per channel that names it.
    """
    if not spec.mask:
        return None
    name = WATER_MASK_CHANNEL if spec.mask == "water" else spec.mask
    if cache is not None and name in cache:
        return cache[name]
    if name not in ds.data_vars:
        extra = (f" (`mask: water` needs the `{WATER_MASK_CHANNEL}` channel, i.e. the "
                 f"landcover product)" if spec.mask == "water" else "")
        raise ValueError(
            f"mask {spec.mask!r} names channel {name!r}, which this cube does not have"
            f"{extra}.")
    if tuple(ds[name].dims) != T2:
        raise ValueError(
            f"mask channel {name!r} has dims {tuple(ds[name].dims)}; a mask must be static "
            f"{T2}.")
    arr = np.asarray(ds[name].values) > 0
    if cache is not None:
        cache[name] = arr
    return arr


# --------------------------------------------------------------------------- #
# The neighbourhood
# --------------------------------------------------------------------------- #
def window(g: AoiGrid, row: int, col: int, px: float, py: float, radius_m: float):
    """The pixel window and its disc mask -> (yslice, xslice, mask, clipped).

    A DISC, not a box: a box's corners reach radius*sqrt(2), so a "300 m neighbourhood"
    would quietly be 424 m along the diagonal, and how much extra depended on the AoI's UTM
    orientation. Distances run from the point's exact projected position -- `px`/`py`, not
    its pixel's centre -- to each pixel centre, both metres in the cube's CRS.

    The containing pixel is ALWAYS in the mask. A radius under ~res/sqrt(2) contains no
    pixel centre for most point positions (which includes a 50 m radius on the default 100 m
    grid), and without this clause the reduction would run over an empty set and return NaN
    for the whole record -- indistinguishable from a channel with no data.

    Pixels exactly ON the edge are IN, within `EDGE_TOL_M` -- see the constant for why a bare
    `<=` makes the disc lopsided.
    """
    res = g.resolution_m
    rad_px = int(np.floor(radius_m / res))
    want = (row - rad_px, row + rad_px + 1, col - rad_px, col + rad_px + 1)
    r0, r1 = max(want[0], 0), min(want[1], g.height)
    c0, c1 = max(want[2], 0), min(want[3], g.width)
    clipped = (r0, r1, c0, c1) != want

    xs, ys = g.xy_centers()
    dx = xs[c0:c1] - px
    dy = ys[r0:r1] - py
    mask = (dy[:, None] ** 2 + dx[None, :] ** 2) <= (radius_m + EDGE_TOL_M) ** 2
    mask[row - r0, col - c0] = True
    return slice(r0, r1), slice(c0, c1), mask, clipped


def reduce_stat(vals: np.ndarray, stat: str) -> np.ndarray:
    """Reduce a (T, n) neighbourhood along its pixel axis -> (T,).

    `vals` already has the disc/mask applied (pixels outside it are NaN and dropped from
    `count` by the caller). NaN in, NaN out: nothing here substitutes a neighbouring pixel
    for a missing one.
    """
    axis = -1
    if stat == "count_valid":
        return np.isfinite(vals).sum(axis=axis).astype("float64")
    if stat == "count":
        # Every pixel the disc selected, finite or not -- how a clipped window or a mask
        # that emptied the neighbourhood becomes visible in the output itself.
        return np.full(vals.shape[0], float(vals.shape[axis]))

    # np.nanmean of an all-NaN slice is a legitimate outcome here (a cloudy day), not a
    # defect, so its RuntimeWarning is noise rather than signal.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if stat in ("mean", "median", "std", "min", "max", "sum"):
            fn = {"mean": np.mean, "median": np.median, "std": np.std,
                  "min": np.min, "max": np.max, "sum": np.sum}[stat]
            kw = {"ddof": 0} if stat == "std" else {}
            return np.asarray(fn(vals, axis=axis, **kw), dtype="float64")
        if stat.startswith("nan"):
            fn = {"nanmean": np.nanmean, "nanmedian": np.nanmedian, "nanstd": np.nanstd,
                  "nanmin": np.nanmin, "nanmax": np.nanmax, "nansum": np.nansum}[stat]
            kw = {"ddof": 0} if stat == "nanstd" else {}
            return np.asarray(fn(vals, axis=axis, **kw), dtype="float64")
        m = PERCENTILE_RE.match(stat)
        if m:
            return np.asarray(np.nanpercentile(vals, float(stat[1:]), axis=axis),
                              dtype="float64")
    raise ValueError(f"unknown stat {stat!r}")   # unreachable: config validates the set


# --------------------------------------------------------------------------- #
# Reading one channel for the points of one AoI
# --------------------------------------------------------------------------- #
def _union_bounds(wins):
    r0 = min(w[0].start for w in wins)
    r1 = max(w[0].stop for w in wins)
    c0 = min(w[1].start for w in wins)
    c1 = max(w[1].stop for w in wins)
    return r0, r1, c0, c1


def read_windows(da: xr.DataArray, wins, budget_bytes: float) -> list[np.ndarray]:
    """Per-point (T, h, w) arrays for one channel, touching as few zarr chunks as possible.

    The cube is chunked (time 64, y 128, x 128), so a naive `.isel` per point re-reads a
    multi-megabyte chunk to serve a 3x3 window -- and a run with a hundred points re-reads
    the same chunks a hundred times.

    UNION READ (the normal case): extraction points are usually clustered, so the union of
    their windows is a small part of the grid. Read that ONE slab and slice it in NumPy;
    each chunk is touched once.

    PER-POINT READ (the fallback): when the union would not fit the budget -- points
    scattered across a large AoI -- fall back to one read per point. Identical output, very
    different cost, so which one ran is logged.
    """
    if not wins:
        return []
    has_time = da.dims[0] == "time"
    nt = da.sizes["time"] if has_time else 1
    r0, r1, c0, c1 = _union_bounds(wins)
    union_bytes = nt * (r1 - r0) * (c1 - c0) * 8

    if union_bytes <= budget_bytes:
        log.debug("    %s: union read %dx%d for %d point(s) (%.1f MB)",
                  da.name, r1 - r0, c1 - c0, len(wins), union_bytes / 1e6)
        slab = np.asarray(da.isel(y=slice(r0, r1), x=slice(c0, c1)).values, dtype="float64")
        if not has_time:
            slab = slab[None, ...]
        return [slab[:, w[0].start - r0:w[0].stop - r0, w[1].start - c0:w[1].stop - c0]
                for w in wins]

    log.info("    %s: points are too scattered for one read (%.1f GB union); reading each "
             "point's window separately.", da.name, union_bytes / 1e9)
    out = []
    for w in wins:
        arr = np.asarray(da.isel(y=w[0], x=w[1]).values, dtype="float64")
        out.append(arr if has_time else arr[None, ...])
    return out


# --------------------------------------------------------------------------- #
# One AoI -> its rows
# --------------------------------------------------------------------------- #
def extract_aoi(ds: xr.Dataset, g: AoiGrid, pts: pd.DataFrame, channels: dict,
                budget_bytes: float = 1e9) -> pd.DataFrame:
    """One open cube plus the points inside it -> that AoI's long-format rows.

    `g` must already be the CUBE's grid (see `grid_from_cube`), and `pts` must carry the
    row/col/px/py that `points.assign_aois` produced against it.
    """
    plan = plan_channels(ds, channels)
    times = pd.DatetimeIndex(ds["time"].values) if "time" in ds.coords else pd.DatetimeIndex([])
    mask_cache: dict[str, np.ndarray] = {}
    frames = []

    for name, spec, dims in plan:
        radius = float(spec.radius_m)
        if 0 < radius < g.resolution_m:
            log.warning("  %s: radius_m=%.0f is smaller than the %.0f m grid posting, so "
                        "the neighbourhood is the single pixel each point falls in and "
                        "every statistic equals `nearest`.", name, radius, g.resolution_m)

        if dims == T1:
            # AoI-wide: one read, tiled over the points. Never per point.
            vals = np.asarray(ds[name].values, dtype="float64")
            for rec in pts.itertuples(index=False):
                frames.append(_frame(rec, name, "nearest", 0.0, times, vals))
            continue

        mask2d = resolve_mask(ds, spec, mask_cache)
        wins = [window(g, int(r.row), int(r.col), float(r.px), float(r.py), radius)
                for r in pts.itertuples(index=False)]
        blocks = read_windows(ds[name], [(w[0], w[1]) for w in wins], budget_bytes)

        for rec, win, block in zip(pts.itertuples(index=False), wins, blocks):
            ys, xs, disc, _clipped = win
            keep = disc
            if mask2d is not None:
                # The mask WINS over the always-include-the-containing-pixel rule: a point
                # whose own cell is land must contribute nothing rather than smuggle a land
                # value into a water-only statistic.
                keep = keep & mask2d[ys, xs]
            # (T, n) over the selected pixels; an empty selection is a real, visible outcome.
            sel = block[:, keep] if keep.any() else block[:, :0].reshape(block.shape[0], 0)
            centre = block[:, int(rec.row) - ys.start, int(rec.col) - xs.start]

            for stat in spec.stat:
                if stat == "nearest":
                    # One specific pixel: the radius and the mask do not apply, by definition.
                    vals, used_radius = centre, 0.0
                elif sel.shape[1] == 0:
                    vals = np.zeros(block.shape[0]) if stat.startswith("count") \
                        else np.full(block.shape[0], np.nan)
                    used_radius = radius
                else:
                    vals, used_radius = reduce_stat(sel, stat), radius
                t = times if dims == T3 else pd.DatetimeIndex([pd.NaT])
                frames.append(_frame(rec, name, stat, used_radius, t, np.asarray(vals)))

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _frame(rec, variable: str, stat: str, radius_m: float,
           times: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    """One (point, variable, stat) block of rows."""
    n = len(times)
    return pd.DataFrame({
        "point_id": np.repeat(rec.point_id, n),
        "lat": np.repeat(float(rec.lat), n),        # the INPUT coords, not the pixel centre
        "lon": np.repeat(float(rec.lon), n),
        "aoi": np.repeat(rec.aoi, n),
        "time": times,
        "variable": np.repeat(variable, n),
        "stat": np.repeat(stat, n),
        "radius_m": np.repeat(float(radius_m), n),  # what this row USED, not what was asked
        "value": np.asarray(values, dtype="float64").reshape(n),
    })


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #
def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    ex = project.extract
    root = Path(project.output_dir)
    return {
        "project": project,
        "cube_dir": root / project.datacube.output_subdir,
        "out_dir": root / ex.output_subdir,
        "stem": ex.stem,
        "points": ex.points,
        "columns": dict(ex.columns),
        "channels": dict(ex.channels),
        "format": ex.format,
        "overwrite": bool(ex.overwrite),
        # Falls back to the assembler's budget: the stages usually want the same answer.
        "memory_budget_gb": (ex.memory_budget_gb if ex.memory_budget_gb is not None
                             else project.datacube.memory_budget_gb),
    }


def _budget_bytes(eff: dict) -> float:
    """Bytes one channel read may hold. Reuses the assembler's budget detection."""
    from . import datacube
    nbytes, src = datacube.budget_bytes(eff)
    log.debug("extract: read budget %.1f GB (%s)", nbytes / 1024**3, src)
    return float(nbytes)


def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Extract every configured channel at every point -> one table."""
    channels = eff["channels"]
    if not channels:
        raise SystemExit(
            "extract: no channels configured. Add `extract.channels` to the config naming "
            "the cube channels to pull (e.g. `eco_sst_clean: {radius_m: 300, stat: nanmean}`).")
    if not eff["points"]:
        raise SystemExit(
            "extract: no points file. Set `extract.points` in the config or pass --points.")

    out_dir, stem, fmt = eff["out_dir"], eff["stem"], eff["format"]
    names = select_aois(grids, only_aoi)     # a typo'd --aoi raises before any cube is opened
    out_path = out_dir / f"{stem}.{fmt}"

    if out_path.exists() and not eff["overwrite"] and not dry_run:
        log.info("=== %s already exists, skipping (use --overwrite to replace) ===",
                 out_path.name)
        rep = report.ProductReport("extract")
        rep.skip()
        rep.log_summary()
        return rep

    pts = points.read_points(eff["points"], eff["columns"])
    assigned = points.assign_aois(pts, grids)
    if not assigned.empty:
        dropped = assigned[~assigned["aoi"].isin(names)]
        if len(dropped):
            log.warning("  %d point(s) belong to AoI(s) this run did not select (%s); "
                        "they are not in the output.", len(dropped),
                        ", ".join(sorted(set(dropped["aoi"]))))
        assigned = assigned[assigned["aoi"].isin(names)]
    if assigned.empty:
        # An empty table is indistinguishable from a mistyped coordinate column or a wrong
        # AoI selection, so this is a failure rather than a zero-row file.
        raise SystemExit(
            f"extract: none of the {len(pts)} point(s) fall inside the selected AoI grid(s) "
            f"({', '.join(names)}). Check the points file's coordinates and `--aoi`.")

    max_radius = max((c.radius_m for c in channels.values()), default=0.0)
    points.flag_edge_points(assigned, grids, max_radius)

    rep = report.ProductReport("extract")
    budget = _budget_bytes(eff)
    frames = []

    for name in names:
        sub = assigned[assigned["aoi"] == name]
        if sub.empty:
            log.info("=== %s: no points inside this AoI, skipping ===", name)
            continue
        zpath = eff["cube_dir"] / f"{name}.zarr"
        if not zpath.exists():
            log.warning("=== %s: no assembled cube at %s; run `assemble` first -- skipping ===",
                        name, zpath.name)
            rep.skip()
            continue

        with xr.open_zarr(zpath) as ds:
            cube_g = grid_from_cube(ds, grids[name])
            if dry_run:
                plan = plan_channels(ds, channels)
                nt = ds.sizes.get("time", 0)
                rows = sum(len(sub) * (nt if dims != T2 else 1)
                           * (len(spec.stat) if dims != T1 else 1)
                           for _n, spec, dims in plan)
                log.info("=== %s: %d point(s) x %d channel(s) -> ~%d rows (dry run) ===",
                         name, len(sub), len(plan), rows)
                for cname, spec, dims in plan:
                    log.info("  %-28s %-12s radius=%-6.0f stat=%s",
                             cname, "/".join(dims), spec.radius_m, ",".join(spec.stat))
                continue
            # Re-place the points on the CUBE's grid. A no-op when it matches the config's
            # (grid_from_cube has already refused the cases where it does not), and it means
            # exactly one authority decides which pixel is read.
            placed = points.assign_aois(sub[["point_id", "lat", "lon"]], {name: cube_g})
            df = extract_aoi(ds, cube_g, placed, channels, budget)
            log.info("  %s: %d point(s) -> %d rows", name, len(sub), len(df))
            frames.append(df)
            rep.wrote()

    if dry_run:
        log.info("extract: dry run, nothing written (would write %s)", out_path)
        rep.log_summary()
        return rep
    if not frames:
        raise SystemExit("extract: no cube produced any rows; run `assemble` first.")

    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    df = df.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    dup = df.duplicated(subset=KEY)
    if dup.any():
        raise RuntimeError(
            f"extract: {int(dup.sum())} duplicate row(s) for the same {tuple(KEY)}; this is "
            f"a bug. First: {df[dup].head(1).to_dict('records')}")
    for c in ("point_id", "aoi", "variable", "stat"):
        df[c] = df[c].astype("category")

    if fmt == "csv" and len(df) > CSV_ROW_WARN:
        log.warning("  %d rows as CSV is a large text file; parquet is roughly a tenth of "
                    "the size and keeps the dtypes (`--format parquet`).", len(df))

    path = store.write_table(df, out_dir, stem, fmt)
    log.info("extract: wrote %d rows to %s", len(df), path)
    rep.log_summary()
    return rep


def extract(project: Project, *, grids=None, aois=None, points_file=None, out=None,
            fmt=None, dry_run=False, overwrite=False) -> report.ProductReport | None:
    """Extract long-format point time series from the assembled cubes.

    Terminal stage: reads only the assembled `<aoi>.zarr` cubes, writes one table, and never
    modifies a cube. Nothing calls it implicitly -- it runs when someone asks for it.
    """
    eff = _build_eff(project)
    if points_file is not None:
        eff["points"] = Path(points_file)
    if fmt is not None:
        eff["format"] = fmt
    if overwrite:
        eff["overwrite"] = True
    if out is not None:
        out = Path(out)
        eff["out_dir"], eff["stem"] = out.parent, out.stem
        if out.suffix:
            eff["format"] = out.suffix.lstrip(".")
    elif aois:
        # Without this, `extract --aoi one` would overwrite the complete table from a full
        # run, at the same path, with a one-AoI subset and no error anywhere.
        eff["stem"] = f"{eff['stem']}_" + "_".join(sorted(aois))
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(extract, "coastal_sst_data point extraction.")


if __name__ == "__main__":
    main()
