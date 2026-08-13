#!/usr/bin/env python3
"""
coastal_sst_data -- post-assembly preprocessing (runs AFTER the assembler, into the SAME cube).

The assembler ships RAW ingredients on a common grid + daily axis (see `processes.datacube`):
masking, water-filling, and multi-input derivations are DOWNSTREAM modelling determinations,
so assembly does not bake them in. This stage is that downstream layer, given a structured,
config-driven home. It opens the assembled

    <output_dir>/<datacube.output_subdir>/<aoi>.zarr

adds its derived channels, and rewrites THAT SAME store atomically -- one cube per AoI holding
raw and derived side by side, so a consumer never has to join two stores.

RAW CHANNELS ARE NEVER OVERWRITTEN. Every derived channel gets its own name (`_gapfilled`,
`_clean`, `_water_elev`, `_georef_*`, ...), so `eco_sst_v002` still holds the values the
sensor delivered while `eco_sst_v002_clean` holds the filtered product. That is what keeps the
stage IDEMPOTENT: each step seeds from the raw channel and rewrites its own outputs, so
re-running with new thresholds needs no re-assembly and never composes onto last run's drops.

Like the acquisition stages it is a thin loop over a REGISTRY: each preprocessing STEP declares
what cube channels it reads/writes and a compute function, and adding a new step is one
`PreprocessStep` registration -- exactly mirroring the datacube's contributor protocol
(`datacube.CONTRIBUTORS`), but reading an OPENED xarray cube rather than aligned files. A step
must not emit a channel the assembler wrote; see `processes.channels`.

Steps shipped today (both re-introduce computations the raw-output refactor deliberately
pushed downstream -- D6/D7/D12):

  * water_line   -- the tide-adjusted waterline per thermal sensor, from that sensor's
                    overpass tide + the DEM. Emits `<sensor>_water_elev` (m, +above/-below the
                    waterline) and `<sensor>_water_class` (submerged/exposed/unknown). Pure
                    glue over `processes.water_level`.
  * fill_water   -- nearest-neighbour fill of the level-4 SST products' (MUR, CMEMS) NaN gaps
                    over water (`landcover_water==1`) into `<channel>_gapfilled`, with a
                    `<channel>_filled` companion mask so an invented value stays
                    distinguishable from an observed one.
  * filter_clouds -- screen ECOSTRESS pixels against a gap-free baseline L4 SST (MUR/CMEMS):
                    a fixed cold offset (`baseline - eco > threshold_k`) or a distribution-based,
                    seasonally-aware outlier floor (`eco < mean - n_sigma*sigma`). Folds drops
                    into `<sensor>_sst_<ver>_clean`/`_valid_<ver>_clean` + a `*_cloudfiltered` flag.
  * filter_cloud_cover -- gate ECOSTRESS on the met total-cloud-cover field (HRRR/ERA5, percent):
                    a scene-level rejection and a per-pixel cutoff, with a `*_metcloudfiltered`
                    flag and a `<sensor>_scene_cloud_pct_<src>` diagnostic. Composes with
                    filter_clouds (both read the WORKING `_clean` product, so drops union).
  * filter_land_clouds -- screen OVER-LAND pixels against the near-surface air temperature
                    (HRRR/ERA5): drop where `airtemp - sst > threshold_k` on cells the landcover
                    mask or the `water_line` step's tide-adjusted water line marks as land. Folds
                    into the same `_clean` product + a `*_landcloudfiltered` flag.

Usage:
    coastal-sst-data run --config config.yaml --assemble --preprocess
    coastal-sst-data preprocess --config config.yaml --aoi hood_canal
    python -m coastal_sst_data.processes.preprocess --config config.yaml
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import get_close_matches
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Project, resolve_step_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, products, provenance, report, store
from . import datacube, water_level
from .channels import GAPFILLED, is_derived
from .cloud_filter import (_step_filter_clouds, _step_filter_cloud_cover,
                           _step_filter_land_clouds,
                           sigma_passes, sigma_accumulate_step, sigma_reduce_step)
from .georef import _step_flag_georef, _step_correct_georef
from .datacube import build_encoding, write_zarr
from .water_level import EXPOSED, SUBMERGED, UNKNOWN

log = logging.getLogger(__name__)

T3 = ("time", "y", "x")


# --------------------------------------------------------------------------- #
# Nearest-neighbour water fill (restored verbatim from the pre-raw-output cube,
# commit 3d99ec5^ -- the same distance_transform_edt technique as insitu.station_pixels).
# --------------------------------------------------------------------------- #
def fill_water_nn(arr, water):
    """Nearest-neighbour fill of NaNs over `water` pixels, per time slice.

    For each day, water pixels with no value take the nearest finite value (typically just-
    offshore open water). Land / non-water NaNs are left as-is. `arr` is (T,H,W); `water` is
    (H,W) bool.
    """
    from scipy.ndimage import distance_transform_edt
    out = arr.copy()
    for t in range(out.shape[0]):
        m = out[t]
        finite = np.isfinite(m)
        need = (~finite) & water
        if need.any() and finite.any():
            idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)
            nn = m[tuple(idx)]
            m[need] = nn[need]
            out[t] = m
    return out


# --------------------------------------------------------------------------- #
# The step contract
#
# A PreprocessStep is the post-assembly analog of a datacube Contributor: a declarative
# spec + a `(ctx) -> None` compute function. `reads`/`writes` name cube-channel families
# (documentation + a hook for a future writer->reader ordering edge); `depends_on` is the
# primary ordering mechanism (mirrors ProductSpec.depends_on / pipeline.process_order).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowStat:
    """A whole-window statistic a step needs before it can be evaluated on any single block.

    Every other step is DAY-LOCAL: hand it a block and it gives the same answer it would have
    given for those days inside the whole cube. A step that REDUCES over the time axis cannot --
    a climatology fitted to one block is a different climatology. So the orchestrator streams
    the blocks past this first, accumulating, and the step's own `fn` then reads the finished
    statistic out of `ctx.window`.

    `passes(ctx)` is asked ONCE, on the zero-day census context, how many prepasses this step's
    CONFIG needs -- 0 when it is configured day-locally, so an `offset` cloud filter pays
    nothing at all. `accumulate(ctx, i)` runs on every block of pass `i`; `reduce(ctx, i)` runs
    once after that pass, turning accumulators into the statistic.

    More than one pass exists because a residual needs the fit it is a residual ABOUT: pass 0
    accumulates the normal equations and solves them, pass 1 accumulates the squared residuals.
    """
    passes: "Callable[[PreprocessContext], int]"
    accumulate: "Callable[[PreprocessContext, int], None]"
    reduce: "Callable[[PreprocessContext, int], None]"


@dataclass(frozen=True)
class PreprocessStep:
    key: str                                       # step name; also the config selector key
    reads: tuple[str, ...]                         # cube channel families it consumes
    writes: tuple[str, ...]                        # cube channel families it emits
    fn: "Callable[[PreprocessContext], None]"
    depends_on: tuple[str, ...] = ()               # step keys that must run first
    option_keys: frozenset[str] = frozenset()      # per-step config options it reads
    region_option_keys: frozenset[str] = frozenset()   # subset a region may override (<= option_keys)
    provenance_inputs: tuple[str, ...] = ()        # products a derived channel attributes to
    window: WindowStat | None = None               # set only by a step that reduces over time


@dataclass
class PreprocessContext:
    """The shared state a step reads and writes while one AoI's derived channels are built.

    A step reads channels off the OPENED cube (`read`, `has`, `base_channels`, `sensor_hours`)
    and emits derived channels (`emit`). It never mutates `ds_cube`. The orchestrator merges
    `channels` back onto the cube once every step has run.

    `ds_cube` may already hold the derived channels of an EARLIER preprocess run -- the stage
    rewrites the cube in place. So a step scans for its inputs with `base_channels`, which hides
    them, and seeds its outputs from the raw channel rather than from whatever is on disk.

    TIME BLOCKS. A cube too large to hold at once is preprocessed a BLOCK of days at a time, so
    `ds_cube` may be one time slice of the store and `days` its days. Because `read` reads off
    `ds_cube`, a DAY-LOCAL step needs no changes at all for this -- it asks for a channel and
    gets that block's days. Two rules cover the rest:

      * `days` / `ds_cube` are THIS BLOCK. Emit on that axis.
      * `all_days` / `ds_all` are the WHOLE CUBE. Use them for any question whose answer is a
        property of the cube rather than of the block -- `_harmonic_design`'s "is the window
        long enough to fit a seasonal cycle" is the worked example, and asked of a block it
        answers "no" every time.

    A step that must REDUCE over the time axis cannot be answered from one block at all; it
    declares a `WindowStat` on its registration, and reads the finished statistic out of
    `window` (see `WindowStat` and `_run_window_stats`).

    A step's CHANNEL SET must be a function of the cube's channels, never of `days` -- a channel
    emitted for some blocks and not others builds a store whose variables disagree about the
    length of the time axis, which writes cleanly and then cannot be opened.

    `cache` is shared across an AoI's blocks: use it for anything derived from the grid or the
    tree rather than from the days (georef's distance transform, the tide series).
    """
    g: AoiGrid
    eff: dict
    days: pd.DatetimeIndex
    aid: str
    H: int
    W: int
    ds_cube: xr.Dataset
    channels: dict[str, tuple] = field(default_factory=dict)
    var_attrs: dict[str, dict] = field(default_factory=dict)
    global_attrs: dict[str, Any] = field(default_factory=dict)
    all_days: pd.DatetimeIndex | None = None      # the whole cube's axis; defaults to `days`
    ds_all: xr.Dataset | None = None              # the whole cube; defaults to `ds_cube`
    window: dict = field(default_factory=dict)    # whole-window statistics, by WindowStat key
    cache: dict = field(default_factory=dict)     # per-AoI scratch, shared across blocks
    # {name: (dims, dtype)} for every channel READ through `read` -- collected during the census
    # so the memory model can count what a step materialises, not just what it emits.
    reads_seen: dict = field(default_factory=dict)
    _read_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.all_days is None:
            self.all_days = self.days
        if self.ds_all is None:
            self.ds_all = self.ds_cube

    # ---- reading the cube (defensive: a missing/misshaped channel -> None) ----------- #
    def has(self, name: str) -> bool:
        return name in self.ds_cube.variables

    def read(self, name: str, *, dims=None, dtype="float32") -> np.ndarray | None:
        """A raw-cube channel as a plain array, or None if absent / wrong-shaped.

        Mirrors the datacube loaders' discipline: a step must degrade (skip, or emit
        all-UNKNOWN) when an input product was never selected, not crash the whole stage.

        Memoised for the life of one block. Several steps read the same channel independently
        (`_working`, `_target`, `_baseline_field` all call this), and each call used to
        decompress and materialise the whole array again -- a hidden multiplier on the peak
        that the block size is supposed to bound. The cached array is made READ-ONLY so a
        future in-place mutation fails loudly here rather than corrupting a later reader; every
        caller today copies before it writes.
        """
        key = (name, tuple(dims) if dims is not None else None, dtype)
        if key in self._read_cache:
            return self._read_cache[key]
        if name not in self.ds_cube.variables:
            return None
        da = self.ds_cube[name]
        if dims is not None and tuple(da.dims) != tuple(dims):
            return None
        arr = np.asarray(da.values).astype(dtype, copy=False)
        arr.setflags(write=False)
        self._read_cache[key] = arr
        self.reads_seen[name] = (tuple(da.dims), np.dtype(dtype))
        return arr

    def read_all(self, name: str, *, dims=("time",), dtype="float32") -> np.ndarray | None:
        """A channel over the WHOLE cube, for a whole-window decision.

        Only 1-D (`time`,) reads are legal: pulling a (time,y,x) channel through here would
        materialise the full cube and undo the blocking this exists to support.
        """
        if tuple(dims) != ("time",):
            raise ValueError("read_all is for 1-D (time,) channels only; a (t,y,x) read over "
                             f"the whole window would defeat blocking (asked for {name!r})")
        if self.ds_all is None or name not in self.ds_all.variables:
            return None
        da = self.ds_all[name]
        if tuple(da.dims) != tuple(dims):
            return None
        return np.asarray(da.values).astype(dtype, copy=False)

    def channels_with_prefix(self, prefix: str) -> list[str]:
        """EVERY cube channel starting with `prefix`, derived ones included.

        Almost every caller wants `base_channels` instead -- see there. This stays for the
        scans whose prefix cannot reach a derived name (the met fields).
        """
        return sorted(str(v) for v in self.ds_cube.data_vars if str(v).startswith(prefix))

    def base_channels(self, prefix: str) -> list[str]:
        """The channels starting with `prefix` that the ASSEMBLER wrote, newest run's derived
        channels excluded (`processes.channels.DERIVED_SUFFIXES`).

        This is what a step scans to discover its inputs. The cube it reads may already hold a
        previous preprocess run's output, and `<pre>_sst<ver>_clean` starts with `<pre>_sst`
        just as `<pre>_sst<ver>` does -- so a plain prefix scan would filter last run's product
        a second time and emit `<pre>_sst<ver>_clean_clean`.
        """
        return [n for n in self.channels_with_prefix(prefix) if not is_derived(n)]

    def sensor_hours(self, prefix: str) -> np.ndarray | None:
        """Per-day fractional overpass hour for a sensor, or None if the sensor is absent.

        A stacked-data sensor (ECOSTRESS) has no single `<pre>_hour`, only `<pre>_hour_<ver>`;
        this coalesces `<pre>_hour` and every `<pre>_hour_<ver>`, taking the first finite value
        per day (the sensor's versions describe the same physical overpasses). None means the
        sensor emitted no hour channel at all -- i.e. it was not acquired -- so the caller
        emits nothing for it rather than an all-UNKNOWN channel.
        """
        names = sorted(str(v) for v in self.ds_cube.data_vars
                       if (str(v) == f"{prefix}_hour" or str(v).startswith(f"{prefix}_hour_"))
                       and self.ds_cube[v].dims == ("time",))
        if not names:
            return None
        out = np.full(len(self.days), np.nan, "float32")
        for n in names:
            arr = np.asarray(self.ds_cube[n].values, "float32")
            take = np.isnan(out) & np.isfinite(arr)
            out[take] = arr[take]
        return out

    def elevation_source(self) -> str | None:
        """The DEM tag to use when `dem_source` is unset: the sole `elevation_<dem>` present."""
        tags = sorted(str(v)[len("elevation_"):] for v in self.ds_cube.data_vars
                      if str(v).startswith("elevation_"))
        if not tags:
            return None
        if len(tags) > 1:
            log.info("  water_line: %d DEM sources present %s; using %r "
                     "(set water_line.dem_source to choose)", len(tags), tags, tags[0])
        return tags[0]

    def tide_source(self) -> str | None:
        """The tide tag to use when `tide_source` is unset: the sole daily `tide_<src>`."""
        tags = sorted(str(v)[len("tide_"):] for v in self.ds_cube.data_vars
                      if str(v).startswith("tide_") and not str(v).startswith("tide_range_"))
        if not tags:
            return None
        if len(tags) > 1:
            log.info("  water_line: %d tide sources present %s; using %r "
                     "(set water_line.tide_source to choose)", len(tags), tags, tags[0])
        return tags[0]

    def aligned_dir(self, product: str, source: str | None = None) -> Path:
        """The `<DIR>[/<source>]/aligned/<aoi>` folder a product's acquisition stage wrote
        (for reaching the full-resolution tide series the cube reduced to daily statistics)."""
        return (self.eff["aligned_root"]
                / products.aligned_rel(datacube.PRODUCT_DIRS[product], source) / self.aid)

    def step_opts(self, key: str) -> dict:
        """This AoI's options for step `key`: region overrides over the global bag.

        Resolved per-AoI (via `self.aid`) so a `regions[].preprocess_steps.<key>` override lands
        for the AoIs in that region only -- the step analog of how products resolve `resolve_opts`.
        """
        return resolve_step_opts(self.eff["project"], self.aid, key)

    # ---- writing the derived channels -------------------------------------------------- #
    def emit(self, name: str, dims, arr, **attrs) -> None:
        self.channels[name] = (dims, arr)
        if attrs:
            self.var_attrs.setdefault(name, {}).update(attrs)

    # (There is no `carry`. When the derived channels lived in a cube of their own, each step
    # copied its raw inputs across so that cube was legible standalone. They now land in the
    # cube those inputs came from, so a copy would just be a second name for the same array.)


# --------------------------------------------------------------------------- #
# Step: water_line
# --------------------------------------------------------------------------- #
def _step_water_line(ctx: PreprocessContext) -> None:
    """Per-sensor tide-adjusted waterline, at each sensor's own overpass.

    Pure glue over `processes.water_level` (the reference math the raw-output refactor left in
    place for exactly this): subtract the DEM's DEM->MSL datum offset to put the ground on MSL,
    take that sensor's overpass tide, and re-reference each cell to the tide-adjusted waterline.
    Emits `<sensor>_water_elev` and `<sensor>_water_class`, beside the DEM elevation, the
    overpass-hour channel(s) and the overpass-tide channel it read them from.
    """
    opts = ctx.step_opts("water_line")
    dem_source = opts.get("dem_source") or ctx.elevation_source()
    tide_source = opts.get("tide_source") or ctx.tide_source()
    sensors = opts.get("sensors") or [sp.sensor.prefix for sp in products.sensors()]
    if isinstance(sensors, str):
        sensors = [sensors]

    if dem_source is None:
        log.warning("  water_line: no elevation_<dem> channel in the cube; skipping "
                    "(bathymetry not assembled?)")
        return
    elev = ctx.read(f"elevation_{dem_source}", dims=("y", "x"))
    if elev is None:
        log.warning("  water_line: no usable elevation_%s channel; skipping", dem_source)
        return
    datum = float(ctx.ds_cube[f"elevation_{dem_source}"].attrs.get(
        "datum_offset_m", water_level.DEFAULT_DATUM_OFFSET_M))

    for s in sensors:
        hours = ctx.sensor_hours(s)
        if hours is None:
            continue                         # sensor not acquired -> no all-UNKNOWN channel
        tide = _overpass_tide(ctx, s, tide_source, hours)
        water_elev, water_class = water_level.water_level_fields(
            elev, tide, datum_offset_m=datum)
        ctx.emit(f"{s}_water_elev", T3, water_elev, units="m",
                 long_name=(f"{s} ground elevation relative to the overpass tide-adjusted "
                            "waterline (0 at it, + exposed, - submerged)"),
                 datum_source=dem_source, tide_source=str(tide_source))
        ctx.emit(f"{s}_water_class", T3, water_class,
                 long_name=f"{s} submerged/exposed at its overpass",
                 flag_values=[SUBMERGED, EXPOSED, UNKNOWN],
                 flag_meanings="submerged exposed unknown")


def _overpass_tide(ctx: PreprocessContext, sensor: str, src, hours) -> np.ndarray:
    """The tide (m, rel. MSL) at `sensor`'s overpass, per day. Prefer the cube's ready-made
    `<sensor>_tide_<src>` (the tide_overpass channel); else interpolate the full-resolution
    tide series on disk to the overpass hour; else NaN (-> UNKNOWN classes)."""
    if src is not None and ctx.has(f"{sensor}_tide_{src}"):
        got = ctx.read(f"{sensor}_tide_{src}", dims=("time",))
        if got is not None:
            return got
    if src is not None:
        series = water_level.load_tide_series(ctx.aligned_dir("tides", src), ctx.aid)
        if series is not None:
            return water_level.tide_at_overpass(series, ctx.days, hours)
    return np.full(len(ctx.days), np.nan, "float32")


# --------------------------------------------------------------------------- #
# Step: fill_water
# --------------------------------------------------------------------------- #
def _step_fill_water(ctx: PreprocessContext) -> None:
    """Nearest-neighbour fill of the level-4 SST products over water.

    A cell is fillable where the mask channel (default `landcover_water`) is water and the
    source channel is NaN -- the model's ~9 km land mask can swallow an estuary, and MUR ships
    honest NaN gaps. Emits `<channel>_gapfilled` -- the filled field, BESIDE the untouched
    source channel -- plus a `<channel>_filled` uint8 mask (1 = invented over unobserved water,
    0 = observed), so a filled value never passes for an observed one. Filling everywhere would
    fabricate data over land the source legitimately never covered, so an absent mask means:
    fill nothing.
    """
    opts = ctx.step_opts("fill_water")
    mask_channel = opts.get("mask_channel", "landcover_water")
    water = ctx.read(mask_channel, dims=("y", "x"))
    if water is None:
        log.warning("  fill_water: no %s channel in the cube; nothing filled "
                    "(filling over an unknown mask would fabricate data over land)",
                    mask_channel)
        return
    water = water > 0.5

    channels = _fill_channels(ctx, opts.get("sources"))
    if not channels:
        log.warning("  fill_water: no level-4 source channels (mur_sst / cmems_*) to fill")
        return

    for c in channels:
        raw = ctx.read(c, dims=T3)
        if raw is None:
            continue
        observed = np.isfinite(raw)
        filled = fill_water_nn(raw, water)
        filled_mask = (np.isfinite(filled) & ~observed).astype("uint8")
        ctx.emit(f"{c}{GAPFILLED}", T3, filled, preprocess=f"nn_filled over {mask_channel}",
                 long_name=f"{c} (nearest-neighbour filled over water)")
        ctx.emit(f"{c}_filled", T3, filled_mask,
                 long_name=(f"{c} was nearest-neighbour filled over water the source did not "
                            "observe (1 = invented, 0 = observed)"),
                 flag_values=[0, 1], flag_meanings="observed filled")


def _fill_channels(ctx: PreprocessContext, sources) -> list[str]:
    """The level-4 channels to fill: `mur_sst` and every `cmems_*` (bar `_valid` and this
    step's own output), restricted to the `sources` product tags (`mur` / `cmems`) when given."""
    pairs: list[tuple[str, str]] = []
    if ctx.has("mur_sst"):
        pairs.append(("mur", "mur_sst"))
    for c in ctx.base_channels("cmems_"):
        if not c.endswith("_valid"):
            pairs.append(("cmems", c))
    if sources is None:
        return [c for _, c in pairs]
    if isinstance(sources, str):
        sources = [sources]
    want = set(sources)
    return [c for prod, c in pairs if prod in want]


# --------------------------------------------------------------------------- #
# The registry. Declaration order is the tie-break for the topological sort (see
# `_topo_order`); it is NOT the run order, which honours `depends_on`.
# --------------------------------------------------------------------------- #
# Shared by each cloud filter and its corrected-pass variant, so their option surfaces can't drift.
_CLOUDS_OPTS = frozenset({"method", "threshold_k", "n_sigma", "baseline", "stat_scope",
                          "seasonality", "sensors", "mask_sst", "use_cloud_raster"})
_CLOUD_COVER_OPTS = frozenset({"source", "scene_max_pct", "pixel_max_pct", "sensors",
                               "mask_sst", "water_mask_channel"})
_LAND_OPTS = frozenset({"threshold_k", "source", "land_source", "mask_channel",
                        "sensors", "mask_sst"})

STEPS: tuple[PreprocessStep, ...] = (
    PreprocessStep(
        key="water_line",
        reads=("elevation_", "tide_", "_hour"),
        writes=("_water_elev", "_water_class"),
        fn=_step_water_line,
        option_keys=frozenset({"dem_source", "tide_source", "sensors"}),
        provenance_inputs=("bathymetry", "tides"),
    ),
    PreprocessStep(
        key="fill_water",
        reads=("landcover_water", "mur_sst", "cmems_"),
        writes=("_gapfilled", "_filled"),
        fn=_step_fill_water,
        option_keys=frozenset({"sources", "mask_channel"}),
    ),
    PreprocessStep(
        key="filter_clouds",
        reads=("eco_sst", "eco_valid", "eco_cloud", "mur_sst", "cmems_", "doy_sin", "doy_cos"),
        writes=("_clean", "_cloudfiltered"),
        fn=_step_filter_clouds,
        depends_on=("fill_water",),        # so offset mode sees the gap-filled baseline
        # `method: sigma` fits a climatology over the WHOLE time axis, so it cannot be answered
        # from one block; `method: offset` declares 0 passes and pays nothing.
        window=WindowStat(
            passes=partial(sigma_passes, key="filter_clouds"),
            accumulate=partial(sigma_accumulate_step, key="filter_clouds"),
            reduce=partial(sigma_reduce_step, key="filter_clouds")),
        option_keys=_CLOUDS_OPTS,
        provenance_inputs=("ecostress", "mur"),
    ),
    PreprocessStep(
        key="filter_cloud_cover",
        reads=("eco_cloud_cover_", "cloud_cover_", "eco_sst", "eco_valid", "landcover_water"),
        writes=("_clean", "_metcloudfiltered", "_scene_cloud_pct"),
        fn=_step_filter_cloud_cover,
        depends_on=("filter_clouds",),     # deterministic order; drops compose via `_working`
        option_keys=_CLOUD_COVER_OPTS,
        provenance_inputs=("met_overpass", "met", "ecostress"),
    ),
    PreprocessStep(
        key="filter_land_clouds",
        reads=("eco_sst", "eco_valid", "eco_airtemp_", "airtemp_",
               "landcover_water", "_water_class"),
        writes=("_clean", "_landcloudfiltered"),
        fn=_step_filter_land_clouds,
        # after water_line (its `<pre>_water_class` feeds land_source=water_line) and the other
        # cloud filters (deterministic order; every filter's drops union via `_working`). A
        # depends_on to an UNSELECTED step is ignored, so none of these are pulled in implicitly.
        depends_on=("water_line", "filter_clouds", "filter_cloud_cover"),
        option_keys=_LAND_OPTS,
        provenance_inputs=("met_overpass", "met", "ecostress"),
    ),
    PreprocessStep(
        key="flag_georef",
        reads=("_sst", "landcover_water"),
        writes=("_georef_",),
        fn=_step_flag_georef,
        # the fit must read the FILTERED sst -- cloud edges are the dominant noise source, and the
        # cloud filters mutate `<pre>_sst` via `_working`. A depends_on to an unselected step is
        # ignored, so flagging still runs (on the raw sst) if no cloud filter is selected.
        depends_on=("filter_clouds", "filter_cloud_cover", "filter_land_clouds"),
        option_keys=frozenset({"sensors", "tol_m", "max_shift_m", "coarse_stride", "n_refine",
                               "sigma", "lo_pct", "hi_pct", "min_coast_obs", "min_valid_pct",
                               "min_edges", "z_min", "lift_min", "gain_min", "ok_shift_m",
                               "stability_windows_km"}),
        region_option_keys=frozenset({"min_coast_obs", "min_edges", "min_valid_pct"}),
        provenance_inputs=("ecostress", "landcover"),
    ),
    PreprocessStep(
        key="correct_georef",
        reads=("_sst", "_valid", "_georef_"),
        writes=("_georef_corrected", "_georef_applied"),
        fn=_step_correct_georef,
        depends_on=("flag_georef",),        # applies the (dy,dx) flag_georef stored this run
        option_keys=frozenset({"sensors", "fields", "fill"}),
        provenance_inputs=("ecostress", "landcover"),
    ),
    # Re-run the cloud filters on the CORRECTED geometry (the pre-fit pass compared misregistered
    # pixels). Each reuses its base filter's math + config (base_key), reads the `_georef_corrected`
    # channels and writes a SEPARATE `_georef_corrected_clean` product, composing among themselves.
    PreprocessStep(
        key="filter_clouds_corrected",
        reads=("_georef_corrected", "mur_sst", "cmems_"),
        writes=("_georef_corrected_clean", "_cloudfiltered"),
        fn=partial(_step_filter_clouds, key="filter_clouds_corrected",
                   base_key="filter_clouds", mode="corrected"),
        depends_on=("correct_georef",),
        # Same climatology as the raw pass -- the corrected geometry never touches the baseline
        # -- and `_sigma_key` is keyed on the maths, so the two share one prepass.
        window=WindowStat(
            passes=partial(sigma_passes, key="filter_clouds_corrected",
                           base_key="filter_clouds"),
            accumulate=partial(sigma_accumulate_step, key="filter_clouds_corrected",
                               base_key="filter_clouds"),
            reduce=partial(sigma_reduce_step, key="filter_clouds_corrected",
                           base_key="filter_clouds")),
        option_keys=_CLOUDS_OPTS,
        provenance_inputs=("ecostress", "landcover", "mur"),
    ),
    PreprocessStep(
        key="filter_cloud_cover_corrected",
        reads=("eco_cloud_cover_", "cloud_cover_", "_georef_corrected", "landcover_water"),
        writes=("_georef_corrected_clean", "_metcloudfiltered"),
        fn=partial(_step_filter_cloud_cover, key="filter_cloud_cover_corrected",
                   base_key="filter_cloud_cover", mode="corrected"),
        depends_on=("correct_georef", "filter_clouds_corrected"),
        option_keys=_CLOUD_COVER_OPTS,
        provenance_inputs=("met_overpass", "met", "ecostress", "landcover"),
    ),
    PreprocessStep(
        key="filter_land_clouds_corrected",
        reads=("_georef_corrected", "eco_airtemp_", "airtemp_", "landcover_water", "_water_class"),
        writes=("_georef_corrected_clean", "_landcloudfiltered"),
        fn=partial(_step_filter_land_clouds, key="filter_land_clouds_corrected",
                   base_key="filter_land_clouds", mode="corrected"),
        depends_on=("correct_georef", "filter_clouds_corrected", "filter_cloud_cover_corrected",
                    "water_line"),
        option_keys=_LAND_OPTS,
        provenance_inputs=("met_overpass", "met", "ecostress", "landcover"),
    ),
)

BY_KEY: dict[str, PreprocessStep] = {s.key: s for s in STEPS}


def _topo_order(steps: list[PreprocessStep]) -> list[PreprocessStep]:
    """Order steps so every `depends_on` runs first; stable by declaration order (mirrors
    `pipeline.process_order`). A dependency on an UNSELECTED step is ignored -- depends_on is
    an ordering hint among the steps that actually run, not a requirement to pull one in."""
    sel = {s.key for s in steps}
    order: list[PreprocessStep] = []
    placed: set[str] = set()
    remaining = list(steps)
    while remaining:
        ready = [s for s in remaining
                 if all(d in placed for d in s.depends_on if d in sel)]
        if not ready:
            raise RuntimeError(
                f"preprocess step dependency cycle among {[s.key for s in remaining]}")
        nxt = ready[0]
        order.append(nxt)
        placed.add(nxt.key)
        remaining.remove(nxt)
    return order


# --------------------------------------------------------------------------- #
# Import-time invariants (mirror products._check_registry / datacube._check_contributors)
# --------------------------------------------------------------------------- #
def _check_steps() -> None:
    keys = [s.key for s in STEPS]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"preprocess step keys must be unique; got {sorted(keys)}")
    known = set(keys)
    for s in STEPS:
        for d in s.depends_on:
            if d not in known:
                raise RuntimeError(
                    f"preprocess step {s.key!r}: depends_on unknown step {d!r}.")
        if not s.region_option_keys <= s.option_keys:
            raise RuntimeError(
                f"preprocess step {s.key!r}: region_option_keys "
                f"{sorted(s.region_option_keys - s.option_keys)} are not in option_keys.")
    _topo_order(list(STEPS))            # raises on a dependency cycle


_check_steps()


def _check_step_options(eff: dict) -> None:
    """Reject a selected step / per-step option the registry does not recognise.

    Deferred here from config validation (config cannot import the step registry without a
    cycle), but it is the same rule and the same `did you mean` hint as
    `config._option_keys_are_known`: a key nothing reads is a config that LIES.
    """
    problems: list[str] = []
    for key, opts in eff["steps"].items():
        if key not in BY_KEY:
            hint = get_close_matches(key, sorted(BY_KEY), n=1, cutoff=0.6)
            suggest = f" (did you mean {hint[0]!r}?)" if hint else ""
            problems.append(f"preprocess.steps.{key} is not a known step{suggest}. "
                            f"Valid: {', '.join(sorted(BY_KEY))}")
            continue
        allowed = BY_KEY[key].option_keys
        for opt_key in sorted(set(opts) - set(allowed)):
            hint = get_close_matches(opt_key, sorted(allowed), n=1, cutoff=0.6)
            suggest = f" (did you mean {hint[0]!r}?)" if hint else ""
            problems.append(f"preprocess.steps.{key}.{opt_key} is not a recognised option"
                            f"{suggest}. Valid: {', '.join(sorted(allowed)) or '(none)'}")

    # Region overrides (regions[].preprocess_steps.<key>): a known step, an option the step
    # reads, AND one it declares region-overridable -- a region tunes coverage/thresholds, it
    # cannot silently redefine a channel's meaning (the product `region_options` rule for steps).
    for region in eff["project"].regions:
        for key, step_opts in region.preprocess_steps.items():
            opts = dict(step_opts.model_extra or {})
            where = f"regions.{region.name}.preprocess_steps.{key}"
            if key not in BY_KEY:
                hint = get_close_matches(key, sorted(BY_KEY), n=1, cutoff=0.6)
                suggest = f" (did you mean {hint[0]!r}?)" if hint else ""
                problems.append(f"{where} is not a known step{suggest}. "
                                f"Valid: {', '.join(sorted(BY_KEY))}")
                continue
            step = BY_KEY[key]
            for opt_key in sorted(set(opts) - set(step.option_keys)):
                hint = get_close_matches(opt_key, sorted(step.option_keys), n=1, cutoff=0.6)
                suggest = f" (did you mean {hint[0]!r}?)" if hint else ""
                problems.append(f"{where}.{opt_key} is not a recognised option{suggest}. "
                                f"Valid: {', '.join(sorted(step.option_keys)) or '(none)'}")
            for opt_key in sorted((set(opts) & set(step.option_keys)) - set(step.region_option_keys)):
                problems.append(
                    f"{where}.{opt_key} is not region-overridable (it would be SILENTLY IGNORED "
                    f"per AoI). Region-overridable options for {key}: "
                    f"{', '.join(sorted(step.region_option_keys)) or '(none)'}")
    if problems:
        raise ValueError(
            "unrecognised preprocess option(s) -- these would be SILENTLY IGNORED:\n  "
            + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Build one AoI's cube: the assembled channels + this stage's derived ones
# --------------------------------------------------------------------------- #
def _new_ctx(ds_cube: xr.Dataset, g: AoiGrid, eff: dict, days, **kw) -> PreprocessContext:
    return PreprocessContext(
        g=g, eff=eff, days=days, aid=g.name,
        H=int(ds_cube.sizes["y"]), W=int(ds_cube.sizes["x"]), ds_cube=ds_cube, **kw)


def selected_steps(eff: dict) -> list["PreprocessStep"]:
    """The selected steps in run order (registry order, then topologically sorted)."""
    return _topo_order([s for s in STEPS if s.key in eff["steps"]])


def preprocess_census(ds_cube: xr.Dataset, g: AoiGrid, eff: dict, all_days,
                      *, window=None, cache=None) -> tuple[dict, dict]:
    """({emitted: (dims, dtype)}, {read: (dims, dtype)}) for this cube and step selection.

    Runs the REAL steps over a ZERO-LENGTH time slice, so every channel is named by the code
    that emits it -- there is no second list to drift -- and nothing is materialised: each array
    is (0,H,W). This is legitimate precisely because a step's channel SET is a function of the
    cube's channels, never of the days (`_channel_sets`, `base_channels` and `_fill_channels`
    all scan variable NAMES).

    The reads are collected too, by instrumenting `ctx.read`: they are what a block actually
    materialises, and the memory model needs them as much as it needs the emissions.

    Stops after the step loop -- deliberately. `preprocess_aoi` goes on to collect provenance
    over the whole aligned tree, which is the most expensive call in the stage and has nothing
    to say about which channels exist.
    """
    ctx = _new_ctx(ds_cube.isel(time=slice(0, 0)), g, eff, all_days[:0],
                   all_days=all_days, ds_all=ds_cube,
                   window={} if window is None else window,
                   cache={} if cache is None else cache)
    with datacube._quiet(logging.getLogger(__package__.split(".")[0])):
        for step in selected_steps(eff):
            step.fn(ctx)
    emitted = {n: (dims, np.asarray(arr).dtype) for n, (dims, arr) in ctx.channels.items()}
    return emitted, dict(ctx.reads_seen)


def resolve_channel_plan(ds_cube: xr.Dataset, census: dict) -> tuple[list[str], dict]:
    """(stale, expected) for this rewrite. Raises if a step would clobber an assembled channel.

    Both are whole-cube facts, and both used to be answered after the steps had run over the
    whole cube. A blocked rewrite has no such moment: block 0 must ALREADY know which variables
    to drop, because a variable dropped from only some blocks is either never extended or
    created at one block's length -- and the cube then fails to open.
    """
    # What the LAST run added, so this one can tell its own output from the assembler's. Without
    # it a re-run cannot distinguish "replacing my `eco_georef_flag`" from "clobbering a channel
    # the assembler wrote", since neither name carries a derived suffix.
    was_derived = set(json.loads(ds_cube.attrs.get("preprocess_channels", "[]")))

    clobbered = sorted(n for n in census if n in ds_cube.variables and n not in was_derived)
    if clobbered:
        # An import-time check can't catch this: what a step emits depends on the channels the
        # cube happens to hold. Failing here beats silently overwriting the sensor's own data.
        raise RuntimeError(
            f"preprocess would overwrite assembled channel(s) {clobbered} -- a step must emit "
            f"under a name of its own (see processes.channels) so the raw values survive.")

    # Channels the last run added that this step selection no longer produces: drop them, or the
    # cube keeps shipping output its own `preprocess` attr no longer claims.
    stale = sorted(n for n in was_derived - set(census) if n in ds_cube.variables)
    if stale:
        log.info("  dropping %d channel(s) from a previous step selection: %s",
                 len(stale), ", ".join(stale))
    expected = {n: (ds_cube[n].dims, ds_cube[n].dtype)
                for n in ds_cube.data_vars if n not in stale}
    expected.update(census)                 # a re-emitted derived channel: the census wins
    return stale, expected


def preprocess_block(ds_cube: xr.Dataset, g: AoiGrid, eff: dict, *, all_days=None, ds_all=None,
                     stale=(), window=None, cache=None) -> xr.Dataset:
    """One span of days, with the selected steps' derived channels merged onto it.

    Every selected step runs through the uniform `(ctx) -> None` protocol in `depends_on`
    order, then its emissions are assigned onto the cube it read them from -- on the cube's own
    coords, never re-derived, so nothing can drift out of alignment.

    `ds_cube` may be one time BLOCK of the store; `all_days`/`ds_all` are then the whole cube,
    for the steps that must ask a whole-window question. The whole-cube attrs are NOT stamped
    here -- see `finalize_preprocess_attrs`.
    """
    days = pd.DatetimeIndex(ds_cube["time"].values)
    ctx = _new_ctx(ds_cube, g, eff, days, all_days=all_days, ds_all=ds_all,
                   window={} if window is None else window,
                   cache={} if cache is None else cache)
    for step in selected_steps(eff):
        step.fn(ctx)

    ds_out = ds_cube.drop_vars(list(stale)).assign(
        {n: (dims, arr) for n, (dims, arr) in ctx.channels.items()})
    ds_out.attrs.update(ctx.global_attrs)
    for name, attrs in ctx.var_attrs.items():
        if name in ds_out:
            ds_out[name].attrs.update(attrs)
    return ds_out


def preprocess_aoi(ds_cube: xr.Dataset, g: AoiGrid, eff: dict) -> xr.Dataset:
    """The assembled cube for one AoI WITH the selected steps' derived channels merged in.

    The whole cube in one pass: what `run` still does for a cube that fits in memory. A
    derived channel never takes an assembled channel's name (`processes.channels`), so this
    only ever ADDS variables; re-running replaces the previous run's derived channels in place.
    `preprocess_steps_stale` is what decides whether that re-run is worth doing.
    """
    all_days = pd.DatetimeIndex(ds_cube["time"].values)
    census, _reads = preprocess_census(ds_cube, g, eff, all_days)
    stale, _expected = resolve_channel_plan(ds_cube, census)
    ds_out = preprocess_block(ds_cube, g, eff, all_days=all_days, stale=stale)
    ds_out.attrs.update(finalize_preprocess_attrs(
        dict(ds_cube.attrs), {}, eff, g, list(ds_out.data_vars), census))
    return ds_out


def finalize_preprocess_attrs(src_attrs: dict, block_attrs: dict, eff: dict, g: AoiGrid,
                              fields, census: dict) -> dict:
    """Every global attr the rewritten cube must carry.

    Split out because a blocked rewrite has no in-memory whole cube to hang them on -- and
    because appending to a Zarr store REPLACES the group's attrs, so they have to be re-stamped
    once at the end whatever the path. Losing them is quiet and expensive: without
    `preprocess_channels` the NEXT run cannot tell its own output from the assembler's and
    refuses to run at all; without `preprocess`/`code_version` the stage never looks finished
    and silently redoes itself on every invocation.

    `src_attrs` is the cube's attrs as OPENED -- it carries everything the assembler stamped
    (coverage, provenance of the raw products, the in-situ station table), which this stage must
    not drop. Spread first, so this run's values win over a previous run's.
    """
    # PROVENANCE: re-stamped over the WHOLE cube, so the derived fields are recorded beside the
    # assembled ones. `created_at` stays the assembly's -- it dates the observations, which this
    # stage does not touch -- and `preprocessed_at` dates the derivation.
    prod = provenance.collect(eff["aligned_root"], g.name, datacube.PRODUCT_DIRS)
    rec = provenance.build(eff["project"], sorted(fields), prod)
    selected = selected_steps(eff)
    return {
        **src_attrs,
        **block_attrs,
        "aoi_id": g.name,
        "crs": src_attrs.get("crs", g.target_crs),
        "preprocess": json.dumps({k: eff["steps"][k] for k in (s.key for s in selected)},
                                 sort_keys=True),
        # Which channels this stage owns -- the next run reads it back, and a consumer can tell
        # a derived channel from an observed one without parsing names.
        "preprocess_channels": json.dumps(sorted(census), sort_keys=True),
        "created_at": src_attrs.get("created_at", rec["created_at"]),
        "preprocessed_at": rec["created_at"],
        "package_version": rec["package_version"], "code_version": rec["code_version"],
        "config_sha256": rec["config_sha256"] or "", "config_path": rec["config_path"] or "",
        "config_yaml": rec["config_yaml"] or "",
        "provenance": json.dumps(rec["fields"], sort_keys=True),
        "provenance_products": json.dumps(rec["products"], sort_keys=True),
    }


# --------------------------------------------------------------------------- #
# Blocked rewrite
# --------------------------------------------------------------------------- #
# Preprocess holds more per day than the assembler does at the same block size: the baseline
# climatology promotes to float64, every filter copies its target to fold drops into it, and a
# step reads channels it does not emit. So the channel arithmetic is scaled harder here.
_PP_TRANSIENT_FACTOR = 3.0


def source_time_chunk(ds_cube: xr.Dataset) -> int | None:
    """The on-disk time chunk of the store `ds_cube` was opened from, or None.

    From `encoding["chunks"]`, which the Zarr backend fills with the array's REAL chunk shape --
    not `.chunks`, which is the dask chunking `chunks="auto"` may have fused, and not
    `datacube.chunks.time`, which is the CONFIGURED value. Those differ on exactly the cubes
    this matters for: the assembler reduces the time chunk when the memory budget cannot hold
    one chunk's worth of days, and preprocess must inherit that reduction rather than quietly
    undo it by re-chunking the whole store back to the config.
    """
    seen = set()
    for name, da in ds_cube.data_vars.items():
        if "time" not in da.dims:
            continue
        ch = da.encoding.get("chunks")
        if ch:
            seen.add(int(ch[list(da.dims).index("time")]))
    if not seen:
        return None
    if len(seen) > 1:
        log.warning("  the cube's variables disagree on their time chunk (%s); taking the "
                    "largest so every append stays aligned", sorted(seen))
    return max(seen)


def _run_window_stats(eff: dict, ds_cube: xr.Dataset, g: AoiGrid, all_days, blocks,
                      *, window: dict, cache: dict) -> None:
    """Build every selected step's whole-window statistics, streaming the blocks.

    In topological order, because one step's statistic could in principle need an earlier one's.
    None does today: the sigma climatology reads the RAW baseline through `ctx.read` -- never
    fill_water's invented cells, which would collapse sigma -- so it depends on no other step's
    output, which is exactly what makes a prepass legal.
    """
    for step in selected_steps(eff):
        if step.window is None:
            continue
        probe = _new_ctx(ds_cube.isel(time=slice(0, 0)), g, eff, all_days[:0],
                         all_days=all_days, ds_all=ds_cube, window=window, cache=cache)
        n = int(step.window.passes(probe))
        if n:
            log.info("    %s: %d whole-window pass(es) over %d block(s)", step.key, n, len(blocks))
        for p in range(n):
            for sl in blocks:
                step.window.accumulate(
                    _new_ctx(ds_cube.isel(time=sl), g, eff,
                             pd.DatetimeIndex(all_days[sl]), all_days=all_days, ds_all=ds_cube,
                             window=window, cache=cache), p)
            step.window.reduce(probe, p)


def _for_write(ds_blk: xr.Dataset, eff: dict, time_chunk: int) -> xr.Dataset:
    """A block ready for `to_zarr`: the SOURCE store's encoding scrubbed, dask chunks matched.

    A block is `ds_cube.isel(time=...)`, so every channel carried over from the source still
    holds the encoding of the store it was OPENED from -- `chunks`, `preferred_chunks`, codecs,
    all measured against the OLD layout. Zarr would then be told two different chunkings for one
    array, and on an append (which passes no `encoding=`) the stale one wins. The assembler
    never had this problem: its blocks are freshly built arrays.

    The rechunk is the other half. A zarr chunk spanning more than one dask chunk makes xarray
    refuse the write outright; a dask chunk spanning the whole grid makes every `to_zarr` task
    materialise a full slab, times the thread pool -- which is how the untouched channels, the
    ones that are supposed to just stream through, would blow the budget anyway.
    """
    out = ds_blk.copy()
    for v in out.data_vars:            # data_vars only: the `time` coord's units/calendar must
        out[v].encoding = {}           # survive, or the axis is rewritten in a different epoch
    return out.chunk({"time": time_chunk,
                      "y": eff["chunks"].get("y", -1), "x": eff["chunks"].get("x", -1)})


def _preprocess_blocked(ds_cube: xr.Dataset, g: AoiGrid, eff: dict, zpath: Path, *,
                        all_days, block_days: int, time_chunk: int, census: dict,
                        expected: dict, stale: list, src_attrs: dict,
                        window: dict, cache: dict) -> None:
    """Rewrite one AoI's cube a block of days at a time, inside one atomic swap.

    The stage reads and writes the SAME path, so the atomicity matters more here than anywhere
    else in the package: a botched write destroys the assembled cube, not merely this stage's
    derived channels. `store.atomic` builds the new cube beside the old one and swaps only on a
    clean return, and the source stays open for the whole loop -- which is why the source's
    `with` must enclose this call, and the swap happen after it closes.
    """
    n = len(all_days)
    blocks = [slice(i, min(i + block_days, n)) for i in range(0, n, block_days)]
    _run_window_stats(eff, ds_cube, g, all_days, blocks, window=window, cache=cache)

    block_attrs: dict = {}
    quiet = datacube._LogOnce()
    # One filter per EMITTING logger: a Logger's filters are skipped for records propagating up
    # from a child, so filtering the package logger would catch none of these.
    loggers = [log, logging.getLogger("coastal_sst_data.processes.cloud_filter"),
               logging.getLogger("coastal_sst_data.processes.georef")]
    for lg in loggers:
        lg.addFilter(quiet)
    try:
        with store.atomic(zpath) as tmp:
            for i, sl in enumerate(blocks):
                ds_blk = preprocess_block(
                    ds_cube.isel(time=sl), g, eff, all_days=all_days, ds_all=ds_cube,
                    stale=stale, window=window, cache=cache)
                datacube._check_channel_set(
                    {k: (ds_blk[k].dims, ds_blk[k].dtype) for k in ds_blk.data_vars},
                    expected, i, pd.DatetimeIndex(all_days[sl]))
                datacube._merge_block_attrs(block_attrs, dict(ds_blk.attrs), i)
                ds_blk = _for_write(ds_blk, eff, time_chunk)
                if i == 0:
                    # Encoding settled here, against the FINISHED cube's time length -- this
                    # block's would chunk the new store in blocks.
                    write_zarr(ds_blk, tmp, build_encoding(
                        ds_blk, eff["compression"], {**eff["chunks"], "time": time_chunk},
                        sizes={"time": n}), consolidated=False)
                else:
                    datacube.append_zarr(ds_blk, tmp)
                log.info("    block %d/%d: %s..%s", i + 1, len(blocks),
                         all_days[sl][0].date(), all_days[sl][-1].date())
                del ds_blk
            fields = (set(ds_cube.data_vars) - set(stale)) | set(census)
            datacube.finalize_cube(tmp, finalize_preprocess_attrs(
                src_attrs, block_attrs, eff, g, fields, census))
    finally:
        for lg in loggers:
            lg.removeFilter(quiet)


def preprocess_steps_stale(ds_cube: xr.Dataset, eff: dict) -> bool:
    """Would re-running the selected steps on `ds_cube` change anything?

    False when the cube already carries the output of THIS step selection, built by THIS code
    version -- the stage is idempotent, so that re-run would rewrite an identical cube. A config
    or code change makes it True and the stage re-runs on its own; `overwrite` forces it.
    """
    selected = json.dumps({k: eff["steps"][k] for k in
                           (s.key for s in STEPS if s.key in eff["steps"])}, sort_keys=True)
    return (ds_cube.attrs.get("preprocess") != selected
            or ds_cube.attrs.get("code_version") != provenance.code_version())


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    pp = project.preprocess
    root = Path(project.output_dir)
    return {
        "project": project,
        "aligned_root": root,                         # per-product <DIR>/aligned/<aoi>
        "cube_dir": root / project.datacube.output_subdir,
        "steps": {k: dict(v.model_extra or {}) for k, v in pp.steps.items()},
        # One cube, ONE encoding: this stage rewrites the store the assembler wrote, so its own
        # chunking/compression would silently re-chunk the assembled channels too. Both come
        # from `datacube`, which is also what keeps the re-write chunk-aligned with what it read.
        "chunks": dict(project.datacube.chunks),
        "compression": project.datacube.compression,
        "overwrite": bool(pp.overwrite),
        # Blocking: this stage's own knobs, falling back to the assembler's. `None` (unset) is
        # distinct from an explicit "auto", so "inherit" and "size it yourself" are separable.
        "block_days": (pp.block_days if pp.block_days is not None
                       else project.datacube.block_days),
        "memory_budget_gb": (pp.memory_budget_gb if pp.memory_budget_gb is not None
                             else project.datacube.memory_budget_gb),
    }


def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Add each AoI's derived channels to its assembled cube, rewriting that cube in place."""
    _check_step_options(eff)
    if not eff["steps"]:
        log.info("preprocess: no steps selected; nothing to do.")
        return None

    cube_dir = eff["cube_dir"]
    overwrite = eff["overwrite"]
    names = select_aois(grids, only_aoi)
    rep = report.ProductReport("preprocess")

    for name in names:
        g = grids[name]
        zpath = cube_dir / f"{name}.zarr"

        if not zpath.exists():
            log.warning("=== %s: no assembled cube at %s; run `assemble` first -- skipping ===",
                        name, zpath.name)
            rep.skip()
            continue
        store.sweep_scratch(zpath)      # clear scratch from a run that died mid-write
        if not overwrite:
            with xr.open_zarr(zpath) as ds_cube:
                stale = preprocess_steps_stale(ds_cube, eff)
            if not stale:
                log.info("=== %s: %s already holds this step selection's output, skipping "
                         "(use overwrite to force) ===", name, zpath.name)
                rep.skip()
                continue
        if dry_run:
            log.info("=== %s: [dry-run] would preprocess into %s ===", name, zpath.name)
            continue

        log.info("=== preprocessing %s (steps: %s) ===",
                 name, ", ".join(sorted(eff["steps"])))
        # The cube is BOTH the input and the output. `store.atomic` gives us a scratch path to
        # build the new one in and swaps it over only on a clean return -- and the swap has to
        # happen after the source is CLOSED, which is why this drives `atomic` itself instead
        # of handing the finished dataset to `write_zarr_safe`. A run killed part-way leaves
        # the assembled cube exactly as it was.
        with xr.open_zarr(zpath) as ds_cube:
            src_attrs = dict(ds_cube.attrs)          # BEFORE anything; not re-readable later
            all_days = pd.DatetimeIndex(ds_cube["time"].values)
            H, W = int(ds_cube.sizes["y"]), int(ds_cube.sizes["x"])
            n_before = len(ds_cube.data_vars)
            window: dict = {}
            cache: dict = {}

            # What this rewrite will hold, and what it will cost per day -- both asked of the
            # real steps over a zero-length slice, so neither can drift from what they emit.
            census, reads = preprocess_census(ds_cube, g, eff, all_days,
                                              window=window, cache=cache)
            stale, expected = resolve_channel_plan(ds_cube, census)
            per_day = (datacube.bytes_per_day(census, H, W)
                       + datacube.bytes_per_day(reads, H, W))
            # The block must not outrun the store it READS: inherit the on-disk time chunk
            # rather than the configured one, which the assembler may have deliberately reduced.
            tc_src = source_time_chunk(ds_cube) or eff["chunks"].get("time", len(all_days))
            block_days, time_chunk = datacube.resolve_block_days(
                {**eff, "chunks": {**eff["chunks"], "time": tc_src}}, per_day, len(all_days),
                transient=_PP_TRANSIENT_FACTOR)
            budget, src = datacube.budget_bytes(eff)
            log.info("  %d derived + %d read channel(s): %.0f MB/day; budget %.1f GiB (%s) "
                     "-> %d block(s) of %d day(s), time chunk %d",
                     len(census), len(reads), per_day / 1e6, budget / 1024**3, src,
                     -(-len(all_days) // block_days), block_days, time_chunk)
            if time_chunk != tc_src:
                log.warning("  %s: the memory budget fits only %d day(s) per block, fewer than "
                            "the store's time chunk of %d; the rewritten cube is chunked at %d "
                            "instead. Raise preprocess.memory_budget_gb to keep the layout.",
                            name, block_days, tc_src, time_chunk)

            if block_days >= len(all_days):
                # The cube fits: rewrite it in one pass, exactly as before.
                with store.atomic(zpath) as tmp:
                    ds_out = preprocess_block(ds_cube, g, eff, all_days=all_days,
                                              stale=stale, window=window, cache=cache)
                    ds_out.attrs.update(finalize_preprocess_attrs(
                        src_attrs, {}, eff, g, list(ds_out.data_vars), census))
                    shape = (ds_out.sizes["time"], ds_out.sizes["y"], ds_out.sizes["x"])
                    n_after = len(ds_out.data_vars)
                    write_zarr(ds_out, tmp,
                               build_encoding(ds_out, eff["compression"], eff["chunks"]))
            else:
                _preprocess_blocked(ds_cube, g, eff, zpath, all_days=all_days,
                                    block_days=block_days, time_chunk=time_chunk,
                                    census=census, expected=expected, stale=stale,
                                    src_attrs=src_attrs, window=window, cache=cache)
                n_after = n_before - len(stale) + len(census)
                shape = (len(all_days), H, W)
        log.info("  rewrote %s  vars=%d (+%d derived) shape=(t=%d,y=%d,x=%d)",
                 zpath.name, n_after, n_after - n_before, *shape)
        rep.wrote()

    rep.log_summary()
    return rep


def preprocess(project: Project, *, grids=None, aois=None, dry_run=False,
               overwrite=False) -> report.ProductReport | None:
    """Preprocess assembled cubes for a validated Project. Terminal stage, runs after assembly.

    Same signature as every product's acquire() and datacube.assemble(); reads only the
    assembled `<aoi>.zarr` cubes (and the aligned tide series on disk) and rewrites each in
    place, so it must run AFTER the assembler. A no-op unless `project.preprocess.enabled`
    (checked by the callers). Safe to re-run: it is a no-op when the cube already carries this
    step selection's output, and rebuilds the derived channels from the raw ones otherwise.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(preprocess, "coastal_sst_data post-assembly preprocessing.")


if __name__ == "__main__":
    main()
