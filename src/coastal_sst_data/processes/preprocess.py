#!/usr/bin/env python3
"""
coastal_sst_data -- post-assembly preprocessing (derived cube, runs AFTER the assembler).

The assembled datacube ships RAW ingredients on a common grid + daily axis (see
`processes.datacube`): masking, water-filling, and multi-input derivations are DOWNSTREAM
modelling determinations, so the raw cube does not bake them in. This stage is that
downstream layer, given a structured, config-driven home. It reads each assembled
`<output_dir>/<datacube.output_subdir>/<aoi>.zarr` and writes a SEPARATE derived cube

    <output_dir>/<preprocess.output_subdir>/<aoi>.zarr

leaving the raw cube byte-untouched. Like the acquisition stages it is a thin loop over a
REGISTRY: each preprocessing STEP declares what cube channels it reads/writes and a compute
function, and adding a new step is one `PreprocessStep` registration -- exactly mirroring the
datacube's contributor protocol (`datacube.CONTRIBUTORS`), but reading an OPENED xarray cube
rather than aligned files.

Steps shipped today (both re-introduce computations the raw-output refactor deliberately
pushed downstream -- D6/D7/D12):

  * water_line   -- the tide-adjusted waterline per thermal sensor, from that sensor's
                    overpass tide + the DEM. Emits `<sensor>_water_elev` (m, +above/-below the
                    waterline) and `<sensor>_water_class` (submerged/exposed/unknown). Pure
                    glue over `processes.water_level`.
  * fill_water   -- nearest-neighbour fill of the level-4 SST products' (MUR, CMEMS) NaN gaps
                    over water (`landcover_water==1`), with a `<channel>_filled` companion
                    mask so an invented value stays distinguishable from an observed one.
  * filter_clouds -- screen ECOSTRESS pixels against a gap-free baseline L4 SST (MUR/CMEMS):
                    a fixed cold offset (`baseline - eco > threshold_k`) or a distribution-based,
                    seasonally-aware outlier floor (`eco < mean - n_sigma*sigma`). Folds drops
                    into `<sensor>_valid_<ver>`/`<sensor>_sst_<ver>` + a `*_cloudfiltered` flag.
  * filter_cloud_cover -- gate ECOSTRESS on the met total-cloud-cover field (HRRR/ERA5, percent):
                    a scene-level rejection and a per-pixel cutoff, with a `*_metcloudfiltered`
                    flag and a `<sensor>_scene_cloud_pct_<src>` diagnostic. Composes with
                    filter_clouds (both read the WORKING sst/valid, so drops union).

The derived cube is self-describing: alongside each derived channel it carries the specific
raw inputs it was built from (the DEM elevation + its datum attrs, the water mask, the tide /
overpass-hour channels), so it is legible standalone rather than only as a join against the
raw cube.

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
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Project
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, products, provenance, report, store
from . import datacube, water_level
from .cloud_filter import _step_filter_clouds, _step_filter_cloud_cover
from .datacube import build_encoding, write_zarr_safe
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
class PreprocessStep:
    key: str                                       # step name; also the config selector key
    reads: tuple[str, ...]                         # cube channel families it consumes
    writes: tuple[str, ...]                        # cube channel families it emits
    fn: "Callable[[PreprocessContext], None]"
    depends_on: tuple[str, ...] = ()               # step keys that must run first
    option_keys: frozenset[str] = frozenset()      # per-step config options it reads
    provenance_inputs: tuple[str, ...] = ()        # products a derived channel attributes to


@dataclass
class PreprocessContext:
    """The shared state a step reads and writes while one AoI's derived cube is built.

    A step reads channels off the OPENED raw cube (`read`, `has`, `channels_with_prefix`,
    `sensor_hours`), emits derived channels (`emit`), and copies raw inputs forward so the
    derived cube is self-describing (`carry`). It never mutates `ds_raw`. The orchestrator
    assembles `channels` into the derived Dataset once every step has run.
    """
    g: AoiGrid
    eff: dict
    days: pd.DatetimeIndex
    aid: str
    H: int
    W: int
    ds_raw: xr.Dataset
    channels: dict[str, tuple] = field(default_factory=dict)
    var_attrs: dict[str, dict] = field(default_factory=dict)
    global_attrs: dict[str, Any] = field(default_factory=dict)

    # ---- reading the raw cube (defensive: a missing/misshaped channel -> None) -------- #
    def has(self, name: str) -> bool:
        return name in self.ds_raw.variables

    def read(self, name: str, *, dims=None, dtype="float32") -> np.ndarray | None:
        """A raw-cube channel as a plain array, or None if absent / wrong-shaped.

        Mirrors the datacube loaders' discipline: a step must degrade (skip, or emit
        all-UNKNOWN) when an input product was never selected, not crash the whole stage.
        """
        if name not in self.ds_raw.variables:
            return None
        da = self.ds_raw[name]
        if dims is not None and tuple(da.dims) != tuple(dims):
            return None
        return np.asarray(da.values).astype(dtype, copy=False)

    def channels_with_prefix(self, prefix: str) -> list[str]:
        return sorted(str(v) for v in self.ds_raw.data_vars if str(v).startswith(prefix))

    def sensor_hours(self, prefix: str) -> np.ndarray | None:
        """Per-day fractional overpass hour for a sensor, or None if the sensor is absent.

        A stacked-data sensor (ECOSTRESS) has no single `<pre>_hour`, only `<pre>_hour_<ver>`;
        this coalesces `<pre>_hour` and every `<pre>_hour_<ver>`, taking the first finite value
        per day (the sensor's versions describe the same physical overpasses). None means the
        sensor emitted no hour channel at all -- i.e. it was not acquired -- so the caller
        emits nothing for it rather than an all-UNKNOWN channel.
        """
        names = sorted(str(v) for v in self.ds_raw.data_vars
                       if (str(v) == f"{prefix}_hour" or str(v).startswith(f"{prefix}_hour_"))
                       and self.ds_raw[v].dims == ("time",))
        if not names:
            return None
        out = np.full(len(self.days), np.nan, "float32")
        for n in names:
            arr = np.asarray(self.ds_raw[n].values, "float32")
            take = np.isnan(out) & np.isfinite(arr)
            out[take] = arr[take]
        return out

    def elevation_source(self) -> str | None:
        """The DEM tag to use when `dem_source` is unset: the sole `elevation_<dem>` present."""
        tags = sorted(str(v)[len("elevation_"):] for v in self.ds_raw.data_vars
                      if str(v).startswith("elevation_"))
        if not tags:
            return None
        if len(tags) > 1:
            log.info("  water_line: %d DEM sources present %s; using %r "
                     "(set water_line.dem_source to choose)", len(tags), tags, tags[0])
        return tags[0]

    def tide_source(self) -> str | None:
        """The tide tag to use when `tide_source` is unset: the sole daily `tide_<src>`."""
        tags = sorted(str(v)[len("tide_"):] for v in self.ds_raw.data_vars
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
        return self.eff["steps"].get(key, {})

    # ---- writing the derived cube ---------------------------------------------------- #
    def emit(self, name: str, dims, arr, **attrs) -> None:
        self.channels[name] = (dims, arr)
        if attrs:
            self.var_attrs.setdefault(name, {}).update(attrs)

    def carry(self, name: str) -> None:
        """Copy a raw-cube channel VERBATIM into the derived cube (with its attrs), so a
        derived channel travels beside the input it was built from. No-op if the channel is
        absent or already present (a later `emit` of the same name wins)."""
        if name in self.channels or name not in self.ds_raw.variables:
            return
        da = self.ds_raw[name]
        self.channels[name] = (da.dims, np.asarray(da.values))
        if da.attrs:
            self.var_attrs.setdefault(name, {}).update(dict(da.attrs))


# --------------------------------------------------------------------------- #
# Step: water_line
# --------------------------------------------------------------------------- #
def _step_water_line(ctx: PreprocessContext) -> None:
    """Per-sensor tide-adjusted waterline, at each sensor's own overpass.

    Pure glue over `processes.water_level` (the reference math the raw-output refactor left in
    place for exactly this): subtract the DEM's DEM->MSL datum offset to put the ground on MSL,
    take that sensor's overpass tide, and re-reference each cell to the tide-adjusted waterline.
    Emits `<sensor>_water_elev` and `<sensor>_water_class`; carries the DEM elevation, the
    overpass-hour channel(s), and the overpass-tide channel it used.
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
    datum = float(ctx.ds_raw[f"elevation_{dem_source}"].attrs.get(
        "datum_offset_m", water_level.DEFAULT_DATUM_OFFSET_M))
    ctx.carry(f"elevation_{dem_source}")

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
        # Carry the raw ingredients this sensor's fields were built from.
        for n in ctx.channels_with_prefix(f"{s}_hour"):
            ctx.carry(n)
        if tide_source is not None and ctx.has(f"{s}_tide_{tide_source}"):
            ctx.carry(f"{s}_tide_{tide_source}")


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
    honest NaN gaps. Emits the FILLED source channel (same name) plus a `<channel>_filled`
    uint8 mask (1 = invented over unobserved water, 0 = observed), so a filled value never
    passes for an observed one. Filling everywhere would fabricate data over land the source
    legitimately never covered, so an absent mask means: fill nothing.
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
    ctx.carry(mask_channel)

    for c in channels:
        raw = ctx.read(c, dims=T3)
        if raw is None:
            continue
        observed = np.isfinite(raw)
        filled = fill_water_nn(raw, water)
        filled_mask = (np.isfinite(filled) & ~observed).astype("uint8")
        ctx.emit(c, T3, filled, preprocess=f"nn_filled over {mask_channel}",
                 long_name=f"{c} (nearest-neighbour filled over water)")
        ctx.emit(f"{c}_filled", T3, filled_mask,
                 long_name=(f"{c} was nearest-neighbour filled over water the source did not "
                            "observe (1 = invented, 0 = observed)"),
                 flag_values=[0, 1], flag_meanings="observed filled")


def _fill_channels(ctx: PreprocessContext, sources) -> list[str]:
    """The level-4 channels to fill: `mur_sst` and every `cmems_*` (bar `_valid`/`_filled`),
    restricted to the `sources` product tags (`mur` / `cmems`) when given."""
    pairs: list[tuple[str, str]] = []
    if ctx.has("mur_sst"):
        pairs.append(("mur", "mur_sst"))
    for c in ctx.channels_with_prefix("cmems_"):
        if not c.endswith(("_valid", "_filled")):
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
        writes=("_filled",),
        fn=_step_fill_water,
        option_keys=frozenset({"sources", "mask_channel"}),
    ),
    PreprocessStep(
        key="filter_clouds",
        reads=("eco_sst", "eco_valid", "eco_cloud", "mur_sst", "cmems_", "doy_sin", "doy_cos"),
        writes=("_sst", "_valid", "_cloudfiltered"),
        fn=_step_filter_clouds,
        depends_on=("fill_water",),        # so offset mode sees the gap-filled baseline
        option_keys=frozenset({"method", "threshold_k", "n_sigma", "baseline", "stat_scope",
                               "seasonality", "sensors", "mask_sst", "use_cloud_raster"}),
        provenance_inputs=("ecostress", "mur"),
    ),
    PreprocessStep(
        key="filter_cloud_cover",
        reads=("eco_cloud_cover_", "cloud_cover_", "eco_sst", "eco_valid", "landcover_water"),
        writes=("_sst", "_valid", "_metcloudfiltered", "_scene_cloud_pct"),
        fn=_step_filter_cloud_cover,
        depends_on=("filter_clouds",),     # deterministic order; drops compose via `_working`
        option_keys=frozenset({"source", "scene_max_pct", "pixel_max_pct", "sensors",
                               "mask_sst", "water_mask_channel"}),
        provenance_inputs=("met_overpass", "met", "ecostress"),
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
    if problems:
        raise ValueError(
            "unrecognised preprocess option(s) -- these would be SILENTLY IGNORED:\n  "
            + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Build one AoI's derived cube
# --------------------------------------------------------------------------- #
def preprocess_aoi(ds_raw: xr.Dataset, g: AoiGrid, eff: dict) -> xr.Dataset:
    """Build the derived Dataset for one AoI from its assembled raw cube.

    Every selected step runs through the uniform `(ctx) -> None` protocol in `depends_on`
    order. The derived cube is built on the RAW cube's own coords (never re-derived), so the
    two align cell-for-cell, and it carries the raw inputs each derived channel came from.
    """
    days = pd.DatetimeIndex(ds_raw["time"].values)
    ctx = PreprocessContext(g=g, eff=eff, days=days, aid=g.name,
                            H=int(ds_raw.sizes["y"]), W=int(ds_raw.sizes["x"]), ds_raw=ds_raw)

    selected = [s for s in STEPS if s.key in eff["steps"]]
    for step in _topo_order(selected):
        step.fn(ctx)

    ds_out = xr.Dataset(
        ctx.channels,
        coords={"time": ds_raw["time"].values,
                "y": ds_raw["y"].values, "x": ds_raw["x"].values})
    ds_out.attrs.update(
        aoi_id=ctx.aid,
        crs=ds_raw.attrs.get("crs", g.target_crs),
        derived_from=f"{eff['project'].datacube.output_subdir}/{ctx.aid}.zarr",
        preprocess=json.dumps({k: eff["steps"][k] for k in
                               (s.key for s in selected)}, sort_keys=True))
    ds_out.attrs.update(ctx.global_attrs)
    for name, attrs in ctx.var_attrs.items():
        if name in ds_out:
            ds_out[name].attrs.update(attrs)

    # PROVENANCE: the config that built this cube, and for every field the source(s) it came
    # from. Same machinery as the assembler; the derived channels are mapped in
    # provenance.field_inputs (water_line -> bathymetry/tides/sensor; *_filled -> its base).
    prod = provenance.collect(eff["aligned_root"], ctx.aid, datacube.PRODUCT_DIRS)
    rec = provenance.build(eff["project"], list(ds_out.data_vars), prod)
    ds_out.attrs.update(
        created_at=rec["created_at"], package_version=rec["package_version"],
        code_version=rec["code_version"],
        config_sha256=rec["config_sha256"] or "", config_path=rec["config_path"] or "",
        config_yaml=rec["config_yaml"] or "",
        provenance=json.dumps(rec["fields"], sort_keys=True),
        provenance_products=json.dumps(rec["products"], sort_keys=True))
    return ds_out


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
        "raw_root": root / project.datacube.output_subdir,
        "out_dir": root / pp.output_subdir,
        "steps": {k: dict(v.model_extra or {}) for k, v in pp.steps.items()},
        "chunks": dict(pp.chunks),
        "compression": pp.compression,
        "overwrite": bool(pp.overwrite),
    }


def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Build one derived Zarr cube per AoI from its assembled raw cube."""
    _check_step_options(eff)
    if not eff["steps"]:
        log.info("preprocess: no steps selected; nothing to do.")
        return None

    out_dir = eff["out_dir"]
    overwrite = eff["overwrite"]
    names = select_aois(grids, only_aoi)
    rep = report.ProductReport("preprocess")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        g = grids[name]
        raw_zpath = eff["raw_root"] / f"{name}.zarr"
        zpath = out_dir / f"{name}.zarr"

        if not raw_zpath.exists():
            log.warning("=== %s: no assembled cube at %s; run `assemble` first -- skipping ===",
                        name, raw_zpath.name)
            rep.skip()
            continue
        if zpath.exists() and not overwrite:
            log.info("=== %s: %s exists, skipping (use overwrite) ===", name, zpath.name)
            rep.skip()
            continue
        store.sweep_scratch(zpath)      # clear scratch from a run that died mid-write
        if dry_run:
            log.info("=== %s: [dry-run] would preprocess -> %s ===", name, zpath.name)
            continue

        log.info("=== preprocessing %s (steps: %s) ===",
                 name, ", ".join(sorted(eff["steps"])))
        with xr.open_zarr(raw_zpath) as ds_raw:
            ds_out = preprocess_aoi(ds_raw, g, eff)
        write_zarr_safe(ds_out, zpath, build_encoding(ds_out, eff["compression"], eff["chunks"]))
        log.info("  wrote %s  vars=%d shape=(t=%d,y=%d,x=%d)", zpath.name,
                 len(ds_out.data_vars), ds_out.sizes["time"], ds_out.sizes["y"],
                 ds_out.sizes["x"])
        rep.wrote()

    rep.log_summary()
    return rep


def preprocess(project: Project, *, grids=None, aois=None, dry_run=False,
               overwrite=False) -> report.ProductReport | None:
    """Preprocess assembled cubes for a validated Project. Terminal stage, runs after assembly.

    Same signature as every product's acquire() and datacube.assemble(); reads only the
    assembled `<aoi>.zarr` cubes (and the aligned tide series on disk), so it must run AFTER
    the assembler. A no-op unless `project.preprocess.enabled` (checked by the callers).
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
