"""Configuration loading and validation.

This module does three jobs:

1. **Describe** what a valid config looks like (the Pydantic *models* below).
2. **Load** a YAML file from disk and turn it into a validated object.
3. **Read secrets** (API keys, passwords) from environment variables, kept
   completely separate from the config file so credentials never live in the repo.

"""

from __future__ import annotations

import logging
import re
from datetime import date
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal
from math import cos, radians

import yaml
from pydantic import (BaseModel, Field, PrivateAttr, ValidationError, field_validator,
                      model_validator)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0. Classes to define data types and areas of interest
# ---------------------------------------------------------------------------

def wrap_lon(lon: float) -> float:
    """Normalize a longitude into [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


class BoundingBox(BaseModel):
    """Axis-aligned box in decimal degrees (WGS84).

    The min_/max_ fields are really WEST/EAST and SOUTH/NORTH edges, following
    the [W, S, E, N] convention: ``min_lon`` is the western edge and ``max_lon``
    the eastern edge, walking EASTWARD. For a box that straddles the antimeridian
    this means ``min_lon > max_lon`` (e.g. W=179.4, E=-179.3) -- that is a valid
    narrow box across 180 deg, NOT an inverted one that wraps the globe the long
    way. Use ``lon_span`` / ``center_lon`` rather than raw subtraction so the
    wrap is handled correctly.
    """
    min_lon: float = Field(..., ge=-180, le=180)   # western edge
    min_lat: float = Field(..., ge=-90, le=90)     # southern edge
    max_lon: float = Field(..., ge=-180, le=180)   # eastern edge
    max_lat: float = Field(..., ge=-90, le=90)     # northern edge

    @model_validator(mode="after")
    def _ordered(self):
        # Latitude never wraps: south must be strictly below north.
        if self.min_lat >= self.max_lat:
            raise ValueError("bbox min_lat must be < max_lat")
        # Longitude MAY have west > east (an antimeridian crossing); only a
        # zero-width box (west == east) is degenerate and rejected.
        if self.min_lon == self.max_lon:
            raise ValueError("bbox min_lon and max_lon must differ")
        return self

    @property
    def crosses_antimeridian(self) -> bool:
        """True when the box straddles the 180 deg meridian (west > east)."""
        return self.min_lon > self.max_lon

    @property
    def lon_span(self) -> float:
        """East-west width in degrees, walking eastward (handles the wrap)."""
        span = self.max_lon - self.min_lon
        return span + 360.0 if span < 0 else span

    @property
    def center_lon(self) -> float:
        """Longitude of the box center, correct across the antimeridian."""
        return wrap_lon(self.min_lon + self.lon_span / 2.0)

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2.0


class AreaOfInterest(BaseModel):
    """A location plus the box drawn around it to acquire data for."""
    name: str
    center_lat: float = Field(..., ge=-90, le=90)
    center_lon: float = Field(..., ge=-180, le=180)
    buffer_ns_km: float = Field(..., gt=0)          # half-height (north-south)
    buffer_ew_km: float = Field(..., gt=0)          # half-width (east-west)

    @property
    def bbox(self) -> BoundingBox:
        """Derive the bounding box, degrading gracefully at the poles.

        Rather than raising when a buffer runs past +/-90 deg latitude (or, near
        a pole, balloons across all longitudes), the box is CLAMPED and a warning
        is logged. That way one pathological AoI degrades to a clipped box
        instead of crashing a batch run over all the other AoIs.
        """
        # approx degrees-per-km; fine for defining acquisition windows
        dlat = self.buffer_ns_km / 111.0
        south, north = self.center_lat - dlat, self.center_lat + dlat
        if south < -90.0 or north > 90.0:
            log.warning(
                "AoI %r extends past the pole (lat %.2f..%.2f); clamping to [-90, 90].",
                self.name, south, north,
            )
            south, north = max(south, -90.0), min(north, 90.0)

        dlon = self.buffer_ew_km / (111.0 * cos(radians(self.center_lat)))
        if 2.0 * dlon >= 360.0:
            # Near a pole all meridians converge, so the box spans every
            # longitude. Clamp to the full range instead of wrapping the globe.
            log.warning(
                "AoI %r spans all longitudes (near-pole, dlon=%.1f deg); "
                "clamping longitude to [-180, 180].", self.name, dlon,
            )
            west, east = -180.0, 180.0
        else:
            # Wrap each edge into [-180, 180) WITHOUT reordering, so a box near
            # the antimeridian crosses it the short way (west may end up > east)
            # rather than flipping to span the globe the long way around.
            west, east = wrap_lon(self.center_lon - dlon), wrap_lon(self.center_lon + dlon)

        return BoundingBox(min_lon=west, max_lon=east, min_lat=south, max_lat=north)


# ---------------------------------------------------------------------------
# Data products and their per-process options
# ---------------------------------------------------------------------------
# DataProduct and the product REGISTRY live in coastal_sst_data.products, which imports
# nothing from the package -- see that module's docstring for why (config needs the
# options/auth tables at validation time, and every process module imports config, so a
# spec declared inside a process module would close an import cycle).
#
# These are re-exported DELIBERATELY: `from ..config import DataProduct` is how every module
# already reaches them, and a caller should not have to know which of the two files a name
# lives in. A product's config surface is still config's business to DESCRIBE -- it is just
# no longer config's business to REMEMBER.
from .products import (                                         # noqa: E402,F401
    BY_PRODUCT, DataProduct, ProductSpec, REGISTRY, spec,
)


class ProductOptions(BaseModel):
    """GLOBAL (project-wide) options for one data product.

    Set once under the project `products` block. Keys stay open at the pydantic level (one
    bag serves every product), but they are CHECKED against PRODUCT_OPTIONS below -- see
    `Project._option_keys_are_known`.
    """
    model_config = {"extra": "allow"}


def opt(opts, name: str, default=None):
    """Read an optional override off a product-options bag (extra='allow').

    The bags are open at the pydantic level, so a key that was never set is simply absent
    and `getattr` needs a default. `opts` itself is None when the product is unselected.

    Every process module used to carry a private copy of this three-line function -- ten
    byte-identical definitions. It lives here, next to the ProductOptions/SourceOptions it
    reads, because it is part of how those bags are consumed.
    """
    return getattr(opts, name, default) if opts is not None else default


class SourceOptions(BaseModel):
    """REGION-DEPENDENT options for one data product's source.

    Set per region under `sources`, for things that vary geographically (e.g. which DEM or
    tide model has coverage there). Checked against REGION_OPTIONS below.
    """
    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# Which options each product READS, and which a REGION may override
# --------------------------------------------------------------------------- #
# DERIVED from the product registry -- these used to be hand-maintained lists that had to be
# kept in step with the `_opt(opts, "<key>", ...)` calls in each module, and with each other.
#
# The rules they enforce are unchanged, and both are about the config not LYING:
#
#   * An option no module reads does NOTHING. Accepting it in silence is the worst of both
#     worlds -- the config says one thing, the run does another, and the provenance
#     faithfully records the run, so the lie lives in the config file where nobody looks.
#     `bathymetry.source` was exactly this: documented, settable, and silently discarded
#     (the module read `default_source`), so a config asking for 3 m CUDEM quietly got
#     ~100 m GMRT.
#
#   * A REGION may override only what genuinely varies geographically:
#         "WHICH SOURCE HAS COVERAGE HERE"  -> yes
#         "WHAT THE CUBE MEANS"             -> no
#     The first is a fact about the world and it changes as the project expands (HRRR is
#     North America only, CO-OPS is U.S. only, CUDEM is CONUS only, IOOS is North America,
#     CMEMS publishes regional models). The second must stay uniform: `variables`, `depths`,
#     `qc_flags`, `reference_time` decide the cube's CHANNEL SET, and letting a region
#     change those makes two AoIs' cubes silently non-comparable. For met it is worse than
#     non-comparable -- the assembler names airtemp/wind_*/swrad/cloud_cover explicitly, so
#     a region that dropped one would ship an all-NaN channel indistinguishable from a
#     forcing that was fetched and came back empty.
#
# products.py enforces the second rule at the registry level (a region_option must be an
# option the module actually reads); this file enforces it against the user's config.
PRODUCT_OPTIONS: dict[DataProduct, set[str]] = {
    s.product: set(s.options) for s in REGISTRY
}

REGION_OPTIONS: dict[DataProduct, set[str]] = {
    s.product: set(s.region_options) for s in REGISTRY if s.region_options
}

REGION_ONLY_OPTIONS: dict[DataProduct, set[str]] = {
    s.product: set(s.region_only_options) for s in REGISTRY if s.region_only_options
}


def resolve_opts(project: "Project", aoi_name: str, product: DataProduct):
    """The options ONE AoI runs `product` with: its region's overrides, over the global bag.

    This is the single two-level lookup every product resolves its options through. It used
    to exist only inside bathymetry and tides (as their private `_resolve_source`), which is
    why they were the only two products whose source could vary by region -- every other
    module read `project.products[...]` directly and was therefore locked to one source for
    the entire project. A project with a Pacific Northwest region and a Mediterranean one
    simply could not use different in-situ networks, whatever the config said.

    Returns a merged ProductOptions bag (or None if the product is not selected), so callers
    read it exactly as they read the global one -- with `config.opt`.
    """
    global_opts = project.products.get(product)
    if global_opts is None:
        return None
    merged = dict(global_opts.model_extra or {})
    region_opts = project.region_of(aoi_name).sources.get(product)
    if region_opts is not None:
        merged.update(region_opts.model_extra or {})
    return ProductOptions(**merged)


def resolve_step_opts(project: "Project", aoi_name: str, key: str) -> dict:
    """The options ONE AoI runs preprocess step `key` with: region overrides over the global bag.

    The step analog of `resolve_opts`. Preprocess step bags (`preprocess.steps.<key>`) are
    project-global; a region layers per-key overrides through `regions[].preprocess_steps.<key>`
    for options that must vary geographically -- e.g. `flag_georef.min_coast_obs`, a gate counted
    in absolute coastline CELLS, which is an order of magnitude stricter at an AoI whose coastline
    is 10x shorter. Which keys a region may override is guarded per step at stage time
    (`preprocess._check_step_options` against `PreprocessStep.region_option_keys`).

    Returns a plain dict (steps read options straight off it, unlike products' ProductOptions bag).
    A step not selected globally simply has an empty global bag; a region override on it is still
    merged here but the stage never runs it (selection is global).
    """
    global_opts = project.preprocess.steps.get(key)
    merged = dict(global_opts.model_extra or {}) if global_opts is not None else {}
    region_opts = project.region_of(aoi_name).preprocess_steps.get(key)
    if region_opts is not None:
        merged.update(region_opts.model_extra or {})
    return merged


def _options_by_product(value: Any) -> Any:
    """Normalize a `{product: options}` mapping before validation.

    A bare `product:` (null value) is treated as 'selected, with default
    options' -> {}, so listing a product with no options is just its name.
    """
    if isinstance(value, dict):
        return {k: ({} if v is None else v) for k, v in value.items()}
    return value


class PreprocessStepOptions(BaseModel):
    """Per-step options for one post-assembly preprocessing step.

    Kept open at the pydantic level (one bag serves every step), then CHECKED against the
    step's declared `option_keys` INSIDE the preprocess stage -- not here. The step registry
    lives in `processes/preprocess.py`, which imports `config`; importing it back here to
    build a validation table would close an import cycle, so the check runs at stage time
    (`preprocess._check_step_options`). This mirrors how PRODUCT_OPTIONS validates a product
    bag, just deferred one hop to keep the registry the single source of truth. The same bag
    serves both the global `preprocess.steps.<key>` and a region's `preprocess_steps.<key>`
    override; the latter is additionally checked against the step's `region_option_keys`.
    """
    model_config = {"extra": "allow"}


class Region(BaseModel):
    """AoIs that share the same region-dependent data sources."""
    model_config = {"extra": "forbid"}
    name: str
    # Region-dependent, per-product source options (e.g. bathymetry.dem_source).
    sources: dict[DataProduct, SourceOptions] = Field(default_factory=dict)
    # Region-dependent, per-preprocess-step option overrides (e.g. flag_georef.min_coast_obs).
    # The step analog of `sources`: a gate counted in absolute cells needs per-region tuning.
    # Resolved by `resolve_step_opts`, guarded by the step's `region_option_keys` at stage time
    # (`preprocess._check_step_options`) -- regions tune coverage/thresholds, not the cube's meaning.
    preprocess_steps: dict[str, PreprocessStepOptions] = Field(default_factory=dict)
    areas: list[AreaOfInterest] = Field(..., min_length=1)

    @field_validator("sources", mode="before")
    @classmethod
    def _fill_source_defaults(cls, v):
        # A blank `sources:` in YAML parses as None -> treat as "no source
        # options" ({}), so regions that only use globally-available products
        # can leave the section empty (or omit it entirely).
        if v is None:
            return {}
        return _options_by_product(v)

    @field_validator("preprocess_steps", mode="before")
    @classmethod
    def _fill_step_defaults(cls, v):
        # Mirror PreprocessSpec.steps: a blank section is {}, a bare `step:` (null) is {}.
        if v is None:
            return {}
        if isinstance(v, dict):
            return {k: (o if o is not None else {}) for k, o in v.items()}
        return v


class TimeWindow(BaseModel):
    """Inclusive UTC date range that every data stream is acquired over."""
    model_config = {"extra": "forbid"}
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_date < self.start_date:
            raise ValueError("time.end_date must be on or after time.start_date")
        return self


class GridSpec(BaseModel):
    """Target grid every product is regridded onto within each AoI.

    Every field has a default (the values from examples/config.test.yaml), so a
    config can omit the `grid:` block, or any individual field, and still get a
    complete, valid grid. Unknown keys are rejected so a typo fails loudly.
    """
    model_config = {"extra": "forbid"}
    resolution_m: float = Field(100.0, gt=0)   # target posting in metres
    target_crs: str = "auto"                   # "auto" -> local UTM per AoI, or e.g. "EPSG:32610"
    resampling_continuous: str = "bilinear"    # continuous rasters (SST, DEM)
    resampling_categorical: str = "nearest"    # masks / class rasters
    snap_origin: bool = True                   # snap grid origin to a resolution multiple


class CompressionSpec(BaseModel):
    """Per-variable Zarr compression for the assembled datacube.

    Lossless: values are kept as-is (float32 / uint8); only a Blosc entropy codec
    is applied. Smooth/interpolated fields still compress well via byte-shuffle.
    The block exists so the codec can be tuned (or later extended to packing/
    quantization) without changing the cube layout.
    """
    model_config = {"extra": "forbid"}
    codec: str = "zstd"                                        # Blosc cname
    level: int = Field(5, ge=0, le=9)                          # Blosc clevel
    shuffle: Literal["shuffle", "bitshuffle", "noshuffle"] = "shuffle"


class DataCubeSpec(BaseModel):
    """How the assembler knits the aligned per-product files into one Zarr cube.

    Every field has a default, so the whole `datacube:` block is optional. The cube
    ships raw ingredients on a common grid; masking / filling / derivation are downstream
    modelling determinations, so there are no mask or fill knobs here.
    """
    model_config = {"extra": "forbid"}
    chunks: dict[str, int] = Field(default_factory=lambda: {"time": 64, "y": 128, "x": 128})
    # NOTE: `fill_mur_water`, `fill_cmems_water`, and `water_level` were REMOVED -- the cube
    # now ships raw ingredients (observed values with honest NaN gaps; `elevation` + `depth`
    # + `tide` for a downstream water-level computation) and masking/filling/derivation are
    # downstream modelling determinations. `extra="forbid"` means an old config that still
    # sets any of these three keys fails validation loudly rather than being silently ignored.
    #
    # The DEM->MSL datum offset still ships (as the cube attrs datum_offset_m/datum_status);
    # it is RESOLVED automatically by the `datum` stage (processes.datum) whenever bathymetry
    # is selected, not configured. The only knob is an optional per-region FALLBACK,
    # regions[].sources.bathymetry.datum_offset_m, used only when VDatum/CO-OPS cannot resolve
    # it (a resolved value always wins).
    #
    # Which met file feeds the cube's met channels (airtemp, wind_*, swrad, cloud_cover):
    #   "reference"  -- the daily snapshot at products.met.reference_time (default 10:30
    #                   local solar, Landsat's overpass). One time of day, every day.
    #   "daily_mean" -- the mean over products.met.daily_mean_hours (the old behaviour),
    #                   which smears the diurnal cycle.
    # Falls back to the other if the chosen file is absent.
    met_time: Literal["reference", "daily_mean"] = "reference"
    # NOTE: `overpass_met` was REMOVED. Met-at-overpass is now the `met_overpass` PRODUCT
    # (D14): the cube emits <sensor>_<var>_<src> for that product's `(sensor, source)`
    # combinations. `extra="forbid"` makes an old config still setting `datacube.overpass_met`
    # fail loudly (pointing the user at the met_overpass product) rather than be ignored.
    #
    # Emit the in-situ channels: the buoy value at the reference time, plus a matchup at
    # each sensor's own overpass (the ground truth a satellite scene is validated against).
    insitu: bool = True
    # Matchup tolerance, minutes: beyond this an overpass gets NaN rather than a stale
    # observation -- a buoy reading two hours off is not a matchup.
    insitu_max_dt_min: float = 60.0
    compression: CompressionSpec = Field(default_factory=CompressionSpec)
    output_subdir: str = "datacube"            # cube dir under output_dir
    overwrite: bool = False                    # rebuild existing <aoi>.zarr cubes
    # How many days the assembler builds and writes at a time. The whole cube used to be held
    # in memory before anything was written, so peak memory was
    #     channels x days x height x width x 4 bytes
    # -- a number that grows with the AoI AND with the date range, and that killed a large
    # region outright. Blocking makes it `channels x block_days x height x width x 4`, i.e.
    # independent of how long the window is.
    #
    # "auto" sizes the block from a memory budget and the AoI's own grid, per AoI (a cube that
    # fits is still assembled in ONE pass, exactly as before). An integer forces it.
    block_days: int | Literal["auto"] = "auto"
    # The budget "auto" spends, in GB. None detects it: $COASTAL_SST_DATA_MEM_GB, then
    # $SLURM_MEM_PER_NODE, then the cgroup limit, then physical RAM -- and takes HALF of
    # whatever it detects. Set this when the assembler shares a machine, or when the detected
    # figure is not the allowance the job actually has (physical RAM on a scheduled node is
    # the case that bites: the kernel kills the process at the cgroup limit, not at the
    # hardware's). An explicit value is used as given, not halved.
    memory_budget_gb: float | None = Field(None, gt=0)


class PreprocessSpec(BaseModel):
    """Optional POST-ASSEMBLY preprocessing: derived channels added to the assembled cube.

    The assembler ships RAW ingredients (see DataCubeSpec); this stage adds the derived
    channels to that same `<output_dir>/<datacube.output_subdir>/<aoi>.zarr`, under names of
    its own so the assembled channels keep their values. It is opt-in (`enabled`), so existing
    runs are unaffected. Each entry in `steps` selects a step from the `preprocess.STEPS`
    registry (e.g. `water_line`, `fill_water`, `filter_clouds`, `filter_cloud_cover`,
    `filter_land_clouds`) and carries its per-step options; unknown step keys / options fail
    loudly at stage time.

    REMOVED KEYS (`extra="forbid"`, so a config that still sets one fails validation rather
    than being quietly ignored): `output_subdir` -- there is no separate derived cube to place
    any more; and `compression` / `chunks` -- one cube has one encoding, taken from `datacube`.
    """
    model_config = {"extra": "forbid"}
    enabled: bool = False                      # opt-in; nothing runs unless set true
    steps: dict[str, PreprocessStepOptions] = Field(default_factory=dict)
    # Memory blocking, exactly as `datacube.block_days` / `memory_budget_gb` -- this stage reads
    # a whole cube and writes a larger one, so it has the same problem. Both default to None
    # meaning "use the datacube value": the stages usually want the same answer, and one knob is
    # enough until they do not. Set these when preprocessing needs a smaller block than assembly
    # did, which it can: it holds the channels it READS as well as the ones it derives.
    block_days: int | Literal["auto"] | None = None
    memory_budget_gb: float | None = Field(None, gt=0)
    # Re-derive channels a cube already carries. Off by default because the stage detects a
    # changed step selection on its own; this is for forcing a rebuild after a code change.
    overwrite: bool = False

    @field_validator("steps", mode="before")
    @classmethod
    def _fill_step_defaults(cls, v):
        """A bare `step:` (null value) means 'selected, default options' -> {}."""
        if isinstance(v, dict):
            return {k: (o if o is not None else {}) for k, o in v.items()}
        return v


# ---------------------------------------------------------------------------
# Point extraction (OPTIONAL, and optional in a load-bearing way)
# ---------------------------------------------------------------------------
# Most projects build cubes and stop. A minority need the transpose -- for a handful of
# lat/lon sites, the time series of every channel, as a table -- and that minority must
# cost the majority NOTHING: no dependency (pyarrow is an extra, lazy-imported only on the
# parquet branch of store.write_table), no import (processes.extract is imported inside the
# CLI handler), no config key (this whole block defaults), and no pipeline stage (nothing
# runs unless someone types `coastal-sst-data extract`).

# The reductions a neighbourhood may be collapsed with. A CLOSED set, because a typo'd stat
# must fail at config-load time and not after an hour of cube reads.
#
# The nan* variants are the NaN-SKIPPING counterparts of the plain ones, and BOTH are
# offered on purpose. Over a 300 m neighbourhood of satellite SST, `mean` returning NaN
# because one pixel was cloudy and `nanmean` returning the mean of the other 28 are
# different scientific questions, and which one you got must be visible in the output rather
# than decided for you here. `nearest` is not a reduction at all -- it is the value of the
# pixel the point falls in.
STATS: tuple[str, ...] = (
    "nearest",
    "mean", "median", "std", "min", "max", "sum",
    "nanmean", "nanmedian", "nanstd", "nanmin", "nanmax", "nansum",
    "count", "count_valid",
)

# p10, p90, p97.5 -- NaN-skipping percentiles. Spelled as a pattern rather than enumerated
# so the set stays closed (`p9O` with a letter O still fails) without listing 101 names.
PERCENTILE_RE = re.compile(r"^p(100|\d{1,2})(\.\d+)?$")

# What `mask: water` resolves to. The alias exists because "restrict the mean to water" is
# the question people actually ask; the channel it needs is an implementation detail of
# whichever landcover source ran.
WATER_MASK_CHANNEL = "landcover_water"


class ExtractChannel(BaseModel):
    """One cube channel to pull at each point, and how to reduce it around the point.

    The neighbourhood is a CIRCLE of `radius_m` in the AoI's projected CRS -- metres on the
    ground, never degrees and never pixels. It is the set of pixels whose CENTRES fall
    within that radius, plus the pixel the point itself falls in. That last clause is not a
    nicety: at the package's default 100 m posting a `radius_m: 50` circle contains no pixel
    centre at all for most point positions, so without it the reduction would run over an
    empty set and the whole column would be NaN -- which reads exactly like a channel that
    was cloudy for the entire record.

    `stat` may be one name or a list; each becomes its OWN ROW, tagged in the output's `stat`
    column, so `[nanmean, nanstd, count_valid]` never has to be decoded back out of a
    variable name.

    `mask` restricts the neighbourhood before the reduction -- `water` for the cube's water
    mask, or any 2-D (y,x) channel name. `nearest` ignores it, because `nearest` is one
    specific pixel by definition.

    NONE of these options applies to an AoI-WIDE channel -- one that is 1-D (time,), like an
    overpass time (`lst_hour`), a tide (`tide_coops`) or a day-of-year term. Those are one
    value per day for the whole grid, so there is no neighbourhood to reduce over: write them
    bare (`lst_hour:`) and they ship one row per date. Which channels those are depends on
    the cube, so the check lives in `processes.extract.plan_channels`, where one is open.
    """
    model_config = {"extra": "forbid"}
    radius_m: float = Field(0.0, ge=0)          # 0 -> just the pixel the point falls in
    stat: list[str] = Field(default_factory=lambda: ["nearest"])
    mask: str | None = None                     # "water", or a 2-D channel name

    @field_validator("stat", mode="before")
    @classmethod
    def _listify(cls, v):
        """`stat: mean` and `stat: [mean]` are the same thing."""
        return [v] if isinstance(v, str) else v

    @field_validator("stat")
    @classmethod
    def _known_stats(cls, v):
        if not v:
            raise ValueError("stat cannot be empty; drop it to get the default `nearest`.")
        bad = [s for s in v if s not in STATS and not PERCENTILE_RE.match(str(s))]
        if bad:
            hint = ""
            near = get_close_matches(str(bad[0]), STATS, n=3, cutoff=0.6)
            if near:
                hint = f" (did you mean {', '.join(near)}?)"
            raise ValueError(
                f"unknown stat(s) {bad}{hint}; choose from {', '.join(STATS)}, "
                f"or a percentile like p90.")
        if len(set(v)) != len(v):
            # Duplicates would emit two identical rows and break the output table's
            # (point_id, aoi, time, variable, stat) primary key.
            raise ValueError(f"duplicate stat(s) in {v}")
        return list(v)

    @model_validator(mode="after")
    def _radius_is_used(self):
        """A radius nothing reduces over is a config that LIES about what was extracted.

        `stat: nearest` reads ONE pixel; asking for it over a 500 m circle states an intent
        the run ignores, and the output's `radius_m` column would then honestly record 0
        while the config said 500 -- with nothing anywhere contradicting the other.
        """
        if self.radius_m > 0 and set(self.stat) == {"nearest"}:
            raise ValueError(
                "radius_m is set but the only stat is `nearest`, which reads the single "
                "pixel the point falls in and ignores the radius. Add a reducing stat "
                "(nanmean/mean/median/...) or drop radius_m.")
        return self


class ExtractSpec(BaseModel):
    """Long-format time series pulled from the assembled cubes at user-supplied points.

    Entirely opt-in: with no `channels` nothing can run, and the whole block may be omitted.

    Channels are declared EXPLICITLY -- a channel not listed is not extracted, and a channel
    listed but absent from the cube is a hard error naming it. A quietly-missing column in a
    modelling table is indistinguishable from a channel that was genuinely all-NaN, and the
    difference is a modelling result.
    """
    model_config = {"extra": "forbid"}
    # CSV of points; at minimum lat/lon, ideally an id. Overridable with `--points`.
    points: Path | None = None
    # canonical field -> the column name in YOUR file, for when the aliases in points.py
    # do not cover it (see points.ALIASES).
    columns: dict[str, str] = Field(default_factory=dict)
    channels: dict[str, ExtractChannel] = Field(default_factory=dict)
    format: Literal["parquet", "csv"] = "parquet"
    output_subdir: str = "extract"              # <output_dir>/extract/
    stem: str = "points"                        # -> points.parquet
    overwrite: bool = False
    # Falls back to the assembler's budget, exactly as PreprocessSpec does: the stages
    # usually want the same answer, and one knob is enough until they do not.
    memory_budget_gb: float | None = Field(None, gt=0)

    @field_validator("channels", mode="before")
    @classmethod
    def _fill_channel_defaults(cls, v):
        """Accept the terse spellings.

        `chan:`               -> extract it, nearest pixel
        `chan: nanmean`       -> that stat
        `chan: [mean, std]`   -> those stats
        `chan: {radius_m: 300, stat: nanmean}` -> the full form
        """
        if isinstance(v, dict):
            out = {}
            for k, o in v.items():
                if o is None:
                    out[k] = {}
                elif isinstance(o, (str, list)):
                    out[k] = {"stat": o}
                else:
                    out[k] = o
            return out
        return v


# ---------------------------------------------------------------------------
# Authentication (NON-SECRET settings only)
# ---------------------------------------------------------------------------
# These say *how* to authenticate; the real secrets live OUTSIDE the repo and
# config -- in ~/.netrc, a service-account key file, or env vars -- and are read
# by earthaccess / earthengine directly, never from this file.
class EarthdataAuth(BaseModel):
    """How earthaccess authenticates to NASA Earthdata (ECOSTRESS, MUR).

    Only the strategy is set here; earthaccess reads the actual credentials from
    ~/.netrc ("netrc"), the EARTHDATA_USERNAME/PASSWORD env vars ("environment"),
    or an interactive prompt ("interactive").
    """
    model_config = {"extra": "forbid"}
    auth_strategy: Literal["netrc", "environment", "interactive"] = "netrc"


class GeeAuth(BaseModel):
    """How to initialize Google Earth Engine (Landsat, landcover).

    `project` and `service_account` are identifiers, not secrets. `key_file` is
    a PATH to a service-account JSON key whose contents stay outside the repo.
    Omit both service_account and key_file to use application-default creds.
    """
    model_config = {"extra": "forbid"}
    project: str
    service_account: str | None = None
    key_file: Path | None = None

    @field_validator("key_file")
    @classmethod
    def _expand_key_file(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else v

    @model_validator(mode="after")
    def _service_account_needs_key(self):
        if bool(self.service_account) != bool(self.key_file):
            raise ValueError(
                "gee.service_account and gee.key_file must be set together "
                "(or both omitted to use application-default credentials)"
            )
        return self


class CopernicusAuth(BaseModel):
    """How the copernicusmarine toolbox authenticates to Copernicus Marine (CMEMS).

    Only the strategy is set here; the toolbox reads the actual credentials from
    ~/.netrc under `machine auth.marine.copernicus.eu` ("netrc"), the
    COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars ("environment"), or its own
    ~/.copernicusmarine credentials file / a prompt ("interactive").
    """
    model_config = {"extra": "forbid"}
    auth_strategy: Literal["netrc", "environment", "interactive"] = "netrc"


class AuthConfig(BaseModel):
    """Non-secret authentication settings, one block per backend, plus run-wide refresh policy.

    NAMING CONSTRAINT: `auth.required_backends` reaches a backend's settings with
    `getattr(project.auth, backend)`, so every BACKEND NAME must be an attribute here -- and
    conversely, a non-backend field added here must not collide with one. `max_age_s` and
    `max_refreshes` are safe; a field called `earthdata_options` would not be.
    """
    model_config = {"extra": "forbid"}
    earthdata: EarthdataAuth | None = None
    gee: GeeAuth | None = None
    copernicus: CopernicusAuth | None = None

    # Credentials expire and runs are long. These bound the mid-run re-authentication the
    # acquisition stages do; see `coastal_sst_data.auth` for what each one prevents.
    max_age_s: float = 1800.0     # proactively re-login a credential older than this
    max_refreshes: int = 20       # per backend per run; exceeded -> a real failure, not a retry


class RuntimeSpec(BaseModel):
    """How much of the run happens at once.

    Nothing here changes WHAT a run produces -- only how many pieces of it are in flight.
    The defaults are the serial pipeline exactly as it has always behaved, so parallelism is
    opt-in and `jobs: 1` remains a true escape hatch.

    TWO LIMITS, because one number cannot express the constraint. `jobs` bounds the worker
    pool. `gates` bounds each SERVICE, and services differ by an order of magnitude in what
    they tolerate: Earthdata is content with several granule reads at once, while CMEMS hands
    out a dataset handle that is not safe to share at all and the NOAA metadata endpoints sit
    behind a module-global `requests.Session`. Which gate a product belongs to is declared on
    its ProductSpec (`products.ProductSpec.gate`); this sizes the buckets.

    The caps are config rather than constants because they move when the run moves: in-region
    on AWS, Earthdata tolerates far more than it does over a home link.
    """
    model_config = {"extra": "forbid"}

    # Concurrent ACQUISITION tasks, where a task is one (product, AoI). 1 == today's serial
    # path, taken verbatim rather than emulated.
    jobs: int = Field(1, ge=1)
    # Concurrent AoIs in the assemble/preprocess stages. Separate from `jobs`, and much
    # smaller by default, because these are bounded by MEMORY rather than by the network --
    # and the memory budget is DIVIDED between them (see datacube.budget_bytes). A run that
    # OOMs here has already done all the downloading, which is the expensive part to lose.
    assemble_jobs: int = Field(1, ge=1)
    # Per-service caps, overriding DEFAULT_GATES. Keys are `ProductSpec.gate` names.
    gates: dict[str, int] = Field(default_factory=dict)


# Starting caps per service. Tuned to what each endpoint tolerates, NOT to the machine:
#
#   earthdata   several granule reads at once are normal; `earthaccess.open()` already
#               threads internally, so the real connection count is a multiple of this.
#               CMR *search* is the throttled part, and there is one search per (product, AoI).
#   pc          anonymous -- there is no account and no per-account limit. Throttling is
#               per-IP, and the STAC search endpoint is the sensitive half.
#   copernicus  the lazy dataset handle carries its own client and is not safe to share; the
#               toolbox parallelises internally already.
#   herbie      hardened in this cycle (retry + a requests deadline); before that a stalled
#               mirror could hang a worker forever, which on a pool is unrecoverable.
#   noaa_small  small public metadata APIs, behind a module-global `requests.Session`.
#   erddap      same -- raise once `insitu_ioos` uses a thread-local session.
#   dem         the CUDEM tile index is one cache file shared by every AoI, refreshed with a
#               non-atomic expire-then-write.
DEFAULT_GATES: dict[str, int] = {
    "earthdata": 6,
    "pc": 4,
    "herbie": 4,
    "copernicus": 1,
    "noaa_small": 1,
    "erddap": 1,
    "dem": 1,
}


def gate_caps(project: "Project") -> dict[str, int]:
    """The effective per-service caps for a run: defaults, overridden by the config."""
    return {**DEFAULT_GATES, **{k: int(v) for k, v in project.runtime.gates.items()}}


# ---------------------------------------------------------------------------
# Auth requirements: which backend each product needs. DERIVED from the registry.
#
# A value is either:
#   * a backend name (str)         -- always requires that backend
#   * None / absent                -- public, needs no auth
#   * {source: backend | None}     -- source-selectable; backend depends on
#                                     products.<product>.source
# The validator here and the runtime auth layer (coastal_sst_data.auth) both read this, so
# a new product (or a new source for one) declares its auth ONCE, in its ProductSpec.
# ---------------------------------------------------------------------------
AUTH_REQUIREMENTS: dict[DataProduct, "str | None | dict[str, str | None]"] = {
    s.product: s.auth for s in REGISTRY if s.auth is not None
}

# Default `source` for source-selectable products (when the config omits it).
DEFAULT_SOURCE: dict[DataProduct, str] = {
    s.product: s.default_source for s in REGISTRY if s.default_source
}


def required_backend(product: DataProduct, opts) -> "str | None":
    """The auth backend a selected product needs, or None if it needs none.

    For a PICK-ONE (ACCESS) product, resolves via its `source` option. For a STACKED (DATA)
    product there is no single source to resolve -- it acquires every entry in its `sources`
    list -- so the requirement is the union over that list. Both validate the names here,
    failing loudly on a typo rather than at the first request.
    """
    req = AUTH_REQUIREMENTS.get(product)
    if isinstance(req, dict):
        s = spec(product)
        if s.is_stacked_data:
            key = s.sources_option
            # Unset -> the product's OWN default list, which is what the acquisition stage will
            # actually run. It used to fall back to every known source, which was invisible
            # while all of a product's sources were public: the union of Nones is None either
            # way. It stops being invisible the moment ONE source needs a credential -- the
            # preflight then demands that credential from every config with the product
            # selected and no explicit list, for a source the run will never touch.
            # `default_sources` empty keeps the old reading, for products where it is right.
            declared = (list(s.default_sources) if s.default_sources is not None
                        else list(req))
            names = getattr(opts, key, None) or declared
            if isinstance(names, str):
                names = [names]
            unknown = sorted(n for n in names if n not in req)
            if unknown:
                raise ValueError(
                    f"{product.value}.{key} {unknown} is not recognized; "
                    f"choose from {sorted(req)}.")
            backends = {req[n] for n in names if req[n]}
            if len(backends) > 1:
                # No product does this yet. Say so plainly rather than picking one and
                # leaving the other's credentials unverified until it fails mid-download.
                raise ValueError(
                    f"{product.value} stacks sources needing different credentials "
                    f"({sorted(backends)}), and the auth preflight resolves ONE backend per "
                    "product. Give them separate products, or widen required_backend(s) to "
                    "return a set.")
            return next(iter(backends), None)

        source = getattr(opts, "source", DEFAULT_SOURCE.get(product))
        if source not in req:
            raise ValueError(
                f"{product.value}.source {source!r} is not recognized; "
                f"choose from {sorted(req)}."
            )
        return req[source]
    return req


class Project(BaseModel):
    """Every AoI in the project, grouped into regions."""
    model_config = {"extra": "forbid"}
    name: str
    # Root directory where acquired/organized data is written.
    output_dir: Path
    time: TimeWindow
    # Which products to run + their GLOBAL options. Keys (the closed DataProduct
    # set) are the selection; dict keys are inherently unique, so no dup check.
    products: dict[DataProduct, ProductOptions] = Field(..., min_length=1)
    regions: list[Region] = Field(..., min_length=1)
    # All grid fields have defaults, so the whole block is optional.
    grid: GridSpec = Field(default_factory=GridSpec)
    # Datacube assembler settings; all default, so the block is optional.
    datacube: DataCubeSpec = Field(default_factory=DataCubeSpec)
    # Post-assembly preprocessing (opt-in); all default, so the block is optional.
    preprocess: PreprocessSpec = Field(default_factory=PreprocessSpec)
    # Point extraction (opt-in via `extract.channels`); all default, so the block is
    # optional -- a project that never extracts writes nothing here and pays nothing.
    extract: ExtractSpec = Field(default_factory=ExtractSpec)
    # Non-secret auth settings; required per selected product (see validator).
    auth: AuthConfig = Field(default_factory=AuthConfig)
    # How much runs at once. Defaults are the serial pipeline, so the block is optional.
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)


    @field_validator("output_dir")
    @classmethod
    def _expand_output_dir(cls, v: Path) -> Path:
        # Expand a leading `~` so downstream code gets a usable path. We do NOT
        # create the directory or require it to exist -- that's a runtime concern
        # for the pipeline, not config validation.
        return v.expanduser()

    @field_validator("products", mode="before")
    @classmethod
    def _fill_product_defaults(cls, v):
        return _options_by_product(v)

    @model_validator(mode="after")
    def _unique_names(self):
        rnames = [r.name for r in self.regions]
        if len(rnames) != len(set(rnames)):
            raise ValueError("region names must be unique")
        anames = [a.name for r in self.regions for a in r.areas]
        if len(anames) != len(set(anames)):
            raise ValueError("AoI names must be unique across the project")
        return self

    @model_validator(mode="after")
    def _auth_present_for_products(self):
        """Every selected product that needs auth must have it configured.

        Uniform over all products via the AUTH_REQUIREMENTS table -- ECOSTRESS/
        MUR/MODIS need `auth.earthdata`; Landsat/landcover depend on their
        `source`; public products need nothing. Fails loudly at load time.
        """
        for product, opts in self.products.items():
            backend = required_backend(product, opts)   # resolves source, validates it
            if backend and getattr(self.auth, backend) is None:
                raise ValueError(
                    f"product {product.value!r} requires `auth.{backend}` but it "
                    f"is not configured. Add an `auth.{backend}` section."
                )
        return self

    @model_validator(mode="after")
    def _sources_are_selected_products(self):
        """Every region `sources` entry must name a product in `products`.

        A source is a region-level option for a product the project acquires;
        options for a product that isn't selected are almost always a typo or a
        stale entry, so fail loudly rather than silently ignore them.
        """
        selected = set(self.products)
        for r in self.regions:
            unknown = set(r.sources) - selected
            if unknown:
                names = ", ".join(sorted(p.value for p in unknown))
                raise ValueError(
                    f"region {r.name!r} has sources for non-selected product(s): "
                    f"{names}. Add them to `products` or remove the source entries."
                )
        return self

    @model_validator(mode="after")
    def _option_keys_are_known(self):
        """Reject product/source options that no code reads.

        A key nothing reads is not harmless -- it is a config that LIES. It states an
        intent, the run ignores it, and the provenance honestly records what the run did,
        so nothing anywhere contradicts the config file. `bathymetry.source: cudem` was
        documented, settable, and silently discarded (the module reads `default_source`),
        which means a config asking for 3 m topobathy would quietly produce ~100 m GMRT.

        Failing at load time is the only point at which this is cheap to notice.
        """
        problems: list[str] = []

        def check(where: str, product: DataProduct, opts, allowed: set[str]):
            extra = set(getattr(opts, "model_extra", None) or {})
            unknown = sorted(extra - allowed)
            for key in unknown:
                hint = get_close_matches(key, sorted(allowed), n=1, cutoff=0.6)
                suggest = f" (did you mean {hint[0]!r}?)" if hint else ""
                problems.append(
                    f"{where}.{product.value}.{key} is not a recognised option{suggest}. "
                    f"Valid: {', '.join(sorted(allowed)) or '(none)'}")

        for product, opts in self.products.items():
            check("products", product, opts, PRODUCT_OPTIONS.get(product, set()))
        for r in self.regions:
            for product, opts in r.sources.items():
                check(f"regions[{r.name}].sources", product, opts,
                      REGION_OPTIONS.get(product, set()))

        if problems:
            raise ValueError(
                "unrecognised config option(s) -- these would be SILENTLY IGNORED:\n  "
                + "\n  ".join(problems))
        return self

    @model_validator(mode="after")
    def _stacked_source_lists_are_valid(self):
        """A DATA (stacked) product's `sources` must be a non-empty list of known sources.

        Distinct-data products (bathymetry) STACK the sources the user lists -- so an empty
        list means "acquire nothing" and an unknown name is a typo that would silently drop a
        DEM. Both fail here, at load time, rather than producing a cube missing a channel the
        config asked for. (Absent `sources` is fine: the module defaults to every known one.)
        """
        problems: list[str] = []

        # Some DATA products have an OPEN source set: a `datasets: {tag: id}` map registers
        # extra source tags (CMEMS regional models). Gather every registered tag per product,
        # from the global bag AND any region, so a tag defined anywhere is a valid source name.
        def _datasets(opts) -> set:
            d = (getattr(opts, "model_extra", None) or {}).get("datasets") or {}
            return set(d) if isinstance(d, dict) else set()

        registered: dict[DataProduct, set] = {}
        for product, opts in self.products.items():
            registered.setdefault(product, set()).update(_datasets(opts))
        for r in self.regions:
            for product, opts in r.sources.items():
                registered.setdefault(product, set()).update(_datasets(opts))

        def check(where: str, product: DataProduct, opts):
            s = spec(product)
            if not s.is_stacked_data:
                return
            # The key naming the stacked list is usually `sources`, but a product may name it
            # something truer to what its sources ARE (ECOSTRESS: `versions`). See
            # ProductSpec.sources_option.
            key = s.sources_option
            val = (getattr(opts, "model_extra", None) or {}).get(key)
            if val is None:
                return
            allowed = set(s.known_sources) | registered.get(product, set())
            names = [val] if isinstance(val, str) else list(val)
            if not names:
                problems.append(f"{where}.{product.value}.{key} is empty; list at least "
                                f"one of {sorted(allowed)}.")
            for name in names:
                if name not in allowed:
                    problems.append(f"{where}.{product.value}.{key} has unknown source "
                                    f"{name!r}; choose from {sorted(allowed)} (register a "
                                    "regional tag with `datasets: {tag: dataset_id}`).")

        for product, opts in self.products.items():
            check("products", product, opts)
        for r in self.regions:
            for product, opts in r.sources.items():
                check(f"regions[{r.name}].sources", product, opts)

        if problems:
            raise ValueError("invalid stacked-source list(s):\n  " + "\n  ".join(problems))
        return self

    @model_validator(mode="after")
    def _overpass_combinations_are_valid(self):
        """Overpass `(sensor, source)` combinations must pair a LOADED sensor with a valid
        source of the RIGHT product -- `met_overpass.combinations` against met sources,
        `tides.overpass_combinations` (D17) against tide sources.

        This is the sharpest silent-regression risk in the met/tide split: a combo naming a
        sensor that isn't selected, or a source typo, would otherwise quietly produce an empty
        overpass channel. Fail at load instead. (Absent = no overpass channels, which is fine.)
        """
        from .products import sensors as _sensors
        sensor_of = {s.sensor.prefix: s.product for s in _sensors()}    # prefix -> DataProduct
        problems: list[str] = []

        def check(where: str, opts, key: str, valid_sources: set):
            raw = (getattr(opts, "model_extra", None) or {}).get(key)
            if raw is None:
                return
            for combo in raw:
                try:
                    sensor, source = str(combo[0]), str(combo[1])
                except (TypeError, IndexError, KeyError):
                    problems.append(f"{where}: {combo!r} is not a [sensor, source] pair.")
                    continue
                if sensor not in sensor_of:
                    problems.append(f"{where}: sensor {sensor!r} is not a sensor "
                                    f"(choose from {sorted(sensor_of)}).")
                elif sensor_of[sensor] not in self.products:
                    problems.append(f"{where}: sensor {sensor!r} names product "
                                    f"{sensor_of[sensor].value!r}, which is not selected.")
                if source not in valid_sources:
                    problems.append(f"{where}: source {source!r} is not valid here "
                                    f"(choose from {sorted(valid_sources)}).")

        # (product, option key, the product whose sources the combo's source must belong to)
        for host, key, src_product in (
                (DataProduct.met_overpass, "combinations", DataProduct.met_overpass),
                (DataProduct.tides, "overpass_combinations", DataProduct.tides)):
            if host not in self.products:
                continue
            valid = set(spec(src_product).known_sources)
            check(f"products.{host.value}.{key}", self.products[host], key, valid)
            for r in self.regions:
                if host in r.sources:
                    check(f"regions[{r.name}].sources.{host.value}.{key}",
                          r.sources[host], key, valid)

        if problems:
            raise ValueError("invalid overpass combinations:\n  " + "\n  ".join(problems))
        return self

    @model_validator(mode="after")
    def _overpass_sensor_lists_are_valid(self):
        """`mur.overpass_sensors` must name LOADED sensors by their channel PREFIX.

        The prefix is `eco`, not `ecostress`: the filter finds a sensor's days by globbing the
        aligned tree that prefix names, so a product name matches nothing and MUR would restrict
        itself to ZERO days -- a run that downloads nothing and reports no error. An unselected
        sensor is the same failure by a different route. Both fail at load, which is what makes
        the runtime message ("the sensors have not run in this output dir") unambiguous: after
        this validator, an empty sensor tree cannot mean a typo.

        Absent = no restriction (every day, the default). An EMPTY list is rejected rather than
        read as either extreme.
        """
        from .products import sensors as _sensors
        sensor_of = {s.sensor.prefix: s.product for s in _sensors()}    # prefix -> DataProduct
        problems: list[str] = []

        def check(where: str, opts, key: str):
            raw = (getattr(opts, "model_extra", None) or {}).get(key)
            if raw is None:
                return
            names = [raw] if isinstance(raw, str) else list(raw)
            if not names:
                problems.append(f"{where} is empty; omit the key to fetch every day, or list "
                                f"at least one of {sorted(sensor_of)}.")
            for name in names:
                name = str(name)
                if name not in sensor_of:
                    hint = get_close_matches(name, sorted(sensor_of), n=1, cutoff=0.4)
                    suggest = f" (did you mean {hint[0]!r}?)" if hint else ""
                    problems.append(f"{where}: {name!r} is not a sensor{suggest}; choose from "
                                    f"{sorted(sensor_of)}.")
                elif sensor_of[name] not in self.products:
                    problems.append(f"{where}: {name!r} names product "
                                    f"{sensor_of[name].value!r}, which is not selected -- it "
                                    "would never write the overpasses to filter on.")

        # (product, option key). One entry today; a table so a second product adopting the
        # same key does not reopen this validator.
        for host, key in ((DataProduct.mur, "overpass_sensors"),):
            if host not in self.products:
                continue
            check(f"products.{host.value}.{key}", self.products[host], key)
            for r in self.regions:
                if host in r.sources:
                    check(f"regions[{r.name}].sources.{host.value}.{key}",
                          r.sources[host], key)

        if problems:
            raise ValueError("invalid overpass sensor list(s):\n  " + "\n  ".join(problems))
        return self

    # What this project was LOADED FROM. Kept so an assembled datacube can embed the
    # exact config that produced it -- a cube whose config has since been edited, moved,
    # or deleted is still reproducible from its own attrs. Private (not config fields), so
    # `extra="forbid"` and round-tripping are untouched.
    _config_path: str | None = PrivateAttr(default=None)
    _config_text: str | None = PrivateAttr(default=None)

    @property
    def config_path(self) -> str | None:
        """Path the config was read from; None when built from a dict."""
        return self._config_path

    @property
    def config_text(self) -> str:
        """The config's YAML text.

        For a project built from a dict (parse_config) there is no file, so the validated
        model is serialized back to YAML -- a programmatically-built project stays just as
        self-describing as a file-backed one.
        """
        if self._config_text is None:
            self._config_text = yaml.safe_dump(
                self.model_dump(mode="json", exclude_defaults=True), sort_keys=False)
        return self._config_text

    @property
    def config_sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self.config_text.encode("utf-8")).hexdigest()

    @property
    def all_areas(self) -> list[AreaOfInterest]:
        return [a for r in self.regions for a in r.areas]

    def region_of(self, area_name: str) -> Region:
        for r in self.regions:
            if any(a.name == area_name for a in r.areas):
                return r
        raise KeyError(area_name)
    
# ---- Load and parse config file ---------------------------
def parse_config(data: dict[str, Any]) -> Project:
    """Validate an already-parsed dict into a Project"""
    return Project(**data)


def load_config(path: str | Path) -> Project:
    """Read a YAML file from disk and validate it into a Project."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}  # `or {}` handles an empty file

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a mapping/dict, got {type(raw).__name__}"
        )

    project = parse_config(raw)
    # Keep the file VERBATIM, not a re-serialization: the cube should embed what you
    # actually wrote, comments and all.
    project._config_text = path.read_text(encoding="utf-8")
    project._config_path = str(path.resolve())
    return project


if __name__ == "__main__":
    
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "examples/config.test.yaml"
    try:
        cfg = load_config(cfg_path)
    except ValidationError as exc:
        # ValidationError prints a readable, field-by-field summary of what's wrong.
        print(f"Config is invalid:\n{exc}")
        sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not load config: {exc}")
        sys.exit(1)

    print("Config loaded and validated:")
    print(cfg.model_dump_json(indent=2))




# # ---------------------------------------------------------------------------
# # 3. Credentials — read from the environment, never from the config file
# # ---------------------------------------------------------------------------


# class Credentials(BaseSettings):
#     """Secrets pulled from environment variables.

#     With env_prefix="COASTAL_DATA_", the fields below are read from COASTAL_DATA_API_KEY and
#     COASTAL_DATA_DB_PASSWORD. A local .env file is also read if present (handy for dev)
#     — add it to .gitignore so secrets never get committed.
#     """

#     model_config = SettingsConfigDict(
#         env_prefix="COASTAL_DATA_",
#         env_file=".env",
#         extra="ignore",
#     )

#     api_key: str  # required: fails if COASTAL_DATA_API_KEY is unset
#     db_password: str | None = None  # optional


# # ---------------------------------------------------------------------------
# # 4. Loading / parsing functions
# # ---------------------------------------------------------------------------


# def parse_config(data: dict[str, Any]) -> AppConfig:
#     """Validate an already-parsed dict into an AppConfig.

#     Kept separate from file reading so it's trivial to unit-test with an
#     in-memory dict (no temp files needed).
#     """
#     return AppConfig(**data)


# def load_config(path: str | Path) -> AppConfig:
#     """Read a YAML file from disk and validate it into an AppConfig."""
#     path = Path(path)
#     if not path.is_file():
#         raise FileNotFoundError(f"Config file not found: {path}")

#     with path.open("r", encoding="utf-8") as f:
#         raw = yaml.safe_load(f) or {}  # `or {}` handles an empty file

#     if not isinstance(raw, dict):
#         raise ValueError(
#             f"Config root must be a mapping/dict, got {type(raw).__name__}"
#         )

#     return parse_config(raw)


# def load_credentials() -> Credentials:
#     """Load secrets from the environment (and .env if present)."""
#     return Credentials()


