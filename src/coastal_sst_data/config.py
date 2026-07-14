"""Configuration loading and validation.

This module does three jobs:

1. **Describe** what a valid config looks like (the Pydantic *models* below).
2. **Load** a YAML file from disk and turn it into a validated object.
3. **Read secrets** (API keys, passwords) from environment variables, kept
   completely separate from the config file so credentials never live in the repo.

"""

from __future__ import annotations

import logging
from datetime import date
from enum import Enum
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
class DataProduct(str, Enum):
    """A data stream the pipeline can acquire and organize onto the AoI grids.

    Values must match the acquisition process names. Unknown names in the config
    fail validation (a typo like `ecostres` is rejected, not silently skipped).
    """
    bathymetry = "bathymetry"
    ecostress = "ecostress"
    mur = "mur"
    cmems = "cmems"
    landsat = "landsat"
    modis = "modis"
    met = "met"
    tides = "tides"
    landcover = "landcover"
    insitu = "insitu"


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
# Which options each product actually READS
# --------------------------------------------------------------------------- #
# A key that is not in this registry does NOTHING. It used to be accepted in silence, which
# is the worst of both worlds: the config says one thing, the run does another, and the
# provenance faithfully records the run -- so the lie lives in the config file, where nobody
# looks. `bathymetry.source` was exactly this: documented, set, and never read (the module
# reads `default_source`), so a config asking for 3 m CUDEM would quietly get ~100 m GMRT.
#
# Keep these in step with the `_opt(opts, "<key>", ...)` calls in each process module.
_COMMON = {"output_format", "overwrite"}

PRODUCT_OPTIONS: dict[DataProduct, set[str]] = {
    DataProduct.bathymetry: _COMMON | {
        "source", "default_source", "fallback", "pad_deg", "layer", "resolution",
        "stats_subgrid_m", "min_cudem_cover", "cudem_urllist", "cudem_index_cache"},
    DataProduct.mur: _COMMON | {"short_name", "variable", "pad_deg"},
    DataProduct.modis: _COMMON | {
        "short_name", "variable", "quality_min", "regrid_radius_m", "access",
        "match_landsat", "max_time_diff_minutes", "daytime_only", "footprint_id"},
    DataProduct.ecostress: _COMMON | {"short_name", "version", "layers", "categorical"},
    DataProduct.landsat: _COMMON | {
        "source", "collection", "stac_url", "platforms", "cloud_cover_max", "masking"},
    DataProduct.cmems: _COMMON | {
        "source", "fallback", "dataset_id", "variables", "depths", "pad_deg"},
    DataProduct.met: _COMMON | {
        "source", "fallback", "variables", "model", "product", "fxx", "era5_zarr",
        "regrid_radius_m", "pad_deg", "reference_time", "reference_basis",
        "daily_mean_hours", "overpass_from"},
    DataProduct.tides: _COMMON | {
        "source", "default_source", "fallback", "model", "model_directory", "interval",
        "stations", "warn_distance_km", "fallback_distance_km"},
    DataProduct.landcover: _COMMON | {
        "source", "collection", "stac_url", "year", "water_classes"},
    DataProduct.insitu: _COMMON | {
        "source", "variables", "stations", "exclude_stations", "qc_flags", "pad_deg",
        "max_sensor_depth_m"},
}

# --------------------------------------------------------------------------- #
# Region-level overrides: what genuinely varies GEOGRAPHICALLY
# --------------------------------------------------------------------------- #
# The line these draw is deliberate and load-bearing:
#
#   "WHICH SOURCE HAS COVERAGE HERE"  is region-varying.   -> allowed below
#   "WHAT THE CUBE MEANS"             is not.              -> project-global only
#
# The first is a fact about the world and it changes as the project expands: HRRR is North
# America only, CO-OPS gauges are U.S. only, CUDEM is CONUS only, IOOS is North America,
# and CMEMS publishes distinct regional models. A project spanning two continents CANNOT
# name one source for those and be telling the truth, which is the whole reason this block
# exists.
#
# The second must stay uniform, and the validator enforces it. `variables`, `depths`,
# `qc_flags`, `reference_time` and friends decide the cube's CHANNEL SET and what each
# channel means. Let a region override those and two AoIs' cubes silently stop being
# comparable -- and for met it is worse than incomparable, because the assembler names
# airtemp/wind_*/swrad/cloud_cover explicitly, so a region that dropped one would ship an
# all-NaN channel that looks exactly like a forcing that was fetched and came back empty.
#
# So: source selectors, their fallbacks, the dataset/collection/model they resolve to, the
# station lists, and the datum. Nothing that reshapes a channel.
REGION_OPTIONS: dict[DataProduct, set[str]] = {
    # CUDEM is CONUS-only -> a region outside it must name its own DEM. `datum_offset_m`
    # is region-only (it has no project-level counterpart): the offset is a property of the
    # DEM that WON and of where the AoI is, so no project-wide constant can be right.
    DataProduct.bathymetry: {"source", "dem_source", "fallback", "datum_offset_m"},
    # CO-OPS gauges are U.S.-only -> elsewhere, a global model + its downloaded directory.
    DataProduct.tides: {"source", "model", "model_directory", "stations"},
    # HRRR is North America only. Outside it the chain MUST start at era5, and saying so
    # per region is the difference between a deliberate choice and a silent fallback.
    DataProduct.met: {"source", "fallback", "model"},
    # CMEMS publishes regional models (Baltic, Mediterranean, NW Shelf) alongside the
    # global one; which is right is a fact about where the AoI is.
    DataProduct.cmems: {"source", "fallback", "dataset_id"},
    # IOOS covers North America; another continent needs another network. `variables` is a
    # per-NETWORK naming preference (sea_water_temperature vs sea_surface_temperature), not
    # a channel choice -- every network still yields the one `insitu_sst` channel -- so it
    # is safe here. Station lists are inherently local.
    DataProduct.insitu: {"source", "stations", "exclude_stations", "variables"},
    # The source and the catalogue it is served from move together, so they travel together.
    DataProduct.landsat: {"source", "collection", "stac_url"},
    DataProduct.landcover: {"source", "collection", "stac_url"},
    # mur, modis, ecostress: global products with no regional variation. A region key here
    # would be a no-op, so there is none.
}

# Keys a region may set that have NO project-level counterpart. Everything else in
# REGION_OPTIONS must be an option the product actually reads (see the invariant below), so
# a region cannot override a key the module will never look at.
REGION_ONLY_OPTIONS: dict[DataProduct, set[str]] = {
    DataProduct.bathymetry: {"dem_source", "datum_offset_m"},
}


def _check_region_options_invariant() -> None:
    """A region key must be one the product READS -- or an explicit region-only key.

    Checked at import, because the failure it prevents is invisible at runtime: a region
    override that no module reads does nothing, the run proceeds, and the config file is
    left stating an intent the data does not honour. That is the same class of lie
    `_option_keys_are_known` exists to catch, one level up.
    """
    for product, keys in REGION_OPTIONS.items():
        allowed = PRODUCT_OPTIONS.get(product, set()) | REGION_ONLY_OPTIONS.get(product, set())
        orphans = keys - allowed
        if orphans:
            raise RuntimeError(
                f"REGION_OPTIONS[{product.value}] names {sorted(orphans)}, which "
                f"{product.value} does not read. Add them to PRODUCT_OPTIONS (and read them "
                f"in the module), or to REGION_ONLY_OPTIONS if they are region-only.")


_check_region_options_invariant()


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


def _options_by_product(value: Any) -> Any:
    """Normalize a `{product: options}` mapping before validation.

    A bare `product:` (null value) is treated as 'selected, with default
    options' -> {}, so listing a product with no options is just its name.
    """
    if isinstance(value, dict):
        return {k: ({} if v is None else v) for k, v in value.items()}
    return value


class Region(BaseModel):
    """AoIs that share the same region-dependent data sources."""
    model_config = {"extra": "forbid"}
    name: str
    # Region-dependent, per-product source options (e.g. bathymetry.dem_source).
    sources: dict[DataProduct, SourceOptions] = Field(default_factory=dict)
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

    Every field has a default, so the whole `datacube:` block is optional. The
    water/land mask is derived from land-cover (authoritative where known), so
    there are no elevation/depth threshold knobs here.
    """
    model_config = {"extra": "forbid"}
    chunks: dict[str, int] = Field(default_factory=lambda: {"time": 64, "y": 128, "x": 128})
    fill_mur_water: bool = True                # NN-fill the MUR backbone over land-cover water
    # Same for the CMEMS ocean-model channels: at ~9 km its land mask can swallow a
    # whole estuary, so fill those cells from the nearest resolved water column.
    fill_cmems_water: bool = True
    # Emit the per-sensor water-level fields (bathymetry + tide re-referenced to the
    # tide-adjusted waterline at each overpass). Needs both products. The DEM->MSL
    # datum offset is RESOLVED automatically by the `datum` stage (processes.datum),
    # not configured: it is a property of the DEM that actually ran and of where the
    # AoI is, so no project-wide constant can be right across a study area. The only
    # knob is an optional per-region override, regions[].sources.bathymetry.datum_offset_m.
    water_level: bool = True
    # Which met file feeds the cube's met channels (airtemp, wind_*, swrad, cloud_cover):
    #   "reference"  -- the daily snapshot at products.met.reference_time (default 10:30
    #                   local solar, Landsat's overpass). One time of day, every day.
    #   "daily_mean" -- the mean over products.met.daily_mean_hours (the old behaviour),
    #                   which smears the diurnal cycle.
    # Falls back to the other if the chosen file is absent.
    met_time: Literal["reference", "daily_mean"] = "reference"
    # Met variables ALSO emitted at each sensor's own overpass, as <sensor>_<var>
    # (e.g. eco_airtemp, lst_wind_speed) -- so a scene is paired with the forcing that
    # was actually present when it was taken. [] disables. Needs products.met.overpass_from
    # to include the sensor, so the snapshots exist.
    overpass_met: list[str] = Field(
        default_factory=lambda: ["airtemp", "wind_speed", "swrad", "cloud_cover"])
    # Emit the in-situ channels: the buoy value at the reference time, plus a matchup at
    # each sensor's own overpass (the ground truth a satellite scene is validated against).
    insitu: bool = True
    # Matchup tolerance, minutes: beyond this an overpass gets NaN rather than a stale
    # observation -- a buoy reading two hours off is not a matchup.
    insitu_max_dt_min: float = 60.0
    compression: CompressionSpec = Field(default_factory=CompressionSpec)
    output_subdir: str = "datacube"            # cube dir under output_dir
    overwrite: bool = False                    # rebuild existing <aoi>.zarr cubes


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
    """Non-secret authentication settings, one block per backend."""
    model_config = {"extra": "forbid"}
    earthdata: EarthdataAuth | None = None
    gee: GeeAuth | None = None
    copernicus: CopernicusAuth | None = None


# ---------------------------------------------------------------------------
# Auth requirements: the SINGLE source of truth for which backend each product
# needs. One table + one resolver replace the old per-product special-casing.
#
# A value is either:
#   * a backend name (str)         -- always requires that backend
#   * None / absent                -- public, needs no auth
#   * {source: backend | None}     -- source-selectable; backend depends on
#                                     products.<product>.source
# Adding a product (or a new source) is a one-line edit here; the validator and
# the runtime auth layer (coastal_sst_data.auth) both read this table, so nothing
# else needs to change.
# ---------------------------------------------------------------------------
AUTH_REQUIREMENTS: dict[DataProduct, "str | None | dict[str, str | None]"] = {
    DataProduct.ecostress: "earthdata",
    DataProduct.mur: "earthdata",
    DataProduct.modis: "earthdata",
    DataProduct.cmems: "copernicus",
    # Source-selectable: PC/ESA (anonymous) and AWS (env creds) need no config
    # auth; only the GEE source needs auth.gee.
    DataProduct.landsat: {"pc": None, "planetary_computer": None, "aws": None, "gee": "gee"},
    DataProduct.landcover: {"esa": None, "worldcover": None, "gee": "gee"},
    # In-situ networks are public (IOOS ERDDAP needs no key); a future network that
    # does need one adds it here as {source: backend}.
    DataProduct.insitu: {"ioos": None},
    # bathymetry, met, tides: absent -> public, no auth.
}

# Default `source` for source-selectable products (when the config omits it).
DEFAULT_SOURCE: dict[DataProduct, str] = {
    DataProduct.landsat: "pc",
    DataProduct.landcover: "esa",
    DataProduct.insitu: "ioos",
}


def required_backend(product: DataProduct, opts) -> "str | None":
    """The auth backend a selected product needs, or None if it needs none.

    For source-selectable products, resolves via the product's `source` option
    (validated against the known set here, failing loudly on a typo).
    """
    req = AUTH_REQUIREMENTS.get(product)
    if isinstance(req, dict):
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
    # Non-secret auth settings; required per selected product (see validator).
    auth: AuthConfig = Field(default_factory=AuthConfig)


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


