#!/usr/bin/env python3
"""
coastal_sst_data -- the product registry: ONE declaration per data product.

Adding a data product used to mean editing about a dozen hand-maintained tables spread
across five files -- the DataProduct enum, PRODUCT_OPTIONS, REGION_OPTIONS,
AUTH_REQUIREMENTS, DEFAULT_SOURCE, the pipeline's PROCESSES + source registries +
PROCESS_ORDER, store.REQUIRED_VARS, the assembler's PRODUCT_DIRS + DAILY_CHANNELS, and
provenance's SENSORS + field map. Miss any one and you fail DIFFERENTLY: forget
REQUIRED_VARS and the skip guard cannot see a truncated file; forget the provenance map and
the channel ships with a blank source record; forget DAILY_CHANNELS and coverage silently
stops being checked. None of those raise. They just quietly do less than you asked.

The tell that this had already gone wrong is the alias tables it grew: `_DIR_KEY_ALIASES`
and `_COVERAGE_ALIASES` both existed for exactly one reason -- one table called the tide
product `tide` and another called it `tides`. Two independent workarounds for two
hand-maintained lists disagreeing about a name. With one declaration there is one name, and
both aliases are gone.

So a product is declared ONCE, here, and every registry is DERIVED from it.

WHY THIS FILE IMPORTS NOTHING FROM THE PACKAGE
----------------------------------------------
The natural place for a spec is next to its module (`SPEC = ProductSpec(...)` at the bottom
of processes/mur.py). That cannot work: `config` needs the options/auth tables at VALIDATION
time, and every process module imports `config` -- so registering from inside the process
modules closes an import cycle. This module therefore sits BELOW config and imports nothing
internal; a spec names its module as a dotted STRING, resolved lazily at dispatch:

    products.py   DataProduct, ProductSpec, SensorSpec, REGISTRY   (no internal imports)
         v
    config.py     derives PRODUCT_OPTIONS / REGION_OPTIONS / AUTH_REQUIREMENTS /
                  DEFAULT_SOURCE; re-exports DataProduct
         v
    store / provenance / grid / report
         v
    processes/*   import config
         v
    pipeline / datacube    resolve spec.module by dotted path

Adding a product is now: write the module, add a ProductSpec here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataProduct(str, Enum):
    """A data stream the pipeline can acquire and organize onto the AoI grids.

    Values must match the acquisition process names. Unknown names in the config fail
    validation (a typo like `ecostres` is rejected, not silently skipped).
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


class Kind(str, Enum):
    """How a product's aligned output is SHAPED -- which decides how the assembler reads it.

    This is about the output's structure, not its subject: MUR and CMEMS are both
    DAILY_RASTER (one file per day on the AoI grid) even though one is a satellite analysis
    and the other an ocean model.
    """
    DAILY_RASTER = "daily_raster"        # <aoi>_<YYYYMMDD>.nc      -- MUR, CMEMS, met
    OVERPASS_SENSOR = "overpass_sensor"  # <aoi>_<YYYYMMDDThhmmss>.nc -- ECOSTRESS, Landsat, MODIS
    STATIC_RASTER = "static_raster"      # <aoi>.nc, no time dim    -- bathymetry, land-cover
    SERIES_1D = "series_1d"              # <aoi>_tides.nc, dims (time,) -- tides
    STATION_TABLE = "station_table"      # <aoi>_insitu.nc, dims (station, time) -- in-situ


@dataclass(frozen=True)
class SensorSpec:
    """A per-overpass thermal sensor, and how to read its validity.

    These fields ARE the arguments `datacube.load_clearest_overpass` already takes -- they
    were passed as three hand-written call sites, one per sensor. Declaring them makes the
    sensor family a LOOP, so a fourth sensor gets its `<prefix>_sst`, `_cloud`, `_valid`,
    `_hour`, `_water_elev`, `_water_class`, `_tide`, its overpass-met snapshots and its
    in-situ matchups for free -- every one of those channel names is already generated from
    the prefix.
    """
    prefix: str                                  # eco | lst | modis -- names every channel
    water_is_land: bool = False                  # ECOSTRESS's water layer has inverted polarity
    use_cloud: bool = True                       # Landsat's cloud mask is reliable...
    qc_levels: tuple[int, ...] | None = None     # ...ECOSTRESS's over-masks cold water: gate on QC
    trust_valid: bool = False                    # MODIS is quality-filtered upstream already


@dataclass(frozen=True)
class ProductSpec:
    """Everything the rest of the package needs to know about one data product."""

    product: DataProduct
    # The ALLCAPS output folder: <output_dir>/<dir>/aligned/<aoi>/. Held here rather than
    # inferred from the product name because they genuinely differ -- the `tides` product
    # writes to `TIDE/`, which is the whole reason two alias tables used to exist.
    dir: str
    kind: Kind

    # --- implementation -------------------------------------------------- #
    # Exactly one of these. `module` for a product with a single implementation;
    # `sources` for a source-selectable one, mapping each source name to its module (None =
    # the source is a recognised name with no implementation yet, e.g. Landsat via AWS).
    # Dotted strings, resolved lazily -- see the import-cycle note in the module docstring.
    module: str | None = None
    sources: dict[str, str | None] | None = None
    default_source: str | None = None

    # --- config surface -------------------------------------------------- #
    # Which options the module actually READS. A key not listed here does NOTHING, and
    # accepting it in silence is a config that LIES -- so config validation rejects it.
    options: frozenset[str] = frozenset()
    # Which of those a REGION may override. The line is deliberate and enforced:
    #   "which source has coverage here" -> yes.   "what the cube means" -> no.
    # (See config.REGION_OPTIONS for why letting a region reshape a channel is fatal.)
    region_options: frozenset[str] = frozenset()
    # Keys a region may set that have no project-level counterpart (bathymetry.datum_offset_m).
    region_only_options: frozenset[str] = frozenset()

    # --- auth ------------------------------------------------------------ #
    # A backend name, None (public), or {source: backend|None} for a source-selectable
    # product whose auth depends on which source is chosen.
    auth: "str | None | dict[str, str | None]" = None

    # --- durability ------------------------------------------------------ #
    # The variables a FINISHED file of this product must carry. A file missing one is not
    # "a file with a gap" -- it is a write, or a set of source layers, that did not
    # complete, and it must be re-fetched rather than skipped. Empty means the channel set
    # is config-dependent (met, CMEMS), so the invariant is only "it opens and holds >= 1
    # data variable"; truncated payloads in those are caught by the deep `check` pass.
    required_vars: tuple[str, ...] = ()

    # --- ordering -------------------------------------------------------- #
    # Products that must run BEFORE this one, and why:
    #   modis -> landsat   its coincidence filter reads Landsat's aligned files
    #   met   -> sensors   its overpass snapshots are taken at times read from their dirs
    # Declared as a dependency rather than a position in a hand-kept list, so adding a
    # product never means reasoning about a global order (see pipeline.process_order).
    depends_on: tuple[DataProduct, ...] = ()

    # --- assembly -------------------------------------------------------- #
    sensor: SensorSpec | None = None
    # The cube channel that proves this product produced data on a given day, for the
    # coverage check. Only DAILY products can be judged this way: an overpass sensor with no
    # scene on a day is normal, not a defect, and warning about it would train the user to
    # ignore the warning.
    coverage_channel: str | None = None
    # The product(s) a cube field built from this one should be attributed to. Usually just
    # itself; a derived field names all its inputs (see provenance.field_inputs).
    provenance_inputs: tuple[str, ...] = ()

    def module_for(self, source: str | None = None) -> str | None:
        """The dotted module path serving this product for a given source.

        None means "no implementation": either the product has none at all, or the named
        source is recognised but unimplemented. The caller reports it and skips -- never
        silently drops it, because an absent product looks exactly like one that found no
        data.
        """
        if self.sources is None:
            return self.module
        return self.sources.get(source or self.default_source)

    @property
    def known_sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.sources)) if self.sources else ()


# --------------------------------------------------------------------------- #
# THE REGISTRY
#
# Declaration order is the DEFAULT run order; `depends_on` corrects it where correctness
# requires (see pipeline.process_order, which topologically sorts this stably -- so this
# list reads as "statics and backbone first" while the hard constraints are enforced
# rather than merely hoped for).
# --------------------------------------------------------------------------- #
_COMMON = frozenset({"output_format", "overwrite"})

_PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

REGISTRY: tuple[ProductSpec, ...] = (

    ProductSpec(
        product=DataProduct.bathymetry,
        dir="BATHYMETRY",
        kind=Kind.STATIC_RASTER,
        module="coastal_sst_data.processes.bathymetry",
        options=_COMMON | {
            "source", "default_source", "fallback", "pad_deg", "layer", "resolution",
            "stats_subgrid_m", "min_cudem_cover", "cudem_urllist", "cudem_index_cache"},
        # CUDEM is CONUS-only, so an AoI outside it must name its own DEM.
        region_options=frozenset({"source", "dem_source", "fallback", "datum_offset_m"}),
        region_only_options=frozenset({"dem_source", "datum_offset_m"}),
        required_vars=("elevation", "depth", "depth_p25", "depth_p75"),
        provenance_inputs=("bathymetry",),
    ),

    ProductSpec(
        product=DataProduct.mur,
        dir="MUR",
        kind=Kind.DAILY_RASTER,
        module="coastal_sst_data.processes.mur",
        auth="earthdata",
        options=_COMMON | {"short_name", "variable", "pad_deg"},
        required_vars=("sst", "valid"),
        coverage_channel="mur_sst",
        provenance_inputs=("mur",),
    ),

    ProductSpec(
        product=DataProduct.cmems,
        dir="CMEMS",
        kind=Kind.DAILY_RASTER,
        module="coastal_sst_data.processes.cmems",
        auth="copernicus",
        options=_COMMON | {
            "source", "fallback", "dataset_id", "variables", "depths", "pad_deg"},
        # CMEMS publishes regional models (Baltic, Mediterranean, NW Shelf) alongside the
        # global one; which is right is a fact about where the AoI is.
        region_options=frozenset({"source", "fallback", "dataset_id"}),
        # + >= 1 configured variable, asserted by the >=1-data-var rule (the channel set is
        # config-dependent, so it cannot be named up front).
        required_vars=("valid",),
        # coverage_channel is discovered from the files (cmems_*), not named here.
        provenance_inputs=("cmems",),
    ),

    ProductSpec(
        product=DataProduct.ecostress,
        dir="ECOSTRESS",
        kind=Kind.OVERPASS_SENSOR,
        module="coastal_sst_data.processes.ecostress",
        auth="earthdata",
        options=_COMMON | {"short_name", "version", "layers", "categorical"},
        required_vars=("sst", "water", "cloud", "valid"),
        # ECOSTRESS's water layer has inverted polarity, and its cloud mask over-masks cold
        # water -- so validity is gated on the QC mandatory-QA bits instead.
        sensor=SensorSpec(prefix="eco", water_is_land=True, use_cloud=False,
                          qc_levels=(0, 1)),
        provenance_inputs=("ecostress",),
    ),

    ProductSpec(
        product=DataProduct.landsat,
        dir="LANDSAT",
        kind=Kind.OVERPASS_SENSOR,
        # Source-selectable. Only the free/anonymous Planetary Computer path is implemented;
        # `aws` and `gee` are recognised names awaiting a module (so a config naming one
        # fails as "not implemented" rather than as "unknown source").
        sources={
            "pc": "coastal_sst_data.processes.landsat_pc",
            "planetary_computer": "coastal_sst_data.processes.landsat_pc",
            "aws": None,
            "gee": None,
        },
        default_source="pc",
        # PC/AWS need no config auth; only the GEE source does.
        auth={"pc": None, "planetary_computer": None, "aws": None, "gee": "gee"},
        options=_COMMON | {
            "source", "collection", "stac_url", "platforms", "cloud_cover_max", "masking"},
        region_options=frozenset({"source", "collection", "stac_url"}),
        required_vars=("sst", "water", "cloud", "valid"),
        # Landsat's QA_PIXEL-based cloud mask is reliable, so validity gates on it.
        sensor=SensorSpec(prefix="lst", water_is_land=False, use_cloud=True),
        provenance_inputs=("landsat",),
    ),

    ProductSpec(
        product=DataProduct.modis,
        dir="MODIS",
        kind=Kind.OVERPASS_SENSOR,
        module="coastal_sst_data.processes.modis",
        auth="earthdata",
        options=_COMMON | {
            "short_name", "variable", "quality_min", "regrid_radius_m", "access",
            "match_landsat", "max_time_diff_minutes", "daytime_only", "footprint_id"},
        required_vars=("sst", "valid"),
        # MODIS coincidence (match_landsat) reads the Landsat aligned files, so Landsat must
        # have run first. This is WHY the old PROCESS_ORDER put landsat before modis; here
        # it is a constraint the sort enforces rather than a comment on a hand-kept list.
        depends_on=(DataProduct.landsat,),
        # Already quality-filtered upstream -> trust the file's own `valid` layer; it has no
        # water or cloud layer to recompute from.
        sensor=SensorSpec(prefix="modis", trust_valid=True),
        provenance_inputs=("modis",),
    ),

    ProductSpec(
        product=DataProduct.met,
        dir="MET",
        kind=Kind.DAILY_RASTER,
        module="coastal_sst_data.processes.met",
        options=_COMMON | {
            "source", "fallback", "variables", "model", "product", "fxx", "era5_zarr",
            "regrid_radius_m", "pad_deg", "reference_time", "reference_basis",
            "daily_mean_hours", "overpass_from"},
        # HRRR is North America only: outside it the chain MUST start at ERA5, and saying so
        # per region is the difference between a deliberate choice and a silent fallback.
        region_options=frozenset({"source", "fallback", "model"}),
        required_vars=(),          # channel set is config-dependent (see ProductSpec)
        # Its overpass snapshots are taken at times read from the sensors' aligned dirs, so
        # every sensor must have run first.
        depends_on=(DataProduct.ecostress, DataProduct.landsat, DataProduct.modis),
        coverage_channel="airtemp",
        provenance_inputs=("met",),
    ),

    ProductSpec(
        product=DataProduct.tides,
        dir="TIDE",                # NOT "TIDES" -- the one name that used to need two aliases
        kind=Kind.SERIES_1D,
        module="coastal_sst_data.processes.tides",
        options=_COMMON | {
            "source", "default_source", "fallback", "model", "model_directory", "interval",
            "stations", "warn_distance_km", "fallback_distance_km"},
        # CO-OPS gauges exist only in U.S. waters -> elsewhere, a global model + its
        # downloaded directory, which is a property of the machine and the region.
        region_options=frozenset({"source", "model", "model_directory", "stations"}),
        required_vars=("tide",),
        coverage_channel="tide",
        provenance_inputs=("tides",),
    ),

    ProductSpec(
        product=DataProduct.landcover,
        dir="LANDCOVER",
        kind=Kind.STATIC_RASTER,
        sources={
            "esa": "coastal_sst_data.processes.landcover_esa",
            "worldcover": "coastal_sst_data.processes.landcover_esa",
            "gee": None,           # JRC + NDWI water mask: a future landcover_gee module
        },
        default_source="esa",
        auth={"esa": None, "worldcover": None, "gee": "gee"},
        options=_COMMON | {"source", "collection", "stac_url", "year", "water_classes"},
        region_options=frozenset({"source", "collection", "stac_url"}),
        required_vars=("landcover", "water"),
        provenance_inputs=("landcover",),
    ),

    ProductSpec(
        product=DataProduct.insitu,
        dir="INSITU",
        kind=Kind.STATION_TABLE,
        # IOOS (ERDDAP) covers most of North America; another network registers here and
        # honours the same output contract.
        sources={"ioos": "coastal_sst_data.processes.insitu_ioos"},
        default_source="ioos",
        auth={"ioos": None},       # public: IOOS ERDDAP needs no key
        options=_COMMON | {
            "source", "variables", "stations", "exclude_stations", "qc_flags", "pad_deg",
            "max_sensor_depth_m"},
        # Station lists are inherently local; `variables` is a per-NETWORK naming preference
        # (sea_water_temperature vs sea_surface_temperature), not a channel choice.
        region_options=frozenset({"source", "stations", "exclude_stations", "variables"}),
        required_vars=("sst", "qc"),
        provenance_inputs=("insitu",),
    ),
)


BY_PRODUCT: dict[DataProduct, ProductSpec] = {s.product: s for s in REGISTRY}


def spec(product: DataProduct) -> ProductSpec:
    return BY_PRODUCT[product]


def sensors() -> tuple[ProductSpec, ...]:
    """The per-overpass thermal sensors, in registry order.

    The three hand-written `("eco", "lst", "modis")` tuples in the assembler become this.
    """
    return tuple(s for s in REGISTRY if s.sensor is not None)


# --------------------------------------------------------------------------- #
# Derived stages: not products, so not selectable in a config, but they own an output
# directory that provenance and the assembler must be able to find.
# --------------------------------------------------------------------------- #
DERIVED_DIRS: dict[str, str] = {"datum": "DATUM"}


def product_dirs() -> dict[str, str]:
    """{key: ALLCAPS dir} for every product AND derived stage that writes aligned output.

    Keyed by the product's own name throughout -- so `tides` is `tides` here, in provenance,
    and in the coverage report. The two alias tables that existed because one table said
    `tide` and another said `tides` are gone.
    """
    return {s.product.value: s.dir for s in REGISTRY} | DERIVED_DIRS


# --------------------------------------------------------------------------- #
# Invariants -- checked at import, because every one of these fails SILENTLY at runtime
# --------------------------------------------------------------------------- #
def _check_registry() -> None:
    seen_prefix: dict[str, str] = {}
    for s in REGISTRY:
        # A product must have exactly one way to be implemented.
        if (s.module is None) == (s.sources is None):
            raise RuntimeError(
                f"{s.product.value}: set exactly one of `module` (single implementation) "
                "or `sources` (source-selectable).")
        if s.sources is not None and s.default_source not in s.sources:
            raise RuntimeError(
                f"{s.product.value}: default_source {s.default_source!r} is not one of "
                f"{sorted(s.sources)}.")
        # Auth keyed by source must cover exactly the declared sources, or a config naming a
        # valid source would fail auth resolution with a confusing "not recognized".
        if isinstance(s.auth, dict):
            if s.sources is None:
                raise RuntimeError(f"{s.product.value}: per-source auth needs `sources`.")
            if set(s.auth) != set(s.sources):
                raise RuntimeError(
                    f"{s.product.value}: auth keys {sorted(s.auth)} do not match sources "
                    f"{sorted(s.sources)}.")
        # A region key must be an option the module READS, or an explicit region-only key.
        # A region override nothing reads is a no-op, and the config is left stating an
        # intent the run silently ignores.
        orphans = s.region_options - (s.options | s.region_only_options)
        if orphans:
            raise RuntimeError(
                f"{s.product.value}: region_options {sorted(orphans)} are not read. Add them "
                "to `options`, or to `region_only_options` if they have no project-level form.")
        # Two sensors sharing a channel prefix would silently overwrite each other's cube
        # channels -- `eco_sst` can only mean one thing.
        if s.sensor is not None:
            clash = seen_prefix.get(s.sensor.prefix)
            if clash:
                raise RuntimeError(
                    f"sensor prefix {s.sensor.prefix!r} is claimed by both {clash} and "
                    f"{s.product.value}; every cube channel it names would collide.")
            seen_prefix[s.sensor.prefix] = s.product.value
        # A dependency must be a real product.
        for dep in s.depends_on:
            if dep not in BY_PRODUCT:
                raise RuntimeError(f"{s.product.value}: depends_on unknown product {dep!r}.")
    # Directory names must be unique, or two products would read each other's outputs.
    dirs = [s.dir for s in REGISTRY]
    if len(dirs) != len(set(dirs)):
        raise RuntimeError(f"product `dir` values must be unique; got {sorted(dirs)}")


_check_registry()
