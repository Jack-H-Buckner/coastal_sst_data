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

Adding a product is now: write the module, add a ProductSpec here -- and every registry in
the ladder above is derived, so acquisition, dispatch, ordering, auth and the skip guard all
pick it up for free. The datacube ASSEMBLER is uniform too: every product contributes through
one `(ctx) -> channels` protocol (`datacube.CONTRIBUTORS`), the run order is topologically
sorted from each contributor's declared slot reads/writes, and a non-sensor product with no
registered contributor (and no `cube_opt_out=True`) is a hard error at import
(`datacube._check_contributors`) rather than a silent omission. So a new SENSOR is one
declaration; a new non-sensor covariate is a ProductSpec + a module + one registered
Contributor (+ a provenance mapping for its channels) -- and forgetting the contributor
fails loudly. The full walkthrough -- protocol and the acquire() contract -- is
docs/DEVELOPMENT.md.
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
    modis_ref = "modis_ref"
    met = "met"
    met_overpass = "met_overpass"
    tides = "tides"
    landcover = "landcover"
    insitu = "insitu"
    insitu_mobile = "insitu_mobile"


class Kind(str, Enum):
    """How a product's aligned output is SHAPED -- which decides how the assembler reads it.

    This is about the output's structure, not its subject: MUR and CMEMS are both
    DAILY_RASTER (one file per day on the AoI grid) even though one is a satellite analysis
    and the other an ocean model.
    """
    DAILY_RASTER = "daily_raster"        # <aoi>_<YYYYMMDD>.nc      -- MUR, CMEMS, met
    OVERPASS_SENSOR = "overpass_sensor"  # <aoi>_<YYYYMMDDThhmmss>.nc -- ECOSTRESS, Landsat, MODIS
    # Timestamped rasters like a sensor, but NOT an instrument: read at ANOTHER product's
    # chosen overpass times (no clearest-scene pick, no SensorSpec). `met_overpass` documents
    # a weather model's value at each thermal sensor's overpass instant.
    OVERPASS_ALIGNED = "overpass_aligned"  # <aoi>_<YYYYMMDDThhmmss>.nc -- met_overpass
    STATIC_RASTER = "static_raster"      # <aoi>.nc, no time dim    -- bathymetry, land-cover
    SERIES_1D = "series_1d"              # <aoi>_tides.nc, dims (time,) -- tides
    STATION_TABLE = "station_table"      # <aoi>_insitu.nc, dims (station, time) -- in-situ
    OBS_TABLE = "obs_table"              # <aoi>_insitu.nc, dims (obs,) -- moving platforms


class SourceKind(str, Enum):
    """For a source-selectable product (`sources=`), what its multiple sources MEAN.

    ACCESS -- REDUNDANT access to the SAME data through different pipes (Landsat via
              Planetary Computer vs AWS; ESA land-cover vs WorldCover). Pick ONE; the
              cube gets a single channel and the output lives in one `<DIR>/aligned/<aoi>`.
    DATA   -- DISTINCT data that the user STACKS (bathymetry CUDEM + GMRT; later met
              HRRR + ERA5). Each source is acquired independently, writes its own
              `<DIR>/<source>/aligned/<aoi>` tree, and the cube emits ONE channel per
              source (`depth_cudem`, `depth_gmrt`). There is no fallback and no
              default -- the user loads as many sources as they need to span coverage.
    """
    ACCESS = "access"
    DATA = "data"


@dataclass(frozen=True)
class SensorSpec:
    """A per-overpass thermal sensor, and how to read its validity.

    These fields ARE the arguments `datacube.load_clearest_overpass` already takes -- they
    were passed as three hand-written call sites, one per sensor. Declaring them makes the
    sensor family a LOOP, so a fourth sensor gets its `<prefix>_sst`, `_cloud`, `_valid`,
    `_hour`, `_footprint_id`, its overpass-met snapshots and its in-situ matchups for free --
    every one of those channel names is already generated from the prefix.
    """
    prefix: str                                  # eco | lst | modis -- names every channel
    water_is_land: bool = False                  # ECOSTRESS's water layer has inverted polarity
    use_cloud: bool = True                       # Landsat's cloud mask is reliable...
    qc_levels: tuple[int, ...] | None = None     # ...ECOSTRESS's over-masks cold water: gate on QC
    trust_valid: bool = False                    # MODIS is quality-filtered upstream already
    # Whether this sensor's aligned files carry a cloud LAYER to publish as a cube channel.
    # MODIS does not (it arrives pre-filtered, with only sst + valid), so it emits no
    # `modis_cloud` -- an all-zero cloud channel would read as "this scene was never cloudy",
    # which is a claim its files do not make.
    has_cloud: bool = True
    # Whether this sensor's aligned files MAY carry a `footprint_id` layer -- the index of the
    # native sensor pixel each grid cell was resampled from, which only a sensor gridded from a
    # COARSER swath has to say (MODIS: ~1 km observations on a 100 m grid). This flag is
    # permission, not a promise: the layer is optional at acquisition time, so the assembler
    # emits `<prefix>_footprint_id` only when a granule on disk actually carried one.
    has_footprint: bool = False
    # Whether MULTIPLE aligned granules on the same day are MOSAICKED -- the clearest is the
    # base, the rest fill ONLY the cells it left invalid -- rather than the clearest being kept
    # and the others DISCARDED. True for a sensor whose scene footprint can be SMALLER than an
    # AoI: two Landsat path/rows covering different halves of an AoI used to leave half of it
    # NaN for that day, because the loser's exclusive coverage was thrown away wholesale.
    # MODIS opts out -- see its spec.
    #
    # Mutually exclusive with `has_footprint` (enforced in `_check_registry`): footprint ids
    # restart at 0 in EVERY granule, so a mosaicked day would put two granules' unrelated
    # native-pixel indices in one channel.
    #
    # The day's overpass IDENTITY is unaffected -- `<prefix>_hour` and the scene time every
    # forcing matchup keys off stay the BASE granule's. So a mosaicked slice can hold pixels
    # acquired at a time the cube does not report, and its overpass met / tide / in-situ values
    # describe the base granule alone. That is why `cloud_filter`'s whole-slice cloud drop and
    # `georef`'s per-slice shift -- both defaulting to `eco`, which mosaics -- judge a
    # mosaicked day on evidence from one of its granules. Documented, not yet handled.
    mosaic_same_day: bool = False


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
    # For a `sources=` product, whether its sources are REDUNDANT ACCESS (pick-one, one
    # channel, one directory) or DISTINCT DATA (stacked, one channel + one directory PER
    # source). Only meaningful when `sources` is set; ignored for `module=` singletons.
    source_kind: SourceKind = SourceKind.ACCESS
    # The config option key that NAMES a stacked-DATA product's sources. Almost always
    # "sources"; ECOSTRESS names its stacked collections "versions", because that is what
    # they are (v002/v003 of the same product), not distinct providers. Only meaningful for
    # a stacked-DATA product; it must be one of that product's `options`.
    sources_option: str = "sources"
    # Which stacked sources are acquired when the config NAMES NONE. `None` means "unknown --
    # assume every one", which is the conservative reading for a product that has not declared
    # its default. An explicit tuple means exactly that tuple, INCLUDING the empty one: `()` is
    # "none of them", not "all of them", and the distinction is load-bearing. `insitu_mobile`
    # defaults to no sources, and under the old "empty means all" reading merely SELECTING it
    # demanded a Copernicus credential for a source the run would never touch.
    #
    # It exists because the auth PREFLIGHT and the acquisition stage have to agree about that
    # default. They did not: `required_backend` assumed every known source while
    # `insitu_acquire` acquired only its own `DEFAULT_SOURCES`. Harmless while every in-situ
    # source was public -- and a breaking change the moment one needed a credential, because
    # every config with an `insitu:` block and no `sources:` list would start demanding an
    # `auth.copernicus` block for a source it was never going to run. One definition, read by
    # both. See config.required_backend and _check_registry's agreement check.
    default_sources: tuple[str, ...] | None = None

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

    # --- concurrency ----------------------------------------------------- #
    # The SERVICE this product loads from, which is what a concurrency cap has to be applied
    # to. Deliberately NOT derived from `auth`: Landsat and land-cover authenticate to
    # nothing (Planetary Computer is anonymous) yet share one rate-limited endpoint, while
    # `mur`/`modis`/`ecostress` are three products behind one Earthdata account. "How many of
    # these may be in flight at once" is a question about the SERVER, so products that share
    # a server must share a gate -- and only the product itself knows which server that is.
    #
    # The CAP is not here: it is deployment policy (`runtime.gates` in the config), and it
    # moves when the run moves -- in-region on AWS, Earthdata tolerates far more than it does
    # over a home link. This names the bucket; the config sizes it.
    gate: str = ""

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
    # A product that acquires to disk but deliberately has NO cube channel. Default False, so
    # the loud-omission invariant (datacube._check_contributors) fails at import for any
    # non-sensor product without a registered contributor -- opt out here to say "on purpose".
    cube_opt_out: bool = False
    # A product whose cube presence is produced by ANOTHER product's contributor, named here.
    # Not the same as `cube_opt_out`: this product DOES reach the cube, just not through a
    # contributor of its own. `insitu_mobile` is the case -- its observations merge into the
    # `insitu_*` channel set, because ground truth is ground truth however it was collected,
    # and a second contributor writing the same channels would double-count instead of merge.
    # Saying so here keeps `_check_contributors`'s loud-omission invariant intact rather than
    # buying silence with `cube_opt_out=True`, which would claim the product has no channels.
    cube_via: str | None = None
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

    @property
    def is_stacked_data(self) -> bool:
        """True when this product's sources are DISTINCT DATA stacked one-channel-per-source
        (bathymetry), as opposed to REDUNDANT ACCESS picked one-at-a-time (Landsat)."""
        return self.sources is not None and self.source_kind is SourceKind.DATA

    def one_module(self) -> str | None:
        """The single module a DATA product's sources all share (it fans out over its
        sources internally). Only valid for a `is_stacked_data` product."""
        return next(iter(self.sources.values()))


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
        gate="dem",
        dir="BATHYMETRY",
        kind=Kind.STATIC_RASTER,
        # DISTINCT-DATA sources, STACKED one channel per source (D10). Both DEMs are served
        # by the one bathymetry module, which fans out over the configured `sources` list
        # internally; there is no default and no fallback -- the user stacks what they need.
        sources={
            "cudem": "coastal_sst_data.processes.bathymetry",
            "gmrt": "coastal_sst_data.processes.bathymetry",
        },
        source_kind=SourceKind.DATA,
        options=_COMMON | {
            "sources", "pad_deg", "layer", "resolution",
            "stats_subgrid_m", "min_cudem_cover", "cudem_urllist", "cudem_index_cache"},
        # CUDEM is CONUS-only, so an AoI outside it stacks GMRT (or its own DEM) instead.
        region_options=frozenset({"sources", "datum_offset_m"}),
        region_only_options=frozenset({"datum_offset_m"}),
        required_vars=("elevation", "depth", "depth_p25", "depth_p75"),
        provenance_inputs=("bathymetry",),
    ),

    ProductSpec(
        product=DataProduct.mur,
        gate="earthdata",
        dir="MUR",
        kind=Kind.DAILY_RASTER,
        module="coastal_sst_data.processes.mur",
        auth="earthdata",
        options=_COMMON | {"short_name", "variable", "pad_deg", "access", "overpass_sensors"},
        # MUR is a GLOBAL product, so nothing about the DATA varies by region -- but which
        # sensors are worth restricting its days to does (an AoI may be ECOSTRESS-only).
        region_options=frozenset({"overpass_sensors"}),
        required_vars=("sst", "valid"),
        coverage_channel="mur_sst",
        # `overpass_sensors` restricts MUR's days to the ones a sensor actually flew, read
        # from the sensors' aligned dirs -- so they must have run first. Declared even when
        # the option is unset: a dependency that appears only for some configs would make the
        # process order config-dependent, and the edge costs nothing when the filter is off
        # (nothing reads MUR's output during acquisition).
        depends_on=(DataProduct.ecostress, DataProduct.landsat, DataProduct.modis),
        provenance_inputs=("mur",),
    ),

    ProductSpec(
        product=DataProduct.cmems,
        gate="copernicus",
        dir="CMEMS",
        kind=Kind.DAILY_RASTER,
        # DISTINCT-DATA sources, STACKED one channel per source (D10). Each source is a TAG
        # naming an exact CMEMS dataset -- `my_global`/`anfc_global` are built in; regional
        # tags (`my_baltic`, `anfc_med`, ...) are registered per config via `datasets`. The
        # tag IS the provenance identity, so a `cmems_<var>_<tag>` channel is self-describing
        # and there is no fallback chain. All tags are served by the one cmems module.
        sources={
            "my_global": "coastal_sst_data.processes.cmems",
            "anfc_global": "coastal_sst_data.processes.cmems",
        },
        source_kind=SourceKind.DATA,
        auth="copernicus",
        options=_COMMON | {
            "sources", "datasets", "variables", "depths", "pad_deg"},
        # Which regional models cover this AoI is a fact about where it is -> region-settable.
        region_options=frozenset({"sources", "datasets"}),
        # + >= 1 configured variable, asserted by the >=1-data-var rule (the channel set is
        # config-dependent, so it cannot be named up front).
        required_vars=("valid",),
        # coverage_channel is discovered from the files (cmems_*), not named here.
        provenance_inputs=("cmems",),
    ),

    ProductSpec(
        product=DataProduct.ecostress,
        gate="earthdata",
        dir="ECOSTRESS",
        kind=Kind.OVERPASS_SENSOR,
        # DISTINCT-DATA collection VERSIONS, STACKED one channel-set per version (D10) -- the
        # first (and so far only) SENSOR that is also stacked-data. v002/v003 have asymmetric
        # temporal coverage (v002 starts earlier, v003 reaches the present), so the user stacks
        # the versions needed to span their range; each writes its own `ECOSTRESS/<ver>/aligned`
        # tree and its own `eco_sst_<ver>` cube channel. No fallback, no default_source; the one
        # ecostress module fans out over the configured `versions` internally. The config names
        # them `versions` (see `sources_option`), because that is what they are.
        sources={
            "v002": "coastal_sst_data.processes.ecostress",
            "v003": "coastal_sst_data.processes.ecostress",
        },
        source_kind=SourceKind.DATA,
        sources_option="versions",
        auth="earthdata",
        options=_COMMON | {"short_name", "versions", "layers", "categorical"},
        required_vars=("sst", "water", "cloud", "valid"),
        # ECOSTRESS's water layer has inverted polarity, and its cloud mask over-masks cold
        # water -- so validity is gated on the QC mandatory-QA bits instead.
        sensor=SensorSpec(prefix="eco", water_is_land=True, use_cloud=False,
                          qc_levels=(0, 1), mosaic_same_day=True),
        provenance_inputs=("ecostress",),
    ),

    ProductSpec(
        product=DataProduct.landsat,
        gate="pc",
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
        # Landsat's QA_PIXEL-based cloud mask is reliable, so validity gates on it. A 185 km
        # scene is wider than most AoIs, but an AoI straddling two path/rows gets one granule
        # per row -- non-overlapping, same day -- so the day is mosaicked rather than halved.
        sensor=SensorSpec(prefix="lst", water_is_land=False, use_cloud=True,
                          mosaic_same_day=True),
        provenance_inputs=("landsat",),
    ),

    ProductSpec(
        product=DataProduct.modis,
        gate="earthdata",
        dir="MODIS",
        kind=Kind.OVERPASS_SENSOR,
        # DISTINCT-DATA PLATFORMS, STACKED one channel-set per platform (D10). Terra and Aqua
        # are two instruments on two spacecraft in two different orbits -- they never observe
        # the same overpass, so they are stacked (`modis_sst_terra` / `modis_sst_aqua`) rather
        # than one falling back to the other. That separation is not cosmetic: since NASA
        # stopped maintaining the orbits (Terra Feb 2020, Aqua Mar 2021) the two equator
        # crossings have drifted in OPPOSITE directions -- Terra earlier, Aqua later -- so one
        # merged channel would blend two diverging times of day into a single number.
        sources={
            "terra": "coastal_sst_data.processes.modis",
            "aqua": "coastal_sst_data.processes.modis",
        },
        source_kind=SourceKind.DATA,
        sources_option="platforms",
        auth="earthdata",
        options=_COMMON | {
            "platforms", "short_name", "variable", "quality_min", "regrid_radius_m",
            "access", "time_of_day", "night_solar_hours", "daytime_only"},
        required_vars=("sst", "valid"),
        # STANDALONE: no Landsat edge. The Landsat-coincident behaviour that used to live on
        # this spec is now its own product (`modis_ref`), so a project can load MODIS as a
        # thermal sensor in its own right without acquiring Landsat at all.
        depends_on=(),
        # Already quality-filtered upstream -> trust the file's own `valid` layer; it has no
        # water or cloud layer to recompute from, and so publishes no `modis_cloud` channel.
        # No `footprint_id` here -- that is `modis_ref`'s job, and its ABSENCE is what lets
        # this product MOSAIC. Above ~32 deg latitude consecutive same-night orbits overlap,
        # so a platform sees an AoI about twice a night: once near nadir, once near the swath
        # edge, each covering a different part of it. Keeping only the clearest threw the
        # other's exclusive coverage away wholesale -- the case mosaicking was added for.
        sensor=SensorSpec(prefix="modis", trust_valid=True, has_cloud=False,
                          has_footprint=False, mosaic_same_day=True),
        provenance_inputs=("modis",),
    ),

    ProductSpec(
        product=DataProduct.modis_ref,
        gate="earthdata",
        dir="MODIS_REF",
        kind=Kind.OVERPASS_SENSOR,
        module="coastal_sst_data.processes.modis_ref",
        auth="earthdata",
        options=_COMMON | {
            "short_name", "variable", "quality_min", "regrid_radius_m", "access",
            "match_landsat", "max_time_diff_minutes", "time_of_day", "daytime_only",
            "footprint_id"},
        required_vars=("sst", "valid"),
        # MODIS coincidence (match_landsat) reads the Landsat aligned files, so Landsat must
        # have run first. This is WHY the old PROCESS_ORDER put landsat before modis; here
        # it is a constraint the sort enforces rather than a comment on a hand-kept list.
        depends_on=(DataProduct.landsat,),
        # The CALIBRATION REFERENCE: coarse, well-calibrated MODIS observations matched to
        # Landsat overpasses so a downstream step can regress the drifting high-resolution
        # field onto them. Same validity rules as `modis` (quality-filtered upstream, no
        # water/cloud layer), but it is the one sensor gridded from a COARSER swath, so it
        # alone can say which native ~1 km observation each grid cell came from --
        # `modisref_footprint_id`. That is also why it is the one sensor NOT mosaicked: those
        # ids restart at 0 in every granule, so merging two granules into a day would make the
        # channel mean two different things at once. Its granules are whole overpasses hours
        # apart, not adjoining halves of one, so a day loses no coverage by keeping the
        # clearest.
        sensor=SensorSpec(prefix="modisref", trust_valid=True, has_cloud=False,
                          has_footprint=True),
        provenance_inputs=("modis_ref",),
    ),

    ProductSpec(
        product=DataProduct.met,
        gate="herbie",
        dir="MET",
        kind=Kind.DAILY_RASTER,
        # FORCING only now (D14): daily reference-time / daily-mean fields, NO sensor
        # dependency. The overpass documentation split out to `met_overpass`. DISTINCT-DATA
        # sources STACKED per channel (D10): `airtemp_hrrr`, `airtemp_era5`, ... no fallback.
        sources={
            "hrrr": "coastal_sst_data.processes.met",
            "era5": "coastal_sst_data.processes.met",
        },
        source_kind=SourceKind.DATA,
        options=_COMMON | {
            "sources", "variables", "model", "product", "fxx", "era5_zarr",
            "regrid_radius_m", "pad_deg", "reference_time", "reference_basis",
            "daily_mean_hours"},
        # HRRR is North America only: outside it a region stacks only ERA5.
        region_options=frozenset({"sources", "model"}),
        required_vars=(),          # channel set is config-dependent (see ProductSpec)
        coverage_channel="airtemp",
        provenance_inputs=("met",),
    ),

    ProductSpec(
        product=DataProduct.met_overpass,
        gate="herbie",
        dir="MET_OVERPASS",
        # Timestamped snapshots read at each thermal sensor's overpass instant -- NOT a
        # sensor itself, so a distinct Kind (see Kind.OVERPASS_ALIGNED). DISTINCT-DATA sources
        # STACKED (same hrrr/era5 as forcing), but the CUBE emits `<sensor>_<var>_<src>` only
        # for the user's `(sensor, source)` combinations (D13), not the full cross-product.
        kind=Kind.OVERPASS_ALIGNED,
        sources={
            "hrrr": "coastal_sst_data.processes.met_overpass",
            "era5": "coastal_sst_data.processes.met_overpass",
        },
        source_kind=SourceKind.DATA,
        options=_COMMON | {
            "sources", "combinations", "variables", "model", "product", "fxx", "era5_zarr",
            "regrid_radius_m", "pad_deg"},
        region_options=frozenset({"sources", "combinations", "model"}),
        required_vars=(),          # channel set is config-dependent
        # Snapshots are taken at times read from the sensors' aligned dirs, so the sensors
        # must have run first.
        depends_on=(DataProduct.ecostress, DataProduct.landsat, DataProduct.modis),
        provenance_inputs=("met_overpass",),
    ),

    ProductSpec(
        product=DataProduct.tides,
        gate="noaa_small",
        dir="TIDE",                # NOT "TIDES" -- the one name that used to need two aliases
        kind=Kind.SERIES_1D,
        # DISTINCT-DATA sources, STACKED one channel per source (D10): CO-OPS gauge synthesis
        # (U.S. waters) and a global ocean-tide model (everywhere). No fallback -- a source
        # with no coverage here (e.g. no CO-OPS gauge nearby) simply contributes no channel.
        sources={
            "coops": "coastal_sst_data.processes.tides",
            "eo_tides": "coastal_sst_data.processes.tides",
        },
        source_kind=SourceKind.DATA,
        options=_COMMON | {
            "sources", "model", "model_directory", "interval",
            "stations", "warn_distance_km", "max_distance_km", "overpass_combinations"},
        # CO-OPS gauges exist only in U.S. waters -> elsewhere, stack the global model, whose
        # downloaded directory is a property of the machine and the region.
        region_options=frozenset({"sources", "model", "model_directory", "stations",
                                  "overpass_combinations"}),
        required_vars=("tide",),
        coverage_channel="tide",
        provenance_inputs=("tides",),
    ),

    ProductSpec(
        product=DataProduct.landcover,
        gate="pc",
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
        gate="erddap",
        dir="INSITU",
        kind=Kind.STATION_TABLE,
        # DISTINCT DATA, STACKED (D10): a public network and the user's own thermometers are
        # not two pipes to the same observations -- they are different platforms, and a cube
        # wants BOTH. Every source is served by `insitu_acquire`, which fans out over the
        # configured `sources` list and delegates the fetch per network.
        #
        # DEVIATION from the usual DATA shape, and it is deliberate: the other stacked products
        # emit ONE CHANNEL PER SOURCE (`depth_cudem`, `depth_gmrt`). In-situ merges every source
        # into ONE channel set instead, because stations are ROWS, not channels -- they occupy
        # disjoint pixels anyway, and splitting them would multiply the whole `<sensor>_insitu_*`
        # family by the source count while making `insitu_sst` stop meaning "ground truth". Each
        # platform records which source it came from in the cube's station table.
        sources={"ioos": "coastal_sst_data.processes.insitu_acquire",
                 "csv": "coastal_sst_data.processes.insitu_acquire",
                 "marineinsitu": "coastal_sst_data.processes.insitu_acquire"},
        source_kind=SourceKind.DATA,
        # IOOS only. `csv` cannot do anything without a `path`, and `marineinsitu` needs a
        # Copernicus credential -- neither may be switched on by a config that never named it.
        default_sources=("ioos",),
        # Public network; local files; and the Copernicus Marine account `cmems` already uses,
        # so one ~/.netrc entry serves both.
        auth={"ioos": None, "csv": None, "marineinsitu": "copernicus"},
        options=_COMMON | {
            "sources", "variables", "stations", "exclude_stations", "qc_flags", "pad_deg",
            "max_sensor_depth_m", "max_position_drift_m",
            # csv source: where the user's files are and how to read them.
            "path", "columns", "time_zone", "units", "qc_pass_values", "default_station_id",
            # marineinsitu source: which In-Situ TAC product, which part of it, and which
            # platform classes are placeable by a fixed-station cube.
            "dataset_id", "dataset_part", "platform_types"},
        # Station lists and FILE PATHS are inherently local; `variables` is a per-NETWORK naming
        # preference (sea_water_temperature vs sea_surface_temperature), not a channel choice.
        # `columns`/`units`/`qc_pass_values` decide what the channel MEANS and so stay global --
        # a region that changed them would make two AoIs' cubes silently non-comparable.
        # `dataset_id`/`dataset_part` join the list for the same reason `path` is on it: WHICH
        # catalogue holds this region's observations is a fact about the world, and a regional
        # In-Situ product ingests national networks the global one never sees. `platform_types`
        # is NOT region-overridable -- it decides what the channel MEANS, and two AoIs whose
        # ground truth came from different classes of instrument are not comparable.
        region_options=frozenset({"sources", "stations", "exclude_stations", "variables",
                                  "path", "dataset_id", "dataset_part"}),
        required_vars=("sst", "qc"),
        provenance_inputs=("insitu",),
    ),

    ProductSpec(
        product=DataProduct.insitu_mobile,
        gate="erddap",
        dir="INSITU_MOBILE",
        kind=Kind.OBS_TABLE,
        # A SEPARATE PRODUCT, not another `insitu` source, and the reason is the SCHEMA.
        #
        # `insitu` writes (station, time) with ONE position per station -- which is not a
        # limitation to route around but the definition of a fixed station. A glider measures
        # as it moves, so its position belongs to the OBSERVATION; there is no rectangle to
        # reindex it onto, and forcing one would build a dense (station, time) block for
        # platforms whose time axes have nothing in common. So this product writes a FLAT
        # observation list instead, and the two schemas live in two trees.
        #
        # Keeping them apart is also what makes this additive. Folding per-observation
        # positions into the shared model would turn `insitu_station` from a static (y,x) map
        # into (time,y,x) -- a breaking change for any config using it as an `extract` mask,
        # and worse, a silent one: `append_zarr(mode="a-")` leaves static channels alone after
        # the first block, so a time-varying map written as (y,x) would freeze at block 0 with
        # no error at all. Nothing about the fixed path changes here.
        #
        # The CUBE still merges them: both trees feed the one `insitu_sst` channel set, because
        # ground truth is ground truth. `insitu_station` stays fixed-only.
        sources={"ioos": "coastal_sst_data.processes.insitu_mobile",
                 "csv": "coastal_sst_data.processes.insitu_mobile",
                 "marineinsitu": "coastal_sst_data.processes.insitu_mobile"},
        source_kind=SourceKind.DATA,
        # NOTHING by default. Unlike `insitu`, no source here is the obvious one to want, and
        # `marineinsitu` needs a credential -- selecting the product is the opt-in.
        default_sources=(),
        auth={"ioos": None, "csv": None, "marineinsitu": "copernicus"},
        options=_COMMON | {
            "sources", "variables", "stations", "exclude_stations", "qc_flags", "pad_deg",
            "max_sensor_depth_m", "min_position_drift_m",
            "path", "columns", "time_zone", "units", "qc_pass_values", "default_station_id",
            "dataset_id", "dataset_part", "platform_types"},
        region_options=frozenset({"sources", "stations", "exclude_stations", "variables",
                                  "path", "dataset_id", "dataset_part"}),
        # No `qc`: a flat observation list has already been QC-filtered at acquisition, and
        # carrying a per-observation flag nothing reads would double the file (see
        # InsituTable's note on the same decision).
        required_vars=("sst",),
        provenance_inputs=("insitu_mobile",),
        cube_via="insitu",          # merged into the `insitu_*` channels -- see above
    ),
)


BY_PRODUCT: dict[DataProduct, ProductSpec] = {s.product: s for s in REGISTRY}


def water_cells(w, *, water_is_land: bool):
    """The cells of a raw water layer that are WATER, for a sensor of this polarity.

    ONE definition, because there were two and they contradicted each other: the ECOSTRESS
    acquisition stage built every granule's `valid` from `water > 0` while the assembler
    recomputed the same mask from `water < 0.5`. Exact opposites on the same layer, so
    whichever ran second was wrong -- and nothing anywhere compared them, because for a long
    while both were reading the wrong asset entirely and neither number meant anything.

    The polarity now has exactly one home, `SensorSpec.water_is_land`, and both stages route
    through here. `w` is the RAW layer, so a caller still excludes non-finite cells itself;
    this answers only the polarity question.
    """
    return (w < 0.5) if water_is_land else (w > 0.5)


def spec(product: DataProduct) -> ProductSpec:
    return BY_PRODUCT[product]


def sensors() -> tuple[ProductSpec, ...]:
    """The per-overpass thermal sensors, in registry order.

    The three hand-written `("eco", "lst", "modis")` tuples in the assembler become this.
    """
    return tuple(s for s in REGISTRY if s.sensor is not None)


# --------------------------------------------------------------------------- #
# Derived stages: not products, so not selectable in a config, but they own an output
# directory that provenance and the assembler must be able to find. (The datum offset is
# resolved INSIDE the bathymetry module now and ships with each DEM source's output, so
# there is no longer a standalone DATUM stage or sidecar directory.)
# --------------------------------------------------------------------------- #
DERIVED_DIRS: dict[str, str] = {}


def product_dirs() -> dict[str, str]:
    """{key: ALLCAPS dir} for every product AND derived stage that writes aligned output.

    Keyed by the product's own name throughout -- so `tides` is `tides` here, in provenance,
    and in the coverage report. The two alias tables that existed because one table said
    `tide` and another said `tides` are gone.
    """
    return {s.product.value: s.dir for s in REGISTRY} | DERIVED_DIRS


def aligned_rel(dir_name: str, source: str | None = None) -> str:
    """The '<DIR>[/<source>]/aligned' path segment for a product's aligned tree.

    A DATA (stacked) product nests each source under its own `<source>` level so the DEMs
    do not overwrite one another; every other product keeps the flat `<DIR>/aligned`.
    """
    return f"{dir_name}/{source}/aligned" if source else f"{dir_name}/aligned"


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
        if s.sources is not None and s.source_kind is SourceKind.ACCESS \
                and s.default_source not in s.sources:
            raise RuntimeError(
                f"{s.product.value}: default_source {s.default_source!r} is not one of "
                f"{sorted(s.sources)}.")
        if s.is_stacked_data:
            # A DATA product has no pick-one default (the user stacks sources), and its one
            # module fans out over all of them, so every source must resolve to that module.
            if s.default_source is not None:
                raise RuntimeError(
                    f"{s.product.value}: a DATA (stacked) product takes no default_source.")
            if len(set(s.sources.values())) != 1 or None in set(s.sources.values()):
                raise RuntimeError(
                    f"{s.product.value}: DATA sources must all map to ONE implemented module "
                    f"(it fans out over sources internally); got {s.sources}.")
            # The key that names the stacked sources must be an option the config accepts and
            # the module reads -- otherwise a `versions:`/`sources:` list would be rejected as
            # unknown (or worse, silently ignored) at validation.
            if s.sources_option not in s.options:
                raise RuntimeError(
                    f"{s.product.value}: sources_option {s.sources_option!r} is not in "
                    f"`options`; add it so the stacked-source list is a recognised key.")
            # `default_sources` is what an unset list means, so it can only name real sources.
            # A typo here would send the auth preflight looking for a backend under a key that
            # does not exist, which is a KeyError deep inside validation rather than a message.
            unknown = sorted(set(s.default_sources or ()) - set(s.sources))
            if unknown:
                raise RuntimeError(
                    f"{s.product.value}: default_sources {unknown} are not among "
                    f"{sorted(s.sources)}.")
        elif s.default_sources:
            raise RuntimeError(
                f"{s.product.value}: default_sources is only meaningful for a stacked-DATA "
                "product; a pick-one product uses default_source.")
        elif s.sources_option != "sources":
            # Only a stacked-DATA product has a stacked-source list to name.
            raise RuntimeError(
                f"{s.product.value}: sources_option is only meaningful for a stacked-DATA "
                "product; leave it at the default for everything else.")
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
            # Unreachable today (only MODIS carries footprints, and it does not mosaic) --
            # this is here for the FOURTH sensor, because the two features are silently
            # incompatible rather than obviously so.
            if s.sensor.mosaic_same_day and s.sensor.has_footprint:
                raise RuntimeError(
                    f"{s.product.value}: mosaic_same_day and has_footprint are mutually "
                    "exclusive. footprint_id restarts at 0 in EVERY granule, so a mosaicked "
                    "day would hold two granules' unrelated native-pixel indices in one "
                    "channel, and grouping by it would average unrelated observations.")
        # A dependency must be a real product.
        for dep in s.depends_on:
            if dep not in BY_PRODUCT:
                raise RuntimeError(f"{s.product.value}: depends_on unknown product {dep!r}.")
        # Every product must name the SERVICE it loads from. Defaulting an unset gate to the
        # product's own name would be the worst possible failure mode: it silently gives that
        # product a private cap, so a fourth Earthdata product would quietly run at full
        # concurrency ALONGSIDE the three that are sharing one -- and the account sees the
        # sum. Nothing raises, nothing logs, and the only symptom is a service deciding it is
        # being abused. So it is required, and only the author knows the answer.
        if not s.gate:
            raise RuntimeError(
                f"{s.product.value}: set `gate` to the service this product loads from "
                "(products sharing a server MUST share a gate name, so their concurrency "
                "cap applies to the server rather than to each product separately).")
    # Directory names must be unique, or two products would read each other's outputs.
    dirs = [s.dir for s in REGISTRY]
    if len(dirs) != len(set(dirs)):
        raise RuntimeError(f"product `dir` values must be unique; got {sorted(dirs)}")


_check_registry()
