"""The product registry (coastal_sst_data.products) and the tables DERIVED from it.

Every registry in the package -- config's options/auth tables, store's completeness check,
the pipeline's dispatch and ordering, the assembler's directories and coverage channels,
provenance's field map -- used to be a hand-maintained list that had to be kept in step with
all the others. Missing an entry never RAISED. It just quietly did less than you asked:
without a store.REQUIRED_VARS entry the skip guard cannot see a truncated file; without a
provenance entry the channel ships with a blank source record; without a DAILY_CHANNELS
entry coverage silently stops being checked.

These tests pin the derivations, so a future edit cannot desync one table from the specs.
"""

import pytest

from coastal_sst_data import config, pipeline, products, provenance, store
from coastal_sst_data.products import BY_PRODUCT, DataProduct, Kind, REGISTRY


def test_every_product_has_exactly_one_spec():
    assert {s.product for s in REGISTRY} == set(DataProduct)
    assert len(REGISTRY) == len(DataProduct)


def test_registry_invariants_hold():
    """products._check_registry runs at import; this is its explicit assertion."""
    products._check_registry()          # must not raise


def test_every_product_names_the_service_it_loads_from():
    """A concurrency cap has to apply to the SERVER, not to each product -- mur, modis and
    ecostress are three products behind one Earthdata account."""
    assert all(s.gate for s in products.REGISTRY)
    shared = {s.product.value for s in products.REGISTRY if s.gate == "earthdata"}
    assert {"mur", "modis", "ecostress"} <= shared


def test_a_product_without_a_gate_is_rejected(monkeypatch):
    """Defaulting an unset gate would be the worst failure mode available: it silently hands
    that product a PRIVATE cap, so it runs at full concurrency alongside the products it
    shares a server with and the account sees the sum. Nothing raises, nothing logs."""
    import dataclasses
    ungated = dataclasses.replace(products.REGISTRY[0], gate="")
    monkeypatch.setattr(products, "REGISTRY", (ungated,) + products.REGISTRY[1:])
    with pytest.raises(RuntimeError, match="gate"):
        products._check_registry()


# --------------------------------------------------------------------------- #
# The derivations
# --------------------------------------------------------------------------- #
def test_config_tables_are_derived_from_the_registry():
    for s in REGISTRY:
        assert config.PRODUCT_OPTIONS[s.product] == set(s.options)
        if s.region_options:
            assert config.REGION_OPTIONS[s.product] == set(s.region_options)
        if s.auth is not None:
            assert config.AUTH_REQUIREMENTS[s.product] == s.auth
        if s.default_source:
            assert config.DEFAULT_SOURCE[s.product] == s.default_source


def test_store_required_vars_is_derived_and_keyed_by_output_dir():
    """The skip guard reads this. A product with no entry here would have its truncated
    downloads taken for done on every subsequent run."""
    for s in REGISTRY:
        assert store.REQUIRED_VARS[s.dir] == s.required_vars


def test_assembler_dirs_and_coverage_are_derived():
    from coastal_sst_data.processes import datacube
    for s in REGISTRY:
        assert datacube.PRODUCT_DIRS[s.product.value] == s.dir
        if s.coverage_channel:
            assert datacube.DAILY_CHANNELS[s.product.value] == s.coverage_channel
        else:
            # Overpass sensors must NOT be coverage-checked: a day with no scene is normal,
            # and warning about it would train the user to ignore the warning.
            assert s.product.value not in datacube.DAILY_CHANNELS


def test_provenance_sensor_map_is_derived():
    assert provenance.SENSORS == {s.sensor.prefix: s.product.value for s in products.sensors()}


def test_the_tide_alias_tables_are_gone():
    """`_DIR_KEY_ALIASES` and `_COVERAGE_ALIASES` existed for exactly one reason: one table
    called the tide product `tide` and another called it `tides`. One registry, one name."""
    from coastal_sst_data.processes import datacube
    assert not hasattr(provenance, "_DIR_KEY_ALIASES")
    assert not hasattr(datacube, "_COVERAGE_ALIASES")
    # The product is `tides` everywhere; only its OUTPUT DIRECTORY is TIDE.
    assert BY_PRODUCT[DataProduct.tides].dir == "TIDE"
    assert datacube.PRODUCT_DIRS["tides"] == "TIDE"
    assert datacube.DAILY_CHANNELS["tides"] == "tide"


# --------------------------------------------------------------------------- #
# Ordering: declared as dependencies, not as a hand-kept list
# --------------------------------------------------------------------------- #
def test_process_order_covers_every_product_exactly_once():
    order = pipeline.process_order()
    assert sorted(p.value for p in order) == sorted(p.value for p in DataProduct)


def test_process_order_honours_the_real_constraints():
    """The two orderings that are load-bearing, and WHY -- previously enforced only by a
    comment above a hand-sorted list."""
    order = pipeline.process_order()
    pos = {p: i for i, p in enumerate(order)}

    # MODIS coincidence (match_landsat) reads the Landsat aligned files.
    assert pos[DataProduct.landsat] < pos[DataProduct.modis]

    # met's overpass snapshots are taken at times read from the sensors' aligned dirs.
    for sensor in (DataProduct.ecostress, DataProduct.landsat, DataProduct.modis):
        assert pos[sensor] < pos[DataProduct.met]

    # MUR's `overpass_sensors` filter restricts its days to the ones a sensor flew, which it
    # discovers from those same aligned dirs -- so MUR can no longer lead the run.
    for sensor in (DataProduct.ecostress, DataProduct.landsat, DataProduct.modis):
        assert pos[sensor] < pos[DataProduct.mur]


def test_process_order_is_stable_and_matches_the_previous_hand_kept_list():
    """The topological sort must not have quietly reshuffled a working pipeline."""
    assert [p.value for p in pipeline.process_order()] == [
        "bathymetry", "cmems", "ecostress", "landsat", "modis", "mur", "modis_ref",
        "met", "met_overpass", "tides", "landcover", "insitu",
    ]


def test_a_dependency_cycle_is_reported_not_looped_on(monkeypatch):
    from dataclasses import replace
    a = replace(BY_PRODUCT[DataProduct.mur], depends_on=(DataProduct.cmems,))
    b = replace(BY_PRODUCT[DataProduct.cmems], depends_on=(DataProduct.mur,))
    monkeypatch.setattr(products, "REGISTRY", (a, b))
    monkeypatch.setattr(products, "BY_PRODUCT", {a.product: a, b.product: b})
    with pytest.raises(RuntimeError, match="dependency cycle"):
        pipeline.process_order()


# --------------------------------------------------------------------------- #
# Dispatch: modules are dotted strings, resolved lazily
# --------------------------------------------------------------------------- #
def test_every_declared_module_actually_imports():
    """A dotted path is only checked when it is resolved, so a typo would surface as a
    mid-run ImportError. Resolve them all up front instead."""
    for s in REGISTRY:
        for source in (s.known_sources or (None,)):
            dotted = s.module_for(source)
            if dotted is None:
                continue                       # a recognised source with no module yet
            assert pipeline._resolve(dotted) is not None
            assert hasattr(pipeline._resolve(dotted), "acquire")


def test_unimplemented_sources_resolve_to_none_not_an_error():
    """Landsat via aws/gee and landcover via gee are RECOGNISED names awaiting a module, so
    a config naming one fails as 'not implemented' rather than as 'unknown source'."""
    ls = BY_PRODUCT[DataProduct.landsat]
    assert ls.module_for("pc") is not None
    assert ls.module_for("aws") is None
    assert ls.module_for("gee") is None
    assert "gee" in ls.known_sources           # recognised, so auth resolution works


def test_sensor_specs_carry_the_validity_rules_the_assembler_needs():
    by_prefix = {s.sensor.prefix: s.sensor for s in products.sensors()}
    assert set(by_prefix) == {"eco", "lst", "modis", "modisref"}
    # ECOSTRESS: inverted water polarity, and cloud over-masks cold water -> gate on QC.
    assert by_prefix["eco"].water_is_land is True
    assert by_prefix["eco"].use_cloud is False
    assert by_prefix["eco"].qc_levels == (0, 1)
    # Landsat: QA_PIXEL cloud is reliable.
    assert by_prefix["lst"].use_cloud is True
    assert by_prefix["lst"].water_is_land is False
    # MODIS: quality-filtered upstream -> trust its own `valid` layer. Both MODIS products
    # read the same L2P files, so both say so.
    assert by_prefix["modis"].trust_valid is True
    assert by_prefix["modisref"].trust_valid is True
    # Which sensors MOSAIC a day's granules rather than keeping only the clearest. Landsat and
    # ECOSTRESS can put two non-overlapping granules over one AoI on one day, and so can MODIS
    # -- above ~32 deg latitude consecutive orbits overlap, so a platform sees an AoI twice a
    # night. `modis_ref` opts out because its `footprint_id` ids restart per granule.
    assert by_prefix["lst"].mosaic_same_day is True
    assert by_prefix["eco"].mosaic_same_day is True
    assert by_prefix["modis"].mosaic_same_day is True
    assert by_prefix["modis"].has_footprint is False
    assert by_prefix["modisref"].mosaic_same_day is False
    assert by_prefix["modisref"].has_footprint is True


def test_a_mosaicking_sensor_may_not_also_carry_footprint_ids(monkeypatch):
    """footprint_id restarts at 0 in every granule, so a mosaicked day would hold two granules'
    unrelated native-pixel indices in one channel -- and grouping by it would then average
    unrelated observations together, silently.

    `modis_ref` is the live case: it is the one sensor carrying footprints, and it is the one
    sensor that does not mosaic. Turning mosaicking on for it must be refused at import."""
    import dataclasses
    ref = BY_PRODUCT[DataProduct.modis_ref]
    both = dataclasses.replace(
        ref, sensor=dataclasses.replace(ref.sensor, mosaic_same_day=True))
    monkeypatch.setattr(products, "REGISTRY",
                        tuple(both if s is ref else s for s in products.REGISTRY))
    with pytest.raises(RuntimeError, match="footprint"):
        products._check_registry()


def test_ecostress_is_a_stacked_data_sensor_keyed_on_versions():
    """ECOSTRESS is the first SENSOR that is also STACKED-DATA (D10): its sources are collection
    version tags, named by `versions` in the config, and it keeps its SensorSpec."""
    s = products.spec(DataProduct.ecostress)
    assert s.is_stacked_data is True                 # stacked, one channel-set per version
    assert s.sensor is not None                      # ...but still a sensor
    assert s.sources_option == "versions"            # config names them `versions`, not `sources`
    assert s.sources_option in s.options             # ...and it is a recognised option
    assert set(s.known_sources) == {"v002", "v003"}
    assert s.default_source is None                  # stacked products have no pick-one default
    assert s.one_module() == "coastal_sst_data.processes.ecostress"
    # sources_option only means anything for a stacked-data product.
    for other in REGISTRY:
        if not other.is_stacked_data:
            assert other.sources_option == "sources"


def test_kinds_are_assigned_as_the_assembler_expects():
    kind = {s.product: s.kind for s in REGISTRY}
    assert kind[DataProduct.mur] == Kind.DAILY_RASTER
    assert kind[DataProduct.cmems] == Kind.DAILY_RASTER
    assert kind[DataProduct.met] == Kind.DAILY_RASTER
    assert kind[DataProduct.ecostress] == Kind.OVERPASS_SENSOR
    assert kind[DataProduct.landsat] == Kind.OVERPASS_SENSOR
    assert kind[DataProduct.modis] == Kind.OVERPASS_SENSOR
    assert kind[DataProduct.bathymetry] == Kind.STATIC_RASTER
    assert kind[DataProduct.landcover] == Kind.STATIC_RASTER
    assert kind[DataProduct.tides] == Kind.SERIES_1D
    assert kind[DataProduct.insitu] == Kind.STATION_TABLE
    # Every OVERPASS_SENSOR must carry a SensorSpec, and nothing else may.
    for s in REGISTRY:
        assert (s.sensor is not None) == (s.kind == Kind.OVERPASS_SENSOR)
