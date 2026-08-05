"""Per-region data sources: the two-continent case.

This is the config the package could not express before. `Region.sources` existed, but only
bathymetry and tides ever READ it -- every other module resolved its options from
`project.products[...]`, which is one answer for the whole project. So a study area with a
Pacific Northwest region and a Mediterranean one was locked to ONE met chain, ONE CMEMS
model, ONE in-situ network, whatever the region block said. And the defaults are not
neutral: HRRR is North America only, CO-OPS gauges are U.S. only, CUDEM is CONUS only, IOOS
is North America. A European AoI therefore got a silent ERA5 fallback and an empty in-situ
channel, and nothing in the run said why.

Two invariants are pinned here, and they are the whole design:

  * COVERAGE is region-varying     -- source selectors, fallbacks, dataset ids, station
                                      lists, model directories, the datum.
  * MEANING is not                 -- variables, depths, qc, the temporal conventions. Let a
                                      region override those and two AoIs' cubes stop having
                                      the same channels, so they stop being comparable. The
                                      config validator REJECTS it rather than letting a
                                      region quietly redefine what a channel is.
"""

import pytest

from coastal_sst_data import pipeline
from coastal_sst_data.config import DataProduct, parse_config, opt, resolve_opts
from coastal_sst_data.processes import cmems, insitu_acquire, landcover_esa, met, tides


def _two_continent_project(tmp_path):
    """One project, two regions, genuinely different data coverage.

    pnw   -- Oregon. HRRR covers it, CO-OPS has gauges, CUDEM has topobathy, IOOS has buoys.
    medit -- the Ligurian Sea. None of those reach it: the met chain must start at ERA5, the
             tide must come from a global model, and CMEMS has a dedicated regional product.
    """
    return parse_config({
        "name": "two_continents",
        "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {
            # Project-wide defaults are the North American ones...
            "met": {"sources": ["hrrr", "era5"], "variables": ["airtemp", "wind"]},
            "cmems": {"sources": ["my_global", "anfc_global"],
                      "variables": ["thetao"], "depths": [0.0, 10.0]},
            "insitu": {"sources": ["ioos"], "qc_flags": [1, 2]},
            "tides": {"sources": ["coops", "eo_tides"]},
            "bathymetry": {"sources": ["cudem"]},
            "landcover": {"source": "esa"},
        },
        "auth": {"earthdata": {"auth_strategy": "netrc"},
                 "copernicus": {"auth_strategy": "netrc"}},
        "regions": [
            {"name": "pnw",
             "areas": [{"name": "tillamook", "center_lat": 45.5, "center_lon": -123.9,
                        "buffer_ns_km": 8, "buffer_ew_km": 8}]},
            {"name": "medit",
             # ...and the Mediterranean region says, per source, what actually reaches it.
             "sources": {
                 "met": {"sources": ["era5"]},
                 "cmems": {"sources": ["anfc_med"],
                           "datasets": {"anfc_med": "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"}},
                 "insitu": {"sources": ["ioos"], "exclude_stations": ["bogus1"]},
                 "tides": {"sources": ["eo_tides"], "model": "FES2022"},
                 "bathymetry": {"sources": ["gmrt"]},
                 "landcover": {"source": "esa"},
             },
             "areas": [{"name": "ligurian", "center_lat": 44.0, "center_lon": 9.0,
                        "buffer_ns_km": 8, "buffer_ew_km": 8}]},
        ],
    })


# --------------------------------------------------------------------------- #
# Coverage: each product resolves a DIFFERENT source per region
# --------------------------------------------------------------------------- #
def test_met_sources_differ_per_region(tmp_path):
    """The one that matters most: HRRR does not reach the Mediterranean, so that region stacks
    only ERA5 -- a deliberate choice, not a silent fallback."""
    ds = met._build_eff(_two_continent_project(tmp_path))["ds"]
    assert ds["tillamook"]["sources"] == ["hrrr", "era5"]   # both stacked
    assert ds["ligurian"]["sources"] == ["era5"]            # ERA5 only


def test_cmems_sources_differ_per_region(tmp_path):
    """CMEMS publishes regional models; the Ligurian AoI stacks the Mediterranean tag, whose
    dataset id it registers via `datasets`. Distinct-data sources are a stacked LIST now."""
    ds = cmems._build_eff(_two_continent_project(tmp_path))["ds"]
    assert ds["tillamook"]["sources"] == ["my_global", "anfc_global"]   # the global tags
    assert ds["ligurian"]["sources"] == ["anfc_med"]
    assert ds["ligurian"]["datasets"]["anfc_med"] == "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"


def test_tide_sources_differ_per_region(tmp_path):
    """CO-OPS gauges are U.S.-only -> the Mediterranean AoI stacks only the global model."""
    ds = tides._build_eff(_two_continent_project(tmp_path))["ds"]
    assert ds["tillamook"]["sources"] == ["coops", "eo_tides"]
    assert ds["ligurian"]["sources"] == ["eo_tides"]
    assert ds["ligurian"]["model"] == "FES2022"


def test_bathymetry_dems_differ_per_region(tmp_path):
    """CUDEM is CONUS-only -> the Mediterranean AoI stacks the global GMRT instead. Distinct
    -data sources are a LIST now (stacked), region-overridable, with no fallback."""
    from coastal_sst_data.processes import bathymetry
    ds = bathymetry._build_eff(_two_continent_project(tmp_path))["ds"]
    assert ds["tillamook"]["sources"] == ["cudem"]
    assert ds["ligurian"]["sources"] == ["gmrt"]


def test_insitu_station_excludes_are_per_region(tmp_path):
    """Station lists are inherently local, so they resolve per region."""
    ds = insitu_acquire._build_eff(_two_continent_project(tmp_path))["ds"]
    assert ds["tillamook"]["exclude_stations"] == []
    assert ds["ligurian"]["exclude_stations"] == ["bogus1"]


# --------------------------------------------------------------------------- #
# Meaning: a region may NOT reshape the cube's channels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("product,key,value", [
    ("cmems", "depths", [0.0, 50.0]),        # would give this region different cube channels
    ("cmems", "variables", ["so"]),          # ditto
    ("met", "variables", ["airtemp"]),       # would ship all-NaN wind/swrad/cloud channels
    ("met", "reference_time", "06:00"),      # would sample a different time of day
    ("insitu", "qc_flags", [1]),             # would apply a different QC bar
])
def test_region_cannot_override_a_channel_shaping_option(tmp_path, product, key, value):
    """A region key that would change what the cube MEANS is rejected at load time.

    Silently accepting it is the worst outcome: the run honours one thing, the config says
    another, and the provenance faithfully records the run -- so the lie lives in the config
    file, where nobody looks. Two AoIs would end up with quietly non-comparable cubes.
    """
    cfg = {
        "name": "p", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {product: None},
        "auth": {"copernicus": {"auth_strategy": "netrc"}},
        "regions": [{"name": "r", "sources": {product: {key: value}},
                     "areas": [{"name": "a", "center_lat": 44.0, "center_lon": 9.0,
                                "buffer_ns_km": 8, "buffer_ew_km": 8}]}],
    }
    with pytest.raises(Exception, match="not a recognised option"):
        parse_config(cfg)


def test_global_products_have_no_region_keys():
    """MODIS/ECOSTRESS are global -- a region source for them would be a no-op, so the
    registry has none and the validator will reject one.

    MUR is global in its DATA too, and its data keys are still project-wide; the one region
    key it has (`overpass_sensors`) does not reshape the cube -- it says which sensors are
    worth restricting MUR's DOWNLOADS to here, which genuinely varies (an AoI may be
    ECOSTRESS-only). `mur_sst` means the same thing in every region either way.
    """
    from coastal_sst_data.config import REGION_OPTIONS
    for p in (DataProduct.modis, DataProduct.ecostress):
        assert p not in REGION_OPTIONS
    assert REGION_OPTIONS[DataProduct.mur] == {"overpass_sensors"}


# --------------------------------------------------------------------------- #
# Dispatch: one product, two source MODULES, in one run
# --------------------------------------------------------------------------- #
# These exercise the PICK-ONE (SourceKind.ACCESS) dispatch path, where the resolved source
# decides which MODULE serves an AoI. Land-cover is the vehicle: in-situ used to be the
# example, but its sources now STACK (every source is served by one fan-out module), so it no
# longer exercises per-AoI module selection at all. `insitu` covers the stacked path below.
def test_pipeline_groups_aois_by_their_resolved_source_module(tmp_path):
    """The dispatch half. `_modules_for` groups AoIs by the module that will serve them, so
    a project whose regions use different land-cover backends runs BOTH in one pass -- which
    the old project-wide `project.products[...].source` lookup could not express at all."""
    project = _two_continent_project(tmp_path)
    groups = pipeline._modules_for(project, DataProduct.landcover, ["tillamook", "ligurian"])
    # Both regions name `esa` here, so they coalesce onto one module -- the grouping is by
    # RESOLVED MODULE, not by region, so identical sources are not needlessly split.
    assert len(groups) == 1
    module, aois = groups[0]
    assert module is landcover_esa
    assert sorted(aois) == ["ligurian", "tillamook"]


def _set_region_source(project, region_idx, product, source):
    """Point one region's product at a different source (the extras bag is what is read)."""
    project.regions[region_idx].sources[product].model_extra["source"] = source


def test_pipeline_splits_when_regions_resolve_to_different_modules(tmp_path, monkeypatch):
    """Register a second land-cover backend for one region only: the pipeline must dispatch
    BOTH modules, each with just its own AoIs. This is the case the old project-wide
    `project.products[<product>].source` lookup could not express at all."""
    # Register a stub backend. `SOURCE_MODULES` holds dotted module paths, resolved lazily,
    # but an already-imported module object passes straight through -- which is exactly what
    # lets a test register a source without shipping a module for it.
    sentinel = object()
    monkeypatch.setitem(pipeline.SOURCE_MODULES[DataProduct.landcover], "corine", sentinel)

    project = _two_continent_project(tmp_path)
    # The Mediterranean region switches to the (newly registered) European product.
    _set_region_source(project, 1, DataProduct.landcover, "corine")

    groups = dict(pipeline._modules_for(project, DataProduct.landcover,
                                        ["tillamook", "ligurian"]))
    assert groups[landcover_esa] == ["tillamook"]    # N. America -> ESA WorldCover
    assert groups[sentinel] == ["ligurian"]          # Mediterranean -> the new backend


def test_an_aoi_whose_source_has_no_module_is_reported_not_dropped(tmp_path):
    """A source with no implementation must be LOUD. That AoI produces nothing, and a cube
    missing a product looks exactly like one whose product found no data."""
    project = _two_continent_project(tmp_path)
    _set_region_source(project, 1, DataProduct.landcover, "nonesuch")
    groups = dict(pipeline._modules_for(project, DataProduct.landcover,
                                        ["tillamook", "ligurian"]))
    assert groups[landcover_esa] == ["tillamook"]
    assert groups[None] == ["ligurian"]              # unimplemented -> surfaced, not silent


def test_stacked_insitu_dispatches_one_module_for_every_source(tmp_path):
    """The STACKED counterpart: in-situ sources are not alternatives to choose between, so
    every AoI resolves to the one fan-out module, which acquires each configured source into
    its own tree. There is no per-AoI source selector left to split on."""
    project = _two_continent_project(tmp_path)
    groups = pipeline._modules_for(project, DataProduct.insitu, ["tillamook", "ligurian"])
    assert len(groups) == 1
    module, aois = groups[0]
    assert module is insitu_acquire
    assert sorted(aois) == ["ligurian", "tillamook"]


# --------------------------------------------------------------------------- #
# The resolver itself
# --------------------------------------------------------------------------- #
def test_resolve_opts_layers_region_over_global(tmp_path):
    project = _two_continent_project(tmp_path)
    pnw = resolve_opts(project, "tillamook", DataProduct.met)
    med = resolve_opts(project, "ligurian", DataProduct.met)
    # region override wins where it is set...
    assert opt(pnw, "sources") == ["hrrr", "era5"] and opt(med, "sources") == ["era5"]
    # ...and the project-global value shows through where it is not.
    assert opt(pnw, "variables") == opt(med, "variables") == ["airtemp", "wind"]


def test_resolve_opts_returns_none_for_an_unselected_product(tmp_path):
    project = _two_continent_project(tmp_path)
    assert resolve_opts(project, "tillamook", DataProduct.landsat) is None
