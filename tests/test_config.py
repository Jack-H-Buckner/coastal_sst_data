import copy
from pathlib import Path

import pytest

from pydantic import ValidationError

from coastal_sst_data.config import (
    load_config, parse_config, wrap_lon, Project, AreaOfInterest, BoundingBox,
    TimeWindow, GridSpec, DataProduct, EarthdataAuth, GeeAuth,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"

def test_load_config():
    cfg = load_config(EXAMPLE)
    assert isinstance(cfg, Project)
    assert cfg.name == "COASTAL_DATA_TESTS"
    assert len(cfg.regions) == 2
    assert cfg.regions[0].areas[0].name == "tillamook_bay"
    # products is a mapping: keys = selection, values = global options.
    assert list(cfg.products) == [DataProduct.bathymetry, DataProduct.ecostress,
                                  DataProduct.mur, DataProduct.cmems, DataProduct.landsat,
                                  DataProduct.met, DataProduct.met_overpass, DataProduct.tides,
                                  DataProduct.insitu, DataProduct.landcover]
    # a bare `bathymetry:` -> default (empty) global options
    assert cfg.products[DataProduct.bathymetry].model_dump() == {}
    # global options land on the product's ProductOptions bag (extra=allow)
    assert cfg.products[DataProduct.ecostress].versions == ["v003", "v002"]
    assert cfg.products[DataProduct.mur].variable == "analysed_sst"
    # depths is a user-tunable list; assert its shape, not one particular choice.
    assert all(isinstance(d, (int, float)) for d in cfg.products[DataProduct.cmems].depths)
    assert cfg.products[DataProduct.landsat].source == "pc"
    # region-dependent source options live under the region's `sources`
    assert cfg.regions[0].sources[DataProduct.bathymetry].sources == ["cudem"]


# ---------------------------------------------------------------------------
# load_config I/O: file-level errors should be clear, not cryptic parse errors.
# ---------------------------------------------------------------------------
def test_load_config_missing_file_raises(tmp_path):
    """A path that doesn't exist -> FileNotFoundError with a clear message."""
    missing = tmp_path / "does_not_exist.yaml"   # tmp_path is empty, so it won't exist
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)


def test_load_config_non_mapping_root_raises(tmp_path):
    """A YAML file whose top level is a list (not a mapping) -> ValueError."""
    bad = tmp_path / "list_root.yaml"
    bad.write_text("- one\n- two\n")   # parses to a list, not a dict
    with pytest.raises(ValueError, match="Config root must be a mapping"):
        load_config(bad)


def test_load_config_empty_file_raises_validation_error(tmp_path):
    """An empty/comment-only file -> {} -> ValidationError for missing fields."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("# just a comment, no content\n")   # safe_load -> None -> {}
    with pytest.raises(ValidationError) as exc:
        load_config(empty)
    # every required top-level field should be reported as missing
    missing = {e["loc"][0] for e in exc.value.errors() if e["type"] == "missing"}
    assert {"name", "output_dir", "time", "products", "regions"} <= missing


# ---------------------------------------------------------------------------
# AreaOfInterest.bbox: the km -> degree conversion (the only real computation
# in the config layer). These lock the math, the latitude correction, and the
# buffer convention so a subtle change can't silently move everyone's boxes.
# ---------------------------------------------------------------------------
def test_bbox_km_to_degrees():
    """111 km of N/S buffer -> ~1 degree of latitude, centered on the point."""
    aoi = AreaOfInterest(name="x", center_lat=0.0, center_lon=0.0,
                         buffer_ns_km=111.0, buffer_ew_km=111.0)
    bb = aoi.bbox
    assert bb.max_lat == pytest.approx(1.0, abs=1e-3)   # 111 km / 111 = 1 deg
    assert bb.min_lat == pytest.approx(-1.0, abs=1e-3)


def test_bbox_cosine_latitude_correction():
    """The same E/W km buffer spans MORE longitude degrees at higher latitude.

    At 60 deg, cos(60) = 0.5, so a given E/W distance covers ~2x the longitude
    span it does at the equator. This is the bug-prone line -- a missing cos()
    passes at the equator and only shows up away from it.
    """
    eq = AreaOfInterest(name="e", center_lat=0.0, center_lon=0.0,
                        buffer_ns_km=111.0, buffer_ew_km=111.0)
    high = AreaOfInterest(name="h", center_lat=60.0, center_lon=0.0,
                          buffer_ns_km=111.0, buffer_ew_km=111.0)
    assert high.bbox.max_lon == pytest.approx(2.0, abs=1e-2)   # 111 / (111 * 0.5)
    assert high.bbox.max_lon > eq.bbox.max_lon


def test_bbox_centered_and_symmetric():
    """The center point is the exact midpoint of the box in both axes."""
    aoi = AreaOfInterest(name="x", center_lat=45.52, center_lon=-123.95,
                         buffer_ns_km=25.0, buffer_ew_km=50.0)
    bb = aoi.bbox
    assert (bb.min_lon + bb.max_lon) / 2 == pytest.approx(aoi.center_lon)
    assert (bb.min_lat + bb.max_lat) / 2 == pytest.approx(aoi.center_lat)


def test_bbox_centered_near_antimeridian():
    """A box straddling 180 deg wraps the SHORT way: west > east, edges in range.

    Naive (min_lon + max_lon)/2 is wrong across the seam, so the wrap-aware
    center_lon/lon_span helpers must recover the true center and a small span.
    """
    aoi = AreaOfInterest(name="x", center_lat=0.00, center_lon=-179.95,
                         buffer_ns_km=25.0, buffer_ew_km=50.0)
    bb = aoi.bbox
    # Both edges stay within valid longitude range (no -180.x overflow).
    assert -180.0 <= bb.min_lon <= 180.0
    assert -180.0 <= bb.max_lon <= 180.0
    # It crosses the antimeridian, so the western edge is numerically > eastern.
    assert bb.crosses_antimeridian
    assert bb.min_lon > bb.max_lon
    # The span is the intended narrow box, NOT ~360 the long way around.
    assert bb.lon_span == pytest.approx(2 * 50.0 / 111.0, abs=1e-3)
    # Wrap-aware center and midpoint recover the original point.
    assert bb.center_lon == pytest.approx(aoi.center_lon, abs=1e-6)
    assert bb.center_lat == pytest.approx(aoi.center_lat)


def test_bbox_buffer_is_half_extent():
    """buffer_ns_km is a HALF-height: total N/S extent is 2 x the buffer."""
    aoi = AreaOfInterest(name="x", center_lat=0.0, center_lon=0.0,
                         buffer_ns_km=111.0, buffer_ew_km=111.0)
    bb = aoi.bbox
    assert bb.max_lat - bb.min_lat == pytest.approx(2.0, abs=1e-3)  # full = 2 x buffer


# ---------------------------------------------------------------------------
# Poles: bbox must degrade gracefully (clamp + warn), never crash, so a batch
# over many AoIs keeps running when one AoI is pathological.
# ---------------------------------------------------------------------------
def test_bbox_clamps_latitude_at_north_pole(caplog):
    """A buffer running past 90 deg is clamped to 90 and logs a warning."""
    aoi = AreaOfInterest(name="arctic", center_lat=89.9, center_lon=0.0,
                         buffer_ns_km=50.0, buffer_ew_km=10.0)
    with caplog.at_level("WARNING"):
        bb = aoi.bbox                       # must NOT raise
    assert bb.max_lat == pytest.approx(90.0)   # clamped at the pole
    assert bb.min_lat < 90.0
    assert "past the pole" in caplog.text
    assert "arctic" in caplog.text


def test_bbox_clamps_longitude_near_pole(caplog):
    """Near a pole the box spans all longitudes -> clamp to [-180, 180] + warn."""
    aoi = AreaOfInterest(name="polar", center_lat=89.99, center_lon=10.0,
                         buffer_ns_km=5.0, buffer_ew_km=200.0)
    with caplog.at_level("WARNING"):
        bb = aoi.bbox                       # must NOT raise
    assert bb.min_lon == pytest.approx(-180.0)
    assert bb.max_lon == pytest.approx(180.0)
    assert bb.lon_span == pytest.approx(360.0)
    assert "all longitudes" in caplog.text


def test_bbox_normal_aoi_does_not_warn(caplog):
    """A well-behaved AoI produces no warnings."""
    aoi = AreaOfInterest(name="ok", center_lat=45.0, center_lon=-123.0,
                         buffer_ns_km=25.0, buffer_ew_km=25.0)
    with caplog.at_level("WARNING"):
        _ = aoi.bbox
    assert caplog.records == []


# ---------------------------------------------------------------------------
# AreaOfInterest field constraints: latitude/longitude ranges and positive
# buffers. These encode intent, so pin the boundaries in both directions.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field, value", [
    ("center_lat", 91.0),      # above +90
    ("center_lat", -91.0),     # below -90
    ("center_lon", 181.0),     # above +180
    ("center_lon", -181.0),    # below -180
    ("buffer_ns_km", 0.0),     # must be strictly > 0
    ("buffer_ns_km", -5.0),
    ("buffer_ew_km", 0.0),
    ("buffer_ew_km", -5.0),
])
def test_area_of_interest_rejects_out_of_range(field, value):
    kwargs = {"name": "a", "center_lat": 45.0, "center_lon": -123.0,
              "buffer_ns_km": 25.0, "buffer_ew_km": 15.0}
    kwargs[field] = value                      # spoil exactly one field
    with pytest.raises(ValidationError):
        AreaOfInterest(**kwargs)


def test_area_of_interest_accepts_boundary_values():
    """Control: the exact limits (+/-90 lat, +/-180 lon) are allowed."""
    aoi = AreaOfInterest(name="edge", center_lat=90.0, center_lon=180.0,
                         buffer_ns_km=1.0, buffer_ew_km=1.0)
    assert aoi.center_lat == 90.0 and aoi.center_lon == 180.0


# ---------------------------------------------------------------------------
# TimeWindow: an inclusive date range; end must be on or after start.
# ---------------------------------------------------------------------------
def test_time_window_end_before_start_rejected():
    """end_date earlier than start_date -> ValidationError."""
    with pytest.raises(ValidationError, match="on or after"):
        TimeWindow(start_date="2026-06-30", end_date="2026-06-01")


def test_time_window_equal_dates_allowed():
    """Inclusive range: a single-day window (end == start) is valid."""
    tw = TimeWindow(start_date="2026-06-01", end_date="2026-06-01")
    assert tw.start_date == tw.end_date


def test_time_window_bad_date_string_rejected():
    """A non-calendar date string (month 13, day 40) -> ValidationError."""
    with pytest.raises(ValidationError):
        TimeWindow(start_date="2026-13-40", end_date="2026-06-30")


def test_time_window_extra_key_rejected():
    """An unknown key is rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        TimeWindow(start_date="2026-06-01", end_date="2026-06-30", timezone="UTC")


# ---------------------------------------------------------------------------
# GridSpec: defaults come from config.test.yaml; resolution must be positive.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_grid_spec_rejects_non_positive_resolution(bad):
    """resolution_m must be strictly > 0."""
    with pytest.raises(ValidationError):
        GridSpec(resolution_m=bad)


# ---------------------------------------------------------------------------
# DataCubeSpec: optional block, all defaults; unknown keys rejected.
# ---------------------------------------------------------------------------
def test_datacube_defaults_when_omitted(base_project):
    cfg = parse_config(base_project)                       # no datacube block
    assert cfg.datacube.met_time == "reference"
    assert cfg.datacube.output_subdir == "datacube"
    assert cfg.datacube.compression.codec == "zstd"
    assert cfg.datacube.compression.shuffle == "shuffle"


def test_datacube_rejects_unknown_key(base_project):
    base_project["datacube"] = {"bogus": 1}
    with pytest.raises(ValidationError):
        parse_config(base_project)


@pytest.mark.parametrize("removed", ["fill_mur_water", "fill_cmems_water", "water_level"])
def test_datacube_rejects_removed_raw_output_keys(base_project, removed):
    """S1 removed these keys; an un-migrated config must FAIL loudly (extra='forbid'), never
    be silently ignored -- the cube now ships raw ingredients and downstream owns the fill /
    mask / water-level determinations."""
    base_project["datacube"] = {removed: True}
    with pytest.raises(ValidationError):
        parse_config(base_project)


def test_datacube_rejects_bad_compression_level(base_project):
    base_project["datacube"] = {"compression": {"level": 99}}   # blosc clevel 0-9
    with pytest.raises(ValidationError):
        parse_config(base_project)


# ---------------------------------------------------------------------------
# wrap_lon: the seam helper everything else leans on. Normalizes any longitude
# into [-180, 180), with +180 folding to -180.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    (0.0, 0.0),
    (45.0, 45.0),        # in-range values are unchanged
    (-45.0, -45.0),
    (180.0, -180.0),     # +180 folds to -180 (half-open range)
    (-180.0, -180.0),
    (190.0, -170.0),     # just past the antimeridian eastward
    (-190.0, 170.0),     # just past it westward
    (360.0, 0.0),        # full turns wrap back
    (540.0, -180.0),
])
def test_wrap_lon_values(raw, expected):
    assert wrap_lon(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [0.0, 179.9, -179.9, 200.0, -200.0, 1234.5])
def test_wrap_lon_in_range_and_idempotent(raw):
    """Result is always in [-180, 180), and re-wrapping changes nothing."""
    w = wrap_lon(raw)
    assert -180.0 <= w < 180.0
    assert wrap_lon(w) == pytest.approx(w)


# ---------------------------------------------------------------------------
# BoundingBox._ordered: tested directly (not just via AreaOfInterest.bbox).
# Latitude must be strictly ordered; longitude may cross the antimeridian but
# must not be zero-width.
# ---------------------------------------------------------------------------
def test_boundingbox_rejects_equal_latitude():
    """min_lat == max_lat (zero N-S height) -> ValidationError."""
    with pytest.raises(ValidationError, match="min_lat must be < max_lat"):
        BoundingBox(min_lon=-1.0, min_lat=10.0, max_lon=1.0, max_lat=10.0)


def test_boundingbox_rejects_inverted_latitude():
    """min_lat > max_lat (south above north) -> ValidationError."""
    with pytest.raises(ValidationError, match="min_lat must be < max_lat"):
        BoundingBox(min_lon=-1.0, min_lat=20.0, max_lon=1.0, max_lat=10.0)


def test_boundingbox_rejects_zero_width_longitude():
    """min_lon == max_lon (zero E-W width) -> ValidationError."""
    with pytest.raises(ValidationError, match="min_lon and max_lon must differ"):
        BoundingBox(min_lon=5.0, min_lat=-1.0, max_lon=5.0, max_lat=1.0)


def test_boundingbox_accepts_antimeridian_crossing():
    """A directly-built box with west > east is valid (crosses 180 deg)."""
    bb = BoundingBox(min_lon=179.0, min_lat=-1.0, max_lon=-179.0, max_lat=1.0)
    assert bb.crosses_antimeridian
    assert bb.lon_span == pytest.approx(2.0)   # short way across the seam


# ---------------------------------------------------------------------------
# Auth: a selected product that needs a backend must have it configured.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("product, backend", [
    ("ecostress", "earthdata"),
    ("mur", "earthdata"),
])
def test_missing_auth_for_selected_product_rejected(base_project, product, backend):
    """Selecting an auth-requiring product with no matching auth -> ValidationError."""
    base_project["products"] = {product: None}   # only the auth-requiring product
    base_project["auth"] = {}                     # no backend configured
    base_project["regions"][0]["sources"] = {}    # drop bathymetry source (now unselected)
    with pytest.raises(ValidationError, match=f"auth.{backend}"):
        parse_config(base_project)


@pytest.mark.parametrize("product, auth", [
    ("ecostress", {"earthdata": {"auth_strategy": "netrc"}}),
    ("mur", {"earthdata": {"auth_strategy": "netrc"}}),
])
def test_auth_present_for_selected_product_ok(base_project, product, auth):
    """Control: the required backend present -> the project validates."""
    base_project["products"] = {product: None}
    base_project["auth"] = auth
    base_project["regions"][0]["sources"] = {}    # drop bathymetry source (now unselected)
    cfg = parse_config(base_project)
    assert list(cfg.products) == [DataProduct(product)]


# --- Landsat auth depends on the `source` selector -------------------------- #
def test_landsat_default_source_pc_needs_no_auth(base_project):
    """Landsat defaults to the Planetary Computer source -> no auth required."""
    base_project["products"] = {"landsat": None}   # bare -> default source 'pc'
    base_project["auth"] = {}
    base_project["regions"][0]["sources"] = {}
    cfg = parse_config(base_project)               # must NOT raise
    assert list(cfg.products) == [DataProduct.landsat]


# --- Landcover auth likewise depends on the `source` selector --------------- #
def test_landcover_default_source_esa_needs_no_auth(base_project):
    """Landcover defaults to the ESA WorldCover (PC) source -> no auth required."""
    base_project["products"] = {"landcover": None}   # bare -> default source 'esa'
    base_project["auth"] = {}
    base_project["regions"][0]["sources"] = {}
    cfg = parse_config(base_project)                 # must NOT raise
    assert list(cfg.products) == [DataProduct.landcover]


def test_landcover_gee_source_requires_auth(base_project):
    """The landcover 'gee' source (JRC + NDWI) needs auth.gee."""
    base_project["products"] = {"landcover": {"source": "gee"}}
    base_project["auth"] = {}
    base_project["regions"][0]["sources"] = {}
    with pytest.raises(ValidationError, match="auth.gee"):
        parse_config(base_project)


def test_landcover_gee_source_with_auth_ok(base_project):
    base_project["products"] = {"landcover": {"source": "gee"}}
    base_project["auth"] = {"gee": {"project": "my-gcp-project"}}
    base_project["regions"][0]["sources"] = {}
    cfg = parse_config(base_project)
    assert list(cfg.products) == [DataProduct.landcover]


def test_landsat_gee_source_requires_gee_auth(base_project):
    """Landsat with source='gee' and no auth.gee -> ValidationError."""
    base_project["products"] = {"landsat": {"source": "gee"}}
    base_project["auth"] = {}
    base_project["regions"][0]["sources"] = {}
    with pytest.raises(ValidationError, match="auth.gee"):
        parse_config(base_project)


def test_landsat_gee_source_with_auth_ok(base_project):
    """Landsat with source='gee' and auth.gee present -> validates."""
    base_project["products"] = {"landsat": {"source": "gee"}}
    base_project["auth"] = {"gee": {"project": "my-gcp-project"}}
    base_project["regions"][0]["sources"] = {}
    cfg = parse_config(base_project)
    assert list(cfg.products) == [DataProduct.landsat]


def test_landsat_unknown_source_rejected(base_project):
    """An unrecognized landsat.source fails loudly (not silently accepted)."""
    base_project["products"] = {"landsat": {"source": "banana"}}
    base_project["regions"][0]["sources"] = {}
    with pytest.raises(ValidationError, match="not recognized"):
        parse_config(base_project)


def test_public_only_project_needs_no_auth(base_project):
    """Only public-source products -> loads with no `auth` block at all."""
    base_project["products"] = {"bathymetry": None, "tides": None}
    del base_project["auth"]                        # no auth section whatsoever
    cfg = parse_config(base_project)                # must NOT raise
    assert cfg.auth.earthdata is None and cfg.auth.gee is None
    assert list(cfg.products) == [DataProduct.bathymetry, DataProduct.tides]


# ---------------------------------------------------------------------------
# GeeAuth: service_account and key_file are all-or-nothing.
# ---------------------------------------------------------------------------
def test_gee_auth_service_account_without_key_rejected():
    """A service account with no key file -> ValidationError."""
    with pytest.raises(ValidationError, match="set together"):
        GeeAuth(project="p", service_account="sa@x.iam.gserviceaccount.com")


def test_gee_auth_key_without_service_account_rejected():
    """A key file with no service account -> ValidationError."""
    with pytest.raises(ValidationError, match="set together"):
        GeeAuth(project="p", key_file="/keys/ee.json")


def test_gee_auth_both_omitted_allowed():
    """Neither set -> application-default credentials (valid)."""
    a = GeeAuth(project="p")
    assert a.service_account is None and a.key_file is None


def test_gee_auth_both_present_allowed_and_expands_key():
    """Both set is valid, and key_file gets `~` expanded."""
    a = GeeAuth(project="p", service_account="sa@x.iam", key_file="~/ee.json")
    assert a.key_file == Path("~/ee.json").expanduser()   # no literal ~ remains


# ---------------------------------------------------------------------------
# EarthdataAuth: auth_strategy is a closed set.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["netrc", "environment", "interactive"])
def test_earthdata_auth_valid_strategies(strategy):
    assert EarthdataAuth(auth_strategy=strategy).auth_strategy == strategy


def test_earthdata_auth_invalid_strategy_rejected():
    """An unknown strategy (e.g. 'oauth') -> ValidationError."""
    with pytest.raises(ValidationError):
        EarthdataAuth(auth_strategy="oauth")


# ---------------------------------------------------------------------------
# sources <-> products: a region source must name a selected product, and a
# region that only uses global products can omit / blank the sources section.
# ---------------------------------------------------------------------------
def test_source_for_selected_product_ok(base_project):
    """Control: a source for a product that IS in `products` validates."""
    cfg = parse_config(base_project)                  # bathymetry is selected
    assert cfg.regions[0].sources[DataProduct.bathymetry].sources == ["cudem"]


def test_source_for_unselected_product_rejected(base_project):
    """A source for a product missing from `products` fails loudly."""
    base_project["products"] = {"mur": {"variable": "analysed_sst"}}   # drop bathymetry
    # region still has a bathymetry source -> now unselected
    with pytest.raises(ValidationError, match="non-selected product"):
        parse_config(base_project)


def test_blank_sources_allowed(base_project):
    """A region can leave `sources:` blank (null) -> no source options."""
    base_project["regions"][0]["sources"] = None
    cfg = parse_config(base_project)
    assert cfg.regions[0].sources == {}


def test_omitted_sources_allowed(base_project):
    """A region can omit `sources` entirely -> defaults to empty."""
    del base_project["regions"][0]["sources"]
    cfg = parse_config(base_project)
    assert cfg.regions[0].sources == {}


# ---------------------------------------------------------------------------
# Project._unique_names: names that must be unique across the project.
# ---------------------------------------------------------------------------
def test_duplicate_region_names_rejected(base_project):
    """Two regions sharing a name -> ValidationError."""
    # A second region with the SAME name but a unique AoI name, so the failure
    # is the region-name clash (not the AoI-name check).
    base_project["regions"].append({
        "name": "r1",   # duplicates the existing region's name
        "areas": [{"name": "a2", "center_lat": 46.0, "center_lon": -124.0,
                   "buffer_ns_km": 10, "buffer_ew_km": 10}],
    })
    with pytest.raises(ValidationError, match="region names must be unique"):
        parse_config(base_project)


def test_duplicate_aoi_names_across_project_rejected(base_project):
    """The same AoI name in two DIFFERENT regions -> ValidationError."""
    # A second, uniquely-named region whose AoI reuses "a1" (already in r1), so
    # region names stay unique and the failure is the cross-project AoI clash.
    base_project["regions"].append({
        "name": "r2",   # unique region name
        "areas": [{"name": "a1", "center_lat": 46.0, "center_lon": -124.0,
                   "buffer_ns_km": 10, "buffer_ew_km": 10}],
    })
    with pytest.raises(ValidationError, match="AoI names must be unique"):
        parse_config(base_project)


# --------------------------------------------------------------------------- #
# Unknown option keys. A key nothing reads is not harmless: it is a config that LIES --
# it states an intent, the run ignores it, and the provenance honestly records what the
# run did, so nothing anywhere contradicts the config file.
# --------------------------------------------------------------------------- #
def _cfg(base_project, product, opts):
    cfg = copy.deepcopy(base_project)
    cfg["products"][product] = opts
    return cfg


def test_unknown_product_option_is_rejected(base_project):
    with pytest.raises(ValidationError, match="SILENTLY IGNORED"):
        parse_config(_cfg(base_project, "bathymetry", {"not_a_real_key": 1}))


def test_a_typo_gets_a_suggestion(base_project):
    with pytest.raises(ValidationError, match="did you mean 'stats_subgrid_m'"):
        parse_config(_cfg(base_project, "bathymetry", {"stats_subgrid": 10}))


def test_bathymetry_sources_is_a_list_read_by_the_module(base_project):
    """Distinct-data sources are a STACKED list (D10): the config lists the DEMs, and the
    module reads exactly that list (no `source`/`fallback`/`default_source` any more)."""
    from coastal_sst_data.processes import bathymetry
    cfg = _cfg(base_project, "bathymetry", {"sources": ["cudem", "gmrt"]})
    cfg["regions"][0]["sources"] = None      # no region override -> the global list flows through
    p = parse_config(cfg)
    assert bathymetry._build_eff(p)["ds"]["a1"]["sources"] == ["cudem", "gmrt"]


@pytest.mark.parametrize("removed", ["source", "fallback", "default_source", "dem_source"])
def test_removed_bathymetry_source_keys_are_rejected(base_project, removed):
    """S3 removed pick-one/fallback for bathymetry. An un-migrated config setting any of these
    fails loudly (pointing at `sources`) rather than being silently ignored."""
    with pytest.raises(ValidationError, match="SILENTLY IGNORED"):
        parse_config(_cfg(base_project, "bathymetry", {removed: "cudem"}))


def test_empty_or_unknown_stacked_sources_are_rejected(base_project):
    with pytest.raises(ValidationError, match="is empty"):
        parse_config(_cfg(base_project, "bathymetry", {"sources": []}))
    with pytest.raises(ValidationError, match="unknown source"):
        parse_config(_cfg(base_project, "bathymetry", {"sources": ["gebco"]}))


# S4 removed pick-one/fallback for cmems, met, tides too -- an un-migrated config setting any
# of these must FAIL loudly (extra="forbid"), never be silently ignored (§6.2).
@pytest.mark.parametrize("product,removed", [
    ("cmems", "source"), ("cmems", "fallback"), ("cmems", "dataset_id"),
    ("met", "source"), ("met", "fallback"), ("met", "overpass_from"),
    ("tides", "source"), ("tides", "default_source"), ("tides", "fallback"),
    ("tides", "fallback_distance_km"),
])
def test_removed_source_keys_are_rejected(base_project, product, removed):
    base_project.setdefault("auth", {})["copernicus"] = {"auth_strategy": "netrc"}  # cmems
    with pytest.raises(ValidationError, match="SILENTLY IGNORED"):
        parse_config(_cfg(base_project, product, {removed: "x"}))


def test_datacube_overpass_met_key_is_rejected(base_project):
    """`datacube.overpass_met` moved to the `met_overpass` product (D14). An old config still
    setting it fails loudly rather than quietly producing no overpass channels."""
    base_project["datacube"] = {"overpass_met": ["airtemp"]}
    with pytest.raises(ValidationError):
        parse_config(base_project)


def test_bad_met_overpass_combinations_are_rejected(base_project):
    """A combo naming an unloaded sensor or a bad source fails at load, not silently as an
    empty overpass channel (the sharpest regression risk in the met split)."""
    base_project["products"]["met_overpass"] = {"combinations": [["eco", "hrrr"]]}
    # ecostress isn't selected in base_project -> combo names an unloaded sensor
    with pytest.raises(ValidationError, match="not selected"):
        parse_config(base_project)
    # a good sensor but a bad source
    base_project["products"]["ecostress"] = None
    base_project["products"]["met_overpass"] = {"combinations": [["eco", "gfs"]]}
    with pytest.raises(ValidationError, match="not valid here"):
        parse_config(base_project)


def test_bad_tide_overpass_combinations_are_rejected(base_project):
    """tide_overpass (D17) combos live on the tides product and are validated against TIDE
    sources -- a bad source or unloaded sensor fails at load."""
    base_project["products"]["ecostress"] = None
    base_project["products"]["tides"] = {"overpass_combinations": [["eco", "gfs"]]}
    with pytest.raises(ValidationError, match="not valid here"):
        parse_config(base_project)
    # a source that is valid for MET but not for TIDES must still be rejected here
    base_project["products"]["tides"] = {"overpass_combinations": [["eco", "hrrr"]]}
    with pytest.raises(ValidationError, match="not valid here"):
        parse_config(base_project)


def test_unknown_region_source_key_is_rejected(base_project):
    cfg = copy.deepcopy(base_project)
    cfg["regions"][0]["sources"] = {"bathymetry": {"sourcez": ["cudem"]}}
    with pytest.raises(ValidationError, match="did you mean 'sources'"):
        parse_config(cfg)


def test_known_region_source_key_is_accepted(base_project):
    cfg = copy.deepcopy(base_project)
    cfg["regions"][0]["sources"] = {"bathymetry": {"sources": ["cudem"],
                                                   "datum_offset_m": 1.3}}
    parse_config(cfg)          # must not raise


def test_the_working_example_config_still_loads():
    """The registry must describe what real configs actually use."""
    load_config(Path(__file__).parents[1] / "examples" / "config.test.yaml")
