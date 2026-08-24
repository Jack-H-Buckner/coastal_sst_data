"""Config -> ECOSTRESS wiring. These exercise the adapter (`_build_eff`) that
maps a validated Project into the flat dict the acquisition code consumes.
No network: they never call earthaccess, only the pure mapping."""

import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import pytest
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from coastal_sst_data.config import (
    load_config, parse_config, DataProduct, AreaOfInterest, GridSpec,
)
from coastal_sst_data import grid, products
from coastal_sst_data.processes import ecostress
from .conftest import UniformDs

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


def _ds(eff):
    """The settings ONE AoI runs with. `eff["ds"]` is keyed by AoI, because every product
    resolves its options per AoI (region override -> project default). ECOSTRESS is a
    global product with no region-varying options, so every AoI resolves alike -- take any."""
    return next(iter(eff["ds"].values()))



# A realistic ECOSTRESS granule id (without the trailing _<LAYER>.tif).
GRANULE_STEM = "ECOv002_L2T_LSTE_25520_009_10UEU_20230105T123623_0710_01"


class FakeGranule:
    """Stand-in for an earthaccess DataGranule: only needs data_links()."""
    def __init__(self, *suffixes):
        self._links = [f"https://data/{GRANULE_STEM}_{s}.tif" for s in suffixes]

    def data_links(self):
        return self._links


class BadGranule:
    """A granule whose data_links() blows up (network/metadata failure)."""
    def data_links(self):
        raise RuntimeError("boom")


def test_build_eff_maps_example_config():
    """The example config maps to the expected acquisition parameters."""
    eff = ecostress._build_eff(load_config(EXAMPLE))

    # product constants (module defaults) + config overrides
    assert _ds(eff)["short_name"] == "ECO_L2T_LSTE"
    assert _ds(eff)["versions"] == ["v003", "v002"]  # STACKED collection versions (D10)
    assert _ds(eff)["layers"] == ecostress.LAYERS
    assert _ds(eff)["categorical"] == ecostress.CATEGORICAL

    # shared project settings flow through
    assert eff["earthdata"]["auth_strategy"] == "netrc"
    assert eff["fmt"] == "netcdf"                      # default output format
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["grid"]["resolution_m"] == 100.0
    # Per-version trees hang under eco_root: ECOSTRESS/<tag>/aligned/<aoi>.
    assert eff["eco_root"] == Path("path/to/data") / "ECOSTRESS"

    # AoI geometry now lives in the shared grid (grid.py), not in eff.
    assert "aois" not in eff


def test_build_eff_requires_ecostress_selected(base_project):
    """Calling the adapter when ecostress isn't a selected product is an error."""
    # base_project selects bathymetry + mur (no ecostress)
    with pytest.raises(ValueError, match="ecostress is not a selected product"):
        ecostress._build_eff(parse_config(base_project))


# ---------------------------------------------------------------------------
# Granule parsing helpers: pure, no network (fed a FakeGranule).
# ---------------------------------------------------------------------------
def test_parse_acq_time_extracts_datetime():
    """The YYYYMMDDTHHMMSS stamp is pulled out of the filename."""
    assert ecostress.parse_acq_time(f"{GRANULE_STEM}_LST.tif") == datetime(2023, 1, 5, 12, 36, 23)


def test_parse_acq_time_returns_none_without_timestamp():
    assert ecostress.parse_acq_time("no_timestamp_here.tif") is None


def test_granule_name_strips_layer_suffix():
    """The trailing _<LAYER>.tif is stripped back to the granule id."""
    assert ecostress.granule_name(FakeGranule("LST")) == GRANULE_STEM


def test_granule_name_handles_broken_links():
    """A granule whose data_links() raises degrades to a placeholder, not a crash."""
    assert ecostress.granule_name(BadGranule()) == "<granule>"


def test_filter_links_selects_requested_layers():
    """Each requested role maps to the URL ending in its suffix."""
    g = FakeGranule("LST", "cloud", "water")
    out = ecostress.filter_links_for_granule(g, {"sst": "LST", "cloud": "cloud", "water": "water"})
    assert set(out) == {"sst", "cloud", "water"}
    assert out["sst"].endswith("_LST.tif")           # sst role -> the LST asset
    assert out["cloud"].endswith("_cloud.tif")


def test_filter_links_omits_and_warns_missing_layers(caplog):
    """A requested layer absent from the granule is omitted (and warned), not faked."""
    g = FakeGranule("LST")                            # only the LST asset present
    with caplog.at_level("WARNING"):
        out = ecostress.filter_links_for_granule(g, {"sst": "LST", "cloud": "cloud"})
    assert set(out) == {"sst"}                        # cloud dropped
    assert "cloud" in caplog.text                     # and reported


# ---------------------------------------------------------------------------
# Synthetic COGs: small local GeoTIFFs standing in for the remote ECOSTRESS
# assets, so the raster core (read_window_reproject / process_granule) can be
# exercised offline. The layout is fixed and known so tests can predict output.
# ---------------------------------------------------------------------------
@pytest.fixture
def aoi_grid():
    """A small AoI's shared grid (100 m, auto UTM) to build/reproject onto."""
    area = AreaOfInterest(name="test_aoi", center_lat=45.52, center_lon=-123.925,
                          buffer_ns_km=8.0, buffer_ew_km=8.0)
    return grid.compute_aoi_grid(area, GridSpec())


def make_granule_cogs(dir_path, g, *, sst_kelvin=290.0):
    """Write synthetic sst/water/cloud GeoTIFFs for one 'overpass' over grid `g`.

    Source is native ~70 m in the SAME CRS as the target grid, spanning the AoI
    bounds plus a pad. Fixed, known layout:
        sst   = sst_kelvin everywhere (all finite)
        water = all WATER (the value for that is taken from the registry, see below)
        cloud = 1 over the WEST half, 0 over the EAST half
    So the expected `valid` mask on the target grid is 1 in the east, 0 in the
    west. Returns {role: path-str}, ready to pass as process_granule's role_to_file.

    The water VALUE is derived from `SensorSpec.water_is_land` rather than written as a
    literal. This fixture used to hardcode 1 = water while the registry said 1 = LAND and
    the assembler's fixtures in test_datacube.py encoded the registry's answer -- so the two
    test modules asserted opposite things about the same layer and neither noticed. Deriving
    it means flipping the registry flag (once real granules settle the question) moves the
    fixtures with it instead of turning them red for the wrong reason.
    """
    src_res = 70.0
    minx, miny, maxx, maxy = g.geom_proj.bounds
    minx, miny, maxx, maxy = minx - 1500, miny - 1500, maxx + 1500, maxy + 1500
    W = int(round((maxx - minx) / src_res))
    H = int(round((maxy - miny) / src_res))
    transform = from_origin(minx, maxy, src_res, src_res)

    sst = np.full((H, W), sst_kelvin, dtype="float32")
    water_val = 0.0 if products.spec(DataProduct.ecostress).sensor.water_is_land else 1.0
    water = np.full((H, W), water_val, dtype="float32")
    cloud = np.zeros((H, W), dtype="float32")
    cloud[:, : W // 2] = 1.0                          # west (low-x) columns cloudy

    paths = {}
    for role, arr, nodata in [("sst", sst, np.nan), ("water", water, None),
                              ("cloud", cloud, None)]:
        p = dir_path / f"{role}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1,
                           dtype="float32", crs=g.target_crs, transform=transform,
                           nodata=nodata) as dst:
            dst.write(arr, 1)
        paths[role] = str(p)
    return paths


def test_synthetic_cogs_are_readable_and_cover_aoi(tmp_path, aoi_grid):
    """Smoke check: the generator writes valid rasters that span the AoI."""
    paths = make_granule_cogs(tmp_path, aoi_grid)
    assert set(paths) == {"sst", "water", "cloud"}
    da = rioxarray.open_rasterio(paths["sst"], masked=True)
    assert str(da.rio.crs) == aoi_grid.target_crs           # same CRS as the grid
    # source bounds contain the AoI's projected bounds (so the window read is valid)
    left, bottom, right, top = da.rio.bounds()
    minx, miny, maxx, maxy = aoi_grid.geom_proj.bounds
    assert left <= minx and bottom <= miny and right >= maxx and top >= maxy


def test_read_cog_window(tmp_path, aoi_grid):
    """The shared windowed COG reader (grid.read_cog_window), which ECOSTRESS, Landsat,
    land-cover and bathymetry all read their rasters through."""
    paths = make_granule_cogs(tmp_path, aoi_grid)
    assert set(paths) == {"sst", "water", "cloud"}
    out = grid.read_cog_window(paths["sst"], aoi_grid, resampling=Resampling.bilinear,
                               pad_m=ecostress.ECO_PAD_M)
    assert out.rio.shape == (aoi_grid.height, aoi_grid.width)      # exact target dimensions
    assert str(out.rio.crs) == aoi_grid.target_crs          # reprojected to the target CRS
    assert out.rio.transform() == aoi_grid.transform        # pixel-for-pixel alignment
    vals = out.values
    assert np.isfinite(vals).any()                   # not silently all-NaN
    finite = vals[np.isfinite(vals)]
    assert finite.min() == pytest.approx(290.0, abs=1e-3)   # bilinear of a constant == constant
    assert finite.max() == pytest.approx(290.0, abs=1e-3)


def test_process_granule(tmp_path, aoi_grid, base_project):
    """tests the proces_granual file against a sunthetic dataset and check that
    misisng observations align with the values baked into in the synthetic data"""
    # Select ecostress in the config BEFORE validating it. Assigning into
    # `parsed.products` afterwards would insert a raw str key and a raw dict where the
    # model expects a DataProduct and a ProductOptions -- pydantic does not validate an
    # in-place dict mutation, so the model would be left in an invalid state (and the
    # options bag, being a plain dict, would silently read back as defaults anyway).
    cfg = copy.deepcopy(base_project)
    cfg["products"]["ecostress"] = {"versions": ["v002"]}
    eff = ecostress._build_eff(parse_config(cfg))
    role_to_file = make_granule_cogs(tmp_path, aoi_grid)
    # run() processes ONE version at a time, injecting its collection number as `version`; a
    # direct process_granule call mirrors that.
    vcfg = {**_ds(eff), "version": "002"}
    out = ecostress.process_granule(role_to_file, vcfg, eff["grid"], aoi_grid,
                                    "test-aoi", "12:00pm")
    v = out["valid"].isel(time=0).values          # (y, x), uint8
    assert out["valid"].dtype == "uint8"
    assert v[:, : v.shape[1] // 4].sum() == 0     # deep west: cloudy -> all invalid
    assert v[:, 3 * v.shape[1] // 4 :].sum() > 0  # deep east: clear water -> some valid
    assert {"sst", "water", "cloud", "valid"} <= set(out.data_vars)
    assert out.sizes["y"] == aoi_grid.height and out.sizes["x"] == aoi_grid.width
    assert str(out["sst"].rio.crs) == aoi_grid.target_crs


def _granule_args(tmp_path, aoi_grid, base_project):
    cfg = copy.deepcopy(base_project)
    cfg["products"]["ecostress"] = {"versions": ["v002"]}
    eff = ecostress._build_eff(parse_config(cfg))
    # process_granule works on ONE version -> inject its collection number, as run() does.
    return ({**_ds(eff), "version": "002"}, eff["grid"], aoi_grid, "test-aoi", "12:00pm")


@pytest.mark.parametrize("lost", ["cloud", "water"])
def test_granule_is_dropped_when_a_mask_layer_fails_to_read(tmp_path, aoi_grid,
                                                            base_project, caplog, lost):
    """A granule whose mask COG failed is WORSE than no granule: without `water`/`cloud`
    the `valid` mask is never built, and the assembler reads that as a scene with nothing
    valid in it -- a silent total cloud-out, indistinguishable from a real overcast day."""
    roles = make_granule_cogs(tmp_path, aoi_grid)
    roles[lost] = str(tmp_path / "does_not_exist.tif")     # the COG read will fail

    with caplog.at_level("WARNING"):
        out = ecostress.process_granule(roles, *_granule_args(tmp_path, aoi_grid, base_project))

    assert out is None                                     # dropped, not written degraded
    assert "dropping granule" in caplog.text and lost in caplog.text


def test_granule_is_dropped_when_the_mask_asset_was_never_published(tmp_path, aoi_grid,
                                                                    base_project, caplog):
    """The other way a layer goes missing: the granule simply has no cloud asset, so it is
    filtered out upstream and never even attempted. Same outcome required."""
    roles = make_granule_cogs(tmp_path, aoi_grid)
    roles.pop("cloud")                                     # as filter_links_for_granule leaves it

    with caplog.at_level("WARNING"):
        out = ecostress.process_granule(roles, *_granule_args(tmp_path, aoi_grid, base_project))

    assert out is None
    assert "dropping granule" in caplog.text


def test_expected_vars_follows_the_configured_layers(tmp_path, aoi_grid, base_project):
    """The skip guard's completeness check is derived from config, not hardcoded: demanding
    a layer the user never asked for would re-fetch every granule forever."""
    assert set(ecostress.expected_vars({"layers": ecostress.LAYERS})) == {
        "sst", "water", "cloud", "valid"}
    # a config that never asks for the masks cannot be required to have them
    assert ecostress.expected_vars({"layers": {"sst": "LST"}}) == ("sst",)


# ---------------------------------------------------------------------------
# run() orchestration: control flow around the (expensive) network + raster
# work. The boundary calls are stubbed with spies, so these are offline and
# only assert WHICH work happened -- never touching Earthdata or real rasters.
# ---------------------------------------------------------------------------
# Suffixes that make filter_links_for_granule find every role in LAYERS.
_GRANULE_SUFFIXES = ("LST", "cloud", "water", "QC")
# Deterministic aligned-file stem for a FakeGranule (from GRANULE_STEM's stamp and tile).
_GRANULE_TSTR = "20230105T123623"
_GRANULE_TILE = "10UEU"


def _eff(tmp_path, *, overwrite=False, fmt="netcdf"):
    """A minimal `eff` dict for run(); eco_root points at a temp dir. One version (v002) keeps
    the run-orchestration tests single-collection; per-version fan-out is covered separately."""
    return {
        "ds": UniformDs({"short_name": "ECO_L2T_LSTE", "versions": ["v002"],
                         "layers": ecostress.LAYERS,
                         "categorical": ecostress.CATEGORICAL}),
        "grid": {"resampling_continuous": "bilinear",
                 "resampling_categorical": "nearest", "to_celsius": False},
        "eco_root": tmp_path,
        "fmt": fmt,
        "overwrite": overwrite,
        "earthdata": {"auth_strategy": "netrc"},
        "time": {"start_date": "2023-01-01", "end_date": "2023-01-31"},
    }


def _grid(name, lat=45.52, lon=-123.925):
    area = AreaOfInterest(name=name, center_lat=lat, center_lon=lon,
                          buffer_ns_km=8.0, buffer_ew_km=8.0)
    return grid.compute_aoi_grid(area, GridSpec())


@pytest.fixture
def run_stubs(monkeypatch):
    """Replace run()'s boundary calls with spies. Set calls['granules'] to
    control what the stubbed search returns; the lists record what happened."""
    calls = {"login": [], "search": [], "open": [], "process": [], "write": [],
             "granules": []}
    monkeypatch.setattr(ecostress, "login",
                        lambda strategy: calls["login"].append(strategy))
    monkeypatch.setattr(ecostress, "search_granules",
                        lambda ds_cfg, bbox, start, end: (calls["search"].append(bbox) or calls["granules"]))
    monkeypatch.setattr(ecostress.earthaccess, "open",
                        lambda urls: (calls["open"].append(list(urls)) or [None] * len(list(urls))))
    # A bare object() would do, except run() now stamps provenance onto the Dataset it
    # gets back -- so the stub has to look like one.
    monkeypatch.setattr(ecostress, "process_granule",
                        lambda *a, **k: (calls["process"].append(True) or xr.Dataset()))
    # The writer now lives in `store` (one implementation for every product), so the spy
    # goes there. `stem` is the filename without an extension -- `<aoi>_<stamp>`.
    monkeypatch.setattr(ecostress.store, "write_output",
                        lambda ds, out_dir, stem, fmt, **k: calls["write"].append(
                            (out_dir, stem, fmt)))
    return calls


# --- 1. dry-run must search but never open or write ---
def test_run_dry_run_searches_but_does_not_open(tmp_path, aoi_grid, run_stubs):
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None,
                  dry_run=True, list_layers=False)
    assert run_stubs["search"]          # it searched
    assert run_stubs["open"] == []      # ...but never opened
    assert run_stubs["write"] == []     # ...nor wrote


# --- 2. skip-if-exists vs overwrite ---
def _aligned_path(tmp_path, name, tag="v002"):
    # Per-version tree: <eco_root>/<tag>/aligned/<aoi> (eco_root == tmp_path in these tests).
    aoi_out = tmp_path / tag / "aligned" / name
    aoi_out.mkdir(parents=True, exist_ok=True)
    # TILED: the tile is part of the name, or every granule of one overpass collides here.
    return aoi_out / f"{name}_{_GRANULE_TSTR}_{_GRANULE_TILE}.nc"


def _write_aligned_file(tmp_path, aoi_grid, *, drop=()):
    """A COMPLETE aligned granule on disk -- or, with `drop`, one missing a layer, as a
    granule whose mask COG failed to download would be."""
    H, W = aoi_grid.height, aoi_grid.width
    layers = {v: (("y", "x"), np.zeros((H, W), "float32"))
              for v in ("sst", "water", "cloud") if v not in drop}
    if "valid" not in drop:
        layers["valid"] = (("y", "x"), np.zeros((H, W), "uint8"))
    xr.Dataset(layers).to_netcdf(_aligned_path(tmp_path, aoi_grid.name))


def test_run_skips_existing_output(tmp_path, aoi_grid, run_stubs):
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    _write_aligned_file(tmp_path, aoi_grid)          # a COMPLETE output already on disk
    ecostress.run(_eff(tmp_path, overwrite=False), {aoi_grid.name: aoi_grid},
                  None, False, False)
    assert run_stubs["open"] == []       # complete file -> skipped, never opened
    assert run_stubs["write"] == []


def test_run_refetches_a_truncated_output(tmp_path, aoi_grid, run_stubs):
    """The whole point of the completeness check: an empty/truncated file left by a run
    that died mid-write must NOT be mistaken for a finished one and skipped forever."""
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    _aligned_path(tmp_path, aoi_grid.name).touch()   # 0-byte file, as a killed write leaves
    ecostress.run(_eff(tmp_path, overwrite=False), {aoi_grid.name: aoi_grid},
                  None, False, False)
    assert run_stubs["open"]             # re-fetched, not skipped
    assert run_stubs["write"]


def test_run_refetches_a_granule_missing_its_cloud_mask(tmp_path, aoi_grid, run_stubs):
    """A COMPLETE write of degraded content: the cloud COG failed, so the granule has sst
    but no cloud/valid. Atomicity cannot see this -- only the layer check can."""
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    _write_aligned_file(tmp_path, aoi_grid, drop=("cloud", "valid"))
    ecostress.run(_eff(tmp_path, overwrite=False), {aoi_grid.name: aoi_grid},
                  None, False, False)
    assert run_stubs["open"]             # re-fetched, not trusted


def test_run_overwrite_reprocesses_existing(tmp_path, aoi_grid, run_stubs):
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    _write_aligned_file(tmp_path, aoi_grid)
    ecostress.run(_eff(tmp_path, overwrite=True), {aoi_grid.name: aoi_grid},
                  None, False, False)
    assert run_stubs["open"]             # overwrite -> reprocessed
    assert run_stubs["write"]


# --- 3. only_aoi filtering ---
def test_run_unknown_aoi_raises_systemexit(tmp_path, aoi_grid, run_stubs):
    with pytest.raises(SystemExit):
        ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, ["nope"], False, False)


def test_run_only_aoi_processes_requested_subset(tmp_path, run_stubs):
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    grids = {"a": _grid("a"), "b": _grid("b", lat=48.45, lon=-122.58)}
    ecostress.run(_eff(tmp_path), grids, ["a"], False, False)
    assert len(run_stubs["search"]) == 1     # only the requested AoI was searched


# --- 4. no granules -> skip cleanly ---
def test_run_no_granules_skips_cleanly(tmp_path, aoi_grid, run_stubs):
    run_stubs["granules"] = []               # search finds nothing
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False, False)
    assert run_stubs["search"]               # searched
    assert run_stubs["open"] == []           # nothing to open, no crash
    assert run_stubs["write"] == []


# --- 5. STACKED collection versions (D10) ---
@pytest.fixture
def version_stubs(monkeypatch):
    """Spies that record the collection version each search ran and the per-version write dir."""
    calls = {"searched": [], "written": []}
    monkeypatch.setattr(ecostress, "login", lambda strategy: None)
    monkeypatch.setattr(ecostress, "search_granules",
                        lambda ds_cfg, bbox, start, end:
                        calls["searched"].append(ds_cfg["version"]) or
                        [FakeGranule(*_GRANULE_SUFFIXES)])
    monkeypatch.setattr(ecostress.earthaccess, "open", lambda urls: [None] * len(list(urls)))
    monkeypatch.setattr(ecostress, "process_granule", lambda *a, **k: xr.Dataset())
    monkeypatch.setattr(ecostress.store, "write_output",
                        lambda ds, out_dir, stem, fmt, **k: calls["written"].append(Path(out_dir)))
    return calls


def _multi_eff(tmp_path, versions):
    eff = _eff(tmp_path)
    eff["ds"] = UniformDs({"short_name": "ECO_L2T_LSTE", "versions": list(versions),
                           "layers": ecostress.LAYERS, "categorical": ecostress.CATEGORICAL})
    return eff


def test_run_fans_out_over_versions_into_per_version_trees(tmp_path, aoi_grid, version_stubs):
    """Each configured version searches its OWN collection (Earthdata number, `v` stripped) and
    writes its OWN ECOSTRESS/<tag>/aligned tree, in config order."""
    ecostress.run(_multi_eff(tmp_path, ["v003", "v002"]),
                  {aoi_grid.name: aoi_grid}, None, False, False)
    assert version_stubs["searched"] == ["003", "002"]       # v-stripped, config order
    # write dir is <eco_root>/<tag>/aligned/<aoi> -> the tag is two levels up from the aoi dir.
    tags = {p.parent.parent.name for p in version_stubs["written"]}
    assert tags == {"v002", "v003"}


def test_run_only_source_narrows_to_one_version(tmp_path, aoi_grid, version_stubs):
    """`only_source` (the pipeline/CLI narrower) restricts the fan-out to one version tag."""
    ecostress.run(_multi_eff(tmp_path, ["v003", "v002"]),
                  {aoi_grid.name: aoi_grid}, None, False, False, only_source="v002")
    assert version_stubs["searched"] == ["002"]
    assert {p.parent.parent.name for p in version_stubs["written"]} == {"v002"}


def test_acquire_rejects_an_unknown_version(base_project):
    """A version tag no code recognises fails loudly rather than silently dropping a collection
    (config validation catches it at load; acquire guards directly too)."""
    cfg = copy.deepcopy(base_project)
    cfg["products"]["ecostress"] = {"versions": ["v999"]}
    with pytest.raises(ValueError, match="version not recognized|unknown source"):
        ecostress.acquire(parse_config(cfg), grids={}, dry_run=True)

        
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])




def test_source_attr_names_the_version_actually_searched(tmp_path, aoi_grid, base_project):
    """The attr provenance.source_of() reads, and that every eco_* field in every cube is
    stamped with. It used to be the literal "v003" while the search ran v002."""
    roles = make_granule_cogs(tmp_path, aoi_grid)
    args = list(_granule_args(tmp_path, aoi_grid, base_project))
    ds_cfg = dict(args[0])
    ds_cfg.update(short_name="ECO_L2T_LSTE", version="002")
    args[0] = ds_cfg
    out = ecostress.process_granule(roles, *args)
    assert out.attrs["source"] == "ECOSTRESS ECO_L2T_LSTE v002"

    ds_cfg = dict(ds_cfg); ds_cfg["version"] = "003"      # a config asking for v003
    args[0] = ds_cfg
    out = ecostress.process_granule(roles, *args)
    assert out.attrs["source"] == "ECOSTRESS ECO_L2T_LSTE v003"   # follows config, not a literal


# --------------------------------------------------------------------------- #
# TILED product: one overpass, several granules, one filename each.
#
# ECO_L2T delivers an overpass as granules that share an acquisition time EXACTLY and differ
# only by MGRS tile. Named by time alone they all resolve to one output path, so the first
# tile written made `store.done` report every other tile of that overpass as already
# processed -- and an AoI wider than a tile silently kept one tile's footprint with nodata
# across the rest. Nothing failed; the files looked valid and the run reported success.
# --------------------------------------------------------------------------- #
class TiledGranule:
    """A FakeGranule for a chosen tile, optionally carrying a catalogue bounding box."""

    def __init__(self, tile, *, stamp=_GRANULE_TSTR, bbox=None,
                 suffixes=_GRANULE_SUFFIXES):
        self.stem = f"ECOv002_L2T_LSTE_25520_009_{tile}_{stamp}_0710_01"
        self._links = [f"https://data/{self.stem}_{s}.tif" for s in suffixes]
        self._bbox = bbox

    def data_links(self):
        return self._links

    def __getitem__(self, key):
        if key != "umm" or self._bbox is None:
            raise KeyError(key)
        w, s, e, n = self._bbox
        return {"SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {
            "BoundingRectangles": [{"WestBoundingCoordinate": w,
                                    "SouthBoundingCoordinate": s,
                                    "EastBoundingCoordinate": e,
                                    "NorthBoundingCoordinate": n}]}}}}


def test_tile_id_reads_the_mgrs_field():
    assert ecostress.tile_id(GRANULE_STEM) == "10UEU"
    assert ecostress.tile_id(
        "ECOv003_L2T_LSTE_36012_005_55GDP_20260122T222901_0713_01") == "55GDP"


def test_tile_id_is_read_relative_to_the_stamp_not_by_position():
    """A positional split that quietly returned the orbit number instead would name every
    tile of an overpass identically again -- the exact failure the tile is in the name to
    stop. An id it cannot read must say None, not guess."""
    assert ecostress.tile_id("ECOv002_L2T_LSTE_25520_009_20230105T123623_0710_01") is None
    assert ecostress.tile_id("not_a_granule_id") is None


def test_two_tiles_of_one_overpass_both_get_written(tmp_path, aoi_grid, run_stubs):
    """The regression. Two granules, same instant, different tiles: both must reach disk
    under names that differ, or the AoI keeps whichever tile was reached first."""
    run_stubs["granules"] = [TiledGranule("10UEU"), TiledGranule("10UEV")]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False, False)

    stems = [stem for _out, stem, _fmt in run_stubs["write"]]
    assert stems == [f"{aoi_grid.name}_{_GRANULE_TSTR}_10UEU",
                     f"{aoi_grid.name}_{_GRANULE_TSTR}_10UEV"]
    assert len(run_stubs["open"]) == 2, "the second tile was skipped as already processed"


def test_a_tile_on_disk_does_not_skip_its_neighbour(tmp_path, aoi_grid, run_stubs):
    """`store.done` is a statement about ONE granule's output. Sharing a filename turned it
    into a statement about the whole overpass, which is how the other tiles were lost."""
    run_stubs["granules"] = [TiledGranule(_GRANULE_TILE), TiledGranule("10UEV")]
    _write_aligned_file(tmp_path, aoi_grid)          # only the 10UEU tile is complete

    ecostress.run(_eff(tmp_path, overwrite=False), {aoi_grid.name: aoi_grid},
                  None, False, False)

    assert len(run_stubs["open"]) == 1, "the absent tile was skipped along with the present one"
    assert [stem for _o, stem, _f in run_stubs["write"]] == [
        f"{aoi_grid.name}_{_GRANULE_TSTR}_10UEV"]


def test_a_granule_outside_the_aoi_polygon_is_never_opened(tmp_path, aoi_grid, run_stubs):
    """The search is by BOUNDING BOX, so the catalogue returns 110 km tiles that touch the
    box and miss the AoI. Reading one costs five COG opens to produce an all-nodata file."""
    inside = aoi_grid.geom_lonlat().bounds
    run_stubs["granules"] = [
        TiledGranule("10UEU", bbox=inside),
        TiledGranule("10UEV", bbox=(inside[0] + 40, inside[1] + 40,
                                    inside[0] + 41, inside[1] + 41)),
    ]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False, False)

    assert len(run_stubs["open"]) == 1
    assert [stem for _o, stem, _f in run_stubs["write"]] == [
        f"{aoi_grid.name}_{_GRANULE_TSTR}_10UEU"]


def test_a_granule_with_no_extent_metadata_is_kept(tmp_path, aoi_grid, run_stubs):
    """"The metadata does not say" must never mean "drop it": that would lose real data on a
    catalogue quirk, which is a far worse failure than reading one tile too many."""
    run_stubs["granules"] = [TiledGranule("10UEU", bbox=None)]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False, False)
    assert len(run_stubs["open"]) == 1


class _Handle:
    """An fsspec-like handle that records whether it was closed."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _handle_stub(run_stubs, monkeypatch, opened):
    def fake_open(urls):
        urls = list(urls)
        run_stubs["open"].append(urls)
        made = [_Handle() for _ in urls]
        opened.extend(made)
        return made
    monkeypatch.setattr(ecostress.earthaccess, "open", fake_open)


def test_the_fsspec_handles_are_closed(tmp_path, aoi_grid, run_stubs, monkeypatch):
    """Each handle holds an HTTPS connection and a one-thread executor for its background
    block cache, released only when the object is collected. A multi-year AoI is thousands of
    granules, and tiling multiplied the handles per overpass -- the abandoned sockets were
    still on the process, in CLOSE-WAIT, hours later."""
    opened = []
    _handle_stub(run_stubs, monkeypatch, opened)
    run_stubs["granules"] = [TiledGranule("10UEU")]

    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False, False)

    assert opened, "the stub never ran"
    assert all(h.closed for h in opened)


def test_the_handles_are_closed_when_the_read_raises(tmp_path, aoi_grid, run_stubs,
                                                     monkeypatch):
    """The failing path is the one that matters: a run that leaks only when it errors leaks
    exactly when a DAAC is misbehaving and the retries are piling handles up fastest."""
    opened = []
    _handle_stub(run_stubs, monkeypatch, opened)
    monkeypatch.setattr(ecostress, "process_granule",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("read failed")))
    run_stubs["granules"] = [TiledGranule("10UEU")]

    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False, False)

    assert opened, "the stub never ran"
    assert all(h.closed for h in opened)
    assert run_stubs["write"] == []


def test_a_granule_crossing_the_antimeridian_is_kept(tmp_path, aoi_grid, run_stubs):
    """W > E is two boxes, and `shapely.box` would build the inside-out one -- which
    intersects nothing, so the filter would drop a real granule and say nothing."""
    run_stubs["granules"] = [TiledGranule("10UEU", bbox=(179.0, -45.0, -179.0, -44.0))]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False, False)
    assert len(run_stubs["open"]) == 1


# --------------------------------------------------------------------------- #
# ROLE -> ASSET binding.
#
# `LAYERS` maps TWO roles onto one asset (`sst` and `lst` are both _LST.tif), so the URL
# list is SHORTER than the role list and `earthaccess.open` returns one handle per DISTINCT
# url. Binding roles to handles positionally therefore paired every role after `sst` with the
# NEXT role's asset and dropped the last role entirely: `cloud` got the water mask, `water`
# got the QC bit field, `quality` was never opened. Nothing failed -- the aligned files were
# written full of plausible numbers, and the assembler then computed ECOSTRESS validity from
# a QC field with a water rule, got an empty mask on every granule, and collapsed each day's
# mosaic onto its first tile: an AoI with data in one corner and a run that reported success.
#
# The old stub returned one handle per URL *including duplicates*, which is why the tests
# never saw it. These stubs dedupe, like the real earthaccess.
# --------------------------------------------------------------------------- #
def _dedup_open(calls):
    """An `earthaccess.open` that returns one handle per DISTINCT url, as the real one does.

    The handle is the url itself, so the role->asset binding is directly assertable.
    """
    def _open(urls):
        uniq = list(dict.fromkeys(list(urls)))
        calls.append(uniq)
        return list(uniq)
    return _open


def test_every_role_is_bound_to_its_own_asset(tmp_path, aoi_grid, run_stubs, monkeypatch):
    """The regression. Each role must receive the asset ITS OWN suffix names."""
    seen = {}
    monkeypatch.setattr(ecostress.earthaccess, "open", _dedup_open(run_stubs["open"]))
    monkeypatch.setattr(ecostress, "process_granule",
                        lambda role_to_file, *a, **k: (seen.update(role_to_file)
                                                       or xr.Dataset()))
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None,
                  dry_run=False, list_layers=False)

    assert seen, "process_granule was never reached"
    for role, suffix in ecostress.LAYERS.items():
        assert role in seen, f"role {role!r} never got an asset"
        assert seen[role].endswith(f"_{suffix}.tif"), (
            f"role {role!r} was bound to {seen[role]!r}, which is not its _{suffix}.tif")
    # the two roles that share an asset must share the SAME handle, not two opens
    assert seen["sst"] == seen["lst"]
    assert len(run_stubs["open"][0]) == 4        # 5 roles, 4 distinct URLs


def test_quality_role_is_not_dropped(tmp_path, aoi_grid, run_stubs, monkeypatch):
    """`quality` sits last in LAYERS, so a positional zip over a deduped list loses exactly
    it -- and the QC gate the assembler applies to ECOSTRESS then never runs at all."""
    seen = {}
    monkeypatch.setattr(ecostress.earthaccess, "open", _dedup_open(run_stubs["open"]))
    monkeypatch.setattr(ecostress, "process_granule",
                        lambda role_to_file, *a, **k: (seen.update(role_to_file)
                                                       or xr.Dataset()))
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None,
                  dry_run=False, list_layers=False)
    assert "quality" in seen and seen["quality"].endswith("_QC.tif")


def test_short_open_result_raises_instead_of_mispairing(tmp_path, aoi_grid, run_stubs,
                                                        monkeypatch):
    """Fewer handles than URLs must fail loudly. Silently getting a short list is exactly
    how the roles came to be shifted, and a guess is worse than a failure here."""
    monkeypatch.setattr(ecostress.earthaccess, "open",
                        lambda urls: [None] * (len(list(urls)) - 1))
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    ecostress.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None,
                  dry_run=False, list_layers=False)
    assert run_stubs["write"] == []            # nothing written from a mis-paired granule


def test_categorical_layers_are_chosen_by_role_not_by_suffix(tmp_path, aoi_grid,
                                                             base_project, monkeypatch):
    """`categorical` lists ROLE names; indexing `layers[role]` yields the SUFFIX.

    It matched for cloud/water only because those spell both the same. `quality`'s suffix is
    `QC`, so the bit-packed QA layer fell through to the CONTINUOUS resampler -- bilinear
    interpolation between bit patterns, which produces flags that mean nothing.
    """
    used = {}
    real = ecostress.read_cog_window

    def spy(fobj, g, *, resampling, **kw):
        used[str(fobj)] = resampling
        return real(fobj, g, resampling=resampling, **kw)

    monkeypatch.setattr(ecostress, "read_cog_window", spy)
    roles = make_granule_cogs(tmp_path, aoi_grid)
    roles["quality"] = roles["water"]          # any readable raster; the ROLE is the point
    args = list(_granule_args(tmp_path, aoi_grid, base_project))
    ds_cfg = dict(args[0])
    ds_cfg["layers"] = {**ecostress.LAYERS}
    args[0] = ds_cfg
    ecostress.process_granule(roles, *args)

    from rasterio.enums import Resampling
    assert used[roles["quality"]] == Resampling.nearest
    assert used[roles["cloud"]] == Resampling.nearest
    assert used[roles["sst"]] == Resampling.bilinear


def test_acquisition_and_assembler_agree_about_which_cells_are_water(tmp_path, aoi_grid,
                                                                     base_project):
    """Both stages must read the water layer with the SAME polarity.

    They did not. `process_granule` built `valid` from `water > 0`; the assembler recomputed
    the very same mask from `water < 0.5`. Whichever was wrong produced an EMPTY validity
    mask -- and an empty mask is not a visible failure, it is a day that looks cloudy. Worse,
    with every granule of a day tying at zero valid pixels the first one becomes the mosaic
    base and no other granule can outrank it, so a tiled AoI silently keeps one tile.
    """
    from coastal_sst_data.processes import datacube

    roles = make_granule_cogs(tmp_path, aoi_grid)          # all water, east half clear
    args = list(_granule_args(tmp_path, aoi_grid, base_project))
    ds = ecostress.process_granule(roles, *args)
    acq = ds["valid"].isel(time=0).values.astype(bool)

    out = tmp_path / "aligned.nc"
    ds.to_netcdf(out)
    sensor = products.spec(DataProduct.ecostress).sensor
    s, c, asm, fp, _pid = datacube._read_granule(
        out, aoi_grid.height, aoi_grid.width,
        water_is_land=sensor.water_is_land, use_cloud=sensor.use_cloud,
        qset=list(sensor.qc_levels) if sensor.qc_levels else None,
        trust_valid=sensor.trust_valid, read_fp=False)

    assert acq.any(), "acquisition produced an all-empty valid mask over all-water input"
    assert asm.any(), "the assembler produced an all-empty valid mask over the same file"
    # The two gates differ by design (acquisition screens on `cloud`, the assembler on QC),
    # so they need not be equal -- but neither may be a subset-of-nothing, and every cell the
    # assembler keeps must be one the acquisition also called water.
    assert (asm & ~np.isfinite(s)).sum() == 0
