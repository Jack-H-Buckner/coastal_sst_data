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
from rasterio.transform import from_origin

from coastal_sst_data.config import (
    load_config, parse_config, DataProduct, AreaOfInterest, GridSpec,
)
from coastal_sst_data import grid
from coastal_sst_data.processes import ecostress

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"

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
    assert eff["ds"]["short_name"] == "ECO_L2T_LSTE"
    assert eff["ds"]["version"] == "002"              # from ecostress options
    assert eff["ds"]["layers"] == ecostress.LAYERS
    assert eff["ds"]["categorical"] == ecostress.CATEGORICAL

    # shared project settings flow through
    assert eff["earthdata"]["auth_strategy"] == "netrc"
    assert eff["fmt"] == "netcdf"                      # default output format
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["grid"]["resolution_m"] == 100.0
    assert eff["out_dir"] == Path("path/to/data") / "ECOSTRESS" / "aligned"

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
        water = 1 everywhere (all water)
        cloud = 1 over the WEST half, 0 over the EAST half
    So the expected `valid` mask on the target grid is 1 in the east, 0 in the
    west. Returns {role: path-str}, ready to pass as process_granule's role_to_file.
    """
    src_res = 70.0
    minx, miny, maxx, maxy = g.geom_proj.bounds
    minx, miny, maxx, maxy = minx - 1500, miny - 1500, maxx + 1500, maxy + 1500
    W = int(round((maxx - minx) / src_res))
    H = int(round((maxy - miny) / src_res))
    transform = from_origin(minx, maxy, src_res, src_res)

    sst = np.full((H, W), sst_kelvin, dtype="float32")
    water = np.ones((H, W), dtype="float32")
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


def test_read_window_reproject(tmp_path, aoi_grid):
    paths = make_granule_cogs(tmp_path, aoi_grid)
    assert set(paths) == {"sst", "water", "cloud"}
    target_crs = aoi_grid.target_crs
    geom_target = aoi_grid.geom_proj
    transform = aoi_grid.transform
    width = aoi_grid.width
    height = aoi_grid.height
    out = ecostress.read_window_reproject(paths["sst"], geom_target, target_crs, transform, width,
                          height, "bilinear")
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
    cfg["products"]["ecostress"] = {"version": "002"}
    parsed = parse_config(cfg)
    eff = ecostress._build_eff(parsed)
    role_to_file  = make_granule_cogs(tmp_path, aoi_grid)
    target_crs = aoi_grid.target_crs
    geom_target = aoi_grid.geom_proj
    transform = aoi_grid.transform
    width = aoi_grid.width
    height = aoi_grid.height
    ds_cfg = eff["ds"]
    grid_cfg = eff["grid"]
    out = ecostress.process_granule(role_to_file, ds_cfg, grid_cfg, target_crs, transform,
                    width, height, geom_target, "test-aoi", "12:00pm")
    v = out["valid"].isel(time=0).values          # (y, x), uint8
    assert out["valid"].dtype == "uint8"
    assert v[:, : v.shape[1] // 4].sum() == 0     # deep west: cloudy -> all invalid
    assert v[:, 3 * v.shape[1] // 4 :].sum() > 0  # deep east: clear water -> some valid
    assert {"sst", "water", "cloud", "valid"} <= set(out.data_vars)
    assert out.sizes["y"] == aoi_grid.height and out.sizes["x"] == aoi_grid.width
    assert str(out["sst"].rio.crs) == aoi_grid.target_crs


# ---------------------------------------------------------------------------
# run() orchestration: control flow around the (expensive) network + raster
# work. The boundary calls are stubbed with spies, so these are offline and
# only assert WHICH work happened -- never touching Earthdata or real rasters.
# ---------------------------------------------------------------------------
# Suffixes that make filter_links_for_granule find every role in LAYERS.
_GRANULE_SUFFIXES = ("LST", "cloud", "water", "QC")
# Deterministic aligned-file stem for a FakeGranule (from GRANULE_STEM's stamp).
_GRANULE_TSTR = "20230105T123623"


def _eff(tmp_path, *, overwrite=False, fmt="netcdf"):
    """A minimal `eff` dict for run(); out_dir points at a temp dir."""
    return {
        "ds": {"short_name": "ECO_L2T_LSTE", "version": "002",
               "layers": ecostress.LAYERS, "categorical": ecostress.CATEGORICAL},
        "grid": {"resampling_continuous": "bilinear",
                 "resampling_categorical": "nearest", "to_celsius": False},
        "out_dir": tmp_path,
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
    monkeypatch.setattr(ecostress, "write_output",
                        lambda ds, out_dir, name, fmt: calls["write"].append((out_dir, name, fmt)))
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
def _touch_aligned_file(tmp_path, name):
    aoi_out = tmp_path / name
    aoi_out.mkdir(parents=True, exist_ok=True)
    (aoi_out / f"{name}_{_GRANULE_TSTR}.nc").touch()


def test_run_skips_existing_output(tmp_path, aoi_grid, run_stubs):
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    _touch_aligned_file(tmp_path, aoi_grid.name)     # output already on disk
    ecostress.run(_eff(tmp_path, overwrite=False), {aoi_grid.name: aoi_grid},
                  None, False, False)
    assert run_stubs["open"] == []       # existing file -> skipped, never opened
    assert run_stubs["write"] == []


def test_run_overwrite_reprocesses_existing(tmp_path, aoi_grid, run_stubs):
    run_stubs["granules"] = [FakeGranule(*_GRANULE_SUFFIXES)]
    _touch_aligned_file(tmp_path, aoi_grid.name)
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

        




