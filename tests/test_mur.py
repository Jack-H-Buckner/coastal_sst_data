
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from coastal_sst_data.config import load_config, parse_config, AreaOfInterest, GridSpec
from coastal_sst_data.processes import mur
from coastal_sst_data import grid


EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


def test_build_eff_maps_example_config():
    """The example config maps to the expected MUR acquisition parameters."""
    eff = mur._build_eff(load_config(EXAMPLE))
    # product constants (module defaults) + config overrides
    assert eff["ds"]["short_name"] == "MUR-JPL-L4-GLOB-v4.1"
    assert eff["ds"]["variable"] == "analysed_sst"     # from the mur options
    assert eff["ds"]["pad_deg"] == 0.05                # default
    # shared project settings flow through
    assert eff["earthdata"]["auth_strategy"] == "netrc"
    assert eff["fmt"] == "netcdf"                       # default output format
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["grid"]["resolution_m"] == 100.0
    assert eff["out_dir"] == Path("path/to/data") / "MUR" / "aligned"
    # AoI geometry now lives in the shared grid (grid.py), not in eff.
    assert "aois" not in eff


def test_build_eff_requires_mur_selected(base_project):
    """Calling the adapter when mur isn't a selected product is an error."""
    base_project["products"] = {"bathymetry": None}    # drop mur (public source, no auth)
    with pytest.raises(ValueError, match="mur is not a selected product"):
        mur._build_eff(parse_config(base_project))


def test_build_eff_defaults_when_options_omitted(base_project):
    """A bare `mur:` (no options) falls back to product defaults."""
    base_project["products"]["mur"] = None             # bare -> default options
    eff = mur._build_eff(parse_config(base_project))
    assert eff["ds"]["variable"] == mur.DEFAULT_VARIABLE       # "analysed_sst"
    assert eff["ds"]["pad_deg"] == mur.DEFAULT_PAD_DEG         # 0.05
    assert eff["ds"]["short_name"] == mur.SHORT_NAME
    assert eff["fmt"] == "netcdf"
    assert eff["overwrite"] is False


def test_build_eff_applies_option_overrides(base_project):
    """mur options override the product defaults."""
    base_project["products"]["mur"] = {
        "variable": "analysed_sst_anomaly", "pad_deg": 0.2,
        "output_format": "geotiff", "overwrite": True,
    }
    eff = mur._build_eff(parse_config(base_project))
    assert eff["ds"]["variable"] == "analysed_sst_anomaly"
    assert eff["ds"]["pad_deg"] == 0.2
    assert eff["fmt"] == "geotiff"
    assert eff["overwrite"] is True



@pytest.fixture
def aoi_grid():
    """A small AoI's shared grid to subset/reproject onto."""
    area = AreaOfInterest(name="test_aoi", center_lat=45.52, center_lon=-123.925,
                          buffer_ns_km=8.0, buffer_ew_km=8.0)
    return grid.compute_aoi_grid(area, GridSpec())


def make_mur_granule(path, g, *, sst_kelvin=290.0, when="2026-06-15",
                     res_deg=0.01, margin_deg=0.3, variable="analysed_sst"):
    """Write a synthetic daily MUR NetCDF for one 'day' over grid `g`.

    `variable` (Kelvin) on an ascending lat/lon grid at ~1 km (0.01 deg), like
    real MUR, spanning g.search_bbox + a margin. dims (time, lat, lon). Ascending
    lat/lon so xarray's .sel(lat=slice(...)) works; margin > the search pad so the
    subset has data on every edge. Returns the path (str).
    """
    w, s, e, n = g.search_bbox
    lat = np.arange(s - margin_deg, n + margin_deg, res_deg, dtype="float64")
    lon = np.arange(w - margin_deg, e + margin_deg, res_deg, dtype="float64")
    sst = np.full((1, lat.size, lon.size), sst_kelvin, dtype="float32")
    ds = xr.Dataset(
        {variable: (("time", "lat", "lon"), sst)},
        coords={"time": [np.datetime64(when)], "lat": lat, "lon": lon},
    )
    ds[variable].attrs["units"] = "kelvin"
    ds.to_netcdf(path)          # NETCDF4/HDF5 -> the h5netcdf reader opens it
    return str(path)


def test_subset_and_reproject(tmp_path, aoi_grid):
    g = aoi_grid
    path = make_mur_granule(tmp_path / "mur_20260615.nc", g, sst_kelvin=290.0)
    grid_cfg = {"resampling_continuous": "bilinear", "to_celsius": False}
    out, t = mur.subset_and_reproject(
        path, "analysed_sst", g.search_bbox, 0.05,
        g.target_crs, g.transform, g.width, g.height, g.geom_proj, grid_cfg)
    # lands on the shared grid
    assert out.rio.shape == (g.height, g.width)
    assert str(out.rio.crs) == g.target_crs
    assert out.rio.transform() == g.transform
    # data came through, constant preserved; timestamp parsed from the granule
    finite = out.values[np.isfinite(out.values)]
    assert finite.size and finite.min() == pytest.approx(290.0, abs=1e-3)
    assert t == pd.Timestamp("2026-06-15")


def test_subset_and_reproject_to_celsius(tmp_path, aoi_grid):
    g = aoi_grid
    path = make_mur_granule(tmp_path / "mur.nc", g, sst_kelvin=290.0)
    out, _ = mur.subset_and_reproject(
        path, "analysed_sst", g.search_bbox, 0.05,
        g.target_crs, g.transform, g.width, g.height, g.geom_proj,
        {"resampling_continuous": "bilinear", "to_celsius": True})
    finite = out.values[np.isfinite(out.values)]
    assert finite.min() == pytest.approx(290.0 - 273.15, abs=1e-3)   # 16.85
