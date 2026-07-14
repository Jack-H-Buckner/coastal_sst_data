
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from coastal_sst_data.config import load_config, parse_config, AreaOfInterest, GridSpec
from coastal_sst_data.processes import met
from coastal_sst_data import grid


EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


# --------------------------------------------------------------------------- #
# _build_eff: config -> acquisition params
# --------------------------------------------------------------------------- #
def test_build_eff_maps_example_config():
    """The example config maps to the expected met acquisition parameters."""
    eff = met._build_eff(load_config(EXAMPLE))
    assert eff["ds"]["chain"] == ["hrrr", "era5"]          # source: auto, fallback: era5
    assert eff["ds"]["variables"] == ["airtemp", "wind", "swrad", "cloud"]
    assert eff["ds"]["daily_mean_hours"] == [0, 6, 12, 18]
    assert eff["ds"]["era5_zarr"] == met.ARCO_ERA5_URI     # default ARCO store
    # shared project settings flow through
    assert eff["fmt"] == "netcdf"
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["grid"]["resolution_m"] == 100.0
    assert eff["out_dir"] == Path("path/to/data") / "MET" / "aligned"
    # overpass dirs resolve to the thermal-scene aligned dirs
    assert eff["overpass_dirs"] == [
        Path("path/to/data") / "ECOSTRESS" / "aligned",
        Path("path/to/data") / "LANDSAT" / "aligned",
    ]
    # AoI geometry lives in the shared grid (grid.py), not in eff.
    assert "aois" not in eff


def test_build_eff_requires_met_selected(base_project):
    """Calling the adapter when met isn't a selected product is an error."""
    base_project["products"] = {"bathymetry": None}        # drop met
    with pytest.raises(ValueError, match="met is not a selected product"):
        met._build_eff(parse_config(base_project))


def test_build_eff_defaults_when_options_omitted(base_project):
    """A bare `met:` (no options) falls back to product defaults."""
    base_project["products"]["met"] = None                 # bare -> default options
    eff = met._build_eff(parse_config(base_project))
    assert eff["ds"]["chain"] == ["hrrr", "era5"]
    assert eff["ds"]["variables"] == met.DEFAULT_VARIABLES
    assert eff["ds"]["daily_mean_hours"] == met.DEFAULT_MEAN_HOURS
    assert eff["overpass_dirs"] == []                       # no overpass_from set
    assert eff["overwrite"] is False


def test_build_eff_applies_option_overrides(base_project):
    """met options override the product defaults."""
    base_project["products"]["met"] = {
        "source": "era5", "variables": ["airtemp", "wind"],
        "regrid_radius_m": 3000, "output_format": "geotiff", "overwrite": True,
    }
    eff = met._build_eff(parse_config(base_project))
    assert eff["ds"]["chain"] == ["era5"]                  # era5-only, no fallback dup
    assert eff["ds"]["variables"] == ["airtemp", "wind"]
    assert eff["ds"]["regrid_radius_m"] == 3000.0
    assert eff["fmt"] == "geotiff"
    assert eff["overwrite"] is True


# --------------------------------------------------------------------------- #
# Source chain resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source,fallback,expected", [
    ("auto", "era5", ["hrrr", "era5"]),   # default
    ("hrrr", "era5", ["hrrr", "era5"]),
    ("era5", "era5", ["era5"]),           # fallback == primary -> not duplicated
    ("auto", "none", ["hrrr"]),           # fallback disabled
    ("era5", "none", ["era5"]),
])
def test_resolve_chain(source, fallback, expected):
    assert met._resolve_chain(source, fallback) == expected


def test_resolve_chain_rejects_unknown_source():
    with pytest.raises(ValueError, match="source 'gfs' not recognized"):
        met._resolve_chain("gfs", "era5")


# --------------------------------------------------------------------------- #
# HRRR domain selection
# --------------------------------------------------------------------------- #
def _grid_at(lat, lon):
    area = AreaOfInterest(name="a", center_lat=lat, center_lon=lon,
                          buffer_ns_km=8.0, buffer_ew_km=8.0)
    return grid.compute_aoi_grid(area, GridSpec())


def test_hrrr_model_for_grid():
    assert met._hrrr_model_for_grid(_grid_at(45.5, -123.9), "auto") == "hrrr"    # PNW
    assert met._hrrr_model_for_grid(_grid_at(58.0, -135.0), "auto") == "hrrrak"  # SE Alaska
    assert met._hrrr_model_for_grid(_grid_at(50.0, 0.0), "auto") is None         # Europe -> ERA5
    # explicit model overrides the auto domain check
    assert met._hrrr_model_for_grid(_grid_at(50.0, 0.0), "hrrr") == "hrrr"


# --------------------------------------------------------------------------- #
# ERA5 unit harmonization
# --------------------------------------------------------------------------- #
def test_era5_normalize():
    fields = {
        "airtemp": np.array([290.0], dtype="float32"),
        "swrad": np.array([3.6e6], dtype="float32"),      # 3.6 MJ/m2 over 1 h
        "cloud_cover": np.array([0.5], dtype="float32"),  # fraction
    }
    out = met._era5_normalize(fields)
    assert out["airtemp"][0] == pytest.approx(290.0)      # K unchanged
    assert out["swrad"][0] == pytest.approx(1000.0)       # 3.6e6 / 3600 -> W/m2
    assert out["cloud_cover"][0] == pytest.approx(50.0)   # 0.5 -> 50 %


# --------------------------------------------------------------------------- #
# Dataset assembly (source-agnostic)
# --------------------------------------------------------------------------- #
def test_to_dataset_builds_wind_speed_and_units(aoi_grid):
    g = aoi_grid
    shp = g.shape
    grids = {
        "airtemp": np.full(shp, 290.0, dtype="float32"),
        "wind_u": np.full(shp, 3.0, dtype="float32"),
        "wind_v": np.full(shp, 4.0, dtype="float32"),
        "swrad": np.full(shp, 500.0, dtype="float32"),
        "cloud_cover": np.full(shp, 40.0, dtype="float32"),
    }
    ds = met.to_dataset(grids, g, "2026-06-15", to_celsius=False)
    assert ds["airtemp"].isel(time=0).shape == shp
    assert ds["airtemp"].attrs["units"] == "K"
    assert float(ds["wind_speed"].isel(time=0).values[0, 0]) == pytest.approx(5.0)  # 3-4-5
    assert ds["wind_speed"].attrs["units"] == "m s-1"
    assert ds["swrad"].attrs["units"] == "W m-2"
    assert ds["cloud_cover"].attrs["units"] == "%"
    assert "time" in ds.dims
    assert str(ds.rio.crs) == g.target_crs


def test_to_dataset_to_celsius(aoi_grid):
    g = aoi_grid
    grids = {"airtemp": np.full(g.shape, 290.0, dtype="float32")}
    ds = met.to_dataset(grids, g, "2026-06-15", to_celsius=True)
    assert float(ds["airtemp"].isel(time=0).values[0, 0]) == pytest.approx(290.0 - 273.15)
    assert ds["airtemp"].attrs["units"] == "degC"


# --------------------------------------------------------------------------- #
# Reference time of day (the cube's met channel; default 10:30 Landsat overpass)
# --------------------------------------------------------------------------- #
def test_parse_hhmm():
    assert met.parse_hhmm("10:30") == pytest.approx(10.5)
    assert met.parse_hhmm("6") == pytest.approx(6.0)
    assert met.parse_hhmm(None) is None          # None -> no reference snapshot
    assert met.parse_hhmm("") is None


def test_solar_reference_is_the_same_time_of_day_in_every_aoi():
    """10:30 LOCAL SOLAR is a different UTC hour per AoI -- that is the whole point.

    A fixed UTC hour would be mid-morning in Oregon and the middle of the night in
    Maine, so cross-AoI forcing would not be like-for-like.
    """
    day = pd.Timestamp("2026-06-15")
    # Tillamook (-123.9): UTC = 10.5 + 8.26 = 18.76 -> 19:00
    assert met.reference_time_utc(day, -123.925, 10.5, "solar").hour == 19
    # Chesapeake (-76.0): UTC = 10.5 + 5.07 = 15.57 -> 16:00
    assert met.reference_time_utc(day, -76.0, 10.5, "solar").hour == 16


def test_utc_basis_is_taken_literally():
    day = pd.Timestamp("2026-06-15")
    t = met.reference_time_utc(day, -123.925, 10.5, "utc")
    assert (t.hour, t.day) == (10, 15)           # longitude ignored


def test_solar_reference_rolls_the_date_across_the_dateline():
    day = pd.Timestamp("2026-06-15")
    t = met.reference_time_utc(day, 170.0, 10.5, "solar")   # 10.5 - 11.33 = -0.83 h
    assert t.day == 14 and t.hour == 23                     # previous UTC day


def test_reference_defaults_are_the_landsat_overpass():
    eff = met._build_eff(load_config(EXAMPLE))
    assert eff["ds"]["reference_time"] == "10:30"
    assert eff["ds"]["reference_basis"] == "solar"
