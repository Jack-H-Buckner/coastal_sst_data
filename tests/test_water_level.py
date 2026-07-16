"""Water level (the DOWNSTREAM reconstruction from the cube's raw ingredients). Unit-tests
the tide interpolation to the overpass hour and the waterline re-referencing / submerged-
exposed classes, then checks the cube ships the raw ingredients (per-source elevation + datum
attrs, daily tide, overpass hour) rather than pre-derived channels. No network, no real data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import datacube, water_level
from coastal_sst_data.processes.water_level import EXPOSED, SUBMERGED, UNKNOWN


AOI = "aoi1"


def _project(tmp_path, **extra):
    cfg = {
        "name": "wl", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {"bathymetry": None, "tides": None},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    }
    cfg.update(extra)
    return parse_config(cfg)


@pytest.fixture
def project(tmp_path):
    return _project(tmp_path)


@pytest.fixture
def grids(project):
    return grid.project_grids(project)


@pytest.fixture
def days(project):
    return pd.date_range(project.time.start_date, project.time.end_date, freq="D")


def _write(project, sub, fname, ds):
    d = project.output_dir / sub / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / fname)


def write_tides(project, days, *, amplitude=1.0):
    """Hourly tide over the window: a 12 h sinusoid, 0 m at each day's midnight."""
    times = pd.date_range(days[0], days[-1] + pd.Timedelta("23h"), freq="h")
    hours = np.arange(len(times), dtype="float64")
    tide = amplitude * np.sin(2 * np.pi * hours / 12.0)
    ds = xr.Dataset({"tide": (("time",), tide.astype("float32"))},
                    coords={"time": times})
    _write(project, "TIDE", f"{AOI}_tides.nc", ds)


def write_bathymetry(project, g, elev, src="cudem", *, datum_offset_m=0.0,
                     datum_status="unresolved_assumed_zero"):
    """One DEM source's aligned output, with its datum offset stamped on (as the bathymetry
    module now produces it). Writes to the per-source tree BATHYMETRY/<src>/aligned/<aoi>."""
    xs, ys = g.xy_centers()
    depth = np.where(elev < 0, -elev, 0.0).astype("float32")
    ds = xr.Dataset({"elevation": (("y", "x"), elev), "depth": (("y", "x"), depth),
                     "depth_p25": (("y", "x"), depth), "depth_p75": (("y", "x"), depth)},
                    coords={"y": ys, "x": xs})
    ds.attrs.update(bathy_source=src, datum_offset_m=datum_offset_m, datum_status=datum_status,
                    datum_method="stub",
                    dem_vertical_datum="NAVD88" if src == "cudem" else "MSL_APPROX")
    d = project.output_dir / "BATHYMETRY" / src / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}.nc")


def write_ecostress(project, g, day, hour):
    H, W = g.height, g.width
    xs, ys = g.xy_centers()
    ds = xr.Dataset({
        "sst": (("time", "y", "x"), np.full((1, H, W), 286.0, "float32")),
        "cloud": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
        "water": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
        "quality": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
    }, coords={"time": [day], "y": ys, "x": xs})
    stamp = day.strftime("%Y%m%d") + f"T{hour:02d}0000"
    _write(project, "ECOSTRESS", f"{AOI}_{stamp}.nc", ds)


# --------------------------------------------------------------------------- #
# tide_at_overpass
# --------------------------------------------------------------------------- #
def test_tide_interpolated_to_the_overpass_hour(project, days):
    write_tides(project, days)
    s = water_level.load_tide_series(
        project.output_dir / "TIDE" / "aligned" / AOI, AOI)
    # Scene at 03:30 on day 0; the series is hourly, so the value is the linear
    # interpolant between 03:00 and 04:00.
    tide = water_level.tide_at_overpass(s, days, [3.5, np.nan, np.nan])
    expect = 0.5 * (np.sin(2 * np.pi * 3 / 12) + np.sin(2 * np.pi * 4 / 12))
    assert tide[0] == pytest.approx(expect, abs=1e-6)
    assert np.isnan(tide[1:]).all()             # no overpass -> no tide


def test_tide_is_nan_without_a_series(days):
    tide = water_level.tide_at_overpass(None, days, [3.5, 4.0, 5.0])
    assert np.isnan(tide).all()


def test_tide_outside_the_series_is_nan_not_clamped(project, days):
    write_tides(project, days)
    s = water_level.load_tide_series(
        project.output_dir / "TIDE" / "aligned" / AOI, AOI)
    # The series stops at 23:00 on the last day; an overpass past its end must not
    # inherit the final height.
    late = pd.date_range(days[-1] + pd.Timedelta("1D"), periods=1, freq="D")
    assert np.isnan(water_level.tide_at_overpass(s, late, [12.0])).all()


# --------------------------------------------------------------------------- #
# water_level_fields
# --------------------------------------------------------------------------- #
def test_waterline_is_zero_and_classes_follow_its_sign():
    # A ramp of ground elevations spanning the waterline, at a +2 m tide.
    elev = np.array([[-10.0, 1.0, 2.0, 3.0]], "float32")
    rel, cls = water_level.water_level_fields(elev, np.array([2.0]))

    assert rel.shape == (1, 1, 4) and rel.dtype == np.float32
    np.testing.assert_allclose(rel[0, 0], [-12.0, -1.0, 0.0, 1.0])
    # Ground BELOW the tide-adjusted waterline is submerged (0 itself is water).
    assert list(cls[0, 0]) == [SUBMERGED, SUBMERGED, SUBMERGED, EXPOSED]
    assert cls.dtype == np.uint8


def test_a_flat_is_submerged_at_high_tide_and_exposed_at_low():
    elev = np.array([[-0.5]], "float32")          # a tidal flat 0.5 m below MSL
    rel, cls = water_level.water_level_fields(elev, np.array([1.5, -1.5]))
    assert rel[0, 0, 0] == pytest.approx(-2.0)    # 2 m of water over it
    assert rel[1, 0, 0] == pytest.approx(1.0)     # 1 m above the waterline
    assert list(cls[:, 0, 0]) == [SUBMERGED, EXPOSED]


def test_unknown_where_the_dem_or_the_tide_is_missing():
    elev = np.array([[-1.0, np.nan]], "float32")
    _, cls = water_level.water_level_fields(elev, np.array([0.5, np.nan]))
    assert cls[0, 0, 0] == SUBMERGED
    assert cls[0, 0, 1] == UNKNOWN                # no DEM here
    assert (cls[1] == UNKNOWN).all()              # no overpass -> no tide that day


def test_datum_offset_shifts_the_waterline():
    # datum_offset_m = where MSL sits in the DEM's datum. With MSL 1 m up in DEM
    # units (the NAVD88 case), ground at +0.5 m in DEM units is really 0.5 m BELOW
    # MSL, so a 0 m tide leaves it submerged.
    elev = np.array([[0.5]], "float32")
    rel, cls = water_level.water_level_fields(elev, np.array([0.0]), datum_offset_m=1.0)
    assert rel[0, 0, 0] == pytest.approx(-0.5)
    assert cls[0, 0, 0] == SUBMERGED
    # Without the offset the same ground reads as exposed.
    rel0, cls0 = water_level.water_level_fields(elev, np.array([0.0]))
    assert rel0[0, 0, 0] == pytest.approx(0.5)
    assert cls0[0, 0, 0] == EXPOSED


# --------------------------------------------------------------------------- #
# datum offset: resolved by the `datum` stage, or asserted per region.
# (The resolution chain itself is covered in tests/test_datum.py.)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Through the assembler
# --------------------------------------------------------------------------- #
def test_cube_ships_raw_water_level_ingredients_not_derived_channels(project, grids, days):
    """S1 (D12): the per-sensor water-level channels are GONE from the cube -- water level
    is a downstream computation now. What the cube ships instead are the RAW ingredients to
    reconstruct it (per-source elevation, per-day tide, each sensor's overpass hour) plus the
    DEM->MSL datum offset on each elevation_<src> channel, so downstream references it to MSL."""
    g = grids[AOI]
    elev = np.full((g.height, g.width), -0.5, "float32")
    write_bathymetry(project, g, elev, "cudem", datum_offset_m=1.3, datum_status="ok")
    write_tides(project, days)
    write_ecostress(project, g, days[0], hour=9)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    # The derived channels are gone...
    for pre in ("eco", "lst", "modis"):
        for suf in ("water_elev", "water_class", "tide"):
            assert f"{pre}_{suf}" not in ds.data_vars

    # ...and the raw ingredients to rebuild them ship instead: the per-source DEM elevation,
    # the daily tide series, and each sensor's overpass hour.
    assert ds["elevation_cudem"].isel(y=0, x=0).item() == pytest.approx(-0.5)
    assert np.isfinite(ds["tide"].isel(time=0).item())                      # daily tide ships
    assert ds["eco_hour"].isel(time=0).item() == pytest.approx(9.0)         # overpass hour

    # The DEM->MSL offset travels PER SOURCE, as attrs on that source's elevation channel.
    assert ds["elevation_cudem"].attrs["datum_offset_m"] == pytest.approx(1.3)
    assert ds["elevation_cudem"].attrs["datum_status"] == "ok"


def test_datum_offset_ships_on_the_elevation_channel(project, grids, days):
    """The datum offset is bathymetry-owned now -- it publishes as an attr on elevation_<src>
    regardless of whether tides ran (water level moved downstream, but MSL referencing needs it)."""
    g = grids[AOI]
    write_bathymetry(project, g, np.full((g.height, g.width), -1.0, "float32"),
                     "gmrt", datum_offset_m=0.0, datum_status="ok")
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert "datum_offset_m" in ds["elevation_gmrt"].attrs
    assert "datum_status" in ds["elevation_gmrt"].attrs

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
