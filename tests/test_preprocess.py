"""Post-assembly preprocessing stage. Assembles a tiny SYNTHETIC cube (reusing the datacube
test writers), then runs the preprocess steps and asserts what they added to it: the waterline
is pure glue over water_level.water_level_fields, the level-4 NN fill respects the land-cover
water mask and flags invented cells, the ASSEMBLED channels come through the rewrite untouched,
re-running is idempotent, and the import-time step invariants fire. No network, no real data.

The golden test (`test_preprocessed_golden_is_unchanged`) snapshots the cube AFTER preprocess,
kept separate from datacube_golden.json (the cube as `assemble` leaves it) so a change to one
stage does not force a regeneration of the other's golden."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import datacube, preprocess, water_level
from coastal_sst_data.processes.water_level import EXPOSED, SUBMERGED, UNKNOWN

# Reuse the datacube test's synthetic aligned-file writers + golden-snapshot helpers.
from tests.test_datacube import (
    AOI, write_mur, write_bathymetry, write_landcover, write_ecostress_two_scenes,
    write_cmems, write_tides, _write_full_fixture, _snapshot, _diff_snapshots)


def _project(tmp_path, **preprocess_cfg):
    return parse_config({
        "name": "pp", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {"bathymetry": None, "tides": None, "mur": None, "cmems": None,
                     "landcover": None, "ecostress": None},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
        "auth": {"earthdata": {"auth_strategy": "netrc"},
                 "copernicus": {"auth_strategy": "netrc"}},
        "preprocess": {"enabled": True, **preprocess_cfg},
    })


@pytest.fixture
def project(tmp_path):
    return _project(tmp_path, steps={"water_line": None, "fill_water": None,
                                     "filter_clouds": None})


@pytest.fixture
def grids(project):
    return grid.project_grids(project)


@pytest.fixture
def days(project):
    return pd.date_range(project.time.start_date, project.time.end_date, freq="D")


def _raw_cube(project, g, days):
    """The in-memory assembled raw cube from whatever aligned files are on disk."""
    return datacube.assemble_aoi(g, datacube._build_eff(project), days)


# --------------------------------------------------------------------------- #
# water_line: pure glue over water_level.water_level_fields
# --------------------------------------------------------------------------- #
def test_water_line_matches_the_reference_water_level_math(project, grids, days):
    g = grids[AOI]
    write_bathymetry(project, g, "cudem", datum_offset_m=1.3, datum_status="ok")
    write_landcover(project, g, land_cols=slice(0, 0))
    write_tides(project, g, days)                          # 12 h sinusoid, hourly on disk
    write_ecostress_two_scenes(project, g, days[0])        # eco flies on day 0 (clearest 20:00)

    ds_raw = _raw_cube(project, g, days)
    ds_out = preprocess.preprocess_aoi(ds_raw, g, preprocess._build_eff(project))

    assert "eco_water_elev" in ds_out.data_vars
    assert "eco_water_class" in ds_out.data_vars
    assert ds_out["eco_water_elev"].dtype == np.float32
    assert ds_out["eco_water_class"].dtype == np.uint8

    # The step is glue: recompute the reference fields directly and demand equality.
    elev = ds_raw["elevation_cudem"].values
    hours = ds_raw["eco_hour_v002"].values
    series = water_level.load_tide_series(
        project.output_dir / "TIDE" / "coops" / "aligned" / AOI, AOI)
    tide = water_level.tide_at_overpass(series, days, hours)
    exp_elev, exp_cls = water_level.water_level_fields(elev, tide, datum_offset_m=1.3)
    np.testing.assert_allclose(ds_out["eco_water_elev"].values, exp_elev, equal_nan=True)
    np.testing.assert_array_equal(ds_out["eco_water_class"].values, exp_cls)

    # Day 0 has an overpass -> a real class; the other days have none -> UNKNOWN.
    assert (ds_out["eco_water_class"].isel(time=0).values != UNKNOWN).any()
    assert (ds_out["eco_water_class"].isel(time=1).values == UNKNOWN).all()
    # The DEM elevation it was built from travels along, with its datum attrs.
    assert ds_out["elevation_cudem"].attrs["datum_offset_m"] == pytest.approx(1.3)


def test_water_line_no_tide_gives_all_unknown_not_a_crash(project, grids, days):
    g = grids[AOI]
    # DEM present, but NO tides on disk -> the overpass tide is NaN -> all-UNKNOWN, no crash.
    write_bathymetry(project, g, "cudem", datum_offset_m=0.0, datum_status="ok")
    write_landcover(project, g, land_cols=slice(0, 0))
    write_ecostress_two_scenes(project, g, days[0])
    ds_out = preprocess.preprocess_aoi(_raw_cube(project, g, days), g,
                                       preprocess._build_eff(project))
    assert (ds_out["eco_water_class"].values == UNKNOWN).all()


def test_water_line_no_dem_emits_nothing_not_a_crash(tmp_path, days, caplog):
    proj = _project(tmp_path, steps={"water_line": None})
    g = grid.project_grids(proj)[AOI]
    write_tides(proj, g, days)                     # tides + a sensor, but deliberately NO DEM
    write_ecostress_two_scenes(proj, g, days[0])
    with caplog.at_level("WARNING"):
        ds = preprocess.preprocess_aoi(_raw_cube(proj, g, days), g,
                                       preprocess._build_eff(proj))
    assert not [v for v in ds.data_vars if v.endswith(("_water_elev", "_water_class"))]
    assert "no elevation" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# fill_water: NN fill over the land-cover water mask, with a *_filled flag
# --------------------------------------------------------------------------- #
def test_fill_water_fills_over_water_flags_it_and_leaves_land_alone(project, grids, days):
    g = grids[AOI]
    # Land strip in cols 0-2; MUR holes at a land col (0) and water cols (5,6,7).
    write_landcover(project, g, land_cols=slice(0, 3))
    write_mur(project, g, days, water_hole_cols=[0, 5, 6, 7])
    ds_raw = _raw_cube(project, g, days)
    ds_out = preprocess.preprocess_aoi(ds_raw, g, preprocess._build_eff(project))

    mur0 = ds_out["mur_sst_gapfilled"].isel(time=0).values
    flag0 = ds_out["mur_sst_filled"].isel(time=0).values
    # Water holes were filled from the nearest finite pixel; the land hole stayed NaN.
    assert np.isfinite(mur0[:, 5:8]).all()
    assert np.isnan(mur0[:, 0]).all()
    # The flag marks exactly the invented-over-water cells (1), observed cells 0.
    assert (flag0[:, 5:8] == 1).all()
    assert (flag0[:, 0] == 0).all()          # a land NaN is not "filled"
    assert (flag0[:, 4] == 0).all()          # an observed water cell is not "filled"
    assert ds_out["mur_sst_filled"].dtype == np.uint8
    # The mask it filled against is in the same cube, and the SOURCE keeps its holes: the
    # gap-filled field is a companion to `mur_sst`, not a replacement for it.
    assert "landcover_water" in ds_out.data_vars
    assert np.isnan(ds_out["mur_sst"].isel(time=0).values[:, 5:8]).all()


def test_fill_water_also_fills_cmems_per_source(project, grids, days):
    g = grids[AOI]
    write_landcover(project, g, land_cols=slice(0, 0))          # all water
    write_cmems(project, g, days, land_cols=[6, 7], src="my_global")
    ds_raw = _raw_cube(project, g, days)
    ds_out = preprocess.preprocess_aoi(ds_raw, g, preprocess._build_eff(project))
    c = "cmems_thetao_0m_my_global"
    assert np.isfinite(ds_out[f"{c}_gapfilled"].isel(time=0).values[:, 6:8]).all()
    assert (ds_out[f"{c}_filled"].isel(time=0).values[:, 6:8] == 1).all()
    assert np.isnan(ds_out[c].isel(time=0).values[:, 6:8]).all()          # source untouched


def test_fill_water_without_a_mask_channel_fills_nothing(project, grids, days, caplog):
    """Defensive branch: a cube LACKING the mask channel entirely (filling over an unknown
    mask would fabricate data over land). The real assembler always emits landcover_water
    (unknown -> water), so build a bare cube by hand to exercise the guard."""
    g = grids[AOI]
    xs, ys = g.xy_centers()
    arr = np.full((len(days), g.height, g.width), 285.0, "float32")
    arr[:, :, 5:7] = np.nan
    ds_raw = xr.Dataset({"mur_sst": (("time", "y", "x"), arr)},
                        coords={"time": days, "y": ys, "x": xs}, attrs={"crs": g.target_crs})
    with caplog.at_level("WARNING"):
        ds_out = preprocess.preprocess_aoi(ds_raw, g, preprocess._build_eff(project))
    assert "mur_sst_gapfilled" not in ds_out.data_vars           # step declined
    assert "fill_water" in caplog.text


# --------------------------------------------------------------------------- #
# filter_clouds + filter_cloud_cover: cloud screening (see processes/cloud_filter.py).
# These build the raw cube BY HAND (bypassing assembly) so the cold anomaly / cloud field is
# exact, then run preprocess_aoi and assert the drops fold into sst/valid + the companion flags.
# --------------------------------------------------------------------------- #
def _setup(tmp_path, steps):
    proj = _project(tmp_path, steps=steps)
    return proj, grid.project_grids(proj)[AOI]


def _hand_cube(g, times, **channels):
    """A raw cube built by hand for a precise cloud-filter fixture. Arrays are keyed by name;
    3-D -> (time,y,x), 2-D -> (y,x), 1-D -> (time,)."""
    xs, ys = g.xy_centers()
    dims = {3: ("time", "y", "x"), 2: ("y", "x"), 1: ("time",)}
    data = {name: (dims[arr.ndim], arr) for name, arr in channels.items()}
    return xr.Dataset(data, coords={"time": pd.DatetimeIndex(times), "y": ys, "x": xs},
                      attrs={"crs": g.target_crs})


def _pp(proj, g, cube):
    return preprocess.preprocess_aoi(cube, g, preprocess._build_eff(proj))


def _pp_blocked(proj, g, cube, block_days):
    """`_pp`, but driven a block of days at a time -- the orchestration `_preprocess_blocked`
    performs, without the Zarr round trip, so a test can compare the two results directly."""
    eff = preprocess._build_eff(proj)
    all_days = pd.DatetimeIndex(cube["time"].values)
    window, cache = {}, {}
    census, _reads = preprocess.preprocess_census(cube, g, eff, all_days,
                                                  window=window, cache=cache)
    stale, _expected = preprocess.resolve_channel_plan(cube, census)
    blocks = [slice(i, min(i + block_days, len(all_days)))
              for i in range(0, len(all_days), block_days)]
    preprocess._run_window_stats(eff, cube, g, all_days, blocks, window=window, cache=cache)
    parts = [preprocess.preprocess_block(cube.isel(time=sl), g, eff, all_days=all_days,
                                         ds_all=cube, stale=stale, window=window, cache=cache)
             for sl in blocks]
    return xr.concat(parts, dim="time", data_vars="minimal", coords="minimal")


def _ones_valid(T, H, W):
    return np.ones((T, H, W), "uint8")


# ---- filter_clouds: offset method ---------------------------------------------------------- #
def test_filter_clouds_offset_drops_cold_anomalies(tmp_path):
    proj, g = _setup(tmp_path, {"filter_clouds": None})       # offset, threshold_k=5 (default)
    H, W = g.height, g.width
    eco = np.full((1, H, W), 286.0, "float32")
    eco[0, :, 5] = 278.0                                      # 7 K colder than MUR -> cloud
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      mur_sst=np.full((1, H, W), 285.0, "float32"))
    ds = _pp(proj, g, cube)

    flag = ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values
    assert (flag[:, 5] == 1).all() and flag.sum() == H        # only column 5 dropped
    assert np.isnan(ds["eco_sst_v002_clean"].isel(time=0).values[:, 5]).all()
    assert np.isfinite(ds["eco_sst_v002_clean"].isel(time=0).values[:, 6]).all()
    assert (ds["eco_valid_v002_clean"].isel(time=0).values[:, 5] == 0).all()
    assert ds["eco_sst_v002_clean_cloudfiltered"].dtype == np.uint8
    assert ds["eco_valid_v002_clean"].dtype == np.uint8
    assert ds["eco_sst_v002_clean"].dtype == np.float32


def test_filter_clouds_threshold_zero_is_a_noop(tmp_path):
    proj, g = _setup(tmp_path, {"filter_clouds": {"threshold_k": 0}})
    H, W = g.height, g.width
    eco = np.full((1, H, W), 286.0, "float32"); eco[0, :, 5] = 278.0
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      mur_sst=np.full((1, H, W), 285.0, "float32"))
    ds = _pp(proj, g, cube)
    # Disabled -> the step emits nothing, so there is no screened product at all and the
    # assembled channel is left exactly as it came in.
    assert "eco_sst_v002_clean_cloudfiltered" not in ds.data_vars
    assert "eco_sst_v002_clean" not in ds.data_vars
    np.testing.assert_array_equal(ds["eco_sst_v002"].values, eco)


def test_filter_clouds_keeps_pixels_where_baseline_is_nan(tmp_path):
    proj, g = _setup(tmp_path, {"filter_clouds": None})
    H, W = g.height, g.width
    mur = np.full((1, H, W), 285.0, "float32"); mur[0, :, 5] = np.nan
    eco = np.full((1, H, W), 286.0, "float32"); eco[0, :, 5] = 278.0
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W), mur_sst=mur)
    ds = _pp(proj, g, cube)
    # NaN baseline -> NaN diff -> kept, never flagged.
    assert (ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values[:, 5] == 0).all()
    assert np.isfinite(ds["eco_sst_v002_clean"].isel(time=0).values[:, 5]).all()


def test_filter_clouds_uses_fill_waters_filled_baseline(tmp_path):
    """With fill_water selected first, a MUR water hole is NN-filled with a warm neighbour, so a
    cold eco pixel there now trips the offset filter -- proving filter_clouds read the WORKING
    (filled) baseline out of ctx.channels, and that depends_on ordered fill_water first."""
    proj, g = _setup(tmp_path, {"fill_water": None, "filter_clouds": None})
    H, W = g.height, g.width
    mur = np.full((1, H, W), 285.0, "float32"); mur[0, :, 5] = np.nan
    eco = np.full((1, H, W), 286.0, "float32"); eco[0, :, 5] = 278.0
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      mur_sst=mur, landcover_water=np.ones((H, W), "float32"))
    ds = _pp(proj, g, cube)
    assert (ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values[:, 5] == 1).all()


# ---- filter_clouds: sigma method (baseline-floor climatology) ------------------------------ #
def _sigma_cube(g, times, cold_col=5, warm_col=6):
    """A cube whose per-cell MUR climatology has mean 287, sample std ~1.58 over 5 days; eco on
    day 0 has a cold outlier (280) in `cold_col` and an in-range value (285) in `warm_col`."""
    H, W = g.height, g.width
    T = len(times)
    mur = np.stack([np.full((H, W), 285.0 + t, "float32") for t in range(T)])
    eco = np.full((T, H, W), np.nan, "float32")
    eco[0] = 287.0
    eco[0, :, cold_col] = 280.0
    eco[0, :, warm_col] = 285.0
    return _hand_cube(g, times, eco_sst_v002=eco, eco_valid_v002=_ones_valid(T, H, W),
                      mur_sst=mur)


def test_filter_clouds_sigma_pixel_drops_below_floor(tmp_path):
    proj, g = _setup(tmp_path, {"filter_clouds": {"method": "sigma", "n_sigma": 3.0}})
    ds = _pp(proj, g, _sigma_cube(g, pd.date_range("2026-06-01", periods=5)))
    flag = ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values
    # cutoff = 287 - 3*1.58 = 282.3: 280 < cutoff (drop), 285 > cutoff (keep).
    assert (flag[:, 5] == 1).all()
    assert (flag[:, 6] == 0).all()
    assert flag[:, :5].sum() == 0 and flag[:, 7:].sum() == 0


def test_filter_clouds_sigma_pooled_scope(tmp_path):
    proj, g = _setup(tmp_path,
                     {"filter_clouds": {"method": "sigma", "n_sigma": 3.0, "stat_scope": "pooled"}})
    ds = _pp(proj, g, _sigma_cube(g, pd.date_range("2026-06-01", periods=5)))
    flag = ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values
    assert (flag[:, 5] == 1).all() and (flag[:, 6] == 0).all()


def test_filter_clouds_sigma_keeps_nan_baseline_pixels(tmp_path):
    proj, g = _setup(tmp_path, {"filter_clouds": {"method": "sigma"}})
    H, W = g.height, g.width
    times = pd.date_range("2026-06-01", periods=5)
    mur = np.stack([np.full((H, W), 285.0 + t, "float32") for t in range(5)])
    mur[:, :, 7] = np.nan                                    # no climatology in column 7
    eco = np.full((5, H, W), np.nan, "float32"); eco[0] = 287.0; eco[0, :, 7] = 280.0
    cube = _hand_cube(g, times, eco_sst_v002=eco, eco_valid_v002=_ones_valid(5, H, W), mur_sst=mur)
    ds = _pp(proj, g, cube)
    assert (ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values[:, 7] == 0).all()
    assert np.isfinite(ds["eco_sst_v002_clean"].isel(time=0).values[:, 7]).all()


def test_filter_clouds_sigma_harmonic_catches_seasonal_outlier(tmp_path):
    """A summer-peak cloud pixel (8 K below the seasonal mean) is a global-mean INLIER but a
    seasonal outlier: the day-of-year harmonic flags it; a constant climatology misses it."""
    proj_h, g = _setup(tmp_path, {"filter_clouds": {"method": "sigma", "seasonality": "harmonic"}})
    proj_o, _ = _setup(tmp_path, {"filter_clouds": {"method": "sigma", "seasonality": "off"}})
    H, W = g.height, g.width
    times = pd.date_range("2026-01-01", periods=365)
    doy = np.arange(1, 366)
    seasonal = 285.0 + 8.0 * np.sin(2 * np.pi * doy / 365.25) + 0.3 * np.sin(2 * np.pi * doy / 30)
    mur = np.broadcast_to(seasonal[:, None, None], (365, H, W)).astype("float32")
    day = 90                                                 # doy 91 ~ seasonal peak (~293)
    eco = np.full((365, H, W), np.nan, "float32")
    eco[day] = 295.0
    eco[day, :, 5] = 283.0                                   # summer cloud pixel
    cube = _hand_cube(g, times, eco_sst_v002=eco, eco_valid_v002=_ones_valid(365, H, W),
                      mur_sst=mur)

    fh = _pp(proj_h, g, cube)["eco_sst_v002_clean_cloudfiltered"].isel(time=day).values
    fo = _pp(proj_o, g, cube)["eco_sst_v002_clean_cloudfiltered"].isel(time=day).values
    assert (fh[:, 5] == 1).all() and (fh[:, 6] == 0).all()   # harmonic drops the seasonal outlier
    assert (fo[:, 5] == 0).all()                             # constant climatology misses it


def test_the_sigma_climatology_is_the_same_fit_however_it_is_blocked(tmp_path):
    """The climatology is a masked least-squares fit, and every term of its normal equations is
    a plain sum over time -- so accumulating block by block is the SAME arithmetic in a
    different order, not an approximation. This is the claim the whole change rests on."""
    for scope in ("pixel", "pooled"):
        proj, g = _setup(tmp_path / scope, {"filter_clouds": {
            "method": "sigma", "n_sigma": 3.0, "stat_scope": scope}})
        cube = _sigma_cube(g, pd.date_range("2026-06-01", periods=20))

        one = _pp(proj, g, cube)["eco_sst_v002_clean"].values
        many = _pp_blocked(proj, g, cube, 3)["eco_sst_v002_clean"].values

        np.testing.assert_array_equal(np.isnan(one), np.isnan(many))     # the DROPS must match
        np.testing.assert_allclose(one[~np.isnan(one)], many[~np.isnan(many)], rtol=1e-9)


def test_a_blocked_sigma_fit_still_catches_the_seasonal_outlier(tmp_path):
    """The 365-day harmonic case, blocked 30 days at a time.

    Two things could break it and both are silent: a per-block climatology would fit each
    month's own mean, and a per-block answer to "is the DOY span long enough for a harmonic?"
    is "no" for every block -- degrading to a constant climatology, which this fixture proves
    misses the outlier entirely.
    """
    proj, g = _setup(tmp_path, {"filter_clouds": {"method": "sigma", "seasonality": "harmonic"}})
    H, W = g.height, g.width
    times = pd.date_range("2026-01-01", periods=365)
    doy = np.arange(1, 366)
    seasonal = 285.0 + 8.0 * np.sin(2 * np.pi * doy / 365.25) + 0.3 * np.sin(2 * np.pi * doy / 30)
    mur = np.broadcast_to(seasonal[:, None, None], (365, H, W)).astype("float32")
    day = 90
    eco = np.full((365, H, W), np.nan, "float32")
    eco[day] = 295.0
    eco[day, :, 5] = 283.0                                   # summer cloud pixel
    cube = _hand_cube(g, times, eco_sst_v002=eco, eco_valid_v002=_ones_valid(365, H, W),
                      mur_sst=mur)

    flag = _pp_blocked(proj, g, cube, 30)["eco_sst_v002_clean_cloudfiltered"]
    flag = flag.isel(time=day).values
    assert (flag[:, 5] == 1).all(), "a blocked fit no longer sees the seasonal outlier"
    assert (flag[:, 6] == 0).all()


def test_filter_clouds_harmonic_short_span_falls_back_to_constant(tmp_path, caplog):
    proj, g = _setup(tmp_path, {"filter_clouds": {"method": "sigma", "seasonality": "harmonic"}})
    with caplog.at_level("INFO"):
        ds = _pp(proj, g, _sigma_cube(g, pd.date_range("2026-06-01", periods=5)))
    assert "too short" in caplog.text                        # harmonic degraded to constant
    assert (ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values[:, 5] == 1).all()


# ---- filter_cloud_cover: meteorological total cloud cover ---------------------------------- #
def test_filter_cloud_cover_scene_rejects_cloudy_overpass(tmp_path):
    proj, g = _setup(tmp_path, {"filter_cloud_cover": None})     # scene 30, pixel 80 (defaults)
    H, W = g.height, g.width
    tcc = np.full((2, H, W), 10.0, "float32"); tcc[0] = 40.0     # day 0 cloudy, day 1 clear
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=2),
                      eco_sst_v002=np.full((2, H, W), 286.0, "float32"),
                      eco_valid_v002=_ones_valid(2, H, W),
                      eco_cloud_cover_hrrr=tcc, landcover_water=np.ones((H, W), "float32"))
    ds = _pp(proj, g, cube)
    assert (ds["eco_valid_v002_clean"].isel(time=0).values == 0).all()
    assert np.isnan(ds["eco_sst_v002_clean"].isel(time=0).values).all()
    assert (ds["eco_sst_v002_clean_metcloudfiltered"].isel(time=0).values == 1).all()
    assert ds["eco_scene_cloud_pct_hrrr"].isel(time=0).item() == pytest.approx(40.0)
    # the clear overpass is untouched
    assert (ds["eco_valid_v002_clean"].isel(time=1).values == 1).all()
    assert (ds["eco_sst_v002_clean_metcloudfiltered"].isel(time=1).values == 0).all()


def test_filter_cloud_cover_pixel_gate_drops_only_cloudy_pixels(tmp_path):
    proj, g = _setup(tmp_path, {"filter_cloud_cover": None})
    H, W = g.height, g.width
    tcc = np.full((1, H, W), 10.0, "float32"); tcc[0, :, 5] = 90.0   # one very cloudy column
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=np.full((1, H, W), 286.0, "float32"),
                      eco_valid_v002=_ones_valid(1, H, W),
                      eco_cloud_cover_hrrr=tcc, landcover_water=np.ones((H, W), "float32"))
    flag = _pp(proj, g, cube)["eco_sst_v002_clean_metcloudfiltered"].isel(time=0).values
    assert (flag[:, 5] == 1).all() and flag.sum() == H          # only column 5 (scene mean ~12<30)


def test_filter_cloud_cover_disable_pixel_gate(tmp_path):
    proj, g = _setup(tmp_path, {"filter_cloud_cover": {"pixel_max_pct": None}})
    H, W = g.height, g.width
    tcc = np.full((1, H, W), 10.0, "float32"); tcc[0, :, 5] = 90.0
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=np.full((1, H, W), 286.0, "float32"),
                      eco_valid_v002=_ones_valid(1, H, W),
                      eco_cloud_cover_hrrr=tcc, landcover_water=np.ones((H, W), "float32"))
    flag = _pp(proj, g, cube)["eco_sst_v002_clean_metcloudfiltered"].isel(time=0).values
    assert flag.sum() == 0                                       # pixel gate off, scene mean <30


def test_filter_cloud_cover_works_with_era5(tmp_path):
    proj, g = _setup(tmp_path, {"filter_cloud_cover": {"source": "era5"}})
    H, W = g.height, g.width
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=np.full((1, H, W), 286.0, "float32"),
                      eco_valid_v002=_ones_valid(1, H, W),
                      eco_cloud_cover_era5=np.full((1, H, W), 40.0, "float32"),
                      landcover_water=np.ones((H, W), "float32"))
    ds = _pp(proj, g, cube)
    assert "eco_scene_cloud_pct_era5" in ds.data_vars
    assert (ds["eco_sst_v002_clean_metcloudfiltered"].isel(time=0).values == 1).all()


def test_filter_cloud_cover_no_met_channel_is_a_noop(tmp_path, caplog):
    proj, g = _setup(tmp_path, {"filter_cloud_cover": None})
    H, W = g.height, g.width
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=np.full((1, H, W), 286.0, "float32"),
                      eco_valid_v002=_ones_valid(1, H, W))
    with caplog.at_level("WARNING"):
        ds = _pp(proj, g, cube)
    assert "eco_sst_v002_clean_metcloudfiltered" not in ds.data_vars
    assert "filter_cloud_cover" in caplog.text


def test_cloud_filters_compose(tmp_path):
    """Both steps on: a baseline-cold pixel (col 3) and a met-cloudy pixel (col 8) are BOTH
    invalid in the final cube -- filter_cloud_cover read filter_clouds' working sst/valid."""
    proj, g = _setup(tmp_path, {"filter_clouds": None, "filter_cloud_cover": None})
    H, W = g.height, g.width
    eco = np.full((1, H, W), 286.0, "float32"); eco[0, :, 3] = 278.0
    tcc = np.full((1, H, W), 10.0, "float32"); tcc[0, :, 8] = 90.0
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      mur_sst=np.full((1, H, W), 285.0, "float32"),
                      eco_cloud_cover_hrrr=tcc, landcover_water=np.ones((H, W), "float32"))
    ds = _pp(proj, g, cube)
    valid = ds["eco_valid_v002_clean"].isel(time=0).values
    sst = ds["eco_sst_v002_clean"].isel(time=0).values
    assert (valid[:, 3] == 0).all() and (valid[:, 8] == 0).all()
    assert np.isnan(sst[:, 3]).all() and np.isnan(sst[:, 8]).all()
    # each flag marks only its own gate's pixels
    assert (ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values[:, 3] == 1).all()
    assert (ds["eco_sst_v002_clean_cloudfiltered"].isel(time=0).values[:, 8] == 0).all()
    assert (ds["eco_sst_v002_clean_metcloudfiltered"].isel(time=0).values[:, 8] == 1).all()
    assert (ds["eco_sst_v002_clean_metcloudfiltered"].isel(time=0).values[:, 3] == 0).all()


# ---- filter_land_clouds: over-land air-temperature deviation ------------------------------- #
def test_filter_land_clouds_drops_cold_land_pixels(tmp_path):
    """Over land (landcover), a pixel far COLDER than the air temp is dropped; a warm land pixel
    is kept. Condition: airtemp - sst > threshold_k (default 5 K)."""
    proj, g = _setup(tmp_path, {"filter_land_clouds": None})
    H, W = g.height, g.width
    eco = np.full((1, H, W), 300.0, "float32")
    eco[0, :, 5] = 287.0                                      # air(295) - 287 = 8 K > 5 -> cloud
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      eco_airtemp_hrrr=np.full((1, H, W), 295.0, "float32"),
                      landcover_water=np.zeros((H, W), "float32"))   # all LAND
    ds = _pp(proj, g, cube)
    flag = ds["eco_sst_v002_clean_landcloudfiltered"].isel(time=0).values
    assert (flag[:, 5] == 1).all() and flag.sum() == H       # only column 5 dropped
    assert np.isnan(ds["eco_sst_v002_clean"].isel(time=0).values[:, 5]).all()
    assert (ds["eco_valid_v002_clean"].isel(time=0).values[:, 5] == 0).all()
    assert np.isfinite(ds["eco_sst_v002_clean"].isel(time=0).values[:, 6]).all()
    assert ds["eco_sst_v002_clean_landcloudfiltered"].dtype == np.uint8


def test_filter_land_clouds_leaves_water_pixels_alone(tmp_path):
    """The SAME cold anomaly over WATER (landcover_water==1) is NOT dropped -- the land gate is
    exactly what separates this from the SST-baseline filter."""
    proj, g = _setup(tmp_path, {"filter_land_clouds": None})
    H, W = g.height, g.width
    eco = np.full((1, H, W), 300.0, "float32"); eco[0, :, 5] = 287.0
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      eco_airtemp_hrrr=np.full((1, H, W), 295.0, "float32"),
                      landcover_water=np.ones((H, W), "float32"))    # all WATER
    ds = _pp(proj, g, cube)
    assert (ds["eco_sst_v002_clean_landcloudfiltered"].isel(time=0).values == 0).all()
    assert np.isfinite(ds["eco_sst_v002_clean"].isel(time=0).values[:, 5]).all()


def test_filter_land_clouds_water_line_source(tmp_path):
    """land_source=water_line gates on `<pre>_water_class`==EXPOSED. Inject the class directly (it
    is what the water_line step emits) so a cold EXPOSED cell drops and a cold SUBMERGED one is
    kept -- the filter reads it through the WORKING channel."""
    proj, g = _setup(tmp_path, {"filter_land_clouds": {"land_source": "water_line"}})
    H, W = g.height, g.width
    eco = np.full((1, H, W), 300.0, "float32")
    eco[0, :, 5] = 287.0; eco[0, :, 6] = 287.0               # both cold
    wc = np.full((1, H, W), SUBMERGED, "uint8"); wc[0, :, 5] = EXPOSED    # col 5 land, col 6 water
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      eco_airtemp_hrrr=np.full((1, H, W), 295.0, "float32"),
                      eco_water_class=wc)
    ds = _pp(proj, g, cube)
    flag = ds["eco_sst_v002_clean_landcloudfiltered"].isel(time=0).values
    assert (flag[:, 5] == 1).all()                           # EXPOSED + cold -> dropped
    assert (flag[:, 6] == 0).all()                           # SUBMERGED + cold -> kept
    assert np.isfinite(ds["eco_sst_v002_clean"].isel(time=0).values[:, 6]).all()


def test_filter_land_clouds_threshold_zero_is_a_noop(tmp_path):
    proj, g = _setup(tmp_path, {"filter_land_clouds": {"threshold_k": 0}})
    H, W = g.height, g.width
    eco = np.full((1, H, W), 300.0, "float32"); eco[0, :, 5] = 287.0
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(1, H, W),
                      eco_airtemp_hrrr=np.full((1, H, W), 295.0, "float32"),
                      landcover_water=np.zeros((H, W), "float32"))
    ds = _pp(proj, g, cube)
    assert "eco_sst_v002_clean_landcloudfiltered" not in ds.data_vars


def test_filter_land_clouds_no_air_temp_is_a_noop(tmp_path, caplog):
    proj, g = _setup(tmp_path, {"filter_land_clouds": None})
    H, W = g.height, g.width
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=np.full((1, H, W), 300.0, "float32"),
                      eco_valid_v002=_ones_valid(1, H, W),
                      landcover_water=np.zeros((H, W), "float32"))
    with caplog.at_level("WARNING"):
        ds = _pp(proj, g, cube)
    assert "eco_sst_v002_clean_landcloudfiltered" not in ds.data_vars
    assert "filter_land_clouds" in caplog.text


# --------------------------------------------------------------------------- #
# Import-time invariants (mirror test_add_a_covariate's technique)
# --------------------------------------------------------------------------- #
def test_duplicate_step_key_is_rejected(monkeypatch):
    dup = preprocess.PreprocessStep("water_line", (), (), preprocess._step_water_line)
    monkeypatch.setattr(preprocess, "STEPS", preprocess.STEPS + (dup,))
    with pytest.raises(RuntimeError, match="unique"):
        preprocess._check_steps()


def test_unknown_dependency_is_rejected(monkeypatch):
    bad = preprocess.PreprocessStep("x", (), (), preprocess._step_fill_water,
                                    depends_on=("nope",))
    monkeypatch.setattr(preprocess, "STEPS", preprocess.STEPS + (bad,))
    with pytest.raises(RuntimeError, match="unknown step"):
        preprocess._check_steps()


def test_dependency_cycle_is_rejected(monkeypatch):
    a = preprocess.PreprocessStep("a", (), (), preprocess._step_fill_water, depends_on=("b",))
    b = preprocess.PreprocessStep("b", (), (), preprocess._step_fill_water, depends_on=("a",))
    monkeypatch.setattr(preprocess, "STEPS", (a, b))
    with pytest.raises(RuntimeError, match="cycle"):
        preprocess._check_steps()


def test_unknown_step_or_option_is_rejected_at_stage_time(tmp_path):
    eff = preprocess._build_eff(_project(tmp_path, steps={"watr_line": None}))
    with pytest.raises(ValueError, match="not a known step"):
        preprocess._check_step_options(eff)
    eff = preprocess._build_eff(
        _project(tmp_path, steps={"water_line": {"dem_sauce": "x"}}))
    with pytest.raises(ValueError, match="not a recognised option"):
        preprocess._check_step_options(eff)
    eff = preprocess._build_eff(
        _project(tmp_path, steps={"filter_clouds": {"n_sgima": 3}}))
    with pytest.raises(ValueError, match="not a recognised option"):
        preprocess._check_step_options(eff)
    eff = preprocess._build_eff(
        _project(tmp_path, steps={"filter_cloud_cover": {"scene_maxpct": 5}}))
    with pytest.raises(ValueError, match="not a recognised option"):
        preprocess._check_step_options(eff)


# --------------------------------------------------------------------------- #
# Through run(): the write path -- ONE cube, derived channels added, raw ones intact
# --------------------------------------------------------------------------- #
def _assembled(project, grids, days):
    """Assemble the full fixture and return (cube path, snapshot of it)."""
    _write_full_fixture(project, grids[AOI], days)
    datacube.assemble(project, grids=grids)                # writes datacube/aoi1.zarr
    zpath = project.output_dir / "datacube" / f"{AOI}.zarr"
    with xr.open_zarr(zpath) as ds:
        return zpath, _snapshot(ds)


# --------------------------------------------------------------------------- #
# Blocked rewrite
#
# A cube too large to hold at once is preprocessed a BLOCK of days at a time. The contract is
# that this changes nothing about the result, so the central test compares a blocked rewrite
# against a single-pass one, channel for channel.
# --------------------------------------------------------------------------- #
def _long_pp_project(tmp_path, n_days=10, **preprocess_cfg):
    """`_project`'s selection over a longer window -- 3 days cannot be split into blocks."""
    end = (pd.Timestamp("2026-06-01") + pd.Timedelta(days=n_days - 1)).date().isoformat()
    return parse_config({
        "name": "pp", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": end},
        "products": {"bathymetry": None, "tides": None, "mur": None, "cmems": None,
                     "landcover": None, "ecostress": None},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
        "auth": {"earthdata": {"auth_strategy": "netrc"},
                 "copernicus": {"auth_strategy": "netrc"}},
        "preprocess": {"enabled": True, **preprocess_cfg},
    })


def _assemble_and_preprocess(tmp_path, *, n_days, block_days, steps=None, chunks=None):
    """Assemble the full fixture, preprocess it at `block_days`, and snapshot the result."""
    p = _long_pp_project(tmp_path, n_days=n_days,
                         steps=steps or {"water_line": None, "fill_water": None,
                                         "filter_clouds": None})
    if chunks:
        p.datacube.chunks.update(chunks)
    grids = grid.project_grids(p)
    days = pd.date_range(p.time.start_date, p.time.end_date, freq="D")
    _write_full_fixture(p, grids[AOI], days)
    datacube.assemble(p, grids=grids)

    eff = preprocess._build_eff(p)
    eff["block_days"] = block_days
    preprocess.run(eff, grids, None, False)
    zpath = p.output_dir / "datacube" / f"{AOI}.zarr"
    with xr.open_zarr(zpath) as ds:
        return p, zpath, _snapshot(ds)


def test_blocked_and_unblocked_preprocessed_cubes_are_identical(tmp_path):
    """HOW the cube is rewritten must not change WHAT it holds.

    With the default step selection every step is day-local, so this is exact -- values, dtypes,
    dims, per-variable attrs, coverage, insitu_stations and the channel set, in one assertion.
    """
    _, _, one = _assemble_and_preprocess(tmp_path / "one", n_days=10, block_days=10)
    _, _, many = _assemble_and_preprocess(tmp_path / "many", n_days=10, block_days=3)

    problems = _diff_snapshots(one, many)
    assert problems == [], ("a blocked rewrite differs from a single-pass one:\n  "
                           + "\n  ".join(problems))


def test_a_blocked_rewrite_keeps_the_assembled_channels_untouched(tmp_path):
    """The failure that would matter most: the stage rewrites the cube it reads, so a blocked
    write that mangled a source channel would destroy the assembled data, not just its own."""
    p = _long_pp_project(tmp_path, n_days=8,
                         steps={"water_line": None, "fill_water": None})
    grids = grid.project_grids(p)
    days = pd.date_range(p.time.start_date, p.time.end_date, freq="D")
    _write_full_fixture(p, grids[AOI], days)
    datacube.assemble(p, grids=grids)
    zpath = p.output_dir / "datacube" / f"{AOI}.zarr"
    with xr.open_zarr(zpath) as ds:
        before = _snapshot(ds)

    eff = preprocess._build_eff(p)
    eff["block_days"] = 3
    preprocess.run(eff, grids, None, False)

    with xr.open_zarr(zpath) as ds:
        after = _snapshot(ds)
    assert set(before["data_vars"]) < set(after["data_vars"])
    for name, snap in before["data_vars"].items():
        assert after["data_vars"][name] == snap, f"{name} was modified by a blocked preprocess"
    for c in ("time", "y", "x"):
        assert after["coords"][c] == before["coords"][c]


def test_staleness_and_the_channel_ledger_survive_a_blocked_write(tmp_path):
    """Appending REPLACES a Zarr group's attrs, so the whole-cube ones are re-stamped at the end.

    Three runs, because two are not enough to catch the real failure: `preprocess_steps_stale`
    reads only `preprocess` and `code_version`, so a lost `preprocess_channels` still looks
    idempotent on run 2. It surfaces on run 3 with `overwrite`, where `was_derived` is empty and
    every derived channel the cube already holds looks like an assembled one being clobbered.
    """
    p, zpath, _ = _assemble_and_preprocess(tmp_path, n_days=8, block_days=3)
    grids = grid.project_grids(p)

    eff = preprocess._build_eff(p)
    eff["block_days"] = 3
    rep = preprocess.run(eff, grids, None, False)          # run 2: nothing to do
    assert rep.skipped == 1, "a blocked cube did not look finished to the next run"

    eff["overwrite"] = True
    rep = preprocess.run(eff, grids, None, False)          # run 3: must not raise
    assert rep.written == 1

    with xr.open_zarr(zpath) as ds:
        owned = json.loads(ds.attrs["preprocess_channels"])
        assert "mur_sst_gapfilled" in owned and "eco_water_elev" in owned
        for key in ("preprocess", "code_version", "provenance", "config_yaml",
                    "coverage", "insitu_stations", "preprocessed_at"):
            assert key in ds.attrs, f"{key} did not survive the blocked write"
        assert ds.attrs["created_at"] != ds.attrs["preprocessed_at"]


def test_a_blocked_rewrite_preserves_the_stores_time_chunk(tmp_path):
    """Preprocess must not re-chunk the store back to the configured value: the assembler may
    have reduced it deliberately, and undoing that silently is how the cube stops being
    writable at all."""
    _, zpath, _ = _assemble_and_preprocess(tmp_path, n_days=10, block_days=4,
                                           chunks={"time": 2})
    with xr.open_zarr(zpath) as ds:
        for name in ("mur_sst", "mur_sst_gapfilled"):
            assert ds[name].encoding["chunks"][0] == 2, (
                f"{name} was re-chunked to {ds[name].encoding['chunks'][0]}, not the store's 2")


def test_a_failure_part_way_through_a_blocked_rewrite_keeps_the_assembled_cube(
        tmp_path, monkeypatch):
    """The stage reads and writes the same path, so a half-finished write would take the
    assembled cube with it. `store.atomic` is what stops that -- pinned here."""
    p, zpath, before = _assemble_and_preprocess(tmp_path, n_days=8, block_days=2)
    grids = grid.project_grids(p)

    calls = {"n": 0}
    real = datacube.append_zarr

    def flaky(ds, path, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("the filesystem went away mid-cube")
        return real(ds, path, **kw)

    monkeypatch.setattr(datacube, "append_zarr", flaky)
    eff = preprocess._build_eff(p)
    eff["block_days"] = 2
    eff["overwrite"] = True
    with pytest.raises(ConnectionError):
        preprocess.run(eff, grids, None, False)

    with xr.open_zarr(zpath) as ds:
        assert _diff_snapshots(before, _snapshot(ds)) == []
    assert not list(zpath.parent.glob(f"{AOI}.zarr.part-*")), "scratch left behind"
    assert not list(zpath.parent.glob(f"{AOI}.zarr.old-*")), "stash left behind"


def test_preprocess_block_days_falls_back_to_the_datacube_value(tmp_path):
    p = _long_pp_project(tmp_path, n_days=4, steps={"fill_water": None})
    p.datacube.block_days = 7
    assert preprocess._build_eff(p)["block_days"] == 7      # unset -> inherit
    p.preprocess.block_days = 3
    assert preprocess._build_eff(p)["block_days"] == 3      # set -> wins


def test_run_adds_derived_channels_to_the_assembled_cube(project, grids, days):
    zpath, before = _assembled(project, grids, days)

    preprocess.preprocess(project, grids=grids)            # rewrites datacube/aoi1.zarr
    assert not (project.output_dir / "preprocessed").exists()   # no second store any more

    with xr.open_zarr(zpath) as ds:
        assert "eco_water_elev" in ds.data_vars            # derived channels landed
        assert "mur_sst_gapfilled" in ds.data_vars
        after = _snapshot(ds)
    # Every assembled channel survived the rewrite with its values, dtype and attrs intact --
    # the whole point of giving the derived channels names of their own.
    assert set(before["data_vars"]) < set(after["data_vars"])
    for name, snap in before["data_vars"].items():
        assert after["data_vars"][name] == snap, f"{name} was modified by preprocess"
    for c in ("time", "y", "x"):
        assert after["coords"][c] == before["coords"][c]


def test_run_is_idempotent(project, grids, days):
    """Re-running with the same steps rebuilds an IDENTICAL cube. The filters seed from the raw
    channels, so a second pass cannot union its drops onto the first pass's."""
    zpath, _ = _assembled(project, grids, days)
    preprocess.preprocess(project, grids=grids)
    with xr.open_zarr(zpath) as ds:
        once = _snapshot(ds)

    rep = preprocess.preprocess(project, grids=grids, overwrite=True)   # force the rebuild
    assert rep.written == 1
    with xr.open_zarr(zpath) as ds:
        twice = _snapshot(ds)
    assert _diff_snapshots(once, twice) == []


def test_rerun_does_not_nest_derived_suffixes(project, grids, days):
    """The cube a re-run reads already holds `_clean` / `_gapfilled` channels. The input scans
    skip them, so no `_clean_clean` or `_gapfilled_gapfilled` can appear."""
    zpath, _ = _assembled(project, grids, days)
    preprocess.preprocess(project, grids=grids)
    preprocess.preprocess(project, grids=grids, overwrite=True)
    with xr.open_zarr(zpath) as ds:
        names = list(ds.data_vars)
    assert not [n for n in names if "_clean_clean" in n or "_gapfilled_gapfilled" in n]


def test_dropping_a_step_drops_the_channels_it_owned(project, grids, days, tmp_path):
    """The cube must not keep shipping output its own `preprocess` attr no longer claims."""
    zpath, _ = _assembled(project, grids, days)
    preprocess.preprocess(project, grids=grids)
    with xr.open_zarr(zpath) as ds:
        assert "eco_water_elev" in ds.data_vars

    narrowed = _project(tmp_path, steps={"fill_water": None})       # water_line deselected
    preprocess.preprocess(narrowed, grids=grids)
    with xr.open_zarr(zpath) as ds:
        assert "eco_water_elev" not in ds.data_vars
        assert "mur_sst_gapfilled" in ds.data_vars


def test_run_skips_when_the_cube_is_missing(project, grids, caplog):
    with caplog.at_level("WARNING"):
        rep = preprocess.preprocess(project, grids=grids)   # no assemble first
    assert rep.written == 0 and rep.skipped == 1
    assert "run `assemble` first" in caplog.text
    assert not list((project.output_dir / "datacube").glob("*.zarr"))


def test_run_skips_a_cube_that_already_holds_this_step_selection(project, grids, days):
    """No `overwrite` needed for the common case: the stage recognises its own output and does
    not redo it, but a CHANGED step selection re-runs on its own."""
    _assembled(project, grids, days)
    assert preprocess.preprocess(project, grids=grids).written == 1
    rep = preprocess.preprocess(project, grids=grids)
    assert rep.skipped == 1 and rep.written == 0
    assert preprocess.preprocess(project, grids=grids, overwrite=True).written == 1


def test_a_changed_step_selection_re_runs_without_overwrite(project, grids, days, tmp_path):
    _assembled(project, grids, days)
    preprocess.preprocess(project, grids=grids)
    changed = _project(tmp_path, steps={"water_line": None, "fill_water": None,
                                        "filter_clouds": {"threshold_k": 2.0}})
    assert preprocess.preprocess(changed, grids=grids).written == 1


def test_a_step_may_not_overwrite_an_assembled_channel(project, grids, days, monkeypatch):
    """The guard that protects the raw values, exercised with a deliberately misbehaving step."""
    g = grids[AOI]
    _write_full_fixture(project, g, days)
    ds_raw = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    def _clobber(ctx):
        ctx.emit("mur_sst", ("time", "y", "x"), np.zeros((len(days), g.height, g.width), "f4"))

    bad = preprocess.PreprocessStep(key="fill_water", reads=(), writes=(), fn=_clobber)
    monkeypatch.setattr(preprocess, "STEPS", (bad,))
    with pytest.raises(RuntimeError, match="would overwrite assembled channel"):
        preprocess.preprocess_aoi(ds_raw, g, preprocess._build_eff(project))


# --------------------------------------------------------------------------- #
# GOLDEN derived cube (separate from datacube_golden.json)
# --------------------------------------------------------------------------- #
GOLDEN = Path(__file__).parent / "golden" / "preprocessed_golden.json"


def test_preprocessed_golden_is_unchanged(project, grids, days):
    """Assemble the full fixture, preprocess it, and assert the derived cube matches the
    committed golden channel for channel. Set UPDATE_GOLDEN=1 to regenerate after an
    INTENDED change, then review the golden's git diff."""
    g = grids[AOI]
    _write_full_fixture(project, g, days)
    eff = datacube._build_eff(project)
    # Pin the stacked sensor's source preference, as a real `platforms: [terra, aqua]` config
    # would: without it the merged overpass identity falls back to on-disk ALPHABETICAL order,
    # so `aqua` silently becomes what every matchup keys off. (Same note in test_datacube.)
    eff["sensor_version_pref"]["modis"] = ["terra", "aqua"]
    ds_raw = datacube.assemble_aoi(g, eff, days)
    ds = preprocess.preprocess_aoi(ds_raw, g, preprocess._build_eff(project))
    actual = _snapshot(ds)

    existed = GOLDEN.exists()
    if os.environ.get("UPDATE_GOLDEN") or not existed:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden {'regenerated' if existed else 'created'} at {GOLDEN}; "
                    "review its git diff and re-run")

    problems = _diff_snapshots(json.loads(GOLDEN.read_text()), actual)
    assert not problems, (
        "preprocessed cube drifted from the golden snapshot:\n  " + "\n  ".join(problems) +
        "\n\nIf this change is intended, regenerate with UPDATE_GOLDEN=1 and review the diff.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
