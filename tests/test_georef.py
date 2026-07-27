"""ECOSTRESS georeferencing diagnosis + correction (preprocess.georef).

Three layers, no network / no real data:

  * ALGORITHM unit tests on the ported pure functions -- the compass sign convention and
    whole-cell shift recovery (the prototype's decisive self-tests), plus gate behaviour.
  * INTEGRATION through `preprocess_aoi` on hand-built synthetic cubes -- flag_georef classifies
    a displaced scene vs a registered one, and correct_georef re-aligns the raw geometry (fit on
    the filtered SST, shift applied to the raw fields) while leaving the raw cube untouched.
  * REGION-OVERRIDE of a step option (min_coast_obs) resolved per-AoI, with the stage-time guard.

The synthetic coastline is a water DISK: its curved boundary pins the whole-cell translation
uniquely (a straight or L-shaped coast leaves a 1-cell along-shore slack at the 2-cell tolerance --
exactly the along-shore ambiguity the design warns about), so recovery is exact at a 1-cell tol."""

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from coastal_sst_data.config import parse_config, resolve_step_opts
from coastal_sst_data.processes import georef, preprocess

# Reuse the preprocess-test synthetic-cube helpers.
from tests.test_preprocess import _setup, _hand_cube, _pp, _ones_valid


# --------------------------------------------------------------------------- #
# Synthetic scene builders
# --------------------------------------------------------------------------- #
def _disk_water(H, W):
    """A centred water disk -> a curved coastline that pins the whole-cell translation uniquely."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = min(H, W) // 2 - 4
    return (yy - H // 2) ** 2 + (xx - W // 2) ** 2 < r ** 2


def _thermal(water, cold=280.0, warm=290.0, seed=0):
    """A thermal scene whose strong gradient sits on the land/water boundary."""
    rng = np.random.default_rng(seed)
    sst = np.where(water, cold, warm).astype("float32")
    return (sst + rng.normal(0, 0.01, sst.shape)).astype("float32")


def _flag_opts(**over):
    """Gates loosened for a tiny synthetic scene; a 1-cell tol makes recovery exact; search small."""
    base = dict(min_edges=10, min_coast_obs=5, min_valid_pct=0.0,
                tol_m=100, max_shift_m=1500, stability_windows_km=[0.5, 1.0])
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# ALGORITHM: compass sign convention (synthetic, unambiguous)
# --------------------------------------------------------------------------- #
def test_sign_convention_on_synthetic_coastlines():
    # E-W coast at row 50; a probe NORTH of it (smaller row) must point SOUTH (north < 0).
    ref = np.zeros((100, 100), bool); ref[50, :] = True
    _, (iy, ix) = ndimage.distance_transform_edt(~ref, return_distances=True, return_indices=True)
    east, north = georef.coast_offsets(np.array([40]), np.array([50]), iy, ix, 100.0)
    assert north[0] < 0 and abs(east[0]) < 1e-6

    # N-S coast at col 50; a probe WEST of it must point EAST (east > 0).
    ref = np.zeros((100, 100), bool); ref[:, 50] = True
    _, (iy, ix) = ndimage.distance_transform_edt(~ref, return_distances=True, return_indices=True)
    east, north = georef.coast_offsets(np.array([50]), np.array([40]), iy, ix, 100.0)
    assert east[0] > 0 and abs(north[0]) < 1e-6


# --------------------------------------------------------------------------- #
# ALGORITHM: whole-cell shift recovery, scored against the scene's own baseline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("roll", [(0, 0), (0, 3), (3, 0), (-2, 0), (0, -2), (3, -2), (-4, 3)])
def test_shift_recovery_relative_to_baseline(roll):
    H = W = 60
    water = _disk_water(H, W)
    dref = ndimage.distance_transform_edt(~georef.water_boundary(water))
    sst = _thermal(water)

    e0, _ = georef.canny(sst, 1.5, 80, 97)
    base = georef.search_shift(*np.nonzero(e0), dref, (H, W), 12, 1.0)

    ry, rx = roll
    e, _ = georef.canny(georef.shift_array(sst, ry, rx), 1.5, 80, 97)
    st = georef.search_shift(*np.nonzero(e), dref, (H, W), 12, 1.0)
    # Rolling the content by (ry,rx) must move the recovered shift by exactly -(ry,rx).
    assert (st["dy"], st["dx"]) == (base["dy"] - ry, base["dx"] - rx)


# --------------------------------------------------------------------------- #
# ALGORITHM: the quality gate is what rejects a low-coverage scene
# --------------------------------------------------------------------------- #
def test_quality_gate_rejects_then_admits_a_sparse_scene():
    H = W = 60
    water = _disk_water(H, W)
    ref = georef.water_boundary(water)
    sst = _thermal(water)
    # Observe only a thin strip -> almost no coastline seen.
    sparse = sst.copy(); sparse[10:, :] = np.nan
    edges, _ = georef.canny(sparse, 1.5, 80, 97)
    q = georef.scene_quality(sparse, ref, water)

    strict = dict(min_coast_obs=500, min_valid_frac=0.02, min_edges=300)
    assert georef.quality_reject(q, int(edges.sum()), strict) is not None      # gated OUT
    loose = dict(min_coast_obs=1, min_valid_frac=0.0, min_edges=1)
    assert georef.quality_reject(q, int(edges.sum()), loose) is None           # ablated -> admitted


# --------------------------------------------------------------------------- #
# INTEGRATION: flag_georef classifies displaced vs registered
# --------------------------------------------------------------------------- #
def _two_scene_cube(g, roll=(3, -2)):
    H, W = g.height, g.width
    water = _disk_water(H, W)
    ok = _thermal(water)
    disp = georef.shift_array(ok, *roll)                     # content moved by `roll`
    eco = np.stack([ok, disp]).astype("float32")
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=2),
                      eco_sst_v002=eco, eco_valid_v002=_ones_valid(2, H, W),
                      landcover_water=water.astype("uint8"))
    return cube, ok, disp


def test_flag_georef_flags_displaced_and_registered(tmp_path):
    proj, g = _setup(tmp_path, {"flag_georef": _flag_opts()})
    cube, _, _ = _two_scene_cube(g, roll=(3, -2))
    ds = _pp(proj, g, cube)

    assert ds["eco_georef_flag"].dtype == np.uint8
    assert ds["eco_georef_dy"].dtype == np.int16 and ds["eco_georef_dx"].dtype == np.int16
    assert ds["eco_georef_flag"].dims == ("time",)

    flag = ds["eco_georef_flag"].values
    assert flag[0] == georef.FLAG["ok"]
    assert flag[1] == georef.FLAG["displaced"]

    # Direction is measured against the registered scene's OWN baseline (a constant Canny offset
    # of a cell must not make a correct estimator look wrong): rolling by (+3,-2) moves it by -(+3,-2).
    base = (int(ds["eco_georef_dy"][0]), int(ds["eco_georef_dx"][0]))
    assert (int(ds["eco_georef_dy"][1]), int(ds["eco_georef_dx"][1])) == (base[0] - 3, base[1] + 2)


def test_flag_georef_degrades_on_degenerate_coastline(tmp_path):
    proj, g = _setup(tmp_path, {"flag_georef": _flag_opts()})
    H, W = g.height, g.width
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=_thermal(_disk_water(H, W))[None], eco_valid_v002=_ones_valid(1, H, W),
                      landcover_water=np.ones((H, W), "uint8"))   # all-water -> no coastline
    ds = _pp(proj, g, cube)
    assert not [v for v in ds.data_vars if "georef" in v]         # emitted nothing, did not raise


def test_flag_georef_skips_when_landcover_absent(tmp_path):
    proj, g = _setup(tmp_path, {"flag_georef": _flag_opts()})
    H, W = g.height, g.width
    cube = _hand_cube(g, pd.date_range("2026-06-01", periods=1),
                      eco_sst_v002=_thermal(_disk_water(H, W))[None],
                      eco_valid_v002=_ones_valid(1, H, W))         # no landcover_water at all
    ds = _pp(proj, g, cube)
    assert not [v for v in ds.data_vars if "georef" in v]


# --------------------------------------------------------------------------- #
# INTEGRATION: correct_georef shifts the RAW geometry, leaves the raw cube untouched
# --------------------------------------------------------------------------- #
def test_correct_georef_realigns_raw_and_copies_registered_verbatim(tmp_path):
    proj, g = _setup(tmp_path, {"flag_georef": _flag_opts(), "correct_georef": None})
    cube, ok, disp = _two_scene_cube(g, roll=(3, -2))
    disp_in = cube["eco_sst_v002"].values[1].copy()             # the raw displaced scene, pre-run
    dref = ndimage.distance_transform_edt(
        ~georef.water_boundary(cube["landcover_water"].values > 0.5))

    ds = _pp(proj, g, cube)

    assert "eco_sst_v002_georef_corrected" in ds.data_vars
    assert "eco_valid_v002_georef_corrected" in ds.data_vars
    assert ds["eco_sst_v002_georef_corrected"].dtype == np.float32
    assert ds["eco_valid_v002_georef_corrected"].dtype == np.uint8

    applied = ds["eco_georef_applied"].values
    assert applied[0] == 0 and applied[1] == 1                  # only the displaced scene shifted

    corr = ds["eco_sst_v002_georef_corrected"].values
    # The registered scene is copied byte-for-byte (its correction is a no-op).
    np.testing.assert_array_equal(np.nan_to_num(corr[0], nan=-9e9),
                                  np.nan_to_num(ok, nan=-9e9))

    # The correction moves the displaced scene's edge ONTO the coast: agreement rises sharply.
    e_before, _ = georef.canny(disp, 1.5, 80, 97)
    e_after, _ = georef.canny(corr[1], 1.5, 80, 97)
    agree_before = float((dref[e_before] < 2.0).mean())
    agree_after = float((dref[e_after] < 2.0).mean())
    assert agree_after > agree_before and agree_after > 0.8

    # The vacated margin is NaN-filled, never wrapped (np.roll would fabricate a coastline).
    assert np.isnan(corr[1]).sum() >= g.width                   # at least a shifted-out band
    # The corrected validity mask fills its margin with 0 (unobserved), not NaN.
    vc = ds["eco_valid_v002_georef_corrected"].values
    assert vc[1].min() == 0 and vc[1].max() == 1

    # The raw cube handed in was not mutated by either step.
    np.testing.assert_array_equal(cube["eco_sst_v002"].values[1], disp_in)


def test_correct_georef_without_flag_emits_nothing(tmp_path):
    # correct_georef depends_on flag_georef; selected alone it finds no (dy,dx) and degrades.
    proj, g = _setup(tmp_path, {"correct_georef": None})
    cube, _, _ = _two_scene_cube(g)
    ds = _pp(proj, g, cube)
    assert not [v for v in ds.data_vars if "georef" in v]


# --------------------------------------------------------------------------- #
# REGION-OVERRIDE of a preprocess step option
# --------------------------------------------------------------------------- #
def _two_region_project(tmp_path, **region_steps):
    return parse_config({
        "name": "pp", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {"landcover": None, "ecostress": None},
        "auth": {"earthdata": {"auth_strategy": "netrc"}},
        "preprocess": {"enabled": True, "steps": {"flag_georef": {"min_coast_obs": 500}}},
        "regions": [
            {"name": "sparse", "areas": [{"name": "a1", "center_lat": 45.5, "center_lon": -123.9,
                                          "buffer_ns_km": 2, "buffer_ew_km": 2}]},
            {"name": "dense", "preprocess_steps": region_steps,
             "areas": [{"name": "a2", "center_lat": 47.5, "center_lon": -122.5,
                        "buffer_ns_km": 2, "buffer_ew_km": 2}]},
        ],
    })


def test_region_override_resolves_per_aoi():
    proj = _two_region_project("/tmp/pp_region", flag_georef={"min_coast_obs": 1500})
    assert resolve_step_opts(proj, "a1", "flag_georef")["min_coast_obs"] == 500     # global
    assert resolve_step_opts(proj, "a2", "flag_georef")["min_coast_obs"] == 1500    # region wins


def test_region_override_rejects_non_overridable_option():
    proj = _two_region_project("/tmp/pp_region2", flag_georef={"sensors": ["lst"]})
    eff = preprocess._build_eff(proj)
    with pytest.raises(ValueError, match="not region-overridable"):
        preprocess._check_step_options(eff)


def test_region_override_rejects_unknown_option():
    proj = _two_region_project("/tmp/pp_region3", flag_georef={"min_coast_obz": 1500})
    eff = preprocess._build_eff(proj)
    with pytest.raises(ValueError, match="not a recognised option"):
        preprocess._check_step_options(eff)


# --------------------------------------------------------------------------- #
# RE-RUN the cloud filters on the corrected geometry (*_corrected step variants)
# --------------------------------------------------------------------------- #
# Water is 280 K (so it never trips the cold-deviation gate against a 280 K MUR baseline); land is
# 290 K. Anomalies are painted AFTER, at known cells, so a filter's drop is unambiguous.
_WATER_K, _LAND_K = 280.0, 290.0


def _cloud_cube(g, *, water_anom=None, land_anom=None, mur=280.0, airtemp=None):
    """A single well-registered scene (-> `ok`, corrected verbatim) with optional painted anomalies.
    `water_anom`/`land_anom` = (value, (row, col-slice)); `airtemp` adds an `airtemp_hrrr` channel."""
    H, W = g.height, g.width
    water = _disk_water(H, W)
    sst = np.where(water, _WATER_K, _LAND_K).astype("float32")
    if water_anom is not None:
        v, (r, cs) = water_anom; sst[r, cs] = v
    if land_anom is not None:
        v, (r, c) = land_anom; sst[r, c] = v
    ch = dict(eco_sst_v002=sst[None], eco_valid_v002=_ones_valid(1, H, W),
              landcover_water=water.astype("uint8"),
              mur_sst=np.full((1, H, W), mur, "float32"))
    if airtemp is not None:
        ch["airtemp_hrrr"] = np.full((1, H, W), airtemp, "float32")
    return _hand_cube(g, pd.date_range("2026-06-01", periods=1), **ch), water


def test_corrected_refilter_writes_separate_clean_channel(tmp_path):
    proj, g = _setup(tmp_path, {"flag_georef": _flag_opts(), "correct_georef": None,
                                "filter_clouds_corrected": {"method": "offset",
                                                            "baseline": "mur_sst",
                                                            "threshold_k": 5.0}})
    r, cs = g.height // 2, slice(g.width // 2 - 2, g.width // 2 + 2)
    cube, _ = _cloud_cube(g, water_anom=(272.0, (r, cs)))     # 8 K below MUR -> cloud
    ds = _pp(proj, g, cube)

    assert ds["eco_georef_applied"].values[0] == 0           # aligned scene -> corrected verbatim
    corr = ds["eco_sst_v002_georef_corrected"].isel(time=0).values
    clean = ds["eco_sst_v002_georef_corrected_clean"].isel(time=0).values
    flag = ds["eco_sst_v002_georef_corrected_clean_cloudfiltered"].isel(time=0).values

    # The corrected-but-unfiltered channel keeps the anomaly; the clean channel removes it.
    assert np.all(corr[r, cs] == 272.0)
    assert np.all(np.isnan(clean[r, cs])) and np.all(flag[r, cs] == 1)
    # Everywhere the filter did not drop, clean == corrected (it seeds from it).
    keep = flag == 0
    np.testing.assert_array_equal(np.nan_to_num(clean, nan=-9e9)[keep],
                                  np.nan_to_num(corr, nan=-9e9)[keep])
    assert ds["eco_sst_v002_georef_corrected_clean"].dtype == np.float32
    assert ds["eco_valid_v002_georef_corrected_clean"].dtype == np.uint8
    assert (ds["eco_valid_v002_georef_corrected_clean"].isel(time=0).values[r, cs] == 0).all()


def test_corrected_filter_inherits_base_config(tmp_path):
    # Base threshold 3 K; a 4 K anomaly. Inherited -> dropped; the default (5) would NOT drop it,
    # so a drop proves `filter_clouds_corrected: {}` picked up the base config.
    proj, g = _setup(tmp_path, {
        "filter_clouds": {"method": "offset", "baseline": "mur_sst", "threshold_k": 3.0},
        "flag_georef": _flag_opts(), "correct_georef": None, "filter_clouds_corrected": {}})
    r, cs = g.height // 2, slice(g.width // 2 - 2, g.width // 2 + 2)
    cube, _ = _cloud_cube(g, water_anom=(276.0, (r, cs)))     # 4 K below MUR
    ds = _pp(proj, g, cube)
    clean = ds["eco_sst_v002_georef_corrected_clean"].isel(time=0).values
    assert np.all(np.isnan(clean[r, cs]))                    # inherited threshold_k=3 dropped it


def test_corrected_filter_override_wins_over_base(tmp_path):
    # Same 4 K anomaly, but the corrected pass overrides threshold to 5 -> NOT dropped in clean.
    proj, g = _setup(tmp_path, {
        "filter_clouds": {"method": "offset", "baseline": "mur_sst", "threshold_k": 3.0},
        "flag_georef": _flag_opts(), "correct_georef": None,
        "filter_clouds_corrected": {"threshold_k": 5.0}})
    r, cs = g.height // 2, slice(g.width // 2 - 2, g.width // 2 + 2)
    cube, _ = _cloud_cube(g, water_anom=(276.0, (r, cs)))
    ds = _pp(proj, g, cube)
    clean = ds["eco_sst_v002_georef_corrected_clean"].isel(time=0).values
    assert np.all(np.isfinite(clean[r, cs]))                 # override 5 K > 4 K -> kept


def test_corrected_filters_compose_on_clean_channel(tmp_path):
    # A cold-water cloud (filter_clouds_corrected) AND a cold-land cloud (filter_land_clouds_corrected)
    # both fold into the SAME clean channel, each leaving its own audit flag.
    proj, g = _setup(tmp_path, {
        "flag_georef": _flag_opts(), "correct_georef": None,
        "filter_clouds_corrected": {"method": "offset", "baseline": "mur_sst", "threshold_k": 5.0},
        "filter_land_clouds_corrected": {"threshold_k": 5.0, "land_source": "landcover"}})
    r, cs = g.height // 2, slice(g.width // 2 - 2, g.width // 2 + 2)
    cube, water = _cloud_cube(g, water_anom=(272.0, (r, cs)),
                              land_anom=(283.0, (2, 2)), airtemp=290.0)   # land cell (2,2) cold
    assert not water[2, 2]                                    # (2,2) really is land
    ds = _pp(proj, g, cube)

    clean = ds["eco_sst_v002_georef_corrected_clean"].isel(time=0).values
    cloud_flag = ds["eco_sst_v002_georef_corrected_clean_cloudfiltered"].isel(time=0).values
    land_flag = ds["eco_sst_v002_georef_corrected_clean_landcloudfiltered"].isel(time=0).values
    assert np.all(cloud_flag[r, cs] == 1) and land_flag[2, 2] == 1     # each filter's own flag
    assert np.all(np.isnan(clean[r, cs])) and np.isnan(clean[2, 2])    # union folded into clean


def test_corrected_pipeline_end_to_end_write_path(tmp_path):
    # Through preprocess() (assemble + zarr write), assert both channels land with right dtypes.
    from coastal_sst_data import grid as _grid
    from coastal_sst_data.processes import datacube
    from tests.test_preprocess import _project, _write_full_fixture, AOI
    proj = _project(tmp_path, steps={
        "fill_water": None, "filter_clouds": None,
        "flag_georef": {"min_edges": 10, "min_coast_obs": 5, "min_valid_pct": 0.0,
                        "tol_m": 100, "max_shift_m": 1500, "stability_windows_km": [0.5, 1.0]},
        "correct_georef": None, "filter_clouds_corrected": {}})
    grids = _grid.project_grids(proj); g = grids[AOI]
    days = pd.date_range(proj.time.start_date, proj.time.end_date, freq="D")
    _write_full_fixture(proj, g, days)
    datacube.assemble(proj, grids=grids)
    rep = preprocess.preprocess(proj, grids=grids)
    assert rep.written == 1

    import xarray as xr
    ds = xr.open_zarr(proj.output_dir / "preprocessed" / f"{AOI}.zarr")
    corrected = [v for v in ds.data_vars if v.endswith("_georef_corrected")]
    clean = [v for v in ds.data_vars if v.endswith("_georef_corrected_clean")]
    assert corrected and clean                               # both stages materialised
    assert ds["eco_sst_v002_georef_corrected_clean"].dtype == np.float32
