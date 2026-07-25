"""Post-assembly preprocessing stage. Assembles a tiny SYNTHETIC raw cube (reusing the
datacube test writers), then runs the preprocess steps and asserts the derived cube: the
waterline is pure glue over water_level.water_level_fields, the level-4 NN fill respects the
land-cover water mask and flags invented cells, the raw cube is left untouched, and the
import-time step invariants fire. No network, no real data.

The golden test (`test_preprocessed_golden_is_unchanged`) is the derived cube's own safety
net, kept SEPARATE from datacube_golden.json so the two stages don't couple."""

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
    return _project(tmp_path, steps={"water_line": None, "fill_water": None})


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

    mur0 = ds_out["mur_sst"].isel(time=0).values
    flag0 = ds_out["mur_sst_filled"].isel(time=0).values
    # Water holes were filled from the nearest finite pixel; the land hole stayed NaN.
    assert np.isfinite(mur0[:, 5:8]).all()
    assert np.isnan(mur0[:, 0]).all()
    # The flag marks exactly the invented-over-water cells (1), observed cells 0.
    assert (flag0[:, 5:8] == 1).all()
    assert (flag0[:, 0] == 0).all()          # a land NaN is not "filled"
    assert (flag0[:, 4] == 0).all()          # an observed water cell is not "filled"
    assert ds_out["mur_sst_filled"].dtype == np.uint8
    # The mask it filled against travels along.
    assert "landcover_water" in ds_out.data_vars


def test_fill_water_also_fills_cmems_per_source(project, grids, days):
    g = grids[AOI]
    write_landcover(project, g, land_cols=slice(0, 0))          # all water
    write_cmems(project, g, days, land_cols=[6, 7], src="my_global")
    ds_raw = _raw_cube(project, g, days)
    ds_out = preprocess.preprocess_aoi(ds_raw, g, preprocess._build_eff(project))
    c = "cmems_thetao_0m_my_global"
    assert np.isfinite(ds_out[c].isel(time=0).values[:, 6:8]).all()
    assert (ds_out[f"{c}_filled"].isel(time=0).values[:, 6:8] == 1).all()


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
    assert "mur_sst_filled" not in ds_out.data_vars              # step declined
    assert "fill_water" in caplog.text


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


# --------------------------------------------------------------------------- #
# Through run(): the write path, and the raw cube left untouched
# --------------------------------------------------------------------------- #
def test_run_writes_a_separate_cube_and_leaves_the_raw_one_untouched(project, grids, days):
    g = grids[AOI]
    _write_full_fixture(project, g, days)
    datacube.assemble(project, grids=grids)                # writes datacube/aoi1.zarr
    raw_zpath = project.output_dir / "datacube" / f"{AOI}.zarr"
    before = _snapshot(xr.open_zarr(raw_zpath))

    preprocess.preprocess(project, grids=grids)            # writes preprocessed/aoi1.zarr
    derived = project.output_dir / "preprocessed" / f"{AOI}.zarr"
    assert derived.exists()

    ds = xr.open_zarr(derived)
    assert "eco_water_elev" in ds.data_vars
    assert "mur_sst_filled" in ds.data_vars
    # Coords are the raw cube's own -> the two align cell-for-cell.
    raw = xr.open_zarr(raw_zpath)
    for c in ("time", "y", "x"):
        np.testing.assert_array_equal(ds[c].values, raw[c].values)
    # The raw cube is byte-for-value unchanged.
    assert _diff_snapshots(before, _snapshot(raw)) == []


def test_run_skips_when_the_raw_cube_is_missing(project, grids, caplog):
    with caplog.at_level("WARNING"):
        rep = preprocess.preprocess(project, grids=grids)   # no assemble first
    assert rep.written == 0 and rep.skipped == 1
    assert "run `assemble` first" in caplog.text
    assert not list((project.output_dir / "preprocessed").glob("*.zarr"))


def test_run_overwrite_semantics(project, grids, days):
    g = grids[AOI]
    _write_full_fixture(project, g, days)
    datacube.assemble(project, grids=grids)
    preprocess.preprocess(project, grids=grids)
    derived = project.output_dir / "preprocessed" / f"{AOI}.zarr"
    mtime = derived.stat().st_mtime

    rep = preprocess.preprocess(project, grids=grids)       # exists, no overwrite -> skip
    assert rep.skipped == 1 and rep.written == 0
    rep = preprocess.preprocess(project, grids=grids, overwrite=True)
    assert rep.written == 1


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
    ds_raw = datacube.assemble_aoi(g, datacube._build_eff(project), days)
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
