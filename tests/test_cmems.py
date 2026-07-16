"""CMEMS global-physics acquisition. The network seam is `cmems.open_window` (the one
function that calls copernicusmarine), monkeypatched with a synthetic GLORYS-shaped
Dataset -- so these run offline. The emphasis is on depth snapping (a value must be one
the model actually computed, at the level we claim) and the my->anfc chain (a reanalysis
day and a forecast day must never be silently conflated)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import cmems, datacube


AOI = "aoi1"
# GLORYS12's real upper levels -- deliberately NOT round numbers, which is the point.
LEVELS = [0.494, 1.541, 2.646, 3.819, 5.078, 6.441, 7.930, 9.573, 11.405, 13.467,
          15.810, 18.496, 21.599, 25.211, 29.445]


def _project(tmp_path, **met):
    opts = {"variables": ["thetao"], "depths": [0, 10, 30]}
    opts.update(met)
    return parse_config({
        "name": "c", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {"cmems": opts},
        "auth": {"copernicus": {"auth_strategy": "netrc"}},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.52, "center_lon": -123.925,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


@pytest.fixture
def project(tmp_path):
    return _project(tmp_path)


@pytest.fixture
def g(project):
    return grid.project_grids(project)[AOI]


@pytest.fixture
def days(project):
    return pd.date_range(project.time.start_date, project.time.end_date, freq="D")


def fake_source(days, g, *, temp_at_depth=None, variables=("thetao",)):
    """A GLORYS-shaped lazy dataset: (time, depth, latitude, longitude)."""
    w, s, e, n = g.search_bbox
    lons = np.linspace(w - 0.1, e + 0.1, 6)
    lats = np.linspace(s - 0.1, n + 0.1, 6)
    t = pd.DatetimeIndex(days)
    dv = {}
    for var in variables:
        if var in cmems.SURFACE_ONLY:
            dv[var] = (("time", "latitude", "longitude"),
                       np.full((len(t), len(lats), len(lons)), 0.5, "float32"))
            continue
        arr = np.empty((len(t), len(LEVELS), len(lats), len(lons)), "float32")
        for k, lvl in enumerate(LEVELS):
            arr[:, k] = temp_at_depth(lvl) if temp_at_depth else 10.0
        dv[var] = (("time", "depth", "latitude", "longitude"), arr)
    return xr.Dataset(dv, coords={"time": t, "depth": LEVELS,
                                  "latitude": lats, "longitude": lons})


# --------------------------------------------------------------------------- #
# Config -> acquisition params
# --------------------------------------------------------------------------- #
# `eff["ds"]` is keyed by AoI: which CMEMS model covers an AoI is region-dependent, so the
# chain is resolved per AoI (region override -> project default) rather than once per run.
def test_default_chain_is_reanalysis_then_forecast(project):
    eff = cmems._build_eff(project)
    assert eff["ds"][AOI]["chain"] == ["my", "anfc"]
    assert cmems.DATASET_IDS["my"] == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    assert eff["ds"][AOI]["depths"] == [0.0, 10.0, 30.0]
    assert eff["ds"][AOI]["variables"] == ["thetao"]


def test_fallback_can_be_disabled(tmp_path):
    eff = cmems._build_eff(_project(tmp_path, fallback="none"))
    assert eff["ds"][AOI]["chain"] == ["my"]


def test_an_unknown_source_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="not recognized"):
        cmems._build_eff(_project(tmp_path, source="glorys4"))


def test_netrc_strategy_points_the_toolbox_at_netrc(project):
    # The secret stays in ~/.netrc, exactly like Earthdata -- never in the config.
    assert cmems._build_eff(project)["creds"]["credentials_file"].endswith(".netrc")


def test_environment_strategy_passes_no_credentials_file(tmp_path):
    p = _project(tmp_path)
    cfg = p.model_dump(mode="json")
    cfg["auth"]["copernicus"]["auth_strategy"] = "environment"
    assert cmems._build_eff(parse_config(cfg))["creds"] == {}


def test_cmems_requires_its_auth_block(tmp_path):
    with pytest.raises(Exception, match="auth.copernicus"):
        parse_config({
            "name": "c", "output_dir": str(tmp_path),
            "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
            "products": {"cmems": None},
            "regions": [{"name": "r", "areas": [
                {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
                 "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
        })


# --------------------------------------------------------------------------- #
# Depth snapping: the value must be one the model actually computed
# --------------------------------------------------------------------------- #
def test_depths_snap_to_the_nearest_model_level(days, g):
    src = fake_source(days, g)
    got = cmems.snap_depths(src, [0, 10, 30])
    assert got == {0.0: 0.494, 10.0: 9.573, 30.0: 29.445}   # never interpolated


def test_a_far_off_depth_warns(days, g, caplog):
    with caplog.at_level("WARNING"):
        got = cmems.snap_depths(fake_source(days, g), [100])
    assert got == {100.0: 29.445}                            # deepest level available
    assert "snaps to model level" in caplog.text


def test_the_level_actually_used_is_recorded(days, g):
    src = fake_source(days, g, temp_at_depth=lambda lvl: 15.0 - 0.2 * lvl)
    ds = cmems.day_dataset(src, days[0], g, ["thetao"],
                           cmems.snap_depths(src, [0, 10]), {}, False)
    # The channel is named for what was ASKED; the attr says what was USED.
    assert ds["thetao_10m"].attrs["requested_depth_m"] == 10.0
    assert ds["thetao_10m"].attrs["model_depth_m"] == pytest.approx(9.573)
    # ...and the value is the one at that level, not an interpolation to 10 m.
    assert float(np.nanmean(ds["thetao_10m"].values)) == pytest.approx(15.0 - 0.2 * 9.573,
                                                                       abs=1e-3)


def test_each_depth_becomes_its_own_channel(days, g):
    src = fake_source(days, g, temp_at_depth=lambda lvl: 15.0 - 0.2 * lvl)
    ds = cmems.day_dataset(src, days[0], g, ["thetao"],
                           cmems.snap_depths(src, [0, 10, 30]), {}, False)
    assert {"thetao_0m", "thetao_10m", "thetao_30m", "valid"} == set(ds.data_vars)
    # a real thermocline: deeper is colder
    surf = float(np.nanmean(ds["thetao_0m"].values))
    deep = float(np.nanmean(ds["thetao_30m"].values))
    assert surf > deep


def test_surface_only_variables_get_no_depth_suffix(days, g):
    src = fake_source(days, g, variables=("thetao", "zos"))
    ds = cmems.day_dataset(src, days[0], g, ["thetao", "zos"],
                           cmems.snap_depths(src, [0]), {}, False)
    assert "zos" in ds.data_vars                     # 2D: written once, unsuffixed
    assert "zos_0m" not in ds.data_vars
    assert "thetao_0m" in ds.data_vars


def test_a_day_the_product_does_not_cover_returns_none(days, g):
    src = fake_source(days[:1], g)                  # only the first day exists
    assert cmems.day_dataset(src, days[2], g, ["thetao"],
                             cmems.snap_depths(src, [0]), {}, False) is None


# --------------------------------------------------------------------------- #
# The my -> anfc chain
# --------------------------------------------------------------------------- #
def test_forecast_backfills_the_days_the_reanalysis_does_not_reach(monkeypatch, project,
                                                                   g, days):
    """The reanalysis ends partway through the window; the forecast covers the rest, and
    each day's file records WHICH product produced it."""
    calls = []

    def fake_open(dataset_id, variables, bbox, pad, start, end, depths, creds):
        calls.append(dataset_id)
        if dataset_id == cmems.DATASET_IDS["my"]:
            return fake_source(days[:2], g)          # reanalysis: days 0-1 only
        return fake_source(days, g)                  # forecast: everything

    monkeypatch.setattr(cmems, "open_window", fake_open)
    cmems.acquire(project, grids={AOI: g})

    out = project.output_dir / "CMEMS" / "aligned" / AOI
    srcs = {}
    for d in days:
        with xr.open_dataset(out / f"{AOI}_{d.strftime('%Y%m%d')}.nc") as ds:
            srcs[d.strftime("%Y%m%d")] = ds.attrs["cmems_source"]
    assert srcs == {"20260601": "my", "20260602": "my", "20260603": "anfc"}
    assert calls == [cmems.DATASET_IDS["my"], cmems.DATASET_IDS["anfc"]]


def test_already_acquired_days_are_not_refetched(monkeypatch, project, g, days):
    n = []

    def fake_open(dataset_id, *a, **kw):
        n.append(dataset_id)
        return fake_source(days, g)

    monkeypatch.setattr(cmems, "open_window", fake_open)
    cmems.acquire(project, grids={AOI: g})
    before = len(n)
    cmems.acquire(project, grids={AOI: g})           # second run: all days on disk
    assert len(n) == before                          # ...no further opens


def test_dry_run_opens_nothing(monkeypatch, project, g):
    monkeypatch.setattr(cmems, "open_window",
                        lambda *a, **kw: pytest.fail("dry-run must not open a dataset"))
    cmems.acquire(project, grids={AOI: g}, dry_run=True)
    assert not (project.output_dir / "CMEMS").exists()


# --------------------------------------------------------------------------- #
# Into the cube: channels + the nearest-neighbour water fill
# --------------------------------------------------------------------------- #
def _write_cmems(project, g, days, *, hole=None):
    """Aligned CMEMS files, optionally with a NaN hole (the model's coarse land mask)."""
    xs, ys = g.xy_centers()
    H, W = g.height, g.width
    for day in days:
        arr = np.full((H, W), 12.0, "float32")
        if hole is not None:
            arr[:, hole] = np.nan
        ds = xr.Dataset({"thetao_0m": (("time", "y", "x"), arr[None]),
                         "valid": (("time", "y", "x"), np.isfinite(arr[None]).astype("uint8"))},
                        coords={"time": [day], "y": ys, "x": xs})
        d = project.output_dir / "CMEMS" / "aligned" / AOI
        d.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(d / f"{AOI}_{day.strftime('%Y%m%d')}.nc")


def _write_landcover(project, g, *, land_cols):
    xs, ys = g.xy_centers()
    water = np.ones((g.height, g.width), "float32")
    water[:, land_cols] = 0.0
    ds = xr.Dataset({"water": (("y", "x"), water)}, coords={"y": ys, "x": xs})
    d = project.output_dir / "LANDCOVER" / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}.nc")


def test_cmems_channels_reach_the_cube(project, g, days):
    _write_cmems(project, g, days)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert "cmems_thetao_0m" in ds.data_vars
    assert float(np.nanmean(ds["cmems_thetao_0m"].isel(time=0).values)) == pytest.approx(12.0)


def test_missing_water_pixels_ship_as_honest_nan_gaps(project, g, days):
    """S1: CMEMS is no longer NN-filled. The 9 km model's land holes stay NaN in the cube
    and carry no `_filled` mask -- filling is a downstream determination now (Goal 3)."""
    _write_cmems(project, g, days, hole=slice(0, 7))
    _write_landcover(project, g, land_cols=slice(0, 5))
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    arr = ds["cmems_thetao_0m"].isel(time=0).values
    assert np.isnan(arr[:, :7]).all()                       # the model hole is untouched
    assert np.isfinite(arr[:, 7:]).all()                    # resolved cells survive
    assert "cmems_thetao_0m_filled" not in ds.data_vars

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])