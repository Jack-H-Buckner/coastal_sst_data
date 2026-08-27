"""Moving in-situ platforms: gliders, ship transects, drifters.

A SEPARATE PRODUCT from `insitu`, writing a flat `(obs,)` observation table instead of the
fixed product's `(station, time)` rectangle, and merged into the same `insitu_*` cube channels.
The headline claim these tests defend is that the fixed path is untouched: `insitu_station`
stays a static `(y,x)` map naming only fixed stations, and a fixed-only project's cube is
exactly what it was.

No network -- the CSV source reads local files, which is all these need.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data import grid, products
from coastal_sst_data.config import DataProduct, parse_config
from coastal_sst_data.processes import (datacube, insitu, insitu_acquire, insitu_cmems,
                                        insitu_mobile)

AOI = "aoi1"
LAT, LON = 45.52, -123.925


def _project(tmp_path, mobile=None, fixed=None):
    products_cfg = {}
    if fixed is not None:
        products_cfg["insitu"] = fixed
    if mobile is not None:
        products_cfg["insitu_mobile"] = mobile
    return parse_config({
        "name": "m", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": products_cfg,
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": LAT, "center_lon": LON,
             "buffer_ns_km": 5, "buffer_ew_km": 5}]}],
    })


@pytest.fixture
def g(tmp_path):
    return grid.project_grids(_project(tmp_path, mobile={"sources": ["csv"]}))[AOI]


@pytest.fixture
def days(tmp_path):
    p = _project(tmp_path, mobile={"sources": ["csv"]})
    return pd.date_range(p.time.start_date, p.time.end_date, freq="D")


def _track_csv(path, *, day="2026-06-01", hour=19, n=12, dlat=0.02, temp=12.0):
    """A transect: `n` observations marching north-east across the AoI within one hour."""
    t0 = pd.Timestamp(f"{day}T{hour:02d}:00")
    rows = pd.DataFrame({
        "station_id": "glider1",
        "time": [t0 + pd.Timedelta(minutes=2 * i) for i in range(n)],
        "latitude": LAT - dlat + np.linspace(0, 2 * dlat, n),
        "longitude": LON - dlat + np.linspace(0, 2 * dlat, n),
        "value": np.full(n, temp),
    })
    path.write_text(rows.to_csv(index=False))
    return rows


def _write_tracks(project, g, records, source="csv"):
    """Acquire straight through the module's own writer, so the on-disk schema is the real one."""
    ds = insitu_mobile.build_track_dataset(records)
    d = insitu_mobile.aligned_dir(project.output_dir, source, AOI)
    insitu_mobile.write_output(ds, d, AOI)
    return ds


def _rec(rows, pid="glider1", ptype="GL"):
    """A fetch record. `ptype=None` is the `csv`/`ioos` case: no declared platform class, so
    measured drift is the only evidence available."""
    return {"id": pid, "title": pid, "var": "TEMP", "platform_type": ptype,
            "df": rows.rename(columns={"value": "value"})[
                ["time", "latitude", "longitude", "value"]],
            "lat": float(rows["latitude"].median()),
            "lon": float(rows["longitude"].median())}


# --------------------------------------------------------------------------- #
# The two products partition the platforms
# --------------------------------------------------------------------------- #
def test_the_products_partition_platforms_with_no_gap_or_overlap(tmp_path, g):
    """`split_fixed_platforms` is the exact complement of `split_moving_platforms`, sharing a
    threshold -- so every platform lands in exactly one product. An overlap would let the same
    instrument reach `insitu_sst` twice, once per tree."""
    # Neither declares a platform class -- the `csv`/`ioos` case, where measured drift is the
    # only evidence there is.
    moving = _rec(_track_csv(tmp_path / "t.csv"), ptype=None)
    still = _rec(pd.DataFrame({
        "time": pd.date_range("2026-06-01T19:00", periods=6, freq="10min"),
        "latitude": LAT, "longitude": LON, "value": 12.0}), pid="buoy1", ptype=None)
    recs = [moving, still]
    thr = g.resolution_m

    keep_fixed, dropped_moving = insitu_acquire.split_moving_platforms(recs, thr)
    keep_mobile, dropped_fixed = insitu_mobile.split_fixed_platforms(recs, thr)

    assert [r["id"] for r in keep_fixed] == ["buoy1"]
    assert [r["id"] for r, _ in dropped_moving] == ["glider1"]
    assert [r["id"] for r in keep_mobile] == ["glider1"]
    assert [r["id"] for r, _ in dropped_fixed] == ["buoy1"]


def test_a_declared_mobile_class_beats_a_drift_of_zero(tmp_path, g):
    """A track that clips the AoI may leave ONE observation inside it, and one position has no
    drift by construction. Classified on drift alone it went to the fixed product -- which for
    `marineinsitu` never fetches mobile classes at all, so the observation fell between the two
    and vanished. Seen live: two of five platforms over Hobart, a drifter and a ship."""
    one = pd.DataFrame({"time": [pd.Timestamp("2026-06-01T19:00")],
                        "latitude": [LAT], "longitude": [LON], "value": [12.0]})
    drifter = _rec(one, pid="drifter1", ptype="DB")
    assert insitu_acquire.platform_drift_m(one) == 0.0      # the trap
    mobile, fixed = insitu_mobile.split_fixed_platforms([drifter], g.resolution_m)
    assert [r["id"] for r in mobile] == ["drifter1"] and fixed == []

    # ...while a DECLARED fixed class with no drift still belongs to the other product.
    mooring = _rec(one, pid="mooring1", ptype="MO")
    mobile, fixed = insitu_mobile.split_fixed_platforms([mooring], g.resolution_m)
    assert mobile == [] and [r["id"] for r, _ in fixed] == ["mooring1"]


def test_it_is_not_on_by_default(tmp_path):
    """Selecting the product is the opt-in; no source here is the obvious one to want."""
    assert products.spec(DataProduct.insitu_mobile).default_sources == ()
    eff = insitu_mobile._build_eff(_project(tmp_path, mobile={}))
    assert eff["ds"][AOI]["sources"] == []


# --------------------------------------------------------------------------- #
# The flat observation schema
# --------------------------------------------------------------------------- #
def test_the_table_is_flat_and_carries_a_position_per_observation(tmp_path):
    rows = _track_csv(tmp_path / "t.csv", n=8)
    ds = insitu_mobile.build_track_dataset([_rec(rows)])
    assert ds["sst"].dims == ("obs",) and ds.sizes["obs"] == 8
    # THE point of the product: every observation has its own place, not one per platform.
    assert ds["lat"].dims == ("obs",) and len(np.unique(ds["lat"].values)) == 8
    assert list(ds["platform_id"].values) == ["glider1"] * 8


def test_platforms_with_unrelated_time_axes_do_not_multiply(tmp_path):
    """Flat, not (station, time). A union time axis over platforms that share no timestamps
    would be the SUM of their lengths with a block that is quadratic and almost all NaN."""
    a = _rec(_track_csv(tmp_path / "a.csv", n=10, hour=19), pid="a")
    b = _rec(_track_csv(tmp_path / "b.csv", n=10, hour=21), pid="b")
    ds = insitu_mobile.build_track_dataset([a, b])
    assert ds.sizes["obs"] == 20                       # not 20 x 20


def test_the_table_round_trips_through_disk(tmp_path):
    ds = insitu_mobile.build_track_dataset([_rec(_track_csv(tmp_path / "t.csv", n=5))])
    out = insitu_mobile.write_output(ds, tmp_path / "o", AOI)
    from coastal_sst_data import store
    with store.open_netcdf(out) as reopened:
        tab = insitu.TrackTable.from_dataset(reopened)
    assert tab.n_obs == 5
    assert np.isfinite(tab.sst).all() and len(np.unique(tab.lat)) == 5


# --------------------------------------------------------------------------- #
# Placement -- observation_pixels and the index trap
# --------------------------------------------------------------------------- #
def test_observation_pixels_places_each_point_separately(g):
    lats = [LAT - 0.02, LAT, LAT + 0.02]
    lons = [LON, LON, LON]
    rows, cols, inside = insitu.observation_pixels(lons, lats, g)
    assert inside.all()
    assert len(set(rows.tolist())) == 3, "three latitudes collapsed into one row"


def test_a_position_outside_the_grid_is_flagged_not_indexed(g):
    rows, cols, inside = insitu.observation_pixels([LON, LON + 40.0], [LAT, LAT], g)
    assert inside.tolist() == [True, False]
    assert 0 <= rows[1] < g.height and 0 <= cols[1] < g.width, (
        "an out-of-grid point must still carry a SAFE index; callers mask on `inside`")


def test_a_nonfinite_position_is_not_a_pixel(g):
    _, _, inside = insitu.observation_pixels([LON, np.nan], [LAT, LAT], g)
    assert inside.tolist() == [True, False]


def test_nearest_index_returns_an_index_into_the_original_axis():
    """THE trap. `value_at` compacted by `finite` first, so its internal index pointed into the
    filtered array. A track uses that index to look up the observation's POSITION -- off-by-N
    there places observations at the wrong coordinates, plausibly and silently."""
    t = pd.DatetimeIndex(["2026-06-01T00:00", "2026-06-01T01:00",
                          "2026-06-01T02:00", "2026-06-01T03:00"])
    v = np.array([np.nan, np.nan, 12.0, 13.0])         # two NaNs BEFORE the winner
    k, dt = insitu.nearest_index(t, v, pd.Timestamp("2026-06-01T02:05"), 60)
    assert k == 2 and dt == pytest.approx(-5.0)
    assert insitu.value_at(t, v, pd.Timestamp("2026-06-01T02:05"), 60) == (12.0, dt)


def test_nearest_index_says_none_beyond_the_tolerance():
    t = pd.DatetimeIndex(["2026-06-01T00:00"])
    assert insitu.nearest_index(t, np.array([12.0]),
                                pd.Timestamp("2026-06-01T06:00"), 60) == (None, pytest.approx(np.nan, nan_ok=True))


# --------------------------------------------------------------------------- #
# Into the cube
# --------------------------------------------------------------------------- #
def _write_landsat(project, g, day, hour):
    H, W = g.height, g.width
    xs, ys = g.xy_centers()
    ds = xr.Dataset({
        "sst": (("time", "y", "x"), np.full((1, H, W), 285.0, "float32")),
        "cloud": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
        "water": (("time", "y", "x"), np.ones((1, H, W), "float32")),
    }, coords={"time": [day], "y": ys, "x": xs})
    d = project.output_dir / "LANDSAT" / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}_{day.strftime('%Y%m%d')}T{hour:02d}0000.nc")


def test_a_transect_paints_a_strip_not_a_point(tmp_path, g, days):
    """The user-visible feature: a ship or glider crossing the AoI leaves a line of pixels on
    the day it crossed, instead of being dropped or collapsed onto one median pixel."""
    project = _project(tmp_path, mobile={"sources": ["csv"]})
    _write_tracks(project, g, [_rec(_track_csv(tmp_path / "t.csv", n=12, hour=19))])
    _write_landsat(project, g, days[0], hour=19)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    arr = ds["insitu_sst"].isel(time=0).values
    n_px = int(np.isfinite(arr).sum())
    assert n_px > 1, "the track collapsed to a single pixel"
    assert np.allclose(arr[np.isfinite(arr)], 12.0)
    # ...and on a day the platform never passed, nothing.
    assert not np.isfinite(ds["insitu_sst"].isel(time=1).values).any()


def test_the_whole_days_transect_lands_not_just_the_reference_hour(tmp_path, g, days):
    """A transect is a spatial sample of its DAY. Gating it to the hour either side of the
    reference instant would discard most of it -- on one real Hobart day, 96 pixels collapse to
    1. So `insitu_sst` takes every observation, bounded by the day it was recorded."""
    project = _project(tmp_path, mobile={"sources": ["csv"]})
    # A crossing that starts at 02:00 and runs for hours -- nowhere near the reference time.
    _write_tracks(project, g, [_rec(_track_csv(tmp_path / "t.csv", n=12, hour=2))])
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    arr = ds["insitu_sst"].isel(time=0).values
    assert int(np.isfinite(arr).sum()) > 1, "the pre-dawn transect was gated away"
    # ...and it does NOT bleed onto the following day, which is what the day bound is for.
    assert not np.isfinite(ds["insitu_sst"].isel(time=1).values).any()


def test_insitu_hour_says_when_each_cell_was_observed(tmp_path, g, days):
    """`insitu_sst` no longer means one thing everywhere, so the cube has to record WHEN."""
    project = _project(tmp_path, mobile={"sources": ["csv"]})
    _write_tracks(project, g, [_rec(_track_csv(tmp_path / "t.csv", n=12, hour=2))])
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert "insitu_hour" in ds.data_vars
    hrs = ds["insitu_hour"].isel(time=0).values
    got = hrs[np.isfinite(hrs)]
    assert len(got) > 1 and got.min() >= 2.0 and got.max() < 3.0   # the 02:00-02:22 crossing
    assert "moving platform" in ds["insitu_sst"].attrs["comment"]


def test_a_fixed_only_cube_gains_no_hour_channel(tmp_path, g, days):
    """The design's promise, restated as a channel-set assertion: nothing about a project
    without moving platforms changes."""
    project = _project(tmp_path, fixed={"sources": ["ioos"]})
    times = pd.DatetimeIndex(["2026-06-01T19:00"])
    fixed = xr.Dataset(
        {"sst": (("station", "time"), np.array([[8.0]], dtype="float32")),
         "qc": (("station", "time"), np.ones((1, 1), "uint8"))},
        coords={"station": [0], "time": times, "station_id": ("station", ["buoy1"]),
                "station_name": ("station", ["Buoy"]),
                "lat": ("station", [LAT]), "lon": ("station", [LON]),
                "variable": ("station", ["sea_water_temperature"])})
    d = project.output_dir / "INSITU" / "ioos" / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    fixed.to_netcdf(d / f"{AOI}_insitu.nc")

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert "insitu_hour" not in ds.data_vars
    assert "insitu_tracks" not in ds.attrs
    assert "comment" not in ds["insitu_sst"].attrs


def test_the_matchup_tolerance_still_gates_a_track(tmp_path, g, days):
    """A glider that passed at 03:00 is not a matchup for a satellite that flew at 19:00. The
    tolerance is what keeps a validation set honest, and it applies per pixel."""
    project = _project(tmp_path, mobile={"sources": ["csv"]})
    _write_tracks(project, g, [_rec(_track_csv(tmp_path / "t.csv", n=12, hour=3))])
    _write_landsat(project, g, days[0], hour=19)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert not np.isfinite(ds["lst_insitu_sst"].isel(time=0).values).any()


def test_insitu_station_stays_static_and_fixed_only(tmp_path, g, days):
    """The design's core promise. A track has no static pixel, so it contributes to `insitu_sst`
    but never to the station map -- which is what keeps `insitu_station` a `(y,x)` channel and
    `mask: insitu_station` working in `extract`."""
    project = _project(tmp_path, mobile={"sources": ["csv"]})
    _write_tracks(project, g, [_rec(_track_csv(tmp_path / "t.csv", n=12, hour=19))])
    _write_landsat(project, g, days[0], hour=19)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert ds["insitu_station"].dims == ("y", "x")
    assert int(ds["insitu_station"].values.max()) == 0     # no FIXED station in this project
    assert np.isfinite(ds["lst_insitu_sst"].isel(time=0).values).sum() > 1


def test_the_track_roster_travels_with_the_cube(tmp_path, g, days):
    project = _project(tmp_path, mobile={"sources": ["csv"]})
    _write_tracks(project, g, [_rec(_track_csv(tmp_path / "t.csv", n=12, hour=19))])
    _write_landsat(project, g, days[0], hour=19)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    roster = json.loads(ds.attrs["insitu_tracks"])
    assert [r["id"] for r in roster] == ["glider1"]
    assert roster[0]["source"] == "csv" and roster[0]["n_obs"] == 12


def test_an_observation_outside_the_grid_drops_itself_not_the_platform(tmp_path, g, days):
    """A track legitimately leaves and re-enters the AoI; only the observations outside are
    lost."""
    project = _project(tmp_path, mobile={"sources": ["csv"]})
    rows = _track_csv(tmp_path / "t.csv", n=6, hour=19)
    rows.loc[0, "longitude"] = LON + 40.0                  # one point on another continent
    _write_tracks(project, g, [_rec(rows)])
    _write_landsat(project, g, days[0], hour=19)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.isfinite(ds["lst_insitu_sst"].isel(time=0).values).sum() >= 1


def test_a_track_and_a_buoy_share_one_channel(tmp_path, g, days):
    """Ground truth is ground truth: both products feed `insitu_sst`, and the fixed station
    still appears in the station map while the track does not."""
    project = _project(tmp_path, mobile={"sources": ["csv"]}, fixed={"sources": ["ioos"]})
    _write_tracks(project, g, [_rec(_track_csv(tmp_path / "t.csv", n=12, hour=19))])
    # A fixed station, in the fixed product's schema, well away from the track.
    times = pd.DatetimeIndex(["2026-06-01T19:00"])
    fixed = xr.Dataset(
        {"sst": (("station", "time"), np.array([[8.0]], dtype="float32")),
         "qc": (("station", "time"), np.ones((1, 1), "uint8"))},
        coords={"station": [0], "time": times, "station_id": ("station", ["buoy1"]),
                "station_name": ("station", ["Buoy"]),
                "lat": ("station", [LAT - 0.03]), "lon": ("station", [LON + 0.03]),
                "variable": ("station", ["sea_water_temperature"])})
    d = project.output_dir / "INSITU" / "ioos" / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    fixed.to_netcdf(d / f"{AOI}_insitu.nc")
    _write_landsat(project, g, days[0], hour=19)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    arr = ds["lst_insitu_sst"].isel(time=0).values
    vals = arr[np.isfinite(arr)]
    assert 8.0 in vals and 12.0 in vals                    # both kinds present
    assert int(ds["insitu_station"].values.max()) == 1     # only the buoy is indexed
    assert [s["id"] for s in json.loads(ds.attrs["insitu_stations"])] == ["buoy1"]
    assert [t["id"] for t in json.loads(ds.attrs["insitu_tracks"])] == ["glider1"]


# --------------------------------------------------------------------------- #
# The Copernicus mobile route
# --------------------------------------------------------------------------- #
def test_the_window_is_clamped_to_the_sparse_cube():
    """A project reaching back to 1984 must acquire the part that exists, not lose the AoI: a
    range starting before the cube RAISES rather than returning empty."""
    assert insitu_cmems._clamp_window("1984-01-01", "2026-01-01") == ("2020-01-01", "2026-01-01")
    assert insitu_cmems._clamp_window("2021-03-01", "2024-01-01") == ("2021-03-01", "2024-01-01")
    assert insitu_cmems._clamp_window("1990-01-01", "1995-01-01") is None


def test_mobile_selection_takes_the_complement_of_the_fixed_classes(monkeypatch, g):
    """By EXCLUSION, so a platform class the archive adds later is picked up without this list
    having to track the fixed product's."""
    df = pd.DataFrame({
        "platform_id": ["a", "b", "c", "d"],
        "platform_type": ["GL", "MO", "TS", "DB"],       # MO is fixed; the rest are not
        "time": pd.date_range("2021-06-01", periods=4, freq="h"),
        "latitude": LAT, "longitude": LON,
        "value": [12.0, 9.0, 13.0, 14.0], "value_qc": 1.0})
    monkeypatch.setattr(insitu_cmems, "_read_arco",
                        lambda *a, **k: df)
    cfg = {"dataset_id": "glo", "dataset_part": "monthly", "platform_types": [],
           "qc_flags": [1, 2], "max_sensor_depth_m": 5.0, "stations": [],
           "exclude_stations": [], "pad_deg": 0.0}
    recs = insitu_cmems.fetch_aoi_mobile(g, "2021-01-01", "2022-01-01", cfg)
    assert sorted(r["id"] for r in recs) == ["a", "c", "d"]


def test_the_history_part_is_never_used_for_tracks(monkeypatch, g):
    """`history` publishes original files only -- the route that cost 348 MB to find four
    in-AoI observations. Asking for it must not silently take it."""
    seen = {}
    monkeypatch.setattr(insitu_cmems, "_read_arco",
                        lambda ds_id, part, *a, **k: seen.setdefault("part", part) and None)
    cfg = {"dataset_id": "glo", "dataset_part": "history", "platform_types": [],
           "qc_flags": [1], "max_sensor_depth_m": 5.0, "stations": [],
           "exclude_stations": [], "pad_deg": 0.0}
    insitu_cmems.fetch_aoi_mobile(g, "2021-01-01", "2022-01-01", cfg)
    assert seen["part"] == insitu_cmems.DEFAULT_MOBILE_PART


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
