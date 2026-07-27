"""In-situ (IOOS/ERDDAP). All network goes through the single seam `insitu_ioos._get`,
monkeypatched with table-driven fakes -- offline. The emphasis is on the two ways this
product fails SILENTLY (an empty in-situ channel that looks like 'no buoys here') and on
the matchup, which is the whole reason to carry in-situ at all: a stale observation
passed off as a matchup quietly biases a validation set."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import datacube, insitu, insitu_ioos


AOI = "aoi1"
# The AoI centre; Garibaldi-like coordinates sit inside it.
LAT, LON = 45.52, -123.925

SEARCH_JSON = """{"table":{"columnNames":["Dataset ID","Title"],"rows":[
  ["noaa_nos_co_ops_9437540","Garibaldi, OR (TLBO3)"],
  ["gov-ndbc-46120","Wave buoy, no thermometer"]]}}"""

INFO_WITH_SWT = """{"table":{"rows":[
  ["variable","time"],["variable","latitude"],["variable","longitude"],["variable","z"],
  ["variable","sea_water_temperature"],["variable","sea_water_temperature_qc_agg"]]}}"""
INFO_ONLY_SST = """{"table":{"rows":[
  ["variable","time"],["variable","latitude"],["variable","longitude"],
  ["variable","sea_surface_temperature"],["variable","sea_surface_temperature_qc_agg"]]}}"""
INFO_NO_TEMP = """{"table":{"rows":[
  ["variable","time"],["variable","latitude"],["variable","longitude"],
  ["variable","wave_height"]]}}"""

HDR = ("time (UTC),latitude (degrees_north),longitude (degrees_east),z (m),"
       "sea_water_temperature (degree_Celsius),sea_water_temperature_qc_agg")


def csv_rows(rows):
    return HDR + "\n" + "\n".join(rows) + "\n"


GOOD_CSV = csv_rows([
    f"2026-06-01T09:00:00Z,{LAT},{LON},0.0,11.5,1",
    f"2026-06-01T18:00:00Z,{LAT},{LON},0.0,12.5,1",
    f"2026-06-01T19:30:00Z,{LAT},{LON},0.0,13.0,1",
])
# The 46120 trap: the variable exists, and is never reported.
ALL_NAN_CSV = csv_rows([
    f"2026-06-01T09:00:00Z,{LAT},{LON},0.0,NaN,2",
    f"2026-06-01T18:00:00Z,{LAT},{LON},0.0,NaN,2",
])


class FakeERDDAP:
    """Serves search / info / tabledap from a table, and records every URL asked for."""

    def __init__(self, info=None, data=None, search=SEARCH_JSON):
        self.info = info or {}
        self.data = data or {}
        self.search = search
        self.calls = []

    def __call__(self, url, params=None, **kw):
        self.calls.append(url)
        if "/search/advanced" in url:
            return self.search
        if "/info/" in url:
            sid = url.split("/info/")[1].split("/")[0]
            return self.info.get(sid, INFO_NO_TEMP)
        if "/tabledap/" in url:
            sid = url.split("/tabledap/")[1].split(".")[0]
            return self.data.get(sid, "Error { code=400; }")
        raise AssertionError(f"unexpected URL {url}")


def _project(tmp_path, **opts):
    return parse_config({
        "name": "i", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-02"},
        "products": {"insitu": opts or {}},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": LAT, "center_lon": LON,
             "buffer_ns_km": 5, "buffer_ew_km": 5}]}],
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


# --------------------------------------------------------------------------- #
# The two silent-failure traps
# --------------------------------------------------------------------------- #
def test_a_station_that_never_reports_is_dropped_and_logged(monkeypatch, project, g, caplog):
    """NDBC 46120 advertises the temperature variable and returns all-NaN forever. It must
    be dropped LOUDLY -- an empty in-situ channel that reads as 'no buoys here' is the
    failure this product cannot afford."""
    fake = FakeERDDAP(
        info={"noaa_nos_co_ops_9437540": INFO_WITH_SWT, "gov-ndbc-46120": INFO_WITH_SWT},
        data={"noaa_nos_co_ops_9437540": GOOD_CSV, "gov-ndbc-46120": ALL_NAN_CSV})
    monkeypatch.setattr(insitu_ioos, "_get", fake)

    with caplog.at_level("WARNING"):
        insitu_ioos.acquire(project, grids={AOI: g})

    with xr.open_dataset(project.output_dir / "INSITU" / "aligned" / AOI /
                         f"{AOI}_insitu.nc") as ds:
        assert list(ds["station_id"].values) == ["noaa_nos_co_ops_9437540"]
    assert "dropped gov-ndbc-46120" in caplog.text
    assert "no QC-passing values" in caplog.text


def test_extending_the_date_range_rebuilds_the_single_file_series(monkeypatch, tmp_path):
    """Insitu writes ONE file for the whole window, so extending end_date must REBUILD it --
    the file already exists and passes is_complete, so the added dates would silently stay NaN.
    `covers=(start, end)` on the skip guard is what forces the rebuild."""
    fake = FakeERDDAP(info={"noaa_nos_co_ops_9437540": INFO_WITH_SWT},
                      data={"noaa_nos_co_ops_9437540": GOOD_CSV})
    monkeypatch.setattr(insitu_ioos, "_get", fake)

    def run_for(end):
        proj = parse_config({
            "name": "i", "output_dir": str(tmp_path),
            "time": {"start_date": "2026-06-01", "end_date": end},
            "products": {"insitu": {}},
            "regions": [{"name": "r", "areas": [
                {"name": AOI, "center_lat": LAT, "center_lon": LON,
                 "buffer_ns_km": 5, "buffer_ew_km": 5}]}],
        })
        insitu_ioos.acquire(proj, grids={AOI: grid.project_grids(proj)[AOI]})
        return proj.output_dir / "INSITU" / "aligned" / AOI / f"{AOI}_insitu.nc"

    n_fetch = lambda: sum("/tabledap/" in u for u in fake.calls)

    out = run_for("2026-06-02")
    with xr.open_dataset(out) as ds:
        assert ds.attrs["requested_end"] == "2026-06-02"
    after_first = n_fetch()

    run_for("2026-06-02")                       # same window -> covered -> skipped, no new fetch
    assert n_fetch() == after_first

    run_for("2026-06-30")                       # extended -> old file no longer spans it -> rebuilt
    assert n_fetch() > after_first
    with xr.open_dataset(out) as ds:
        assert ds.attrs["requested_end"] == "2026-06-30"


def test_a_station_lacking_the_variable_is_never_queried_for_it(monkeypatch, project, g):
    """Asking ERDDAP for a variable a station lacks is an HTTP 400, which would kill the
    station. So its variable list is read first, and it is simply not asked."""
    fake = FakeERDDAP(
        info={"noaa_nos_co_ops_9437540": INFO_WITH_SWT, "gov-ndbc-46120": INFO_NO_TEMP},
        data={"noaa_nos_co_ops_9437540": GOOD_CSV})
    monkeypatch.setattr(insitu_ioos, "_get", fake)
    insitu_ioos.acquire(project, grids={AOI: g})

    assert not any("/tabledap/gov-ndbc-46120" in u for u in fake.calls)
    assert any("/tabledap/noaa_nos_co_ops_9437540" in u for u in fake.calls)


def test_the_variable_name_falls_back_per_station():
    """Providers do not agree on the name: some expose only sea_surface_temperature."""
    prefer = ["sea_water_temperature", "sea_surface_temperature"]
    assert insitu_ioos.pick_variable({"sea_water_temperature"}, prefer) == "sea_water_temperature"
    assert insitu_ioos.pick_variable({"sea_surface_temperature"}, prefer) == "sea_surface_temperature"
    assert insitu_ioos.pick_variable({"wave_height"}, prefer) is None


def test_z_and_qc_columns_are_only_requested_when_present(monkeypatch):
    """Requesting a column the station lacks is the same 400. Build the projection from
    what the station actually has."""
    fake = FakeERDDAP(data={"s": csv_rows([f"2026-06-01T09:00:00Z,{LAT},{LON},0.0,11.5,1"])})
    monkeypatch.setattr(insitu_ioos, "_get", fake)
    insitu_ioos.fetch_station("s", "sea_water_temperature", "2026-06-01", "2026-06-02",
                              [1, 2], 5.0, available={"sea_water_temperature"})
    url = fake.calls[0]
    assert "z" not in url.split("?")[1].split("&")[0].split("%2C")
    assert "qc_agg" not in url


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
def test_suspect_and_failed_observations_are_dropped(monkeypatch):
    fake = FakeERDDAP(data={"s": csv_rows([
        f"2026-06-01T09:00:00Z,{LAT},{LON},0.0,11.5,1",    # pass
        f"2026-06-01T10:00:00Z,{LAT},{LON},0.0,12.0,2",    # not evaluated -> kept
        f"2026-06-01T11:00:00Z,{LAT},{LON},0.0,99.0,3",    # suspect -> dropped
        f"2026-06-01T12:00:00Z,{LAT},{LON},0.0,99.0,4",    # fail    -> dropped
        f"2026-06-01T13:00:00Z,{LAT},{LON},0.0,99.0,9",    # missing -> dropped
    ])})
    monkeypatch.setattr(insitu_ioos, "_get", fake)
    df = insitu_ioos.fetch_station("s", "sea_water_temperature", "2026-06-01",
                                   "2026-06-02", [1, 2], 5.0,
                                   available={"z", "sea_water_temperature",
                                              "sea_water_temperature_qc_agg"})
    assert list(df["value"]) == [11.5, 12.0]           # 99.0 never survives


def test_deep_sensors_are_ignored(monkeypatch):
    fake = FakeERDDAP(data={"s": csv_rows([
        f"2026-06-01T09:00:00Z,{LAT},{LON},0.0,11.5,1",
        f"2026-06-01T10:00:00Z,{LAT},{LON},50.0,7.0,1",    # a deep sensor on a mooring
    ])})
    monkeypatch.setattr(insitu_ioos, "_get", fake)
    df = insitu_ioos.fetch_station("s", "sea_water_temperature", "2026-06-01",
                                   "2026-06-02", [1, 2], 5.0,
                                   available={"z", "sea_water_temperature",
                                              "sea_water_temperature_qc_agg"})
    assert list(df["value"]) == [11.5]


# --------------------------------------------------------------------------- #
# Time matching (pure)
# --------------------------------------------------------------------------- #
def test_nearest_observation_within_tolerance_is_used():
    t = pd.DatetimeIndex(["2026-06-01T18:00", "2026-06-01T19:30"])
    v = np.array([12.5, 13.0])
    val, dt = insitu.value_at(t, v, pd.Timestamp("2026-06-01T19:00"), 60)
    assert val == pytest.approx(13.0)                  # 19:30 is 30 min away; 18:00 is 60
    assert dt == pytest.approx(30.0)                   # signed: the obs came AFTER


def test_an_observation_beyond_the_tolerance_is_not_a_matchup():
    """A buoy reading 90 min from the overpass is not truth for that scene. NaN, never a
    stale value -- that is how a validation set quietly acquires a bias."""
    t = pd.DatetimeIndex(["2026-06-01T18:00"])
    val, dt = insitu.value_at(t, np.array([12.5]), pd.Timestamp("2026-06-01T19:30"), 60)
    assert np.isnan(val) and np.isnan(dt)


def test_a_gap_is_not_a_nearest_neighbour():
    """The nearest record in time is useless if its value is NaN."""
    t = pd.DatetimeIndex(["2026-06-01T18:55", "2026-06-01T18:00"])
    val, _ = insitu.value_at(t, np.array([np.nan, 12.5]), pd.Timestamp("2026-06-01T19:00"), 60)
    assert val == pytest.approx(12.5)


def test_no_observations_at_all():
    assert all(np.isnan(x) for x in insitu.value_at(pd.DatetimeIndex([]), np.array([]),
                                                    pd.Timestamp("2026-06-01"), 60))
    assert all(np.isnan(x) for x in insitu.value_at(
        pd.DatetimeIndex(["2026-06-01T18:00"]), np.array([12.5]), None, 60))


# --------------------------------------------------------------------------- #
# Station -> pixel
# --------------------------------------------------------------------------- #
def test_a_station_lands_in_its_own_cell(g):
    water = np.ones((g.height, g.width), dtype=bool)
    [p] = insitu.station_pixels([LON], [LAT], g, water)
    assert p["inside"] and p["snap_m"] == 0.0
    # ...and that cell really is the one containing the station.
    lons, lats = g.lonlat_centers()
    assert abs(lons[p["row"], p["col"]] - LON) < 0.002
    assert abs(lats[p["row"], p["col"]] - LAT) < 0.002


def test_a_station_on_a_land_pixel_snaps_to_the_nearest_water(g, caplog):
    """A mooring near shore, on a coarse water mask, can land in a cell the cube calls
    LAND -- where it would be masked out of every downstream loss."""
    water = np.ones((g.height, g.width), dtype=bool)
    [home] = insitu.station_pixels([LON], [LAT], g, water)
    water[home["row"], home["col"]] = False          # its own cell is now land
    [p] = insitu.station_pixels([LON], [LAT], g, water)
    assert (p["row"], p["col"]) != (home["row"], home["col"])
    assert p["snap_m"] == pytest.approx(g.resolution_m, abs=1e-6)   # one cell over


def test_a_station_outside_the_grid_is_reported_not_placed(g):
    [p] = insitu.station_pixels([LON + 5.0], [LAT + 5.0], g, None)
    assert p["inside"] is False


# --------------------------------------------------------------------------- #
# Into the cube
# --------------------------------------------------------------------------- #
def _write_insitu(project, g, times, values, station="s1"):
    ds = xr.Dataset(
        {"sst": (("station", "time"), np.array([values], dtype="float32")),
         "qc": (("station", "time"), np.ones((1, len(times)), "uint8"))},
        coords={"station": [0], "time": pd.DatetimeIndex(times),
                "station_id": ("station", [station]),
                "station_name": ("station", ["Test Station"]),
                "lat": ("station", [LAT]), "lon": ("station", [LON]),
                "variable": ("station", ["sea_water_temperature"])})
    d = project.output_dir / "INSITU" / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}_insitu.nc")


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


def test_the_buoy_value_lands_in_the_buoys_pixel_at_the_overpass(project, g, days):
    """The point of the whole product: the satellite pixel and the buoy pixel are the same
    pixel, at the same instant."""
    _write_insitu(project, g,
                  ["2026-06-01T18:00", "2026-06-01T19:10", "2026-06-01T23:00"],
                  [12.5, 13.0, 9.9])
    _write_landsat(project, g, days[0], hour=19)      # overpass 19:00

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    arr = ds["lst_insitu_sst"].isel(time=0).values
    assert np.isfinite(arr).sum() == 1                 # exactly one pixel, not a blob
    r, c = np.argwhere(np.isfinite(arr))[0]
    assert arr[r, c] == pytest.approx(13.0)            # the 19:10 obs, 10 min out
    assert ds["lst_insitu_dt_min"].isel(time=0).values[r, c] == pytest.approx(10.0)
    # the station map indexes the table that travels with the cube
    assert ds["insitu_station"].values[r, c] == 1
    import json
    assert json.loads(ds.attrs["insitu_stations"])[0]["id"] == "s1"


def test_no_scene_means_no_matchup(project, g, days):
    _write_insitu(project, g, ["2026-06-01T18:00"], [12.5])
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.isnan(ds["lst_insitu_sst"].values).all()     # Landsat never flew


def test_an_out_of_tolerance_observation_gives_no_matchup(project, g, days):
    _write_insitu(project, g, ["2026-06-01T12:00"], [12.5])   # 7 h from the overpass
    _write_landsat(project, g, days[0], hour=19)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.isnan(ds["lst_insitu_sst"].isel(time=0).values).all()


def test_the_reference_time_channel_is_sampled_like_met(project, g, days):
    """insitu_sst is taken at 10:30 local solar -- the same instant as the met channels --
    so the two are contemporaneous. At this longitude that is ~18:30 UTC."""
    _write_insitu(project, g,
                  ["2026-06-01T03:00", "2026-06-01T18:30", "2026-06-01T23:00"],
                  [8.0, 14.0, 9.0])
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    arr = ds["insitu_sst"].isel(time=0).values
    assert np.nanmax(arr) == pytest.approx(14.0)      # the 18:30 obs, not the 03:00 one


def test_insitu_can_be_turned_off(project, g, days):
    _write_insitu(project, g, ["2026-06-01T18:00"], [12.5])
    eff = datacube._build_eff(project)
    eff["insitu"] = False
    ds = datacube.assemble_aoi(g, eff, days)
    assert "insitu_sst" not in ds.data_vars

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])