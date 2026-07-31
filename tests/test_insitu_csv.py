"""User-provided in-situ observations from long-format CSVs.

Entirely offline -- this source has no network at all, which is the point: everything here
is the user's file and our parsing of it. The emphasis is on the failures that would
otherwise be SILENT, because a wrong in-situ channel is worse than a missing one: it is the
cube's only ground truth, so a bad value does not look like an error, it looks like the
truth. Specifically:

  * a multi-station file that gets collapsed into one platform,
  * a units mistake (54 degF is a plausible degC sea temperature),
  * a mistyped path, which yields an empty channel indistinguishable from "no thermometers".
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import datacube, insitu_acquire, insitu_csv


AOI = "aoi1"
LAT, LON = 45.52, -123.925

HDR = "station_id,time,latitude,longitude,value"


def write_csv(path, rows, header=HDR):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


# Three platforms at three distinct positions, in ONE file -- the normal case.
THREE_STATIONS = [
    f"mooring_a,2026-06-01T18:00:00Z,{LAT},{LON},11.0",
    f"mooring_a,2026-06-01T18:30:00Z,{LAT},{LON},11.2",
    f"mooring_b,2026-06-01T18:00:00Z,{LAT + 0.01},{LON},12.0",
    f"mooring_b,2026-06-01T18:30:00Z,{LAT + 0.01},{LON},12.2",
    f"mooring_c,2026-06-01T18:00:00Z,{LAT},{LON + 0.01},13.0",
    f"mooring_c,2026-06-01T18:30:00Z,{LAT},{LON + 0.01},13.2",
]


def _project(tmp_path, **opts):
    return parse_config({
        "name": "i", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-02"},
        "products": {"insitu": {"sources": ["csv"], **opts}},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": LAT, "center_lon": LON,
             "buffer_ns_km": 5, "buffer_ew_km": 5}]}],
    })


@pytest.fixture
def g(tmp_path):
    return grid.project_grids(_project(tmp_path, path="/nowhere"))[AOI]


def cfg(**over):
    """A resolved options bag, as insitu_acquire._ds_cfg would build it."""
    base = {"path": None, "columns": {}, "time_zone": "UTC", "units": "degC",
            "qc_pass_values": None, "default_station_id": None,
            "stations": [], "exclude_stations": [], "max_sensor_depth_m": 5.0}
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# Many stations in one file -- the case this loader exists for
# --------------------------------------------------------------------------- #
def test_one_file_holding_three_stations_yields_three_platforms(tmp_path, g):
    """Rows are GROUPED BY station_id. A logger-fleet export is one file, and each station in
    it must keep its own identity and its own position."""
    f = write_csv(tmp_path / "fleet.csv", THREE_STATIONS)
    recs = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))

    assert [r["id"] for r in recs] == ["mooring_a", "mooring_b", "mooring_c"]
    # Three DISTINCT positions -- not one file-wide average.
    assert len({(round(r["lat"], 4), round(r["lon"], 4)) for r in recs}) == 3
    assert recs[0]["lat"] == pytest.approx(LAT)
    assert recs[1]["lat"] == pytest.approx(LAT + 0.01)
    assert recs[2]["lon"] == pytest.approx(LON + 0.01)


def test_three_stations_land_in_three_pixels_of_the_cube(tmp_path):
    """The end-to-end payoff: three rows in one CSV become three separate ground-truth
    pixels, each with its own value."""
    project = _project(tmp_path, path=str(write_csv(tmp_path / "fleet.csv", THREE_STATIONS)))
    g = grid.project_grids(project)[AOI]
    insitu_acquire.acquire(project, grids={AOI: g})

    days = pd.date_range(project.time.start_date, project.time.end_date, freq="D")
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    arr = ds["insitu_sst"].isel(time=0).values
    assert np.isfinite(arr).sum() == 3                    # three pixels, not one blob
    # insitu_sst is sampled at the daily reference time (10:30 local solar ~= 18:30 UTC
    # here), so each platform contributes its 18:30 observation.
    assert sorted(np.round(arr[np.isfinite(arr)], 1)) == [11.2, 12.2, 13.2]
    import json
    assert [s["source"] for s in json.loads(ds.attrs["insitu_stations"])] == ["csv"] * 3


def test_the_same_station_split_across_files_merges_into_one_platform(tmp_path, g):
    """A per-deployment or per-year export set must not become several platforms stacked in
    one pixel."""
    write_csv(tmp_path / "a.csv", THREE_STATIONS[:2])
    write_csv(tmp_path / "b.csv", [f"mooring_a,2026-06-02T18:00:00Z,{LAT},{LON},11.9"])
    recs = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(tmp_path)))

    assert [r["id"] for r in recs] == ["mooring_a"]
    assert len(recs[0]["df"]) == 3                        # 2 + 1, in time order
    assert recs[0]["df"]["time"].is_monotonic_increasing


def test_station_lists_narrow_a_multi_station_file(tmp_path, g):
    """A 50-platform file should not have to be edited to use 5 of them."""
    f = write_csv(tmp_path / "fleet.csv", THREE_STATIONS)
    keep = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                                cfg(path=str(f), stations=["mooring_b"]))
    assert [r["id"] for r in keep] == ["mooring_b"]

    rest = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                                cfg(path=str(f), exclude_stations=["mooring_b"]))
    assert [r["id"] for r in rest] == ["mooring_a", "mooring_c"]


def test_a_file_with_no_station_id_column_is_one_platform(tmp_path, g):
    """A single-logger export has no id column; the file itself is the platform."""
    f = write_csv(tmp_path / "logger7.csv",
                  [f"2026-06-01T18:00:00Z,{LAT},{LON},11.0"],
                  header="time,latitude,longitude,value")
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert rec["id"] == "logger7"                          # the file stem

    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                                 cfg(path=str(f), default_station_id="pier_a"))
    assert rec["id"] == "pier_a"


def test_an_unmapped_station_id_on_a_synchronised_fleet_is_rejected(tmp_path, g):
    """THE trap, in its nastiest form. Loggers usually share a time base, so three merged
    stations report the SAME instants from three places. De-duplicating by time would keep
    one station's rows and silently discard the other two -- leaving one station's data
    wearing the whole file's name, with zero apparent drift for any guard to notice."""
    f = write_csv(tmp_path / "fleet.csv", THREE_STATIONS,
                  header="platform,time,latitude,longitude,value")
    with pytest.raises(ValueError) as e:
        insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert "station_id" in str(e.value)
    assert "SEVERAL STATIONS" in str(e.value)

    # ...and mapping the column is the fix.
    ok = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                              cfg(path=str(f), columns={"station_id": "platform"}))
    assert len(ok) == 3


def test_an_unmapped_station_id_on_staggered_stations_is_caught_as_drift(tmp_path, g):
    """When merged stations DON'T share timestamps there are no duplicate instants to catch,
    so the fallback is the drift guard -- whose message must also name `station_id`, or the
    user goes hunting for a bug in their coordinates."""
    f = write_csv(tmp_path / "fleet.csv", [
        f"a,2026-06-01T18:00:00Z,{LAT},{LON},11.0",
        f"b,2026-06-01T18:05:00Z,{LAT + 0.02},{LON},12.0",     # ~2.2 km away, 5 min later
    ], header="platform,time,latitude,longitude,value")
    recs = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert len(recs) == 1                                  # merged, undetectably so far

    keep, moving = insitu_acquire.split_moving_platforms(recs, g.resolution_m)
    assert keep == []
    assert "station_id" in insitu_acquire._moving_platform_message(*moving[0])


def test_repeated_rows_at_one_position_are_not_mistaken_for_merged_stations(tmp_path, g):
    """A duplicated row is a duplicated row -- only a duplicate instant from a DIFFERENT
    position is evidence of merged stations."""
    f = write_csv(tmp_path / "dup.csv", [
        f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5",
        f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5",
    ])
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert len(rec["df"]) == 1


# --------------------------------------------------------------------------- #
# Column mapping
# --------------------------------------------------------------------------- #
def test_columns_are_remappable(tmp_path, g):
    """Users should not have to rewrite files they already have."""
    f = write_csv(tmp_path / "odd.csv",
                  [f"buoy1,Deep Water,2026-06-01T18:00:00Z,{LAT},{LON},11.5"],
                  header="platform,label,datetime_utc,lat,lon,temp_c")
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(
        path=str(f),
        columns={"station_id": "platform", "station_name": "label",
                 "time": "datetime_utc", "latitude": "lat", "longitude": "lon",
                 "value": "temp_c"}))
    assert rec["id"] == "buoy1" and rec["title"] == "Deep Water"
    assert rec["df"]["value"].iloc[0] == pytest.approx(11.5)


def test_a_missing_required_column_names_the_file_and_the_column(tmp_path, g):
    """The error has to be actionable: which file, which field, and what was actually there."""
    f = write_csv(tmp_path / "bad.csv", [f"b1,2026-06-01T18:00:00Z,{LAT},{LON}"],
                  header="station_id,time,latitude,longitude")
    with pytest.raises(ValueError) as e:
        insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    msg = str(e.value)
    assert "bad.csv" in msg and "value" in msg and "longitude" in msg   # names what IS there


def test_an_unknown_column_field_is_rejected(tmp_path):
    """A `columns:` key nothing reads would silently do nothing."""
    with pytest.raises(ValueError, match="not a field this loader reads"):
        insitu_csv.column_map({"temperature": "temp_c"})


# --------------------------------------------------------------------------- #
# Units -- the silent one
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("units,raw,expect", [
    ("degC", "11.5", 11.5),
    ("degF", "52.7", 11.5),
    ("K", "284.65", 11.5),
])
def test_units_are_converted_to_degc(tmp_path, g, units, raw, expect):
    f = write_csv(tmp_path / "u.csv", [f"b1,2026-06-01T18:00:00Z,{LAT},{LON},{raw}"])
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                                 cfg(path=str(f), units=units))
    assert rec["df"]["value"].iloc[0] == pytest.approx(expect, abs=0.05)


def test_degf_read_as_degc_is_flagged_loudly(tmp_path, g, caplog):
    """54 degF is a plausible degC sea temperature, so nothing downstream would ever catch
    this. The range gate is the only chance to say something."""
    f = write_csv(tmp_path / "u.csv", [f"b1,2026-06-01T18:00:00Z,{LAT},{LON},127.4"])
    with caplog.at_level("WARNING"):
        insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert "insitu.units" in caplog.text


def test_an_unrecognised_unit_is_rejected(tmp_path, g):
    f = write_csv(tmp_path / "u.csv", [f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5"])
    with pytest.raises(ValueError, match="units"):
        insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                             cfg(path=str(f), units="millikelvin"))


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def test_a_naive_timestamp_is_read_in_the_configured_zone(tmp_path, g):
    """The assumption most likely to be wrong in a user's file, so it is configurable."""
    f = write_csv(tmp_path / "t.csv", [f"b1,2026-06-01 11:00:00,{LAT},{LON},11.5"])
    [utc] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert utc["df"]["time"].iloc[0] == pd.Timestamp("2026-06-01T11:00")

    [pac] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                                 cfg(path=str(f), time_zone="America/Los_Angeles"))
    assert pac["df"]["time"].iloc[0] == pd.Timestamp("2026-06-01T18:00")   # PDT = UTC-7


def test_an_offset_bearing_timestamp_is_honoured(tmp_path, g):
    f = write_csv(tmp_path / "t.csv", [f"b1,2026-06-01T11:00:00-07:00,{LAT},{LON},11.5"])
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert rec["df"]["time"].iloc[0] == pd.Timestamp("2026-06-01T18:00")


def test_unparseable_times_are_dropped_with_a_count(tmp_path, g, caplog):
    f = write_csv(tmp_path / "t.csv", [
        f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5",
        f"b1,not-a-date,{LAT},{LON},12.5",
    ])
    with caplog.at_level("WARNING"):
        [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert len(rec["df"]) == 1
    assert "unparseable time" in caplog.text


def test_observations_outside_the_project_window_are_dropped(tmp_path, g):
    f = write_csv(tmp_path / "t.csv", [
        f"b1,2026-05-01T18:00:00Z,{LAT},{LON},9.0",       # before
        f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5",      # inside
        f"b1,2026-06-02T23:30:00Z,{LAT},{LON},12.0",      # inside: end date is a whole DAY
        f"b1,2026-07-01T18:00:00Z,{LAT},{LON},15.0",      # after
    ])
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert list(np.round(rec["df"]["value"], 1)) == [11.5, 12.0]


# --------------------------------------------------------------------------- #
# QC and depth
# --------------------------------------------------------------------------- #
def test_no_qc_column_means_not_evaluated(tmp_path, g):
    """QARTOD 2 is what a station that does not run QC asserts -- and what the default
    qc_flags accept, so the data is not thrown away for lacking a column."""
    f = write_csv(tmp_path / "q.csv", [f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5"])
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert rec["df"]["qc"].iloc[0] == insitu_csv.DEFAULT_QC_FLAG


def test_qc_pass_values_filter_rows(tmp_path, g):
    f = write_csv(tmp_path / "q.csv", [
        f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5,1",
        f"b1,2026-06-01T19:00:00Z,{LAT},{LON},99.0,4",
    ], header=HDR + ",qc")
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                                 cfg(path=str(f), qc_pass_values=[1, 2]))
    assert list(rec["df"]["value"]) == [11.5]


def test_deep_sensors_are_ignored(tmp_path, g):
    f = write_csv(tmp_path / "z.csv", [
        f"b1,2026-06-01T18:00:00Z,{LAT},{LON},11.5,0.0",
        f"b1,2026-06-01T19:00:00Z,{LAT},{LON},7.0,50.0",
    ], header=HDR + ",z")
    [rec] = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                                 cfg(path=str(f), max_sensor_depth_m=5.0))
    assert list(rec["df"]["value"]) == [11.5]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_a_file_a_directory_and_a_glob_all_resolve(tmp_path):
    write_csv(tmp_path / "a.csv", THREE_STATIONS[:2])
    write_csv(tmp_path / "b.csv", THREE_STATIONS[2:4])
    (tmp_path / "notes.txt").write_text("ignore me")

    assert len(insitu_csv.resolve_paths(tmp_path / "a.csv")) == 1
    assert len(insitu_csv.resolve_paths(tmp_path)) == 2            # *.csv only
    assert len(insitu_csv.resolve_paths(str(tmp_path / "*.csv"))) == 2


def test_a_path_matching_nothing_is_an_error_not_an_empty_channel(tmp_path, g):
    """An in-situ channel that is empty because a path was mistyped looks exactly like one
    that is empty because there are no thermometers here. Never the same outcome."""
    with pytest.raises(ValueError, match="matched no CSV files"):
        insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02",
                             cfg(path=str(tmp_path / "typo" / "*.csv")))


def test_the_csv_source_without_a_path_says_so(tmp_path, g):
    with pytest.raises(ValueError, match="needs `insitu.path`"):
        insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg())


def test_a_platform_outside_the_aoi_is_dropped(tmp_path, g):
    """One file can serve every AoI in a project; each keeps only what falls in its grid."""
    f = write_csv(tmp_path / "wide.csv", [
        f"here,2026-06-01T18:00:00Z,{LAT},{LON},11.5",
        f"far_away,2026-06-01T18:00:00Z,{LAT + 5},{LON + 5},20.0",
    ])
    recs = insitu_csv.fetch_aoi(g, "2026-06-01", "2026-06-02", cfg(path=str(f)))
    assert [r["id"] for r in recs] == ["here"]


# --------------------------------------------------------------------------- #
# Stacking with the public network
# --------------------------------------------------------------------------- #
def test_csv_and_ioos_platforms_stack_into_one_station_table(tmp_path):
    """The whole point of making in-situ a stacked product: public buoys AND your own
    thermometers, in one cube, in one channel set."""
    project = _project(tmp_path, path=str(write_csv(tmp_path / "mine.csv",
                                                    THREE_STATIONS[:2])))
    g = grid.project_grids(project)[AOI]
    insitu_acquire.acquire(project, grids={AOI: g})

    # Stand in for the IOOS tree with a file in its own source dir.
    ioos_dir = project.output_dir / "INSITU" / "ioos" / "aligned" / AOI
    ioos_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"sst": (("station", "time"), np.array([[12.5]], dtype="float32")),
         "qc": (("station", "time"), np.ones((1, 1), "uint8"))},
        coords={"station": [0], "time": pd.DatetimeIndex(["2026-06-01T18:00"]),
                "station_id": ("station", ["ndbc_1"]),
                "station_name": ("station", ["A Public Buoy"]),
                "lat": ("station", [LAT + 0.02]), "lon": ("station", [LON]),
                "variable": ("station", ["sea_water_temperature"])},
    ).to_netcdf(ioos_dir / f"{AOI}_insitu.nc")

    days = pd.date_range(project.time.start_date, project.time.end_date, freq="D")
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    import json
    table = json.loads(ds.attrs["insitu_stations"])
    assert {s["source"] for s in table} == {"csv", "ioos"}
    assert {s["id"] for s in table} == {"mooring_a", "ndbc_1"}


def test_a_stray_directory_under_insitu_is_ignored_not_read_as_a_source(tmp_path, caplog):
    """The bug docs/bug-empty-version-tag-channels.md records: every subdirectory taken as a
    source tag, one leftover directory enough to produce silent sentinel channels."""
    project = _project(tmp_path, path=str(write_csv(tmp_path / "mine.csv",
                                                    THREE_STATIONS[:2])))
    g = grid.project_grids(project)[AOI]
    insitu_acquire.acquire(project, grids={AOI: g})
    (project.output_dir / "INSITU" / "_tmp_scratch").mkdir(parents=True, exist_ok=True)

    with caplog.at_level("WARNING"):
        sources = datacube.load_insitu(project.output_dir / "INSITU", AOI)
    assert [s for s, _ in sources] == ["csv"]
    assert "_tmp_scratch" in caplog.text
    for _, d in sources:
        d.close()
