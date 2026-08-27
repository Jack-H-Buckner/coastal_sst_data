"""In-situ station DISCOVERY -- the metadata-only seam behind `grids --plot --insitu`.

Everything here runs offline. IOOS goes through the module's one `_get` seam (the
`test_insitu` pattern -- the transport is what has traps in it here, since both ERDDAP
endpoints are hand-built query strings); Copernicus gets a synthetic index catalog; CSV reads
real temp files, because it always does.

The behaviours worth pinning are the ones that were wrong at some point in writing this:

  * intersection, not containment, in the `allDatasets` constraints, or every track vanishes;
  * a track whose reported extent ENGULFS the searched box is NOT placed anywhere, or three
    North Pacific gliders come out stacked on a Hood Canal AoI claiming `km_from_aoi=0.00`;
  * a wide-bounded "mooring" is demoted to mobile, because the index lies about bounds and
    nothing here opens a file to check.
"""

import json
import math

import pytest

from coastal_sst_data.processes import insitu_cmems, insitu_csv, insitu_ioos, insitu_stations
from coastal_sst_data.processes.insitu_stations import Station

BOX = (-123.2, 47.3, -123.0, 47.5)

# The index catalog's five comment lines then the `# `-prefixed header, as in
# `test_insitu_cmems` -- `parse_index` skips exactly five lines and strips exactly that prefix.
_INDEX_HEAD = (
    "# Title : in-situ files catalog\n# Description : catalog\n"
    "# Project : Copernicus Marine In Situ TAC\n# Format version : 3.0\n"
    "# Date of update : 2026-08-24T20:34:16Z\n"
    "# product_id,file_name,geospatial_lat_min,geospatial_lat_max,geospatial_lon_min,"
    "geospatial_lon_max,time_coverage_start,time_coverage_end,institution,date_update,"
    "data_mode,parameters\n")


def _row(code, dtype, lat0, lat1, lon0, lon1, t0, t1, params="TEMP PSAL"):
    return (f"COP-X,INSITU_GLO/x/history/{dtype}/GL_TS_{dtype}_{code}.nc,"
            f"{lat0},{lat1},{lon0},{lon1},{t0},{t1},Inst,2026-01-01T00:00:00Z,R,{params}\n")


# A mooring in the box; a mooring whose bounds are absurdly WIDE; a glider and a ship track;
# a mooring with no TEMP; and one that ran decades before the window.
INDEX_ROWS = [
    _row("MOORED", "MO", 46.90, 46.94, -124.12, -124.08,
         "2004-09-08T16:00:00Z", "2026-07-31T23:56:00Z"),
    _row("WIDE", "MO", 40.0, 50.0, -130.0, -120.0,
         "1987-05-06T00:00:00Z", "1987-11-30T00:00:00Z"),
    _row("GLIDER", "GL", 46.90, 46.94, -124.12, -124.08,
         "2010-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    _row("SHIP", "TS", 46.90, 46.94, -124.12, -124.08,
         "2010-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    _row("NOTEMP", "MO", 46.90, 46.94, -124.12, -124.08,
         "2010-01-01T00:00:00Z", "2020-01-01T00:00:00Z", params="PSAL VHM0"),
    _row("ELSEWHEN", "MO", 46.90, 46.94, -124.12, -124.08,
         "1950-01-01T00:00:00Z", "1955-01-01T00:00:00Z"),
]


def make_index(rows) -> bytes:
    return (_INDEX_HEAD + "".join(rows)).encode()


# --------------------------------------------------------------------------- #
# The halo
# --------------------------------------------------------------------------- #
def test_halo_grows_by_a_real_distance_not_a_degree():
    """10 km is 10 km at 47 N too -- the E/W growth is divided by cos(lat)."""
    w, s, e, n = insitu_stations.halo_bbox(BOX, 10.0)
    assert (n - BOX[3]) == pytest.approx(10.0 / 111.0, rel=1e-6)
    # A longitude degree is only cos(47.4) ~ 0.68 as long here, so the box has to grow ~1.5x
    # further in lon to cover the same 10 km. Without that division the halo would be 10 km
    # N-S and 6.8 km E-W, and quietly wrong by more the further north you work.
    cos_lat = math.cos(math.radians(47.4))                 # the box's centre latitude
    assert (e - BOX[2]) == pytest.approx((10.0 / 111.0) / cos_lat, rel=1e-9)
    assert (BOX[0] - w) == pytest.approx(e - BOX[2], rel=1e-9)


def test_halo_of_zero_is_the_identity():
    assert insitu_stations.halo_bbox(BOX, 0.0) == BOX


def test_km_from_bbox_is_zero_inside_and_a_distance_outside():
    assert insitu_stations.km_from_bbox(47.4, -123.1, BOX) == 0.0
    # 0.09 deg north of the top edge is ~10 km.
    assert insitu_stations.km_from_bbox(47.59, -123.1, BOX) == pytest.approx(10.0, rel=0.02)
    # Diagonal: the corner distance, not either edge's.
    corner = insitu_stations.km_from_bbox(47.59, -123.33, BOX)
    assert corner > insitu_stations.km_from_bbox(47.59, -123.1, BOX)


# --------------------------------------------------------------------------- #
# Station.locate -- the "known to be around, not known to be anywhere" case
# --------------------------------------------------------------------------- #
def _st(**kw):
    base = dict(id="x", title="x", source="ioos", lat=47.4, lon=-123.1,
                start="2020-01-01", end="2024-01-01", mobile=False, bbox=None)
    return Station(**{**base, **kw})


def test_fixed_station_locates_at_its_own_position():
    assert _st().locate(BOX) == (-123.1, 47.4)


def test_narrow_track_locates_at_the_centre_of_the_overlap():
    """A track that only clips a corner of the box IS localized -- to that corner."""
    st = _st(mobile=True, bbox=(-123.30, 47.44, -123.15, 47.60))
    lon, lat = st.locate(BOX)
    assert lon == pytest.approx((-123.15 + -123.2) / 2)     # overlap in lon: -123.2..-123.15
    assert lat == pytest.approx((47.44 + 47.5) / 2)         # overlap in lat: 47.44..47.5


def test_track_engulfing_the_box_is_not_placed_at_all():
    """The regression this whole option type exists for.

    `ioos-gliderdac-gp-276` advertises lon -144.9..-70.9 / lat 39.8..50.5. Its extent centre
    is in Wyoming and its OVERLAP centre is just the middle of the map, so neither is a
    position. None is the only true answer.
    """
    papa = _st(mobile=True, lat=45.17, lon=-107.87, bbox=(-144.87, 39.84, -70.87, 50.51))
    assert papa.locate(BOX) is None
    assert papa.overlap(BOX) == BOX                # it does overlap; it is just not locatable


def test_track_that_misses_the_box_entirely_is_not_placed():
    assert _st(mobile=True, bbox=(-100.0, 10.0, -99.0, 11.0)).locate(BOX) is None


def test_station_with_no_position_is_not_placeable():
    assert not _st(lat=None, lon=None).placeable
    assert _st(lat=None, lon=None).locate(BOX) is None


def test_span_label_trims_to_year_and_month():
    assert _st(start="2015-01-01T20:13:00Z", end=None).span_label() == "2015-01 - ?"


# --------------------------------------------------------------------------- #
# IOOS
# --------------------------------------------------------------------------- #
_SEARCH = {"table": {
    "columnNames": ["Dataset ID", "Title"],
    "rows": [["gov-ndbc-46123", "Twanoh"], ["glider-1", "A glider"],
             ["ghost-station", "In the search, not in the catalogue"]]}}

_ALL = {"table": {
    "columnNames": ["datasetID", "cdm_data_type", "title", "minLongitude", "maxLongitude",
                    "minLatitude", "maxLatitude", "minTime", "maxTime"],
    "rows": [
        ["gov-ndbc-46123", "TimeSeries", "Twanoh", -123.008, -123.008, 47.375, 47.375,
         "2015-01-01T20:13:00Z", "2026-08-27T11:20:00Z"],
        ["glider-1", "TrajectoryProfile", "A glider", -127.0, -122.4, 46.88, 47.98,
         "2022-03-25T15:21:33Z", "2022-11-18T21:52:45Z"],
        ["a-barometer", "TimeSeries", "No temperature here", -123.1, -123.1, 47.4, 47.4,
         "2015-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]]}}


@pytest.fixture
def ioos_seam(monkeypatch):
    """Replace the one network seam; record the URLs it was asked for."""
    urls = []

    def fake_get(url, params=None, **kw):
        urls.append((url, params))
        return json.dumps(_ALL if "allDatasets" in url else _SEARCH)

    monkeypatch.setattr(insitu_ioos, "_get", fake_get)
    return urls


def test_ioos_classifies_trajectories_as_mobile(ioos_seam):
    found = {s.id: s for s in insitu_ioos.stations(BOX, "2020-01-01", "2024-12-31", {})}
    assert found["gov-ndbc-46123"].mobile is False
    assert found["glider-1"].mobile is True


def test_ioos_carries_position_span_and_extent(ioos_seam):
    st = {s.id: s for s in insitu_ioos.stations(BOX, "2020-01-01", "2024-12-31", {})}
    buoy = st["gov-ndbc-46123"]
    assert (buoy.lat, buoy.lon) == (47.375, -123.008)
    assert buoy.span_label() == "2015-01 - 2026-08"
    assert buoy.bbox is None                       # a mooring's bounds are a point
    # A track keeps its extent, which is what lets the map refuse to place it.
    assert st["glider-1"].bbox == (-127.0, 46.88, -122.4, 47.98)


def test_ioos_only_returns_what_the_temperature_search_found(ioos_seam):
    """`allDatasets` lists every dataset on the server; the map must show the buoys the
    acquire path would actually fetch, not every barometer on the coast."""
    ids = {s.id for s in insitu_ioos.stations(BOX, "2020-01-01", "2024-12-31", {})}
    assert "a-barometer" not in ids


def test_ioos_names_stations_it_cannot_place(ioos_seam, caplog):
    """A platform that vanishes between the two endpoints is reported, never silent."""
    with caplog.at_level("WARNING"):
        ids = {s.id for s in insitu_ioos.stations(BOX, "2020-01-01", "2024-12-31", {})}
    assert "ghost-station" not in ids
    assert "ghost-station" in caplog.text


def test_ioos_alldatasets_query_intersects_rather_than_contains(ioos_seam):
    """The containment form reads naturally and drops every trajectory. Pin the operators.

    With the box (-123.2, 47.3, -123.0, 47.5): a glider spanning -127..-122.4 satisfies
    `maxLongitude >= -123.2` and `minLongitude <= -123.0`, and fails `minLongitude >= -123.2`.
    """
    insitu_ioos.stations(BOX, "2020-01-01", "2024-12-31", {})
    url = next(u for u, _ in ioos_seam if "allDatasets" in u)
    assert "maxLongitude%3E=-123.2" in url and "minLongitude%3C=-123.0" in url
    assert "maxLatitude%3E=47.3" in url and "minLatitude%3C=47.5" in url
    assert "minLongitude%3E=" not in url           # the containment form, explicitly absent


def test_ioos_applies_the_station_allow_and_deny_lists(ioos_seam):
    ids = {s.id for s in insitu_ioos.stations(
        BOX, "2020-01-01", "2024-12-31", {"exclude_stations": ["glider-1"]})}
    assert ids == {"gov-ndbc-46123"}


def test_ioos_adds_no_padding_of_its_own(ioos_seam):
    """The halo is applied once, by `discover`. A source that padded again would search an
    area the map then crops away."""
    insitu_ioos.stations(BOX, "2020-01-01", "2024-12-31", {"pad_deg": 5.0})
    _, params = next((u, p) for u, p in ioos_seam if "advanced" in u)
    assert (params["minLon"], params["maxLon"]) == (BOX[0], BOX[2])


# --------------------------------------------------------------------------- #
# Copernicus
# --------------------------------------------------------------------------- #
@pytest.fixture
def cmems_index(monkeypatch, tmp_path):
    monkeypatch.setattr(insitu_cmems, "_catalogue_root", lambda ds, part: "https://x.invalid")
    monkeypatch.setattr(insitu_cmems, "_fetch_index", lambda url: make_index(INDEX_ROWS))
    monkeypatch.setattr(insitu_cmems, "_CATALOG_CACHE", {})
    return {"cache_dir": tmp_path, "dataset_part": "history"}


CM_BOX = (-124.2, 46.85, -124.0, 47.0)


def test_cmems_splits_fixed_from_mobile_by_declared_class(cmems_index):
    found = {s.id: s for s in insitu_cmems.stations(
        CM_BOX, "2015-01-01", "2021-01-01", cmems_index)}
    assert found["MOORED"].mobile is False         # MO
    assert found["GLIDER"].mobile is True          # GL
    assert found["SHIP"].mobile is True            # TS


def test_cmems_demotes_a_wide_bounded_mooring_to_mobile(cmems_index):
    """`WIDE` is declared `MO` and advertises a 10 x 10 degree box.

    The real catalogue does this: `GL_TS_MO_31261` claims half the planet while sitting off
    Brazil. `within()` catches it only once the file is open, which this path never does -- so
    a labelled mooring would be asserted where there is none. Demoted, and its extent kept.
    """
    st = {s.id: s for s in insitu_cmems.stations(
        CM_BOX, "1987-01-01", "1988-01-01", cmems_index)}["WIDE"]
    assert st.mobile is True
    assert st.bbox == (-130.0, 40.0, -120.0, 50.0)
    assert st.locate(CM_BOX) is None               # and so it is not placed


def test_cmems_ignores_the_configured_platform_types(cmems_index):
    """`fetch_aoi` keeps only `platform_types`; the MAP is about what exists nearby."""
    ids = {s.id for s in insitu_cmems.stations(
        CM_BOX, "2015-01-01", "2021-01-01", {**cmems_index, "platform_types": ["MO"]})}
    assert {"GLIDER", "SHIP"} <= ids


def test_cmems_drops_rows_without_temperature_or_outside_the_window(cmems_index):
    ids = {s.id for s in insitu_cmems.stations(
        CM_BOX, "2015-01-01", "2021-01-01", cmems_index)}
    assert "NOTEMP" not in ids                     # carries PSAL/VHM0 only
    assert "ELSEWHEN" not in ids                   # 1950-1955


def test_cmems_reports_the_platforms_record_span(cmems_index):
    st = {s.id: s for s in insitu_cmems.stations(
        CM_BOX, "2015-01-01", "2021-01-01", cmems_index)}["MOORED"]
    # The WHOLE record, not the window it was filtered against -- that is what makes the
    # label worth reading when choosing a time range.
    assert st.start == "2004-09-08" and st.end == "2026-07-31"


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def _csv_file(tmp_path):
    rows = ["station_id,time,latitude,longitude,value"]
    for h in range(6):                             # a mooring: one position
        rows.append(f"pier,2020-06-0{h + 1}T00:00:00Z,47.40,-123.10,12.{h}")
    for h in range(6):                             # a drifter: kilometres of drift
        rows.append(f"drifter,2020-06-0{h + 1}T00:00:00Z,{47.40 + h * 0.01},-123.10,12.{h}")
    rows.append("faraway,2020-06-01T00:00:00Z,10.0,-80.0,25.0")
    rows.append("longago,1975-06-01T00:00:00Z,47.41,-123.11,11.0")
    p = tmp_path / "obs.csv"
    p.write_text("\n".join(rows) + "\n")
    return p


def _csv_cfg(tmp_path, **kw):
    return {"path": str(_csv_file(tmp_path)), "resolution_m": 100.0, **kw}


def test_csv_splits_by_measured_drift(tmp_path):
    found = {s.id: s for s in insitu_csv.stations(
        BOX, "2020-01-01", "2020-12-31", _csv_cfg(tmp_path))}
    assert found["pier"].mobile is False
    assert found["drifter"].mobile is True         # ~5.5 km, far past the 100 m cell


def test_csv_drift_threshold_follows_the_grid_cell(tmp_path):
    """The same threshold the two in-situ products partition on, so the map agrees with the
    cube about which platform lands where."""
    cfg = _csv_cfg(tmp_path, resolution_m=100_000.0)      # a 100 km cell: nothing drifts
    found = {s.id: s for s in insitu_csv.stations(BOX, "2020-01-01", "2020-12-31", cfg)}
    assert found["drifter"].mobile is False


def test_csv_keeps_only_platforms_in_the_box_and_the_window(tmp_path):
    ids = {s.id for s in insitu_csv.stations(
        BOX, "2020-01-01", "2020-12-31", _csv_cfg(tmp_path))}
    assert ids == {"pier", "drifter"}              # not `faraway`, not `longago`


def test_csv_reports_real_first_and_last_observations(tmp_path):
    st = {s.id: s for s in insitu_csv.stations(
        BOX, "2020-01-01", "2020-12-31", _csv_cfg(tmp_path))}["pier"]
    assert st.start == "2020-06-01" and st.end == "2020-06-06"


def test_csv_without_a_path_is_silent_rather_than_an_error(tmp_path):
    """`csv` in the source list with no `path` is the normal state of a config still being
    written; the other sources' stations must still reach the map."""
    assert insitu_csv.stations(BOX, "2020-01-01", "2020-12-31", {}) == []


def test_csv_says_so_when_a_path_matches_nothing(tmp_path):
    """A mistyped path and an empty ocean must never look the same."""
    with pytest.raises(ValueError, match="matched no CSV files"):
        insitu_csv.stations(BOX, "2020-01-01", "2020-12-31",
                            {"path": str(tmp_path / "nope-*.csv")})


# --------------------------------------------------------------------------- #
# discover() -- the aggregation across sources
# --------------------------------------------------------------------------- #
def _project(tmp_path, **insitu_opts):
    from coastal_sst_data.config import parse_config
    return parse_config({
        "name": "p", "output_dir": str(tmp_path),
        "time": {"start_date": "2020-01-01", "end_date": "2024-12-31"},
        "auth": {"copernicus": {"auth_strategy": "netrc"}},
        "products": ({"insitu": insitu_opts} if insitu_opts else {"bathymetry": None}),
        "regions": [{"name": "r", "areas": [
            {"name": "a1", "center_lat": 47.4, "center_lon": -123.1,
             "buffer_ns_km": 6, "buffer_ew_km": 5}]}],
    })


def _grid(project):
    from coastal_sst_data.grid import project_grids
    return project_grids(project)["a1"]


def test_discover_applies_the_halo_once(tmp_path, monkeypatch, ioos_seam):
    """The halo is `discover`'s job. A source that padded again would search an area the map
    then crops away, and the two paddings would compound silently."""
    proj = _project(tmp_path, sources=["ioos"])
    g = _grid(proj)
    insitu_stations.discover(proj, g, halo_km=10.0)

    _, params = next((u, p) for u, p in ioos_seam if "advanced" in u)
    expected = insitu_stations.halo_bbox(g.search_bbox, 10.0)
    assert (params["minLon"], params["maxLon"]) == (expected[0], expected[2])


def test_discover_works_without_insitu_being_a_selected_product(tmp_path, ioos_seam):
    """The interesting case: someone is still DECIDING where the AoIs go, so in-situ is not
    configured yet. Falling back to the defaults is what makes the map useful then."""
    found = insitu_stations.discover(_project(tmp_path), _grid(_project(tmp_path)))
    assert {s.id for s in found} == {"gov-ndbc-46123", "glider-1"}


def test_discover_survives_one_source_failing(tmp_path, monkeypatch, ioos_seam):
    """A map with two of three sources beats a traceback."""
    def boom(*a, **kw):
        raise RuntimeError("no Copernicus credential")
    monkeypatch.setattr(insitu_cmems, "stations", boom)
    monkeypatch.setitem(insitu_stations.SOURCES, "marineinsitu", boom)

    proj = _project(tmp_path, sources=["ioos", "marineinsitu"])
    found = insitu_stations.discover(proj, _grid(proj))
    assert {s.id for s in found} == {"gov-ndbc-46123", "glider-1"}   # IOOS still landed


def test_discover_sorts_fixed_before_mobile(tmp_path, ioos_seam):
    proj = _project(tmp_path, sources=["ioos"])
    found = insitu_stations.discover(proj, _grid(proj))
    assert [s.mobile for s in found] == sorted(s.mobile for s in found)


def test_discover_honours_a_region_override(tmp_path, ioos_seam):
    """The map reads the SAME resolved config acquisition would, region overrides included."""
    from coastal_sst_data.config import parse_config
    proj = parse_config({
        "name": "p", "output_dir": str(tmp_path),
        "time": {"start_date": "2020-01-01", "end_date": "2024-12-31"},
        "products": {"insitu": {"sources": ["ioos"]}},
        "regions": [{"name": "r",
                     "sources": {"insitu": {"exclude_stations": ["glider-1"]}},
                     "areas": [{"name": "a1", "center_lat": 47.4, "center_lon": -123.1,
                                "buffer_ns_km": 6, "buffer_ew_km": 5}]}],
    })
    found = insitu_stations.discover(proj, _grid(proj))
    assert {s.id for s in found} == {"gov-ndbc-46123"}


def test_discover_raises_when_every_source_fails(tmp_path, monkeypatch):
    """"We do not know" must not be drawn as "no buoys here".

    A partial failure degrades (see above); a TOTAL one raises, so the caller writes no figure
    rather than an empty coastline that reads as an answer.
    """
    def boom(*a, **kw):
        raise RuntimeError("ERDDAP is down")
    monkeypatch.setitem(insitu_stations.SOURCES, "ioos", boom)

    proj = _project(tmp_path, sources=["ioos"])
    with pytest.raises(RuntimeError, match="every in-situ source failed"):
        insitu_stations.discover(proj, _grid(proj))


def test_discover_returns_empty_when_a_source_genuinely_finds_nothing(tmp_path, monkeypatch):
    """The other half of the same distinction: a source that ANSWERS "none" is not a failure."""
    monkeypatch.setitem(insitu_stations.SOURCES, "ioos", lambda *a, **kw: [])
    proj = _project(tmp_path, sources=["ioos"])
    assert insitu_stations.discover(proj, _grid(proj)) == []
