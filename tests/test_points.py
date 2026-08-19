"""points.py -- reading a user's points file, and placing each point on ONE AoI grid.

Every test here guards a failure that produces a plausible-looking table rather than an
error: a coordinate column read in the wrong units, a point placed in the wrong AoI, or a
point silently dropped.
"""

import numpy as np
import pandas as pd
import pytest

from coastal_sst_data import points
from coastal_sst_data.config import AreaOfInterest, GridSpec
from coastal_sst_data.grid import compute_aoi_grid

from .conftest import pixel_lonlat


def _csv(tmp_path, rows, name="p.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# read_points
# --------------------------------------------------------------------------- #
def test_column_aliases_resolve(tmp_path):
    """`station/latitude/longitude` and `id/lat/lon` must both be readable."""
    a = points.read_points(_csv(tmp_path, [{"station": "s1", "latitude": 45.5,
                                            "longitude": -123.9}], "a.csv"))
    b = points.read_points(_csv(tmp_path, [{"id": "s1", "lat": 45.5, "lon": -123.9}], "b.csv"))
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == list(points.OUT_COLUMNS)


def test_explicit_column_map_wins(tmp_path):
    p = _csv(tmp_path, [{"name": "s1", "deg_n": 45.5, "deg_e": -123.9, "lat": 0.0}])
    df = points.read_points(p, {"lat": "deg_n", "lon": "deg_e", "point_id": "name"})
    assert float(df["lat"].iloc[0]) == 45.5


def test_unknown_column_map_field_is_rejected(tmp_path):
    p = _csv(tmp_path, [{"lat": 45.5, "lon": -123.9}])
    with pytest.raises(ValueError, match="unknown field"):
        points.read_points(p, {"latitude": "lat"})


def test_ambiguous_columns_are_rejected(tmp_path):
    """Both `lat` and `latitude` present: we cannot know which one is live.

    Picking by priority would extract at the wrong coordinates for whoever's second column
    was the real one, and nothing downstream could tell.
    """
    p = _csv(tmp_path, [{"lat": 45.5, "latitude": 46.5, "lon": -123.9}])
    with pytest.raises(ValueError, match="more than one column"):
        points.read_points(p)


def test_projected_metres_are_rejected(tmp_path):
    """An x/y UTM file must not be read as degrees.

    It would not raise anywhere: the coordinates simply land in the ocean, match no AoI, and
    produce an empty table that reads like a configuration mistake.
    """
    p = _csv(tmp_path, [{"id": "s1", "x": 512345.0, "y": 5041234.0}])
    with pytest.raises(ValueError, match="outside WGS84 range"):
        points.read_points(p)


def test_swapped_lat_lon_is_rejected(tmp_path):
    p = _csv(tmp_path, [{"id": "s1", "lat": -123.9, "lon": 45.5}])
    with pytest.raises(ValueError, match="outside WGS84 range"):
        points.read_points(p)


def test_missing_id_is_synthesised_with_a_warning(tmp_path, caplog):
    """Ids that are really row numbers cannot be joined back -- that must be visible."""
    p = _csv(tmp_path, [{"lat": 45.5, "lon": -123.9}, {"lat": 45.6, "lon": -123.8}],
             "sites.csv")
    with caplog.at_level("WARNING"):
        df = points.read_points(p)
    assert list(df["point_id"]) == ["sites_0001", "sites_0002"]
    assert "no id column" in caplog.text


def test_duplicate_ids_are_rejected(tmp_path):
    p = _csv(tmp_path, [{"id": "s1", "lat": 45.5, "lon": -123.9},
                        {"id": "s1", "lat": 45.6, "lon": -123.8}])
    with pytest.raises(ValueError, match="duplicate point id"):
        points.read_points(p)


def test_colocated_points_warn_but_are_kept(tmp_path, caplog):
    p = _csv(tmp_path, [{"id": "s1", "lat": 45.5, "lon": -123.9},
                        {"id": "s2", "lat": 45.5, "lon": -123.9}])
    with caplog.at_level("WARNING"):
        df = points.read_points(p)
    assert len(df) == 2
    assert "share coordinates" in caplog.text


def test_non_numeric_rows_are_dropped_with_a_count(tmp_path, caplog):
    p = _csv(tmp_path, [{"id": "s1", "lat": 45.5, "lon": -123.9},
                        {"id": "s2", "lat": "", "lon": -123.8}])
    with caplog.at_level("WARNING"):
        df = points.read_points(p)
    assert list(df["point_id"]) == ["s1"]
    assert "dropped 1 row" in caplog.text


def test_missing_coordinate_column_names_what_it_found(tmp_path):
    p = _csv(tmp_path, [{"id": "s1", "northing": 45.5}])
    with pytest.raises(ValueError, match="no column found for lat, lon"):
        points.read_points(p)


def test_extra_columns_are_dropped(tmp_path):
    """The output schema is fixed however wide the input is; join back on point_id."""
    p = _csv(tmp_path, [{"id": "s1", "lat": 45.5, "lon": -123.9, "depth": 12, "note": "x"}])
    assert list(points.read_points(p).columns) == list(points.OUT_COLUMNS)


# --------------------------------------------------------------------------- #
# assign_aois
# --------------------------------------------------------------------------- #
def _grid(name, lat, lon, km=8.0):
    return compute_aoi_grid(
        AreaOfInterest(name=name, center_lat=lat, center_lon=lon,
                       buffer_ns_km=km, buffer_ew_km=km),
        GridSpec())


def test_point_is_assigned_to_the_grid_that_contains_it(aoi_grid):
    lon, lat = pixel_lonlat(aoi_grid, 20, 20)
    pts = pd.DataFrame({"point_id": ["s1"], "lat": [lat], "lon": [lon]})
    out = points.assign_aois(pts, {aoi_grid.name: aoi_grid})
    assert list(out["aoi"]) == [aoi_grid.name]
    assert (int(out["row"].iloc[0]), int(out["col"].iloc[0])) == (20, 20)


def test_point_outside_every_aoi_is_dropped_with_a_warning(aoi_grid, caplog):
    pts = pd.DataFrame({"point_id": ["far"], "lat": [0.0], "lon": [0.0]})
    with caplog.at_level("WARNING"):
        out = points.assign_aois(pts, {aoi_grid.name: aoi_grid})
    assert out.empty
    assert "outside every AoI" in caplog.text and "far" in caplog.text


def test_overlapping_aois_tie_break_on_nearest_centre():
    """Two grids contain the point; it gets exactly one, and it is the near one."""
    from coastal_sst_data.processes.insitu import station_pixels

    west = _grid("west", 45.52, -123.99)
    east = _grid("east", 45.52, -123.90)
    for lon, expect in [(-123.93, "east"), (-123.96, "west")]:
        pts = pd.DataFrame({"point_id": ["s1"], "lat": [45.52], "lon": [lon]})
        # sanity: it is inside BOTH, so this is a tie-break and not a containment test
        assert all(station_pixels([lon], [45.52], g)[0]["inside"] for g in (west, east))
        out = points.assign_aois(pts, {"west": west, "east": east})
        assert len(out) == 1
        assert out["aoi"].iloc[0] == expect


def test_tie_break_is_deterministic_under_grid_order():
    """Dict insertion order must not decide which AoI a point lands in."""
    west, east = _grid("west", 45.52, -123.99), _grid("east", 45.52, -123.90)
    pts = pd.DataFrame({"point_id": ["s1"], "lat": [45.52], "lon": [-123.95]})
    a = points.assign_aois(pts, {"west": west, "east": east})["aoi"].iloc[0]
    b = points.assign_aois(pts, {"east": east, "west": west})["aoi"].iloc[0]
    assert a == b


def test_projected_position_is_the_point_not_its_pixel_centre(aoi_grid):
    """px/py must be the point itself -- the extraction disc is centred on it."""
    lon, lat = pixel_lonlat(aoi_grid, 20, 20)
    off_lon = lon + 0.0003                          # a fraction of a pixel east
    pts = pd.DataFrame({"point_id": ["s1"], "lat": [lat], "lon": [off_lon]})
    out = points.assign_aois(pts, {aoi_grid.name: aoi_grid})
    xs, _ = aoi_grid.xy_centers()
    assert out["px"].iloc[0] != pytest.approx(xs[int(out["col"].iloc[0])], abs=1e-9)


def test_edge_point_is_flagged(aoi_grid, caplog):
    lon, lat = pixel_lonlat(aoi_grid, 1, 1)
    pts = pd.DataFrame({"point_id": ["edge"], "lat": [lat], "lon": [lon]})
    out = points.assign_aois(pts, {aoi_grid.name: aoi_grid})
    with caplog.at_level("WARNING"):
        points.flag_edge_points(out, {aoi_grid.name: aoi_grid}, 500.0)
    assert "clipped" in caplog.text and "edge" in caplog.text


def test_interior_point_is_not_flagged(aoi_grid, caplog):
    lon, lat = pixel_lonlat(aoi_grid, 80, 80)
    pts = pd.DataFrame({"point_id": ["mid"], "lat": [lat], "lon": [lon]})
    out = points.assign_aois(pts, {aoi_grid.name: aoi_grid})
    with caplog.at_level("WARNING"):
        points.flag_edge_points(out, {aoi_grid.name: aoi_grid}, 500.0)
    assert "clipped" not in caplog.text


def test_no_snapping_to_water(aoi_grid):
    """A point must land in the cell it is in, not the nearest water cell.

    `insitu.station_pixels` can snap; this stage must never ask it to. A snapped point moves
    up to hundreds of metres with nothing in the output recording it.
    """
    lon, lat = pixel_lonlat(aoi_grid, 20, 160)      # deep in the "land" half
    pts = pd.DataFrame({"point_id": ["s1"], "lat": [lat], "lon": [lon]})
    out = points.assign_aois(pts, {aoi_grid.name: aoi_grid})
    assert int(out["col"].iloc[0]) == 160


def test_no_grids_is_an_error():
    pts = pd.DataFrame({"point_id": ["s1"], "lat": [45.5], "lon": [-123.9]})
    with pytest.raises(ValueError, match="no AoI grids"):
        points.assign_aois(pts, {})


def test_grid_center_is_the_middle_of_the_grid(aoi_grid):
    cx, cy = points.grid_center(aoi_grid)
    xs, ys = aoi_grid.xy_centers()
    assert cx == pytest.approx(float(np.mean(xs)), abs=aoi_grid.resolution_m)
    assert cy == pytest.approx(float(np.mean(ys)), abs=aoi_grid.resolution_m)
