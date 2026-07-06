"""AoI map plotting. matplotlib is optional, so these skip if it's absent; they
run headless (Agg) and assert WHICH files get written, not pixel content."""

import pytest

pytest.importorskip("matplotlib")   # plotting is an optional capability

from coastal_sst_data.config import parse_config
from coastal_sst_data import plot


def _two_region_project():
    """A project with two regions (2 AoIs + 1 AoI) for map tests."""
    return parse_config({
        "name": "plots", "output_dir": "out",
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-30"},
        "products": {"bathymetry": None},
        "regions": [
            {"name": "pnw estuaries", "areas": [
                {"name": "tillamook_bay", "center_lat": 45.52, "center_lon": -123.925,
                 "buffer_ns_km": 25, "buffer_ew_km": 15},
                {"name": "coos_bay", "center_lat": 43.37, "center_lon": -124.22,
                 "buffer_ns_km": 10, "buffer_ew_km": 10}]},
            {"name": "puget_sound", "areas": [
                {"name": "padilla_bay", "center_lat": 48.45, "center_lon": -122.58,
                 "buffer_ns_km": 10, "buffer_ew_km": 10}]},
        ],
    })


def test_writes_overview_and_one_map_per_region(tmp_path):
    proj = _two_region_project()
    paths = plot.plot_project_aois(proj, out_dir=tmp_path)
    names = {p.name for p in paths}
    assert names == {"aoi_overview.png",
                     "aoi_region_pnw_estuaries.png",     # space -> underscore
                     "aoi_region_puget_sound.png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_overview_only(tmp_path):
    paths = plot.plot_project_aois(_two_region_project(), out_dir=tmp_path,
                                   per_region=False)
    assert [p.name for p in paths] == ["aoi_overview.png"]


def test_per_region_only(tmp_path):
    paths = plot.plot_project_aois(_two_region_project(), out_dir=tmp_path,
                                   overview=False)
    assert all(p.name.startswith("aoi_region_") for p in paths)
    assert len(paths) == 2


def test_default_out_dir_is_output_dir_figures(tmp_path):
    proj = parse_config({
        "name": "p", "output_dir": str(tmp_path / "data"),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-30"},
        "products": {"bathymetry": None},
        "regions": [{"name": "r", "areas": [
            {"name": "a1", "center_lat": 45.0, "center_lon": -123.0,
             "buffer_ns_km": 8, "buffer_ew_km": 8}]}],
    })
    paths = plot.plot_project_aois(proj)
    assert all((tmp_path / "data" / "figures") == p.parent for p in paths)


def test_grids_filter_drops_ungridded_aois(tmp_path):
    """AoIs absent from `grids` are omitted; a region left empty is skipped."""
    proj = _two_region_project()
    # Only the pnw AoIs gridded successfully; puget_sound's AoI is missing.
    grids = {"tillamook_bay": object(), "coos_bay": object()}
    paths = plot.plot_project_aois(proj, grids=grids, out_dir=tmp_path)
    names = {p.name for p in paths}
    assert "aoi_region_pnw_estuaries.png" in names
    assert "aoi_region_puget_sound.png" not in names    # no drawable AoIs -> skipped


def test_no_drawable_aois_returns_empty(tmp_path):
    """If nothing is drawable, no files are written and an empty list is returned."""
    proj = _two_region_project()
    paths = plot.plot_project_aois(proj, grids={}, out_dir=tmp_path)   # grid nothing
    assert paths == []
    assert not any(tmp_path.iterdir())


def test_slug_sanitizes_names():
    assert plot._slug("puget sound") == "puget_sound"
    assert plot._slug("a/b c") == "a_b_c"
    assert plot._slug("  ") == "unnamed"
