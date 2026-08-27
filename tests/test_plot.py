"""AoI map plotting. matplotlib is optional, so these skip if it's absent; they
run headless (Agg) and assert WHICH files get written, not pixel content."""

import pytest

pytest.importorskip("matplotlib")   # plotting is an optional capability

from coastal_sst_data.config import parse_config
from coastal_sst_data import plot
from pathlib import Path

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

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])

# --------------------------------------------------------------------------- #
# The in-situ context maps (`grids --plot --insitu`)
# --------------------------------------------------------------------------- #
def _one_aoi_project(tmp_path):
    return parse_config({
        "name": "p", "output_dir": str(tmp_path),
        "time": {"start_date": "2020-01-01", "end_date": "2024-12-31"},
        "products": {"bathymetry": None},
        "regions": [{"name": "r", "areas": [
            {"name": "twanoh", "center_lat": 47.375, "center_lon": -123.008,
             "buffer_ns_km": 6, "buffer_ew_km": 5}]}],
    })


def _station(**kw):
    from coastal_sst_data.processes.insitu_stations import Station
    base = dict(id="s", title="s", source="ioos", lat=47.375, lon=-123.008,
                start="2015-01-01", end="2026-08-01", mobile=False, bbox=None)
    return Station(**{**base, **kw})


def _fake_stations():
    return [
        _station(id="in-aoi"),
        _station(id="in-halo", lat=47.47),                     # ~4.5 km outside the box
        _station(id="near-track", mobile=True,
                 bbox=(-123.06, 47.42, -123.04, 47.46)),       # small: placeable
        _station(id="ocean-glider", mobile=True, lat=45.2, lon=-107.9,
                 bbox=(-144.9, 39.8, -70.9, 50.5)),            # engulfs the box: not placed
    ]


def _patched(monkeypatch, stations):
    """Discovery replaced wholesale -- these tests are about the FIGURE, not the network."""
    monkeypatch.setattr("coastal_sst_data.processes.insitu_stations.discover",
                        lambda project, g, **kw: list(stations))


def test_insitu_is_off_by_default(tmp_path, monkeypatch):
    """`--plot` alone must stay offline: discovery is never even called."""
    def boom(*a, **kw):
        raise AssertionError("discovery ran without --insitu")
    monkeypatch.setattr("coastal_sst_data.processes.insitu_stations.discover", boom)
    names = {p.name for p in plot.plot_project_aois(_one_aoi_project(tmp_path),
                                                    out_dir=tmp_path)}
    assert not any(n.startswith("aoi_insitu_") for n in names)


def test_insitu_writes_a_map_and_a_sidecar_csv_per_aoi(tmp_path, monkeypatch):
    _patched(monkeypatch, _fake_stations())
    paths = plot.plot_project_aois(_one_aoi_project(tmp_path), out_dir=tmp_path,
                                   overview=False, per_region=False, insitu=True)
    assert {p.name for p in paths} == {"aoi_insitu_twanoh.png", "aoi_insitu_twanoh.csv"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_insitu_csv_never_places_a_track_that_engulfs_the_box(tmp_path, monkeypatch):
    """The regression: an open-ocean glider once came out as `inside_aoi=1, km=0.00`.

    A blank position is a fact ("we know it is around, not where"); a zero is a claim.
    """
    import csv as _csv

    _patched(monkeypatch, _fake_stations())
    plot.plot_project_aois(_one_aoi_project(tmp_path), out_dir=tmp_path,
                           overview=False, per_region=False, insitu=True)
    rows = {r["id"]: r for r in
            _csv.DictReader((tmp_path / "aoi_insitu_twanoh.csv").read_text().splitlines())}

    assert rows["ocean-glider"]["lat"] == "" and rows["ocean-glider"]["km_from_aoi"] == ""
    assert rows["ocean-glider"]["inside_aoi"] == ""
    assert rows["ocean-glider"]["reported_bbox"]        # ... but the claim is still recorded

    assert rows["in-aoi"]["inside_aoi"] == "1" and rows["in-aoi"]["km_from_aoi"] == "0.00"
    # The number the map exists to produce: how far the box would have to move.
    assert rows["in-halo"]["inside_aoi"] == "0"
    assert float(rows["in-halo"]["km_from_aoi"]) == pytest.approx(4.6, abs=0.2)
    # A narrow track IS placed -- to the centre of where its extent meets the searched box.
    assert rows["near-track"]["lat"] and rows["near-track"]["mobile"] == "1"


def test_insitu_csv_lists_stations_the_label_budget_left_off(tmp_path, monkeypatch):
    """Labels are capped so a 200-station box stays readable; the CSV never is."""
    many = [_station(id=f"s{i}", lat=47.36 + i * 0.001) for i in range(40)]
    _patched(monkeypatch, many)
    plot.plot_project_aois(_one_aoi_project(tmp_path), out_dir=tmp_path, overview=False,
                           per_region=False, insitu=True, insitu_max_labels=5)
    lines = (tmp_path / "aoi_insitu_twanoh.csv").read_text().strip().splitlines()
    assert len(lines) == 41                       # header + every one of the 40


def test_insitu_halo_widens_the_frame_past_the_pad_factor(tmp_path, monkeypatch):
    """The extent is the UNION of the two knobs, so a big halo is not searched then cropped."""
    _patched(monkeypatch, [])
    proj = _one_aoi_project(tmp_path)
    area = proj.regions[0].areas[0]
    bb = plot._aoi_bbox(area)

    from coastal_sst_data.processes.insitu_stations import halo_bbox
    wide = halo_bbox(bb, 200.0)
    padded = plot._scale_bbox(bb, 2.5)
    frame = plot._pad_extent([padded, wide], pad_frac=0.02, min_pad_deg=0.005)
    assert frame[0] < wide[0] and frame[2] > wide[2]        # the 200 km halo fits inside
    assert frame[0] < padded[0]                              # ... and it is what set the width

    # A 1 km halo must not SHRINK the frame below the pad factor.
    narrow = plot._pad_extent([padded, halo_bbox(bb, 1.0)], pad_frac=0.02, min_pad_deg=0.005)
    assert narrow[0] <= padded[0] and narrow[2] >= padded[2]


def test_insitu_survives_a_source_that_fails(tmp_path, monkeypatch):
    """One AoI's discovery failing must not cost the other figures."""
    def boom(project, g, **kw):
        raise RuntimeError("ERDDAP is down")
    monkeypatch.setattr("coastal_sst_data.processes.insitu_stations.discover", boom)
    paths = plot.plot_project_aois(_one_aoi_project(tmp_path), out_dir=tmp_path, insitu=True)
    assert {p.name for p in paths} == {"aoi_overview.png", "aoi_region_r.png"}


def test_stack_labels_separates_colocated_rows():
    """Twanoh has an NDBC buoy and a MAPCO2 mooring 0.005 deg apart; one smear is not a map."""
    taken = []
    assert plot._stack_labels(taken, 47.40, 0.01) == 47.40
    # Pushed down until it clears EVERY taken row by a full dy, not just the last one:
    # 47.391 is still within 0.01 of 47.40, so it keeps going to 47.381.
    assert plot._stack_labels(taken, 47.401, 0.01) == pytest.approx(47.381)
    assert plot._stack_labels(taken, 47.60, 0.01) == 47.60                   # far off: kept
