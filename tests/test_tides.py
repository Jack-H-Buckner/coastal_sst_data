
import sys
import types

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from coastal_sst_data.config import load_config, parse_config, AreaOfInterest, GridSpec
from coastal_sst_data.processes import tides
from coastal_sst_data import grid


EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


def _ds(eff):
    """The tide settings ONE AoI runs with.

    `eff["ds"]` is keyed by AoI: CO-OPS gauges exist only in U.S. waters, so which tide
    source serves an AoI is region-dependent. The configs below set no region override, so
    every AoI resolves alike -- take any. (The per-region cases are tested separately.)
    """
    return next(iter(eff["ds"].values()))




# --------------------------------------------------------------------------- #
# _build_eff: config -> acquisition params
# --------------------------------------------------------------------------- #
def test_build_eff_maps_example_config():
    """The example config maps to the expected tides acquisition parameters."""
    eff = tides._build_eff(load_config(EXAMPLE))
    assert _ds(eff)["interval"] == "h"
    assert _ds(eff)["stations"] == {}                     # no overrides in example
    assert _ds(eff)["warn_distance_km"] == tides.DEFAULT_WARN_KM
    assert eff["fmt"] == "netcdf"
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["tide_root"] == Path("path/to/data") / "TIDE"
    assert "aois" not in eff


def test_build_eff_requires_tides_selected(base_project):
    """Calling the adapter when tides isn't a selected product is an error."""
    base_project["products"] = {"bathymetry": None}        # drop tides
    with pytest.raises(ValueError, match="tides is not a selected product"):
        tides._build_eff(parse_config(base_project))


def test_build_eff_defaults_when_options_omitted(base_project):
    """A bare `tides:` (no options) falls back to product defaults."""
    base_project["products"]["tides"] = None               # bare -> default options
    eff = tides._build_eff(parse_config(base_project))
    assert _ds(eff)["interval"] == tides.DEFAULT_INTERVAL
    assert _ds(eff)["stations"] == {}
    assert eff["overwrite"] is False


def test_build_eff_reads_station_overrides(base_project):
    """Per-AoI gauge overrides come through the `stations` mapping."""
    base_project["products"]["tides"] = {
        "interval": "6",
        "stations": {"a1": "9445133"},
        "warn_distance_km": 50,
        "output_format": "geotiff",
        "overwrite": True,
    }
    eff = tides._build_eff(parse_config(base_project))
    assert _ds(eff)["interval"] == "6"
    assert _ds(eff)["stations"] == {"a1": "9445133"}
    assert _ds(eff)["warn_distance_km"] == 50.0
    # tide is always NetCDF, but the option is still carried through
    assert eff["fmt"] == "geotiff"
    assert eff["overwrite"] is True


# --------------------------------------------------------------------------- #
# Geometry + gauge selection (offline)
# --------------------------------------------------------------------------- #
def test_grid_centroid_lonlat(aoi_grid):
    lon, lat = tides.grid_centroid_lonlat(aoi_grid)
    w, s, e, n = aoi_grid.search_bbox
    assert lon == pytest.approx((w + e) / 2.0)
    assert lat == pytest.approx((s + n) / 2.0)


def test_haversine_km():
    # ~1 deg of latitude is ~111 km
    assert tides.haversine_km(-123.0, 45.0, -123.0, 46.0) == pytest.approx(111.19, abs=0.5)
    assert tides.haversine_km(-123.0, 45.0, -123.0, 45.0) == pytest.approx(0.0)


def test_nearest_station_picks_closest():
    stations = [
        {"id": "A", "name": "far", "lon": -120.0, "lat": 40.0},
        {"id": "B", "name": "near", "lon": -123.9, "lat": 45.5},
        {"id": "C", "name": "mid", "lon": -122.0, "lat": 47.0},
    ]
    station, dist = tides.nearest_station(-123.925, 45.52, stations)
    assert station["id"] == "B"
    assert dist == pytest.approx(0.0, abs=5.0)


# --------------------------------------------------------------------------- #
# Stacked sources (D10): coops + the global model, no fallback
# --------------------------------------------------------------------------- #
def test_build_eff_source_defaults(base_project):
    """A bare `tides:` stacks both built-in sources; the model knobs default too."""
    base_project["products"]["tides"] = None
    eff = tides._build_eff(parse_config(base_project))
    assert _ds(eff)["sources"] == tides.DEFAULT_SOURCES == ["coops", "eo_tides"]
    assert _ds(eff)["max_distance_km"] == tides.DEFAULT_MAX_KM
    assert _ds(eff)["model"] == tides.DEFAULT_MODEL == "EOT20"
    assert _ds(eff)["model_directory"] is None


def test_build_eff_reads_source_options(base_project):
    """The stacked `sources` list + model options come through the ds config."""
    base_project["products"]["tides"] = {
        "sources": ["eo_tides"], "max_distance_km": 200, "model": "FES2022",
        "model_directory": "/data/tide_models",
    }
    eff = tides._build_eff(parse_config(base_project))
    assert _ds(eff)["sources"] == ["eo_tides"]
    assert _ds(eff)["max_distance_km"] == 200.0
    assert _ds(eff)["model"] == "FES2022"
    assert _ds(eff)["model_directory"] == "/data/tide_models"


def test_sources_region_override(base_project):
    """A region's `sources.tides.sources` overrides the project default stack."""
    base_project["products"]["tides"] = None
    base_project["regions"][0]["sources"]["tides"] = {"sources": ["eo_tides"]}
    eff = tides._build_eff(parse_config(base_project))
    assert eff["ds"]["a1"]["sources"] == ["eo_tides"]


def test_sources_default_when_unset(base_project):
    """With no override, the project default stack is used."""
    base_project["products"]["tides"] = None
    eff = tides._build_eff(parse_config(base_project))
    assert eff["ds"]["a1"]["sources"] == ["coops", "eo_tides"]


def test_unknown_source_is_rejected_at_config_load(base_project):
    """A typo'd source in the stacked list fails at config validation, before any work."""
    base_project["products"]["tides"] = {"sources": ["bogus"]}
    with pytest.raises(Exception, match="unknown source"):
        parse_config(base_project)


def _install_fake_eo_tides(monkeypatch):
    """Inject a fake `eo_tides.model.model_tides` so predict_global runs offline.

    Returns long-format output (multi-indexed by time/x/y) matching the real API:
    a linear ramp of heights so the reduction to a time-only Series is checkable.
    """
    def fake_model_tides(x, y, time, model, directory, output_units):
        idx = pd.MultiIndex.from_product([pd.DatetimeIndex(time), [x], [y]],
                                         names=["time", "x", "y"])
        return pd.DataFrame(
            {"tide_height": np.arange(len(time), dtype="float64"), "tide_model": model},
            index=idx)

    pkg = types.ModuleType("eo_tides")
    mod = types.ModuleType("eo_tides.model")
    mod.model_tides = fake_model_tides
    monkeypatch.setitem(sys.modules, "eo_tides", pkg)
    monkeypatch.setitem(sys.modules, "eo_tides.model", mod)


def test_predict_global_reduces_long_format(monkeypatch):
    """predict_global samples the model and flattens (time,x,y) -> a time Series."""
    _install_fake_eo_tides(monkeypatch)
    s = tides.predict_global(-123.0, 45.0, "2026-06-01", "2026-06-01", "h", "EOT20", None)
    # start..end+1day hourly, inclusive left -> 24 hourly steps for one day.
    assert len(s) == 24
    assert s.name == "tide"
    assert s.dtype == np.float32
    assert list(s.values[:3]) == [0.0, 1.0, 2.0]


def test_predict_global_raises_helpful_error_without_eo_tides(monkeypatch):
    """Missing eo-tides gives an actionable install message, not an ImportError."""
    monkeypatch.setitem(sys.modules, "eo_tides", None)     # force ImportError
    with pytest.raises(RuntimeError, match="eo-tides is not installed"):
        tides.predict_global(-123.0, 45.0, "2026-06-01", "2026-06-01")


def test_a_source_that_fails_yields_no_channel_no_fallback(monkeypatch, base_project, aoi_grid):
    """No fallback (D10): if a stacked source raises, it simply contributes no tree here --
    the OTHER stacked source is unaffected. (The old code fell back coops->eo_tides.)"""
    def bad(*a):
        raise RuntimeError("boom")

    def good(lon, lat, start, end, ds_cfg, station):
        s = pd.Series([1.0], index=pd.DatetimeIndex(["2026-06-01"]), name="tide")
        return s, {"method": "model"}

    monkeypatch.setitem(tides.SOURCES, "coops", bad)
    monkeypatch.setitem(tides.SOURCES, "eo_tides", good)
    monkeypatch.setattr(tides, "fetch_stations",
                        lambda: [{"id": "X", "name": "near", "lon": -123.9, "lat": 45.5}])
    base_project["products"]["tides"] = None
    proj = parse_config(base_project)
    tides.acquire(proj, grids={"a1": aoi_grid})

    root = Path(proj.output_dir) / "TIDE"
    assert not (root / "coops" / "aligned" / "a1" / "a1_tides.nc").exists()   # failed -> no tree
    assert (root / "eo_tides" / "aligned" / "a1" / "a1_tides.nc").exists()    # the other is fine

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])