
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


# --------------------------------------------------------------------------- #
# _build_eff: config -> acquisition params
# --------------------------------------------------------------------------- #
def test_build_eff_maps_example_config():
    """The example config maps to the expected tides acquisition parameters."""
    eff = tides._build_eff(load_config(EXAMPLE))
    assert eff["ds"]["interval"] == "h"
    assert eff["ds"]["stations"] == {}                     # no overrides in example
    assert eff["ds"]["warn_distance_km"] == tides.DEFAULT_WARN_KM
    assert eff["fmt"] == "netcdf"
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["out_dir"] == Path("path/to/data") / "TIDE" / "aligned"
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
    assert eff["ds"]["interval"] == tides.DEFAULT_INTERVAL
    assert eff["ds"]["stations"] == {}
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
    assert eff["ds"]["interval"] == "6"
    assert eff["ds"]["stations"] == {"a1": "9445133"}
    assert eff["ds"]["warn_distance_km"] == 50.0
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
# Source selection + global (eo-tides) backup
# --------------------------------------------------------------------------- #
def test_build_eff_source_defaults(base_project):
    """A bare `tides:` gets the coops-primary / eo_tides-backup defaults."""
    base_project["products"]["tides"] = None
    eff = tides._build_eff(parse_config(base_project))
    assert eff["ds"]["default_source"] == tides.DEFAULT_SOURCE == "coops"
    assert eff["ds"]["fallback"] == tides.DEFAULT_FALLBACK == "eo_tides"
    assert eff["ds"]["fallback_distance_km"] == tides.DEFAULT_FALLBACK_KM
    assert eff["ds"]["model"] == tides.DEFAULT_MODEL == "EOT20"
    assert eff["ds"]["model_directory"] is None


def test_build_eff_fallback_none_disables_backup(base_project):
    """`fallback: none` normalizes to a disabled (None) fallback."""
    base_project["products"]["tides"] = {"fallback": "none"}
    eff = tides._build_eff(parse_config(base_project))
    assert eff["ds"]["fallback"] is None


def test_build_eff_reads_source_options(base_project):
    """Source / model options come through the ds config."""
    base_project["products"]["tides"] = {
        "default_source": "eo_tides", "fallback": "coops",
        "fallback_distance_km": 200, "model": "FES2022",
        "model_directory": "/data/tide_models",
    }
    eff = tides._build_eff(parse_config(base_project))
    assert eff["ds"]["default_source"] == "eo_tides"
    assert eff["ds"]["fallback"] == "coops"
    assert eff["ds"]["fallback_distance_km"] == 200.0
    assert eff["ds"]["model"] == "FES2022"
    assert eff["ds"]["model_directory"] == "/data/tide_models"


def test_resolve_source_region_override(base_project):
    """A region's `sources.tides.source` overrides the project default."""
    base_project["products"]["tides"] = None
    base_project["regions"][0]["sources"]["tides"] = {"source": "eo_tides"}
    proj = parse_config(base_project)
    assert tides._resolve_source(proj, "a1", "coops") == "eo_tides"


def test_resolve_source_falls_back_to_default(base_project):
    """With no region override, the project default source is used."""
    proj = parse_config(base_project)                      # no tides source set
    assert tides._resolve_source(proj, "a1", "coops") == "coops"


def test_acquire_rejects_unknown_source(base_project):
    """A typo'd source fails loudly before any network work."""
    base_project["products"]["tides"] = {"default_source": "bogus"}
    proj = parse_config(base_project)
    with pytest.raises(ValueError, match="tides source not recognized"):
        tides.acquire(proj, dry_run=True)


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


def test_predict_with_fallback_uses_backup_on_error(monkeypatch):
    """When the primary source raises, the fallback source is used."""
    def bad(*a):
        raise RuntimeError("boom")

    def good(lon, lat, start, end, ds_cfg, station):
        s = pd.Series([1.0], index=pd.DatetimeIndex(["2026-06-01"]), name="tide")
        return s, {"method": "backup"}

    monkeypatch.setitem(tides.SOURCES, "coops", bad)
    monkeypatch.setitem(tides.SOURCES, "eo_tides", good)
    res = tides._predict_with_fallback("coops", "eo_tides", -123.0, 45.0,
                                       "2026-06-01", "2026-06-01",
                                       {"interval": "h"}, None, "a1")
    assert res is not None
    _, attrs, used = res
    assert used == "eo_tides"
    assert attrs["method"] == "backup"


def test_predict_with_fallback_returns_none_when_both_fail(monkeypatch):
    """If primary and fallback both fail, the AoI yields None (is skipped)."""
    def bad(*a):
        raise RuntimeError("boom")

    monkeypatch.setitem(tides.SOURCES, "coops", bad)
    monkeypatch.setitem(tides.SOURCES, "eo_tides", bad)
    res = tides._predict_with_fallback("coops", "eo_tides", -123.0, 45.0,
                                       "2026-06-01", "2026-06-01",
                                       {"interval": "h"}, None, "a1")
    assert res is None
