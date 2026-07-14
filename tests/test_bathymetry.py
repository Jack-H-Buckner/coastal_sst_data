"""Bathymetry: config -> params, per-region DEM source resolution, and the
source-registry / fallback machinery. Pure/offline (no CUDEM or GMRT network).

The fallback + registry tests inject STUB sources into `bathymetry.SOURCES`, so
they exercise the machinery independently of cudem/gmrt -- proving new sources
(gebco, copernicus_glo30, ...) will plug in without touching this logic."""

from pathlib import Path

import numpy as np
import pytest

from coastal_sst_data.config import load_config, parse_config
from coastal_sst_data.processes import bathymetry as B

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


# --- stub source fetchers (registry is name -> fetcher) -------------------- #
def _ok(tag):
    """A fetcher that 'succeeds', tagging its output so tests know who ran."""
    def f(g, params):
        a = np.zeros((g.height, g.width), dtype="float32")
        return a, a, a, a, tag
    return f


def _none(g, params):
    return None                     # signals insufficient coverage -> fallback


def _boom(g, params):
    raise RuntimeError("read failed")


# ---------------------------------------------------------------------------
# _build_eff: global (source-agnostic) params.
# ---------------------------------------------------------------------------
def test_build_eff_maps_example_config():
    eff = B._build_eff(load_config(EXAMPLE))
    assert eff["out_dir"] == Path("path/to/data") / "BATHYMETRY" / "aligned"
    assert eff["default_source"] == "gmrt"          # default
    assert eff["fallback"] == "gmrt"
    assert eff["params"]["stats_subgrid_m"] == 10.0
    assert eff["params"]["min_cudem_cover"] == 0.5


def test_build_eff_requires_bathymetry_selected(base_project):
    base_project["products"] = {"mur": {"variable": "analysed_sst"}}   # drop bathymetry
    base_project["regions"][0]["sources"] = {}                          # and its region source
    with pytest.raises(ValueError, match="bathymetry is not a selected product"):
        B._build_eff(parse_config(base_project))


def test_build_eff_applies_option_overrides(base_project):
    base_project["products"]["bathymetry"] = {
        "default_source": "cudem", "fallback": None, "stats_subgrid_m": 5.0,
        "min_cudem_cover": 0.8, "output_format": "geotiff",
    }
    eff = B._build_eff(parse_config(base_project))
    assert eff["default_source"] == "cudem"
    assert eff["fallback"] is None
    assert eff["params"]["stats_subgrid_m"] == 5.0
    assert eff["fmt"] == "geotiff"


# ---------------------------------------------------------------------------
# _resolve_source: the two-level lookup (region override -> project default).
# ---------------------------------------------------------------------------
def test_resolve_source_region_override(base_project):
    base_project["products"] = {"bathymetry": None}
    base_project["regions"][0]["sources"] = {"bathymetry": {"dem_source": "cudem"}}
    proj = parse_config(base_project)
    assert B._resolve_source(proj, "a1", "gmrt") == "cudem"


def test_resolve_source_falls_back_to_default(base_project):
    base_project["products"] = {"bathymetry": None}
    base_project["regions"][0]["sources"] = {}      # region names no source
    proj = parse_config(base_project)
    assert B._resolve_source(proj, "a1", "gmrt") == "gmrt"


def test_resolve_source_differs_per_region(base_project):
    """The whole point: two regions, two different DEM sources."""
    base_project["products"] = {"bathymetry": None}
    base_project["regions"] = [
        {"name": "r_cudem",
         "sources": {"bathymetry": {"dem_source": "cudem"}},
         "areas": [{"name": "a1", "center_lat": 45.0, "center_lon": -123.0,
                    "buffer_ns_km": 10, "buffer_ew_km": 10}]},
        {"name": "r_default",   # no bathymetry source -> default
         "areas": [{"name": "a2", "center_lat": 58.0, "center_lon": -135.0,
                    "buffer_ns_km": 10, "buffer_ew_km": 10}]},
    ]
    proj = parse_config(base_project)
    assert B._resolve_source(proj, "a1", "gmrt") == "cudem"
    assert B._resolve_source(proj, "a2", "gmrt") == "gmrt"


# ---------------------------------------------------------------------------
# acquire: per-AoI source is validated against the SOURCES registry.
# ---------------------------------------------------------------------------
def test_acquire_rejects_unknown_source(base_project):
    base_project["products"] = {"bathymetry": None}
    base_project["regions"][0]["sources"] = {"bathymetry": {"dem_source": "banana"}}
    with pytest.raises(ValueError, match="not recognized"):
        B.acquire(parse_config(base_project), dry_run=True)


def test_acquire_accepts_newly_registered_source(monkeypatch, base_project):
    """Registering a source in SOURCES makes it usable -- no validation edits."""
    monkeypatch.setitem(B.SOURCES, "gebco", _ok("gebco"))
    base_project["products"] = {"bathymetry": None}
    base_project["regions"][0]["sources"] = {"bathymetry": {"dem_source": "gebco"}}
    B.acquire(parse_config(base_project), dry_run=True)   # must NOT raise


# ---------------------------------------------------------------------------
# _fetch_with_fallback: source-agnostic machinery (stub registry).
# ---------------------------------------------------------------------------
def test_fetch_primary_success_skips_fallback(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _ok("x"), "gmrt": _ok("gmrt")})
    res = B._fetch_with_fallback("x", aoi_grid, {}, "gmrt")
    assert res[-1] == "x"                     # primary used; fallback untouched


def test_fetch_none_triggers_fallback(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _none, "gmrt": _ok("gmrt")})
    res = B._fetch_with_fallback("x", aoi_grid, {}, "gmrt")
    assert res[-1] == "gmrt"                   # None -> fell back


def test_fetch_error_triggers_fallback(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _boom, "gmrt": _ok("gmrt")})
    res = B._fetch_with_fallback("x", aoi_grid, {}, "gmrt")
    assert res[-1] == "gmrt"                   # exception -> fell back


def test_fetch_no_fallback_returns_none(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _none})
    assert B._fetch_with_fallback("x", aoi_grid, {}, None) is None


def test_fetch_fallback_also_fails_returns_none(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _none, "gmrt": _none})
    assert B._fetch_with_fallback("x", aoi_grid, {}, "gmrt") is None


# ---------------------------------------------------------------------------
# block_stats: pure per-cell depth aggregation (no network/rasters).
# ---------------------------------------------------------------------------
def test_block_stats_depth_from_elevation():
    """4x4 fine elevation -> 2x2 coarse (k=2): mean elev + depth (0 on land)."""
    elev_fine = np.array([
        [-10, -10, 5, 5],
        [-10, -10, 5, 5],
        [-20, -20, np.nan, np.nan],
        [-20, -20, np.nan, np.nan],
    ], dtype="float32")
    elev_mean, d_mean, d_p25, d_p75 = B.block_stats(elev_fine, k=2, H=2, W=2)
    # cell (0,0): all -10 -> depth 10 ; (0,1): all +5 (land) -> depth 0
    assert elev_mean[0, 0] == -10 and d_mean[0, 0] == 10
    assert elev_mean[0, 1] == 5 and d_mean[0, 1] == 0
    assert elev_mean[1, 0] == -20 and d_mean[1, 0] == 20
    assert np.isnan(elev_mean[1, 1])                     # all-NaN cell stays NaN
    # uniform cells -> percentiles equal the mean
    assert d_p25[0, 0] == 10 and d_p75[0, 0] == 10

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])