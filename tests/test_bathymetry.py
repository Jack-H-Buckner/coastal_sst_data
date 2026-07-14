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



# --------------------------------------------------------------------------- #
# The CUDEM tile index cache. Every way this broke was STICKY, and every way it broke
# ended in a silent downgrade to ~100 m GMRT that no later run would ever undo.
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, text="", status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            exc = RuntimeError(f"HTTP {self.status_code}")
            exc.response = self
            raise exc


_GOOD_INDEX = "https://x/ncei19_n47x75_w122x50_2021.tif\nhttps://x/ncei19_n47x50_w122x50_2021.tif\n"


def test_index_is_cached_and_reused(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(B.requests, "get",
                        lambda u, **kw: calls.append(u) or _Resp(_GOOD_INDEX))
    cache = tmp_path / "urllist.txt"

    assert len(B._fetch_index("http://idx", cache)) == 2
    assert len(B._fetch_index("http://idx", cache)) == 2
    assert len(calls) == 1                      # second call served from the cache


def test_an_http_error_page_is_NEVER_cached(tmp_path, monkeypatch):
    """No raise_for_status meant a 500 HTML body was written to the cache as if it were
    data. `.endswith('.tif')` then filtered every line out, CUDEM found no tiles, and the
    AoI was quietly downgraded to GMRT -- forever, because the cache file existed."""
    monkeypatch.setattr(B.requests, "get",
                        lambda u, **kw: _Resp("<html>500 Internal Server Error</html>", 500))
    cache = tmp_path / "urllist.txt"

    with pytest.raises(RuntimeError):
        B._fetch_index("http://idx", cache)
    assert not cache.exists()                   # nothing poisoned was left behind


def test_a_200_that_is_not_a_tile_list_is_NEVER_cached(tmp_path, monkeypatch):
    """A redirect notice or a maintenance page comes back 200 with a perfectly good body."""
    monkeypatch.setattr(B.requests, "get",
                        lambda u, **kw: _Resp("<html>we have moved</html>"))
    cache = tmp_path / "urllist.txt"

    with pytest.raises(RuntimeError, match="no .tif entries"):
        B._fetch_index("http://idx", cache)
    assert not cache.exists()


def test_a_poisoned_cache_from_an_older_run_is_discarded(tmp_path, monkeypatch, caplog):
    """The repair path: a tree that ALREADY holds a bad cache must heal itself."""
    cache = tmp_path / "urllist.txt"
    cache.write_text("<html>500 Internal Server Error</html>")     # as an old run left it
    monkeypatch.setattr(B.requests, "get", lambda u, **kw: _Resp(_GOOD_INDEX))

    with caplog.at_level("WARNING"):
        urls = B._fetch_index("http://idx", cache)

    assert len(urls) == 2                        # re-fetched, not trusted
    assert "no .tif entries" in caplog.text


def test_a_stale_cache_is_refreshed(tmp_path, monkeypatch):
    """NCEI adds tiles; a year-old index quietly misses them."""
    import os
    import time
    cache = tmp_path / "urllist.txt"
    cache.write_text("https://x/ncei19_n47x75_w122x50_old.tif\n")
    old = time.time() - (B.INDEX_MAX_AGE_S + 86400)
    os.utime(cache, (old, old))

    monkeypatch.setattr(B.requests, "get", lambda u, **kw: _Resp(_GOOD_INDEX))
    urls = B._fetch_index("http://idx", cache)
    assert len(urls) == 2                        # the fresh index, not the stale one


# --------------------------------------------------------------------------- #
# A tile that fails to READ is not a tile that does not EXIST
# --------------------------------------------------------------------------- #
def test_a_network_failure_does_not_masquerade_as_missing_coverage(monkeypatch, aoi_grid):
    """The old code dropped unreadable tiles, computed `cover` from the SURVIVORS, and
    wrote a DEM with a network-shaped hole in it labelled '60% cover'."""
    monkeypatch.setattr(B, "_fetch_index",
                        lambda u, c: ["https://x/ncei19_n47x75_w122x50_a.tif"])
    monkeypatch.setattr(B, "_tile_bounds", lambda n: (-180.0, -90.0, 180.0, 90.0))

    def dead(*a, **kw):
        raise ConnectionError("connection reset")

    monkeypatch.setattr(B.rioxarray, "open_rasterio", dead)

    with pytest.raises(B.TileReadError, match="network-shaped hole"):
        B.read_cudem(aoi_grid.search_bbox, aoi_grid.target_crs,
                              aoi_grid.transform, aoi_grid.width, aoi_grid.height,
                              aoi_grid.geom_proj, "http://idx", "/tmp/idx.txt", 10.0)


def test_a_tile_read_error_does_NOT_fall_back_to_gmrt(monkeypatch, aoi_grid):
    """Falling back on a transient error swaps a 3 m NAVD88 DEM for a ~100 m MSL one
    because a CDN hiccuped -- and nothing ever puts it back, because the next run finds a
    complete B file and skips the AoI."""
    gmrt_called = []
    monkeypatch.setitem(B.SOURCES, "cudem",
                        lambda g, p: (_ for _ in ()).throw(B.TileReadError("boom")))
    monkeypatch.setitem(B.SOURCES, "gmrt",
                        lambda g, p: gmrt_called.append(1) or "gmrt-result")

    with pytest.raises(B.TileReadError):
        B._fetch_with_fallback("cudem", aoi_grid, {}, "gmrt")
    assert not gmrt_called          # the AoI FAILS and is retried, not downgraded


def test_genuine_missing_coverage_DOES_fall_back(monkeypatch, aoi_grid, caplog):
    """...while an area CUDEM genuinely does not cover still falls back, loudly."""
    monkeypatch.setitem(B.SOURCES, "cudem", lambda g, p: None)   # no coverage
    monkeypatch.setitem(B.SOURCES, "gmrt", lambda g, p: "gmrt-result")

    with caplog.at_level("WARNING"):
        res = B._fetch_with_fallback("cudem", aoi_grid, {}, "gmrt")

    assert res == "gmrt-result"
    assert "FALLING BACK" in caplog.text and "different vertical datum" in caplog.text

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])
