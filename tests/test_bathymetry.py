"""Bathymetry: config -> params, per-region STACKED DEM sources (D10), and the per-source
acquisition (each DEM its own tree, its datum resolved inline). Pure/offline (no CUDEM or
GMRT network; the datum resolver + CO-OPS station fetch are stubbed).

The registry tests inject STUB sources into `bathymetry.SOURCES`, so they exercise the
machinery independently of cudem/gmrt -- proving new sources (gebco, ...) plug in cleanly."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from coastal_sst_data import grid, products
from coastal_sst_data.config import DataProduct, load_config, parse_config
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
    return None                     # signals insufficient coverage -> no channel from it


def _boom(g, params):
    raise RuntimeError("read failed")


def _stub_datum(monkeypatch, offsets=None):
    """Stub the inline datum resolution so no network is touched. `offsets` maps source->m."""
    offsets = offsets or {"cudem": 1.3, "gmrt": 0.0}
    monkeypatch.setattr(B.tides, "fetch_stations", lambda: [])
    monkeypatch.setattr(B.datum, "resolve_aoi", lambda g, elev, src, **kw: {
        "datum_offset_m": offsets.get(src, 0.0), "status": "ok", "method": "stub",
        "dem_vertical_datum": "NAVD88" if src == "cudem" else "MSL_APPROX"})


def _one_aoi_project(tmp_path, sources):
    return parse_config({
        "name": "b", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-02"},
        "products": {"bathymetry": {"sources": sources}},
        "regions": [{"name": "r", "areas": [
            {"name": "a1", "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


# ---------------------------------------------------------------------------
# _build_eff: global (source-agnostic) params + the per-AoI STACKED source list.
# ---------------------------------------------------------------------------
def test_build_eff_maps_example_config():
    eff = B._build_eff(load_config(EXAMPLE))
    assert eff["bathy_root"] == Path("path/to/data") / "BATHYMETRY"
    assert eff["params"]["stats_subgrid_m"] == 10.0
    assert eff["params"]["min_cudem_cover"] == 0.5

    # `ds` is keyed by AoI: pnw_estuaries stacks just [cudem]; puget_sound names nothing and
    # takes the project default (every known source). One config, per-AoI source lists.
    assert eff["ds"]["tillamook_bay"]["sources"] == ["cudem"]       # region override
    assert eff["ds"]["padilla_bay"]["sources"] == ["cudem", "gmrt"]  # project default (all)


def test_build_eff_requires_bathymetry_selected(base_project):
    base_project["products"] = {"mur": {"variable": "analysed_sst"}}   # drop bathymetry
    base_project["regions"][0]["sources"] = {}                          # and its region source
    with pytest.raises(ValueError, match="bathymetry is not a selected product"):
        B._build_eff(parse_config(base_project))


def test_build_eff_applies_option_overrides(base_project):
    base_project["products"]["bathymetry"] = {
        "sources": ["cudem"], "stats_subgrid_m": 5.0,
        "min_cudem_cover": 0.8, "output_format": "geotiff",
    }
    base_project["regions"][0]["sources"] = None       # no region override
    eff = B._build_eff(parse_config(base_project))
    assert eff["ds"]["a1"]["sources"] == ["cudem"]
    assert eff["params"]["stats_subgrid_m"] == 5.0
    assert eff["fmt"] == "geotiff"


# ---------------------------------------------------------------------------
# Region override -> project default, resolved through config.resolve_opts and landing in
# eff["ds"][<aoi>]["sources"] like every other product's per-AoI settings.
# ---------------------------------------------------------------------------
def test_sources_region_override(base_project):
    base_project["products"] = {"bathymetry": None}
    base_project["regions"][0]["sources"] = {"bathymetry": {"sources": ["gmrt"]}}
    eff = B._build_eff(parse_config(base_project))
    assert eff["ds"]["a1"]["sources"] == ["gmrt"]


def test_sources_default_when_unset(base_project):
    base_project["products"] = {"bathymetry": None}
    base_project["regions"][0]["sources"] = {}      # region names no source
    eff = B._build_eff(parse_config(base_project))
    assert eff["ds"]["a1"]["sources"] == ["cudem", "gmrt"]   # the module default (all known)


def test_sources_differ_per_region(base_project):
    """The whole point: two regions stack different DEM sources."""
    base_project["products"] = {"bathymetry": None}
    base_project["regions"] = [
        {"name": "r_cudem",
         "sources": {"bathymetry": {"sources": ["cudem"]}},
         "areas": [{"name": "a1", "center_lat": 45.0, "center_lon": -123.0,
                    "buffer_ns_km": 10, "buffer_ew_km": 10}]},
        {"name": "r_gmrt",
         "sources": {"bathymetry": {"sources": ["gmrt"]}},
         "areas": [{"name": "a2", "center_lat": 58.0, "center_lon": -135.0,
                    "buffer_ns_km": 10, "buffer_ew_km": 10}]},
    ]
    eff = B._build_eff(parse_config(base_project))
    assert eff["ds"]["a1"]["sources"] == ["cudem"]
    assert eff["ds"]["a2"]["sources"] == ["gmrt"]


# ---------------------------------------------------------------------------
# acquire: every configured source is validated against the SOURCES registry.
# ---------------------------------------------------------------------------
def test_unknown_source_is_rejected_at_config_load(base_project):
    """An unknown DEM in a `sources` list fails at config validation -- before acquire runs
    at all -- so a typo can never silently drop a source the user asked to stack."""
    base_project["products"] = {"bathymetry": {"sources": ["cudem"]}}
    base_project["regions"][0]["sources"] = {"bathymetry": {"sources": ["banana"]}}
    with pytest.raises(ValueError, match="unknown source"):
        parse_config(base_project)


def test_acquire_accepts_newly_registered_source(monkeypatch, base_project):
    """Adding a DEM source is: register it in the spec's `sources` (so config validation
    knows it), implement it in `SOURCES`, and give it a `DEM_DATUM` entry -- then a config
    naming it just works, no edits to the dispatch or acquire logic."""
    bathy = products.BY_PRODUCT[DataProduct.bathymetry]
    monkeypatch.setitem(bathy.sources, "gebco", "coastal_sst_data.processes.bathymetry")
    monkeypatch.setitem(B.SOURCES, "gebco", _ok("gebco"))
    monkeypatch.setitem(B.datum.DEM_DATUM, "gebco", "MSL_APPROX")
    base_project["products"] = {"bathymetry": {"sources": ["gebco"]}}
    base_project["regions"][0]["sources"] = None
    B.acquire(parse_config(base_project), dry_run=True)   # must NOT raise


# ---------------------------------------------------------------------------
# _fetch_one: ONE source, NO fallback (distinct-data sources are stacked, not substituted).
# ---------------------------------------------------------------------------
def test_fetch_one_returns_the_source_result(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _ok("x")})
    assert B._fetch_one("x", aoi_grid, {})[-1] == "x"


def test_fetch_one_no_coverage_returns_none(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _none})
    assert B._fetch_one("x", aoi_grid, {}) is None        # contributes no channel here


def test_fetch_one_read_error_returns_none(monkeypatch, aoi_grid):
    monkeypatch.setattr(B, "SOURCES", {"x": _boom})
    assert B._fetch_one("x", aoi_grid, {}) is None        # non-tile error -> skip the source


def test_fetch_one_tile_read_error_PROPAGATES(monkeypatch, aoi_grid):
    """A transient read failure must NOT be silently swallowed (that used to become a
    permanent CUDEM->GMRT downgrade); it fails loudly so the next run retries."""
    monkeypatch.setitem(B.SOURCES, "cudem",
                        lambda g, p: (_ for _ in ()).throw(B.TileReadError("boom")))
    with pytest.raises(B.TileReadError):
        B._fetch_one("cudem", aoi_grid, {})


# ---------------------------------------------------------------------------
# acquire end-to-end: one tree PER source, each with its own datum offset stamped.
# ---------------------------------------------------------------------------
def test_acquire_writes_one_tree_per_source_with_its_own_datum(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "SOURCES", {"cudem": _ok("cudem-DEM"), "gmrt": _ok("gmrt-DEM")})
    _stub_datum(monkeypatch, {"cudem": 1.3, "gmrt": 0.0})
    p = _one_aoi_project(tmp_path, ["cudem", "gmrt"])
    B.acquire(p, grids=grid.project_grids(p))

    for src, off in [("cudem", 1.3), ("gmrt", 0.0)]:
        f = tmp_path / "BATHYMETRY" / src / "aligned" / "a1" / "a1.nc"
        assert f.exists(), f"{src} tree not written"
        with xr.open_dataset(f) as ds:
            assert ds.attrs["bathy_source"] == src
            assert ds.attrs["datum_offset_m"] == pytest.approx(off)
            assert ds.attrs["datum_status"] == "ok"


def test_acquire_skips_a_source_with_no_coverage(tmp_path, monkeypatch):
    """A source that has no coverage here simply writes no tree -- the user stacks another."""
    monkeypatch.setattr(B, "SOURCES", {"cudem": _none, "gmrt": _ok("gmrt-DEM")})
    _stub_datum(monkeypatch)
    p = _one_aoi_project(tmp_path, ["cudem", "gmrt"])
    B.acquire(p, grids=grid.project_grids(p))
    assert not (tmp_path / "BATHYMETRY" / "cudem" / "aligned" / "a1" / "a1.nc").exists()
    assert (tmp_path / "BATHYMETRY" / "gmrt" / "aligned" / "a1" / "a1.nc").exists()


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])
