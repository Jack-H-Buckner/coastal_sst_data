"""CUDEM DEM source: the tile-index cache, tile geometry, per-cell depth aggregation, and the
read-error-is-not-missing-coverage guarantee. Pure/offline (no CUDEM network).

These exercise `processes.bathymetry_cudem` directly -- the CUDEM half of the bathymetry
sources, split out of the orchestrator so a CUDEM failure cannot abort GMRT (see
test_bathymetry.py for the orchestration + isolation tests)."""

import numpy as np
import pytest

from coastal_sst_data.processes import bathymetry_cudem as C


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
    elev_mean, d_mean, d_p25, d_p75 = C.block_stats(elev_fine, k=2, H=2, W=2)
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
    monkeypatch.setattr(C.requests, "get",
                        lambda u, **kw: calls.append(u) or _Resp(_GOOD_INDEX))
    cache = tmp_path / "urllist.txt"

    assert len(C._fetch_index("http://idx", cache)) == 2
    assert len(C._fetch_index("http://idx", cache)) == 2
    assert len(calls) == 1                      # second call served from the cache


def test_an_http_error_page_is_NEVER_cached(tmp_path, monkeypatch):
    """No raise_for_status meant a 500 HTML body was written to the cache as if it were
    data. `.endswith('.tif')` then filtered every line out, CUDEM found no tiles, and the
    AoI was quietly downgraded to GMRT -- forever, because the cache file existed."""
    monkeypatch.setattr(C.requests, "get",
                        lambda u, **kw: _Resp("<html>500 Internal Server Error</html>", 500))
    cache = tmp_path / "urllist.txt"

    with pytest.raises(RuntimeError):
        C._fetch_index("http://idx", cache)
    assert not cache.exists()                   # nothing poisoned was left behind


def test_a_200_that_is_not_a_tile_list_is_NEVER_cached(tmp_path, monkeypatch):
    """A redirect notice or a maintenance page comes back 200 with a perfectly good body."""
    monkeypatch.setattr(C.requests, "get",
                        lambda u, **kw: _Resp("<html>we have moved</html>"))
    cache = tmp_path / "urllist.txt"

    with pytest.raises(RuntimeError, match="no .tif entries"):
        C._fetch_index("http://idx", cache)
    assert not cache.exists()


def test_a_poisoned_cache_from_an_older_run_is_discarded(tmp_path, monkeypatch, caplog):
    """The repair path: a tree that ALREADY holds a bad cache must heal itself."""
    cache = tmp_path / "urllist.txt"
    cache.write_text("<html>500 Internal Server Error</html>")     # as an old run left it
    monkeypatch.setattr(C.requests, "get", lambda u, **kw: _Resp(_GOOD_INDEX))

    with caplog.at_level("WARNING"):
        urls = C._fetch_index("http://idx", cache)

    assert len(urls) == 2                        # re-fetched, not trusted
    assert "no .tif entries" in caplog.text


def test_a_stale_cache_is_refreshed(tmp_path, monkeypatch):
    """NCEI adds tiles; a year-old index quietly misses them."""
    import os
    import time
    cache = tmp_path / "urllist.txt"
    cache.write_text("https://x/ncei19_n47x75_w122x50_old.tif\n")
    old = time.time() - (C.INDEX_MAX_AGE_S + 86400)
    os.utime(cache, (old, old))

    monkeypatch.setattr(C.requests, "get", lambda u, **kw: _Resp(_GOOD_INDEX))
    urls = C._fetch_index("http://idx", cache)
    assert len(urls) == 2                        # the fresh index, not the stale one


# --------------------------------------------------------------------------- #
# A tile that fails to READ is not a tile that does not EXIST
# --------------------------------------------------------------------------- #
def test_a_network_failure_does_not_masquerade_as_missing_coverage(monkeypatch, aoi_grid):
    """The old code dropped unreadable tiles, computed `cover` from the SURVIVORS, and
    wrote a DEM with a network-shaped hole in it labelled '60% cover'."""
    monkeypatch.setattr(C, "_fetch_index",
                        lambda u, c: ["https://x/ncei19_n47x75_w122x50_a.tif"])
    monkeypatch.setattr(C, "_tile_bounds", lambda n: (-180.0, -90.0, 180.0, 90.0))

    def dead(*a, **kw):
        raise ConnectionError("connection reset")

    monkeypatch.setattr(C.rioxarray, "open_rasterio", dead)

    with pytest.raises(C.TileReadError, match="network-shaped hole"):
        C.read_cudem(aoi_grid.search_bbox, aoi_grid.target_crs,
                     aoi_grid.transform, aoi_grid.width, aoi_grid.height,
                     aoi_grid.geom_proj, "http://idx", "/tmp/idx.txt", 10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])
