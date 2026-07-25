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
# Failure isolation: one (AoI, source) failing must not abort the others.
# ---------------------------------------------------------------------------
def _two_aoi_project(tmp_path, sources):
    return parse_config({
        "name": "b", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-02"},
        "products": {"bathymetry": {"sources": sources}},
        "regions": [{"name": "r", "areas": [
            {"name": "a1", "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2},
            {"name": "a2", "center_lat": 46.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


def test_a_cudem_tile_read_error_does_not_abort_gmrt_or_other_aois(tmp_path, monkeypatch):
    """The whole point of the refactor: a transient CUDEM read (a 503) fails ONLY that
    (AoI, source) row. GMRT for that AoI, and every source for every other AoI, still run --
    the stage is no longer taken down by one CDN hiccup."""
    def cudem(g, params):
        if g.name == "a1":
            raise B.TileReadError("503 on a tile")          # transient, a1 only
        return _ok("cudem-DEM")(g, params)

    monkeypatch.setattr(B, "SOURCES", {"cudem": cudem, "gmrt": _ok("gmrt-DEM")})
    _stub_datum(monkeypatch)
    p = _two_aoi_project(tmp_path, ["cudem", "gmrt"])
    rep = B.acquire(p, grids=grid.project_grids(p))         # must NOT raise

    root = tmp_path / "BATHYMETRY"
    assert not (root / "cudem" / "aligned" / "a1" / "a1.nc").exists()   # the one failure
    assert (root / "gmrt" / "aligned" / "a1" / "a1.nc").exists()        # sibling source ran
    assert (root / "cudem" / "aligned" / "a2" / "a2.nc").exists()       # other AoI ran
    assert (root / "gmrt" / "aligned" / "a2" / "a2.nc").exists()
    assert rep.failed == 1                                  # exactly one row failed
    assert rep.written == 3


# ---------------------------------------------------------------------------
# Datum decoupled from the DEM download: a VDatum outage saves the DEM and retries
# the offset ALONE from disk on the next run (no tile re-download).
# ---------------------------------------------------------------------------
def test_a_vdatum_outage_saves_the_dem_and_retries_the_datum_from_disk(tmp_path, monkeypatch):
    fetches = []                                            # how many times CUDEM was READ
    def cudem(g, params):
        fetches.append(g.name)
        a = np.full((g.height, g.width), -3.0, dtype="float32")
        return a, np.abs(a), np.abs(a), np.abs(a), "cudem-DEM"

    monkeypatch.setattr(B, "SOURCES", {"cudem": cudem})
    monkeypatch.setattr(B.tides, "fetch_stations", lambda: [])

    # resolve_aoi raises on run 1 (VDatum down), succeeds on run 2.
    calls = {"n": 0}
    def resolve(g, elev, src, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("VDatum 503")
        return {"datum_offset_m": 1.3, "status": "ok", "method": "vdatum_navd88_to_lmsl",
                "dem_vertical_datum": "NAVD88"}
    monkeypatch.setattr(B.datum, "resolve_aoi", resolve)

    p = _one_aoi_project(tmp_path, ["cudem"])
    f = tmp_path / "BATHYMETRY" / "cudem" / "aligned" / "a1" / "a1.nc"

    # Run 1: tiles OK, VDatum down -> DEM saved, datum pending.
    B.acquire(p, grids=grid.project_grids(p))
    assert f.exists(), "the DEM must be saved even though the datum could not resolve"
    with xr.open_dataset(f) as ds:
        assert ds.attrs["datum_status"] == B.PENDING_DATUM
    assert fetches == ["a1"]

    # Run 2: DEM already complete but datum pending -> retry the OFFSET ALONE from disk.
    B.acquire(p, grids=grid.project_grids(p))
    with xr.open_dataset(f) as ds:
        assert ds.attrs["datum_status"] == "ok"
        assert ds.attrs["datum_offset_m"] == pytest.approx(1.3)
    assert fetches == ["a1"], "the datum retry must NOT re-download the CUDEM tiles"


def test_a_resolved_datum_is_not_re_resolved_on_a_later_run(tmp_path, monkeypatch):
    """A DEM whose datum already resolved is a plain skip -- the retry path is only for
    PENDING datums, so a healthy output is never needlessly re-resolved."""
    monkeypatch.setattr(B, "SOURCES", {"cudem": _ok("cudem-DEM")})
    calls = {"n": 0}
    def resolve(g, elev, src, **kw):
        calls["n"] += 1
        return {"datum_offset_m": 1.3, "status": "ok", "method": "stub",
                "dem_vertical_datum": "NAVD88"}
    monkeypatch.setattr(B.tides, "fetch_stations", lambda: [])
    monkeypatch.setattr(B.datum, "resolve_aoi", resolve)

    p = _one_aoi_project(tmp_path, ["cudem"])
    B.acquire(p, grids=grid.project_grids(p))
    B.acquire(p, grids=grid.project_grids(p))               # second run: nothing to do
    assert calls["n"] == 1                                  # resolved once, never re-resolved


def test_a_vdatum_outage_stamps_the_region_fallback_while_still_pending(tmp_path, monkeypatch):
    """On a transient VDatum outage WITH a region datum_offset_m set, the offset stays PENDING
    (so the retry loop tries VDatum again next run for the authoritative value) but the cube
    uses the region fallback in the meantime rather than a ~1 m-biased 0.0."""
    monkeypatch.setattr(B.datum, "resolve_aoi",
                        lambda g, elev, src, **kw: (_ for _ in ()).throw(RuntimeError("VDatum 503")))
    g = grid.project_grids(_one_aoi_project(tmp_path, ["cudem"]))["a1"]
    elev = np.full((g.height, g.width), -3.0, "float32")

    # With a fallback set: pending, but the fallback value is stamped (not 0.0).
    rec = B._resolve_datum(g, elev, "cudem", 1.2, [])
    assert rec["status"] == B.PENDING_DATUM                 # keeps retrying next run
    assert rec["method"] == "config_fallback_pending"
    assert rec["datum_offset_m"] == pytest.approx(1.2)

    # Without a fallback: the old behaviour -- pending, biased 0.0.
    rec0 = B._resolve_datum(g, elev, "cudem", None, [])
    assert rec0["status"] == B.PENDING_DATUM
    assert rec0["method"] == "datum_fetch_failed"
    assert rec0["datum_offset_m"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])
