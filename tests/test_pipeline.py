"""Pipeline orchestration. Each product module's acquire() is stubbed into a
recorder, so these assert WHAT the orchestrator does -- order, dispatch, the
Landsat source selector, skip-unimplemented, resilience, and subsetting -- with
no network and no real acquisition."""

import pytest

from coastal_sst_data.config import DataProduct, parse_config
from coastal_sst_data import pipeline
from coastal_sst_data.processes import (
    bathymetry, ecostress, landcover_esa, landsat_pc, met, modis, mur, tides,
)

EARTHDATA = {"earthdata": {"auth_strategy": "netrc"}}


def _make_project(products, *, auth=None, extra_area=None):
    """Smallest valid project selecting `products` (dict), for the orchestrator."""
    d = {
        "name": "t", "output_dir": "out",
        "time": {"start_date": "2023-08-01", "end_date": "2023-08-31"},
        "products": products,
        "regions": [{"name": "r", "areas": [
            {"name": "a1", "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 8, "buffer_ew_km": 8}]}],
    }
    if auth:
        d["auth"] = auth
    if extra_area:
        d["regions"][0]["areas"].append(extra_area)
    return parse_config(d)


@pytest.fixture
def acquire_calls(monkeypatch):
    """Stub every product module's acquire() to record its call. Returns the list
    of {name, grids, aois, dry_run, overwrite} in call order."""
    calls = []

    def make(name):
        def f(project, *, grids=None, aois=None, dry_run=False, overwrite=False, **kw):
            calls.append({"name": name, "grids": grids, "aois": aois,
                          "dry_run": dry_run, "overwrite": overwrite})
        return f

    for name, m in [("bathymetry", bathymetry), ("ecostress", ecostress),
                    ("mur", mur), ("landsat_pc", landsat_pc), ("met", met),
                    ("modis", modis), ("tides", tides), ("landcover_esa", landcover_esa)]:
        monkeypatch.setattr(m, "acquire", make(name))
    return calls


# ---------------------------------------------------------------------------
# A. Dispatch & ordering
# ---------------------------------------------------------------------------
def test_products_run_in_process_order(acquire_calls):
    # config lists them out of order; pipeline must impose PROCESS_ORDER.
    proj = _make_project({"mur": None, "bathymetry": None, "ecostress": None}, auth=EARTHDATA)
    pipeline.run_pipeline(proj, dry_run=True)
    assert [c["name"] for c in acquire_calls] == ["bathymetry", "mur", "ecostress"]


def test_landsat_runs_before_modis(acquire_calls):
    proj = _make_project({"modis": None, "landsat": None}, auth=EARTHDATA)
    pipeline.run_pipeline(proj, dry_run=True)
    assert [c["name"] for c in acquire_calls] == ["landsat_pc", "modis"]


def test_landsat_dispatched_to_pc_source(acquire_calls):
    proj = _make_project({"landsat": {"source": "pc"}})   # pc -> no auth
    pipeline.run_pipeline(proj, dry_run=True)
    assert [c["name"] for c in acquire_calls] == ["landsat_pc"]


def test_grids_computed_once_and_shared(acquire_calls):
    proj = _make_project({"bathymetry": None, "mur": None}, auth=EARTHDATA)
    pipeline.run_pipeline(proj, dry_run=True)
    grids = [c["grids"] for c in acquire_calls]
    assert len(grids) == 2
    assert grids[0] is grids[1]              # the SAME grids object, not recomputed
    assert set(grids[0]) == {"a1"}


# ---------------------------------------------------------------------------
# B. Skip / not implemented
# ---------------------------------------------------------------------------
def test_unimplemented_product_skipped(acquire_calls):
    # landsat via the aws source has no module yet -> skipped; mur still runs.
    proj = _make_project({"mur": None, "landsat": {"source": "aws"}}, auth=EARTHDATA)
    out = pipeline.run_pipeline(proj, dry_run=True)
    assert [c["name"] for c in acquire_calls] == ["mur"]         # landsat(aws) never called
    assert out[DataProduct.landsat] == "skipped (not implemented)"
    assert out[DataProduct.mur] == "ok"


def test_landsat_unimplemented_source_skipped(acquire_calls):
    proj = _make_project({"landsat": {"source": "aws"}})   # valid config, no module yet
    out = pipeline.run_pipeline(proj, dry_run=True)
    assert acquire_calls == []
    assert out[DataProduct.landsat] == "skipped (not implemented)"


# ---------------------------------------------------------------------------
# B'. Datacube assembly (terminal stage, opt-in).
# ---------------------------------------------------------------------------
def test_assemble_runs_after_products_when_requested(acquire_calls, monkeypatch):
    seen = {}
    def stub(project, *, grids=None, aois=None, dry_run=False, overwrite=False):
        seen.update(called=True, n_grids=len(grids))
    monkeypatch.setattr(pipeline.datacube, "assemble", stub)
    proj = _make_project({"mur": None, "bathymetry": None}, auth=EARTHDATA)
    out = pipeline.run_pipeline(proj, dry_run=True, assemble=True)
    assert seen["called"] and seen["n_grids"] == 1
    assert out["datacube"] == "ok"
    assert [c["name"] for c in acquire_calls] == ["bathymetry", "mur"]   # products ran first


def test_assemble_not_run_by_default(acquire_calls, monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline.datacube, "assemble",
                        lambda *a, **k: seen.setdefault("called", True))
    out = pipeline.run_pipeline(_make_project({"mur": None}, auth=EARTHDATA), dry_run=True)
    assert "called" not in seen
    assert "datacube" not in out


def test_assemble_failure_recorded_not_fatal(acquire_calls, monkeypatch):
    def boom(project, **kw):
        raise RuntimeError("nope")
    monkeypatch.setattr(pipeline.datacube, "assemble", boom)
    out = pipeline.run_pipeline(_make_project({"mur": None}, auth=EARTHDATA),
                                dry_run=True, assemble=True)
    assert out["datacube"].startswith("failed")
    assert out[DataProduct.mur] == "ok"                    # product still succeeded


# ---------------------------------------------------------------------------
# C. Resilience & reporting
# ---------------------------------------------------------------------------
def test_product_failure_does_not_abort_run(acquire_calls, monkeypatch):
    def boom(project, **kw):
        raise RuntimeError("nope")
    monkeypatch.setattr(mur, "acquire", boom)             # override the recorder for mur
    proj = _make_project({"mur": None, "ecostress": None}, auth=EARTHDATA)
    out = pipeline.run_pipeline(proj, dry_run=True)
    assert out[DataProduct.mur].startswith("failed")
    assert out[DataProduct.ecostress] == "ok"             # ran despite mur failing
    assert [c["name"] for c in acquire_calls] == ["ecostress"]


def test_outcomes_summary_returned(acquire_calls):
    proj = _make_project({"mur": None, "landsat": {"source": "aws"}}, auth=EARTHDATA)
    out = pipeline.run_pipeline(proj, dry_run=True)
    assert out == {DataProduct.mur: "ok",
                   DataProduct.landsat: "skipped (not implemented)"}


def test_keyboard_interrupt_propagates(monkeypatch):
    def interrupt(project, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(mur, "acquire", interrupt)
    proj = _make_project({"mur": None}, auth=EARTHDATA)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_pipeline(proj, dry_run=True)


# ---------------------------------------------------------------------------
# D. Subsetting & validation
# ---------------------------------------------------------------------------
def test_products_filter_runs_subset(acquire_calls):
    proj = _make_project({"mur": None, "bathymetry": None}, auth=EARTHDATA)
    pipeline.run_pipeline(proj, products=[DataProduct.mur], dry_run=True)
    assert [c["name"] for c in acquire_calls] == ["mur"]        # bathymetry filtered out


def test_products_filter_rejects_unselected():
    proj = _make_project({"mur": None}, auth=EARTHDATA)
    with pytest.raises(SystemExit, match="not selected"):
        pipeline.run_pipeline(proj, products=[DataProduct.ecostress])


def test_unknown_aoi_rejected():
    proj = _make_project({"mur": None}, auth=EARTHDATA)
    with pytest.raises(SystemExit, match="not in config"):
        pipeline.run_pipeline(proj, aois=["nope"])


def test_aois_and_flags_passed_through(acquire_calls):
    proj = _make_project({"mur": None}, auth=EARTHDATA)
    pipeline.run_pipeline(proj, aois=["a1"], dry_run=True, overwrite=True)
    c = acquire_calls[0]
    assert c["aois"] == ["a1"] and c["dry_run"] is True and c["overwrite"] is True


def test_empty_grids_raises(monkeypatch):
    proj = _make_project({"mur": None}, auth=EARTHDATA)
    monkeypatch.setattr(pipeline, "compute_grids", lambda p: {})   # all AoIs failed
    with pytest.raises(SystemExit, match="no usable AoI grids"):
        pipeline.run_pipeline(proj, dry_run=True)


# ---------------------------------------------------------------------------
# E. compute_grids resilience
# ---------------------------------------------------------------------------
def test_compute_grids_skips_failing_aoi(caplog):
    # a1 is normal; `am` crosses the antimeridian -> compute_aoi_grid raises.
    proj = _make_project({"mur": None}, auth=EARTHDATA, extra_area={
        "name": "am", "center_lat": 0.0, "center_lon": -179.95,
        "buffer_ns_km": 25, "buffer_ew_km": 50})
    with caplog.at_level("WARNING"):
        grids = pipeline.compute_grids(proj)
    assert set(grids) == {"a1"}                    # pathological AoI dropped, not fatal
    assert "am" in caplog.text


def test_compute_grids_all_normal():
    proj = _make_project({"mur": None}, auth=EARTHDATA, extra_area={
        "name": "a2", "center_lat": 46.0, "center_lon": -124.0,
        "buffer_ns_km": 8, "buffer_ew_km": 8})
    assert set(pipeline.compute_grids(proj)) == {"a1", "a2"}


# ---------------------------------------------------------------------------
# F. _module_for (unit)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source, auth, is_pc", [
    ("pc", None, True),
    ("planetary_computer", None, True),
    ("aws", None, False),
    ("gee", {"gee": {"project": "p"}}, False),
])
def test_module_for_landsat_by_source(source, auth, is_pc):
    proj = _make_project({"landsat": {"source": source}}, auth=auth)
    mod = pipeline._module_for(proj, DataProduct.landsat)
    assert (mod is landsat_pc) if is_pc else (mod is None)


def test_module_for_implemented_products():
    proj = _make_project({"mur": None}, auth=EARTHDATA)   # project need not select them
    assert pipeline._module_for(proj, DataProduct.bathymetry) is bathymetry
    assert pipeline._module_for(proj, DataProduct.mur) is mur
    assert pipeline._module_for(proj, DataProduct.ecostress) is ecostress
    assert pipeline._module_for(proj, DataProduct.met) is met
    assert pipeline._module_for(proj, DataProduct.modis) is modis
    assert pipeline._module_for(proj, DataProduct.tides) is tides


def test_module_for_unimplemented_source():
    # No top-level product is unimplemented now; only certain SOURCES are (e.g.
    # landsat via aws/gee), which _module_for resolves to None.
    proj = _make_project({"landsat": {"source": "aws"}})
    assert pipeline._module_for(proj, DataProduct.landsat) is None


@pytest.mark.parametrize("source, auth, is_esa", [
    ("esa", None, True),
    ("worldcover", None, True),
    ("gee", {"gee": {"project": "p"}}, False),
])
def test_module_for_landcover_by_source(source, auth, is_esa):
    proj = _make_project({"landcover": {"source": source}}, auth=auth)
    mod = pipeline._module_for(proj, DataProduct.landcover)
    assert (mod is landcover_esa) if is_esa else (mod is None)


# ---------------------------------------------------------------------------
# G. Credential preflight (auth.verify) -- stubbed, no network.
# ---------------------------------------------------------------------------
def test_preflight_skipped_on_dry_run(acquire_calls, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.auth, "verify", lambda p, products=None: seen.append(True) or {})
    pipeline.run_pipeline(_make_project({"mur": None}, auth=EARTHDATA), dry_run=True)
    assert seen == []                       # dry-run preview -> no preflight


def test_preflight_runs_on_real_run(acquire_calls, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.auth, "verify",
                        lambda p, products=None: seen.append(list(products)) or {})
    pipeline.run_pipeline(_make_project({"mur": None}, auth=EARTHDATA), dry_run=False)
    assert seen and DataProduct.mur in seen[0]   # verified the products being run


def test_preflight_failure_aborts_before_any_acquire(acquire_calls, monkeypatch):
    def boom(p, products=None):
        raise RuntimeError("bad creds")
    monkeypatch.setattr(pipeline.auth, "verify", boom)
    with pytest.raises(SystemExit, match="preflight failed"):
        pipeline.run_pipeline(_make_project({"mur": None}, auth=EARTHDATA), dry_run=False)
    assert acquire_calls == []               # aborted before running any product


def test_no_verify_skips_preflight(acquire_calls, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.auth, "verify", lambda p, products=None: seen.append(True) or {})
    pipeline.run_pipeline(_make_project({"mur": None}, auth=EARTHDATA),
                          dry_run=False, verify_auth=False)
    assert seen == []                        # explicitly opted out
