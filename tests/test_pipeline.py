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
    # config lists them out of order; pipeline must impose PROCESS_ORDER. MUR follows the
    # sensors because its `overpass_sensors` filter reads their aligned dirs.
    proj = _make_project({"mur": None, "bathymetry": None, "ecostress": None}, auth=EARTHDATA)
    pipeline.run_pipeline(proj, dry_run=True)
    assert [c["name"] for c in acquire_calls] == ["bathymetry", "ecostress", "mur"]


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

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])

# ---------------------------------------------------------------------------
# G. Parallel acquisition
#
# The unit of work is (product, AoI), and the dependency edges are PER-AoI: MODIS's
# coincidence filter reads Landsat's aligned files FOR THAT AoI (`match_landsat`), and MUR's
# `overpass_sensors` reads the sensor dirs FOR THAT AoI. So `mur(a1)` waits for
# `landsat(a1)`, not for `landsat(a2)` -- which is what removes the barriers between AoIs.
# ---------------------------------------------------------------------------
A2 = {"name": "a2", "center_lat": 46.5, "center_lon": -124.0,
      "buffer_ns_km": 8, "buffer_ew_km": 8}


@pytest.fixture
def timed_calls(monkeypatch):
    """Record (name, aoi, start, end) for every acquire(), with a real pause so overlap is
    observable."""
    import threading
    import time
    calls, lock = [], threading.Lock()

    def make(name):
        def f(project, *, grids=None, aois=None, dry_run=False, overwrite=False, **kw):
            t0 = time.monotonic()
            time.sleep(0.02)
            with lock:
                calls.append({"name": name, "aoi": (aois or [None])[0],
                              "start": t0, "end": time.monotonic()})
        return f

    for name, m in [("bathymetry", bathymetry), ("ecostress", ecostress),
                    ("mur", mur), ("landsat_pc", landsat_pc), ("met", met),
                    ("modis", modis), ("tides", tides), ("landcover_esa", landcover_esa)]:
        monkeypatch.setattr(m, "acquire", make(name))
    return calls


def _span(calls, name, aoi):
    got, = [c for c in calls if c["name"] == name and c["aoi"] == aoi]
    return got["start"], got["end"]


def test_parallel_dispatches_one_call_per_aoi(timed_calls):
    """Splitting an AoI list into one call per AoI is what makes AoIs parallelisable, and it
    needs no change inside any module -- acquire() already takes a list."""
    proj = _make_project({"bathymetry": None}, extra_area=A2)
    pipeline.run_pipeline(proj, dry_run=True, jobs=4)
    assert sorted(c["aoi"] for c in timed_calls) == ["a1", "a2"]


def test_parallel_honours_the_dependency_within_each_aoi(timed_calls):
    proj = _make_project({"modis": None, "landsat": None}, auth=EARTHDATA, extra_area=A2)
    pipeline.run_pipeline(proj, dry_run=True, jobs=4)

    for aoi in ("a1", "a2"):
        _, landsat_end = _span(timed_calls, "landsat_pc", aoi)
        modis_start, _ = _span(timed_calls, "modis", aoi)
        assert modis_start >= landsat_end, f"{aoi}: modis started before landsat finished"


def test_parallel_does_not_serialise_across_aois(timed_calls):
    """The payoff of per-AoI edges: a2's Landsat must not wait for a1's."""
    proj = _make_project({"modis": None, "landsat": None}, auth=EARTHDATA, extra_area=A2)
    pipeline.run_pipeline(proj, dry_run=True, jobs=4)

    a1 = _span(timed_calls, "landsat_pc", "a1")
    a2 = _span(timed_calls, "landsat_pc", "a2")
    assert a1[0] < a2[1] and a2[0] < a1[1], "the two AoIs' Landsat stages did not overlap"


def test_parallel_runs_independent_products_together(timed_calls):
    proj = _make_project({"bathymetry": None, "tides": None, "landcover": None})
    pipeline.run_pipeline(proj, dry_run=True, jobs=4)

    spans = [(c["start"], c["end"]) for c in timed_calls]
    overlap = any(a[0] < b[1] and b[0] < a[1]
                  for i, a in enumerate(spans) for b in spans[i + 1:])
    assert overlap, "independent products ran strictly one after another"


def test_a_service_gate_caps_its_products(monkeypatch, timed_calls):
    """mur/modis/ecostress are three products behind ONE Earthdata account, so the cap has
    to apply to the service, not to each product separately."""
    proj = _make_project({"mur": None, "ecostress": None}, auth=EARTHDATA, extra_area=A2)
    monkeypatch.setattr(proj.runtime, "gates", {"earthdata": 1})
    pipeline.run_pipeline(proj, dry_run=True, jobs=8)

    spans = sorted((c["start"], c["end"]) for c in timed_calls)
    assert len(spans) == 4
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= prev_end, "two Earthdata tasks overlapped under a cap of 1"


def test_a_failure_in_one_aoi_does_not_stop_another(monkeypatch, acquire_calls):
    def boom(project, *, grids=None, aois=None, **kw):
        if aois == ["a1"]:
            raise RuntimeError("a1 landsat died")
        acquire_calls.append({"name": "landsat_pc", "aois": aois})
    monkeypatch.setattr(landsat_pc, "acquire", boom)

    proj = _make_project({"landsat": None}, extra_area=A2)
    outcomes = pipeline.run_pipeline(proj, dry_run=True, jobs=4)

    assert [c["aois"] for c in acquire_calls] == [["a2"]]
    assert "failed" in outcomes[DataProduct.landsat]


def test_a_dependent_is_skipped_when_its_aois_dependency_fails(monkeypatch, acquire_calls):
    """MODIS reading a Landsat directory that was never written produces a channel that
    looks like 'no scenes matched' rather than 'the stage it depends on failed'."""
    def boom(project, *, grids=None, aois=None, **kw):
        raise RuntimeError("landsat died")
    monkeypatch.setattr(landsat_pc, "acquire", boom)

    ran = []
    monkeypatch.setattr(modis, "acquire",
                        lambda project, **kw: ran.append(kw.get("aois")))

    proj = _make_project({"landsat": None, "modis": None}, auth=EARTHDATA)
    outcomes = pipeline.run_pipeline(proj, dry_run=True, jobs=4)

    assert ran == []
    assert "skipped" in outcomes[DataProduct.modis]
    assert "dependency failed" in outcomes[DataProduct.modis]


def test_jobs_1_takes_the_original_serial_path(acquire_calls):
    """The escape hatch is a different code path, not a differently-tuned one: one batched
    call per module, exactly as before."""
    proj = _make_project({"bathymetry": None}, extra_area=A2)
    pipeline.run_pipeline(proj, dry_run=True, jobs=1)
    assert [c["aois"] for c in acquire_calls] == [["a1", "a2"]]


def test_jobs_defaults_to_the_config(acquire_calls):
    proj = _make_project({"bathymetry": None}, extra_area=A2)
    proj.runtime.jobs = 4
    pipeline.run_pipeline(proj, dry_run=True)
    assert sorted(c["aois"] for c in acquire_calls) == [["a1"], ["a2"]]


# ---------------------------------------------------------------------------
# H. Parallel terminal stages (assemble -> preprocess), on a DIVIDED memory budget
#
# Each AoI owns its own <aoi>.zarr, so AoIs are independent. assemble(aoi) and
# preprocess(aoi) are NOT: preprocess reads and rewrites the very store assemble wrote, so
# that pair is a dependency edge rather than two phases -- which is also the win, since AoI A
# can preprocess while AoI B is still assembling.
# ---------------------------------------------------------------------------
@pytest.fixture
def terminal_calls(monkeypatch, acquire_calls):
    """Record (stage, aoi, budget, start, end) for assemble/preprocess.

    Depends on `acquire_calls` so the ACQUISITION stages are stubbed too -- these tests run
    with dry_run=False (the terminal stages are skipped under --dry-run), and an unstubbed
    acquire would go to the network.
    """
    import threading
    import time
    from coastal_sst_data.processes import datacube
    calls, lock = [], threading.Lock()

    def make(stage):
        def f(project, *, grids=None, aois=None, dry_run=False, overwrite=False,
              memory_budget_gb=None):
            t0 = time.monotonic()
            time.sleep(0.02)
            with lock:
                calls.append({"stage": stage, "aoi": (aois or [None])[0],
                              "budget": memory_budget_gb,
                              "start": t0, "end": time.monotonic()})
        return f

    monkeypatch.setattr(pipeline.datacube, "assemble", make("assemble"))
    monkeypatch.setattr(pipeline.preprocess_stage, "preprocess", make("preprocess"))
    # A fixed budget, so the division is checked rather than the detection chain.
    monkeypatch.setattr(datacube, "budget_bytes", lambda eff: (64 * 1024**3, "test"))
    return calls


def _pp_project(**kw):
    proj = _make_project({"bathymetry": None}, extra_area=A2, **kw)
    proj.preprocess.enabled = True
    return proj


def test_terminal_stages_run_per_aoi_and_divide_the_budget(terminal_calls):
    """`budget_bytes` halves the detected memory on the assumption it owns the machine.
    Four AoIs each claiming half of 200 GB is a 400 GB working set -- and this stage is
    exactly the one that gets OOM-killed when that arithmetic is wrong."""
    proj = _pp_project()
    proj.runtime.assemble_jobs = 2
    pipeline.run_pipeline(proj, jobs=1, assemble=True, preprocess=True, verify_auth=False)

    assembles = [c for c in terminal_calls if c["stage"] == "assemble"]
    assert sorted(c["aoi"] for c in assembles) == ["a1", "a2"]
    assert all(c["budget"] == pytest.approx(32.0) for c in assembles)   # 64 GiB / 2


def test_preprocess_waits_for_its_own_aois_assembly(terminal_calls):
    """They read and write the SAME <aoi>.zarr; overlapping them would corrupt it."""
    proj = _pp_project()
    proj.runtime.assemble_jobs = 4
    pipeline.run_pipeline(proj, jobs=1, assemble=True, preprocess=True, verify_auth=False)

    for aoi in ("a1", "a2"):
        asm, = [c for c in terminal_calls if c["stage"] == "assemble" and c["aoi"] == aoi]
        pre, = [c for c in terminal_calls if c["stage"] == "preprocess" and c["aoi"] == aoi]
        assert pre["start"] >= asm["end"], f"{aoi}: preprocess overlapped its own assembly"


def test_different_aois_terminal_stages_overlap(terminal_calls):
    proj = _pp_project()
    proj.runtime.assemble_jobs = 4
    pipeline.run_pipeline(proj, jobs=1, assemble=True, preprocess=True, verify_auth=False)

    a1 = [c for c in terminal_calls if c["aoi"] == "a1"]
    a2 = [c for c in terminal_calls if c["aoi"] == "a2"]
    assert any(x["start"] < y["end"] and y["start"] < x["end"] for x in a1 for y in a2), \
        "the two AoIs' terminal stages ran strictly one after another"


def test_preprocess_is_skipped_when_its_assembly_failed(monkeypatch, terminal_calls):
    """Preprocessing a cube that was never written is not a smaller result -- it is a
    different failure wearing the wrong name."""
    def boom(project, *, aois=None, **kw):
        raise RuntimeError(f"assembly died for {aois}")
    monkeypatch.setattr(pipeline.datacube, "assemble", boom)

    proj = _pp_project()
    proj.runtime.assemble_jobs = 2
    outcomes = pipeline.run_pipeline(proj, jobs=1, assemble=True, preprocess=True,
                                     verify_auth=False)

    assert [c for c in terminal_calls if c["stage"] == "preprocess"] == []
    assert "failed" in outcomes["datacube"]
    assert "skipped" in outcomes["preprocess"]


def test_assemble_jobs_1_keeps_the_single_batched_call(terminal_calls):
    proj = _pp_project()
    proj.runtime.assemble_jobs = 1
    pipeline.run_pipeline(proj, jobs=1, assemble=True, preprocess=True, verify_auth=False)

    assembles = [c for c in terminal_calls if c["stage"] == "assemble"]
    assert len(assembles) == 1
    assert assembles[0]["aoi"] is None          # the whole-project call, as before
    assert assembles[0]["budget"] is None       # and no budget override
