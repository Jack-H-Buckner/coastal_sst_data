"""Run accounting. The failure these exist to prevent: a run that downloaded 100 days and
lost 40 of them printed exactly the same `Done.` / `ok` as a run that lost none."""

from coastal_sst_data import report


def test_a_lossy_run_is_NOT_ok():
    """The core of it. `ok` used to be recorded unconditionally by the pipeline."""
    r = report.ProductReport("mur")
    for _ in range(60):
        r.wrote(source="GHRSST MUR")
    for d in range(40):
        r.fail(f"2023{d:04d}", "HTTP 503")

    assert r.outcome == "degraded (40 failed)"      # NOT "ok"
    assert not r.ok
    assert r.attempted == 100 and r.success_rate == 0.6


def test_a_clean_run_is_ok():
    r = report.ProductReport("mur")
    r.wrote(source="GHRSST MUR")
    r.skip()
    assert r.ok and r.outcome == "ok"
    assert r.success_rate == 1.0


def test_a_skipped_item_was_not_attempted():
    """Skipping a complete file is not a network attempt, so it must not dilute the rate."""
    r = report.ProductReport("mur")
    r.skip(50)
    r.wrote()
    assert r.attempted == 1 and r.success_rate == 1.0


def test_a_product_that_produced_nothing_says_so():
    assert report.ProductReport("insitu").outcome == "no data"


def test_failure_list_is_capped_but_the_count_is_not():
    r = report.ProductReport("met")
    for i in range(100):
        r.fail(f"item{i}", "boom")
    assert r.failed == 100                                   # the COUNT is exact
    assert len(r.failures) == report.MAX_FAILURES_KEPT       # the detail is bounded
    assert r.dropped == 100 - report.MAX_FAILURES_KEPT


def test_sources_are_never_truncated():
    """An elided source name defeats the column: `cmems_mod_glo_phy_...` truncates to the
    same string whether it is the reanalysis or the forecast."""
    r = report.ProductReport("cmems")
    for _ in range(300):
        r.wrote(source="cmems_mod_glo_phy_my_0.083deg_P1D-m")
    for _ in range(65):
        r.wrote(source="cmems_mod_glo_phy_anfc_0.083deg_P1D-m")
    s = r.sources_str()
    assert "cmems_mod_glo_phy_my_0.083deg_P1D-m x300" in s
    assert "cmems_mod_glo_phy_anfc_0.083deg_P1D-m x65" in s
    assert "…" not in s


def test_run_report_surfaces_the_dem_downgrade():
    """`bathymetry ok` used to hide a silent 3 m CUDEM -> ~100 m GMRT fallback."""
    rr = report.RunReport()
    b = report.ProductReport("bathymetry")
    b.wrote(source="GMRT (topo, max)")
    rr.add("bathymetry", b.finish())

    out = rr.render()
    assert "GMRT (topo, max) x1" in out
    assert "what actually served this run" in out


def test_run_report_lists_the_lost_items():
    rr = report.RunReport()
    m = report.ProductReport("mur")
    m.wrote(); m.fail("hood_canal 20230314", "HTTPError 503")
    rr.add("mur", m.finish())

    out = rr.render()
    assert rr.any_failures
    assert "degraded (1 failed)" in out
    assert "hood_canal 20230314: HTTPError 503" in out
    assert "looks exactly like a cloudy day" in out


def test_a_stage_that_raised_is_recorded_without_a_report():
    rr = report.RunReport()
    rr.add("met", None, outcome="stage raised: boom")
    assert "stage raised: boom" in rr.render()
