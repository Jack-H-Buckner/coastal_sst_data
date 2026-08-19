"""CLI (coastal_sst_data.cli). Tests drive `cli.main([...])` and stub the
underlying functions (run_pipeline / auth.verify) so nothing hits the network;
the offline subcommands (validate / grids) run for real against a temp config.

The `grids --plot` flag is tested only at the WIRING level here (the plotting
function is stubbed); the maps themselves are covered by test_plot.py."""

import pytest

from coastal_sst_data import cli
from coastal_sst_data.config import DataProduct, Project

CONFIG_YAML = """\
name: cli-test
output_dir: out
time:
  start_date: "2023-08-01"
  end_date: "2023-08-31"
auth:
  earthdata:
    auth_strategy: netrc
products:
  mur:
  bathymetry:
regions:
  - name: r
    areas:
      - name: a1
        center_lat: 45.5
        center_lon: -123.9
        buffer_ns_km: 8
        buffer_ew_km: 8
"""


@pytest.fixture
def config_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(CONFIG_YAML)
    return str(p)


# ---------------------------------------------------------------------------
# run: forwards parsed args to run_pipeline (stubbed).
# ---------------------------------------------------------------------------
def test_run_dispatches_with_defaults(config_file, monkeypatch):
    calls = {}
    def stub(project, **kw):
        calls["project"] = project
        calls.update(kw)
    monkeypatch.setattr(cli, "run_pipeline", stub)

    cli.main(["run", "--config", config_file, "--dry-run"])

    assert isinstance(calls["project"], Project)
    assert calls["dry_run"] is True
    assert calls["overwrite"] is False
    assert calls["aois"] is None and calls["products"] is None
    assert calls["verify_auth"] is None          # no --no-verify -> pipeline decides


def test_run_forwards_all_flags(config_file, monkeypatch):
    calls = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda project, **kw: calls.update(kw))

    cli.main(["run", "--config", config_file, "--aoi", "a1",
              "--products", "mur", "--overwrite", "--no-verify"])

    assert calls["aois"] == ["a1"]
    assert calls["products"] == [DataProduct.mur]
    assert calls["overwrite"] is True
    assert calls["verify_auth"] is False         # --no-verify
    assert calls["dry_run"] is False


def test_run_unknown_product_rejected(config_file, monkeypatch):
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: None)   # never reached
    with pytest.raises(SystemExit):
        cli.main(["run", "--config", config_file, "--products", "banana"])


# ---------------------------------------------------------------------------
# verify: dispatches to auth.verify (stubbed).
# ---------------------------------------------------------------------------
def test_verify_dispatches(config_file, monkeypatch, capsys):
    seen = {}
    def stub(project, products=None):
        seen["products"] = products
        return {"earthdata": "ok"}
    monkeypatch.setattr(cli.auth, "verify", stub)

    cli.main(["verify", "--config", config_file])

    assert seen["products"] is None
    assert "verified" in capsys.readouterr().out.lower()


def test_verify_products_filter(config_file, monkeypatch):
    seen = {}
    def stub(project, products=None):
        seen["products"] = products
        return {}
    monkeypatch.setattr(cli.auth, "verify", stub)

    cli.main(["verify", "--config", config_file, "--products", "bathymetry"])
    assert seen["products"] == [DataProduct.bathymetry]


def test_verify_failure_exits_nonzero(config_file, monkeypatch):
    def boom(project, products=None):
        raise RuntimeError("bad creds")
    monkeypatch.setattr(cli.auth, "verify", boom)
    with pytest.raises(SystemExit):
        cli.main(["verify", "--config", config_file])


# ---------------------------------------------------------------------------
# validate / grids: offline, run for real against the temp config.
# ---------------------------------------------------------------------------
def test_validate_prints_summary(config_file, capsys):
    cli.main(["validate", "--config", config_file])
    out = capsys.readouterr().out
    assert "Config OK" in out and "cli-test" in out
    assert "mur" in out and "bathymetry" in out
    assert "earthdata" in out                     # mur's resolved auth backend is shown


def test_grids_prints_each_aoi(config_file, capsys):
    cli.main(["grids", "--config", config_file])
    out = capsys.readouterr().out
    assert "a1" in out and "EPSG:" in out         # base text output (no --plot)


def test_assemble_subcommand_dispatches(config_file, monkeypatch):
    seen = {}
    def stub(project, *, aois=None, dry_run=False, overwrite=False, memory_budget_gb=None):
        seen.update(aois=aois, dry_run=dry_run, overwrite=overwrite,
                    memory_budget_gb=memory_budget_gb)
    monkeypatch.setattr("coastal_sst_data.processes.datacube.assemble", stub)

    cli.main(["assemble", "--config", config_file, "--aoi", "a1", "--overwrite"])

    assert seen["aois"] == ["a1"]
    assert seen["overwrite"] is True and seen["dry_run"] is False
    assert seen["memory_budget_gb"] is None      # unset -> detect, exactly as before


def test_assemble_forwards_an_explicit_memory_budget(config_file, monkeypatch):
    """The knob an orchestrator uses to DIVIDE the budget between concurrent AoIs -- and a
    user needs by hand when the machine is shared."""
    seen = {}
    def stub(project, *, aois=None, dry_run=False, overwrite=False, memory_budget_gb=None):
        seen.update(memory_budget_gb=memory_budget_gb)
    monkeypatch.setattr("coastal_sst_data.processes.datacube.assemble", stub)

    cli.main(["assemble", "--config", config_file, "--memory-budget-gb", "12.5"])

    assert seen["memory_budget_gb"] == 12.5


def test_run_forwards_assemble_flag(config_file, monkeypatch):
    calls = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda project, **kw: calls.update(kw))
    cli.main(["run", "--config", config_file, "--dry-run", "--assemble"])
    assert calls["assemble"] is True


def test_grids_plot_flag_wires_to_plotting(config_file, monkeypatch, capsys):
    """--plot forwards grids/plot-dir/show to plot_project_aois and reports paths."""
    from pathlib import Path
    seen = {}

    def stub(project, *, grids=None, out_dir=None, show=False):
        seen.update(out_dir=out_dir, show=show, n_grids=len(grids))
        return [Path("figs/aoi_overview.png")]

    monkeypatch.setattr("coastal_sst_data.plot.plot_project_aois", stub)
    cli.main(["grids", "--config", config_file, "--plot", "--plot-dir", "figs"])

    out = capsys.readouterr().out
    assert seen["out_dir"] == "figs" and seen["show"] is False
    assert seen["n_grids"] == 1                    # a1 gridded and passed through
    assert "a1" in out                             # base text still printed
    assert "Wrote map(s)" in out and "aoi_overview.png" in out


# ---------------------------------------------------------------------------
# check / --repair
# ---------------------------------------------------------------------------
def _tree_with_live_scratch(tmp_path):
    """An output tree holding one file another RUNNING job is in the middle of writing.

    Attributed to our parent process: on this host, so `os.kill` can answer for it, and
    alive, so the answer is the one that matters.
    """
    import os
    from coastal_sst_data import store

    root = tmp_path / "out"
    d = root / "MUR" / "aligned" / "a1"
    d.mkdir(parents=True)
    scratch = d / f"a1_20230801.nc{store.PART_SUFFIX}{store._HOST}-{os.getppid()}-1-1"
    scratch.write_bytes(b"half a day")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.replace("output_dir: out", f"output_dir: {root}"))
    return str(cfg), scratch


def test_check_reports_scratch_another_run_is_still_writing(tmp_path, capsys):
    config, scratch = _tree_with_live_scratch(tmp_path)
    cli.main(["check", "--config", config, "--quick"])

    out = capsys.readouterr().out
    assert "IN USE" in out and "still writing it" in out
    assert scratch.exists()


def test_check_refuses_to_repair_a_tree_with_a_live_write_in_it(tmp_path):
    """Deleting a running job's in-flight scratch is the exact failure this pass exists to
    prevent, so `--repair` declines rather than guessing."""
    config, scratch = _tree_with_live_scratch(tmp_path)
    with pytest.raises(SystemExit, match="Refusing to repair"):
        cli.main(["check", "--config", config, "--quick", "--repair"])
    assert scratch.exists()


def test_force_overrides_the_refusal(tmp_path, capsys):
    """For the case the liveness rule cannot get right on its own: a reboot leaves scratch
    whose owning pid has since been reused, so nothing ever calls it dead."""
    config, scratch = _tree_with_live_scratch(tmp_path)
    cli.main(["check", "--config", config, "--quick", "--repair", "--force"])

    assert not scratch.exists()
    assert "Removed 1 path(s)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Argument parsing errors.
# ---------------------------------------------------------------------------
def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.main([])


def test_missing_config_errors():
    with pytest.raises(SystemExit):
        cli.main(["validate"])

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])

# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def test_extract_subcommand_dispatches(config_file, monkeypatch):
    seen = {}
    def stub(project, **kw):
        seen.update(kw)
    monkeypatch.setattr("coastal_sst_data.processes.extract.extract", stub)

    cli.main(["extract", "--config", config_file, "--aoi", "a1",
              "--points", "sites.csv", "--format", "csv", "--overwrite"])

    assert seen["aois"] == ["a1"]
    assert seen["points_file"] == "sites.csv"
    assert seen["fmt"] == "csv" and seen["overwrite"] is True
    assert seen["dry_run"] is False and seen["out"] is None


def test_extract_format_flag_defaults_to_none(config_file, monkeypatch):
    """The config's `extract.format` must win when --format is absent.

    A `default='parquet'` here would silently override a config that says csv, on every
    invocation, and nothing would report it.
    """
    seen = {}
    monkeypatch.setattr("coastal_sst_data.processes.extract.extract",
                        lambda project, **kw: seen.update(kw))
    cli.main(["extract", "--config", config_file])
    assert seen["fmt"] is None


def test_extract_missing_pyarrow_exits_with_advice(config_file, monkeypatch):
    """The ImportError from store.write_table must become actionable advice, not a traceback."""
    def boom(project, **kw):
        raise ImportError("parquet output needs pyarrow, which is optional")
    monkeypatch.setattr("coastal_sst_data.processes.extract.extract", boom)

    with pytest.raises(SystemExit) as exc:
        cli.main(["extract", "--config", config_file])
    msg = str(exc.value)
    assert "coastal_sst_data[extract]" in msg and "--format csv" in msg
