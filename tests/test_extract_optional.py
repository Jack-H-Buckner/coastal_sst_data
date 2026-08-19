"""The `extract` stage's OPTIONALITY CONTRACT.

Only a minority of projects extract point time series, so the feature must cost the others
nothing: no config key they have to write, no dependency, no import, no pipeline stage.
That is a promise about the whole package rather than about one module, which is why it has
its own file -- each test below pins one clause, and a change that quietly breaks one is a
regression even though every extraction test still passes.
"""

import subprocess
import sys

import pytest

from coastal_sst_data import cli, pipeline, store
from coastal_sst_data.config import DataProduct, parse_config


# 1. The config block is optional ------------------------------------------- #
def test_a_config_with_no_extract_block_is_valid(base_project):
    """Every existing config must keep validating untouched."""
    project = parse_config(base_project)
    assert project.extract.channels == {}
    assert project.extract.points is None


def test_validate_says_nothing_about_extraction(base_project, tmp_path, capsys):
    """A project that never extracts must not be told about extraction."""
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(base_project))
    cli.main(["validate", "--config", str(p)])
    assert "extract" not in capsys.readouterr().out.lower()


def test_a_typo_in_the_block_name_still_fails_loudly(base_project):
    """`extra='forbid'` on Project must not have been loosened to accept the new block."""
    base_project["extrac"] = {"channels": {}}
    with pytest.raises(Exception, match="[Ee]xtra"):
        parse_config(base_project)


# 2/3. No new core dependency, no import cost -------------------------------- #
def test_importing_the_package_does_not_import_the_extract_stage():
    """The stage (and its table machinery) must stay off every other command's import path.

    Run in a FRESH interpreter: this test file has already imported the stage indirectly, so
    checking `sys.modules` in-process would prove nothing.
    """
    code = (
        "import coastal_sst_data, coastal_sst_data.cli, coastal_sst_data.pipeline, sys;"
        "print('coastal_sst_data.processes.extract' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# A subprocess preamble that makes `import pyarrow` fail exactly as it would on a machine
# that never installed it. It has to be a meta-path finder rather than a `sys.modules`
# entry: pandas imports pyarrow itself when it is present, so by the time a test runs it is
# already imported and deleting it proves nothing about a fresh install.
_BLOCK_PYARROW = """
import sys, importlib.abc
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("No module named 'pyarrow' (simulated)")
        return None
sys.meta_path.insert(0, _Block())
"""


def _without_pyarrow(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _BLOCK_PYARROW + code],
                          capture_output=True, text=True)


def test_the_package_imports_with_pyarrow_uninstalled():
    """Nothing may import it at module scope -- including the extract stage itself."""
    out = _without_pyarrow(
        "import coastal_sst_data, coastal_sst_data.cli, coastal_sst_data.store;"
        "from coastal_sst_data.processes import extract;"
        "print('ok')")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_csv_output_works_with_pyarrow_uninstalled(tmp_path):
    """The dependency-free path must stay dependency-free."""
    out = _without_pyarrow(
        "import pandas as pd; from coastal_sst_data import store;"
        f"print(store.write_table(pd.DataFrame({{'a': [1, 2]}}), r'{tmp_path}', 't', 'csv'))")
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "t.csv").exists()


def test_parquet_without_pyarrow_names_the_extra(tmp_path):
    """And it must say what to install, not silently write something else."""
    out = _without_pyarrow("\n".join([
        "import pandas as pd",
        "from coastal_sst_data import store",
        "try:",
        f"    store.write_table(pd.DataFrame({{'a': [1]}}), r'{tmp_path}', 't', 'parquet')",
        "except ImportError as e:",
        "    print('RAISED', e)",
    ]))
    assert "RAISED" in out.stdout and "pyarrow" in out.stdout
    assert not (tmp_path / "t.parquet").exists()


def test_points_module_adds_no_imports_of_its_own():
    """`points.py` must not drag in a backend the package did not already load.

    Asserted as a DELTA against a bare `import coastal_sst_data`, because the package's own
    __init__ already pulls xarray in and an absolute assertion would just be measuring that.
    """
    code = ("import sys, coastal_sst_data;"
            "before = set(sys.modules);"
            "import coastal_sst_data.points;"
            "print(sorted(m for m in set(sys.modules) - before if '.' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# 4. Nothing runs implicitly ------------------------------------------------- #
def test_extraction_is_not_a_product_or_a_pipeline_stage():
    """No registry entry, no dispatch table entry, so no run can reach it by accident."""
    assert "extract" not in {p.value for p in DataProduct}
    assert not any("extract" in str(m) for m in pipeline.PROCESS_MODULES.values())


def test_run_pipeline_does_not_call_extract(base_project, tmp_path, monkeypatch):
    """A normal `run` must not touch the stage even when the config declares channels."""
    base_project["output_dir"] = str(tmp_path)
    base_project["extract"] = {"points": "sites.csv",
                               "channels": {"mur_sst": {"stat": "nearest"}}}
    project = parse_config(base_project)

    called = []
    from coastal_sst_data.processes import extract as extract_mod
    monkeypatch.setattr(extract_mod, "extract", lambda *a, **k: called.append(1))
    pipeline.run_pipeline(project, dry_run=True, verify_auth=False)
    assert called == []


# 5. The cube is unchanged --------------------------------------------------- #
def test_the_stage_never_writes_into_a_cube(tmp_path, aoi_grid, cube_dir, points_csv):
    """It reads cubes and writes one table; the store must come back byte-identical."""
    import hashlib

    zpath = cube_dir / f"{aoi_grid.name}.zarr"

    def fingerprint():
        h = hashlib.sha256()
        for f in sorted(zpath.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(zpath)).encode())
                h.update(f.read_bytes())
        return h.hexdigest()

    before = fingerprint()
    project = parse_config({
        "name": "t", "output_dir": str(tmp_path),
        "time": {"start_date": "2023-06-01", "end_date": "2023-06-04"},
        "auth": {"earthdata": {"auth_strategy": "netrc"}},
        "products": {"mur": None},
        "regions": [{"name": "r", "areas": [
            {"name": "test_aoi", "center_lat": 45.52, "center_lon": -123.925,
             "buffer_ns_km": 8.0, "buffer_ew_km": 8.0}]}],
        "extract": {"points": str(points_csv), "format": "csv",
                    "channels": {"eco_sst": None, "elevation_cudem": None}},
    })
    from coastal_sst_data.processes import extract as extract_mod
    extract_mod.extract(project)
    assert (tmp_path / "extract" / "points.csv").exists()
    assert fingerprint() == before
