"""Provenance: the config that built a cube, and per field the source(s) it came from and
when they were accessed. The emphasis is on not LYING -- a date derived from a file mtime
must never be passed off as a recorded one, and a field must never ship with blank
provenance just because someone added a channel and forgot the mapping."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data import provenance
from coastal_sst_data.config import load_config, parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import datacube


EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"
AOI = "aoi1"


def _project(tmp_path):
    return parse_config({
        "name": "p", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-02"},
        "products": {"bathymetry": None, "tides": None},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


@pytest.fixture
def project(tmp_path):
    return _project(tmp_path)


@pytest.fixture
def g(project):
    return grid.project_grids(project)[AOI]


@pytest.fixture
def days(project):
    return pd.date_range(project.time.start_date, project.time.end_date, freq="D")


def _write_mur(project, g, days, *, stamped=True):
    xs, ys = g.xy_centers()
    H, W = g.height, g.width
    d = project.output_dir / "MUR" / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    for day in days:
        ds = xr.Dataset({"sst": (("time", "y", "x"), np.full((1, H, W), 285.0, "float32"))},
                        coords={"time": [day], "y": ys, "x": xs})
        ds.attrs["source"] = "GHRSST MUR-JPL-L4-GLOB-v4.1"
        if stamped:
            ds.attrs["acquired_at"] = "2026-07-14T09:31:02Z"
        ds.to_netcdf(d / f"{AOI}_{day.strftime('%Y%m%d')}.nc")


# --------------------------------------------------------------------------- #
# The config that built the cube
# --------------------------------------------------------------------------- #
def test_load_config_keeps_the_file_verbatim():
    """The cube embeds what you actually WROTE -- comments and all -- not a
    re-serialization of it."""
    cfg = load_config(EXAMPLE)
    assert cfg.config_text == EXAMPLE.read_text()
    assert cfg.config_path == str(EXAMPLE.resolve())
    assert cfg.config_sha256 == provenance.sha256_text(EXAMPLE.read_text())


def test_a_dict_built_project_is_still_self_describing(project):
    """No file to embed, so the validated model is serialized back to YAML."""
    assert project.config_path is None
    assert "name: p" in project.config_text
    assert len(project.config_sha256) == 64


def test_editing_the_config_changes_its_hash(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(EXAMPLE.read_text())
    before = load_config(f).config_sha256
    f.write_text(EXAMPLE.read_text() + "\n# a change\n")
    assert load_config(f).config_sha256 != before


# --------------------------------------------------------------------------- #
# Access dates: recorded vs guessed
# --------------------------------------------------------------------------- #
def test_a_stamped_file_reports_its_recorded_date(project, g, days):
    _write_mur(project, g, days, stamped=True)
    f = project.output_dir / "MUR" / "aligned" / AOI / f"{AOI}_20260601.nc"
    when, basis = provenance.access_of(f)
    assert when == "2026-07-14T09:31:02Z"
    assert basis == provenance.STAMPED


def test_an_unstamped_file_falls_back_to_mtime_and_says_so(project, g, days):
    """Data acquired before provenance existed has no stamp. Using the mtime is fine --
    silently presenting it as a recorded acquisition date is not."""
    _write_mur(project, g, days, stamped=False)
    f = project.output_dir / "MUR" / "aligned" / AOI / f"{AOI}_20260601.nc"
    when, basis = provenance.access_of(f)
    assert basis == provenance.FILE_MTIME
    assert when.endswith("Z") and len(when) >= 19          # a real timestamp, just a guess


def test_one_unstamped_file_downgrades_the_whole_products_basis(project, g, days):
    """A window is only as trustworthy as its weakest entry."""
    _write_mur(project, g, days, stamped=True)
    d = project.output_dir / "MUR" / "aligned" / AOI
    with xr.open_dataset(d / f"{AOI}_20260602.nc") as ds:
        bad = ds.load()
    del bad.attrs["acquired_at"]                            # one file loses its stamp
    bad.to_netcdf(d / f"{AOI}_20260602.nc", mode="w")

    rec = provenance.collect_product(d, "mur")
    assert rec["basis"] == provenance.FILE_MTIME
    assert rec["n_files"] == 2


def test_stamp_records_the_time_and_the_config_it_ran_under():
    st = provenance.stamp({"config_sha256": "abc123"})
    assert st["acquired_at"].endswith("Z")
    assert st["config_sha256"] == "abc123"
    assert st["package_version"]


# --------------------------------------------------------------------------- #
# field -> source mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,inputs", [
    ("mur_sst", ["mur"]),
    ("eco_sst", ["ecostress"]),
    ("lst_valid", ["landsat"]),
    ("cmems_thetao_10m", ["cmems"]),
    ("airtemp", ["met"]),
    ("tide", ["tides"]),
    ("depth_p25", ["bathymetry"]),
    # the derived ones: several genuine inputs, all of them load-bearing
    ("eco_water_elev", ["bathymetry", "tides", "datum", "ecostress"]),
    ("lst_airtemp", ["met", "landsat"]),
    ("modis_insitu_sst", ["insitu", "modis"]),
    ("landmask", ["landcover", "bathymetry"]),
    ("doy_sin", []),                                        # computed; no data source
])
def test_field_inputs(field, inputs):
    assert provenance.field_inputs(field) == inputs


def test_an_unmapped_field_is_logged_not_silently_blank(caplog):
    with caplog.at_level("WARNING"):
        assert provenance.field_inputs("some_new_channel") == []
    assert "no source mapping" in caplog.text


# --------------------------------------------------------------------------- #
# Through the assembler
# --------------------------------------------------------------------------- #
def test_every_field_in_the_cube_has_provenance(project, g, days):
    """A channel added later without a mapping must fail HERE rather than ship blank."""
    _write_mur(project, g, days)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    rec = json.loads(ds.attrs["provenance"])

    assert set(rec) == set(ds.data_vars)                    # nothing unaccounted for
    unmapped = [f for f, r in rec.items()
                if not r["inputs"] and f not in ("doy_sin", "doy_cos")]
    assert unmapped == [], f"fields with no source mapping: {unmapped}"


def test_the_cube_carries_the_config_and_the_sources(project, g, days):
    _write_mur(project, g, days)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    assert ds.attrs["config_sha256"] == project.config_sha256
    assert "name: p" in ds.attrs["config_yaml"]
    assert ds.attrs["created_at"].endswith("Z")

    prods = json.loads(ds.attrs["provenance_products"])
    assert prods["mur"]["sources"] == ["GHRSST MUR-JPL-L4-GLOB-v4.1"]
    assert prods["mur"]["n_files"] == 2
    assert prods["mur"]["basis"] == provenance.STAMPED

    fields = json.loads(ds.attrs["provenance"])
    assert fields["mur_sst"]["sources"] == ["GHRSST MUR-JPL-L4-GLOB-v4.1"]
    assert fields["mur_sst"]["accessed"] == "2026-07-14T09:31:02Z"


def test_a_guessed_date_stays_visible_in_the_cube(project, g, days, caplog):
    _write_mur(project, g, days, stamped=False)
    with caplog.at_level("WARNING"):
        ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert json.loads(ds.attrs["provenance"])["mur_sst"]["basis"] == provenance.FILE_MTIME
    assert "FILE MTIMES" in caplog.text


def test_provenance_survives_the_zarr_round_trip(project, g, days, tmp_path):
    _write_mur(project, g, days)
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)
    z = tmp_path / "c.zarr"
    datacube.write_zarr_safe(ds, z, datacube.build_encoding(ds, eff["compression"],
                                                            eff["chunks"]))
    back = xr.open_zarr(z)
    assert back.attrs["config_yaml"] == project.config_text     # byte-for-byte
    assert json.loads(back.attrs["provenance"])["mur_sst"]["inputs"] == ["mur"]
    back.close()


# --------------------------------------------------------------------------- #
# WHICH CODE built this cube. `package_version` is pinned in pyproject and never moves, so
# every cube ever built was stamped "0.0.1" whatever commit produced it -- while the code
# changed what the numbers MEAN (mur_valid once counted NN-filled pixels; depth was once
# fabricated zeros where the DEM was missing).
# --------------------------------------------------------------------------- #
def test_code_version_is_the_git_sha():
    v = provenance.code_version()
    bare = v.removesuffix("-dirty")
    assert v == "unknown" or (len(bare) == 40 and all(c in "0123456789abcdef" for c in bare))


def test_a_dirty_tree_says_so():
    """A cube built from uncommitted edits is not reproducible from ANY commit. Claiming a
    bare SHA for it would be the most dangerous kind of lie: the one a reader trusts."""
    import subprocess
    repo = Path(__file__).parents[1]
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    v = provenance.code_version()
    if v != "unknown":
        assert v.endswith("-dirty") == bool(dirty)


def test_stamp_carries_the_code_version():
    s = provenance.stamp({"config_sha256": "abc"})
    assert s["code_version"] == provenance.code_version()
    assert s["config_sha256"] == "abc"          # the config is still stamped too
    assert "acquired_at" in s
