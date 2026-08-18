"""MODIS_REF: the Landsat-coincident calibration reference.

Same swath machinery as `processes.modis` (imported, not forked), so this file covers only
what is different: the coincidence filter, the footprint layer, the daytime default, and the
refusal to combine server-side subsetting with footprint ids. All offline.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data.processes import modis, modis_ref
from coastal_sst_data import auth, naming
from .conftest import UniformDs
from .test_modis import write_modis_granule


def _ds(eff):
    return next(iter(eff["ds"].values()))


def _granule(iso, night=False):
    return {"umm": {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": iso}}},
            "meta": {"native-id": ("X-N-Y" if night else "X-D-Y"),
                     "concept-id": "G1", "collection-concept-id": "C1"}}


# ---------------------------------------------------------------------------
# _build_eff: config -> acquisition params.
# ---------------------------------------------------------------------------
def test_build_eff_maps_defaults(base_project):
    base_project["products"]["modis_ref"] = None       # bare -> defaults
    eff = modis_ref._build_eff(parse_config(base_project))
    # Terra: its 10:30 crossing sits within minutes of Landsat's, which is the whole point.
    assert _ds(eff)["short_name"] == modis_ref.SHORT_NAME
    assert "MODIS_T" in _ds(eff)["short_name"]
    assert _ds(eff)["match_landsat"] is True
    assert _ds(eff)["max_time_diff_minutes"] == 360
    assert _ds(eff)["footprint_id"] is True
    # DAYTIME, the opposite default from the standalone sensor -- match what is calibrated.
    assert _ds(eff)["time_of_day"] == "day"
    # Whole granules: the footprint ids index the NATIVE swath.
    assert _ds(eff)["access"] == "download"
    assert eff["out_dir"] == Path("path/to/data") / "MODIS_REF" / "aligned"
    assert eff["landsat_dir"] == Path("path/to/data") / "LANDSAT" / "aligned"


def test_build_eff_requires_modis_ref_selected(base_project):
    with pytest.raises(ValueError, match="modis_ref is not a selected product"):
        modis_ref._build_eff(parse_config(base_project))


def test_build_eff_applies_overrides(base_project):
    base_project["products"]["modis_ref"] = {
        "quality_min": 5, "match_landsat": False, "max_time_diff_minutes": 30,
        "regrid_radius_m": 1000, "footprint_id": False, "output_format": "geotiff",
        "short_name": "MODIS_A-JPL-L2P-v2019.0",
    }
    eff = modis_ref._build_eff(parse_config(base_project))
    assert _ds(eff)["quality_min"] == 5
    assert _ds(eff)["match_landsat"] is False
    assert _ds(eff)["max_time_diff_minutes"] == 30
    assert _ds(eff)["regrid_radius_m"] == 1000
    assert _ds(eff)["footprint_id"] is False
    assert eff["fmt"] == "geotiff"


def test_harmony_and_footprints_are_refused_together(base_project):
    """A Harmony subset TRIMS the swath, so `arange(sst.size)` indexes the SUBSET, not the
    granule. Two AoIs would then write the same id for different native observations and the
    footprint-median matchup this product exists for would silently group the wrong cells."""
    base_project["products"]["modis_ref"] = {"access": "harmony", "footprint_id": True}
    with pytest.raises(ValueError, match="renumbers the native swath indices"):
        modis_ref._build_eff(parse_config(base_project))


def test_harmony_is_allowed_when_footprints_are_off(base_project):
    """The refusal is about the ids, not about Harmony -- turn them off and it is fine."""
    base_project["products"]["modis_ref"] = {"access": "harmony", "footprint_id": False}
    eff = modis_ref._build_eff(parse_config(base_project))
    assert _ds(eff)["access"] == "harmony"


# ---------------------------------------------------------------------------
# The coincidence filter.
# ---------------------------------------------------------------------------
_LS = [datetime(2023, 8, 15, 19, 0, 0)]
_MAXDT = timedelta(minutes=360)


def _kept(isos):
    """Granules already past the time-of-day filter, as [(granule, t), ...]."""
    return [(_granule(i), datetime.strptime(i, "%Y-%m-%dT%H:%M:%S.%fZ")) for i in isos]


def test_keeps_a_granule_within_the_window():
    got = modis_ref.select_coincident(_kept(["2023-08-15T18:56:00.000000Z"]), _LS, _MAXDT)
    assert len(got) == 1                                  # 4 min from Landsat


def test_drops_a_granule_outside_the_window():
    got = modis_ref.select_coincident(_kept(["2023-08-15T03:00:00.000000Z"]), _LS, _MAXDT)
    assert got == []                                      # 16 h from Landsat


def test_with_no_landsat_scenes_nothing_is_coincident():
    got = modis_ref.select_coincident(_kept(["2023-08-15T18:56:00.000000Z"]), [], _MAXDT)
    assert got == []


def test_landsat_times_are_read_through_the_shared_naming_convention(tmp_path):
    """This is the one place one product parses another's filenames. If the two ever drift,
    every day silently becomes a non-matchup rather than an error."""
    d = tmp_path / "aoi1"
    d.mkdir(parents=True)
    t = datetime(2023, 8, 15, 18, 56, 0)
    (d / f"{naming.time_stem('aoi1', t)}.nc").write_bytes(b"")
    (d / "aoi1_20230815.nc").write_bytes(b"")             # a per-DAY file, not an overpass
    assert modis_ref._landsat_times(tmp_path, "aoi1") == [t]


def test_missing_landsat_tree_is_not_an_error(tmp_path):
    assert modis_ref._landsat_times(tmp_path, "nope") == []


# ---------------------------------------------------------------------------
# The run loop: footprints, output tree, and the full-series escape hatch.
# ---------------------------------------------------------------------------
class _Gran:
    """A DAYTIME granule over Puget Sound: 18:45 UTC is 10:35 local solar."""

    def __init__(self, t="2023-07-15T18:45:00.000Z"):
        self._t = t

    def __getitem__(self, k):
        if k == "umm":
            return {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": self._t}}}
        return {"native-id": "MODIS-D-granule", "concept-id": "G1",
                "collection-concept-id": "C1"}


def _eff(tmp_path, **over):
    ds = {"short_name": modis_ref.SHORT_NAME, "variable": modis.DEFAULT_VARIABLE,
          "quality_min": 4, "regrid_radius_m": 1500.0, "access": "download",
          "match_landsat": False, "max_time_diff_minutes": 360,
          "time_of_day": "day", "night_solar_hours": modis.DEFAULT_NIGHT_SOLAR_HOURS,
          "footprint_id": True}
    ds.update(over)
    return {
        "ds": UniformDs(ds),
        "grid": {"to_celsius": False},
        "out_dir": tmp_path / "MODIS_REF" / "aligned",
        "landsat_dir": tmp_path / "LANDSAT" / "aligned",
        "tmp_dir": tmp_path / "MODIS_REF" / "_tmp",
        "fmt": "netcdf", "overwrite": False,
        "earthdata": {"auth_strategy": "netrc"},
        "time": {"start_date": "2023-07-15", "end_date": "2023-07-15"},
        "config_sha256": "x",
    }


def _stub(monkeypatch, tmp_path, aoi_grid):
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis_ref.earthaccess, "search_data", lambda **kw: [_Gran()])
    src = write_modis_granule(tmp_path / "src.nc", aoi_grid.search_bbox,
                              sst_kelvin=290.0, quality=5)

    def fetch(granule, bbox, tmp_dir, *, variables=None):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / "granule.nc"
        dst.write_bytes(Path(src).read_bytes())
        return dst

    monkeypatch.setitem(modis._ACCESS, "download", fetch)


def test_a_scene_lands_in_the_flat_tree_with_footprints(monkeypatch, tmp_path, aoi_grid):
    _stub(monkeypatch, tmp_path, aoi_grid)
    rep = modis_ref.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False)

    assert rep.written == 1 and rep.failed == 0
    files = list((tmp_path / "MODIS_REF" / "aligned").rglob("*.nc"))
    assert len(files) == 1
    # FLAT, not per-platform: this product has one collection, not a stack.
    assert files[0].parent.name == aoi_grid.name
    with xr.open_dataset(files[0]) as ds:
        assert "footprint_id" in ds.data_vars
        assert ds["footprint_id"].dtype == np.int32
        assert ds.attrs["source"] == f"GHRSST {modis_ref.SHORT_NAME}"


def test_footprints_can_be_turned_off(monkeypatch, tmp_path, aoi_grid):
    _stub(monkeypatch, tmp_path, aoi_grid)
    modis_ref.run(_eff(tmp_path, footprint_id=False), {aoi_grid.name: aoi_grid}, None, False)
    files = list((tmp_path / "MODIS_REF" / "aligned").rglob("*.nc"))
    with xr.open_dataset(files[0]) as ds:
        assert "footprint_id" not in ds.data_vars


def test_match_landsat_with_no_landsat_acquires_nothing(monkeypatch, tmp_path, aoi_grid):
    """Running before Landsat must produce an empty tree and a warning, never a silently
    unfiltered full time series -- which is what the reference is explicitly NOT."""
    _stub(monkeypatch, tmp_path, aoi_grid)
    rep = modis_ref.run(_eff(tmp_path, match_landsat=True), {aoi_grid.name: aoi_grid},
                        None, False)
    assert rep.written == 0
    assert list((tmp_path / "MODIS_REF" / "aligned").rglob("*.nc")) == []


def test_full_series_turns_the_coincidence_filter_off(base_project, monkeypatch, aoi_grid,
                                                     tmp_path):
    base_project["output_dir"] = str(tmp_path)
    base_project["products"]["modis_ref"] = {"match_landsat": True}
    project = parse_config(base_project)
    _stub(monkeypatch, tmp_path, aoi_grid)

    seen = {}
    monkeypatch.setattr(modis_ref, "run",
                        lambda eff, *a, **kw: seen.update(eff["ds"]) or None)
    modis_ref.acquire(project, grids={aoi_grid.name: aoi_grid}, full_series=True)
    assert all(c["match_landsat"] is False for c in seen.values())


def test_a_failed_granule_leaves_no_tmp_file_behind(monkeypatch, tmp_path, aoi_grid):
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis_ref.earthaccess, "search_data", lambda **kw: [_Gran()])

    def fetch_then_die(granule, bbox, tmp_dir, *, variables=None):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "granule.nc").write_bytes(b"half a granule")
        raise ConnectionError("connection reset mid-download")

    monkeypatch.setitem(modis._ACCESS, "download", fetch_then_die)

    rep = modis_ref.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False)
    assert rep.failed == 1
    assert list((tmp_path / "MODIS_REF" / "_tmp").rglob("*.nc")) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
