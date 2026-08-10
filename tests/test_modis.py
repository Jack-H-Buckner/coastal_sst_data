"""MODIS: config -> params, the Landsat coincidence filter, and the swath read +
pyresample nearest regrid onto the shared AoiGrid. All offline (no earthaccess).
The swath tests write a synthetic MODIS L2P NetCDF (root-level sst/quality/lat/lon
2D swath) and run the real read + resample."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data.processes import modis
from coastal_sst_data import auth
from .conftest import UniformDs


def _ds(eff):
    """The settings ONE AoI runs with. `eff["ds"]` is keyed by AoI, because every product
    now resolves its options per AoI (region override -> project default). This is a global
    product with no region-varying options, so every AoI resolves alike -- take any."""
    return next(iter(eff["ds"].values()))




# --- fake earthaccess granule (dict-like: what _select_granules reads) ------- #
def _granule(iso, night=False):
    return {"umm": {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": iso}}},
            "meta": {"native-id": ("X-N-Y" if night else "X-D-Y")}}


# --- synthetic MODIS L2P swath granule -------------------------------------- #
def write_modis_granule(path, bbox, *, sst_kelvin=290.0, quality=5, res_deg=0.01):
    """A minimal MODIS L2P NetCDF: sea_surface_temperature/quality_level on a 2D
    swath, with lat/lon covering `bbox` + margin. quality 5=best, 1=bad."""
    w, s, e, n = bbox
    lon2d, lat2d = np.meshgrid(np.arange(w - 0.1, e + 0.1, res_deg),
                               np.arange(s - 0.1, n + 0.1, res_deg))
    shape = (1,) + lon2d.shape
    ds = xr.Dataset(
        {"sea_surface_temperature": (("time", "nj", "ni"),
                                     np.full(shape, sst_kelvin, dtype="float32")),
         "quality_level": (("time", "nj", "ni"), np.full(shape, quality, dtype="int8"))},
        coords={"time": [np.datetime64("2023-08-15T18:56:19")],
                "lat": (("nj", "ni"), lat2d), "lon": (("nj", "ni"), lon2d)},
    )
    ds.to_netcdf(path, engine="netcdf4")
    return str(path)


# ---------------------------------------------------------------------------
# _build_eff: config -> acquisition params (modis needs earthdata auth).
# ---------------------------------------------------------------------------
def test_build_eff_maps_defaults(base_project):
    base_project["products"]["modis"] = None       # bare -> defaults
    eff = modis._build_eff(parse_config(base_project))
    assert _ds(eff)["short_name"] == modis.SHORT_NAME
    assert _ds(eff)["variable"] == modis.DEFAULT_VARIABLE
    assert _ds(eff)["quality_min"] == 4
    assert _ds(eff)["access"] == "download"
    assert _ds(eff)["match_landsat"] is True
    assert _ds(eff)["max_time_diff_minutes"] == 360
    assert _ds(eff)["footprint_id"] is True
    assert eff["earthdata"]["auth_strategy"] == "netrc"
    assert eff["out_dir"] == Path("path/to/data") / "MODIS" / "aligned"
    assert eff["landsat_dir"] == Path("path/to/data") / "LANDSAT" / "aligned"


def test_build_eff_requires_modis_selected(base_project):
    with pytest.raises(ValueError, match="modis is not a selected product"):
        modis._build_eff(parse_config(base_project))


def test_build_eff_applies_overrides(base_project):
    base_project["products"]["modis"] = {
        "quality_min": 5, "match_landsat": False, "max_time_diff_minutes": 30,
        "regrid_radius_m": 1000, "footprint_id": False, "output_format": "geotiff",
    }
    eff = modis._build_eff(parse_config(base_project))
    assert _ds(eff)["quality_min"] == 5
    assert _ds(eff)["match_landsat"] is False
    assert _ds(eff)["max_time_diff_minutes"] == 30
    assert _ds(eff)["regrid_radius_m"] == 1000
    assert _ds(eff)["footprint_id"] is False
    assert eff["fmt"] == "geotiff"


def test_build_eff_rejects_unknown_access(base_project):
    base_project["products"]["modis"] = {"access": "ftp"}
    with pytest.raises(ValueError, match="not recognized"):
        modis._build_eff(parse_config(base_project))


# ---------------------------------------------------------------------------
# _select_granules: daytime + Landsat coincidence (the configurable matchup).
# ---------------------------------------------------------------------------
_LS = [datetime(2023, 8, 15, 19, 0, 0)]
_MAXDT = timedelta(minutes=360)


def test_select_keeps_granule_within_time_window():
    g = [_granule("2023-08-15T18:56:00.000000Z")]           # 4 min from Landsat
    assert len(modis._select_granules(g, _LS, True, _MAXDT, True)) == 1


def test_select_drops_granule_outside_time_window():
    g = [_granule("2023-08-15T03:00:00.000000Z")]           # 16 h from Landsat
    assert modis._select_granules(g, _LS, True, _MAXDT, True) == []


def test_select_drops_night_granule():
    g = [_granule("2023-08-15T18:56:00.000000Z", night=True)]
    assert modis._select_granules(g, [], False, _MAXDT, True) == []


def test_select_daytime_only_false_keeps_night():
    g = [_granule("2023-08-15T18:56:00.000000Z", night=True)]
    assert len(modis._select_granules(g, [], False, _MAXDT, daytime_only=False)) == 1


def test_select_match_landsat_with_no_landsat_keeps_nothing():
    g = [_granule("2023-08-15T18:56:00.000000Z")]
    assert modis._select_granules(g, [], True, _MAXDT, True) == []


def test_select_full_series_keeps_all_daytime():
    g = [_granule("2023-08-15T18:56:00.000000Z"), _granule("2023-08-16T18:56:00.000000Z")]
    assert len(modis._select_granules(g, [], False, _MAXDT, True)) == 2


def test_select_returns_sorted_by_time():
    g = [_granule("2023-08-16T18:56:00.000000Z"), _granule("2023-08-15T18:56:00.000000Z")]
    kept = modis._select_granules(g, [], False, _MAXDT, True)
    assert kept[0][1] < kept[1][1]


# ---------------------------------------------------------------------------
# read_swath: quality masking + Kelvin/Celsius (real synthetic granule).
# ---------------------------------------------------------------------------
def test_read_swath_good_quality_kelvin(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, sst_kelvin=290.0, quality=5)
    sst, lat, lon = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    finite = sst[np.isfinite(sst)]
    assert finite.size and np.allclose(finite, 290.0)
    assert lat.ndim == 2 and lon.ndim == 2                  # 2D swath coords


def test_read_swath_masks_low_quality(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, quality=1)  # bad
    sst, _, _ = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    assert np.isnan(sst).all()                              # all below quality_min -> NaN


def test_read_swath_to_celsius(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, sst_kelvin=290.0)
    sst, _, _ = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, True)
    assert np.allclose(sst[np.isfinite(sst)], 290.0 - 273.15)


# ---------------------------------------------------------------------------
# resample_to_grid + _scene_dataset: swath -> shared grid, output schema.
# ---------------------------------------------------------------------------
def test_swath_resamples_onto_shared_grid(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, sst_kelvin=290.0)
    sst, lat, lon = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    fp = np.arange(sst.size, dtype="int32").reshape(sst.shape)
    sst_g, fp_g = modis.resample_to_grid(sst, lat, lon, aoi_grid, 1500.0, fp)
    assert sst_g.shape == aoi_grid.shape
    finite = sst_g[np.isfinite(sst_g)]
    assert finite.size and np.allclose(finite, 290.0, atol=0.5)   # nearest preserves value
    assert (fp_g >= 0).any()                                       # footprint ids assigned


def test_scene_dataset_schema(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox)
    sst, lat, lon = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    fp = np.arange(sst.size, dtype="int32").reshape(sst.shape)
    sst_g, fp_g = modis.resample_to_grid(sst, lat, lon, aoi_grid, 1500.0, fp)
    ds = modis._scene_dataset(sst_g, fp_g, aoi_grid,
                              pd.Timestamp("2023-08-15T18:56:19"), "test_aoi", False)
    assert {"sst", "valid", "footprint_id"} <= set(ds.data_vars)
    assert ds["sst"].attrs["units"] == "K"
    assert ds.sizes["y"] == aoi_grid.height and ds.sizes["x"] == aoi_grid.width
    assert "time" in ds.coords


def test_source_attr_names_the_configured_sensor(aoi_grid):
    """Configure Aqua and every file used to still claim Terra, because the attr was built
    from the module constant rather than the short_name the search actually used."""
    g = aoi_grid
    sst = np.full((g.height, g.width), 290.0, "float32")

    terra = modis._scene_dataset(sst, None, g, datetime(2023, 7, 15, 21, 0), "aoi", False)
    assert terra.attrs["source"] == f"GHRSST {modis.SHORT_NAME}"          # default

    aqua = modis._scene_dataset(sst, None, g, datetime(2023, 7, 15, 21, 0), "aoi", False,
                                short_name="MODIS_A-JPL-L2P-v2019.0")
    assert aqua.attrs["source"] == "GHRSST MODIS_A-JPL-L2P-v2019.0"       # follows config
    assert "MODIS_T" not in aqua.attrs["source"]                          # NOT Terra


# --------------------------------------------------------------------------- #
# The tmp-granule lifecycle. `earthaccess.download` skips a file that already exists BY
# NAME, so a truncated granule left by a killed run was handed straight back to the reader.
# --------------------------------------------------------------------------- #
class _Gran:
    def __init__(self, t="2023-07-15T21:00:00.000Z"):
        self._t = t

    def __getitem__(self, k):
        if k == "umm":
            return {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": self._t}}}
        return {"native-id": "MODIS-D-granule"}


def _modis_eff(tmp_path):
    return {
        "ds": UniformDs({
            "short_name": modis.SHORT_NAME, "variable": modis.DEFAULT_VARIABLE,
            "quality_min": 4, "regrid_radius_m": 1500.0, "access": "download",
            "match_landsat": False, "max_time_diff_minutes": 360,
            "daytime_only": True, "footprint_id": False}),
        "grid": {"to_celsius": False},
        "out_dir": tmp_path / "out", "tmp_dir": tmp_path / "_tmp",
        "landsat_dir": tmp_path / "LANDSAT",
        "fmt": "netcdf", "overwrite": False,
        "earthdata": {"auth_strategy": "netrc"},
        "time": {"start_date": "2023-07-15", "end_date": "2023-07-15"},
        "config_sha256": "x",
    }


def test_a_failed_granule_leaves_no_tmp_file_behind(monkeypatch, tmp_path, aoi_grid):
    """A partial download must not survive the attempt: the next run would see the name
    already present, skip the download, and read the truncated file."""
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis.earthaccess, "search_data", lambda **kw: [_Gran()])

    def fetch_then_die(granule, bbox, tmp_dir):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        p = tmp_dir / "granule.nc"
        p.write_bytes(b"half a granule")          # a partial download, as a kill would leave
        raise ConnectionError("connection reset mid-download")

    monkeypatch.setitem(modis._ACCESS, "download", fetch_then_die)

    eff = _modis_eff(tmp_path)
    rep = modis.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.failed == 1                        # counted as a LOSS, not a silent skip
    leftovers = list((tmp_path / "_tmp").rglob("*.nc"))
    assert leftovers == []                        # ...and nothing truncated survives


def test_a_successful_granule_also_cleans_up_its_scratch(monkeypatch, tmp_path, aoi_grid):
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis.earthaccess, "search_data", lambda **kw: [_Gran()])

    src = write_modis_granule(tmp_path / "src.nc", aoi_grid.search_bbox,
                              sst_kelvin=290.0, quality=5)

    def fetch(granule, bbox, tmp_dir):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / "granule.nc"
        dst.write_bytes(Path(src).read_bytes())
        return dst

    monkeypatch.setitem(modis._ACCESS, "download", fetch)

    eff = _modis_eff(tmp_path)
    rep = modis.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.written == 1 and rep.failed == 0
    assert list((tmp_path / "_tmp").rglob("*.nc")) == []     # scratch removed on success too

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])

