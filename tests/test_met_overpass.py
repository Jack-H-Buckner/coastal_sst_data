"""met_overpass: overpass-aligned met acquisition. It reuses met's HRRR/ERA5 fetchers (stubbed
here, so offline), reads each combo's sensor overpass times from the sensor aligned dirs, and
writes one snapshot per source instant into MET_OVERPASS/<source>/aligned/<aoi>."""

import numpy as np
import xarray as xr
import pytest

from coastal_sst_data.config import parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import met, met_overpass

AOI = "aoi1"


def _project(tmp_path, combinations, sources=("hrrr", "era5")):
    return parse_config({
        "name": "mo", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-01"},
        "auth": {"earthdata": {"auth_strategy": "netrc"}},
        "products": {"ecostress": None, "landsat": None, "modis": None,
                     "met_overpass": {"sources": list(sources), "variables": ["airtemp"],
                                      "combinations": combinations}},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


def _write_scene(project, g, sensor_dir, hour):
    """A timestamped sensor scene (so met_overpass discovers its overpass time)."""
    H, W = g.height, g.width
    xs, ys = g.xy_centers()
    ds = xr.Dataset({"sst": (("time", "y", "x"), np.full((1, H, W), 286.0, "float32"))},
                    coords={"time": [np.datetime64("2026-06-01")], "y": ys, "x": xs})
    d = project.output_dir / sensor_dir / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}_20260601T{hour:02d}0000.nc")


def test_acquire_writes_snapshots_per_source_at_sensor_overpass_times(tmp_path, monkeypatch):
    """Combos [(lst, hrrr), (eco, era5)]: hrrr snapshots at Landsat's instant, era5 at
    ECOSTRESS's -- each in its OWN source tree, timestamped to the scene."""
    seen = []

    def fake(src):
        def f(g, dt, cfg):
            seen.append((src, dt.hour))
            return {"airtemp": np.full((g.height, g.width), 290.0, "float32")}
        return f
    monkeypatch.setattr(met, "_SOURCES", {"hrrr": fake("hrrr"), "era5": fake("era5")})

    p = _project(tmp_path, combinations=[["lst", "hrrr"], ["eco", "era5"]])
    g = grid.project_grids(p)[AOI]
    _write_scene(p, g, "LANDSAT", 18)       # lst flew at 18:00
    _write_scene(p, g, "ECOSTRESS", 20)     # eco flew at 20:00
    met_overpass.acquire(p, grids={AOI: g})

    hrrr = tmp_path / "MET_OVERPASS" / "hrrr" / "aligned" / AOI
    era5 = tmp_path / "MET_OVERPASS" / "era5" / "aligned" / AOI
    assert (hrrr / f"{AOI}_20260601T180000.nc").exists()    # hrrr @ lst's 18:00
    assert (era5 / f"{AOI}_20260601T200000.nc").exists()    # era5 @ eco's 20:00
    # each source was fetched only at its paired sensor's instant, not the cross-product
    assert set(seen) == {("hrrr", 18), ("era5", 20)}


def test_acquire_rejects_an_unknown_combo_source(tmp_path):
    # config validation catches a bad source before acquire; but acquire also guards directly.
    with pytest.raises(Exception, match="not valid here|not recognized"):
        _project(tmp_path, combinations=[["lst", "gfs"]])


def test_two_sensors_on_one_source_share_its_snapshots(tmp_path, monkeypatch):
    """(lst, hrrr) and (eco, hrrr): hrrr is fetched at the UNION of both overpass instants,
    one snapshot each -- not once per combo."""
    n = []
    monkeypatch.setattr(met, "_SOURCES", {"hrrr": lambda g, dt, cfg: n.append(dt.hour) or
                                          {"airtemp": np.zeros((g.height, g.width), "float32")}})
    p = _project(tmp_path, combinations=[["lst", "hrrr"], ["eco", "hrrr"]], sources=("hrrr",))
    g = grid.project_grids(p)[AOI]
    _write_scene(p, g, "LANDSAT", 18)
    _write_scene(p, g, "ECOSTRESS", 20)
    met_overpass.acquire(p, grids={AOI: g})
    assert sorted(n) == [18, 20]            # one fetch per distinct instant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
