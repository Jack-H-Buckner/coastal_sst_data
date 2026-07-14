"""Datacube assembler. Builds tiny SYNTHETIC aligned per-product files in a temp
output tree, assembles them, and asserts the knit result: channel layout, the
clearest-overpass pick, the land-cover water mask + MUR fill, and that the Zarr
was written with the expected chunking/compressor. No network, no real data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config, CompressionSpec
from coastal_sst_data import grid
from coastal_sst_data.processes import datacube


AOI = "aoi1"


@pytest.fixture
def project(tmp_path):
    """A one-AoI project pointed at a temp output_dir, over a 3-day window."""
    return parse_config({
        "name": "dc", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {"bathymetry": None},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


@pytest.fixture
def grids(project):
    return grid.project_grids(project)


@pytest.fixture
def days(project):
    return pd.date_range(project.time.start_date, project.time.end_date, freq="D")


def _write(project, sub, fname, ds):
    d = project.output_dir / sub / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / fname)


def _grid_hw(g):
    xs, ys = g.xy_centers()
    return g.height, g.width, xs, ys


# --------------------------------------------------------------------------- #
# Synthetic aligned writers
# --------------------------------------------------------------------------- #
def write_mur(project, g, days, *, water_hole_cols):
    """MUR daily files; a NaN hole in the given columns (to test the water fill)."""
    H, W, xs, ys = _grid_hw(g)
    for i, day in enumerate(days):
        arr = np.full((H, W), 285.0 + i, "float32")
        arr[:, water_hole_cols] = np.nan
        ds = xr.Dataset({"sst": (("time", "y", "x"), arr[None]),
                         "valid": (("time", "y", "x"), np.ones((1, H, W), "uint8"))},
                        coords={"time": [day], "y": ys, "x": xs})
        _write(project, "MUR", f"{AOI}_{day.strftime('%Y%m%d')}.nc", ds)


def write_ecostress_two_scenes(project, g, day):
    """Two ECOSTRESS scenes on one day; the 20:00 scene is clearer (more valid)."""
    H, W, xs, ys = _grid_hw(g)
    for hh, nan_frac in [(18, 0.5), (20, 0.0)]:
        sst = np.full((H, W), 286.0, "float32")
        if nan_frac:
            sst[: int(H * nan_frac)] = np.nan          # fewer valid px in the 18:00 scene
        ds = xr.Dataset({
            "sst": (("time", "y", "x"), sst[None]),
            "cloud": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
            "water": (("time", "y", "x"), np.zeros((1, H, W), "float32")),  # eco: <0.5 = water
            "quality": (("time", "y", "x"), np.zeros((1, H, W), "float32")),  # QC bits 0-1 = good
            "valid": (("time", "y", "x"), np.ones((1, H, W), "uint8")),
        }, coords={"time": [day], "y": ys, "x": xs})
        stamp = day.strftime("%Y%m%d") + f"T{hh:02d}0000"
        _write(project, "ECOSTRESS", f"{AOI}_{stamp}.nc", ds)


def write_landsat(project, g, day, hour, temp=288.0):
    """One Landsat scene. Landsat polarity: water = 1 (unlike ECOSTRESS), cloud reliable."""
    H, W, xs, ys = _grid_hw(g)
    ds = xr.Dataset({
        "sst": (("time", "y", "x"), np.full((1, H, W), temp, "float32")),
        "cloud": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
        "water": (("time", "y", "x"), np.ones((1, H, W), "float32")),
    }, coords={"time": [day], "y": ys, "x": xs})
    stamp = day.strftime("%Y%m%d") + f"T{hour:02d}0000"
    _write(project, "LANDSAT", f"{AOI}_{stamp}.nc", ds)


def write_modis(project, g, day, temp=287.0):
    H, W, xs, ys = _grid_hw(g)
    ds = xr.Dataset({"sst": (("time", "y", "x"), np.full((1, H, W), temp, "float32")),
                     "valid": (("time", "y", "x"), np.ones((1, H, W), "uint8"))},
                    coords={"time": [day], "y": ys, "x": xs})
    _write(project, "MODIS", f"{AOI}_{day.strftime('%Y%m%d')}T210000.nc", ds)


def write_bathymetry(project, g):
    H, W, xs, ys = _grid_hw(g)
    ds = xr.Dataset({"elevation": (("y", "x"), np.full((H, W), -10.0, "float32")),
                     "depth": (("y", "x"), np.full((H, W), 10.0, "float32")),
                     "depth_p25": (("y", "x"), np.full((H, W), 8.0, "float32")),
                     "depth_p75": (("y", "x"), np.full((H, W), 12.0, "float32"))},
                    coords={"y": ys, "x": xs})
    _write(project, "BATHYMETRY", f"{AOI}.nc", ds)


def write_landcover(project, g, *, land_cols):
    """Land-cover water everywhere except a land strip in the given columns."""
    H, W, xs, ys = _grid_hw(g)
    water = np.ones((H, W), "float32")
    water[:, land_cols] = 0.0
    ds = xr.Dataset({"landcover": (("y", "x"), np.full((H, W), 80, "int16")),
                     "water": (("y", "x"), water)}, coords={"y": ys, "x": xs})
    _write(project, "LANDCOVER", f"{AOI}.nc", ds)


# --------------------------------------------------------------------------- #
# assemble_aoi (the knit logic)
# --------------------------------------------------------------------------- #
def test_channel_layout_and_dims(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 3))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 5))
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)

    assert ds.sizes == {"time": len(days), "y": g.height, "x": g.width}
    for v in ["mur_sst", "eco_sst", "lst_sst", "modis_sst", "airtemp",
              "depth", "landmask", "landcover_water", "tide", "doy_sin"]:
        assert v in ds.data_vars
    assert ds["mur_sst"].dtype == np.float32
    assert ds["landmask"].dtype == np.uint8
    assert list(pd.to_datetime(ds["time"].values)) == list(days)


def test_landmask_from_landcover(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 5))     # cols 0-4 are land
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)
    lm = ds["landmask"].values
    assert (lm[:, :5] == 1).all()                          # land strip
    assert (lm[:, 5:] == 0).all()                          # water elsewhere


def test_mur_filled_over_landcover_water_only(project, grids, days):
    g = grids[AOI]
    # MUR hole spans cols 0-6; land-cover marks cols 0-4 land, 5+ water.
    write_mur(project, g, days, water_hole_cols=slice(0, 7))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 5))
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)
    mur0 = ds["mur_sst"].isel(time=0).values
    assert np.isfinite(mur0[:, 5:7]).all()                 # water hole filled
    assert np.isnan(mur0[:, :5]).all()                     # land hole NOT filled


def test_mur_fill_disabled(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 7))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 5))
    eff = datacube._build_eff(project)
    eff["fill_mur_water"] = False
    ds = datacube.assemble_aoi(g, eff, days)
    mur0 = ds["mur_sst"].isel(time=0).values
    assert np.isnan(mur0[:, :7]).all()                     # nothing filled


def test_clearest_overpass_is_kept(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    write_ecostress_two_scenes(project, g, days[0])        # 18:00 (half NaN) vs 20:00 (clear)
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)
    # The clearer 20:00 scene wins -> eco_hour == 20 and full valid coverage.
    assert ds["eco_hour"].isel(time=0).item() == pytest.approx(20.0)
    assert int(ds["eco_valid"].isel(time=0).values.sum()) == g.height * g.width


def test_modis_trusts_valid_layer(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    write_modis(project, g, days[1], temp=290.0)
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)
    assert np.nanmean(ds["modis_sst"].isel(time=1).values) == pytest.approx(290.0)
    assert int(ds["modis_valid"].isel(time=1).values.sum()) == g.height * g.width
    assert np.isnan(ds["modis_sst"].isel(time=0).values).all()   # no granule that day


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def test_build_encoding_shuffle_and_chunks():
    ds = xr.Dataset(
        {"f": (("time", "y", "x"), np.zeros((4, 10, 10), "float32")),
         "m": (("time", "y", "x"), np.zeros((4, 10, 10), "uint8"))},
        coords={"time": range(4), "y": range(10), "x": range(10)})
    enc = datacube.build_encoding(ds, CompressionSpec(), {"time": 2, "y": 8, "x": 8})
    # chunks clamped to axis length
    assert enc["f"]["chunks"] == (2, 8, 8)
    key = "compressors" if "compressors" in enc["f"] else "compressor"
    assert key in enc["m"]                                  # a codec was attached


# --------------------------------------------------------------------------- #
# assemble() end-to-end -> Zarr on disk
# --------------------------------------------------------------------------- #
def test_assemble_writes_zarr_with_compression(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 5))
    datacube.assemble(project, grids=grids, aois=[AOI])

    zpath = project.output_dir / "datacube" / f"{AOI}.zarr"
    assert zpath.exists()
    cube = xr.open_zarr(zpath)
    assert cube.sizes["time"] == len(days)
    assert cube["mur_sst"].dtype == np.float32 and cube["landmask"].dtype == np.uint8

    import zarr
    zg = zarr.open(str(zpath), mode="r")
    assert zg["mur_sst"].compressors            # a compressor is set (lossless float32)
    assert zg["mur_sst"].chunks[1:] == (128, 128) or zg["mur_sst"].chunks[1:] == (g.height, g.width)


def test_assemble_dry_run_writes_nothing(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_landcover(project, g, land_cols=slice(0, 0))
    datacube.assemble(project, grids=grids, aois=[AOI], dry_run=True)
    assert not (project.output_dir / "datacube").exists()


def test_assemble_skips_existing_without_overwrite(project, grids, days, caplog):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_landcover(project, g, land_cols=slice(0, 0))
    datacube.assemble(project, grids=grids, aois=[AOI])
    mtime = (project.output_dir / "datacube" / f"{AOI}.zarr").stat().st_mtime
    with caplog.at_level("INFO"):
        datacube.assemble(project, grids=grids, aois=[AOI])     # no overwrite
    assert "exists, skipping" in caplog.text
    # unchanged
    assert (project.output_dir / "datacube" / f"{AOI}.zarr").stat().st_mtime == mtime


def test_assemble_unknown_aoi_errors(project, grids):
    with pytest.raises(SystemExit, match="not found"):
        datacube.assemble(project, grids=grids, aois=["nope"])


# --------------------------------------------------------------------------- #
# write_zarr_safe: a run that dies mid-write must not leave a cube behind
# --------------------------------------------------------------------------- #
def _cube(n=4, h=8, w=8):
    return xr.Dataset(
        {"f": (("time", "y", "x"), np.arange(n * h * w, dtype="float32").reshape(n, h, w))},
        coords={"time": range(n), "y": range(h), "x": range(w)})


def _write_cube(ds, zpath):
    datacube.write_zarr_safe(ds, zpath, datacube.build_encoding(ds, CompressionSpec(), {}))


def test_write_zarr_safe_leaves_nothing_at_the_final_path_when_the_write_dies(tmp_path, monkeypatch):
    """The point of the whole exercise: a killed write must not park a partial cube at the
    final path, where the next run's exists() check would take it for a finished one."""
    zpath = tmp_path / "aoi.zarr"

    def die(self, store, *a, **kw):
        Path(store).mkdir()                   # half-write the scratch dir, then drop the wire
        (Path(store) / "chunk.0.0").write_text("truncated")
        raise ConnectionError("connection reset mid-write")

    monkeypatch.setattr(xr.Dataset, "to_zarr", die)
    with pytest.raises(ConnectionError):
        _write_cube(_cube(), zpath)

    assert not zpath.exists()                          # nothing at the final path
    assert not list(tmp_path.glob("*.tmp-*"))          # and the scratch was cleaned up


def test_write_zarr_safe_keeps_the_previous_cube_when_the_rewrite_dies(tmp_path, monkeypatch):
    """An overwrite that fails must leave the OLD cube intact -- losing a good cube to a
    failed refresh would be a worse outcome than not refreshing it."""
    zpath = tmp_path / "aoi.zarr"
    _write_cube(_cube(), zpath)
    before = xr.open_zarr(zpath)["f"].values.copy()

    def die(self, *a, **kw):
        raise ConnectionError("connection reset mid-write")

    monkeypatch.setattr(xr.Dataset, "to_zarr", die)
    with pytest.raises(ConnectionError):
        _write_cube(_cube() * 99, zpath)

    assert np.array_equal(xr.open_zarr(zpath)["f"].values, before)   # old cube survived
    assert not list(tmp_path.glob("*.old-*"))                        # not stranded aside


def test_write_zarr_safe_overwrites_and_cleans_up(tmp_path):
    zpath = tmp_path / "aoi.zarr"
    _write_cube(_cube(), zpath)
    _write_cube(_cube() + 100.0, zpath)

    assert xr.open_zarr(zpath)["f"].values[0, 0, 0] == 100.0        # new cube won
    assert not list(tmp_path.glob("*.tmp-*")) and not list(tmp_path.glob("*.old-*"))


def test_write_zarr_safe_sweeps_scratch_left_by_an_earlier_crash(tmp_path, caplog):
    zpath = tmp_path / "aoi.zarr"
    stale = tmp_path / "aoi.zarr.tmp-999-1"
    stale.mkdir()
    (stale / "junk").write_text("half a chunk")

    with caplog.at_level("WARNING"):
        _write_cube(_cube(), zpath)

    assert not stale.exists()
    assert "partial cube" in caplog.text            # and the user is TOLD a run had died
    assert xr.open_zarr(zpath).sizes["time"] == 4


# --------------------------------------------------------------------------- #
# Met: reference time of day, and forcing at each sensor's own overpass
# --------------------------------------------------------------------------- #
def write_met_daily(project, g, days, *, temp=280.0, prefix=""):
    """Met daily-mean (prefix='') or reference-time (prefix='ref_') files."""
    H, W, xs, ys = _grid_hw(g)
    for day in days:
        ds = xr.Dataset(
            {"airtemp": (("time", "y", "x"), np.full((1, H, W), temp, "float32")),
             "wind_speed": (("time", "y", "x"), np.full((1, H, W), 3.0, "float32")),
             "swrad": (("time", "y", "x"), np.full((1, H, W), 500.0, "float32")),
             "cloud_cover": (("time", "y", "x"), np.full((1, H, W), 10.0, "float32"))},
            coords={"time": [day], "y": ys, "x": xs})
        _write(project, "MET", f"{AOI}_{prefix}{day.strftime('%Y%m%d')}.nc", ds)


def write_met_snapshot(project, g, day, hour, *, temp):
    """A met snapshot at one overpass instant."""
    H, W, xs, ys = _grid_hw(g)
    stamp = day.strftime("%Y%m%d") + f"T{hour:02d}0000"
    ds = xr.Dataset(
        {"airtemp": (("time", "y", "x"), np.full((1, H, W), temp, "float32")),
         "wind_speed": (("time", "y", "x"), np.full((1, H, W), 7.0, "float32")),
         "swrad": (("time", "y", "x"), np.full((1, H, W), 800.0, "float32")),
         "cloud_cover": (("time", "y", "x"), np.full((1, H, W), 20.0, "float32"))},
        coords={"time": [day], "y": ys, "x": xs})
    _write(project, "MET", f"{AOI}_{stamp}.nc", ds)


def test_met_comes_from_the_reference_snapshot_not_the_daily_mean(project, grids, days):
    g = grids[AOI]
    write_met_daily(project, g, days, temp=280.0)                 # daily mean
    write_met_daily(project, g, days, temp=291.0, prefix="ref_")  # 10:30 reference
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.nanmean(ds["airtemp"].isel(time=0).values) == pytest.approx(291.0)
    assert ds.attrs["met_time"] == "reference"


def test_daily_mean_can_still_be_selected(project, grids, days):
    g = grids[AOI]
    write_met_daily(project, g, days, temp=280.0)
    write_met_daily(project, g, days, temp=291.0, prefix="ref_")
    eff = datacube._build_eff(project)
    eff["met_time"] = "daily_mean"
    ds = datacube.assemble_aoi(g, eff, days)
    assert np.nanmean(ds["airtemp"].isel(time=0).values) == pytest.approx(280.0)
    assert ds.attrs["met_time"] == "daily_mean"


def test_met_falls_back_when_no_reference_files_exist(project, grids, days, caplog):
    """An older MET tree has no reference snapshots -- use the daily mean rather than
    emitting an all-NaN forcing channel."""
    g = grids[AOI]
    write_met_daily(project, g, days, temp=280.0)
    with caplog.at_level("WARNING"):
        ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.nanmean(ds["airtemp"].isel(time=0).values) == pytest.approx(280.0)
    assert ds.attrs["met_time"] == "daily_mean"


def test_each_sensor_gets_the_forcing_from_its_own_overpass(project, grids, days):
    """Two sensors, hours apart on one day, must not share one met value."""
    g = grids[AOI]
    write_met_daily(project, g, days, temp=291.0, prefix="ref_")
    write_ecostress_two_scenes(project, g, days[0])           # clearest scene is 20:00
    write_landsat(project, g, days[0], hour=18)
    write_met_snapshot(project, g, days[0], 20, temp=286.0)   # the ECOSTRESS instant
    write_met_snapshot(project, g, days[0], 18, temp=299.0)   # the Landsat instant

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.nanmean(ds["eco_airtemp"].isel(time=0).values) == pytest.approx(286.0)
    assert np.nanmean(ds["lst_airtemp"].isel(time=0).values) == pytest.approx(299.0)
    # ...and the daily channel is still the reference time, independent of both.
    assert np.nanmean(ds["airtemp"].isel(time=0).values) == pytest.approx(291.0)
    # a day with no scene has no overpass forcing (not a stale value)
    assert np.isnan(ds["eco_airtemp"].isel(time=1).values).all()


def test_overpass_met_follows_the_scene_the_cube_actually_kept(project, grids, days):
    """ECOSTRESS flies twice; the cube keeps the CLEAREST (20:00). The forcing must come
    from that scene's instant, not from the discarded 18:00 one."""
    g = grids[AOI]
    write_ecostress_two_scenes(project, g, days[0])           # 18:00 (half NaN), 20:00 (clear)
    write_met_snapshot(project, g, days[0], 18, temp=270.0)   # discarded scene
    write_met_snapshot(project, g, days[0], 20, temp=295.0)   # kept scene
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert ds["eco_hour"].isel(time=0).item() == pytest.approx(20.0)
    assert np.nanmean(ds["eco_airtemp"].isel(time=0).values) == pytest.approx(295.0)


def test_overpass_met_is_configurable(project, grids, days):
    g = grids[AOI]
    write_ecostress_two_scenes(project, g, days[0])
    write_met_snapshot(project, g, days[0], 20, temp=295.0)

    eff = datacube._build_eff(project)
    eff["overpass_met"] = ["airtemp"]                         # only this one
    ds = datacube.assemble_aoi(g, eff, days)
    assert "eco_airtemp" in ds.data_vars
    assert "eco_swrad" not in ds.data_vars

    eff["overpass_met"] = []                                  # disabled entirely
    ds = datacube.assemble_aoi(g, eff, days)
    assert not [v for v in ds.data_vars if v.startswith("eco_") and "airtemp" in v]
