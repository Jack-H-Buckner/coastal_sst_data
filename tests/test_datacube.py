"""Datacube assembler. Builds tiny SYNTHETIC aligned per-product files in a temp
output tree, assembles them, and asserts the knit result: channel layout, the
clearest-overpass pick, the land-cover water mask + MUR fill, and that the Zarr
was written with the expected chunking/compressor. No network, no real data."""

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
