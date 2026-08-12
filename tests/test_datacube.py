"""Datacube assembler. Builds tiny SYNTHETIC aligned per-product files in a temp
output tree, assembles them, and asserts the knit result: channel layout, the
clearest-overpass pick, the land-cover water mask + MUR fill, and that the Zarr
was written with the expected chunking/compressor. No network, no real data."""

import hashlib
import json
import os
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


def _write_eco(project, fname, ds, tag="v002"):
    """ECOSTRESS is STACKED per collection version (D10): each writes its own
    ECOSTRESS/<tag>/aligned/<aoi> tree, and the cube ships one channel-set per version
    (eco_sst_<tag>, ...). The default tag mirrors the single-collection behaviour."""
    d = project.output_dir / "ECOSTRESS" / tag / "aligned" / AOI
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


def write_ecostress_two_scenes(project, g, day, tag="v002"):
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
        _write_eco(project, f"{AOI}_{stamp}.nc", ds, tag=tag)


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


def write_modis(project, g, day, temp=287.0, *, footprint=True, block=3, temp_step=0.0):
    """One MODIS scene, with the `footprint_id` layer MODIS writes by default.

    The ids are laid out as `block` x `block` tiles, which is the shape a real nearest-neighbour
    resample from a ~1 km swath onto a fine grid produces: many grid cells per native pixel.
    `temp_step` makes SST vary per footprint (block-constant, as that resample always is), so a
    footprint-grouped average has something to average; the 0.0 default keeps SST constant for
    the tests that predate the layer. `footprint=False` writes the pre-feature file (sst+valid),
    which is what a granule acquired with `footprint_id: false` looks like.
    """
    H, W, xs, ys = _grid_hw(g)
    rows, cols = np.arange(H)[:, None], np.arange(W)[None, :]
    fp = (rows // block) * ((W + block - 1) // block) + (cols // block)
    fp = np.broadcast_to(fp, (H, W)).astype("int32")
    data = {"sst": (("time", "y", "x"), (temp + temp_step * fp)[None].astype("float32")),
            "valid": (("time", "y", "x"), np.ones((1, H, W), "uint8"))}
    if footprint:
        data["footprint_id"] = (("time", "y", "x"), fp[None])
    ds = xr.Dataset(data, coords={"time": [day], "y": ys, "x": xs})
    _write(project, "MODIS", f"{AOI}_{day.strftime('%Y%m%d')}T210000.nc", ds)


def write_cmems(project, g, days, *, land_cols, src="my_global"):
    """One CMEMS source tag's daily files (CMEMS/<src>/aligned/<aoi>). The ~9 km model's land
    mask swallows `land_cols` (NaN there)."""
    H, W, xs, ys = _grid_hw(g)
    d = project.output_dir / "CMEMS" / src / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    for day in days:
        arr = np.full((H, W), 284.0, "float32")
        arr[:, land_cols] = np.nan                     # model never resolved these cells
        ds = xr.Dataset({"thetao_0m": (("time", "y", "x"), arr[None]),
                         "valid": (("time", "y", "x"),
                                   np.isfinite(arr)[None].astype("uint8"))},
                        coords={"time": [day], "y": ys, "x": xs})
        ds.to_netcdf(d / f"{AOI}_{day.strftime('%Y%m%d')}.nc")


def write_bathymetry(project, g, src="cudem", *, datum_offset_m=1.3, datum_status="ok"):
    """One DEM source's aligned output (BATHYMETRY/<src>/aligned/<aoi>), datum stamped on."""
    H, W, xs, ys = _grid_hw(g)
    ds = xr.Dataset({"elevation": (("y", "x"), np.full((H, W), -10.0, "float32")),
                     "depth": (("y", "x"), np.full((H, W), 10.0, "float32")),
                     "depth_p25": (("y", "x"), np.full((H, W), 8.0, "float32")),
                     "depth_p75": (("y", "x"), np.full((H, W), 12.0, "float32"))},
                    coords={"y": ys, "x": xs})
    ds.attrs.update(bathy_source=src, datum_offset_m=datum_offset_m, datum_status=datum_status,
                    datum_method="stub",
                    dem_vertical_datum="NAVD88" if src == "cudem" else "MSL_APPROX")
    d = project.output_dir / "BATHYMETRY" / src / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}.nc")


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
    write_ecostress_two_scenes(project, g, days[0])   # stacked sensor: emits eco_sst_v002 only
    write_bathymetry(project, g)                       # when a version tree is actually present
    write_landcover(project, g, land_cols=slice(0, 5))
    write_tides(project, g, days)
    write_met_daily(project, g, days, prefix="ref_")
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)

    assert ds.sizes == {"time": len(days), "y": g.height, "x": g.width}
    for v in ["mur_sst", "eco_sst_v002", "lst_sst", "modis_sst", "airtemp_hrrr",
              "elevation_cudem", "depth_cudem", "landcover_water", "tide_coops", "doy_sin"]:
        assert v in ds.data_vars
    # A stacked-data sensor emits per-version channels, not a bare `eco_sst`.
    assert "eco_sst" not in ds.data_vars
    # S1 removed these derived/fill channels; S3 made bathymetry per-source (no bare names).
    for gone in ["landmask", "mur_valid", "mur_filled", "eco_water_elev", "eco_tide",
                 "elevation", "depth"]:
        assert gone not in ds.data_vars
    # Each DEM source's datum offset rides on its own elevation channel (S3).
    assert ds["elevation_cudem"].attrs["datum_offset_m"] == pytest.approx(1.3)
    assert ds["mur_sst"].dtype == np.float32
    assert ds["landcover_water"].dtype == np.uint8
    assert list(pd.to_datetime(ds["time"].values)) == list(days)


def test_mur_ships_raw_observed_values_with_honest_nan_gaps(project, grids, days):
    """Goal 3 / S1: the cube no longer NN-fills MUR. A hole in the input stays NaN in the
    cube, and there is no mur_valid / mur_filled channel -- filling is a downstream call."""
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 7))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 5))
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    mur0 = ds["mur_sst"].isel(time=0).values
    assert np.isnan(mur0[:, :7]).all()                     # the hole is untouched
    assert np.isfinite(mur0[:, 7:]).all()                  # observed cells survive
    assert "mur_valid" not in ds.data_vars
    assert "mur_filled" not in ds.data_vars


def test_no_bathymetry_means_no_channels_not_a_fabricated_sea_level(project, grids, days):
    """With no DEM acquired, the cube ships NO bathymetry channels -- an ABSENT product, not a
    fabricated one. (The old bug shipped a flawless 'everything at sea level' DEM.)"""
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_landcover(project, g, land_cols=slice(0, 5))
    # deliberately NO write_bathymetry(...)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert not [v for v in ds.data_vars if v.startswith(("elevation", "depth"))]


def test_a_dem_without_usable_elevation_gives_nan_not_zeros(project, grids, days, caplog):
    """The worst audit bug: `np.where(elev < 0, -elev, 0.0)` took the 0.0 branch for every
    cell when elev was all-NaN, fabricating a NaN-free sea-level DEM. A per-source file that
    lacks a usable elevation must yield NaN, not zeros."""
    g = grids[AOI]
    H, W, xs, ys = _grid_hw(g)
    d = project.output_dir / "BATHYMETRY" / "cudem" / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    # a file with NO usable elevation (and no depth) -> nothing to derive depth from
    xr.Dataset({"junk": (("y", "x"), np.zeros((H, W), "float32"))},
               coords={"y": ys, "x": xs}).to_netcdf(d / f"{AOI}.nc")
    with caplog.at_level("WARNING"):
        ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.isnan(ds["elevation_cudem"].values).all()
    assert np.isnan(ds["depth_cudem"].values).all(), "depth was fabricated, not NaN"
    assert "no usable `elevation`" in caplog.text


def test_clearest_overpass_is_kept(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    write_ecostress_two_scenes(project, g, days[0])        # 18:00 (half NaN) vs 20:00 (clear)
    eff = datacube._build_eff(project)
    ds = datacube.assemble_aoi(g, eff, days)
    # The clearer 20:00 scene wins -> eco_hour_v002 == 20 and full valid coverage.
    assert ds["eco_hour_v002"].isel(time=0).item() == pytest.approx(20.0)
    assert int(ds["eco_valid_v002"].isel(time=0).values.sum()) == g.height * g.width


# --------------------------------------------------------------------------- #
# STACKED-DATA sensor: ECOSTRESS collection versions (D10) get one channel-set PER version,
# but the matchups (met/tide/in-situ) key off ONE merged overpass identity (D5).
# --------------------------------------------------------------------------- #
def _eco_versions_project(tmp_path, versions):
    """A one-AoI project selecting ECOSTRESS with an ordered `versions` list (the first-listed
    version wins the merged overpass identity per day)."""
    return parse_config({
        "name": "dc", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "products": {"ecostress": {"versions": list(versions)}},
        "auth": {"earthdata": {"auth_strategy": "netrc"}},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


def test_each_ecostress_version_gets_its_own_channel_set(tmp_path, days):
    """The whole point of the feature: v002 and v003 each ship their own eco_sst_<ver>/valid/
    cloud/hour, discovered from their own tree -- a bare `eco_sst` never appears."""
    project = _eco_versions_project(tmp_path, ["v003", "v002"])
    g = grid.project_grids(project)[AOI]
    write_ecostress(project, g, days[0], hour=20, sst=286.0, tag="v003")
    write_ecostress(project, g, days[0], hour=18, sst=290.0, tag="v002")

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    for ver, sst, hour in [("v003", 286.0, 20.0), ("v002", 290.0, 18.0)]:
        assert f"eco_sst_{ver}" in ds.data_vars
        assert np.nanmean(ds[f"eco_sst_{ver}"].isel(time=0).values) == pytest.approx(sst)
        assert ds[f"eco_hour_{ver}"].isel(time=0).item() == pytest.approx(hour)
        assert int(ds[f"eco_valid_{ver}"].isel(time=0).values.sum()) == g.height * g.width
    assert "eco_sst" not in ds.data_vars          # no bare (unversioned) sensor channel


def test_versions_share_one_overpass_identity_preferring_the_first_listed(tmp_path, days):
    """D5: the matchups (here overpass-met) key off ONE merged `eco` scene time -- the config's
    first-listed version wins each day, later versions only fill days it did not cover."""
    project = _eco_versions_project(tmp_path, ["v003", "v002"])
    g = grid.project_grids(project)[AOI]
    # day0: BOTH versions flew, at different hours -> v003 (first-listed) must win.
    write_ecostress(project, g, days[0], hour=20, tag="v003")
    write_ecostress(project, g, days[0], hour=18, tag="v002")
    # day1: ONLY v002 flew -> it fills the gap in the merged identity.
    write_ecostress(project, g, days[1], hour=16, tag="v002")
    write_met_snapshot(project, g, days[0], 20, temp=295.0)   # v003's instant
    write_met_snapshot(project, g, days[0], 18, temp=270.0)   # v002's instant (must be ignored)
    write_met_snapshot(project, g, days[1], 16, temp=250.0)   # v002's instant, day1

    ds = datacube.assemble_aoi(g, _eff_with_overpass(project, met=[("eco", "hrrr")]), days)

    # one matchup channel, not per-version, taking v003's scene on day0 and v002's on day1.
    assert "eco_airtemp_hrrr" in ds.data_vars
    assert np.nanmean(ds["eco_airtemp_hrrr"].isel(time=0).values) == pytest.approx(295.0)
    assert np.nanmean(ds["eco_airtemp_hrrr"].isel(time=1).values) == pytest.approx(250.0)
    assert "eco_airtemp_hrrr_v003" not in ds.data_vars    # matchups are never per-version


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
# `modis_footprint_id` -- which native ~1 km observation each fine cell came from.
#
# MODIS is gridded from a COARSER swath, so a block of grid cells shares one observation.
# Carrying that grouping into the cube is what lets a downstream calibration average a
# high-res sensor over exactly the pixels one MODIS observation covers, without reopening
# the per-granule NetCDFs. Three things have to hold for that to be worth anything: the
# ids must be integers with a real "no pixel" sentinel, they must come from the SAME scene
# the SST came from, and they must be absent (not all -1) when the granules never had them.
# --------------------------------------------------------------------------- #
def _modis_base(project, g, days):
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))


def test_modis_footprint_id_reaches_the_cube(project, grids, days):
    g = grids[AOI]
    _modis_base(project, g, days)
    write_modis(project, g, days[1], temp=290.0, block=3)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    assert "modis_footprint_id" in ds.data_vars
    # An INDEX, not a measurement: int32 with a -1 sentinel, never a float with NaN.
    assert ds["modis_footprint_id"].dtype == np.int32
    fp = ds["modis_footprint_id"].isel(time=1).values
    assert (fp >= 0).all()                               # the whole AoI was in swath
    # Days with no granule are -1 throughout -- distinguishable from any real id.
    assert (ds["modis_footprint_id"].isel(time=0).values == -1).all()
    # A footprint spans many grid cells; that is the entire point of the channel.
    _, counts = np.unique(fp, return_counts=True)
    assert counts.max() == 9                              # 3x3 blocks
    assert "NOT comparable across days" in ds["modis_footprint_id"].attrs["comment"]


def test_footprint_ids_group_cells_that_share_one_modis_observation(project, grids, days):
    """The property the channel exists to support: cells sharing an id share a MODIS value,
    so averaging another sensor within an id is averaging over one MODIS observation."""
    g = grids[AOI]
    _modis_base(project, g, days)
    write_modis(project, g, days[1], temp=290.0, block=3, temp_step=0.5)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    fp = ds["modis_footprint_id"].isel(time=1).values
    sst = ds["modis_sst"].isel(time=1).values
    for fid in np.unique(fp):
        block = sst[fp == fid]
        assert np.ptp(block) == 0.0, f"footprint {fid} spans >1 MODIS value"
    # ...and distinct footprints really are distinct observations (temp_step made them vary),
    # so a group-by is not silently collapsing the whole scene into one bucket.
    assert len(np.unique(sst)) == len(np.unique(fp))


def test_no_footprint_channel_when_the_granules_never_carried_one(project, grids, days):
    """`footprint_id` is optional at acquisition. When no granule has it, the channel is
    ABSENT rather than all -1 -- which would read as 'nothing was ever in swath'."""
    g = grids[AOI]
    _modis_base(project, g, days)
    write_modis(project, g, days[1], temp=290.0, footprint=False)
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    assert "modis_footprint_id" not in ds.data_vars
    assert "modis_sst" in ds.data_vars                    # the rest of MODIS is unaffected


def test_no_int32_footprint_is_allocated_when_no_granule_carried_the_layer(
        project, grids, days, monkeypatch):
    """The absent-footprint case must not COST anything either.

    The `(T,H,W)` int32 index is the same size as an SST channel, and it used to be filled
    with -1 up front and thrown away on the way out -- 16 GiB on a large AoI, for a channel
    the caller never emits. `np.full` touches every page, so this was committed memory, not a
    lazy reservation.
    """
    g = grids[AOI]
    _modis_base(project, g, days)
    write_modis(project, g, days[1], temp=290.0, footprint=False)

    big = []
    real_full = np.full

    def spy(shape, *a, **kw):
        out = real_full(shape, *a, **kw)
        if out.ndim == 3:
            big.append((out.shape, out.dtype))
        return out

    monkeypatch.setattr(datacube.np, "full", spy)
    d = project.output_dir / "MODIS" / "aligned" / AOI
    *_, footprint = datacube.load_clearest_overpass(
        d, AOI, days, g.height, g.width, trust_valid=True, read_footprint=True)

    assert footprint is None
    assert not [s for s, dt in big if dt == np.dtype("int32")], (
        f"allocated a 3-D int32 array for a footprint that does not exist: {big}")


def test_footprint_comes_from_the_same_scene_as_the_sst(project, grids, days):
    """Two granules on one day: the clearest wins, and its footprint must win with it.
    Reading the footprint outside that choice would pair one granule's pixel indices with
    another granule's temperatures -- wrong, and invisible."""
    g = grids[AOI]
    _modis_base(project, g, days)
    H, W, xs, ys = _grid_hw(g)
    for hh, valid_rows, block in [(19, H // 2, 2), (21, H, 3)]:   # the 21:00 scene is clearer
        rows, cols = np.arange(H)[:, None], np.arange(W)[None, :]
        fp = ((rows // block) * ((W + block - 1) // block) + (cols // block)).astype("int32")
        fp = np.broadcast_to(fp, (H, W)).astype("int32")
        valid = np.zeros((H, W), "uint8")
        valid[:valid_rows] = 1
        ds_in = xr.Dataset(
            {"sst": (("time", "y", "x"), np.full((1, H, W), 290.0, "float32")),
             "valid": (("time", "y", "x"), valid[None]),
             "footprint_id": (("time", "y", "x"), fp[None])},
            coords={"time": [days[1]], "y": ys, "x": xs})
        _write(project, "MODIS", f"{AOI}_{days[1].strftime('%Y%m%d')}T{hh:02d}0000.nc", ds_in)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert ds["modis_hour"].values[1] == pytest.approx(21.0)      # the clearer scene won
    _, counts = np.unique(ds["modis_footprint_id"].isel(time=1).values, return_counts=True)
    assert counts.max() == 9        # the 21:00 scene's 3x3 blocks, not the 19:00 scene's 2x2


# --------------------------------------------------------------------------- #
# The per-sensor VALIDITY RULES.
#
# Each thermal sensor decides `valid` differently, and the differences are not cosmetic --
# they are corrections for how each instrument's own masks misbehave:
#
#   ECOSTRESS  water layer has INVERTED polarity (<0.5 means water), and its cloud mask
#              over-masks cold water -- so validity gates on the QC mandatory-QA bits
#              instead of on cloud.
#   Landsat    water = 1, and its QA_PIXEL-derived cloud mask IS reliable -> gate on cloud.
#   MODIS      already quality-filtered upstream -> trust its own `valid` layer; it carries
#              no water or cloud layer to recompute from.
#
# Until now those three rules lived only as arguments at three hardcoded call sites, and
# NOTHING asserted them: the happy-path tests above pass just as well with the polarity
# reversed or the QC gate removed. They are about to become DATA (products.SensorSpec), so
# pin the behaviour first -- a mis-wired spec would silently change what counts as an
# observation, which is exactly the kind of degradation that leaves no trace in the cube.
# --------------------------------------------------------------------------- #
def write_ecostress(project, g, day, hour=20, *, sst=286.0, water=0.0, cloud=0.0, quality=0,
                    tag="v002"):
    """One ECOSTRESS scene with every mask settable. Defaults are the all-clear case:
    water=0 (eco polarity: <0.5 IS water), no cloud, QC bits 0-1 = 0 (good)."""
    H, W, xs, ys = _grid_hw(g)
    full = lambda v, dt: (("time", "y", "x"), np.full((1, H, W), v, dt))   # noqa: E731
    ds = xr.Dataset({
        "sst": full(sst, "float32"),
        "cloud": full(cloud, "float32"),
        "water": full(water, "float32"),
        "quality": full(quality, "float32"),
    }, coords={"time": [day], "y": ys, "x": xs})
    _write_eco(project, f"{AOI}_{day.strftime('%Y%m%d')}T{hour:02d}0000.nc", ds, tag=tag)


def _bare_cube(project, g, days):
    """Assemble with only the statics written, so a sensor's channels are what is under test."""
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    return datacube.assemble_aoi(g, datacube._build_eff(project), days)


def test_ecostress_water_layer_is_read_with_INVERTED_polarity(project, grids, days):
    """eco `water` >= 0.5 means LAND. Read with Landsat's polarity, a land pixel would be
    counted as a valid sea-surface observation -- and the cube gives no hint that it was."""
    g = grids[AOI]
    write_ecostress(project, g, days[0], water=1.0)         # eco polarity: 1 == LAND
    ds = _bare_cube(project, g, days)
    assert int(ds["eco_valid_v002"].isel(time=0).values.sum()) == 0


def test_ecostress_validity_gates_on_QC_not_cloud(project, grids, days):
    """Two halves of the same rule, and they pull in opposite directions:

    * a BAD QC scene must be rejected even though it is cloud-free, and
    * a CLOUDY scene must NOT be rejected, because eco's cloud mask over-masks cold water
      (that is the entire reason this sensor gates on QC instead).

    Drop the QC gate and the first scene silently becomes valid data. Add a cloud gate and
    the second silently disappears. Neither would raise.
    """
    g = grids[AOI]
    # quality=3 -> mandatory-QA bits 0-1 == 0b11, which is not in the accepted {0, 1}
    write_ecostress(project, g, days[0], quality=3, cloud=0.0)
    # cloudy, but QC-good: eco must keep it
    write_ecostress(project, g, days[1], quality=0, cloud=1.0)
    ds = _bare_cube(project, g, days)

    assert int(ds["eco_valid_v002"].isel(time=0).values.sum()) == 0                  # bad QC -> out
    assert int(ds["eco_valid_v002"].isel(time=1).values.sum()) == g.height * g.width  # cloudy KEPT


def test_landsat_validity_gates_on_CLOUD(project, grids, days):
    """The asymmetry with ECOSTRESS above: Landsat's QA_PIXEL cloud mask is reliable, so a
    cloudy Landsat pixel IS invalid -- where the identical eco pixel is kept."""
    g = grids[AOI]
    write_landsat(project, g, days[0], hour=19)                     # clear
    H, W, xs, ys = _grid_hw(g)
    cloudy = xr.Dataset({
        "sst": (("time", "y", "x"), np.full((1, H, W), 288.0, "float32")),
        "cloud": (("time", "y", "x"), np.ones((1, H, W), "float32")),   # fully cloudy
        "water": (("time", "y", "x"), np.ones((1, H, W), "float32")),   # lst polarity: 1 == water
    }, coords={"time": [days[1]], "y": ys, "x": xs})
    _write(project, "LANDSAT", f"{AOI}_{days[1].strftime('%Y%m%d')}T190000.nc", cloudy)

    ds = _bare_cube(project, g, days)
    assert int(ds["lst_valid"].isel(time=0).values.sum()) == g.height * g.width   # clear -> valid
    assert int(ds["lst_valid"].isel(time=1).values.sum()) == 0                    # cloudy -> out


def test_landsat_water_layer_is_read_with_NORMAL_polarity(project, grids, days):
    """lst `water` == 1 means water -- the opposite of ECOSTRESS. A pixel the classifier
    calls land is not a sea-surface observation."""
    g = grids[AOI]
    H, W, xs, ys = _grid_hw(g)
    land = xr.Dataset({
        "sst": (("time", "y", "x"), np.full((1, H, W), 288.0, "float32")),
        "cloud": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
        "water": (("time", "y", "x"), np.zeros((1, H, W), "float32")),  # 0 == LAND for Landsat
    }, coords={"time": [days[0]], "y": ys, "x": xs})
    _write(project, "LANDSAT", f"{AOI}_{days[0].strftime('%Y%m%d')}T190000.nc", land)

    ds = _bare_cube(project, g, days)
    assert int(ds["lst_valid"].isel(time=0).values.sum()) == 0


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
    assert cube["mur_sst"].dtype == np.float32 and cube["landcover_water"].dtype == np.uint8

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
    assert not list(tmp_path.glob("*.part-*"))          # and the scratch was cleaned up


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
    assert not list(tmp_path.glob("*.part-*")) and not list(tmp_path.glob("*.old-*"))


def test_write_zarr_safe_sweeps_scratch_left_by_an_earlier_crash(tmp_path, caplog):
    zpath = tmp_path / "aoi.zarr"
    stale = tmp_path / "aoi.zarr.part-999-1"
    stale.mkdir()
    (stale / "junk").write_text("half a chunk")

    with caplog.at_level("WARNING"):
        _write_cube(_cube(), zpath)

    assert not stale.exists()
    assert "unfinished run" in caplog.text          # and the user is TOLD a run had died
    assert xr.open_zarr(zpath).sizes["time"] == 4


# --------------------------------------------------------------------------- #
# Met: reference time of day, and forcing at each sensor's own overpass
# --------------------------------------------------------------------------- #
def write_met_daily(project, g, days, *, temp=280.0, prefix="", src="hrrr"):
    """One met FORCING source's daily-mean (prefix='') or reference (prefix='ref_') files,
    under MET/<src>/aligned/<aoi>."""
    H, W, xs, ys = _grid_hw(g)
    d = project.output_dir / "MET" / src / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    for day in days:
        ds = xr.Dataset(
            {"airtemp": (("time", "y", "x"), np.full((1, H, W), temp, "float32")),
             "wind_speed": (("time", "y", "x"), np.full((1, H, W), 3.0, "float32")),
             "swrad": (("time", "y", "x"), np.full((1, H, W), 500.0, "float32")),
             "cloud_cover": (("time", "y", "x"), np.full((1, H, W), 10.0, "float32"))},
            coords={"time": [day], "y": ys, "x": xs})
        ds.to_netcdf(d / f"{AOI}_{prefix}{day.strftime('%Y%m%d')}.nc")


def write_met_snapshot(project, g, day, hour, *, temp, src="hrrr"):
    """One met_overpass source's snapshot at one overpass instant, under
    MET_OVERPASS/<src>/aligned/<aoi>."""
    H, W, xs, ys = _grid_hw(g)
    stamp = day.strftime("%Y%m%d") + f"T{hour:02d}0000"
    d = project.output_dir / "MET_OVERPASS" / src / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"airtemp": (("time", "y", "x"), np.full((1, H, W), temp, "float32")),
         "wind_speed": (("time", "y", "x"), np.full((1, H, W), 7.0, "float32")),
         "swrad": (("time", "y", "x"), np.full((1, H, W), 800.0, "float32")),
         "cloud_cover": (("time", "y", "x"), np.full((1, H, W), 20.0, "float32"))},
        coords={"time": [day], "y": ys, "x": xs})
    ds.to_netcdf(d / f"{AOI}_{stamp}.nc")


def _eff_with_overpass(project, met=(("eco", "hrrr"), ("lst", "hrrr"), ("modis", "hrrr"))):
    """Build a datacube eff and inject overpass-met combos (the test project doesn't select
    the met_overpass product, so its combos would otherwise be empty)."""
    eff = datacube._build_eff(project)
    eff["met_overpass_combos"] = list(met)
    return eff


def test_met_comes_from_the_reference_snapshot_not_the_daily_mean(project, grids, days):
    g = grids[AOI]
    write_met_daily(project, g, days, temp=280.0)                 # daily mean
    write_met_daily(project, g, days, temp=291.0, prefix="ref_")  # 10:30 reference
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.nanmean(ds["airtemp_hrrr"].isel(time=0).values) == pytest.approx(291.0)
    assert ds.attrs["met_time"] == "reference"


def test_daily_mean_can_still_be_selected(project, grids, days):
    g = grids[AOI]
    write_met_daily(project, g, days, temp=280.0)
    write_met_daily(project, g, days, temp=291.0, prefix="ref_")
    eff = datacube._build_eff(project)
    eff["met_time"] = "daily_mean"
    ds = datacube.assemble_aoi(g, eff, days)
    assert np.nanmean(ds["airtemp_hrrr"].isel(time=0).values) == pytest.approx(280.0)
    assert ds.attrs["met_time"] == "daily_mean"


def test_met_falls_back_when_no_reference_files_exist(project, grids, days, caplog):
    """An older MET tree has no reference snapshots -- use the daily mean rather than
    emitting an all-NaN forcing channel."""
    g = grids[AOI]
    write_met_daily(project, g, days, temp=280.0)
    with caplog.at_level("WARNING"):
        ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert np.nanmean(ds["airtemp_hrrr"].isel(time=0).values) == pytest.approx(280.0)
    assert ds.attrs["met_time"] == "daily_mean"


def test_each_sensor_gets_the_forcing_from_its_own_overpass(project, grids, days):
    """Two sensors, hours apart on one day, must not share one met value."""
    g = grids[AOI]
    write_met_daily(project, g, days, temp=291.0, prefix="ref_")
    write_ecostress_two_scenes(project, g, days[0])           # clearest scene is 20:00
    write_landsat(project, g, days[0], hour=18)
    write_met_snapshot(project, g, days[0], 20, temp=286.0)   # the ECOSTRESS instant
    write_met_snapshot(project, g, days[0], 18, temp=299.0)   # the Landsat instant

    ds = datacube.assemble_aoi(g, _eff_with_overpass(project), days)
    assert np.nanmean(ds["eco_airtemp_hrrr"].isel(time=0).values) == pytest.approx(286.0)
    assert np.nanmean(ds["lst_airtemp_hrrr"].isel(time=0).values) == pytest.approx(299.0)
    # ...and the daily forcing channel is still the reference time, independent of both.
    assert np.nanmean(ds["airtemp_hrrr"].isel(time=0).values) == pytest.approx(291.0)
    # a day with no scene has no overpass forcing (not a stale value)
    assert np.isnan(ds["eco_airtemp_hrrr"].isel(time=1).values).all()


def test_overpass_met_follows_the_scene_the_cube_actually_kept(project, grids, days):
    """ECOSTRESS flies twice; the cube keeps the CLEAREST (20:00). The forcing must come
    from that scene's instant, not from the discarded 18:00 one."""
    g = grids[AOI]
    write_ecostress_two_scenes(project, g, days[0])           # 18:00 (half NaN), 20:00 (clear)
    write_met_snapshot(project, g, days[0], 18, temp=270.0)   # discarded scene
    write_met_snapshot(project, g, days[0], 20, temp=295.0)   # kept scene
    ds = datacube.assemble_aoi(g, _eff_with_overpass(project), days)
    assert ds["eco_hour_v002"].isel(time=0).item() == pytest.approx(20.0)
    assert np.nanmean(ds["eco_airtemp_hrrr"].isel(time=0).values) == pytest.approx(295.0)


def test_overpass_met_is_configurable_via_combinations(project, grids, days):
    """Which `<sensor>_<var>_<src>` channels appear is exactly the (sensor, source) combos."""
    g = grids[AOI]
    write_ecostress_two_scenes(project, g, days[0])
    write_met_snapshot(project, g, days[0], 20, temp=295.0, src="hrrr")

    eff = _eff_with_overpass(project, met=[("eco", "hrrr")])
    ds = datacube.assemble_aoi(g, eff, days)
    assert "eco_airtemp_hrrr" in ds.data_vars
    assert "lst_airtemp_hrrr" not in ds.data_vars            # lst not in the combos

    eff = _eff_with_overpass(project, met=[])                # no combos -> no overpass channels
    ds = datacube.assemble_aoi(g, eff, days)
    assert not [v for v in ds.data_vars if v.startswith("eco_") and "airtemp" in v]

def test_cmems_ships_raw_observed_values_with_honest_nan_gaps(project, grids, days):
    """Goal 3 / S1: CMEMS is no longer NN-filled and carries no `_filled` mask. Cells the
    ~9 km model never resolved stay NaN; downstream fills as it sees fit."""
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))          # all water per land-cover
    write_cmems(project, g, days, land_cols=slice(0, 4))        # model has no data cols 0-3
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    assert "cmems_thetao_0m_my_global" in ds.data_vars
    assert "cmems_thetao_0m_my_global_filled" not in ds.data_vars
    assert "cmems_source" not in ds.data_vars                   # per-day source code is gone
    val = ds["cmems_thetao_0m_my_global"].isel(time=0).values
    assert np.isnan(val[:, :4]).all()        # unresolved cells stay NaN, not filled
    assert np.isfinite(val[:, 4:]).all()     # resolved cells survive


# --------------------------------------------------------------------------- #
# Coverage: the cube's time axis is built from the CONFIG, so a lost day is a NaN
# slice that looks exactly like a cloudy day. Count what actually landed.
# --------------------------------------------------------------------------- #
def test_coverage_counts_the_days_that_actually_landed(project, grids, days):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    # drop one of the three MUR days, as a failed download would
    (project.output_dir / "MUR" / "aligned" / AOI /
     f"{AOI}_{days[1].strftime('%Y%m%d')}.nc").unlink()

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    cov = json.loads(ds.attrs["coverage"])

    assert cov["mur"]["days_expected"] == 3
    assert cov["mur"]["days_with_data"] == 2      # the lost day is COUNTED, not hidden
    assert cov["mur"]["fraction"] == 2 / 3
    assert ds.sizes["time"] == 3                  # ...while the axis is still full-length


def test_thin_coverage_is_warned_about(project, grids, days, caplog):
    g = grids[AOI]
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    write_mur(project, g, [days[0]], water_hole_cols=slice(0, 0))   # 1 of 3 days only

    with caplog.at_level("WARNING"):
        datacube.assemble_aoi(g, datacube._build_eff(project), days)

    assert "covers only 1 of 3" in caplog.text
    assert "look exactly like cloudy days" in caplog.text


def test_deliberately_sparse_mur_is_reported_but_not_warned_about(tmp_path, days, caplog):
    """MUR restricted to overpass days (`mur.overpass_sensors`) is thin ON PURPOSE. The
    fraction is still measured and still stamped on the cube -- a reader must see it -- but
    the warning would fire on every assemble of a correct config, which is how the real
    thin-coverage warning gets ignored."""
    project = parse_config({
        "name": "dc", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
        "auth": {"earthdata": {"auth_strategy": "netrc"}},
        "products": {"bathymetry": None, "ecostress": None,
                     "mur": {"overpass_sensors": ["eco"]}},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })
    g = grid.project_grids(project)[AOI]
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    write_mur(project, g, [days[0]], water_hole_cols=slice(0, 0))   # 1 of 3 days, as asked

    with caplog.at_level("WARNING"):
        ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    cov = json.loads(ds.attrs["coverage"])["mur"]
    assert (cov["fraction"], cov["sparse"]) == (1 / 3, True)   # measured AND labelled
    assert "covers only" not in caplog.text


def test_full_coverage_does_not_warn(project, grids, days, caplog):
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    with caplog.at_level("WARNING"):
        ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert json.loads(ds.attrs["coverage"])["mur"]["fraction"] == 1.0
    assert "covers only" not in caplog.text


def test_overpass_sensors_are_not_judged_on_daily_coverage(project, grids, days):
    """ECOSTRESS/Landsat/MODIS are overpass sensors: a day with no scene is NORMAL. Warning
    on those would train the user to ignore the warning."""
    g = grids[AOI]
    write_mur(project, g, days, water_hole_cols=slice(0, 0))
    write_bathymetry(project, g)
    write_landcover(project, g, land_cols=slice(0, 0))
    write_ecostress_two_scenes(project, g, days[0])          # a scene on ONE day of three
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert "ecostress" not in json.loads(ds.attrs["coverage"])

# --------------------------------------------------------------------------- #
# GOLDEN CUBE (refactor safety net, plan stage S0)
#
# Assemble ONE richly-populated AoI -- every product written -- and snapshot every
# channel's values + attrs into a committed JSON golden. The whole assembler refactor
# (docs/refactor-plan-assembler-and-sources.md) is gated on this: the contributor-protocol
# migration (S2) must keep it BYTE-IDENTICAL, and the deletions (S1) / per-source emission
# (S3-S4) regenerate it as a REVIEWED diff -- the git diff of the golden IS the review
# artifact (channel set + attrs are human-readable; bulk float values are fingerprinted).
#
# The comparator EXCLUDES the run-stamp attrs, because any working-tree edit flips
# `code_version` to `...-dirty` and rewrites `created_at`, so a naive full-attr snapshot
# would fail on every phase for reasons unrelated to the cube's data.
# --------------------------------------------------------------------------- #
GOLDEN = Path(__file__).parent / "golden" / "datacube_golden.json"

# provenance / run-stamp attrs: excluded from the snapshot (they change on every tree edit).
RUNSTAMP_ATTRS = ("created_at", "preprocessed_at", "code_version", "package_version",
                  "config_yaml", "config_sha256", "config_path",
                  "provenance", "provenance_products")


def write_tides(project, g, days, *, amplitude=1.0, src="coops"):
    """One tide source's hourly series (TIDE/<src>/aligned/<aoi>): a 12 h sinusoid."""
    hours = pd.date_range(days[0], periods=24 * len(days), freq="h")
    tide = amplitude * np.sin(2 * np.pi * np.arange(len(hours)) / 12.0)
    ds = xr.Dataset({"tide": (("time",), tide.astype("float32"))}, coords={"time": hours})
    d = project.output_dir / "TIDE" / src / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}_tides.nc")


def write_insitu(project, g, times, values):
    """One in-situ station at the AoI centre, with a value at each of `times`."""
    ds = xr.Dataset(
        {"sst": (("station", "time"), np.array([values], "float32")),
         "qc": (("station", "time"), np.ones((1, len(times)), "uint8"))},
        coords={"station": [0], "time": pd.DatetimeIndex(times),
                "station_id": ("station", ["s1"]), "station_name": ("station", ["Test Station"]),
                "lat": ("station", [45.5]), "lon": ("station", [-123.9])})
    _write(project, "INSITU", f"{AOI}_insitu.nc", ds)


def _write_full_fixture(project, g, days):
    """Every product written, chosen so as many channels as possible carry real values:
    MUR (+ a water hole to fill), bathymetry, land-cover (a land strip), all three thermal
    sensors, CMEMS (+ a model-land hole), met (daily + reference + per-overpass snapshots),
    tides, and one in-situ station. Deterministic: all synthetic values are constants and
    dates are fixed, so every derived channel (fills, water level, coverage, doy) is stable.
    """
    write_mur(project, g, days, water_hole_cols=slice(0, 3))
    write_bathymetry(project, g, "cudem", datum_offset_m=1.3, datum_status="ok")
    write_bathymetry(project, g, "gmrt", datum_offset_m=0.0, datum_status="ok")  # stacked (D10)
    write_landcover(project, g, land_cols=slice(0, 5))
    write_ecostress_two_scenes(project, g, days[0])      # clearest scene 20:00
    write_landsat(project, g, days[0], hour=18)
    write_modis(project, g, days[1], temp=287.0)         # overpass 21:00
    write_cmems(project, g, days, land_cols=slice(0, 4), src="my_global")
    write_cmems(project, g, days, land_cols=slice(0, 4), src="anfc_global")  # stacked (D10)
    write_met_daily(project, g, days, temp=280.0)                 # daily mean
    write_met_daily(project, g, days, temp=291.0, prefix="ref_")  # reference snapshot
    write_met_snapshot(project, g, days[0], 20, temp=286.0)       # eco overpass
    write_met_snapshot(project, g, days[0], 18, temp=299.0)       # lst overpass
    write_met_snapshot(project, g, days[1], 21, temp=285.0)       # modis overpass
    write_insitu(project, g, ["2026-06-01T18:00", "2026-06-01T20:00", "2026-06-02T21:00"],
                 [12.5, 13.0, 14.0])
    write_tides(project, g, days)


def _fingerprint(arr) -> str:
    """A NaN-aware content hash of an array: value-level, not bit-payload-level.

    NaN positions are hashed separately from the (NaN-zeroed) values, so two arrays that
    agree on which cells are missing and on every finite value fingerprint identically --
    regardless of the particular NaN bit pattern a given code path happened to produce.
    """
    b = np.ascontiguousarray(np.asarray(arr))
    if b.dtype.kind == "M":                    # datetime64 -> integer ns
        b = b.astype("int64")
    h = hashlib.sha256()
    h.update(str(b.dtype).encode()); h.update(str(b.shape).encode())
    if b.dtype.kind == "f":
        b = b.copy()
        nan = np.isnan(b)
        b[nan] = 0.0
        h.update(nan.tobytes())
    h.update(b.tobytes())
    return h.hexdigest()


def _stats(arr) -> dict:
    """Human-readable summary (finite count + range) so the golden diff is legible."""
    a = np.asarray(arr)
    if a.dtype.kind not in "fiu":
        return {}
    af = a.astype("float64")
    finite = np.isfinite(af)
    n = int(finite.sum())
    if not n:
        return {"n_finite": 0}
    return {"n_finite": n, "min": round(float(np.nanmin(af)), 6),
            "max": round(float(np.nanmax(af)), 6), "mean": round(float(np.nanmean(af)), 6)}


def _jsonify(obj):
    """Make attrs JSON-comparable: numpy scalars -> python, numpy arrays -> lists."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _snapshot(ds: xr.Dataset) -> dict:
    """Snapshot a cube: dims, stripped global attrs, and per coord/channel dtype + shape +
    attrs + stats + value fingerprint. Everything here is deterministic for a fixed fixture."""
    attrs = {k: v for k, v in ds.attrs.items() if k not in RUNSTAMP_ATTRS}
    snap = {"dims": {k: int(v) for k, v in ds.sizes.items()},
            "global_attrs": _jsonify(attrs), "coords": {}, "data_vars": {}}
    for name, c in ds.coords.items():
        snap["coords"][str(name)] = {"dtype": str(c.dtype), "shape": list(c.shape),
                                     "fingerprint": _fingerprint(c.values)}
    for name in ds.data_vars:
        da = ds[name]
        snap["data_vars"][str(name)] = {
            "dims": list(da.dims), "dtype": str(da.dtype), "shape": list(da.shape),
            "attrs": _jsonify(dict(da.attrs)), "stats": _stats(da.values),
            "fingerprint": _fingerprint(da.values)}
    return snap


def _diff_snapshots(golden: dict, actual: dict) -> list[str]:
    """A human-readable list of every way `actual` drifts from `golden` (empty == identical)."""
    out = []
    if golden["dims"] != actual["dims"]:
        out.append(f"dims: {golden['dims']} -> {actual['dims']}")
    ga, aa = golden["global_attrs"], actual["global_attrs"]
    for k in sorted(set(ga) - set(aa)):
        out.append(f"global attr REMOVED: {k}")
    for k in sorted(set(aa) - set(ga)):
        out.append(f"global attr ADDED: {k} = {aa[k]!r}")
    for k in sorted(set(ga) & set(aa)):
        if ga[k] != aa[k]:
            out.append(f"global attr CHANGED {k}: {ga[k]!r} -> {aa[k]!r}")
    for sect in ("coords", "data_vars"):
        gs, as_ = golden[sect], actual[sect]
        for k in sorted(set(gs) - set(as_)):
            out.append(f"{sect}: channel REMOVED: {k}")
        for k in sorted(set(as_) - set(gs)):
            out.append(f"{sect}: channel ADDED: {k}")
        for k in sorted(set(gs) & set(as_)):
            gk, ak = gs[k], as_[k]
            for field in ("dtype", "shape", "dims"):
                if gk.get(field) != ak.get(field):
                    out.append(f"{sect}[{k}].{field}: {gk.get(field)} -> {ak.get(field)}")
            if gk.get("attrs") != ak.get("attrs"):
                out.append(f"{sect}[{k}].attrs: {gk.get('attrs')} -> {ak.get('attrs')}")
            if gk.get("fingerprint") != ak.get("fingerprint"):
                out.append(f"{sect}[{k}] VALUES changed: stats {gk.get('stats')} -> {ak.get('stats')}")
    return out


def test_golden_cube_is_unchanged(project, grids, days):
    """The refactor safety net. Assemble the full fixture and assert the cube matches the
    committed golden, channel for channel (run-stamp attrs excluded). Set UPDATE_GOLDEN=1 to
    regenerate after an INTENDED change, then review the golden's git diff."""
    g = grids[AOI]
    _write_full_fixture(project, g, days)
    # Exercise the overpass products too (the minimal project fixture doesn't select them):
    # met_overpass snapshots and tide_overpass, for a couple of (sensor, source) combos.
    eff = datacube._build_eff(project)
    eff["met_overpass_combos"] = [("eco", "hrrr"), ("lst", "hrrr")]
    eff["tide_overpass_combos"] = [("eco", "coops"), ("lst", "coops")]
    ds = datacube.assemble_aoi(g, eff, days)
    actual = _snapshot(ds)

    existed = GOLDEN.exists()
    if os.environ.get("UPDATE_GOLDEN") or not existed:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden {'regenerated' if existed else 'created'} at {GOLDEN}; "
                    "review its git diff and re-run")

    problems = _diff_snapshots(json.loads(GOLDEN.read_text()), actual)
    assert not problems, (
        "datacube drifted from the golden snapshot:\n  " + "\n  ".join(problems) +
        "\n\nIf this change is intended, regenerate with UPDATE_GOLDEN=1 and review the diff.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])
