#!/usr/bin/env python3
"""
coastal_sst_data -- datacube assembler (final stage).

Knits the per-AoI aligned outputs of every acquisition stage (MUR, ECOSTRESS,
Landsat, MODIS, met, bathymetry, tide, land-cover) into ONE analysis-ready,
chunked+compressed Zarr cube per AoI on a common DAILY time axis:

    <output_dir>/<datacube.output_subdir>/<aoi>.zarr

Unlike the acquisition modules this reads no network -- it only reads files the
other stages already wrote, so run it LAST. It drives off the validated Project
and the SHARED per-AoI grid (coastal_sst_data.grid), so the cube lands on exactly
the grid every product was regridded onto (no grid is re-derived here).

Design (locked with the maintainer):
  * Zarr per AoI, chunked in (time, y, x), lossless float32 + Blosc(zstd) codec.
  * SST kept SEPARATE per sensor (mur/eco/lst/modis) so the model's learned
    per-source offsets survive; each high-res sensor carries its own valid mask
    and overpass hour.
  * Multiple scenes of one sensor on a day -> keep the CLEAREST (most valid px).
  * Water/land mask comes from LAND-COVER (authoritative where known), falling
    back to the sensor water union + sea-level bathymetry where it is unknown.
  * The MUR backbone is nearest-neighbour filled over land-cover water so the
    always-present channel has no holes in narrow estuaries.

  * Water level: bathymetry + tide give, per sensor and at that sensor's OVERPASS
    time, the ground elevation relative to the tide-adjusted waterline and a
    submerged/exposed class (see processes.water_level).

Channel layout in each <aoi>.zarr:
  3D (time,y,x): mur_sst, mur_valid, eco_sst, eco_cloud, eco_valid,
                 lst_sst, lst_cloud, lst_valid, modis_sst, modis_valid,
                 airtemp, wind_u, wind_v, wind_speed, swrad, cloud_cover,
                 {eco,lst,modis}_water_elev, {eco,lst,modis}_water_class
  2D (y,x) static: depth, depth_p25, depth_p75, landmask, landcover_water
  1D (time): tide, tide_range, eco_hour, lst_hour, modis_hour, doy_sin, doy_cos,
             {eco,lst,modis}_tide

Usage:
    python -m coastal_sst_data.processes.datacube --config config.yaml
    python -m coastal_sst_data.processes.datacube --config config.yaml --aoi hood_canal
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import CompressionSpec, Project, load_config
from ..grid import AoiGrid, project_grids
from . import water_level

log = logging.getLogger(__name__)

_DT_RE = re.compile(r"(\d{8}T\d{6})")     # per-overpass filename stamp
_D_RE = re.compile(r"_(\d{8})\.nc$")      # per-day filename stamp

# source -> the ALLCAPS "<DIR>/aligned/<aoi>/" folder each acquisition stage wrote.
PRODUCT_DIRS = {
    "mur": "MUR", "ecostress": "ECOSTRESS", "landsat": "LANDSAT", "modis": "MODIS",
    "met": "MET", "bathymetry": "BATHYMETRY", "tide": "TIDE", "landcover": "LANDCOVER",
    "datum": "DATUM",
}


# --------------------------------------------------------------------------- #
# Loaders (each returns arrays on the daily axis / shared AoI grid)
# --------------------------------------------------------------------------- #
def _empty3d(days, H, W):
    return np.full((len(days), H, W), np.nan, dtype="float32")


def load_daily_sensor(d: Path, aoi_id, days, H, W, var):
    """MUR/met style: one file per day (<aoi>_YYYYMMDD.nc). Returns a (T,H,W) array."""
    out = _empty3d(days, H, W)
    if not d.exists():
        return out
    didx = {dd.strftime("%Y%m%d"): i for i, dd in enumerate(days)}
    for f in d.glob(f"{aoi_id}_*.nc"):
        m = _D_RE.search(f.name)
        if not m or m.group(1) not in didx:
            continue
        ds = xr.open_dataset(f)
        if var in ds:
            arr = ds[var].isel(time=0).values if "time" in ds[var].dims else ds[var].values
            if arr.shape == (H, W):
                out[didx[m.group(1)]] = arr
        ds.close()
    return out


def load_clearest_overpass(d: Path, aoi_id, days, H, W, *, water_is_land=False,
                           use_cloud=True, qc_levels=None, trust_valid=False):
    """Per-overpass sensors (ECOSTRESS/Landsat/MODIS): keep the clearest scene/day.

    Validity per sensor:
      * trust_valid=True (MODIS -- already quality-filtered): use the file's
        `valid` layer directly; the sensor has no water/cloud layer.
      * else recompute finite(sst) & water [& clear] [& QC]:
          - water: the sensor water layer with per-sensor polarity (water_is_land).
          - use_cloud gates on the binary cloud layer (Landsat: reliable).
          - qc_levels (e.g. {0,1}) gates on QC mandatory-QA bits 0-1 instead of
            cloud (ECOSTRESS: cloud over-masks cold water, so gate on QC).
    Returns (sst, cloud, valid, hour, water_union). `water_union` is the OR of the
    water mask over scenes -- a high-res static water hint for narrow estuaries.
    """
    sst, cloud = _empty3d(days, H, W), _empty3d(days, H, W)
    valid = np.zeros((len(days), H, W), dtype="uint8")
    hour = np.full(len(days), np.nan, dtype="float32")
    water_union = np.zeros((H, W), dtype=bool)
    if not d.exists():
        return sst, cloud, valid, hour, water_union
    qset = list(qc_levels) if qc_levels is not None else None
    didx = {dd.strftime("%Y%m%d"): i for i, dd in enumerate(days)}
    best = {}  # day -> (valid_count, sst, cloud, valid, datetime)
    for f in d.glob(f"{aoi_id}_*T*.nc"):
        m = _DT_RE.search(f.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        day = dt.strftime("%Y%m%d")
        if day not in didx:
            continue
        ds = xr.open_dataset(f)
        if "time" in ds.dims:
            ds = ds.isel(time=0)
        if "sst" not in ds or ds["sst"].shape != (H, W):
            ds.close(); continue
        s = ds["sst"].values.astype("float32")
        c = (ds["cloud"].values.astype("float32")
             if "cloud" in ds and ds["cloud"].shape == (H, W) else np.zeros((H, W), "float32"))
        if trust_valid and "valid" in ds and ds["valid"].shape == (H, W):
            v = (ds["valid"].values > 0) & np.isfinite(s)
            wp = np.zeros((H, W), dtype=bool)              # no water layer to contribute
        else:
            if "water" in ds and ds["water"].shape == (H, W):
                w = ds["water"].values.astype("float32")
                wp = np.isfinite(w) & ((w < 0.5) if water_is_land else (w > 0.5))
            else:
                wp = np.zeros((H, W), dtype=bool)          # no water layer -> claim NOTHING
            q = (ds["quality"].values if "quality" in ds and ds["quality"].shape == (H, W) else None)
            v = np.isfinite(s) & wp
            if use_cloud:
                v &= ~(np.nan_to_num(c, nan=1.0) > 0)
            if qset is not None and q is not None:
                mqa = np.full((H, W), -1, dtype="int64")
                fin = np.isfinite(q)
                mqa[fin] = q[fin].astype("int64") & 0b11   # mandatory-QA bits 0-1
                v &= np.isin(mqa, qset)
        ds.close()

        water_union |= wp
        vc = int(v.sum())
        if day not in best or vc > best[day][0]:
            best[day] = (vc, s, c, v, dt)
    for day, (_, s, c, v, dt) in best.items():
        i = didx[day]
        sst[i] = s
        cloud[i] = np.nan_to_num(c, nan=0.0)
        valid[i] = v.astype("uint8")
        hour[i] = dt.hour + dt.minute / 60.0
    return sst, cloud, valid, hour, water_union


def load_tide_daily(d: Path, aoi_id, days):
    """Tide 1D series -> (daily_mean, daily_range) on the daily axis."""
    mean = np.full(len(days), np.nan, "float32")
    rng = np.full(len(days), np.nan, "float32")
    f = d / f"{aoi_id}_tides.nc"
    if not f.exists():
        return mean, rng
    ds = xr.open_dataset(f)
    t = ds["tide"]
    dm = t.resample(time="1D").mean()
    dr = t.resample(time="1D").max() - t.resample(time="1D").min()
    lut_m = dict(zip(dm["time"].dt.strftime("%Y%m%d").values, dm.values))
    lut_r = dict(zip(dr["time"].dt.strftime("%Y%m%d").values, dr.values))
    for i, dd in enumerate(days):
        k = dd.strftime("%Y%m%d")
        if k in lut_m:
            mean[i] = lut_m[k]
            rng[i] = lut_r[k]
    ds.close()
    return mean, rng


def load_bathy(d: Path, aoi_id, H, W):
    """Static bathymetry: (elevation, depth, depth_p25, depth_p75), NaN where absent."""
    elev = np.full((H, W), np.nan, "float32")
    depth = dp25 = dp75 = None
    f = d / f"{aoi_id}.nc"
    if f.exists():
        ds = xr.open_dataset(f)

        def g(name):
            return (ds[name].values.astype("float32")
                    if name in ds and ds[name].shape == (H, W) else None)
        if g("elevation") is not None:
            elev = g("elevation")
        depth, dp25, dp75 = g("depth"), g("depth_p25"), g("depth_p75")
        ds.close()
    if depth is None:                                    # derive mean depth from elevation
        depth = np.where(elev < 0, -elev, 0.0).astype("float32")
    if dp25 is None:
        dp25 = depth.copy()
    if dp75 is None:
        dp75 = depth.copy()
    return elev, depth, dp25, dp75


def load_bathy_attrs(d: Path, aoi_id) -> dict:
    """The bathymetry file's global attrs (its `source` is the DEM fingerprint)."""
    f = d / f"{aoi_id}.nc"
    if not f.exists():
        return {}
    with xr.open_dataset(f) as ds:
        return dict(ds.attrs)


def load_landcover(d: Path, aoi_id, H, W):
    """Static land-cover water mask -> float (1=water, 0=land, NaN=unknown/absent)."""
    water = np.full((H, W), np.nan, "float32")
    f = d / f"{aoi_id}.nc"
    if f.exists():
        ds = xr.open_dataset(f)
        if "water" in ds and ds["water"].shape == (H, W):
            water = ds["water"].values.astype("float32")
        ds.close()
    return water


def fill_water_nn(arr, water):
    """Nearest-neighbour fill of NaNs over `water` pixels, per time slice.

    For each day, water pixels with no value take the nearest finite value
    (typically just-offshore open water). Land / non-water NaNs are left as-is.
    `arr` is (T,H,W); `water` is (H,W) bool.
    """
    from scipy.ndimage import distance_transform_edt
    out = arr.copy()
    for t in range(out.shape[0]):
        m = out[t]
        finite = np.isfinite(m)
        need = (~finite) & water
        if need.any() and finite.any():
            idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)
            nn = m[tuple(idx)]
            m[need] = nn[need]
            out[t] = m
    return out


# --------------------------------------------------------------------------- #
# Assemble one AoI onto its shared grid
# --------------------------------------------------------------------------- #
def assemble_aoi(g: AoiGrid, eff: dict, days) -> xr.Dataset:
    """Build the analysis-ready Dataset for one AoI from its aligned files."""
    H, W = g.height, g.width
    xs, ys = g.xy_centers()
    aid = g.name

    def adir(src):
        return eff["aligned_root"] / PRODUCT_DIRS[src] / "aligned" / aid

    elev, depth, depth_p25, depth_p75 = load_bathy(adir("bathymetry"), aid, H, W)

    mur_sst = load_daily_sensor(adir("mur"), aid, days, H, W, "sst")
    # ECOSTRESS: water + QC-produced (its cloud over-masks cold water; gate on QC).
    eco_sst, eco_cloud, eco_valid, eco_hour, eco_wu = load_clearest_overpass(
        adir("ecostress"), aid, days, H, W, water_is_land=True, use_cloud=False,
        qc_levels=[0, 1])
    # Landsat: water + cloud (its QA_PIXEL-based cloud is reliable).
    lst_sst, lst_cloud, lst_valid, lst_hour, lst_wu = load_clearest_overpass(
        adir("landsat"), aid, days, H, W, water_is_land=False, use_cloud=True)
    # MODIS: already quality-filtered upstream -> trust its `valid` layer.
    modis_sst, _, modis_valid, modis_hour, _ = load_clearest_overpass(
        adir("modis"), aid, days, H, W, trust_valid=True)

    airtemp = load_daily_sensor(adir("met"), aid, days, H, W, "airtemp")
    wind_u = load_daily_sensor(adir("met"), aid, days, H, W, "wind_u")
    wind_v = load_daily_sensor(adir("met"), aid, days, H, W, "wind_v")
    wind_speed = load_daily_sensor(adir("met"), aid, days, H, W, "wind_speed")
    swrad = load_daily_sensor(adir("met"), aid, days, H, W, "swrad")
    cloud_cover = load_daily_sensor(adir("met"), aid, days, H, W, "cloud_cover")
    tide, tide_range = load_tide_daily(adir("tide"), aid, days)

    # Water/land mask: land-cover is authoritative where known; elsewhere fall back
    # to the sensor water union + sea-level bathymetry (no tunable thresholds).
    lc_raw = load_landcover(adir("landcover"), aid, H, W)     # 1=water,0=land,NaN=unknown
    lc_known = np.isfinite(lc_raw)
    fallback = eco_wu | lst_wu | (np.isfinite(elev) & (elev < 0))
    water = np.where(lc_known, lc_raw > 0.5, fallback)
    landmask = (~water).astype("uint8")                       # 1 = land
    # Raw land-cover water as a loss-filter channel (unknown -> water, a no-op filter).
    landcover_water = np.where(lc_known, lc_raw > 0.5, True).astype("uint8")
    wf = float(water.mean())
    if wf > 0.98:
        log.warning("  %s: water is %.0f%% of the tile -- check the land-cover layer", aid, 100 * wf)

    # MUR is 1 km upsampled: NN-fill it over land-cover water so the backbone has
    # no holes in narrow estuaries.
    if eff["fill_mur_water"]:
        mur_sst = fill_water_nn(mur_sst, water)
    mur_valid = np.isfinite(mur_sst).astype("uint8")

    doy = days.dayofyear.values.astype("float32")
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    T = ("time", "y", "x")

    # Water level per sensor, at that sensor's overpass time: the DEM re-referenced
    # to the tide-adjusted waterline, plus its submerged/exposed class. All-NaN /
    # all-UNKNOWN when bathymetry or tide is absent.
    wl = {}
    datum_attrs = {}
    if eff["water_level"]:
        series = water_level.load_tide_series(adir("tide"), aid)
        # The offset follows the DEM that actually ran, so it is resolved against that
        # file's fingerprint (see water_level.resolve_datum_offset / processes.datum).
        offset, datum_attrs = water_level.resolve_datum_offset(
            eff["project"], aid, bathy_attrs=load_bathy_attrs(adir("bathymetry"), aid))
        for pre, hours in (("eco", eco_hour), ("lst", lst_hour), ("modis", modis_hour)):
            th = water_level.tide_at_overpass(series, days, hours)
            elev_rel, cls = water_level.water_level_fields(elev, th, datum_offset_m=offset)
            wl[f"{pre}_tide"] = (("time",), th)
            wl[f"{pre}_water_elev"] = (T, elev_rel)
            wl[f"{pre}_water_class"] = (T, cls)

    ds = xr.Dataset(
        {
            **wl,
            "mur_sst": (T, mur_sst), "mur_valid": (T, mur_valid),
            "eco_sst": (T, eco_sst), "eco_cloud": (T, eco_cloud), "eco_valid": (T, eco_valid),
            "lst_sst": (T, lst_sst), "lst_cloud": (T, lst_cloud), "lst_valid": (T, lst_valid),
            "modis_sst": (T, modis_sst), "modis_valid": (T, modis_valid),
            "airtemp": (T, airtemp), "wind_u": (T, wind_u), "wind_v": (T, wind_v),
            "wind_speed": (T, wind_speed), "swrad": (T, swrad), "cloud_cover": (T, cloud_cover),
            "depth": (("y", "x"), depth),
            "depth_p25": (("y", "x"), depth_p25), "depth_p75": (("y", "x"), depth_p75),
            "landmask": (("y", "x"), landmask),
            "landcover_water": (("y", "x"), landcover_water),
            "tide": (("time",), tide), "tide_range": (("time",), tide_range),
            "eco_hour": (("time",), eco_hour), "lst_hour": (("time",), lst_hour),
            "modis_hour": (("time",), modis_hour),
            "doy_sin": (("time",), doy_sin), "doy_cos": (("time",), doy_cos),
        },
        coords={"time": days, "y": ys, "x": xs},
    )
    ds.attrs.update(aoi_id=aid, crs=g.target_crs)
    # Datum provenance is a GLOBAL cube attr, not just a per-variable one: a cube whose
    # offset could not be resolved is complete and plausible-looking, so the only thing
    # that tells a downstream user it is biased is `datum_status` travelling with it.
    ds.attrs.update({k: v for k, v in datum_attrs.items() if v is not None})
    for pre in ("eco", "lst", "modis"):
        if f"{pre}_water_elev" not in ds:
            continue
        ds[f"{pre}_tide"].attrs.update(
            units="m", long_name=f"tide height at the {pre} overpass (rel. MSL)")
        ds[f"{pre}_water_elev"].attrs.update(
            units="m",
            long_name=f"ground elevation rel. to the waterline at the {pre} overpass "
                      "(0 at the waterline, + exposed, - submerged)",
            datum_offset_m=offset)
        ds[f"{pre}_water_class"].attrs.update(
            long_name=f"submerged/exposed at the {pre} overpass",
            flag_values=np.array([water_level.SUBMERGED, water_level.EXPOSED,
                                  water_level.UNKNOWN], dtype="uint8"),
            flag_meanings="submerged exposed unknown")
    return ds


# --------------------------------------------------------------------------- #
# Compression / encoding
# --------------------------------------------------------------------------- #
_SHUFFLE = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def _blosc_codec(cname: str, clevel: int, shuffle: str):
    """A Blosc codec for the installed Zarr (v3 BloscCodec, else numcodecs Blosc)."""
    import zarr
    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.codecs import BloscCodec
        return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle), "compressors"
    from numcodecs import Blosc
    return Blosc(cname=cname, clevel=clevel, shuffle=_SHUFFLE[shuffle]), "compressor"


def build_encoding(ds: xr.Dataset, compression: CompressionSpec, chunks: dict) -> dict:
    """Per-variable chunk + Blosc(zstd) encoding.

    Lossless: dtypes are untouched. Integer masks get BITSHUFFLE (long runs of a
    constant class compress hard); floats get the configured shuffle. The result
    keys ('compressors' for Zarr v3, 'compressor' for v2) match the installed Zarr.
    """
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        ch = tuple(min(chunks.get(d, ds.sizes[d]), ds.sizes[d]) for d in dims)
        shuffle = "bitshuffle" if ds[v].dtype == np.uint8 else compression.shuffle
        codec, key = _blosc_codec(compression.codec, compression.level, shuffle)
        e = {key: (codec,) if key == "compressors" else codec}
        if ch:
            e["chunks"] = ch
        enc[v] = e
    return enc


# --------------------------------------------------------------------------- #
# Write (NFS-safe overwrite)
# --------------------------------------------------------------------------- #
def write_zarr_safe(ds: xr.Dataset, zpath: Path, encoding: dict):
    """Write a Zarr cube, tolerating NFS 'Directory not empty' on overwrite.

    Overwriting rmtree's the old dir in place, which fails on networked
    filesystems when a chunk is still open (silly-renamed to a hidden .nfs*). So
    move any existing cube ASIDE (atomic rename, works with open handles), write
    fresh, then best-effort delete the stash.
    """
    zpath = Path(zpath)
    stash = None
    if zpath.exists():
        stash = zpath.with_name(f"{zpath.name}.old-{os.getpid()}-{int(time.time())}")
        zpath.rename(stash)
    with warnings.catch_warnings():
        # Zarr v3 warns that consolidated metadata isn't in the v3 spec; xarray
        # still writes+reads it and it speeds opening a many-variable cube.
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        ds.to_zarr(zpath, mode="w-", consolidated=True, encoding=encoding)
    if stash is not None:
        try:
            shutil.rmtree(stash)
        except OSError as exc:                 # NFS .nfs* leftovers -> non-fatal
            log.warning("  wrote %s but could not remove old cube %s (%s); delete it later",
                        zpath.name, stash.name, exc)


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    dc = project.datacube
    root = Path(project.output_dir)
    return {
        "aligned_root": root,                         # per-product <DIR>/aligned/<aoi>
        "out_dir": root / dc.output_subdir,
        "chunks": dict(dc.chunks),
        "fill_mur_water": bool(dc.fill_mur_water),
        "water_level": bool(dc.water_level),
        # Needed to resolve the datum offset per AoI (region override + sidecar lookup).
        "project": project,
        "compression": dc.compression,
        "overwrite": bool(dc.overwrite),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Assemble one Zarr cube per AoI from the pre-computed shared grids."""
    out_dir = eff["out_dir"]
    overwrite = eff["overwrite"]
    days = pd.date_range(eff["time"]["start_date"], eff["time"]["end_date"], freq="D")

    names = list(grids)
    if only_aoi:
        req = set(only_aoi)
        missing = req - set(names)
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        names = [n for n in names if n in req]

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        g = grids[name]
        zpath = out_dir / f"{name}.zarr"
        if zpath.exists() and not overwrite:
            log.info("=== %s: %s exists, skipping (use overwrite) ===", name, zpath.name)
            continue
        if dry_run:
            log.info("=== %s: [dry-run] would assemble %d day(s) -> %s ===",
                     name, len(days), zpath.name)
            continue

        log.info("=== assembling %s (%d days, grid=%dx%d) ===", name, len(days), g.width, g.height)
        ds = assemble_aoi(g, eff, days)
        write_zarr_safe(ds, zpath, build_encoding(ds, eff["compression"], eff["chunks"]))
        log.info("  wrote %s  vars=%d shape=(t=%d,y=%d,x=%d)", zpath.name,
                 len(ds.data_vars), ds.sizes["time"], ds.sizes["y"], ds.sizes["x"])
    log.info("Done.")


def assemble(project: Project, *, grids=None, aois=None, dry_run=False,
             overwrite=False) -> None:
    """Assemble datacubes for a validated Project. Terminal pipeline stage.

    Same signature as every product's acquire(); reads only the aligned files the
    acquisition stages wrote, so it must run AFTER them.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    run(eff, grids, aois, dry_run)


def main():
    ap = argparse.ArgumentParser(description="coastal_sst_data datacube assembler.")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Assemble only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing .zarr cubes")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    project = load_config(args.config)
    assemble(project, aois=args.aoi, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
