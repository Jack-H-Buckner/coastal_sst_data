#!/usr/bin/env python3
"""
coastal_sst_data -- Landsat C2 L2 Surface Temperature via Microsoft Planetary
Computer (free STAC + Cloud-Optimized GeoTIFF access).

This is ONE of several interchangeable Landsat SOURCE modules. Every
`landsat_<source>.py` module (landsat_pc, and future landsat_aws / landsat_gee)
honours the SAME CONTRACT so downstream code is source-agnostic:

  * entry point:  acquire(project, *, grids=None, aois=None, dry_run=False, overwrite=False)
  * output:       one aligned NetCDF per scene at
                    <output_dir>/LANDSAT/aligned/<aoi>/<aoi>_<YYYYMMDDTHHMMSS>.nc
  * variables:    sst (K|degC), cloud (1=cloud), water (1=water), valid (uint8),
                  on dims (time, y, x), reprojected onto the shared AoiGrid.

Only *scene discovery + pixel fetch* differs per source. Here: a STAC search of
Planetary Computer's `landsat-c2-l2` collection, then windowed COG reads of the
thermal / QA / SR assets (HTTP range requests -- the full 185 km scene is never
downloaded). The alignment, mask logic and output schema are identical across
sources, so the datacube assembler reads LANDSAT/aligned/ without caring which
source produced it.

Auth: NONE. Planetary Computer signs asset URLs anonymously (free). AWS/GEE
sources will need their own credentials; see the config `landsat.source` selector.

Usage:
    python -m coastal_sst_data.processes.landsat_pc --config config.yaml
    python -m coastal_sst_data.processes.landsat_pc --config config.yaml --dry-run
    python -m coastal_sst_data.processes.landsat_pc --config config.yaml --aoi tillamook_bay
"""

from __future__ import annotations

import logging
from datetime import timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling

from ..config import Project, DataProduct, opt as _opt
from ..grid import AoiGrid, project_grids, read_cog_window, select_aois
from .. import entry, naming, net, provenance, report, store

log = logging.getLogger(__name__)

# --- Landsat C2 L2 product constants (shared by all landsat_<source> modules) - #
SOURCE = "landsat"
COLLECTION = "landsat-c2-l2"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Level-2 scale/offset (USGS): ST digital number -> Kelvin; SR DN -> reflectance.
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
CDIST_SCALE = 0.01  # `cdist` (ST_CDIST) DN -> km
# Planetary Computer uses lowercase-hyphen platform names.
DEFAULT_PLATFORMS = ["landsat-8", "landsat-9"]


def _pc_platform(name: str) -> str:
    """Normalize a platform name to Planetary Computer style ('LANDSAT_8' -> 'landsat-8')."""
    return name.strip().lower().replace("_", "-")


# --------------------------------------------------------------------------- #
# STAC discovery (Planetary Computer; lazy import so the module loads without
# the PC client installed -- only `acquire` needs it).
# --------------------------------------------------------------------------- #
def search_scenes(collection, stac_url, bbox, start, end, platforms, cloud_max):
    """Return signed STAC items for Landsat scenes over the AoI bbox + dates."""
    import planetary_computer
    from pystac_client import Client

    def _search():
        cat = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
        search = cat.search(
            collections=[collection],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": cloud_max * 100.0},
                   "platform": {"in": [_pc_platform(p) for p in platforms]}},
        )
        return list(search.items())      # inside the retry: paging is where it fails

    return net.retry(_search, what="Landsat STAC search")


# --------------------------------------------------------------------------- #
# Per-scene: windowed COG reads -> shared grid -> sst/cloud/water/valid
# --------------------------------------------------------------------------- #
def _read_asset(href, g: AoiGrid, *, resampling, masked=True):
    """One Landsat COG asset, windowed onto the shared grid (see grid.read_cog_window).

    `masked=False` keeps raw integer values, which is what the bit-packed QA_PIXEL band
    needs. 300 m of pad is a few Landsat pixels of slack for the reprojection edges.
    """
    return read_cog_window(href, g, resampling=resampling, masked=masked, pad_m=300.0)


def scene_to_dataset(item, g: AoiGrid, mask_cfg: dict, to_celsius: bool,
                     acq_time, aoi_id: str) -> Optional[xr.Dataset]:
    """Build the aligned sst/cloud/water/valid Dataset for one STAC item.

    sst: thermal DN -> Kelvin (or degC). water: NDWI from green/NIR SR. cloud:
    QA_PIXEL cloud/shadow/dilated bits, optionally buffered by ST_CDIST distance.
    """
    a = item.assets
    thermal_key = "lwir11" if "lwir11" in a else "lwir"   # L8/9 vs L4-7

    dn = _read_asset(a[thermal_key].href, g, resampling=Resampling.bilinear)
    kelvin = dn * ST_SCALE + ST_OFFSET
    sst = (kelvin - 273.15) if to_celsius else kelvin

    # Water via NDWI = (green - nir) / (green + nir).
    green = _read_asset(a["green"].href, g, resampling=Resampling.bilinear) * SR_SCALE + SR_OFFSET
    nir = _read_asset(a["nir08"].href, g, resampling=Resampling.bilinear) * SR_SCALE + SR_OFFSET
    ndwi = (green - nir) / (green + nir)
    water = ndwi >= float(mask_cfg.get("ndwi_threshold", 0.0))

    # Cloud via QA_PIXEL bits: dilated(1) | cloud(3) | shadow(4), + ST_CDIST buffer.
    qa = _read_asset(a["qa_pixel"].href, g, resampling=Resampling.nearest, masked=False).astype("int64")
    cloudy = ((qa & (1 << 1)) != 0) | ((qa & (1 << 3)) != 0) | ((qa & (1 << 4)) != 0)
    buf_km = float(mask_cfg.get("cloud_buffer_km", 1.0))
    if buf_km > 0 and "cdist" in a:
        cdist_km = _read_asset(a["cdist"].href, g, resampling=Resampling.bilinear) * CDIST_SCALE
        cloudy = cloudy | (cdist_km < buf_km)

    ds = xr.Dataset({
        "sst": sst.astype("float32"),
        "cloud": cloudy.astype("float32"),
        "water": water.astype("float32"),
    })
    ds["sst"].attrs["units"] = "degC" if to_celsius else "K"
    # Masks may be NaN outside the AoI; treat NaN as 0 for the logic mask.
    valid = (np.isfinite(ds["sst"])
             & (ds["water"].fillna(0) > 0)
             & ~(ds["cloud"].fillna(0) > 0))
    ds["valid"] = valid.astype("uint8")
    ds["valid"].attrs["long_name"] = "water & clear & finite SST"

    ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    ds.attrs.update(
        aoi_id=aoi_id,
        source=f"Landsat C2 L2 ST ({item.properties.get('platform')}) via Planetary Computer",
        processing="PC STAC + COG windowed read -> reprojected/clipped to AoI grid",
    )
    return ds


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Acquire Landsat (via Planetary Computer) onto the pre-computed AoI grids."""
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    mask_cfg = ds_cfg.get("masking", {})
    platforms = ds_cfg["platforms"]
    cloud_max = ds_cfg["cloud_cover_max"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]
    to_celsius = grid_cfg.get("to_celsius", False)

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("landsat")

    for name in names:
        g = grids[name]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d @ %.0fm) ===",
                 name, g.target_crs, g.width, g.height, g.resolution_m)

        items = search_scenes(ds_cfg["collection"], ds_cfg["stac_url"], g.search_bbox,
                              start, end, platforms, cloud_max)
        log.info("  %d Landsat scene(s) (cloud < %.0f%%)", len(items), cloud_max * 100)
        if not items:
            continue
        if dry_run:
            log.info("  [dry-run] would process %d scene(s)", len(items))
            continue

        aoi_out = out_root / name
        for it in sorted(items, key=lambda i: i.properties["datetime"]):
            acq = pd.Timestamp(it.datetime.astimezone(timezone.utc).replace(tzinfo=None))
            stem = naming.time_stem(name, acq)
            if store.done(aoi_out / f"{stem}.nc", store.REQUIRED_VARS["LANDSAT"],
                          shape=(g.height, g.width), overwrite=overwrite):
                log.info("  %s already processed, skipping", naming.time_stamp(acq))
                continue
            try:
                ds = scene_to_dataset(it, g, mask_cfg, to_celsius, acq, name)
            except Exception as exc:
                log.warning("    FAILED %s (%s)", it.id, exc)
                rep.fail(f"{name} {it.id}", exc)
                continue
            if ds is None:
                continue
            ds.attrs.update(**provenance.stamp(eff))
            log.info("      wrote %s", store.write_output(ds, aoi_out, stem, fmt))
            rep.wrote(source="Landsat C2 L2 (Planetary Computer)")
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.landsat)
    if opts is None:
        raise ValueError("landsat is not a selected product in this config")
    masking = _opt(opts, "masking", {}) or {}

    ds_cfg = {
        "collection": _opt(opts, "collection", COLLECTION),
        "stac_url": _opt(opts, "stac_url", STAC_URL),
        "platforms": list(_opt(opts, "platforms", DEFAULT_PLATFORMS)),
        "cloud_cover_max": float(_opt(opts, "cloud_cover_max", 0.7)),
        "masking": dict(masking),
    }
    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    return {
        "config_sha256": project.config_sha256,
        "ds": ds_cfg,
        "grid": grid_cfg,
        "out_dir": Path(project.output_dir) / "LANDSAT" / "aligned",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire Landsat via Planetary Computer for a validated Project.

    Entry point for pipeline.py -- same signature as every other product's
    acquire(). No credentials required (PC signs assets anonymously).
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    net.setup_gdal_env()      # windowed COG reads: deadline + retries
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(
        acquire, "coastal_sst_data Landsat C2 L2 acquisition (Planetary Computer).")


if __name__ == "__main__":
    main()
