#!/usr/bin/env python3
"""
coastal_sst_data -- static land-cover / water mask from ESA WorldCover via
Microsoft Planetary Computer (free STAC + Cloud-Optimized GeoTIFF access).

This is ONE of several interchangeable landcover SOURCE modules. Every
`landcover_<source>.py` module (landcover_esa today; a future landcover_gee using
the JRC + Landsat-NDWI water mask) honours the SAME CONTRACT so downstream code
is source-agnostic:

  * entry point:  acquire(project, *, grids=None, aois=None, dry_run=False, overwrite=False)
  * output:       one STATIC aligned NetCDF per AoI at
                    <output_dir>/LANDCOVER/aligned/<aoi>/<aoi>.nc
  * variables:    landcover (class code, int16) and water (uint8, 1=water) on
                  dims (y, x), reprojected onto the shared AoiGrid. No time dim.

The assembler uses `water` as an authoritative LAND override: pixels the
classifier calls non-water are forced to land even when bathymetry is below sea
level -- which fixes diked/reclaimed farmland (negative elevation but not water).

Here the source is ESA WorldCover 10 m: a STAC search of Planetary Computer's
`esa-worldcover` collection, then a windowed COG read of the `map` asset (HTTP
range requests -- the full 3x3 deg tile is never downloaded) reprojected with
nearest-neighbour (categorical) onto the AoI grid. WorldCover classes:
  10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 70 snow/ice,
  80 permanent water, 90 herbaceous wetland, 95 mangrove, 100 moss/lichen.

Auth: NONE. Planetary Computer signs asset URLs anonymously (free).

Usage:
    python -m coastal_sst_data.processes.landcover_esa --config config.yaml
    python -m coastal_sst_data.processes.landcover_esa --config config.yaml --dry-run
    python -m coastal_sst_data.processes.landcover_esa --config config.yaml --aoi padilla_bay
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from pyproj import Transformer
from shapely.ops import transform as shp_transform

from ..config import Project, DataProduct, load_config
from ..grid import AoiGrid, project_grids

log = logging.getLogger(__name__)

# --- ESA WorldCover product constants (shared by all landcover_<source> defaults) #
SOURCE = "landcover"
COLLECTION = "esa-worldcover"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
ASSET = "map"                       # the classification COG asset
DEFAULT_YEAR = 2021                 # WorldCover v200 (2021); 2020 -> v100
DEFAULT_WATER_CLASSES = [80]        # 80 = permanent water bodies


# --------------------------------------------------------------------------- #
# STAC discovery (Planetary Computer; lazy import so the module loads without the
# PC client installed -- only `acquire` needs it).
# --------------------------------------------------------------------------- #
def _item_year(item) -> Optional[int]:
    """Acquisition year of a STAC item (from start_datetime/datetime), or None."""
    dt = item.properties.get("start_datetime") or item.properties.get("datetime")
    try:
        return int(str(dt)[:4])
    except (TypeError, ValueError):
        return None


def search_items(collection, stac_url, bbox, year):
    """Signed WorldCover STAC items over the AoI bbox, filtered to `year`.

    Falls back to all returned items if none match the year (so an unexpected
    version labelling still yields a mask rather than nothing).
    """
    import planetary_computer
    from pystac_client import Client

    cat = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=[collection], bbox=list(bbox)).items())
    matched = [it for it in items if _item_year(it) == year]
    if items and not matched:
        log.warning("  no WorldCover tiles labelled year=%d; using all %d returned",
                    year, len(items))
    return matched or items


# --------------------------------------------------------------------------- #
# Per-tile windowed COG read -> shared grid; mosaic tiles -> landcover + water
# --------------------------------------------------------------------------- #
def _read_map(href, g: AoiGrid):
    """Open the WorldCover COG, read ONLY the AoI window, reproject onto the grid.

    Range reads (via rioxarray/GDAL) pull just the windowed blocks. Class codes
    are categorical, so nearest-neighbour resampling; nodata = 0 (not a class).
    """
    da = rioxarray.open_rasterio(href, masked=False)     # uint8 class codes
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    cog_crs = da.rio.crs
    to_cog = Transformer.from_crs(g.target_crs, cog_crs, always_xy=True).transform
    minx, miny, maxx, maxy = shp_transform(to_cog, g.geom_proj).bounds
    pad = 300.0  # a few pixels of slack (metres)
    da = da.rio.clip_box(minx - pad, miny - pad, maxx + pad, maxy + pad, crs=cog_crs)
    return da.rio.reproject(dst_crs=g.target_crs, shape=(g.height, g.width),
                            transform=g.transform, resampling=Resampling.nearest, nodata=0)


def items_to_dataset(items, g: AoiGrid, water_classes, aoi_id: str) -> Optional[xr.Dataset]:
    """Mosaic WorldCover tiles onto the grid -> static landcover + water Dataset."""
    lc = None
    for it in items:
        layer = _read_map(it.assets[ASSET].href, g)
        # Mosaic: keep what we have, fill nodata (0) cells from the next tile.
        lc = layer if lc is None else lc.where(lc > 0, layer)
    if lc is None:
        return None

    lc = lc.astype("int16")
    water = lc.isin(list(water_classes)).astype("uint8")
    ds = xr.Dataset({"landcover": lc, "water": water})
    ds["landcover"].attrs["long_name"] = "ESA WorldCover class code (0 = no data)"
    ds["water"].attrs["long_name"] = f"WorldCover water (class in {list(water_classes)})"
    ds = ds.rio.write_crs(g.target_crs)
    ds.attrs.update(aoi_id=aoi_id, source="ESA WorldCover via Planetary Computer",
                    processing="PC STAC + COG windowed read -> nearest reproject to AoI grid")
    return ds


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "netcdf":
        path = out_dir / f"{aoi_id}.nc"
        ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    elif fmt == "geotiff":
        path = out_dir / aoi_id
        path.mkdir(exist_ok=True)
        for v in ds.data_vars:
            ds[v].rio.to_raster(path / f"{v}.tif")
    else:
        raise ValueError(f"Unknown output format: {fmt}")
    return path


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Acquire the WorldCover landcover/water mask onto the pre-computed AoI grids."""
    ds_cfg = eff["ds"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    water_classes = ds_cfg["water_classes"]
    year = ds_cfg["year"]

    names = list(grids)
    if only_aoi:
        req = set(only_aoi)
        missing = req - set(names)
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        names = [n for n in names if n in req]

    for name in names:
        g = grids[name]
        aoi_out = out_root / name
        out_f = aoi_out / f"{name}.nc"
        if not overwrite and out_f.exists():
            log.info("=== %s: %s exists, skipping (use overwrite) ===", name, out_f.name)
            continue

        log.info("=== AOI: %s (CRS=%s grid=%dx%d) | WorldCover %d | water_classes=%s ===",
                 name, g.target_crs, g.width, g.height, year, water_classes)
        if dry_run:
            log.info("  [dry-run] would fetch WorldCover %d and align to grid", year)
            continue

        try:
            items = search_items(ds_cfg["collection"], ds_cfg["stac_url"], g.search_bbox, year)
            if not items:
                log.warning("  no WorldCover tiles over AOI; skipping")
                continue
            ds = items_to_dataset(items, g, water_classes, name)
        except Exception as exc:
            log.warning("  skipping %s (%s)", name, exc)
            continue
        if ds is None:
            continue

        wf = float((ds["water"].values == 1).mean())
        log.info("  wrote %s  (water=%.0f%% of grid)", write_output(ds, aoi_out, name, fmt), 100 * wf)
    log.info("Done.")


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _opt(opts, name, default):
    """Read an optional override off a product-options bag (extra='allow')."""
    return getattr(opts, name, default) if opts is not None else default


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.landcover)
    if opts is None:
        raise ValueError("landcover is not a selected product in this config")

    ds_cfg = {
        "collection": _opt(opts, "collection", COLLECTION),
        "stac_url": _opt(opts, "stac_url", STAC_URL),
        "year": int(_opt(opts, "year", DEFAULT_YEAR)),
        "water_classes": list(_opt(opts, "water_classes", DEFAULT_WATER_CLASSES)),
    }
    return {
        "ds": ds_cfg,
        "out_dir": Path(project.output_dir) / "LANDCOVER" / "aligned",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire ESA WorldCover landcover/water for a validated Project.

    Entry point for pipeline.py -- same signature as every other product's
    acquire(). No credentials required (PC signs assets anonymously). Static
    layer, so it is safe to run at any point in the pipeline.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    run(eff, grids, aois, dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="coastal_sst_data ESA WorldCover landcover/water mask (Planetary Computer).")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Process only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess AoIs even if the aligned file exists")
    ap.add_argument("--dry-run", action="store_true", help="Search only; no download.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    project = load_config(args.config)
    acquire(project, aois=args.aoi, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
