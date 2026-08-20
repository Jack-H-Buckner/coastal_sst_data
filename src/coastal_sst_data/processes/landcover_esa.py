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

import logging
from pathlib import Path
from typing import Optional

import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, read_cog_window, select_aois
from .. import entry, net, provenance, report, store

log = logging.getLogger(__name__)

# --- ESA WorldCover product constants (shared by all landcover_<source> defaults) #
SOURCE = "landcover"
COLLECTION = "esa-worldcover"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
ASSET = "map"                       # the classification COG asset
DEFAULT_YEAR = 2021                 # WorldCover v200 (2021); 2020 -> v100
DEFAULT_WATER_CLASSES = [80]        # 80 = permanent water bodies

# `water` is a 0/1 mask where 0 means LAND -- a meaningful value, NOT nodata. It must be
# written with NO _FillValue, or xarray's default mask_and_scale=True decodes every land
# cell (0) to NaN on read, collapsing the mask to uniformly-water downstream. The COG read
# leaks _FillValue: 0 onto `water` (rio.reproject(nodata=0)); we scrub it and pin uint8.
WATER_ENCODING: dict = {"zlib": True, "complevel": 4, "dtype": "uint8", "_FillValue": None}


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

    def _search():
        cat = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
        return list(cat.search(collections=[collection], bbox=list(bbox)).items())

    items = net.retry(_search, what="WorldCover STAC search")
    matched = [it for it in items if _item_year(it) == year]
    if items and not matched:
        log.warning("  no WorldCover tiles labelled year=%d; using all %d returned",
                    year, len(items))
    return matched or items


# --------------------------------------------------------------------------- #
# Per-tile windowed COG read -> shared grid; mosaic tiles -> landcover + water
# --------------------------------------------------------------------------- #
def _read_map(href, g: AoiGrid):
    """The WorldCover COG, windowed onto the shared grid (see grid.read_cog_window).

    Class codes are categorical: nearest-neighbour resampling, raw uint8 (`masked=False`),
    and nodata = 0, which is not a WorldCover class -- so the mosaic below can tell an
    unfilled cell from a real one.
    """
    return read_cog_window(href, g, resampling=Resampling.nearest, masked=False,
                           pad_m=300.0, nodata=0)


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
    # `water` is a 0/1 mask -- 0 means LAND, not nodata -- and the COG read leaks CF decode
    # attrs onto it. See `store.clear_cf_decode_attrs` for what they do to a mask; this used
    # to be an inline copy of it, and the copy is why Landsat's `cloud` repeated the bug
    # years later. The explicit WATER_ENCODING then pins uint8 with no fill on write.
    water = store.clear_cf_decode_attrs(water)
    ds = xr.Dataset({"landcover": lc, "water": water})
    ds["landcover"].attrs["long_name"] = "ESA WorldCover class code (0 = no data)"
    ds["water"].attrs["long_name"] = f"WorldCover water (class in {list(water_classes)})"
    ds = ds.rio.write_crs(g.target_crs)
    ds.attrs.update(aoi_id=aoi_id, source="ESA WorldCover via Planetary Computer",
                    processing="PC STAC + COG windowed read -> nearest reproject to AoI grid")
    return ds


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Acquire the WorldCover landcover/water mask onto the pre-computed AoI grids."""
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("landcover")

    for name in names:
        g = grids[name]
        # Resolved PER AoI: `collection`/`stac_url` travel with `source` and are
        # region-overridable. `water_classes` is NOT -- it decides what the cube's `water`
        # channel MEANS, and a landmask that means one thing in one region and another
        # elsewhere is not a landmask.
        ds_cfg = eff["ds"][name]
        water_classes = ds_cfg["water_classes"]
        year = ds_cfg["year"]
        aoi_out = out_root / name
        out_f = aoi_out / f"{name}.nc"
        if store.done(out_f, store.REQUIRED_VARS["LANDCOVER"],
                      shape=(g.height, g.width), overwrite=overwrite):
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
            log.warning("  FAILED %s (%s)", name, exc)
            rep.fail(name, exc)
            continue
        if ds is None:
            continue

        wf = float((ds["water"].values == 1).mean())
        ds.attrs.update(**provenance.stamp(eff))
        # `water` needs an explicit fill-free uint8 encoding (see WATER_ENCODING); `landcover`
        # keeps the zlib default -- there 0 IS genuine nodata and it never reaches the cube.
        encoding = {"water": dict(WATER_ENCODING), "landcover": {"zlib": True, "complevel": 4}}
        # Static layer: the stem is the bare AoI name -- no time stamp to encode.
        log.info("  wrote %s  (water=%.0f%% of grid)",
                 store.write_output(ds, aoi_out, name, fmt, encoding=encoding), 100 * wf)
        rep.wrote(source="ESA WorldCover (Planetary Computer)")
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's land-cover settings, from its region-resolved options bag."""
    return {
        "collection": _opt(opts, "collection", COLLECTION),
        "stac_url": _opt(opts, "stac_url", STAC_URL),
        "year": int(_opt(opts, "year", DEFAULT_YEAR)),
        "water_classes": list(_opt(opts, "water_classes", DEFAULT_WATER_CLASSES)),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.landcover)
    if opts is None:
        raise ValueError("landcover is not a selected product in this config")

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.landcover))
               for a in project.all_areas},
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
    net.setup_gdal_env()      # windowed COG reads: deadline + retries
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(
        acquire,
        "coastal_sst_data ESA WorldCover landcover/water mask (Planetary Computer).")


if __name__ == "__main__":
    main()
