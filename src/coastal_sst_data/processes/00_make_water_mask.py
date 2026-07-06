"""
00_make_water_mask.py
=====================
Create a high-resolution permanent water mask for the Tasmania east coast
study region. Run this once before 03_apply_calibration.py.

WHY THIS IS BETTER THAN NATURAL EARTH POLYGONS
-----------------------------------------------
Natural Earth 50m shorelines resolve features at ~500 m. At 100 m pixel
resolution the land mask misclassifies a 4–5 pixel fringe along every
coastline — exactly where validation sites (Maria Island NRS) and
estuarine sub-regions of interest lie.

This script builds a water mask from actual Landsat spectral data at 30 m
native resolution, resampled to 100 m for use in the SST pipeline.

DATA SOURCES (accessed via Google Earth Engine)
------------------------------------------------
Primary   : JRC Global Surface Water v1.4 (Pekel et al. 2016)
            'occurrence' band = % of observations 1984–2021 with open water
            Threshold ≥ 50 % → permanent / semi-permanent water

Secondary : Landsat 8 OLI NDWI composite (2015–2021)
            NDWI = (Green − NIR) / (Green + NIR)
            Band assignments (C02 L2 surface reflectance):
              Green = SR_B3,  NIR = SR_B5
            Median composite over cloud-free scenes; threshold ≥ 0.0

COMBINING THE TWO
-----------------
A pixel is marked WATER if EITHER source says water:
  water = (JRC_occurrence ≥ 50) OR (NDWI_median ≥ 0.0)

JRC is the backbone (broad temporal coverage, robust).
NDWI composite catches tidal flats and recent land-use change not fully
captured by JRC and handles estuarine turbid water slightly better.

OUTPUT
------
  data/processed/water_mask/water_mask_tas_east_coast.tif
  Band 1: uint8, 1 = water, 0 = land/no-data
  CRS: EPSG:4326, resolution: 100 m (same as Landsat BT exports)
  Spatial extent: matches config.yaml region bbox

GEE DRIVE FOLDER: tas_sst_landsat  (same as Landsat BT exports)
FILE PREFIX:      water_mask_tas_east_coast

After the GEE task completes, download the file from Google Drive and place
it at: data/processed/water_mask/water_mask_tas_east_coast.tif

Usage:
    python src/00_make_water_mask.py           # queue GEE export
    python src/00_make_water_mask.py --dry-run # inspect stats, no export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    import ee
except ImportError:
    print("earthengine-api required:  pip install earthengine-api")
    sys.exit(1)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
MASK_SUBDIR = "data/processed/water_mask"
MASK_FILENAME = "water_mask_tas_east_coast"
GEE_FOLDER = "tas_sst_landsat"


# ── Landsat surface reflectance band names (C02 L2) ───────────────────────────
_SR_BANDS = {
    "LANDSAT_5": {"green": "SR_B2", "nir": "SR_B4"},
    "LANDSAT_7": {"green": "SR_B2", "nir": "SR_B4"},
    "LANDSAT_8": {"green": "SR_B3", "nir": "SR_B5"},
    "LANDSAT_9": {"green": "SR_B3", "nir": "SR_B5"},
}
_SR_COLLECTIONS = {
    "LANDSAT_5": "LANDSAT/LT05/C02/T1_L2",
    "LANDSAT_7": "LANDSAT/LE07/C02/T1_L2",
    "LANDSAT_8": "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT_9": "LANDSAT/LC09/C02/T1_L2",
}

# Scale factor for Landsat C02 L2 surface reflectance
_SR_SCALE  = 0.0000275
_SR_OFFSET = -0.2


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _mask_target(cfg: dict) -> tuple[str, str]:
    """(subdir, filename-stem) for the water mask, from config paths.water_mask
    if set, else the legacy east-coast constants. Lets each region write its own
    distinctly-named mask (e.g. water_mask_nw_tas.tif)."""
    wm = (cfg.get("paths") or {}).get("water_mask")
    if wm:
        p = Path(wm)
        return str(p.parent), p.stem
    return MASK_SUBDIR, MASK_FILENAME


def _ndwi_image(img: "ee.Image", green_band: str, nir_band: str) -> "ee.Image":
    """Compute per-image NDWI, masking clouds via QA_PIXEL."""
    # Cloud / shadow mask from QA_PIXEL (bit 3 = cloud, bit 4 = cloud shadow)
    qa = img.select("QA_PIXEL")
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(
                 qa.bitwiseAnd(1 << 4).eq(0))

    green = img.select(green_band).multiply(_SR_SCALE).add(_SR_OFFSET)
    nir   = img.select(nir_band).multiply(_SR_SCALE).add(_SR_OFFSET)
    ndwi  = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    return ndwi.updateMask(cloud_mask)


def download_water_mask_direct(image: "ee.Image", bbox: list, out_path: Path) -> bool:
    """Download the water mask GeoTIFF immediately via getDownloadURL() — no Drive.

    The mask is a small uint8 raster (~1000x550 px over these bboxes, <1 MB), well
    within the getDownloadURL() size limit, so a single synchronous request works.
    Returns True on success.
    """
    import io
    import time
    import zipfile

    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_region = ee.Geometry.Rectangle(bbox)
    print(f"  Requesting download URL for {out_path.name}…")
    url = image.toUint8().getDownloadURL({
        "name": out_path.stem,
        "scale": 100,
        "crs": "EPSG:4326",
        "region": export_region,
        "format": "GeoTIFF",
    })
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  FAILED after 3 attempts: {e}")
                return False
            print(f"  retry {attempt + 1}/3: {e}")
            time.sleep(5)

    # GEE returns a zip containing the GeoTIFF.
    if "zip" in resp.headers.get("Content-Type", "") or resp.content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            tifs = [n for n in zf.namelist() if n.endswith(".tif")]
            if not tifs:
                print("  ERROR: no .tif in downloaded zip")
                return False
            with zf.open(tifs[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    else:
        out_path.write_bytes(resp.content)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    return True


def make_water_mask(cfg: dict, dry_run: bool = False, direct: bool = False,
                    max_ndwi_scenes: int | None = None) -> None:
    region = cfg["region"]
    bbox = [region["lon_min"], region["lat_min"],
            region["lon_max"], region["lat_max"]]
    study_area = ee.Geometry.Rectangle(bbox)

    # ── 1. JRC Global Surface Water (permanent water occurrence ≥ 50 %) ──────
    print("  Loading JRC Global Surface Water…")
    jrc = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
             .select("occurrence")
             .clip(study_area))
    jrc_water = jrc.gte(50).rename("jrc_water")

    if dry_run:
        mean_occ = jrc.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=study_area,
            scale=100,
            maxPixels=1e8,
        ).getInfo()
        print(f"  JRC mean occurrence in bbox: {mean_occ.get('occurrence', 'N/A'):.1f} %")

    # ── 2. Landsat NDWI composite (2015–2021) ────────────────────────────────
    print("  Building Landsat NDWI composite (2015–2021)…")
    ndwi_images = []
    for platform, collection_id in _SR_COLLECTIONS.items():
        if platform == "LANDSAT_5":
            # L5 decommissioned June 2013; use 1984-2012
            date_start, date_end = "1984-03-01", "2012-12-31"
        elif platform == "LANDSAT_9":
            # L9 has limited coverage before 2022; include from 2022
            date_start, date_end = "2022-01-01", "2022-12-31"
        else:
            date_start, date_end = "2015-01-01", "2021-12-31"

        bands = _SR_BANDS[platform]
        col = (ee.ImageCollection(collection_id)
                 .filterBounds(study_area)
                 .filterDate(date_start, date_end)
                 .filter(ee.Filter.lt("CLOUD_COVER", 20)))

        # Cap the composite to the least-cloudy N scenes per platform. The median
        # NDWI water mask is stable with a few dozen clear scenes, and capping
        # keeps the on-the-fly computation light enough for getDownloadURL()
        # (--direct) over large regions, which otherwise 400s on a 400+ scene median.
        if max_ndwi_scenes:
            col = col.sort("CLOUD_COVER").limit(int(max_ndwi_scenes))

        n = col.size().getInfo()
        cap_note = f", capped to {max_ndwi_scenes}" if max_ndwi_scenes else ""
        print(f"    {platform}: {n} scenes (cloud < 20 %{cap_note}) in {date_start[:4]}–{date_end[:4]}")

        if n > 0:
            ndwi_col = col.map(
                lambda img: _ndwi_image(img, bands["green"], bands["nir"])
            )
            ndwi_images.append(ndwi_col)

    if not ndwi_images:
        print("  WARNING: No Landsat scenes found for NDWI composite — using JRC only.")
        combined_water = jrc_water.rename("water")
    else:
        all_ndwi = ndwi_images[0]
        for extra in ndwi_images[1:]:
            all_ndwi = all_ndwi.merge(extra)

        ndwi_median = all_ndwi.median().clip(study_area)
        ndwi_water  = ndwi_median.gte(0.0).rename("ndwi_water")

        # ── 3. Combine: water if EITHER JRC or NDWI ──────────────────────────
        # Apply a 1-pixel morphological closing to fill single-pixel holes in
        # water bodies (e.g. glint-affected pixels inside bays).
        combined_water = (jrc_water.Or(ndwi_water)
                                   .focal_max(radius=1, kernelType="square")
                                   .focal_min(radius=1, kernelType="square")
                                   .rename("water"))

        if dry_run:
            water_frac = combined_water.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=study_area,
                scale=100,
                maxPixels=1e8,
            ).getInfo()
            print(f"  Combined water fraction in bbox: {water_frac.get('water', 0):.3f}")

    print()
    if dry_run:
        print("Dry-run complete — no GEE export submitted.")
        print("Inspect the statistics above and re-run without --dry-run to export.")
        return

    # ── 4. Export ─────────────────────────────────────────────────────────────
    mask_subdir, mask_filename = _mask_target(cfg)

    if direct:
        out_path = Path(mask_subdir) / f"{mask_filename}.tif"
        ok = download_water_mask_direct(combined_water, bbox, out_path)
        if ok:
            print()
            print(f"Water mask written directly to {out_path} — no Drive step.")
            print("  Next: python src/03_apply_calibration.py (uses it automatically)")
        return

    export_region = ee.Geometry.Rectangle(bbox)
    task = ee.batch.Export.image.toDrive(
        image=combined_water.toUint8(),
        description=mask_filename,
        folder=GEE_FOLDER,
        fileNamePrefix=mask_filename,
        region=export_region,
        scale=100,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    task.start()
    print(f"GEE export task submitted: {mask_filename}")
    print(f"  Monitor: https://code.earthengine.google.com/tasks")
    print(f"  Output:  Google Drive / {GEE_FOLDER} / {mask_filename}.tif")
    print()
    print("After downloading from Drive:")
    print(f"  Place at: {mask_subdir}/{mask_filename}.tif")
    print("  Then re-run: python src/03_apply_calibration.py")
    print("  (it will automatically use the new water mask)")


def parse_args():
    p = argparse.ArgumentParser(description="Create GEE water mask for Tasmania SST pipeline")
    p.add_argument("--dry-run", action="store_true",
                   help="Print scene counts and water fraction stats; no GEE export.")
    p.add_argument("--direct", action="store_true",
                   help="Download the mask GeoTIFF directly via getDownloadURL() to "
                        "paths.water_mask instead of exporting to Google Drive.")
    p.add_argument("--max-ndwi-scenes", type=int, default=None,
                   help="Cap the NDWI composite to the least-cloudy N scenes per "
                        "platform (default: 80 in --direct mode, uncapped for Drive).")
    p.add_argument("--config", default=str(CONFIG_PATH))
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(Path(args.config))

    print("=" * 60)
    print("Tasmania SST — Water Mask Generation")
    print("=" * 60)
    region = cfg["region"]
    print(f"Region: lon [{region['lon_min']}, {region['lon_max']}]  "
          f"lat [{region['lat_min']}, {region['lat_max']}]")
    _msub, _mname = _mask_target(cfg)
    print(f"Output: {_msub}/{_mname}.tif")
    print()

    print("Initialising Google Earth Engine…")
    gee_project = cfg["project"].get("gee_project")
    ee.Initialize(project=gee_project)
    print(f"GEE initialised (project={gee_project}).")
    print()

    max_ndwi = args.max_ndwi_scenes
    if max_ndwi is None and args.direct:
        max_ndwi = 80  # keep getDownloadURL compute within budget for large regions
    make_water_mask(cfg, dry_run=args.dry_run, direct=args.direct, max_ndwi_scenes=max_ndwi)


if __name__ == "__main__":
    main()
