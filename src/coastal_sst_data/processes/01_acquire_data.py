"""
01_acquire_data.py
==================
Stage 1 of the SST calibration pipeline.

Downloads MODIS Terra L2 SST granules and queries Google Earth Engine
for coincident Landsat scenes over the Tasmania study region.

Methodology: Speiser & Largier (2024), Remote Sensing 16(23), 4477
Adapted for Tasmania East Coast

MODES
-----
  Default (calibration) mode
    Finds Landsat scenes that are coincident with clear-sky MODIS overpasses.
    Downloads MODIS granules and exports coincident Landsat BT to Google Drive.
    Use this when deriving or re-deriving calibration constants (02_pixel_matching.py).

  --landsat-only mode (recommended for extending the time-series)
    Exports all clear-sky Landsat BT scenes without touching MODIS.
    Use when the calibration is already done and you just want more SST scenes.
    Storage: ~1.7 MB/scene → ~7 years of L7+L8 ≈ 1–2 GB total.

STORAGE ESTIMATES (18-year 2005–2022, path 91 row 90)
------------------------------------------------------
  ~80 scenes/year × 18 years = ~1440 candidate scenes (50 % cloud filter)
  Landsat BT GeoTIFFs : ~1440 × 1.7 MB ≈  2.4 GB
  SST calibrated TIFs : ~1440 × 0.7 MB ≈  1.0 GB
  MODIS raw (default) : NOT needed for --landsat-only
  ─────────────────────────────────────────────────
  Landsat-only total  : ≈ 3.4 GB   (well within 10 GB budget)

Usage:
    python src/01_acquire_data.py                              # calibration mode
    python src/01_acquire_data.py --landsat-only               # all clear-sky scenes
    python src/01_acquire_data.py --start 2014-01-01 --end 2021-02-28 --landsat-only
    python src/01_acquire_data.py --dry-run                    # query only, no downloads
    python src/01_acquire_data.py --landsat-only --dry-run     # count scenes, estimate storage
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# ---- Optional imports (install as needed) ----
try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False
    print("Warning: earthaccess not installed. MODIS download will not work.")
    print("  Install with: pip install earthaccess")

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("Warning: earthengine-api not installed. Landsat query will not work.")
    print("  Install with: pip install earthengine-api")

# ---- Setup ----
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"acquire_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ==============================================================
# MODIS ACQUISITION (via earthaccess + NASA OB.DAAC)
# ==============================================================

def search_modis_granules(cfg: dict, start: str, end: str, logger: logging.Logger) -> list:
    """
    Search NASA OB.DAAC for MODIS Terra L2 SST granules covering the study region.
    Returns a list of earthaccess DataGranule objects.
    """
    if not EARTHACCESS_AVAILABLE:
        raise ImportError("earthaccess is required for MODIS download.")

    region = cfg["region"]
    bbox = (
        region["lon_min"],
        region["lat_min"],
        region["lon_max"],
        region["lat_max"],
    )

    logger.info(f"Searching MODIS Terra L2 SST: {start} to {end}")
    logger.info(f"Bounding box: {bbox}")

    # Authenticate (uses ~/.netrc or prompts interactively)
    # You need a free NASA Earthdata account: https://urs.earthdata.nasa.gov
    earthaccess.login(strategy="netrc")

    results = earthaccess.search_data(
        short_name="MODIS_T-JPL-L2P-v2019.0",
        temporal=(start, end),
        bounding_box=bbox,
        count=-1,  # Return all matching granules
    )

    logger.info(f"Found {len(results)} MODIS granules")
    return results


def download_modis_granules(granules: list, output_dir: Path, logger: logging.Logger) -> list:
    """
    Download MODIS granules to local disk.
    Returns list of downloaded file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {len(granules)} MODIS granules to {output_dir}")

    downloaded = earthaccess.download(granules, local_path=str(output_dir))
    logger.info(f"Downloaded {len(downloaded)} files")
    return downloaded


def filter_granules_to_dates(granules: list, target_dates: set, logger: logging.Logger,
                             daytime_only: bool = True,
                             day_window: int = 1) -> list:
    """
    Filter a list of MODIS DataGranule objects to only those whose acquisition
    date (UTC) falls on or within ±day_window days of one of the target_dates.

    The day_window buffer catches cross-date-boundary cases where MODIS and Landsat
    overpass the same region near midnight UTC (common at Tasmania, ~UTC+10).

    Parameters:
        target_dates: set of date strings in 'YYYY-MM-DD' format
        daytime_only: if True, skip nighttime granules (filename contains '-N-')
        day_window: number of calendar days either side of Landsat date to search (default 1)

    Returns:
        Filtered list of DataGranule objects.
    """
    # Expand target dates to include ±day_window for date-boundary matching
    expanded = set()
    for ds in target_dates:
        dt = datetime.strptime(ds, "%Y-%m-%d")
        for offset in range(-day_window, day_window + 1):
            expanded.add((dt + timedelta(days=offset)).strftime("%Y-%m-%d"))

    filtered = []
    skipped = 0
    skipped_night = 0
    for granule in granules:
        # Skip nighttime granules if requested
        if daytime_only:
            try:
                native_id = granule["meta"].get("native-id", "")
                if "-N-" in native_id:
                    skipped_night += 1
                    continue
            except (KeyError, TypeError):
                pass

        try:
            modis_time = datetime.strptime(
                granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
                "%Y-%m-%dT%H:%M:%S.%fZ",
            )
            date_str = modis_time.strftime("%Y-%m-%d")
            if date_str in expanded:
                filtered.append(granule)
            else:
                skipped += 1
        except (KeyError, ValueError):
            skipped += 1

    logger.info(
        f"MODIS date filter: {len(filtered)} granules kept, {skipped} discarded "
        f"(no matching clear-sky Landsat date within ±{day_window} day(s))"
    )
    if daytime_only:
        logger.info(f"  Nighttime granules skipped: {skipped_night}")
    return filtered


def check_modis_cloud_fraction(
    filepath: Path,
    quality_min: int = 4,
    lon_min: float = None,
    lon_max: float = None,
    lat_min: float = None,
    lat_max: float = None,
) -> float:
    """
    Check the fraction of cloud-contaminated pixels in a MODIS granule.

    Uses the GHRSST quality_level convention:
        0 = no data, 1 = bad, 2 = worst, 3 = low, 4 = acceptable, 5 = best
    Pixels with quality_level >= quality_min are counted as 'good/clear'.

    GHRSST L2P files (MODIS_T-JPL-L2P-v2019.0) store quality_level at the root
    level. Older MODIS HDF products may use a 'geophysical_data' group.

    When lat/lon bounds are provided the cloud fraction is computed only over
    pixels that fall within that bounding box. This avoids the whole-swath bias
    where a clear-sky patch over Tasmania is masked by clouds elsewhere in the
    2000 km × 1400 km MODIS granule.

    Returns:
        cloud_fraction (float): 0.0 = all clear, 1.0 = fully cloudy/bad
    """
    try:
        import xarray as xr
        import numpy as np

        # Try root level first (GHRSST L2P NetCDF format)
        try:
            ds = xr.open_dataset(filepath, engine="netcdf4")
            qual = ds["quality_level"].values
            lats = ds["lat"].values if "lat" in ds else None
            lons = ds["lon"].values if "lon" in ds else None
            ds.close()
        except (KeyError, Exception):
            # Fall back to geophysical_data group (older MODIS HDF-EOS format)
            ds = xr.open_dataset(filepath, group="geophysical_data", engine="netcdf4")
            qual = ds["quality_level"].values
            lats = None
            lons = None
            ds.close()

        qual = np.squeeze(qual)

        # Crop to bounding box if coordinates and bounds are available
        bbox_provided = all(v is not None for v in [lon_min, lon_max, lat_min, lat_max])
        if bbox_provided and lats is not None and lons is not None:
            lats = np.squeeze(lats)
            lons = np.squeeze(lons)
            in_box = (
                (lats >= lat_min) & (lats <= lat_max) &
                (lons >= lon_min) & (lons <= lon_max)
            )
            if in_box.any():
                qual = qual[in_box]
            else:
                # Granule does not overlap the study region at all
                return 1.0

        good_pixels = (qual >= quality_min).sum()
        total_pixels = qual.size
        cloud_fraction = 1.0 - (good_pixels / total_pixels)
        return float(cloud_fraction)

    except Exception:
        # If quality layer missing or file malformed, treat as cloudy
        return 1.0


def filter_dates_by_landsat_coverage(dates: list, bt_dir: Path, min_valid_frac: float = 0.05,
                                      logger: logging.Logger = None) -> set:
    """
    Filter Landsat dates to only those with enough valid (non-cloudy) water pixels.

    Reads existing Landsat BT GeoTIFFs and checks the fraction of valid pixels.
    Dates without a local BT file are kept (they haven't been exported yet).

    Parameters:
        dates: list of date strings 'YYYY-MM-DD'
        bt_dir: directory containing Landsat BT GeoTIFFs
        min_valid_frac: minimum fraction of valid pixels (default 5%)

    Returns:
        set of date strings that pass the coverage check
    """
    import rasterio
    import numpy as np

    kept = set()
    removed = 0
    no_file = 0

    for date_str in dates:
        date_compact = date_str.replace("-", "")
        # Find any BT file for this date
        matches = list(bt_dir.glob(f"{date_compact}_*_BT_*.tif"))
        if not matches:
            # No local file — keep the date (may not be exported yet)
            kept.add(date_str)
            no_file += 1
            continue

        # Check the first matching file
        try:
            with rasterio.open(matches[0]) as src:
                data = src.read(1)
                valid = np.sum(~np.isnan(data) & (data > 0))
                frac = valid / data.size if data.size > 0 else 0
                if frac >= min_valid_frac:
                    kept.add(date_str)
                else:
                    removed += 1
        except Exception:
            kept.add(date_str)  # keep on error

    if logger:
        logger.info(f"Landsat coverage filter: {len(kept)} dates kept, {removed} too cloudy "
                    f"(<{min_valid_frac*100:.0f}% valid pixels), {no_file} without local BT file")
    return kept


# ==============================================================
# LANDSAT ACQUISITION (via Google Earth Engine)
# ==============================================================

def initialise_gee(logger: logging.Logger, project: str = None):
    """Authenticate and initialise the GEE Python API.

    Parameters:
        project: Google Cloud project ID. Required for earthengine-api >= 0.1.370.
                 Set via config.yaml project.gee_project.
    """
    if not GEE_AVAILABLE:
        raise ImportError("earthengine-api is required for Landsat query.")

    logger.info("Initialising Google Earth Engine...")
    # First-time setup: run `earthengine authenticate` in terminal
    ee.Initialize(project=project)
    logger.info(f"GEE initialised (project={project}).")


def get_landsat_collection_id(platform: str) -> str:
    """Map platform name to GEE collection ID (Collection 2, Tier 1, Level 2)."""
    mapping = {
        "LANDSAT_5":  "LANDSAT/LT05/C02/T1_L2",
        "LANDSAT_7":  "LANDSAT/LE07/C02/T1_L2",
        "LANDSAT_8":  "LANDSAT/LC08/C02/T1_L2",
        "LANDSAT_9":  "LANDSAT/LC09/C02/T1_L2",
    }
    if platform not in mapping:
        raise ValueError(f"Unknown platform: {platform}. Choose from {list(mapping.keys())}")
    return mapping[platform]


def query_landsat_scenes(
    cfg: dict,
    start: str,
    end: str,
    cloud_cover_max: float,
    logger: logging.Logger,
) -> dict:
    """
    Query GEE for Landsat scenes covering the study region.
    Applies a cloud cover pre-filter (metadata only - no image download).

    Returns:
        dict mapping platform -> ee.ImageCollection
    """
    region = cfg["region"]
    study_area = ee.Geometry.Rectangle([
        region["lon_min"], region["lat_min"],
        region["lon_max"], region["lat_max"],
    ])

    collections = {}
    for platform in cfg["data_sources"]["landsat"]["platforms"]:
        collection_id = get_landsat_collection_id(platform)
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(study_area)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max * 100))  # GEE uses 0-100
        )
        count = collection.size().getInfo()
        logger.info(f"{platform}: {count} scenes found (cloud < {cloud_cover_max*100:.0f}%)")
        collections[platform] = collection

    return collections


def get_scene_dates(collection: "ee.ImageCollection") -> list:
    """Extract acquisition dates from a GEE ImageCollection."""
    dates = collection.aggregate_array("DATE_ACQUIRED").getInfo()
    return sorted(set(dates))


def export_landsat_bt(
    image: "ee.Image",
    platform: str,
    scene_id: str,
    output_dir: Path,
    region: dict,
    cfg: dict = None,
    scale_m: int = 100,
    mask_clouds: bool = True,
    aux_bands: bool = False,
    logger: logging.Logger = None,
):
    """
    Export Landsat brightness temperature as a GeoTIFF via GEE.

    Two modes controlled by aux_bands:

      aux_bands=False (default, production):
        Single-band export (BT °C). Server-side masks: QA_PIXEL + NDWI + ST_CDIST.
        Fast export, small files (~1.7 MB).

      aux_bands=True (diagnostic):
        4-band export (BT °C, ST_CDIST km, ST_QA K, NDWI). Server-side masks: QA_PIXEL + NDWI only.
        All filter variables preserved for post-download threshold experiments.
        Slower export, larger files.

    Parameters:
        cfg:         Full config dict (masking thresholds read from cfg["masking"])
        scale_m:     Output resolution in metres (100m target)
        mask_clouds: Apply per-scene cloud/water masking (default True)
        aux_bands:   Include ST_CDIST and ST_QA as bands 2-3 (default False)
    """
    # Band configuration per platform
    _BANDS = {
        "L5": {"bt": "ST_B6",  "green": "SR_B2", "nir": "SR_B4"},
        "L7": {"bt": "ST_B6",  "green": "SR_B2", "nir": "SR_B4"},
        "L8": {"bt": "ST_B10", "green": "SR_B3", "nir": "SR_B5"},
        "L9": {"bt": "ST_B10", "green": "SR_B3", "nir": "SR_B5"},
    }
    platform_short = platform.split("_")[1]  # "7", "8", "9"
    bands = _BANDS[f"L{platform_short}"]

    # Read masking thresholds from config (with defaults for backward compatibility)
    mask_cfg = cfg.get("masking", {}) if cfg else {}
    ndwi_thresh      = mask_cfg.get("ndwi_threshold", 0.1)
    cloud_buffer_px  = mask_cfg.get("cloud_buffer_pixels", 10)

    # --- BT in °C ---
    bt_image = image.select(bands["bt"])
    # Landsat L2 ST bands: scale 0.00341802, offset 149.0 → Kelvin → Celsius
    bt_kelvin = bt_image.multiply(0.00341802).add(149.0)
    bt_celsius = bt_kelvin.subtract(273.15).rename("BT").toFloat()

    if mask_clouds:
        # --- QA_PIXEL cloud mask ---
        # Bit 1 = dilated cloud, bit 3 = cloud, bit 4 = cloud shadow
        qa = image.select("QA_PIXEL")
        clear_sky = (qa.bitwiseAnd(1 << 1).eq(0)
                     .And(qa.bitwiseAnd(1 << 3).eq(0))
                     .And(qa.bitwiseAnd(1 << 4).eq(0)))

        # --- NDWI water filter ---
        _SR_SCALE  = 0.0000275
        _SR_OFFSET = -0.2
        green = image.select(bands["green"]).multiply(_SR_SCALE).add(_SR_OFFSET)
        nir   = image.select(bands["nir"]).multiply(_SR_SCALE).add(_SR_OFFSET)
        ndwi  = green.subtract(nir).divide(green.add(nir))
        water = ndwi.gte(ndwi_thresh)

        if aux_bands:
            # Diagnostic mode: QA + NDWI only; preserve ST_CDIST/ST_QA as bands
            combined_mask = clear_sky.And(water)
            mask_tag = f" [QA+NDWI({ndwi_thresh}), 4-band diagnostic]"
        else:
            # Production mode: also apply ST_CDIST cloud distance buffer
            # ST_CDIST is pre-scaled to km in GEE; convert pixel threshold to km
            cloud_dist = image.select("ST_CDIST")
            cloud_buffer_km = cloud_buffer_px * (scale_m / 1000)
            far_from_cloud = cloud_dist.gte(cloud_buffer_km)
            combined_mask = clear_sky.And(water).And(far_from_cloud)
            mask_tag = f" [QA+NDWI({ndwi_thresh})+CDIST({cloud_buffer_px}px)]"

        bt_celsius = bt_celsius.updateMask(combined_mask)
    else:
        mask_tag = " [unmasked]"

    # Build export image
    if aux_bands:
        st_cdist = image.select("ST_CDIST").multiply(0.01).rename("ST_CDIST").toFloat()
        st_qa = image.select("ST_QA").multiply(0.01).rename("ST_QA").toFloat()
        # Compute NDWI for export (may already exist from masking; recompute if not)
        if not mask_clouds:
            _SR_SCALE  = 0.0000275
            _SR_OFFSET = -0.2
            green = image.select(bands["green"]).multiply(_SR_SCALE).add(_SR_OFFSET)
            nir   = image.select(bands["nir"]).multiply(_SR_SCALE).add(_SR_OFFSET)
            ndwi  = green.subtract(nir).divide(green.add(nir))
        ndwi_band = ndwi.rename("NDWI").toFloat()
        if mask_clouds:
            st_cdist  = st_cdist.updateMask(combined_mask)
            st_qa     = st_qa.updateMask(combined_mask)
            ndwi_band = ndwi_band.updateMask(combined_mask)
        export_image = (bt_celsius.addBands(st_cdist)
                        .addBands(st_qa).addBands(ndwi_band))
        band_label = "4-band"
    else:
        export_image = bt_celsius
        band_label = "1-band"

    export_region = ee.Geometry.Rectangle([
        region["lon_min"], region["lat_min"],
        region["lon_max"], region["lat_max"],
    ])

    output_dir.mkdir(parents=True, exist_ok=True)

    task = ee.batch.Export.image.toDrive(
        image=export_image,
        description=f"BT_{scene_id}",
        folder="tas_sst_landsat",  # Google Drive folder name
        fileNamePrefix=f"{scene_id}_BT_{platform_short}",
        region=export_region,
        scale=scale_m,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    try:
        task.start()
    except ee.ee_exception.EEException as exc:
        if "already started" in str(exc).lower():
            if logger:
                logger.info(f"GEE export already queued: {scene_id} (skipping duplicate)")
            return
        raise

    if logger:
        logger.info(f"GEE export task started: {scene_id} ({band_label}: BT @ {scale_m}m){mask_tag}")
    return task


def download_landsat_bt_direct(
    image: "ee.Image",
    platform: str,
    scene_id: str,
    output_dir: Path,
    region: dict,
    cfg: dict = None,
    scale_m: int = 100,
    mask_clouds: bool = True,
    logger: logging.Logger = None,
    force: bool = False,
) -> "Path | None":
    """
    Download Landsat BT directly via getDownloadURL() — no Drive export needed.

    Applies the same masking as export_landsat_bt() (QA_PIXEL + NDWI + ST_CDIST)
    but downloads the GeoTIFF immediately instead of queuing a Drive export.

    Returns the saved file path, or None on failure.
    """
    import io
    import time
    import zipfile
    import requests

    _BANDS = {
        "L5": {"bt": "ST_B6",  "green": "SR_B2", "nir": "SR_B4"},
        "L7": {"bt": "ST_B6",  "green": "SR_B2", "nir": "SR_B4"},
        "L8": {"bt": "ST_B10", "green": "SR_B3", "nir": "SR_B5"},
        "L9": {"bt": "ST_B10", "green": "SR_B3", "nir": "SR_B5"},
    }
    platform_short = platform.split("_")[1]
    bands = _BANDS[f"L{platform_short}"]

    filename = f"{scene_id}_BT_{platform_short}.tif"
    out_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and out_path.exists() and out_path.stat().st_size > 1000:
        if logger:
            logger.info(f"  SKIP (exists): {filename}")
        return out_path

    # Read masking thresholds from config
    mask_cfg = cfg.get("masking", {}) if cfg else {}
    ndwi_thresh = mask_cfg.get("ndwi_threshold", 0.1)
    cloud_buffer_px = mask_cfg.get("cloud_buffer_pixels", 10)

    # BT in °C
    bt_image = image.select(bands["bt"])
    bt_kelvin = bt_image.multiply(0.00341802).add(149.0)
    bt_celsius = bt_kelvin.subtract(273.15).rename("BT").toFloat()

    if mask_clouds:
        qa = image.select("QA_PIXEL")
        clear_sky = (qa.bitwiseAnd(1 << 1).eq(0)
                     .And(qa.bitwiseAnd(1 << 3).eq(0))
                     .And(qa.bitwiseAnd(1 << 4).eq(0)))

        _SR_SCALE = 0.0000275
        _SR_OFFSET = -0.2
        green = image.select(bands["green"]).multiply(_SR_SCALE).add(_SR_OFFSET)
        nir = image.select(bands["nir"]).multiply(_SR_SCALE).add(_SR_OFFSET)
        ndwi = green.subtract(nir).divide(green.add(nir))
        water = ndwi.gte(ndwi_thresh)

        cloud_dist = image.select("ST_CDIST")
        cloud_buffer_km = cloud_buffer_px * (scale_m / 1000)
        far_from_cloud = cloud_dist.gte(cloud_buffer_km)
        combined_mask = clear_sky.And(water).And(far_from_cloud)
        bt_celsius = bt_celsius.updateMask(combined_mask)

    export_region = ee.Geometry.Rectangle([
        region["lon_min"], region["lat_min"],
        region["lon_max"], region["lat_max"],
    ])

    if logger:
        logger.info(f"  Downloading {filename} via getDownloadURL()...")
    try:
        url = bt_celsius.getDownloadURL({
            "name": filename.replace(".tif", ""),
            "scale": scale_m,
            "crs": "EPSG:4326",
            "region": export_region,
            "format": "GeoTIFF",
        })
    except Exception as e:
        if logger:
            logger.error(f"  ERROR getting download URL for {scene_id}: {e}")
        return None

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                if logger:
                    logger.error(f"  FAILED after 3 attempts for {scene_id}: {e}")
                return None
            if logger:
                logger.warning(f"  Retry {attempt+1}/3: {e}")
            time.sleep(5)

    # GEE returns a zip containing the GeoTIFF
    content_type = resp.headers.get("Content-Type", "")
    if "zip" in content_type or resp.content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
            if not tif_names:
                if logger:
                    logger.error(f"  ERROR: no .tif in zip for {filename}")
                return None
            with zf.open(tif_names[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    else:
        out_path.write_bytes(resp.content)

    size_mb = out_path.stat().st_size / 1e6
    if logger:
        logger.info(f"  Saved: {filename}  ({size_mb:.1f} MB)")
    return out_path


# ==============================================================
# COINCIDENCE MATCHING
# ==============================================================

def find_coincident_pairs(
    modis_granules: list,
    landsat_dates: list,
    max_diff_minutes: int = 360,
    logger: logging.Logger = None,
    best_per_date: bool = False,
    day_window: int = 1,
) -> list:
    """
    Find MODIS granules that are temporally coincident with Landsat acquisitions.

    MODIS Terra and Landsat both have ~10:30 AM local overpass times,
    so coincident pairs are common when skies are clear.

    Matching uses ±day_window calendar days to handle date-boundary cases (e.g.
    at Tasmania ~UTC+10, both instruments overpass near midnight UTC). The
    max_diff_minutes parameter is stored in config for documentation.

    If best_per_date=True, only the single closest MODIS granule per Landsat
    date is kept. This dramatically reduces MODIS download volume (~5× fewer
    granules) at the cost of fewer fallback options if the best granule is cloudy.

    Returns:
        List of (modis_granule, landsat_date) tuples
    """
    pairs = []

    for granule in modis_granules:
        # earthaccess DataGranule temporal metadata
        try:
            modis_time = datetime.strptime(
                granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
                "%Y-%m-%dT%H:%M:%S.%fZ",
            )
        except (KeyError, ValueError):
            continue

        for ls_date in landsat_dates:
            ls_dt = datetime.strptime(ls_date, "%Y-%m-%d")
            # Allow ±day_window calendar days to catch cross-date-boundary overpasses
            if abs((modis_time.date() - ls_dt.date()).days) <= day_window:
                pairs.append((granule, ls_date, modis_time))
                if logger:
                    logger.debug(f"Coincident pair: MODIS {modis_time} / Landsat {ls_date}")

    if logger:
        logger.info(f"Found {len(pairs)} coincident MODIS/Landsat pairs "
                    f"(±{day_window} day(s) matching, max_diff={max_diff_minutes}min)")

    if best_per_date and pairs:
        # Keep only the closest MODIS granule per Landsat date
        best = {}  # ls_date → (granule, ls_date, abs_diff)
        for granule, ls_date, modis_time in pairs:
            ls_dt = datetime.strptime(ls_date, "%Y-%m-%d").replace(hour=12)
            diff = abs((modis_time - ls_dt).total_seconds())
            if ls_date not in best or diff < best[ls_date][2]:
                best[ls_date] = (granule, ls_date, diff)
        pairs = [(g, d, t) for g, d, t in pairs if d in best
                 and g is best[d][0]]
        if logger:
            logger.info(f"  best_per_date: reduced to {len(pairs)} pairs "
                        f"({len(best)} unique Landsat dates)")

    # Strip the modis_time from the returned tuples for backward compatibility
    return [(g, d) for g, d, *_ in pairs]


# ==============================================================
# MAIN
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Acquire MODIS and Landsat data for Tasmania SST calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (overrides config)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (overrides config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Query only — no downloads or GEE exports. Prints scene counts and "
                             "storage estimates.")
    parser.add_argument("--landsat-only", action="store_true",
                        help="Export all clear-sky Landsat BT scenes without MODIS download. "
                             "Use when calibration is already derived and you want more SST scenes.")
    parser.add_argument("--no-mask", action="store_true",
                        help="Disable per-scene QA_PIXEL + NDWI cloud/water masking. "
                             "By default, cloud and non-water pixels are set to NoData.")
    parser.add_argument("--force", action="store_true",
                        help="Re-export scenes even if local BT file already exists. "
                             "Use when re-exporting with updated masking parameters.")
    parser.add_argument("--platforms", nargs="+", default=None,
                        help="Limit to specific platforms, e.g. --platforms LANDSAT_8 LANDSAT_9. "
                             "Default: all platforms from config.")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit total number of scenes to export. "
                             "Useful for test batches (e.g. --max-scenes 10).")
    parser.add_argument("--aux-bands", action="store_true",
                        help="Export ST_CDIST and ST_QA as auxiliary bands (3-band diagnostic mode). "
                             "Default: single-band BT with ST_CDIST applied as server-side mask.")
    parser.add_argument("--best-per-date", action="store_true",
                        help="Download only the single closest MODIS granule per Landsat date. "
                             "Reduces MODIS storage ~5× at cost of fewer fallback options.")
    parser.add_argument("--direct", action="store_true",
                        help="Download Landsat BT directly via getDownloadURL() instead of "
                             "exporting to Google Drive. No Drive API required.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml")
    return parser.parse_args()


def _estimate_storage(n_scenes: int, logger: logging.Logger):
    """Print storage estimates for n_scenes Landsat BT + SST files."""
    bt_mb  = n_scenes * 1.7
    sst_mb = n_scenes * 0.7
    total_mb = bt_mb + sst_mb
    logger.info("")
    logger.info("─" * 50)
    logger.info("STORAGE ESTIMATE")
    logger.info("─" * 50)
    logger.info(f"  Scenes (after cloud filter) : {n_scenes:,}")
    logger.info(f"  Landsat BT GeoTIFFs         : ~{bt_mb/1024:.1f} GB  ({n_scenes} × 1.7 MB)")
    logger.info(f"  SST calibrated GeoTIFFs     : ~{sst_mb/1024:.1f} GB  ({n_scenes} × 0.7 MB)")
    logger.info(f"  Total (Landsat-only)        : ~{total_mb/1024:.1f} GB")
    logger.info("─" * 50)


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    base_dir = Path(args.config).parent.parent

    logger = setup_logging(base_dir / cfg["paths"]["logs"])
    logger.info("=" * 60)
    logger.info("Tasmania SST Calibration - Data Acquisition")
    if args.landsat_only:
        logger.info("MODE: Landsat-only (MODIS download skipped)")
    logger.info("=" * 60)

    start = args.start or cfg["dates"]["start"]
    end   = args.end   or cfg["dates"]["end"]
    logger.info(f"Date range : {start} → {end}")
    logger.info(f"Dry-run    : {args.dry_run}")

    # ── 1. Query Landsat via GEE (metadata only — cheap) ─────────────────────
    gee_project = cfg["project"].get("gee_project")
    initialise_gee(logger, project=gee_project)
    cloud_max = cfg["data_sources"]["landsat"]["cloud_cover_max"]

    # Optionally limit to specific platforms via --platforms
    if args.platforms:
        cfg = {**cfg, "data_sources": {**cfg["data_sources"],
               "landsat": {**cfg["data_sources"]["landsat"],
                           "platforms": args.platforms}}}
        logger.info(f"Platforms  : {args.platforms}")

    ls_collections = query_landsat_scenes(cfg, start, end, cloud_max, logger)

    all_ls_dates = []
    platform_scenes: dict[str, list] = {}   # platform → [(date, ee.Image), …]
    for platform, collection in ls_collections.items():
        dates = get_scene_dates(collection)
        all_ls_dates.extend(dates)
        platform_scenes[platform] = dates
        logger.info(f"  {platform}: {len(dates)} clear-sky scenes")

    total_scenes = sum(len(d) for d in platform_scenes.values())
    logger.info(f"Total clear-sky Landsat scenes: {total_scenes:,}")

    if total_scenes == 0:
        logger.warning("No clear Landsat scenes found in date range. Nothing to do.")
        return

    # ── Storage estimate (always show) ───────────────────────────────────────
    _estimate_storage(total_scenes, logger)

    # ════════════════════════════════════════════════════════════════════════
    # LANDSAT-ONLY MODE — export all clear-sky scenes, skip MODIS entirely
    # ════════════════════════════════════════════════════════════════════════
    if args.landsat_only:
        if args.dry_run:
            logger.info("Dry-run complete. Re-run without --dry-run to start GEE exports.")
            return

        dl_method = "direct (getDownloadURL)" if args.direct else "Drive export"
        logger.info(f"\nDownloading ALL clear-sky Landsat scenes via {dl_method}…")
        if not args.direct:
            logger.info("Files will appear in Google Drive folder: tas_sst_landsat")
        ls_dir = base_dir / cfg["paths"]["proc_landsat"]
        n_queued = 0
        for platform, collection in ls_collections.items():
            dates = platform_scenes[platform]
            for ls_date in dates:
                scene_id = f"{ls_date.replace('-', '')}_{platform.split('_')[1]}"
                plat_short = platform.split('_')[1]
                out_path = ls_dir / f"{scene_id}_BT_{plat_short}.tif"
                if not args.force and out_path.exists():
                    logger.info(f"  SKIP (exists): {out_path.name}")
                    continue
                scene = (
                    collection
                    .filterDate(
                        ls_date,
                        (datetime.strptime(ls_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                    )
                    .first()
                )
                if scene:
                    if args.direct:
                        download_landsat_bt_direct(
                            scene, platform, scene_id, ls_dir, cfg["region"],
                            cfg=cfg, mask_clouds=not args.no_mask,
                            logger=logger, force=args.force)
                    else:
                        export_landsat_bt(scene, platform, scene_id, ls_dir, cfg["region"],
                                         cfg=cfg, mask_clouds=not args.no_mask,
                                         aux_bands=args.aux_bands, logger=logger)
                    n_queued += 1
                    if args.max_scenes and n_queued >= args.max_scenes:
                        break
            if args.max_scenes and n_queued >= args.max_scenes:
                break

        if args.no_mask:
            mask_str = "OFF (--no-mask)"
        elif args.aux_bands and not args.direct:
            mask_str = "ON (QA_PIXEL + NDWI); ST_CDIST + ST_QA as aux bands"
        else:
            mask_str = "ON (QA_PIXEL + NDWI + ST_CDIST)"
        logger.info(f"\n{n_queued} scenes processed.  Cloud masking: {mask_str}")
        if not args.direct:
            logger.info("Monitor progress at: https://code.earthengine.google.com/tasks")
            logger.info("Files will appear in your Google Drive under: tas_sst_landsat/")
        logger.info("After downloading, run: python src/03_apply_calibration.py")
        return

    # ════════════════════════════════════════════════════════════════════════
    # DEFAULT (CALIBRATION) MODE — MODIS + coincident Landsat
    # ════════════════════════════════════════════════════════════════════════
    ls_date_set = set(all_ls_dates)

    # ── 1b. Pre-filter Landsat dates by pixel coverage ────────────────────
    ls_dir = base_dir / cfg["paths"]["proc_landsat"]
    ls_date_set = filter_dates_by_landsat_coverage(
        all_ls_dates, ls_dir, min_valid_frac=0.05, logger=logger,
    )

    # ── 2. Search MODIS granules ──────────────────────────────────────────
    all_modis_granules = search_modis_granules(cfg, start, end, logger)

    # ── 3. Filter MODIS to clear-sky Landsat dates (daytime only) ─────────
    day_window = cfg["data_sources"]["modis"].get("day_window", 1)
    modis_granules = filter_granules_to_dates(all_modis_granules, ls_date_set, logger,
                                              daytime_only=True, day_window=day_window)

    # ── 4. Find coincident pairs ─────────────────────────────────────────
    max_diff = cfg["data_sources"]["modis"]["max_time_diff_minutes"]
    pairs = find_coincident_pairs(modis_granules, all_ls_dates, max_diff, logger,
                                   best_per_date=args.best_per_date,
                                   day_window=day_window)

    if not pairs:
        logger.warning("No coincident MODIS/Landsat pairs found. Nothing to download.")
        return

    if args.dry_run:
        logger.info(f"Dry-run: {len(pairs)} coincident pairs found. "
                    "Re-run without --dry-run to download.")
        return

    # ── 5. Download coincident MODIS granules ───────────────────────────
    seen_ids = set()
    unique_modis = []
    for granule, _ in pairs:
        gid = granule["meta"]["concept-id"]
        if gid not in seen_ids:
            seen_ids.add(gid)
            unique_modis.append(granule)

    modis_dir = base_dir / cfg["paths"]["raw_modis"]
    downloaded = download_modis_granules(unique_modis, modis_dir, logger)

    # ── 6. Post-download cloud check ─────────────────────────────────────
    nc_files = [Path(fp) for fp in downloaded if Path(fp).suffix == ".nc"]
    logger.info(f"Checking cloud fractions for {len(nc_files)} .nc files…")
    region = cfg["region"]
    clear_granules = []
    for fp in nc_files:
        cf = check_modis_cloud_fraction(
            fp,
            lon_min=region["lon_min"],
            lon_max=region["lon_max"],
            lat_min=region["lat_min"],
            lat_max=region["lat_max"],
        )
        if cf < 0.5:
            clear_granules.append(fp)
            logger.info(f"  CLEAR ({cf:.2%} cloud): {fp.name}")
        else:
            logger.info(f"  CLOUDY ({cf:.2%} cloud): {fp.name} — skipping")
    logger.info(f"{len(clear_granules)}/{len(nc_files)} MODIS granules passed cloud check")

    # ── 7. Download/export coincident Landsat scenes ─────────────────────
    dl_method = "direct (getDownloadURL)" if args.direct else "Drive export"
    logger.info(f"Fetching coincident Landsat scenes via {dl_method}…")
    ls_dir = base_dir / cfg["paths"]["proc_landsat"]
    n_queued = 0
    n_skipped = 0
    for granule, ls_date in pairs:
        for platform, collection in ls_collections.items():
            scene_id = f"{ls_date.replace('-', '')}_{platform.split('_')[1]}"
            plat_short = platform.split('_')[1]
            out_path = ls_dir / f"{scene_id}_BT_{plat_short}.tif"
            if not args.force and out_path.exists():
                n_skipped += 1
                continue
            scene = (
                collection
                .filterDate(
                    ls_date,
                    (datetime.strptime(ls_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                .first()
            )
            if scene:
                if args.direct:
                    download_landsat_bt_direct(
                        scene, platform, scene_id, ls_dir, cfg["region"],
                        cfg=cfg, mask_clouds=not args.no_mask,
                        logger=logger, force=args.force)
                else:
                    export_landsat_bt(scene, platform, scene_id, ls_dir, cfg["region"],
                                     cfg=cfg, mask_clouds=not args.no_mask, logger=logger)
                n_queued += 1
    if n_skipped:
        logger.info(f"  Skipped {n_skipped} scenes (BT file already exists locally)")

    logger.info("Acquisition step complete.")
    logger.info("Next step: run src/02_pixel_matching.py")


if __name__ == "__main__":
    main()
