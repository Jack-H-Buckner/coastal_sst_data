"""
03_apply_calibration.py
=======================
Stage 3 of the SST calibration pipeline.

Applies the generalised calibration equation to all available Landsat
brightness temperature images to produce 100m resolution SST.

This step can be applied to:
  - Calibration period images (2000-2021, MODIS era)
  - Pre-MODIS images (Landsat 5, pre-2000) using generalised constants
  - Post-MODIS images (Landsat 8/9, post-2021) using generalised constants

Note on fjords/estuaries:
  The generalised equation derived from open coastal pixels is applied
  uniformly across all water pixels. For inland fjords where MODIS
  calibration data was unavailable, this is the best available approach
  pending validation against in situ data (see 04_validation.py).

Usage:
    python src/03_apply_calibration.py
    python src/03_apply_calibration.py --calibration R0    # Use R0 dataset constants
    python src/03_apply_calibration.py --calibration all   # Use all-data constants
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    print("rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "apply_calibration.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_calibration(results_dir: Path, dataset: str = "R0") -> dict:
    """Load saved generalised calibration constants."""
    cal_file = results_dir / f"generalised_{dataset}_calibration.json"
    if not cal_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {cal_file}")
    with open(cal_file) as f:
        return json.load(f)


def load_landsat_bt(filepath: Path) -> tuple:
    """Load Landsat Bt GeoTIFF. Returns (bt_array, profile, aux_dict).

    For 4-band files (BT, ST_CDIST, ST_QA, NDWI), aux_dict contains:
        "st_cdist": cloud distance in km
        "st_qa":    temperature uncertainty in K
        "ndwi":     normalised difference water index
    For single-band files, aux_dict is empty.
    """
    with rasterio.open(filepath) as src:
        bt = src.read(1).astype(np.float32)
        if src.nodata is not None:
            bt[bt == src.nodata] = np.nan
        profile = src.profile.copy()
        aux = {}
        if src.count >= 4:
            aux["st_cdist"] = src.read(2).astype(np.float32)
            aux["st_qa"]    = src.read(3).astype(np.float32)
            aux["ndwi"]     = src.read(4).astype(np.float32)
            # Set nodata to NaN in aux bands
            for k in aux:
                if src.nodata is not None:
                    aux[k][aux[k] == src.nodata] = np.nan
    return bt, profile, aux


def filter_aux_bands(
    bt: np.ndarray,
    aux: dict,
    ndwi_min: float | None = None,
    cdist_min_km: float | None = None,
    st_qa_max_k: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply post-download filters using auxiliary bands.

    Returns (filtered_bt, stats_dict) where stats_dict records how many
    pixels each filter removed.
    """
    stats = {}
    bt_out = bt.copy()

    if ndwi_min is not None and "ndwi" in aux:
        mask = aux["ndwi"] < ndwi_min
        mask = mask & ~np.isnan(bt_out)  # only count currently-valid pixels
        n = int(mask.sum())
        bt_out[mask] = np.nan
        stats["ndwi"] = n

    if cdist_min_km is not None and "st_cdist" in aux:
        mask = aux["st_cdist"] < cdist_min_km
        mask = mask & ~np.isnan(bt_out)
        n = int(mask.sum())
        bt_out[mask] = np.nan
        stats["st_cdist"] = n

    if st_qa_max_k is not None and "st_qa" in aux:
        mask = aux["st_qa"] > st_qa_max_k
        mask = mask & ~np.isnan(bt_out)
        n = int(mask.sum())
        bt_out[mask] = np.nan
        stats["st_qa"] = n

    return bt_out, stats


def _load_precomputed_water_mask(profile: dict, bt_shape: tuple) -> np.ndarray | None:
    """
    Load the GEE-derived water mask (from 00_make_water_mask.py) if it exists.

    The mask is a single-band uint8 GeoTIFF: 1 = water, 0 = land/nodata.
    It is reprojected/resampled on-the-fly to match the BT raster.

    Returns boolean array (True = water) or None if the mask file is missing.
    """
    mask_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / "water_mask"
        / "water_mask_tas_east_coast.tif"
    )
    if not mask_path.exists():
        return None

    try:
        from rasterio.warp import reproject, Resampling

        h, w = bt_shape
        dst_transform = profile["transform"]
        dst_crs       = profile["crs"]

        with rasterio.open(mask_path) as src:
            water = np.zeros((h, w), dtype=np.uint8)
            reproject(
                source      = rasterio.band(src, 1),
                destination = water,
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = dst_transform,
                dst_crs       = dst_crs,
                resampling    = Resampling.nearest,
            )
        return water.astype(bool)

    except Exception as e:
        print(f"  [water mask] failed to load pre-computed mask ({e})")
        return None


def create_water_mask(bt: np.ndarray, profile: dict) -> np.ndarray:
    """
    Build a water mask for the BT raster.

    Priority order:
    1. Pre-computed GEE water mask (data/processed/water_mask/*.tif)
       Built from JRC Global Surface Water + Landsat NDWI composite at 30 m.
       Resolves Tasmania's complex coastline far better than polygon shapefiles.
       Run src/00_make_water_mask.py once to generate this file.

    2. Natural Earth 50m land polygons (cartopy fallback).
       Coarser (~500 m effective resolution at the shoreline).

    3. BT threshold > −10 °C (last resort — no spatial dependency).

    Returns boolean array: True = ocean pixel to calibrate, False = land/nodata.
    """
    not_nodata = ~np.isnan(bt)

    # ── Priority 1: pre-computed spectral water mask ──────────────────────────
    precomputed = _load_precomputed_water_mask(profile, bt.shape)
    if precomputed is not None:
        print("  [water mask] using pre-computed GEE water mask (JRC + NDWI)")
        return not_nodata & precomputed

    # ── Priority 2: Natural Earth 50m polygons (cartopy) ─────────────────────
    print("  [water mask] pre-computed mask not found — falling back to Natural Earth 50m polygons")
    print("               Run src/00_make_water_mask.py once for a better mask.")
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box, mapping
        from rasterio.features import geometry_mask
        import rasterio.transform as rt

        transform = profile.get("transform")
        h, w = bt.shape

        corners_r = [0, 0, h - 1, h - 1]
        corners_c = [0, w - 1, 0, w - 1]
        xs, ys = rt.xy(transform, corners_r, corners_c)
        lon_min, lon_max = min(xs) - 1, max(xs) + 1
        lat_min, lat_max = min(ys) - 1, max(ys) + 1
        study = box(lon_min, lat_min, lon_max, lat_max)

        shp   = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        geoms = [
            mapping(g)
            for g in shpreader.Reader(shp).geometries()
            if g.intersects(study)
        ]
        if not geoms:
            return not_nodata

        land = geometry_mask(geoms, out_shape=(h, w), transform=transform, invert=True)
        return not_nodata & ~land

    except Exception as e:
        # ── Priority 3: BT threshold ──────────────────────────────────────────
        print(f"  [water mask] cartopy unavailable ({e}) — falling back to BT threshold")
        return not_nodata & (bt > -10.0)


def filter_implausible_bt(bt: np.ndarray, method: str = "tukey") -> tuple[np.ndarray, int]:
    """
    Remove implausibly cold BT pixels that survived GEE-side cloud masking.

    These are typically cloud-edge, thin cirrus, or haze-contaminated pixels
    that appear as cold outliers in the scene's BT distribution.

    Methods:
        "tukey" : Standard Tukey fence — mask pixels below Q1 - 1.5 * IQR.
                  Adjusts to seasonal variation (cold winter scenes vs warm summer).

    Parameters:
        bt     : 2D BT array (°C), NaN where already masked.
        method : Outlier detection method (default "tukey").

    Returns:
        bt_filtered : Copy of bt with cold outliers set to NaN.
        n_removed   : Number of pixels removed.
    """
    bt_out = bt.copy()
    valid = bt_out[~np.isnan(bt_out)]
    if valid.size < 20:
        return bt_out, 0

    if method == "tukey":
        p25, p75 = np.percentile(valid, [25, 75])
        iqr = p75 - p25
        # Standard Tukey lower fence: Q1 - 1.5 * IQR
        lower_fence = p25 - 1.5 * iqr
        cold_mask = bt_out < lower_fence
        n_removed = int(cold_mask.sum())
        bt_out[cold_mask] = np.nan
    else:
        n_removed = 0

    return bt_out, n_removed


def apply_calibration_equation(
    bt: np.ndarray,
    slope: float,
    intercept: float,
    water_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply linear calibration: SST = slope * Bt + intercept

    Only applied to water pixels; land pixels are set to NaN.
    """
    sst = np.full_like(bt, np.nan)
    sst[water_mask] = slope * bt[water_mask] + intercept
    return sst


def load_climatology(clim_path: Path) -> pd.DataFrame | None:
    """Load DOY SST climatology CSV (columns: doy, sst_mean, sst_std)."""
    if not clim_path.exists():
        return None
    df = pd.read_csv(clim_path)
    return df.set_index("doy")


def filter_by_climatology(
    sst: np.ndarray,
    scene_date: date,
    climatology: "pd.DataFrame",
    n_sigma: float = 5.0,
) -> tuple[np.ndarray, int, float, float]:
    """Mask SST pixels outside ±n_sigma of the DOY climatological mean.

    Returns (filtered_sst, n_removed, lower_bound, upper_bound).
    """
    doy = scene_date.timetuple().tm_yday
    if doy not in climatology.index:
        # Leap-year DOY 366 fallback to 365
        doy = min(doy, climatology.index.max())

    clim_mean = climatology.loc[doy, "sst_mean"]
    clim_std  = climatology.loc[doy, "sst_std"]

    lower = clim_mean - n_sigma * clim_std
    upper = clim_mean + n_sigma * clim_std

    outlier = (sst < lower) | (sst > upper)
    # Only count valid (non-NaN) pixels being removed
    n_removed = int(np.nansum(outlier & ~np.isnan(sst)))

    sst_out = sst.copy()
    sst_out[outlier] = np.nan
    return sst_out, n_removed, lower, upper


def flag_fjord_pixels(
    sst: np.ndarray,
    profile: dict,
    fjord_bounds: list[dict],
) -> np.ndarray:
    """
    Build a flag band indicating pixels in fjord/estuary sub-regions where
    MODIS calibration data was unavailable (land adjacency effects).

    Uses vectorised numpy operations — builds a lon/lat grid once from the
    raster transform, then applies bounding-box tests for each sub-region.

    Parameters:
        fjord_bounds : List of dicts with {lon_min, lon_max, lat_min, lat_max, name}

    Returns:
        flag array (uint8): 0 = open coast, 1 = fjord/estuary (extrapolated calibration)
    """
    import rasterio.transform as rt

    flags = np.zeros(sst.shape, dtype=np.uint8)
    transform = profile.get("transform")
    if transform is None or not fjord_bounds:
        return flags

    # Build coordinate grids once — vectorised, no pixel loop
    rows, cols = np.mgrid[0:sst.shape[0], 0:sst.shape[1]]
    xs, ys = rt.xy(transform, rows.ravel(), cols.ravel())
    lons = np.asarray(xs).reshape(sst.shape)
    lats = np.asarray(ys).reshape(sst.shape)

    for fb in fjord_bounds:
        in_fjord = (
            (lons >= fb["lon_min"]) & (lons <= fb["lon_max"]) &
            (lats >= fb["lat_min"]) & (lats <= fb["lat_max"])
        )
        flags[in_fjord] = 1

    return flags


def save_sst_geotiff(
    sst: np.ndarray,
    flags: np.ndarray,
    profile: dict,
    output_path: Path,
):
    """
    Save calibrated SST as a 2-band GeoTIFF.
    Band 1: SST (°C, float32)
    Band 2: Quality flags (uint8: 0=open coast, 1=fjord/extrapolated)
    """
    out_profile = profile.copy()
    out_profile.update({
        "count":   2,
        "dtype":   "float32",
        "nodata":  np.nan,
        "compress": "lzw",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(sst.astype(np.float32), 1)
        dst.write(flags.astype(np.float32), 2)
        dst.update_tags(1, description="Calibrated SST (°C)")
        dst.update_tags(2, description="Flags: 0=open coast, 1=fjord/estuary (extrapolated)")

    return output_path


# ==============================================================
# MAIN
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Apply calibration to produce SST products")
    parser.add_argument("--calibration", choices=["R0", "all"], default="R0",
                        help="Which calibration dataset to use (default: R0)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    base_dir = Path(args.config).parent.parent

    logger = setup_logging(base_dir / cfg["paths"]["logs"])
    logger.info("=" * 60)
    logger.info("Tasmania SST Calibration - Apply Calibration")
    logger.info("=" * 60)

    # Load calibration constants
    results_dir = base_dir / cfg["paths"]["results"]
    try:
        cal = load_calibration(results_dir, args.calibration)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run 02_pixel_matching.py first.")
        return

    slope     = cal["generalised_slope"]
    intercept = cal["generalised_intercept"]
    logger.info(f"Using {args.calibration} calibration: {cal['equation']}")
    logger.info(f"  Derived from {cal['n_images']} images")

    # Reference constants from paper (Northern California) - for comparison
    ref_slope, ref_intercept = 0.00297, -105.88
    logger.info(f"Reference (Speiser & Largier 2024, N. California): "
                f"SST = {ref_slope} * Bt + {ref_intercept}")
    logger.warning("NOTE: Reference constants are from N. California and will NOT "
                   "transfer to Tasmania. Always use locally derived constants.")

    # Identify fjord/estuary sub-regions (from config) for flagging
    fjord_bounds = [
        {**v, "name": k}
        for k, v in cfg.get("sub_regions", {}).items()
        if not v.get("modis_suitable", True)
    ]
    if fjord_bounds:
        logger.info(f"Flagging {len(fjord_bounds)} fjord/estuary sub-regions as extrapolated")

    # Process all Landsat BT files
    landsat_dir = base_dir / cfg["paths"]["proc_landsat"]
    output_dir  = base_dir / cfg["paths"]["proc_modis"]  # reuse processed dir
    sst_output_dir = base_dir / "data" / "processed" / "sst_calibrated"
    sst_output_dir.mkdir(parents=True, exist_ok=True)

    bt_files = sorted(landsat_dir.glob("*.tif"))
    logger.info(f"\nProcessing {len(bt_files)} Landsat BT files...")

    # Read masking config (with defaults for backward compatibility)
    mask_cfg = cfg.get("masking", {})
    bt_outlier_method = mask_cfg.get("bt_outlier_method", "tukey")
    ndwi_post_min = mask_cfg.get("ndwi_postdownload_min")
    cdist_min_km  = mask_cfg.get("cdist_min_km")
    st_qa_max_k   = mask_cfg.get("st_qa_max_k")
    has_aux_filters = any(v is not None for v in [ndwi_post_min, cdist_min_km, st_qa_max_k])

    # Climatology-based outlier filter (replaces fixed SST floor)
    clim_file  = mask_cfg.get("climatology_file")
    clim_sigma = mask_cfg.get("climatology_sigma", 5.0)
    climatology = None
    if clim_file:
        clim_path = base_dir / clim_file
        climatology = load_climatology(clim_path)
        if climatology is not None:
            logger.info(f"Climatology filter: ±{clim_sigma}σ from DOY mean "
                        f"({clim_path.name}, {len(climatology)} DOYs)")
        else:
            logger.warning(f"Climatology file not found: {clim_path}")
            logger.warning("  Run: python src/build_sst_climatology.py")

    # Fallback to fixed SST floor if no climatology
    sst_floor = mask_cfg.get("sst_floor_celsius")
    if climatology is None and sst_floor is not None:
        logger.info(f"Using fixed SST floor: {sst_floor}°C (no climatology file)")

    logger.info(f"Post-download filters: BT outlier method={bt_outlier_method}")
    if has_aux_filters:
        parts = []
        if ndwi_post_min is not None: parts.append(f"NDWI>={ndwi_post_min}")
        if cdist_min_km  is not None: parts.append(f"CDIST>={cdist_min_km}km")
        if st_qa_max_k   is not None: parts.append(f"QA<={st_qa_max_k}K")
        logger.info(f"Aux-band filters: {', '.join(parts)}")

    # Track totals for summary
    total_aux_removed = {"ndwi": 0, "st_cdist": 0, "st_qa": 0}
    total_clim_removed = 0
    n_files_with_aux = 0

    for bt_path in bt_files:
        try:
            bt, profile, aux = load_landsat_bt(bt_path)

            # Aux-band post-download filters (4-band files only)
            aux_str = ""
            if aux and has_aux_filters:
                n_files_with_aux += 1
                bt, aux_stats = filter_aux_bands(
                    bt, aux,
                    ndwi_min=ndwi_post_min,
                    cdist_min_km=cdist_min_km,
                    st_qa_max_k=st_qa_max_k,
                )
                for k, v in aux_stats.items():
                    total_aux_removed[k] += v
                aux_parts = [f"{v} {k}" for k, v in aux_stats.items() if v > 0]
                if aux_parts:
                    aux_str = f", aux removed: {', '.join(aux_parts)}"

            # Post-download BT statistical filter — remove cold outliers
            bt, n_bt_removed = filter_implausible_bt(bt, method=bt_outlier_method)
            bt_filter_str = f", {n_bt_removed} BT outliers removed" if n_bt_removed > 0 else ""

            water_mask  = create_water_mask(bt, profile)
            sst         = apply_calibration_equation(bt, slope, intercept, water_mask)

            # Climatology-based outlier filter (or fixed SST floor fallback)
            clim_str = ""
            # Parse scene date from filename: YYYYMMDD_P_BT_P.tif
            scene_date_str = bt_path.stem.split("_")[0]
            scene_date = date(int(scene_date_str[:4]),
                              int(scene_date_str[4:6]),
                              int(scene_date_str[6:8]))

            if climatology is not None:
                sst, n_clim, clim_lo, clim_hi = filter_by_climatology(
                    sst, scene_date, climatology, n_sigma=clim_sigma
                )
                total_clim_removed += n_clim
                if n_clim > 0:
                    clim_str = f", {n_clim} clim outliers [{clim_lo:.1f}–{clim_hi:.1f}°C]"
            elif sst_floor is not None:
                n_cold = int(np.nansum(sst < sst_floor))
                if n_cold > 0:
                    sst[sst < sst_floor] = np.nan
                    clim_str = f", {n_cold} below SST floor"

            flags       = flag_fjord_pixels(sst, profile, fjord_bounds)

            out_name    = bt_path.stem.replace("_BT", "_SST") + ".tif"
            out_path    = sst_output_dir / out_name
            save_sst_geotiff(sst, flags, profile, out_path)

            valid_pct  = (~np.isnan(sst)).mean() * 100
            fjord_pct  = (flags == 1).sum() / flags.size * 100
            logger.info(f"  {out_name}: {valid_pct:.1f}% valid{bt_filter_str}{aux_str}{clim_str}, "
                        f"{fjord_pct:.1f}% fjord")

        except Exception as e:
            logger.warning(f"  Failed: {bt_path.name} — {e}")

    if has_aux_filters and n_files_with_aux > 0:
        logger.info(f"\nAux-band filter totals ({n_files_with_aux} files with aux bands):")
        for k, v in total_aux_removed.items():
            if v > 0:
                logger.info(f"  {k}: {v:,} pixels removed")
    if climatology is not None:
        logger.info(f"Climatology filter total: {total_clim_removed:,} pixels removed")

    logger.info(f"\nSST products saved to: {sst_output_dir}")
    logger.info("Next step: run src/04_validation.py")


if __name__ == "__main__":
    main()
