"""
02_pixel_matching.py
====================
Stage 2 of the SST calibration pipeline.

For each coincident MODIS/Landsat image pair:
  1. Reproject Landsat Bt to match MODIS grid
  2. For each valid MODIS SST pixel, compute the MEDIAN of all Landsat Bt
     pixels whose centroids fall within that MODIS footprint (~100 pixels)
  3. Pair median Landsat Bt with MODIS SST value
  4. Run iterative outlier removal until Pearson r stabilises
  5. Fit per-image OLS calibration (SST = m * Bt + b)
  6. Save matched pairs and per-image calibration results

Methodology: Speiser & Largier (2024), Remote Sensing 16(23), 4477
Section 2.3 - Pixel Matching and Calibration

Usage:
    python src/02_pixel_matching.py
    python src/02_pixel_matching.py --min-r 0.70    # R0 dataset (strict)
    python src/02_pixel_matching.py --min-r 0.00    # All-data dataset
"""

import argparse
import json
import logging
import pickle
import signal
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import rowcol
except ImportError:
    print("rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

try:
    import xarray as xr
except ImportError:
    print("xarray not installed. Run: pip install xarray netCDF4")
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
            logging.FileHandler(log_dir / "pixel_matching.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ==============================================================
# DATA LOADING
# ==============================================================

def load_modis_sst(filepath: Path, quality_min: int = 4) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load MODIS Terra L2 SST from GHRSST L2P NetCDF file.

    GHRSST L2P format stores all data at the root level (no groups).
    xarray automatically decodes int16 raw data to float32 in Kelvin using
    scale_factor/add_offset. SST is converted to Celsius on load.
    Lat/lon are 2D swath coordinate arrays (nj x ni) at root level.

    GHRSST quality_level convention (OPPOSITE of legacy MODIS convention):
        0 = no data
        1 = bad data
        2 = worst quality
        3 = low quality
        4 = acceptable quality   ← default minimum
        5 = best quality

    Parameters:
        quality_min : Minimum acceptable quality_level (inclusive). Pixels
                      with quality_level < quality_min are masked to NaN.
                      Default 4 retains only acceptable + best quality pixels.

    Returns:
        sst     : 2D array of SST values (°C), NaN where quality masked
        quality : 2D int array of quality flags (0–5)
        meta    : dict with lats (2D), lons (2D), shape
    """
    ds = xr.open_dataset(filepath, engine="netcdf4")  # no group — all at root

    # SST has shape (time=1, nj, ni); squeeze the time dimension
    sst_raw = ds["sea_surface_temperature"].values.squeeze().astype(np.float32)
    qual    = ds["quality_level"].values.squeeze().astype(np.int8)

    # xarray has already applied scale_factor/add_offset → values are in Kelvin
    # Convert to Celsius for consistency with Landsat brightness temperature
    sst = sst_raw - 273.15

    # Mask pixels below the quality threshold or with NaN fill values
    # quality_level < quality_min covers no-data (0), bad (1), and low-quality pixels
    mask = (qual < quality_min) | np.isnan(sst_raw)
    sst[mask] = np.nan

    # lat/lon are 2D swath coordinate arrays at root level in GHRSST format
    lats = ds["lat"].values  # (nj, ni)
    lons = ds["lon"].values  # (nj, ni)
    ds.close()

    meta = {"lats": lats, "lons": lons, "shape": sst.shape}
    return sst, qual, meta


def filter_implausible_bt(bt: np.ndarray, method: str = "tukey") -> tuple[np.ndarray, int]:
    """
    Remove implausibly cold BT pixels that survived GEE-side cloud masking.

    These are typically cloud-edge, thin cirrus, or haze-contaminated pixels
    that appear as cold outliers in the scene's BT distribution.

    Methods:
        "tukey" : Adaptive Tukey fence — mask pixels below p15 - 1.5 * IQR.
                  Adjusts to seasonal variation (cold winter scenes vs warm summer).

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
        p15 = np.percentile(valid, 15)
        lower_fence = p15 - 1.5 * iqr
        cold_mask = bt_out < lower_fence
        n_removed = int(cold_mask.sum())
        bt_out[cold_mask] = np.nan
    else:
        n_removed = 0

    return bt_out, n_removed


def load_landsat_bt(
    filepath: Path,
    bt_outlier_method: str = "tukey",
) -> tuple[np.ndarray, "rasterio.transform.Affine", str]:
    """
    Load Landsat brightness temperature GeoTIFF (already in °C from GEE export).

    Applies three quality filters:
    1. nodata mask  : pixels equal to the GeoTIFF nodata value are set to NaN.
    2. physical bounds : Landsat Collection 2 fill pixels (raw DN=0) are converted
       by the scale/offset formula to −124.15°C; other artefacts from Landsat 7
       SLC-off gaps or bad scan lines can produce extreme values. Pixels outside
       the physical range [−50, 60] °C are masked as NaN.
    3. BT outlier filter : Adaptive Tukey fence removes implausibly cold pixels
       that survived GEE-side cloud masking (cloud-edge / haze contamination).

    Returns:
        bt        : 2D array of brightness temperature (°C), NaN where masked
        transform : rasterio affine transform
        crs       : CRS string
    """
    with rasterio.open(filepath) as src:
        bt = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            bt[bt == nodata] = np.nan
        # Physical bounds filter: mask artefact values outside Earth surface range
        bt[(bt < -50.0) | (bt > 60.0)] = np.nan
        transform = src.transform
        crs = src.crs.to_string()

    # Adaptive BT outlier filter — catches residual cloud-contaminated cold pixels
    bt, n_removed = filter_implausible_bt(bt, method=bt_outlier_method)

    return bt, transform, crs


# ==============================================================
# COLD-TAIL FILTER
# ==============================================================

def filter_cold_tail(
    bt: np.ndarray,
    asymmetry_threshold: float = 2.0,
    trim_percentile: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """
    Detect a heavy left tail in the BT distribution and remove the coldest pixels.

    Cloud-edge pixels adjacent to QA-masked clouds can be anomalously cold while
    technically passing the QA_PIXEL filter.  These appear as a long left tail in
    the BT histogram, with the asymmetry check:

        left_span  = p50 − p5
        right_span = p95 − p50
        heavy_tail = (left_span / right_span) > asymmetry_threshold

    If a heavy tail is detected all pixels below the ``trim_percentile`` are set
    to NaN.

    Parameters
    ----------
    bt                  : 2-D array of brightness temperature (°C), NaN where masked
    asymmetry_threshold : Ratio left_span/right_span above which the tail is
                          considered heavy.  Default 2.0 (left side is twice as
                          wide as the right side).
    trim_percentile     : Bottom percentile to cut when a heavy tail is detected.
                          Default 5.0 (bottom 5 %).

    Returns
    -------
    bt_filtered   : Copy of *bt* with cold-tail pixels set to NaN (or unchanged
                    if no heavy tail was detected).
    tail_detected : True if a heavy left tail was found and trimming applied.
    """
    valid = bt[~np.isnan(bt)]
    if valid.size < 10:
        return bt.copy(), False

    p5, p50, p95 = np.percentile(valid, [5, 50, 95])
    left_span  = float(p50 - p5)
    right_span = float(p95 - p50)

    if right_span <= 0:
        return bt.copy(), False

    tail_detected = (left_span / right_span) > asymmetry_threshold

    bt_out = bt.copy()
    if tail_detected:
        cutoff = float(np.percentile(valid, trim_percentile))
        bt_out[bt_out < cutoff] = np.nan

    return bt_out, tail_detected


# ==============================================================
# REPROJECTION
# ==============================================================

def reproject_landsat_to_modis_grid(
    bt: np.ndarray,
    bt_transform: "rasterio.transform.Affine",
    bt_crs: str,
    modis_lats: np.ndarray,
    modis_lons: np.ndarray,
) -> tuple[np.ndarray, "rasterio.transform.Affine"]:
    """
    Reproject Landsat Bt array to WGS84 to match MODIS geolocation arrays.

    MODIS L2 data is in a swath projection (not a regular grid), so we
    reproject Landsat to WGS84 at ~0.01° (~1km) spacing to approximate
    the MODIS footprint geometry.
    """
    import rasterio.crs as rcrs
    from rasterio.transform import from_bounds

    # Define target grid from MODIS lat/lon extent
    lat_min, lat_max = float(np.nanmin(modis_lats)), float(np.nanmax(modis_lats))
    lon_min, lon_max = float(np.nanmin(modis_lons)), float(np.nanmax(modis_lons))

    # ~1km in degrees at Tasmania latitude (42°S): 0.009°
    res_deg = 0.009
    n_rows = int((lat_max - lat_min) / res_deg) + 1
    n_cols = int((lon_max - lon_min) / res_deg) + 1

    dst_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, n_cols, n_rows)
    dst_crs = "EPSG:4326"

    dst_bt = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=bt.shape[0], width=bt.shape[1],
            count=1, dtype="float32",
            crs=bt_crs, transform=bt_transform,
            nodata=np.nan,
        ) as src_ds:
            src_ds.write(np.where(np.isnan(bt), src_ds.nodata, bt), 1)

            reproject(
                source=rasterio.band(src_ds, 1),
                destination=dst_bt,
                src_transform=bt_transform,
                src_crs=rcrs.CRS.from_string(bt_crs),
                dst_transform=dst_transform,
                dst_crs=rcrs.CRS.from_string(dst_crs),
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )

    return dst_bt, dst_transform


# ==============================================================
# CORE PIXEL MATCHING (Speiser & Largier 2024, Section 2.3)
# ==============================================================

def aggregate_landsat_to_modis_footprints(
    bt_reprojected: np.ndarray,
    bt_transform: "rasterio.transform.Affine",
    modis_sst: np.ndarray,
    modis_lats: np.ndarray,
    modis_lons: np.ndarray,
    modis_pixel_size_deg: float = 0.009,
) -> pd.DataFrame:
    """
    For each valid MODIS SST pixel, find all Landsat Bt pixels within its
    approximate footprint and compute the MEDIAN.

    This is the core aggregation step from Speiser & Largier (2024):
    the median reduces noise from sub-pixel heterogeneity and partially
    accounts for the spatial mismatch between 100m Landsat and 1km MODIS.

    Parameters:
        modis_pixel_size_deg : Half-width of MODIS footprint in degrees (~0.009° = 1km)

    Returns:
        DataFrame with columns: [modis_sst, median_landsat_bt, modis_lat, modis_lon, n_pixels]
    """
    half_pixel = modis_pixel_size_deg / 2.0
    rows_list = []

    n_rows, n_cols = modis_sst.shape

    for i in range(n_rows):
        for j in range(n_cols):
            sst_val = modis_sst[i, j]
            if np.isnan(sst_val):
                continue

            lat_c = modis_lats[i, j]
            lon_c = modis_lons[i, j]

            # Define bounding box of MODIS pixel footprint
            lat_lo = lat_c - half_pixel
            lat_hi = lat_c + half_pixel
            lon_lo = lon_c - half_pixel
            lon_hi = lon_c + half_pixel

            # Convert to row/col indices in the reprojected Landsat array
            # rasterio rowcol uses (transform, xs, ys)
            row_hi, col_lo = rowcol(bt_transform, lon_lo, lat_hi)  # top-left
            row_lo, col_hi = rowcol(bt_transform, lon_hi, lat_lo)  # bottom-right

            # Clamp to array bounds
            r0 = max(0, min(row_hi, row_lo))
            r1 = min(bt_reprojected.shape[0], max(row_hi, row_lo) + 1)
            c0 = max(0, min(col_lo, col_hi))
            c1 = min(bt_reprojected.shape[1], max(col_lo, col_hi) + 1)

            if r1 <= r0 or c1 <= c0:
                continue

            # Extract Landsat Bt pixels within footprint
            bt_patch = bt_reprojected[r0:r1, c0:c1]
            bt_valid = bt_patch[~np.isnan(bt_patch)]

            if len(bt_valid) == 0:
                continue

            median_bt = float(np.median(bt_valid))
            rows_list.append({
                "modis_sst":        sst_val,
                "median_landsat_bt": median_bt,
                "modis_lat":        lat_c,
                "modis_lon":        lon_c,
                "n_landsat_pixels": len(bt_valid),
            })

    if not rows_list:
        return pd.DataFrame(columns=["modis_sst", "median_landsat_bt", "modis_lat", "modis_lon", "n_landsat_pixels"])

    return pd.DataFrame(rows_list)


# ==============================================================
# ITERATIVE OUTLIER REMOVAL (Speiser & Largier 2024, Section 2.3)
# ==============================================================

def iterative_outlier_removal(
    pairs: pd.DataFrame,
    std_threshold: float = 1.0,
    max_iterations: int = 20,
    logger: logging.Logger = None,
) -> tuple[pd.DataFrame, float, int]:
    """
    Iteratively remove outlier pairs until Pearson r stabilises.

    At each iteration:
      1. Fit OLS regression (Bt -> SST)
      2. Compute residuals
      3. Remove pairs where |residual| > std_threshold * std(residuals)
      4. Recompute r; stop if r has not improved or if < 10 pairs remain

    Returns:
        filtered_pairs : DataFrame with outliers removed
        final_r        : Final Pearson r
        n_iterations   : Number of iterations performed
    """
    df = pairs.copy()
    prev_r = 0.0
    n_iter = 0

    for i in range(max_iterations):
        if len(df) < 10:
            break

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df["median_landsat_bt"], df["modis_sst"]
        )
        predicted = slope * df["median_landsat_bt"] + intercept
        residuals = df["modis_sst"] - predicted
        std_res = residuals.std()

        # Remove outliers beyond threshold
        mask = np.abs(residuals) <= std_threshold * std_res
        n_removed = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        n_iter += 1

        r_abs = abs(r_value)
        if logger:
            logger.debug(f"  Iteration {i+1}: r={r_abs:.4f}, removed {n_removed} outliers, n={len(df)}")

        # Stop if r has stopped improving (stabilised or worsened)
        if r_abs <= prev_r or n_removed == 0:
            break
        prev_r = r_abs

    # Final r with cleaned dataset
    if len(df) >= 2:
        _, _, final_r, _, _ = stats.linregress(df["median_landsat_bt"], df["modis_sst"])
    else:
        final_r = 0.0

    return df, float(final_r), n_iter


def fit_per_image_calibration(pairs: pd.DataFrame) -> dict:
    """
    Fit OLS linear regression for a single image pair.

    Returns dict with slope, intercept, r, p, stderr, n.
    Equation: SST = slope * Bt + intercept
    """
    if len(pairs) < 2:
        return None

    slope, intercept, r, p, se = stats.linregress(
        pairs["median_landsat_bt"], pairs["modis_sst"]
    )
    return {
        "slope":     slope,
        "intercept": intercept,
        "r":         r,
        "r_squared": r ** 2,
        "p_value":   p,
        "std_err":   se,
        "n_pairs":   len(pairs),
    }


# ==============================================================
# MAIN PROCESSING LOOP
# ==============================================================

def process_image_pair(
    modis_path: Path,
    landsat_path: Path,
    cfg: dict,
    logger: logging.Logger,
) -> dict | None:
    """
    Process a single MODIS/Landsat coincident image pair through the full
    pixel matching and calibration pipeline.

    Returns per-image calibration results dict, or None if rejected.
    """
    pm_cfg = cfg["pixel_matching"]
    min_r   = pm_cfg["outlier_removal"]["min_r_threshold"]
    std_thr = pm_cfg["outlier_removal"]["std_threshold"]
    max_it  = pm_cfg["outlier_removal"]["max_iterations"]

    logger.info(f"Processing: {modis_path.name} + {landsat_path.name}")

    # --- Image-level metadata from filenames ---
    name8       = modis_path.name[:8]          # "20191021"
    time6       = modis_path.name[8:14]         # "000501"
    ls_parts    = landsat_path.stem.split("_")
    ls_platform = f"L{ls_parts[1]}" if len(ls_parts) > 1 else "Lx"
    acq_date    = f"{name8[:4]}-{name8[4:6]}-{name8[6:8]}"
    acq_month   = int(name8[4:6])
    acq_time    = f"{time6[:2]}:{time6[2:4]}"
    overpass    = "night" if "-N-" in modis_path.name else "day"

    # Load data
    try:
        modis_sst, modis_qual, modis_meta = load_modis_sst(modis_path)
        bt, bt_transform, bt_crs = load_landsat_bt(landsat_path)
    except Exception as e:
        logger.warning(f"  Failed to load data: {e}")
        return None

    # --- MODIS SST statistics within the study bounding box ---
    region   = cfg["region"]
    lats_m, lons_m = modis_meta["lats"], modis_meta["lons"]
    in_bbox  = (
        (lats_m >= region["lat_min"]) & (lats_m <= region["lat_max"]) &
        (lons_m >= region["lon_min"]) & (lons_m <= region["lon_max"])
    )
    sst_bbox      = modis_sst[in_bbox]
    valid_bbox    = sst_bbox[~np.isnan(sst_bbox)]
    n_bbox_total  = int(in_bbox.sum())
    n_modis_valid = int(valid_bbox.size)
    modis_cloud_pct = 100.0 * (1.0 - n_modis_valid / max(n_bbox_total, 1))
    modis_sst_mean = float(np.mean(valid_bbox))            if valid_bbox.size else float("nan")
    modis_sst_std  = float(np.std(valid_bbox))             if valid_bbox.size else float("nan")
    modis_sst_p2   = float(np.percentile(valid_bbox,  2))  if valid_bbox.size else float("nan")
    modis_sst_p98  = float(np.percentile(valid_bbox, 98))  if valid_bbox.size else float("nan")

    # Reproject Landsat to match MODIS geolocation
    bt_reproj, bt_reproj_transform = reproject_landsat_to_modis_grid(
        bt, bt_transform, bt_crs,
        modis_meta["lats"], modis_meta["lons"],
    )

    # Aggregate Landsat to MODIS footprints
    pairs = aggregate_landsat_to_modis_footprints(
        bt_reproj, bt_reproj_transform,
        modis_sst, modis_meta["lats"], modis_meta["lons"],
    )

    if len(pairs) < pm_cfg["min_valid_pairs"]:
        logger.info(f"  Rejected: only {len(pairs)} valid pairs (min={pm_cfg['min_valid_pairs']})")
        return None

    # Cold-tail filter on ocean-only matched pairs (land already excluded by MODIS mask)
    n_pairs_before_ct   = len(pairs)
    cold_tail_triggered = False
    n_cold_tail_removed = 0
    bt_vals = pairs["median_landsat_bt"].values.astype(np.float32)
    bt_filt, tail_detected = filter_cold_tail(bt_vals)
    if tail_detected:
        keep = ~np.isnan(bt_filt)
        n_cold_tail_removed = int((~keep).sum())
        cold_tail_triggered = True
        pairs = pairs[keep].reset_index(drop=True)
        logger.info(f"  Cold-tail filter: heavy left tail — removed {n_cold_tail_removed} bottom-5% pairs")

    logger.info(f"  {len(pairs)} matched pairs before outlier removal")

    # Initial r (used for R0 dataset classification)
    _, initial_r, _ = stats.linregress(pairs["median_landsat_bt"], pairs["modis_sst"])[:3], 0, 0
    _, _, initial_r, _, _ = stats.linregress(pairs["median_landsat_bt"], pairs["modis_sst"])

    # Iterative outlier removal
    filtered_pairs, final_r, n_iter = iterative_outlier_removal(
        pairs, std_thr, max_it, logger
    )
    logger.info(f"  After {n_iter} iterations: r={final_r:.4f}, n={len(filtered_pairs)}")

    # Apply r threshold — require positive correlation (negative r is physically wrong)
    if final_r < min_r:
        logger.info(f"  Rejected: final r={final_r:.4f} < threshold {min_r}")
        return None

    # Fit per-image calibration
    cal = fit_per_image_calibration(filtered_pairs)
    if cal is None:
        return None

    # Regression residual diagnostics
    pred     = cal["slope"] * filtered_pairs["median_landsat_bt"] + cal["intercept"]
    resids   = filtered_pairs["modis_sst"] - pred
    res_mean = float(resids.mean())
    res_std  = float(resids.std())

    # BT and SST statistics from the clean matched pairs
    bt_clean      = filtered_pairs["median_landsat_bt"]
    sst_clean     = filtered_pairs["modis_sst"]
    bt_mean       = float(bt_clean.mean())
    bt_p2         = float(bt_clean.quantile(0.02))
    bt_p98        = float(bt_clean.quantile(0.98))
    sst_bt_offset = float(sst_clean.mean() - bt_clean.mean())   # positive → SST warmer than BT
    pair_coverage = 100.0 * len(filtered_pairs) / max(n_modis_valid, 1)

    result = {
        # --- Identification ---
        "date":              acq_date,
        "month":             acq_month,
        "overpass_time_utc": acq_time,
        "overpass_type":     overpass,
        "landsat_platform":  ls_platform,
        "modis_file":        modis_path.name,
        "landsat_file":      landsat_path.name,
        # --- MODIS coverage and SST distribution (study bbox) ---
        "n_modis_valid_bbox":  n_modis_valid,
        "modis_cloud_pct":     round(modis_cloud_pct, 1),
        "modis_sst_mean":      round(modis_sst_mean, 3),
        "modis_sst_std":       round(modis_sst_std,  3),
        "modis_sst_p2":        round(modis_sst_p2,   3),
        "modis_sst_p98":       round(modis_sst_p98,  3),
        # --- Matched pair counts and filtering ---
        "n_pairs_raw":          n_pairs_before_ct,
        "cold_tail_triggered":  cold_tail_triggered,
        "n_cold_tail_removed":  n_cold_tail_removed,
        "n_pairs_clean":        len(filtered_pairs),
        "n_iterations":         n_iter,
        "pair_coverage_pct":    round(pair_coverage, 1),
        # --- BT distribution from clean matched pairs ---
        "bt_mean":       round(bt_mean,       3),
        "bt_p2":         round(bt_p2,         3),
        "bt_p98":        round(bt_p98,        3),
        "sst_bt_offset": round(sst_bt_offset, 3),  # SST − BT mean offset
        # --- Calibration regression ---
        "initial_r":   round(float(initial_r), 4),
        "final_r":     round(float(final_r),   4),
        "slope":       cal["slope"],
        "intercept":   cal["intercept"],
        "r_squared":   cal["r_squared"],
        "p_value":     cal["p_value"],
        "std_err":     cal["std_err"],
        # --- Residual diagnostics ---
        "residual_mean": round(res_mean, 4),
        "residual_std":  round(res_std,  4),
        # --- Dataset classification ---
        "r0_dataset":    initial_r >= min_r,
        # Internal: kept for downstream diagnostics, excluded from CSV
        "filtered_pairs": filtered_pairs,
    }

    logger.info(f"  Calibration: SST = {cal['slope']:.5f} * Bt + {cal['intercept']:.3f} "
                f"(r²={cal['r_squared']:.3f}, offset={sst_bt_offset:+.2f}°C)")
    return result


def derive_generalised_calibration(results: list[dict], use_r0_only: bool = True) -> dict:
    """
    Derive generalised calibration constants from per-image results.

    Following Speiser & Largier (2024): the generalised slope and intercept
    are the MEDIANS of all per-image values (from the R0 dataset).

    Parameters:
        use_r0_only : If True, use only images that pass the R0 (strict) criterion
    """
    if use_r0_only:
        subset = [r for r in results if r.get("r0_dataset", False)]
    else:
        subset = results

    if not subset:
        raise ValueError("No images available for generalised calibration.")

    slopes     = [r["slope"] for r in subset]
    intercepts = [r["intercept"] for r in subset]

    gen_slope     = float(np.median(slopes))
    gen_intercept = float(np.median(intercepts))

    return {
        "generalised_slope":     gen_slope,
        "generalised_intercept": gen_intercept,
        "n_images":              len(subset),
        "slope_std":             float(np.std(slopes)),
        "intercept_std":         float(np.std(intercepts)),
        "equation":              f"SST = {gen_slope:.5f} * Bt + {gen_intercept:.3f}",
        "dataset":               "R0" if use_r0_only else "all_data",
    }


# ==============================================================
# MAIN
# ==============================================================

def _process_pair_worker(args_tuple):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    modis_path, landsat_path, cfg_dict = args_tuple
    _logger = logging.getLogger(f"worker_{modis_path.stem[:8]}")
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        _logger.addHandler(logging.StreamHandler(sys.stdout))
    return process_image_pair(modis_path, landsat_path, cfg_dict, _logger)


# ==============================================================
# STREAMING MODIS (on-the-fly download via earthaccess)
# ==============================================================

def search_modis_for_landsat_dates(
    landsat_dates: list[str],
    cfg: dict,
    logger: logging.Logger,
) -> list[tuple]:
    """
    Search NASA earthaccess for MODIS granules coincident with Landsat dates.

    Returns list of (granule, landsat_date_str, modis_filename) tuples.
    """
    import earthaccess

    earthaccess.login(strategy="netrc")

    region = cfg["region"]
    bbox = (region["lon_min"], region["lat_min"], region["lon_max"], region["lat_max"])
    day_window = cfg["data_sources"]["modis"].get("day_window", 0)

    # Build expanded date set
    all_dates = set()
    for ds in landsat_dates:
        dt = datetime.strptime(ds, "%Y-%m-%d")
        for offset in range(-day_window, day_window + 1):
            all_dates.add((dt + timedelta(days=offset)).strftime("%Y-%m-%d"))

    # Search in yearly chunks to avoid huge single queries
    date_objs = sorted(datetime.strptime(d, "%Y-%m-%d") for d in all_dates)
    year_min, year_max = date_objs[0].year, date_objs[-1].year

    all_granules = []
    for year in range(year_min, year_max + 1):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        logger.info(f"  Searching MODIS granules for {year}...")
        try:
            results = earthaccess.search_data(
                short_name="MODIS_T-JPL-L2P-v2019.0",
                temporal=(start, end),
                bounding_box=bbox,
                count=-1,
            )
            all_granules.extend(results)
        except Exception as e:
            logger.warning(f"  earthaccess search failed for {year}: {e}")

    logger.info(f"  Total MODIS granules found: {len(all_granules)}")

    # Filter to coincident daytime granules — use set for O(1) date lookups
    landsat_date_set = set(landsat_dates)
    pairs = []
    for granule in all_granules:
        try:
            modis_time = datetime.strptime(
                granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
                "%Y-%m-%dT%H:%M:%S.%fZ",
            )
        except (KeyError, ValueError):
            continue

        # Daytime only (MODIS Terra ~10:30 local; at Tasmania UTC+10, this is ~00:30 UTC)
        # Night passes have "-N-" in filename or hour > 12 UTC
        try:
            native_id = granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
        except (KeyError, IndexError):
            native_id = ""
        if "-N-" in native_id:
            continue

        modis_date = modis_time.date()
        # Check if any Landsat date falls within ±day_window using expanded set lookup
        matched_ls_date = None
        for offset in range(-day_window, day_window + 1):
            candidate = (modis_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            if candidate in landsat_date_set:
                matched_ls_date = candidate
                break
        if matched_ls_date:
            modis_fname = modis_time.strftime("%Y%m%d%H%M%S") + "-JPL-L2P_GHRSST-SSTskin-MODIS_T-D-v02.0-fv01.0.nc"
            pairs.append((granule, matched_ls_date, modis_fname))

    logger.info(f"  Coincident MODIS/Landsat pairs: {len(pairs)}")
    return pairs


def stream_and_process_pair(
    granule,
    modis_fname: str,
    landsat_path: Path,
    cfg: dict,
    logger: logging.Logger,
    timeout_seconds: int = 300,
) -> dict | None:
    """
    Download a single MODIS granule to a temp directory, process it with
    the Landsat scene, then delete it immediately. Only ~23 MB on disk at a time.

    Parameters:
        timeout_seconds : Maximum time (seconds) for the entire download +
                          processing operation. If exceeded, the pair is skipped
                          with a warning. Default 300s (5 minutes).
    """
    import earthaccess

    logger.info(f"  Streaming MODIS: {modis_fname}")

    # Set up alarm-based timeout (Unix only; ignored on Windows)
    old_handler = None
    if hasattr(signal, "SIGALRM") and timeout_seconds > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download granule to temp dir (more robust than earthaccess.open())
            downloaded = earthaccess.download([granule], local_path=tmpdir)
            if not downloaded:
                logger.warning(f"  Failed to download MODIS granule")
                return None

            # Find the downloaded .nc file
            nc_files = list(Path(tmpdir).glob("*.nc"))
            if not nc_files:
                logger.warning(f"  No .nc file in download")
                return None

            # Rename to expected filename (process_image_pair parses date from name)
            modis_path = Path(tmpdir) / modis_fname
            nc_files[0].rename(modis_path)

            result = process_image_pair(modis_path, landsat_path, cfg, logger)
            # Temp dir and contents auto-deleted on context exit
            return result

    except _StreamTimeout:
        logger.warning(f"  Timed out after {timeout_seconds}s: {modis_fname}")
        return None
    except Exception as e:
        logger.warning(f"  Stream error for {modis_fname}: {e}")
        return None
    finally:
        # Cancel any pending alarm and restore previous handler
        if hasattr(signal, "SIGALRM") and timeout_seconds > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


class _StreamTimeout(Exception):
    """Raised when a MODIS streaming operation exceeds the time limit."""
    pass


def _timeout_handler(signum, frame):
    raise _StreamTimeout("MODIS streaming timed out")


def save_checkpoint(
    all_results: list[dict],
    results_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Save current results incrementally — CSV + generalised JSON files.

    Called periodically during the processing loop so that progress is
    not lost if the process hangs or crashes.
    """
    if not all_results:
        return

    # Per-image CSV
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "filtered_pairs"}
        for r in all_results
    ])
    csv_path = results_dir / "per_image_calibrations.csv"
    results_df.to_csv(csv_path, index=False)

    # Generalised calibration JSON files
    for use_r0, name in [(True, "generalised_R0"), (False, "generalised_all")]:
        try:
            gen = derive_generalised_calibration(all_results, use_r0_only=use_r0)
            out_path = results_dir / f"{name}_calibration.json"
            with open(out_path, "w") as f:
                json.dump({k: v for k, v in gen.items()}, f, indent=2)
        except ValueError:
            pass  # Not enough R0 images yet

    logger.info(f"  [checkpoint] Saved {len(all_results)} calibrations to {csv_path}")


def load_checkpoint(results_dir: Path, logger: logging.Logger) -> tuple[list[dict], set[str]]:
    """
    Load previously saved checkpoint CSV.

    Returns:
        all_results  : List of result dicts (without filtered_pairs).
        done_keys    : Set of 'modis_file|landsat_file' keys already processed,
                       used to skip completed work items on resume.
    """
    csv_path = results_dir / "per_image_calibrations.csv"
    if not csv_path.exists():
        return [], set()

    df = pd.read_csv(csv_path)
    logger.info(f"  [resume] Loaded {len(df)} calibrations from {csv_path}")

    all_results = df.to_dict("records")
    done_keys = {
        f"{r['modis_file']}|{r['landsat_file']}" for r in all_results
    }
    return all_results, done_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Pixel matching and per-image calibration")
    parser.add_argument("--min-r",  type=float, default=None, help="Override min Pearson r threshold")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = serial). "
                             "Use e.g. --workers 4 to process 4 image pairs concurrently.")
    parser.add_argument("--stream", action="store_true",
                        help="Stream MODIS granules on-the-fly from NASA instead of reading "
                             "local files. No MODIS storage required.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous run. Loads per_image_calibrations.csv "
                             "and skips already-processed pairs.")
    parser.add_argument("--stream-timeout", type=int, default=300,
                        help="Timeout in seconds for each MODIS streaming operation (default: 300)")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save checkpoint every N accepted calibrations (default: 50)")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    base_dir = Path(args.config).parent.parent

    logger = setup_logging(base_dir / cfg["paths"]["logs"])
    logger.info("=" * 60)
    logger.info("Tasmania SST Calibration - Pixel Matching")
    logger.info("=" * 60)

    if args.min_r is not None:
        cfg["pixel_matching"]["outlier_removal"]["min_r_threshold"] = args.min_r
        logger.info(f"Overriding min r threshold: {args.min_r}")

    landsat_dir = base_dir / cfg["paths"]["proc_landsat"]
    pairs_dir   = base_dir / cfg["paths"]["matched_pairs"]
    results_dir = base_dir / cfg["paths"]["results"]
    pairs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    landsat_files = sorted(landsat_dir.glob("*.tif"))
    logger.info(f"Landsat files found: {len(landsat_files)}")

    if not landsat_files:
        logger.warning("No Landsat files found. Run fetch_landsat_direct.py first.")
        return

    # ---- Resume from checkpoint if requested ----
    all_results = []
    done_keys = set()
    if args.resume:
        all_results, done_keys = load_checkpoint(results_dir, logger)
        if done_keys:
            logger.info(f"  [resume] Will skip {len(done_keys)} already-processed pairs")

    checkpoint_interval = args.checkpoint_interval
    last_checkpoint_count = len(all_results)

    if args.stream:
        # ── STREAMING MODE: fetch MODIS on-the-fly from NASA ──
        logger.info("Mode: STREAMING (MODIS fetched on-the-fly, no local storage)")

        # Extract unique Landsat dates
        landsat_dates = sorted({f.stem[:8] for f in landsat_files})
        landsat_dates_fmt = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in landsat_dates]
        logger.info(f"Unique Landsat dates: {len(landsat_dates_fmt)}")

        # Build lookup: date_str (YYYYMMDD) -> [landsat_path, ...]
        ls_by_date = {}
        for f in landsat_files:
            d8 = f.stem[:8]
            ls_by_date.setdefault(d8, []).append(f)

        # Search MODIS granules via earthaccess
        logger.info("Searching NASA earthaccess for coincident MODIS granules...")
        modis_pairs = search_modis_for_landsat_dates(landsat_dates_fmt, cfg, logger)

        n_total = len(modis_pairs)
        logger.info(f"Pairs to process: {n_total}")

        # Build flat list of (granule, modis_fname, ls_path) work items
        work_items = []
        for granule, ls_date, modis_fname in modis_pairs:
            ls_date_compact = ls_date.replace("-", "")
            ls_paths = ls_by_date.get(ls_date_compact, [])
            for ls_path in ls_paths:
                work_items.append((granule, modis_fname, ls_path))

        n_work = len(work_items)
        logger.info(f"Work items (MODIS x Landsat): {n_work}")

        n_workers = args.workers

        # Filter out already-done pairs when resuming
        if done_keys:
            before = len(work_items)
            work_items = [
                (g, mf, lp) for g, mf, lp in work_items
                if f"{mf}|{lp.name}" not in done_keys
            ]
            logger.info(f"  [resume] Skipped {before - len(work_items)} already-done pairs, "
                        f"{len(work_items)} remaining")
            n_work = len(work_items)

        timeout_s = args.stream_timeout

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            logger.info(f"Running streaming mode with {n_workers} parallel workers")
            logger.info(f"  Stream timeout: {timeout_s}s, checkpoint every {checkpoint_interval} calibrations")

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        stream_and_process_pair,
                        granule, modis_fname, ls_path, cfg, logger,
                        timeout_seconds=timeout_s,
                    ): (modis_fname, ls_path.name)
                    for granule, modis_fname, ls_path in work_items
                }
                for i, future in enumerate(as_completed(futures), 1):
                    mf, lf = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        logger.warning(f"  Stream worker error for {mf}: {e}")
                    if i % 25 == 0:
                        logger.info(f"  Progress: {i}/{n_work} streamed, "
                                    f"{len(all_results)} accepted so far")
                    # Incremental checkpoint
                    if len(all_results) - last_checkpoint_count >= checkpoint_interval:
                        save_checkpoint(all_results, results_dir, logger)
                        last_checkpoint_count = len(all_results)
        else:
            logger.info(f"  Stream timeout: {timeout_s}s, checkpoint every {checkpoint_interval} calibrations")
            for i, (granule, modis_fname, ls_path) in enumerate(work_items, 1):
                logger.info(f"\n── [{i}/{n_work}] {modis_fname} + {ls_path.name} ──")
                result = stream_and_process_pair(
                    granule, modis_fname, ls_path, cfg, logger,
                    timeout_seconds=timeout_s,
                )
                if result is not None:
                    all_results.append(result)
                if i % 25 == 0:
                    logger.info(f"  Progress: {i}/{n_work} streamed, "
                                f"{len(all_results)} accepted so far")
                # Incremental checkpoint
                if len(all_results) - last_checkpoint_count >= checkpoint_interval:
                    save_checkpoint(all_results, results_dir, logger)
                    last_checkpoint_count = len(all_results)

    else:
        # ── LOCAL MODE: read MODIS from disk ──
        modis_dir = base_dir / cfg["paths"]["raw_modis"]
        modis_files = sorted(modis_dir.glob("*.nc"))

        logger.info(f"MODIS files found:   {len(modis_files)}")

        if not modis_files:
            logger.warning("No MODIS files found. Run with --stream or download MODIS first.")
            return

        # Build list of (modis_path, landsat_path) pairs
        pair_list = []
        for modis_path in modis_files:
            date_str = modis_path.name[:8]
            matching_ls = [f for f in landsat_files if date_str in f.stem]
            for ls_path in matching_ls:
                pair_list.append((modis_path, ls_path))

        logger.info(f"Image pairs to process: {len(pair_list)}")

        # Filter out already-done pairs when resuming
        if done_keys:
            before = len(pair_list)
            pair_list = [
                (mp, lp) for mp, lp in pair_list
                if f"{mp.name}|{lp.name}" not in done_keys
            ]
            logger.info(f"  [resume] Skipped {before - len(pair_list)} already-done pairs, "
                        f"{len(pair_list)} remaining")

        n_workers = args.workers
        n_pairs = len(pair_list)

        if n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            logger.info(f"Running with {n_workers} parallel workers")
            logger.info(f"  Checkpoint every {checkpoint_interval} calibrations")

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_process_pair_worker, (mp, lp, cfg)): (mp, lp)
                    for mp, lp in pair_list
                }
                for i, future in enumerate(as_completed(futures), 1):
                    mp, lp = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        logger.warning(f"  Worker error for {mp.name}: {e}")
                    if i % 50 == 0:
                        logger.info(f"  Progress: {i}/{n_pairs} pairs processed, "
                                    f"{len(all_results)} accepted so far")
                    # Incremental checkpoint
                    if len(all_results) - last_checkpoint_count >= checkpoint_interval:
                        save_checkpoint(all_results, results_dir, logger)
                        last_checkpoint_count = len(all_results)
        else:
            logger.info(f"  Checkpoint every {checkpoint_interval} calibrations")
            for i, (modis_path, ls_path) in enumerate(pair_list, 1):
                logger.info(f"\n── [{i}/{n_pairs}] {modis_path.name} + {ls_path.name} ──")
                result = process_image_pair(modis_path, ls_path, cfg, logger)
                if result is not None:
                    all_results.append(result)
                # Incremental checkpoint
                if len(all_results) - last_checkpoint_count >= checkpoint_interval:
                    save_checkpoint(all_results, results_dir, logger)
                    last_checkpoint_count = len(all_results)

    logger.info(f"\nTotal accepted image pairs: {len(all_results)}")
    if not all_results:
        logger.warning("No valid pairs found.")
        return

    # ---- Derive generalised calibration ----
    try:
        gen_cal_r0 = derive_generalised_calibration(all_results, use_r0_only=True)
        logger.info(f"\nGeneralised calibration (R0 dataset, n={gen_cal_r0['n_images']}):")
        logger.info(f"  {gen_cal_r0['equation']}")
        logger.info(f"  Slope SD: {gen_cal_r0['slope_std']:.6f}, Intercept SD: {gen_cal_r0['intercept_std']:.4f}")
    except ValueError as e:
        logger.warning(f"R0 dataset calibration failed: {e}")
        gen_cal_r0 = None

    gen_cal_all = derive_generalised_calibration(all_results, use_r0_only=False)
    logger.info(f"\nGeneralised calibration (all data, n={gen_cal_all['n_images']}):")
    logger.info(f"  {gen_cal_all['equation']}")

    # ---- Save results ----
    # Per-image results as CSV
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "filtered_pairs"}
        for r in all_results
    ])
    csv_path = results_dir / "per_image_calibrations.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nPer-image results saved to: {csv_path}")

    # Generalised calibration as JSON
    for gen, name in [(gen_cal_r0, "generalised_R0"), (gen_cal_all, "generalised_all")]:
        if gen:
            out_path = results_dir / f"{name}_calibration.json"
            with open(out_path, "w") as f:
                json.dump({k: v for k, v in gen.items()}, f, indent=2)
            logger.info(f"Saved: {out_path}")

    logger.info("\nNext step: run src/03_apply_calibration.py")


if __name__ == "__main__":
    main()
