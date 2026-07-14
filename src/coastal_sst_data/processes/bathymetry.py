#!/usr/bin/env python3
"""
OCEANSR -- bathymetry static covariate (NOAA NCEI CUDEM, GMRT fallback).

Settings come from `sources.bathymetry`. `source: cudem` reads the NOAA NCEI
Continuously Updated DEM (1/9 arc-second ~3 m seamless topobathy) straight from
its /vsicurl VRT, aggregates the fine pixels within each 100 m grid cell to
depth statistics, and (where CUDEM has no coverage, e.g. SE Alaska) falls back
to `source: gmrt` (GMRT GridServer, ~100 m). Writes ONE static NetCDF per AOI:

    data/BATHYMETRY/aligned/<aoi_id>/<aoi_id>.nc

Variables (all m):
  elevation  : mean elevation (neg below sea level) -- used by the landmask
  depth      : mean water depth over the cell (= mean of max(-elev,0))
  depth_p25  : 25th-percentile depth within the cell (sub-grid variability)
  depth_p75  : 75th-percentile depth within the cell

For GMRT (no sub-grid) depth_p25 = depth_p75 = depth. CUDEM is referenced to
NAVD88, not MSL -- expect the 0 contour (and water_min_depth_m) to shift slightly.

Usage (from the project root):
    python src/acquire_bathymetry.py --config configs/config.yaml --aoi hood_canal
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import warnings
from pathlib import Path

import numpy as np
import requests
import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from ..config import Project, DataProduct, load_config
from ..grid import AoiGrid, project_grids

log = logging.getLogger(__name__)
SOURCE = "bathymetry"
GMRT_URL = "https://www.gmrt.org/services/GridServer"


# Config is loaded/validated by coastal_sst_data.config; the per-AoI CRS + coarse
# grid come from coastal_sst_data.grid (AoiGrid). Only the sub-grid used for
# CUDEM depth statistics is derived here (fine_grid), from the shared grid.


def fine_grid(transform, width, height, k):
    """Sub-grid aligned to the coarse grid: same origin, resolution / k."""
    r = transform.a / k
    return from_origin(transform.c, transform.f, r, r), width * k, height * k


def block_stats(elev_fine, k, H, W):
    """(H*k, W*k) fine elevation -> per-coarse-cell (elev_mean, depth stats)."""
    ef = elev_fine.reshape(H, k, W, k)
    depth_fine = np.where(np.isnan(elev_fine), np.nan,
                          np.where(elev_fine < 0, -elev_fine, 0.0)).reshape(H, k, W, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN cells -> NaN
        elev_mean = np.nanmean(ef, axis=(1, 3))
        d_mean = np.nanmean(depth_fine, axis=(1, 3))
        d_p25 = np.nanpercentile(depth_fine, 25, axis=(1, 3))
        d_p75 = np.nanpercentile(depth_fine, 75, axis=(1, 3))
    return tuple(a.astype("float32") for a in (elev_mean, d_mean, d_p25, d_p75))


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def fetch_gmrt(bbox_ll, pad, layer, resolution, tmp_path: Path) -> Path:
    w, s, e, n = bbox_ll
    params = {"west": w - pad, "east": e + pad, "south": s - pad, "north": n + pad,
              "format": "geotiff", "layer": layer, "resolution": resolution}
    r = requests.get(GMRT_URL, params=params, timeout=180)
    r.raise_for_status()
    if r.content[:2] not in (b"II", b"MM"):
        raise RuntimeError(f"GMRT did not return a GeoTIFF (got {r.content[:80]!r})")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(r.content)
    return tmp_path


# CUDEM 1/9" tiles are 0.25-deg COGs named by their NW corner, e.g.
# ncei19_n47x75_w122x50_...tif -> lat 47.50-47.75, lon -122.50..-122.25.
CUDEM_URLLIST = ("https://coast.noaa.gov/htdata/raster2/elevation/"
                 "NCEI_ninth_Topobathy_2014_8483/urllist8483.txt")
_TILE_RE = re.compile(r"ncei19_n(\d+)x(\d+)_w(\d+)x(\d+)_", re.IGNORECASE)
CUDEM_NATIVE_M = 3.0


def _fetch_index(urllist, cache: Path):
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(requests.get(urllist, timeout=60).text)
    return [u.strip() for u in cache.read_text().splitlines() if u.strip().endswith(".tif")]


def _tile_bounds(name):
    m = _TILE_RE.search(name)
    if not m:
        return None
    top = int(m.group(1)) + int(m.group(2)) / 100.0
    left = -(int(m.group(3)) + int(m.group(4)) / 100.0)
    return (left, top - 0.25, left + 0.25, top)          # (W, S, E, N)


def _overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _choose_overview(target_m):
    """Map a target resolution to a COG overview level (None = full ~3 m)."""
    if target_m <= CUDEM_NATIVE_M * 1.5:
        return None
    return max(0, min(3, round(math.log2(target_m / CUDEM_NATIVE_M)) - 1))


def read_cudem(bbox_ll, target_crs, ftransform, Wf, Hf, geom_proj, urllist, cache, target_res_m):
    """Window-read ONLY the CUDEM tiles overlapping the AOI (COG overviews via
    /vsicurl + clip_box, so no full-mosaic allocation), merge, and reproject onto
    the fine sub-grid. Returns a (Hf, Wf) elevation array (NaN off-cover)."""
    from rioxarray.merge import merge_arrays
    urls = _fetch_index(urllist, Path(cache))
    sel = [u for u in urls if (tb := _tile_bounds(u)) and _overlaps(tb, bbox_ll)]
    if not sel:
        raise RuntimeError(f"no CUDEM tiles overlap bbox "
                           f"{tuple(round(b, 3) for b in bbox_ll)} ({len(urls)} in index)")
    ovr = _choose_overview(target_res_m)
    arrays = []
    for u in sel:
        da = None
        for lvl in dict.fromkeys([ovr, None]):           # requested overview, else full-res
            try:
                da = rioxarray.open_rasterio("/vsicurl/" + u, masked=True, overview_level=lvl)
                break
            except Exception:
                da = None
        if da is None:
            continue
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)
        tb = _tile_bounds(u)
        clip = (max(bbox_ll[0], tb[0]), max(bbox_ll[1], tb[1]),
                min(bbox_ll[2], tb[2]), min(bbox_ll[3], tb[3]))
        try:
            arrays.append(da.rio.clip_box(*clip))        # windowed read of the AOI portion
        except Exception:
            continue
    if not arrays:
        raise RuntimeError("all overlapping CUDEM tiles failed to read")
    mosaic = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
    fine = mosaic.rio.reproject(dst_crs=target_crs, shape=(Hf, Wf), transform=ftransform,
                                resampling=Resampling.nearest, nodata=np.nan)
    fine = fine.rio.clip([geom_proj], target_crs, drop=False)
    return fine.values.astype("float32")


def from_gmrt(bbox_ll, pad, layer, resolution, target_crs, transform, W, H, geom_proj, tmp):
    tif = fetch_gmrt(bbox_ll, pad, layer, resolution, tmp)
    da = rioxarray.open_rasterio(tif, masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    elev = da.rio.reproject(dst_crs=target_crs, shape=(H, W), transform=transform,
                            resampling=Resampling.bilinear, nodata=np.nan)
    elev = elev.rio.clip([geom_proj], target_crs, drop=False).values.astype("float32")
    tif.unlink(missing_ok=True)
    depth = np.where(np.isnan(elev), np.nan,
                     np.where(elev < 0, -elev, 0.0)).astype("float32")
    return elev, depth, depth.copy(), depth.copy()          # no sub-grid -> p25=p75=mean


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def write_output(ds, out_dir, aoi_id, fmt) -> Path:
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
# Source registry. Each fetcher: (g: AoiGrid, params) -> (elev, depth, p25, p75,
# used) on the shared grid, or None to signal "insufficient coverage" (-> the
# configured fallback source is tried). Extensible: gebco, copernicus_glo30, ...
# --------------------------------------------------------------------------- #
def _source_cudem(g: AoiGrid, params: dict):
    """NOAA NCEI CUDEM 1/9\" topobathy, aggregated to per-cell depth stats."""
    sub_m = params["stats_subgrid_m"]
    k = max(1, int(round(g.resolution_m / sub_m)))
    ftr, Wf, Hf = fine_grid(g.transform, g.width, g.height, k)
    elev_fine = read_cudem(g.search_bbox, g.target_crs, ftr, Wf, Hf, g.geom_proj,
                           params["cudem_urllist"], params["cudem_cache"], sub_m)
    cover = float(np.isfinite(elev_fine).mean())
    if cover < params["min_cudem_cover"]:
        log.info("  %s: CUDEM cover %.0f%% < %.0f%%", g.name, 100 * cover,
                 100 * params["min_cudem_cover"])
        return None                                          # -> fallback
    elev, depth, dp25, dp75 = block_stats(elev_fine, k, g.height, g.width)
    return elev, depth, dp25, dp75, f'NCEI CUDEM 1/9" ({cover:.0%} cover, {k}x{k} subgrid)'


def _source_gmrt(g: AoiGrid, params: dict):
    """GMRT GridServer (~100 m, global). No sub-grid -> p25 = p75 = mean depth."""
    elev, depth, dp25, dp75 = from_gmrt(
        g.search_bbox, params["pad_deg"], params["layer"], params["resolution"],
        g.target_crs, g.transform, g.width, g.height, g.geom_proj,
        params["tmp_dir"] / f"{g.name}.tif")
    return elev, depth, dp25, dp75, f'GMRT ({params["layer"]}, {params["resolution"]})'


SOURCES = {"cudem": _source_cudem, "gmrt": _source_gmrt}


def _fetch_with_fallback(source, g: AoiGrid, params, fallback):
    """Run the resolved source's fetcher; on None/error, try `fallback` if set."""
    try:
        res = SOURCES[source](g, params)
    except Exception as exc:
        log.warning("  %s: %s read failed (%s)", g.name, source, exc)
        res = None
    if res is None and fallback and fallback != source:
        log.info("  %s: falling back to %s", g.name, fallback)
        try:
            res = SOURCES[fallback](g, params)
        except Exception as exc:
            log.warning("  %s: %s fallback failed (%s)", g.name, fallback, exc)
            res = None
    return res


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _opt(opts, name, default):
    """Read an optional override off a product-options bag (extra='allow')."""
    return getattr(opts, name, default) if opts is not None else default


def _resolve_source(project: Project, aoi_name: str, default: str) -> str:
    """The DEM source for one AoI: its region's `dem_source`, else the default.

    This is the two-level lookup -- region-level override (region.sources.
    bathymetry.dem_source) on top of a project-level default_source.
    """
    opts = project.region_of(aoi_name).sources.get(DataProduct.bathymetry)
    src = getattr(opts, "dem_source", None) if opts is not None else None
    return src or default


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes.

    Only the source-agnostic global params live here; the per-AoI source is
    resolved separately (see acquire / _resolve_source).
    """
    opts = project.products.get(DataProduct.bathymetry)
    if opts is None:
        raise ValueError("bathymetry is not a selected product in this config")

    out_root = Path(project.output_dir) / "BATHYMETRY" / "aligned"
    cache = _opt(opts, "cudem_index_cache", None)
    params = {
        "pad_deg": float(_opt(opts, "pad_deg", 0.02)),
        "layer": _opt(opts, "layer", "topo"),
        "resolution": _opt(opts, "resolution", "max"),
        "stats_subgrid_m": float(_opt(opts, "stats_subgrid_m", 10.0)),
        "min_cudem_cover": float(_opt(opts, "min_cudem_cover", 0.5)),
        "cudem_urllist": _opt(opts, "cudem_urllist", CUDEM_URLLIST),
        "cudem_cache": Path(cache) if cache else out_root.parent / "urllist_cudem.txt",
        "tmp_dir": Path(project.output_dir) / "BATHYMETRY" / "_tmp",
    }
    return {
        "params": params,
        "out_dir": out_root,
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "default_source": _opt(opts, "default_source", "gmrt"),
        "fallback": _opt(opts, "fallback", "gmrt"),
    }


def run(eff, grids: dict[str, AoiGrid], aoi_sources: dict[str, str], only_aoi, dry_run):
    """Build one static bathymetry NetCDF per AoI, each from its resolved source."""
    params = eff["params"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    fallback = eff["fallback"]

    names = list(grids)
    if only_aoi:
        req = set(only_aoi)
        missing = req - set(names)
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        names = [n for n in names if n in req]

    for name in names:
        g = grids[name]
        source = aoi_sources[name]
        log.info("=== AOI: %s | grid=%dx%d @ %.0fm | source=%s ===",
                 name, g.width, g.height, g.resolution_m, source)

        out_path = out_root / name / f"{name}.nc"
        if not overwrite and out_path.exists():
            log.info("  already processed, skipping"); continue
        if dry_run:
            log.info("  [dry-run] would build bathymetry (%s) for %s", source, name); continue

        res = _fetch_with_fallback(source, g, params, fallback)
        if res is None:
            log.warning("  skipping %s (no bathymetry from %s or fallback)", name, source); continue
        elev, depth, dp25, dp75, used = res

        xs = g.transform.c + (np.arange(g.width) + 0.5) * g.transform.a
        ys = g.transform.f - (np.arange(g.height) + 0.5) * g.transform.a
        ds = xr.Dataset(
            {"elevation": (("y", "x"), elev), "depth": (("y", "x"), depth),
             "depth_p25": (("y", "x"), dp25), "depth_p75": (("y", "x"), dp75)},
            coords={"y": ys, "x": xs})
        ds["elevation"].attrs.update(units="m", long_name="mean elevation (neg below sea level)")
        ds["depth"].attrs.update(units="m", long_name="mean water depth (0 on land)")
        ds["depth_p25"].attrs.update(units="m", long_name="25th-percentile depth in cell")
        ds["depth_p75"].attrs.update(units="m", long_name="75th-percentile depth in cell")
        ds = ds.rio.write_crs(g.target_crs)
        ds.attrs.update(aoi_id=name, source=used,
                        processing="aggregated to AOI grid (mean, p25, p75 depth per cell)")
        log.info("  wrote %s  [%s]", write_output(ds, out_root / name, name, fmt), used)
    log.info("Done.")


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire bathymetry for a validated Project. Entry point for pipeline.py.

    The DEM source is resolved PER AoI (region override -> project default), then
    validated against the SOURCES registry, so a typo'd source fails loudly.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    aoi_sources = {name: _resolve_source(project, name, eff["default_source"])
                   for name in grids}
    unknown = sorted((n, s) for n, s in aoi_sources.items() if s not in SOURCES)
    if unknown:
        bad = ", ".join(f"{n}:{s}" for n, s in unknown)
        raise ValueError(f"bathymetry dem_source not recognized ({bad}); "
                         f"choose from {sorted(SOURCES)}.")
    run(eff, grids, aoi_sources, aois, dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="coastal_sst_data bathymetry (per-region CUDEM/GMRT) acquisition.")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Process only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if the static file exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    project = load_config(args.config)
    acquire(project, aois=args.aoi, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
