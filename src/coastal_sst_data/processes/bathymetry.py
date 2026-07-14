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

import logging
import math
import re
import time
import warnings
from pathlib import Path

import numpy as np
import requests
import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from ..config import Project, DataProduct, opt as _opt
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, net, provenance, report, store

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
    def _get():
        r = requests.get(GMRT_URL, params=params, timeout=180)
        r.raise_for_status()
        if r.content[:2] not in (b"II", b"MM"):     # an error page is not a GeoTIFF
            raise RuntimeError(f"GMRT did not return a GeoTIFF (got {r.content[:80]!r})")
        return r.content

    content = net.retry(_get, what=f"GMRT grid {bbox_ll}")
    # r.content is fully materialised before we write, and the magic bytes are checked
    # above, so this cannot leave a truncated GeoTIFF -- but write it atomically anyway,
    # because two AoIs sharing a tmp name must not tear each other's file.
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with store.atomic(tmp_path) as tmp:
        tmp.write_bytes(content)
    return tmp_path


# CUDEM 1/9" tiles are 0.25-deg COGs named by their NW corner, e.g.
# ncei19_n47x75_w122x50_...tif -> lat 47.50-47.75, lon -122.50..-122.25.
CUDEM_URLLIST = ("https://coast.noaa.gov/htdata/raster2/elevation/"
                 "NCEI_ninth_Topobathy_2014_8483/urllist8483.txt")
_TILE_RE = re.compile(r"ncei19_n(\d+)x(\d+)_w(\d+)x(\d+)_", re.IGNORECASE)
CUDEM_NATIVE_M = 3.0


INDEX_MAX_AGE_S = 30 * 86400        # NCEI adds tiles; a year-old index quietly misses them


class TileReadError(RuntimeError):
    """A CUDEM tile could not be READ -- distinct from CUDEM not COVERING the AoI.

    The difference decides whether falling back to GMRT is right. No coverage is a fact
    about the world and GMRT is the correct answer. A failed read is a fact about the
    network, and answering it with a permanent ~100 m DEM (which also flips the vertical
    datum from NAVD88 to MSL, and which no later run will ever re-attempt, because the
    output then exists and is complete) turns a transient blip into a permanent downgrade.
    """


def _tif_urls(text: str) -> list[str]:
    return [u.strip() for u in text.splitlines() if u.strip().endswith(".tif")]


def _download_index(urllist: str) -> str:
    """Fetch the CUDEM tile index and PROVE it is a tile index before anyone caches it."""
    def _get():
        r = requests.get(urllist, timeout=60)
        r.raise_for_status()            # was missing: a 500 HTML page has a .text too
        return r.text

    text = net.retry(_get, what="CUDEM tile index")
    if not _tif_urls(text):
        # An error page, a redirect notice, or an empty body. Every one of them is a
        # perfectly good string, and the old code cached it verbatim.
        raise RuntimeError(
            f"CUDEM index at {urllist} contained no .tif entries "
            f"(got {text[:120]!r}) -- refusing to cache it")
    return text


def _fetch_index(urllist, cache: Path) -> list[str]:
    """The CUDEM tile list, cached on disk.

    This cache was the nastiest failure in the module, because every way it broke was
    STICKY and every way it broke ended in a SILENT DOWNGRADE. There was no
    raise_for_status, so a 500 HTML error page was written to the cache as if it were data;
    no validation, so an empty body was cached too; no atomicity, so an interrupted write
    left a truncated list; and no expiry, so any of those survived forever. In each case
    `_tif_urls` then returned [] or a short list, `_source_cudem` found no overlapping
    tiles, and `_fetch_with_fallback` quietly served ~100 m GMRT instead of 3 m CUDEM --
    for the life of the cache file, with nothing in the log to say why.

    So now: validate BEFORE caching, write atomically, expire, and treat a cache that
    yields no tiles as poisoned rather than as an answer.
    """
    cache = Path(cache)
    if cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age > INDEX_MAX_AGE_S:
            log.info("  CUDEM index is %.0f days old; refreshing", age / 86400)
            cache.unlink()
        elif not _tif_urls(cache.read_text()):
            log.warning("  cached CUDEM index holds no .tif entries (an error page, or a "
                        "truncated write); discarding it and re-fetching")
            cache.unlink()

    if not cache.exists():
        store.write_text(cache, _download_index(urllist))   # atomic; validated above

    urls = _tif_urls(cache.read_text())
    if not urls:                        # cannot happen now -- but never downgrade silently
        raise RuntimeError(f"CUDEM index {cache} yielded no tiles")
    return urls


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
    arrays, unread = [], []
    for u in sel:
        def _read(url=u):
            for lvl in dict.fromkeys([ovr, None]):       # requested overview, else full-res
                try:
                    return rioxarray.open_rasterio("/vsicurl/" + url, masked=True,
                                                   overview_level=lvl)
                except Exception:
                    continue                             # this overview is absent; try full-res
            raise RuntimeError("no readable overview level")

        try:
            da = net.retry(_read, what=f"CUDEM tile {u.rsplit('/', 1)[-1]}")
            if "band" in da.dims:
                da = da.squeeze("band", drop=True)
            tb = _tile_bounds(u)
            clip = (max(bbox_ll[0], tb[0]), max(bbox_ll[1], tb[1]),
                    min(bbox_ll[2], tb[2]), min(bbox_ll[3], tb[3]))
            arrays.append(da.rio.clip_box(*clip))        # windowed read of the AOI portion
        except Exception as exc:
            unread.append((u.rsplit("/", 1)[-1], str(exc)))

    # A tile we could not READ is not a tile that does not EXIST, and conflating the two is
    # how a network blip used to become permanent. The dropped tiles left a hole in the
    # mosaic; `cover` was then computed from the SURVIVORS, so 6 of 10 tiles reading gave
    # cover=60% > the 50% threshold and the DEM was written -- correct-looking, labelled
    # "60% cover", with a network-shaped hole in it that propagated into landmask, depth and
    # every water-level channel. Refuse: an incomplete tile set is a failed read, not a
    # low-coverage area, and the caller must not paper over it with GMRT.
    if unread:
        raise TileReadError(
            f"{len(unread)} of {len(sel)} CUDEM tile(s) could not be read after retries "
            f"({', '.join(n for n, _ in unread[:3])}{'...' if len(unread) > 3 else ''}); "
            "refusing to build a DEM with a network-shaped hole in it")
    if not arrays:
        raise TileReadError("all overlapping CUDEM tiles failed to read")
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
    """Run the resolved source's fetcher; on NO-COVERAGE, try `fallback` if set.

    A NETWORK failure is deliberately NOT fallback-worthy. Falling back on one would swap a
    3 m NAVD88 DEM for a ~100 m MSL one because a CDN hiccuped -- and nothing would ever
    put it back, since the next run finds a complete bathymetry file and skips the AoI. So
    a TileReadError propagates: the AoI fails, it is named in the run report, and the next
    run retries CUDEM properly. Only genuine absence of coverage falls back.
    """
    try:
        res = SOURCES[source](g, params)
    except TileReadError:
        raise                                   # transient: fail loudly, retry next run
    except Exception as exc:
        log.warning("  %s: %s read failed (%s)", g.name, source, exc)
        res = None
    if res is None and fallback and fallback != source:
        log.warning("  %s: %s has no coverage here; FALLING BACK to %s "
                    "(coarser, and a different vertical datum)", g.name, source, fallback)
        try:
            res = SOURCES[fallback](g, params)
        except Exception as exc:
            log.warning("  %s: %s fallback failed (%s)", g.name, fallback, exc)
            res = None
    return res


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
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
        "config_sha256": project.config_sha256,
        "params": params,
        "out_dir": out_root,
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        # `source` is what the docs (and every user) call it; `default_source` is what the
        # code originally read. Both work, `source` wins. They diverged silently for a long
        # time: a config that said `source: cudem` got the `default_source` DEFAULT -- gmrt.
        "default_source": _opt(opts, "source", None) or _opt(opts, "default_source", "gmrt"),
        "fallback": _opt(opts, "fallback", "gmrt"),
    }


def run(eff, grids: dict[str, AoiGrid], aoi_sources: dict[str, str], only_aoi, dry_run):
    """Build one static bathymetry NetCDF per AoI, each from its resolved source."""
    params = eff["params"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    fallback = eff["fallback"]

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("bathymetry")

    for name in names:
        g = grids[name]
        source = aoi_sources[name]
        log.info("=== AOI: %s | grid=%dx%d @ %.0fm | source=%s ===",
                 name, g.width, g.height, g.resolution_m, source)

        out_path = out_root / name / f"{name}.nc"
        if store.done(out_path, store.REQUIRED_VARS["BATHYMETRY"],
                      shape=(g.height, g.width), overwrite=overwrite):
            log.info("  already processed, skipping"); rep.skip(); continue
        if dry_run:
            log.info("  [dry-run] would build bathymetry (%s) for %s", source, name); continue

        res = _fetch_with_fallback(source, g, params, fallback)
        if res is None:
            log.warning("  FAILED %s (no bathymetry from %s or fallback)", name, source)
            rep.fail(name, f"no bathymetry from {source} or fallback")
            continue
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
                        processing="aggregated to AOI grid (mean, p25, p75 depth per cell)",
                        **provenance.stamp(eff))
        # Static layer: the stem is the bare AoI name -- no time stamp to encode.
        log.info("  wrote %s  [%s]",
                 store.write_output(ds, out_root / name, name, fmt), used)
        # `used` is the DEM that actually won -- this is what stops "bathymetry ok" from
        # hiding a silent 3 m CUDEM -> ~100 m GMRT downgrade.
        rep.wrote(source=used)
    rep.log_summary()
    return rep


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
    net.setup_gdal_env()      # /vsicurl CUDEM tile reads: deadline + retries
    return run(eff, grids, aoi_sources, aois, dry_run)


def main():
    entry.process_main(
        acquire, "coastal_sst_data bathymetry (per-region CUDEM/GMRT) acquisition.")


if __name__ == "__main__":
    main()
