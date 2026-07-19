#!/usr/bin/env python3
"""
OCEANSR -- CUDEM DEM source for the bathymetry process (NOAA NCEI Continuously Updated DEM).

A helper library for `processes.bathymetry`: the CUDEM half of the DEM sources. It reads the
NCEI 1/9 arc-second (~3 m) seamless topobathy tiles straight from their /vsicurl COGs,
window-reads only the tiles overlapping the AOI, merges and reprojects them onto a fine
sub-grid, and aggregates that to per-coarse-cell depth statistics (mean, p25, p75).

CUDEM is referenced to NAVD88, so this is the source whose DEM->MSL offset is resolved from
NOAA VDatum (see `processes.datum`, keyed by `DEM_DATUM["cudem"] = "NAVD88"`). GMRT lives in
`processes.bathymetry_gmrt`; the orchestrator (`processes.bathymetry`) fans out over both and
resolves each source's datum inline, so a failure in one source no longer aborts the other.

`_source_cudem(g, params) -> (elev, depth, p25, p75, used) | None` is the fetcher the
orchestrator's SOURCES registry binds to `"cudem"`. It returns None for genuine no-coverage
(too few overlapping tiles / below the cover threshold) and raises `TileReadError` for a
transient read failure -- the two must never be conflated (see TileReadError below).
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

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from ..grid import AoiGrid
from .. import net, store

log = logging.getLogger(__name__)


# Only the sub-grid used for CUDEM depth statistics is derived here (fine_grid), from the
# shared coarse grid (coastal_sst_data.grid.AoiGrid).
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
    tiles, and the AoI was quietly served ~100 m GMRT instead of 3 m CUDEM -- for the life
    of the cache file, with nothing in the log to say why.

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


def _source_cudem(g: AoiGrid, params: dict):
    """NOAA NCEI CUDEM 1/9\" topobathy, aggregated to per-cell depth stats.

    Returns (elev, depth, p25, p75, used) on the shared grid, or None for insufficient
    coverage. Raises TileReadError for a transient read failure (retried next run).
    """
    sub_m = params["stats_subgrid_m"]
    k = max(1, int(round(g.resolution_m / sub_m)))
    ftr, Wf, Hf = fine_grid(g.transform, g.width, g.height, k)
    elev_fine = read_cudem(g.search_bbox, g.target_crs, ftr, Wf, Hf, g.geom_proj,
                           params["cudem_urllist"], params["cudem_cache"], sub_m)
    cover = float(np.isfinite(elev_fine).mean())
    if cover < params["min_cudem_cover"]:
        log.info("  %s: CUDEM cover %.0f%% < %.0f%%", g.name, 100 * cover,
                 100 * params["min_cudem_cover"])
        return None                                          # -> no channel from this source
    elev, depth, dp25, dp75 = block_stats(elev_fine, k, g.height, g.width)
    return elev, depth, dp25, dp75, f'NCEI CUDEM 1/9" ({cover:.0%} cover, {k}x{k} subgrid)'
