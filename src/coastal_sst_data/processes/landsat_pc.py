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

Auth: NO LOGIN, but not no credential. Planetary Computer signs asset URLs anonymously
(free) -- yet the SAS token it issues expires in ~30-60 min, which is a credential with a
TTL by any other name. So assets are signed PER SCENE, not once at search time (see
`sign_item`), and a signature that dies mid-scene is replaced through the same
`net.retry(refresh=...)` seam the credentialed backends use, with `auth.refresher("pc")`
evicting PC's token cache (see `read_scene`). AWS/GEE sources will need their own
credentials; see the config `landsat.source` selector.

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
from shapely.geometry import shape as shp_shape

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, read_cog_window, select_aois
from .. import auth, entry, naming, net, provenance, report, store

log = logging.getLogger(__name__)

# --- Landsat C2 L2 product constants (shared by all landsat_<source> modules) - #
SOURCE = "landsat"
COLLECTION = "landsat-c2-l2"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Level-2 scale/offset (USGS): ST digital number -> Kelvin; SR DN -> reflectance.
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
CDIST_SCALE = 0.01  # `cdist` (ST_CDIST) DN -> km
# Green surface reflectance above which a cell is too BRIGHT to be water, whatever its NDWI.
# See `scene_to_dataset` for the measurement this comes from. Overridable per project as
# `masking.brightness_max`; <= 0 disables the gate entirely.
BRIGHTNESS_MAX = 0.15
# Planetary Computer uses lowercase-hyphen platform names.
DEFAULT_PLATFORMS = ["landsat-8", "landsat-9"]

# The missions that carry a THERMAL band, and therefore an ST layer in `landsat-c2-l2`.
#
# Verified against the live catalogue: a TM (L4/L5) or ETM+ (L7) item serves `lwir`, `green`,
# `nir08`, `qa_pixel` and `cdist` -- every asset this module reads -- with the SAME `ST_SCALE` /
# `ST_OFFSET` / `SR_SCALE` / `SR_OFFSET` as OLI-TIRS. So the pre-2013 record needs no per-mission
# arithmetic; only the thermal asset is named differently (see `scene_to_dataset`).
#
# Landsat 1/2/3 and the MSS sensor are NOT here and never can be: MSS has no thermal band at all
# (its assets are green/red/nir08/nir09), and there is no Collection-2 Level-2 product for it --
# MSS lives in `landsat-c2-l1` only. Asking for one is not a thin-coverage request that might
# return a few scenes, it is a request for data that does not exist, which is why `_ds_cfg`
# refuses it outright. Left to run, every scene would raise `KeyError` on the missing asset inside
# the per-scene handler in `run`, and an all-MSS date range would look like a network outage --
# hundreds of failures and no output -- rather than an impossible ask.
THERMAL_PLATFORMS = frozenset({
    "landsat-4", "landsat-5", "landsat-7", "landsat-8", "landsat-9"})

# Platform name -> the small integer stamped on each granule as `platform_id`, which the
# assembler surfaces as the 1-D `lst_platform` channel.
#
# A forty-year Landsat record is THREE sensors -- TM, ETM+, OLI-TIRS -- with independent
# calibration histories, different cloud masks (no cirrus band before OLI) and, after
# 2003-05-31, Landsat 7's SLC-off gaps. They all merge into ONE `lst_*` channel set, which is
# the right call (unlike MODIS Terra/Aqua, every Landsat flies the same ~10:00 descending WRS-2
# orbit, so time-of-day is consistent and splitting per mission would only force a re-merge
# downstream) -- but merged and UNLABELLED means a step change at a mission boundary is
# indistinguishable from a real trend. The code travels with the data so it can be filtered on.
#
# 0 is reserved for "not recorded": a granule written before this existed, not a mission.
PLATFORM_CODES = {"landsat-4": 4, "landsat-5": 5, "landsat-7": 7,
                  "landsat-8": 8, "landsat-9": 9}

# How much of the date range one STAC search covers. A 1984-2026 AoI is thousands of scenes, and
# `search_scenes` materialises a whole window (paging as it goes) inside ONE `net.retry`: a
# failure on the last page discards every page before it, and nothing is downloaded until the
# entire multi-decade search has succeeded. Windowing bounds both -- a failure costs one year,
# and the first year's scenes start downloading while the rest is still being searched, so the
# `store.done` skip guard begins doing useful work on a resumed run almost immediately.
SEARCH_WINDOW_DAYS = 366

# Per-asset read retries. Deliberately fewer than net.MAX_RETRY: a Landsat scene is FIVE
# windowed COG reads and a multi-year AoI is hundreds of scenes, so the budget is spent per
# ASSET, five times over, on every scene. GDAL already retries 5xx/429 inside each attempt
# (GDAL_HTTP_MAX_RETRY), so net.retry here is the backstop for what GDAL does not retry -- a
# connection reset mid-stream, a curl/SSL error. Three attempts sleep 2s + 4s, bounding a
# scene at ~30s of backoff instead of the ~70s four would allow.
READ_ATTEMPTS = 3


def _pc_platform(name: str) -> str:
    """Normalize a platform name to Planetary Computer style ('LANDSAT_8' -> 'landsat-8')."""
    return name.strip().lower().replace("_", "-")


def resolve_platforms(raw) -> list[str]:
    """Normalized platform names, with the ones that have no thermal product REFUSED.

    Normalizing HERE rather than at search time is what makes the refusal reachable: `LANDSAT_5`
    and `landsat-5` have to become the same string before a membership test on
    `THERMAL_PLATFORMS` means anything, and the config is the last place the user's own spelling
    is still visible to quote back at them.

    Raises rather than dropping the bad name. A config asking for Landsat 1 wants a 1970s record,
    and quietly running the remaining platforms would deliver a 1984-onwards cube that looks like
    a successful answer to a question about 1972 -- the `platforms` list is the only place that
    intent is written down, so it is the only place the impossibility can be reported.
    """
    names = [_pc_platform(p) for p in raw]
    bad = [n for n in names if n not in THERMAL_PLATFORMS]
    if bad:
        raise ValueError(
            f"landsat.platforms: {', '.join(sorted(set(bad)))} has no thermal band in "
            f"{COLLECTION}, so no surface temperature can be derived from it. Landsat 1-3 and "
            "the MSS sensor carry no thermal band at all and have no Collection-2 Level-2 "
            "product (MSS is Level-1 only), so the Landsat SST record cannot start before 1982. "
            f"Choose from: {', '.join(sorted(THERMAL_PLATFORMS))}.")
    return names


def _search_windows(start: str, end: str, days: int = SEARCH_WINDOW_DAYS):
    """`[(start, end), ...]` covering `start`..`end` in <= `days` chunks, inclusive and disjoint.

    Dates, not datetimes: the STAC `datetime` interval is built as `f"{s}/{e}"` exactly as the
    unwindowed search did, so a single-window range produces the byte-identical query it always
    did and nothing changes for the 2013-onwards case.
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out, cur = [], s
    while cur <= e:
        stop = min(cur + pd.Timedelta(days=days - 1), e)
        out.append((cur.date().isoformat(), stop.date().isoformat()))
        cur = stop + pd.Timedelta(days=1)
    return out or [(s.date().isoformat(), e.date().isoformat())]


# --------------------------------------------------------------------------- #
# STAC discovery (Planetary Computer; lazy import so the module loads without
# the PC client installed -- only `acquire` needs it).
# --------------------------------------------------------------------------- #
def search_scenes(collection, stac_url, bbox, start, end, platforms, cloud_max):
    """Return UNSIGNED STAC items for Landsat scenes over the AoI bbox + dates.

    Unsigned on purpose. The catalogue used to be opened with
    `modifier=planetary_computer.sign_inplace`, which signs every item as it is materialised
    -- once, here, for a search that may span years. Signing now happens per scene
    (`sign_item`), and it HAS to happen there: `planetary_computer.sign_url` returns an href
    that already carries `st`/`se`/`sp` UNTOUCHED, so signing here would make every later
    re-sign a silent no-op handing back the same dead token.

    Searched one `SEARCH_WINDOW_DAYS` window at a time, each its own `net.retry`. The whole
    range used to be one call, which was fine while Landsat meant 2013-onwards; a pre-2013
    record makes it thousands of scenes and dozens of pages, and `list(search.items())` pages
    to exhaustion INSIDE the retry -- so a failure on the last page threw away every page
    before it and started the decade over. Windows are disjoint and cover the range exactly,
    so the concatenation needs no de-duplication.
    """
    from pystac_client import Client

    def _search(s, e):
        cat = Client.open(stac_url)
        search = cat.search(
            collections=[collection],
            bbox=list(bbox),
            datetime=f"{s}/{e}",
            query={"eo:cloud_cover": {"lt": cloud_max * 100.0},
                   "platform": {"in": [_pc_platform(p) for p in platforms]}},
        )
        return list(search.items())      # inside the retry: paging is where it fails

    windows = _search_windows(start, end)
    items = []
    for s, e in windows:
        # `s`/`e` pinned as default args (the CUDEM idiom): the lambda is called by net.retry
        # after the loop variable may have moved on.
        items.extend(net.retry(lambda s=s, e=e: _search(s, e),
                               what=f"Landsat STAC search {s}..{e}"))
    if len(windows) > 1:
        log.debug("  searched %s..%s in %d windows -> %d scene(s)",
                  start, end, len(windows), len(items))
    return items


def sign_item(item):
    """A COPY of `item` whose asset hrefs carry a FRESH Planetary Computer SAS token.

    PC signs anonymously, but the token it issues lives ~30-60 minutes. Signing at search
    time meant ONE token for a whole AoI: an hour into a multi-year date range every
    remaining href 403'd, each scene was caught, logged FAILED and skipped -- and because
    scenes are processed in DATE order, an expiring token looked exactly like a date cutoff.
    The AoI appeared to simply stop partway through, and the next AoI, with a fresh search and
    a fresh token, worked for another hour. So: sign per scene, as late as possible, after the
    skip guard has already decided we want this one.

    `copy=True` (the default) is load-bearing twice over. The item in the caller's list keeps
    its UNSIGNED href, so a second call really does mint a new token; and
    `planetary_computer.sign_url` returns an already-signed href UNCHANGED, so signing an
    already-signed item would quietly hand back the dead one.

    FRESHNESS IS NOT AUTOMATIC, though this once claimed it was. PC's cache re-fetches only
    when the cached token has under a minute left:

        if not token or token.ttl() < 60:

    so a token that just produced a 403 but still looks valid by that clock -- skew, an
    Azure-side revocation, a short grace window -- is handed straight back, and the "re-sign"
    silently returns the dead token. `read_scene` therefore passes `auth.refresher("pc")`,
    which evicts the cache first; `sign_item` on its own only guarantees a token, not a NEW one.

    Lazy import, so the module still loads without the PC client installed -- and so this
    stays the single seam the tests replace.
    """
    import planetary_computer

    return net.retry(lambda: planetary_computer.sign(item), what=f"Landsat sign {item.id}")


def reaches_aoi(item, aoi_lonlat) -> bool:
    """Does this scene's DATA footprint reach the AoI polygon?

    The search is by `search_bbox`, so the catalogue returns every scene whose BOUNDING BOX
    touches the AoI's. A Landsat scene is a rotated 185 km parallelogram inside a much larger
    axis-aligned bbox, so a whole WRS-2 path can be returned for an AoI its imagery never
    covers -- and nothing downstream notices: `rio.clip_box` succeeds on the intersection,
    `reproject` fills the rest with nodata, and the result is written as a COMPLETE granule
    (every `store.REQUIRED_VARS["LANDSAT"]` present) that the skip guard then treats as done.
    It becomes the day's base and the day reads as observed while holding nothing. Measured on
    one AoI: an entire path, 15 dates, 15 x 5 windowed COG reads, zero pixels delivered.

    The STAC item's `geometry` is a true footprint polygon -- BETTER than the bounding box
    ECOSTRESS has to settle for (`ecostress.granule_bbox`) -- so this is the tighter form of
    the same guard.

    A missing or unreadable geometry means KEEP, as it does for ECOSTRESS. Dropping a scene
    because its metadata was thin trades a known failure (a dead download, now reported) for a
    silent one (a real overpass that never appears).
    """
    geom = getattr(item, "geometry", None)
    if not geom:
        return True
    try:
        return bool(shp_shape(geom).intersects(aoi_lonlat))
    except Exception as exc:          # any malformed geometry -- keep, and say why
        log.debug("    %s: unreadable geometry (%s), keeping the scene", item.id, exc)
        return True


# --------------------------------------------------------------------------- #
# Per-scene: windowed COG reads -> shared grid -> sst/cloud/water/valid
# --------------------------------------------------------------------------- #
def _read_asset(item, key: str, g: AoiGrid, *, resampling, masked=True, nodata=None):
    """One Landsat COG asset, windowed onto the shared grid (see grid.read_cog_window).

    `masked=False` keeps raw integer values, which is what the bit-packed QA_PIXEL band
    needs. 300 m of pad is a few Landsat pixels of slack for the reprojection edges.

    Wrapped in `net.retry` -- this was the only windowed COG read in the tree without one. A
    scene is five independent range-read sequences, and one transient 503 on any of the five
    lost the WHOLE scene: the caller catches, logs FAILED, moves on, and the skip guard never
    revisits it. The href is pinned as a default argument (the CUDEM idiom) so every attempt
    reads the URL we signed for this pass.

    No `refresh` here, deliberately: re-reading the SAME signed href cannot be helped by a new
    token. `markers` still widens the vocabulary, which is what stops a dead-signature wording
    from being retried three times as a transient hiccup before it reaches `read_scene` -- the
    one level that can actually re-sign.
    """
    def _read(href=item.assets[key].href):
        return read_cog_window(href, g, resampling=resampling, masked=masked, pad_m=300.0,
                               nodata=nodata)

    return net.retry(_read, what=f"Landsat {item.id} {key}", attempts=READ_ATTEMPTS,
                     markers=net.SIGNED_URL_MARKERS)


def scene_to_dataset(item, g: AoiGrid, mask_cfg: dict, to_celsius: bool,
                     acq_time, aoi_id: str) -> Optional[xr.Dataset]:
    """Build the aligned sst/cloud/water/valid Dataset for one STAC item.

    sst: thermal DN -> Kelvin (or degC). water: NDWI from green/NIR SR. cloud:
    QA_PIXEL cloud/shadow/dilated bits, optionally buffered by ST_CDIST distance.
    """
    a = item.assets
    thermal_key = "lwir11" if "lwir11" in a else "lwir"   # L8/9 vs L4-7
    if thermal_key not in a:
        # An L2SR scene: surface reflectance was produced but surface temperature was not, so
        # the catalogue serves the item with NO thermal asset at all. Roughly 8% of the archive
        # globally, and the search cannot exclude them -- `landsat:correction` is not a field the
        # `platform`/`eo:cloud_cover` query filters on here.
        #
        # Returning None rather than raising, because this is not a failure: `run` counts a None
        # as "nothing to write" and moves on, where a KeyError would be caught one level up,
        # logged FAILED and counted against the report. A scene that never had a temperature
        # band is not a broken download, and a date range full of them should not read like an
        # outage. Said at debug level so a run that produces less than expected can still be
        # explained.
        log.debug("    %s: no thermal asset (L2SR -- surface reflectance only), skipping",
                  item.id)
        return None

    dn = _read_asset(item, thermal_key, g, resampling=Resampling.bilinear)
    kelvin = dn * ST_SCALE + ST_OFFSET
    sst = (kelvin - 273.15) if to_celsius else kelvin

    # Water via NDWI = (green - nir) / (green + nir), AND dark in the visible.
    green = _read_asset(item, "green", g, resampling=Resampling.bilinear) * SR_SCALE + SR_OFFSET
    nir = _read_asset(item, "nir08", g, resampling=Resampling.bilinear) * SR_SCALE + SR_OFFSET
    ndwi = (green - nir) / (green + nir)
    water = ndwi >= float(mask_cfg.get("ndwi_threshold", 0.0))

    # NDWI ALONE CALLS BRIGHT CLOUD "WATER", and the failure is not rare or theoretical.
    #
    # NDWI is a RATIO, so it is blind to absolute brightness: any surface whose green exceeds
    # its NIR passes, however luminous. Cloud usually has green < NIR and fails -- but a thick
    # cloud whose green channel SATURATES does not. The reflectance is then pinned at the top of
    # the valid SR range while NIR is merely high, the ratio goes positive, and a cloud top is
    # admitted as water. It is not caught downstream either: `valid` gates on the QA cloud bits,
    # and the scenes where this bites are exactly the ones CFMask rated low-confidence.
    #
    # Measured over 9.0M pixels -- three AoIs (Tillamook, Grays Harbor, Puget Sound) x three
    # missions (TM, ETM+, OLI-TIRS) -- restricted to the pixels `ndwi >= 0` currently admits:
    #
    #     QA says water (real)          n=1,419,279   green P50 0.017  P95 0.041  P99 0.075
    #     QA says cloud (FALSE water)   n=1,035,794   green P50 0.671  P90 1.602  P99 1.602
    #
    # Two populations ~40x apart in median, and 12.6% of everything NDWI admits has green > 1.0
    # -- NONPHYSICAL, since reflectance cannot exceed 1. Water is dark in the visible almost by
    # definition; that is the discriminator NDWI throws away, and this puts it back.
    #
    # The default keeps essentially all real water: at 0.15, 99.91% of QA-confirmed water
    # survives (99.13% at 0.08, 100% at 0.20) while the bright false-water population is cut by
    # ~79%. Chosen high rather than tight on purpose -- turbid, sediment-laden coastal water is
    # the whole point of this project and is brighter than open ocean, so the cost of a
    # too-strict cut (silently dropping real estuary observations, worst exactly where the
    # science is) far exceeds the cost of a too-loose one. Even so it sits 2x above the P99 of
    # every water pixel measured.
    #
    # What it CANNOT do: cloud SHADOW is dark, so brightness never catches it -- that is what the
    # QA shadow bit and the ST_CDIST buffer below are for. The two gates are complementary.
    bright_max = float(mask_cfg.get("brightness_max", BRIGHTNESS_MAX))
    if bright_max > 0:
        # `green <= bright_max` is False wherever green is NaN, which is the same answer the
        # NDWI test already gives there -- an unobserved cell claims nothing.
        water = water & (green <= bright_max)

    # Cloud via QA_PIXEL bits: dilated(1) | cloud(3) | shadow(4), + ST_CDIST buffer.
    # `nodata=1` -- QA_PIXEL's OWN fill value, which the catalogue declares ("nodata": 1) and
    # which is bit 0 set, "fill". `read_cog_window` defaults an unmasked read to 0 because NaN
    # is not representable in an integer band, and 0 is the WORST possible choice for this one:
    # it is a perfectly valid QA word meaning "no flag set", i.e. CLEAR. A Landsat scene is a
    # rotated parallelogram, so an AoI straddling a scene edge is largely OUTSIDE the source
    # raster -- and every one of those cells was reprojected in as 0 and read as clear sky.
    # Measured on a real Landsat 7 scene over Tillamook: 81.6% of the AoI had no temperature,
    # and 28 percentage points of that claimed to be cloud-free. Declaring the true fill value
    # puts outside-the-footprint cells and SLC-off gaps on the SAME footing -- both arrive with
    # bit 0 set, and the one test below handles both.
    qa = _read_asset(item, "qa_pixel", g, resampling=Resampling.nearest, masked=False,
                     nodata=1).astype("int64")
    cloudy = ((qa & (1 << 1)) != 0) | ((qa & (1 << 3)) != 0) | ((qa & (1 << 4)) != 0)
    buf_km = float(mask_cfg.get("cloud_buffer_km", 1.0))
    if buf_km > 0 and "cdist" in a:
        cdist_km = _read_asset(item, "cdist", g, resampling=Resampling.bilinear) * CDIST_SCALE
        cloudy = cloudy | (cdist_km < buf_km)
    # QA_PIXEL bit 0 is FILL -- no observation here. Not a cloud bit, and that is the trap: a
    # fill pixel has bit 0 and nothing else, so the three cloud bits above are all clear and the
    # cell reads `cloud = 0`, "clear sky", where the sensor recorded nothing at all.
    #
    # Which was harmless while Landsat meant 2013-onwards, because fill is then just the scene's
    # rotated corners. It stops being harmless with Landsat 7: the Scan Line Corrector failed
    # 2003-05-31, and every ETM+ scene since carries diagonal no-data wedges -- ~22% of the scene
    # on average, absent at nadir and widening toward the edges, so an AoI near a scene edge is
    # mostly gap. Measured on LE07_L2SP_047029_20040927_02_T1, a 1000x1000 window at scene
    # centre: 3.5% of pixels fill, EVERY one of them `QA_PIXEL == 1` exactly, 0.0% with any cloud
    # bit set, and 100% agreement with `lwir == 0`.
    #
    # `valid` was never wrong -- the `np.isfinite(sst)` gate below catches these cells, because
    # the thermal read is masked and its nodata is 0. But `cloud` is a published channel in its
    # own right, so it must not assert clear sky over a fifth of every post-2003 ETM+ scene. NaN
    # is already this module's "no observation" (it is what the masked reads produce) and is read
    # conservatively downstream -- `datacube._read_granule` does `np.nan_to_num(c, nan=1.0)`, so
    # an unknown cell counts as cloudy rather than as an observation.
    cloudy = cloudy.astype("float32").where((qa & 1) == 0)

    # The masks are DERIVED from windowed COG reads, so they arrive carrying the reads'
    # `_FillValue` (0, from the `masked=False` QA read) and an identity scale/offset. On a
    # 0/1 mask a fill of 0 is not "no data" -- it is CLEAR, and half the layer. Left on, every
    # clear cell decodes back as NaN, the assembler reads a NaN cloud cell as CLOUDY, and the
    # sensor's whole validity mask goes empty. `store` refuses such a fill value on write as
    # well; scrubbing here is saying what these layers ARE rather than relying on that net.
    ds = xr.Dataset({
        "sst": sst.astype("float32"),
        "cloud": store.clear_cf_decode_attrs(cloudy.astype("float32")),
        "water": store.clear_cf_decode_attrs(water.astype("float32")),
    })
    ds["sst"].attrs["units"] = "degC" if to_celsius else "K"
    ds["cloud"].attrs["long_name"] = "1=cloud/shadow, 0=clear, NaN=no observation (QA fill)"
    # Masks may be NaN outside the AoI; treat NaN as 0 for the logic mask.
    valid = (np.isfinite(ds["sst"])
             & (ds["water"].fillna(0) > 0)
             & ~(ds["cloud"].fillna(0) > 0))
    ds["valid"] = store.clear_cf_decode_attrs(valid.astype("uint8"))
    ds["valid"].attrs["long_name"] = "water & clear & finite SST"

    ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    # `platform` was already in the free-text `source` string, which is fine for a human reading
    # one file and useless to the assembler. `platform_id` is the machine-readable half: the
    # assembler lifts it into the 1-D `lst_platform` channel so a forty-year cube can say which
    # sensor produced each date. See PLATFORM_CODES; 0 means "not recorded", never a mission.
    plat = _pc_platform(str(item.properties.get("platform") or ""))
    instruments = item.properties.get("instruments") or []
    ds.attrs.update(
        aoi_id=aoi_id,
        platform=plat,
        platform_id=PLATFORM_CODES.get(plat, 0),
        instrument=",".join(instruments) if isinstance(instruments, (list, tuple))
                   else str(instruments),
        source=f"Landsat C2 L2 ST ({item.properties.get('platform')}) via Planetary Computer",
        processing="PC STAC + COG windowed read -> reprojected/clipped to AoI grid",
    )
    return ds


def read_scene(item, g: AoiGrid, mask_cfg: dict, to_celsius: bool, acq_time,
               aoi_id: str) -> Optional[xr.Dataset]:
    """`scene_to_dataset` on freshly signed assets, re-signed ONCE if the token expired.

    Signing per scene is necessary but not quite sufficient: a scene is five windowed reads,
    and a token with a minute left on it can die between the first asset and the last. That
    arrives as a 403 -- which `net.retry` correctly refuses to retry, because re-reading a
    dead URL is pointless. Re-reading a LIVE one is not, so the retry lives HERE, at the only
    level that can mint a new signature. Exactly once: if a second, freshly signed attempt
    fails too, the token was never the problem, and run()'s handler records a real failure.

    The wordings a dead SAS token arrives in used to live here as a private list; they are now
    `net.SIGNED_URL_MARKERS`, shared with every other per-URL credential. `attempts=1` because
    the transient budget belongs to `_read_asset` -- the auth retry is a separate budget and
    does not spend it.

    `refresh` has to be a REAL action even though PC is anonymous and `fn` re-signs itself:
    see `auth._refresh_pc`. `sign_item` alone can hand back the very token that just 403'd,
    because planetary_computer's cache only re-fetches below a minute of nominal TTL.
    """
    return net.retry(
        lambda: scene_to_dataset(sign_item(item), g, mask_cfg, to_celsius, acq_time, aoi_id),
        what=f"Landsat scene {item.id}", attempts=1,
        refresh=auth.refresher("pc"), markers=net.SIGNED_URL_MARKERS)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Acquire Landsat (via Planetary Computer) onto the pre-computed AoI grids."""
    grid_cfg = eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]
    to_celsius = grid_cfg.get("to_celsius", False)

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("landsat")

    for name in names:
        g = grids[name]
        # Resolved PER AoI: `collection`/`stac_url` are region-overridable (a region served
        # by a different catalogue names its own), while the scene-selection knobs
        # (platforms, cloud_cover_max, masking) stay project-global so every AoI's scenes
        # are chosen and masked the same way.
        ds_cfg = eff["ds"][name]
        mask_cfg = ds_cfg.get("masking", {})
        platforms = ds_cfg["platforms"]
        cloud_max = ds_cfg["cloud_cover_max"]

        log.info("=== AOI: %s (CRS=%s grid=%dx%d @ %.0fm) ===",
                 name, g.target_crs, g.width, g.height, g.resolution_m)
        # Said out loud, because these decide what `valid` (and every `_clean` channel built
        # from it) accepts, and a run whose masks look wrong is diagnosed from the log first.
        log.info("  masking: ndwi_threshold=%.2f brightness_max=%s cloud_buffer_km=%.2f "
                 "cloud_cover_max=%.2f",
                 float(mask_cfg.get("ndwi_threshold", 0.0)),
                 (f"{_bmax:.2f}" if (_bmax := float(mask_cfg.get("brightness_max",
                                                                 BRIGHTNESS_MAX))) > 0
                  else "off"),
                 float(mask_cfg.get("cloud_buffer_km", 1.0)), cloud_max)

        items = search_scenes(ds_cfg["collection"], ds_cfg["stac_url"], g.search_bbox,
                              start, end, platforms, cloud_max)
        log.info("  %d Landsat scene(s) (cloud < %.0f%%)", len(items), cloud_max * 100)
        if not items:
            continue
        if dry_run:
            log.info("  [dry-run] would process %d scene(s)", len(items))
            continue

        # Once per AoI, not once per scene: the projection back to lon/lat is the same answer
        # every time and a multi-year AoI is hundreds of scenes (as `ecostress.run` does).
        aoi_lonlat = g.geom_lonlat()
        off_aoi = 0
        # Scenes that DID reach the AoI by their footprint yet read back with no finite SST
        # over the grid. The acquisition-side twin of the assembler's "finite SST but NO valid
        # pixels" warning: without it a dead download is indistinguishable from a live one.
        empty_scenes = 0

        aoi_out = out_root / name
        for it in sorted(items, key=lambda i: i.properties["datetime"]):
            if not reaches_aoi(it, aoi_lonlat):
                # NOT `rep.skip()`: that counter means "already complete on disk", and these
                # scenes have nothing on disk and never will.
                log.debug("  [%s] %s does not reach the AoI polygon, skipping", name, it.id)
                off_aoi += 1
                continue
            acq = pd.Timestamp(it.datetime.astimezone(timezone.utc).replace(tzinfo=None))
            stem = naming.time_stem(name, acq)
            if store.done(aoi_out / f"{stem}.nc", store.REQUIRED_VARS["LANDSAT"],
                          shape=(g.height, g.width), overwrite=overwrite):
                log.info("  %s already processed, skipping", naming.time_stamp(acq))
                continue
            try:
                ds = read_scene(it, g, mask_cfg, to_celsius, acq, name)
            except Exception as exc:
                log.warning("    FAILED %s (%s)", it.id, exc)
                rep.fail(f"{name} {it.id}", exc)
                continue
            if ds is None:
                continue
            if not bool(np.isfinite(ds["sst"].values).any()):
                # Written anyway -- it IS what the catalogue served for this AoI, and a
                # complete granule is what stops the next run downloading it again. Counted,
                # so it stops looking like a successful acquisition.
                empty_scenes += 1
                log.debug("    %s: no finite SST over the AoI", it.id)
            ds.attrs.update(**provenance.stamp(eff))
            try:
                out = store.write_output(ds, aoi_out, stem, fmt)
            except (OSError, RuntimeError) as exc:       # this scene only -- see cmems.run
                log.warning("      WRITE FAILED %s (%s)", stem, exc)
                rep.fail(f"{name} {it.id}", f"write failed: {exc}")
                continue
            log.info("      wrote %s", out)
            rep.wrote(source="Landsat C2 L2 (Planetary Computer)")

        if off_aoi:
            log.info("  %s: %d of %d scene(s) touched the search box but not the AoI polygon "
                     "-- not downloaded", name, off_aoi, len(items))
        if empty_scenes:
            log.warning(
                "  %s: %d scene(s) read back with NO finite SST over the AoI. Their footprint "
                "polygon reaches the AoI but their imagery does not cover it; those days look "
                "observed in the cube and hold nothing.", name, empty_scenes)
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's Landsat settings, from its region-resolved options bag.

    `collection`/`stac_url` travel with `source` and are region-overridable; the
    scene-selection knobs are not, so every AoI's scenes are chosen and masked alike.
    """
    return {
        "collection": _opt(opts, "collection", COLLECTION),
        "stac_url": _opt(opts, "stac_url", STAC_URL),
        # Normalized AND checked here, at config time, so an impossible mission is reported once
        # against the line that asked for it rather than as one KeyError per scene.
        "platforms": resolve_platforms(_opt(opts, "platforms", DEFAULT_PLATFORMS)),
        "cloud_cover_max": float(_opt(opts, "cloud_cover_max", 0.7)),
        "masking": dict(_opt(opts, "masking", {}) or {}),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.landsat)
    if opts is None:
        raise ValueError("landsat is not a selected product in this config")

    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.landsat))
               for a in project.all_areas},
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
    # Credentials expire and runs are long: apply this project's refresh policy before any
    # network call. acquire() is the one entry point every invocation path goes through.
    auth.configure(project.auth)
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
