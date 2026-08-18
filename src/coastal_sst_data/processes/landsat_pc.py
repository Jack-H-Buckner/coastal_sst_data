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
# Planetary Computer uses lowercase-hyphen platform names.
DEFAULT_PLATFORMS = ["landsat-8", "landsat-9"]

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
    """
    from pystac_client import Client

    def _search():
        cat = Client.open(stac_url)
        search = cat.search(
            collections=[collection],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": cloud_max * 100.0},
                   "platform": {"in": [_pc_platform(p) for p in platforms]}},
        )
        return list(search.items())      # inside the retry: paging is where it fails

    return net.retry(_search, what="Landsat STAC search")


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


# --------------------------------------------------------------------------- #
# Per-scene: windowed COG reads -> shared grid -> sst/cloud/water/valid
# --------------------------------------------------------------------------- #
def _read_asset(item, key: str, g: AoiGrid, *, resampling, masked=True):
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
        return read_cog_window(href, g, resampling=resampling, masked=masked, pad_m=300.0)

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

    dn = _read_asset(item, thermal_key, g, resampling=Resampling.bilinear)
    kelvin = dn * ST_SCALE + ST_OFFSET
    sst = (kelvin - 273.15) if to_celsius else kelvin

    # Water via NDWI = (green - nir) / (green + nir).
    green = _read_asset(item, "green", g, resampling=Resampling.bilinear) * SR_SCALE + SR_OFFSET
    nir = _read_asset(item, "nir08", g, resampling=Resampling.bilinear) * SR_SCALE + SR_OFFSET
    ndwi = (green - nir) / (green + nir)
    water = ndwi >= float(mask_cfg.get("ndwi_threshold", 0.0))

    # Cloud via QA_PIXEL bits: dilated(1) | cloud(3) | shadow(4), + ST_CDIST buffer.
    qa = _read_asset(item, "qa_pixel", g, resampling=Resampling.nearest, masked=False).astype("int64")
    cloudy = ((qa & (1 << 1)) != 0) | ((qa & (1 << 3)) != 0) | ((qa & (1 << 4)) != 0)
    buf_km = float(mask_cfg.get("cloud_buffer_km", 1.0))
    if buf_km > 0 and "cdist" in a:
        cdist_km = _read_asset(item, "cdist", g, resampling=Resampling.bilinear) * CDIST_SCALE
        cloudy = cloudy | (cdist_km < buf_km)

    ds = xr.Dataset({
        "sst": sst.astype("float32"),
        "cloud": cloudy.astype("float32"),
        "water": water.astype("float32"),
    })
    ds["sst"].attrs["units"] = "degC" if to_celsius else "K"
    # Masks may be NaN outside the AoI; treat NaN as 0 for the logic mask.
    valid = (np.isfinite(ds["sst"])
             & (ds["water"].fillna(0) > 0)
             & ~(ds["cloud"].fillna(0) > 0))
    ds["valid"] = valid.astype("uint8")
    ds["valid"].attrs["long_name"] = "water & clear & finite SST"

    ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    ds.attrs.update(
        aoi_id=aoi_id,
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

        items = search_scenes(ds_cfg["collection"], ds_cfg["stac_url"], g.search_bbox,
                              start, end, platforms, cloud_max)
        log.info("  %d Landsat scene(s) (cloud < %.0f%%)", len(items), cloud_max * 100)
        if not items:
            continue
        if dry_run:
            log.info("  [dry-run] would process %d scene(s)", len(items))
            continue

        aoi_out = out_root / name
        for it in sorted(items, key=lambda i: i.properties["datetime"]):
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
            ds.attrs.update(**provenance.stamp(eff))
            try:
                out = store.write_output(ds, aoi_out, stem, fmt)
            except (OSError, RuntimeError) as exc:       # this scene only -- see cmems.run
                log.warning("      WRITE FAILED %s (%s)", stem, exc)
                rep.fail(f"{name} {it.id}", f"write failed: {exc}")
                continue
            log.info("      wrote %s", out)
            rep.wrote(source="Landsat C2 L2 (Planetary Computer)")
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
        "platforms": list(_opt(opts, "platforms", DEFAULT_PLATFORMS)),
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
