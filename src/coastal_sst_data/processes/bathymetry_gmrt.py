#!/usr/bin/env python3
"""
OCEANSR -- GMRT DEM source for the bathymetry process (Global Multi-Resolution Topography).

A helper library for `processes.bathymetry`: the GMRT half of the DEM sources. It fetches a
GeoTIFF from the GMRT GridServer (~100 m, global) and reprojects it onto the shared AOI grid.
GMRT has no sub-grid detail, so the per-cell depth percentiles collapse to the mean.

GMRT is already ~sea-level referenced (MSL_APPROX), so its DEM->MSL datum offset is 0 with no
network call -- there is no VDatum coupling here. CUDEM (the NAVD88 source that needs VDatum)
lives in `processes.bathymetry_cudem`; the orchestrator (`processes.bathymetry`) fans out over
both, so a failure in one source no longer aborts the other.

`_source_gmrt(g, params) -> (elev, depth, p25, p75, used)` is the fetcher the orchestrator's
SOURCES registry binds to `"gmrt"`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import requests

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling

from ..grid import AoiGrid
from .. import net, store

log = logging.getLogger(__name__)

GMRT_URL = "https://www.gmrt.org/services/GridServer"


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


def _source_gmrt(g: AoiGrid, params: dict):
    """GMRT GridServer (~100 m, global). No sub-grid -> p25 = p75 = mean depth.

    Returns (elev, depth, p25, p75, used) on the shared grid.
    """
    elev, depth, dp25, dp75 = from_gmrt(
        g.search_bbox, params["pad_deg"], params["layer"], params["resolution"],
        g.target_crs, g.transform, g.width, g.height, g.geom_proj,
        params["tmp_dir"] / f"{g.name}.tif")
    return elev, depth, dp25, dp75, f'GMRT ({params["layer"]}, {params["resolution"]})'
