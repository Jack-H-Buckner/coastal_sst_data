"""Per-AoI target grid: computed ONCE, shared by every data product.

Every acquisition process must regrid onto the SAME grid for a given AoI, so the
products stack pixel-for-pixel. This module is the single source of truth for
that grid: given an ``AreaOfInterest`` and the project ``GridSpec``, it resolves
the target CRS and builds the affine transform + shape that every process reuses.

Relationship to config:
    GridSpec   -- the user's *request* (resolution, "auto" CRS, snap...).
    AoiGrid    -- the *realized* grid for one AoI (concrete CRS, transform, size).

Typical use (the pipeline computes grids once, then passes them to each process):
    grids = project_grids(project)          # {aoi_name: AoiGrid}
    ecostress.acquire(project, grids=grids)
    mur.acquire(project, grids=grids)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from .config import AreaOfInterest, GridSpec, Project

log = logging.getLogger(__name__)


def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """EPSG code of the local UTM zone containing a point (WGS84)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def resolve_target_crs(area: AreaOfInterest, grid: GridSpec) -> str:
    """CRS for this AoI: explicit from config, or auto local-UTM from its center."""
    tc = grid.target_crs.strip()
    if tc.lower() == "auto":
        # Use the AoI center directly (robust; no polygon-centroid edge cases).
        return utm_epsg_from_lonlat(area.center_lon, area.center_lat)
    return tc


@dataclass(frozen=True)
class AoiGrid:
    """The realized target grid for one AoI, shared across all products."""
    name: str
    target_crs: str                                  # e.g. "EPSG:32610"
    transform: Affine                                # rasterio affine (top-left origin)
    width: int
    height: int
    resolution_m: float
    search_bbox: tuple[float, float, float, float]   # (W, S, E, N) in EPSG:4326
    geom_proj: BaseGeometry                          # AoI polygon in target_crs (for clipping)

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) -- the array shape products are reprojected to."""
        return (self.height, self.width)


def compute_aoi_grid(area: AreaOfInterest, grid: GridSpec) -> AoiGrid:
    """Resolve the CRS and build the target grid for one AoI.

    The AoI bounding box (already buffered in the config) is projected into the
    target CRS; the grid spans its bounds, optionally snapped to a resolution
    multiple. Every product for this AoI reprojects onto this exact grid.
    """
    bb = area.bbox
    if bb.crosses_antimeridian:
        # Projecting a box that wraps 180 deg into a single UTM zone is ambiguous
        # (its corners fall in different zones). Fail loudly until it's handled.
        raise NotImplementedError(
            f"AoI {area.name!r} crosses the antimeridian; shared-grid support "
            "for that case is not implemented yet."
        )

    crs = resolve_target_crs(area, grid)
    geom_4326 = box(bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat)
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geom_proj = shp_transform(fwd, geom_4326)

    minx, miny, maxx, maxy = geom_proj.bounds
    r = float(grid.resolution_m)
    if grid.snap_origin:
        minx, miny = math.floor(minx / r) * r, math.floor(miny / r) * r
        maxx, maxy = math.ceil(maxx / r) * r, math.ceil(maxy / r) * r
    width = int(round((maxx - minx) / r))
    height = int(round((maxy - miny) / r))
    transform = from_origin(minx, maxy, r, r)

    log.debug("grid %s: crs=%s %dx%d @ %.1fm", area.name, crs, width, height, r)
    return AoiGrid(
        name=area.name, target_crs=crs, transform=transform,
        width=width, height=height, resolution_m=r,
        search_bbox=(bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat),
        geom_proj=geom_proj,
    )


def project_grids(project: Project) -> dict[str, AoiGrid]:
    """Compute the shared grid for every AoI in the project, keyed by AoI name."""
    return {a.name: compute_aoi_grid(a, project.grid) for a in project.all_areas}
