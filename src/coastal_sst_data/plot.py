"""Quick-look maps of a project's regions and AoIs.

Given a validated ``Project`` this draws two kinds of figure:

  * an **overview** map with every AoI in the project, coloured by region, so you
    can see the whole study area and how the AoIs group into regions; and
  * one **per-region** map, zoomed to that region, with its AoIs highlighted (and
    any neighbouring AoIs from other regions faded in for context).

Each AoI is drawn as its acquisition bounding box (the buffered box from the
config) plus a centre marker and label. Plotting is intentionally an *optional*
capability: matplotlib is imported lazily so the core package has no plotting
dependency, and coastlines are added only if ``cartopy`` happens to be installed
(otherwise the maps are plain lon/lat axes).

Wired into the CLI as ``coastal-sst-data grids --config config.yaml --plot``.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .config import Project

if TYPE_CHECKING:
    from .grid import AoiGrid

log = logging.getLogger(__name__)

# Bounding box tuple convention used throughout: (W, S, E, N) in EPSG:4326.
BBox = tuple[float, float, float, float]


def _aoi_bbox(area) -> Optional[BBox]:
    """(W, S, E, N) for an AoI, or None if it crosses the antimeridian.

    An antimeridian-crossing box (west > east) can't be drawn as a single lon/lat
    rectangle, so we skip it here -- the same case the grid code refuses to build.
    """
    bb = area.bbox
    if bb.crosses_antimeridian:
        log.warning("AoI %r crosses the antimeridian; skipping in map", area.name)
        return None
    return (bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat)


def _pad_extent(bboxes: list[BBox], pad_frac: float = 0.15,
                min_pad_deg: float = 0.05) -> BBox:
    """Union of ``bboxes`` grown by a margin, as (W, S, E, N) for axis limits."""
    w = min(b[0] for b in bboxes)
    s = min(b[1] for b in bboxes)
    e = max(b[2] for b in bboxes)
    n = max(b[3] for b in bboxes)
    dx = max((e - w) * pad_frac, min_pad_deg)
    dy = max((n - s) * pad_frac, min_pad_deg)
    return (w - dx, s - dy, e + dx, n + dy)


def _slug(name: str) -> str:
    """Filesystem-safe version of a region/project name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()) or "unnamed"


def _make_axes(fig, extent: BBox):
    """An axes spanning ``extent``; a cartopy GeoAxes with coastlines if available.

    Returns ``(ax, transform)`` where ``transform`` is the PlateCarree transform to
    pass to plotting calls (cartopy) or ``None`` (plain axes -> data coords).
    """
    w, s, e, n = extent
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        # Correct the lon/lat aspect for latitude so shapes aren't stretched.
        mean_lat = (s + n) / 2.0
        ax.set_aspect(1.0 / max(math.cos(math.radians(mean_lat)), 1e-6))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        return ax, None

    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([w, e, s, n], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#eeeee8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#dbeaf5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    return ax, proj


def _draw_aoi(ax, area, color, transform, *, highlight=True, label=True):
    """Draw one AoI: its bbox rectangle, a centre marker, and (optionally) a label."""
    from matplotlib.patches import Rectangle

    bb = _aoi_bbox(area)
    if bb is None:
        return
    w, s, e, n = bb
    tkw = {"transform": transform} if transform is not None else {}

    face_alpha = 0.25 if highlight else 0.06
    lw = 1.8 if highlight else 0.8
    ax.add_patch(Rectangle((w, s), e - w, n - s, facecolor=color, edgecolor="none",
                           alpha=face_alpha, zorder=2, **tkw))
    ax.add_patch(Rectangle((w, s), e - w, n - s, facecolor="none", edgecolor=color,
                           linewidth=lw, zorder=3, **tkw))
    ax.plot([area.center_lon], [area.center_lat], marker="o", markersize=4 if highlight else 3,
            color=color, zorder=4, **tkw)
    if label:
        ax.text(area.center_lon, area.center_lat, "  " + area.name, fontsize=7,
                va="center", ha="left", color="black" if highlight else "0.4",
                zorder=5, **tkw)


def _region_colors(project: Project):
    """Map each region name to a stable colour from a qualitative colormap."""
    import matplotlib.pyplot as plt

    names = [r.name for r in project.regions]
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(names)}


def plot_project_aois(project: Project, *, grids: "dict[str, AoiGrid] | None" = None,
                      out_dir: "str | Path | None" = None, overview: bool = True,
                      per_region: bool = True, show: bool = False,
                      dpi: int = 150) -> list[Path]:
    """Save an overview map of all AoIs plus one map per region.

    Parameters
    ----------
    project     validated config to visualize.
    grids       optional {aoi_name: AoiGrid}; when given, AoIs missing from it
                (e.g. ones whose grid failed to compute) are omitted from the maps.
    out_dir     directory for the PNGs (default ``<output_dir>/figures``).
    overview    write the all-regions overview map.
    per_region  write one zoomed map per region.
    show        also display the figures interactively (plt.show()).
    dpi         raster resolution of the saved PNGs.

    Returns the list of written file paths.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")           # headless: save without a display
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir) if out_dir is not None else Path(project.output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = set(grids) if grids is not None else None
    colors = _region_colors(project)
    # Region -> [drawable areas], and a flat (area, region_name) list for context.
    region_areas: dict[str, list] = {}
    all_drawable: list = []
    for r in project.regions:
        areas = [a for a in r.areas
                 if (keep is None or a.name in keep) and _aoi_bbox(a) is not None]
        region_areas[r.name] = areas
        all_drawable.extend(areas)

    if not all_drawable:
        log.warning("No drawable AoIs (nothing gridded / all antimeridian); no maps written.")
        return []

    written: list[Path] = []

    # --- Overview: every AoI, coloured by region ---------------------------- #
    if overview:
        all_bboxes = [_aoi_bbox(a) for a in all_drawable]
        fig = plt.figure(figsize=(9, 8))
        ax, transform = _make_axes(fig, _pad_extent(all_bboxes))
        for rname, areas in region_areas.items():
            for a in areas:
                _draw_aoi(ax, a, colors[rname], transform, highlight=True, label=True)
        handles = [Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=colors[rn],
                          markeredgecolor=colors[rn], markersize=9,
                          label=f"{rn} ({len(region_areas[rn])})")
                   for rn in region_areas if region_areas[rn]]
        if handles:
            ax.legend(handles=handles, title="regions", loc="best", fontsize=8, framealpha=0.9)
        ax.set_title(f"{project.name}: {len(all_drawable)} AoI(s) in {len(region_areas)} region(s)")
        path = out_dir / "aoi_overview.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
        log.info("wrote %s", path)
        if not show:
            plt.close(fig)

    # --- Per region: that region's AoIs highlighted, others faded ----------- #
    if per_region:
        for r in project.regions:
            areas = region_areas.get(r.name, [])
            if not areas:
                log.info("region %r has no drawable AoIs; skipping its map", r.name)
                continue
            fig = plt.figure(figsize=(8, 7))
            ax, transform = _make_axes(fig, _pad_extent([_aoi_bbox(a) for a in areas]))
            # Context: other regions' AoIs faded (only those falling in the frame
            # actually show once the extent clips them).
            for other in project.regions:
                if other.name == r.name:
                    continue
                for a in region_areas.get(other.name, []):
                    _draw_aoi(ax, a, colors[other.name], transform, highlight=False, label=False)
            for a in areas:
                _draw_aoi(ax, a, colors[r.name], transform, highlight=True, label=True)
            ax.set_title(f"{project.name} -- region: {r.name} ({len(areas)} AoI(s))")
            path = out_dir / f"aoi_region_{_slug(r.name)}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
            log.info("wrote %s", path)
            if not show:
                plt.close(fig)

    if show:
        plt.show()
    return written
