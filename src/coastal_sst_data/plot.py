"""Quick-look maps of a project's regions and AoIs.

Given a validated ``Project`` this draws three kinds of figure:

  * an **overview** map with every AoI in the project, coloured by region, so you
    can see the whole study area and how the AoIs group into regions;
  * one **per-region** map, zoomed to that region, with its AoIs highlighted (and
    any neighbouring AoIs from other regions faded in for context); and
  * one **in-situ context** map per AoI (opt-in), showing every temperature
    platform within a halo of the box -- fixed stations labelled with the years
    they ran, moving platforms as small unlabelled dots.

Each AoI is drawn as its acquisition bounding box (the buffered box from the
config) plus a centre marker and label. Plotting is intentionally an *optional*
capability: matplotlib is imported lazily so the core package has no plotting
dependency, and coastlines are added only if ``cartopy`` happens to be installed
(otherwise the maps are plain lon/lat axes).

The in-situ layer is opt-in for the same reason it is useful: it is the only part
of this module that touches the network. The first two figures answer "where are
my AoIs" from the config alone; the third answers "is there a thermometer in this
box, and did I miss one by 3 km" -- which cannot be known without asking ERDDAP
and the Copernicus index. Discovery is metadata-only (see
``processes/insitu_stations.py``): no observation is downloaded to draw a map.

Wired into the CLI as::

    coastal-sst-data grids --config config.yaml --plot
    coastal-sst-data grids --config config.yaml --plot --insitu --insitu-halo-km 25
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


def _scale_bbox(bbox: BBox, factor: float, min_pad_deg: float = 0.05) -> BBox:
    """``bbox`` grown about its centre by a LINEAR ``factor`` (2.5 -> half-span x 2.5)."""
    w, s, e, n = bbox
    cx, cy = (w + e) / 2.0, (s + n) / 2.0
    hx = max((e - w) / 2.0 * factor, (e - w) / 2.0 + min_pad_deg)
    hy = max((n - s) / 2.0 * factor, (n - s) / 2.0 + min_pad_deg)
    return (cx - hx, cy - hy, cx + hx, cy + hy)


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


def _draw_aoi(ax, area, color, transform, *, highlight=True, label=True, center=True):
    """Draw one AoI: its bbox rectangle, a centre marker, and (optionally) a label.

    `center=False` drops the centre dot, which the in-situ maps need: there it would be the
    same colour and roughly the same size as an in-AoI station marker, and a mark that reads
    as a buoy but is not one is worse than no mark at all.
    """
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
    if center:
        ax.plot([area.center_lon], [area.center_lat], marker="o",
                markersize=4 if highlight else 3, color=color, zorder=4, **tkw)
    if label:
        ax.text(area.center_lon, area.center_lat, "  " + area.name, fontsize=7,
                va="center", ha="left", color="black" if highlight else "0.4",
                zorder=5, **tkw)


def _inside(lon: float, lat: float, bbox: BBox) -> bool:
    """Does this point fall inside the AoI's own box (as opposed to the halo)?"""
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


def _stack_labels(placed: list[float], y: float, dy: float) -> float:
    """A free label row near ``y``, pushing down past rows already taken.

    Co-located buoys are the norm, not the exception -- Twanoh has an NDBC buoy and a historic
    MAPCO2 mooring 0.005 deg apart -- and two labels drawn at the same point are one unreadable
    smear. This is a deliberately crude de-overlap: it only separates labels vertically, which
    is all that is needed when the collisions are pairs and triples.
    """
    while any(abs(y - taken) < dy for taken in placed):
        y -= dy
    placed.append(y)
    return y


def _draw_stations(ax, stations, transform, *, aoi_bbox: BBox, halo: BBox, color,
                   max_labels: int = 25) -> tuple[int, int, int]:
    """Draw one AoI's platforms -> (n_fixed, n_moving_placed, n_moving_unplaced).

    Three marks, because the three carry different amounts of truth.

    A FIXED station has a real position and a real record span, so it gets a full marker and a
    label saying when it ran -- that label is the whole point, since "there is a buoy here" and
    "there was a buoy here until 2019" are different answers for a 2020-2024 config.

    A MOVING platform has a position only in the loosest sense. Where its reported extent is
    small enough to localize, it gets a small unlabelled dot and a faint dotted outline of that
    extent. Labelling them would crowd the map with names of things that are not there any
    more, which is the opposite of what this figure is for.

    A moving platform whose extent ENGULFS the searched box gets NOTHING drawn -- see
    `Station.locate`. Marking it would place an open-ocean glider on top of the AoI. It is
    counted in the return value so the caller can say how many were left off.

    Stations already inside the AoI are drawn in the region colour and ones in the halo in a
    muted colour, so "captured" and "one nudge away" separate at a glance.
    """
    from matplotlib.patches import Rectangle

    tkw = {"transform": transform} if transform is not None else {}
    cx, cy = (aoi_bbox[0] + aoi_bbox[2]) / 2.0, (aoi_bbox[1] + aoi_bbox[3]) / 2.0

    fixed = [s for s in stations if s.placeable and not s.mobile]
    moving = [s for s in stations if s.placeable and s.mobile]

    unplaced = 0
    for st in moving:
        at = st.locate(halo)
        if at is None:
            unplaced += 1
            continue
        ov = st.overlap(halo)
        if st.bbox is not None and ov is not None:
            # The extent CLIPPED to the halo: a glider's reported bounds span tens of degrees,
            # and an unclipped rectangle would drag the figure out to ocean-basin scale.
            ax.add_patch(Rectangle((ov[0], ov[1]), ov[2] - ov[0], ov[3] - ov[1],
                                   facecolor="none", edgecolor="0.55", linewidth=0.5,
                                   linestyle=":", alpha=0.6, zorder=5, **tkw))
        ax.plot([at[0]], [at[1]], marker=".", markersize=4, color="0.45",
                alpha=0.8, linestyle="none", zorder=6, **tkw)

    # Label budget. A 1.5 deg Puget Sound box really does return ~200 stations, and 200 labels
    # is an unreadable figure -- but a silently truncated one would be worse, so what is left
    # off is said out loud and every station is in the sidecar CSV regardless.
    by_distance = sorted(fixed, key=lambda s: math.hypot(s.lon - cx, s.lat - cy))
    labelled = {id(s) for s in by_distance[:max_labels]}
    if len(fixed) > max_labels:
        log.info("  %d fixed station(s) exceed the %d-label budget; labelling the %d nearest "
                 "the AoI centre (all of them are in the sidecar CSV)",
                 len(fixed), max_labels, max_labels)

    # ~2 label rows per 100 of the frame's height, so the spacing follows the zoom.
    w_, s_, e_, n_ = halo
    dy = (n_ - s_) * 0.055
    dx = dy * 0.45
    # Past this longitude a right-hand label runs off the frame, so it goes on the left
    # instead. Station ids here are long (`noaa_nos_co_ops_9447130`) and the frame edge is
    # exactly where the interesting ones sit -- the ones a bigger AoI would reach.
    flip_at = e_ - (e_ - w_) * 0.25
    taken: list[float] = []
    for st in by_distance:                       # nearest first, so the centre wins the space
        here = _inside(st.lon, st.lat, aoi_bbox)
        c = color if here else "0.35"
        ax.plot([st.lon], [st.lat], marker="o", markersize=6 if here else 5,
                markerfacecolor=c, markeredgecolor="black", markeredgewidth=0.6,
                linestyle="none", zorder=7, **tkw)
        if id(st) not in labelled:
            continue
        ly = _stack_labels(taken, st.lat, dy)
        left = st.lon > flip_at
        lx = st.lon - dx if left else st.lon + dx
        if abs(ly - st.lat) > 1e-9:              # displaced: say which marker it belongs to
            ax.plot([st.lon, lx], [st.lat, ly], linewidth=0.4, color="0.5", zorder=6, **tkw)
        ax.text(lx, ly, f"{st.id}\n{st.span_label()}", fontsize=5.5, va="center",
                ha="right" if left else "left", color="black" if here else "0.3",
                zorder=8, **tkw)
    return len(fixed), len(moving) - unplaced, unplaced


def _region_colors(project: Project):
    """Map each region name to a stable colour from a qualitative colormap."""
    import matplotlib.pyplot as plt

    names = [r.name for r in project.regions]
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(names)}


def _write_station_csv(path: Path, stations, aoi_bbox: BBox, halo: BBox) -> None:
    """The complete station list beside the figure -- what the label budget could not show.

    `km_from_aoi` is the number the map is really about: how far outside the box a platform
    sits, and therefore how far the box would have to move to capture it. 0 means it is
    already inside.

    `lat`/`lon` are the position DRAWN, which for a track is the centre of where its extent
    meets the searched box rather than the centre of the extent itself (see
    `Station.draw_position`). The raw claim is kept in `reported_bbox` so nothing the
    catalogue said is lost.
    """
    import csv as _csv

    from .processes.insitu_stations import km_from_bbox

    with path.open("w", newline="") as fh:
        wr = _csv.writer(fh)
        wr.writerow(["source", "id", "title", "lat", "lon", "start", "end",
                     "mobile", "inside_aoi", "km_from_aoi", "reported_bbox"])
        for st in sorted(stations, key=lambda s: (s.mobile, s.source, s.id)):
            bb = "" if st.bbox is None else " ".join(f"{v:.4f}" for v in st.bbox)
            at = st.locate(halo)
            if at is None:
                # Known to be around, not known to be anywhere: the position columns stay
                # EMPTY rather than carrying a guess. A blank is a fact; 0.00 km would be a
                # claim, and for a Station Papa glider a false one.
                wr.writerow([st.source, st.id, st.title, "", "", st.start, st.end,
                             int(st.mobile), "", "", bb])
                continue
            lon, lat = at
            wr.writerow([st.source, st.id, st.title, f"{lat:.5f}", f"{lon:.5f}",
                         st.start, st.end, int(st.mobile), int(_inside(lon, lat, aoi_bbox)),
                         f"{km_from_bbox(lat, lon, aoi_bbox):.2f}", bb])


def plot_aoi_insitu(project: Project, area, stations, *, out_dir: Path, color,
                    halo_km: float, pad_factor: float = 2.5, max_labels: int = 25,
                    show: bool = False, dpi: int = 150) -> list[Path]:
    """One AoI's in-situ context map -> [png, csv].

    The extent is the UNION of the pad box and the halo box, so raising `--insitu-halo-km`
    past the pad factor widens the figure rather than searching an area the map then crops
    away. The halo itself is drawn as a dashed rectangle: the search distance should be
    visible on the figure, not implied by it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from .processes.insitu_stations import halo_bbox

    bb = _aoi_bbox(area)
    if bb is None:
        return []
    halo = halo_bbox(bb, halo_km)
    extent = _pad_extent([_scale_bbox(bb, pad_factor), halo], pad_frac=0.02, min_pad_deg=0.005)

    fig = plt.figure(figsize=(8, 7))
    ax, transform = _make_axes(fig, extent)
    tkw = {"transform": transform} if transform is not None else {}
    w, s, e, n = halo
    ax.add_patch(Rectangle((w, s), e - w, n - s, facecolor="none", edgecolor="0.4",
                           linewidth=0.8, linestyle="--", zorder=2, **tkw))
    _draw_aoi(ax, area, color, transform, highlight=True, label=False, center=False)
    n_fixed, n_moving, n_wide = _draw_stations(ax, stations, transform, aoi_bbox=bb, halo=halo,
                                               color=color, max_labels=max_labels)

    # Only the keys that are actually on the map: a legend entry for a symbol that was never
    # drawn invites a hunt for something that is not there.
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor="black", markersize=6, label="fixed station, in AoI"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.35",
               markeredgecolor="black", markersize=5, label="fixed station, in halo"),
    ]
    if n_moving:
        handles.append(Line2D([0], [0], marker=".", linestyle="none", color="0.45",
                              markersize=6, label="moving platform"))
    handles.append(Line2D([0], [0], linestyle="--", color="0.4", linewidth=0.8,
                          label=f"{halo_km:g} km halo"))
    ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.9)
    title = (f"{area.name} -- {n_fixed} fixed station(s), {n_moving} moving platform(s) "
             f"within {halo_km:g} km")
    if n_wide:
        # Said on the figure, not just in the log: a count that only appears in the CSV reads
        # as "no gliders here" to anyone looking at the map.
        title += (f"\n(+{n_wide} wide-ranging track(s) not placed -- see the sidecar CSV)")
    ax.set_title(title, fontsize=10)

    png = out_dir / f"aoi_insitu_{_slug(area.name)}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    log.info("wrote %s", png)
    if not show:
        plt.close(fig)

    csv_path = out_dir / f"aoi_insitu_{_slug(area.name)}.csv"
    _write_station_csv(csv_path, stations, bb, halo)
    log.info("wrote %s", csv_path)
    return [png, csv_path]


def plot_project_aois(project: Project, *, grids: "dict[str, AoiGrid] | None" = None,
                      out_dir: "str | Path | None" = None, overview: bool = True,
                      per_region: bool = True, show: bool = False,
                      dpi: int = 150, insitu: bool = False,
                      insitu_halo_km: "float | None" = None, insitu_pad: float = 2.5,
                      insitu_max_labels: int = 25) -> list[Path]:
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
    insitu      also write one in-situ context map per AoI. OFF BY DEFAULT because it
                queries ERDDAP and the Copernicus index -- ``--plot`` alone stays offline.
    insitu_halo_km   how far beyond each AoI to look for stations (default 10 km).
    insitu_pad       how much wider than the AoI to draw those maps (linear factor).
    insitu_max_labels  cap on labelled fixed stations per map; the rest go to the CSV.

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

    # --- Per AoI: the in-situ context map (opt-in; this one hits the network) ---- #
    if insitu:
        from .processes import insitu_stations

        halo = insitu_stations.DEFAULT_HALO_KM if insitu_halo_km is None else insitu_halo_km
        if grids is None:
            from .grid import project_grids
            grids = project_grids(project)
        for rname, areas in region_areas.items():
            for a in areas:
                g = grids.get(a.name)
                if g is None:                  # gridding failed; nothing to search around
                    continue
                log.info("=== in-situ context: %s ===", a.name)
                try:
                    found = insitu_stations.discover(project, g, halo_km=halo)
                except Exception as exc:
                    # One AoI's discovery failing must not cost the other AoIs' maps.
                    log.warning("  %s: station discovery failed (%s); no in-situ map", a.name,
                                exc)
                    continue
                written.extend(plot_aoi_insitu(
                    project, a, found, out_dir=out_dir, color=colors[rname], halo_km=halo,
                    pad_factor=insitu_pad, max_labels=insitu_max_labels, show=show, dpi=dpi))

    if show:
        plt.show()
    return written
