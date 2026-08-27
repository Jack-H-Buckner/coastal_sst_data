#!/usr/bin/env python3
"""
coastal_sst_data -- in-situ station DISCOVERY: who is out there, without downloading anything.

`insitu_acquire` answers "what did the thermometers read". This module answers the question
that comes BEFORE it, while the AoI boxes are still being drawn: *is there a thermometer in
this box at all, and did I miss one by 3 km?* An AoI is a few numbers in a config, and moving
its centre 5 km is free -- but only if you can SEE that a mooring sits just outside the corner.
Discovering that after a multi-hour acquisition run is discovering it too late.

So this is a second seam beside the fetch seam, deliberately CHEAPER than it:

    stations(bbox, start, end, cfg) -> [Station, ...]

Every source implements it out of metadata it can get without touching an observation:

  * `ioos`         -- one ERDDAP advanced-search call (which datasets carry temperature here)
                      plus one `allDatasets` call (where they are, when they ran, and whether
                      they move). Two requests for a whole coast.
  * `marineinsitu` -- the index catalog `insitu_cmems` already fetches and caches for
                      acquisition. Zero extra network on a warm cache.
  * `csv`          -- the user's own files, read locally, so the answer is exact.

The result feeds `plot.plot_aoi_insitu`, wired in as
``coastal-sst-data grids --plot --insitu``.

THE HALO IS DEFINED HERE, ONCE. `discover` grows the AoI's search bbox by `halo_km` and hands
every source the already-grown box; no source applies padding of its own. That keeps
`insitu.pad_deg` -- an acquisition knob, about how generously to search for data you intend to
KEEP -- from quietly changing what a map shows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import DataProduct
from . import insitu_cmems, insitu_csv, insitu_ioos

log = logging.getLogger(__name__)

# How far beyond the AoI to look, in km. A DISTANCE rather than a ratio on purpose: the
# question this answers is "would a small move of the box capture this platform", and a small
# move is a small move whether the AoI is 5 km across or 50.
DEFAULT_HALO_KM = 10.0

# Same degrees-per-km convention as `AreaOfInterest.bbox` (config.py), so a 10 km halo and a
# 10 km `buffer_ew_km` grow a box by the same amount.
_KM_PER_DEG = 111.0

# How much of the searched box a track's reported extent may cover before placing it anywhere
# inside stops being a claim about position. See `Station.locate`.
_LOCALIZABLE_FRAC = 0.5


@dataclass(frozen=True)
class Station:
    """One platform a source knows about, from METADATA ALONE -- no observations were read.

    `lat`/`lon` is the reported position: a point for a mooring, the centre of the reported
    extent for a track (which is a fiction, but a useful one -- it says "something mobile
    passes through here" and the map draws it small and unlabelled to say no more than that).

    `start`/`end` are the platform's WHOLE record, not the project window. That is the point
    of showing them: a station whose record runs to 2019 tells you something about a 2020-2024
    config that a window-clipped span would hide.
    """
    id: str
    title: str
    source: str
    lat: Optional[float]
    lon: Optional[float]
    start: Optional[str]                                  # ISO date, or None if unknown
    end: Optional[str]
    mobile: bool
    bbox: Optional[tuple[float, float, float, float]] = None   # reported extent, when wide

    @property
    def placeable(self) -> bool:
        """Can this station be drawn? A station with no position is reported, not plotted."""
        return (self.lat is not None and self.lon is not None
                and math.isfinite(self.lat) and math.isfinite(self.lon))

    def overlap(self, search_bbox):
        """This platform's reported extent intersected with `search_bbox`, or None.

        A fixed station has no extent, so its "overlap" is its own point.
        """
        w, s, e, n = search_bbox
        bw, bs, be, bn = self.bbox if self.bbox is not None else (self.lon, self.lat,
                                                                 self.lon, self.lat)
        ow, oe, os_, on = max(w, bw), min(e, be), max(s, bs), min(n, bn)
        return None if (ow > oe or os_ > on) else (ow, os_, oe, on)

    def locate(self, search_bbox):
        """Where to mark this platform on a map of `search_bbox` -> (lon, lat), or **None**.

        None means *known to be here, but not known to be anywhere in particular*, and it is
        the whole reason this returns an option rather than a point.

        For a fixed station the answer is simply its position. For a TRACK there may be no
        answer at all, and the two naive ones are both WRONG:

          * The centre of the reported extent is not on the track. Measured on the live IOOS
            catalogue, `ioos-gliderdac-gp-276` advertises lon -144.9..-70.9 -- its midpoint is
            -107.9, in Wyoming, for a platform whose bounds overlap a Puget Sound box.
          * The centre of the OVERLAP is no better when the extent ENGULFS the box, because
            then the overlap *is* the box and its centre is just the middle of the map. That
            is how three North Pacific gliders first came out of this code stacked on the AoI
            centre and written to the sidecar CSV as `inside_aoi=1, km_from_aoi=0.00` -- a
            confident false claim about an open-ocean glider, which is exactly the failure this
            product cannot afford.

        So: a track is placed only when its extent is small enough for the overlap to mean
        something -- less than `_LOCALIZABLE_FRAC` of the searched box in at least one
        dimension. Otherwise it is counted and named, never placed.
        """
        if not self.placeable:
            return None
        if self.bbox is None:
            return (self.lon, self.lat)
        ov = self.overlap(search_bbox)
        if ov is None:
            return None
        w, s, e, n = search_bbox
        fx = (ov[2] - ov[0]) / max(e - w, 1e-9)
        fy = (ov[3] - ov[1]) / max(n - s, 1e-9)
        if fx >= _LOCALIZABLE_FRAC and fy >= _LOCALIZABLE_FRAC:
            return None                          # engulfs the box: no locality to report
        return ((ov[0] + ov[2]) / 2.0, (ov[1] + ov[3]) / 2.0)

    def span_label(self) -> str:
        """`2015-01 - 2026-08` -- the record span, at the precision a map label can carry."""
        def ym(v):
            return str(v)[:7] if v else "?"
        return f"{ym(self.start)} - {ym(self.end)}"


SOURCES = {
    "ioos": insitu_ioos.stations,
    "csv": insitu_csv.stations,
    "marineinsitu": insitu_cmems.stations,
}


# --------------------------------------------------------------------------- #
# The halo
# --------------------------------------------------------------------------- #
def halo_bbox(bbox, halo_km: float = DEFAULT_HALO_KM):
    """(W, S, E, N) grown outward by `halo_km` on every side.

    Longitude degrees shrink with latitude, so the E/W growth is divided by cos(lat) -- the
    same correction `AreaOfInterest.bbox` applies, so the halo is a real distance rather than
    a distance at the equator and something else everywhere people actually work.
    """
    w, s, e, n = bbox
    if halo_km <= 0:
        return (w, s, e, n)
    dlat = halo_km / _KM_PER_DEG
    lat0 = (s + n) / 2.0
    # Guard the pole, where cos -> 0 and the correction diverges.
    dlon = halo_km / (_KM_PER_DEG * max(math.cos(math.radians(lat0)), 1e-6))
    return (w - dlon, max(s - dlat, -90.0), e + dlon, min(n + dlat, 90.0))


def km_from_bbox(lat: float, lon: float, bbox) -> float:
    """How far a point lies OUTSIDE `bbox`, in km. 0.0 for a point inside it.

    The number that answers "how far would I have to move the box to capture this": the
    perpendicular distance to the nearest edge, or the corner distance when the point is
    diagonal to the box.
    """
    w, s, e, n = bbox
    dlat = max(s - lat, lat - n, 0.0)
    dlon = max(w - lon, lon - e, 0.0)
    lat0 = max(min(lat, 90.0), -90.0)
    dy = dlat * _KM_PER_DEG
    dx = dlon * _KM_PER_DEG * math.cos(math.radians(lat0))
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------- #
# Discovery across every configured source
# --------------------------------------------------------------------------- #
def _effective(project):
    """The resolved per-AoI in-situ config, or None when in-situ is not configured at all.

    Reuses `insitu_acquire._build_eff` so the map reads exactly what acquisition would --
    region overrides, station allow/deny lists, the CMEMS dataset id. A config that has not
    selected the product yet is the interesting case, not an error: that is precisely when
    someone is still deciding where the AoIs go, so fall back to the module defaults and
    still draw the map.
    """
    from . import insitu_acquire

    try:
        return insitu_acquire._build_eff(project)
    except Exception as exc:
        log.info("in-situ is not configured (%s); discovering with default settings", exc)
        return None


def _default_cfg():
    """The cfg a source gets when the project has not configured in-situ at all."""
    from . import insitu_acquire

    return insitu_acquire._ds_cfg(None)


def discover(project, g, *, halo_km: float = DEFAULT_HALO_KM,
             sources=None) -> list[Station]:
    """Every platform any configured source knows about within `halo_km` of this AoI.

    One source failing -- a network blip, a missing credential, a mistyped CSV path -- must
    not cost the whole map, so each is tried independently and its failure is logged and
    stepped over. A map with two of three sources on it is worth far more than a traceback.
    """
    from . import insitu_acquire

    eff = _effective(project)
    if eff is not None:
        cfg = eff["ds"].get(g.name) or _default_cfg()
        start, end = eff["time"]["start_date"], eff["time"]["end_date"]
        out_root = eff["out_dir"]
    else:
        cfg = _default_cfg()
        start = project.time.start_date.isoformat()
        end = project.time.end_date.isoformat()
        out_root = Path(project.output_dir)

    want = list(sources) if sources else list(cfg["sources"])
    # `csv` without a path and `marineinsitu` without a credential are the normal state of a
    # config that never opted into them; asking anyway would be noise, not information.
    want = [s for s in want if s in SOURCES]

    bbox = halo_bbox(g.search_bbox, halo_km)
    found: dict[tuple[str, str], Station] = {}
    ok, failed = 0, []
    for src in want:
        # Shared with acquisition: the CMEMS index is ~28 MB and there is no reason for the
        # map to fetch its own copy of a file the acquire run already has on disk.
        src_cfg = {**cfg, "cache_dir": out_root / products_dir() / src / "_cache",
                   "resolution_m": g.resolution_m}
        try:
            if eff is not None:
                insitu_acquire._ensure_source_auth(src, eff)
            for st in SOURCES[src](bbox, start, end, src_cfg):
                found[(st.source, st.id)] = st
            ok += 1
        except Exception as exc:
            log.warning("  %s station discovery failed (%s); it is missing from this map",
                        src, exc)
            failed.append(f"{src} ({exc})")

    # EVERY source failing is not "no stations here" -- it is "we do not know", and a map
    # drawn from it would assert an empty coastline. That is the same failure the in-situ
    # products refuse (`insitu_acquire.run`: an empty channel must be loud, not silent), so
    # this raises and the caller writes no figure at all rather than a misleading one.
    if want and not ok:
        raise RuntimeError("every in-situ source failed: " + "; ".join(failed))

    out = sorted(found.values(), key=lambda s: (s.mobile, s.source, s.id))
    log.info("  %d platform(s) within %.1f km of %s (%d fixed, %d moving)",
             len(out), halo_km, g.name,
             sum(not s.mobile for s in out), sum(s.mobile for s in out))
    return out


def products_dir() -> str:
    """`INSITU` -- the product directory, read from the registry rather than spelled again."""
    from .. import products

    return products.spec(DataProduct.insitu).dir
