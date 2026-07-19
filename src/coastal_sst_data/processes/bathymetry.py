#!/usr/bin/env python3
"""
OCEANSR -- bathymetry static covariate (NOAA NCEI CUDEM + GMRT, STACKED per source).

`sources.bathymetry.sources` lists the DEMs to acquire (default: both). Each is fetched
INDEPENDENTLY -- there is no fallback (D10): CUDEM (NOAA NCEI Continuously Updated DEM,
1/9 arc-second ~3 m seamless topobathy) and GMRT (GridServer, ~100 m global) each write
their own tree:

    data/BATHYMETRY/<source>/aligned/<aoi_id>/<aoi_id>.nc

so the datacube can ship one channel per source (`depth_cudem`, `depth_gmrt`). A source with
no coverage simply contributes no channel there; the user closes the gap by stacking another.

This module is the thin ORCHESTRATOR. The DEM readers live in sibling helpers so the three
data sources are no longer wired together and a failure in one cannot abort another:

  * `processes.bathymetry_cudem` -- the CUDEM COG reader + tile-index cache (NAVD88).
  * `processes.bathymetry_gmrt`  -- the GMRT GridServer reader (already ~MSL).
  * `processes.datum`            -- the DEM->MSL offset resolver (VDatum for CUDEM/NAVD88).

Failure isolation:
  * Each (AoI, source) is acquired in its own try/except. A transient read failure
    (`bathymetry_cudem.TileReadError`, a 503/timeout) fails ONLY that row -- it is named in
    the run report and retried next run -- while every other source and AoI keeps going. It
    is never turned into a permanent GMRT downgrade.
  * The datum resolution is likewise decoupled from the DEM download. If VDatum is down when
    the CUDEM tiles came back fine, the DEM is SAVED (the expensive part) with
    `datum_status="pending_retry"`, and the offset alone is re-resolved on a later run from
    the on-disk elevation -- no tile re-download. This is distinct from datum's own
    `unresolved_assumed_zero` (a final no-coverage answer that must not loop).

Variables (all m):
  elevation  : mean elevation (neg below sea level)
  depth      : mean water depth over the cell (= mean of max(-elev,0))
  depth_p25  : 25th-percentile depth within the cell (sub-grid variability)
  depth_p75  : 75th-percentile depth within the cell

For GMRT (no sub-grid) depth_p25 = depth_p75 = depth. Each source's DEM->MSL datum offset is
resolved INLINE (0 for GMRT/MSL, VDatum for CUDEM/NAVD88) and stamped onto its output as
`datum_offset_m` / `datum_status`, so `elevation_<source>` can be referenced to MSL
downstream. Adding a DEM source means adding its reader + `datum.DEM_DATUM` entry.

Usage (from the project root):
    python -m coastal_sst_data.processes.bathymetry --config configs/config.yaml --aoi hood_canal
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, net, provenance, report, store
from . import bathymetry_cudem as cudem, bathymetry_gmrt as gmrt, datum, tides

log = logging.getLogger(__name__)
SOURCE = "bathymetry"

# Re-exported so `acquire`/`run` and the tests can reference it from the orchestrator without
# reaching into the CUDEM helper: a transient CUDEM read failure raises this.
TileReadError = cudem.TileReadError

# The datum_status stamped when the DEM came back fine but its offset could not be resolved
# (e.g. VDatum was down). It is retryable -- and deliberately NOT datum's own
# `unresolved_assumed_zero`, which is a FINAL no-coverage answer that must not loop forever.
PENDING_DATUM = "pending_retry"


# --------------------------------------------------------------------------- #
# Source registry. Each fetcher: (g: AoiGrid, params) -> (elev, depth, p25, p75,
# used) on the shared grid, or None to signal "insufficient coverage" (that source
# contributes no channel here). Extensible: gebco, copernicus_glo30, ...
# --------------------------------------------------------------------------- #
SOURCES = {"cudem": cudem._source_cudem, "gmrt": gmrt._source_gmrt}


def _fetch_one(source, g: AoiGrid, params):
    """Run ONE source's fetcher. No fallback (D3): sources are stacked, not substituted.

    A NETWORK failure still propagates as `TileReadError` -- the (AoI, source) fails, is
    named in the run report, and the next run retries it properly, rather than a CDN hiccup
    being frozen into a permanent downgrade. Genuine NO-COVERAGE returns None: that source
    simply contributes no channel here, and the user closes the gap by stacking another.
    """
    try:
        return SOURCES[source](g, params)
    except TileReadError:
        raise                                   # transient: caller records it, retry next run
    except Exception as exc:
        log.warning("  %s: %s read failed (%s)", g.name, source, exc)
        return None


# --------------------------------------------------------------------------- #
# Datum (decoupled from the DEM download): resolve best-effort, retry from disk.
# --------------------------------------------------------------------------- #
def _datum_attrs(drec: dict) -> dict:
    """The datum_* attrs stamped onto a source's output, from a resolver record."""
    return {
        "datum_offset_m": float(drec["datum_offset_m"]),
        "datum_status": str(drec["status"]),
        "datum_method": str(drec["method"]),
        "dem_vertical_datum": str(drec.get("dem_vertical_datum", "unknown")),
    }


def _resolve_datum(g: AoiGrid, elev, source, override, stations) -> dict:
    """Resolve THIS DEM source's DEM->MSL offset, best-effort.

    A failure here (VDatum/CO-OPS outage) must NOT discard a DEM we already downloaded, so it
    is caught and returned as a PENDING record: the DEM is saved and only the offset is
    retried on a later run. GMRT resolves to 0 with no network, so it never goes pending.
    """
    try:
        return datum.resolve_aoi(g, elev, source, override=override, stations=stations)
    except Exception as exc:
        log.warning("  %s: %s datum resolution failed (%s); saving the DEM with the datum "
                    "PENDING -- the offset alone is retried next run", g.name, source, exc)
        return {"datum_offset_m": 0.0, "status": PENDING_DATUM,
                "method": "datum_fetch_failed",
                "dem_vertical_datum": datum.DEM_DATUM.get(source, "unknown")}


def _datum_pending(path: Path) -> bool:
    """True if an existing output was written with its datum offset still PENDING."""
    try:
        with xr.open_dataset(path) as ds:
            return str(ds.attrs.get("datum_status", "")) == PENDING_DATUM
    except Exception:
        return False


def _reresolve_datum(path: Path, g: AoiGrid, source, override, stations) -> dict | None:
    """Re-resolve a pending datum from the ON-DISK elevation -- no tile re-download.

    Reads the DEM we already saved, resolves the offset against its `elevation`, restamps the
    datum_* attrs, and rewrites the file atomically. Returns the record on success (still
    PENDING if the resolver could not reach a source), or None if the file is unreadable.
    """
    try:
        with xr.open_dataset(path) as ds:
            ds = ds.load()
    except Exception as exc:
        log.warning("  %s: could not re-open %s to retry its datum (%s)", g.name, path.name, exc)
        return None
    drec = _resolve_datum(g, ds["elevation"].values, source, override, stations)
    ds.attrs.update(_datum_attrs(drec))
    ds = ds.rio.write_crs(g.target_crs)
    store.write_output(ds, path.parent, path.stem, _fmt_of(path))
    return drec


def _fmt_of(path: Path) -> str:
    return "geotiff" if path.suffix.lower() in (".tif", ".tiff") else "netcdf"


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's bathymetry settings: which DEM SOURCES to STACK, region-overridable.

    Distinct-data sources (D10): the user lists the DEMs to acquire, and each writes its own
    `<source>/` tree and its own cube channel (`depth_cudem`, `depth_gmrt`). CUDEM is
    CONUS-only, so an AoI outside it stacks GMRT (or its own DEM) instead. There is no
    `source`/`fallback`/`default_source` any more -- a config still setting one fails
    validation rather than being silently ignored.

    The tuning knobs (pad, subgrid, cover threshold, the CUDEM index) are project-global:
    they decide how a DEM is AGGREGATED, not which DEM has coverage.
    """
    srcs = _opt(opts, "sources", None)
    if srcs is None:
        srcs = list(SOURCES)                     # default: stack every known DEM source
    elif isinstance(srcs, str):
        srcs = [srcs]
    # `datum_offset_m` is the region-only override honoured per DEM source by the datum resolver.
    return {"sources": [str(s) for s in srcs],
            "datum_override": (None if _opt(opts, "datum_offset_m", None) is None
                               else float(_opt(opts, "datum_offset_m")))}


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.bathymetry)
    if opts is None:
        raise ValueError("bathymetry is not a selected product in this config")

    bathy_root = Path(project.output_dir) / "BATHYMETRY"
    cache = _opt(opts, "cudem_index_cache", None)
    params = {
        "pad_deg": float(_opt(opts, "pad_deg", 0.02)),
        "layer": _opt(opts, "layer", "topo"),
        "resolution": _opt(opts, "resolution", "max"),
        "stats_subgrid_m": float(_opt(opts, "stats_subgrid_m", 10.0)),
        "min_cudem_cover": float(_opt(opts, "min_cudem_cover", 0.5)),
        "cudem_urllist": _opt(opts, "cudem_urllist", cudem.CUDEM_URLLIST),
        "cudem_cache": Path(cache) if cache else bathy_root / "urllist_cudem.txt",
        "tmp_dir": bathy_root / "_tmp",
    }
    return {
        "project": project,
        "config_sha256": project.config_sha256,
        "params": params,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.bathymetry))
               for a in project.all_areas},
        "bathy_root": bathy_root,
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
    }


def _source_out_root(bathy_root: Path, source: str) -> Path:
    """Per-source aligned dir: BATHYMETRY/<source>/aligned. Each DEM is kept separate so
    stacking CUDEM and GMRT does not have one overwrite the other."""
    return bathy_root / source / "aligned"


def run(eff, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    """Build one static bathymetry NetCDF per (AoI, SOURCE), each in its own `<source>/` tree.

    Every configured source is acquired independently (no fallback); `only_source` narrows
    the run to one DEM. Each (AoI, source) is isolated: a failure -- a transient CUDEM tile
    read, or any other -- fails ONLY that row and the loop keeps going. Each source's
    DEM->MSL datum offset is resolved INLINE and stamped onto that source's output; a datum
    outage saves the DEM with the offset PENDING and retries it (from disk) on a later run.
    """
    params = eff["params"]
    bathy_root, fmt, overwrite = eff["bathy_root"], eff["fmt"], eff["overwrite"]

    names = select_aois(grids, only_aoi)
    rep = report.ProductReport("bathymetry")
    stations = None                       # CO-OPS gauge list for the datum cross-check (lazy)

    def _stations():
        """Lazily fetch the CO-OPS station list once (only when a NAVD88 DEM needs it)."""
        nonlocal stations
        if stations is None:
            try:
                stations = tides.fetch_stations()
            except Exception as exc:
                log.warning("  CO-OPS station list unavailable (%s); no datum cross-check", exc)
                stations = []
        return stations

    for name in names:
        g = grids[name]
        ds_cfg = eff["ds"][name]
        override = ds_cfg["datum_override"]
        sources = [s for s in ds_cfg["sources"] if only_source is None or s == only_source]

        for source in sources:
            log.info("=== AOI: %s | grid=%dx%d @ %.0fm | source=%s ===",
                     name, g.width, g.height, g.resolution_m, source)
            out_dir = _source_out_root(bathy_root, source)
            out_path = out_dir / name / f"{name}.nc"

            if store.done(out_path, store.REQUIRED_VARS["BATHYMETRY"],
                          shape=(g.height, g.width), overwrite=overwrite):
                # The DEM is complete. If only its datum is still pending (VDatum was down when
                # it was built), retry the OFFSET ALONE from the on-disk elevation -- no tiles.
                if not dry_run and _datum_pending(out_path):
                    if datum.DEM_DATUM.get(source) == "NAVD88":
                        _stations()
                    drec = _reresolve_datum(out_path, g, source, override, stations)
                    if drec is not None and drec["status"] != PENDING_DATUM:
                        log.info("  datum retry succeeded: offset=%.3f m (%s)",
                                 drec["datum_offset_m"], drec["status"])
                        rep.wrote(source=str(out_path))
                    else:
                        log.warning("  %s: %s datum still pending; will retry next run",
                                    name, source)
                        rep.skip()
                    continue
                log.info("  already processed, skipping"); rep.skip(); continue
            if dry_run:
                log.info("  [dry-run] would build bathymetry (%s) for %s", source, name); continue

            try:
                res = _fetch_one(source, g, params)
            except Exception as exc:
                # Isolated: a transient CUDEM tile read (TileReadError) or any other error
                # fails ONLY this (AoI, source). Other sources/AoIs continue; retried next run.
                log.warning("  %s: %s fetch failed (%s); other sources and AoIs continue",
                            name, source, exc)
                rep.fail(f"{name}:{source}", str(exc))
                continue
            if res is None:
                log.warning("  %s: %s has no coverage here; no channel from it (stack "
                            "another source to fill the gap)", name, source)
                rep.fail(name, f"{source}: no coverage")
                continue
            elev, depth, dp25, dp75, used = res

            # The datum offset follows THIS DEM source (CUDEM=NAVD88 -> VDatum; GMRT=MSL -> 0),
            # resolved from the elevation we just built and stamped onto the output so a
            # downstream user can reference `elevation_<source>` to MSL. A resolver outage does
            # NOT discard the DEM: it is saved with the offset pending and retried later.
            if datum.DEM_DATUM.get(source) == "NAVD88":
                _stations()
            drec = _resolve_datum(g, elev, source, override, stations)

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
            ds.attrs.update(aoi_id=name, source=used, bathy_source=source,
                            processing="aggregated to AOI grid (mean, p25, p75 depth per cell)",
                            **_datum_attrs(drec),
                            **provenance.stamp(eff))
            # Static layer: the stem is the bare AoI name -- no time stamp to encode.
            log.info("  wrote %s  [%s]  datum_offset=%.3f m (%s)",
                     store.write_output(ds, out_dir / name, name, fmt), used,
                     drec["datum_offset_m"], drec["status"])
            rep.wrote(source=used)
    rep.log_summary()
    return rep


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None) -> None:
    """Acquire bathymetry for a validated Project. Entry point for pipeline.py.

    The DEM sources are the region-resolved `sources` LIST per AoI (D10), validated against
    the SOURCES registry so a typo fails loudly before a tile is fetched. Every configured
    source is acquired and stacked; `source` (from the pipeline, or unset on a direct CLI
    run) narrows to one DEM. There is no fallback -- a source with no coverage simply
    contributes no channel, and a source that fails to read fails only its own (AoI, source).
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    bad = sorted(f"{n}:{s}" for n, c in eff["ds"].items() for s in c["sources"]
                 if s not in SOURCES)
    if bad:
        raise ValueError(f"bathymetry source not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(SOURCES)}.")
    net.setup_gdal_env()      # /vsicurl CUDEM tile reads: deadline + retries
    return run(eff, grids, aois, dry_run, only_source=source)


def main():
    entry.process_main(
        acquire, "coastal_sst_data bathymetry (per-region CUDEM/GMRT) acquisition.")


if __name__ == "__main__":
    main()
