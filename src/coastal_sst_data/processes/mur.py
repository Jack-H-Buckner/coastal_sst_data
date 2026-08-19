#!/usr/bin/env python3
"""
OCEANSR -- MUR L4 SST backbone acquisition.

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; MUR settings come from `sources.mur`. For each AOI and each
day it subsets `analysed_sst` from the daily GHRSST MUR L4 granule at PODAAC to
the AOI lat/lon window, upsamples onto the AOI grid (identical to the
ECOSTRESS/Landsat grid), and writes one aligned NetCDF per day. The global 1 km
file is never fully downloaded.

WHERE the subsetting happens is the `access` option:

  * download (default) -- stream the granule (earthaccess.open) and subset in
    this process, over HDF5 range reads.
  * opendap            -- ask PO.DAAC's Hyrax for a DAP4 hyperslab: the server
    sends the window and nothing else.

They produce bit-identical grids. `download` moves far more than the window
contains, because the remote HDF5 is read through fsspec in 8 MB blocks with
background prefetch and the header reads that locate the window are scattered:
measured live on one granule over a 50 km AOI, 26.83 MB against 0.03 MB.
`download` stays the default only because it is the long-proven path.

MUR is a gap-free L4 analysis, so it has no cloud mask; `valid` = finite SST
(i.e. water). It is the always-present backbone the model fills high-res detail
onto. A later stage bins these to the daily datacube.

`overpass_sensors` narrows the days fetched to the ones a named thermal sensor
actually flew over this AoI, read from the sensors' aligned dirs. MUR's main job
downstream is the cloud-filter baseline, which can only ever read a day a sensor
also imaged -- so over a multi-year window most of the daily downloads produce a
file nothing reads. Opt-in: omit the option and every day is fetched, as before.

Usage (run from the OCEANSR project root, Earthdata auth via ~/.netrc):
    python src/acquire_mur.py --config configs/config.yaml
    python src/acquire_mur.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_mur.py --config configs/config.yaml --dry-run
"""

from __future__ import annotations

import io
import logging
import re
import threading
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import xarray as xr

import earthaccess
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import auth, entry, naming, net, overpass, provenance, report, store

log = logging.getLogger(__name__)

# --- MUR product constants -------------------------------------------------- #
# Describe the MUR L4 product itself (not user choices); overridable via the
# mur product options block in the config.
SOURCE = "mur"
SHORT_NAME = "MUR-JPL-L4-GLOB-v4.1"
DEFAULT_VARIABLE = "analysed_sst"
DEFAULT_PAD_DEG = 0.05
# `download` streams the granule and subsets client-side; `opendap` asks Hyrax for the window.
# Measured live on one granule over a 50 km AOI: 26.83 MB / 13.8 s vs 0.03 MB / 4.1 s, for a
# bit-identical gridded result. `download` stays the default because it is the path with years
# of runs behind it -- switch per config once opendap has proved itself on your own AOIs.
DEFAULT_ACCESS = "download"


# The nominal stamp that opens a MUR granule id: 20230715090000-JPL-L4_GHRSST-...
_UR_STAMP_RE = re.compile(r"(\d{8})\d{6}")


# Config is loaded/validated by coastal_sst_data.config; the per-AoI CRS + grid
# come from coastal_sst_data.grid (AoiGrid), shared with every other product.


# --------------------------------------------------------------------------- #
# Granule selection (the `overpass_sensors` filter)
# --------------------------------------------------------------------------- #
def _iso(s: str):
    """A CMR timestamp -> datetime, with or without fractional seconds. None if unparseable."""
    try:
        return pd.Timestamp(str(s).replace("Z", "+00:00")).tz_convert(None).to_pydatetime()
    except Exception:
        return None


def _granule_day_stamp(granule) -> str | None:
    """The YYYYMMDD a daily MUR granule IS, from its CATALOGUE metadata.

    Read without opening the file, because not opening it is the whole point of the filter.

    NOT `RangeDateTime.BeginningDateTime`, which is what the MODIS coincidence filter reads
    (modis._granule_time): a MUR L4 granule's temporal extent is the 24 h ANALYSIS WINDOW
    centred on its 09:00Z nominal time, so its BeginningDateTime falls on the PREVIOUS
    calendar day -- selecting on it would shift every chosen day by one, silently. The
    granule id carries the nominal stamp; the fallback is the MIDPOINT of the range, which
    lands back on 09:00Z of the right day. None when neither is readable.
    """
    umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    meta = granule.get("meta", {}) if hasattr(granule, "get") else {}
    for ur in (umm.get("GranuleUR"), meta.get("native-id")):
        m = _UR_STAMP_RE.match(str(ur or ""))
        if m:
            return m.group(1)

    rng = (umm.get("TemporalExtent") or {}).get("RangeDateTime") or {}
    beg, end = _iso(rng.get("BeginningDateTime")), _iso(rng.get("EndingDateTime"))
    if beg and end:
        return naming.day_stamp(beg + (end - beg) / 2)
    return None


def _select_granules(granules, day_stamps: set[str] | None) -> list:
    """Restrict the daily granule list to the days a named sensor actually flew.

    `day_stamps=None` -> unrestricted (the default: MUR is a daily backbone).

    A granule whose day cannot be READ is kept, with a warning. Fail-open on identification
    only: the worst case is downloading what the unfiltered code downloaded anyway, whereas
    fail-closed on a CMR metadata change would silently delete days from the record.
    """
    if day_stamps is None:
        return list(granules)
    kept, unknown = [], 0
    for granule in granules:
        stamp = _granule_day_stamp(granule)
        if stamp is None:
            unknown += 1
            kept.append(granule)
        elif stamp in day_stamps:
            kept.append(granule)
    if unknown:
        log.warning("    %d granule(s) carry no readable day in their catalogue metadata; "
                    "kept rather than dropped (the overpass filter cannot identify them)",
                    unknown)
    return kept


# --------------------------------------------------------------------------- #
# MUR per-granule processing
# --------------------------------------------------------------------------- #
def _to_grid(da, target_crs, transform, width, height, geom_proj, grid_cfg) -> xr.DataArray:
    """A lat/lon-indexed MUR window -> the AOI grid. Shared by every access backend.

    Split out of `subset_and_reproject` so the OPeNDAP backend, which never holds a file
    object, reaches the identical reprojection rather than a second copy of it.
    """
    if grid_cfg.get("to_celsius", False):
        da = da - 273.15

    # Standard orientation + CRS, then reproject (bilinear upsample 1 km -> grid).
    da = da.rename({"lon": "x", "lat": "y"}).sortby("y", ascending=False)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    rs = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    out = da.rio.reproject(dst_crs=target_crs, shape=(height, width),
                           transform=transform, resampling=rs, nodata=np.nan)
    return out.rio.clip([geom_proj], target_crs, drop=False)


def subset_and_reproject(fobj, variable, bbox_ll, pad, target_crs, transform,
                         width, height, geom_proj, grid_cfg) -> tuple[xr.DataArray, pd.Timestamp]:
    """Open one daily MUR granule lazily, subset to the AOI, upsample to grid."""
    w, s, e, n = bbox_ll
    # `with`, because the window is `.load()`ed and the time read out before we return: the
    # Dataset is finished with at that point, and leaving it open pins the HDF5 handle -- and
    # through it the caller's fsspec object -- for as long as the DataArray is alive.
    with xr.open_dataset(fobj, engine="h5netcdf", mask_and_scale=True) as ds:
        da = ds[variable].isel(time=0)
        da = da.sel(lat=slice(s - pad, n + pad), lon=slice(w - pad, e + pad)).load()
        t = pd.Timestamp(ds["time"].values[0]).tz_localize(None)
    return _to_grid(da, target_crs, transform, width, height, geom_proj, grid_cfg), t


# --------------------------------------------------------------------------- #
# Access backends: granule -> (gridded DataArray, time). Swappable via `access`.
# --------------------------------------------------------------------------- #
def _fetch_download(granule, variable, g: AoiGrid, pad, grid_cfg):
    """Stream the granule over HTTPS and subset client-side (the original path).

    The OPEN is here, inside the caller's `net.retry`, not outside it: `earthaccess.open`
    returns a handle bound to the fsspec session it was created with, so re-authenticating
    cannot heal a handle we already hold -- only re-opening can.
    """
    fobj = earthaccess.open([granule])[0]
    try:
        # CLOSED here: the handle holds an HTTPS connection and a one-thread executor for its
        # background block cache, and neither is released until the object is collected. A
        # multi-year AoI is hundreds of granules, and the abandoned sockets were still on the
        # process as CLOSE-WAIT hours later. `subset_and_reproject` materialises its window,
        # so nothing reads back through this afterwards.
        return subset_and_reproject(fobj, variable, g.search_bbox, pad, g.target_crs,
                                    g.transform, g.width, g.height, g.geom_proj, grid_cfg)
    finally:
        try:
            fobj.close()
        except Exception:      # noqa: BLE001 -- a close that fails must not mask the result
            pass


# The coordinate axes are a property of the COLLECTION, not of a granule, so they are read
# once (~216 KB) and shared. Read rather than computed: hard-coding MUR's -89.99/0.01 origin
# would silently mis-index the day the collection is re-gridded, and a wrong window is a wrong
# answer that still looks like data.
_DAP_AXES: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DAP_LOCK = threading.Lock()


def _dap_base(granule) -> str:
    """The granule's own OPeNDAP endpoint, from its catalogue metadata.

    Read from the granule rather than built from a template: the Hyrax path embeds the full
    collection title, and a hand-assembled URL would break on any collection but this one.
    """
    for r in (granule.get("umm", {}) or {}).get("RelatedUrls", []) or []:
        url = str(r.get("URL", ""))
        if r.get("Type") == "USE SERVICE API" and "opendap" in url.lower():
            return url.rstrip("/")
    raise RuntimeError(
        "this granule publishes no OPeNDAP service URL; use mur access='download'")


def _dap_get(url: str, what: str) -> bytes:
    """One authenticated DAP request -> raw bytes.

    Earthdata Login, the same credential everything else here uses: Hyrax 302s to URS and
    `earthaccess`'s session already carries the cookie jar.
    """
    resp = earthaccess.get_requests_https_session().get(
        url, timeout=(net.CONNECT_TIMEOUT_S, net.HTTP_TIMEOUT_S))
    if resp.status_code in (401, 403):
        # Deliberately a bare status number in the text: net.err_text strips path-like tokens,
        # and net.is_auth_error reads the number -- so this still costs exactly one credential
        # refresh before failing permanently, with the hint attached. The hint matters because
        # an un-approved EDL application and an expired token are the SAME status, and only one
        # of them is fixable by re-authenticating (net.retry says so on the second rejection).
        raise RuntimeError(
            f"OPeNDAP rejected {what} with status {resp.status_code}. If a credential refresh "
            "does not clear this, the Earthdata profile has most likely not authorized the "
            "OPeNDAP application yet -- approve it once under Applications -> Authorized Apps "
            "at urs.earthdata.nasa.gov. A fresh token cannot fix an unapproved application.")
    resp.raise_for_status()
    return resp.content


def _dap_axes(base: str, short_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The collection's (lat, lon) axes, fetched once per process."""
    with _DAP_LOCK:
        if short_name not in _DAP_AXES:
            raw = _dap_get(f"{base}.dap.nc4?dap4.ce={quote('/lat;/lon', safe='')}",
                           f"{short_name} coordinate axes")
            with xr.open_dataset(io.BytesIO(raw), engine="h5netcdf") as d:
                _DAP_AXES[short_name] = (np.asarray(d["lat"].values).copy(),
                                         np.asarray(d["lon"].values).copy())
            log.info("  OPeNDAP: read %s axes (%d lat x %d lon)", short_name,
                     *(a.size for a in _DAP_AXES[short_name]))
        return _DAP_AXES[short_name]


def _fetch_opendap(granule, variable, g: AoiGrid, pad, grid_cfg):
    """Ask Hyrax for ONLY the AOI window (a DAP4 hyperslab).

    MUR is a regular 0.01 deg grid, so the AOI's array indices are a lookup on the coordinate
    axes -- no geolocation search. The server returns the window itself rather than the ~27 MB
    of 8 MB-aligned HDF5 blocks a client-side subset drags over the wire for the same pixels.
    """
    base = _dap_base(granule)
    lats, lons = _dap_axes(base, str(granule["umm"].get("CollectionReference", {})
                                     .get("ShortName") or SHORT_NAME))
    w, s, e, n = g.search_bbox

    def span(axis, lo, hi):
        # EXACTLY the index range `.sel(slice(lo, hi))` selects on an ascending axis -- first
        # index >= lo, last index <= hi -- so the two backends reproject from the identical
        # source window. Widening this by a cell "for interpolation slack" is what it looks
        # like it should do, and it silently breaks equivalence: `analysed_sst` is NaN over
        # land, so one extra ring can pull a land NaN into an edge cell's bilinear
        # neighbourhood and drop it from `valid`. The slack is `pad` (0.05 deg = 5 cells),
        # applied by the caller to BOTH paths. Clamped, so an AOI running off the edge of the
        # grid shrinks to the edge rather than wrapping to the far side of the world.
        i0 = int(np.searchsorted(axis, lo, "left"))
        i1 = int(np.searchsorted(axis, hi, "right")) - 1
        return max(i0, 0), min(i1, axis.size - 1)

    i0, i1 = span(lats, s - pad, n + pad)
    j0, j1 = span(lons, w - pad, e + pad)
    if i1 < i0 or j1 < j0:
        raise RuntimeError(f"AOI window is empty in the {variable} grid "
                           f"(lat {i0}:{i1}, lon {j0}:{j1})")

    ce = (f"/{variable}[0][{i0}:{i1}][{j0}:{j1}];"
          f"/lat[{i0}:{i1}];/lon[{j0}:{j1}];/time[0]")
    raw = _dap_get(f"{base}.dap.nc4?dap4.ce={quote(ce, safe='')}",
                   f"{variable} window {i0}:{i1},{j0}:{j1}")

    # mask_and_scale on the RESPONSE: Hyrax preserves the packing and the CF attributes, so
    # the int16 -> Kelvin conversion is the same arithmetic the streamed path applies.
    with xr.open_dataset(io.BytesIO(raw), engine="h5netcdf", mask_and_scale=True) as ds:
        da = ds[variable].isel(time=0).load()
        t = pd.Timestamp(ds["time"].values[0]).tz_localize(None)
    return _to_grid(da, g.target_crs, g.transform, g.width, g.height,
                    g.geom_proj, grid_cfg), t


_ACCESS = {"download": _fetch_download, "opendap": _fetch_opendap}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Acquire MUR onto the pre-computed per-AoI grids (shared across products)."""
    grid_cfg = eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    auth.login("earthdata", eff["earthdata"])

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("mur")

    # Whether a sensor tree EXISTS is a fact about the project; whether THIS AoI has scenes in
    # it is per-AoI (the tree is <DIR>/aligned/<aoi>). So the preflight splits there.
    wanted = sorted({s for c in eff["ds"].values() for s in (c["overpass_sensors"] or ())})
    sdirs = overpass.sensor_dirs(eff["root"]) if wanted else {}
    absent = [p for p in wanted if not any(d.exists() for d in sdirs.get(p, []))]
    if wanted and absent == wanted:
        msg = (f"mur.overpass_sensors names {wanted}, but none has an aligned tree under "
               f"{eff['root']}: the sensors have not run in this output dir, so restricting "
               "MUR to overpass days would fetch NOTHING. Run the sensors first (a full "
               "pipeline run does -- mur depends on them), or drop `overpass_sensors`.")
        # Raise rather than warn: pipeline.run_pipeline records this as `mur: failed: ...` and
        # carries on with the other products, so it is maximally loud AND non-fatal. The
        # dry-run downgrade is load-bearing -- the sensors' own dry run writes nothing, so on a
        # fresh tree every --dry-run would otherwise report MUR failed.
        if dry_run:
            log.warning("  %s", msg)
        else:
            raise ValueError(msg)
    elif absent:
        log.warning("  overpass_sensors: %s has no aligned tree; filtering on the rest (%s)",
                    absent, [p for p in wanted if p not in absent])

    for name in names:
        g = grids[name]
        # MUR is a GLOBAL product, so nothing about the DATA varies by region -- but its
        # settings are still resolved per AoI (and `overpass_sensors` genuinely does vary),
        # so every product answers to the same contract.
        ds_cfg = eff["ds"][name]
        variable = ds_cfg["variable"]
        pad = float(ds_cfg["pad_deg"])
        fetch = _ACCESS[ds_cfg["access"]]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d @ %.0fm) | access=%s ===",
                 name, g.target_crs, g.width, g.height, g.resolution_m, ds_cfg["access"])
        auth.ensure_fresh("earthdata", eff["earthdata"])

        want = ds_cfg["overpass_sensors"]
        day_stamps = None
        if want:
            dirs = [d for p in want for d in sdirs.get(p, [])]
            day_stamps = overpass.days_with_scenes(dirs, name)
            if not day_stamps:
                # Two very different situations reach here, and saying which is the whole
                # value of the message: a tree that exists but holds no scene for this AoI is
                # a fact about the DATA; no tree at all is a stage that has not run (already
                # reported by the preflight, which only warns under --dry-run).
                why = ("the sensor trees exist, so this is a fact about the data, not a stage "
                       "that was skipped" if any(d.exists() for d in dirs)
                       else "no aligned tree for it under this output dir -- see above")
                log.warning("  %s: %s recorded no scene on ANY day -> MUR fetches nothing "
                            "here (%s).", name, "/".join(want), why)
                rep.note = "; ".join(filter(None, [rep.note, f"{name}: no overpass days"]))
                continue
            log.info("  overpass filter: %s flew on %d day(s)", "/".join(want), len(day_stamps))

        granules = net.retry(
            lambda: earthaccess.search_data(
                short_name=ds_cfg["short_name"], temporal=(start, end),
                bounding_box=tuple(g.search_bbox)),
            what=f"MUR search {name}")
        kept = _select_granules(granules, day_stamps)
        log.info("  %d daily MUR granule(s)%s", len(granules),
                 "" if day_stamps is None else f" -> {len(kept)} on overpass days")
        # AFTER the filter: `expected` is what the stage MEANT to produce (report.py), so
        # expecting the unfiltered count would make every filtered run read as having lost days.
        rep.expect(len(kept))
        if not kept:
            continue
        if dry_run:
            log.info("  [dry-run] would process %d day(s)%s", len(kept),
                     "" if day_stamps is None else f" of {len(granules)}")
            continue

        aoi_out = out_root / name
        ed = auth.refresher("earthdata", eff["earthdata"])
        for gi, granule in enumerate(kept, 1):
            # Proactive: a multi-year AoI is hundreds of granules on one login, so top the
            # credential up between them rather than discovering it died on granule 400.
            if gi % auth.CHECK_EVERY == 0:
                auth.ensure_fresh("earthdata", eff["earthdata"])
            try:
                # The WHOLE fetch sits inside the retry -- open included, for `download`. An
                # `earthaccess.open` handle is bound to the fsspec session it was created with,
                # so re-authenticating cannot heal a handle we already hold; only re-opening
                # can, and retrying the read alone would go through the same dead session
                # forever. `granule=granule` pins the loop variable (the CUDEM idiom).
                def _fetch(granule=granule):
                    return fetch(granule, variable, g, pad, grid_cfg)

                da, t = net.retry(_fetch, what=f"MUR granule {gi}", refresh=ed)
            except Exception as exc:
                # A failed download and a day that genuinely has no data used to look
                # identical: one warning, no tally, and `Done.` all the same.
                log.warning("    [%d/%d] FAILED (%s)", gi, len(kept), exc)
                rep.fail(f"{name} granule {gi}", exc)
                continue

            # The file's own time is the authority for the stem; this only reports that the
            # catalogue metadata the filter SELECTED on disagreed with it, which would mean
            # `_granule_day_stamp` is reading the wrong field for this collection. A visible
            # warning rather than a silent off-by-one -- the output stays correct either way.
            if day_stamps is not None and naming.day_stamp(t) not in day_stamps:
                log.warning("    [%d/%d] catalogue day and file day disagree (%s); the "
                            "overpass filter selected on catalogue metadata",
                            gi, len(kept), naming.day_stamp(t))

            stem = naming.day_stem(name, t)
            if store.done(aoi_out / f"{stem}.nc", store.REQUIRED_VARS["MUR"],
                          shape=(g.height, g.width), overwrite=overwrite):
                log.info("  [%d/%d] %s already processed, skipping", gi, len(kept),
                         naming.day_stamp(t))
                rep.skip()
                continue

            ds = xr.Dataset({"sst": da})
            ds["sst"].attrs["units"] = "degC" if grid_cfg.get("to_celsius", False) else "K"
            ds["valid"] = np.isfinite(ds["sst"]).astype("uint8")
            ds["valid"].attrs["long_name"] = "finite MUR SST (water)"
            ds = ds.expand_dims(time=[t])
            src = f"GHRSST {ds_cfg['short_name']}"
            ds.attrs.update(aoi_id=name, source=src,
                            processing="subset + bilinear upsample to AOI grid",
                            **provenance.stamp(eff))
            try:
                out = store.write_output(ds, aoi_out, stem, fmt)
            except (OSError, RuntimeError) as exc:   # this day only -- see cmems.run
                log.warning("  [%d/%d] WRITE FAILED %s (%s)", gi, len(kept), stem, exc)
                rep.fail(f"{name} {stem}", f"write failed: {exc}")
                continue
            log.info("  [%d/%d] wrote %s", gi, len(kept), out)
            rep.wrote(source=src)

    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _overpass_sensors(raw) -> tuple[str, ...] | None:
    """The sensor prefixes MUR restricts its days to; None = every day (the default).

    A bare string is one sensor. The names were checked against the loaded sensors at config
    load (config._overpass_sensor_lists_are_valid), so anything here is real.
    """
    if raw is None:
        return None
    names = [raw] if isinstance(raw, str) else list(raw)
    return tuple(str(n) for n in names) or None


def _access(raw) -> str:
    """The fetch backend for this AoI. Fails at CONFIG time, not on granule 400.

    `config._option_keys_are_known` validates option NAMES; nothing validates a value, so an
    unknown backend would otherwise surface as a KeyError deep inside the granule loop after
    the search has already run.
    """
    name = str(raw)
    if name not in _ACCESS:
        raise ValueError(f"mur access={name!r} is not a known backend; "
                         f"choose from {sorted(_ACCESS)}.")
    return name


def _ds_cfg(opts) -> dict:
    """One AoI's MUR settings. MUR is a global product, so nothing about the DATA varies by
    region -- but `overpass_sensors` does (which sensors are worth restricting days to here),
    so it is resolved region-then-global like every other option."""
    return {
        "short_name": _opt(opts, "short_name", SHORT_NAME),
        "variable": _opt(opts, "variable", DEFAULT_VARIABLE),
        "pad_deg": float(_opt(opts, "pad_deg", DEFAULT_PAD_DEG)),
        "access": _access(_opt(opts, "access", DEFAULT_ACCESS)),
        "overpass_sensors": _overpass_sensors(_opt(opts, "overpass_sensors", None)),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.mur)
    if opts is None:
        raise ValueError("mur is not a selected product in this config")
    if project.auth.earthdata is None:            # guaranteed by config validation
        raise ValueError("mur requires an auth.earthdata block")

    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.mur))
               for a in project.all_areas},
        "grid": grid_cfg,
        # The output root, not just MUR's own tree: the overpass filter reads the SENSORS'
        # aligned dirs, which hang off the same root.
        "root": Path(project.output_dir),
        "out_dir": Path(project.output_dir) / "MUR" / "aligned",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "earthdata": {"auth_strategy": project.auth.earthdata.auth_strategy},
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire MUR for a validated Project. Entry point for pipeline.py.

    Parameters
    ----------
    project      validated config (coastal_sst_data.config.Project)
    grids        pre-computed {aoi_name: AoiGrid}; if None, computed here so all
                 products share one grid computation.
    aois         restrict to these AoI name(s); default all
    dry_run      search only, no download/write
    overwrite    reprocess days even if the aligned file exists
    """
    eff = _build_eff(project)
    # Credentials expire and runs are long: apply this project's refresh policy before any
    # network call. acquire() is the one entry point every invocation path goes through.
    auth.configure(project.auth)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(acquire, "coastal_sst_data MUR L4 SST acquisition.")


if __name__ == "__main__":
    main()