#!/usr/bin/env python3
"""
coastal_sst_data -- MODIS L2P Sea-Surface Temperature acquisition (Terra + Aqua).

Loads GHRSST MODIS L2P skin SST (NASA OB.DAAC via earthaccess) and grids it onto the shared
AoiGrid. MODIS L2P is a SWATH product (2D curvilinear lat/lon), so it is regridded with
pyresample nearest-neighbour (`resample_nearest`) -- NOT rioxarray reproject. Nearest is
deliberate: it preserves the actual observed MODIS values (block-constant on the fine grid)
rather than inventing intermediate ones a 1 km sensor never measured.

STACKED PLATFORMS (D10). Terra and Aqua are two instruments on two spacecraft in two orbits;
they never observe the same overpass, so each is acquired into its own tree and contributes
its own cube channel set (`modis_sst_terra`, `modis_sst_aqua`) rather than one falling back
to the other:

    <output_dir>/MODIS/<platform>/aligned/<aoi>/<aoi>_<YYYYMMDDTHHMMSS>.nc

Keeping them apart is not cosmetic. NASA stopped maintaining both orbits (Terra's last
inclination maneuver Feb 2020, Aqua's Mar 2021) and the two equator crossings have since
drifted in OPPOSITE directions -- Terra earlier, Aqua later -- so a single merged channel
would blend two diverging times of day into one number.

TIME OF DAY. Nominal local crossings are Terra 10:30 / 22:30 and Aqua 13:30 / 01:30, but the
drift above means those are the START of the record, not a property of it: by end of mission
Terra's morning crossing had reached ~09:00 and Aqua's afternoon ~15:50. So the day/night
selection is made on COMPUTED LOCAL MEAN SOLAR TIME (`solar_hour`), not on a fixed clock and
not on the granule filename's `-D-`/`-N-` token alone -- see `select_by_time_of_day`.

Above ~32 deg latitude consecutive same-night orbits overlap, so a platform typically sees an
AoI twice a night -- once near nadir, once near the swath edge, each covering a different part
of it. Both are kept; the assembler mosaics them (`mosaic_same_day=True` on the spec).

Access backends (configurable via `access`):
  * harmony (default) -- server-side bbox + variable subsetting, so only the AoI crosses the
    wire. A full L2P granule is ~15-25 MB of a ~2030x1354 swath with ~15 variables, of which
    this module reads four; a multi-year night-time series over both platforms is tens of
    thousands of granules, which is the whole reason this backend exists.
  * download -- earthaccess.download the full granule, crop in memory. No extra dependency,
    and the only option for a caller that needs NATIVE swath indices (see processes.modis_ref).

Output variables: sst (K|degC) and valid (uint8). MODIS L2P arrives quality-filtered, with no
water or cloud layer to recompute from, so this product publishes no `modis_cloud` channel.

Usage:
    python -m coastal_sst_data.processes.modis --config config.yaml
    python -m coastal_sst_data.processes.modis --config config.yaml --dry-run
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import earthaccess
import rioxarray  # noqa: F401  (registers the .rio accessor)

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import auth, entry, naming, net, products, provenance, report, store

log = logging.getLogger(__name__)

# --- MODIS L2P product constants (overridable via the modis product options) - #
SOURCE = "modis"

# Platform TAG -> GHRSST collection short name. The tag is the provenance identity: it names
# the output tree (`MODIS/<tag>/aligned`) and the cube channel suffix (`modis_sst_<tag>`).
PLATFORMS = {
    "terra": "MODIS_T-JPL-L2P-v2019.0",
    "aqua": "MODIS_A-JPL-L2P-v2019.0",
}
DEFAULT_PLATFORMS = ("terra", "aqua")

DEFAULT_VARIABLE = "sea_surface_temperature"  # Kelvin (xarray auto-unscales)
QUALITY_VAR = "quality_level"                 # GHRSST: >=4 acceptable, 5 best
GEO_VARS = ("lat", "lon")                     # the 2D swath geolocation
DEFAULT_QUALITY_MIN = 4
DEFAULT_RADIUS_M = 1500.0                     # pyresample search radius (~1 km px)

# Day/night selection. `night_solar_hours` is a LOCAL MEAN SOLAR time window and it WRAPS
# through midnight, which is the only shape that can describe a night. The default spans both
# platforms' night crossings across the whole 2000-2026 record including the orbit drift
# (Terra 22:30 -> ~21:00, Aqua 01:30 -> ~03:50); `day` is its complement.
DEFAULT_TIME_OF_DAY = "night"
DEFAULT_NIGHT_SOLAR_HOURS = (19.0, 5.0)
TIMES_OF_DAY = ("night", "day", "both")


# --------------------------------------------------------------------------- #
# Time of day: local solar hour, and selecting granules by it
# --------------------------------------------------------------------------- #
def solar_hour(t: datetime, lon: float) -> float:
    """Local MEAN SOLAR hour at longitude `lon` for a UTC instant, in [0, 24).

    The inverse of the conversion `processes.met.reference_time_utc` already applies to turn
    a solar reference time into UTC (`utc = solar - lon/15`), so the two stages agree on what
    "local time" means by construction rather than by coincidence.

    Mean solar, not apparent: no equation-of-time term. That is worth up to ~16 minutes, which
    is an order of magnitude below the multi-hour windows this feeds and two orders below the
    orbit drift it exists to absorb.
    """
    utc_h = t.hour + t.minute / 60.0 + t.second / 3600.0
    return (utc_h + lon / 15.0) % 24.0


def _in_window(h: float, lo: float, hi: float) -> bool:
    """Is hour `h` inside [lo, hi), where lo > hi means the window WRAPS through midnight."""
    return (lo <= h < hi) if lo <= hi else (h >= lo or h < hi)


def _granule_time(granule) -> datetime:
    t = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")


def _day_night(granule) -> str | None:
    """'day' | 'night' from the GHRSST filename token, or None when it is absent.

    The `-D-`/`-N-` token in a GHRSST native-id is OBPG's own classification of which side of
    the orbit the granule is on, so it is authoritative about the granule. It is NOT
    authoritative about the local hour, which is what drifts -- hence it is used to CONFIRM the
    solar-time decision rather than to make it.
    """
    nid = granule["meta"].get("native-id", "")
    if "-N-" in nid:
        return "night"
    if "-D-" in nid:
        return "day"
    return None


def select_by_time_of_day(granules, *, time_of_day: str, night_hours, lon: float):
    """Filter granules to a time of day. Returns [(granule, acq_time), ...] sorted by time.

    The decision is made on LOCAL MEAN SOLAR HOUR at the AoI, because the thing being selected
    for -- "this is a night-time observation" -- is a fact about the sun, and MODIS's UTC
    overpass time for a fixed place moved by hours over the mission as the orbits drifted. A
    fixed UTC window, or a hard-coded 22:30/01:30, silently loses granules at the ends of the
    record.

    The filename token is then required to AGREE where it exists. It almost always does; a
    disagreement means the granule straddles the terminator (or the window has been set to cut
    through one), and dropping it keeps the channel honest about what "night" meant.
    """
    kept = []
    for gr in granules:
        t = _granule_time(gr)
        if time_of_day != "both":
            want_night = time_of_day == "night"
            if _in_window(solar_hour(t, lon), *night_hours) != want_night:
                continue
            token = _day_night(gr)
            if token is not None and token != time_of_day:
                log.debug("  %s: solar hour says %s but the granule is tagged %s; dropping",
                          gr["meta"].get("native-id", "?"), time_of_day, token)
                continue
        kept.append((gr, t))
    return sorted(kept, key=lambda x: x[1])


# --------------------------------------------------------------------------- #
# Fetch backends: granule -> local NetCDF path. Swappable via `access`.
# --------------------------------------------------------------------------- #
def _fetch_download(granule, bbox, tmp_dir: Path, *, variables=None) -> Path:
    """Download the FULL granule into a SCRATCH dir the caller will delete.

    `variables` is accepted and ignored: this backend has no way to ask for less than the whole
    file. That is the trade -- no extra dependency, no service round-trip, every byte.

    `earthaccess.download` skips a file that already exists BY NAME. That is a caching
    feature and, combined with the old lifecycle (unlink inside the try, so a killed run
    left the partial granule behind), it was a corruption vector: the truncated granule
    survived, the next run saw the name already present, skipped the download, and handed
    the truncated file straight to read_swath. Hence the per-granule scratch dir, deleted
    in a `finally` -- a granule directory that outlives its attempt cannot be reused.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # `download` re-derives its session from `earthaccess.__store__` on each call, and a
    # refresh rebuilds that store -- so unlike an already-open handle, a retried download
    # genuinely benefits. No settings needed: the strategy was recorded when the stage
    # logged in (see auth._strategy).
    paths = net.retry(lambda: earthaccess.download([granule], local_path=str(tmp_dir)),
                      what="MODIS granule download",
                      refresh=auth.refresher("earthdata"))
    if not paths:
        raise RuntimeError("earthaccess.download returned no path")
    return Path(paths[0])


def _fetch_harmony(granule, bbox, tmp_dir: Path, *, variables=None) -> Path:
    """Server-side subset via Harmony (PO.DAAC's l2ss-py), so only the AoI crosses the wire.

    Two multipliers, and both matter at this product's scale: the BBOX trims the swath to the
    AoI, and `variables` drops the ~11 L2P layers this module never reads (`sses_bias`,
    `l2p_flags`, `sst_dtime`, `wind_speed`, ...). The variable list is derived from what
    `read_swath` actually opens, so the request and the reader cannot drift apart.

    Both CMR concept-ids come off the granule `earthaccess.search_data` already returned --
    there is no second catalogue round-trip to address the granule to Harmony.

    A single-granule request often completes without Harmony creating a job at all, in which
    case `submit` returns the result JSON instead of a job id; `download_all` accepts either,
    so the two paths do not need distinguishing here.
    """
    from harmony import BBox, Client, Collection, Request

    tmp_dir.mkdir(parents=True, exist_ok=True)
    meta = granule["meta"]
    collection_id = meta.get("collection-concept-id")
    granule_id = meta.get("concept-id")
    if not collection_id or not granule_id:
        raise RuntimeError(
            "granule metadata carries no CMR concept-ids, so Harmony cannot address it; "
            "use access='download' for this collection")

    def _subset() -> list[str]:
        # The client is built INSIDE the retry, like ECOSTRESS's `earthaccess.open`: its
        # requests session carries the credential it was created with, so re-authenticating
        # cannot heal a client already held -- only rebuilding it can.
        #
        # should_validate_auth=False because auth.login already verified this credential
        # against EDL; leaving it on would add a round-trip per granule to re-ask a question
        # the preflight answered.
        client = Client(should_validate_auth=False, **auth.earthdata_client_auth())
        req = Request(collection=Collection(id=collection_id),
                      granule_id=[granule_id],
                      spatial=BBox(*bbox),
                      **({"variables": list(variables)} if variables else {}))
        if not req.is_valid():
            raise RuntimeError("invalid Harmony request: "
                               + ", ".join(req.error_messages()))
        job = client.submit(req)
        # `download_all` is a GENERATOR and it SWALLOWS a failed job -- it prints to stderr and
        # yields nothing. Materialising it here is what turns "the job failed" into an
        # exception the retry and the ProductReport can both see, rather than a granule that
        # silently produced no file.
        futures = list(client.download_all(job, directory=str(tmp_dir), overwrite=True))
        return [f.result() for f in futures]

    paths = net.retry(_subset, what="MODIS Harmony subset",
                      refresh=auth.refresher("earthdata"))
    if not paths:
        raise RuntimeError(
            "Harmony returned no file for this granule (the job failed, or the subset was "
            "empty); re-run with access='download' to see the unsubsetted granule")
    return Path(paths[0])


_ACCESS = {"download": _fetch_download, "harmony": _fetch_harmony}


def harmony_variables(variable: str) -> tuple[str, ...]:
    """The L2P variables a Harmony subset must carry -- exactly what `read_swath` opens."""
    return (variable, QUALITY_VAR) + GEO_VARS


# --------------------------------------------------------------------------- #
# Swath read + resample onto the shared grid
# --------------------------------------------------------------------------- #
def read_swath(path, variable, quality_min, to_celsius):
    """Read a MODIS L2P granule -> (sst, lat, lon) 2D swath arrays (NaN where bad)."""
    ds = xr.open_dataset(path, engine="netcdf4")
    sst = ds[variable].values.squeeze().astype("float32")     # Kelvin
    qual = ds[QUALITY_VAR].values.squeeze().astype("int16")
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()
    sst[(qual < quality_min) | ~np.isfinite(sst)] = np.nan
    if to_celsius:
        sst = sst - 273.15
    return sst, lat, lon


def resample_to_grid(sst, lat, lon, g: AoiGrid, radius_m, footprint=None):
    """Nearest-neighbour resample the swath onto the AoI grid (sst, and footprint)."""
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest

    lon = ((lon + 180.0) % 360.0) - 180.0                      # normalize longitudes
    swath = SwathDefinition(lons=lon, lats=lat)
    area = g.to_area_def()
    sst_g = resample_nearest(swath, sst, area, radius_of_influence=radius_m,
                             fill_value=np.nan)
    fp_g = None
    if footprint is not None:
        fp_g = resample_nearest(swath, footprint, area, radius_of_influence=radius_m,
                                fill_value=-1)
    return sst_g, fp_g


def _scene_dataset(sst_g, fp_g, g: AoiGrid, acq_time, aoi_id, to_celsius,
                   short_name=None, platform=None, solar_h=None,
                   day_night=None) -> xr.Dataset:
    xs, ys = g.xy_centers()
    data = {
        "sst": (("y", "x"), sst_g.astype("float32")),
        "valid": (("y", "x"), np.isfinite(sst_g).astype("uint8")),
    }
    if fp_g is not None:
        data["footprint_id"] = (("y", "x"), fp_g.astype("int32"))
    ds = xr.Dataset(data, coords={"y": ys, "x": xs})
    ds["sst"].attrs["units"] = "degC" if to_celsius else "K"
    ds["valid"].attrs["long_name"] = "finite, quality-good MODIS SST"
    if fp_g is not None:
        ds["footprint_id"].attrs["long_name"] = (
            "MODIS swath pixel index (for footprint-median matchups); -1 = none")
    ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    ds = ds.rio.write_crs(g.target_crs)
    # The short_name we actually SEARCHED, not the module default: configure Aqua and the
    # old constant still stamped every file "Terra", so the cube misnamed its own sensor.
    ds.attrs.update(aoi_id=aoi_id, source=f"GHRSST {short_name}",
                    processing="swath -> nearest resample onto AoI grid")
    # The overpass's LOCAL time, recorded per scene. The cube only carries the UTC hour, and
    # the whole point of a multi-decade MODIS series is that the local hour behind that UTC
    # hour MOVED -- so the drift has to be auditable from the granules themselves.
    if platform is not None:
        ds.attrs["platform"] = platform
    if solar_h is not None:
        ds.attrs["solar_hour"] = round(float(solar_h), 4)
    if day_night is not None:
        ds.attrs["day_night"] = day_night
    return ds


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    grid_cfg = eff["grid"]
    modis_root, fmt, overwrite = eff["modis_root"], eff["fmt"], eff["overwrite"]
    to_celsius = grid_cfg.get("to_celsius", False)
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    auth.login("earthdata", eff["earthdata"])

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("modis")

    for name in names:
        g = grids[name]
        # MODIS is a GLOBAL product, so it has no region-varying options -- but its settings
        # are still resolved per AoI, so every product answers to the same contract.
        ds_cfg = eff["ds"][name]
        variable, quality_min = ds_cfg["variable"], ds_cfg["quality_min"]
        radius, access = ds_cfg["regrid_radius_m"], ds_cfg["access"]
        time_of_day, night_hours = ds_cfg["time_of_day"], ds_cfg["night_solar_hours"]
        fetch = _ACCESS[access]
        want_vars = harmony_variables(variable)
        # The AoI's centre longitude drives the solar-time conversion. `search_bbox` is
        # (W, S, E, N) in EPSG:4326, which is the one longitude on the grid that is already
        # in degrees -- the projected grid's own x is metres.
        lon_c = 0.5 * (g.search_bbox[0] + g.search_bbox[2])

        platforms = [p for p in ds_cfg["platforms"]
                     if only_source is None or p == only_source]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d) | platforms=%s time_of_day=%s access=%s ===",
                 name, g.target_crs, g.width, g.height, platforms, time_of_day, access)
        auth.ensure_fresh("earthdata", eff["earthdata"])

        for tag in platforms:
            short_name = ds_cfg["short_name"] or PLATFORMS[tag]

            granules = net.retry(
                lambda sn=short_name: earthaccess.search_data(
                    short_name=sn, temporal=(start, end),
                    bounding_box=tuple(g.search_bbox)),
                what=f"MODIS search {name} {tag}",
                refresh=auth.refresher("earthdata", eff["earthdata"]))
            kept = select_by_time_of_day(granules, time_of_day=time_of_day,
                                         night_hours=night_hours, lon=lon_c)
            log.info("  [%s] %d granule(s) over AOI -> %d after the %s filter",
                     tag, len(granules), len(kept), time_of_day)
            if not kept:
                continue
            if dry_run:
                for gr, t in kept[:5]:
                    log.info("    [dry-run] %s solar_hour=%.2f (%s)", naming.time_stamp(t),
                             solar_hour(t, lon_c), _day_night(gr) or "untagged")
                log.info("  [%s] [dry-run] would process %d MODIS granule(s)", tag, len(kept))
                continue

            aoi_out = modis_root / tag / "aligned" / name
            # Scratch lives UNDER the platform tag, never beside it: a directory that is a
            # direct child of MODIS/ is a candidate source tag to the assembler.
            tmp_root = modis_root / tag / "_tmp"

            for gi, (gr, t) in enumerate(kept, 1):
                # Proactive: top the credential up between granules rather than discovering it
                # died four hours in.
                if gi % auth.CHECK_EVERY == 0:
                    auth.ensure_fresh("earthdata", eff["earthdata"])
                tstr = naming.time_stamp(t)
                stem = naming.time_stem(name, t)
                if store.done(aoi_out / f"{stem}.nc", store.REQUIRED_VARS["MODIS"],
                              shape=(g.height, g.width), overwrite=overwrite):
                    log.info("  [%s] %s already processed, skipping", tag, tstr)
                    rep.skip()
                    continue
                # One scratch dir per granule, removed in `finally`. A partial download can
                # therefore never survive to be mistaken for a complete one by the next run.
                # The name carries WHO is downloading as well as WHAT: keyed on the granule
                # alone, two runs over overlapping dates share the directory, and each one's
                # `finally` deletes the other's half-fetched granule.
                gran_tmp = tmp_root / f"g_{name}_{tstr}_{store.unique_suffix()}"
                try:
                    path = fetch(gr, g.search_bbox, gran_tmp, variables=want_vars)
                    sst, lat, lon = read_swath(path, variable, quality_min, to_celsius)
                    log.debug("  [%s] %s: swath %s", tag, tstr, sst.shape)
                    sst_g, _ = resample_to_grid(sst, lat, lon, g, radius)
                except Exception as exc:
                    log.warning("  [%s] FAILED %s (%s)", tag, tstr, exc)
                    rep.fail(f"{name} {tag} {tstr}", exc)
                    continue
                finally:
                    shutil.rmtree(gran_tmp, ignore_errors=True)
                if not np.isfinite(sst_g).any():
                    log.info("  [%s] %s: no valid MODIS pixels over AOI, skipping", tag, tstr)
                    continue
                ds = _scene_dataset(sst_g, None, g, t, name, to_celsius,
                                    short_name=short_name, platform=tag,
                                    solar_h=solar_hour(t, lon_c),
                                    day_night=_day_night(gr))
                ds.attrs.update(**provenance.stamp(eff))
                try:
                    out = store.write_output(ds, aoi_out, stem, fmt)
                except (OSError, RuntimeError) as exc:   # this scene only -- see cmems.run
                    log.warning("  [%s] WRITE FAILED %s (%s)", tag, stem, exc)
                    rep.fail(f"{name} {tag} {stem}", f"write failed: {exc}")
                    continue
                log.info("  [%s] wrote %s", tag, out)
                rep.wrote(source=f"GHRSST {short_name}")
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def resolve_time_of_day(opts, default: str = DEFAULT_TIME_OF_DAY) -> str:
    """`time_of_day`, honouring the `daytime_only` boolean it replaced.

    The old option could only say "day, or don't filter"; the new one has to be able to say
    "night", which is what a standalone MODIS series is usually after. Accepting the old key
    keeps existing configs running rather than silently changing which granules they select --
    the one failure mode that would look exactly like the archive having changed.
    """
    tod = _opt(opts, "time_of_day", None)
    legacy = _opt(opts, "daytime_only", None)
    if tod is None and legacy is not None:
        tod = "day" if bool(legacy) else "both"
        log.warning("`daytime_only: %s` is deprecated; write `time_of_day: %s`", legacy, tod)
    tod = default if tod is None else str(tod)
    if tod not in TIMES_OF_DAY:
        raise ValueError(f"modis time_of_day {tod!r} not recognized; "
                         f"choose from {list(TIMES_OF_DAY)}.")
    return tod


def resolve_night_hours(opts) -> tuple[float, float]:
    """The local-solar-time window that means "night", as (start, end), wrapping midnight."""
    raw = _opt(opts, "night_solar_hours", None)
    if raw is None:
        return DEFAULT_NIGHT_SOLAR_HOURS
    try:
        lo, hi = (float(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("modis night_solar_hours must be two numbers, "
                         f"e.g. [19, 5]; got {raw!r}") from exc
    if not (0.0 <= lo < 24.0 and 0.0 <= hi < 24.0):
        raise ValueError(f"modis night_solar_hours must lie in [0, 24); got {raw!r}")
    if lo == hi:
        raise ValueError(f"modis night_solar_hours {raw!r} is an empty window")
    return lo, hi


def _access(opts, default: str) -> str:
    access = _opt(opts, "access", default)
    if access not in _ACCESS:
        raise ValueError(f"modis access {access!r} not recognized; "
                         f"choose from {sorted(_ACCESS)}.")
    return access


def _ds_cfg(opts) -> dict:
    """One AoI's MODIS settings. MODIS is global, so nothing here is region-overridable."""
    raw_platforms = _opt(opts, "platforms", None)
    if raw_platforms is None:
        platforms = list(DEFAULT_PLATFORMS)
    elif isinstance(raw_platforms, str):
        platforms = [raw_platforms]
    else:
        platforms = [str(p) for p in raw_platforms]
    return {
        "platforms": platforms,
        # Unset -> each platform's own collection. Setting it PINS every platform to one
        # collection, which is only meaningful when a single platform is configured.
        "short_name": _opt(opts, "short_name", None),
        "variable": _opt(opts, "variable", DEFAULT_VARIABLE),
        "quality_min": int(_opt(opts, "quality_min", DEFAULT_QUALITY_MIN)),
        "regrid_radius_m": float(_opt(opts, "regrid_radius_m", DEFAULT_RADIUS_M)),
        "access": _access(opts, "harmony"),
        "time_of_day": resolve_time_of_day(opts),
        "night_solar_hours": resolve_night_hours(opts),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.modis)
    if opts is None:
        raise ValueError("modis is not a selected product in this config")
    if project.auth.earthdata is None:            # guaranteed by config validation
        raise ValueError("modis requires an auth.earthdata block")

    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    root = Path(project.output_dir)
    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.modis))
               for a in project.all_areas},
        "grid": grid_cfg,
        # Per-platform trees hang under here: MODIS/<tag>/aligned/<aoi> (mirrors eco_root).
        "modis_root": root / "MODIS",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "earthdata": {"auth_strategy": project.auth.earthdata.auth_strategy},
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None) -> None:
    """Acquire MODIS for a validated Project. Entry point for pipeline.py.

    Every configured platform is acquired and STACKED (D10); the pipeline dispatches the
    shared module once and it fans out over the `platforms` list internally. `source` narrows
    to one platform tag (unset on a pipeline run; available for a direct CLI/test run) --
    there is no fallback, a platform with no coverage simply contributes no channel.
    """
    eff = _build_eff(project)
    # Credentials expire and runs are long: apply this project's refresh policy before any
    # network call. acquire() is the one entry point every invocation path goes through.
    auth.configure(project.auth)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    # A platform tag no code recognises would silently drop a satellite -- fail loudly, like
    # ecostress does for an unknown version (config also checks this at load time).
    known = set(products.spec(DataProduct.modis).known_sources)
    bad = sorted(f"{n}:{p}" for n, c in eff["ds"].items() for p in c["platforms"]
                 if p not in known)
    if bad:
        raise ValueError(f"modis platform not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(known)}.")
    return run(eff, grids, aois, dry_run, only_source=source)


def main():
    entry.process_main(acquire, "coastal_sst_data MODIS L2P SST acquisition (Terra + Aqua).")


if __name__ == "__main__":
    main()
