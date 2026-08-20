#!/usr/bin/env python3
"""
OCEANSR — ECOSTRESS L2T LSTE acquisition (collection versions STACKED, D10).

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; ECOSTRESS-specific settings come from `products.ecostress`.

DISTINCT-DATA collection VERSIONS, STACKED (D10) -- the config lists the collection
`versions` to acquire (source tags `v002`, `v003`), and each is fetched INDEPENDENTLY into
its own tree. There is no fallback: v002 and v003 have asymmetric temporal coverage (v002
starts earlier, v003 reaches the present), so the user stacks the versions needed to span
their range, and the datacube ships one channel-set per version (`eco_sst_v002`,
`eco_sst_v003`). A granule present in both collections lands under each rather than colliding:

    data/ECOSTRESS/<version>/aligned/<aoi>/<aoi>_<YYYYMMDDThhmmss>_<tile>.nc

ONE FILE PER TILE, not per overpass. ECO_L2T is a TILED product: an AoI wider than a 110 km
tile is covered by several granules that share an acquisition time EXACTLY and differ only by
MGRS tile. Naming the output by time alone gave them all one filename, so the first tile
written made `store.done` report every other tile of that overpass as already processed and
the AoI kept one tile's footprint with nodata across the rest. The tile is therefore part of
the name, which also lets the assembler do what it was already configured to do: `scene_index`
groups a day's files, and `load_clearest_overpass` unions them because ECOSTRESS's SensorSpec
sets `mosaic_same_day=True`. Nothing is mosaicked here.

For each (AOI, version) it searches ECO_L2T_LSTE by bbox + dates (Earthdata wants the bare
collection number, so the tag's leading `v` is stripped at the search boundary), then for
every overpass *streams a windowed read* of the SST + cloud/water/QC COGs (HTTP range
requests via earthaccess.open -- no full-tile download to disk), reprojects the AOI window
onto the AOI grid, clips to the AOI polygon, and writes one aligned file per overpass. A
later stage bins these to a daily cube.

Because only the AOI window is read and only the small aligned outputs are kept, there is no
multi-GB raw tile cache.

Usage (run from the OCEANSR project root):
    python src/acquire_ecostress.py --config configs/config.yaml
    python src/acquire_ecostress.py --config configs/config.yaml --dry-run
    python src/acquire_ecostress.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_ecostress.py --config configs/config.yaml --list-layers
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

import earthaccess
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from shapely.geometry import box as shp_box

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, read_cog_window, select_aois
from .. import auth, entry, naming, net, products, provenance, report, store

log = logging.getLogger(__name__)

# --- ECOSTRESS product constants -------------------------------------------- #
# These describe the ECO_L2T_LSTE product itself (not user choices), so they
# live here as defaults. Any of them can be overridden per-config via the
# ecostress product options block.
SOURCE = "ecostress"
SHORT_NAME = "ECO_L2T_LSTE"
# STACKED collection VERSIONS (D10), as source TAGS. Each tag is the provenance identity: it
# names the per-version output tree (`ECOSTRESS/<tag>/aligned`) and the cube channel suffix
# (`eco_sst_<tag>`). Earthdata's search wants the bare collection NUMBER, so the leading `v`
# is stripped only at the search boundary (`_collection_version`). Default preserves the
# historical single-collection behaviour (v002); stack v003 to reach the present.
DEFAULT_VERSIONS = ("v002",)


def _collection_version(tag: str) -> str:
    """The Earthdata collection number for a version TAG: 'v002' -> '002'."""
    return tag[1:] if tag.startswith("v") else tag
# role -> COG asset suffix (..._<suffix>.tif). sst uses the LST band (LST ~= SST
# over water); cloud/water/quality are categorical masks.
LAYERS = {"sst": "LST", "lst": "LST", "cloud": "cloud", "water": "water", "quality": "QC"}
CATEGORICAL = ["cloud", "water", "quality"]
# The layers a granule CANNOT be written without: they are what `valid` is built from, and
# a granule lacking one reads downstream as a scene with nothing valid in it -- a silent
# total cloud-out rather than a visible failure. (`lst`/`quality` are optional extras.)
CORE_ROLES = ("sst", "water", "cloud")


def expected_vars(ds_cfg: dict) -> tuple[str, ...]:
    """The variables a COMPLETE aligned granule carries under this config's `layers`.

    Derived from config rather than hardcoded, because `layers` is overridable: demanding a
    layer the user never asked for would make every granule look incomplete and re-fetch on
    every run, forever.
    """
    roles = set(ds_cfg.get("layers", LAYERS))
    core = tuple(r for r in CORE_ROLES if r in roles)
    return core + (("valid",) if set(CORE_ROLES) <= roles else ())


# The per-AoI target CRS + grid now come from coastal_sst_data.grid (the single
# source of truth shared by every product), passed in as AoiGrid objects.


# --------------------------------------------------------------------------- #
# Earthdata search / download
# --------------------------------------------------------------------------- #
def login(strategy: str):
    """Authenticate to Earthdata AND record when, via `auth.login`.

    Kept as a named seam because the tests replace it, but it no longer calls
    `earthaccess.login` directly: a login nobody timestamped cannot be proactively
    refreshed, which is exactly what a multi-hour run needs.
    """
    auth.login("earthdata", {"auth_strategy": strategy})


def search_granules(ds_cfg: dict, bbox, start: str, end: str):
    results = net.retry(
        lambda: earthaccess.search_data(
            short_name=ds_cfg["short_name"],
            version=ds_cfg["version"],
            temporal=(start, end),
            bounding_box=tuple(bbox),
        ),
        what=f"ECOSTRESS search {ds_cfg['short_name']}",
        refresh=auth.refresher("earthdata"))
    log.info("  found %d granule(s)", len(results))
    return results


def granule_name(granule) -> str:
    """Clean granule id (strips the trailing _<LAYER>.tif), e.g.
    ECOv002_L2T_LSTE_25520_009_10UEU_20230105T123623_0710_01."""
    try:
        fn = granule.data_links()[0].split("/")[-1]
        return re.sub(r"_[A-Za-z0-9]+\.tif$", "", fn)
    except Exception:
        return "<granule>"


def filter_links_for_granule(granule, layers: dict) -> dict:
    """{role: url} for the COG assets we want from one granule."""
    links = granule.data_links()
    out, missing = {}, []
    for role, suffix in layers.items():
        match = next((u for u in links if u.endswith(f"_{suffix}.tif")), None)
        if match:
            out[role] = match
        else:
            missing.append(f"{role} (_{suffix}.tif)")
    if missing:
        log.warning("    %s: requested layer(s) not in granule: %s",
                    granule_name(granule), ", ".join(missing))
    return out


def parse_acq_time(filename: str) -> Optional[datetime]:
    """The overpass stamp in a granule id (..._20230715T210043_...), or None."""
    return naming.parse_time(filename)


# The MGRS tile, read from its position RELATIVE TO the acquisition stamp rather than by
# counting underscores from the left: the id's leading fields differ between collection
# versions, and a positional split that silently returns the orbit number instead would name
# every tile of an overpass identically again -- the exact failure this parse exists to stop.
_TILE_RE = re.compile(r"_([0-9]{1,2}[A-Z]{3})_\d{8}T\d{6}_")


def tile_id(granule_id: str) -> Optional[str]:
    """The MGRS tile in an ECO_L2T granule id, or None if it carries none.

    ECO_L2T is a TILED product: one overpass is delivered as several granules sharing an
    acquisition time exactly and differing only here. Without this field in the output name
    they all collapse onto one file (see `naming.tile_stem`).
    """
    m = _TILE_RE.search(granule_id)
    return m.group(1) if m else None


def granule_bbox(granule) -> Optional[tuple[float, float, float, float]]:
    """(W, S, E, N) in EPSG:4326 from the granule's own catalogue metadata, or None.

    Read from the search result -- no extra request. None means "the metadata does not say",
    which every caller must treat as "keep it": dropping a granule because its extent could
    not be read would lose real data on a metadata quirk.
    """
    try:
        geom = (granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"])
        rects = geom.get("BoundingRectangles") or []
        if not rects:
            return None
        r = rects[0]
        w, s = float(r["WestBoundingCoordinate"]), float(r["SouthBoundingCoordinate"])
        e, n = float(r["EastBoundingCoordinate"]), float(r["NorthBoundingCoordinate"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if w > e:
        # Crosses the antimeridian, so this is TWO boxes and `shapely.box` would build the
        # inside-out one -- which intersects nothing and would drop the granule silently.
        # Unrepresentable here is "the metadata does not say", i.e. keep it.
        return None
    return (w, s, e, n)


# --------------------------------------------------------------------------- #
# Per-granule processing
# --------------------------------------------------------------------------- #
# ECOSTRESS COGs are read through the shared windowed reader (grid.read_cog_window). Its
# 110 km tiles want a slightly wider pad than the 300 m the optical products use -- ~6
# pixels of slack at the 70 m native posting -- so that is passed explicitly.
ECO_PAD_M = 600.0


def process_granule(role_to_file, ds_cfg, grid_cfg, g: AoiGrid, aoi_id,
                    acq_time) -> Optional[xr.Dataset]:
    """One granule's COG assets -> the aligned sst/water/cloud/valid Dataset, or None.

    `role_to_file` maps a role to anything rioxarray opens: the fsspec objects
    earthaccess.open() returns for the remote assets, or plain paths in a test.
    """
    categorical = set(ds_cfg.get("categorical", []))
    rs_cont = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    rs_cat = Resampling[grid_cfg.get("resampling_categorical", "nearest")]

    data_vars, failed = {}, []
    for role, fobj in role_to_file.items():
        # By ROLE, not by the role's asset suffix. `categorical` lists role names
        # (cloud/water/quality); looking up `layers[role]` yields the SUFFIX, which matches
        # only because `cloud` and `water` happen to spell both the same. `quality`'s suffix
        # is `QC`, so the bit-packed QA layer fell through to the CONTINUOUS resampler and
        # would have been bilinear-interpolated into bit patterns that mean nothing.
        resampling = rs_cat if role in categorical else rs_cont
        try:
            da = read_cog_window(fobj, g, resampling=resampling, pad_m=ECO_PAD_M)
            da = da.rio.clip([g.geom_proj], g.target_crs, drop=False)
        except Exception as exc:  # window outside this tile, transient COG read failure...
            # ...but NOT a dead credential. An expired S3/EDL token fails every layer
            # identically, so swallowing it here turned an expiry into "core layer(s)
            # missing" and dropped the granule as degraded -- the same silent, data-shaped
            # failure that once made a Landsat AoI look like it ran out of dates. Let it out
            # to the caller, which is the level that can re-authenticate and re-open.
            if net.is_auth_error(exc):
                raise
            log.warning("    layer %s failed to read (%s)", role, exc)
            failed.append(role)
            continue
        data_vars[role] = da

    if "sst" not in data_vars and "lst" not in data_vars:
        log.warning("    no SST/LST after processing; dropping granule")
        return None

    # A granule missing a CORE layer is degraded, not merely partial -- and a degraded
    # granule is worse than none at all. Without `water`/`cloud` the `valid` mask below is
    # never built, and the assembler reads a missing water layer as "claim nothing", so the
    # scene silently becomes an all-zero valid mask: a total cloud-out, indistinguishable
    # from a genuinely overcast day. Drop it and let a later run re-fetch it.
    #
    # This catches BOTH ways a layer goes missing: a read that failed (`failed`, e.g. a
    # transient COG error) and an asset the granule never published (dropped upstream by
    # filter_links_for_granule). Only layers this config actually ASKED for are required,
    # so a custom `layers` list cannot make every granule look degraded.
    missing = [r for r in CORE_ROLES if r in ds_cfg["layers"] and r not in data_vars]
    if missing:
        log.warning("    dropping granule: core layer(s) %s missing (%s); a granule with "
                    "no mask reads as a total cloud-out downstream",
                    ", ".join(missing),
                    "read failed" if failed else "not published by the granule")
        return None

    ds = xr.Dataset(data_vars)

    if grid_cfg.get("to_celsius", False):
        for v in ("sst", "lst"):
            if v in ds:
                ds[v] = ds[v] - 273.15
                ds[v].attrs["units"] = "degC"
    else:
        for v in ("sst", "lst"):
            if v in ds:
                ds[v].attrs["units"] = "K"

    if {"sst", "water", "cloud"} <= set(ds.data_vars):
        # Polarity comes from the REGISTRY, through the same helper the assembler uses. This
        # line used to read `ds["water"] > 0` while `datacube._read_granule` recomputed the
        # very same mask as `water < 0.5` -- exact opposites, and since the assembler
        # recomputes (`trust_valid=False` for this sensor) the cube quietly used the other
        # answer than the file advertised.
        sensor = products.spec(DataProduct.ecostress).sensor
        water = products.water_cells(ds["water"], water_is_land=sensor.water_is_land)
        valid = (np.isfinite(ds["sst"]) & np.isfinite(ds["water"]) & water
                 & ~(ds["cloud"] > 0))
        ds["valid"] = valid.astype("uint8")
        ds["valid"].attrs["long_name"] = "water & clear & finite SST"

    if acq_time is not None:
        ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    # Built from what we ACTUALLY searched, never a literal: this attr is what
    # provenance.source_of() reads and what every eco_* field in every cube is stamped
    # with, so a hardcoded string here is a lie the cube then repeats. (It used to say
    # "v003" while DEFAULT_VERSION -- and the search -- was "002".)
    ds.attrs.update(aoi_id=aoi_id,
                    source=f"ECOSTRESS {ds_cfg['short_name']} v{ds_cfg['version']}",
                    processing="reprojected+clipped to AOI grid")
    return ds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, list_layers,
        only_source=None):
    """Acquire ECOSTRESS onto the pre-computed per-AoI grids, STACKED per version (D10).

    Each configured collection version is acquired INDEPENDENTLY into its own
    `ECOSTRESS/<tag>/aligned/<aoi>` tree, so a granule that exists in both v002 and v003
    lands under each rather than colliding, and the cube ships one channel-set per version.
    `only_source` narrows the run to one version tag (the pipeline dispatches the shared
    module once and lets it fan out; a direct CLI run can restrict it).

    `grids` (from coastal_sst_data.grid) is the single source of truth for the CRS + target
    grid, so every version lands on exactly the same grid as every other product for the AoI.
    """
    grid_cfg = eff["grid"]
    eco_root = eff["eco_root"]
    fmt, overwrite = eff["fmt"], eff["overwrite"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    login(eff["earthdata"]["auth_strategy"])

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("ecostress")

    for name in names:
        g = grids[name]
        # ECOSTRESS is a GLOBAL product, so it has no region-varying options -- but its
        # settings are still resolved per AoI, so every product answers to one contract.
        ds_cfg = eff["ds"][name]
        versions = [v for v in ds_cfg["versions"] if only_source is None or v == only_source]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d @ %.0fm) | versions=%s ===",
                 name, g.target_crs, g.width, g.height, g.resolution_m, versions)
        auth.ensure_fresh("earthdata", eff["earthdata"])

        for tag in versions:
            # The search + the source-attr stamp both read `version` (the bare collection
            # number); the tag (with its `v`) names the output tree and the cube channel.
            vcfg = {**ds_cfg, "version": _collection_version(tag)}
            layers = dict(vcfg["layers"])

            granules = search_granules(vcfg, g.search_bbox, start, end)
            if not granules:
                continue
            if list_layers:
                links = granules[0].data_links()
                tails = sorted({u.rsplit("_", 1)[-1] for u in links if u.endswith(".tif")})
                log.info("  [%s] available COG suffixes: %s", tag, tails)
                continue
            if dry_run:
                log.info("  [%s] [dry-run] would process %d granule(s)", tag, len(granules))
                continue

            aoi_out = eco_root / tag / "aligned" / name
            # Once per (AoI, version), not once per granule: the projection back to lon/lat
            # is the same answer every time and there are thousands of granules.
            aoi_lonlat = g.geom_lonlat()

            for gi, granule in enumerate(granules, 1):
                if gi % auth.CHECK_EVERY == 0:
                    auth.ensure_fresh("earthdata", eff["earthdata"])
                gid = granule_name(granule)

                # Tiles that touch the search BOX but miss the AoI polygon read five COGs to
                # produce a file that is nodata everywhere. Skipped before anything is opened.
                bbox = granule_bbox(granule)
                if bbox is not None and not shp_box(*bbox).intersects(aoi_lonlat):
                    log.debug("  [%s %d/%d] %s outside the AoI polygon, skipping",
                              tag, gi, len(granules), gid)
                    continue

                role_to_url = filter_links_for_granule(granule, layers)
                if not role_to_url:
                    continue
                t = parse_acq_time(gid)
                tstr = naming.time_stamp(t) if t else f"g{gi}"
                # ECO_L2T is TILED: without the tile, every granule of one overpass names the
                # same file and `store.done` reports all but the first as already processed,
                # leaving the AoI covered by whichever tile happened to be reached first. An
                # id we cannot read the tile from falls back to the untiled stem -- it is one
                # granule's name, not a reason to drop the overpass.
                tile = tile_id(gid)
                if tile is None:
                    log.warning("    %s: no MGRS tile in the granule id; a second tile of "
                                "this overpass would collide with it", gid)
                    stem = f"{name}_{tstr}"
                else:
                    stem = naming.tile_stem(name, t, tile) if t else f"{name}_{tstr}_{tile}"
                label = f"{tstr} {tile}" if tile else tstr
                if store.done(aoi_out / f"{stem}.nc", expected_vars(vcfg),
                              shape=(g.height, g.width), overwrite=overwrite):
                    log.info("  [%s %d/%d] %s already processed, skipping",
                             tag, gi, len(granules), label)
                    rep.skip()
                    continue

                log.info("  [%s %d/%d] streaming %d layer(s) for %s",
                         tag, gi, len(granules), len(role_to_url), label)
                try:
                    # The OPEN and the READS go in ONE closure. `earthaccess.open` returns
                    # handles bound to the fsspec session they were created with, so
                    # re-authenticating cannot heal a handle already held -- only re-opening
                    # can. Retrying the reads alone would keep reading through a dead session.
                    def _open_and_read(r2u=dict(role_to_url)):
                        # Roles are mapped to handles BY URL, never by position. Two roles
                        # legitimately share one asset -- `sst` and `lst` are both _LST.tif --
                        # so the URL list is SHORTER than the role list, and `earthaccess.open`
                        # hands back one handle per DISTINCT url. Zipping roles to handles
                        # positionally therefore paired every role after `sst` with the NEXT
                        # role's asset and dropped the last role entirely: `cloud` received the
                        # water mask, `water` received the QC bit field, and `quality` was never
                        # opened. Nothing failed -- the aligned files were written, full of
                        # plausible numbers, and the assembler then computed `valid` from a QC
                        # field with a water rule and got an empty mask on every granule, which
                        # collapsed each day's mosaic onto its first tile.
                        uniq = list(dict.fromkeys(r2u.values()))
                        # CLOSED in a finally: each handle holds an HTTPS connection and a
                        # one-thread executor for its background block cache, and neither is
                        # released until the object is collected. Tiling multiplied the
                        # handles per overpass, so leaking them stopped being survivable.
                        # `process_granule` reprojects to numpy before returning, so the
                        # Dataset it hands back does not read through these again.
                        fobjs = earthaccess.open(uniq)
                        try:
                            if len(fobjs) != len(uniq):
                                # The silent short return is what made the bug above possible;
                                # refuse to guess which asset is which.
                                raise RuntimeError(
                                    f"earthaccess.open returned {len(fobjs)} handle(s) for "
                                    f"{len(uniq)} distinct URL(s); cannot tell which asset is "
                                    f"which")
                            by_url = dict(zip(uniq, fobjs))
                            return process_granule({r: by_url[u] for r, u in r2u.items()},
                                                   vcfg, grid_cfg, g, name, t)
                        finally:
                            for fo in fobjs:
                                try:
                                    fo.close()
                                except Exception:   # noqa: BLE001 -- a close that fails must
                                    pass            # not mask what we are returning/raising

                    ds = net.retry(_open_and_read, what=f"ECOSTRESS {tag} {label}",
                                   refresh=auth.refresher("earthdata", eff["earthdata"]))
                except Exception as exc:
                    log.warning("    FAILED to open %s %s (%s)", tag, label, exc)
                    rep.fail(f"{name} {tag} {label}", exc)
                    continue

                if ds is None:
                    # dropped as degraded (a mask layer was missing) -- a LOSS, not a no-op
                    rep.fail(f"{name} {tag} {label}", "granule dropped (missing core layer)")
                    continue
                ds.attrs.update(**provenance.stamp(eff))
                try:
                    out = store.write_output(ds, aoi_out, stem, fmt)
                except (OSError, RuntimeError) as exc:   # this granule only -- see cmems.run
                    log.warning("      WRITE FAILED %s (%s)", stem, exc)
                    rep.fail(f"{name} {tag} {label}", f"write failed: {exc}")
                    continue
                log.info("      wrote %s", out)
                rep.wrote(source=ds.attrs.get("source"))

    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's ECOSTRESS settings. ECOSTRESS is global -- nothing is region-overridable.

    The stacked collection VERSIONS (D10) are a per-project list of source TAGS; absent, it
    defaults to the historical single collection (`DEFAULT_VERSIONS`). Each tag writes its own
    tree + cube channel-set, so `versions: [v003, v002]` spans a range no single collection
    covers. Validated against the registry's known versions by config._stacked_source_lists.
    """
    vers = _opt(opts, "versions", None)
    if vers is None:
        vers = list(DEFAULT_VERSIONS)
    elif isinstance(vers, str):
        vers = [vers]
    return {
        "short_name": _opt(opts, "short_name", SHORT_NAME),
        "versions": [str(v) for v in vers],
        "layers": dict(_opt(opts, "layers", LAYERS)),
        "categorical": list(_opt(opts, "categorical", CATEGORICAL)),
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes.

    ECOSTRESS product constants (short_name, layers, ...) come from module
    defaults; the config's ecostress product options may override any of them.
    Grid, time, auth and output_dir come from the shared project settings.
    """
    opts = project.products.get(DataProduct.ecostress)
    if opts is None:
        raise ValueError("ecostress is not a selected product in this config")
    if project.auth.earthdata is None:            # guaranteed by config validation
        raise ValueError("ecostress requires an auth.earthdata block")

    # Only the resampling / to_celsius knobs matter here; the CRS, resolution and
    # snapping are applied by coastal_sst_data.grid, not by this process.
    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.ecostress))
               for a in project.all_areas},
        "grid": grid_cfg,
        # Per-version trees hang under here: ECOSTRESS/<tag>/aligned/<aoi> (mirrors cmems_root).
        "eco_root": Path(project.output_dir) / "ECOSTRESS",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "earthdata": {"auth_strategy": project.auth.earthdata.auth_strategy},
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None, list_layers=False) -> None:
    """Acquire ECOSTRESS for a validated Project. Entry point for pipeline.py.

    Every configured collection version is acquired and STACKED (D10); the pipeline
    dispatches the shared module once and it fans out over the `versions` list internally.
    `source` narrows to one version tag (unset on a pipeline run; available for a direct
    CLI/test run) -- there is no fallback, a version with no coverage simply contributes no
    channel.

    Parameters
    ----------
    project      validated config (coastal_sst_data.config.Project)
    grids        pre-computed {aoi_name: AoiGrid} from coastal_sst_data.grid;
                 if None, they are computed here. The pipeline computes them once
                 and passes the same dict to every product so all grids match.
    aois         restrict to these AoI name(s); default all
    dry_run      search only, no download/write
    overwrite    reprocess overpasses even if the aligned file exists
    source       restrict to one version tag (e.g. "v003"); default all configured
    list_layers  print available COG suffixes for the first granule and exit
    """
    eff = _build_eff(project)
    # Credentials expire and runs are long: apply this project's refresh policy before any
    # network call. acquire() is the one entry point every invocation path goes through.
    auth.configure(project.auth)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    # A version tag that no code recognises would silently drop a collection -- fail loudly,
    # like bathymetry does for an unknown DEM source (config also checks this at load time).
    known = set(products.spec(DataProduct.ecostress).known_sources)
    bad = sorted(f"{n}:{v}" for n, c in eff["ds"].items() for v in c["versions"]
                 if v not in known)
    if bad:
        raise ValueError(f"ecostress version not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(known)}.")
    net.setup_gdal_env()      # windowed COG reads: deadline + retries
    return run(eff, grids, aois, dry_run, list_layers, only_source=source)


def main():
    entry.process_main(
        acquire, "coastal_sst_data ECOSTRESS acquisition.",
        extra=[entry.Flag("--list-layers",
                          "Print available COG suffixes for the first granule and exit.")])


if __name__ == "__main__":
    main()