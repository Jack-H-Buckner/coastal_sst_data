#!/usr/bin/env python3
"""
coastal_sst_data -- Copernicus Marine (CMEMS) global physics acquisition.

The offshore ocean state the nearshore exchanges with: temperature, salinity and
currents AT DEPTH, from the Copernicus Marine global physics models. Where MUR gives
one skin temperature at the surface, this gives the water column -- so a model can see
the stratification and the offshore water mass that upwelling and tidal exchange draw
into an estuary.

DISTINCT-DATA sources, STACKED (D10) -- each a source TAG naming an exact dataset, acquired
INDEPENDENTLY into its own tree and emitted as its own `cmems_<var>_<tag>` cube channel. There
is NO fallback chain: the user lists the tags they need, and a day/region a tag does not cover
is a NaN slice they fill by stacking another tag. Built-in tags:

  * my_global   -- cmems_mod_glo_phy_my_0.083deg_P1D-m, the GLORYS12 REANALYSIS (hindcast):
    1/12 deg, daily means, 50 depth levels. Best quality, but stops a year or two behind.
  * anfc_global -- cmems_mod_glo_phy_anfc_0.083deg_P1D-m, the ANALYSIS/FORECAST product,
    which reaches the present.

Regional models (Baltic/Med/NW-Shelf) are registered per config as extra tags:
`datasets: {anfc_med: <dataset_id>}`, then listed in `sources` for the region that needs them.
The tag IS the provenance identity, so each channel is self-describing.

DEPTHS. The model has ~50 fixed levels (0.494, 1.54, 2.65, 5.08 m ...). The config asks
for depths in metres and each is snapped to the NEAREST model level -- no interpolation,
so every value is one the model actually computed. The level actually used is recorded
in the variable's `model_depth_m` attr, which will differ from the requested depth:

    depths: [0, 10, 30]  ->  thetao_0m  (level 0.494 m)
                             thetao_10m (level 9.573 m)
                             thetao_30m (level 29.445 m)

2D variables (zos, mlotst) have no depth dimension and are written once, unsuffixed.

Data is streamed with `copernicusmarine.open_dataset`, which is LAZY: the AoI window is
subset server-side, so the global model is never downloaded. One open per AoI covers the
whole time range.

    <output_dir>/CMEMS/<tag>/aligned/<aoi>/<aoi>_<YYYYMMDD>.nc

CREDENTIALS. Needs a free Copernicus Marine account (https://data.marine.copernicus.eu).
Like Earthdata, the secret never goes in the config: it lives in ~/.netrc under
`machine auth.marine.copernicus.eu`, in COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD, or
in the toolbox's own credentials file. Declare only the strategy, in `auth.copernicus`.

WHAT EXPIRES HERE IS THE HANDLE, NOT A TOKEN. Unlike Earthdata, copernicusmarine caches no
token in-process -- credentials are re-read from env/file on every call, and the ARCO data
path is unsigned (the username rides along only for usage accounting). So a mid-run 401/403
is usually NOT something re-logging-in can fix. What DOES go stale is the LAZY DATASET
HANDLE: `open_window` is called once per (AoI, source) and then read day by day, which for a
5-year range is ~1800 chunk fetches through one client built at open time. The only refresh
that works is re-opening, so that is exactly what the day loop in `run()` does -- and a
per-day failure is no longer swallowed as a coverage gap, which is how an expiry used to
reach the cube as an indistinguishable NaN slice.

Usage:
    python -m coastal_sst_data.processes.cmems --config config.yaml
    python -m coastal_sst_data.processes.cmems --config config.yaml --aoi hood_canal
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling

from ..config import DataProduct, Project, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import auth, entry, naming, net, provenance, report, store

log = logging.getLogger(__name__)

SOURCE = "cmems"

# BUILT-IN source TAG -> dataset id. Each tag names an exact CMEMS dataset and IS the
# provenance identity, so a `cmems_<var>_<tag>` channel is self-describing. Regional tags
# (`my_baltic`, `anfc_med`, ...) are registered per config via the `datasets` option, since
# their ids vary (and some regionals split variables into separate products). No fallback:
# distinct-data sources are STACKED, not substituted (D10) -- the user lists what they need.
DATASET_IDS = {
    "my_global": "cmems_mod_glo_phy_my_0.083deg_P1D-m",       # GLORYS12 reanalysis (hindcast)
    "anfc_global": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",   # analysis / forecast
}
DEFAULT_SOURCES = ["my_global", "anfc_global"]
DEFAULT_VARIABLES = ["thetao"]           # sea water potential temperature
DEFAULT_DEPTHS = [0.0]                   # metres; snapped to the nearest model level
DEFAULT_PAD_DEG = 0.15                   # >= one 1/12 deg cell of padding around the AoI

# Variables with no depth dimension -- written once, unsuffixed.
SURFACE_ONLY = {"zos", "mlotst", "siconc", "sithick", "bottomT"}

OUT_UNITS = {"thetao": "degC", "so": "1e-3", "uo": "m s-1", "vo": "m s-1",
             "zos": "m", "mlotst": "m", "bottomT": "degC"}


def depth_label(d: float) -> str:
    """0.0 -> '0m', 10.0 -> '10m', 2.5 -> '2.5m' (the REQUESTED depth, not the level)."""
    return f"{d:g}m"


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def open_window(dataset_id, variables, bbox_ll, pad, start, end, depths, creds):
    """Lazily open one CMEMS product, subset to the AoI window + depth range.

    Server-side subsetting: only the AoI's cells are ever transferred, so the global
    model is not downloaded. Returns None when the product does not cover this window
    (e.g. dates past the end of the reanalysis).
    """
    import copernicusmarine

    w, s, e, n = bbox_ll
    kw = dict(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=w - pad, maximum_longitude=e + pad,
        minimum_latitude=s - pad, maximum_latitude=n + pad,
        start_datetime=str(start), end_datetime=str(end),
        coordinates_selection_method="outside",   # never clip the AoI to a partial cell
    )
    if depths:
        kw["minimum_depth"] = 0.0
        kw["maximum_depth"] = float(max(depths)) + 5.0   # room to snap to a deeper level
    if creds.get("credentials_file"):
        kw["credentials_file"] = creds["credentials_file"]

    try:
        return net.retry(lambda: copernicusmarine.open_dataset(**kw),
                         what=f"CMEMS open {dataset_id}",
                         refresh=auth.refresher("copernicus"))
    except Exception as exc:
        # Returns None -> this source contributes no data here (a NaN slice in its channel).
        # That is right for a genuine coverage gap and WRONG for an expired credential -- which
        # used to look identical, at INFO. Say which one this is, loudly, so a missing channel
        # is not mistaken for "this source doesn't cover the AoI".
        if net.is_auth_error(exc):
            # Named separately from the generic failure below because the remedy differs and
            # the consequence is severe: an empty channel here is NOT a coverage gap, and if a
            # fallback source is stacked it will quietly serve a different product instead.
            log.error("  %s: CREDENTIAL failure (%s). This is not a coverage gap -- Copernicus "
                      "Marine rejected the credential, so this channel will be empty.",
                      dataset_id, exc)
        elif net.is_transient(exc):
            log.warning("  %s: unreachable after retries (%s); treating as no data",
                        dataset_id, exc)
        else:
            log.warning("  %s: open FAILED (%s). If this is a credential/permission error, "
                        "the fallback will now quietly serve a DIFFERENT product.",
                        dataset_id, exc)
        return None


def snap_depths(ds: xr.Dataset, depths) -> dict[float, float]:
    """{requested depth -> nearest MODEL level}, so we never invent a value."""
    if "depth" not in ds.coords or not depths:
        return {}
    levels = np.asarray(ds["depth"].values, dtype="float64")
    out = {}
    for d in depths:
        lvl = float(levels[int(np.argmin(np.abs(levels - float(d))))])
        out[float(d)] = lvl
        if abs(lvl - float(d)) > 5.0:
            log.warning("  requested depth %.1f m snaps to model level %.2f m "
                        "(%.1f m away)", float(d), lvl, abs(lvl - float(d)))
    return out


def to_grid(da: xr.DataArray, g: AoiGrid, grid_cfg) -> np.ndarray:
    """One lat/lon slice -> the AoI grid (bilinear, like MUR's 1 km upsample)."""
    da = da.rename({"longitude": "x", "latitude": "y"}).sortby("y", ascending=False)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    rs = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    out = da.rio.reproject(dst_crs=g.target_crs, shape=(g.height, g.width),
                           transform=g.transform, resampling=rs, nodata=np.nan)
    out = out.rio.clip([g.geom_proj], g.target_crs, drop=False)
    return out.values.astype("float32")


def day_dataset(src_ds: xr.Dataset, day, g: AoiGrid, variables, level_of, grid_cfg,
                to_celsius: bool) -> xr.Dataset | None:
    """All requested variables x depths for ONE day, on the AoI grid."""
    sel = src_ds.sel(time=slice(pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta("1D")))
    if sel.sizes.get("time", 0) == 0:
        return None
    sel = sel.isel(time=0)

    xs, ys = g.xy_centers()
    out, attrs = {}, {}
    for var in variables:
        if var not in sel:
            continue
        da = sel[var]
        if var in SURFACE_ONLY or "depth" not in da.dims:
            out[var] = (("y", "x"), to_grid(da, g, grid_cfg))
            attrs[var] = {"units": OUT_UNITS.get(var, "")}
            continue
        for req, lvl in level_of.items():
            name = f"{var}_{depth_label(req)}"
            out[name] = (("y", "x"),
                         to_grid(da.sel(depth=lvl, method="nearest"), g, grid_cfg))
            attrs[name] = {"units": OUT_UNITS.get(var, ""),
                           "requested_depth_m": float(req),
                           "model_depth_m": float(lvl),
                           "long_name": f"{var} at the model level nearest {req:g} m"}
    if not out:
        return None

    ds = xr.Dataset(out, coords={"y": ys, "x": xs})
    for name, a in attrs.items():
        ds[name].attrs.update(a)
    if to_celsius:
        for name in ds.data_vars:
            if name.startswith(("thetao", "bottomT")):
                ds[name] = ds[name] - 273.15 if ds[name].attrs.get("units") == "K" else ds[name]
    # Gap-free where there is water; `valid` is finite (the model's land is NaN).
    first = next(iter(ds.data_vars))
    ds["valid"] = np.isfinite(ds[first]).astype("uint8")
    ds["valid"].attrs["long_name"] = "finite CMEMS value (water)"
    ds = ds.expand_dims(time=[pd.Timestamp(day)])
    return ds


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    """Acquire each configured CMEMS SOURCE TAG independently into its own tree.

    No fallback chain (D10): a tag covers whatever days its dataset covers, and a day it does
    not reach simply has no file there -> a NaN slice in that tag's cube channel, which the
    user fills by stacking another tag. `only_source` narrows the run to one tag.
    """
    grid_cfg = eff["grid"]
    cmems_root, fmt, overwrite = eff["cmems_root"], eff["fmt"], eff["overwrite"]
    to_celsius = grid_cfg.get("to_celsius", False)
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]
    days = pd.date_range(start, end, freq="D")

    # CMEMS never logged in during acquisition -- it relied on the preflight and passed a
    # credentials file per open. Logging in here gives the backend a recorded age, so the
    # proactive check below has something to measure.
    auth.login("copernicus", {"auth_strategy": eff["strategy"]})

    names = select_aois(grids, only_aoi)
    rep = report.ProductReport("cmems")

    for name in names:
        g = grids[name]
        ds_cfg = eff["ds"][name]
        variables, depths, pad = ds_cfg["variables"], ds_cfg["depths"], float(ds_cfg["pad_deg"])
        datasets = ds_cfg["datasets"]
        sources = [s for s in ds_cfg["sources"] if only_source is None or s == only_source]

        for src in sources:
            # `open_dataset` is where credentials are actually checked, so this is the
            # meaningful boundary at which to top the credential up.
            auth.ensure_fresh("copernicus", {"auth_strategy": eff["strategy"]})
            dsid = datasets.get(src)
            if dsid is None:
                log.warning("  %s: no dataset id for source %r (register it in `datasets`)",
                            name, src)
                rep.fail(f"{name} {src}", "no dataset id"); continue
            log.info("=== AOI: %s (grid=%dx%d) | source=%s (%s) | vars=%s depths=%s ===",
                     name, g.width, g.height, src, dsid, variables, depths)

            aoi_out = cmems_root / src / "aligned" / name
            remaining = [d for d in days
                         if not store.done(aoi_out / f"{naming.day_stem(name, d)}.nc",
                                           store.REQUIRED_VARS["CMEMS"], shape=(g.height, g.width),
                                           overwrite=overwrite)]
            rep.expect(len(days)); rep.skip(len(days) - len(remaining))
            if not remaining:
                log.info("  all %d day(s) already processed, skipping", len(days)); continue
            if dry_run:
                log.info("  [dry-run] would acquire %d day(s) from %s", len(remaining), dsid); continue

            sds = open_window(dsid, variables, g.search_bbox, pad,
                              min(remaining).date(), max(remaining).date(), depths, eff["creds"])
            if sds is None:
                log.warning("  %s: %s (%s) has no coverage here; no channel from it "
                            "(stack another source)", name, src, dsid)
                for d in remaining:
                    rep.fail(f"{name} {src} {naming.day_stamp(d)}", f"{src}: no coverage")
                continue
            level_of = snap_depths(sds, depths)
            if level_of:
                log.info("  %s: depths %s -> model levels %s", src, [f"{d:g}" for d in level_of],
                         [f"{v:.2f}" for v in level_of.values()])
            # ONE lazy handle, then potentially thousands of days read through it -- a 5-year
            # range is ~1800 chunk fetches against the client `open_window` built once. That
            # handle is what goes stale here, NOT a process-wide token (copernicusmarine caches
            # none, and the ARCO path is unsigned), so the only refresh that can work is
            # RE-OPENING. `holder` is what lets the closure below read through the new handle.
            holder = [sds]

            def _reopen():
                log.info("  %s: re-opening the dataset handle after a credential failure", dsid)
                fresh = open_window(dsid, variables, g.search_bbox, pad,
                                    min(remaining).date(), max(remaining).date(),
                                    depths, eff["creds"])
                if fresh is None:
                    raise RuntimeError(f"{dsid}: could not re-open after a credential failure")
                holder[0] = fresh

            for day in remaining:
                try:
                    ds = net.retry(
                        lambda day=day: day_dataset(holder[0], day, g, variables, level_of,
                                                    grid_cfg, to_celsius),
                        what=f"CMEMS {src} {naming.day_stamp(day)}",
                        attempts=1, refresh=_reopen)
                except Exception as exc:
                    log.warning("    %s: %s", day.strftime("%Y%m%d"), exc); continue
                if ds is None:                       # this source has no such day -> NaN in cube
                    continue
                ds.attrs.update(aoi_id=name, source=dsid, cmems_source=src,
                                processing="subset + bilinear reproject to AOI grid",
                                **provenance.stamp(eff))
                log.info("  [%s] %s -> %s", src, naming.day_stamp(day),
                         store.write_output(ds, aoi_out, naming.day_stem(name, day), fmt).name)
                rep.wrote(source=dsid)
            holder[0].close()
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's CMEMS settings: which SOURCE TAGS to STACK (region-overridable), plus the
    tag->dataset-id map (built-ins + any regional tags registered via `datasets`).

    `sources`/`datasets` are region-overridable (which model covers this AoI); `variables`/
    `depths` are not, so every AoI's cube carries the same CMEMS channels.
    """
    srcs = _opt(opts, "sources", None)
    if srcs is None:
        srcs = list(DEFAULT_SOURCES)
    elif isinstance(srcs, str):
        srcs = [srcs]
    datasets = dict(DATASET_IDS)
    datasets.update(_opt(opts, "datasets", {}) or {})
    return {
        "sources": [str(s) for s in srcs],
        "datasets": datasets,
        "variables": list(_opt(opts, "variables", DEFAULT_VARIABLES)),
        "depths": [float(d) for d in _opt(opts, "depths", DEFAULT_DEPTHS)],
        "pad_deg": float(_opt(opts, "pad_deg", DEFAULT_PAD_DEG)),
    }


def _build_eff(project: Project) -> dict:
    opts = project.products.get(DataProduct.cmems)
    if opts is None:
        raise ValueError("cmems is not a selected product in this config")
    if project.auth.copernicus is None:              # guaranteed by config validation
        raise ValueError("cmems requires an auth.copernicus block")

    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)

    strategy = project.auth.copernicus.auth_strategy
    creds = {"credentials_file": str(Path.home() / ".netrc")} if strategy == "netrc" else {}

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.cmems))
               for a in project.all_areas},
        "grid": grid_cfg,
        "creds": creds,
        "strategy": strategy,
        "cmems_root": Path(project.output_dir) / "CMEMS",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None) -> None:
    """Acquire CMEMS physics for a validated Project. Entry point for pipeline.py.

    Every configured source tag is acquired and stacked; `source` (from the pipeline, or
    unset on a direct CLI run) narrows to one tag. A tag with no dataset id / no coverage
    simply contributes no channel -- there is no fallback.
    """
    eff = _build_eff(project)
    # Credentials expire and runs are long: apply this project's refresh policy before any
    # network call. acquire() is the one entry point every invocation path goes through.
    auth.configure(project.auth)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run, only_source=source)


def main():
    entry.process_main(acquire, "coastal_sst_data CMEMS physics acquisition.")


if __name__ == "__main__":
    main()
