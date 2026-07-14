#!/usr/bin/env python3
"""
coastal_sst_data -- Copernicus Marine (CMEMS) global physics acquisition.

The offshore ocean state the nearshore exchanges with: temperature, salinity and
currents AT DEPTH, from the Copernicus Marine global physics models. Where MUR gives
one skin temperature at the surface, this gives the water column -- so a model can see
the stratification and the offshore water mass that upwelling and tidal exchange draw
into an estuary.

Two products, tried in order (the "source chain", like met's hrrr->era5):

  * my   (default) -- cmems_mod_glo_phy_my_0.083deg_P1D-m, the GLORYS12 REANALYSIS
    (hindcast): 1/12 deg, daily means, 50 depth levels. Best quality, but it stops a
    year or two behind the present.
  * anfc -- cmems_mod_glo_phy_anfc_0.083deg_P1D-m, the ANALYSIS/FORECAST product, which
    covers right up to the present. It backfills whatever days the reanalysis does not
    reach, so a project spanning the reanalysis cut-off is still gap-free.

Each output file records which product actually produced it in `ds.attrs["source"]`, so
a reanalysis day and a forecast day are never silently conflated.

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

    <output_dir>/CMEMS/aligned/<aoi>/<aoi>_<YYYYMMDD>.nc

CREDENTIALS. Needs a free Copernicus Marine account (https://data.marine.copernicus.eu).
Like Earthdata, the secret never goes in the config: it lives in ~/.netrc under
`machine auth.marine.copernicus.eu`, in COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD, or
in the toolbox's own credentials file. Declare only the strategy, in `auth.copernicus`.

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

from ..config import DataProduct, Project, opt as _opt
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, naming, net, provenance, report, store

log = logging.getLogger(__name__)

SOURCE = "cmems"

# source key -> dataset id. The chain is [my, anfc]: reanalysis first, forecast to cover
# the days it does not reach.
DATASET_IDS = {
    "my": "cmems_mod_glo_phy_my_0.083deg_P1D-m",       # GLORYS12 reanalysis (hindcast)
    "anfc": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",   # analysis / forecast
}
DEFAULT_SOURCE = "my"
DEFAULT_FALLBACK = "anfc"                # "none" to disable
DEFAULT_VARIABLES = ["thetao"]           # sea water potential temperature
DEFAULT_DEPTHS = [0.0]                   # metres; snapped to the nearest model level
DEFAULT_PAD_DEG = 0.15                   # >= one 1/12 deg cell of padding around the AoI

# Variables with no depth dimension -- written once, unsuffixed.
SURFACE_ONLY = {"zos", "mlotst", "siconc", "sithick", "bottomT"}

OUT_UNITS = {"thetao": "degC", "so": "1e-3", "uo": "m s-1", "vo": "m s-1",
             "zos": "m", "mlotst": "m", "bottomT": "degC"}


def _resolve_chain(source: str, fallback: str) -> list[str]:
    """['my', 'anfc'] for the defaults; a single source when the fallback is off."""
    chain = [DEFAULT_SOURCE] if source == "auto" else [source]
    if fallback and fallback not in ("none", "", None) and fallback not in chain:
        chain.append(fallback)
    bad = [s for s in chain if s not in DATASET_IDS]
    if bad:
        raise ValueError(f"cmems source(s) not recognized ({bad}); "
                         f"choose from {sorted(DATASET_IDS)}.")
    return chain


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
                         what=f"CMEMS open {dataset_id}")
    except Exception as exc:
        # This returns None, which the chain reads as "this product has no such day" and
        # falls through to the NEXT product. That is right for a genuine coverage gap and
        # WRONG for an expired credential -- which used to look identical, at INFO. Say
        # which one this is, loudly, because the fallback silently serves different data.
        if net.is_transient(exc):
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
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    to_celsius = grid_cfg.get("to_celsius", False)
    chain, variables, depths = ds_cfg["chain"], ds_cfg["variables"], ds_cfg["depths"]
    pad = float(ds_cfg["pad_deg"])
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]
    days = pd.date_range(start, end, freq="D")

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("cmems")

    for name in names:
        g = grids[name]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d) | chain=%s | vars=%s depths=%s ===",
                 name, g.target_crs, g.width, g.height, "->".join(chain),
                 variables, depths)
        if dry_run:
            log.info("  [dry-run] would acquire %d day(s) from %s", len(days),
                     " -> ".join(DATASET_IDS[s] for s in chain))
            continue

        aoi_out = out_root / name
        remaining = [d for d in days
                     if not store.done(aoi_out / f"{naming.day_stem(name, d)}.nc",
                                       store.REQUIRED_VARS["CMEMS"], shape=(g.height, g.width),
                                       overwrite=overwrite)]
        rep.expect(len(days))
        rep.skip(len(days) - len(remaining))
        if not remaining:
            log.info("  all %d day(s) already processed, skipping", len(days))
            continue

        # Walk the chain: the reanalysis covers what it covers, the forecast product
        # picks up the rest. One lazy open per product, not per day.
        for src in chain:
            if not remaining:
                break
            sds = open_window(DATASET_IDS[src], variables, g.search_bbox, pad,
                              min(remaining).date(), max(remaining).date(), depths,
                              eff["creds"])
            if sds is None:
                continue
            level_of = snap_depths(sds, depths)
            if level_of:
                log.info("  %s: depths %s -> model levels %s", src,
                         [f"{d:g}" for d in level_of],
                         [f"{v:.2f}" for v in level_of.values()])

            still: list = []
            for day in remaining:
                try:
                    ds = day_dataset(sds, day, g, variables, level_of, grid_cfg, to_celsius)
                except Exception as exc:
                    log.warning("    %s: %s", day.strftime("%Y%m%d"), exc)
                    still.append(day)
                    continue
                if ds is None:                       # this product has no such day
                    still.append(day)
                    continue
                ds.attrs.update(aoi_id=name, source=DATASET_IDS[src], cmems_source=src,
                                processing="subset + bilinear reproject to AOI grid",
                                **provenance.stamp(eff))
                log.info("  [%s] %s -> %s", src, naming.day_stamp(day),
                         store.write_output(ds, aoi_out, naming.day_stem(name, day), fmt).name)
                rep.wrote(source=DATASET_IDS[src])
            remaining = still
            sds.close()

        if remaining:
            log.warning("  %s: %d day(s) NOT COVERED by %s", name, len(remaining),
                        " or ".join(chain))
            for d in remaining:
                rep.fail(f"{name} {naming.day_stamp(d)}", f"not covered by {' or '.join(chain)}")
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _build_eff(project: Project) -> dict:
    opts = project.products.get(DataProduct.cmems)
    if opts is None:
        raise ValueError("cmems is not a selected product in this config")
    if project.auth.copernicus is None:              # guaranteed by config validation
        raise ValueError("cmems requires an auth.copernicus block")

    chain = _resolve_chain(_opt(opts, "source", DEFAULT_SOURCE),
                           _opt(opts, "fallback", DEFAULT_FALLBACK))
    # An explicit dataset_id overrides the chain entirely (an escape hatch for any of
    # the other CMEMS physics products).
    dataset_id = _opt(opts, "dataset_id", None)
    if dataset_id:
        DATASET_IDS.setdefault(dataset_id, dataset_id)
        chain = [dataset_id]

    ds_cfg = {
        "chain": chain,
        "variables": list(_opt(opts, "variables", DEFAULT_VARIABLES)),
        "depths": [float(d) for d in _opt(opts, "depths", DEFAULT_DEPTHS)],
        "pad_deg": float(_opt(opts, "pad_deg", DEFAULT_PAD_DEG)),
    }
    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)

    strategy = project.auth.copernicus.auth_strategy
    creds = {"credentials_file": str(Path.home() / ".netrc")} if strategy == "netrc" else {}

    return {
        "config_sha256": project.config_sha256,
        "ds": ds_cfg,
        "grid": grid_cfg,
        "creds": creds,
        "out_dir": Path(project.output_dir) / "CMEMS" / "aligned",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False) -> None:
    """Acquire CMEMS physics for a validated Project. Entry point for pipeline.py."""
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(acquire, "coastal_sst_data CMEMS physics acquisition.")


if __name__ == "__main__":
    main()
