#!/usr/bin/env python3
"""
OCEANSR -- MUR L4 SST backbone acquisition.

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; MUR settings come from `sources.mur`. For each AOI and each
day it streams the daily GHRSST MUR L4 granule from PODAAC (earthaccess.open),
subsets `analysed_sst` to the AOI lat/lon window (HDF5 range reads -- the global
1 km file is never fully downloaded), upsamples onto the AOI grid (identical to
the ECOSTRESS/Landsat grid), and writes one aligned NetCDF per day.

MUR is a gap-free L4 analysis, so it has no cloud mask; `valid` = finite SST
(i.e. water). It is the always-present backbone the model fills high-res detail
onto. A later stage bins these to the daily datacube.

Usage (run from the OCEANSR project root, Earthdata auth via ~/.netrc):
    python src/acquire_mur.py --config configs/config.yaml
    python src/acquire_mur.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_mur.py --config configs/config.yaml --dry-run
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import earthaccess
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling

from ..config import Project, DataProduct, opt as _opt
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, naming, net, provenance, report, store

log = logging.getLogger(__name__)

# --- MUR product constants -------------------------------------------------- #
# Describe the MUR L4 product itself (not user choices); overridable via the
# mur product options block in the config.
SOURCE = "mur"
SHORT_NAME = "MUR-JPL-L4-GLOB-v4.1"
DEFAULT_VARIABLE = "analysed_sst"
DEFAULT_PAD_DEG = 0.05


# Config is loaded/validated by coastal_sst_data.config; the per-AoI CRS + grid
# come from coastal_sst_data.grid (AoiGrid), shared with every other product.


# --------------------------------------------------------------------------- #
# MUR per-granule processing
# --------------------------------------------------------------------------- #
def subset_and_reproject(fobj, variable, bbox_ll, pad, target_crs, transform,
                         width, height, geom_proj, grid_cfg) -> tuple[xr.DataArray, pd.Timestamp]:
    """Open one daily MUR granule lazily, subset to the AOI, upsample to grid."""
    w, s, e, n = bbox_ll
    ds = xr.open_dataset(fobj, engine="h5netcdf", mask_and_scale=True)
    da = ds[variable].isel(time=0)
    da = da.sel(lat=slice(s - pad, n + pad), lon=slice(w - pad, e + pad)).load()

    t = pd.Timestamp(ds["time"].values[0]).tz_localize(None)

    if grid_cfg.get("to_celsius", False):
        da = da - 273.15

    # Standard orientation + CRS, then reproject (bilinear upsample 1 km -> grid).
    da = da.rename({"lon": "x", "lat": "y"}).sortby("y", ascending=False)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    rs = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    out = da.rio.reproject(dst_crs=target_crs, shape=(height, width),
                           transform=transform, resampling=rs, nodata=np.nan)
    out = out.rio.clip([geom_proj], target_crs, drop=False)
    return out, t


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    """Acquire MUR onto the pre-computed per-AoI grids (shared across products)."""
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    variable = ds_cfg["variable"]
    pad = float(ds_cfg["pad_deg"])
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    log.info("Authenticating with Earthdata (strategy=%s)", eff["earthdata"]["auth_strategy"])
    earthaccess.login(strategy=eff["earthdata"]["auth_strategy"])

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("mur")

    for name in names:
        g = grids[name]
        log.info("=== AOI: %s (CRS=%s grid=%dx%d @ %.0fm) ===",
                 name, g.target_crs, g.width, g.height, g.resolution_m)

        granules = net.retry(
            lambda: earthaccess.search_data(
                short_name=ds_cfg["short_name"], temporal=(start, end),
                bounding_box=tuple(g.search_bbox)),
            what=f"MUR search {name}")
        log.info("  %d daily MUR granule(s)", len(granules))
        rep.expect(len(granules))
        if not granules:
            continue
        if dry_run:
            log.info("  [dry-run] would process %d day(s)", len(granules))
            continue

        aoi_out = out_root / name
        for gi, granule in enumerate(granules, 1):
            try:
                fobj = net.retry(lambda: earthaccess.open([granule])[0],
                                 what=f"MUR open granule {gi}")
                da, t = subset_and_reproject(fobj, variable, g.search_bbox, pad,
                                             g.target_crs, g.transform, g.width,
                                             g.height, g.geom_proj, grid_cfg)
            except Exception as exc:
                # A failed download and a day that genuinely has no data used to look
                # identical: one warning, no tally, and `Done.` all the same.
                log.warning("    [%d/%d] FAILED (%s)", gi, len(granules), exc)
                rep.fail(f"{name} granule {gi}", exc)
                continue

            stem = naming.day_stem(name, t)
            if store.done(aoi_out / f"{stem}.nc", store.REQUIRED_VARS["MUR"],
                          shape=(g.height, g.width), overwrite=overwrite):
                log.info("  [%d/%d] %s already processed, skipping", gi, len(granules),
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
            log.info("  [%d/%d] wrote %s", gi, len(granules),
                     store.write_output(ds, aoi_out, stem, fmt))
            rep.wrote(source=src)

    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.mur)
    if opts is None:
        raise ValueError("mur is not a selected product in this config")
    if project.auth.earthdata is None:            # guaranteed by config validation
        raise ValueError("mur requires an auth.earthdata block")

    ds_cfg = {
        "short_name": _opt(opts, "short_name", SHORT_NAME),
        "variable": _opt(opts, "variable", DEFAULT_VARIABLE),
        "pad_deg": float(_opt(opts, "pad_deg", DEFAULT_PAD_DEG)),
    }
    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    return {
        "config_sha256": project.config_sha256,
        "ds": ds_cfg,
        "grid": grid_cfg,
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
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(acquire, "coastal_sst_data MUR L4 SST acquisition.")


if __name__ == "__main__":
    main()