#!/usr/bin/env python3
"""
coastal_sst_data -- in-situ observations from MOVING platforms: gliders, ship transects,
drifters, profiling floats.

The sibling of `insitu_acquire`, and deliberately a separate product rather than another
source under it. `insitu` writes `(station, time)` with ONE position per station -- which is
not a limitation to be routed around, it is what a fixed station IS. A glider measures as it
moves, so its position belongs to the OBSERVATION. Those are two different shapes of data and
they get two trees.

Why that matters beyond tidiness: folding per-observation positions into the shared model would
turn `insitu_station` from a static `(y,x)` map into `(time,y,x)`, breaking any config that
uses it as an `extract` mask -- and, worse, doing so silently, because `append_zarr(mode="a-")`
leaves static channels alone after the first block, so a time-varying map written as `(y,x)`
freezes at block 0 with no error. Nothing about the fixed path changes here.

THE CUBE STILL MERGES THEM. Both trees feed the one `insitu_sst` / `<sensor>_insitu_sst`
channel set, because ground truth is ground truth however it was collected. A track paints the
pixels it crossed on the day it crossed them; `insitu_station` stays fixed-only.

  * Where it comes from: the same `fetch_aoi` seams the fixed product uses -- `insitu_ioos`,
    `insitu_csv` -- plus `insitu_cmems.fetch_aoi_mobile`, which takes a DIFFERENT route into
    the Copernicus archive (see that module; the fixed route is unusable for tracks).
  * What it measures: water temperature in degC, one row per observation, each with its own
    time and position.

    <output_dir>/INSITU_MOBILE/<source>/aligned/<aoi>/<aoi>_insitu.nc   dims (obs,)

THE DRIFT GUARD IS INVERTED, not removed. `insitu_acquire` keeps what stays inside one grid
cell and drops the rest; this keeps exactly what that drops. A platform that never moves is a
fixed station filed in the wrong product -- placing it here would work, but it would also make
the same instrument appear in `insitu_sst` twice if the user configured both products, so it is
rejected by name. The threshold is one grid cell either way, so the two products partition the
platforms exactly, with no gap and no overlap.

Usage:
    python -m coastal_sst_data.processes.insitu_mobile --config config.yaml
    python -m coastal_sst_data.processes.insitu_mobile --config config.yaml --aoi hood_canal
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import DataProduct, Project, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import auth, entry, products, provenance, report, store
from . import insitu, insitu_acquire, insitu_cmems, insitu_csv, insitu_ioos

log = logging.getLogger(__name__)

# Nothing by default: selecting the product is the opt-in, and no source here is the obvious
# one to want (see the ProductSpec).
DEFAULT_SOURCES = list(products.spec(DataProduct.insitu_mobile).default_sources)

# `marineinsitu` gets its own entry point rather than the shared `fetch_aoi`, because reaching
# tracks in the Copernicus archive needs a different service entirely -- the fixed product's
# index route downloads a platform's whole life to find the hours it spent in the AoI.
SOURCES = {
    "ioos": insitu_ioos.fetch_aoi,
    "csv": insitu_csv.fetch_aoi,
    "marineinsitu": insitu_cmems.fetch_aoi_mobile,
}

# Platform classes that do NOT move, as the Copernicus vocabulary names them -- shared with the
# fixed product so the two cannot disagree about what "fixed" means. Used only when a source
# DECLARES a class; `csv` and `ioos` do not, and there drift is the only evidence available.
FIXED_PLATFORM_TYPES = frozenset(insitu_cmems.DEFAULT_PLATFORM_TYPES)


# --------------------------------------------------------------------------- #
# Records -> the flat observation table
# --------------------------------------------------------------------------- #
def split_fixed_platforms(records: list[dict], min_drift_m: float):
    """(mobile, [(record, drift_m), ...]) -- the platforms that DO NOT move are separated out.

    The exact complement of `insitu_acquire.split_moving_platforms`, sharing its
    `platform_drift_m` measure and its threshold, so the two products partition the platforms
    with no gap and no overlap.

    A stationary platform here is not an error in the data, it is a platform filed under the
    wrong product -- and admitting it would let the same instrument reach `insitu_sst` twice if
    both products are configured, once per tree. Rejected and named, so the fix is obvious.

    MEASURED DRIFT IS NOT THE ONLY SIGNAL, and relying on it alone silently lost data. A track
    that clips the corner of an AoI may leave ONE observation inside it, and one position has
    no drift -- `platform_drift_m` returns 0.0 below two points, by construction. Such a
    platform was classified fixed, handed to a product that (for `marineinsitu`) never fetches
    mobile classes at all, and its observation vanished between the two. Observed on the first
    live run: two of five platforms over Hobart, a drifter and a ship, each with a single
    in-AoI reading.

    So a DECLARED mobile class wins over a drift of zero. Where no class is declared -- the CSV
    and IOOS sources say nothing about platform type -- drift remains the only evidence there
    is, and a single-observation platform is genuinely indistinguishable from a mooring.
    """
    mobile, fixed = [], []
    for r in records:
        drift = insitu_acquire.platform_drift_m(r["df"])
        declared = str(r.get("platform_type") or "").strip().upper()
        declared_mobile = bool(declared) and declared not in FIXED_PLATFORM_TYPES
        if drift > min_drift_m or declared_mobile:
            mobile.append(r)
        else:
            fixed.append((r, drift))
    return mobile, fixed


def build_track_dataset(records: list[dict]) -> xr.Dataset:
    """Per-platform records -> ONE flat `(obs,)` Dataset, sorted by time.

    A FLAT LIST, not the `(station, time)` rectangle the fixed product writes. Two reasons, and
    the second is the load-bearing one:

      * Every observation carries its own position, so there is nothing for a station axis to
        factor out.
      * Reindexing several platforms onto a union time axis is what the fixed product does, and
        it is right there -- moorings report on regular, comparable schedules. Tracks do not. A
        glider sampling every 10 s and a ship logging hourly share almost no timestamps, so the
        union axis would be the SUM of their lengths and the block `S x T` -- quadratic in the
        number of platforms, and almost entirely NaN. Flat, the same data is `sum(len)` rows.

    `platform_id` is carried per observation rather than as a separate axis for the same reason:
    it is a label on a row, not a dimension of the data.
    """
    frames = []
    for r in records:
        df = r["df"]
        frames.append(pd.DataFrame({
            "time": pd.to_datetime(df["time"]),
            "lat": df["latitude"].to_numpy(dtype="float64"),
            "lon": df["longitude"].to_numpy(dtype="float64"),
            "sst": df["value"].to_numpy(dtype="float32"),
            "platform_id": str(r["id"]),
            "platform_type": str(r.get("platform_type") or "mobile"),
        }))
    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["time", "lat", "lon", "sst", "platform_id", "platform_type"])
    obs = obs.dropna(subset=["time", "lat", "lon", "sst"]).sort_values("time")

    ds = xr.Dataset(
        {"sst": (("obs",), obs["sst"].to_numpy(dtype="float32")),
         "lat": (("obs",), obs["lat"].to_numpy(dtype="float64")),
         "lon": (("obs",), obs["lon"].to_numpy(dtype="float64")),
         "platform_id": (("obs",), obs["platform_id"].to_numpy(dtype=object).astype("U32")),
         "platform_type": (("obs",),
                           obs["platform_type"].to_numpy(dtype=object).astype("U8"))},
        coords={"time": (("obs",), pd.DatetimeIndex(obs["time"]).to_numpy())})
    ds["sst"].attrs.update(units="degC", long_name="in-situ water temperature")
    ds["lat"].attrs["long_name"] = "observation latitude"
    ds["lon"].attrs["long_name"] = "observation longitude"
    return ds


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str) -> Path:
    """A flat observation table, not a raster -- same reasoning as insitu_acquire.write_output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return store.write_netcdf(ds, out_dir / f"{aoi_id}_insitu.nc",
                              encoding={"sst": {"zlib": True, "complevel": 4},
                                        "lat": {"zlib": True, "complevel": 4},
                                        "lon": {"zlib": True, "complevel": 4}})


def aligned_dir(root: Path, source: str, aoi_id: str) -> Path:
    return root / products.aligned_rel(
        products.spec(DataProduct.insitu_mobile).dir, source) / aoi_id


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    """Acquire moving-platform observations per (AoI, SOURCE), each in its own tree."""
    out_root, overwrite = eff["out_dir"], eff["overwrite"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    rep = report.ProductReport("insitu_mobile")

    for name in select_aois(grids, only_aoi):
        g = grids[name]
        ds_cfg = eff["ds"][name]
        sources = [s for s in ds_cfg["sources"] if only_source is None or s == only_source]

        for src in sources:
            out_dir = aligned_dir(out_root, src, name)
            out_path = out_dir / f"{name}_insitu.nc"
            # `covers=` for the same reason the fixed product needs it: one file spans the whole
            # window, so an EXTENDED range keeps the same filename and would be skipped.
            if store.done(out_path, store.REQUIRED_VARS["INSITU_MOBILE"],
                          covers=(start, end), overwrite=overwrite):
                log.info("=== %s [%s]: %s exists, skipping ===", name, src, out_path.name)
                rep.skip()
                continue

            log.info("=== AOI: %s | source=%s (mobile) ===", name, src)
            insitu_acquire._ensure_source_auth(src, eff, DataProduct.insitu_mobile)
            src_cfg = {**ds_cfg, "cache_dir": out_root / products.spec(
                DataProduct.insitu_mobile).dir / src / "_cache"}
            try:
                records = SOURCES[src](g, start, end, src_cfg, dry_run=dry_run)
            except Exception as exc:
                log.warning("  %s: %s mobile source failed (%s); no platforms from it",
                            name, src, exc)
                rep.fail(f"{name} {src}", f"{src} failed: {exc}")
                continue
            if dry_run:
                continue

            min_drift = ds_cfg["min_position_drift_m"]
            if min_drift is None:
                min_drift = g.resolution_m
            records, fixed = split_fixed_platforms(records, float(min_drift))
            for rec, drift in fixed:
                log.info("  %s does not move (%.0f m); it belongs to the `insitu` product, "
                         "not this one -- skipped here to avoid counting it twice",
                         rec["id"], drift)

            if not records:
                # NOT a failure, unlike the fixed product. Most coasts have no track crossing
                # them in a given window, and an AoI with nothing is the ordinary case rather
                # than a sign something broke.
                log.info("  %s [%s]: no moving platforms in this AoI window; nothing written",
                         name, src)
                rep.skip()
                continue

            ds = build_track_dataset(records)
            if ds.sizes.get("obs", 0) == 0:
                log.info("  %s [%s]: platforms found but no usable observations", name, src)
                rep.skip()
                continue
            ds.attrs.update(aoi_id=name, source=src,
                            qc_flags=str(ds_cfg["qc_flags"]),
                            **provenance.requested_range(start, end),
                            **provenance.stamp(eff))
            log.info("  wrote %s  (%d platform(s), %d observation(s)) [%s]",
                     write_output(ds, out_dir, name).name,
                     len(records), ds.sizes["obs"], src)
            rep.wrote(source=src)
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's mobile settings. Mirrors `insitu_acquire._ds_cfg`, sharing every source's
    knobs, and differs only in `min_position_drift_m` -- the same threshold read from the
    other side."""
    base = insitu_acquire._ds_cfg(opts)
    srcs = _opt(opts, "sources", None)
    if srcs is None:
        srcs = list(DEFAULT_SOURCES)
    elif isinstance(srcs, str):
        srcs = [srcs]
    base["sources"] = [str(s) for s in srcs]
    # `insitu_acquire._ds_cfg` reads `max_position_drift_m`; this product's knob is the
    # complementary one. None -> one grid cell, resolved per AoI where the grid is in hand.
    base["min_position_drift_m"] = _opt(opts, "min_position_drift_m", None)
    # Mobile platform classes, for the `marineinsitu` source. The fixed product's default is
    # the fixed classes; here it is deliberately absent so `fetch_aoi_mobile` takes "everything
    # that is not fixed" rather than a list that would have to be kept in sync.
    base["platform_types"] = list(_opt(opts, "platform_types", []) or [])
    return base


def _build_eff(project: Project) -> dict:
    opts = project.products.get(DataProduct.insitu_mobile)
    if opts is None:
        raise ValueError("insitu_mobile is not a selected product in this config")

    strategies = {}
    if getattr(project.auth, "copernicus", None) is not None:
        strategies["copernicus"] = project.auth.copernicus.auth_strategy

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.insitu_mobile))
               for a in project.all_areas},
        "auth_strategy": strategies,
        "out_dir": Path(project.output_dir),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, source=None) -> None:
    """Acquire moving-platform in-situ observations. Entry point for pipeline.py."""
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    bad = sorted(f"{n}:{s}" for n, c in eff["ds"].items() for s in c["sources"]
                 if s not in SOURCES)
    if bad:
        raise ValueError(f"insitu_mobile source not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(SOURCES)}.")

    auth.configure(project.auth)
    wanted = {s for c in eff["ds"].values() for s in c["sources"]
              if insitu_acquire.only_source_matches(s, source)}
    for backend in sorted({b for s in wanted
                           if (b := insitu_acquire._source_backend(s,
                                                                   DataProduct.insitu_mobile))}):
        strategy = eff["auth_strategy"].get(backend)
        if strategy:
            auth.login(backend, {"auth_strategy": strategy})
    return run(eff, grids, aois, dry_run, only_source=source)


def main():
    entry.process_main(acquire, "coastal_sst_data moving-platform in-situ acquisition.")


if __name__ == "__main__":
    main()
