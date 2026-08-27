#!/usr/bin/env python3
"""
coastal_sst_data -- in-situ acquisition: the per-(AoI, source) loop every network shares.

In-situ is the cube's only GROUND TRUTH, and there is more than one place it comes from: a
public network the pipeline discovers for you (IOOS/ERDDAP) and the thermometers the user
deployed themselves and holds in a file. Those are not two pipes to the same observations --
they are DIFFERENT PLATFORMS -- so they STACK (D10): each configured source is acquired
independently into its own `INSITU/<source>/aligned/<aoi>/` tree, and the assembler merges
them into one station table.

This module owns everything that is not network-specific:

  * the (AoI x source) loop, the skip guard, and the atomic write;
  * `build_dataset` -- per-platform series -> one (station, time) Dataset;
  * the report, so a source that yields nothing is VISIBLE rather than an empty channel.

Each source owns only its fetch, behind one seam:

    fetch_aoi(g, start, end, cfg, dry_run=False) -> [record, ...]

    record = {"id", "title", "var", "lat", "lon", "df"}
    df     = DataFrame[time, latitude, longitude, value, (qc)]

so a new network is a fetch function and a registry entry, not another copy of this loop.

    <output_dir>/INSITU/<source>/aligned/<aoi>/<aoi>_insitu.nc   dims (station, time)

The NATIVE sampling interval is kept (6 min for CO-OPS): the assembler needs the sub-hourly
series to match a satellite overpass instant, exactly as `tides` writes a series that
`water_level` later samples.

Usage:
    python -m coastal_sst_data.processes.insitu_acquire --config config.yaml
    python -m coastal_sst_data.processes.insitu_acquire --config config.yaml --aoi hood_canal
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
from . import insitu_cmems, insitu_csv, insitu_ioos

log = logging.getLogger(__name__)

# IOOS only, unless the config says otherwise. NOT "every known source": `csv` cannot do
# anything without a `path`, and `marineinsitu` needs a Copernicus credential, so both have to
# be opted into rather than defaulted on.
#
# READ FROM THE REGISTRY, not spelled again here. The auth preflight has to know the same list
# -- an unset `sources` means THIS, and if the preflight assumed something wider it would demand
# credentials for sources the run will never acquire (see config.required_backend).
DEFAULT_SOURCES = list(products.spec(DataProduct.insitu).default_sources)

SOURCES = {
    "ioos": insitu_ioos.fetch_aoi,
    "csv": insitu_csv.fetch_aoi,
    "marineinsitu": insitu_cmems.fetch_aoi,
}


# --------------------------------------------------------------------------- #
# The moving-platform guard
# --------------------------------------------------------------------------- #
def platform_drift_m(df: pd.DataFrame) -> float:
    """How far this platform's observations stray from its nominal (median) position, in m.

    A local equirectangular approximation is ample: the question is "tens of metres or
    kilometres", and it is asked about points that are by assumption nearly co-located.
    """
    lat = df["latitude"].to_numpy(dtype="float64")
    lon = df["longitude"].to_numpy(dtype="float64")
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return 0.0
    lat, lon = lat[ok], lon[ok]
    lat0, lon0 = float(np.median(lat)), float(np.median(lon))
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = m_per_deg_lat * float(np.cos(np.radians(lat0)))
    dy = (lat - lat0) * m_per_deg_lat
    dx = (lon - lon0) * m_per_deg_lon
    return float(np.max(np.hypot(dx, dy)))


def split_moving_platforms(records: list[dict], max_drift_m: float):
    """(stationary, [(record, drift_m), ...]) -- the platforms that MOVE are separated out.

    The cube's in-situ model is FIXED-STATION: one position per platform, one pixel, for the
    whole window. Handed a track it would place every observation in the median pixel, which
    does not fail -- it produces confidently WRONG matchups, the one outcome this product
    cannot tolerate. So a platform whose observations stray beyond `max_drift_m` (by default
    one grid cell, so anything that never leaves its own pixel passes) is DROPPED and
    reported, never quietly averaged.

    Moving platforms -- drifters, gliders, ship transects -- need per-observation placement,
    which is a change to the shared data model and the cube schema. Until then this is the
    honest answer.
    """
    stationary, moving = [], []
    for r in records:
        drift = platform_drift_m(r["df"])
        (moving.append((r, drift)) if drift > max_drift_m else stationary.append(r))
    return stationary, moving


def _moving_platform_message(rec: dict, drift_m: float) -> str:
    """Why this platform was dropped -- naming BOTH plausible causes.

    For a CSV the likelier explanation is not a drifter at all: it is a file holding several
    stations whose `station_id` column was never mapped, so every station was merged into one
    platform spanning the distance between them.
    """
    return (f"platform {rec['id']!r} spans {drift_m:,.0f} m; moving platforms "
            "(drifters, gliders, transects) are not supported yet. If this file holds "
            "SEVERAL stations, map the `station_id` column so they are not merged into one.")


# --------------------------------------------------------------------------- #
# Per-platform series -> (station, time)
# --------------------------------------------------------------------------- #
def build_dataset(records: list[dict]) -> xr.Dataset:
    """Per-platform series -> one (station, time) Dataset on the union time axis."""
    times = pd.DatetimeIndex(sorted({t for r in records for t in r["df"]["time"]}))
    S, T = len(records), len(times)

    sst = np.full((S, T), np.nan, dtype="float32")
    qc = np.zeros((S, T), dtype="uint8")
    for i, r in enumerate(records):
        df = r["df"].drop_duplicates(subset="time").set_index("time").reindex(times)
        sst[i] = df["value"].to_numpy(dtype="float32")
        if "qc" in df:
            qc[i] = np.nan_to_num(df["qc"].to_numpy(dtype="float32"), nan=0).astype("uint8")

    ds = xr.Dataset(
        {"sst": (("station", "time"), sst), "qc": (("station", "time"), qc)},
        coords={
            "station": np.arange(S, dtype="int32"),
            "time": times,
            "station_id": ("station", [r["id"] for r in records]),
            "station_name": ("station", [r["title"] for r in records]),
            "lat": ("station", np.array([r["lat"] for r in records], "float64")),
            "lon": ("station", np.array([r["lon"] for r in records], "float64")),
            "variable": ("station", [r["var"] for r in records]),
        },
    )
    ds["sst"].attrs.update(units="degC", long_name="in-situ water temperature")
    ds["qc"].attrs.update(long_name="QARTOD aggregate flag",
                          flag_values="1 pass, 2 not_evaluated, 3 suspect, 4 fail, 9 missing")
    return ds


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str) -> Path:
    """In-situ is a (station, time) table, not a raster: it does NOT go through
    store.write_output -- there is no y/x for a GeoTIFF to hold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return store.write_netcdf(ds, out_dir / f"{aoi_id}_insitu.nc",
                              encoding={"sst": {"zlib": True, "complevel": 4},
                                        "qc": {"zlib": True, "complevel": 4}})


# --------------------------------------------------------------------------- #
# Auth, for the sources that need it
# --------------------------------------------------------------------------- #
# Which backend each source authenticates to, read from the registry so it cannot drift from
# what the config preflight validated. `ioos` and `csv` are public and map to None.
def _source_backend(src: str, product: DataProduct = DataProduct.insitu) -> "str | None":
    auth_map = products.spec(product).auth
    return auth_map.get(src) if isinstance(auth_map, dict) else auth_map


def _ensure_source_auth(src: str, eff: dict,
                        product: DataProduct = DataProduct.insitu) -> None:
    """Re-validate this source's credential, if it has one.

    In-situ used to need no auth at all, so `acquire` called none. `marineinsitu` reaches the
    Copernicus Marine Data Store -- the same backend `cmems` uses -- and a multi-year, many-AoI
    in-situ run is long enough to outlive a credential. Called at the (AoI, source) boundary
    because that is a point between writes, so a refresh cannot land mid-file.
    """
    backend = _source_backend(src, product)
    strategy = eff.get("auth_strategy", {}).get(backend) if backend else None
    if backend and strategy:
        auth.ensure_fresh(backend, {"auth_strategy": strategy})


def aligned_dir(root: Path, source: str, aoi_id: str) -> Path:
    """`<out>/INSITU/<source>/aligned/<aoi>` -- built through the registry helper so the
    write side and the assembler's `ctx.adir` cannot drift apart."""
    return root / products.aligned_rel(products.spec(DataProduct.insitu).dir, source) / aoi_id


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run, only_source=None):
    """Acquire in-situ observations per (AoI, SOURCE), each in its own `<source>/` tree.

    No fallback (D10): each configured source is produced independently, so a region with no
    public buoys still gets the user's own file, and vice versa. `only_source` narrows the run
    to one.
    """
    out_root, overwrite = eff["out_dir"], eff["overwrite"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    rep = report.ProductReport("insitu")

    for name in select_aois(grids, only_aoi):
        g = grids[name]
        # Resolved PER AoI: station lists and file paths are inherently local, and `variables`
        # is a per-NETWORK naming preference (sea_water_temperature vs sea_surface_temperature)
        # -- every network still yields the one `insitu_sst` channel, so varying it by region
        # does not change what the cube means.
        ds_cfg = eff["ds"][name]
        sources = [s for s in ds_cfg["sources"] if only_source is None or s == only_source]

        for src in sources:
            out_dir = aligned_dir(out_root, src, name)
            out_path = out_dir / f"{name}_insitu.nc"
            # `covers=` because in-situ writes ONE file for the whole window: without it an
            # EXTENDED date range keeps the same filename, passes `is_complete`, and is
            # silently skipped (see DEVELOPMENT.md §3b).
            if store.done(out_path, store.REQUIRED_VARS["INSITU"],
                          covers=(start, end), overwrite=overwrite):
                log.info("=== %s [%s]: %s exists, skipping ===", name, src, out_path.name)
                rep.skip()
                continue

            log.info("=== AOI: %s | source=%s ===", name, src)
            # A credentialed source outlives its credential on a long run, so it is re-checked
            # at each (AoI, source) boundary -- a safe point, between writes -- exactly as
            # `cmems.run` does. No-op for the public sources.
            _ensure_source_auth(src, eff)
            # Downloaded originals are shared across AoIs and runs: a platform's `history` file
            # is its WHOLE LIFE, so two adjacent AoIs routinely want the same one, and a
            # re-acquisition should not re-fetch what is already on disk.
            src_cfg = {**ds_cfg, "cache_dir": out_root / products.spec(
                DataProduct.insitu).dir / src / "_cache"}
            try:
                records = SOURCES[src](g, start, end, src_cfg, dry_run=dry_run)
            except Exception as exc:
                log.warning("  %s: %s in-situ source failed (%s); no stations from it",
                            name, src, exc)
                rep.fail(f"{name} {src}", f"{src} failed: {exc}")
                continue
            if dry_run:
                continue

            # A platform that MOVES cannot be honoured by a fixed-station cube, and quietly
            # collapsing it to its median position would produce wrong matchups rather than
            # absent ones. Drop it, loudly. Default tolerance: one grid cell -- a platform
            # that never leaves its own pixel is exactly what this model can represent.
            max_drift = ds_cfg["max_position_drift_m"]
            if max_drift is None:
                max_drift = g.resolution_m
            records, moving = split_moving_platforms(records, float(max_drift))
            for rec, drift in moving:
                log.warning("  dropped %s", _moving_platform_message(rec, drift))
                rep.fail(f"{name} {src} {rec['id']}", _moving_platform_message(rec, drift))

            # A source that yields nothing must be VISIBLE: an empty in-situ channel that
            # looks like "no buoys here" is the failure mode this product cannot afford.
            if not records:
                log.warning("  %s [%s]: NO usable in-situ platforms; nothing written",
                            name, src)
                rep.fail(f"{name} {src}", "no usable platforms")
                continue

            ds = build_dataset(records)
            ds.attrs.update(aoi_id=name, source=src,
                            qc_flags=str(ds_cfg["qc_flags"]),
                            **provenance.requested_range(start, end),
                            **provenance.stamp(eff))
            log.info("  wrote %s  (%d platform(s), %d timestep(s)) [%s]",
                     write_output(ds, out_dir, name).name,
                     ds.sizes["station"], ds.sizes["time"], src)
            rep.wrote(source=src)
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's in-situ settings: which SOURCES to STACK (region-overridable), plus each
    network's knobs.

    `columns`/`units`/`qc_pass_values` stay project-global (they decide what the channel
    MEANS); `path`, the station allow/deny lists and `variables` may vary by region, because
    which file holds this region's data -- and which platforms exist here -- is a fact about
    the world.
    """
    srcs = _opt(opts, "sources", None)
    if srcs is None:
        srcs = list(DEFAULT_SOURCES)
    elif isinstance(srcs, str):
        srcs = [srcs]
    return {
        "sources": [str(s) for s in srcs],
        # shared across networks
        "qc_flags": [int(f) for f in _opt(opts, "qc_flags", insitu_ioos.DEFAULT_QC_FLAGS)],
        "max_sensor_depth_m": _opt(opts, "max_sensor_depth_m",
                                   insitu_ioos.DEFAULT_MAX_SENSOR_DEPTH_M),
        "stations": list(_opt(opts, "stations", []) or []),
        "exclude_stations": list(_opt(opts, "exclude_stations", []) or []),
        # None -> one grid cell, resolved per AoI where the grid is in hand.
        "max_position_drift_m": _opt(opts, "max_position_drift_m", None),
        # ioos
        "variables": list(_opt(opts, "variables", insitu_ioos.DEFAULT_VARIABLES)),
        "pad_deg": float(_opt(opts, "pad_deg", insitu_ioos.DEFAULT_PAD_DEG)),
        # marineinsitu (Copernicus Marine In-Situ TAC). `dataset_id` accepts a region shorthand
        # ("nws") as well as a full dataset id; the module resolves it.
        "dataset_id": _opt(opts, "dataset_id", insitu_cmems.DEFAULT_DATASET_ID),
        "dataset_part": _opt(opts, "dataset_part", insitu_cmems.DEFAULT_PART),
        "platform_types": list(_opt(opts, "platform_types",
                                    insitu_cmems.DEFAULT_PLATFORM_TYPES)),
        # csv
        "path": _opt(opts, "path", None),
        "columns": dict(_opt(opts, "columns", {}) or {}),
        "time_zone": _opt(opts, "time_zone", "UTC"),
        "units": _opt(opts, "units", insitu_csv.DEFAULT_UNITS),
        "qc_pass_values": _opt(opts, "qc_pass_values", None),
        "default_station_id": _opt(opts, "default_station_id", None),
    }


def _build_eff(project: Project) -> dict:
    opts = project.products.get(DataProduct.insitu)
    if opts is None:
        raise ValueError("insitu is not a selected product in this config")

    # The auth STRATEGY per backend, not the secret -- `auth.login` reads the credential itself
    # (~/.netrc, the environment, or a prompt). Only backends this product's sources can use,
    # and only when the config actually configured one.
    strategies = {}
    if getattr(project.auth, "copernicus", None) is not None:
        strategies["copernicus"] = project.auth.copernicus.auth_strategy

    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.insitu))
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
    """Acquire in-situ observations. Entry point for pipeline.py.

    Every configured source is acquired and STACKED into its own `INSITU/<source>/` tree;
    `source` (unset on a pipeline run) narrows to one. Sources are resolved PER AoI (region
    override -> project default) and validated against the SOURCES registry, so a typo fails
    loudly -- before a single station is queried.
    """
    eff = _build_eff(project)
    if overwrite:
        eff["overwrite"] = True
    if grids is None:
        grids = project_grids(project)

    bad = sorted(f"{n}:{s}" for n, c in eff["ds"].items() for s in c["sources"]
                 if s not in SOURCES)
    if bad:
        raise ValueError(f"insitu source not recognized ({', '.join(bad)}); "
                         f"choose from {sorted(SOURCES)}.")

    # Credentials expire and runs are long: apply this project's refresh policy, then log in
    # ONCE per run for whichever backends the configured sources actually need. A run of
    # `ioos`/`csv` alone authenticates to nothing, exactly as before.
    auth.configure(project.auth)
    wanted = {s for c in eff["ds"].values() for s in c["sources"]
              if only_source_matches(s, source)}
    for backend in sorted({b for s in wanted if (b := _source_backend(s))}):
        strategy = eff["auth_strategy"].get(backend)
        if strategy:
            auth.login(backend, {"auth_strategy": strategy})
    return run(eff, grids, aois, dry_run, only_source=source)


def only_source_matches(src: str, only_source) -> bool:
    """Whether `src` will actually be acquired on this invocation.

    The per-source CLIs narrow a run to one source, and logging in for the others would prompt
    (or fail) for a credential this invocation has no use for.
    """
    return only_source is None or src == only_source


def main():
    entry.process_main(acquire, "coastal_sst_data in-situ acquisition (IOOS + user CSV).")


if __name__ == "__main__":
    main()
