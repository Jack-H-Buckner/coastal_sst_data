#!/usr/bin/env python3
"""
coastal_sst_data -- MODIS L2P SST as a LANDSAT-COINCIDENT CALIBRATION REFERENCE.

Same instrument, same collection and the same swath machinery as `processes.modis` -- this
module imports all of it rather than forking it. What differs is the QUESTION being asked, and
it is different enough to be its own product:

    modis      "what was the SST over this AoI, per overpass"      -> a thermal SENSOR
    modis_ref  "what did MODIS see AT THE MOMENT LANDSAT DID"      -> a calibration REFERENCE

That difference drives three things this module owns and `modis` does not:

  * COINCIDENCE. Only granules within `max_time_diff_minutes` of an already-acquired Landsat
    scene are loaded, read from Landsat's own aligned filenames. This is the one place where
    one product parses another product's filenames, and it is why the spec declares
    `depends_on=(landsat,)`.
  * FOOTPRINT IDS. Every grid cell records WHICH native ~1 km MODIS observation it was drawn
    from (`modisref_footprint_id`), so a downstream fit can aggregate the fine Landsat field
    to genuine MODIS footprints instead of to arbitrary blocks.
  * WHOLE GRANULES. Those ids are indices into the NATIVE swath, so a server-side subset --
    which renumbers them -- would make the channel a lie. Hence `access` defaults to
    `download` here, and `harmony` is refused outright while footprints are on.

Because footprints restart at 0 in every granule, this product is also the one sensor that is
NOT mosaicked (see the spec) -- a day keeps its clearest granule.

Output: one aligned NetCDF per granule at
    <output_dir>/MODIS_REF/aligned/<aoi>/<aoi>_<YYYYMMDDTHHMMSS>.nc

The calibration itself is out of scope here; this module only LOADS and grids the reference.

Usage:
    python -m coastal_sst_data.processes.modis_ref --config config.yaml
    python -m coastal_sst_data.processes.modis_ref --config config.yaml --full-series
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import earthaccess

from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import auth, entry, naming, net, provenance, report, store
from .modis import (
    _ACCESS,
    _access,
    _day_night,
    harmony_variables,
    read_swath,
    resample_to_grid,
    resolve_night_hours,
    resolve_time_of_day,
    select_by_time_of_day,
    solar_hour,
    _scene_dataset,
)

log = logging.getLogger(__name__)

SOURCE = "modis_ref"
# Terra by default: its 10:30 descending crossing was chosen to sit within minutes of
# Landsat's, which is what makes a same-day matchup meaningful at all. Aqua's 13:30 is three
# hours off and would fold the diurnal warming cycle into the calibration.
SHORT_NAME = "MODIS_T-JPL-L2P-v2019.0"
DEFAULT_MAX_TIME_DIFF_MIN = 360               # +/- 6 h Landsat<->MODIS overpass


# --------------------------------------------------------------------------- #
# Coincidence helpers
# --------------------------------------------------------------------------- #
def _landsat_times(landsat_dir: Path, aoi: str) -> list[datetime]:
    """Acquisition times of the Landsat aligned files already written for an AoI.

    Reads the stamp Landsat WROTE via the shared convention (coastal_sst_data.naming), so
    the two cannot drift: this coincidence filter is the one place where one product's
    filenames are parsed by another product's code.
    """
    d = landsat_dir / aoi
    if not d.exists():
        return []
    return [t for f in d.glob(f"{aoi}_*T*.nc")
            if (t := naming.parse_time(f.name)) is not None]


def select_coincident(kept, ls_times, max_dt: timedelta):
    """Narrow [(granule, time), ...] to those within `max_dt` of any Landsat scene."""
    out = []
    for gr, t in kept:
        nearest = min((abs(t - lt) for lt in ls_times), default=None)
        if nearest is not None and nearest <= max_dt:
            out.append((gr, t))
    return out


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(eff: dict, grids: dict[str, AoiGrid], only_aoi, dry_run):
    grid_cfg = eff["grid"]
    out_root, tmp_dir, fmt, overwrite = eff["out_dir"], eff["tmp_dir"], eff["fmt"], eff["overwrite"]
    landsat_dir = eff["landsat_dir"]
    to_celsius = grid_cfg.get("to_celsius", False)
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    auth.login("earthdata", eff["earthdata"])

    names = select_aois(grids, only_aoi)

    rep = report.ProductReport("modis_ref")

    for name in names:
        g = grids[name]
        ds_cfg = eff["ds"][name]
        variable, quality_min = ds_cfg["variable"], ds_cfg["quality_min"]
        radius, access = ds_cfg["regrid_radius_m"], ds_cfg["access"]
        match_landsat = ds_cfg["match_landsat"]
        max_dt = timedelta(minutes=ds_cfg["max_time_diff_minutes"])
        time_of_day, night_hours = ds_cfg["time_of_day"], ds_cfg["night_solar_hours"]
        do_footprint = ds_cfg["footprint_id"]
        short_name = ds_cfg["short_name"]
        fetch = _ACCESS[access]
        want_vars = harmony_variables(variable)
        lon_c = 0.5 * (g.search_bbox[0] + g.search_bbox[2])

        log.info("=== AOI: %s (CRS=%s grid=%dx%d) | match_landsat=%s access=%s ===",
                 name, g.target_crs, g.width, g.height, match_landsat, access)

        ls_times = _landsat_times(landsat_dir, name) if match_landsat else []
        if match_landsat and not ls_times:
            log.warning("  %s: match_landsat on but no Landsat aligned files in %s; "
                        "run Landsat first (or set match_landsat: false)", name, landsat_dir / name)
            continue

        granules = net.retry(
            lambda: earthaccess.search_data(
                short_name=short_name, temporal=(start, end),
                bounding_box=tuple(g.search_bbox)),
            what=f"MODIS_REF search {name}",
            refresh=auth.refresher("earthdata", eff["earthdata"]))
        kept = select_by_time_of_day(granules, time_of_day=time_of_day,
                                     night_hours=night_hours, lon=lon_c)
        if match_landsat:
            kept = select_coincident(kept, ls_times, max_dt)
        log.info("  %d granule(s) over AOI -> %d after the %s/coincidence filter",
                 len(granules), len(kept), time_of_day)
        if not kept:
            continue
        if dry_run:
            log.info("  [dry-run] would process %d MODIS granule(s)", len(kept))
            continue

        aoi_out = out_root / name
        for gi, (gr, t) in enumerate(kept, 1):
            # Proactive: top the credential up between granules rather than discovering it
            # died four hours in.
            if gi % auth.CHECK_EVERY == 0:
                auth.ensure_fresh("earthdata", eff["earthdata"])
            tstr = naming.time_stamp(t)
            stem = naming.time_stem(name, t)
            if store.done(aoi_out / f"{stem}.nc", store.REQUIRED_VARS["MODIS_REF"],
                          shape=(g.height, g.width), overwrite=overwrite):
                log.info("  %s already processed, skipping", tstr)
                rep.skip()
                continue
            # One scratch dir per granule, removed in `finally`. A partial download can
            # therefore never survive to be mistaken for a complete one by the next run.
            gran_tmp = tmp_dir / f"g_{name}_{tstr}"
            try:
                path = fetch(gr, g.search_bbox, gran_tmp, variables=want_vars)
                sst, lat, lon = read_swath(path, variable, quality_min, to_celsius)
                fp = np.arange(sst.size, dtype="int32").reshape(sst.shape) if do_footprint else None
                sst_g, fp_g = resample_to_grid(sst, lat, lon, g, radius, fp)
            except Exception as exc:
                log.warning("  FAILED %s (%s)", tstr, exc)
                rep.fail(f"{name} {tstr}", exc)
                continue
            finally:
                shutil.rmtree(gran_tmp, ignore_errors=True)
            if not np.isfinite(sst_g).any():
                log.info("  %s: no valid MODIS pixels over AOI, skipping", tstr)
                continue
            ds = _scene_dataset(sst_g, fp_g, g, t, name, to_celsius,
                                short_name=short_name, solar_h=solar_hour(t, lon_c),
                                day_night=_day_night(gr))
            ds.attrs.update(**provenance.stamp(eff))
            log.info("  wrote %s", store.write_output(ds, aoi_out, stem, fmt))
            rep.wrote(source=f"GHRSST {short_name}")
    rep.log_summary()
    return rep


# --------------------------------------------------------------------------- #
# Config adapter + pipeline entry point
# --------------------------------------------------------------------------- #
def _ds_cfg(opts) -> dict:
    """One AoI's settings. MODIS is global, so nothing here is region-overridable."""
    access = _access(opts, "download")
    footprint = bool(_opt(opts, "footprint_id", True))
    if access == "harmony" and footprint:
        # A Harmony subset TRIMS the swath, so `np.arange(sst.size)` over the subset indexes
        # the SUBSET's pixels, not the granule's. Two AoIs -- or two runs with different
        # bounds -- would then write the same id for different native observations, and the
        # footprint-median matchup this product exists to enable would silently group the
        # wrong cells. Refuse rather than emit a channel that means nothing.
        raise ValueError(
            "modis_ref access='harmony' cannot be combined with footprint_id=true: a "
            "server-side subset renumbers the native swath indices the footprint ids ARE. "
            "Use access='download', or set footprint_id: false if you only need the SST.")
    return {
        "short_name": _opt(opts, "short_name", SHORT_NAME),
        "variable": _opt(opts, "variable", "sea_surface_temperature"),
        "quality_min": int(_opt(opts, "quality_min", 4)),
        "regrid_radius_m": float(_opt(opts, "regrid_radius_m", 1500.0)),
        "access": access,
        "match_landsat": bool(_opt(opts, "match_landsat", True)),
        "max_time_diff_minutes": float(_opt(opts, "max_time_diff_minutes",
                                             DEFAULT_MAX_TIME_DIFF_MIN)),
        # Landsat flies in the morning, so the reference is a DAYTIME product -- the opposite
        # default from the standalone sensor, and for the same reason: match what is being
        # calibrated.
        "time_of_day": resolve_time_of_day(opts, default="day"),
        "night_solar_hours": resolve_night_hours(opts),
        "footprint_id": footprint,
    }


def _build_eff(project: Project) -> dict:
    """Map a validated Project into the flat `eff` dict `run()` consumes."""
    opts = project.products.get(DataProduct.modis_ref)
    if opts is None:
        raise ValueError("modis_ref is not a selected product in this config")
    if project.auth.earthdata is None:            # guaranteed by config validation
        raise ValueError("modis_ref requires an auth.earthdata block")

    grid_cfg = project.grid.model_dump()
    grid_cfg.setdefault("to_celsius", False)      # GridSpec has no such field yet

    root = Path(project.output_dir)
    return {
        "config_sha256": project.config_sha256,
        "ds": {a.name: _ds_cfg(resolve_opts(project, a.name, DataProduct.modis_ref))
               for a in project.all_areas},
        "grid": grid_cfg,
        "out_dir": root / "MODIS_REF" / "aligned",
        "landsat_dir": root / "LANDSAT" / "aligned",   # coincidence source
        "tmp_dir": root / "MODIS_REF" / "_tmp",
        "fmt": _opt(opts, "output_format", "netcdf"),
        "overwrite": bool(_opt(opts, "overwrite", False)),
        "earthdata": {"auth_strategy": project.auth.earthdata.auth_strategy},
        "time": {
            "start_date": project.time.start_date.isoformat(),
            "end_date": project.time.end_date.isoformat(),
        },
    }


def acquire(project: Project, *, grids=None, aois=None, dry_run=False,
            overwrite=False, full_series=False) -> None:
    """Acquire the MODIS calibration reference. Entry point for pipeline.py.

    full_series=True forces match_landsat off (load the whole time series). With
    match_landsat on this must run AFTER Landsat so its aligned files exist -- which the
    spec's `depends_on` guarantees for a pipeline run.
    """
    eff = _build_eff(project)
    # Credentials expire and runs are long: apply this project's refresh policy before any
    # network call. acquire() is the one entry point every invocation path goes through.
    auth.configure(project.auth)
    if overwrite:
        eff["overwrite"] = True
    if full_series:
        for ds_cfg in eff["ds"].values():     # `ds` is per-AoI
            ds_cfg["match_landsat"] = False
    if grids is None:
        grids = project_grids(project)
    return run(eff, grids, aois, dry_run)


def main():
    entry.process_main(
        acquire, "coastal_sst_data MODIS Landsat-coincident calibration reference.",
        extra=[entry.Flag("--full-series",
                          "ignore Landsat coincidence; load the full MODIS time series")])


if __name__ == "__main__":
    main()
