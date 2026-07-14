#!/usr/bin/env python3
"""
coastal_sst_data -- provenance: what produced each field of a cube, and when.

A cube is a knitted-together artifact. Open one and you can see the channels, but not
which config built it, which source each field came from, or when those data were pulled
-- and a cube assembled today from files fetched months ago looks identical to one fetched
this morning. That matters here in particular because several products SWITCH SOURCE
underneath you: CMEMS falls back reanalysis->forecast, met falls back HRRR->ERA5, and
bathymetry falls back CUDEM->GMRT. "Where did this number come from" is a real question
with a per-day, per-AoI answer.

So every acquisition stage stamps its output files with `acquired_at`, and the assembler
harvests those stamps into the cube's own attributes:

    config_yaml          the FULL text of the config that built it (+ sha256)
    provenance           per FIELD: its inputs, their sources, when they were accessed
    provenance_products  per PRODUCT: source, file count, access window

Two design points worth stating, because both are about not lying:

  * ACCESS DATES ARE STAMPED, NOT GUESSED -- but data acquired before this existed has no
    stamp, so we fall back to the file's mtime and RECORD THAT WE DID (`basis`). An mtime
    is wrong the moment a tree is rsynced or restored from backup, so a date derived from
    one must never be passed off as a recorded one.
  * DERIVED FIELDS HAVE SEVERAL INPUTS. `eco_water_elev` is bathymetry AND tides AND the
    datum resolution AND the ECOSTRESS overpass that set the time. Picking one to report
    would be tidy and wrong; the record lists them all.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import xarray as xr

log = logging.getLogger(__name__)

STAMPED = "stamped"          # the file recorded its own acquisition time
FILE_MTIME = "file_mtime"    # ...it did not, so we used the filesystem's guess


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_version() -> str:
    from . import __version__
    return __version__


@lru_cache(maxsize=1)
def code_version() -> str:
    """Which CODE built this -- the git commit, not the package version.

    `package_version` is pinned in pyproject and does not move: every cube ever built is
    stamped "0.0.1" whatever commit produced it. That is fine for a release artifact and
    useless for provenance, because the code is exactly what changes between two cubes
    built from the same config. And it changes MEANINGFULLY: `mur_valid` used to include
    NN-filled pixels and now does not; `depth` used to be fabricated zeros where the DEM
    was missing and is now NaN. Two cubes with identical `config_sha256` and identical
    `package_version` can therefore hold different numbers, and nothing in either says so.

    A DIRTY tree is reported as such. A cube built from uncommitted edits is not
    reproducible from any commit, and claiming a bare SHA for it would be a lie -- the
    most dangerous kind here, because it is the one a reader would trust.

    Degrades to "unknown" outside a git checkout (an installed wheel, a container), which
    is honest: we would rather say we do not know than guess.
    """
    import subprocess
    repo = Path(__file__).resolve().parent

    def git(*args) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args],
                                 capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    if not sha:
        return "unknown"
    dirty = git("status", "--porcelain")
    return f"{sha}-dirty" if dirty else sha


def stamp(eff: dict | None = None) -> dict:
    """The attrs every acquisition write adds. `acquired_at` is evaluated HERE -- at write
    time -- so a long run dates each file honestly rather than all of them at start-up."""
    out = {"acquired_at": now_utc(), "package_version": package_version(),
           "code_version": code_version()}
    sha = (eff or {}).get("config_sha256")
    if sha:
        out["config_sha256"] = sha
    return out


def access_of(path: Path) -> tuple[str, str]:
    """(when this file's data was acquired, on what basis).

    Prefers the file's own `acquired_at` stamp. Falls back to its mtime -- and says so, so
    a downstream reader can tell a recorded date from a filesystem guess.
    """
    try:
        with xr.open_dataset(path) as ds:
            got = ds.attrs.get("acquired_at")
        if got:
            return str(got), STAMPED
    except Exception:                       # not readable as a Dataset (e.g. a GeoTIFF dir)
        pass
    mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return mtime.isoformat(timespec="seconds").replace("+00:00", "Z"), FILE_MTIME


def source_of(path: Path) -> str | None:
    try:
        with xr.open_dataset(path) as ds:
            return ds.attrs.get("source")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# field -> the product(s) that made it
# --------------------------------------------------------------------------- #
SENSORS = {"eco": "ecostress", "lst": "landsat", "modis": "modis"}

# Fields whose name is exactly this -> inputs.
_EXACT = {
    "tide": ["tides"], "tide_range": ["tides"],
    "airtemp": ["met"], "wind_u": ["met"], "wind_v": ["met"], "wind_speed": ["met"],
    "swrad": ["met"], "cloud_cover": ["met"],
    # which source served met on each day (see daily_sources); cmems_source is caught by
    # the cmems_ prefix rule below
    "met_source": ["met"],
    "depth": ["bathymetry"], "depth_p25": ["bathymetry"], "depth_p75": ["bathymetry"],
    "landmask": ["landcover", "bathymetry"],
    "landcover_water": ["landcover"],
    "insitu_sst": ["insitu", "met"],      # sampled at met's reference time
    "insitu_n": ["insitu"],
    "insitu_station": ["insitu"],
    # purely computed from the time axis -- no data source at all
    "doy_sin": [], "doy_cos": [],
}

_MET_VARS = ("airtemp", "wind_u", "wind_v", "wind_speed", "swrad", "cloud_cover")


def field_inputs(name: str) -> list[str]:
    """The product(s) a cube field was built from.

    Derived channels genuinely have several inputs and the record says so. An unmapped
    field returns [] and LOGS -- a channel added later that silently ships with blank
    provenance is exactly what this module exists to prevent.
    """
    if name in _EXACT:
        return list(_EXACT[name])
    if name.startswith("cmems_"):
        return ["cmems"]
    if name.startswith("mur_"):
        return ["mur"]

    m = re.match(r"^(eco|lst|modis)_(.+)$", name)
    if m:
        pre, rest = m.group(1), m.group(2)
        sensor = SENSORS[pre]
        if rest in ("sst", "cloud", "valid", "hour"):
            return [sensor]
        if rest in ("water_elev", "water_class"):
            # the DEM, the tide, the datum tying them together, and the overpass that
            # set the instant -- all four are load-bearing
            return ["bathymetry", "tides", "datum", sensor]
        if rest == "tide":
            return ["tides", sensor]
        if rest.startswith("insitu"):
            return ["insitu", sensor]
        if rest in _MET_VARS:
            return ["met", sensor]

    log.warning("provenance: no source mapping for field %r; it will be recorded with an "
                "empty input list. Add it to provenance.field_inputs.", name)
    return []


# --------------------------------------------------------------------------- #
# Harvest the aligned files
# --------------------------------------------------------------------------- #
def collect_product(d: Path, product: str) -> dict | None:
    """Source + access window for one product's aligned files in one AoI."""
    if not d.exists():
        return None
    files = sorted(f for f in d.glob("*.nc"))
    if not files:
        return None

    sources, accesses, bases = set(), [], set()
    for f in files:
        s = source_of(f)
        if s:
            sources.add(str(s))
        when, basis = access_of(f)
        accesses.append(when)
        bases.add(basis)

    return {
        "product": product,
        "sources": sorted(sources),
        "n_files": len(files),
        "accessed_first": min(accesses) if accesses else None,
        "accessed_last": max(accesses) if accesses else None,
        # If ANY file fell back to mtime, say so: the window is only as trustworthy as
        # its weakest entry.
        "basis": STAMPED if bases == {STAMPED} else FILE_MTIME,
    }


def daily_sources(d: Path, aoi_id: str, days, prefix: str = "") -> tuple[list[int], list[str]]:
    """Per-DAY source code for one product in one AoI, plus the legend naming each code.

    `collect_product` above unions a product's sources into a SET, which answers "which
    sources appear somewhere in this cube" and destroys the per-day answer this module's
    docstring promises. That union is exactly wrong for the products that switch source
    underneath you: a CMEMS product with 300 reanalysis days and 65 forecast days reports
    `[glorys, forecast]` and tells you nothing about any given day, and a met product that
    fell back to ERA5 for a fortnight in March looks identical to one that never did.

    So the cube carries the answer per timestep. Code 0 is always "none" (no file that
    day); legend[i] names code i. `prefix` selects a variant written into the same
    directory (met writes both `<aoi>_<date>.nc` and `<aoi>_ref_<date>.nc`), matched WHOLE
    so the two cannot be confused.
    """
    codes = [0] * len(days)
    legend = ["none"]
    if not d.exists():
        return codes, legend

    pat = re.compile(rf"^{re.escape(aoi_id)}_{re.escape(prefix)}(\d{{8}})\.nc$")
    idx = {dd.strftime("%Y%m%d"): i for i, dd in enumerate(days)}
    for f in sorted(d.glob(f"{aoi_id}_{prefix}*.nc")):
        m = pat.match(f.name)
        if not m or m.group(1) not in idx:
            continue
        s = str(source_of(f) or "unknown")
        if s not in legend:
            legend.append(s)
        codes[idx[m.group(1)]] = legend.index(s)
    return codes, legend


# The assembler's directory key for the tide product is singular; the product (and every
# field mapping) is plural. Normalize, or tide-derived fields would silently find no
# provenance record at all.
_DIR_KEY_ALIASES = {"tide": "tides"}


def collect(aligned_root: Path, aoi: str, product_dirs: dict) -> dict:
    """{product: record} for every product that actually wrote files for this AoI."""
    out = {}
    for dir_key, sub in product_dirs.items():
        product = _DIR_KEY_ALIASES.get(dir_key, dir_key)
        rec = collect_product(Path(aligned_root) / sub / "aligned" / aoi, product)
        if rec is not None:
            out[product] = rec
    # The datum stage writes a JSON sidecar, not a NetCDF; pick it up so water-level
    # fields can name the resolution that produced their offset.
    dj = Path(aligned_root) / "DATUM" / "aligned" / aoi / f"{aoi}_datum.json"
    if dj.exists():
        import json
        try:
            rec = json.loads(dj.read_text())
            out["datum"] = {"product": "datum",
                            "sources": [str(rec.get("method", "unknown"))],
                            "n_files": 1,
                            "accessed_first": rec.get("resolved_at"),
                            "accessed_last": rec.get("resolved_at"),
                            "basis": STAMPED}
        except ValueError:
            pass
    return out


def build(project, fields, products: dict) -> dict:
    """The full record: the config that built the cube, plus per-field and per-product.

    Fields whose inputs produced no files (the product was not run) are still listed, with
    an empty source list -- an absent channel and an unprovenanced one must look different.
    """
    per_field = {}
    for name in fields:
        inputs = field_inputs(name)
        recs = [products[p] for p in inputs if p in products]
        accessed = [r["accessed_last"] for r in recs if r["accessed_last"]]
        bases = {r["basis"] for r in recs}
        per_field[name] = {
            "inputs": inputs,
            "sources": sorted({s for r in recs for s in r["sources"]}),
            "accessed": max(accessed) if accessed else None,
            "basis": (STAMPED if bases == {STAMPED} else FILE_MTIME) if bases else None,
        }

    return {
        "created_at": now_utc(),
        "package_version": package_version(),
        "code_version": code_version(),
        "config_path": str(getattr(project, "config_path", None) or "") or None,
        "config_sha256": getattr(project, "config_sha256", None),
        "config_yaml": getattr(project, "config_text", None),
        "fields": per_field,
        "products": products,
    }
