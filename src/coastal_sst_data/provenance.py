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

from . import products

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
# DERIVED from the product registry: a sensor declares its channel prefix ONCE (in its
# ProductSpec.sensor) and every cube channel it names is attributed automatically. Adding a
# fourth thermal sensor used to mean remembering to extend this dict AND the regex below --
# and forgetting either was silent, because an unmapped field does not raise, it just ships
# with a blank provenance record.
SENSORS: dict[str, str] = {s.sensor.prefix: s.product.value for s in products.sensors()}

# `^(eco|lst|modis)_(.+)$`, built from the registry rather than written out.
_SENSOR_RE = re.compile(
    rf"^({'|'.join(re.escape(p) for p in SENSORS)})_(.+)$") if SENSORS else None

# Fields whose name is exactly this -> inputs.
_EXACT = {
    "tide": ["tides"], "tide_range": ["tides"],
    "airtemp": ["met"], "wind_u": ["met"], "wind_v": ["met"], "wind_speed": ["met"],
    "swrad": ["met"], "cloud_cover": ["met"],
    "elevation": ["bathymetry"],
    "depth": ["bathymetry"], "depth_p25": ["bathymetry"], "depth_p75": ["bathymetry"],
    "landcover_water": ["landcover"],
    "insitu_sst": ["insitu", "met"],      # sampled at met's reference time
    "insitu_n": ["insitu"],
    "insitu_station": ["insitu"],
    # purely computed from the time axis -- no data source at all
    "doy_sin": [], "doy_cos": [],
}

_MET_VARS = ("airtemp", "wind_u", "wind_v", "wind_speed", "swrad", "cloud_cover")


def _matching_source_tag(name: str, sources_by_tag: dict) -> str | None:
    """The source tag a per-source channel belongs to: the longest tag `t` such that `name`
    ends with `_<t>` (longest wins, so `tide_range_eo_tides` picks `eo_tides`, not a stray
    shorter tag). None when the channel names no known source tag."""
    best = None
    for tag in sources_by_tag:
        if name.endswith("_" + tag) and (best is None or len(tag) > len(best)):
            best = tag
    return best


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
    # Per-source bathymetry (D4/D5): `elevation_cudem`, `depth_gmrt`, `depth_p25_<src>`, ...
    if any(name.startswith(v + "_") for v in ("elevation", "depth", "depth_p25", "depth_p75")):
        return ["bathymetry"]

    m = _SENSOR_RE.match(name) if _SENSOR_RE else None
    if m:
        pre, rest = m.group(1), m.group(2)
        sensor = SENSORS[pre]
        # The sensor's own DATA channels: `<pre>_sst`/`_cloud`/`_valid`/`_hour`, and -- for a
        # STACKED-DATA sensor (ECOSTRESS) -- their per-version forms `<pre>_sst_v002`, etc. The
        # exact source tag is narrowed later in `build` via sources_by_tag, so here it is enough
        # to attribute the channel to the sensor product. The base token (before any `_<ver>`)
        # distinguishes these from the matchup channels (insitu/met/tide) handled below.
        base = rest.split("_", 1)[0]
        if base in ("sst", "cloud", "valid", "hour"):
            return [sensor]
        if rest.startswith("insitu"):
            return ["insitu", sensor]
        # overpass met `<sensor>_<metvar>_<src>` (e.g. eco_airtemp_hrrr) -- from met_overpass.
        if any(rest.startswith(v + "_") for v in _MET_VARS):
            return ["met_overpass", sensor]
        # overpass tide `<sensor>_tide_<src>` (e.g. eco_tide_coops) -- derived from the tides
        # series interpolated to this sensor's overpass (tide_overpass contributor).
        if rest.startswith("tide_"):
            return ["tides", sensor]
        # POST-ASSEMBLY water line `<sensor>_water_elev` / `<sensor>_water_class` (the
        # preprocess.water_line step): the DEM + this sensor's overpass tide. A DERIVED field,
        # so it names ALL its inputs.
        if rest.startswith("water_"):
            return ["bathymetry", "tides", sensor]

    # Per-source forcing met `<metvar>_<src>` (airtemp_hrrr) and tides `tide_<src>` /
    # `tide_range_<src>` (D4/D5). Match by PREFIX -- a source token can itself contain an
    # underscore (`eo_tides`), so splitting off the last token would misread the base.
    if any(name.startswith(v + "_") for v in _MET_VARS):
        return ["met"]
    if name.startswith("tide_"):
        return ["tides"]

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


def collect(aligned_root: Path, aoi: str, product_dirs: dict) -> dict:
    """{product: record} for every product that actually wrote files for this AoI.

    `product_dirs` is keyed by the PRODUCT's own name (products.product_dirs()). It used to
    be keyed by the assembler's own idea of the name, which called the tide product `tide`
    while every field mapping called it `tides` -- so an alias table existed here purely to
    reconcile two hand-maintained lists that disagreed. With one registry there is one name,
    and the alias is gone.
    """
    by_value = {s.product.value: s for s in products.REGISTRY}
    out = {}
    for product, sub in product_dirs.items():
        s = by_value.get(product)
        if s is not None and s.is_stacked_data:
            # DATA product: gather across its per-source trees (globbed, so config-registered
            # sources are covered), keyed by the source TAG (the `<source>` dir name). The
            # merged record carries `sources_by_tag`, so `build` can attribute a per-source
            # channel (`airtemp_hrrr`) to its ONE source unambiguously rather than the union.
            base = Path(aligned_root) / sub
            by_tag = {sd.parent.name: r
                      for sd in (sorted(base.glob("*/aligned")) if base.exists() else [])
                      if (r := collect_product(sd / aoi, product)) is not None}
            rec = _merge_records(by_tag, product) if by_tag else None
        else:
            rec = collect_product(Path(aligned_root) / sub / "aligned" / aoi, product)
        if rec is not None:
            out[product] = rec
    return out


def _merge_records(by_tag: dict[str, dict], product: str) -> dict:
    """Fold one DATA product's per-source records into one (union window), keeping the
    per-source-tag source list so a channel can be attributed to its exact source."""
    recs = list(by_tag.values())
    accessed_first = [r["accessed_first"] for r in recs if r["accessed_first"]]
    accessed_last = [r["accessed_last"] for r in recs if r["accessed_last"]]
    return {
        "product": product,
        "sources": sorted({s for r in recs for s in r["sources"]}),
        "sources_by_tag": {tag: r["sources"] for tag, r in by_tag.items()},
        "n_files": sum(r["n_files"] for r in recs),
        "accessed_first": min(accessed_first) if accessed_first else None,
        "accessed_last": max(accessed_last) if accessed_last else None,
        "basis": STAMPED if all(r["basis"] == STAMPED for r in recs) else FILE_MTIME,
    }


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
        # Per-source PRECISION: a `<var>_<source>` channel attributes to its ONE source, not
        # the product's union. Where an input product records `sources_by_tag`, narrow to the
        # tag the channel name ends with (longest match, so `eo_tides` beats a shorter tag).
        sources: set = set()
        for r in recs:
            tag = _matching_source_tag(name, r.get("sources_by_tag") or {})
            sources |= set(r["sources_by_tag"][tag]) if tag else set(r["sources"])
        per_field[name] = {
            "inputs": inputs,
            "sources": sorted(sources),
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
