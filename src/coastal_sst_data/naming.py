#!/usr/bin/env python3
"""
coastal_sst_data -- the aligned-file naming convention, in one place.

An aligned output's FILENAME is a load-bearing contract, not a cosmetic choice. The
acquisition stages encode a timestamp into it and the assembler decodes it back out:
`datacube.load_clearest_overpass` finds a sensor's scenes by globbing `<aoi>_*T*.nc` and
parsing the stamp; `datacube.load_at_times` reconstructs a filename from a datetime to
fetch the met snapshot taken at that exact overpass; `met.overpass_times_for_day`
discovers the scenes to snapshot against by parsing the same stamp.

Until now that convention was re-declared wherever it was needed -- the same
`re.compile(r"(\\d{8}T\\d{6})")` appeared in FOUR modules, and the corresponding
`strftime("%Y%m%dT%H%M%S")` that WRITES it appeared in six more, with no single owner.
A writer and a reader of the same contract sitting in different files, agreeing only by
coincidence, is precisely the thing that breaks silently: change the stamp on the write
side and the assembler does not crash, it simply matches nothing and every affected day
becomes a NaN slice -- indistinguishable from a cloudy day.

So the format lives here, once, and the parse and the format are defined next to each
other where they cannot drift apart.

    stem = naming.time_stem("hood_canal", acq)      # hood_canal_20230715T210043
    stem = naming.day_stem("hood_canal", day)       # hood_canal_20230715
    dt   = naming.parse_time(path.name)             # datetime | None
"""

from __future__ import annotations

import re
from datetime import datetime

# The two stamp formats every aligned file uses. `TIME` is per-overpass (a sensor scene,
# or a met snapshot taken at one); `DAY` is per-day (MUR, CMEMS, the met daily mean).
TIME_FMT = "%Y%m%dT%H%M%S"
DAY_FMT = "%Y%m%d"

# Matched anywhere in the name (a granule id carries other fields around the stamp).
TIME_RE = re.compile(r"(\d{8}T\d{6})")
# Anchored to the whole filename: a per-DAY file must not match a per-overpass one.
DAY_RE = re.compile(r"_(\d{8})\.nc$")


def time_stamp(t) -> str:
    """A datetime -> the per-overpass stamp (20230715T210043)."""
    return t.strftime(TIME_FMT)


def day_stamp(d) -> str:
    """A date/datetime -> the per-day stamp (20230715)."""
    return d.strftime(DAY_FMT)


def time_stem(aoi_id: str, t) -> str:
    """The filename stem of a per-overpass aligned file (no extension)."""
    return f"{aoi_id}_{time_stamp(t)}"


def day_stem(aoi_id: str, d, prefix: str = "") -> str:
    """The filename stem of a per-day aligned file (no extension).

    `prefix` selects a variant written into the same directory -- met writes both a daily
    mean (`<aoi>_20230715.nc`) and a reference-time snapshot (`<aoi>_ref_20230715.nc`).
    """
    return f"{aoi_id}_{prefix}{day_stamp(d)}"


def parse_time(name: str) -> datetime | None:
    """The per-overpass stamp in a filename/granule id, or None if it carries none."""
    m = TIME_RE.search(name)
    return datetime.strptime(m.group(1), TIME_FMT) if m else None


def day_pattern(aoi_id: str, prefix: str = "") -> re.Pattern:
    """Regex matching EXACTLY this AoI's per-day files for one variant.

    Anchored and prefix-aware on purpose: a plain `<aoi>_*.nc` glob cannot tell
    `<aoi>_20230715.nc` from `<aoi>_ref_20230715.nc`, and reading one as the other means
    silently feeding a daily mean where a reference snapshot was asked for.
    """
    return re.compile(rf"^{re.escape(aoi_id)}_{re.escape(prefix)}(\d{{8}})\.nc$")
