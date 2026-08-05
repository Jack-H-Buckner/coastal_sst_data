#!/usr/bin/env python3
"""
coastal_sst_data -- discovering WHEN the thermal sensors flew, from what they wrote.

Several stages need to know a sensor's overpass instants without being that sensor:
`met_overpass` snapshots the weather at each instant, and `mur` restricts its daily
downloads to the days a sensor actually flew. Both answer the question the same way --
by globbing the sensors' aligned trees and parsing the stamp through the shared
convention (coastal_sst_data.naming) -- so the seam lives here, once, rather than in
whichever product happened to need it first.

Keeping it out of `processes/met.py` matters: `mur` reading the SENSORS' output should
not mean importing the met product.

    dirs  = overpass.sensor_dirs(Path(output_dir))       # {prefix: [aligned dirs]}
    times = overpass.times_for_day(dirs["eco"], aoi, day)  # instants on one day
    days  = overpass.days_with_scenes(dirs["eco"], aoi)    # {"20230715", ...}
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import naming, products


def sensor_dirs(root: Path) -> dict[str, list[Path]]:
    """{sensor prefix -> its aligned dir(s)}: a flat sensor has one (lst -> [LANDSAT/aligned]);
    a STACKED-DATA sensor (ECOSTRESS) has one per collection-version tree on disk
    (eco -> [ECOSTRESS/v002/aligned, ECOSTRESS/v003/aligned]). The sensor 'flew' if ANY version
    recorded a scene, so overpass discovery searches every version's tree."""
    out: dict[str, list[Path]] = {}
    for s in products.sensors():
        base = Path(root) / s.dir
        out[s.sensor.prefix] = (sorted(base.glob("*/aligned")) if s.is_stacked_data
                                else [base / "aligned"])
    return out


def times_for_day(overpass_dirs, aoi_id: str, day) -> list[datetime]:
    """Datetimes (UTC, naive) of thermal scenes for this AoI on `day`.

    Reads the stamp the SENSOR stages wrote, via the shared convention
    (coastal_sst_data.naming) -- this is the seam where met discovers what to snapshot
    forcing against, so its parse must be the sensors' format by construction.
    """
    daystr = naming.day_stamp(day)
    times = []
    for d in overpass_dirs:
        adir = Path(d) / aoi_id
        if not adir.exists():
            continue
        for f in adir.glob(f"{aoi_id}_{daystr}T*.nc"):
            t = naming.parse_time(f.name)
            if t is not None:
                times.append(t)
    return sorted(set(times))


def days_with_scenes(overpass_dirs, aoi_id: str) -> set[str]:
    """The set of `YYYYMMDD` day stamps on which this AoI has ANY scene in these dirs.

    DAY STAMPS, not dates, and one glob PER DIR rather than one per day:

      * the stamp is exactly the key the cube bins by (`datacube.load_daily_sensor` and
        `load_clearest_overpass` both index days as `naming.day_stamp(dd)`), so "the sensor
        flew on stamp S" and "the per-day file `<aoi>_S.nc` that scene is compared against"
        are the same string by construction -- no date arithmetic, no +/-1 day tolerance;
      * a caller asking about a multi-year window would otherwise glob every sensor dir once
        per day, which is thousands of directory scans to answer one set-membership question.

    The `*T*` in the glob is load-bearing: a per-DAY file (`<aoi>_20230715.nc`) written into
    the same tree must never read as an overpass.
    """
    days: set[str] = set()
    for d in overpass_dirs:
        adir = Path(d) / aoi_id
        if not adir.exists():
            continue
        for f in adir.glob(f"{aoi_id}_*T*.nc"):
            t = naming.parse_time(f.name)
            if t is not None:
                days.add(naming.day_stamp(t))
    return days
