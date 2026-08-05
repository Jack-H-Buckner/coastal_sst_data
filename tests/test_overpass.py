"""The overpass-discovery seam: reading WHEN the sensors flew out of what they wrote.

Two products depend on these answers -- `met_overpass` snapshots the weather at each instant,
`mur` restricts its daily downloads to the days a sensor flew -- and both are silent when the
answer is wrong: an over-narrow read produces empty channels or missing days, never an error.
So the glob and the parse are pinned here.
"""

from datetime import datetime
from pathlib import Path

import pytest

from coastal_sst_data import overpass


def _touch(d: Path, *names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"")


# --------------------------------------------------------------------------- #
# sensor_dirs
# --------------------------------------------------------------------------- #
def test_sensor_dirs_expands_a_stacked_sensor_to_every_version(tmp_path):
    """ECOSTRESS writes one aligned tree PER collection version. The sensor 'flew' if ANY
    version recorded the scene, so discovery must search all of them -- reading only one
    would silently under-report overpasses for a project that stacks v002 and v003."""
    for p in ("ECOSTRESS/v002/aligned", "ECOSTRESS/v003/aligned", "LANDSAT/aligned"):
        (tmp_path / p).mkdir(parents=True)

    dirs = overpass.sensor_dirs(tmp_path)

    assert dirs["eco"] == [tmp_path / "ECOSTRESS/v002/aligned",
                           tmp_path / "ECOSTRESS/v003/aligned"]
    assert dirs["lst"] == [tmp_path / "LANDSAT/aligned"]       # flat: one tree, always
    assert "modis" in dirs                                      # every sensor is answered for


def test_sensor_dirs_answers_for_a_sensor_that_has_not_run(tmp_path):
    """A flat sensor's dir is named whether or not it exists -- the caller decides what an
    absent tree means (mur: a loud error; met_overpass: nothing to snapshot)."""
    dirs = overpass.sensor_dirs(tmp_path)
    assert dirs["lst"] == [tmp_path / "LANDSAT" / "aligned"]
    assert dirs["eco"] == []                                    # stacked: no version on disk


# --------------------------------------------------------------------------- #
# days_with_scenes
# --------------------------------------------------------------------------- #
def test_days_with_scenes_dedupes_the_overpasses_within_a_day(tmp_path):
    d = tmp_path / "aligned"
    _touch(d / "a1", "a1_20260601T180000.nc", "a1_20260601T190000.nc",
           "a1_20260603T180000.nc")
    assert overpass.days_with_scenes([d], "a1") == {"20260601", "20260603"}


def test_days_with_scenes_ignores_other_aois_and_per_day_files(tmp_path):
    """The `*T*` in the glob is load-bearing: a per-DAY file in the same tree is not an
    overpass, and one AoI's scenes must never answer for another's."""
    d = tmp_path / "aligned"
    _touch(d / "a1", "a1_20260601T180000.nc", "a1_20260602.nc")   # the second is per-day
    _touch(d / "a2", "a2_20260605T180000.nc")

    assert overpass.days_with_scenes([d], "a1") == {"20260601"}
    assert overpass.days_with_scenes([d], "a2") == {"20260605"}


def test_days_with_scenes_unions_the_dirs_and_skips_absent_ones(tmp_path):
    v2, v3 = tmp_path / "v002/aligned", tmp_path / "v003/aligned"
    _touch(v2 / "a1", "a1_20260601T180000.nc")
    _touch(v3 / "a1", "a1_20260602T180000.nc")
    missing = tmp_path / "never_ran/aligned"

    assert overpass.days_with_scenes([v2, v3, missing], "a1") == {"20260601", "20260602"}
    assert overpass.days_with_scenes([missing], "a1") == set()


# --------------------------------------------------------------------------- #
# times_for_day (moved here from processes/met.py -- regression on the move)
# --------------------------------------------------------------------------- #
def test_times_for_day_returns_sorted_unique_instants_for_that_day_only(tmp_path):
    d = tmp_path / "aligned"
    _touch(d / "a1", "a1_20260601T190000.nc", "a1_20260601T180000.nc",
           "a1_20260602T180000.nc")

    got = overpass.times_for_day([d], "a1", datetime(2026, 6, 1))

    assert got == [datetime(2026, 6, 1, 18), datetime(2026, 6, 1, 19)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
