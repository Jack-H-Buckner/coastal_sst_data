"""The aligned-file naming convention.

A filename is a contract between an acquisition stage that writes a stamp and an assembler
that parses it back out. The two live in different modules and agree only by construction, so
the failure mode is silence: change the write side and the assembler does not crash, it
matches nothing and every affected day becomes a NaN slice -- indistinguishable from cloud.

These tests pin the round trip, and in particular that a TILED product's name still satisfies
everything `datacube` does to a per-overpass file.
"""

import fnmatch
from datetime import datetime

import pytest

from coastal_sst_data import naming

ACQ = datetime(2026, 1, 22, 22, 29, 1)


def test_tile_stem_extends_the_overpass_stem():
    assert naming.time_stem("hobart", ACQ) == "hobart_20260122T222901"
    assert naming.tile_stem("hobart", ACQ, "55GDP") == "hobart_20260122T222901_55GDP"


def test_a_tiled_name_still_yields_its_acquisition_time():
    """`scene_index` reads the instant back out of the filename to group a day's granules.
    A trailing tile must not hide it -- `TIME_RE` searches rather than anchors precisely so
    that the stamp can carry fields after it."""
    name = naming.tile_stem("hobart", ACQ, "55GDP") + ".nc"
    assert naming.parse_time(name) == ACQ


def test_a_tiled_name_matches_the_per_overpass_glob():
    """`datacube.scene_index` finds a sensor's scenes with `<aoi>_*T*.nc`. A tiled file the
    glob missed would be written and then never read -- the loss this naming exists to stop,
    reintroduced one layer down."""
    name = naming.tile_stem("hobart", ACQ, "55GDP") + ".nc"
    assert fnmatch.fnmatch(name, "hobart_*T*.nc")


def test_an_mgrs_tile_cannot_be_read_as_a_timestamp():
    """The tile sits next to the stamp in the same name, so it must not be able to match it.
    Five characters cannot satisfy `\\d{8}T\\d{6}`, and this says so out loud."""
    assert naming.parse_time("hobart_55GDP.nc") is None


def test_tiles_of_one_overpass_sort_by_tile():
    """`scene_index` orders a day on `(datetime, name)`. For a tiled sensor the datetime is a
    tie across the WHOLE overpass, so the name is what makes the order -- and therefore the
    mosaic's base, and therefore the day's reported overpass -- reproducible between runs."""
    names = [naming.tile_stem("hobart", ACQ, t) for t in ("55GDQ", "55GDP", "55HDA")]
    assert sorted(names) == ["hobart_20260122T222901_55GDP",
                             "hobart_20260122T222901_55GDQ",
                             "hobart_20260122T222901_55HDA"]


def test_a_tiled_name_is_not_mistaken_for_a_per_day_file():
    """`day_pattern` is anchored so a per-overpass file cannot be read as a daily mean."""
    assert not naming.day_pattern("hobart").match(
        naming.tile_stem("hobart", ACQ, "55GDP") + ".nc")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
