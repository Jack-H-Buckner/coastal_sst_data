"""Durable output: writes that cannot be half-done, and skips that cannot be fooled.

The failure these exist to prevent: a run killed mid-write leaves a file that EXISTS and
OPENS but is missing its data; the old `.exists()` skip guard took that for a finished
download and never repaired it, so a dropped connection became a permanent NaN day in
every cube built afterwards.
"""

import numpy as np
import pytest
import xarray as xr

from coastal_sst_data import store


def _ds(h=4, w=4):
    return xr.Dataset(
        {"sst": (("y", "x"), np.ones((h, w), "float32")),
         "valid": (("y", "x"), np.ones((h, w), "uint8"))},
        coords={"y": range(h), "x": range(w)})


# --------------------------------------------------------------------------- #
# atomic()
# --------------------------------------------------------------------------- #
def test_a_failed_write_leaves_nothing_at_the_final_path(tmp_path):
    dest = tmp_path / "a.nc"
    with pytest.raises(ConnectionError):
        with store.atomic(dest) as tmp:
            tmp.write_bytes(b"half a file")      # the write got this far...
            raise ConnectionError("reset")       # ...and then the wire dropped

    assert not dest.exists()                     # nothing at the final path
    assert not list(tmp_path.glob("*.part-*"))   # and no scratch left behind


def test_a_failed_overwrite_keeps_the_previous_file(tmp_path):
    dest = tmp_path / "a.nc"
    store.write_netcdf(_ds(), dest)

    with pytest.raises(ConnectionError):
        with store.atomic(dest) as tmp:
            tmp.write_bytes(b"garbage")
            raise ConnectionError("reset")

    with xr.open_dataset(dest) as ds:            # the good file survived untouched
        assert "sst" in ds and ds["sst"].shape == (4, 4)


def test_keyboard_interrupt_is_cleaned_up_too(tmp_path):
    """A long run usually dies by Ctrl-C, not by an exception -- so `except Exception`
    would miss the very case this is for."""
    dest = tmp_path / "a.nc"
    with pytest.raises(KeyboardInterrupt):
        with store.atomic(dest) as tmp:
            tmp.write_bytes(b"half")
            raise KeyboardInterrupt

    assert not dest.exists()
    assert not list(tmp_path.glob("*.part-*"))


def test_a_writer_that_produces_nothing_is_an_error(tmp_path):
    dest = tmp_path / "a.nc"
    with pytest.raises(RuntimeError, match="nothing was written"):
        with store.atomic(dest):
            pass
    assert not dest.exists()


def test_overwrite_replaces_and_leaves_no_scratch(tmp_path):
    dest = tmp_path / "a.nc"
    store.write_netcdf(_ds(), dest)
    store.write_netcdf(_ds() * 2, dest)

    with xr.open_dataset(dest) as ds:
        assert float(ds["sst"][0, 0]) == 2.0
    assert not list(tmp_path.glob("*.part-*")) and not list(tmp_path.glob("*.old-*"))


def test_scratch_from_an_earlier_crash_is_swept_and_reported(tmp_path, caplog):
    dest = tmp_path / "a.nc"
    (tmp_path / "a.nc.part-999-1").write_bytes(b"junk")

    with caplog.at_level("WARNING"):
        store.write_netcdf(_ds(), dest)

    assert not list(tmp_path.glob("*.part-*"))
    assert "unfinished run" in caplog.text      # the only trace an earlier run died


def test_rasters_swap_the_whole_directory(tmp_path):
    """A directory holding 2 of 3 bands is exactly the half-output a skip guard would
    take for done, so the directory is staged and swapped as a unit."""
    ds = _ds()
    dest = tmp_path / "layers"
    with pytest.raises(ValueError):
        with store.atomic(dest) as tmp:
            tmp.mkdir()
            (tmp / "sst.tif").write_bytes(b"x")
            raise ValueError("second band failed")
    assert not dest.exists()


# --------------------------------------------------------------------------- #
# is_complete() / done()
# --------------------------------------------------------------------------- #
def test_complete_file_is_complete(tmp_path):
    p = store.write_netcdf(_ds(), tmp_path / "a.nc")
    assert store.is_complete(p, ("sst", "valid"))


def test_empty_file_is_not_complete(tmp_path):
    p = tmp_path / "a.nc"
    p.touch()                                   # 0 bytes, as a killed write can leave
    assert not store.is_complete(p, ("sst", "valid"))


def test_missing_layer_is_not_complete(tmp_path):
    """A COMPLETE write of degraded content -- the mask COG failed, so the file is intact
    but has no `valid`. Atomicity cannot see this; only the layer check can."""
    p = store.write_netcdf(_ds()[["sst"]], tmp_path / "a.nc")
    assert store.is_complete(p, ("sst",))
    assert not store.is_complete(p, ("sst", "valid"))


def test_wrong_grid_is_not_complete(tmp_path):
    p = store.write_netcdf(_ds(4, 4), tmp_path / "a.nc")
    assert store.is_complete(p, ("sst",), shape=(4, 4))
    assert not store.is_complete(p, ("sst",), shape=(8, 8))   # written on another grid


def test_deep_read_catches_a_corrupt_chunk_the_metadata_check_misses(tmp_path):
    """Why `check` reads the payload rather than trusting the header.

    A tail-truncated HDF5 file usually fails to OPEN (its object headers live near the
    end), so the cheap check catches it. But a file whose header is intact and whose
    chunk bytes are damaged -- a partial flush, a bad disk -- opens perfectly and reports
    the right dims. Only forcing the data off disk can tell.
    """
    p = store.write_netcdf(
        xr.Dataset({"sst": (("y", "x"), np.random.rand(300, 300).astype("float32"))}),
        tmp_path / "a.nc")
    with open(p, "r+b") as f:
        f.seek(int(p.stat().st_size * 0.45))
        f.write(b"\x00" * 3000)                 # scribble over a compressed chunk

    assert store.is_complete(p, ("sst",), deep=False)          # header still looks fine
    assert not store.is_complete(p, ("sst",), deep=True)       # the data does not


def test_done_reports_an_incomplete_file_rather_than_skipping_it(tmp_path, caplog):
    p = tmp_path / "a.nc"
    p.touch()
    with caplog.at_level("WARNING"):
        assert store.done(p, ("sst", "valid")) is False       # -> caller re-fetches
    assert "INCOMPLETE" in caplog.text


def test_done_is_false_under_overwrite(tmp_path):
    p = store.write_netcdf(_ds(), tmp_path / "a.nc")
    assert store.done(p, ("sst",)) is True
    assert store.done(p, ("sst",), overwrite=True) is False


# --------------------------------------------------------------------------- #
# covers=(start, end) -- the range-aware guard for single-file span products
# (tides, insitu), which write ONE file for the whole window. See store._covers_range.
# --------------------------------------------------------------------------- #
def _span(tmp_path, start, end):
    """A complete file stamped with the requested date window it was built for."""
    ds = _ds()
    ds.attrs.update(requested_start=str(start), requested_end=str(end))
    return store.write_netcdf(ds, tmp_path / "span.nc")


def test_done_skips_when_the_stamped_range_spans_the_request(tmp_path):
    p = _span(tmp_path, "2020-01-01", "2020-12-31")
    assert store.done(p, ("sst",), covers=("2020-03-01", "2020-06-30")) is True
    assert store.done(p, ("sst",), covers=("2020-01-01", "2020-12-31")) is True  # exact edges


def test_done_rebuilds_when_the_request_extends_past_the_stamped_range(tmp_path, caplog):
    """The bug this fixes: an old file built for a NARROWER window would be skipped, so an
    extended date range silently became NaN in the cube."""
    p = _span(tmp_path, "2020-01-01", "2020-06-30")
    with caplog.at_level("INFO"):
        assert store.done(p, ("sst",), covers=("2020-01-01", "2020-12-31")) is False  # end extended
    assert "narrower date range" in caplog.text
    # a start moved earlier must rebuild too
    assert store.done(p, ("sst",), covers=("2019-06-01", "2020-06-30")) is False


def test_done_rebuilds_a_complete_file_that_carries_no_range_stamp(tmp_path):
    """Files written before range-stamping have no requested_start/end; rebuild once so they
    gain the stamp rather than being trusted with an unknown window."""
    p = store.write_netcdf(_ds(), tmp_path / "a.nc")            # complete, but unstamped
    assert store.done(p, ("sst",)) is True                      # covers=None: unchanged behavior
    assert store.done(p, ("sst",), covers=("2020-01-01", "2020-12-31")) is False


def test_covers_is_ignored_for_an_incomplete_file(tmp_path, caplog):
    """Incompleteness is reported first; a truncated file never reaches the range check."""
    p = tmp_path / "a.nc"
    p.touch()
    with caplog.at_level("WARNING"):
        assert store.done(p, ("sst",), covers=("2020-01-01", "2020-12-31")) is False
    assert "INCOMPLETE" in caplog.text


# --------------------------------------------------------------------------- #
# scan() / repair() -- the one-off pass over a tree written before atomicity
# --------------------------------------------------------------------------- #
def _aligned(root, product, aoi):
    d = root / product / "aligned" / aoi
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_scan_finds_truncated_and_degraded_files(tmp_path):
    d = _aligned(tmp_path, "MUR", "aoi1")
    store.write_netcdf(_ds(), d / "aoi1_20230101.nc")        # good
    (d / "aoi1_20230102.nc").touch()                          # truncated
    store.write_netcdf(_ds()[["sst"]], d / "aoi1_20230103.nc")  # missing `valid`
    (d / "aoi1_20230104.nc.part-1-1").write_bytes(b"junk")    # scratch from a crash

    n, bad, leftovers = store.scan(tmp_path)

    assert n == 3
    assert {f.name for _, f in bad} == {"aoi1_20230102.nc", "aoi1_20230103.nc"}
    assert [p.name for p in leftovers] == ["aoi1_20230104.nc.part-1-1"]


def test_repair_deletes_the_bad_and_leaves_the_good(tmp_path):
    d = _aligned(tmp_path, "MUR", "aoi1")
    good = store.write_netcdf(_ds(), d / "aoi1_20230101.nc")
    (d / "aoi1_20230102.nc").touch()

    n, bad, leftovers = store.scan(tmp_path)
    assert store.repair(bad, leftovers) == 1

    assert good.exists()                                      # the good file is untouched
    assert not (d / "aoi1_20230102.nc").exists()              # the bad one is gone
    assert store.scan(tmp_path)[1] == []                      # tree now clean


def test_scan_can_be_limited_to_one_aoi(tmp_path):
    (_aligned(tmp_path, "MUR", "aoi1") / "aoi1_20230101.nc").touch()
    (_aligned(tmp_path, "MUR", "aoi2") / "aoi2_20230101.nc").touch()

    _, bad, _ = store.scan(tmp_path, aois=["aoi1"])
    assert [f.name for _, f in bad] == ["aoi1_20230101.nc"]
