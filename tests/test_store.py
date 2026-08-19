"""Durable output: writes that cannot be half-done, and skips that cannot be fooled.

The failure these exist to prevent: a run killed mid-write leaves a file that EXISTS and
OPENS but is missing its data; the old `.exists()` skip guard took that for a finished
download and never repaired it, so a dropped connection became a permanent NaN day in
every cube built afterwards.

And the failure the OWNERSHIP half prevents: the cleanup that removes a dead run's scratch
could not tell it from a live run's open file, so two runs over one tree deleted each other's
writes mid-flight.
"""

import os
import threading
import time

import numpy as np
import pytest
import xarray as xr

from coastal_sst_data import store


def _ds(h=4, w=4):
    return xr.Dataset(
        {"sst": (("y", "x"), np.ones((h, w), "float32")),
         "valid": (("y", "x"), np.ones((h, w), "uint8"))},
        coords={"y": range(h), "x": range(w)})


def _scratch(dest, owner, *, age_h=0.0, body=b"junk"):
    """Scratch beside `dest` attributed to `owner` ('<host>-<pid>') and last touched
    `age_h` hours ago. BOTH halves decide whether the sweep may delete it, so both are
    spelled out at every call site rather than left to a default.
    """
    p = dest.with_name(f"{dest.name}{store.PART_SUFFIX}{owner}-1-1")
    p.write_bytes(body)
    if age_h:
        stamp = time.time() - age_h * 3600
        os.utime(p, (stamp, stamp))
    return p


# A run we cannot interrogate -- on another machine, so `os.kill` cannot answer for it -- and
# untouched for two days. Only the clock can condemn this one.
DEAD = "oldnode-999"
# This host, but a pid that cannot exist: above every platform's pid_max.
DEAD_PID_HERE = f"{store._HOST}-4000000"


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
    _scratch(dest, DEAD, age_h=48)

    with caplog.at_level("WARNING"):
        store.write_netcdf(_ds(), dest)

    assert not list(tmp_path.glob("*.part-*"))
    assert "unfinished run" in caplog.text      # the only trace an earlier run died


# --------------------------------------------------------------------------- #
# sweep_scratch() -- whose scratch is it?
#
# The sweep runs at the top of EVERY atomic() call, so getting this wrong does not strand a
# file, it deletes one out from under an open HDF5 handle. Each test below pins one tier of
# the evidence, and every one of them errs the same way: when we cannot prove the writer is
# gone, the file stays.
# --------------------------------------------------------------------------- #
def test_scratch_a_live_writer_in_this_process_holds_is_not_swept(tmp_path):
    """The reported production failure, in miniature.

    Two runs over one output tree both reach `Macquarie_Harbour_20040327.nc`. The second
    swept the first's scratch, and the first -- whose netCDF4 handle is reopened BY PATH for
    each dask chunk -- died with `errno = 2` on a day it had already downloaded.
    """
    dest = tmp_path / "a.nc"
    entered, release, seen = threading.Event(), threading.Event(), {}

    def writer():
        with store.atomic(dest) as tmp:
            tmp.write_bytes(b"half a file")      # the write is under way...
            seen["tmp"] = tmp
            entered.set()
            release.wait(5)                      # ...and stalled here, as a slow fetch stalls
            tmp.write_bytes(b"whole file")

    t = threading.Thread(target=writer)
    t.start()
    assert entered.wait(5)

    store.sweep_scratch(dest)                    # what the second run does on arrival
    assert seen["tmp"].exists()                  # ...and must not have done

    release.set()
    t.join(5)
    assert dest.read_bytes() == b"whole file"    # the first writer still landed its file


def test_scratch_owned_by_another_live_process_on_this_host_is_left_alone(tmp_path, caplog):
    """Our parent: deterministically alive, and deterministically not us."""
    dest = tmp_path / "a.nc"
    p = _scratch(dest, f"{store._HOST}-{os.getppid()}")

    with caplog.at_level("WARNING"):
        assert store.sweep_scratch(dest) == [p]
    assert p.exists()
    assert "may still be writing it" in caplog.text


def test_scratch_owned_by_a_dead_process_on_this_host_is_swept_immediately(tmp_path):
    """No waiting out the clock: `os.kill(pid, 0)` PROVES the writer is gone, so a killed
    job's leftovers clear on the next run rather than in six hours."""
    dest = tmp_path / "a.nc"
    p = _scratch(dest, DEAD_PID_HERE)            # fresh mtime, and still condemned
    assert store.sweep_scratch(dest) == []
    assert not p.exists()


def test_our_own_abandoned_scratch_is_swept_even_though_our_pid_is_alive(tmp_path):
    """`os.kill` cannot answer for our own pid -- it is trivially alive -- so without the
    registry this process could never reclaim scratch it abandoned itself."""
    dest = tmp_path / "a.nc"
    p = _scratch(dest, f"{store._HOST}-{os.getpid()}")
    assert store.sweep_scratch(dest) == []
    assert not p.exists()


def test_a_live_process_keeps_its_scratch_however_old_the_file_looks(tmp_path):
    """The clock does not get to overrule `os.kill`.

    This was the first fix's bug with a longer fuse: a live pid fell THROUGH to the age test,
    so a writer stalled on a slow read -- scratch created, mtime frozen, entirely alive --
    had its file deleted once it crossed the staleness threshold.
    """
    dest = tmp_path / "a.nc"
    p = _scratch(dest, f"{store._HOST}-{os.getppid()}", age_h=48)   # ancient, and alive

    assert store.sweep_scratch(dest, max_age_s=0) == [p]
    assert p.exists()


def test_recent_scratch_from_another_host_is_left_alone_until_it_goes_stale(tmp_path):
    """The only tier that is a guess, so it is the only one that waits."""
    dest = tmp_path / "a.nc"
    p = _scratch(dest, DEAD)
    assert store.sweep_scratch(dest) == [p] and p.exists()

    assert store.sweep_scratch(dest, max_age_s=0) == []
    assert not p.exists()


def test_a_staged_zarr_directory_is_judged_live_by_its_chunk_files_not_its_own_mtime(tmp_path):
    """A cube assembly writes into `sst/c/0/0` for hours; the store's OWN mtime stops moving
    after the first block. Trusting it would declare a running assembly dead."""
    dest = tmp_path / "cube.zarr"
    staged = dest.with_name(f"{dest.name}{store.PART_SUFFIX}{DEAD}-1-1")
    (staged / "sst" / "c" / "0").mkdir(parents=True)
    (staged / "sst" / "c" / "0" / "0").write_bytes(b"a fresh chunk")
    old = time.time() - 48 * 3600
    for d in (staged, staged / "sst", staged / "sst" / "c"):
        os.utime(d, (old, old))

    assert store.sweep_scratch(dest) == [staged]
    assert staged.exists()


def test_a_stash_from_a_live_swap_is_never_swept(tmp_path):
    """`_swap` moves the previous output to `<dest>.old-<our own tag>` while it installs the
    new one. That tag is OURS, which reads as provably dead -- so the swap has to claim it,
    or a concurrent sweep deletes the only copy of the old output mid-rename."""
    dest = tmp_path / "layers"
    dest.mkdir()
    (dest / "sst.tif").write_bytes(b"the previous output")
    swept, entered, release = {}, threading.Event(), threading.Event()

    real_replace = os.replace

    def slow_replace(src, dst):                  # pause INSIDE _swap, stash on disk
        entered.set()
        release.wait(5)
        return real_replace(src, dst)

    def writer():
        with store.atomic(dest) as tmp:
            tmp.mkdir()
            (tmp / "sst.tif").write_bytes(b"the new output")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", slow_replace)
        t = threading.Thread(target=writer)
        t.start()
        assert entered.wait(5)
        stash = next(p for p in tmp_path.glob("*.old-*"))
        swept["live"] = store.sweep_scratch(dest)
        swept["survived"] = stash.exists()       # asked NOW: the swap deletes it on success
        release.set()
        t.join(5)

    assert stash in swept["live"] and swept["survived"]
    assert (dest / "sst.tif").read_bytes() == b"the new output"


def test_writing_a_netcdf_off_the_main_thread_prints_no_hdf5_error_stacks(tmp_path, capfd):
    """A `--jobs N` run buried its own log under HDF5 error stacks -- for writes that
    SUCCEEDED.

    libnetcdf probes the target with `H5Fis_accessible()` before every create, with no
    existence check, so writing a NEW file emitted a full `HDF5-DIAG` block ending
    `errno = 2` about a file that was missing only because it was about to be created.
    Serial runs never saw it and parallel runs drowned in it, from identical code: HDF5's
    error stack is thread-local, and libnetcdf silences it once on the thread that
    initialised HDF5 -- the main one -- so every pool worker printed.

    `capfd`, not `caplog`: a C library writes this straight to fd 2, and it never passes
    through Python's logging at all.
    """
    # The main-thread write is LOAD-BEARING, not a warm-up. HDF5 is initialised lazily by
    # the first netCDF call in the process, and whichever thread does that is the one
    # libnetcdf silences. Skip this and the worker below initialises HDF5 itself, silences
    # itself, and the test passes without ever exercising the bug. A real run always
    # initialises on the main thread first.
    store.write_netcdf(_ds(), tmp_path / "init.nc")

    err = {}

    def write():
        capfd.readouterr()                       # discard anything already buffered
        store.write_netcdf(_ds(), tmp_path / "a.nc")
        err["text"] = capfd.readouterr().err

    t = threading.Thread(target=write)
    t.start()
    t.join(30)

    assert "HDF5-DIAG" not in err["text"], f"HDF5 noise leaked:\n{err['text'][:1500]}"
    with xr.open_dataset(tmp_path / "a.nc") as ds:       # ...and the file is real
        assert ds["sst"].shape == (4, 4)


def test_a_placeholder_that_is_never_written_to_is_still_an_error(tmp_path):
    """`placeholder=True` makes the scratch exist before the writer touches it, so the
    emptiness guard cannot be `exists()` any more. A writer that opens the file and writes
    nothing must still fail -- silently swapping a 0-byte file onto the destination is
    exactly the half-output this module exists to prevent."""
    dest = tmp_path / "a.nc"
    with pytest.raises(RuntimeError, match="nothing was written"):
        with store.atomic(dest, placeholder=True) as tmp:
            assert tmp.exists() and tmp.stat().st_size == 0     # handed to us pre-created
    assert not dest.exists()


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
    dead = _scratch(d / "aoi1_20230104.nc", DEAD, age_h=48)    # scratch from a crash

    n, bad, leftovers, in_use = store.scan(tmp_path)

    assert n == 3
    assert {f.name for _, f in bad} == {"aoi1_20230102.nc", "aoi1_20230103.nc"}
    assert leftovers == [dead] and in_use == []


def test_scan_separates_scratch_in_use_from_scratch_to_repair(tmp_path):
    """The distinction `check --repair` turns on: one of these is junk, the other is a file
    another job is in the middle of writing."""
    d = _aligned(tmp_path, "MUR", "aoi1")
    dead = _scratch(d / "aoi1_20230101.nc", DEAD, age_h=48)
    live = _scratch(d / "aoi1_20230102.nc", f"{store._HOST}-{os.getppid()}")

    _, _, leftovers, in_use = store.scan(tmp_path)
    assert leftovers == [dead] and in_use == [live]


def test_scan_does_not_descend_into_an_abandoned_scratch_directory(tmp_path):
    """An abandoned scratch Zarr holds thousands of chunk files. It is ONE dead write, and
    reporting it as one is the difference between a readable `check` and a wall of paths."""
    d = _aligned(tmp_path, "MUR", "aoi1")
    staged = d / f"aoi1.zarr{store.PART_SUFFIX}{DEAD}-1-1"
    (staged / "sst" / "c").mkdir(parents=True)
    (staged / "sst" / "c" / "0").write_bytes(b"chunk")
    old = time.time() - 48 * 3600
    for p in (*staged.rglob("*"), staged):
        os.utime(p, (old, old))

    _, _, leftovers, in_use = store.scan(tmp_path)
    assert leftovers == [staged] and in_use == []


def test_a_stash_left_by_a_swap_that_died_is_reported_by_scan(tmp_path):
    """A process killed between the rename and the replace leaves a full copy of the previous
    output that nothing ever looked for -- for a cube, tens of gigabytes of silent leak."""
    d = _aligned(tmp_path, "MUR", "aoi1")
    stash = d / f"aoi1.zarr{store.OLD_SUFFIX}{DEAD}-1-1"
    stash.mkdir()
    old = time.time() - 48 * 3600
    os.utime(stash, (old, old))

    _, _, leftovers, _ = store.scan(tmp_path)
    assert leftovers == [stash]


def test_repair_deletes_the_bad_and_leaves_the_good(tmp_path):
    d = _aligned(tmp_path, "MUR", "aoi1")
    good = store.write_netcdf(_ds(), d / "aoi1_20230101.nc")
    (d / "aoi1_20230102.nc").touch()

    n, bad, leftovers, _ = store.scan(tmp_path)
    assert store.repair(bad, leftovers) == 1

    assert good.exists()                                      # the good file is untouched
    assert not (d / "aoi1_20230102.nc").exists()              # the bad one is gone
    assert store.scan(tmp_path)[1] == []                      # tree now clean


def test_repair_refuses_to_delete_scratch_that_is_in_use(tmp_path, caplog):
    """`scan` already filters it out; this is the second lock on the same door, for a caller
    that assembled the list some other way."""
    d = _aligned(tmp_path, "MUR", "aoi1")
    live = _scratch(d / "aoi1_20230101.nc", f"{store._HOST}-{os.getppid()}")

    with caplog.at_level("WARNING"):
        assert store.repair([], [live]) == 0
    assert live.exists()
    assert "in use by" in caplog.text


def test_force_deletes_scratch_that_still_looks_in_use(tmp_path):
    """The one escape hatch, for the one case the liveness rule cannot get right: a machine
    rebooted while scratch was open and the owning pid has since been REUSED, so nothing will
    ever call that file dead on its own."""
    d = _aligned(tmp_path, "MUR", "aoi1")
    live = _scratch(d / "aoi1_20230101.nc", f"{store._HOST}-{os.getppid()}")

    assert store.repair([], [live], force=True) == 1
    assert not live.exists()


def test_scan_can_be_limited_to_one_aoi(tmp_path):
    (_aligned(tmp_path, "MUR", "aoi1") / "aoi1_20230101.nc").touch()
    (_aligned(tmp_path, "MUR", "aoi2") / "aoi2_20230101.nc").touch()

    _, bad, _, _ = store.scan(tmp_path, aois=["aoi1"])
    assert [f.name for _, f in bad] == ["aoi1_20230101.nc"]
