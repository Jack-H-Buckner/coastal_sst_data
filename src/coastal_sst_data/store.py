#!/usr/bin/env python3
"""
coastal_sst_data -- durable output: writes that cannot be half-done, skips that cannot be
fooled, and a pass that re-checks what earlier runs left behind.

Every acquisition stage resumes by asking "did I already do this day?", and until now it
asked the FILESYSTEM -- `path.exists()`. That is a different question from the one it
means to ask. `to_netcdf` lays down the HDF5 superblock and the variable definitions
first and streams the payload after, so a run killed mid-write (dropped connection,
Ctrl-C, OOM) leaves a file that EXISTS, OPENS CLEANLY, and reports the right dimensions
while missing some or all of its data. `exists()` says True, the day is logged as "already
processed", and it is never repaired -- so a truncated download silently becomes a NaN day
in every cube built from that tree afterwards.

Two mechanisms fix that, and they only work together:

  * ATOMIC WRITE (`atomic`). Write to scratch beside the target, and os.replace() it into
    place only once the writer has RETURNED. os.replace is atomic on POSIX, so the final
    path only ever holds a finished write; a crash leaves a clearly-named `.part-*` behind
    (swept on the next attempt) and the PREVIOUS file untouched.

  * COMPLETENESS CHECK (`is_complete`). Atomicity is forward-looking. It cannot vouch for
    files an older run already left on disk, and it cannot tell a truncated write from a
    COMPLETE write of degraded content -- an ECOSTRESS granule whose cloud mask failed to
    download is a perfectly intact file that is missing a layer. So the skip guards ask
    whether the file carries the variables and the grid we expect, not whether a path is
    there.

`acquired_at` is NOT usable as a completion marker, tempting though it looks: attrs are
set before `to_netcdf` is called and HDF5 writes them early, so a truncated file still
carries a perfectly good stamp. The stamp proves the write STARTED. Only the data proves
it finished -- which is why the deep check (`--validate`) reads the payload rather than
trusting the header.

WHOSE SCRATCH IS IT. Atomicity assumed one writer per destination, and cleaning up after a
dead run was therefore "delete every `<dest>.part-*` you find". That is wrong the moment two
runs share an output tree -- a Slurm array with overlapping `--aoi` lists, two shells, a job
relaunched before the old one died -- because the sweep cannot tell a dead run's leftovers
from a live run's open file. It deleted the live one, and the writer failed mid-file on a
path it had every reason to think it owned. (The reopen that turns a deleted scratch file
into a hard failure is xarray's `CachingFileManager`: it flips `mode="w"` to `"a"` after the
first open and re-acquires BY PATH, so an eviction from the 128-entry global `FILE_CACHE`
sends it back to a filename that is no longer there. This has nothing to do with dask -- it
happens to plain numpy-backed writes just the same.)

So scratch now says WHO made it (`<host>-<pid>-<tid>-<ms>`) and the sweep asks whether that
writer is still around, using the strongest evidence available:

    this host, this pid   -> the in-process registry: is it open right now?
    this host, other pid  -> os.kill(pid, 0): is that process still running?
    another host, or a
    tag we cannot read    -> the clock: has anything under it been touched recently?

Only the first two are PROOF. The clock is a guess, so it is deliberately generous
(`STALE_SCRATCH_S`) and it errs toward keeping: a stranded file costs a line in `check`,
a wrongly-deleted one costs another run the work it was in the middle of.
"""

from __future__ import annotations

import contextlib
import glob as globlib
import logging
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .products import REGISTRY

log = logging.getLogger(__name__)

PART_SUFFIX = ".part-"     # a write in progress; junk if the run dies
OLD_SUFFIX = ".old-"       # the previous output, kept only until the swap succeeds
SCRATCH_SUFFIXES = (PART_SUFFIX, OLD_SUFFIX)

# How long scratch owned by a run we CANNOT interrogate -- another host, or a tag written by
# an older version -- must sit untouched before we call it dead. Generous on purpose: this
# tier is a guess, and the two tiers above it (the registry, and os.kill) already clear the
# common cases the moment the owning process is gone, so nothing waits on this clock unless
# the writer was on another machine.
STALE_SCRATCH_S = 6 * 3600

# The host half of a scratch file's owner token. Hyphens are NOT sanitised out: `-` is the
# field separator, and the tag is parsed from the RIGHT so a hyphenated hostname survives
# intact. Collapsing `node-1` and `node_1` onto one name would make two machines look like
# one, which is the false-"that's mine" that deletes a live writer's file.
_HOST = socket.gethostname().split(".")[0] or "unknown"

# The variables a finished file of each product MUST carry, keyed by its ALLCAPS output
# directory. A file missing one of these is not "a file with a gap" -- it is a file whose
# write, or whose source layers, did not complete, and it must be re-fetched rather than
# skipped.
#
# DERIVED from the product registry (products.ProductSpec.required_vars), so a new product
# cannot be added with its completeness check forgotten -- which was a silent failure, not a
# loud one: without an entry here the skip guard falls back to "the file opens and holds at
# least one variable", so a truncated download would be taken for done on every subsequent
# run.
#
# Products whose channel set is CONFIG-DEPENDENT (met's variable list, CMEMS's requested
# variables x depths) cannot name their columns up front, so they declare `()` and assert
# only the invariant they do have: the file opens and carries at least one data variable.
# Truncated payloads in those are caught by the deep `check` pass, not the metadata check.
REQUIRED_VARS: dict[str, tuple[str, ...]] = {s.dir: s.required_vars for s in REGISTRY}


# The scratch this process has OPEN right now: {scratch path -> its destination}. The only
# tier of `is_live_scratch` that is certain in both directions -- it is what lets us reclaim
# scratch THIS pid abandoned (a swap that raised, an `_rm` that failed on NFS) without
# waiting out a clock, and what stops one thread sweeping another thread's open file.
_ACTIVE: dict[Path, Path] = {}
_ACTIVE_LOCK = threading.Lock()


def _tag() -> str:
    """A scratch-name suffix unique to this HOST, process, THREAD and moment.

    Unique, and -- the part that matters -- ATTRIBUTABLE: `sweep_scratch` reads the host and
    pid back out to ask whether the writer still exists, instead of assuming that anything
    it finds is junk.

    The thread id is not redundant with the millisecond: once stages run on a pool, two
    workers can enter `atomic()` inside the same millisecond, and pid+ms alone would hand
    them the same scratch path -- one would then swap a file the other was still writing.
    """
    return f"{_HOST}-{os.getpid()}-{threading.get_ident()}-{int(time.time() * 1000)}"


def _owner_of(path: Path) -> tuple[str, int] | None:
    """The (host, pid) that made this scratch, or None if the tag cannot be read.

    Parsed from the RIGHT, because a hostname may contain the separator. None means "written
    by a version that did not stamp a host, or not a scratch name at all" -- and an unknown
    owner is never treated as ours.
    """
    for suffix in SCRATCH_SUFFIXES:
        _, sep, tail = path.name.partition(suffix)
        if not sep:
            continue
        parts = tail.rsplit("-", 3)
        if len(parts) != 4 or not parts[0] or not all(p.isdigit() for p in parts[1:]):
            return None
        return parts[0], int(parts[1])
    return None


def _pid_alive(pid: int) -> bool:
    """Is `pid` a live process on THIS host? Every uncertain answer is 'yes'."""
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:            # another user's process: alive, just not ours to signal
        return True
    except OSError:                    # no signals here (non-POSIX): assume the worst
        return True
    return True


def _newest_mtime(path: Path, *, at_least: float = float("inf")) -> float:
    """The most recent mtime anywhere under `path`, stopping once it reaches `at_least`.

    A DIRECTORY's own mtime is not a liveness signal: a staged Zarr's top level stops moving
    once its first block lands, while the assembler spends hours writing chunks underneath
    it. Ask the chunk files. `at_least` keeps that from being a full walk of a large store --
    the caller only needs to know which side of its cutoff the answer falls on.
    """
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0
    if newest >= at_least or not path.is_dir():
        return newest
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in (dirpath, *(os.path.join(dirpath, f) for f in filenames)):
            try:
                newest = max(newest, os.stat(name).st_mtime)
            except OSError:            # vanished under us; something else is clearly busy here
                continue
            if newest >= at_least:
                return newest
    return newest


def _register(tmp: Path, dest: Path) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE[Path(tmp)] = Path(dest)


def _unregister(tmp: Path) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(Path(tmp), None)


def _is_scratch_name(name: str) -> bool:
    return any(s in name for s in SCRATCH_SUFFIXES)


def scratch_owner(path: Path) -> str:
    """'host:pid' that made this scratch, for a human-readable report."""
    owner = _owner_of(Path(path))
    return f"{owner[0]}:{owner[1]}" if owner else "an unidentified run"


def _why_dead(path: Path) -> str:
    """WHICH tier condemned this scratch -- so a discard in the log can be judged.

    "its pid is gone" is routine. "no host/pid in its name" means scratch from before the
    tag carried an owner, which should stop appearing a few runs after an upgrade. "untouched
    for Nh" is the only guess of the three, and the only one that could ever be wrong about a
    live writer, so it is the one worth noticing in a log.
    """
    owner = _owner_of(Path(path))
    if owner is None:
        return "no host/pid in its name (written before this version)"
    if owner[0] != _HOST:
        return f"{owner[0]}:{owner[1]} is on another host and it has been untouched too long"
    if owner[1] == os.getpid():
        return "we started it ourselves and are no longer writing it"
    return f"its process ({owner[0]}:{owner[1]}) is gone"


def is_live_scratch(path: Path, *, max_age_s: float | None = None) -> bool:
    """Might something still be WRITING this scratch? Errs toward yes.

    The three tiers, strongest evidence first (see the module docstring). Only the first two
    are proof; the third is a clock, and a clock cannot distinguish a dead run from a writer
    stalled on a slow download -- which is exactly what a lazy CMEMS or ARCO read looks like
    from outside, the scratch file created and its mtime frozen while chunks trickle in.
    """
    path = Path(path)
    with _ACTIVE_LOCK:
        if path in _ACTIVE:
            return True                    # a thread here has it open right now
    owner = _owner_of(path)
    if owner is not None and owner[0] == _HOST:
        if owner[1] == os.getpid():
            return False                   # ours, and not open -> we abandoned it
        # The writer's process answers for it in BOTH directions, and the clock does not get
        # to overrule it. Letting a live pid fall through to the age test reintroduces the
        # original bug on a longer fuse: a writer stalled on a slow read has a frozen mtime
        # while being entirely alive, and after STALE_SCRATCH_S its file would be deleted.
        # The cost is a leak -- a pid REUSED after a reboot keeps dead scratch looking live
        # forever -- and a leak is visible in `check` and clears with `--repair --force`.
        return _pid_alive(owner[1])
    cutoff = time.time() - (STALE_SCRATCH_S if max_age_s is None else max_age_s)
    return _newest_mtime(path, at_least=cutoff) > cutoff


def unique_suffix() -> str:
    """A host/process/thread/moment-unique name fragment, for scratch that is NOT an
    `atomic()` destination -- a per-granule download directory, say.

    Such a directory is torn down with `shutil.rmtree` when its item finishes, so a name
    built only from what the item IS (its AoI and timestamp) is the same name in every
    concurrent run: two runs over overlapping dates download into one directory and each
    deletes the other's half-fetched granule. The identity has to be in the name.
    """
    return _tag()


def scratch_beside(dest: Path) -> list[Path]:
    """Every scratch path sharing this destination, whoever wrote it."""
    dest = Path(dest)
    pattern = globlib.escape(dest.name)
    return sorted(p for s in SCRATCH_SUFFIXES for p in dest.parent.glob(f"{pattern}{s}*"))


def _rm(path: Path) -> None:
    """Best-effort recursive delete. Never raises: failing to clean up scratch must not
    fail a run that has already written its output."""
    if not path.exists():          # the writer may have failed before creating anything
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:         # NFS .nfs* leftovers, etc.
        log.warning("  could not remove %s (%s); delete it by hand", path.name, exc)


def sweep_scratch(dest: Path, *, max_age_s: float | None = None) -> list[Path]:
    """Discard scratch beside `dest` left by a run that DIED -- and only that.

    Returns the scratch it LEFT ALONE, because someone may still be filling it. A caller
    that gets a non-empty list has learned that another run is writing this destination.

    The warning on a discard is the ONLY trace the user gets that an earlier run failed
    partway through a write -- the output itself looks untouched, because that is the whole
    point. The warning on a KEEP matters just as much: it is the only sign that two runs are
    writing one file, which is safe (unique scratch names, an atomic swap, last writer wins)
    but is almost always an unintended overlap in `--aoi` lists or date ranges.
    """
    live: list[Path] = []
    for stale in scratch_beside(dest):
        if is_live_scratch(stale, max_age_s=max_age_s):
            live.append(stale)
            log.warning("  leaving %s alone -- %s may still be writing it",
                        stale.name, scratch_owner(stale))
            continue
        log.warning("  discarding %s left by an unfinished run -- %s",
                    stale.name, _why_dead(stale))
        _rm(stale)
    return live


def _swap(tmp: Path, dest: Path) -> None:
    """Move a COMPLETED scratch write onto the final path.

    A DIRECTORY destination is the interesting case, and it is not fully serialisable: two
    processes swapping one store can interleave so that the second finds `dest` already
    moved aside, takes the no-stash branch, and then fails ENOTEMPTY against the first's
    freshly-installed directory. Nothing is corrupted -- the first swap is complete and
    intact -- but the second loses its work to a spurious error, which is why `atomic` warns
    as soon as it sees another run writing the same destination.
    """
    if not tmp.is_dir():
        os.replace(tmp, dest)      # atomic, and overwrites an existing file in one step
        return
    # os.replace() cannot clobber a non-empty directory, so an existing one moves aside
    # first -- by rename, which is atomic and survives open handles (an rmtree in place
    # fails on NFS when a reader still holds a chunk).
    stash = None
    if dest.exists():
        stash = dest.with_name(f"{dest.name}{OLD_SUFFIX}{_tag()}")
        dest.rename(stash)
        # REGISTERED, because the stash is the only copy of the previous output until the
        # replace lands. It carries our own tag, so without this a concurrent sweep in this
        # same process would read it as "ours and not open" -- provably dead -- and delete
        # the rollback out from under us.
        _register(stash, dest)
    try:
        os.replace(tmp, dest)
    except OSError:
        if stash is not None:      # put the old output back: leaving NOTHING is worse
            stash.rename(dest)
        raise
    finally:
        if stash is not None:
            _unregister(stash)
    if stash is not None:
        _rm(stash)


@contextlib.contextmanager
def atomic(dest: Path, *, placeholder: bool = False):
    """Yield a scratch path to write to; swap it onto `dest` only if the write RETURNS.

        with store.atomic(path) as tmp:
            ds.to_netcdf(tmp)

    On any failure -- including KeyboardInterrupt, which is how a long run usually dies --
    the scratch is removed and `dest` is left exactly as it was.

    `placeholder=True` creates the scratch as an EMPTY FILE before yielding, for writers that
    are noisy about being handed a path that does not exist yet. libnetcdf is: its
    `NC_infermodel` probes the target with `H5Fis_accessible()` before every create, with no
    existence check, so creating a new file prints a full `HDF5-DIAG` stack ending
    `errno = 2` -- about a file that is missing only because it is one instruction away from
    being created. netcdf-c ignores the answer and the write succeeds; the block is pure
    stderr noise, and there is no Python exception to catch or suppress.

    Parallel runs saw it and serial runs did not, from identical code: HDF5's error stack is
    thread-local, and libnetcdf silences it once on whichever thread initialised HDF5 -- the
    main one -- so every pool worker started printing again. An empty file is enough to make
    the probe answer "not HDF5" quietly instead of "cannot open", and every writer that wants
    this truncates what it finds anyway.

    The cost is that the emptiness check below can no longer be `exists()`, since we made it
    exist. It becomes a SIZE check, which is a better test regardless: a writer that opened
    the file and wrote nothing used to pass.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    live = sweep_scratch(dest)
    if live:
        log.warning("  %s is being written by more than one run at once (%s); both writes "
                    "complete and whichever finishes LAST wins -- check for overlapping "
                    "--aoi lists or date ranges", dest.name,
                    ", ".join(sorted({scratch_owner(p) for p in live})))
    tmp = dest.with_name(f"{dest.name}{PART_SUFFIX}{_tag()}")
    # Claimed BEFORE the yield, not once the writer has created something: a sweep that ran
    # in the gap would find no file to glob, so registering first leaves no window at all.
    _register(tmp, dest)
    if placeholder:
        tmp.touch()
    try:
        yield tmp
        if not tmp.exists() or (placeholder and tmp.is_file() and tmp.stat().st_size == 0):
            raise RuntimeError(f"nothing was written to {tmp.name}")
        _swap(tmp, dest)
    except BaseException:
        # BaseException, not Exception: a Ctrl-C halfway through the payload must not
        # leave a half-file behind either.
        #
        # SAID OUT LOUD, because this is the other way a scratch file disappears mid-write:
        # not another run sweeping it, but THIS writer cleaning up after a failure of its
        # own. HDF5 will often print an `errno = 2` stack of its own as it unwinds over the
        # file we just removed, and without this line that stack looks identical to the
        # concurrency bug. It is not -- the cause is whatever raised, one frame up.
        log.info("  removing %s after a failed write", tmp.name)
        _rm(tmp)
        raise
    finally:
        _unregister(tmp)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
# The netCDF library every aligned file is written with, named rather than inferred.
#
# Left unset, xarray walks ("netcdf4", "h5netcdf", "scipy") and takes the first INSTALLED
# one -- so a conda env (which has netcdf4) and a bare `pip install .` (which does not; it
# is only in the `modis`/`all` extras) write through different HDF5 stacks, with different
# reopen and locking behaviour, from identical code. `zlib`/`complevel` are accepted by both,
# so nothing fails to reveal which one ran. Naming it makes a cube reproducible off one
# machine, and makes a future HDF5 report attributable to a known writer.
NETCDF_ENGINE = "netcdf4"


# The CF attrs a windowed COG read leaves behind. They are how a raster DECLARES its nodata
# and scaling, and they are meaningful on a raster -- but they ride along onto every array
# derived from one, and on a 0/1 MASK they are actively harmful. See `clear_cf_decode_attrs`.
_CF_DECODE_ATTRS = ("_FillValue", "scale_factor", "add_offset")


def clear_cf_decode_attrs(da: xr.DataArray) -> xr.DataArray:
    """Strip the CF decode attrs a COG read leaks onto a derived MASK -> a new DataArray.

    `grid.read_cog_window` reprojects, and rioxarray stamps the result with `_FillValue`
    (whatever nodata the read used), plus an identity `scale_factor`/`add_offset`. Any mask
    computed from that raster inherits them, and both are wrong for a mask:

      * `_FillValue: 0` on a 0/1 mask means every ZERO cell decodes to NaN the next time
        anything opens the file. 0 is not absent data on a mask -- it is CLEAR, or LAND, and
        it is half the information the layer carries.
      * the identity scale/offset alone promote a uint8 mask to float64 on read.

    Both places must be cleared: `to_netcdf` refuses a key that sits in `attrs` and
    `encoding` at once, and clearing only `encoding` leaves the ATTRIBUTE to be written,
    which is what actually does the damage.
    """
    out = da.copy(deep=False)
    out.attrs = {k: v for k, v in out.attrs.items() if k not in _CF_DECODE_ATTRS}
    out.encoding = {k: v for k, v in out.encoding.items() if k not in _CF_DECODE_ATTRS}
    return out


def _sanitize_fill_values(ds: xr.Dataset, encoding: dict) -> tuple[xr.Dataset, dict]:
    """Never write a `_FillValue` that OCCURS in the data -> (dataset, encoding) to write.

    A fill value marks data that is ABSENT. One that collides with a value actually present
    does not mark anything -- it DELETES that value for every future reader, silently, on
    decode.

    This is not hypothetical. Landsat's `cloud` mask went to disk as 0/1 carrying
    `_FillValue: 0` (inherited from the `masked=False` QA read), so every CLEAR cell came
    back as NaN; `datacube._read_granule` reads a NaN cloud cell as CLOUDY -- correct when
    NaN really means "unknown" -- and the sensor's entire validity mask went empty on every
    granule. With nothing valid anywhere, every granule of a day tied at zero and the day's
    mosaic collapsed onto whichever one happened to be read first, so a multi-scene AoI
    showed a single scene's footprint. No error, at any stage.

    The check is deliberately narrow: only a FINITE fill value can collide (NaN is the normal
    float nodata and never equals anything, including itself), so this costs one comparison
    pass over arrays that are about to be serialised anyway. A non-colliding fill -- NaN on a
    float field, -9999 on a DEM -- is doing its job and is left exactly as it is.
    """
    fixed = []
    out = ds.copy(deep=False)
    enc = {k: dict(v) for k, v in encoding.items()}
    for name in list(out.data_vars):
        da = out[name]
        fv = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
        if fv is None:
            continue
        fv = np.asarray(fv)
        if fv.dtype.kind == "f" and not np.isfinite(fv):
            continue
        values = np.asarray(da.values)
        if not np.any(values == fv):
            continue
        out[name] = clear_cf_decode_attrs(da)
        # Belt and braces: the attr is gone, and the encoding says explicitly "no fill".
        enc.setdefault(name, {})["_FillValue"] = None
        fixed.append(f"{name} (={fv.item()})")
    if fixed:
        log.warning("  %s: refusing to write a _FillValue that occurs in the data -- it "
                    "would decode back as NaN and delete those cells: %s",
                    getattr(ds, "name", "output"), ", ".join(fixed))
    return out, enc


def write_netcdf(ds: xr.Dataset, path: Path, encoding: dict | None = None) -> Path:
    """Atomically write `ds` to `path`."""
    if encoding is None:
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds, encoding = _sanitize_fill_values(ds, encoding)
    with atomic(path, placeholder=True) as tmp:      # see `atomic`: keeps libnetcdf quiet
        ds.to_netcdf(tmp, encoding=encoding, engine=NETCDF_ENGINE)
    return path


def write_rasters(ds: xr.Dataset, path: Path, layers) -> Path:
    """Atomically write one GeoTIFF per layer into the directory `path`.

    The whole DIRECTORY is staged and swapped, not each .tif -- a directory holding three
    of four bands is exactly the kind of half-output the skip guard would take for done.
    """
    with atomic(path) as tmp:
        tmp.mkdir(parents=True)
        for name, da in layers:
            da.rio.to_raster(tmp / f"{name}.tif")
    return path


def write_text(path: Path, text: str) -> Path:
    with atomic(path) as tmp:
        tmp.write_text(text)
    return path


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, fmt: str = "parquet") -> Path:
    """Atomically write one long-format table -> the path written.

    The package's only TABULAR output (the `extract` stage). It lives here, beside the raster
    writers, for one reason: `atomic`. A writer that opened the destination directly would be
    the single durable write in this package that can leave a half-file behind for the next
    run's `exists()` check to take for done.

    Parquet by default. The extraction table is tall and narrow -- one row per point x time x
    variable x stat -- with four label columns that are almost entirely repetition, so
    dictionary encoding makes it an order of magnitude smaller AND typed: a CSV round-trip
    turns the `time` column into strings and every NaT into an empty field that pandas reads
    back as an object column. CSV stays available because it opens in everything.

    pyarrow is OPTIONAL and lazily imported, like every other heavy backend here (extra:
    `extract`). Most projects never call this stage, and a base install must not grow a
    dependency for it. The ImportError is raised rather than downgraded to CSV: silently
    writing a different format than the one asked for is how you discover, a month later,
    that half your outputs are text.
    """
    out_dir = Path(out_dir)
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(f"parquet output needs pyarrow, which is optional: {exc}") from exc
        path = out_dir / f"{stem}.parquet"
        with atomic(path) as tmp:
            df.to_parquet(tmp, engine="pyarrow", index=False)
        return path
    if fmt == "csv":
        path = out_dir / f"{stem}.csv"
        with atomic(path) as tmp:
            df.to_csv(tmp, index=False)
        return path
    raise ValueError(f"unknown table format {fmt!r}; choose 'parquet' or 'csv'.")


def write_output(ds: xr.Dataset, out_dir: Path, stem: str, fmt: str = "netcdf",
                 *, encoding: dict | None = None) -> Path:
    """Write one aligned output in the configured format -> the path written.

    `stem` is the filename without an extension (see coastal_sst_data.naming, which owns
    the stamp convention it encodes). NetCDF gets `<stem>.nc`; GeoTIFF gets a DIRECTORY
    `<stem>/` holding one .tif per variable -- staged and swapped whole, because a
    directory carrying three of four bands is exactly the half-output the skip guard would
    take for done.

    This was ten near-identical `write_output`s, one per process, differing only in how
    they built the stem -- so the stem is now the caller's business and the writing is
    ours. Products whose output has no time dimension (bathymetry, land-cover) fall out of
    the same code path: a variable is squeezed only if it HAS a time dim.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "netcdf":
        return write_netcdf(ds, out_dir / f"{stem}.nc", encoding=encoding)
    if fmt == "geotiff":
        layers = [(v, ds[v].isel(time=0) if "time" in ds[v].dims else ds[v])
                  for v in ds.data_vars]
        return write_rasters(ds, out_dir / stem, layers)
    raise ValueError(f"Unknown output format: {fmt!r}; choose 'netcdf' or 'geotiff'.")


# --------------------------------------------------------------------------- #
# Completeness
# --------------------------------------------------------------------------- #
def is_complete(path: Path, expected_vars=(), *, shape=None, deep: bool = False) -> bool:
    """Does `path` hold a FINISHED output -- not merely a path that exists?

    Cheap by default: opening a Dataset reads metadata only, so this costs milliseconds
    and is safe to call on every skip decision. It catches an unopenable file, a missing
    layer, and a file written against a different grid.

    `deep=True` additionally forces the payload to be read, which is the only way to catch
    a file whose header is intact but whose chunks are truncated. That is a full read of
    every file, so it belongs in the one-off `--validate` pass, not the resume path.
    """
    path = Path(path)
    if not path.exists():
        return False

    if path.is_dir():                              # geotiff layout: one .tif per variable
        tifs = {f.stem for f in path.glob("*.tif")}
        if expected_vars:
            return set(expected_vars) <= tifs
        return bool(tifs)

    try:
        with xr.open_dataset(path) as ds:
            for v in expected_vars:
                if v not in ds.variables:
                    return False
                if shape is not None and ds[v].ndim >= 2 and tuple(ds[v].shape[-2:]) != tuple(shape):
                    return False
            if not expected_vars and not ds.data_vars:
                return False                       # opened, but carries nothing
            if deep:
                for v in (expected_vars or tuple(ds.data_vars)):
                    ds[v].load()                   # force the chunks off disk
    except Exception:                              # unopenable, truncated, wrong format
        return False
    return True


def _covers_range(path: Path, start, end) -> bool:
    """Does this SINGLE-FILE span product already cover [start, end]?

    Tides and insitu write one file for the whole window, so an extended date range yields
    the same filename and `is_complete` still passes -- the extension would be silently
    skipped. They stamp the window they were built for (`provenance.requested_range`), which
    this reads back and compares to the currently-configured range.

    A file that carries no `requested_start`/`requested_end` (built before this stamp, or
    unreadable) is treated as NOT covering, so it is rebuilt once and gains the stamp.
    """
    try:
        with xr.open_dataset(path) as ds:
            rs = ds.attrs.get("requested_start")
            re = ds.attrs.get("requested_end")
    except Exception:
        return False
    if rs is None or re is None:
        return False
    return pd.Timestamp(rs) <= pd.Timestamp(start) and pd.Timestamp(re) >= pd.Timestamp(end)


def done(path: Path, expected_vars=(), *, shape=None, covers=None, overwrite: bool = False) -> bool:
    """The skip guard. True -> this output is finished and can be skipped.

    A path that exists but is NOT complete is reported and returns False, so the caller
    re-fetches it: silently skipping a broken file is the failure this module exists to
    prevent, and silently overwriting one would hide that a run had died.

    `covers=(start, end)` is for SINGLE-FILE span products (tides, insitu): a complete file
    whose stamped requested range does not span the configured window is rebuilt, so extending
    the project's date range is not silently skipped. Per-day/per-scene products leave it None.
    """
    path = Path(path)
    if overwrite or not path.exists():
        return False
    if is_complete(path, expected_vars, shape=shape):
        if covers is not None and not _covers_range(path, *covers):
            log.info("  %s covers a narrower date range than requested; rebuilding", path.name)
            return False
        return True
    log.warning("  %s exists but is INCOMPLETE (missing layers, or a truncated write); "
                "re-fetching", path.name)
    return False


# --------------------------------------------------------------------------- #
# The one-off sweep of an existing tree
# --------------------------------------------------------------------------- #
def scan(root: Path, aois=None, *, deep: bool = True):
    """Walk an output tree and find every file that is not a finished output.

    This is the pass that pays off the debt of everything written BEFORE the writes became
    atomic -- a tree built by older runs can hold any number of truncated files, and each
    one is currently being skipped as done on every run. It reads the payload of every file
    (`deep`), because a truncated chunk is invisible to a metadata check.

    Returns (n_checked, bad, leftovers, in_use). `leftovers` is scratch whose writer is gone
    -- safe to delete. `in_use` is scratch something appears to be filling RIGHT NOW: it is
    reported and never repaired, because deleting it is how a concurrent run loses a file it
    was in the middle of writing.
    """
    root = Path(root)
    bad: list[tuple[str, Path]] = []
    n = 0
    for s in REGISTRY:
        # A DATA (stacked) product nests each source under `<DIR>/<source>/aligned`, so walking
        # only `<DIR>/aligned` would find NOTHING and SILENTLY validate none of it -- exactly
        # the omission this pass exists to catch. GLOB every source subtree (so config-
        # registered sources, e.g. CMEMS regional tags, are covered too, not just built-ins).
        if s.is_stacked_data:
            bases = sorted((root / s.dir).glob("*/aligned")) if (root / s.dir).exists() else []
        else:
            bases = [root / s.dir / "aligned"]
        for base in bases:
            if not base.exists():
                continue
            for aoi_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                if aois and aoi_dir.name not in aois:
                    continue
                for f in sorted(aoi_dir.glob("*.nc")):
                    n += 1
                    if not is_complete(f, s.required_vars, deep=deep):
                        bad.append((s.dir, f))
    leftovers: list[Path] = []
    in_use: list[Path] = []
    for p in _find_scratch(root):
        (in_use if is_live_scratch(p) else leftovers).append(p)
    return n, bad, leftovers, in_use


def _find_scratch(root: Path) -> list[Path]:
    """Every scratch path under `root`, without descending INTO one.

    `rglob("*.part-*")` walks the inside of an abandoned scratch Zarr and reports each of its
    chunk files as a separate leftover -- thousands of paths naming one dead write. The
    scratch directory itself is the unit; what is inside it goes with it.

    `.old-` counts too. A process killed between the rename and the replace in `_swap` leaves
    a full copy of the previous output that nothing has ever looked for, which for a cube is
    tens of gigabytes of silent leak.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            if _is_scratch_name(d):
                found.append(Path(dirpath) / d)
            else:
                keep.append(d)
        dirnames[:] = keep                 # in place: this is what prunes the walk
        found.extend(Path(dirpath) / f for f in filenames if _is_scratch_name(f))
    return sorted(found)


def repair(bad, leftovers, *, force: bool = False) -> int:
    """Delete what `scan` condemned, so the next run re-fetches it.

    Deleting is the repair: with the skip guard now asking `is_complete`, simply REMOVING a
    bad file is enough -- the next run sees no output and fetches it again. We delete rather
    than leave it in place so that a tree can be made trustworthy without a full --overwrite
    re-download of everything that was already fine.

    Scratch that looks live is refused even when it is handed to us. `scan` already filters
    it out; this is the second lock on the same door, because deleting a running job's
    in-flight write is the one mistake this pass must never make.

    `force` overrides that, and exists for exactly one situation: a machine rebooted while
    scratch was open, and the owning pid has since been REUSED by an unrelated process, so
    the liveness check will call a dead file live forever. Only use it when you know nothing
    is running against this tree.

    -> how many paths were actually removed.
    """
    removed = 0
    for _, f in bad:
        _rm(f)
        removed += 1
    for p in leftovers:
        if is_live_scratch(p) and not force:
            log.warning("  %s is in use by %s; NOT deleting it", p.name, scratch_owner(p))
            continue
        _rm(p)
        removed += 1
    return removed
