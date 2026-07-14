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
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
from pathlib import Path

import xarray as xr

log = logging.getLogger(__name__)

PART_SUFFIX = ".part-"     # a write in progress; junk if the run dies
OLD_SUFFIX = ".old-"       # the previous output, kept only until the swap succeeds

# The variables a finished file of each product MUST carry. A file missing one of these is
# not "a file with a gap" -- it is a file whose write or whose source layers did not
# complete, and it must be re-fetched rather than skipped.
#
# Products whose channel set is CONFIG-DEPENDENT (met's variable list, CMEMS's requested
# variables + depths) cannot name their columns up front, so they assert the invariant they
# do have: the file opens and carries at least one data variable. Truncated payloads in
# those are caught by the deep pass, not the metadata check.
REQUIRED_VARS: dict[str, tuple[str, ...]] = {
    "MUR": ("sst", "valid"),
    "MODIS": ("sst", "valid"),
    "ECOSTRESS": ("sst", "water", "cloud", "valid"),
    "LANDSAT": ("sst", "water", "cloud", "valid"),
    "BATHYMETRY": ("elevation", "depth", "depth_p25", "depth_p75"),
    "LANDCOVER": ("landcover", "water"),
    "TIDE": ("tide",),
    "INSITU": ("sst", "qc"),
    "CMEMS": ("valid",),          # + >=1 configured variable, asserted by the >=1 rule
    "MET": (),                    # config-dependent channel set
}


def _tag() -> str:
    """A scratch-name suffix unique to this process and moment, so two runs writing the
    same output cannot land on each other's scratch."""
    return f"{os.getpid()}-{int(time.time() * 1000)}"


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


def sweep_scratch(dest: Path) -> None:
    """Discard scratch left beside `dest` by a run that died mid-write.

    This warns, and it is the ONLY trace the user gets that an earlier run failed partway
    through a write -- the output itself looks untouched, because that is the whole point.
    """
    for stale in sorted(dest.parent.glob(f"{dest.name}{PART_SUFFIX}*")):
        log.warning("  discarding %s left by an unfinished run", stale.name)
        _rm(stale)


def _swap(tmp: Path, dest: Path) -> None:
    """Move a COMPLETED scratch write onto the final path."""
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
    try:
        os.replace(tmp, dest)
    except OSError:
        if stash is not None:      # put the old output back: leaving NOTHING is worse
            stash.rename(dest)
        raise
    if stash is not None:
        _rm(stash)


@contextlib.contextmanager
def atomic(dest: Path):
    """Yield a scratch path to write to; swap it onto `dest` only if the write RETURNS.

        with store.atomic(path) as tmp:
            ds.to_netcdf(tmp)

    On any failure -- including KeyboardInterrupt, which is how a long run usually dies --
    the scratch is removed and `dest` is left exactly as it was.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sweep_scratch(dest)
    tmp = dest.with_name(f"{dest.name}{PART_SUFFIX}{_tag()}")
    try:
        yield tmp
        if not tmp.exists():
            raise RuntimeError(f"nothing was written to {tmp.name}")
        _swap(tmp, dest)
    except BaseException:
        # BaseException, not Exception: a Ctrl-C halfway through the payload must not
        # leave a half-file behind either.
        _rm(tmp)
        raise


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_netcdf(ds: xr.Dataset, path: Path, encoding: dict | None = None) -> Path:
    """Atomically write `ds` to `path`."""
    if encoding is None:
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    with atomic(path) as tmp:
        ds.to_netcdf(tmp, encoding=encoding)
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


def done(path: Path, expected_vars=(), *, shape=None, overwrite: bool = False) -> bool:
    """The skip guard. True -> this output is finished and can be skipped.

    A path that exists but is NOT complete is reported and returns False, so the caller
    re-fetches it: silently skipping a broken file is the failure this module exists to
    prevent, and silently overwriting one would hide that a run had died.
    """
    path = Path(path)
    if overwrite or not path.exists():
        return False
    if is_complete(path, expected_vars, shape=shape):
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

    Returns (n_checked, bad, leftovers): the files that failed, and any scratch stranded by
    a run that died mid-write.
    """
    root = Path(root)
    bad: list[tuple[str, Path]] = []
    n = 0
    for product, required in REQUIRED_VARS.items():
        base = root / product / "aligned"
        if not base.exists():
            continue
        for aoi_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if aois and aoi_dir.name not in aois:
                continue
            for f in sorted(aoi_dir.glob("*.nc")):
                n += 1
                if not is_complete(f, required, deep=deep):
                    bad.append((product, f))
    leftovers = sorted(p for p in root.rglob(f"*{PART_SUFFIX}*"))
    return n, bad, leftovers


def repair(bad, leftovers) -> int:
    """Delete what `scan` condemned, so the next run re-fetches it.

    Deleting is the repair: with the skip guard now asking `is_complete`, simply REMOVING a
    bad file is enough -- the next run sees no output and fetches it again. We delete rather
    than leave it in place so that a tree can be made trustworthy without a full --overwrite
    re-download of everything that was already fine.
    """
    for _, f in bad:
        _rm(f)
    for p in leftovers:
        _rm(p)
    return len(bad) + len(leftovers)
