# Fix: tide & insitu outputs don't extend when the project date range grows

## Context

When a project's date range is extended and the pipeline is re-run, per-day / per-scene
products (MUR, Landsat, ECOSTRESS, MODIS, CMEMS, met) pick up the new dates automatically:
each day/scene is a distinct date-stamped filename, so new dates are simply new files that
don't exist yet and get fetched.

**Tides** and **insitu** are different — each writes a *single file spanning the whole
project window* (`<aoi>_tides.nc`, `<aoi>_insitu.nc`). Their skip guard
(`store.done`) only checks *file existence + required variables are present*; it has no
notion of the time axis or the configured date range. So an old, narrower-range file still
passes the "done" check, is skipped, and is never regenerated. In the assembled datacube the
extended dates then show up as NaN for the tide/insitu channels.

Root-cause locations:
- Tide guard — [tides.py:331](../src/coastal_sst_data/processes/tides.py#L331)
- Insitu guard — [insitu_ioos.py:269](../src/coastal_sst_data/processes/insitu_ioos.py#L269)
- The guard itself — [store.done](../src/coastal_sst_data/store.py#L249) / [store.is_complete](../src/coastal_sst_data/store.py#L211) (metadata only; no time-range awareness)

Today the only workaround is the manual `overwrite` flag.

**Intended outcome:** a normal (non-overwrite) run detects that an existing tide/insitu file
was built for a narrower range than currently configured and rebuilds it, so the datacube
extends cleanly. Scope is deliberately limited to tides + insitu (the two single-file span
products); the datacube's own coarse `zpath.exists() and not overwrite` guard
([datacube.py:1129](../src/coastal_sst_data/processes/datacube.py#L1129)) is left unchanged and
still requires `overwrite` to rebuild the cube.

## Approach — stamp the requested range, compare it in the guard

Chosen over a time-axis coverage check because insitu observations often don't reach the
window boundaries (a station may report nothing near `end_date`); comparing the *requested*
range that was written into the file is exact and produces no false re-fetches.

### 1. Record the requested range at write time

In each of the two writers, add the configured `start`/`end` to the dataset attrs alongside
the existing `provenance.stamp(eff)` call:

- [tides.py ~357](../src/coastal_sst_data/processes/tides.py#L357) — the `ds.attrs.update(...)`
  before `write_output`. `start, end` are already in scope (tides.py:317).
- [insitu_ioos.py ~314](../src/coastal_sst_data/processes/insitu_ioos.py#L314) — the
  `ds.attrs.update(...)` before `write_output`. `start, end` are already in scope
  (passed to `find_stations`, insitu_ioos.py:275).

Add attrs `requested_start=str(start), requested_end=str(end)`. Stamp the *config* values
(the plain `start_date`/`end_date`), NOT the tide builder's `end + 1 day` / `inclusive="left"`
internal convention — comparing config-to-config sidesteps that subtlety entirely.

Optionally factor this into a tiny helper in
[provenance.py](../src/coastal_sst_data/provenance.py) (e.g. `requested_range(start, end) -> dict`)
mirroring `stamp()`, but an inline dict in the two `attrs.update` calls is acceptable and keeps
the change scoped.

### 2. Make `store.done` range-aware (opt-in)

In [store.py](../src/coastal_sst_data/store.py), add an optional `covers=(start, end)` parameter
to `done()`, following the same pattern as the existing optional `shape` check:

- New private helper `_covers_range(path, start, end) -> bool`: open the file, read
  `requested_start`/`requested_end` attrs, return
  `pd.Timestamp(rs) <= pd.Timestamp(start) and pd.Timestamp(re) >= pd.Timestamp(end)`.
  A file **missing** the stamp (built before this change, or unreadable) returns `False` →
  it is rebuilt once, which is safe and gives it the stamp going forward.
- In `done()`: after `is_complete(...)` passes, if `covers is not None and not
  _covers_range(path, *covers)`, log an info line ("covers a narrower date range than
  requested; rebuilding") and return `False`.
- Confirm `pandas` is imported in store.py; add the import if not.

Leave `is_complete` untouched (it stays a pure metadata/payload check); coverage is a
separate concern layered in `done`, so every other product that calls `done` without
`covers` is unaffected.

### 3. Wire the two callers

- [tides.py:331](../src/coastal_sst_data/processes/tides.py#L331) →
  `store.done(out_path, store.REQUIRED_VARS["TIDE"], covers=(start, end), overwrite=overwrite)`
- [insitu_ioos.py:269](../src/coastal_sst_data/processes/insitu_ioos.py#L269) →
  `store.done(out_path, store.REQUIRED_VARS["INSITU"], covers=(start, end), overwrite=overwrite)`

### 4. Document the gotcha in DEVELOPMENT.md

Per the request, add a short subsection to [docs/DEVELOPMENT.md](DEVELOPMENT.md)
warning that **single-file span products** (currently tides & insitu) need extra care:
unlike per-day/per-scene products, one file covers the entire window, so the resume/skip
guard must be *range-aware* (`covers=`) or an extended date range will be silently skipped.
Any future product that writes one file for the whole window must do the same (stamp
`requested_start`/`requested_end` and pass `covers=` to `store.done`).

## Files to modify

- `src/coastal_sst_data/store.py` — `done()` gains `covers=`; new `_covers_range` helper
- `src/coastal_sst_data/processes/tides.py` — stamp requested range; pass `covers=` to guard
- `src/coastal_sst_data/processes/insitu_ioos.py` — stamp requested range; pass `covers=` to guard
- `src/coastal_sst_data/provenance.py` — (optional) `requested_range()` helper
- `docs/DEVELOPMENT.md` — single-file-span-product warning
- `tests/test_store.py`, `tests/test_tides.py`, `tests/test_insitu.py` — see below

## Verification

- **Unit (store):** add tests to `tests/test_store.py` mirroring the existing
  `done`/`is_complete` cases — a file stamped `requested_start/end` that spans a requested
  `covers` range → `done(...) is True`; a file stamped with a *narrower* range → `done(...,
  covers=(wider)) is False`; a complete file with **no** stamp + `covers=` → `False`; and
  `covers=None` preserves current behavior.
- **Round-trip (tides & insitu):** in `tests/test_tides.py` / `tests/test_insitu.py`, build
  an output for a short window, then re-run `run()` with an extended `end_date` and assert the
  file is rebuilt (its `time` axis / `requested_end` now reaches the new end) rather than
  skipped. Reuse the existing fixtures/mocks in those test files.
- **Full suite:** `pytest -q` (or at least `pytest tests/test_store.py tests/test_tides.py
  tests/test_insitu.py`) stays green — no other `store.done` caller passes `covers`, so
  behavior elsewhere is unchanged.
- **End-to-end sanity:** run the pipeline for a small AoI over a short window, extend
  `end_date` in the config, re-run *without* `overwrite`, and confirm the tide/insitu `.nc`
  files are regenerated and the resulting datacube's tide/insitu channels are populated
  (non-NaN) across the extended dates.
