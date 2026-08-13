# Plan: Bounded-memory preprocessing

## Context

The assembler now builds cubes a block of days at a time, so a long window on a large AOI no
longer costs memory proportional to `channels × days × cells` (see
`docs/plan-bounded-memory-assembly.md`, shipped). That work explicitly left one thing undone, and
it is the reason the OOM is not yet fixed end to end:

> **`preprocess.run` will OOM on the same AoI.** It does `xr.open_zarr(zpath)` → `preprocess_aoi`
> → `write_zarr` of the whole cube, and re-chunks the store back to `datacube.chunks` — which
> would undo a reduced time chunk.

`PreprocessContext.read` (preprocess.py:159-170) ends in `np.asarray(da.values)`, so every channel
a step reads becomes a dense `(T, H, W)` array, and every channel a step emits is held in
`ctx.channels` until the stage-end merge. On a Hobart-sized cube that is the same hundreds of GB
the assembler used to want — worse in places, because `_baseline_cutoff` promotes the baseline to
**float64** and holds five `(T,H,W)` arrays at once.

Outcome wanted: preprocessing whose peak memory is bounded by a block size, matching the
assembler, so `assemble` → `preprocess` completes on a large AOI.

---

## Approach

**Nine of the ten steps are already day-local** — each output day depends only on that day's input
— so time-blocking is simply correct for them. `water_line` broadcasts a per-day tide over a static
DEM; `fill_water` already loops `for t in range(...)`; `filter_cloud_cover` and
`filter_land_clouds` are elementwise; `flag_georef` and `correct_georef` work scene by scene.

Several of them are, however, **spatially non-local** — `fill_water` is a nearest-neighbour fill,
`correct_georef` translates whole scenes, `filter_cloud_cover` takes an AOI-wide scene mean. That
settles the axis: **blocking must be in time, not space.** A spatial tiling would need halos for
the fill and could not express a whole-scene translation at all.

**One step reduces over time:** `filter_clouds` with `method: sigma`, whose `_baseline_cutoff`
(cloud_filter.py:312-356) fits a climatology to the baseline and drops pixels below
`mean − n_sigma·σ`. That fit is already expressed as **masked normal equations** —

```python
XtX = np.einsum("ti,tj,tyx->yxij", X, X, M)   # (H,W,k,k)
Xtb = np.einsum("ti,tyx->yxi", X, Bz)         # (H,W,k)
cnt = M.sum(axis=0)                           # (H,W)
```

— every term a plain sum over `t`, with no median, percentile, sort or scipy anywhere in the
module. So the accumulator approach is **exact, not an approximation**: the same numbers, summed
in a different order.

---

## The climatology, as a two-pass accumulator

`σ` needs `β`, and `β` needs the completed sums, so this is genuinely two passes over the baseline
channel. (The one-pass identity `ssr = Σb² − β·Xtb` exists but subtracts two large near-equal
numbers — SST in Kelvin squared is ~81 000 per sample — to recover a residual of order 1. Rejected
deliberately.)

```
pass 1   per block:  XtX += einsum(X_blk, X_blk, M_blk)
                     Xtb += einsum(X_blk, Bz_blk)
                     cnt += M_blk.sum(0)
  solve:             beta = _batched_solve(XtX, Xtb)          # (H,W,k)
pass 2   per block:  ssr += sum((b_blk - X_blk @ beta)**2, axis=0)
  finish:            sigma = sqrt(ssr / max(cnt-k, 1)); sigma[cnt <= k] = NaN
main     per block:  cutoff = X_blk @ beta - n_sigma * sigma   # evaluated per block, never (T,H,W)
```

State is `(H,W,k,k) + (H,W,k) + 2×(H,W)` float64 — **k+… floats per pixel, independent of T**.
For Hobart at `k=3` that is ~206 MB, against a `(T,H,W)` float64 working set today.

The compact representation `(beta, sigma)` is also what the cutoff *is*: today's `(T,H,W)` cutoff
array is `X @ beta - n_sigma·sigma` broadcast, so evaluating it per block is not an approximation
either — it is the same expression, materialised later.

### `_harmonic_design` is the sharper trap

```python
if np.ptp(ctx.days.dayofyear.values) < 60 or (np.std(s) < 1e-3 and np.std(c) < 1e-3):
    log.info("  filter_clouds: DOY span too short for a seasonal harmonic; ...")
    return ones[:, None]
```

Asked per block, this is **true for every block** of any sensible size — so a `harmonic`
climatology would silently degrade to a constant one, everywhere, logged only at INFO. It is a
whole-window question and must be answered from `ctx.all_days`, with the design matrix then
**sliced** per block. `tests/test_preprocess.py:341` pins the fallback log, so the guard itself
must keep working — it just has to be asked about the right axis.

---

## Design

### Nine steps need no changes at all

`PreprocessContext.read` reads off `ctx.ds_cube` (preprocess.py:165-170). So if each block is
handed `ds_cube.isel(time=blk)` as its `ds_cube`, every read returns a block-shaped array and
every day-local step works **unmodified** — no signature changes, no per-step edits. `has`,
`base_channels` and `channels_with_prefix` scan `.variables`, which a time slice preserves.

That is what keeps this change small: the work is in the orchestrator and in `filter_clouds`, not
spread across ten steps.

### The context gains a window view

`PreprocessContext` takes the same shape the assembler's `AssemblyContext` already has:
`days` (this block) and `all_days` (the cube's axis), plus a `cache` shared across blocks and a
`window` dict holding whole-window statistics. The rule for step authors is the one already
documented for contributors in DEVELOPMENT.md §3c: **emit on `ctx.days`, decide on
`ctx.all_days`.**

### Declaring a whole-window statistic

`PreprocessStep` gains one optional field:

```python
window_stats: Callable[[PreprocessContext, list], Any] | None = None
```

Called once per AOI, before the main pass, with the context and the list of day-blocks; its return
value lands in `ctx.window[step.key]` for the step's own `fn` to read. Only `filter_clouds`
implements it. This keeps the registry the single source of truth — a future step that needs a
climatology declares it the same way, rather than the orchestrator growing a special case.

`_baseline_cutoff` splits into `baseline_climatology(ctx, blocks, ...) -> (beta, sigma, X)` (the
`window_stats` hook) and a per-block `_cutoff_from(beta, sigma, X_blk, n_sigma)`. A single-block
run takes the same path, so its numerics are unchanged.

### Chunking

Preprocess must **preserve the source store's on-disk time chunk** rather than recompute it from
`datacube.chunks`. Re-chunking is what would undo a reduced chunk the assembler deliberately chose,
and it forces a read-modify-write of every chunk. The block is then a whole number of *source*
chunks, so each block read is chunk-aligned on the way in and chunk-aligned on the way out.

`build_encoding(..., sizes={"time": len(all_days)})` — the escape hatch added for the assembler —
is what stops block 0 chunking the new store in blocks.

### Write path

Mirrors `_assemble_blocked` exactly, inside the `store.atomic` block preprocess already drives:
block 0 `write_zarr(..., consolidated=False)` with the encoding, blocks 1..N `append_zarr`
(`mode="a-"`, no encoding), then `finalize_cube(tmp, attrs)`.

### Attrs are the highest-risk part

Appending **replaces** a Zarr group's attrs. If `preprocess_channels` is lost, the *next* run's
`clobbered` check (preprocess.py:628-635) sees this run's own output as assembler channels and
**raises**. If `preprocess`/`code_version` are lost, `preprocess_steps_stale` returns True forever
and the stage silently redoes itself on every run. So `preprocess`, `preprocess_channels`,
`code_version`, `provenance`, `provenance_products`, `created_at` and `preprocessed_at` are all
re-stamped once by `finalize_cube` after the last block.

`was_derived` / `clobbered` / `stale` are whole-cube facts: they move out of the per-block path and
are computed once from the frozen channel census, before the loop. `drop_vars(stale)` still has to
be applied to **every** block slice.

### Corrections found while validating the plan

Four things in the approach above were wrong or under-specified:

1. **The census cannot be `preprocess_aoi` on a zero-day slice.** That function calls
   `provenance.collect` (the most expensive call in the stage), computes `clobbered`/`stale`,
   assigns and stamps. It needs a separate `preprocess_census` that stops after the step loop —
   exactly as `channel_census` stops after `_topo_order(CONTRIBUTORS)`.

2. **The blocks are dask-backed, which the assembler's never were.** `ds_cube.isel(time=...)`
   carries the *source store's* `encoding["chunks"]` / `preferred_chunks` / codecs on every
   variable straight into `to_zarr`; the append passes no `encoding=`, so the **stale encoding
   wins**. This is the likeliest "works on the 10-day fixture, explodes on the real cube" failure
   in the change. Needs a `_for_write(ds_blk, ...)` that scrubs `encoding` on `data_vars` only
   (the `time` coord's units/calendar must survive) and rechunks explicitly to the chunks it will
   be written with — which also bounds how much of the untouched source channels streams at once.

3. **`stat_scope: pooled` is a second whole-window reduction**, not just `pixel`. Same hook,
   scalar and `(k,k)` accumulators, and its `dof` sums over cells *and* time.

4. **The staleness test as I wrote it would not have caught the bug it was for.**
   `preprocess_steps_stale` reads only `preprocess` and `code_version`, so "run twice, second is a
   no-op" passes cleanly even if `preprocess_channels` was lost. The loss only surfaces on a
   **third** run with `overwrite=True`, where `was_derived` is empty and every derived channel
   looks clobbered. The test must run: blocked → plain re-run skips → **`overwrite=True` re-runs
   without raising** → `preprocess_channels` equals the derived set.

Smaller ones folded into the work: `_LogOnce` must attach to the `preprocess`, `cloud_filter` and
`georef` loggers (a filter on one does not see another's records); `resolve_block_days` gains a
`transient=` kwarg because preprocess's transients are structurally larger than the assembler's
(`_baseline_cutoff` alone is ~40 bytes/cell/day in float64, invisible to a channel census); the
read set is measured by instrumenting `ctx.read` during the census rather than guessed; and the
window entry is keyed on `(baseline, scope, seasonality)` so `filter_clouds` and
`filter_clouds_corrected` — which fit the *same* climatology — share one prepass.

### Sizing

`preprocess.block_days` and `preprocess.memory_budget_gb`, each falling back to the corresponding
`datacube` value when unset — the stages have genuinely different per-day footprints, since
preprocess holds both what it reads and what it derives.

Bytes-per-day is the sum over the 3-D channels a step *reads* plus the 3-D channels it *emits*.
The channel set comes from a **zero-day census** — `preprocess_aoi` over `ds_cube.isel(time=
slice(0, 0))` — reusing the trick the assembler already relies on, so the emitted set is named by
the code that emits it. Untouched source channels stay dask-backed and stream through `to_zarr`,
so they are not counted; the read set is over-estimated as "all 3-D source channels", which errs
toward smaller blocks.

`resolve_block_days` / `budget_bytes` are reused from `datacube` rather than reimplemented.

---

## Files

- `src/coastal_sst_data/processes/preprocess.py` — the context, the `window_stats` field, the
  census, the block loop in `run()`, and hoisting the whole-cube facts out of `preprocess_aoi`.
- `src/coastal_sst_data/processes/cloud_filter.py` — split `_baseline_cutoff` into the accumulator
  hook and the per-block cutoff; `_harmonic_design` off `ctx.all_days`.
- `src/coastal_sst_data/config.py` — two fields on `PreprocessSpec`.
- `src/coastal_sst_data/processes/georef.py` — cache the static per-AOI precompute (`ref`, `dref`)
  across blocks; otherwise it is recomputed per block.
- `tests/test_preprocess.py` — new tests (below).
- `README.md`, `docs/DEVELOPMENT.md` §5b, `examples/config.test.yaml` — the two knobs and the
  step-author rule.

### Order of work

1. Baseline the suite (expect the known pre-existing `test_preprocessed_golden_is_unchanged`
   signed-zero failure in `tide_coops`, unrelated to this).
2. Context gains `all_days` / `cache` / `window`; loaders and the static georef precompute cached.
   Observably a no-op.
3. `_baseline_cutoff` → `baseline_climatology` + per-block cutoff, with `_harmonic_design` on
   `all_days`. Still single-block; the sigma unit tests must not move.
4. Hoist the whole-cube facts out of `preprocess_aoi`; add the census.
5. Config fields, sizing, the block loop, `finalize_cube`.
6. Tests, then docs.

---

## Verification

```bash
pytest tests/test_preprocess.py tests/test_georef.py tests/test_datacube.py
pytest                                    # full suite
```

New tests, reusing `tests/test_preprocess.py`'s `_hand_cube` / `_pp` / `_sigma_cube` and
`tests/test_datacube.py`'s `_snapshot` / `_diff_snapshots` / `_fingerprint` / `_long_project`:

1. **`test_blocked_and_unblocked_preprocessing_agree`** — the full fixture over ~10 days, run
   through `run()` at `block_days=10` and `block_days=3`; day-local channels compared by
   fingerprint (**exact**), climatology-derived channels by `allclose(rtol=1e-9)`, and the drop
   flags **exactly** — a filter decision must not flip.
2. **`test_the_sigma_climatology_is_the_same_fit_however_it_is_blocked`** — the 365-day harmonic
   fixture (`tests/test_preprocess.py:318`), blocked; β and σ match the single-pass fit, and the
   seasonal outlier is still caught while the constant climatology still misses it.
3. **`test_the_harmonic_guard_reads_the_whole_window_not_a_block`** — 365 days at `block_days=30`;
   assert the harmonic is used (no `"too short"` log) and the outlier is still flagged. Without
   the `all_days` fix every block degrades to a constant.
4. **`test_a_blocked_run_is_idempotent`** — run twice; the second is skipped as not stale, proving
   `preprocess`/`code_version` survived the appends.
5. **`test_a_blocked_cube_does_not_trip_the_clobber_check`** — run twice with `overwrite`, proving
   `preprocess_channels` survived (otherwise the second run raises).
6. **`test_preprocess_preserves_the_stores_time_chunk`** — assemble with a reduced time chunk, then
   preprocess; the chunk is unchanged.
7. **`test_a_failure_part_way_through_leaves_the_assembled_cube_intact`** — monkeypatch
   `append_zarr` to raise on the 3rd call; the cube is byte-identical and no scratch remains.

End to end, and the point of the exercise:

```bash
coastal-sst-data assemble   --config configs/tasi_lst.yaml --aoi Hobart --overwrite
coastal-sst-data preprocess --config configs/tasi_lst.yaml --aoi Hobart
```

Peak RSS from `/usr/bin/time -v` (or `sacct --format=MaxRSS`) against the logged prediction, as was
done for the assembler (measured there: 666 MB → 110 MB on a 200-day cube).
