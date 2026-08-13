# Plan: Bounded-memory datacube assembly

## Context

A production run was **OOM-killed** by the kernel part-way into its fourth AoI:

```
07:54:09 INFO === assembling Macquarie_Harbour (2404 days, grid=616x614) ===
08:04:17 INFO   wrote Macquarie_Harbour.zarr  vars=18 ...
08:14:17 INFO   wrote Tamar_River.zarr        vars=18 ...
08:31:06 INFO   wrote Freycinet_National_Park.zarr vars=18 ...
08:31:06 INFO === assembling Hobart (2404 days, grid=1218x1507) ===
Killed
```

The cause is structural, not a defect in any one product. `assemble_aoi` (`processes/datacube.py`)
runs every contributor over the **whole** time axis, holds each channel as a dense NumPy
`(T, H, W)` array in `ctx.channels`, and only then builds the Dataset and writes it in one shot.
Peak memory is therefore

```
n_channels x T x H x W x 4 bytes
```

with no knob on it, and no dependence on how much memory the machine actually has.

Measured against the golden fixture, an assembled cube carries **34 three-dimensional channels
≈ 127 bytes per cell per day**. On Hobart's 1218x1507 grid that is **233 MB per day**, so the
2404-day window is a **560 GB** working set. A single `(time, y, x)` float32 channel there is
16.4 GiB. The three AoIs that succeeded did so only because they are 2.8-7x smaller in cell count
and because mostly-NaN pages compress well under memory pressure — Freycinet, at 5.9 GiB per
channel, already took roughly twice as long as the small AoIs, which is the swap/compressor cost
showing up as wall-clock.

**Outcome:** assembly whose peak memory is **bounded and configurable**, so the cost scales with a
block size rather than with the AoI size multiplied by the length of the date range.

### Decisions locked with the user

- **The knob is a block size in days**, not a block count: `datacube.block_days`. A count would let
  peak memory grow with the date range, which is the property we are trying to remove.
- **The default is derived from a memory budget**, detected from the environment (cgroup / Slurm /
  physical RAM) and overridable with `datacube.memory_budget_gb`. Physical RAM is not the job's
  allowance — that distinction is exactly why the server run died.
- **Detected memory is spent at 50%**, with a further **2x transient factor** over the raw channel
  arithmetic, to cover loader working memory, the Blosc compressor and the interpreter itself.
- **When the budget cannot hold one time chunk's worth of days, the time chunk is reduced** (loudly)
  rather than writing unaligned appends. See "Block size vs. chunk alignment" below for the cost
  that decision avoids.

---

## Approach

Two tiers, landed as separate commits.

**Tier 1** removes allocations that are simply wasted; it is a pure win with no design change and
must leave the golden cube byte-identical.

**Tier 2** assembles and writes the cube in blocks of days, making peak memory a function of
`block_days` instead of the full window.

**Single-block behaviour is preserved by construction.** When the computed block covers the whole
window, `run()` takes today's exact path (`assemble_aoi` -> `write_zarr_safe`). All new machinery
lives in the multi-block branch, so small AoIs — and the golden test — are untouched by argument
*and* by code path.

---

## Tier 1 — allocation fixes

Gate: `tests/test_datacube.py::test_golden_cube_is_unchanged` stays green **without**
`UPDATE_GOLDEN`.

### 1a. `load_clearest_overpass`: process per day, not per directory

Today the function allocates its destination arrays up front, then globs the whole directory
accumulating one scene's `(sst, cloud, valid, footprint)` per day into `best` — so at the end of the
glob loop the destinations and a full window of scene arrays are resident together, roughly 2x peak.

Group granules by day from their **filenames** first (new `scene_index(d, aoi_id, *, cache=None) ->
{day_stamp: [(dt, path), ...]}`), then process one day at a time and write straight into the
destination. Peak drops from `destinations + T scenes` to `destinations + 2 scenes`.

> Draining `best` as it is consumed does **not** help: the peak has already happened by the time the
> second loop starts.

Iterate each day's granules sorted by `(dt, path)`. Ties in valid-pixel count are currently broken
by filesystem order; sorting makes them deterministic (earliest scene wins a tie). The golden's only
two-scene day has unequal valid counts, so the golden does not move.

### 1b. Allocate the footprint only when it exists

`load_clearest_overpass` unconditionally allocates `np.full((T,H,W), -1, "int32")` — 16.4 GiB on
Hobart, per sensor — and then returns `None` when `saw_footprint` is False. `np.full` touches every
page, so this is committed memory, frequently for nothing.

Allocate after the scan instead. Keep the `saw_footprint` flag rather than allocating on first use,
so today's "some granules carried the layer, but no *chosen* scene did -> all -1 plus a warning"
case is preserved exactly.

### 1c. `build_insitu`: sparse accumulation

`build_insitu` allocates **19** dense `(T,H,W)` arrays — 4 `chans`, 3 `dts`, 1 `counts`, 4 float64
`sums`, 3 float64 `dt_sums`, 4 int32 `hits` — totalling **104 bytes per cell per day**, to hold data
that exists in a handful of station pixels. On Hobart that is 459 GB for the full cube.

Replace `sums` / `dt_sums` / `hits` with `cells: dict[(row, col), dict]` holding `(T,)` arrays, and
densify only the emitted `chans` / `dts` / `counts`. 104 -> 32 bytes per cell per day.

> **Keep the accumulator dtypes exactly as they are** (float64 sums, int32 hits, float32 out,
> `casting="unsafe"` in the `np.divide`). The ufunc's loop selection depends on them, and changing
> `hits` to int64 could change the rounding and move the golden.

**Rejected:** narrowing `cloud` to uint8. It is float32 and may carry fractional values for some
sensors; changing it would move the golden for a saving that blocking already delivers.

---

## Tier 2 — time-blocked assembly

### Config surface

Two additive fields on `config.DataCubeSpec`, which is `extra="forbid"` (so they must be declared
fields — there is no open bag). Both follow the repo's existing `"auto"`-sentinel idiom
(`GridSpec.target_crs`): the schema carries the sentinel, the **consumer** resolves it, because
`H`/`W` are per-AoI and the schema is project-global.

```python
block_days: int | Literal["auto"] = "auto"   # days assembled and written per pass
memory_budget_gb: float | None = None        # None -> detect
```

`datacube._build_eff` passes both through into the flat `eff` dict.

### Sizing

Budget resolution order, first hit wins:

```
datacube.memory_budget_gb  ->  $COASTAL_SST_DATA_MEM_GB  ->  $SLURM_MEM_PER_NODE
  ->  cgroup v2 /sys/fs/cgroup/memory.max  ->  SC_PAGE_SIZE x SC_PHYS_PAGES
```

A **detected** value is multiplied by 0.5; an explicit `memory_budget_gb` is taken as given. Floor
4 GiB. Standard library only — no new `psutil` dependency.

```
bytes_per_day = sum over census channels of  H*W*itemsize (3-D)  |  itemsize (1-D time)
max_days      = max(1, (budget - 512MiB) // (bytes_per_day * 2.0))
```

`resolve_block_days(eff, per_day, n_days) -> (block_days, time_chunk)`:

1. `max_days >= n_days` -> `(n_days, min(tc, n_days))` — the single-block path, today's behaviour.
2. `max_days >= tc` -> `block = (max_days // tc) * tc`, chunk `tc`.
3. otherwise -> step `tc` down the ladder `(64, 32, 16, 8, 4, 2, 1)` to the largest value
   `<= max_days`; `block = tc_eff`; **log a warning** naming the budget, the per-day cost, and the
   reduced chunk.

Worked examples for Hobart at 233 MB/day: 64 GB detected -> block 64, chunk 64, 38 blocks.
32 GB -> block 32, chunk 32. 16 GB -> block 16, chunk 16.

Log the budget, its source, the per-day cost, the block count and the predicted peak on one line, so
a future OOM is diagnosable from the log alone. After block 0, compare the prediction against actual
`sum(v.nbytes ...)` and `resource.getrusage(RUSAGE_SELF).ru_maxrss`, and warn if it is off by >20%.

#### Block size vs. chunk alignment

Rule 3 reduces the on-disk time chunk rather than accepting unaligned appends. An append that lands
mid-chunk forces Zarr to read-modify-write every touched chunk: on a 1218x1507 grid that is roughly
19 GB of decompress+recompress per block boundary, ~1.4 TB over a 2404-day cube. Alignment is not a
tidiness concern here, it is the difference between a working run and an unusable one.

### Channel census

`bytes_per_day` needs the channel set *before* any block runs. Get it by running the real
contributors over a **zero-length** day index and reading `ctx.channels` directly — no Dataset is
built, nothing is materialised, and every channel is named by the code that emits it, so there is no
second list to keep in sync.

```python
def channel_census(g, eff, days, *, cache=None) -> dict[str, tuple[tuple[str, ...], np.dtype]]
```

This is zero-day-safe for every loader and contributor: `didx` is empty, so every globbed file is
skipped *before* `open_dataset`; `_contribute_doy` yields empty arrays; `build_insitu` still builds
the station table (placement is the outer loop). Do **not** call `provenance.collect` or `coverage()`
from the census — the former is the single most expensive call in the assembler. Suppress package
logging during the pass, or every file-presence warning fires twice per AoI.

The census shares the block runs' `cache`, so it *warms* the scene index, the day-file index, the
tide LUT, the in-situ tables, the statics and `met_prefix`. It is close to free.

### The channel set must be frozen before block 0

`<sensor>_footprint_id` is the **only** channel whose existence depends on granule *contents* rather
than on directory or file presence. Under blocking, block 0 and block 7 can therefore disagree — and
both directions produce a store that **writes without error and then fails to open**:

```
ValueError: conflicting sizes for dimension 'time': length 2404 on 'mur_sst'
            and length 32 on {'time': 'modis_footprint_id'}
```

A variable missing from a later block is never extended; a variable absent from block 0 and present
later is created with only that block's length. Either way the cube is silently unreadable.

Fix: hoist the decision to a whole-window probe.

```python
def footprint_available(d, aoi_id, days, H, W, *, cache=None) -> bool
```

It walks the cached scene index applying the loader's exact guards and early-exits on the first hit
(usually one file open). `load_clearest_overpass` gains `footprint_present: bool | None = None`
(`None` keeps today's auto-detect, so standalone and test callers are unaffected); `_load_sensor`
always passes the probe's answer. Over a full window the probe equals today's `saw_footprint`, so
the golden holds.

Backstop: `_check_channel_set(channels, census, block_i, days)` raises if any block's emitted set,
dims or dtypes drift from the census — failing loudly at write time rather than silently at read
time. **This is the new invariant contributors must respect**, and it is documented in
DEVELOPMENT.md §3c: *a contributor's channel set must be a function of the files on disk, never of
`ctx.days`; use `ctx.all_days` for any whole-window decision.*

### Repeated-work cache

Every loader globs its own directory per call, so blocking would multiply whole-window work by the
block count (75 rescans of directories holding thousands of scene files, on a 2404-day run at 32-day
blocks). `AssemblyContext` gains `all_days` (the full cube axis) and `cache: dict`, created once per
AoI in `run()` and threaded into the loaders as a **keyword-only** `cache=None` — every existing
positional signature is unchanged, because user-written contributors call these loaders positionally
(`tests/test_add_a_covariate.py`).

| Site | Cache entry / fix |
|---|---|
| `load_clearest_overpass` glob | `("scenes", dir)` -> `scene_index` |
| `load_daily_sensor` glob (6x per met source per block) | `("dayfiles", dir, prefix)` -> `day_index` |
| **`met_prefix`** | `("met_prefix", dir, want)`, resolved against `ctx.all_days` |
| `load_tide_daily` (resamples the whole series) | `("tide_daily", dir)` -> `(lut_mean, lut_range)` |
| `water_level.load_tide_series` | `("tide_series", dir)` — cached in `datacube`, keeping `water_level` pure |
| `load_insitu` + `insitu.station_pixels` | `("insitu", base)` -> open, materialise to numpy, **close immediately** |
| `load_bathy`, `load_bathy_attrs`, `load_landcover` | `("bathy", dir)`, `("landcover", dir)` |
| `_overpass_met_vars`, `cmems_channels` | `("overpass_vars", dir)`, `("cmems_vars", dir)` |
| `footprint_available` | `("footprint", dir)` |
| `provenance.collect` | **hoisted out of the block path** — called once, at finalize |

> `met_prefix` is a **correctness** fix, not merely a performance one. It decides `ref_` vs
> daily-mean from the days it is handed, so per-block calls could pick different variants in
> different blocks and silently mix them into one forcing channel — with a `met_time` attr
> describing only the last block. Resolving it against `ctx.all_days` restores whole-window
> semantics identical to today's unblocked run.

### Write mechanics

All blocks write into **one** `store.atomic(zpath)` scratch directory. `atomic` already supports
this: it yields the scratch path, checks only that something exists, and swaps on clean return —
`store.write_rasters` is the existing precedent for a multi-write body. A run killed at block 40 of
76 therefore leaves the previous cube intact, exactly as today.

- **Block 0:** `to_zarr(tmp, mode="w-", consolidated=False, encoding=enc0)`
- **Blocks 1..N:** `append_zarr` -> `to_zarr(tmp, mode="a-", append_dim="time", consolidated=False)`,
  passing **no `encoding`** — xarray *raises* `variable ... already exists, but encoding was
  provided`. Use `mode="a-"`, not `"a"`, so the static `(y,x)` channels are written once by block 0
  instead of being rewritten in full on every block.
- **Finalize:** `finalize_cube(tmp, attrs)` — `zarr.open_group(mode="r+")`,
  `attrs.put({**dict(g.attrs), **attrs})`, then a single `zarr.consolidate_metadata(g.store)`.

> Appending **replaces the group's whole attrs dict** (zarr's `Attributes.put` clears first), so
> `coverage`, `provenance`, `config_yaml` and `insitu_stations` would be silently lost without the
> finalize step. This is the single most likely silent-data-loss bug in the change, and there is no
> existing helper in the package to crib from.

Signature changes, all backward compatible — `processes/preprocess.py` imports `build_encoding` and
`write_zarr` and calls them positionally, and drives its own `store.atomic` because its source store
must stay open across the write:

```python
def build_encoding(ds, compression, chunks, *, sizes=None)   # sizes overrides the chunk clamp
def write_zarr(ds, zpath, encoding, *, consolidated=True)    # unchanged default
def append_zarr(ds, zpath, *, append_dim="time")             # new
def finalize_cube(zpath, attrs)                              # new
```

`sizes=` matters: `build_encoding` clamps chunks to `ds.sizes`, so calling it on a *block* dataset
would chunk a 2404-day cube in 8s because block 0 happened to be 8 days long.

### Cross-block accumulators

- `coverage()` splits into `_coverage_hits(ds)` per block and `coverage_from_hits(...)` at finalize.
  `days_expected` stays the **full** length — that is the whole point of the coverage attr.
- Block attrs are folded with `_merge_block_attrs(acc, new, block_i)`, which raises on a key whose
  value changes between blocks. That is precisely how a `met_time`-style whole-window bug announces
  itself.
- `_contribute_stacked_sensor`'s merged overpass identity is computed per day index, so it is
  already day-local and needs no change. `SLOT_SENSOR_TIMES` / `SLOT_REF_UTC` are per-day lists —
  use a **fresh ctx per block**, never a reused one.
- `water_union` is computed (an OR over every granule) and returned, then discarded at both call
  sites. Delete it from the return tuple while restructuring 1a.

---

## Files

- `src/coastal_sst_data/processes/datacube.py` — all of Tier 1, most of Tier 2.
- `src/coastal_sst_data/config.py` — two fields on `DataCubeSpec`, plus docstring.
- `src/coastal_sst_data/processes/insitu.py` — optional batched `values_at` helper.
- `tests/test_datacube.py`, `tests/test_insitu.py` — new tests.
- `docs/DEVELOPMENT.md` §3c, `README.md` (the `datacube:` storage paragraph),
  `examples/config.test.yaml` — the two knobs and the new contributor invariant.

### Order of work

0. Commit this plan.
1. **Baseline** — full suite green. (At the time of writing: 709 passed, plus one *pre-existing*
   failure in `tests/test_preprocess.py::test_preprocessed_golden_is_unchanged`, a signed-zero
   `-0.0` vs `0.0` drift in `tide_coops` that is unrelated to this work.)
2. **Tier 1** (1a-1c). *Gate: the datacube golden is green without `UPDATE_GOLDEN`.*
3. **Infrastructure**, observably a no-op: `all_days`/`cache` on the context, the cached index
   helpers, `footprint_available`, splitting `assemble_aoi` into `_run_contributors` /
   `_dataset_from` / `assemble_block`, and the four write-function signatures. *Gate: suite green.*
4. **Tier 2**: config fields, `channel_census`, `bytes_per_day`, `resolve_block_days`,
   `_assemble_blocked`, and the `run()` branch.
5. **Docs.**

---

## Verification

```bash
pytest                                                          # full suite, offline
pytest tests/test_datacube.py::test_golden_cube_is_unchanged    # the Tier 1 gate
```

New tests, reusing the existing fixtures (`project`, `grids`, `days`, `_write_full_fixture`,
`write_modis`, `write_met_daily`, `_eff_with_overpass`, `_write_cube`) and the golden comparator
(`_snapshot`, `_diff_snapshots`, `_fingerprint`, `RUNSTAMP_ATTRS`). A `_long_project` helper is
needed — the 3-day `project` fixture is too short to block.

1. **`test_blocked_and_single_block_cubes_are_identical`** — the crown jewel. A 10-day full fixture
   assembled at `block_days=10` and at `block_days=3` (4 blocks with a short tail); the
   `_diff_snapshots` of the two stores is empty. `RUNSTAMP_ATTRS` already excludes `created_at`.
   Covers values, dtypes, dims, per-variable attrs, coverage, `met_time`, `insitu_stations` and the
   channel set in one assertion.
2. **`test_channel_census_matches_the_assembled_cube`** — census names, dims and dtypes equal
   `assemble_aoi(...).data_vars`. This is what keeps the census honest as channels are added; it
   would have caught the footprint case.
3. **`test_footprint_channel_survives_blocking_when_only_one_block_has_the_layer`** — MODIS over 6
   days with `footprint_id` on day 5 only, `block_days=2`; assert the cube **opens**, that
   `modis_footprint_id` spans the full axis, and that days 0-3 are -1. Then the mirror case (day 0
   only). Without the probe both raise on open — asserting by opening is the entire point.
4. **`test_met_prefix_is_decided_once_over_the_whole_window`** — daily-mean files for every day,
   `ref_` for the last day only, `block_days=1`; assert `met_time == "reference"` and that days 0-1
   are NaN rather than silently carrying daily-mean values.
5. **`test_auto_block_days_arithmetic`** — a unit test over all three `resolve_block_days` rules;
   `block % time_chunk == 0` in every case; `block >= 1` at an absurdly small budget.
6. **`test_chunking_is_the_full_cubes_not_the_blocks`** — 10 days in 2-day blocks with
   `chunks={"time": 4}` stores a time chunk of 4, not 2 (the `sizes=` guard).
7. **`test_attrs_survive_the_append`** — `coverage`, `insitu_stations`, `provenance`, `config_yaml`
   all present after a blocked write; `coverage["mur"]["days_expected"] == len(days)`.
8. **`test_a_failure_mid_way_through_a_blocked_write_leaves_the_previous_cube_intact`** — modelled
   on `test_write_zarr_safe_keeps_the_previous_cube_when_the_rewrite_dies`; monkeypatch
   `append_zarr` to raise on the 3rd call; the old cube is unchanged and no `*.part-*` / `*.old-*`
   remain.
9. **`test_provenance_is_collected_once_per_aoi`** (and the tide series, and the in-situ opens) —
   counting monkeypatches over a 4-block run.
10. **`test_no_int32_footprint_allocation_when_the_sensor_never_carried_the_layer`** — record
    `np.full` calls; assert no `(T,H,W)` int32 allocation.
11. **`test_two_stations_in_one_cell_average`** in `tests/test_insitu.py` — the sparse accumulator's
    one behavioural risk is the N-station mean, and the golden fixture has only one station.

End to end on the real config:

```bash
coastal-sst-data validate --config configs/tasi_lst.yaml
coastal-sst-data assemble --config configs/tasi_lst.yaml --aoi Hobart --overwrite
```

Confirm from the log that the budget line reports the detected limit and the block count; check peak
RSS (`/usr/bin/time -v`, or `sacct -j <id> --format=MaxRSS`) against the predicted peak; and confirm
the cube opens with `xr.open_zarr` at the full 2404-day length.

---

## Out of scope — flagged, not fixed

**~~`preprocess.run` will OOM on the same AoI.~~** *Done — see
`docs/plan-bounded-memory-preprocessing.md`.* It did `xr.open_zarr(zpath)` -> `preprocess_aoi` ->
`write_zarr` of the whole cube, and re-chunked the store back to `datacube.chunks`, undoing a
reduced time chunk. It is now blocked on the same axis, inherits the store's chunking rather than
overwriting it, and streams the one step that reduces over time (`filter_clouds` with
`method: sigma`) through an accumulator.

**Disk, not just RAM.** The `store.atomic` scratch cube and the previous cube coexist until the
swap, so the output filesystem needs roughly 2x the compressed cube.

**The in-situ channels are ~28% of per-day memory** — 9 dense `(T,H,W)` float32 channels carrying a
handful of station pixels, ~160 GB of a Hobart-sized cube. They compress to nearly nothing on disk,
so this is purely a memory and blocking tax. A `(station, time)` layout indexed by the existing
`insitu_station` map would buy back a third of the budget, but that is a cube-format change, not a
memory fix.
