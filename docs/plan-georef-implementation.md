# Implementation plan: `flag_georef` + `correct_georef` preprocess steps

Companion to [plan-georef-preprocess-step.md](plan-georef-preprocess-step.md) (the *design*). This
document is the *build plan*: it reviews/corrects the design against the current codebase and gives
the ordered implementation steps. Follow [DEVELOPMENT.md §5b](DEVELOPMENT.md#5b-adding-a-post-assembly-preprocess-step).

## Context

Some ECOSTRESS granules are geolocated wrong. A misregistered scene is worse than a missing one:
it contributes wrong SST at every coastal pixel and nothing downstream flags it. The validated
prototype (`/Users/johnbuckner/Documents/test_coastal_sst_data/src/georef_diagnose.py` +
`plot_thermal_edges.py`) registers each thermal scene's coastline against the static land-cover
coastline by brute-force whole-cell translation, yielding the magnitude **and direction** of the
error.

**Scope:** the design doc deferred *correction* to "Phase 2, later." This build does it now —
because the search already returns `(dy,dx)`, correction is a small addition (`shift_array`). Two
steps ship: `flag_georef` (diagnose + classify) and `correct_georef` (apply the shift).

## Review: corrections to the design doc

The design doc asserts a few things that are not true of the codebase; these are fixed as part of
this work.

1. **Region-override does NOT ride `config.resolve_opts`.** The doc claims `min_coast_obs` "maps
   directly onto the existing per-key merge in `config.resolve_opts`, the same mechanism
   `datum_offset_m` uses." It doesn't: `resolve_opts` ([config.py:221](../src/coastal_sst_data/config.py#L221))
   is **product-only** (keyed by `DataProduct`), and preprocess step bags are flattened
   **project-global** at [preprocess.py:549](../src/coastal_sst_data/processes/preprocess.py#L549)
   with no region merge. `regions[].sources` is typed `dict[DataProduct, SourceOptions]`, so the
   doc's `regions[].sources.flag_georef:` YAML would fail validation. We build real per-region
   step-option support instead (step 1).
2. **The 6-way flag classifier and the ±5/10/20/30 km stability sweep are NOT in the prototype.**
   The prototype has the validated search (`search_shift`), quality gates (`quality_reject`,
   `scene_quality`) and scoring (`_score`). The classification into
   `ok/displaced/suspect/unstable/unrecoverable/insufficient_signal` and the stability
   determination (design §5) are **specified but unwritten** — new code, added per the documented
   thresholds and validated against the regression table.
3. **The correction primitive already exists**: `shift_array(a, dy, dx, fill=np.nan)`
   (georef_diagnose.py:39-50) — slice-based (never `np.roll`), NaN-fills the vacated margin.
   Port it verbatim.
4. **Prototype location** is `/Users/johnbuckner/Documents/test_coastal_sst_data/src/`, not
   `github/test_coastal_sst_data/` (that tree holds only data).

## Decisions

- **Region-override:** build genuine per-region step-option support (extend the config).
- **Fit on the cloud-filtered SST; apply the shift to the raw data.** `flag_georef` estimates
  `(dy,dx)` from the working (cloud-filtered) SST — cleaner edges, better fit. `correct_georef`
  applies that same `(dy,dx)` to the **raw** ECOSTRESS fields read from `ds_raw`, never to the
  filtered/derived channels.
- **Correction operates on RAW ECOSTRESS geometry, not derived masks.** Shift the raw
  ECOSTRESS-native fields (SST + validity, and other native bands) into `*_georef_corrected`
  channels. Do **not** shift preprocess-derived products (cloud-filtered SST, water_line, met
  masks). The correction fixes the granule geometry; the cloud/water preprocessing must be
  **re-run** on the corrected geometry later (possibly dedicated georef-aware modules) — that
  re-run is explicit follow-up, out of scope here.
- **Correction applies to `displaced` scenes only**; every other scene is copied verbatim so the
  corrected channel is a complete drop-in.

## Reuse — do not reimplement

| Need | Reuse | Source |
|---|---|---|
| whole-cell shift | `shift_array` | georef_diagnose.py:39-50 |
| NaN-aware edges | `canny`, `nan_gaussian`, `_shift` (NMS) | plot_thermal_edges.py:68-135 |
| reference coastline | `water_boundary`, degenerate-mask guard from `load_water_mask` | plot_thermal_edges.py:42-65,138-141 |
| shift search + surface stats | `search_shift`, `_score` | georef_diagnose.py:65-120 |
| gates / quality | `scene_quality`, `quality_reject` | georef_diagnose.py:123-186 |
| diagnostics vector | `coast_offsets` | georef_diagnose.py:53-62 |
| step scaffolding template | `cloud_filter.py` header/`TYPE_CHECKING`/`_working` pattern | processes/cloud_filter.py |
| in-run "working" read | `_working` (a channel emitted this run wins over raw) | cloud_filter.py:61 |
| 1-D `("time",)` emit | `<pre>_scene_cloud_pct_<src>` pattern | cloud_filter.py:304-305 |
| synthetic-cube tests + golden | `_hand_cube`, `write_ecostress_two_scenes`, `_snapshot`/`_diff_snapshots`, `UPDATE_GOLDEN=1` | tests/test_preprocess.py, tests/test_datacube.py |

## Implementation steps

### 1. Per-region step-option support (config)

Mirror the product `resolve_opts` two-level merge for steps.

- **`config.py`**: add a region field `Region.preprocess_steps: dict[str, PreprocessStepOptions] =
  {}` (open bag, same `_fill_step_defaults` normalization as `PreprocessSpec.steps`). Add
  `resolve_step_opts(project, aoi_name, key) -> dict`: global `project.preprocess.steps.get(key)`
  merged under region `project.region_of(aoi_name).preprocess_steps.get(key)` (region wins
  per-key), returning the merged extra dict.
- **`PreprocessStep`** (preprocess.py:113): add `region_option_keys: frozenset[str] =
  frozenset()` — the subset a region may override (guards "regions tune coverage/thresholds, not
  cube meaning," the product `region_options` principle). For `flag_georef`:
  `{"min_coast_obs", "min_edges", "min_valid_pct"}`.
- **`PreprocessContext.step_opts`** (preprocess.py:213): resolve per-AoI via
  `resolve_step_opts(self.eff["project"], self.aid, key)` instead of the global `eff["steps"]`
  bag. `eff` already carries `project` and `ctx.aid == g.name`, so `_build_eff`/`preprocess_aoi`
  signatures are unchanged.
- **`_check_step_options`** (preprocess.py:462): also validate each region's `preprocess_steps`
  bags against `option_keys`, and reject a region key not in that step's `region_option_keys`.
- Selection stays global (`selected = [s for s in STEPS if s.key in eff["steps"]]`) — a region
  overrides options of a globally-selected step; it cannot add one. Same as products.

### 2. New module `processes/georef.py`

Follow `cloud_filter.py` conventions exactly (`from __future__ import annotations`; module
docstring saying why it's outside `preprocess.py`; `if TYPE_CHECKING: from .preprocess import
PreprocessContext`; string forward-ref hints; `log = getLogger(__name__)`; `T3 =
("time","y","x")`). A step must **never raise** — read defensively via `ctx.read(...)` /
`ctx.has(...)` / `ctx.channels_with_prefix(...)`, degrade with a warning when landcover or the
sensor is absent.

Port the reuse-table functions verbatim (image ops get their package home here). Then **add the
classifier** (not in the prototype):

- `classify(...) -> (flag_int, label)` implementing design §5: `unstable` (fails the window sweep)
  → `unrecoverable` (`P_peak < lift_min*median`) → `ok` (`|shift| <= ok_shift_m` and `z >= z_min`)
  → `displaced` (`z >= z_min` and `gain >= gain_min` and stable) → else `suspect`.
  `gain = agree - agree0`.
- stability sweep: re-run `search_shift` at each `stability_windows_km` half-width; `stable` iff
  the best `(dy,dx)` moves ≤1–2 cells across windows.

`_step_flag_georef(ctx)`:
1. `sensors = opts.get("sensors") or ["eco"]`; iterate each `pre`.
2. Reference coastline once: `water = ctx.read("landcover_water", dims=("y","x"))`; guard a
   degenerate (all-water/all-land) mask per `load_water_mask`; `ref = water_boundary(water)`;
   precompute `distance_transform_edt(~ref)` (+ indices for `coast_offsets`).
3. Per version suffix (`for sst_name in ctx.channels_with_prefix(f"{pre}_sst"): suffix = ...`), per
   time slice: read the **working (cloud-filtered)** SST via the `_working` pattern, `canny` →
   edges, `scene_quality`, `quality_reject` (→ flag `insufficient_signal`, skip fit), else
   `search_shift` + stability sweep + `classify`.
4. Emit per-sensor 1-D `("time",)` diagnostics like `<pre>_scene_cloud_pct_<src>`:
   `<pre>_georef_dy/_dx` (int16), `_shift_m` (f32), `_z` (f32), `_agree/_agree0` (f32), `_n_edge`
   (int32), `_coast_obs` (int32), `_dt_k` (f32), `_flag` (uint8;
   `flag_values=[0..5]`, `flag_meanings="ok displaced suspect unstable unrecoverable
   insufficient_signal"`). Nothing 3-D; raw cube untouched.

Register in `STEPS`:
```python
PreprocessStep(
    key="flag_georef",
    reads=("_sst", "landcover_water"),
    writes=("_georef_",),
    fn=_step_flag_georef,
    depends_on=("filter_clouds", "filter_cloud_cover", "filter_land_clouds"),
    option_keys=frozenset({"sensors","tol_m","max_shift_m","coarse_stride","n_refine",
        "sigma","lo_pct","hi_pct","min_coast_obs","min_valid_pct","min_edges",
        "z_min","lift_min","gain_min","ok_shift_m","stability_windows_km"}),
    region_option_keys=frozenset({"min_coast_obs","min_edges","min_valid_pct"}),
    provenance_inputs=("ecostress","landcover"),
)
```
`depends_on` the three cloud filters is load-bearing: the fit must read the *filtered* SST.

### 3. `correct_georef` step (in `georef.py`)

`_step_correct_georef(ctx)` — `depends_on=("flag_georef",)`:
1. Read the in-run `(dy,dx)` and `_flag` from `flag_georef` (they live in `ctx.channels`, not
   `ds_raw`). Add a small `ctx.working(name)` helper (generalize `cloud_filter._working`) or read
   `ctx.channels[name]` directly.
2. Determine the raw ECOSTRESS-native spatial fields to shift: SST + validity by default, plus any
   other native bands present. Option `fields` (default the native set); shift only channels that
   are **raw** (present in `ds_raw`) and 3-D — never a preprocess-derived channel. Read each raw
   field from `ds_raw` via `ctx.read(name, dims=T3)`.
3. Per time index: if `flag[t] == displaced`, `shift_array(field[t], dy[t], dx[t])`; else copy
   verbatim. Emit `<field>_georef_corrected` (float32, T3). Also emit `<pre>_georef_applied`
   (uint8, `("time",)`, 1 where shifted).
4. Docstring states: corrected channels carry a **different geometry**; cloud/water preprocessing
   must be re-run against them (future work), and companion masks were deliberately NOT shifted.

Register `PreprocessStep(key="correct_georef", reads=("_sst","_valid","_georef_"),
writes=("_georef_corrected","_georef_applied"), fn=_step_correct_georef,
depends_on=("flag_georef",), option_keys=frozenset({"sensors","fields","fill"}),
provenance_inputs=("ecostress","landcover"))`.

### 4. Wire into `preprocess.py` + provenance

- Import both `_step_*` from `.georef` near [preprocess.py:70](../src/coastal_sst_data/processes/preprocess.py#L70);
  add both `PreprocessStep`s to `STEPS`. `_check_steps()` guards keys/deps/cycles at import.
- **`provenance.py` `field_inputs`**: add a georef branch **before** the `base in ("sst",...)`
  check (else `eco_sst_v002_georef_corrected` matches `base=="sst"` and returns `[sensor]` early):
  `if "georef" in rest: return [sensor, "landcover"]` — covers both the flag channels
  (`georef_dy`, `georef_flag`, …) and the corrected channels (`sst_v002_georef_corrected`,
  `valid_v002_georef_corrected`, `georef_applied`).

### 5. Config surface

The open `preprocess.steps.<key>` bag + `option_keys` needs no new validation code. Example:
```yaml
preprocess:
  enabled: true
  steps:
    flag_georef:    { sensors: [eco], tol_m: 200, max_shift_m: 10000, min_coast_obs: 500,
                      min_edges: 300 }
    correct_georef: { sensors: [eco] }         # SST + validity, displaced scenes only
regions:
  - name: puget_sound
    preprocess_steps:                          # NEW region field (step 1)
      flag_georef: { min_coast_obs: 1500 }     # dense archipelago coastline
    areas: [ ... ]
```

## Verification

Port the prototype self-tests (`_sign_test`, `self_test` shift recovery, gate ablation) into
`tests/test_preprocess.py`, and:

1. **Algorithm unit tests** (synthetic, no cube): sign convention; shift recovery scored against
   the scene's own baseline for (0,±5),(±5,0),(+7,−4),(+20,+12); gate ablation (disabling gates
   makes the known-spurious scenes reappear as multi-km fits).
2. **Correction round-trip**: apply a known displacement, confirm `flag_georef` → `displaced` and
   `correct_georef` restores it; assert the vacated margin is NaN (not wrapped) and a
   non-`displaced` scene is copied byte-for-byte.
3. **Step invariants**: emitted 1-D channel names/dtypes + flag encoding (`flag_georef`);
   `*_georef_corrected` dims/dtype (`correct_georef`); **raw cube left byte-unchanged** (existing
   test at test_preprocess.py:562-581 must still pass); derived channels only in the derived cube.
4. **Region-override test**: two AoIs, a region `preprocess_steps.flag_georef.min_coast_obs`
   override changes the gate for one AoI only; an over-`option_keys` / non-`region_option_keys` key
   fails `_check_step_options` loudly.
5. **Provenance**: `coastal-sst-data provenance --fields` shows `[ecostress, landcover]` (no blank)
   for every `*_georef*` channel.
6. **Golden**: the `preprocessed_golden.json` fixture is deliberately left WITHOUT the georef
   steps -- the whole-cell search output on tiny synthetic scenes is numerically fragile across
   scipy versions and would make the golden flaky. Georef is covered by dedicated tests in
   `tests/test_georef.py` (algorithm unit tests + integration through `preprocess_aoi`) instead.
7. **Regression against measured scenes** (design §Verification table) where downloads are
   available: tillamook 07-15 → 5,126 m displaced; north_sound 06-20 → 762 m displaced;
   07-09/07-01/06-05 → skipped by the gates.
8. End-to-end: `coastal-sst-data preprocess --config <cfg> --aoi <one> --overwrite`, then
   `pytest tests/test_preprocess.py`.

## Follow-on: re-filtering the corrected geometry (implemented)

`correct_georef` shifts the **raw** (unfiltered) SST, so `<pre>_sst<ver>_georef_corrected` still
contains clouds — and the pre-fit cloud-filter pass compared misregistered pixels against
MUR/met/landcover. The correct clean product comes from **re-running the same filters on the
corrected geometry**, not from shifting the stale drops. Implemented as three corrected-pass step
variants that reuse the existing filter math:

- **`cloud_filter.py`** — the drop math was already channel-agnostic; a channel-binding seam
  (`_ChannelSet` / `_channel_sets(ctx, pre, mode)` / `_working_or` / `_emitted_channels`) and a
  rewritten `_fold_drop` let each `_step_*(ctx, *, key, base_key, mode)` run against either the raw
  `<pre>_sst<ver>` channels (`mode="raw"`, unchanged) or the emitted
  `<pre>_sst<ver>_georef_corrected` channels (`mode="corrected"`), writing a **separate**
  `<pre>_sst<ver>_georef_corrected_clean` product seeded from the corrected source.
- **`preprocess.py`** — `filter_clouds_corrected` / `filter_cloud_cover_corrected` /
  `filter_land_clouds_corrected`, registered via `functools.partial(..., base_key="filter_*",
  mode="corrected")`, `depends_on` `correct_georef` (+ each other, mirroring the raw order). Each
  **inherits** its base filter's config (`_resolve_opts` layers the step's own bag over the base
  key's), so `filter_clouds_corrected: {}` reuses `filter_clouds`. Shared `option_keys` constants
  (`_CLOUDS_OPTS` / `_CLOUD_COVER_OPTS` / `_LAND_OPTS`) keep the raw and corrected surfaces in lockstep.
- **Provenance** — the `_clean` channels contain `"georef"`, so the existing branch maps them to
  `[sensor, "landcover"]`; the baselines ride the step `provenance_inputs`.
- **`use_cloud_raster` caveat** — set `correct_georef: { fields: [sst, valid, cloud] }` so the native
  cloud band is shifted too; the corrected filter reads `<pre>_cloud<ver>_georef_corrected`.
- Tested in `tests/test_georef.py` (re-filter + separate-channel, config inheritance + override,
  multi-filter composition on the shared `_clean` channel, end-to-end write path).
