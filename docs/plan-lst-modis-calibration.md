# Landsat → MODIS SST calibration as a preprocess step

## Context

> **Updated for the MODIS split.** MODIS is now two products: `modis` (a standalone
> thermal sensor, stacked per platform) and `modis_ref` (the Landsat-coincident
> reference this plan uses). The channels below are therefore `modisref_*`, and the
> step's `reference:` option names `modis_ref`.

Issue #61. The cube already carries everything a Landsat-to-MODIS calibration needs —
`lst_sst`, `modisref_sst`, and `modisref_footprint_id` (added in #52 specifically so a downstream
calibration could group the fine-grid cells covered by one ~1 km MODIS observation). The
calibration itself has never been written; `processes/modis_ref.py` says so explicitly
("Calibration itself is out of scope here").

MODIS is the well-calibrated coarse reference; Landsat is the high-resolution field whose
absolute scale drifts. The two prototype scripts supplied (a footprint-aggregation stage and a
common-slope / per-date-intercept regression) are the validated method. This ports them into
the package as **one post-assembly preprocess step**, which:

1. aggregates the **cloud-filtered** Landsat SST to MODIS footprints, per date;
2. fits `lst_median = m·modis + c_t` — one shared slope, one intercept per date — with
   iterative sigma clipping;
3. applies `lst_cal = (lst − median(c_t)) / m` to the cloud-filtered Landsat field;
4. writes per-date diagnostics into the cube and a QA table + figure beside it.

Branch `61-...` currently has **zero commits** and is one commit behind `main` (v0.2.5) — merge
`main` first.

## Decisions locked with the user

| | |
|---|---|
| Diagnostics | QA dir beside the cube: `<output_dir>/<datacube.output_subdir>/qa/<aoi>_lst_calibration.{csv,png}` |
| Intercept | `median` by default; `intercept: per_date` option, falling back to the median for dates without a fit |
| Source / target | Fit and apply on `lst_sst_clean`; emit `lst_sst_clean_calibrated`. Raw `lst_sst` untouched |
| Fit orientation | `lst ~ modis` (x = `modisref_sst`, y = Landsat footprint median), inverted to apply |
| Table format | **CSV** — `pyarrow` is not a dependency and adding one for a diagnostic sidecar inverts the package's rule that heavy backends live behind extras |
| Step shape | **One step**, `calibrate_sst`. A `apply: false` option gives diagnose-without-applying without a second registry entry |
| Scope | Generic: `sensors: [lst]`, `reference: modis_ref` — ECOSTRESS can be calibrated later with no code change |

## Departures from the prototype scripts (read this before porting)

These are not stylistic — each is a place the script would misbehave against *this* cube.

1. **`flox` is not a dependency** (nor is `dask`). `PreprocessContext.read()` hands back plain
   in-memory numpy, so all grouped statistics are done with `np.unique(return_inverse=True)` +
   `np.bincount` + one `lexsort` per date. See *Grouped statistics* below.
2. **`pyarrow` is not a dependency** — the two-stage parquet handoff collapses into one
   in-memory pass; the CSV is a diagnostic sidecar, not an intermediate.
3. **`cloud_var="lst_clouds"` does not exist.** The raw channel is `lst_cloud`, but it is not
   needed at all: `lst_sst_clean` already has cloud pixels NaN'd by the `filter_clouds` family.
   The whole `_clear_mask` / `cloud_flag` machinery drops out — clear = finite.
4. **`lst_sst_clean_cloudfiltered` is a `uint8` DROP FLAG (1 = dropped), not a stricter LST
   field.** Counting its non-NaN values, as the script does, would count every pixel. Replace
   the `n_filter_clear` / `filter_pass_frac` columns with `n_raw` (cells where the *raw*
   `lst_sst` is finite), which is what actually separates "cloud removed it" from "Landsat
   never covered it".
5. **`lst_sst_clean` only exists if the cloud filters are configured for Landsat.** The filters
   are generic over prefix but `examples/config.test.yaml` lists `sensors: [eco]`. The step
   resolves its source as `lst_sst_clean` (emitted *this run*) → raw `lst_sst`, and records
   which in `calibration_source`. The config block below turns the filter on for `lst`.
6. **`modisref_footprint_id` restarts at 0 in every scene.** Group within a timestep, never across
   the time axis — already what the script does, but it must stay that way and gets its own test.
7. **`sigma == 0` is an exact fit, not a failure.** The script aborts; that makes a perfect
   synthetic fixture untestable. Only a *non-finite* sigma aborts.
8. **Cast every scalar to `float`/`int`/`bool`/`str` before it enters a zarr attribute.** A
   `np.float32` is not JSON-serializable and blows up in the encoder *after* the whole stage has run.

## Files

### New: `src/coastal_sst_data/processes/calibration.py`

Structured like `processes/georef.py` — design-rationale docstring, pure testable functions,
then one `_step_calibrate_sst(ctx)` entry point. Imports **numpy, pandas, `.channels`** only;
`PreprocessContext` under `TYPE_CHECKING` (no cycle — the `cloud_filter.py`/`georef.py` pattern).

**Grouped statistics** (one timestep at a time; ids are only meaningful within a timestep):

- `codes, inv = np.unique(fid[t].ravel()[fid >= 0], return_inverse=True)` — never `bincount` on
  raw int32 ids, which would allocate `max(id)+1`.
- `group_stats(inv, y, G)` → count / mean / std. Two-pass variance
  (`bincount(inv, weights=(y - mean[inv])**2)`); a one-pass `E[x²]−E[x]²` on ~285 K absolute
  temperatures loses ~5 significant digits.
- `group_quantiles(inv, y, G, qs=(.25,.5,.75))` → `(len(qs), G)`. `np.lexsort((yv, gi))`, then
  `start = cumsum(counts)` and an explicit `yv_s[start + offset]` gather with linear
  interpolation, matching `np.percentile(method="linear")`. **Do not use `np.add.reduceat`** —
  a zero-length group silently reads the next group's first element, and empty groups (a fully
  clouded footprint) are the normal case here.
- `group_first_finite(inv, x, G)` → `(first, ptp)`. Every cell of a footprint carries the *same*
  `modisref_sst`; taking the mean would paper over a violated assumption, so take the first finite
  value and **report the peak-to-peak spread** so the violation is visible in the QA table.

**Stage 1 — `footprint_matchups(...) -> pd.DataFrame`.** Per date: `n_total` (geometric cell
count — the `clear_frac` denominator), `n_raw`, `n_clear`, `clear_frac`, mean/median/std/p25/p75
of the working LST, `modisref_sst`, `modis_ptp`. Invalid where
`n_clear < min_count | clear_frac < min_clear_frac | ~isfinite(modis_sst)`; **rows are kept**
with LST stats NaN'd and a `reject_reason`, because the table's job is to show what was rejected
and why. Key is `(t_index, footprint_id)`.

**Stage 2 — `fit_common_slope(x, y, code, G, ...) -> dict`.** The within (fixed-effects)
estimator, iterated:

1. drop groups with `cnt < min_per_group`; abort `too_few_after_clip` if survivors
   `< max(min_n, min_frac·n_start)`;
2. group means **from survivors only**;
3. `dx = x − gx[code]`, `dy = y − gy[code]`, `m = (dx·dy)/(dx·dx)`; abort `no_x_variance` if
   `dx·dx <= 0`;
4. `c = gy − m·gx`;
5. residuals from the **full model over all rows** — `r = y − (m·x + c[code])` — so the keep-set
   is recomputed from scratch each iteration and the answer does not depend on the iteration path;
6. `sigma = 1.4826·MAD(r[keep])` (or `sd`); clip to `[−nsd_low·sigma, +nsd·sigma]`;
7. stop when the keep-set settles or `|m − m_prev| < tol`.

`nsd_low` is separate because cloud contamination biases thermal-IR SST **cold** — the residual
distribution is asymmetric. Both default 3.0. Abort `degenerate_slope` if `m` falls outside
`(0.05, 20.0)`: the applied form divides by `m`, so a near-zero slope is unusable, not merely bad.

Returns `slope, intercepts, intercept_median, keep, residual, n_start/n_used/n_clipped/n_iter/
n_groups, converged, reason, rmse, r, sigma` plus per-group `n_used/mean_x/mean_y/sigma`.

**`fit_free_slopes(...)`** — independent slope+intercept per date, same clip loop, as a
diagnostic. **`common_slope_ok(slope, free, min_dates=5)`** → `None` when fewer than 5 dates got
a free slope (with one or two dates the free and common slopes are the same number). A shared
slope outside the central 90 % of free slopes is a **WARNING naming the QA figure, never an
abort** — the warning is the deliverable.

**Stage 3 — apply.** `(field − c_t[:, None, None]) / m`, where `c_t` is `full(T, c_median)` for
`intercept: median` or `where(isfinite(c), c, c_median)` for `per_date`. The fallback is
unconditional, so a date the fit dropped can never yield an all-NaN calibrated scene.

### Edited: `src/coastal_sst_data/processes/channels.py`

```python
CALIBRATED = "_calibrated"   # calibrate_sst: the working SST on the reference sensor's scale
DERIVED_SUFFIXES = (..., CLEAN, CORRECTED, CALIBRATED, ...)
```

Load-bearing, not bookkeeping: `is_derived` matches on `endswith`, and
`lst_sst_clean_calibrated` ends with `_calibrated`, not `_clean`. Without the entry,
`ctx.base_channels("lst_sst")` on the second run hands this run's own output back as an input
and the cube grows `lst_sst_clean_calibrated_clean_calibrated`. The 1-D `lst_cal_*` diagnostics
need no entry (no scan uses a prefix that reaches them).

### Edited: `src/coastal_sst_data/processes/preprocess.py`

Registry entry, declared **last** in `STEPS` (calibration is the terminal transformation of the
LST product, and declaration order is the topo tie-break):

```python
PreprocessStep(
    key="calibrate_sst",
    reads=("lst_sst", "modisref_sst", "modisref_footprint_id", "_clean"),
    writes=("_cal_", "_calibrated"),
    fn=_step_calibrate_sst,
    # the fit must read the CLOUD-FILTERED LST -- a cloudy pixel inside a footprint drags its
    # median cold and biases the intercept. A depends_on to an UNSELECTED step is ignored, so
    # the step still runs (on raw lst_sst) if no filter is selected.
    depends_on=("filter_clouds", "filter_cloud_cover", "filter_land_clouds", "correct_georef",
                "filter_clouds_corrected", "filter_cloud_cover_corrected",
                "filter_land_clouds_corrected"),
    option_keys=_CAL_OPTS,
    region_option_keys=frozenset({"min_count", "min_clear_frac", "min_per_group",
                                  "min_n", "min_frac"}),
    provenance_inputs=("landsat", "modis"),
)
```

Only coverage/count gates are region-overridable — the rule `flag_georef` already applies: a
region tunes how much data a fit needs, it cannot redefine the estimator.

**QA plumbing** (the one genuinely new piece of machinery):

- `_build_eff` gains `"qa_dir": root / project.datacube.output_subdir / "qa"`.
- `PreprocessContext` gains `artifacts: list = field(default_factory=list)` and
  `artifact(stem, kind, payload)`. A step **queues**, it never touches the filesystem: the cube
  is rewritten inside `store.atomic`, and a sidecar written mid-run would survive a failure that
  rolled the cube back. `kind` is `"csv"` (payload: a DataFrame) or `"figure"` (payload: a
  zero-argument builder returning a Figure, so matplotlib is imported only at flush time).
- `preprocess_aoi(ds_cube, g, eff, *, artifacts: list | None = None)` — keyword-only and
  optional so every existing call site and test keeps working, and so the function stays
  side-effect free.
- New `_write_artifacts(qa_dir, aoi, artifacts)`, called in `run()` **after** the `store.atomic`
  block returns and the log line prints — i.e. only once the swap succeeded. Each file goes
  through `store.atomic` itself. **Never raises**: these are diagnostics and the cube is already
  committed, so an unwritable dir or an absent matplotlib costs a WARNING, not the run.

Behaviour: dry run returns before the atomic block (nothing computed, nothing written); a failed
run never reaches the flush (no orphan QA files); a skipped (not-stale) run leaves the previous
QA files, which still describe the unchanged cube exactly.

### Edited: `src/coastal_sst_data/plot.py`

`plot_lst_calibration(fit, table, *, aoi, sensor, reference, out_path, dpi=150) -> Path`.
It lives here, not in `calibration.py`, so matplotlib never enters the import graph of a module
that runs on every preprocess. `plot.py` imports only `config`, so `preprocess → plot` is a new
edge in a direction nothing travels back along. Lazy `matplotlib.use("Agg")`, mirroring
`plot_project_aois`.

2×3 panels:

1. **Matchup scatter + fit** — survivors coloured by date, clipped points grey ×, the
   common-slope line at the median intercept, and the 1:1 line. The gap between the two lines
   is the bias being corrected.
2. **Per-date intercept series** — `c_t` vs date, marker area ∝ `n_used`, median line + IQR
   band. This is the panel that adjudicates `median` vs `per_date`: a flat cloud says median, a
   seasonal drift says `per_date`.
3. **Free per-date slopes vs the common slope** — histogram, central-90 % band shaded, vertical
   line at `m`. Makes the warned condition visible without reading the log.
4. **Residuals vs fitted** — with the `+nsd·σ` / `−nsd_low·σ` bands and an inset carrying
   `n_used/n_start`, `n_clipped`, `n_iter`, `rmse`, `r`.
5. **Before / after** — overlaid histograms of `lst_median − modis_sst` and
   `(lst_median − c)/m − modis_sst`, each annotated with its mean and sd. Demonstrates the
   calibration did what it claims.
6. **Matchups per date** — grouped bars of `n_matchups` / `n_used` with the `min_per_group` line.

On an aborted fit, still draw panels 1 and 6 under a red `FIT ABORTED: {reason}` suptitle — the
failure is exactly the case a user needs to see, and a missing PNG communicates nothing.

### Edited: `src/coastal_sst_data/provenance.py`

In the `_SENSOR_RE` branch, **before** the base-token test (same placement and reason as the
existing `georef` special case, which would otherwise read `lst_sst_clean_calibrated` as a plain
`sst` channel and attribute it to Landsat alone):

```python
if rest.startswith("cal_") or rest.endswith("_calibrated"):
    return [sensor, "modis"]
```

`"modis"` is hard-coded with a comment: `reference` is a config option the *name* cannot carry,
and MODIS is the only sensor shipping a `_footprint_id` channel.

## Emitted channels

1-D `("time",)`, named `<pre>_cal_<stat><ver>` (version arithmetic mirrors
`cloud_filter._channel_sets`, so Landsat gets `lst_cal_intercept` and a stacked sensor would get
`eco_cal_intercept_v002`):

| name | dtype | notes |
|---|---|---|
| `lst_cal_intercept` | float32 | `c_t`; NaN where the date had no fit |
| `lst_cal_intercept_applied` | float32 | what was actually used (constant in `median` mode) — makes the apply auditable |
| `lst_cal_n_matchups` / `lst_cal_n_used` | int32 | valid matchups; survivors after clipping |
| `lst_cal_sigma` | float32 | robust residual scale within this date |
| `lst_cal_mean_modis` / `lst_cal_mean_lst` | float32 | the group means the intercept is built from |
| `lst_cal_free_slope` | float32 | this date's independent slope (diagnostic only) |
| `lst_cal_flag` | uint8 | `ok / too_few_matchups / clipped_out / no_matchups / overpass_gap / no_fit` |

3-D `("time","y","x")`: **`lst_sst_clean_calibrated`** (float32, K) with
`calibration_source`, `_reference`, `_slope`, `_intercept_k`, `_intercept_mode`, `_n_used`,
`_rmse_k`, `_r`, `_converged`, `_reason` attrs. The name is `f"{source_channel}{CALIBRATED}"`,
so the no-filter fallback yields the self-describing `lst_sst_calibrated`.

AoI scalars go to `ctx.global_attrs["lst_calibration"]` as a JSON blob (slope, intercept_median,
n_*, converged, reason, rmse, r, sigma, free-slope percentiles, `common_slope_ok`).

## Config surface

```yaml
preprocess:
  enabled: true
  steps:
    # Screen Landsat FIRST -- the calibration fits on the clean product.
    filter_clouds: { method: offset, baseline: mur_sst, threshold_k: 5.0, sensors: [lst] }

    calibrate_sst:
      sensors: [lst]           # sensors to calibrate (channel prefixes)
      reference: modis_ref         # target scale; needs <ref>_sst and <ref>_footprint_id
      source: auto             # auto | clean | raw -- which field to fit and apply on
      apply: true              # false = fit and write diagnostics only, move no pixels
      intercept: median        # median (default) | per_date (falls back to median per date)
      # -- Stage 1: what counts as a matchup --------------------------------------------
      min_count: 20            # clear LST pixels the footprint needs
      min_clear_frac: 0.5      # clear pixels / cells in the footprint
      max_dt_hours: null       # null = off; else drop dates where |lst_hour - modis_hour| exceeds it
      # -- Stage 2: the fit --------------------------------------------------------------
      min_per_group: 5         # matchups a date needs to keep its own intercept
      nsd: 3.0                 # upper sigma clip
      nsd_low: 3.0             # lower sigma clip (cloud biases LST cold -- tighten here)
      max_iter: 10
      tol: 1.0e-4
      min_n: 30                # abort if survivors < max(min_n, min_frac * n_start)
      min_frac: 0.5
      scale: mad               # mad (1.4826*MAD) | sd
      free_slope_check: true   # fit free per-date slopes and warn if one slope is indefensible
      # -- QA sidecars, written beside the cube after it is committed ---------------------
      qa_table: true           # <output_dir>/datacube/qa/<aoi>_lst_calibration.csv
      qa_figure: true          # ...png (needs the `plot` extra; absent -> warning, not a crash)
```

`max_dt_hours` exists because it is the scientific hole a reviewer will find: Landsat overpasses
~10:30 local, MODIS-Aqua ~13:30, so a diurnal offset is folded into the intercept. That is *why*
the intercept is per-date and why `median` is the default; the gate is off by default and
documented as a known limit.

## QA table — `<output_dir>/datacube/qa/<aoi>_lst_calibration.csv`

```
aoi, date, t_index, footprint_id,
n_total, n_raw, n_clear, clear_frac,
lst_mean, lst_median, lst_std, lst_p25, lst_p75,
modis_sst, modis_ptp,
valid, reject_reason, residual, used,
date_intercept, date_n_used, date_sigma, date_free_slope, date_flag,
slope, intercept_median, reason
```

Every `(t_index, footprint_id)` seen, including rejected (`valid=0` + `reject_reason`) and
clipped (`used=0`) rows. Per-date and AoI columns are denormalized down every row: CSV has no
metadata slot, a `#`-comment header would break bare `pd.read_csv`, and
`df.groupby("date").first()` reconstructs the per-date table in one line.
`to_csv(index=False, float_format="%.6g")` — sub-mK precision, a few MB for a multi-year record.

## Degradation (a step must never crash the stage)

| condition | behaviour |
|---|---|
| no `modisref_footprint_id` | WARNING naming `footprint_id: true`; emit nothing |
| no `modisref_sst` | WARNING; emit nothing |
| no `lst_sst` (Landsat not acquired) | INFO, `continue` — mirrors `sensor_hours() is None` |
| no cloud filter ran | fit + apply on raw `lst_sst` → `lst_sst_calibrated`; INFO. Not an error |
| stale `lst_sst_clean` on disk, filter deselected | never used — `_clean` resolves only through this run's `ctx.channels` |
| a date with no footprints / no clear pixels | `flag=no_matchups`, intercept NaN, median fallback |
| a date below `min_per_group` | rows kept in the CSV, group dropped from the fit, `flag=too_few_matchups`, median fallback |
| survivors `< max(min_n, min_frac·n_start)` | abort `too_few_matchups`; emit 1-D diagnostics with `flag=no_fit`; **no `_calibrated` channel**; WARNING |
| no x-variance | abort `no_x_variance`; same degradation |
| `sigma == 0` (exact fit) | **converged, `reason="ok"`, calibration applies** |
| slope outside `(0.05, 20.0)` | abort `degenerate_slope`; no `_calibrated`; WARNING quoting the slope |
| common slope outside the free-slope 90 % band | WARNING pointing at the figure; `common_slope_ok=false`; **still applies** |
| `modis_ptp > 0` on >1 % of footprints | one aggregated WARNING: footprint ids may not be aligned with the `modisref_sst` beside them |
| matplotlib absent / QA dir unwritable | WARNING, CSV still written, run still reports `wrote()` |

## Tests — `tests/test_calibration.py`

Reuse `tests/test_preprocess.py`'s `_project`, `_setup`, `_hand_cube`, `_pp`, `_ones_valid` and
`tests/test_datacube.py`'s `AOI`, `write_landsat`, `write_modis(..., footprint=, block=,
temp_step=)`, `write_mur`, `write_landcover` — the established import chain. Add a local
`_matchup_cube(g, times, *, slope, offset, block=3, noise=0.0)` that lays out block-constant
footprint ids and `lst_sst = slope·modis + offset`, so the true coefficients are recoverable.

Primitives: `group_quantiles` matches a per-group `np.percentile` loop (incl. groups of size 1
and 2); `group_stats` returns `n_clear=0` / NaN for an all-NaN group; `group_first_finite` takes
the first, not the mean, and reports `ptp`. **`test_grouping_never_crosses_timesteps`** — two
dates whose ids both restart at 0 with different MODIS values.

Fit: recovers a known affine (rtol 1e-3); **the recovered slope is unchanged when the per-date
offsets are multiplied by 100** (this is what separates the fixed-effects estimator from pooled
OLS, which fails it badly); a planted 15 K-cold footprint is clipped and the slope matches the
clean-data slope; a date below `min_per_group` is dropped and falls back; `temp_step=0.0` gives
`reason="no_x_variance"` and no `_calibrated`; zero-noise data is `converged`/`ok`, not aborted;
a poor common slope warns *and still applies*.

Apply: `lst_sst_clean_calibrated == (lst_sst_clean − c_med)/m` exactly and `≈ modis_sst` at
footprint centres; `per_date` beats `median` on a drifting offset; `apply: false` emits
diagnostics and no 3-D channel; **`lst_sst` is byte-identical**.

Stage invariants (§5b requires both): assembled channels survive the rewrite;
**`test_calibration_is_idempotent`** — feed run 1's output into run 2 and assert every channel
identical (this is the test that catches a missing `_calibrated` in `DERIVED_SUFFIXES`);
`is_derived("lst_sst_clean_calibrated")` is True and `base_channels("lst_sst")` on a post-run
cube returns exactly `["lst_sst"]`; `field_inputs` names both sensors (also add to
`tests/test_provenance.py`); a region override of `nsd` is rejected.

QA: `preprocess_aoi(..., artifacts=arts)` queues one `("lst_calibration", "csv", df)` with the
exact column list, one row per `(t_index, footprint_id)` including rejects; `preprocess_aoi`
leaves `qa_dir` **absent**; the full `run()` path writes both files under `datacube/qa/`
(PNG under `pytest.importorskip("matplotlib")`); a dry run writes nothing; a figure builder
raising `ImportError` still writes the CSV and still reports `wrote()`.

`tests/golden/preprocessed_golden.json` needs **no regeneration** — the default config does not
select the new step. Assert that explicitly in the PR.

## Docs

- **`docs/plan-lst-modis-calibration.md`** (new) — in `docs/plan-georef-preprocess-step.md`'s
  style. Known limits section must carry: the diurnal offset absorbed into the intercept; that
  one slope per AoI is an assumption `free_slope_check` *tests* but cannot repair; that
  `modis_ptp` is how you check the constant-within-footprint assumption; and that the fit
  inherits the cloud filter's biases.
- **`docs/DEVELOPMENT.md` §5b** — the step in the worked YAML block, plus a short paragraph on
  the **QA-artifact protocol**, which is the one new thing a future step author must know.
- **`README.md`** preprocess section — a bullet, the config lines, and a note that
  `calibrate_sst` needs `modisref_sst` + `modisref_footprint_id` (acquire MODIS with
  `footprint_id: true`) and a Landsat SST channel.
- **`examples/config.test.yaml`** — the step commented out, as the corrected-pass filters are.
- `preprocess.py` module docstring "Steps shipped today" list; `config.PreprocessSpec` docstring.
- `pyproject.toml` → `0.2.6` as the final commit, per the repo's convention.

## Verification

```bash
# 1. Merge main (branch 61 is one commit behind) and install.
git merge main && pip install -e '.[all,dev]'

# 2. Unit + invariant tests.
pytest tests/test_calibration.py -v
pytest tests/test_preprocess.py tests/test_provenance.py tests/test_georef.py   # no regressions
pytest                                                                          # full suite

# 3. End-to-end on a real AoI with Landsat + MODIS acquired (footprint_id: true).
coastal-sst-data preprocess --config config.yaml --aoi <aoi> --overwrite -v
```

Then confirm, in order:

- the log prints the shared slope, `n_used/n_start`, iterations, and the reason code — and, if
  the common-slope assumption is poor, the free-slope warning naming the figure;
- `<output_dir>/datacube/qa/<aoi>_lst_calibration.csv` and `.png` exist and the PNG's panel 5
  shows the after-histogram centred nearer zero than the before-histogram;
- `xr.open_zarr(cube)` carries `lst_sst_clean_calibrated`, the `lst_cal_*` 1-D channels, and a
  `lst_calibration` global attr; `lst_sst` is unchanged;
- re-running without `--overwrite` is a **no-op** (`preprocess_steps_stale` is False), and
  re-running *with* `--overwrite` produces bit-identical `lst_cal_*` values — the idempotence
  property, checked on real data as well as in the fixture;
- selecting the step with `apply: false` writes the diagnostics and QA files but no
  `_calibrated` channel.
