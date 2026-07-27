# Design: `flag_georef` / `correct_georef` — ECOSTRESS georeferencing errors as preprocess steps

Port the validated prototype in `/Users/johnbuckner/Documents/test_coastal_sst_data/src/georef_diagnose.py`
(+ `plot_thermal_edges.py`) into the package as post-assembly preprocess steps, following
[DEVELOPMENT.md](DEVELOPMENT.md) §5b.

> **This is the design.** The ordered build plan — and corrections to a few claims below that do
> not hold against the current codebase — live in
> [plan-georef-implementation.md](plan-georef-implementation.md). Key changes there:
> correction is implemented **now** (not deferred); per-region step options require a small config
> extension (they do **not** ride `config.resolve_opts`, which is product-only); and the six-way
> classifier + stability sweep in §5 are new code (the prototype has only the search and gates).

## Context

Some ECOSTRESS granules are geolocated wrong. A misregistered scene is worse than a missing one:
it contributes wrong SST at every coastal pixel and nothing downstream flags it.

The prototype established that a thermal scene's coastline can be registered against the static
land-cover coastline by brute-force whole-cell translation, and — critically — that **most
apparent multi-kilometre errors are artifacts of fitting scenes that barely observe the coast**.
Of 22 ECOSTRESS scenes across two AoIs, one genuine large displacement was found (Tillamook
2026-07-15, **5,126 m**), several real sub-kilometre ones (500–762 m), and three spurious
multi-km "corrections" that quality gates reject outright.

**Scope (updated):** `flag_georef` flags and moves no pixels; a companion `correct_georef` step —
implemented in the same work, see [plan-georef-implementation.md](plan-georef-implementation.md) —
applies the fitted `(dy,dx)` to the **raw** ECOSTRESS geometry. (The original plan deferred
correction; it is now in scope.)

## Method (as validated — port faithfully, these details are load-bearing)

### 1. Reference coastline
Static ESA WorldCover water mask (`landcover_water`) → water cells touching land, one cell wide,
on the water side (8-connected dilation of `~water`, intersected with `water`). Reject a
degenerate all-water/all-land mask rather than using it — that channel was broken until recently
and a single-class mask yields no coastline at all.

**Not** the tide-adjusted waterline. Measured: against `<sensor>_water_class`, 2026-06-14 scores
8% agreement; against the static coastline, 42%. The thermal edge sits at the permanent land
boundary because exposed intertidal reads thermally like land.

### 2. Thermal edges — Canny, hysteresis 80/97
NaN-aware Gaussian (normalised convolution, σ=1.5) → Sobel → non-maximum suppression → hysteresis
at the 80th/97th percentiles of ridge magnitude. Gradients whose 3×3 neighbourhood is not fully
observed are discarded (a gradient across the rim of a data hole is an artifact of the hole).

Two things matter:
- **NMS is required.** The result must be one cell wide to be scored fairly against a one-cell
  reference. A thick gradient ridge scores high agreement for free.
- **Sensitivity is tuned by the LOW threshold only.** Lowering both (85/95) inflates edge counts
  ~65% with land texture, diluting agreement 5–20 points. Lowering only the low threshold
  (90/97 → 80/97) extends the same strong seeds along weak coastal continuations: on-coast edge
  counts rise (north_sound 06-11: 749 → 804) for a 1-point agreement cost.

### 3. Quality gates — run BEFORE any fitting
This is where most of the discriminating power turned out to live.

```
coast_obs   < min_coast_obs   -> skip, do not fit   (coastline cells under valid data)
valid_frac  < min_valid_frac  -> skip, do not fit
n_edge      < min_edges       -> skip, do not fit
```

`coast_obs` is the single strongest predictor of a spurious fit. Use the **absolute count**, not a
fraction — north_sound's coastline is 20,925 cells and Tillamook's is 2,180, so the same fraction
means an order of magnitude different evidence.

**`min_coast_obs` must be region-overridable** for exactly that reason. Default 500; expect to tune
per AoI. **Correction to the original plan:** this does *not* ride `config.resolve_opts` — that
resolver is product-only, and preprocess step bags are flattened project-global with no region
merge. Per-region step options are added as a small config extension (a `regions[].preprocess_steps`
bag + a `resolve_step_opts` merge); see step 1 of
[plan-georef-implementation.md](plan-georef-implementation.md).

**Do NOT gate on land–sea ΔT.** north_sound 06-20 has ΔT = 0.47 K and is a confirmed real 762 m
correction with 2,040 edges. Contrast predicts how *visible* the coastline is, not whether a
usable edge set exists. Emit ΔT as a diagnostic instead.

### 4. Shift search
Edge cells as index arrays `(ys, xs)`; a candidate `(dy,dx)` is scored by gathering the
precomputed `distance_transform_edt(~ref)` at the shifted positions — no image is resampled, so
the whole surface costs one gather per candidate. Coarse stride 4 over ±`max_shift_m`, then ±4
refinement around each of the best 8 coarse peaks.

**Two rules that are NOT optional** — both were bugs in the first prototype and both silently
produced wrong answers:

- **Off-grid cells count as *not agreeing*, with the denominator kept at ALL edges.** Rejecting
  the whole candidate when any single cell leaves the grid makes entire *directions*
  unsearchable, because scene edges normally reach the margin. Tillamook 07-15 needed ~5 km east
  and not one eastward candidate was ever scored; the search confidently returned 5 km west.
  Keeping the full denominator also stops a huge shift from scoring well by retaining a lucky few.
- **Refine around the best 8 coarse peaks, not just the winner.** A stride-4 grid cannot land on
  an optimum like (+3,−1), and refining only around the coarse winner leaves it unreachable.
  north_sound 06-23 reported a phantom 4,300 m this way; the true answer is 316 m.

### 5. Statistics and classification
**New code:** the prototype computes the search surface and its statistics (`search_shift`) but
does *not* classify — the six-way labelling and the stability sweep below are written fresh from
this spec, then validated against the regression table in Verification.

From the search surface: `P0` (agreement at zero shift), `P_peak`, `(dy,dx)`, surface median and
sd, `z = (P_peak − median)/sd`, `gain = P_peak − P0`.

```
NOT stable across a ±5/10/20/30 km sweep           -> unstable      (never correct)
P_peak      < lift_min (2.5) * surface_median      -> unrecoverable
|shift|     <= ok_shift_m (300 m) and z >= z_min   -> ok
z >= z_min (4.5) and gain >= gain_min (0.10)
                            and stable             -> displaced     (correctable candidate)
otherwise                                          -> suspect
```

`stable` = the best `(dy,dx)` moves ≤1–2 cells across the window sweep. Real corrections are
invariant (n_edge 845–2040); the runaways drift 7.5–34.8 km (n_edge 34–143).

Three things the data explicitly rules out:
- **No penalty on shift magnitude.** It would reject Tillamook 07-15, the one confirmed large
  displacement. Magnitude is not evidence of error; low edge count and drift are.
- **`z` alone is insufficient.** The spurious 07-01 and 07-09 reach z = 6.5 and 7.3 — *higher*
  than the real 5,126 m correction (z = 6.1).
- **Absolute agreement thresholds are wrong.** Chance level is AoI-specific (surface median
  12–18% at north_sound vs 3–7% at Tillamook, tracking coastline density). Score on **lift over
  the surface median**.

## Files

### New: `src/coastal_sst_data/processes/georef.py`
More than glue, so its own module imported into `STEPS` — the same arrangement as
[`cloud_filter.py`](../src/coastal_sst_data/processes/cloud_filter.py). Follow that file's header
conventions exactly: `from __future__ import annotations`, a docstring explaining why it lives
outside `preprocess.py`, and

```python
if TYPE_CHECKING:                    # avoid an import cycle: only a type hint needs it
    from .preprocess import PreprocessContext
```

Port from the prototype, unchanged in substance:

| prototype (`georef_diagnose.py`) | notes |
|---|---|
| `_score` | the off-grid/denominator rule — port verbatim |
| `search_shift` | coarse + top-8 refinement |
| `scene_quality` | `valid_frac`, `coast_obs`, `coast_obs_frac`, `dt_k` |
| `quality_reject` | returns a human-readable reason or `None` |
| `coast_offsets` | signed nearest-coast vectors, for diagnostics only |

Reuse rather than reimplement: `canny` / `nan_gaussian` / non-maximum suppression currently live
in the prototype's `plot_thermal_edges.py`. They have no package home yet — put them in
`georef.py` unless a shared image-ops module is wanted.

Read channels defensively via `ctx.read(...)` / `ctx.channels_with_prefix(...)`; degrade with a
warning when landcover or the sensor is absent. A step must never raise.

### Register in `preprocess.py`

```python
PreprocessStep(
    key="flag_georef",
    reads=("_sst", "landcover_water"),
    writes=("_georef_",),
    fn=_step_flag_georef,
    depends_on=("filter_clouds", "filter_cloud_cover", "filter_land_clouds"),
    option_keys=frozenset({"sensors", "tol_m", "max_shift_m", "coarse_stride", "n_refine",
                           "sigma", "lo_pct", "hi_pct", "min_coast_obs", "min_valid_pct",
                           "min_edges", "z_min", "lift_min", "gain_min", "ok_shift_m",
                           "stability_windows_km", "land_mask_channel"}),
    provenance_inputs=("ecostress", "landcover"),
)
```

`depends_on` the three cloud filters is essential — cloud edges are the dominant noise source, and
the step must read the *filtered* `<sensor>_sst` via the `_working` convention.

### Emitted channels — 1-D `("time",)` per sensor
The pattern `filter_cloud_cover` already uses for `<pre>_scene_cloud_pct_<src>`:

```
<pre>_georef_dy, _dx          int16   whole-cell offset that best aligns this scene
<pre>_georef_shift_m          float32 |shift| in metres
<pre>_georef_z                float32 peak significance against the search surface
<pre>_georef_agree, _agree0   float32 agreement at the peak / at zero shift
<pre>_georef_n_edge           int32
<pre>_georef_coast_obs        int32   coastline cells observed
<pre>_georef_dt_k             float32 land-sea contrast (diagnostic, not a gate)
<pre>_georef_flag             uint8   flag_values/flag_meanings for the six labels
```

Flag values: `0=ok, 1=displaced, 2=suspect, 3=unstable, 4=unrecoverable, 5=insufficient_signal`.
Nothing 3-D; nothing mutated.

### `provenance.py`
Add a branch in `field_inputs`'s `_SENSOR_RE` section: `<sensor>_georef_*` → `[sensor,
"landcover"]` (a derived field names **all** its inputs). Without it the channels ship with blank
provenance and the module logs a warning — which is exactly what it exists to prevent.

### Config surface
No code needed; the open `preprocess.steps.<key>` bag validates against `option_keys`.

```yaml
preprocess:
  steps:
    flag_georef:
      sensors: [eco]
      tol_m: 200
      max_shift_m: 10000
      min_coast_obs: 500        # region-overridable — see below
      min_edges: 300
```

Region override — via the new `regions[].preprocess_steps` bag (NOT `sources`, which is typed
`dict[DataProduct, …]` and would reject a step key):

```yaml
regions:
  - name: puget_sound
    preprocess_steps:
      flag_georef:
        min_coast_obs: 1500     # dense archipelago coastline
```

This bag and its `resolve_step_opts` merge are added in step 1 of
[plan-georef-implementation.md](plan-georef-implementation.md).

## Verification

1. **Sign convention on synthetic coastlines.** A probe 10 cells north of an east–west coast must
   report `north < 0`. Cube rows run north→south, so `east = col_off*res` but `north =
   −row_off*res`. A mirrored axis inverts every interpretation and is otherwise invisible.
2. **Synthetic shift recovery — the decisive test.** Roll a well-registered scene by a known
   `(dy,dx)` and confirm the search returns the baseline minus that roll. Score against the
   scene's **own measured baseline**, not zero — a scene may already be displaced (Tillamook
   07-12 sits at (−5,−5) before anything is rolled), and comparing to zero makes a correct
   estimator look broken. The prototype passes for (0,±5), (±5,0), (+7,−4), (+20,+12).
3. **Regression against the measured scenes.** These must reproduce:

   | scene | expected |
   |---|---|
   | tillamook 07-15 | 5,126 m, z≈6.1, stable, **displaced** |
   | north_sound 06-20 | 762 m, z≈11.2, stable, **displaced** |
   | tillamook 07-12 | ~640–707 m, z≈12, stable, **displaced** |
   | north_sound 06-11 | ≤100 m, z≈10, **ok** |
   | north_sound 07-09 | **skipped** — 112 coast cells |
   | north_sound 07-01 | **skipped** — 305 coast cells |
   | north_sound 06-05 | **skipped** — 198 edges |

4. **Gate ablation.** With the quality gates disabled, 07-09/07-01/06-05 must reappear as
   multi-km "corrections" — that is what the gates are for, and the test documents it.
5. **Package-side**, per §5b: extend `tests/test_preprocess.py` with a synthetic cube (reuse the
   `test_datacube` writers), assert the emitted 1-D channels and dtypes, the flag encoding, and
   the step invariants; add to `tests/golden/preprocessed_golden.json`. The raw cube must be left
   byte-unchanged (there is already a test for that).

## Known limits — document these in the module docstring

- **Whole-cell translation only.** No sub-pixel, no rotation, no scale. Adequate for the observed
  failure mode but not general.
- **Along-shore ambiguity.** A straight coastline constrains the across-shore offset well and the
  along-shore one barely; the search surface shows this as a ridge rather than a peak. Both test
  AoIs have convoluted coastlines, so this is untested on a simple one — check before trusting
  the method at such a site.
- **The nearest-coast displacement vector saturates** and must not be used as an estimator. It
  recovers only 0–20% of a known displacement, degrading to 4–6% at 5 km, because a displaced
  point simply finds a different nearby piece of coastline. It is useful for *visual* diagnostics
  (tight-at-origin vs offset-and-diffuse) but the magnitude must come from the search.
- **Thresholds rest on 22 scenes across 2 AoIs**, with exactly one confirmed large displacement.
  `min_coast_obs = 500` currently sits between a rejected scene (305) and a kept marginal one
  (692). Expect re-tuning; the region override exists because of this.

## Phase 2 — `correct_georef` (now in scope)

`correct_georef`, `depends_on=("flag_georef",)`: apply the stored `(dy,dx)` **only** where
`_flag == displaced`, NaN-fill the vacated margin, emit `<pre>_georef_applied`. Use the prototype's
slice-based `shift_array`, never `np.roll` — wrapping fabricates a coastline on the opposite side
of the scene.

**Refinements decided during the build** (see [plan-georef-implementation.md](plan-georef-implementation.md)):

- **Fit on filtered, apply to raw.** The `(dy,dx)` is estimated from the cloud-filtered SST (clean
  edges) but applied to the **raw** ECOSTRESS fields read from the raw cube.
- **Shift raw ECOSTRESS-native fields only** (SST + validity, and other native bands) into
  `*_georef_corrected` channels — *not* the preprocess-derived masks (cloud-filtered SST,
  water_line, met masks). The corrected geometry invalidates those derived products; re-running the
  cloud/water preprocessing on the corrected images is deliberate follow-up work.
