# Plan: cloud filtering for ECOSTRESS SST (two preprocess steps)

## Context

ECOSTRESS thermal-IR SST is biased **cold** wherever clouds contaminate a pixel. The existing
ECOSTRESS validity mask (`eco_valid_<ver>`) gates on QC bits only and deliberately does **not**
use ECOSTRESS's own cloud raster, because that raster over-masks genuinely cold water (design
note at `datacube.py:229-235`). So cloud contamination currently survives into the cube.

This adds cloud screening as **two new, independently-selectable post-assembly preprocess steps**
in this repo's `preprocess.py` registry (issue #29), each mirroring the existing `fill_water`
step (read raw channels → emit a modified channel + a companion flag; the raw cube stays
untouched):

1. **`filter_clouds`** — port of the user's oceanSR protocol: screen ECOSTRESS pixels against a
   gap-free baseline L4 SST product (**MUR** or **CMEMS**), either by a fixed cold offset or a
   distribution-based, seasonally-aware outlier floor.
2. **`filter_cloud_cover`** — gate ECOSTRESS scenes/pixels on **meteorological total cloud cover**
   (HRRR or ERA5) when that data is available.

Both steps apply the same action to a dropped pixel and **compose** (their drops stack), so a
user can enable either or both.

## Design decisions (confirmed with user)

### Shared: filter action + composition
- Per dropped pixel: set `eco_valid_<ver>=0`, set `eco_sst_<ver>=NaN` (`mask_sst` default True),
  and emit a companion `..._cloudfiltered` uint8 flag — the `fill_water` "modified channel +
  companion flag" shape. This keeps oceanSR's two consumption modes (`eco_valid` directly, and
  `valid_from_sst` which rebuilds from finite SST) in agreement, and stays auditable.
- **Composition**: both steps mutate `eco_valid_<ver>` / `eco_sst_<ver>`, but `ctx.read` only sees
  the RAW cube. A shared helper `_working(ctx, name)` returns the already-emitted value from
  `ctx.channels` if present, else `ctx.read` — so whichever step runs second reads the first's
  output and the drops **union** (valids AND, NaNs union) regardless of order. `filter_cloud_cover`
  declares `depends_on=("filter_clouds",)` for deterministic ordering (ignored when `filter_clouds`
  isn't selected).
- **Sensor scope**: `sensors` default `["eco"]` (ECOSTRESS only), configurable.
- Units: SST cutoffs derive from the cube's own baseline values, so they hold in K or °C
  regardless of the grid's `to_celsius` setting. Met cloud cover is **percent (0–100)**, so its
  thresholds are in percent.

### Step 1 — `filter_clouds` (baseline deviation)
Two methods (`method` option):
- `method: "offset"` (**default**) — flag where `baseline − eco > threshold_k` (default 5.0 K,
  oceanSR's value; `<= 0` disables). Uses the **per-day co-located** baseline value (prefers
  `fill_water`'s gap-filled baseline when that step ran).
- `method: "sigma"` — **baseline floor**: build the baseline's climatology `mean`/`σ`, flag any
  ECOSTRESS pixel colder than `mean − n_sigma·σ` (`n_sigma` default 3.0). σ comes from the dense,
  gap-free baseline field.

For the sigma method:
- **Scope** (`stat_scope`): `"pixel"` (**default**, per-cell climatology, preserves spatial
  gradients) or `"pooled"` (one distribution across all cells & times).
- **Seasonality** (`seasonality`, optional): `"off"` (**default**) or `"harmonic"` — annual sin/cos
  climatology `μ(doy) = β0 + β1·sin + β2·cos` fit by least squares (σ = std of residuals about it),
  reusing the cube's `doy_sin`/`doy_cos` `(time,)` channels (`datacube.py:766`, period 365.25) as
  design columns; degrades to `"off"` (logged) when the DOY span can't constrain the harmonic.
- **Baseline** (`baseline`): default `"mur_sst"`; may name a `cmems_*` channel. NaN baseline → no
  cutoff there → pixel kept.

### Step 2 — `filter_cloud_cover` (meteorological TCC)
- **Data**: the met total-cloud-cover channel, **percent 0–100**. Prefer the ECOSTRESS
  overpass-matched channel `eco_cloud_cover_<src>` (`(time,y,x)`, valued at the overpass instant);
  fall back to the daily forcing `cloud_cover_<src>`. Works with **HRRR and ERA5**: `source`
  option (default auto — prefer overpass over forcing, prefer `hrrr` then `era5`; set e.g.
  `source: era5` to force). If no met cloud channel exists (met not acquired), the step logs and
  no-ops.
- **Two gates, both configurable** (a scene is dropped if EITHER fires):
  - **Scene-level** (`scene_max_pct`, default 30.0): per overpass, aggregate the met cloud field
    over the AOI (spatial mean over water pixels via `landcover_water`, else all finite pixels); if
    the scene mean exceeds the threshold, drop the **entire** overpass (all its ECOSTRESS pixels).
  - **Per-pixel** (`pixel_max_pct`, default 80.0): flag individual pixels where the met cloud
    cover at that pixel/overpass exceeds the threshold. (HRRR is ~3 km, coarse vs 70 m ECOSTRESS,
    so this masks smooth regions — default kept permissive.)
  - Either gate is disabled by setting its threshold to `null` (or `>= 100`).
- **Version handling**: the met overpass channel is per **sensor** (`eco_cloud_cover_<src>`, one
  overpass identity), not per SST version — compute `dropped` once from it, then apply to every
  `eco_sst_<ver>` / `eco_valid_<ver>`.

## Changes

All paths under `/Users/johnbuckner/github/coastal_sst_data`.

### 1. `src/coastal_sst_data/processes/preprocess.py` — two new steps + helpers (primary)

Add after `_fill_channels` (~line 346), before the `STEPS` tuple:

- **`_working(ctx, name, dims=T3, dtype=...)`** — the composition helper (prefers `ctx.channels`).
- **`_step_filter_clouds(ctx)`** — baseline-deviation filter. Read+validate options (raise
  `ValueError` on unknown `method`/`stat_scope`/`seasonality` *values*). For each `pre` in
  `sensors`, each `sst_name` in `ctx.channels_with_prefix(f"{pre}_sst")` (`suffix` = `""` flat or
  `"_v002"` stacked): `eco = _working(ctx, sst_name)`; compute `dropped`:
  - offset: `dropped = (_baseline_perday(ctx, baseline) - eco) > threshold_k`;
  - sigma: `dropped = eco < _baseline_cutoff(ctx, baseline, n_sigma, stat_scope, seasonality)`.
  NaN in any operand → kept. Then emit updated `eco_valid{suffix}` (= `_working(valid) & ~dropped`),
  NaN'd `eco_sst{suffix}` (if `mask_sst`), and `eco_sst{suffix}_cloudfiltered` (uint8, flag). Carry
  the baseline once.
  - **`_baseline_perday(ctx, name)`**: prefer `ctx.channels[name]` (filled) else `ctx.read`.
  - **`_baseline_cutoff(...)`** — climatology from the **observed** raw baseline (`ctx.read`; never
    filled values, so invented cells don't collapse σ). Design matrix `X` `(T,k)`: harmonic →
    `[1, doy_sin, doy_cos]` (k=3), off → `[1]` (k=1), with a short-span fallback to k=1. Fully
    vectorized **masked normal equations** (handles gap-free-over-water / NaN-over-land): with
    `M`=finite mask, `Bz`=baseline·M — `pixel` scope: `XᵀX=einsum('ti,tj,tyx->yxij',X,X,M)`,
    `Xᵀb=einsum('ti,tyx->yxi',X,Bz)`, batched `np.linalg.solve`→`β(y,x,k)` (cells with `<k` finite
    times → NaN), `μ=einsum('ti,yxi->tyx',X,β)`, `σ=sqrt(Σ M(base−μ)²/(ΣM−k))`; `pooled` scope:
    reduce the same sums over time **and** cells → one `β`, scalar `σ`, `μ(t)` broadcast.
    `cutoff = μ − n_sigma·σ` (NaN where undefined → pixel kept).

- **`_step_filter_cloud_cover(ctx)`** — met TCC gate. Resolve the met cloud channel per sensor via
  `source` (auto-discovery over `eco_cloud_cover_<src>` then `cloud_cover_<src>`, src pref
  hrrr→era5); None → log + skip that sensor. Compute per-sensor `dropped`:
  `scene_pct(t) = spatial mean of tcc[t] over water` (or all finite); `scene_drop = scene_pct >
  scene_max_pct` (broadcast over y,x); `pixel_drop = tcc > pixel_max_pct`; `dropped = scene_drop |
  pixel_drop` (each gate skipped if its threshold is null/≥100). Apply `dropped` to every
  `eco_sst_<ver>`/`eco_valid_<ver>` (via `_working`, composing with step 1): emit updated valid,
  NaN'd sst, and `eco_sst_<ver>_metcloudfiltered` (uint8 flag). Also emit
  `eco_scene_cloud_pct_<src>` `(time,)` diagnostic (the scene mean %). Carry the met cloud channel.

Register both in the `STEPS` tuple (after `fill_water`, ~line 368):
```python
PreprocessStep(
    key="filter_clouds",
    reads=("eco_sst", "eco_valid", "eco_cloud", "mur_sst", "cmems_", "doy_sin", "doy_cos"),
    writes=("_sst", "_valid", "_cloudfiltered"),
    fn=_step_filter_clouds, depends_on=("fill_water",),
    option_keys=frozenset({"method", "threshold_k", "n_sigma", "baseline",
                           "stat_scope", "seasonality", "sensors", "mask_sst", "use_cloud_raster"}),
    provenance_inputs=("ecostress", "mur"),
),
PreprocessStep(
    key="filter_cloud_cover",
    reads=("eco_cloud_cover_", "cloud_cover_", "eco_sst", "eco_valid", "landcover_water"),
    writes=("_sst", "_valid", "_metcloudfiltered", "_scene_cloud_pct"),
    fn=_step_filter_cloud_cover, depends_on=("filter_clouds",),
    option_keys=frozenset({"source", "scene_max_pct", "pixel_max_pct",
                           "sensors", "mask_sst", "water_mask_channel"}),
    provenance_inputs=("met_overpass", "met", "ecostress"),
),
```
`_check_steps()` (line 411) validates uniqueness/`depends_on`/acyclicity at import. Update the
module blurb (lines 19-33) with the two new bullets.

### 2. `src/coastal_sst_data/config.py` — no code change

`PreprocessStepOptions` is `extra="allow"`; new keys validated at stage time by
`_check_step_options` (414-438). Optionally add the two step names to the `PreprocessSpec`
docstring (380-381).

### 3. `src/coastal_sst_data/provenance.py` — one small branch

The mutated `eco_sst_<ver>`/`eco_valid_<ver>` and the `..._cloudfiltered`/`..._metcloudfiltered`
flags all resolve via the existing `_SENSOR_RE` branch (`base` = `"sst"`/`"valid"` → `[sensor]`,
196-207) — no change. **But** `eco_scene_cloud_pct_<src>` would fall through to the blank-provenance
warning: add a branch in `field_inputs` (near line 216) — `if rest.startswith("scene_cloud_pct"):
return ["met_overpass", sensor]`. The actual met source is recorded via `ctx.carry` + attrs.

### 4. `tests/test_preprocess.py` — new tests

Mirror `test_fill_water_*` (126-172), reusing `write_ecostress_two_scenes`/`write_mur`
(`test_datacube.py:70,82`; eco flies day 0). Add a small synthetic met writer (or extend the
fixture) that writes `eco_cloud_cover_hrrr` (and `cloud_cover_hrrr`) so the cube carries them.
Cover:
- **filter_clouds offset**: warm MUR vs cold eco column → flag/valid/sst as expected; dtypes;
  `threshold_k: 0` no-op; filled-baseline path when `fill_water` ran (proves `depends_on` + reading
  `ctx.channels`).
- **filter_clouds sigma** (per-pixel & pooled, seasonality off) with known mean/σ; NaN-baseline
  pixels never flagged. **harmonic**: build a bespoke multi-month cube by hand (year of daily
  `time`, seasonal baseline sinusoid + noise, a couple eco scenes); a warm-season eco value that is
  a global-mean outlier but seasonal-mean normal is NOT flagged; a genuine cloud value IS; plus a
  short-span fallback-to-constant (logged) case.
- **filter_cloud_cover scene**: an overpass with AOI-mean `eco_cloud_cover_hrrr` > 30 → all eco
  pixels that day dropped, `eco_valid==0`, `eco_sst==NaN`, `eco_scene_cloud_pct_hrrr` recorded; a
  clear overpass kept.
- **filter_cloud_cover pixel**: a cloudy column (> `pixel_max_pct`) in an otherwise-clear scene →
  only those pixels dropped. Disabling a gate (`null`/≥100) is a no-op for that gate.
- **filter_cloud_cover ERA5**: same via `eco_cloud_cover_era5` / `source: era5`. **Met absent** →
  no-op, no crash.
- **Composition**: enable both steps; a pixel dropped by the baseline filter AND a different pixel
  dropped by the met gate are BOTH invalid in the final cube (proves `_working` stacking).
- Extend the stage-time option-rejection test (line 201) with a bad option for each new step.

**Golden**: add both steps (offset mode; scene gate) to the `project` fixture steps (line 47) and
extend the full fixture to write a met cloud channel; regenerate with
`UPDATE_GOLDEN=1 pytest tests/test_preprocess.py` and review the diff.

### 5. `docs/DEVELOPMENT.md` §5b (~434-439)

Add example YAML for both steps, e.g.:
```yaml
preprocess:
  enabled: true
  steps:
    filter_clouds:      { method: sigma, baseline: mur_sst, n_sigma: 3.0,
                          stat_scope: pixel, seasonality: harmonic }
    filter_cloud_cover: { source: hrrr, scene_max_pct: 30, pixel_max_pct: 80 }
```
plus a sentence noting they compose, `filter_clouds` ports oceanSR's cold-deviation filter, and
`filter_cloud_cover` needs the met `(eco,<src>)` overpass combo (or a `cloud_cover_<src>` forcing
channel) in the cube.

## Edge cases handled

All-NaN baseline / undefined cutoff / missing met channel → nothing flagged, no crash. Sensor
absent → step no-ops. Multiple ECOSTRESS versions → each `eco_sst_<ver>` handled (met gate computes
`dropped` once per sensor, applies to all versions). Flat sensor (`suffix=""`) → bare channels.
Missing `eco_valid` → validity update skipped; SST-NaN + flag still keep both consumption modes
consistent. Cells with `<k` finite baseline times → cutoff NaN → kept. Harmonic under-constrained
→ fall back to constant. Days with no overpass → met channel NaN → scene mean NaN → not dropped.
Both steps use `_working`, so their order never changes the union of drops. `threshold_k<=0` or a
`null`/≥100 met threshold → that gate disabled.

## Verification

1. `pytest tests/test_preprocess.py -q` — new `test_filter_clouds_*` / `test_filter_cloud_cover_*`
   pass; existing `fill_water`/`water_line`/golden tests still pass (regenerate golden after
   extending the fixture). Then `pytest -q`.
2. End-to-end on real data (config with `ecostress` + `mur` + `met`/`met_overpass` [(eco,hrrr) or
   (eco,era5) combo] selected, `preprocess.enabled: true`):
   ```
   coastal-sst-data preprocess --config <config>.yaml --aoi <aoi>
   ```
   with both steps configured. Open `preprocessed/<aoi>.zarr`: confirm `eco_sst_<ver>` is NaN and
   `eco_valid_<ver>` is 0 where either `..._cloudfiltered` (cold outliers) or `..._metcloudfiltered`
   (high TCC) fired, `eco_scene_cloud_pct_<src>` matches rejected overpasses, and the raw
   `datacube/<aoi>.zarr` is byte-unchanged. Spot-check that flagged pixels coincide with cloud
   cold-spots / cloudy overpasses and that clear cold water is NOT dropped; sweep `n_sigma`,
   `threshold_k`, `scene_max_pct`, `pixel_max_pct` and toggle `stat_scope`/`seasonality`/`source`.
