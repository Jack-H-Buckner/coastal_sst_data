# Plan: Post-assembly preprocessing stage (issue #29)

## Context

Today the pipeline acquires raw products onto a shared per-AoI grid, then a **terminal
assembler** (`processes/datacube.py`) knits them into one **raw** Zarr cube per AoI at
`<output_dir>/datacube/<aoi>.zarr`. That cube deliberately ships **raw ingredients** — a
prior refactor (decisions D6/D7/D12) *removed* the in-cube `fill_mur_water`,
`fill_cmems_water`, and `water_level` channels because masking / filling / water-level are
downstream modelling determinations, not properties of the raw data. `config.DataCubeSpec`
even rejects those keys with `extra="forbid"`.

Issue #29 wants those determinations back — but as a **structured, config-driven layer that
runs AFTER assembly**, with "a similar structure as the data loading processes: a common
input/output contract and a registry so a new step is easy to add." Two initial steps:

1. **`water_line`** — the tide-adjusted waterline from tide + DEM, per thermal sensor at its overpass.
2. **`fill_water`** — fill NaN water cells of the level-4 SST products (`mur_sst`, `cmems_*`)
   with the nearest finite estimate.

**Outcome:** a new opt-in `preprocess` stage that reads each assembled `datacube/<aoi>.zarr`
and writes a **separate derived cube** `preprocessed/<aoi>.zarr`, leaving the raw cube
untouched. Adding a future step is one registration in a step registry.

### Decisions locked with the user
- **Output:** a **separate** derived cube `<output_dir>/preprocessed/<aoi>.zarr`; the raw
  `datacube/<aoi>.zarr` stays byte-untouched.
- **water_line:** emit per-sensor `<sensor>_water_elev` (time,y,x float32) + `<sensor>_water_class`
  (time,y,x uint8; SUBMERGED=0 / EXPOSED=1 / UNKNOWN=255) for eco/lst/modis, using each
  sensor's own overpass tide.
- **fill_water:** fillable cells = `landcover_water==1`; nearest-finite fill via
  `scipy.ndimage.distance_transform_edt`; emit a `<channel>_filled` uint8 companion mask.

### Design stance
Mirror the datacube's lightweight **Contributor protocol**, NOT the heavy `products.py`
registry. A preprocess step reads an already-opened xarray cube and writes arrays — it has no
auth / `dir` / `Kind` / `required_vars` / dotted-module lazy-resolution. The right analog is
`datacube.CONTRIBUTORS` + `Contributor` + `_topo_order` + `_check_contributors` +
`AssemblyContext.emit`, which *is* the "common contract + registry + topological order + loud
invariant" the user asked for.

---

## Files

### New
- **`src/coastal_sst_data/processes/preprocess.py`** — the whole stage: `PreprocessStep`
  dataclass, `STEPS` registry, the two step functions, `PreprocessContext`, import-time
  `_check_steps()`, and the stage entry (`preprocess()` / `run()` / `_build_eff()` / `main()`).
  Both steps live in this one file (each ~30–50 lines of glue over existing math); split into
  `preprocess_*.py` only if a third, larger step arrives. This module runs dead-last and may
  freely import `config`, `products`, `grid`, `store`, `provenance`, `report`, `entry`,
  `water_level`, and `datacube` (for `build_encoding` / `write_zarr_safe`) — there is no
  import-cycle constraint forcing it low like `products.py`.
- **`tests/test_preprocess.py`** + **`tests/golden/preprocessed_golden.json`**.

### Edited (all additive)
- `src/coastal_sst_data/config.py` — add `PreprocessSpec` (mirror `DataCubeSpec`) and
  `Project.preprocess`.
- `src/coastal_sst_data/provenance.py` — add `field_inputs` branch for the water_line channels.
- `src/coastal_sst_data/pipeline.py` — terminal `preprocess` block in `run_pipeline` +
  `--preprocess`.
- `src/coastal_sst_data/cli.py` — `_cmd_preprocess`, a `preprocess` subparser, `--preprocess`
  on `run`, docstring.
- `docs/DEVELOPMENT.md` / `README.md` — document the new stage (§ "Adding a preprocess step"
  and the CLI/`preprocess:` config surface).

---

## Design detail

### 1. Step spec + registry (mirror `Contributor`/`CONTRIBUTORS`/`_check_contributors`)
```python
@dataclass(frozen=True)
class PreprocessStep:
    key: str                              # step name; also the config selector key
    reads: tuple[str, ...]                # cube channel families it consumes (docs + topo edges)
    writes: tuple[str, ...]               # cube channel families it emits
    fn: Callable[[PreprocessContext], None]
    depends_on: tuple[str, ...] = ()      # other step keys that must run first
    option_keys: frozenset[str] = frozenset()   # per-step config options it reads
    provenance_inputs: tuple[str, ...] = ()

STEPS = (
    PreprocessStep("water_line",
        reads=("elevation_", "tide_", "_hour"), writes=("_water_elev", "_water_class"),
        fn=_step_water_line,
        option_keys=frozenset({"dem_source", "tide_source", "sensors"}),
        provenance_inputs=("bathymetry", "tides")),
    PreprocessStep("fill_water",
        reads=("landcover_water", "mur_sst", "cmems_"), writes=("_filled",),
        fn=_step_fill_water,
        option_keys=frozenset({"sources", "mask_channel"})),
)
```
Ordering: reuse a topo-sort like `datacube._topo_order` — order on `depends_on` (primary) plus
writer→reader edges where one step's `writes` family overlaps another's `reads`. water_line is
declared first so a future "fill-using-waterline" step (`depends_on=("water_line",)`) is placed
correctly. `_check_steps()` (called at import) asserts: unique keys; every `depends_on` names a
real step; the graph is acyclic (call `_topo_order` inside the check). No "every product must
have a step" analog — steps are opt-in.

### 2. `PreprocessContext` (mirror `AssemblyContext`)
Holds `g, eff, days, aid, H, W, ds_raw (opened raw cube), channels, var_attrs, global_attrs`.
Methods: `emit(name, dims, arr, **attrs)`; `read(name, dims=..., dtype="float32") -> ndarray|None`
(defensive — missing or wrong-shape channel returns `None`, mirroring `datacube.load_*`, so a
step never crashes when an input product wasn't selected); `channels_with_prefix(prefix)` (for
`cmems_*` discovery, like `cmems_channels`); `sensor_hours(prefix)` (coalesce `<pre>_hour` and
`<pre>_hour_<ver>`, first finite per day); `aligned_dir(product, source)` (reach the raw
tide series on disk when needed). A step that finds none of its inputs emits nothing and warns.

### 3. Step math (reuse existing code as-is)
**`water_line`** — for each sensor in `sensors` (default `products.sensors()` prefixes):
- `elev = ctx.read(f"elevation_{dem_source}", dims=("y","x"))`; datum from that channel's attr
  `ds_raw[...].attrs.get("datum_offset_m", 0.0)`. If `dem_source` unset and exactly one
  `elevation_*` exists, auto-pick it. Elevation absent → all-UNKNOWN, continue.
- Overpass hours: `hours = ctx.sensor_hours(s)`.
- Overpass tide: prefer the cube's ready-made `f"{s}_tide_{tide_source}"` when present; else
  `water_level.tide_at_overpass(water_level.load_tide_series(ctx.aligned_dir("tides", src), aid),
  days, hours)`. (All three `water_level` functions reused **verbatim**.)
- `water_elev, water_class = water_level.water_level_fields(elev, tide, datum_offset_m=datum)` —
  used **verbatim**. Emit `<s>_water_elev` (units=m, +above/-below the waterline) and
  `<s>_water_class` (flag_values [0,1,255], flag_meanings "submerged exposed unknown").

**`fill_water`** — restore `fill_water_nn` **verbatim** from `git show
3d99ec5^:src/coastal_sst_data/processes/datacube.py` (the `distance_transform_edt` NN fill,
same technique as `insitu.py:67`). For each source channel `c` (default `mur_sst` + all
`cmems_*` discovered, excluding `_valid`/`_filled`):
- `water = ctx.read(mask_channel or "landcover_water", dims=("y","x")) > 0.5`; absent → skip (warn).
- `raw = ctx.read(c, dims=("time","y","x"))`; absent → skip. `observed = isfinite(raw)`.
- `filled = fill_water_nn(raw, water)`; `filled_mask = (isfinite(filled) & ~observed).uint8`.
- Emit the filled `c` and companion `f"{c}_filled"` (1 = invented over unobserved water, 0 = observed).

### 4. Config (`config.py`, mirror `DataCubeSpec`)
```python
class PreprocessStepOptions(BaseModel):
    model_config = {"extra": "allow"}    # validated against the step registry, not here
class PreprocessSpec(BaseModel):
    model_config = {"extra": "forbid"}
    enabled: bool = False                # opt-in; existing runs unaffected
    steps: dict[str, PreprocessStepOptions] = Field(default_factory=dict)
    output_subdir: str = "preprocessed"
    overwrite: bool = False
    compression: CompressionSpec = Field(default_factory=CompressionSpec)
    chunks: dict[str, int] = Field(default_factory=lambda: {"time": 64, "y": 128, "x": 128})
# Project: preprocess: PreprocessSpec = Field(default_factory=PreprocessSpec)
```
**Per-step option validation stays in the stage, not `config.py`** (importing the step registry
into `config.py` would reintroduce a cycle). At the top of `preprocess.run()`, `_check_step_options(eff)`
rejects unknown step keys and unknown per-step option keys, using `difflib.get_close_matches`
for a "did you mean" hint exactly like config's `_option_keys_are_known`. Region-level source
overrides are **out of scope for v1** — options are project-level, and a step auto-picks the sole
present `elevation_*` / `tide_*` when unspecified (covers most single-source AoIs). Example config:
```yaml
preprocess:
  enabled: true
  steps:
    water_line: { dem_source: cudem, tide_source: coops, sensors: [eco, lst] }
    fill_water: { sources: [mur, cmems] }
```

### 5. Stage entry + write path (mirror `datacube.assemble`/`run`)
- Signature `preprocess(project, *, grids=None, aois=None, dry_run=False, overwrite=False)` —
  identical to every `acquire`/`assemble`, so `entry.process_main(preprocess, ...)` works.
- Per AoI: raw cube at `root/datacube.output_subdir/<aoi>.zarr`. **If absent → warn ("run
  assemble first") and skip that AoI, never crash the batch.** Derived cube at
  `root/preprocess.output_subdir/<aoi>.zarr`; skip if it exists and not `overwrite`;
  `store.sweep_scratch` first.
- Open raw with `xr.open_zarr`. Run `_topo_order(selected_steps)`; each `step.fn(ctx)`.
- Build `ds_out` **on the raw cube's own coords** (`ds_raw.coords` — never re-derive from
  `g.xy_centers()`) so the two cubes align cell-for-cell.
- **Derived-cube contents (recommended):** derived channels **+ copy-forward the specific raw
  inputs they reference** (`elevation_<dem>` with datum attrs, `landcover_water`, the
  tide/hour channels used, and the *filled* `mur_sst`/`cmems_*` + their `_filled` masks) — NOT a
  full raw copy, NOT derived-only. This makes `preprocessed/<aoi>.zarr` a legible, standalone
  "cleaned" product. Stamp `derived_from`, `aoi_id`, `crs`, and a `preprocess` attr recording
  which steps ran with which options.
- Write via `datacube.write_zarr_safe(ds_out, zpath, datacube.build_encoding(ds_out, compression,
  chunks))` — reuses `store.atomic` (raw cube stays intact on a mid-write kill) + Blosc encoding
  (uint8 masks get bitshuffle). Return a `report.ProductReport("preprocess")`.
- **No `coverage` pass** on the derived cube (coverage measures *holes*; this stage *fills*
  them, so it would mislead).

### 6. Provenance (`provenance.field_inputs`)
- `<sensor>_water_elev` / `<sensor>_water_class`: add a branch (in the `_SENSOR_RE` section)
  returning `["bathymetry", "tides", sensor]` (derived fields list ALL inputs). The existing
  `elevation_*`/`landcover_water`/`tide_*` copy-forwards already map. `<c>_filled` masks are
  **already covered** by the existing `mur_`/`cmems_` prefix branches — no change needed.
- In `preprocess.run`, build the record with `provenance.build(project, list(ds_out.data_vars),
  prod)` and stamp the same attrs `datacube.assemble_aoi` does.

### 7. Wiring
- **`pipeline.run_pipeline`**: add `preprocess=False` kwarg; a terminal block **after** the
  `if assemble:` block, gated on `preprocess and project.preprocess.enabled`, calling
  `pp.preprocess(...)` and recording an `"preprocess"` outcome/report (same try/except shape as
  the assemble block). `--preprocess` in `pipeline.main()`.
- **`cli.py`**: `--preprocess` on `p_run` (pass to `_cmd_run`); `_cmd_preprocess` mirroring
  `_cmd_assemble`; a `preprocess` subparser (`--aoi`, `--overwrite`, `--dry-run`); update the
  docstring command list.
- **Standalone**: `python -m coastal_sst_data.processes.preprocess --config … --aoi … [--overwrite]`.

---

## Verification (end-to-end)

1. **Config** — `coastal-sst-data validate --config examples/config.test.yaml` after adding a
   `preprocess:` block; confirm an unknown step key / option fails with a clear message.
2. **Unit (offline, synthetic) — new `tests/test_preprocess.py`**, reusing
   `test_water_level.py`/`test_datacube.py` writers (assemble a raw cube from synthetic aligned
   files, then run `preprocess_aoi`):
   - water_line: `eco_water_elev`/`eco_water_class` equal a direct `water_level_fields` call
     (the step is pure glue); datum offset shifts the waterline; UNKNOWN where no overpass.
   - fill_water: NaN cells over `landcover_water==1` become finite via NN; land NaNs stay NaN;
     `mur_sst_filled`==1 exactly on invented-over-water cells, 0 on observed; same for a `cmems_*`.
   - edges: raw cube absent → AoI skipped, no crash; tides absent → `*_water_class` all-UNKNOWN;
     landcover absent → fill skipped; overwrite semantics.
   - invariant: monkeypatch `STEPS` with a bad `depends_on`/duplicate key → `_check_steps()` raises
     (mirror `test_add_a_covariate.py`).
   - write-path: derived cube at `preprocessed/<aoi>.zarr`; **raw `datacube/<aoi>.zarr`
     byte-unchanged**; coords identical between the two.
   - a separate golden `tests/golden/preprocessed_golden.json` + `test_preprocessed_golden_is_unchanged`
     (reuse the `_snapshot`/`_diff`/`RUNSTAMP_ATTRS` helpers). **Do not touch the existing
     `datacube_golden.json`** — `test_golden_cube_is_unchanged` is the guard that the raw cube
     didn't change.
   - extend `test_cli.py`/`test_pipeline.py` for `--preprocess` and the `preprocess` subcommand.
3. **Whole suite** — `pytest` (offline; confirms the additive `config`/`provenance`/`pipeline`/`cli`
   edits break nothing, especially `test_provenance.py`, `test_config.py`, and both goldens).
4. **Real end-to-end (optional, one AoI)**:
   `coastal-sst-data run --config config.yaml --assemble --preprocess --aoi <one>` then
   `python -c "import xarray as xr; ds=xr.open_zarr('data/preprocessed/<one>.zarr'); print(ds)"` —
   confirm `<sensor>_water_elev/_class` and `*_filled` channels, and that
   `data/datacube/<one>.zarr` is unchanged.
