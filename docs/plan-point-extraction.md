# Point time-series extraction from assembled datacubes

*Implemented in v0.5.0. This is the design record: what was decided, why, and which silent failure each guard exists for. The user-facing documentation is the `extract` section of [README.md](../README.md).*

## Context

The pipeline produces per-AoI datacubes at `<output_dir>/datacube/<aoi>.zarr` — dims
`(time, y, x)` on a projected grid, ~60 channels. Downstream applications need the
*opposite* shape: for a set of lat/lon sites, the full time series of every channel, as a
table.

Nothing in the package does this today. There is no tabular writer at all (no CSV/parquet
output, no pyarrow dependency) and no generic point-sampling path. The closest existing
machinery is [insitu.py](src/coastal_sst_data/processes/insitu.py) — `station_pixels()`
already maps lon/lat to `(row, col)` on an `AoiGrid` — but it is used only to *inject*
buoy observations *into* the cube.

**Goal:** a new `coastal-sst-data extract` command. Given a CSV of lat/lon points and an
`extract:` config block naming the channels to pull (each with an optional neighbourhood
radius, an optional pixel mask, and one or more aggregation statistics), it writes one
long-format parquet or CSV file:

```
point_id, lat, lon, aoi, time, variable, stat, radius_m, value
```

## Decisions (settled with you)

| | |
|---|---|
| Channel selection | **Explicit list only.** A listed channel absent from the cube is a hard error naming it; nothing is pulled implicitly. |
| Config home | New `extract:` block in the project YAML (a pydantic model on `Project`, like `datacube:`/`preprocess:`). |
| Point → AoI | Exactly one AoI per point: the AoI whose grid *contains* it; if several do, the one whose grid centre is nearest (then AoI name, so it never depends on dict order). A point in no AoI is dropped with a `WARNING`. |
| Stats | Explicit NaN semantics: `mean` propagates NaN, `nanmean` skips it. Plus `count_valid`. Default `nearest`. |
| Neighbourhood mask | Optional per-channel `mask:` (e.g. water-only aggregation). Off by default. |
| Table shape | Long, with an explicit `stat` column (not folded into the variable name). |
| Output | parquet by default, `--format csv`; pyarrow lazy-imported from a new `extract` extra. |
| Non-3D channels | Static `(y,x)` → one row with a null `time`. 1D `(time,)` → one row per time. |

## Optionality contract

Only a minority of projects need this, so it must cost the others **nothing** — not a
dependency, not a config key they have to write, not an import, not a slower run. This is
a hard requirement, and each clause below has a test in §6:

1. **The config block is optional.** `Project.extract` has a `default_factory`, so every
   existing config still validates untouched and `validate`'s summary says nothing about
   extraction unless `extract.channels` is non-empty.
2. **No new core dependency.** `pyarrow` goes in a new `extract` extra only, lazy-imported
   inside `store.write_table` and only on the parquet branch. `--format csv` needs nothing
   beyond pandas, which is already core. A base install is byte-for-byte as light as today.
3. **No import cost.** `processes/extract.py` is imported *inside* `_cmd_extract` (exactly
   how `_cmd_preprocess` imports `preprocess`), so `import coastal_sst_data` and every
   other subcommand never touch it. `points.py` imports only numpy/pandas (core) and pulls
   `insitu.station_pixels` in lazily.
4. **Nothing runs implicitly.** Not wired into `pipeline.run_pipeline`, not in the
   `products.py` registry, no new auth backend, no new runtime gate. It executes only when
   someone types `coastal-sst-data extract`.
5. **The cube is unchanged.** No new channels, no new attrs, no write into any `.zarr` —
   so the `datacube`/`preprocessed` golden snapshots are untouched and no existing test
   changes.
6. **The whole test suite still passes without pyarrow installed** (parquet cases use
   `pytest.importorskip`), and `pip install coastal_sst_data` followed by
   `extract --format csv` works with no extra.

## Shape of the feature

A consumer stage in the same shape as `preprocess`: it reads the cubes and writes one
durable artefact through `store.atomic`. It never writes into a cube, so re-running is
safe and nothing upstream can be corrupted. A `run --extract` flag could be added later
without changing anything here.

| File | Role |
|---|---|
| [config.py](src/coastal_sst_data/config.py) | `ExtractChannel` + `ExtractSpec`, hung off `Project.extract` |
| `src/coastal_sst_data/points.py` **(new)** | Pure: read the points CSV, canonicalise columns, assign points → AoIs. No xarray, no zarr, no config. |
| `src/coastal_sst_data/processes/extract.py` **(new)** | The stage: open each cube, window/reduce, build the long frame, hand it to the writer |
| [store.py](src/coastal_sst_data/store.py) | `write_table()` — the atomic tabular writer + lazy pyarrow |
| [cli.py](src/coastal_sst_data/cli.py) | the `extract` subcommand |

Splitting `points.py` out of `extract.py` follows
[insitu.py](src/coastal_sst_data/processes/insitu.py)'s stated rationale ("no network, no
config -- so it is trivially testable"): every geometric decision that can silently
produce a wrong number lives in a module with no I/O; the stage module only orchestrates.

---

## 1. Config — `ExtractSpec` in `config.py`

Inserted after `PreprocessSpec`, in its style: every field defaulted, `extra="forbid"`, a
`mode="before"` validator so terse YAML works.

```yaml
extract:
  points: "path/to/sites.csv"        # or pass --points
  format: parquet                    # or csv
  channels:
    eco_sst_clean: {radius_m: 300, stat: [nanmean, nanstd, count_valid], mask: water}
    mur_sst:       {radius_m: 1000, stat: mean}      # NaN-propagating: any gap -> NaN
    eco_valid:     {radius_m: 300,  stat: nanmean}   # 0/1 mask -> fraction valid
    depth_cudem:                                     # bare key: nearest pixel
    tide_coops:                                      # 1-D (time,): AoI-wide series
```

```python
# A CLOSED set: a typo'd stat must fail at config-load time, not after an hour of cube reads.
# The nan* variants are the NaN-SKIPPING counterparts of the plain ones; `nearest` is not a
# reduction at all -- it is the value of the pixel the point falls in.
STATS = ("nearest",
         "mean", "median", "std", "min", "max", "sum",
         "nanmean", "nanmedian", "nanstd", "nanmin", "nanmax", "nansum",
         "count", "count_valid")
_PCT_RE = re.compile(r"^p(100|\d{1,2})(\.\d+)?$")     # p10, p90, p97.5 -- NaN-skipping

class ExtractChannel(BaseModel):
    model_config = {"extra": "forbid"}
    radius_m: float = Field(0.0, ge=0)                # 0 -> just the containing pixel
    stat: list[str] = Field(default_factory=lambda: ["nearest"])
    mask: str | None = None                           # "water", or a 2-D channel name

class ExtractSpec(BaseModel):
    model_config = {"extra": "forbid"}
    points: Path | None = None
    columns: dict[str, str] = Field(default_factory=dict)   # canonical -> your column name
    channels: dict[str, ExtractChannel] = Field(default_factory=dict)
    format: Literal["parquet", "csv"] = "parquet"
    output_subdir: str = "extract"
    stem: str = "points"
    overwrite: bool = False
    memory_budget_gb: float | None = Field(None, gt=0)      # falls back to datacube's
```

`Project` gains `extract: ExtractSpec = Field(default_factory=ExtractSpec)`; the block
stays optional, and `Project`'s existing `extra="forbid"` makes `extrac:` fail at load.

**Validators**
- `stat` accepts a scalar (`stat: mean`) or a list; a bare `chan:` (null) means "extract it,
  nearest pixel"; `chan: [mean, std]` is shorthand for `{stat: [...]}` — the same
  `mode="before"` trick as `PreprocessSpec.steps`.
- Unknown stat → `ValueError` listing `STATS` and the percentile form. Duplicate stats
  rejected (they would emit two identical rows and break the table's primary key).
- `radius_m > 0` with `stat == ["nearest"]` → rejected: the config would state a footprint
  the run ignores (`_option_keys_are_known`'s "a key nothing reads is a config that LIES").
- Channel names cannot be checked against a cube at load time — that check lives in the
  stage, where the cube is open.

### Stat vocabulary

| stat | meaning | all-NaN neighbourhood |
|---|---|---|
| `nearest` (default) | the value of the pixel the point falls in — **not** the nearest *finite* value, and it ignores `radius_m` and `mask` | NaN |
| `mean` `median` `std` `min` `max` `sum` | plain NumPy, **NaN-propagating**: one cloudy pixel in the window ⇒ NaN | NaN |
| `nanmean` `nanmedian` `nanstd` `nanmin` `nanmax` `nansum` | the NaN-skipping counterparts | NaN |
| `count` | pixels in the neighbourhood after masking, NaN or not — the clipped-window detector | 0.0 |
| `count_valid` | **finite** pixels in the neighbourhood after masking — the cloud-cover detector | 0.0 |
| `p10` … `p97.5` | `np.nanpercentile` (NaN-skipping) | NaN |

- `nearest` is deliberately not "nearest finite". Substituting a neighbour when the point's
  own pixel is cloudy is how a validation set silently acquires a warm bias;
  `insitu.value_at` makes the same call in time ("never a stale value") and this mirrors it.
- All values are cast to float64 before reducing, so uint8 channels (`<s>_valid`,
  `landcover_water`) give a fraction under `mean`, not an integer truncation.
- `std` is population (`ddof=0`), documented; `count_valid < 2` gives 0.0.
- All-NaN reductions run under `warnings.catch_warnings()` + `simplefilter("ignore",
  RuntimeWarning)` — an all-NaN slice is a legitimate outcome here, not a defect.
- `count` / `count_valid` land in the same float `value` column (parquet needs one dtype
  per column); documented.

### The `mask:` option

`mask: water` resolves to the cube's `landcover_water` channel; any other string is taken
as a channel name. Either must exist and be 2-D `(y, x)` — a missing one is a hard error,
never a silent unmasked fallback. The mask is `AND`ed into the neighbourhood before the
reduction, so a coastal point's `nanmean` is not contaminated by land pixels.

- The mask **wins** over the always-include-the-containing-pixel rule (below): a point
  whose own pixel is land contributes nothing, and if that empties the neighbourhood the
  value is NaN with `count == 0` — visible, not substituted.
- `nearest` ignores `mask` by definition (it is one specific pixel); documented explicitly
  so `{stat: [nearest, nanmean], mask: water}` has no surprise.

---

## 2. `points.py` — reading and assigning points

### 2a. Column-name flexibility — `read_points(path, columns=None) -> DataFrame[point_id, lat, lon]`

```python
# Canonical field -> the spellings people actually write.
#
# `x`/`y` are the dangerous pair: a file of PROJECTED coordinates uses exactly those names,
# and 512345.0 read as a longitude lands in the Pacific without a single error. The WGS84
# range check below is what makes offering the alias safe at all.
ALIASES = {
    "point_id": ("point_id", "id", "station_id", "station", "name", "site"),
    "lat":      ("lat", "latitude", "y"),
    "lon":      ("lon", "longitude", "x"),
}
```

1. Explicit `extract.columns` overrides win first (same contract as `insitu_csv.column_map`).
2. Otherwise case-insensitive alias match. **Ambiguity is an error** — a file carrying both
   `lat` and `latitude` raises naming both, rather than silently picking one.
3. Missing lat/lon → `ValueError` naming the columns present, in `insitu_csv.read_file`'s style.
4. Missing id → synthesised `<stem>_0001` with one loud `WARNING`: a table whose ids are row
   numbers cannot be joined back to anything you recognise.
5. Non-numeric / non-finite coordinates dropped with a counted warning.
6. **WGS84 range check** — the units tripwire. Out-of-range values raise, naming both causes
   ("the columns are swapped, or they hold projected metres"). Never auto-swapped.
7. Duplicate `point_id` → error (breaks the primary key, fans out on any downstream join).
   Duplicate (lat, lon) under different ids → warning only (co-located sensors are legal).
8. Returns exactly `[point_id, lat, lon]`; extra input columns are dropped so the output
   schema is fixed — join back on `point_id`.

### 2b. Assignment — `assign_aois(pts, grids) -> +[aoi, row, col, px, py]`

No cube is opened. For each AoI, one call to
[insitu.station_pixels](src/coastal_sst_data/processes/insitu.py#L43) with **all** points
(it builds a `pyproj.Transformer` per call, and pass `.tolist()` per its own comment).

**`water=None` on purpose — no snapping.** `insitu` snaps a buoy off a land pixel because
that is a mask artefact; an extraction point is *where you said it is*, and moving it 300 m
would change the answer with nothing in the output recording it.

Reusing `station_pixels` is the most important decision here: the extractor and the cube's
own `insitu_station` channel then invert the affine with the *same* code, so they cannot
disagree about which pixel a lat/lon is.

- Containment is against the **grid** (what the cube has pixels for), which is up to one
  resolution unit larger than the configured bbox because `compute_aoi_grid` snaps outward.
- Tie-break across overlapping AoIs: nearest grid centre in that grid's own projected CRS,
  then AoI name. (Two candidates can be in different UTM zones; both are metres and the
  comparison only breaks a tie inside an overlap — documented, not geodesic.)
- A point in no AoI is dropped with a `WARNING` naming it. Not an error — a regional station
  list legitimately has points outside the project — but never silent.
- `flag_edge_points(...)`: warn once per point whose neighbourhood will be clipped by the
  grid edge (`min(row, col, H-1-row, W-1-col) * res < max_radius_m`). A clipped window still
  returns a mean — of a half-disc — and nothing in the value says so.

---

## 3. `processes/extract.py` — the stage

```python
def extract(project, *, grids=None, aois=None, points=None, out=None,
            fmt=None, dry_run=False, overwrite=False) -> report.ProductReport | None
def run(eff, grids, only_aoi, dry_run)
def extract_aoi(ds, g, pts, channels) -> pd.DataFrame
```

`_build_eff(project)` mirrors `preprocess._build_eff` (flat dict: cube_dir, out_path,
points, columns, channels, format, overwrite, `memory_budget_gb` falling back to
`datacube.memory_budget_gb`). `main()` via `entry.process_main`, so
`python -m coastal_sst_data.processes.extract --config ... --aoi one` works like every
other stage.

### 3a. `grid_from_cube(ds, g)` — reconcile config grid vs cube

Points are assigned with the grid the **config** computes; values are read from a cube some
earlier run wrote. If `grid.resolution_m` or `target_crs` changed since, every row/col is
silently wrong — every value finite, plausible, and from the wrong pixel. So:

- Re-derive the grid from the cube's own `y`/`x` coords (`dataclasses.replace` on the frozen
  `AoiGrid`, `transform = from_origin(xs[0] - res/2, ys[0] + res/2, res, res)`).
- Raise if `y` is not descending — every grid this package writes has a top-left origin, and
  the row arithmetic assumes it; an ascending axis flips every window north-for-south.
- Raise on any CRS / shape / resolution divergence, pointing at `assemble --aoi X --overwrite`.
- A cube with no `crs` attr: `log.warning` naming the CRS being assumed and telling the user
  to rebuild — never a silent assumption.
- Re-map the points through `station_pixels` against the cube's grid: a no-op when the two
  agree, and the cheapest way to guarantee one authority for the read.

### 3b. `plan_channels(ds, channels)` — presence and dimensionality

A listed channel missing from the cube is a **hard error** naming it, its close matches
(`get_close_matches`, already imported in `config.py`), and every channel the cube holds.
Skipping it would put a silently-absent variable into a modelling table, which reads
identically to one that was genuinely all-NaN.

| dims | representation | rules |
|---|---|---|
| `(time,y,x)` | one row per (point, time, stat) | full neighbourhood machinery |
| `(y,x)` static | **one** row per (point, stat), `time = NaT` | radius/mask/stats apply spatially; repeating a constant down the time axis multiplies the file by T for no information |
| `(time,)` | one row per (point, time), `stat = nearest`, `radius_m = 0.0` | AoI-wide: read **once** per AoI, tiled across its points |
| anything else | `ValueError` naming the channel and its dims | |

A 1-D channel configured with a radius, a mask, or a spatial stat fails loudly — it is one
value per day for the whole AoI, so there is nothing to reduce over.

### 3c. The neighbourhood — `window(g, row, col, px, py, radius_m)`

A **circle**, not a box: a box's corners reach `radius·√2`, so a "300 m neighbourhood" would
quietly be 424 m along the diagonal. Distances run from the point's **exact projected
position** (`px, py`, carried from assignment — not the containing pixel's centre) to each
pixel centre, both in metres in the cube's CRS. `radius_m` is never degrees, never pixels.

```python
rad_px = int(np.floor(radius_m / res))
r0, r1 = max(row - rad_px, 0), min(row + rad_px + 1, g.height)   # clipped at the edge
c0, c1 = max(col - rad_px, 0), min(col + rad_px + 1, g.width)
mask = (dy[:, None] ** 2 + dx[None, :] ** 2) <= radius_m ** 2
mask[row - r0, col - c0] = True          # the containing pixel, ALWAYS (unless masked out)
```

**The containing pixel is unconditionally in the set.** At the package's default 100 m
posting, a `radius_m: 50` circle contains no pixel centre at all for a large fraction of
point positions — without this clause the reduction is over an empty set and the whole
column is NaN, which reads exactly like a cloudy record. A one-time `log.warning` also fires
when `0 < radius_m < resolution_m`, saying the neighbourhood degenerates to one pixel rather
than quietly returning `nanmean == nearest`.

`radius_m: 0` ⇒ the single containing pixel. `count` exposes edge clipping and mask shrinkage.

### 3d. Memory — how the cube is actually read

The cube is chunked `(time: 64, y: 128, x: 128)`, so a naive per-point `.isel` re-reads a
multi-MB chunk for a 3×3 window — a 200-point × 8-channel run touches the same chunks
hundreds of times. Two strategies per (AoI, channel), and **which one ran is logged**
(identical output, wildly different cost):

- **Union read** (normal): points in one AoI are clustered, so read the single slab
  `y[min(r0):max(r1)], x[min(c0):max(c1)]` and slice it in NumPy. Each zarr chunk is touched
  once. xarray composes `.isel` into the zarr backend array, so only intersecting chunks are
  fetched — true without dask, which is not a dependency.
- **Per-point fallback**: when the union exceeds the budget (scattered points, large AoI),
  one `.isel` per point, with the time axis blocked so a long window is never materialised
  whole. Budget via `datacube.budget_bytes` — reuse, not a new detection path.

Static 2-D is the same code with `T = 1`; 1-D is one `da.values` per AoI, tiled.

### 3e. The table

```python
COLUMNS = ["point_id", "lat", "lon", "aoi", "time", "variable", "stat", "radius_m", "value"]
KEY     = ["point_id", "aoi", "time", "variable", "stat"]
```

`point_id`/`aoi`/`variable`/`stat` are `category` (dictionary-encoded in parquet — keeps a
multi-million-row frame small); `lat`/`lon` echo the **input** coordinates, not the pixel
centre; `time` is `datetime64[ns]` naive UTC, `NaT` for static channels; `value` is float64.

`radius_m` records **what the row actually used**, not what the config declared: a `nearest`
row is `0.0` even when its channel declares `radius_m: 300`. So `stat: [nearest, nanmean]`
yields two distinguishable rows and the column never asserts an unused footprint.

Final frame is sorted by `KEY` (stable, so two runs diff cleanly), then asserted unique on
`KEY` — one cheap check that catches an entire class of assignment/config bugs.

### 3f. `run()` semantics

1. Empty `channels`, or no points file from either config or `--points` → `SystemExit`
   naming the key to add. An extraction of nothing is never what was meant.
2. Skip guard: existing output + no `--overwrite` → `log.info`, `rep.skip()`, return.
3. `select_aois(grids, only_aoi)` — a typo'd `--aoi` raises before any cube is opened.
   Points assigned to a non-selected AoI are dropped with their own counted warning
   (a different message from "in no AoI" — different cause).
4. **Zero surviving points is a failure, not an empty file** (`insitu_csv`'s stated rule).
5. `--dry-run`: log the per-AoI point counts, the channel plan (dims / radius / stats /
   window size / clipped points) and the estimated row count and file size; write nothing.
6. Missing cube → `log.warning("run `assemble` first")` + `rep.skip()`, matching `preprocess.run`.
7. `store.write_table(...)`; `report.ProductReport("extract")` for the run summary, like
   every other stage. Above ~2 M rows, warn that `--format csv` will be hundreds of MB.

---

## 4. Tabular writer — `store.write_table`

Extends [store.py](src/coastal_sst_data/store.py) (which already imports pandas) next to
`write_output`, so this is not the one durable write in the package bypassing `atomic`:

```python
def write_table(df, out_dir: Path, stem: str, fmt: str = "parquet") -> Path:
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(f"parquet output needs pyarrow (optional): {exc}") from exc
        path = out_dir / f"{stem}.parquet"
        with atomic(path) as tmp:
            df.to_parquet(tmp, engine="pyarrow", index=False)
        return path
    ...
```

- No `placeholder=True` — that exists solely for libnetcdf's HDF5 probe. `to_parquet`/
  `to_csv` create the file themselves, so `atomic`'s emptiness check genuinely catches a
  writer that wrote nothing.
- The `ImportError` is raised, not swallowed; `cli._cmd_extract` turns it into a
  `SystemExit` naming the extra and `--format csv` — the same shape as the matplotlib path
  in `_cmd_grids`.
- `store.scan` only walks registry product dirs for `*.nc`, so the parquet is never
  mis-scanned; but `_find_scratch` walks the whole root, so a `.part-` left by a killed
  extraction *is* reported by `coastal-sst-data check --repair`. Worth a docs line.

`pyproject.toml`: `extract = ["pyarrow>=16"]`, added to the `all` bundle; `pyarrow` added to
the three `environment*.yml` optional blocks. Core install unchanged; `--format csv` needs
no new dependency.

---

## 5. CLI wiring

```
coastal-sst-data extract --config c.yaml [--points sites.csv] [--aoi NAME ...]
                         [--format parquet|csv] [--out PATH] [--overwrite] [--dry-run]
```

Follows the `_cmd_preprocess` pattern (`sub.add_parser` + `add_common(p)` +
`set_defaults(func=...)`), plus a line in the module docstring's subcommand list.

- `--format default=None` is load-bearing: `default="parquet"` would silently override a
  config that says `format: csv` on every invocation.
- Default output `<output_dir>/extract/<stem>.<ext>`; when `--aoi` is given and `--out` is
  not, the stem is suffixed with the selected AoIs (`points_hood_canal.parquet`). Otherwise
  `extract --aoi hood_canal` would overwrite a complete table with a one-AoI subset at the
  same path, with no error. The resolved path is logged either way.

---

## 6. Tests

New `tests/conftest.py` fixtures:

- `cube_dir` — a synthetic cube on the existing `aoi_grid`, written with
  `datacube.write_zarr_safe` + `build_encoding` (the helper `tests/test_datacube.py` already
  uses), with **anisotropic** ramps so a wrong index changes the answer instead of returning
  something plausible: `eco_sst (t,y,x) = 1000*t + row`, `eco_sst_gappy` with a known NaN
  block and one all-NaN pixel, `elevation_cudem (y,x) = col`, `landcover_water (y,x)` split
  east/west, `tide_coops (time,)`.
- `points_csv` — deliberately non-canonical column names (`latitude`/`longitude`/`station`),
  with positions derived by inverting `aoi_grid.xy_centers()` back to lon/lat so a test can
  name the expected `(row, col)` exactly.

`tests/test_points.py` — aliases resolve; ambiguous columns rejected; projected metres
rejected; swapped lat/lon rejected; synthesised ids warn; duplicate ids rejected;
containment; point outside every AoI dropped with a warning; overlapping AoIs tie-break on
nearest centre; equal-distance tie-break deterministic; edge points flagged.

`tests/test_extract.py` — one test per silent-failure mode:
`nearest` reads the pixel the point falls in (catches an off-by-one); north/south points
differ as y descends; the static channel varies along `x` (catches a row/col transposition);
`radius_m: 50` on a 100 m grid returns the containing pixel with `count == 1` and warns;
`radius_m: 250` at 100 m gives `count == 21` — **not** 25 (a square box) and not 1 (degrees
read as metres); an edge-clipped window reduces `count` without raising; `mean` is NaN where
`nanmean` is finite; `count_valid` is the finite count; `mask: water` excludes land and an
all-land neighbourhood gives NaN with `count == 0`; `nearest` never substitutes a finite
neighbour; multiple stats → separate rows; a `nearest` row records `radius_m == 0`; static →
one row with null time; 1-D identical for every point and rejected with a radius; a missing
channel fails with a suggestion; cube/config grid mismatch fails loudly; an ascending-y cube
is rejected; a missing `crs` attr warns; zero points inside any AoI is an error; the primary
key is unique; row order is deterministic; parquet round-trips (`importorskip`) and agrees
with csv; missing pyarrow gives clear advice; existing output skipped without `--overwrite`;
`--dry-run` writes nothing.

`tests/test_config.py` — the `extract` block validates; `extra="forbid"` inside a channel
(`{radiusm: 50}`) fails; `stat: mean` coerces to `["mean"]`; a bare `chan:` defaults;
unknown stat lists the choices; duplicate stats rejected; `radius_m` with only `nearest`
rejected; `p90` accepted, `p9O` rejected.

`tests/test_cli.py` — subcommand dispatches with the right kwargs; `--format` absent leaves
the config's format winning; missing pyarrow exits with advice.

`tests/test_extract_optional.py` — the optionality contract, one test per clause:
a config with **no** `extract:` block loads and `validate` mentions nothing;
`import coastal_sst_data` and a `run --dry-run` leave `coastal_sst_data.processes.extract`
absent from `sys.modules`; `points.py` imports with `pyarrow` blocked
(`monkeypatch.setitem(sys.modules, "pyarrow", None)`) and the csv path still writes;
`DataProduct` and `pipeline.PROCESS_MODULES` are unchanged by the feature.

All offline: a local CSV and a local zarr, no network seam. The cube golden snapshots are
untouched — this feature adds no cube channels.

## 7. Docs

- `README.md`: an `### extract` section in the CLI chapter, opening with a line saying the
  stage is entirely optional — no `extract:` block, no cost — (YAML block, output column table,
  the stat vocabulary with the `mean`/`nanmean` distinction, the circle-plus-containing-pixel
  definition with the 50 m/100 m worked example, how 2-D and 1-D channels appear, the `--aoi`
  filename suffix, the pyarrow extra), plus `extract` in the quick-start sequence.
- `examples/config.test.yaml`: a commented `extract:` block in the file's style, flagged as
  optional (the file exercises every block, so it carries one; nothing else needs to).
- `docs/DEVELOPMENT.md`: `points.py` and `processes/extract.py` in the architecture screen;
  `store.write_table` in the shared-helpers table.
- `docs/plan-point-extraction.md`: this plan, per the repo's `plan-<feature>.md` convention.

## 8. Risk table — how this silently produces wrong numbers, and what stops it

| # | Silent failure | Why it is silent | Guard |
|---|---|---|---|
| 1 | y-axis inverted — rows counted from the south | Every value finite and plausible, from the mirrored latitude | `station_pixels` is the *only* affine inversion (shared with the cube's own in-situ channel); `grid_from_cube` rejects an ascending `y`; the fixture ramps along `row` |
| 2 | Row/col transposed | Invisible on a near-square AoI | The 2-D fixture ramps along `col` while the 3-D ramps along `row` |
| 3 | Off-by-one (pixel edge vs centre) | Wrong by one 100 m pixel — looks like real spatial variation | Expected `(row, col)` derived by inverting `AoiGrid.xy_centers()`, so the test asserts against the grid's own definition |
| 4 | `radius_m` read as degrees | 50 "degrees" clips to the grid and returns a plausible AoI-wide mean | Distances computed in the projected CRS; `count == 21` test |
| 5 | Radius smaller than a pixel → empty set → all NaN (**your `radius_m: 50` example on the default 100 m grid**) | A NaN column reads like a channel with no data | Containing pixel unconditionally included; one-time warning; dedicated test |
| 6 | Square box instead of a circle | A "300 m" neighbourhood quietly reaching 424 m | True-distance circular mask; the test asserts 21, not 25 |
| 7 | Window clipped by the grid edge → half-disc mean | The mean is finite and in range | Edge proximity warned at assignment; `count`/`count_valid` in the vocabulary, recommended in the docs beside any radius |
| 8 | Point outside every AoI | An empty file looks like a mistyped column or an empty region | Counted warning naming the ids; zero surviving points is a hard error |
| 9 | Point in two AoIs → duplicated rows | Doubles the row count, fans out on any join | One AoI per point by construction + the `KEY` uniqueness assertion |
| 10 | Config grid changed since the cube was built | Every read from the wrong pixel of a valid cube | `grid_from_cube` raises on any CRS/shape/resolution divergence |
| 11 | Missing `crs` attr on an older cube | A silent assumption nobody recorded | Explicit warning naming the assumed CRS |
| 12 | Projected metres or swapped lat/lon in the CSV | 512345.0 as a longitude lands in the Pacific → empty file (see #8) | WGS84 range check naming both causes; never auto-swapped |
| 13 | Listed channel missing from the cube | Reads as "that variable was all-NaN" | Hard error with close matches and the cube's real channel list |
| 14 | `nearest` substituting a finite neighbour | A cloudy pixel filled from just offshore — a warm bias | `nearest` is the containing pixel, full stop; dedicated test |
| 15 | Point snapped to the nearest water pixel | The point moves hundreds of metres with nothing recording it | `water=None` passed explicitly, with the reason in the docstring |
| 16 | Mask empties the neighbourhood and the code falls back to unmasked | A land-contaminated mean that looks fine | Explicit NaN + `count == 0`; dedicated test |
| 17 | `--format parquet` default overriding `format: csv` in the config | Silently ignores a config field | `--format default=None`, resolved in `_cmd_extract` |
| 18 | `--aoi` subset overwriting the full table at the same path | A complete table replaced by a one-AoI one | AoI-suffixed default stem; resolved path logged |
| 19 | Killed write leaving a truncated parquet | `exists()` takes it for done on the next skip guard | `store.atomic`; leftovers surfaced by `coastal-sst-data check` |
| 20 | 1-D channel given a radius/stat | An AoI-wide constant presented as a neighbourhood statistic | Hard error naming the channel and its dims |

## 9. What shipped

Built in the order above (config models -> `points.py` -> `store.write_table` -> the
`cube_dir`/`points_csv` fixtures -> `processes/extract.py` -> CLI -> docs). Two details
landed differently from the sketch above:

* The stat vocabulary also carries `sum`/`nansum`, for the same reason as the rest of the
  pairs.
* The optionality contract's "works without pyarrow" clauses are tested with a
  `sys.meta_path` finder that makes `import pyarrow` fail, not by deleting it from
  `sys.modules` -- pandas imports pyarrow itself wherever it is installed, so by the time a
  test runs the module is already loaded and removing it would prove nothing about a fresh
  install. See `_BLOCK_PYARROW` in `tests/test_extract_optional.py`.

Verification:

```bash
pytest tests/test_points.py tests/test_extract.py tests/test_extract_optional.py -q
pytest -q                                        # nothing else regresses

# End to end against a real cube (<output_dir>/datacube/<aoi>.zarr):
printf 'id,lat,lon\nsite_a,45.52,-123.925\n' > sites.csv
coastal-sst-data extract --config examples/config.test.yaml --points sites.csv --dry-run
coastal-sst-data extract --config examples/config.test.yaml --points sites.csv --format csv
```

Spot-check: for one channel and one date, the `nearest` row must equal
`xr.open_zarr(cube)[ch].isel(time=t, y=row, x=col)` for the `(row, col)` the log reports.
