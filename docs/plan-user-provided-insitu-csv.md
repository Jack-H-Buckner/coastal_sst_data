# Plan: moving in-situ platforms (drifters, gliders, transects)

> **Status.** The CSV loading half of issue #47 **SHIPPED in 0.2.0**: `insitu` is a stacked-DATA
> product, `insitu_csv` reads long-format files, and `insitu_acquire` fans out over the sources.
>
> **§1 is SUPERSEDED.** Moving platforms shipped in 2026-08 — but as a SEPARATE PRODUCT
> (`insitu_mobile`), not as the shared data-model change §1 designs. See
> [plan-insitu-mobile-platforms.md](plan-insitu-mobile-platforms.md) for what was actually built
> and why the shape changed; §1 is kept below because its analysis of what breaks is accurate and
> its two new primitives (`observation_pixels`, `nearest_index`) were implemented as written.
>
> The short version: §1 makes `lat`/`lon` `(station, time)` everywhere and `insitu_station`
> `(time, y, x)`. That is a real breaking change — `mask: insitu_station` in `extract` requires a
> static channel — and it carries a silent hazard §1 did not see: `append_zarr(mode="a-")` leaves
> static `(y,x)` channels alone after the first block, so a time-varying station map written as
> `(y,x)` would freeze at block 0 with no error at all. Two products sidestep both, and the fixed
> path did not change by a byte.
>
> Sections 2–7 are DONE and kept as the record of why the stacking is shaped the way it is.

Issue #47 — "add a method to load user-provided in-situ data into the data cubes".

## Context

Today `insitu` has exactly one implementation, `insitu_ioos.py`, and it is a
`SourceKind.ACCESS` product: the config picks *one* source, the module writes one
`INSITU/aligned/<aoi>/<aoi>_insitu.nc`, and `datacube._contribute_insitu` reads that single
file. A user with their own moorings, hand-loggers, or drifters has no way in.

Three decisions frame the work:

1. **Stack, don't replace.** A cube must be able to carry auto-discovered IOOS buoys **and**
   the user's own platforms at once. So `insitu` becomes a `SourceKind.DATA` (stacked)
   product, like `tides`, `met` and `bathymetry`.
2. **Configurable column mapping.** Required fields have conventional default names but each
   is remappable, so users don't rewrite their files.
3. **Positions are per-observation, not per-platform.** The data may include **Lagrangian
   drifters**: a platform that measures temperature as it moves. Each observation must be
   placed in the pixel it was actually taken in — never a track-average position.

Decision 3 is the largest piece of work and it is **not** confined to the CSV loader. It
changes the shared in-situ data model that IOOS also writes into, and it changes the cube
schema. Sections 1–2 below cover it; sections 3–6 cover the stacking and the loader.

---

## 1. The moving-platform data model

### What breaks today

| Where | Assumption |
|---|---|
| [insitu_ioos.py:301-302](../src/coastal_sst_data/processes/insitu_ioos.py#L301-L302) | one `lat`/`lon` per station, taken as the **median** of the series |
| [insitu_ioos.py:222-233](../src/coastal_sst_data/processes/insitu_ioos.py#L222-L233) | `lat`/`lon` written as `(station,)` coords |
| [insitu.station_pixels](../src/coastal_sst_data/processes/insitu.py#L43) | one pixel per station, computed once |
| [datacube.build_insitu](../src/coastal_sst_data/processes/datacube.py#L352-L375) | station `s` writes to a fixed `(r, c)` on every day |
| `insitu_station` channel | a **static** `(y, x)` index map |

A drifter run through this lands its whole track in the median pixel — silently, and the
resulting matchups would be wrong rather than absent, which is the worst failure mode this
package has.

### The new on-disk contract

`<aoi>_insitu.nc`, dims `(station, time)` — `station` now means **platform** (a fixed
mooring is the degenerate case of a track that doesn't move):

| name | dims | change |
|---|---|---|
| `sst` | `(station, time)` | unchanged |
| `qc` | `(station, time)` | unchanged |
| `lat` | `(station, time)` | **was `(station,)`** |
| `lon` | `(station, time)` | **was `(station,)`** |
| `station_id`, `station_name`, `variable` | `(station,)` | unchanged |
| `platform_type` | `(station,)` | **new** — `fixed` \| `mobile`, derived from whether the position varies |

A fixed mooring writes the same value down the time axis — no special case, no branch in the
consumer. **Readers must accept 1-D `lat`/`lon`** and broadcast them along `time`, so files
written by 0.1.x still load.

### New pure functions in `processes/insitu.py`

This module is the source-agnostic half and stays pure/testable. Two additions:

```python
def observation_pixels(lons, lats, g):
    """(S,T) or flat positions -> (rows, cols, inside), vectorized.

    One pyproj transform + one affine inversion for EVERY observation, done once.
    `station_pixels` becomes a thin wrapper over this so its snapping behaviour and
    its tests are untouched."""

def nearest_index(times, values, target, max_dt_min=DEFAULT_MAX_DT_MIN):
    """(k, signed_dt_minutes) for the nearest FINITE observation, or (None, nan).

    The index, not just the value: the caller needs observation k's POSITION.
    `value_at` becomes a wrapper returning (values[k], dt) so existing callers and
    tests are unaffected."""
```

`nearest_index` should use `np.searchsorted` on the (sorted) time axis rather than the
current full `argmin` — a drifter series is long, and the current version is `O(T)` per
(station × day × target).

### `datacube.build_insitu` rewrite

```
precompute rows/cols/inside for EVERY observation          # one vectorized call
for each platform s:
    for each target key, for each day i:
        k, dt = nearest_index(times_s, sst_s, targets[key][i], max_dt_min)
        if k is None or not inside[s, k]: continue
        r, c = rows[s, k], cols[s, k]                      # <- THE observation's pixel
        sums[key][i, r, c] += sst[s, k];  counts[key][i, r, c] += 1
        dt_sums[key][i, r, c] += dt
        station_idx[key][i, r, c] = table_index(s)
emit sums/counts as the mean
```

Two fixes fall out of this rewrite and should ship with it:

- **Co-located platforms are currently mis-averaged.** `chans[k][i,r,c] = (prev + v) / 2.0`
  ([datacube.py:372](../src/coastal_sst_data/processes/datacube.py#L372)) is not a mean for
  three or more contributors — it yields `a/4 + b/4 + c/2`. Drifters make co-location common,
  so switch to sum/count. `insitu_dt_min` has the same defect.
- **Out-of-grid is now per-observation.** A track may leave and re-enter the AoI. Drop the
  *observation*, keep the platform; warn once per platform only if it is *never* inside
  (preserving the existing loud-drop behaviour at
  [datacube.py:355](../src/coastal_sst_data/processes/datacube.py#L355)).

### The `insitu_station` channel — a breaking change

A drifter occupies different pixels on different days, and a *different* pixel again at each
sensor's overpass instant. A static `(y, x)` map cannot express that. Proposal:

- `insitu_station` becomes `(time, y, x)` — the platform sampled at the **reference time**.
- New `<sensor>_insitu_station` `(time, y, x)` per sensor, matching the existing
  `<sensor>_insitu_sst` / `<sensor>_insitu_dt_min` family.

These are `uint16` and almost entirely zero, so they compress to near-nothing in Zarr.

> **Alternative if breakage is unacceptable:** keep `insitu_station` at `(y, x)` when every
> platform is fixed and switch to `(time, y, x)` only when a mobile one is present. I'd
> advise against it — a channel whose dimensionality depends on the input data breaks
> downstream code unpredictably, which is worse than breaking it once, loudly, at a version
> bump.

---

## 2. `insitu` becomes a stacked-DATA product

Mirrors `tides` exactly, which is the closest precedent (one module fanning out over a
region-overridable `sources` list, one output tree per source).

```python
ProductSpec(
    product=DataProduct.insitu,
    dir="INSITU",
    kind=Kind.STATION_TABLE,
    sources={"ioos": "coastal_sst_data.processes.insitu_acquire",
             "csv":  "coastal_sst_data.processes.insitu_acquire"},
    source_kind=SourceKind.DATA,          # was ACCESS
    # default_source must be dropped -- the registry invariant forbids one on a DATA product
    auth={"ioos": None, "csv": None},
    options=_COMMON | {
        "sources", "variables", "stations", "exclude_stations", "qc_flags", "pad_deg",
        "max_sensor_depth_m",
        "path", "columns", "time_zone", "units", "qc_pass_values", "default_station_id"},
    region_options=frozenset({"sources", "stations", "exclude_stations", "variables", "path"}),
    required_vars=("sst", "qc"),
    provenance_inputs=("insitu",),
)
```

Both sources map to **one** module because
[`_check_registry`](../src/coastal_sst_data/products.py#L560) requires it of a DATA product
(and `pipeline._module_for` dispatches `spec.one_module()` once, letting it fan out
internally). Hence the new `insitu_acquire.py` dispatcher in §3.

Notes on the config surface:

- `source` is **removed** from `options`. Leaving it accepted-but-ignored is exactly the
  "config that lies" failure [config.py:185-207](../src/coastal_sst_data/config.py#L185-L207)
  exists to prevent. Add a targeted validator message: *"insitu.source is now
  insitu.sources: [ioos] — in-situ sources stack."*
- `path` **is** region-overridable: which file holds this region's data is inherently local,
  the "which source has coverage here" side of the documented line.
- `columns` / `units` / `qc_pass_values` are **not** region-overridable: they decide what the
  channel *means*, and a region that changed them would make two AoIs' cubes silently
  non-comparable.

Output moves from `INSITU/aligned/<aoi>/` to `INSITU/<source>/aligned/<aoi>/`.
[`provenance.collect`](../src/coastal_sst_data/provenance.py#L305-L314) already globs
`<DIR>/*/aligned` for stacked products, so per-source in-situ provenance comes for free.

---

## 3. `processes/insitu_acquire.py` — the fan-out module (new)

Owns everything that is not source-specific, lifted out of `insitu_ioos.run`:

```python
SOURCES = {"ioos": insitu_ioos.fetch_aoi, "csv": insitu_csv.fetch_aoi}

def run(eff, grids, only_aoi, dry_run, only_source=None):
    for name in select_aois(grids, only_aoi):
        for src in eff["ds"][name]["sources"]:
            out_path = root / src / "aligned" / name / f"{name}_insitu.nc"
            if store.done(out_path, store.REQUIRED_VARS["INSITU"],
                          covers=(start, end), overwrite=overwrite):   # range-aware: DEVELOPMENT.md §3b
                rep.skip(); continue
            records = SOURCES[src](g, start, end, eff["ds"][name], dry_run=dry_run)
            ...
            ds = build_dataset(records)      # (station, time) with (station,time) lat/lon
            ds.attrs.update(aoi_id=name, source=src, **provenance.requested_range(start, end),
                            **provenance.stamp(eff))
            write_output(...); rep.wrote(source=src)
```

The **source seam** — every source returns a list of records:

```python
{"id": str, "title": str, "var": str, "source": str,
 "df": DataFrame[time, latitude, longitude, value, (qc)]}   # positions PER ROW
```

`build_dataset` no longer takes a median position; it reindexes each platform's `latitude` /
`longitude` onto the union time axis alongside `value`, and derives `platform_type` from
whether the position varies.

Validate `sources` against `SOURCES` in `acquire()` before any work, so a typo fails loudly —
copy [tides.acquire:435-439](../src/coastal_sst_data/processes/tides.py#L435-L439).

Keep `entry.process_main` here with a `--source` narrowing, matching `tides`.

## 4. `processes/insitu_ioos.py` — refactor

Keep `find_stations`, `station_variables`, `pick_variable`, `fetch_station` and the `_get`
network seam (the tests monkeypatch `_get`, so they keep working). Replace `run` / `_build_eff`
/ `acquire` / `main` with one `fetch_aoi(g, start, end, cfg, dry_run=False) -> list[record]`
that does the discovery + allow/deny + empty-station-drop logging currently in `run`. Stop
taking the median position — pass `latitude`/`longitude` through per row (a CO-OPS gauge just
reports the same pair every time).

## 5. `processes/insitu_csv.py` — the loader (new)

Zero network, pure parsing, so it is entirely unit-testable.

```yaml
products:
  insitu:
    sources: [ioos, csv]        # STACK the public network with your own data
    qc_flags: [1, 2]            # applies to the IOOS QARTOD flags
    max_sensor_depth_m: 5
    # --- csv source ---
    path: ~/data/moorings/*.csv     # file, directory, or glob; region-overridable
    columns:                        # only the ones you need to remap
      station_id: platform
      time: datetime_utc
      latitude: lat
      longitude: lon
      value: temp_c
      # optional and auto-detected when present: z, qc, station_name
    time_zone: UTC              # naive timestamps are assumed to be this
    units: degC                 # degC | degF | K -> converted to degC
    qc_pass_values: [1, 2]      # values of the qc column that pass; omit to keep every row
    default_station_id: null    # id for a single-platform file with no station_id column
```

Behaviour:

- **Resolve `path`** to a file, a directory (`*.csv` within it), or a glob; concatenate.
  Zero matching files is a **failure**, reported through `rep.fail`, never a silent empty
  channel — the same posture `insitu_ioos` takes for a station that reports nothing.
- **Required columns** after mapping: `time`, `latitude`, `longitude`, `value`. Missing one
  raises naming the file, the missing key, and the columns actually present.
- **`station_id`** optional: absent → `default_station_id`, or the file stem.
- **Time**: parse to UTC then drop tz (matching `insitu_ioos`, which stores naive UTC).
  Unparseable rows are dropped **with a count logged**.
- **Units**: convert `degF`/`K` to `degC`. Add a sanity gate — values outside, say, −5…45 °C
  after conversion warn loudly with a count, because a `units` mistake is silent otherwise.
- **`z` / `qc`**: reuse `max_sensor_depth_m`; filter on `qc_pass_values` when configured.
- **AoI filter**: keep platforms with **at least one** observation inside the AoI grid; one
  CSV can serve every AoI in the project.
- **`dry_run`**: parse and report platform/row counts, write nothing.

## 6. `datacube._contribute_insitu` — read the stack

```python
base = ctx.eff["aligned_root"] / PRODUCT_DIRS["insitu"]
for src in sorted(d.name for d in base.iterdir() if d.is_dir()):
    f = ctx.adir("insitu", src) / f"{ctx.aid}_insitu.nc"
    if not f.exists():
        continue                       # not a source tree -- see the guard note below
    datasets.append((src, xr.open_dataset(f)))
```

> **Guard against the empty-version-tag bug.** [`docs/bug-empty-version-tag-channels.md`](bug-empty-version-tag-channels.md)
> documents exactly this pattern going wrong in `_contribute_stacked_sensor`: every
> subdirectory was taken as a source tag, so a stray leftover directory produced silent
> all-sentinel channels. Requiring the expected file to exist (above) is the minimum; also
> log every directory skipped, so a mis-named tree is visible rather than absorbed.

Also read a **legacy flat** `INSITU/aligned/<aoi>/` if present, tag it `ioos`, and log a
deprecation pointing at the migration — so a 0.1.x output tree doesn't silently lose its
in-situ channels.

**Pass the datasets to `build_insitu` as a list, not a concatenation.** An
`xr.concat(..., dim="station", join="outer")` across sources would union two unrelated time
axes into one dense `(S, T)` NaN block — 6-minute CO-OPS data unioned with a 1-minute logger
is a needless order of magnitude. `build_insitu` iterating a list accumulates into the shared
`(T, H, W)` channel arrays with no such product.

Station ids are namespaced `<source>:<id>` in the station table (which gains a `source`
field), so two networks that both name a platform `01` don't collide.

## 7. Provenance

- `_EXACT` gains `insitu_platform`-family entries as needed;
  `insitu_sst` / `insitu_n` / `insitu_station` keep their current mappings.
- `<sensor>_insitu_station` is already covered by the `rest.startswith("insitu")` branch at
  [provenance.py:228](../src/coastal_sst_data/provenance.py#L228) — verify with a test rather
  than assuming, since an unmapped channel only *logs*.

---

## Files to modify

| File | Change |
|---|---|
| `src/coastal_sst_data/processes/insitu.py` | + `observation_pixels`, `nearest_index`; existing fns become wrappers |
| `src/coastal_sst_data/processes/insitu_acquire.py` | **new** — per-(AoI, source) loop, `build_dataset`, write, report, `acquire`/`main` |
| `src/coastal_sst_data/processes/insitu_csv.py` | **new** — long-format CSV → records |
| `src/coastal_sst_data/processes/insitu_ioos.py` | refactor to `fetch_aoi`; per-row positions; drop its own run/acquire |
| `src/coastal_sst_data/products.py` | insitu spec → `SourceKind.DATA`, new options, drop `default_source`/`source` |
| `src/coastal_sst_data/config.py` | migration error for `insitu.source` |
| `src/coastal_sst_data/processes/datacube.py` | `load_insitu` multi-source + legacy path; `build_insitu` per-observation placement, mean fix, per-target station maps |
| `src/coastal_sst_data/provenance.py` | verify/extend channel mapping |
| `examples/config.test.yaml` | `sources:` + a commented csv block |
| `docs/DEVELOPMENT.md` | insitu joins the stacked-DATA list; note in-situ merges to ONE channel set (see below) |
| `pyproject.toml` | 0.1.1 → **0.2.0** (breaking) |

**One documented deviation from the DATA convention.** `SourceKind.DATA` is described as
"one channel per source" (`depth_cudem`, `depth_gmrt`). In-situ deliberately merges to **one**
channel set: stations are *rows*, not channels, they occupy disjoint pixels anyway, and
per-source channels would multiply the whole `<sensor>_insitu_*` family by the source count
while making `insitu_sst` stop meaning "ground truth". The source is recorded per platform in
the station table instead. This deviation needs a comment in the spec and a line in
`DEVELOPMENT.md`, or the next person will read it as a bug.

## Breaking changes & migration

1. `insitu.source: ioos` → `insitu.sources: [ioos]` (loud validation error with the fix).
2. `INSITU/aligned/<aoi>/` → `INSITU/ioos/aligned/<aoi>/`. The legacy path is still *read* by
   the assembler, so no re-fetch is forced; moving the directory is a one-liner.
3. `insitu_station` becomes `(time, y, x)`.
4. `lat`/`lon` in `<aoi>_insitu.nc` become `(station, time)`. Old files still read.

## Risks

- **Dense `(station, time)` blow-up.** Many platforms on irregular time axes make the union
  axis large. Keeping sources in separate files (above) removes the cross-source case; within
  a source, add a loud warning when `S × T` exceeds a threshold and note a ragged layout as
  future work. A drifter fleet is where this bites first.
- **Matchup cost.** Per-observation placement multiplies the inner loop by the target count.
  `searchsorted` + a single vectorized reprojection keeps it flat; benchmark once on a real
  drifter file.
- **A `units` mistake is silent** — °F read as °C is a plausible number. Hence the range gate.
- **Refactoring `insitu_ioos` risks its two silent-failure guards** (HTTP 400 on an
  unadvertised variable; a station that advertises and never reports). `tests/test_insitu.py`
  covers both — keep them green throughout rather than at the end.

## Verification

- `tests/test_insitu_csv.py` **(new)** — column mapping; missing-required-column error names the
  file and column; unit conversion; naive-vs-aware timestamps; `qc_pass_values`; glob/dir/file
  resolution; zero-files fails rather than returning empty; a platform entirely outside the AoI
  is dropped.
- `tests/test_insitu.py` **(extend)** — **a drifter crossing a pixel boundary lands in two
  different pixels on two days** (the acceptance test for decision 3); three co-located
  observations average correctly; legacy 1-D `lat`/`lon` still load; a stray directory under
  `INSITU/` is ignored with a log; `ioos` + `csv` stack into one station table with namespaced
  ids; per-target station maps disagree for a mobile platform and agree for a fixed one.
- `tests/test_config.py` — `insitu.source` rejected with the migration message; a region
  overriding `columns` rejected; a region overriding `path` accepted.
- `tests/test_provenance.py` — every new channel resolves to a non-empty input list.
- `pytest` is offline throughout; the CSV source needs no stub at all.

## Suggested order

1. `insitu.py` primitives + their tests (pure, no dependencies).
2. `build_insitu` per-observation placement + mean fix, still single-source. **The cube is
   correct for drifters at this point**, before any stacking work.
3. Stacking: registry flip, `insitu_acquire`, `insitu_ioos` refactor, contributor multi-source.
4. `insitu_csv` + config surface.
5. Docs, example config, version bump.

Steps 1–2 are independently shippable and carry most of the risk; step 4 is the easy part.
