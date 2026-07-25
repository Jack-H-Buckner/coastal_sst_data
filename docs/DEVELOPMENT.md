# Developer & contributor guide

This document is for people **extending** `coastal_sst_data` — adding a data product, adding a
new source to an existing product, or changing how the datacube is assembled. If you only want
to *use* the package (write a config, run it, read a cube), the [README](../README.md) is what you
want; this file assumes you've read it.

The package's design intent is that most of what you need is already explained, in depth, in the
module docstrings — `products.py`, `naming.py`, `store.py`, and `datacube.py` in particular each
open with a "why it is built this way / what silently breaks if you get it wrong" essay. This
guide's job is to **assemble those scattered explanations into one path a new contributor can
follow**, and to make explicit a few contracts that today are only learned by reading an existing
module.

- [1. Architecture in one screen](#1-architecture-in-one-screen)
- [2. The extension surface — what "adding X" actually costs](#2-the-extension-surface--what-adding-x-actually-costs)
- [3. Adding a data product](#3-adding-a-data-product)
  - [3a. The `ProductSpec` declaration](#3a-the-productspec-declaration)
  - [3b. The `acquire()` contract](#3b-the-acquire-contract)
  - [3c. Wiring a non-sensor product into the datacube](#3c-wiring-a-non-sensor-product-into-the-datacube)
  - [3d. Teaching provenance about new channels](#3d-teaching-provenance-about-new-channels)
  - [3e. Write a test](#3e-write-a-test)
- [4. Adding a source to an existing product](#4-adding-a-source-to-an-existing-product)
- [5. Adding a derived stage](#5-adding-a-derived-stage)
- [5b. Adding a post-assembly preprocess step](#5b-adding-a-post-assembly-preprocess-step)
- [6. The shared helpers you must use](#6-the-shared-helpers-you-must-use)
- [7. Developing and debugging one process at a time](#7-developing-and-debugging-one-process-at-a-time)

---

## 1. Architecture in one screen

The wireframe at [`wireframes/main_wire_frame_claude.drawio`](../wireframes/main_wire_frame_claude.drawio)
is the picture; the pipeline is:

```
config (YAML)
   │  parse + validate, derive option/auth tables from the registry
   ▼
products.py  ── the REGISTRY: one ProductSpec per product ──┐
   │                                                         │ every other table
config.py / auth.py / grid.py / store.py / provenance.py ◄──┘ is DERIVED from it
   │
pipeline.py  ── compute one shared grid per AoI (once), topo-sort products by
   │            depends_on, dispatch each to its module via acquire()
   ▼
processes/<product>.py  ── acquire() → aligned files on the shared grid
   │            <output_dir>/<DIR>/aligned/<aoi>/<stem>.nc
   ▼
datum.py (derived)  ── DEM→MSL offset, from the bathymetry that actually ran
   ▼
datacube.py  ── knit every aligned output into ONE Zarr cube per AoI,
                on a common daily axis, self-describing (embedded provenance)
```

**The one design decision to internalise before touching anything:** a data product is *declared
once* in [`products.py`](../src/coastal_sst_data/products.py), and every other registry (config
options, auth requirements, dispatch table, run order, the skip guard's `required_vars`, the
assembler's directory map, provenance's sensor list) is a **view derived from that declaration**.
The module docstring at [products.py:1](../src/coastal_sst_data/products.py) lists what those
tables used to be — a dozen hand-maintained lists across five files — and why every one of them
"fails differently, and silently" when forgotten. That history is the reason the registry exists.

**Why `products.py` imports nothing from the package** (and names modules as dotted strings): it
must sit *below* `config` in the import order, because `config` needs the option/auth tables at
validation time and every process module imports `config`. Registering a spec from inside a
process module would close an import cycle. See the ladder in
[products.py:22-42](../src/coastal_sst_data/products.py#L22-L42).

---

## 2. The extension surface — what "adding X" actually costs

This is the single most important table in this document, because it says exactly how much each
kind of extension costs — and, crucially, what happens if you forget a step.

| To add a… | Registry-only? | You also write |
|---|---|---|
| **Thermal sensor** (`Kind.OVERPASS_SENSOR`) | ✅ **Yes** | Nothing else. The assembler's `sensors` contributor loops over `products.sensors()`, so every `<prefix>_sst/_cloud/_valid/_hour`, overpass-met, and in-situ-matchup channel is generated from the spec. This is the "acid test" proven by [`test_add_a_sensor.py`](../tests/test_add_a_sensor.py). |
| **A new source** for an existing product (e.g. `landsat_aws`) | ✅ ~Yes | A process module matching the **existing product's output contract** (§4). The registry already reserves `None` placeholders for these. |
| **A new daily / static / series covariate** (a *non-sensor* product) | ❌ **No** | A `ProductSpec` **+ a process module (§3b) + a registered `Contributor` in `datacube.CONTRIBUTORS` (§3c) + provenance rules if it produces derived channels (§3d)**. |

> ✅ **The trap is now closed by a loud invariant.** A covariate still needs a contributor to
> reach the cube — but forgetting it is no longer *silent*. Every product flows through one uniform
> `(ctx) -> channels` **contributor protocol** in
> [`datacube.assemble_aoi`](../src/coastal_sst_data/processes/datacube.py), and
> [`_check_contributors`](../src/coastal_sst_data/processes/datacube.py) runs **at import**: a
> non-sensor `ProductSpec` with no registered contributor (and no `cube_opt_out=True`) is a hard
> `RuntimeError` before a single cube is built. So a new covariate that you acquire to disk but
> forget to wire will crash the process with a message naming the product, rather than being
> acquired and then silently omitted from every cube. This is the acid test proven by
> [`test_add_a_covariate.py`](../tests/test_add_a_covariate.py).

A covariate is not a *uniform family* the way sensors are — CMEMS emits per-depth channels
discovered from the files, met emits a fixed set and writes the reference-time slot, bathymetry
publishes elevation/depth. So each still writes its own `_contribute_*` function; the protocol does
not try to make them one loop. What it *does* give you is a uniform mechanism, a derived run order
(topological sort over declared slot reads/writes — no hand-kept sequence), and the loud failure
above, so a covariate can never again be half-wired without anyone noticing.

---

## 3. Adding a data product

Worked example throughout: suppose you're adding **chlorophyll** (`chl`), a daily-raster ocean-colour
covariate from some Earthdata collection.

### 3a. The `ProductSpec` declaration

Add an enum member to [`DataProduct`](../src/coastal_sst_data/products.py#L50) and a `ProductSpec`
to [`REGISTRY`](../src/coastal_sst_data/products.py#L197) — **both in `products.py`, the same
file, nothing else for this step.** Fields, with the ones people forget called out:

| Field | What it is | Gotcha |
|---|---|---|
| `product` | the `DataProduct` enum member | must match the config key exactly |
| `dir` | the ALLCAPS output folder (`CHL`) | held explicitly, *not* inferred from the name — `tides` writes to `TIDE`, which is [why two alias tables once existed](../src/coastal_sst_data/products.py#L14). Must be unique. |
| `kind` | one of the [`Kind`](../src/coastal_sst_data/products.py#L68) values | describes the output's **shape**, which decides how the assembler reads it (see below). |
| `module` **or** `sources` | exactly one | `module` for a single implementation; `sources={name: module\|None}` for source-selectable. Dotted strings, resolved lazily. Setting both, or neither, fails the import-time check. |
| `options` | the option keys the module **reads** | a key not listed here is **rejected** by config validation — an accepted-but-unread option is a config that lies. Start from `_COMMON` (`output_format`, `overwrite`). |
| `region_options` / `region_only_options` | which options a region may override | the line is enforced: *"which source has coverage here" → yes; "what the cube channel means" → no.* Every region option must also be in `options` (or `region_only_options`). |
| `auth` | backend name, `None`, or `{source: backend}` | must match `sources` keys exactly when it's a dict. |
| `required_vars` | variables a **finished** file must carry | the skip guard uses this to tell a truncated write from a done one. Empty means "channel set is config-dependent" (met, CMEMS) → the check falls back to "opens and holds ≥1 data var". |
| `depends_on` | products that must run first | declared as a dependency, **not** a position in a hand-kept order. The topo-sort in [`pipeline.process_order`](../src/coastal_sst_data/pipeline.py#L72) places you. |
| `sensor` | a `SensorSpec`, or `None` | set **only** for `OVERPASS_SENSOR` thermal products. |
| `coverage_channel` | the cube channel that proves this product produced data on a day | **daily products only** — an overpass sensor with no scene on a day is normal, so coverage-checking it would train the user to ignore the warning. |
| `provenance_inputs` | the product(s) a field from this one is attributed to | usually just itself. |

The `Kind` you choose determines the filename stamp and which assembler reader services it:

| `Kind` | Filename | Assembler reader |
|---|---|---|
| `DAILY_RASTER` | `<aoi>_YYYYMMDD.nc` | `load_daily_sensor` |
| `OVERPASS_SENSOR` | `<aoi>_YYYYMMDDThhmmss.nc` | `load_clearest_overpass` (registry loop — no wiring) |
| `STATIC_RASTER` | `<aoi>.nc` (no time dim) | a product-specific `load_*` (e.g. `load_bathy`, `load_landcover`) |
| `SERIES_1D` | `<aoi>_tides.nc`, dims `(time,)` | `load_tide_daily` |
| `STATION_TABLE` | `<aoi>_insitu.nc`, dims `(station, time)` | `load_insitu` + `build_insitu` |

For chlorophyll:

```python
# in DataProduct
chl = "chl"

# in REGISTRY
ProductSpec(
    product=DataProduct.chl,
    dir="CHL",
    kind=Kind.DAILY_RASTER,
    module="coastal_sst_data.processes.chl",
    auth="earthdata",
    options=_COMMON | {"short_name", "variable", "pad_deg"},
    required_vars=("chl", "valid"),
    coverage_channel="chl",          # a daily product, so coverage IS checkable
    provenance_inputs=("chl",),
),
```

The import-time invariant checker [`_check_registry`](../src/coastal_sst_data/products.py#L413)
runs on every import and will refuse a spec that sets both/neither of `module`/`sources`, has a
mismatched auth map, declares a region option nothing reads, reuses a `dir` or a sensor `prefix`,
or names an unknown `depends_on`. Read its messages — they name the exact fix.

### 3b. The `acquire()` contract

Every process module exposes **one entry point** that the pipeline calls:

```python
def acquire(project, *, grids=None, aois=None, dry_run=False, overwrite=False) -> ProductReport | None
```

- `project` — the validated [`Project`](../src/coastal_sst_data/config.py).
- `grids` — the pre-computed `{aoi_name: AoiGrid}`, shared across all products (compute it yourself
  via `project_grids(project)` only if `None`, so a standalone run still works).
- `aois` — restrict to these AoI names; `None` means all.
- `dry_run` — search/plan only, **write nothing**.
- `overwrite` — reprocess even where a finished output exists.
- **Returns** a [`ProductReport`](../src/coastal_sst_data/report.py#L53) (or `None` if there was
  genuinely nothing to report). **This return value is load-bearing** — see the accounting rule below.

**[`mur.py`](../src/coastal_sst_data/processes/mur.py) is the reference implementation.** It is the
smallest module that exercises the whole contract; copy its structure. The conventions it follows,
which your module must too:

1. **The `_build_eff` → `run` split.** `acquire` calls `_build_eff(project)` to flatten the
   validated Pydantic `Project` into a plain `eff` dict, then hands that to `run(eff, grids, …)`.
   This adapter boundary is a package-wide convention: `run` works in plain dicts so it is trivial
   to test and doesn't reach back into config internals. Resolve per-AoI options inside `_build_eff`
   with [`resolve_opts(project, aoi, product)`](../src/coastal_sst_data/config.py#L221) +
   [`opt(opts, name, default)`](../src/coastal_sst_data/config.py#L157) — that pair applies the
   region→project override chain for you.

2. **Name files through [`naming.py`](../src/coastal_sst_data/naming.py) — never by hand.** Use
   `naming.day_stem(aoi, day)` (daily), `naming.time_stem(aoi, t)` (overpass), or the bare AoI name
   (static). The filename stamp is a **contract the assembler parses back out**; a `strftime` written
   inline is how the write side and read side drift apart and every affected day silently becomes a
   NaN slice. See the essay at [naming.py:1](../src/coastal_sst_data/naming.py).

3. **Guard every write with the skip guard, and write atomically.** Before processing an item:
   ```python
   if store.done(path, store.REQUIRED_VARS["CHL"], shape=(g.height, g.width), overwrite=overwrite):
       rep.skip(); continue
   ...
   store.write_output(ds, aoi_out, stem, fmt)   # atomic: scratch file + os.replace
   ```
   `store.done` checks the file *carries the expected variables and grid*, not merely that a path
   exists — a run killed mid-write leaves a file that exists and opens cleanly but is truncated. See
   [store.py:1](../src/coastal_sst_data/store.py). `store.REQUIRED_VARS` is keyed by the ALLCAPS
   `dir`, and is itself derived from your spec's `required_vars`.

4. **Wrap everything that touches the network in [`net.retry`](../src/coastal_sst_data/net.py)** —
   `net.retry(lambda: ..., what="CHL search {name}")`. Bare requests have no timeout or retry.

5. **Stamp provenance into every file:** `ds.attrs.update(aoi_id=name, source=..., **provenance.stamp(eff))`.
   `provenance.stamp` records `acquired_at`, the package version, and the config hash — this is what
   lets the assembler report *when* each field was fetched, and distinguish a recorded date from a
   guessed one.

6. **Report honestly.** Create `rep = report.ProductReport("chl")`; call `rep.expect(n)` for the
   number of items you found, `rep.wrote(source=...)` per successful write, `rep.skip()` per skipped
   item, `rep.fail(item, exc)` per failure. **Return `rep`.** This matters because the pipeline marks
   a product `ok` only when nothing was lost ([pipeline.py:257](../src/coastal_sst_data/pipeline.py#L257));
   a stage that drops 40 of 100 days and returns `None` reports "ok" and the loss becomes invisible —
   the exact bug the report machinery exists to prevent.

7. **Provide a `main()` for standalone runs:** `entry.process_main(acquire, "…description…")`. This
   gives you `python -m coastal_sst_data.processes.chl --config … --aoi … --dry-run` for free (§7).

Skeleton (fill in the product-specific processing; lean on `mur.py` for the real thing):

```python
from ..config import Project, DataProduct, opt as _opt, resolve_opts
from ..grid import AoiGrid, project_grids, select_aois
from .. import entry, naming, net, provenance, report, store

def run(eff, grids, only_aoi, dry_run):
    rep = report.ProductReport("chl")
    for name in select_aois(grids, only_aoi):
        g = grids[name]
        granules = net.retry(lambda: _search(...), what=f"CHL search {name}")
        rep.expect(len(granules))
        if dry_run:
            continue
        for item in granules:
            stem = naming.day_stem(name, t)
            path = eff["out_dir"] / name / f"{stem}.nc"
            if store.done(path, store.REQUIRED_VARS["CHL"], shape=(g.height, g.width),
                          overwrite=eff["overwrite"]):
                rep.skip(); continue
            ds = _process(...)                      # -> Dataset with vars chl, valid on the grid
            ds.attrs.update(aoi_id=name, source=src, **provenance.stamp(eff))
            store.write_output(ds, eff["out_dir"] / name, stem, eff["fmt"])
            rep.wrote(source=src)
    rep.log_summary()
    return rep

def _build_eff(project: Project) -> dict: ...        # copy mur._build_eff
def acquire(project, *, grids=None, aois=None, dry_run=False, overwrite=False):
    eff = _build_eff(project)
    if overwrite: eff["overwrite"] = True
    if grids is None: grids = project_grids(project)
    return run(eff, grids, aois, dry_run)
def main(): entry.process_main(acquire, "coastal_sst_data chlorophyll acquisition.")
if __name__ == "__main__": main()
```

At this point your product **acquires correctly** — it downloads, resumes, reports, and is
provenanced at the product level. It is **not yet in any cube.**

### 3c. Wiring a non-sensor product into the datacube (write a contributor)

The registry gets your product acquired to disk; the **contributor protocol** gets it into the
cube. You write **one `_contribute_*(ctx)` function** and register **one `Contributor`** in
[`datacube.CONTRIBUTORS`](../src/coastal_sst_data/processes/datacube.py) — you do *not* hand-edit
`assemble_aoi` (there is no `Dataset` literal to append to any more). Forgetting the contributor is
a hard error at import (`_check_contributors`), not a silent omission.

A contributor receives an `AssemblyContext` (`ctx`) and calls `ctx.emit(name, dims, arr, **attrs)`
to add channels. It reads aligned files from `ctx.adir("<product>")` and may read/write shared
intermediates via `ctx.slots[...]`. Declare which slots you read/write on the `Contributor`; the
run order is **topologically sorted** from those declarations, so you never reason about a global
sequence. Follow the closest existing model:

- **Daily raster** → copy `_contribute_mur`:
  ```python
  def _contribute_chl(ctx):
      d = ctx.adir("chl")
      ctx.emit("chl", ("time", "y", "x"),
               load_daily_sensor(d, ctx.aid, ctx.days, ctx.H, ctx.W, "chl"))
  ```
  then add `Contributor("chl", (), (), _contribute_chl)` to the `CONTRIBUTORS` tuple. (The `key`
  must equal your product's `DataProduct.value`, so the loud invariant matches it to the registry.)
- **Config-dependent channel set** (variables/depths discovered from the files) → copy
  `_contribute_cmems`, which discovers channels via a `*_channels()` helper rather than a fixed list.
- **Static raster** → copy `_contribute_bathymetry` / `_contribute_landcover`.
- **1D series / station table** → copy `_contribute_tides` / `_contribute_insitu` (+ `build_insitu`).
- **Reads a shared intermediate** (e.g. a sensor's chosen overpass time) → declare it in `reads`
  and pull it from `ctx.slots[SLOT_SENSOR_TIMES]`, like `_contribute_met_overpass`/`_contribute_insitu`.

If your product is daily and you set a `coverage_channel`, the coverage check picks it up from
`DAILY_CHANNELS` automatically — no wiring. If your product deliberately ships **no** cube channel,
set `cube_opt_out=True` on its `ProductSpec` so the invariant knows the omission is intentional.

### 3d. Teaching provenance about new channels

Every cube channel must resolve to a source in
[`provenance.field_inputs`](../src/coastal_sst_data/provenance.py#L171). An **unmapped channel
returns `[]` and only logs a warning** — it ships with a blank provenance record, which is precisely
what this module exists to prevent. Map your channels:

- A channel named exactly `chl` / `chl_valid`, etc. with a fixed input list → add to the `_EXACT`
  dict at [provenance.py:151](../src/coastal_sst_data/provenance.py#L151).
- A whole prefix family (`chl_*`) → add a `name.startswith("chl_")` branch beside the `cmems_` /
  `mur_` ones at [provenance.py:180](../src/coastal_sst_data/provenance.py#L180).
- **Derived** channels (built from several products) list *all* their inputs — an overpass-met
  channel like `eco_airtemp` returns `["met", sensor]`, and `insitu_sst` returns `["insitu", "met"]`
  (it is sampled at met's reference time). If you add a derived channel, add its full input list;
  naming one input would be tidy and wrong.

Sensor-prefixed channels (`<prefix>_*`) are already handled generically via the `SENSORS` map, so a
new *sensor* needs nothing here — this step is only for non-sensor and derived channels.

### 3e. Write a test

The suite's convention is one test module per product (`tests/test_<product>.py`), plus **two**
cross-cutting acid tests — [`test_add_a_sensor.py`](../tests/test_add_a_sensor.py) for a sensor and
[`test_add_a_covariate.py`](../tests/test_add_a_covariate.py) for a non-sensor covariate. Read the
one matching your product before writing yours; each is the executable spec for "a new product is
picked up by every derived table *and* an assembled cube gains its channels."

Their key technique: because the derived tables (`store.REQUIRED_VARS`, `datacube.PRODUCT_DIRS`,
`datacube.CONTRIBUTORS`, `provenance.SENSORS`, `pipeline.PROCESS_MODULES`, …) are computed **once at
import**, a test that adds a spec must `monkeypatch` those module-level tables to re-derive them
from the patched `products.REGISTRY` — this simulates "restart the process with the new spec in
`products.py`". `test_add_a_covariate.py` additionally asserts the payoff of the loud invariant:
registering the spec **without** a contributor makes `_check_contributors()` raise, and adding the
one contributor both satisfies the invariant and makes the assembled cube gain the channel with
correct values.

Run `pytest` (offline; the suite stubs the network).

---

## 4. Adding a source to an existing product

Products like `landsat`, `landcover`, and `insitu` are **source-selectable**: their spec sets
`sources={name: module|None}` and the pipeline resolves *which module* per AoI from the
region→project `source` option ([pipeline._modules_for](../src/coastal_sst_data/pipeline.py#L127)).
The registry already reserves recognised-but-unimplemented sources as `None` (Landsat `aws`/`gee`,
landcover `gee`), so a config naming one fails as *"not implemented"* rather than *"unknown source"*.

To implement one (say `landsat_aws`):

1. Point the source at your new module in the spec: `"aws": "coastal_sst_data.processes.landsat_aws"`.
   If it needs auth, set that source's entry in the `auth` dict.
2. Write `landsat_aws.py` honouring the **same `acquire()` contract** (§3b) **and the same output
   contract as the existing source** — this is the part that is asserted everywhere but written down
   nowhere, so make it explicit for yourself by reading the incumbent (`landsat_pc.py`) and matching:
   - the **same output variables** the spec's `required_vars` names (`sst`, `water`, `cloud`,
     `valid`), with the **same meaning and polarity** (e.g. `water==1` means water), the same
     units convention, and the same `valid` semantics (observed & clear & finite);
   - the **same `dir`, `Kind`, and filename stamp**, so the assembler reads it identically;
   - the same dtype/shape (on the AoI grid, `(time, y, x)` or the Kind's shape).

Because the assembler and provenance key off the **product**, not the source, a correct drop-in
source needs **no changes to `datacube.py` or `provenance.py`** — that's the payoff of the contract.
The interchangeability is the whole point: a cube built from `landsat_pc` and one built from
`landsat_aws` must be indistinguishable channel-for-channel.

---

## 5. Adding a derived stage

`datum` is a **derived stage**, not a product — it is not selectable in a config and produces
nothing from the network; it computes from what other products already wrote. It owns an output
directory the assembler and provenance must find, registered in
[`DERIVED_DIRS`](../src/coastal_sst_data/products.py#L397) (so `product_dirs()` includes it).

- A derived stage that writes an aligned sidecar (like `datum`, which writes
  `DATUM/aligned/<aoi>/<aoi>_datum.json`) is dispatched explicitly from
  [`run_pipeline`](../src/coastal_sst_data/pipeline.py#L162) — it is **not** in `PROCESS_ORDER`
  and is called by name after its inputs exist (datum runs after bathymetry, before assembly). Add
  its dir to `DERIVED_DIRS` and its call site to `run_pipeline`. The assembler reads its result to
  publish the `datum_offset_m` / `datum_status` cube attributes.
- A derived stage computed **inside** the assembler that emits a *channel* (rather than an attribute)
  is just another `Contributor` — key it `"derived:<name>"`, declare the slots it reads, and map its
  channels in `provenance.field_inputs` (§3d). `_contribute_doy` is the minimal example.

Keep derived stages **idempotent and cheap to re-run**, and have them **read inputs off disk** rather
than re-fetching, so they can backfill an existing tree — `datum` reads the bathymetry file rather
than re-downloading a DEM tile, which is why it has its own subcommand.

---

## 5b. Adding a post-assembly preprocess step

A **preprocess step** runs *after* the datacube is assembled, reading the raw `<aoi>.zarr` and writing
a **separate** derived cube `<output_dir>/<preprocess.output_subdir>/<aoi>.zarr` — the raw cube stays
untouched. This is where the *downstream modelling determinations* the assembler deliberately omits
(masking, water-filling, water level — see [datacube.py](../src/coastal_sst_data/processes/datacube.py)'s
design note and the raw-output refactor's D6/D7/D12) get a structured, opt-in home. All the machinery
lives in one module, [`processes/preprocess.py`](../src/coastal_sst_data/processes/preprocess.py), and it
**mirrors the datacube's contributor protocol** rather than the heavyweight product registry — a step
reads an already-opened xarray cube and writes arrays, so it needs none of a product's auth / `dir` /
`Kind` / `required_vars` machinery.

Adding a step is **three things in one file** plus (if it emits genuinely new channel names) a
provenance mapping:

1. **Write a `_step_<name>(ctx)` function.** It receives a `PreprocessContext` and calls
   `ctx.emit(name, dims, arr, **attrs)` to add derived channels and `ctx.carry(name)` to copy a raw
   input forward (so the derived cube is self-describing). Read raw channels **defensively** with
   `ctx.read(name, dims=...)` (returns `None` when a channel is absent or wrong-shaped) /
   `ctx.has(name)` / `ctx.channels_with_prefix(...)` / `ctx.sensor_hours(prefix)` — a step must
   **degrade** (emit nothing, or all-`UNKNOWN`) when an input product wasn't selected, never crash the
   stage. `_step_water_line` and `_step_fill_water` are the two worked examples; both are thin glue
   over existing math (`processes.water_level`, and a restored `fill_water_nn`).
2. **Register a `PreprocessStep`** in the `STEPS` tuple: `key` (also the config selector), the
   `option_keys` it reads, `depends_on` for ordering (topologically sorted like
   `pipeline.process_order`), and documentary `reads`/`writes` channel families.
3. **Nothing else for dispatch.** The stage loops over `STEPS`; the config surface is the open
   `preprocess.steps.<key>` bag, validated against your `option_keys` at stage time by
   `_check_step_options` (deferred out of `config.py` to avoid an import cycle — same rule and
   `did you mean` hint as `config._option_keys_are_known`). Import-time `_check_steps()` rejects a
   duplicate key, an unknown `depends_on`, or a dependency cycle.
4. **Map any new channel names** in [`provenance.field_inputs`](../src/coastal_sst_data/provenance.py)
   (§3d) — a *derived* channel lists **all** its inputs (e.g. `<sensor>_water_elev` →
   `["bathymetry", "tides", sensor]`). Channels that keep a raw name (a filled `mur_sst`, a
   `mur_sst_filled` mask) already resolve via the existing `mur_`/`cmems_` prefixes.

Config, invocation, and testing:

```yaml
preprocess:
  enabled: true                 # opt-in; the stage is a no-op otherwise
  steps:
    water_line: { dem_source: cudem, tide_source: coops, sensors: [eco, lst] }
    fill_water: { sources: [mur, cmems] }
```

```bash
coastal-sst-data run --config config.yaml --assemble --preprocess   # assemble, then preprocess
coastal-sst-data preprocess --config config.yaml --aoi <one> --overwrite
python -m coastal_sst_data.processes.preprocess --config config.yaml
```

Test it in [`tests/test_preprocess.py`](../tests/test_preprocess.py): assemble a synthetic raw cube
(reusing the `test_datacube` writers), run `preprocess_aoi`, and assert the derived channels — plus the
step invariants and a **separate** golden (`tests/golden/preprocessed_golden.json`) so the two stages
don't couple. The raw cube must be left byte-unchanged (there's a test for exactly that).

---

## 6. The shared helpers you must use

Every stage leans on these cross-cutting modules; using them is not optional, because each closes a
specific silent-failure hole:

| Module | Use it for | The failure it prevents |
|---|---|---|
| [`store.py`](../src/coastal_sst_data/store.py) | `store.write_output` (atomic), `store.done` (skip guard) | a mid-write crash leaving a file that exists, opens, and is truncated — taken for "done" forever |
| [`naming.py`](../src/coastal_sst_data/naming.py) | encode/decode the aligned-file timestamp | a write-side and read-side stamp drifting apart → every affected day a silent NaN slice |
| [`net.py`](../src/coastal_sst_data/net.py) | `net.retry` around every network call | an unbounded hang or a transient failure aborting a long run |
| [`report.py`](../src/coastal_sst_data/report.py) | `ProductReport` accounting | a lossy stage reporting "ok" and hiding the gap |
| [`provenance.py`](../src/coastal_sst_data/provenance.py) | `provenance.stamp` on write; `field_inputs` for the cube | a channel shipping with a blank/guessed source record |
| [`config.py`](../src/coastal_sst_data/config.py) | `resolve_opts` + `opt` for per-AoI options | a region override silently ignored, or an unread option silently accepted |
| [`grid.py`](../src/coastal_sst_data/grid.py) | the shared `AoiGrid`; `select_aois` | every product must land on the *same* grid — compute once, don't reinvent |

---

## 7. Developing and debugging one process at a time

You do **not** need to run the whole pipeline to work on one product. Every module's `main()` (via
[`entry.process_main`](../src/coastal_sst_data/entry.py#L74)) gives it a standalone CLI sharing the
same flags as the orchestrator:

```bash
python -m coastal_sst_data.processes.chl --config config.yaml --aoi tillamook_bay --dry-run
python -m coastal_sst_data.processes.chl --config config.yaml --overwrite -v
```

This runs *only* your `acquire()`, against a real config, writing real aligned files — the fastest
inner loop for developing acquisition. `entry.process_main` also threads `argv` through, so a test
can drive the CLI in-process (see how `tests/test_cli.py` drives `cli.main([...])`).

The recommended sequence when adding a product:

```bash
coastal-sst-data validate --config config.yaml            # spec picked up? option surface right?
python -m coastal_sst_data.processes.chl --config … --dry-run   # search works, nothing written
python -m coastal_sst_data.processes.chl --config … --aoi <one>  # acquire one AoI for real
coastal-sst-data assemble --config … --aoi <one> --overwrite     # did the contributor land the channels?
coastal-sst-data provenance --config … --fields                  # is every new channel sourced?
pytest tests/test_chl.py tests/test_add_a_covariate.py          # the derived tables + assembly
```

If `provenance --fields` shows a channel with no products behind it, you missed §3d (that one only
*warns*, so it's on the checklist). Forgetting the contributor itself is louder: `_check_contributors`
raises at **import** — `assemble` (indeed any command that imports `datacube`) crashes with a message
naming your product, so a half-wired covariate can't reach a cube unnoticed.
