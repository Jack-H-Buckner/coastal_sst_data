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
| `depends_on` | products that must run first | declared as a dependency, **not** a position in a hand-kept order. The topo-sort in [`pipeline.process_order`](../src/coastal_sst_data/pipeline.py#L72) places you. Applies to *reading another product's output at acquisition time*, whatever the reason: `met_overpass` snapshots at the sensors' instants; `mur` restricts its days to them. Declare the edge unconditionally, even for an opt-in feature — an edge that appears only for some configs would make the process order config-dependent. |
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

> One thing in `mur.py` is **not** part of the contract to copy: the `overpass_sensors` filter
> (`_granule_day_stamp` / `_select_granules` and the preflight in `run`) is a MUR-specific feature —
> restricting a daily backbone's downloads to the days a thermal sensor flew. If your product needs
> the same idea, reuse [`overpass.py`](../src/coastal_sst_data/overpass.py) rather than copying the
> code; everything else below is the contract.

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

   > **⚠️ Single-file span products need extra care.** Nearly every product writes **one file per
   > day or per scene**, so the filename carries the date and extending the project's date range
   > simply produces new filenames that don't exist yet and get fetched. **Tides** and **insitu**
   > are the exceptions: each writes **one file spanning the whole window** (`<aoi>_tides.nc`,
   > `<aoi>_insitu.nc`). For these, the plain skip guard is a trap — the same filename already
   > exists and passes `is_complete`, so an *extended* date range is silently skipped and the new
   > dates show up as NaN in the cube. The fix is a **range-aware guard**: stamp the window with
   > `provenance.requested_range(start, end)` at write time and pass `covers=(start, end)` to
   > `store.done`, which rebuilds a file whose stamped range no longer spans the configured one.
   > Any future product that writes a single file for the whole window **must** do the same.

4. **Wrap everything that touches the network in [`net.retry`](../src/coastal_sst_data/net.py)** —
   `net.retry(lambda: ..., what="CHL search {name}")`. Bare requests have no timeout or retry.

   **If the call needs a credential, pass a refresher too** —
   `net.retry(lambda: ..., what=..., refresh=auth.refresher("earthdata", eff["earthdata"]))`.
   Runs last hours; credentials do not. Without it an expiry is a `403`, which `net.retry`
   correctly refuses to retry, and every remaining item fails the same way — in date order,
   so it reads as the data simply stopping. Two rules go with it:

   * **Log in through [`auth.login`](../src/coastal_sst_data/auth.py), never the client library.**
     A login nobody timestamped cannot be proactively refreshed. Then call `auth.ensure_fresh`
     at loop boundaries (per AoI, and every `auth.CHECK_EVERY` items) so the common case is a
     credential replaced *before* it dies rather than after.
   * **The closure must include the call that MINTS the credential** — the `open()`, the
     `sign()` — not just the read. `earthaccess.open` returns a handle bound to the session it
     was created with, so re-authenticating cannot heal a handle you already hold; only
     re-opening can. A retry that re-reads through a dead handle loops on the same failure.

   Where the credential is per-URL rather than per-process (a presigned href, a lazy dataset
   handle), an expiry can surface as a *missing file* — pass `markers=net.SIGNED_URL_MARKERS`
   to opt into the wider, deliberately ambiguous vocabulary.

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

#### The two rules a contributor must follow about time

A cube too large to hold in memory is assembled a **block of days at a time** (see
`datacube.block_days`), so `ctx.days` may be one slice of the cube's axis rather than the whole
thing. Your contributor is called once per block, with a fresh `ctx` each time. Two rules follow,
and both are enforced — you will get a `RuntimeError` naming your contributor, not a broken cube.

1. **Emit on `ctx.days`, decide on `ctx.all_days`.** Arrays you emit are indexed by the block
   (`ctx.days`). But any question whose answer describes the *cube* — which variant of a product
   feeds it, whether a layer exists anywhere in the tree — must be asked of `ctx.all_days`.
   `met_prefix` is the worked example: asked about one block's days it answers "these are daily
   means", and the cube ends up splicing two different times of day into one channel.

2. **Your channel SET may depend on the files, never on the days.** Emit `chl` on every block or on
   none. A channel that appears in some blocks and not others produces a store whose variables
   disagree about the length of the time axis — which *writes* without error and then fails on
   `open_zarr` with `conflicting sizes for dimension 'time'`, long after the run that caused it.
   `_check_channel_set` compares each block against `channel_census` and stops the run instead.

   If a channel's existence genuinely depends on file contents, resolve it once over the whole
   window and pass the answer down, as `footprint_available` does for `<sensor>_footprint_id` — the
   only channel in the assembler that has this property.

`ctx.cache` is shared across an AoI's blocks. Pass it to any loader that accepts `cache=` so a
directory is scanned once per AoI rather than once per block; `_cached(cache, key, build)` is the
helper if you add a loader of your own. Anything the cache holds **open** must be released in
`close_cache`.

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

Products like `landsat` and `landcover` are **source-selectable** (`SourceKind.ACCESS`): their spec
sets `sources={name: module|None}` and the pipeline resolves *which module* per AoI from the
region→project `source` option ([pipeline._modules_for](../src/coastal_sst_data/pipeline.py#L127)).
The registry already reserves recognised-but-unimplemented sources as `None` (Landsat `aws`/`gee`,
landcover `gee`), so a config naming one fails as *"not implemented"* rather than *"unknown source"*.

> **First decide which KIND of source you are adding.** `SourceKind.ACCESS` means *redundant access
> to the same data* — pick one, one channel, one directory (Landsat via PC or AWS). `SourceKind.DATA`
> means *distinct data the user stacks* — every configured source is acquired into its own
> `<DIR>/<source>/aligned/` tree (`bathymetry`, `tides`, `met`, `cmems`, `insitu`). Getting this
> backwards is the difference between "the user chooses" and "the user gets both". §4a below covers
> the stacked case.

To implement an ACCESS source (say `landsat_aws`):

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

### 4a. Adding a source to a STACKED (`SourceKind.DATA`) product

A DATA product acquires **every** source in its `sources:` list, so there is no per-AoI module to
resolve — `_check_registry` requires all its sources to map to **one** module, which fans out over
them internally. Adding a source is therefore: a fetch function, a registry entry, and a `SOURCES`
entry in the fan-out module. [`insitu_acquire`](../src/coastal_sst_data/processes/insitu_acquire.py)
is the clearest model — it defines the seam explicitly:

```python
fetch_aoi(g, start, end, cfg, dry_run=False) -> [record, ...]
SOURCES = {"ioos": insitu_ioos.fetch_aoi, "csv": insitu_csv.fetch_aoi}
```

Everything after the fetch — the union time axis, the range-aware skip guard, the atomic write, the
report — is shared, so a new network is a fetch function rather than another copy of the loop.
(`tides` and `met` do the same with a `_SOURCES` dict of fetchers inside a single module; split the
sources into their own modules, as in-situ does, once they stop being a few lines each.)

**Per-source auth** on a stacked product resolves as the union over the configured list
([`config.required_backend`](../src/coastal_sst_data/config.py)), and raises if two stacked sources
need *different* credentials — no product does this yet, and the preflight resolves one backend per
product, so it says so rather than leaving one source's credentials unverified.

**Channels: one per source, or one merged set?** The usual DATA shape is **one channel per source**
(`depth_cudem`, `depth_gmrt`) — distinct data deserves distinct channels. **In-situ deliberately
deviates**: every source merges into ONE channel set, because stations are *rows*, not channels.
They occupy disjoint pixels anyway, and splitting them would multiply the whole `<sensor>_insitu_*`
family by the source count while making `insitu_sst` stop meaning "ground truth". Which source a
platform came from is recorded per station in the cube's `insitu_stations` table. If you add a
stacked product whose sources are *rows* rather than *layers*, follow in-situ; otherwise follow
bathymetry.

> ⚠️ **Reading a stacked product's directories.** Do **not** treat every subdirectory of
> `<DIR>/` as a source tag. `_contribute_stacked_sensor` did, and one stray leftover directory was
> enough to produce silently all-sentinel channels across an entire project — see
> [bug-empty-version-tag-channels.md](bug-empty-version-tag-channels.md). Require the expected file
> to exist, and **log every directory you skip**. `datacube.load_insitu` is the pattern to copy.

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

A **preprocess step** runs *after* the datacube is assembled, reading
`<output_dir>/<datacube.output_subdir>/<aoi>.zarr`, adding its derived channels, and rewriting **that
same store** atomically. This is where the *downstream modelling determinations* the assembler deliberately omits
(masking, water-filling, water level — see [datacube.py](../src/coastal_sst_data/processes/datacube.py)'s
design note and the raw-output refactor's D6/D7/D12) get a structured, opt-in home. All the machinery
lives in one module, [`processes/preprocess.py`](../src/coastal_sst_data/processes/preprocess.py), and it
**mirrors the datacube's contributor protocol** rather than the heavyweight product registry — a step
reads an already-opened xarray cube and writes arrays, so it needs none of a product's auth / `dir` /
`Kind` / `required_vars` machinery.

Adding a step is **three things in one file** plus (if it emits genuinely new channel names) a
provenance mapping:

1. **Write a `_step_<name>(ctx)` function.** It receives a `PreprocessContext` and calls
   `ctx.emit(name, dims, arr, **attrs)` to add derived channels. **The name must be one no assembler
   channel uses** — the step writes into the assembled cube, so emitting `eco_sst_v002` would destroy
   what the sensor delivered; give the output a suffix of its own and register it in
   [`processes/channels.py`](../src/coastal_sst_data/processes/channels.py). `preprocess_aoi` raises if
   a step tries. Read channels **defensively** with
   `ctx.read(name, dims=...)` (returns `None` when a channel is absent or wrong-shaped) /
   `ctx.has(name)` / **`ctx.base_channels(prefix)`** (a prefix scan that hides a previous run's derived
   channels — use this, not `channels_with_prefix`, to discover inputs) / `ctx.sensor_hours(prefix)` — a step must
   **degrade** (emit nothing, or all-`UNKNOWN`) when an input product wasn't selected, never crash the
   stage. `_step_water_line` and `_step_fill_water` are the two worked examples; both are thin glue
   over existing math (`processes.water_level`, and a restored `fill_water_nn`). A step that needs
   more than glue can live in its **own module** and be imported into the `STEPS` tuple — the two
   cloud filters (`filter_clouds`, `filter_cloud_cover`) sit in
   [`processes/cloud_filter.py`](../src/coastal_sst_data/processes/cloud_filter.py); it imports
   nothing from `preprocess` at runtime (only a `TYPE_CHECKING` hint for `PreprocessContext`), so
   there is no cycle. Two steps that mutate the **same** channel compose by reading the *working*
   value (a channel already emitted this run wins over the raw cube — see `cloud_filter._working`),
   so their edits stack regardless of run order.
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

#### The two rules a step must follow about time

A cube too large to hold in memory is preprocessed a **block of days at a time**
(`preprocess.block_days`), so `ctx.ds_cube` may be one time slice of the store and your step is
called once per block with a fresh `ctx`. Because `ctx.read` reads off `ctx.ds_cube`, a **day-local**
step — one whose output for a day depends only on that day's input — needs no changes at all for
this: it asks for a channel and gets that block's days. Nine of the ten shipped steps are day-local.

1. **Emit on `ctx.days`, decide on `ctx.all_days`.** Any question whose answer describes the *cube*
   rather than the block must be asked of `ctx.all_days` / `ctx.read_all`. `cloud_filter`'s "is the
   day-of-year span long enough to fit a seasonal harmonic?" is the worked example — asked of a
   block it answers *no* every time, and a ten-year cube silently fits a constant climatology
   everywhere while the config says `harmonic`.

2. **Your channel SET may depend on the cube's channels, never on the days.** Emit a channel for
   every block or for none. A channel present in some blocks and not others builds a store whose
   variables disagree about the length of the time axis — which *writes* without error and then
   fails on `open_zarr` with `conflicting sizes for dimension 'time'`. `_check_channel_set` compares
   each block against `preprocess_census` and stops the run instead.

**If your step reduces over the time axis** — a climatology, a trend, anything needing all days at
once — it cannot be answered from one block. Declare a `WindowStat` on its registration:
`passes(ctx)` says how many prepasses your *config* needs (return `0` when it is configured
day-locally, and you pay nothing), `accumulate(ctx, i)` folds in one block, and `reduce(ctx, i)`
turns the accumulators into the statistic, which lands in `ctx.window` for your `fn` to read per
block. `cloud_filter`'s sigma climatology is the worked example: its fit is masked normal
equations, every term a plain sum over `t`, so accumulating block by block is the same arithmetic
in a different order — exact, not approximate.

Use `ctx.cache` (shared across an AoI's blocks) for anything derived from the grid or the tree
rather than from the days — `flag_georef`'s coastline distance transform is cached this way, or it
would run once per block.

Config, invocation, and testing:

```yaml
preprocess:
  enabled: true                 # opt-in; the stage is a no-op otherwise
  steps:
    water_line: { dem_source: cudem, tide_source: coops, sensors: [eco, lst] }
    fill_water: { sources: [mur, cmems] }
    # Screen cloud-contaminated ECOSTRESS pixels (ported from oceanSR's cold-deviation filter).
    # method: offset -> drop where `baseline - eco > threshold_k`; sigma -> drop where eco is
    # below the baseline climatology's `mean - n_sigma*sigma` (per-pixel or pooled; optional
    # day-of-year harmonic seasonality).
    filter_clouds: { method: sigma, baseline: mur_sst, n_sigma: 3.0,
                     stat_scope: pixel, seasonality: harmonic }
    # Gate ECOSTRESS on met total cloud cover (HRRR/ERA5, percent): reject a scene whose AOI-mean
    # exceeds scene_max_pct, and/or drop individual pixels above pixel_max_pct. Needs the met
    # `(eco, <src>)` overpass combo (or a `cloud_cover_<src>` forcing channel) in the cube.
    # Composes with filter_clouds; either gate is disabled with a null / >=100 threshold.
    filter_cloud_cover: { source: hrrr, scene_max_pct: 30, pixel_max_pct: 80 }
```

```bash
coastal-sst-data run --config config.yaml --assemble --preprocess   # assemble, then preprocess
coastal-sst-data preprocess --config config.yaml --aoi <one> --overwrite
python -m coastal_sst_data.processes.preprocess --config config.yaml
```

Test it in [`tests/test_preprocess.py`](../tests/test_preprocess.py): assemble a synthetic cube
(reusing the `test_datacube` writers), run `preprocess_aoi`, and assert the derived channels — plus the
step invariants and a golden (`tests/golden/preprocessed_golden.json`, the cube *after* preprocess) kept
separate from `datacube_golden.json` (the cube as `assemble` leaves it) so a change to one stage does not
force the other's golden to be regenerated.

Two properties every step must keep, each with a test: the **assembled channels come through unchanged**,
and the stage is **idempotent** — a step seeds its target from the raw channel, never from whatever it
finds on disk under its own output name, or a second run composes onto the first run's results.

---

## 6. The shared helpers you must use

Every stage leans on these cross-cutting modules; using them is not optional, because each closes a
specific silent-failure hole:

| Module | Use it for | The failure it prevents |
|---|---|---|
| [`store.py`](../src/coastal_sst_data/store.py) | `store.write_output` (atomic), `store.done` (skip guard) | a mid-write crash leaving a file that exists, opens, and is truncated — taken for "done" forever |
| [`naming.py`](../src/coastal_sst_data/naming.py) | encode/decode the aligned-file timestamp | a write-side and read-side stamp drifting apart → every affected day a silent NaN slice |
| [`net.py`](../src/coastal_sst_data/net.py) | `net.retry` around every network call; `net.is_auth_error` to tell a dead credential from a dead server | an unbounded hang, a transient failure aborting a long run, or a credential expiring four hours in and reading as a permission error that kills the rest of the range |
| [`auth.py`](../src/coastal_sst_data/auth.py) | `auth.login` (never the client library directly), `auth.refresher` on credentialed `net.retry` calls, `auth.ensure_fresh` at loop boundaries | a login nobody timestamped — and therefore cannot refresh — leaving a multi-hour run to die on an expired token that looks exactly like the data running out |
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
