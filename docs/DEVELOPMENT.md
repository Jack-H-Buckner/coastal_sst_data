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

This is the single most important table in this document, because the registry makes *some*
extensions genuinely one declaration and leaves *others* needing hand-wiring — and the difference
is not obvious from the code.

| To add a… | Registry-only? | You also write |
|---|---|---|
| **Thermal sensor** (`Kind.OVERPASS_SENSOR`) | ✅ **Yes** | Nothing else. The assembler loops over `products.sensors()`, so every `<prefix>_sst/_cloud/_valid/_hour/_tide/_water_elev/_water_class`, overpass-met, and in-situ-matchup channel is generated from the spec. This is the "acid test" proven by [`test_add_a_sensor.py`](../tests/test_add_a_sensor.py). |
| **A new source** for an existing product (e.g. `landsat_aws`) | ✅ ~Yes | A process module matching the **existing product's output contract** (§4). The registry already reserves `None` placeholders for these. |
| **A new daily / static / series covariate** (a *non-sensor* product) | ❌ **No** | A `ProductSpec` **+ a process module (§3b) + a hand-wired block in `datacube.assemble_aoi` (§3c) + provenance rules if it produces derived channels (§3d)**. |

> ⚠️ **The trap.** The docstring at [products.py:41](../src/coastal_sst_data/products.py#L41) says
> "Adding a product is now: write the module, add a ProductSpec here." That is completely true for
> *acquisition, dispatch, ordering, auth, and the skip guard* — a new covariate will be downloaded,
> resumed, and provenanced at the product level with nothing but a spec. It is **not** true for
> *assembly*: [`datacube.assemble_aoi`](../src/coastal_sst_data/processes/datacube.py#L448) reads
> each non-sensor product through a **hand-written block** (MUR, met, CMEMS, tides, bathymetry,
> land-cover, in-situ each have their own `load_*` call and `Dataset` entry). A new covariate with
> a perfect spec and a working `acquire()` will be **acquired to disk and then silently omitted
> from every cube** until you wire it into the assembler. That silent gap is exactly the class of
> failure the registry was built to eliminate for *sensors* — it just doesn't extend to arbitrary
> covariates yet.

Sensors are registry-driven end-to-end because they are a **uniform family** — every sensor
produces the same channel shapes and differs only in a handful of validity flags
([`SensorSpec`](../src/coastal_sst_data/products.py#L82)). A covariate is not a uniform family:
CMEMS emits per-depth channels discovered from the files, met emits a fixed set with source-code
channels, bathymetry feeds the land mask. There is no single loop that could serve all of them, so
each is wired by hand.

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

### 3c. Wiring a non-sensor product into the datacube

This is the step the registry does *not* do for you (see §2). Open
[`datacube.assemble_aoi`](../src/coastal_sst_data/processes/datacube.py#L448) and add a block that
loads your product and contributes its channels to the final `Dataset`. Follow the closest existing
model:

- **Daily raster** → copy the MUR pattern ([datacube.py:459](../src/coastal_sst_data/processes/datacube.py#L459)):
  ```python
  chl = load_daily_sensor(adir("chl"), aid, days, H, W, "chl")
  # optionally NN-fill over land-cover water like MUR/CMEMS do, if it's an ocean field
  ```
  then add `"chl": (T, chl)` to the `xr.Dataset({...})` literal at
  [datacube.py:617](../src/coastal_sst_data/processes/datacube.py#L617).
- **Config-dependent channel set** (variables/depths discovered from the files) → copy the CMEMS
  pattern ([datacube.py:555](../src/coastal_sst_data/processes/datacube.py#L555)), which discovers
  channels via a `*_channels()` helper rather than a fixed list.
- **Static raster** → copy `load_bathy` / `load_landcover`.
- **1D series / station table** → copy `load_tide_daily` / `load_insitu` + `build_insitu`.

If your product **falls back between sources day-to-day** (like CMEMS reanalysis→forecast or met
HRRR→ERA5), also add it to the per-day source-channel loop at
[datacube.py:542](../src/coastal_sst_data/processes/datacube.py#L542) so a row of the cube can be
traced to the file that produced it.

If your product is daily and you set a `coverage_channel`, the coverage check at
[datacube.py:744](../src/coastal_sst_data/processes/datacube.py#L744) picks it up from
`DAILY_CHANNELS` automatically — no wiring.

### 3d. Teaching provenance about new channels

Every cube channel must resolve to a source in
[`provenance.field_inputs`](../src/coastal_sst_data/provenance.py#L171). An **unmapped channel
returns `[]` and only logs a warning** — it ships with a blank provenance record, which is precisely
what this module exists to prevent. Map your channels:

- A channel named exactly `chl` / `chl_valid`, etc. with a fixed input list → add to the `_EXACT`
  dict at [provenance.py:151](../src/coastal_sst_data/provenance.py#L151).
- A whole prefix family (`chl_*`) → add a `name.startswith("chl_")` branch beside the `cmems_` /
  `mur_` ones at [provenance.py:180](../src/coastal_sst_data/provenance.py#L180).
- **Derived** channels (built from several products) list *all* their inputs — `water_elev` returns
  `["bathymetry", "tides", "datum", sensor]`. If you add a derived channel, add its full input list;
  naming one input would be tidy and wrong.

Sensor-prefixed channels (`<prefix>_*`) are already handled generically via the `SENSORS` map, so a
new *sensor* needs nothing here — this step is only for non-sensor and derived channels.

### 3e. Write a test

The suite's convention is one test module per product (`tests/test_<product>.py`), plus the
cross-cutting **acid test** [`test_add_a_sensor.py`](../tests/test_add_a_sensor.py) — read it before
writing yours, it is the executable spec for "a new product is picked up by every derived table."

Its key technique: because the derived tables (`store.REQUIRED_VARS`, `datacube.PRODUCT_DIRS`,
`provenance.SENSORS`, `pipeline.PROCESS_MODULES`, …) are computed **once at import**, a test that
adds a spec must `monkeypatch` those module-level tables to re-derive them from the patched
`products.REGISTRY` — this simulates "restart the process with the new spec in `products.py`". See
the `registered` fixture at [test_add_a_sensor.py:61](../tests/test_add_a_sensor.py#L61). For a
non-sensor product, also assert (as that test does for sensors) that an assembled cube actually
**gains your channels with correct values** — that's the assertion that would have caught a missing
§3c wiring.

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

`datum` and `water_level` are **derived stages**, not products — they are not selectable in a config
and produce nothing from the network; they compute from what other products already wrote. They own
an output directory the assembler and provenance must find, registered in
[`DERIVED_DIRS`](../src/coastal_sst_data/products.py#L397) (so `product_dirs()` includes them).

- A derived stage that writes an aligned sidecar (like `datum`, which writes
  `DATUM/aligned/<aoi>/<aoi>_datum.json`) is dispatched explicitly from
  [`run_pipeline`](../src/coastal_sst_data/pipeline.py#L162) — it is **not** in `PROCESS_ORDER`
  and is called by name after its inputs exist (datum runs after bathymetry, before assembly). Add
  its dir to `DERIVED_DIRS` and its call site to `run_pipeline`.
- A derived stage computed **inside** the assembler (like `water_level`, called from within
  `assemble_aoi`) needs no directory of its own; it just needs its channels mapped in
  `provenance.field_inputs` (§3d).

Keep derived stages **idempotent and cheap to re-run**, and have them **read inputs off disk** rather
than re-fetching, so they can backfill an existing tree — `datum` reads the bathymetry file rather
than re-downloading a DEM tile, which is why it has its own subcommand.

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
coastal-sst-data assemble --config … --aoi <one> --overwrite     # did §3c wiring land the channels?
coastal-sst-data provenance --config … --fields                  # is every new channel sourced?
pytest tests/test_chl.py tests/test_add_a_sensor.py              # the derived tables + assembly
```

If `provenance --fields` shows a channel with no products behind it, you missed §3d. If `assemble`
produces a cube without your channels, you missed §3c. Neither raises — which is exactly why they're
on the checklist.
