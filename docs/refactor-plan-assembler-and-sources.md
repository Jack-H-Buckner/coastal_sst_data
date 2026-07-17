# Refactor plan — assembler symmetry, multi-source channels, and raw-output simplification

**Status:** draft for review
**Author:** drafted with Claude, 2026-07-15
**Scope:** three work themes of `coastal_sst_data` (contributor protocol, multi-source acquisition,
per-source channels + raw-output simplification) — sequenced into stages S0–S5 in [§4](#4-sequencing)
— plus a README deliverable.

This document consolidates the design discussion into one reviewable plan. It supersedes any
earlier informal notes. All open items (O1–O5) are now **resolved** and folded into
[§2 Confirmed decisions](#2-confirmed-decisions) (see [§3](#3-open-items-for-your-review) for the
resolution trail); no decision blocks implementation. **Execution order is set in [§4](#4-sequencing)
and is not the section order** — the raw-output deletions run early. Read [§5.3](#53-the-contributor-set)
before building the contributor protocol (stage S2): the wiring shape depends on the §4 deletions
fork, and building the wrong shape breaks the byte-identity S2 depends on.

---

## 1. Goals

1. **Remove the sensor-vs-non-sensor asymmetry in the datacube assembler.** Today the
   per-overpass thermal sensors are registry-driven (a loop over `products.sensors()`), while
   every other product is read by a hand-written block in
   [`datacube.assemble_aoi`](../src/coastal_sst_data/processes/datacube.py). A new non-sensor
   covariate is acquired to disk from its `ProductSpec` alone and then **silently omitted from
   every cube** until someone remembers to hand-wire it — the exact class of silent failure the
   registry was built to eliminate everywhere else. (See
   [DEVELOPMENT.md §2](DEVELOPMENT.md) for the current statement of the asymmetry.)

2. **Replace default-source + fallback with one channel per source.** Products that today pick a
   primary source and silently fall back to a backup (bathymetry CUDEM→GMRT, met HRRR→ERA5, CMEMS
   reanalysis→forecast, tides CO-OPS→model) instead expose **one channel per source**
   (`depth_cudem`, `depth_gmrt`, `airtemp_hrrr`, `airtemp_era5`, …). The user loads as many sources
   as they need to span their AoIs and time range; source coverage gaps are **documented in the
   README** and the user closes them by stacking sources, not by a hidden fallback.

3. **Ship raw data in a common format; push masking/filling/matching/derivation decisions
   downstream.** The module's job is to land every product on a common grid and daily axis in a
   self-describing cube. Water-filling, land-masking, station-to-pixel snapping, and multi-input
   derivations (water level / submerged-exposed class) are **modeling determinations** that belong to
   the downstream model, which can make them per-process. The cube therefore drops the NN water fills,
   the opinionated derived land mask, the water-mask-dependent in-situ snapping, and the multi-input
   derived channels — shipping instead the raw per-source ingredients to reconstruct any of them.

4. **Separate met's two roles into two processes.** Weather serves two distinct purposes that the
   codebase currently fulfills in one process: a **forcing variable** (daily conditions at a reference
   time) and **overpass documentation** (conditions at the instant a satellite flew). These become two
   products — `met` and `met_overpass` — each independently configured, and the overpass alignments
   are chosen explicitly by the user per region rather than produced as an automatic cross-product.

---

## 2. Confirmed decisions

Settled during the design discussion; fixed for this plan.

| # | Decision | Consequence |
|---|---|---|
| D1 | **Full contributor protocol**, centralized in `datacube.py` (contributors live in the assembler module, not colocated in each process module). | Every product — sensor and non-sensor — flows through one uniform `(product[, source]) → channels` mechanism. |
| D2 | **Run-order is derived from declared slot reads/writes** (topo-sort), not a hand-kept order list. | Mirrors `pipeline.process_order`; no ordering list to drift. |
| D3 | **One channel per source; remove `default_source`/`fallback`.** Applies to *distinct-data* sources only (D10). | The user stacks sources to span coverage gaps. |
| D4 | **Naming: source is the last token on the variable** — `depth_cudem`, `cmems_thetao_0m_glorys`, `airtemp_hrrr`. | Keeps existing variable grouping; extends cleanly to CMEMS's per-variable/per-depth names. |
| D5 | **Per-source channels** for every distinct-data source — `depth_<src>`, `elevation_<src>`, `landcover_water_<src>`, `tide_<src>`, `cmems_<var>_<src>`, `<metvar>_<src>`. | Channel count scales with the number of stacked sources (bounded by what the user loads). |
| D6 | **Drop the NN water fills** (`fill_water_nn`, `fill_mur_water`, `fill_cmems_water`) and the `_filled`/`mur_filled` channels. | MUR and CMEMS ship observed values with honest NaN gaps; downstream fills as it sees fit. |
| D7 | **Drop the water-mask-dependent operations**: the singular `water` union slot, the derived `landmask` channel, and in-situ **snapping**. | The cube ships raw per-source ingredients; downstream builds masks. |
| D8 | **Add a loud-omission invariant** — a product with cube presence that has no registered contributor (and no explicit opt-out) is a hard error at import. | Closes the silent-omission trap for non-sensor products. |
| D9 | **README source-coverage gap table** is a first-class deliverable of the per-source work (stage S5). | Per product: which sources cover which regions / time ranges, so the user can choose what to stack. |
| D10 | **(was O1) Distinct-data sources are per-channel; redundant-access sources feed one channel.** *Distinct-data*: bathymetry (CUDEM/GMRT), met (HRRR/ERA5), CMEMS (reanalysis/forecast, regional models), tides (CO-OPS/model), landcover `gee` (JRC+NDWI). *Redundant-access* (same data, different pipe → single channel, pick-one): Landsat `pc`/`aws`/`gee`, landcover `esa`/`worldcover`. | `ProductSpec` gains an **access** vs **data** source distinction; only `data` sources get D3/D5. |
| D11 | **(was O2) In-situ: keep rasterization + temporal overpass matching; drop only the snapping.** A station goes into the cell it falls in (call `station_pixels(..., water=None)`); a station in a land cell stays there. | Preserves pixel-for-pixel / minute-for-minute validation; removes the last water-mask consumer. |
| D12 | **(was O3) Drop the multi-input derived channels entirely.** Remove `<sensor>_water_elev` / `<sensor>_water_class` / `<sensor>_tide` and the `derived:water_level` contributor. Ship the raw ingredients instead: per-source `elevation_<src>` and `depth_<src>`, per-source daily `tide_<src>`, plus each sensor's `<pre>_hour`. The **datum offset** (DEM→MSL) that water level consumed ships as a **per-source cube attribute** (`datum_offset_m`, `datum_status`) so downstream can reference elevation to MSL. | Water level / submerged-exposed classification becomes a downstream computation; avoids the sensors × bathy × tide channel multiplication. |
| D13 | **(was O4) Keep the overpass-met channels** (`<sensor>_<metvar>_<src>`) — met matched to a sensor's exact overpass time is a real information source, not a reconstructable derivation. **The combinations are user-specified per region**, not an automatic sensor × source cross-product. | The channel count is exactly what the user opts into, which resolves the "multiply to excess" concern that made O4 borderline. |
| D14 | **Split met into two products / processes.** `met` = **daily forcing** (standalone per-source variables at a reference time; **no sensor dependency**). `met_overpass` = **overpass documentation** (snapshots time-aligned to a sensor; **depends on the sensors**). Today one module does both simultaneously. | Two clear responsibilities; `met` loses its `depends_on=(eco,lst,modis)`; the overpass concern is isolated in its own product with its own config. |
| D15 | **Config surface for met (per region).** `met`: a list of `sources` (standalone daily variables) + a `reference_time` for the daily sample. `met_overpass`: a list of `(sensor, source)` **combinations** the user wants. Both region-overridable. | The user picks exactly which forcing sources and which sensor-met alignments are produced. |
| D16 | **(was O5) Forcing met keeps both `reference_time` snapshot and `daily_mean` modes**, config-selectable via `met_time` (default: reference-time snapshot). | No change to the existing dual-mode behavior; the config chooses. |
| D17 | **(S4-review) Add `tide_overpass`, correcting D12's tide claim.** D12 said downstream reconstructs overpass tide from "daily `tide` + `<s>_hour`", but the cube ships only the daily **mean** tide, which for a ~zero-mean tidal signal is ≈0 and carries no phase — the instantaneous overpass tide (needed for water level) is **not** recoverable from it. So the cube emits `<sensor>_tide_<src>` per user-specified `(sensor, source)` combo, matching `met_overpass`'s channel shape and config. **Implementation differs from `met_overpass`, though:** the tide series is a smooth signal already fully on disk (the tides product's series), so `tide_overpass` is a **DERIVED contributor** at assembly (interpolate the per-source series to each sensor's `<s>_hour`, reusing `water_level.tide_at_overpass`) — NOT a new acquisition/`Kind.OVERPASS_ALIGNED` product. `met_overpass` stays an acquisition product because a weather model's value at an instant is not interpolatable from a daily sample. | Downstream can reconstruct overpass water level from the cube alone; the daily-mean `tide` channel (near-useless for a zero-mean signal) is superseded. |

**Supersession notes.** D7 supersedes an earlier tentative "keep one union `landmask`" — Goal 3
(downstream owns land-masking) means the cube ships raw per-source ingredients, not an opinionated
mask. D12 supersedes keeping `water_elev`/`water_class`.

**Datum resolution folded into bathymetry (S3 outcome).** The offset is still needed downstream to
reference `elevation_<src>` to MSL, but rather than a standalone stage/sidecar it is now resolved by
the datum *library* INSIDE the bathymetry module as each DEM source is acquired, and ships as
attributes on that source's `elevation_<src>` channel (per-source, since CUDEM/NAVD88 and GMRT/MSL
differ). The standalone `datum` stage and `coastal-sst-data datum` subcommand were removed.

---

## 3. Open items for your review

**All open items are resolved** (O1–O4 → D10–D15; O5 → D16). No decisions block implementation.

- **O5 — forcing-met representation. RESOLVED (D16): keep both modes.** The `met` (forcing) product
  samples the daily met at the configured `reference_time`; both a **snapshot at the reference time**
  (default — 10:30 local solar, avoids smearing the diurnal cycle) and a **daily mean** remain
  config-selectable via `met_time`, unchanged from today. The user's config picks.

---

## 4. Sequencing

The work splits into three **themes**, documented in §5 (contributor protocol, goal 1), §6
(multi-source acquisition, goal 2), and §7 (per-source channels + raw-output simplification, goals
2–3). **Execution order is NOT section order.** The stages below are the plan of record; each names
the section holding its mechanics. Two decisions drive the order:

1. **The raw-output deletions (goal 3, detailed in §7.2) are pulled to the front, ahead of the
   contributor protocol.** They are self-contained edits to today's `assemble_aoi` and depend on
   nothing but the golden test. Doing them first shrinks the shared-intermediate surface from four
   slots to two (`sensor_times`, `ref_utc`) — see the §5.1 note — so the protocol becomes a trivial
   2-slot migration (the §5.3b wiring) instead of having to model, then demolish, the `water` slot
   (the §5.3a wiring). Build-then-demolish is avoided.
2. **Multi-source acquisition (goal 2) is done as vertical slices — one product end-to-end — not as
   horizontal layers.** Bathymetry (the simplest data source: static, no reanalysis/forecast switch,
   no met split) goes all the way through — acquisition → per-source dirs → config → per-source
   channels → provenance — proving the whole pattern before it is replicated for met/cmems/tides.

### Stage order (plan of record)

```
S0  Golden-cube test                                      (§5.5 Phase 0)
        │  safety net; run-stamp attrs excluded from the snapshot
        ▼
S1  Raw-output simplification / deletions                 (§7.2)
        │  fills, landmask, snapping, water_level, daily_sources removed; raw ingredients
        │  (elevation, tide, <s>_hour) shipped; datum attrs retained. ONE reviewed
        │  behavior-diff. ⚠ downstream-gated — see the fork below.
        ▼
S2  Contributor protocol + loud invariant                 (§5, built directly in the §5.3b 2-slot shape)
        │  byte-identical against the S1 baseline
        ▼
S3  Multi-source vertical slice: bathymetry end-to-end     (§6 mechanics + §7.1 emission, bathymetry only)
        │  access/data split; per-source dirs across store/provenance; config; per-source channels
        ▼
S4  Replicate the slice: met (+met_overpass split), cmems, tides + config migration  (§6, §6.1, §6.2)
        ▼
S5  Provenance per-source finish + README source-coverage gap table    (§7.3, §7.4)
```

### The downstream fork at S1

S1 is the only stage that changes cube outputs a consumer sees (fills/landmask/water_level/snapping
gone). That break lands at the very start, and consumers stay on the old shape until they implement
the raw-ingredient reconstruction (Goal 3). So S1's position is gated on downstream readiness:

- **Consumer ready soon (or you control it):** run S1 first, as above. The protocol (S2) is then
  trivial, and the §5.3a 4-slot wiring is never built.
- **Consumer not ready / external:** move S1 to the end (after S5) — i.e. `S0 → S2 → S3 → S4 →
  deletions`. The cube stays fully functional until the last step, but the protocol (S2) must then be
  built in the §5.3a 4-slot shape and migrated to §5.3b when the deletions land. Strictly more
  protocol work; choose it only if the downstream break cannot be absorbed early.

Either branch keeps the same **two** reviewed behavior-diffs (the S1 deletions, and per-source
emission in S3–S4); the fork only decides whether the deletions diff comes first or last.

---

## 5. Theme: the contributor protocol (goal 1)

> Execution order lives in §4; this section is the design detail. The protocol is stage **S2**, built
> directly in the §5.3b 2-slot shape in the default (deletions-first) order.

### 5.1 Core types (new, in `datacube.py`)

```python
# Slot names are module constants so a typo is a NameError, not a silent miss.
SLOT_SENSOR_TIMES = "sensor_times"  # {prefix: [datetime|None per day]}
SLOT_REF_UTC      = "ref_utc"       # [datetime|None per day]  (met reference time)
# --- deletions-last-only slots (removed by the S1 simplification: fills / land mask / water level) ---
SLOT_WATER        = "water"         # (H,W) bool water mask (land-cover ∪ sensor union ∪ elev<0)
SLOT_ELEV         = "elev"          # (H,W) float32 DEM elevation (read by the water_level stage)

@dataclass
class AssemblyContext:
    g: AoiGrid; eff: dict; days; aid: str; H: int; W: int
    slots: dict[str, Any]                 # shared intermediates keyed by SLOT_*
    channels: dict[str, tuple]            # name -> (dims, array); merged into the Dataset
    global_attrs: dict                    # ds.attrs to set
    var_attrs: dict[str, dict]            # per-channel attrs (long_name, flag_values, legend)
    def adir(self, product, source=None): ...      # per-source aligned dir after the S3–S4 slices
    def emit(self, name, dims, arr, **attrs): ...   # add a channel (+ its attrs)

@dataclass(frozen=True)
class Contributor:
    key: str                              # product name, or "derived:<name>"
    reads: tuple[str, ...]                # SLOT_* it consumes
    writes: tuple[str, ...]               # SLOT_* it produces
    fn: Callable[[AssemblyContext], None] # mutates ctx.channels / ctx.slots / ctx.*_attrs

CONTRIBUTORS: tuple[Contributor, ...] = (...)   # the registry
```

> **The slot surface depends on whether the S1 deletions have happened — get this right before you
> wire `CONTRIBUTORS`.** Once the S1 simplification (D6/D7/D12) has landed, the shared-intermediate
> surface is just two slots: `sensor_times` (written by `sensors`, read by `met_overpass` and
> `insitu`) and `ref_utc` (written by `met`, read by `insitu`). In the **default deletions-first
> order** that is already true when you build the protocol (S2), so `SLOT_WATER`/`SLOT_ELEV` never
> exist — omit them. Only in the **deletions-last branch** does the assembler still compute today's
> `water` mask and consume it in four places, giving the four-slot set
> `{sensor_times, ref_utc, water, elev}` — see [§5.3a](#53a-deletions-last-wiring-assembler-still-computes-water)
> for that wiring. Match the slot set to your branch.

### 5.2 Orchestrator (replaces the body of `assemble_aoi`)

```python
def assemble_aoi(g, eff, days):
    ctx = AssemblyContext(g=g, eff=eff, days=days, aid=g.name, H=g.height, W=g.width,
                          slots={}, channels={}, global_attrs={}, var_attrs={})
    for c in _topo_order(CONTRIBUTORS):   # edges: writer(slot) -> reader(slot); ties = registry order
        c.fn(ctx)
    xs, ys = g.xy_centers()
    ds = xr.Dataset(ctx.channels, coords={"time": days, "y": ys, "x": xs})
    ds.attrs.update(ctx.global_attrs)
    _apply_var_attrs(ds, ctx.var_attrs)
    # coverage + provenance stamping stay exactly as today (already registry-driven)
    ...
    return ds
```

`_topo_order` builds edges from `writes → reads` on shared slot names and topologically sorts,
breaking ties by registry order for determinism. `_check_contributors()` (below) guarantees every
`reads` slot has a producer, so the sort cannot starve on a missing slot.

### 5.3 The contributor set — two wiring shapes

The contributor graph has **two possible shapes, and which one you build depends on the §4 fork:**

- **Deletions-first (default §4 order):** the S1 deletions have already removed the `water`/`elev`
  slots, so build the protocol directly in the **§5.3b** 2-slot shape. This is the target.
- **Deletions-last:** the assembler still computes today's `water` mask, so the protocol must be
  built in the **§5.3a** 4-slot shape and later migrated to §5.3b when the deletions land.

Building the §5.3b shape while the deletions have *not* happened is the trap the earlier single-table
draft invited: the topo-sort would omit the `water` dependency edges and could legally order
`insitu`/`mur`/`cmems` *before* the land-mask contributor, silently breaking byte-identity. **Match
the table to your branch.**

#### 5.3a Deletions-last wiring (assembler still computes `water`)

Emissions are **today's single-source names**. Note `mur` (the backbone), the `water` slot with its
three writers and four readers, and the `water_level`+`datum` contributors — all of which exist
before the S1 deletions and none of which appear in the §5.3b table.

| Contributor | reads | writes | emits (today's names) |
|---|---|---|---|
| `bathymetry` | — | `elev` | `depth`, `depth_p25`, `depth_p75` |
| `sensors` (existing loop) | — | `sensor_times`, `water` (its water-union share) | `<pre>_sst/_cloud/_valid/_hour` |
| `landcover` | `elev` (fallback), `water` (sensor union) | `water` | `landmask`, `landcover_water` |
| `mur` | `water` | — | `mur_sst`, `mur_valid`, `mur_filled` |
| `met` (forcing) | — | `ref_utc` | `airtemp`, `wind_u`, `wind_v`, `wind_speed`, `swrad`, `cloud_cover`, `met_source` |
| `overpass_met` (derived) | `sensor_times` | — | `<pre>_<metvar>` (auto cross-product over sensors × `overpass_met` vars) |
| `tides` | — | — | `tide`, `tide_range` |
| `cmems` | `water` | — | `cmems_<var>`, `cmems_<var>_filled`, `cmems_source` |
| `derived:water_level` | `elev` | — | `<pre>_tide`, `<pre>_water_elev`, `<pre>_water_class` (needs the `datum` stage's offset) |
| `insitu` | `sensor_times`, `ref_utc`, `water` | — | `insitu_sst`, `insitu_n`, `<pre>_insitu_sst`, `<pre>_insitu_dt_min` |
| `derived:doy` | — | — | `doy_sin`, `doy_cos` |

> The `water` slot is written **collaboratively**: `bathymetry` supplies `elev` (contributes the
> `elev<0` term), each `sensors` scene contributes its water-union, and `landcover` folds those into
> the authoritative mask ([`datacube.assemble_aoi`](../src/coastal_sst_data/processes/datacube.py)
> lines ~514–523). Model this as `landcover` being the sole *writer* of the final `water` mask that
> `mur`/`cmems`/`insitu` read, with `elev` and the sensor unions as its declared `reads`, so the sort
> places it after both — that is exactly today's order and keeps Phase 2 byte-identical.

Topo-sort must yield today's evaluation order: `bathymetry, sensors → landcover → mur, cmems, insitu`
with `met`/`tides`/`overpass_met`/`water_level`/`doy` slotted where their reads allow. Phase 2's
golden test is what *proves* the derived order equals the hand order; if it diverges, a missing edge
(most likely a `water` reader that wasn't declared) is the first thing to check.

#### 5.3b Deletions-first / end-state wiring (no `water`/`elev` slots)

Per-source emissions (D4 names); the `water`/`elev` slots and the `mur`-fill / `water_level` / `daily
source` machinery are gone. This is what the protocol is built as directly in the default §4 order,
and what the §5.3a shape migrates *to* in the deletions-last branch.

| Contributor | reads | writes | emits (final, per-source) |
|---|---|---|---|
| `bathymetry` | — | — | `elevation_<src>`, `depth_<src>`, `depth_p25_<src>`, `depth_p75_<src>` |
| `sensors` (existing loop) | — | `sensor_times` | `<pre>_sst/_cloud/_valid/_hour` |
| `mur` | — | — | `mur_sst` (no fill; honest NaN gaps) |
| `met` (forcing) | — | `ref_utc` | `airtemp_<src>`, `wind_u_<src>`, `wind_v_<src>`, `wind_speed_<src>`, `swrad_<src>`, `cloud_cover_<src>` |
| `met_overpass` | `sensor_times` | — | `<pre>_<metvar>_<src>` — **only for the user-specified `(sensor, source)` combos** (D13/D14) |
| `tides` | — | — | `tide_<src>`, `tide_range_<src>` |
| `cmems` | — | — | `cmems_<var>_<src>` |
| `landcover` | — | — | `landcover_water_<src>` |
| `insitu` | `sensor_times`, `ref_utc` | — | `insitu_sst`, `insitu_n`, `<pre>_insitu_sst`, `<pre>_insitu_dt_min` |
| `derived:doy` | — | — | `doy_sin`, `doy_cos` |

The end-state topo-sort yields: `bathymetry, sensors, mur, met, tides, cmems, landcover` →
`met_overpass` → `insitu` → `doy`. The `sensors` contributor stays **collective** (covers every `SensorSpec` product),
so adding a fourth sensor remains pure-registry. `met_overpass` is now a **product** (D14) with its
own contributor keyed on the user's combos — not a derived auto-cross-product. The `datum` stage is
retained (D12) but no longer feeds a contributor — its offset ships as a per-source attribute.

**Deleted by the S1 simplification (D6/D7/D12):** `fill_water_nn`, the `water`/`landmask` block, the `elev`
and `water` slots, `mur_valid`/`mur_filled`, CMEMS `_filled` masks, the `src_channels`/`daily_sources`
block (`met_source`/`cmems_source`), the `derived:water_level` contributor
(`water_elev`/`water_class`/`<pre>_tide`), and the `station_pixels(..., water)` snapping argument.

### 5.4 The loud-omission invariant

Lives in `datacube.py` (which imports `products`; `products.py` still imports nothing internal, so
the import ladder is respected):

```python
def _check_contributors():
    keys = {c.key for c in CONTRIBUTORS}
    for s in products.REGISTRY:
        if s.sensor is not None:        # covered collectively by the `sensors` contributor
            continue
        if s.cube_opt_out:              # new ProductSpec field, default False
            continue
        if s.product.value not in keys:
            raise RuntimeError(
                f"{s.product.value}: no cube contributor registered. Add one to "
                "datacube.CONTRIBUTORS, or set cube_opt_out=True on its ProductSpec.")
    produced = {w for c in CONTRIBUTORS for w in c.writes}
    for c in CONTRIBUTORS:              # every `reads` slot must have a producer
        missing = set(c.reads) - produced
        if missing:
            raise RuntimeError(f"contributor {c.key!r} reads unproduced slot(s) {sorted(missing)}.")

_check_contributors()
```

Add `cube_opt_out: bool = False` to `ProductSpec`.

### 5.5 Phases of the protocol migration (stage S2; each independently reviewable, all gated on the S0 golden test)

| Phase | Change | Behavior |
|---|---|---|
| **0** | **Golden-cube test** in [`test_datacube.py`](../tests/test_datacube.py): assemble a fixture AoI, snapshot every channel's values + attrs. Every S2 phase must keep it identical. **The comparator MUST exclude the provenance/run-stamp attrs** — `created_at`, `code_version`, `package_version`, `config_yaml`/`config_sha256`/`config_path`, `provenance`, `provenance_products` — because any tree edit flips `code_version` to `…-dirty` and rewrites `created_at` ([`provenance.build`](../src/coastal_sst_data/provenance.py)), so a naïve full-attr snapshot would fail on every phase for reasons unrelated to the cube's data. | no prod change |
| **1** | **Mechanical extraction**: pull each inline block of `assemble_aoi` into a `_contribute_*(ctx)` function; the two surviving locals become `ctx.slots`. Still called in today's fixed sequence. | byte-identical |
| **2** | **The protocol**: add `AssemblyContext`, `Contributor`, `CONTRIBUTORS`, `_topo_order`; delete the fixed sequence. | byte-identical (proves derived order == hand order) |
| **3** | **The invariant**: add `cube_opt_out`, `_check_contributors()`; wire every product's contributor. | no output change |
| **4** | **Acid test**: new `tests/test_add_a_covariate.py` mirroring [`test_add_a_sensor.py`](../tests/test_add_a_sensor.py) — register a fake daily product + contributor, assert the cube gains its channels with correct values, and assert that **removing the contributor now raises**. | test-only |
| **5** | **Docs**: update [DEVELOPMENT.md §2/§3c](DEVELOPMENT.md) — non-sensor covariates now need `ProductSpec` + module + **a registered contributor**; §3c becomes "write a `contribute()`," no longer "hand-edit `assemble_aoi`." | docs |

> **The protocol migration (S2) is behavior-preserving** and byte-identical against its baseline. In
> the default §4 order the D6/D7/D12 simplifications have already landed in S1, so S2's baseline is the
> *simplified* cube; in the deletions-last branch they land after, and the golden test guards the
> migration against today's cube. Either way the golden test is the safety net for the whole migration.

---

## 6. Theme: multi-source acquisition (goal 2)

> Execution order lives in §4; this section is the design detail. It is done as vertical slices —
> stages **S3** (bathymetry end-to-end) and **S4** (replicate for met/cmems/tides) — not as one
> horizontal pass.

"Source" stops meaning *pick one* and becomes *a set, each acquired independently* — for **data**
sources only (D10).

> **Do this as vertical slices, not horizontal layers.** Take one product — **bathymetry**, the
> simplest data source (static, no reanalysis/forecast switch, no met split) — all the way through
> the bullets below *and* its per-source channels (§7.1) before touching met/cmems/tides (S3). The
> first slice proves the whole access/data + per-source-dir + config + channel + provenance pattern
> end-to-end on the easy case; met/cmems/tides then replicate it (S4), with CMEMS's product-switch
> and the met split as the known-hard variants. The bullets below are the mechanics each slice
> applies — not four separate horizontal passes.

> **Reality check — the dispatch machinery is currently wired to the *opposite* set of products,
> and this is the bulk of the multi-source lift's cost.** The registry `sources={module}` dispatch
> ([`pipeline._modules_for`](../src/coastal_sst_data/pipeline.py), `ProductSpec.module_for`) is used
> **today only by `landsat`, `landcover`, `insitu`** — which are exactly the D10 *access / pick-one*
> products we are **not** changing. The D10 *distinct-data* products (`bathymetry`, `met`, `cmems`,
> `tides`) are `module=` **singletons** that take `source`/`fallback` as **internal config options**
> and run the fallback chain *inside the module* (`bathymetry._fetch_with_fallback`,
> `cmems._resolve_chain`), writing one directory. So "make `sources` a loadable set and dispatch per
> source" is **not** a registry tweak that reuses existing plumbing — it is a rewrite of these four
> acquisition modules to (a) accept an externally-chosen source, (b) **delete** their internal
> fallback logic, and (c) write to a per-source directory. Budget the S3–S4 slices accordingly; the
> "just loop over sources" ease belongs to the per-source *emission*, not this acquisition rewrite.

- **`products.py`** — add the `access` vs `data` source distinction (D10). For data-source products,
  migrate `module=` → a `sources={cudem: <module>, gmrt: <module>}` map (the same module may serve
  several sources), `sources` becomes a loadable set, and drop `default_source` and every `fallback`
  option from `options`/`region_options`. `module_for(source)` then resolves per source for these too.
  Update `_check_registry` invariants (e.g. a `data` product needs ≥1 source; auth map still covers
  all sources — note `met`/`cmems` carry a single scalar `auth` today, so per-source auth needs the
  dict form if two sources ever differ in credentials).
- **acquisition modules** (`bathymetry.py`, `met.py`, `cmems.py`, `tides.py`) — remove the internal
  fallback chain; `acquire()` is dispatched once per source and writes that source's directory only.
- **`config.py`** — data-source products take `sources: [cudem, gmrt]` (a list); validation rejects an
  unknown source and an empty list. Region override remains "which sources have coverage here."
  Access-source products keep the single `source` selector.
- **`pipeline.py`** — the acquisition loop dispatches once **per (product × selected data source)**;
  `process_order`/`depends_on` unchanged in spirit (still topo-sorted).
- **Output layout** — aligned files gain a per-source directory:
  `BATHYMETRY/<source>/aligned/<aoi>/…`. `products.product_dirs()`, `datacube.PRODUCT_DIRS`,
  `store.done`, **`store.scan` + `store.REQUIRED_VARS`** (the `check`/`--repair` validate pass walks
  `root/<DIR>/aligned` and is keyed by `s.dir`; leave it and it will **silently stop validating every
  data-source product** — its `if not base.exists(): continue` makes the omission invisible), and
  `provenance.collect` all follow the new layout. Access-source products keep today's single directory.
- **`store` / naming** — unchanged filename stamps; the source lives in the directory, not the stem.

In a vertical slice the per-source *files* (this section) and the per-source *channels* (§7.1) land
together for that product — there is no separate "files exist but channels don't" checkpoint. Each
slice carries one product all the way to its cube channels and its reviewed golden diff.

### 6.1 Split `met` into `met` (forcing) + `met_overpass` (D14/D15)

Land this with the met vertical slice (stage S4.2), since it is a product/registry/config restructuring:

- **`met` (forcing)** — daily forcing at `reference_time` (mode config-selectable per O5). Its
  `depends_on=(eco,lst,modis)` is **removed** — forcing does not need the sensors. Config: `sources`
  (a list of data sources) + `reference_time` + `variables`. Emits standalone `<var>_<src>` channels.
- **`met_overpass` (new product)** — snapshots time-aligned to a sensor's chosen overpass. `depends_on`
  the sensors (it reads their overpass times). Config: a list of `(sensor, source)` **combinations**
  (region-overridable) + `variables`. Its aligned files are read at each day's chosen scene time (the
  existing `load_at_times` path), and its contributor emits `<pre>_<metvar>_<src>` for exactly those
  combos.
- **Module split** — today's `met.py` does both; separate the forcing acquisition from the
  overpass-snapshot acquisition (two modules, or one module exposing two `acquire()` entry points via
  two product specs). The overpass module keeps the "read sensor dirs for overpass times" logic; the
  forcing module drops it.
- **Registry** — two `ProductSpec`s (`met`, `met_overpass`), each source-selectable over the same met
  data sources. `met_overpass` sets `cube_opt_out=False` and registers its contributor; the
  loud-omission invariant (D8) then guarantees it can't be silently dropped.
- **`Kind` — add a new `OVERPASS_ALIGNED` member** (resolved; see the design note). `met_overpass`
  is declared `kind=Kind.OVERPASS_ALIGNED`, `sensor=None`. Use this exact name at S4.
- **Validation** — a `met_overpass` combination naming a sensor that isn't loaded, or a source with no
  coverage, fails config validation rather than silently producing an empty channel.

> **Design note — `Kind` for `met_overpass` (RESOLVED at S2 review: add `Kind.OVERPASS_ALIGNED`).**
> `met_overpass`'s aligned files are timestamped (`<aoi>_<YYYYMMDDThhmmss>.nc`) and read at *another*
> product's chosen overpass times via `load_at_times` — with no clearest-scene selection and no
> `SensorSpec`. Neither existing `Kind` fits: `DAILY_RASTER` is wrong on file shape (these are not
> `<aoi>_<YYYYMMDD>.nc` daily files), and `OVERPASS_SENSOR` is wrong on semantics *and* would break
> the registry invariant `(s.sensor is not None) == (s.kind == Kind.OVERPASS_SENSOR)`
> ([`test_products.py`](../tests/test_products.py)), since `met_overpass` carries no `SensorSpec`. So
> add a distinct `Kind.OVERPASS_ALIGNED` = "timestamped rasters read at another product's overpass
> times." `Kind` is descriptive metadata today (nothing branches on it; it only feeds that invariant),
> so this is a labelling choice that keeps the `sensor ⟺ OVERPASS_SENSOR` invariant intact. After S1,
> `met_overpass` is the *only* overpass-aligned non-instrument product — tide-at-overpass was dropped
> (D12; downstream reconstructs it from the daily `tide` series + `<s>_hour`) — so this one `Kind`
> covers the whole category. If that invariant is stated as a biconditional, widen it at S4 to
> "`sensor is not None` ⟹ `OVERPASS_SENSOR`" (one-directional) so `OVERPASS_ALIGNED` is permitted.

### 6.2 Config migration & back-compat (do not silently no-op an existing config)

These refactors change the config surface in ways that will **fail or silently no-op** existing user
configs. Each change needs an explicit, loud migration — a silently-ignored key is the exact failure
mode this codebase is built to reject.

| Existing config | Change | Required behaviour on an un-migrated config |
|---|---|---|
| `bathymetry.source` + `bathymetry.fallback` (and `met`/`cmems`/`tides` likewise) | `source`/`fallback`/`default_source` removed; replaced by `sources: [..]` | Validation **rejects** the removed keys with a message pointing at `sources: [..]` — never accept-and-ignore. |
| `datacube.overpass_met: [airtemp, wind_speed]` (auto cross-product over all sensors) | Moves to the `met_overpass` product's `combinations: [(sensor, source), …]` | A config with the old key (or with no `met_overpass` block) must **fail validation**, not quietly produce **empty overpass channels**. This is the sharpest silent-regression risk: today every sensor auto-gets overpass met; after D13 only listed combos do. |
| `datacube.fill_mur_water` / `fill_cmems_water` / `water_level` flags | Removed with the fills / water-level channels | Reject the removed keys (they now describe nothing); re-gate the `datum` stage off the `water_level` flag (see §7.2). |

Provide a one-shot migration note in the README/CHANGELOG and, where cheap, a validation error string
that shows the before→after config shape. The test suite gains a case per removed key asserting the
loud rejection.

---

## 7. Theme: per-source channels + raw-output simplification (goals 2–3)

> Execution order lives in §4; this section is the design detail. Its two halves run at **different**
> times: the raw-output simplification (§7.2) is stage **S1** (early, deletions-first — or last), while
> per-source emission (§7.1) rides the multi-source slices (**S3–S4**) and the provenance/gap-table
> finish (§7.3–§7.4) is **S5**. They are documented together for cohesion, not because they execute
> together.

### 7.1 Per-source channels (D3/D4/D5/D10)

- Each **data**-source contributor loops over the AoI's active sources, emitting `_<source>`
  channels (D4 naming) — the same pattern the `sensors()` loop already uses.
- `bathymetry` now also publishes **raw `elevation_<src>`** (not just depth), so downstream can
  compute water level (D12).
- CMEMS: `cmems_<var>_<source>`. Tides: `tide_<source>`, `tide_range_<source>`. Landcover:
  `landcover_water_<source>`.
- Met (forcing): standalone `<var>_<source>` (e.g. `airtemp_hrrr`, `airtemp_era5`).
- Met (overpass): `<sensor>_<var>_<source>` for **only** the user-specified `(sensor, source)` combos
  (D13) — e.g. if the config lists `{lst, hrrr}` and `{eco, era5}`, only `lst_airtemp_hrrr` and
  `eco_airtemp_era5` (× the overpass `variables`) are emitted, not the full cross-product.
- Access-source products (Landsat sensors, single-network in-situ) emit single-channel as today.

### 7.2 Deletions (D6/D7/D12)

- Remove `fill_water_nn`, the `fill_mur_water` / `fill_cmems_water` config, and the `_filled` /
  `mur_filled` / `mur_valid` channels.
- Remove the `water` union slot, the derived `landmask` channel, and the `station_pixels(..., water)`
  snapping (call with `water=None`; the `SNAP_WARN_M` path goes dormant/removed).
- Remove the `derived:water_level` contributor and the `<sensor>_water_elev` / `<sensor>_water_class`
  / `<sensor>_tide` channels. Publish `elevation_<src>` + `depth_<src>` + `tide_<src>` + `<pre>_hour`
  as the raw ingredients, and the **datum offset** as per-source attributes (`datum_offset_m`,
  `datum_status`). Keep the datum derived stage (D12).
- Remove the `daily_sources` / `src_channels` block and the `met_source` / `cmems_source` channels
  and their legend attrs — one channel per source makes the per-day source code redundant.
  `provenance.daily_sources` is deleted.
- **`coverage_channel` must move per-source.** `datacube.DAILY_CHANNELS` is derived from
  `ProductSpec.coverage_channel` (today `met`→`"airtemp"`, `tides`→`"tide"`). After D4/D5 those exact
  channels no longer exist (`airtemp_hrrr`, `tide_coops`, …), so the thin-coverage warning silently
  stops firing. And with stacking, a source with **no regional coverage is all-NaN by design** — so
  "is this product thin?" must become "does *any* loaded source have a finite day," not one fixed
  channel. Update `coverage()` / `DAILY_CHANNELS` to judge a product present if any of its per-source
  channels is finite (the CMEMS `sorted(cm)[0]` prefix trick already does a version of this and can
  be generalised).
- **Datum pipeline gate.** Datum runs today only `if DataProduct.bathymetry in selected and
  project.datacube.water_level` ([`pipeline.run_pipeline`](../src/coastal_sst_data/pipeline.py)). With
  the `water_level` cube channels (and likely the `datacube.water_level` config flag) gone, re-gate
  datum so it still runs to publish the per-source `datum_offset_m`/`datum_status` attributes (D12) —
  e.g. gate on bathymetry being selected alone. Do not let the flag removal silently disable datum.
- **O4 is already resolved (D13);** no decision blocks this phase.

### 7.3 Provenance (`provenance.py`)

- `field_inputs` gains a `<var>_<source>` suffix rule; per-source channels attribute to their one
  source unambiguously — simpler and more precise than today's per-day source union.
- `collect` / `collect_product` follow the per-source directory layout from the S3–S4 slices.
- Delete `daily_sources`. Datum offset attributes still sourced from the retained datum stage.

### 7.4 README source-coverage gap table (D9)

A first-class deliverable: per product, which sources cover which regions / time ranges, so the user
can decide what to stack. Gaps to document:

- **bathymetry** — CUDEM is CONUS-only; GMRT is global but coarser. Outside CONUS, stack GMRT (or a
  region-specific DEM).
- **met** — HRRR is North America only; ERA5 is global. Outside NA, stack ERA5.
- **CMEMS** — regional models (Baltic/Med/NW-Shelf) vs the global model; forecast vs reanalysis
  windows.
- **tides** — CO-OPS gauges are U.S.-only; the global model covers elsewhere.

### 7.5 Golden test update

The golden snapshot is regenerated at **two** reviewed moments, not one:

- **At S1** (the deletions): fills gone, `landmask` gone, `water_level` gone, `_source`/`daily_sources`
  gone; raw `elevation`/`tide`/`<s>_hour` shipped.
- **At the end of S3–S4** (per-source emission): channels now `_<source>`, `elevation_<src>` added.

Each diff is itself part of that stage's review artifact. (In the deletions-last branch the two
diffs land in the opposite order; the diffs themselves are unchanged.)

---

## 8. Channel layout: before → after

Illustrative, assuming bathymetry stacked as `[cudem, gmrt]`, met as `[hrrr, era5]`, tides as
`[coops, model]`, `<s>` over sensors `{eco, lst, modis}`.

| Today | After |
|---|---|
| `mur_sst`, `mur_valid`, `mur_filled` | `mur_sst` |
| `cmems_thetao_0m`, `cmems_thetao_0m_filled`, `cmems_source` | `cmems_thetao_0m_glorys`, `cmems_thetao_0m_<other>` |
| `airtemp`, `met_source` (forcing) | `airtemp_hrrr`, `airtemp_era5` (forcing product `met`) |
| `depth`, `depth_p25`, `depth_p75` | `depth_cudem`, `depth_gmrt`, `depth_p25_cudem`, …; **+ `elevation_cudem`, `elevation_gmrt`** |
| `landmask`, `landcover_water` | `landcover_water_<src>` (per source); **no `landmask`** |
| `<s>_water_elev`, `<s>_water_class`, `<s>_tide` | **dropped** (downstream computes from `elevation_<src>` + `tide_<src>` + `<s>_hour` + datum attrs) |
| `tide`, `tide_range` | `tide_coops`, `tide_model`, `tide_range_coops`, … |
| `insitu_sst` (snapped) | `insitu_sst` (unsnapped) |
| `<s>_<metvar>` (overpass met) | `<s>_<metvar>_<metsrc>` — **kept**, only for user-specified `(sensor, source)` combos (D13); now produced by the separate `met_overpass` product (D14) |
| `<s>_sst/_cloud/_valid/_hour` | unchanged (single-source sensors) |
| `doy_sin`, `doy_cos` | unchanged |

---

## 9. Test strategy

- **Golden-cube snapshot** ([`test_datacube.py`](../tests/test_datacube.py)) — the safety net for the
  whole migration (run-stamp attrs excluded); regenerated with review at **S1** (deletions) and again
  at the end of **S3–S4** (per-source emission), per §7.5.
- **Acid test for a non-sensor covariate** (`tests/test_add_a_covariate.py`, new) — the executable
  proof that a new covariate is picked up by every derived table *and* that forgetting its
  contributor now raises.
- **Multi-source tests** — a product with two stacked sources produces both channels with correct
  per-source NaN patterns; an unknown source fails validation; per-source output dirs are read
  correctly by assembler + provenance; an access-source product still yields a single channel.
- **Provenance tests** — `field_inputs` maps `<var>_<source>` channels to the right single source;
  datum offset attributes present per source.
- Existing per-product tests updated for the removed channels and the per-source layout.
- Suite stays offline (network stubbed), per current convention.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Regression in a correctness-critical module | Golden-cube test written **first** (comparator excludes run-stamp attrs, §5.5); every S2 phase gated on it byte-identical. |
| **Protocol built in the wrong wiring shape for the branch** (the §5.3b 2-slot shape while `water` still exists) | §5.3 ties each shape to the §4 fork — §5.3b only after the S1 deletions, §5.3a otherwise; the `water` slot's 3 writers / 4 readers are stated explicitly, and S2's golden test fails loudly if an edge is missing. |
| Ordering drift (hand-order relocated into a list) | Order is topo-derived from slot reads/writes and *verified* against today's order by Phase 2's golden test (D2). |
| Slot-name typos (string keys) | `SLOT_*` constants; `_check_contributors` asserts every `reads` has a producer. |
| **Multi-source lift under-scoped** — the four distinct-data modules need internal-fallback → external-per-source rewrites, not a registry tweak (§6 reality-check) | Called out explicitly with the per-module work listed; done as vertical slices (S3 bathymetry first, then S4) so the pattern is proven on the easy case before the hard ones. |
| **Existing configs silently no-op** after removing `fallback`/`source`/`overpass_met` | §6.2 migration table: validation **rejects** every removed key loudly; a test per key asserts the rejection; the `overpass_met` → `met_overpass` move fails rather than emitting empty channels. |
| **`store.scan` / `coverage_channel` silently stop firing** under the per-source layout | Both listed in the §6 layout list and the §7.2 deletions; scan walks the per-source dirs, and coverage judges "any source finite," with round-trip tests. |
| Losing overpass-time met (was O4) | **Resolved (D13):** overpass met is kept, but only for user-specified `(sensor, source)` combos — no auto cross-product. |
| Silent per-source dir mismatch after the S3–S4 slices | Provenance + assembler read dirs from one `PRODUCT_DIRS` map; tests assert round-trip. |
| Losing the fill/mask/water-level value users relied on | Intentional (Goal 3); the raw ingredients to reconstruct all of them ship in the cube, and the datum offset ships as an attribute. **Coordination risk:** any downstream consumer (e.g. sibling model repos) must implement the reconstruction before this lands, or it is a net regression — sequence the downstream change with the S1 deletions (and see the §4 fork). |

---

## 11. Explicitly out of scope

- **Pure-declarative assembly** (all assembly logic as `ProductSpec` data). The validity logic is
  genuinely code; forcing it into flags would recreate `SensorSpec`'s flag-explosion for products
  that are not a uniform family. The win is a *uniform mechanism and a loud failure*, not zero code.
- **Colocating contributors in each process module.** D1 keeps them centralized in `datacube.py`.
- **Per-source treatment of redundant-access sources** (Landsat pipes, ESA/WorldCover) — single
  channel, pick-one (D10).
- **Shipping met at sub-daily resolution.** Overpass met (D13) ships one snapshot per user-specified
  `(sensor, source)` combo, not a sub-daily series.

---

## 12. Execution checklist

Stages per §4 (**deletions-first** branch). For the deletions-last branch, move all of **S1** to the
end and build **S2** in the §5.3a 4-slot shape instead of §5.3b.

**S0 — safety net**
- [x] **S0** Golden-cube test (run-stamp attrs excluded from the snapshot).

**S1 — raw-output simplification** *(⚠ downstream-gated; the reviewed deletions diff)*
- [x] **S1.1** Delete fills (`fill_water_nn`, `fill_mur_water`/`fill_cmems_water`), `landmask`, in-situ snapping (`station_pixels(..., water=None)`), `water_level` channels, `daily_sources`/`_source` channels.
- [x] **S1.2** Ship raw ingredients (`elevation`, `tide`, `<s>_hour`) in place of the derived channels; retain the datum stage, publish `datum_offset_m`/`datum_status` as attrs, and re-gate the datum stage off the removed `water_level` flag.
- [x] **S1.3** Regenerate the golden snapshot with review (deletions diff).

**S2 — contributor protocol** *(byte-identical vs the S1 baseline; built in the §5.3b 2-slot shape)*
- [x] **S2.1** Extract inline blocks to `_contribute_*`; the two surviving locals become `ctx.slots` (`sensor_times`, `ref_utc`).
- [x] **S2.2** `AssemblyContext` + `Contributor` + `CONTRIBUTORS` + topo-sort orchestrator.
- [x] **S2.3** `cube_opt_out` + `_check_contributors`; wire every product's contributor.
- [x] **S2.4** `test_add_a_covariate.py` acid test.
- [x] **S2.5** DEVELOPMENT.md §2/§3c: a non-sensor covariate now needs a registered contributor, not a hand-edited `assemble_aoi`.

**S3 — multi-source vertical slice: bathymetry end-to-end**
- [x] **S3.1** `access`/`data` split in `products.py`; `bathymetry` `module=` → `sources={}`; remove its internal fallback chain.
- [x] **S3.2** Per-source layout for bathymetry across `store.done`, **`store.scan`/`REQUIRED_VARS`**, `product_dirs`, `provenance.collect`; config `sources: [..]` with empty/unknown rejection.
- [x] **S3.3** Bathymetry per-source channels (`depth_<src>`, `elevation_<src>`, `depth_p25/p75_<src>`) + provenance — proving the whole pattern on one product.

**S4 — replicate the slice.** The bundled S4.2 is split into per-source *layering* (mechanical S3
replication) and the *new overpass products*, each its own reviewed golden diff.
- [x] **S4.1** `cmems` per-source (data sources: reanalysis/forecast + regional models); remove its internal fallback chain. Golden diff.
- [x] **S4.2** `tides` per-source (data sources: co-ops / model); remove its internal fallback chain. Golden diff.
- [x] **S4.3** `met` FORCING per-source (`<var>_<src>`), drop `depends_on=(sensors)`; keep the existing overpass-met contributor unchanged for now. Golden diff.
- [x] **S4.4** `met_overpass` as a real product (D14): `DataProduct` + `ProductSpec` (`kind=Kind.OVERPASS_ALIGNED`), module split, its OWN `combinations: [(sensor, met_source)]` config (D15), `<sensor>_<var>_<src>` for user combos only (D13). Golden diff.
- [x] **S4.5** `tide_overpass` (D17): a DERIVED contributor emitting `<sensor>_tide_<src>` for its OWN `combinations: [(sensor, tide_source)]` (interpolate the per-source tide series to `<s>_hour` via `water_level.tide_at_overpass`). Golden diff.
- **Combos are PER PRODUCT (S4-review decision):** `met_overpass` and `tide_overpass` each carry their own `combinations` list referencing their own product's sources (met vs tide) — independent pairings, not one shared sensor list (which would recreate the sensor×source cross-product D13 rejects). A combo's sensor must be loaded and its source must be one of that product's sources (validated at config load).
- [x] **S4.6** Config migration: loudly reject removed `source`/`fallback`/`default_source`/`overpass_met` keys (§6.2) + a test per key; add the `met_overpass`/`tide_overpass` combos config.
- [x] **S4.7** Update `coverage_channel`/`coverage()` to "any loaded source finite".

**S5 — finish**
- [x] **S5.1** Provenance per-source (`<var>_<source>` field mapping; delete `daily_sources`).
- [x] **S5.2** README source-coverage gap table.
- [x] **S5.3** Final golden snapshot regeneration with review (per-source emission diff).
