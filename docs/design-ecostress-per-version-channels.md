# Design — per-version ECOSTRESS channels

**Status:** implemented (2026-07-18); this document is the design of record
**Author:** drafted with Claude, 2026-07-18
**Issue / branch:** `26-add-each-ecostress-version-as-a-unique-channel`

> **Implementation note.** One detail beyond this plan: because the config names the stacked
> sources `versions` (not `sources`), a `sources_option` field was added to `ProductSpec`
> (default `"sources"`, ECOSTRESS sets `"versions"`), and `config._stacked_source_lists_are_valid`
> reads that key. A second, non-obvious touch-point surfaced during implementation: the
> `met_overpass` acquisition also discovers sensor overpass times from disk, so its
> `_sensor_dirs` now expands a stacked-data sensor to its per-version trees (§5 covered only the
> datacube assembler). Everything else matches the plan below.
**Scope:** ship ECOSTRESS collection versions (v002, v003) as **distinct stacked channels** in the
datacube, mirroring the D10 "one channel per source" pattern already used by CMEMS, bathymetry,
tides, and met — while keeping a single overpass identity so the met/tide/in-situ matchup surface
does not fork.

This consolidates the design discussion into one reviewable plan. It supersedes any earlier
informal notes. All three open decisions are **resolved** (see [§2](#2-confirmed-decisions)); no
decision blocks implementation. Execution order is set in [§6](#6-sequencing).

---

## 1. Goals & motivation

1. **Cover the full time range no single ECOSTRESS collection spans.** The two collections have
   asymmetric temporal coverage (Puget Sound probe, 2026-07-18):

   | version | coverage | role |
   |---|---|---|
   | **v002** | 2018-07-29 → ~2024+ | starts ~13 months earlier; covers 2018-07 → 2019-08 that v003 lacks |
   | **v003** | 2019-08-23 → present (2026-07) | newer reprocessed collection; reaches the present |

   Neither alone spans 2018→now. Today the config takes a single scalar `version`, so a cube can
   carry only one collection. This is the same coverage-gap problem the D10 stacking model already
   solves for CMEMS (reanalysis + analysis/forecast).

2. **Do it the D10 way — one channel per source, no fallback.** Rather than a hidden preference
   merge into one `eco_sst` (which the codebase deliberately removed elsewhere), ship
   `eco_sst_v002` and `eco_sst_v003` as distinct, self-describing channels. The consumer picks
   per day; a version that does not cover a day is an honest NaN slice.

3. **Do NOT explode the matchup surface.** v002 and v003 are largely reprocessings of the *same*
   physical overpasses, so the overpass time per day is usually identical across versions. The
   met-overpass, tide-overpass, and in-situ matchups therefore keep a **single `eco` overpass
   identity** rather than forking per version.

---

## 2. Confirmed decisions

Settled during the design discussion; fixed for this plan.

| # | Decision | Consequence |
|---|---|---|
| D1 | **Data channels only.** Ship per-version raw sensor channels; keep ONE `eco` overpass identity for downstream matchups. | `eco_sst_v002/_v003`, `eco_valid_*`, `eco_cloud_*`, `eco_hour_*` fork per version; `eco_airtemp_<src>`, `eco_tide_<src>`, `eco_insitu_*` stay singular. No config churn, no doubled matchup channels. |
| D2 | **Tag-last naming:** `eco_sst_v002`. | Consistent with the D10 convention everywhere else (`cmems_thetao_my_global`, `depth_cudem`, `tide_coops`). |
| D3 | **Clean break on config.** Replace scalar `version` with a `versions` list; a config still setting `version` **fails validation** with a message pointing to `versions`. | Mirrors the bathymetry `source`/`fallback` and cmems removals — no silently-ignored keys. |
| D4 | **Source tag = `v002`/`v003`; Earthdata collection version = the tag minus the leading `v`.** | The subtree dir (`ECOSTRESS/v002/aligned`) and cube tag (`eco_sst_v002`) are the same token, so tag-last naming falls straight out of `aligned_rel()`, exactly like CMEMS source tags. The acquisition strips `v` for the Earthdata `version=` query. |
| D5 | **Single `eco` overpass stream is preference-ordered by the config `versions` list.** Prefer the first-listed version's chosen scene per day, fall back to the next. | `versions: ["v003", "v002"]` → `sensor_times["eco"]` uses v003's scene where present, else v002's. The list order IS the preference order. |
| D6 | **ECOSTRESS becomes the first stacked-DATA *sensor*.** | Reuses the existing `sources=` + `SourceKind.DATA` + `aligned_rel()` + pipeline fan-out + provenance descent. The only genuinely new code is the sensor contributor's per-version branch. |

---

## 3. Why this is not just "another stacked-data product"

CMEMS/bathymetry/tides/met are stacked-DATA products read by **hand-written contributors** that
already loop over per-source dirs. ECOSTRESS is different: it is a **sensor**, handled collectively
by [`datacube._contribute_sensors`](../src/coastal_sst_data/processes/datacube.py) over
`products.sensors()`, and the sensor machinery is **prefix-based and flat**:

- It reads the flat `ECOSTRESS/aligned/<aoi>` dir (`ctx.adir(product)`, `source=None`).
- It emits `eco_sst / eco_cloud / eco_valid / eco_hour` from the sensor's single `prefix`.
- It registers **one** `sensor_times["eco"]` stream, which
  [`_contribute_met_overpass`](../src/coastal_sst_data/processes/datacube.py),
  [`_contribute_tide_overpass`](../src/coastal_sst_data/processes/datacube.py), and
  [`_contribute_insitu`](../src/coastal_sst_data/processes/datacube.py) all align to.

No existing sensor uses stacked-DATA sources (Landsat has `sources=` but it is `SourceKind.ACCESS`
— pick-one, flat tree, one channel). So **"a sensor that is also stacked-DATA" is new ground.** The
blast radius is deliberately contained to `_contribute_sensors` by D1/D5: everything downstream
keys off `sensor_times["eco"]`, which stays singular.

---

## 4. Channel layout

Per-version data channels; matchups unchanged.

```
3D (time,y,x):  eco_sst_v002,   eco_sst_v003
                eco_valid_v002, eco_valid_v003
                eco_cloud_v002, eco_cloud_v003
1D (time):      eco_hour_v002,  eco_hour_v003

matchups UNCHANGED (single 'eco' overpass identity, D1/D5):
                eco_airtemp_<src>, eco_wind_u_<src>, ...      (met_overpass)
                eco_tide_<src>                                 (tide_overpass)
                eco_insitu_sst, eco_insitu_dt_min             (in-situ)
```

`lst` (Landsat) and `modis` sensor channels are untouched.

---

## 5. Implementation

### 5.1 Registry — [products.py](../src/coastal_sst_data/products.py)
Convert the ECOSTRESS spec from `module=` to stacked-data:
- `sources={"v002": "coastal_sst_data.processes.ecostress", "v003": "coastal_sst_data.processes.ecostress"}`,
  `source_kind=SourceKind.DATA`, drop `module=`.
- Keep `sensor=SensorSpec(prefix="eco", water_is_land=True, use_cloud=False, qc_levels=(0, 1))`.
- Options: drop `version`, add `versions`.
- Leave `versions` **project-level** (not `region_options`): versions are a data-availability /
  time-range choice, not a regional-coverage one. (Revisit only if a region genuinely needs a
  different collection set.)
- `_check_registry`: confirm the **sensor + DATA** combination passes the existing invariants (one
  shared module, no `default_source`, unique prefix). Add an assertion if any existing check
  implicitly assumed sensors are `module=`.
- Known sources are `{v002, v003}`; validate the `versions` option against these (extensible later,
  like cmems `datasets`, if further collections appear).

### 5.2 Acquisition — [ecostress.py](../src/coastal_sst_data/processes/ecostress.py)
- `_ds_cfg`: replace scalar `version` with `versions` list (default `["v002"]` to preserve today's
  single-collection behavior). Reject legacy `version` (via config validation, §5.4).
- `_build_eff` / `acquire` / `run`: fan out over versions exactly like cmems/bathymetry — honor the
  `source` / `only_source` param the pipeline already threads for stacked-data
  ([pipeline.py:120-124](../src/coastal_sst_data/pipeline.py)).
- Output tree per version: `ECOSTRESS/<tag>/aligned/<aoi>/` via `products.aligned_rel(dir, tag)`.
  This also **fixes the filename-collision risk** of the interim two-pass workaround — the same
  overpass in each collection lands in its own subtree.
- `search_granules`: query Earthdata with `version=tag.lstrip("v")` per version (`v002` → `"002"`).
- Keep the per-file `source` attr self-stamp (`ECOSTRESS … v{version}`,
  [ecostress.py:207-209](../src/coastal_sst_data/processes/ecostress.py)) so provenance is correct.
- `store.done` in `run()` must point at the per-version `aoi_out` subdir.

### 5.3 Assembler — [`_contribute_sensors`](../src/coastal_sst_data/processes/datacube.py)
Branch on `spec.is_stacked_data`:
- **Stacked sensor (eco):** discover per-version dirs from disk (`ECOSTRESS/<tag>/aligned/<aoi>`,
  the same globbed-discovery the other stacked contributors use), and emit
  `eco_sst_<tag>`, `eco_valid_<tag>`, `eco_cloud_<tag>`, `eco_hour_<tag>` per version.
- **Single overpass identity (D5):** build **one** `sensor_times["eco"]` by merging the per-version
  chosen-scene times per day, preferring the first-listed version and falling back to later ones.
  met_overpass, tide_overpass, and in-situ remain keyed by `eco` and **unchanged**.
- **Non-stacked sensors (lst, modis):** unchanged flat-dir path.
- Update the channel-layout docstring ([datacube.py:44-58](../src/coastal_sst_data/processes/datacube.py)).

### 5.4 Config validation — [config.py](../src/coastal_sst_data/config.py)
- Add `versions` to the derived ECOSTRESS option surface.
- Explicitly reject the removed `version` key with a message pointing to `versions` (mirror the
  bathymetry `source`/`fallback` clean-break).
- Validate `versions` is a non-empty list of known tags.

### 5.5 Provenance — [provenance.py](../src/coastal_sst_data/provenance.py)
- **Descent is free:** `provenance.collect` already branches on `is_stacked_data` and globs
  `<DIR>/*/aligned` ([provenance.py:270-277](../src/coastal_sst_data/provenance.py)), so per-version
  trees are picked up once ECOSTRESS is stacked-data.
- **Verify** the per-field mapping attributes `eco_sst_v002` → tag `v002` under `sources_by_tag`
  ([provenance.py:168-190](../src/coastal_sst_data/provenance.py)) rather than the source union. The
  `eco_*` channels currently attribute via `provenance_inputs=("ecostress",)`; confirm the
  tag-suffix match resolves the version.

### 5.6 Store / done-guard
- `REQUIRED_VARS["ECOSTRESS"]` unchanged — the per-file contract (`sst, water, cloud, valid`) is
  identical within each version subtree.

---

## 6. Sequencing

1. **Registry** (§5.1) — the source of truth every other layer derives from.
2. **Config validation** (§5.4) — so `versions` parses and legacy `version` fails loudly.
3. **Acquisition** (§5.2) — per-version fan-out + subtree.
4. **Assembler** (§5.3) — per-version channels + single overpass merge.
5. **Provenance verify** (§5.5).
6. **Tests + docs** (§7, §8), running the suite throughout.

---

## 7. Tests
- [test_ecostress.py](../tests/test_ecostress.py): multi-version fan-out; per-version subtree
  layout; legacy-`version` rejection; Earthdata `version=` derivation from the tag.
- [test_datacube.py](../tests/test_datacube.py): per-version channels present; single `eco` matchup
  identity with preference-ordered merge (first-listed version wins, later fills gaps); lst/modis
  unaffected.
- [test_products.py](../tests/test_products.py): sensor + DATA registry invariants.
- [test_provenance.py](../tests/test_provenance.py): `eco_sst_v002` attributed to `v002`.

## 8. Docs / example config
- `examples/config.test.yaml`: ECOSTRESS block → `versions: ["v003", "v002"]` with a comment.
- README ECOSTRESS section; remove the stale scalar-`version` reference.

---

## 9. Risks & call-outs
- **New combination:** a sensor that is also stacked-DATA. Contained to `_contribute_sensors` by
  design (D1/D5). Ordering, auth (`earthdata`), and dispatch need no changes.
- **Preference-merge asymmetry (D5):** the single `eco` overpass stream *does* encode a per-day
  preference between versions — a deliberate, documented exception to "no fallback," justified
  because the versions describe the same physical overpasses. The *data* channels remain
  fallback-free (one per version); only the matchup-alignment time is merged.
- **Default `versions: ["v002"]`** preserves current single-collection behavior for existing
  configs that simply drop the now-invalid `version` key and add `versions`.

## 10. Verification (pre-merge)
- `pytest tests/test_ecostress.py tests/test_datacube.py tests/test_products.py tests/test_provenance.py`
- A small live check: one AoI, short date range, `versions: ["v003", "v002"]` — confirm both
  subtrees populate and the cube shows `eco_sst_v002` + `eco_sst_v003` alongside a single
  `eco_insitu_*` / overpass stream.
