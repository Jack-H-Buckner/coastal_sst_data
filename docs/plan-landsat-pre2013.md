# Landsat before 2013: what the archive actually holds, and what the pipeline had to change

*Status: IMPLEMENTED 2026-08-24. Prompted by a request to extend the coastal SST record back
past the Landsat 8 era. Everything marked ✅ was verified live against the Planetary Computer
STAC and against real COGs on the day this was written; it is recorded here so the next person
does not re-derive it.*

---

## 1. The headline: acquisition already worked

`landsat_pc.py` was written for Landsat 8/9, but the pre-2013 missions are in the **same
`landsat-c2-l2` collection**, produced by the **same single-channel algorithm** (`st_1.3.0`,
identical version string in the L4–7 and L8–9 product guides), with the **same scale and
offset**. ✅ Read from a real 1995 Landsat 5 item:

```
lwir     -> {"unit": "kelvin", "scale": 0.00341802, "offset": 149.0, "nodata": 0}
green    -> {"scale": 2.75e-05, "offset": -0.2, "nodata": 0}
nir08    -> {"scale": 2.75e-05, "offset": -0.2, "nodata": 0}
qa_pixel -> {"unit": "bit index", "nodata": 1}
cdist    -> {"unit": "kilometer", "scale": 0.01, "nodata": -9999}
```

Every asset `scene_to_dataset` reads is present for TM and ETM+, and the only difference is the
thermal asset's **name** — `lwir` rather than `lwir11` — which the module had handled since it
was written. `platforms` was already a validated config option.

**So the work was never enablement.** It was the four things that mislead once the record spans
three sensors and forty years, and they are what §3 covers.

Worth stating plainly, because it is the opposite of the usual expectation: **L8/9's second
thermal band buys nothing here.** C2 Level-2 uses band 10 only and applies no split-window, so
TM band 6, ETM+ band 6 and TIRS band 10 all go through one identical retrieval.

## 2. What exists, and what does not

| Mission | Sensor | Thermal | Native GSD | C2 L2 ST record ✅ |
|---|---|---|---|---|
| Landsat 1–2 | MSS | **none** | — | — |
| Landsat 3 | MSS | band 8 (10.4–12.6 µm) | ~238 m | **not distributed** |
| Landsat 4 | TM | band 6 | 120 m | 1982-08-22 → 1993-11-18 |
| Landsat 5 | TM | band 6 | 120 m | 1984-03-05 → **2012-05-05** |
| Landsat 7 | ETM+ | band 6 (dual gain) | 60 m | 1999-05-28 → 2024-01-19 |
| Landsat 8/9 | TIRS | band 10 | 100 m | 2013-02 → present |

✅ Verified directly: MSS items live in `landsat-c2-l1` and serve `green / red / nir08 / nir09`
and nothing else. **There is no thermal band, at any processing level, for Landsat 1–3.** The
Landsat 3 MSS band 8 did exist in orbit but degraded within a year (of 374 daytime scenes, two
were rated good quality) and was never distributed in Collection 2. *The Landsat SST record
cannot begin before 1982, at any effort.*

Two traps worth knowing:

- **Landsat 5 thermal ends November 2011**, not 2012 or 2013. The tail to 2012-05 is the MSS
  backup instrument, which has no thermal band. ✅ Consistent with a probe over a Tillamook-sized
  AoI: 72 L5 scenes in 2010, **zero** in 2012.
- **Landsat 4 is not a time series.** ✅ Ten scenes across 1982–1994 over that same AoI. It is in
  `THERMAL_PLATFORMS` because it is real data, but treat it as opportunistic single scenes.

Measured scene counts over that AoI, which is the honest picture of what a long record looks
like: L5 27/yr (1984) rising to ~70–80/yr (1995–2010); L7 44–92/yr. Before cloud rejection.

## 3. What changed, and why

### 3.1 Missions with no thermal product are refused at config time

`_read_asset` indexes `item.assets[key]` with no guard, so an MSS scene raised `KeyError` inside
the per-scene handler in `run` — one `rep.fail` per scene. An all-MSS date range therefore
produced hundreds of failures and no output, which reads like a network outage rather than a
request for data that does not exist. `resolve_platforms` now normalizes and refuses, naming the
platform and saying why.

The refusal *raises* rather than dropping the bad name: a config asking for Landsat 1 wants a
1970s record, and quietly running the rest would deliver a 1984-onwards cube that looks like a
successful answer to a question about 1972.

### 3.2 `lst_platform` — which mission produced each date

This is the substantive addition, and the reason is a number:

| Sensor | Mean bias vs. offshore buoys | RMSE |
|---|---|---|
| Landsat 5 TM | **−0.28 °C** | 0.96 °C |
| Landsat 7 ETM+ | +0.10 °C | 1.05 °C |
| Landsat 8 TIRS | **+0.65 °C** | 0.98 °C |

(Wachmann et al. 2024, *Remote Sensing* 16(5):920 — C2 ARD Level-2 ST, NE Pacific, >10 km
offshore, bulk→skin corrected.)

**A ~0.9 °C TM→TIRS step with essentially identical scatter.** Collection 2 did *not* deliver a
homogenized thermal record. That step lands exactly at the 2013 mission boundary and is the same
order as several decades of regional coastal warming — so merged and unlabelled, it is
indistinguishable from a trend.

Landsat still merges into one `lst_*` channel set, and that is right: unlike MODIS Terra/Aqua
(two spacecraft, two orbits, diverging equator crossings — hence stacked), every Landsat flies
the same ~10:00 descending WRS-2 orbit, so time-of-day is consistent and splitting per mission
would only force a re-merge downstream. But the mission is now recorded:

- `landsat_pc.scene_to_dataset` stamps `platform_id` (and `platform`, `instrument`) on each
  granule. A **global attr**, not a layer — it describes the whole granule.
- `datacube.load_clearest_overpass` carries the **base granule's** id per day, chosen by exactly
  the rule that picks `_hour`, so a mosaicked day reports one mission alongside one instant.
- `_contribute_flat_sensor` emits `lst_platform`, 1-D `(time,)`, int8, `0` = not recorded.

It could not be a global attr on the cube: `_merge_block_attrs` treats a block-varying attr as
fatal, and platform varies per scene by construction.

**Block invariance is why `platform_available` exists.** `lst_platform` is the second channel
(after `_footprint_id`) whose existence depends on file *contents* rather than on which files
exist. Decided per block it would appear in some and vanish from others, and a cube whose
variables disagree on their time length does not fail to write, it fails to *read*
(`conflicting sizes for dimension 'time'`). A partly re-acquired tree — 2013-onwards granules
written before the label existed, pre-2013 ones written after — is the **ordinary** state during
this migration, not a corner case.

### 3.3 SLC-off gaps must not read as "clear"

Landsat 7's Scan Line Corrector failed 2003-05-31. Every subsequent scene carries diagonal
no-data wedges — ~22% of the scene, absent at nadir and widening toward the edges.

`QA_PIXEL` bit 0 is **fill**, and it is not a cloud bit. A fill pixel carries bit 0 and *nothing
else*, so the cloud/shadow/dilated test came out false and the cell read `cloud = 0` — clear sky
— where the sensor recorded nothing at all.

✅ Measured on `LE07_L2SP_047029_20040927_02_T1`, a 1000×1000 window at scene centre:

```
fill(bit0)=3.5%   lwir==0 nodata=3.5%
of fill px, cloud-bits set: 0.0%  -> 100.0% read CLEAR
unique qa in fill px: [1]
fill <-> lwir-nodata agreement: 100.0%
```

Every gap pixel is `QA_PIXEL == 1` exactly, and gap agrees perfectly with thermal nodata. (3.5%
is the scene-*centre* figure — the wedges widen toward the edges to the ~22% scene average.)

`valid` was never wrong; the `isfinite(sst)` gate caught these cells. But `lst_cloud` is a
published channel in its own right. `cloud` is now NaN wherever bit 0 is set, and
`load_clearest_overpass` no longer flattens NaN to 0 on the way into the cube — NaN is already
that channel's value for a day with no scene at all, so 0 was claiming *more* about an equally
unobserved cell. The conservative reading downstream is unchanged: `_read_granule` still counts
a NaN cloud cell as cloudy for validity.

**No gap-filling**, deliberately. Nothing is gap-filled upstream (the Phase One/Two products
ended in 2008), the retained 78% is radiometrically identical to SLC-on, and standard fill
methods interpolate from *other dates* — which for an instantaneous SST field means inventing
temperatures in a spatially systematic pattern.

#### 🚩 The bigger half of the same defect, found while verifying the above

Chasing SLC-off turned up a **much larger** instance of the same problem, and it is **not a
pre-2013 issue at all — it has always affected Landsat 8/9 too**.

`read_cog_window` fills an unmasked read with `nodata=0` by default, because NaN is not
representable in an integer band. For `qa_pixel` that is the worst possible choice: **0 is a
perfectly valid QA word meaning "no flag set" — clear.** A Landsat scene is a rotated
parallelogram, so an AoI straddling a scene edge is largely *outside* the source raster, and
every one of those cells was reprojected in as 0 and read as cloud-free.

✅ Measured on a real acquisition (Landsat 7 over Tillamook, 2004-06-07, 25 km buffers), before
and after:

| | sst NaN | cloud NaN | agreement | cells with no temperature claiming CLEAR |
|---|---|---|---|---|
| before | 81.6% | 53.6% | 72.0% | **72,278** |
| after | 81.6% | 81.6% | **100.0%** | **0** |

The fix is to read `qa_pixel` with **its own declared fill value** (`nodata=1` — bit 0, fill),
which the catalogue publishes. Off-footprint cells and SLC-off gaps then arrive on the same
footing and one bit-0 test handles both. ✅ Across all eight scenes of the re-acquired window,
`sst` NaN and `cloud` NaN now agree at 100.0% with zero false-clear cells.

`valid` was correct before and after — this never corrupted the temperature record — but any
analysis that read `lst_cloud` directly was being told a scene edge was clear sky.

### 3.4 A forty-year STAC search is windowed

`search_scenes` did `list(search.items())` inside one `net.retry`. Fine at 2013-onwards; at
1984–2026 it is thousands of scenes and dozens of pages, paged to exhaustion inside the retry,
so a failure on the last page discarded every page before it and nothing downloaded until the
whole multi-decade search had succeeded. Now searched in ≤366-day windows, each its own retry.
Windows are disjoint and cover the range exactly, so no de-duplication is needed, and the skip
guard starts doing useful work on a resumed run almost immediately.

### 3.5 L2SR scenes are skipped, not failed

Roughly 8% of the archive has surface reflectance but no surface temperature
(`landsat:correction == "L2SR"`). On Planetary Computer that surfaces as a **missing thermal
asset key**, not a masked band, and the STAC query here does not filter on it.
`scene_to_dataset` returns `None` — `run` counts that as nothing to write. A scene that never
had a temperature band is not a broken download.

## 4. Corrections to things people believe

- 🚩 **"USGS says Level-2 ST is not validated over water."** No such statement exists in the C2
  ST page, LSDS-1618, LSDS-1619, LSDS-1330, or the FAQs. It is a misremembering of the
  **Surface Reflectance** docs, which *do* warn about water ("not ideal for water bodies due to
  the inherently low level of water leaving radiance") and list "coastal regions where land area
  is small relative to adjacent water" as an adverse condition — for SR, not ST. Do not
  propagate it, and do not let it stand unchallenged in review.
- **What the algorithm actually does over water** is more specific and more useful. ASTER GED
  covers non-frozen *land*; where emissivity is missing over water, the product substitutes a
  hard-coded **ε = 0.988**. ✅ Confirmed on real `ST_EMIS` COGs — three sites × three missions,
  every open-water pixel reads exactly 0.9880 with `ST_EMSD = 0`. Two consequences: the
  spectral-library value is 0.9926, so retrieved temperature runs **≈ +0.2 K warm**; and because
  `ST_EMSD` is zero there, **`ST_QA` under-represents uncertainty over water**. The atmospheric
  term (~0.8 K) is roughly 4× larger than the emissivity term either way.
- **The existing bits-1|3|4 cloud test is correct and should not be "improved" to bit 6.** USGS
  documents a Landsat 4–7 bug where the Clear bit is erroneously off when shadow, snow and water
  are all on, and explicitly recommends the dilated-cloud AND cloud OFF condition instead — which
  is what this module already does. Relatedly, bits 10–13 carry no independent information on
  TM/ETM+ (they mirror bits 2/4/5), so any filter on "medium-confidence shadow" filters on
  nothing.

## 5. Known-open follow-ups

Not implemented here; recorded so they are not rediscovered as novel.

1. ~~The NDWI water test admits bright cloud.~~ **FIXED — see §6.**
2. **Sensor-dependent shoreline buffering.** Mixed pixels extend ~2 native thermal pixels from
   shore: ~240 m for TM, ~120 m ETM+, ~200 m TIRS. Published MAE against in-situ is ~1.3 °C at
   >180 m from shore, **4.9 °C at <59 m**, and 6.7 °C at <10 m.
3. **Per-mission calibration.** The `modis_ref` product with `match_landsat` is exactly the
   published method. Chain L5→L7 across the 1999–2011 overlap and L7→L8 across 2013–2021 rather
   than bridging 1984→2013 in one step; MODIS itself only starts in 2000 and its own orbits
   drifted after 2020/2021.
4. **Resampling in SLC-off gaps.** `lwir` is read with `Resampling.bilinear`. Confirm the warp is
   nodata-aware at gap edges for L7, or use nearest for that mission.
5. **Other reasons a long record has steps**, all worth testing for before trusting a trend: the
   reanalysis input changes at **2000-01-01** (MERRA-2 → GEOS-5 FP-IT) and **2024-01-01**
   (→ GEOS-IT); Landsat 7's mean local time drifted 10:00 → <8:30 after 2017; Landsat 9 thermal
   was retied to Landsat 8 on 2023-03-01; and ~250,000 Landsat 5 scenes from 1986–1999 lack
   Payload Correction Data, which carries the instrument temperatures behind the thermal gains.

---

## 6. The brightness gate on the water mask

*Added after the above, as the fix for follow-up #1.*

**NDWI is a ratio, so it is blind to absolute brightness.** Anything whose green exceeds its NIR
passes, however luminous. Cloud usually has green < NIR and fails — but a thick cloud whose green
channel **saturates** does not: the reflectance pins at the top of the valid SR range while NIR is
merely high, the ratio goes positive, and a cloud top is admitted as water. Nothing downstream
catches it, because the scenes where this bites are exactly the ones CFMask rated low-confidence.

The original finding: `LE07_L2SP_047026_20100912_02_T2`, reported `eo:cloud_cover = 14%`, over an
open-water box — `green` = **1.602** (nonphysical; reflectance cannot exceed 1) against
`nir08` = 0.72, so NDWI = +0.38 and `ndwi_threshold: 0.0` classified dense cloud at **−24.5 °C**
as water.

✅ **The measurement the threshold comes from.** 9.0M pixels — Tillamook, Grays Harbor and Puget
Sound × TM, ETM+ and OLI-TIRS — restricted to the pixels `ndwi >= 0` currently admits:

| population | n | green P50 | P90 | P95 | P99 |
|---|---|---|---|---|---|
| QA says water (real) | 1,419,279 | **0.017** | 0.030 | 0.041 | **0.075** |
| QA says cloud (false water) | 1,035,794 | **0.671** | 1.602 | 1.602 | 1.602 |
| neither (land/other) | 13,582 | 0.234 | 0.387 | 0.425 | 0.484 |

Two populations ~**40× apart in median**, and **12.6% of everything NDWI admits has green > 1.0**.

Threshold trade-off over that set:

| `brightness_max` | real water kept | bright false-water dropped |
|---|---|---|
| 0.05 | 96.62% | 93.1% |
| 0.08 | 99.13% | 87.8% |
| **0.15 (default)** | **99.91%** | **79.1%** |
| 0.20 | 100.00% | 76.1% |

**0.15 was chosen loose on purpose.** It sits 2× above the P99 of every water pixel measured.
Turbid, sediment-laden estuary water is the whole point of this project and is brighter than open
ocean, so the cost of a too-strict cut — silently dropping real estuary observations, worst
exactly where the science is — far exceeds the cost of a too-loose one. `<= 0` disables the gate,
the same idiom `cloud_buffer_km` uses, so an older extract stays reproducible.

✅ **End-to-end on a real 15-scene acquisition** (Tillamook, 2002, L5+L7):

```
TOTAL water 27200 -> 21652   (20.4% of "water" cells were too bright to be water)
TOTAL valid  7364 ->  7344   ( 0.3% of valid observations dropped)
lst_sst: byte-identical in all 15 scenes
```

The large `water` correction against the tiny `valid` cost is the signature of a gate doing the
right thing: most over-bright cells were **already** excluded by the QA cloud mask, so the gate is
cutting the residue QA missed rather than competing with it. Scenes where CFMask was confident
barely move (2030 → 2030 valid); scenes where it was not move a lot (water 4533 → 1405).

**What it cannot do:** cloud *shadow* is dark, so brightness never catches it. That is what the QA
shadow bit and the `ST_CDIST` buffer are for — the gates are complementary, not redundant.

**Side effect worth knowing:** the shared test fixture's synthetic water was green 0.20 / NIR 0.10
— the right NDWI, but four times brighter than real water and squarely in cloud-top territory. It
is now 0.05 / 0.02. A fixture that could not survive a realism check was itself a small bug.

**Not done here:** `QA_PIXEL` bit 7 (water) is an *independent* USGS water verdict that reported
0% on the failing scene and 99.9% on genuine water. It is a stronger signal than either NDWI or
brightness and worth considering as the primary gate — but it would replace the per-scene
reflectance logic rather than tighten it, which is a larger change than this one.

---

**Primary sources**: [LSDS-1618 (L4–7 C2 L2 Product Guide)](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) ·
[C2 Surface Temperature](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature) ·
[C2 Known Issues](https://www.usgs.gov/landsat-missions/landsat-collection-2-known-issues) ·
[C2 QA bands](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands) ·
[Landsat Global Archive Consolidation](https://www.usgs.gov/landsat-missions/landsat-global-archive-consolidation) ·
Wachmann et al. 2024, *Remote Sensing* 16(5):920 ·
Foga et al. 2017, *RSE* 194:379 ·
Vanhellemont et al. 2022, *ECSS* 265:107650 ·
Schaeffer et al. 2018.
