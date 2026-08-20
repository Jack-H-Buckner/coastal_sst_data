# Landsat coverage: why most sites see a small fraction of the overpasses

*Status: DIAGNOSED, NOT FIXED. Written 2026-08-20 from a real extract; no code changed. Two
defects are described here with their measured cost and the fix for each.*

Prompted by: in an extract over the Hobart AoI, most sites had far fewer Landsat observations
than the 8-day revisit predicts, while MODIS and ECOSTRESS looked normal.

---

## 1. What the data shows

From `aux_info/salmon_small_eco_points_5.csv` — 53 sites x 182 dates, one long-format row per
(site, date, variable, stat). ECOSTRESS columns in that file are stale (that tree was
mid-migration when it was written) and are ignored throughout.

The AoI has **44 Landsat overpass dates**. `lst_hour` is 1-D `(time,)` — the AoI-wide overpass
time — so grouping the dates by it separates the WRS-2 **paths**. Three clusters fall out,
about six minutes apart:

| overpass hour (UTC) | dates | dates yielding ZERO sites | max sites reached |
|---|---|---|---|
| ~23.77 | 13 | 0 | 17 — **east only**; the 36 western sites get 0 on all 13 |
| ~23.87 | 16 | 2 | 53 — the only path that ever reaches the west |
| ~23.97 | **15** | **15** | **0 — never delivers to any site, ever** |

Two facts drive everything below:

1. **An entire path is acquired 15 times and contributes nothing.** On all 15 of those dates
   `lst_hour` is set, meaning a granule was read and became that day's base — yet not one
   site gets a value. Those days look observed and are empty.
2. **The 36 western sites get data on 4 dates**, and all four are dates when the whole AoI was
   covered (2022-10-20, 2022-12-31, 2023-01-16, 2023-03-13). The ~23.87 path demonstrably
   *can* cover the entire AoI, but does so on only 4 of its 16 dates; on the other 12 it
   delivers the east alone.

Net: 15 dead + 13 east-only = **28 of 44 dates can never help a western site**. Part of the
deficit is therefore real geometry, not a defect — see §5 for what a fix can and cannot
recover.

### Cloud is not involved

`<prefix>_sst` is the RAW channel. `_read_granule` returns the scene array untouched
(`datacube.py`, `s = ds["sst"].values`), and the merge copies it verbatim — on a
single-granule day `sst[i] = s`, and on a mosaicked day the base's `take = v | (score < 0)`
is all-True. **A cloudy, land, or NDWI-fails cell keeps its temperature in `lst_sst`.**

So a NaN in `lst_sst` is scene footprint or the merge, never weather. This also means
`masking.ndwi_threshold` and `masking.cloud_buffer_km` are *not* the lever for missing
observations — they only affect `lst_valid` and the `_clean` channels. Anyone starting from
"it must be cloud" will spend the day in the wrong file.

---

## 2. Defect 1 — scenes that never reach the AoI are acquired anyway

`landsat_pc.search_scenes` filters on collection, **bbox**, datetime, `eo:cloud_cover` and
platform, and nothing anywhere checks that a returned scene's *data* reaches the AoI.
ECOSTRESS has exactly this check (`ecostress.run`, the `granule_bbox(...).intersects(
aoi_lonlat)` guard); Landsat does not.

A Landsat COG's bounding rectangle is much larger than its rotated data parallelogram, so a
scene whose bbox intersects the AoI while its imagery clips the edge is:

1. downloaded (five windowed COG reads),
2. read as mostly or entirely nodata — `rio.clip_box` succeeds on the intersection and
   `reproject` fills the rest with nodata,
3. written as a **complete** granule: every one of `store.REQUIRED_VARS["LANDSAT"]`
   (`sst`, `water`, `cloud`, `valid`) is present, so the skip guard treats it as done,
4. and, being the day's only granule, becomes the base and defines the day.

That is the ~23.97 cluster: 15 downloads, 15 dead days, no warning at any stage.

### Fix

Add the parity check in `landsat_pc.run()`, using the STAC item's `geometry` — a true
footprint polygon, which is *better* than the bounding box ECOSTRESS has to settle for:

```python
aoi_lonlat = g.geom_lonlat()          # once per AoI, as ecostress does
...
geom = getattr(it, "geometry", None)
if geom is not None and not shape(geom).intersects(aoi_lonlat):
    log.debug("  [%s] %s does not reach the AoI polygon, skipping", name, it.id)
    skipped += 1
    continue
```

**A missing or unreadable geometry means KEEP** — ECOSTRESS's convention. Never drop a scene
because its metadata was thin; that trades a known failure for a silent one. Log the skips at
DEBUG with a count at INFO.

---

## 3. Defect 2 — the merge discards most of a non-overlapping granule

Two defects in `datacube.load_clearest_overpass`, both from treating the raw SST channel as
if every granule covered the whole grid:

```python
take = (v | (score < 0)) if new_base else (v & (score < vc))
np.copyto(sst[i], s, where=take)
```

They bite only when granules are **non-overlapping** — WRS-2 rows/paths, MGRS tiles — which
is precisely the case `mosaic_same_day` exists for. They are the likely reason the ~23.87
path reaches the west on only 4 of its 16 dates.

### 3a — a new base overwrites earlier granules' data with its own NaN

`take = v | (score < 0)` is true wherever nothing has *validly* claimed a cell — including
cells an earlier granule already painted with a real temperature. `np.copyto` then writes
this granule's value there, and **outside its own footprint that value is NaN**. Each
successive new base wipes its predecessors, so only the LAST new base keeps a full footprint.

Simulated on three non-overlapping tiles arriving in increasing valid-count order:

```
today  tile0   5.1%   tile1  10.1%   tile2 100.0%   TOTAL  38.4%
fixed  tile0 100.0%   tile1 100.0%   tile2 100.0%   TOTAL 100.0%
```

### 3b — a non-base granule contributes only where it is valid

`take = v & (score < vc)` requires the granule's own `v`, so its finite-but-invalid pixels
never enter the channel — while the base's do. With Landsat's mask (NDWI water AND QA cloud
AND a 1 km CDIST buffer, the strictest of the three sensors) a granule covering 100% of its
half delivers about 10% of it:

```
Landsat-like strict mask                                       9.9%
lower-ranked granule almost entirely masked                    2.1%
validity BROKEN entirely (the ECOSTRESS fill-value failure)    0.0%
```

The docstring's stated rationale is that filling those cells "would put unvalidated pixels
into the sst channel, which is a stronger claim than the files make". **The base is already
exempt from that rule** — its own unvalidated pixels are written by `v | (score < 0)`. The
channel already carries unvalidated pixels; only ever from whichever granule sorted first.
The asymmetry has no justification, and for non-overlapping granules its cost is a whole
footprint.

### Fix — two clauses, both about NaNs in the raw channel

```python
    if new_base:
        take = v | (score < 0)
        # Never overwrite a real temperature with this granule's NaN. Outside its own
        # footprint `s` IS NaN, so without this a later base wipes every earlier granule's
        # contribution and only the last one's footprint survives.
        take = take & (np.isfinite(s) | ~np.isfinite(sst[i]))
    else:
        take = v & (score < vc)
        # A cell NO granule has any reading for is a HOLE, not a contested cell. Filling it
        # from a lower-ranked granule's raw value is exactly the claim the base already makes
        # about its own unvalidated pixels; refusing it only for granules that did not happen
        # to sort first is what let a 185 km scene deliver a tenth of its own footprint.
        take = take | (np.isfinite(s) & (score < 0) & ~np.isfinite(sst[i]))
```

Together they recover **100%** in every case above.

What is preserved, and must stay preserved:

- A cell the base **observed and masked** still keeps the base's value — the documented
  intent. `score < 0` excludes every validly-claimed cell; `~isfinite(sst[i])` excludes every
  cell already holding a real number.
- **`<prefix>_valid` is unchanged.** `np.copyto(valid[i], v, where=take)` writes the
  contributing granule's own verdict, and `score` still advances only on `take & v`. A filled
  hole is honestly labelled invalid — and the cells the fix adds are, by construction, exactly
  the ones that granule's mask rejected, since its valid cells are already filled today.
- sst / cloud / valid still come from ONE granule for any given cell.

It also makes the mosaic robust to a broken validity mask: with `valid` empty everywhere — the
ECOSTRESS fill-value failure — non-base granules still fill the holes. That retires the whole
"one granule wins" class rather than the instances of it chased one at a time.

---

## 4. Defect 3 — a granule that contributes nothing is silent

Nothing reports a scene that read back with no finite SST over the AoI, which is why 15 dead
downloads looked like 15 successful ones. Count them and report per AoI — the
acquisition-side twin of the assembler's "finite SST but NO valid pixels" warning.

---

## 5. What a fix will NOT recover

Recorded so this is not re-opened on a false expectation:

- The **15 dates in the ~23.97 cluster stay empty**. That path genuinely does not reach these
  sites. The fix stops downloading it; it does not invent coverage.
- The **13 east-only dates stay east-only**. Real WRS-2 geometry.
- A realistic ceiling is **~16-20 Landsat dates per site in 182 days, not 44**, because two of
  the three paths do not cover the western sites at all.
- Landsat is legitimately sparse: one path/row is imaged every 8 days (L8 and L9 in 8-day
  phase), so ~11-12 scene-days per path per 92 days, fewer after `cloud_cover_max`. Sparse
  dates are correct; *zero* dates at a site whose neighbour has values is not.

---

## 6. Confirm it on a real tree first

Read-only, no re-download. Reports each granule's actual AoI coverage grouped by overpass
hour, so a dead path is visible directly:

```python
import numpy as np, xarray as xr, glob, os, re, collections
D = "data/salmon_regions_small_eco/LANDSAT/aligned/Hobart"
rows = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(D, "*.nc"))):
    d = xr.open_dataset(f); d = d.isel(time=0) if "time" in d.dims else d
    cov = float(np.isfinite(d["sst"].values).mean())
    hh = re.search(r"T(\d{2})(\d{2})", os.path.basename(f))
    rows[f"{hh.group(1)}:{hh.group(2)[0]}0"].append(cov)
    d.close()
print(f"{'overpass ~hour':16s} {'granules':>9s} {'dead(0%)':>9s} {'median cov':>11s} {'max cov':>8s}")
for k in sorted(rows):
    c = np.array(rows[k])
    print(f"{k:16s} {len(c):9d} {int((c == 0).sum()):9d} {np.median(c):10.1%} {c.max():8.1%}")
```

A cluster at or near 0% coverage is Defect 1 — those scenes should never have been
downloaded. Clusters with healthy per-file coverage that still lose sites in the cube are
Defect 2.

---

## 7. Files, tests, verification

| file | change |
|---|---|
| `src/coastal_sst_data/processes/landsat_pc.py` | AoI-polygon pre-filter via `item.geometry`; all-NaN-granule warning; log the masking params in force |
| `src/coastal_sst_data/processes/datacube.py` | the two merge clauses, and the docstring's "two consequences" paragraph, which becomes one |
| `README.md` | Landsat is the only sensor carrying both a water gate and a cloud gate, and the only one recomputing water per-scene from reflectance; the `ndwi_threshold` / `cloud_buffer_km` / `cloud_cover_max` knobs |

**Deliberately not changing** `ndwi_threshold`, `cloud_buffer_km` or `cloud_cover_max`. They
change what the SST values mean, they are a config decision, and they are not what is costing
observations here (see §1).

Tests:

- A scene whose `geometry` misses the AoI polygon is **never opened**; one with no geometry is
  kept; one that covers the AoI is unaffected. (`conftest.FakeStacItem` needs an optional
  `geometry`, defaulting to absent so existing tests keep exercising the keep-path.)
- A granule reading all-NaN over the AoI is counted and reported.
- **Defect 3a**: three non-overlapping granules in increasing valid-count order — all three
  footprints survive. Fails today at 38%.
- **Defect 3b**: two non-overlapping granules, the lower-ranked one mostly invalid — its
  footprint is still filled. Fails today at ~10%.
- A cell the base observed and masked keeps the base's value.
- `<prefix>_valid` stays 0 in a filled hole and equals the contributing granule's own mask.

The golden snapshots will move. Regenerate with `UPDATE_GOLDEN=1` and confirm the diff is
**only** additional finite cells in the sensor SST channels, with `*_valid` unchanged —
anything else means the fix reaches further than intended.

```bash
pytest tests/test_landsat_pc.py tests/test_datacube.py -q && pytest -q
```

The merge fix is assembly-only; the polygon filter needs a re-acquire to stop the dead
downloads:

```bash
coastal-sst-data assemble --config <config> --overwrite
coastal-sst-data extract  --config <config> --overwrite
```

Expected after the fix, against the numbers in §1: the 36 western sites rise from **4** dates
toward the ~16 the ~23.87 path offers; the 2 dead dates inside that cluster populate; the 15
dates in the ~23.97 cluster stay empty; east sites (17) are roughly unchanged.
