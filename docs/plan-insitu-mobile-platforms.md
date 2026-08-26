# Moving in-situ platforms: gliders, ship transects, drifters

*Status: IMPLEMENTED 2026-08-25. Supersedes §1 of
[plan-user-provided-insitu-csv.md](plan-user-provided-insitu-csv.md), which designed the same
feature as a shared data-model change. Everything marked ✅ was measured live.*

---

## 1. What was actually blocking it, and what wasn't

Nothing was accidentally blocking mobile platforms. `insitu_acquire.split_moving_platforms`
drops them **on purpose**: the cube stores one position per station for the whole window, and a
track collapsed onto its median position produces confidently *wrong* matchups rather than
absent ones. What was missing was somewhere to put per-observation positions.

The payoff is large — **under 5% of the Copernicus In-Situ archive is fixed platforms**. Hobart
has zero moorings and plenty of tracks.

## 2. Why a separate product rather than the shared schema change

The superseded design makes `lat`/`lon` `(station, time)` everywhere and promotes
`insitu_station` to `(time, y, x)`. Two problems, the second of which is the decisive one:

1. **It breaks `extract`.** A `mask:` channel must be static `(y,x)`; `mask: insitu_station` is
   a natural "only where there's ground truth" idiom and would become a hard error.
2. 🚩 **It has a silent failure mode.** `append_zarr` uses `mode="a-"` so that static `(y,x)`
   channels are written once and left alone by later blocks. A station map that varied with time
   but was still declared `(y,x)` would therefore **freeze at whatever block 0 computed**, and
   every later block's map would be discarded with no error and no warning.

A separate product sidesteps both. `insitu` is untouched — ✅ the golden-cube test passes
unchanged, which is the machine-checked form of that claim. `insitu_station` stays `(y,x)` and
fixed-only.

**The cube still merges them**, because ground truth is ground truth: both trees feed the one
`insitu_sst` channel set. That is what `ProductSpec.cube_via` records — a new field, because
`insitu_mobile` genuinely reaches the cube but not through a contributor of its own, and
claiming `cube_opt_out=True` would have bought silence from the loud-omission guard by lying.

## 3. The on-disk schema: flat, not a rectangle

`INSITU_MOBILE/<source>/aligned/<aoi>/<aoi>_insitu.nc`, dims `(obs,)` — `sst`, `lat`, `lon`,
`time`, `platform_id`, `platform_type`.

Flat rather than `(station, time)` for a reason the superseded design flags as its own top risk:
reindexing platforms onto a union time axis, when a glider sampling every 10 s and a ship logging
hourly share almost no timestamps, makes the axis the **sum** of their lengths and the block
`S × T` — quadratic in the number of platforms and almost entirely NaN. Flat, the same data is
`sum(len)` rows. ✅ 15,286 Hobart observations is a trivially small table.

## 4. The route into the Copernicus archive is different for tracks

✅ This is the measurement that shaped the whole feature:

| route | download | observations parsed | **inside the AoI** |
|---|---|---|---|
| `history` index (what fixed platforms use) | **348 MB** for 25 of 324 files | 7,481,318 | **4** (0.000%) |
| `monthly` ARCO, server-side bbox | a few MB | 16,398 | **16,088** (100%) |

A `history` file is a platform's *whole life*, so its catalogued bounds are the bounding box of
that whole life — a drifter that crossed the AoI once carries a bbox spanning an ocean.
Extrapolated, that route is **~4.5 GB per AoI** to find a handful of observations.

The sparse ARCO service subsets by bounding box server-side, which is the shape of the question a
track asks. The cost is temporal: only `latest` and `monthly` are published as sparse cubes, and
`monthly` **begins 2020-01-01**. A range entirely before it does not come back empty — it raises
`CoordinatesOutOfDatasetBounds` — so `_clamp_window` is not a convenience: without it a project
whose window starts in 1984 (the ordinary case, since the satellite record reaches that far) would
lose the AoI outright.

## 5. `insitu_sst` means two things now, and says so

The fork, settled with real numbers rather than taste. On 2021-03-17 off Hobart, one ship made
1,304 in-AoI observations spanning 01:00–22:00:

- gated to ±60 min of the daily reference instant → **1 pixel**
- every observation placed on its own day → **96 pixels**

A transect is a spatial sample of its *day*; gating it discards 95% of the thing that makes a ship
track worth having. So `insitu_sst` takes every track observation, **bounded by the day it was
recorded** — the day bound is load-bearing, or the nearest-to-reference search would reach across
weeks and paint a March transect onto a day in April. Where a track revisits a pixel, the
observation nearest the reference time wins it.

✅ Verified end to end: 96 pixels on 2021-03-17, 80 on 2022-01-22, hours 1.0–22.8, and nothing on
days no platform passed.

Two things keep that from being muddy:

- **`insitu_hour` (time,y,x)** records the UTC hour behind each `insitu_sst` cell. Shipped only
  when a moving platform is present — for a fixed-station cube every value is at the reference
  time by construction, and adding the channel unconditionally would change a fixed-only cube.
  Presence of a track is a property of the *tree*, identical in every block, so the channel set
  stays block-invariant.
- **The `<sensor>_insitu_sst` matchup channels stay strictly gated**, for tracks as for buoys. A
  glider that passed at 03:00 is not ground truth for a satellite that flew at 18:30. This is not
  a tunable.

`insitu_sst` carries a `comment` attr stating both meanings, and `insitu_tracks` lists each
platform with its source, observation count and time span — deliberately **not** per-day, because
`_merge_block_attrs` makes a global attr that differs between blocks a hard `RuntimeError`.

## 6. 🚩 Drift alone misclassified platforms, and lost their data

The two products partition platforms on measured drift, sharing `platform_drift_m` and a
threshold of one grid cell. That is complete only if drift is measurable.

It is not. `platform_drift_m` returns `0.0` below two positions, by construction — and a track
that clips the corner of an AoI may leave exactly **one** observation inside it. Such a platform
was classified fixed, handed to a product that (for `marineinsitu`) never fetches mobile classes
at all, and its observation vanished between the two.

✅ Not hypothetical: on the first live Hobart run, **two of five platforms** — a drifter and a
ship, one observation each.

So a **declared** mobile class now beats a drift of zero. Where no class is declared (`csv` and
`ioos` say nothing about platform type) drift remains the only evidence there is, and a
single-observation platform is genuinely indistinguishable from a mooring.

## 7. One shared-registry fix this required

`ProductSpec.default_sources` was `tuple = ()` with `()` meaning "fall back to every known
source". `insitu_mobile` defaults to *no* sources, so under that reading merely selecting the
product demanded a Copernicus credential for a source the run would never touch — the same class
of bug the field was added to fix last month, reintroduced by the sentinel.

It is now `tuple | None = None`: `None` means "undeclared, assume all" (every existing product,
unchanged), and an explicit tuple means exactly that tuple, **including the empty one**.

## 8. Known wrinkles

- **`insitu_n` means slightly different things for the two halves.** For a fixed station it
  counts stations *resident* in the cell, incremented per day regardless of whether a matchup
  landed; for a track it counts observations *placed* there that day. Changing the fixed half is
  precisely what "leave the fixed stations as is" ruled out. Documented rather than fixed.
- **`extract`'s `nanmean` over a disc gets stranger with tracks.** With a fixed station a disc
  holds at most one finite cell per day; with a transect it may hold several *different*
  observations, and `nanmean` returns their spatial average as though it were a point
  measurement. `count_valid` is the existing mitigation.
- **Single-observation platforms from `csv`/`ioos` still go to the fixed product**, since those
  sources declare no platform type. Unavoidable without a type: one position is one position.
- **The `erddap` gate (cap 1) is per-product**, so this product inherits it despite reaching a
  different service. Correct and safe, just serialised.

**Files**: `processes/insitu_mobile.py` (new), `processes/insitu.py` (`TrackTable`,
`observation_pixels`, `nearest_index`), `processes/insitu_cmems.py` (`fetch_aoi_mobile`,
`_read_arco`, `_clamp_window`), `processes/datacube.py` (`load_tracks`, `_accumulate_track`,
`build_insitu(tracks=)`), `products.py` (the spec, `cube_via`, `default_sources` sentinel),
`config.py` (the preflight fallback), `provenance.py`, `tests/test_insitu_mobile.py` (new).
