# Copernicus Marine In-Situ TAC as an in-situ source

*Status: IMPLEMENTED 2026-08-25. Prompted by a request to add marineinsitu.eu as a data source.
Everything marked ✅ was verified live against the real catalogue and real files on the day this
was written; the docs on this topic are unusually stale, so the measurements are recorded here
rather than the documentation's claims.*

---

## 1. Why it was worth adding

`insitu` is the cube's only ground truth, and it had two sources: `ioos` (North America only)
and `csv` (whatever the user can supply). **Any AoI outside US waters had no discoverable ground
truth at all.** The In-Situ TAC is the global equivalent — seven production units harmonising
Argo, OceanSITES, GOSUD, the GTS and the EuroGOOS national coastal networks into one format with
one QC scale.

It fitted the existing architecture almost exactly. `insitu` was already a stacked-DATA product
whose sources fan out through `insitu_acquire` behind one seam:

```
fetch_aoi(g, start, end, cfg, dry_run=False) -> [{"id","title","var","lat","lon","df"}, ...]
```

and the Copernicus credential was already built — `auth.py` has had a `copernicus` login/refresh
handler reading `~/.netrc` since `cmems` was added, and `copernicusmarine>=2` was already a
declared dependency. So the new code is one module plus a registry entry.

## 2. The access route, and why the obvious one is wrong

The toolbox's tidy-DataFrame path (`read_dataframe` / `subset`) only works where a **sparse ARCO
cube** has been published. ✅ For the global product that is the `latest` and `monthly` parts
only; the `history` part publishes `original-files` and nothing else, so `subset` on it raises
`NoServiceAvailable`.

That matters because of what `monthly` covers:

```
monthly 1990-1999: CoordinatesOutOfDatasetBounds
monthly 2000-2009: CoordinatesOutOfDatasetBounds
monthly 2010-2014: CoordinatesOutOfDatasetBounds
monthly 2015-2019: CoordinatesOutOfDatasetBounds
monthly 2020-2026: rows=124,717  first=2020-01-02  last=2026-07-31
```

✅ **The ARCO cube begins 2020-01-01.** A range entirely before it is an error, not an empty
result. Measured against `history` over the same AoIs:

| AoI | fixed platforms | earliest via `history` | via `monthly` |
|---|---|---|---|
| grays_harbor | 2 | **1987-05-06** | 2020 |
| puget_sound | 4 | 2009-05-15 | 2020 |
| tillamook_bay | 1 | 2023-07-26 | 2020 |
| North Sea (NL) | 10 | 2001-11-20 | 2020 |
| Galicia (ES) | 11 | **1998-07-06** | 2020 |
| hobart / all Tasmania | **0** | — | — |

A pipeline whose satellite record starts in 1984 cannot use the 2020 door, so this module takes
the **index-catalog** route: `index_<part>.txt` → rows intersecting the AoI → `get(file_list=…)`
→ per-platform NetCDF.

Product roots are **derived from the catalogue, never hard-coded**: ✅ the native bucket number
differs per product (`mdl-native-01` for GLO, `mdl-native-03` for IBI/NWS/MED), so a literal path
works for whichever product it was copied from and 404s for the rest.

## 3. What the archive actually holds

**Mostly things this cube cannot place.** Globally, drifters and profiling floats alone are ~70%
of platforms while moorings and tide gauges together are under 5%. ✅ At Grays Harbor the index
lists 421 files carrying `TEMP`, of which **2** are fixed. At Hobart, and across all of Tasmania,
**zero** — only gliders, ship thermosalinographs and drifters.

So `platform_types` defaults to `["MO", "TG"]` and mobile platforms are counted but never
downloaded. Fetching them would be pointless as well as slow:
`insitu_acquire.split_moving_platforms` drops anything straying beyond one grid cell, because the
cube places one position per station for the whole window and a track collapsed to its median
position produces confidently *wrong* matchups rather than absent ones.

**An AoI returning nothing is an ordinary outcome**, and the module says so in those words.

## 4. Two traps in the data, both found live

### 🚩 The index's geographic bounds are sometimes wrong by continents

The published Help Center example filters with **containment**
(`lon_min >= box_min and lon_max <= box_max`), which keeps only files whose bounds fall entirely
inside the AoI — silently discarding every platform whose recorded extent is merely generous.
`select_files` therefore uses **interval intersection**.

Which then exposed the opposite problem. ✅ Running a Puget Sound AoI returned two "moorings"
reporting 0.0–38.9 °C:

```
INDEX BOUNDS                                    ACTUAL POSITION IN THE FILE
GL_TS_MO_31261  lat -31.5..48.7 lon -123.4..-34.6    lat  -8.156  lon  -34.564   <- Brazil
GL_TS_MO_31375  lat -31.6..48.7 lon -123.4..-43.1    lat -28.520  lon  -47.394   <- Brazil
GL_TS_MO_46120  lat  47.761..47.761 lon -122.397..-122.397  (correct, a point)
```

Two platforms advertise a box spanning **half the planet** while sitting off Brazil. Reverting to
containment would also have excluded them — at the cost of every genuinely wide-bounded platform
— so the fix belongs *after* the download, where the file's own position is known: `within()`
re-checks each platform against the AoI and drops a disagreement by name. Puget Sound then
returns its two real stations, `Pt Wells` and `Hansville - Hood Canal`.

### The file format

✅ Dumped from `GL_TS_MO_46211.nc` (Grays Harbor, 3.5 MB):

```
dims: TIME=325320, DEPTH=2
LATITUDE/LONGITUDE shape ()      <- SCALAR for a fixed platform, (TIME,) for a mobile one
DEPH (2,) values [0.0, 0.46]     <- the VARIABLE is DEPH; the DIMENSION is DEPTH
TEMP (TIME, DEPTH); TEMP_QC decoded by xarray as FLOAT32, not the int8 it is stored as
attrs: platform_code='46211' platform_name='Grays Harbor' data_mode='R'
327,272 QC-good rows, 2004-09-08 -> 2026-07-31, 5.30..20.10 degC
```

Three details, each of which silently empties a series if missed: compare `TEMP_QC` as float;
prefer `DEPH` and fall back to `PRES`; and **ask for `LATITUDE`/`LONGITUDE` explicitly** — they
are *coordinates* in the files this was built against, so a variable subset carries them along
for free, but the format permits them as data variables, and a file storing them that way would
come back with no position and be dropped as unplaceable while the index still insisted the
platform was there.

A mooring reports several depths; the **shallowest** surviving `max_sensor_depth_m` wins per
timestamp, that being the level comparable to a satellite's surface retrieval.

## 5. The index is cached on disk, and may be served stale

✅ It is ~28 MB in a single unresumed GET, fetched once per (dataset, part) for **every** AoI —
so a blip does not cost one AoI, it costs the run. That is not hypothetical; it happened during
verification:

```
FAILED grays_harbor marineinsitu: ('Connection broken: IncompleteRead(12243941 bytes read,
16141403 more expected)', ...)
```

`net.retry` classified it correctly and still exhausted all four attempts, on an endpoint that
served the same file cleanly in 2.8 s three times a minute later. So the catalog is cached under
`INSITU/marineinsitu/_cache/`, refreshed after 24 h (it is rebuilt about daily), and **when a
refresh fails but a stale copy exists the stale copy is used with a warning** — a day-old catalog
is a far better answer than no data. With no cache at all a failure still raises, so a first-run
outage cannot masquerade as an empty channel. Second runs drop from 31 s to 1 s.

## 6. One shared-code fix this required

`config.required_backend` resolved an unset stacked `sources` list as **every known source**,
while `insitu_acquire` acquired only its own `DEFAULT_SOURCES = ["ioos"]`. Invisible while every
in-situ source was public — the union of `None` is `None` either way — and a breaking change the
moment one needed a credential: every existing config with an `insitu:` block and no explicit
`sources:` list would have started demanding an `auth.copernicus` block for a source it was never
going to run.

`ProductSpec.default_sources` is now the single definition, read by both the preflight and the
acquisition stage, with a registry invariant checking it names real sources. The change can only
ever *relax* what the preflight demands.

## 7. Known wrinkles

- **An AoI with no fixed platforms is reported as `FAILED`.** That is pre-existing shared
  behaviour (`insitu_acquire.run` calls `rep.fail` when a source yields nothing), and it is right
  for `ioos`/`csv`, where empty is suspicious. For this source empty is *routine* — most of the
  world's coastline has no mooring — so a multi-AoI run will show `FAILED` lines for expected
  outcomes. Distinguishing "nothing here, normally" from "nothing here, something broke" means
  changing the shared seam, which was left alone deliberately.
- 🚩 **Overlap with `ioos` in US waters.** The moorings found off North America are largely the
  same NDBC buoys `ioos` serves, under a different id (`46211` here, `gov-ndbc-46211` there). The
  assembler's `seen_ids` guard only catches *identical* ids, so stacking both enters one buoy
  twice and averages it into one pixel — not wrong, but it double-counts one instrument as two
  independent observations. In US waters, pick one.
- **These are bulk temperatures at a stated depth** (~1 m on a moored buoy), never skin. Expect a
  skin–bulk difference of ~0.1–0.5 K plus diurnal warming in the top metre when validating a
  satellite retrieval against them. `DEPH` is filtered but not carried into the cube, which has
  no depth dimension.
- **The `erddap` gate (cap 1) is per-product, so this source inherits it** despite talking to a
  different service. Correct and safe, just serialised; raising `runtime.gates.erddap` would
  speed a many-AoI in-situ run at the cost of politeness to both endpoints.
- **Mobile platforms remain out of scope.** `platform_types` widens the *fetch* side by adding
  `GL`/`TS`/`DB`, but the drift guard drops them again downstream. Real support means
  per-observation placement, which changes the shared data model and the cube schema.

**Sources**: [marineinsitu.eu](https://marineinsitu.eu/) ·
[Copernicus Marine Toolbox docs](https://toolbox-docs.marine.copernicus.eu/) ·
[In Situ NetCDF Format Manual v2.1.0](https://archimer.ifremer.fr/doc/00488/59938/) ·
[How to filter INSITU data files using the index files](https://help.marine.copernicus.eu/en/articles/9630028-how-to-filter-insitu-data-files-using-the-index-files-on-python)
(the containment bug lives here) ·
live endpoints `stac.marine.copernicus.eu` and `s3.waw3-1.cloudferro.com` (2026-08-24/25).
