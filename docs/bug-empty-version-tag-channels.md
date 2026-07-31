# Empty version-tag channels silently disable the georeferencing correction

Status: diagnosed against the `test_coastal_sst_data` project on 2026-07-27; worked around there by
moving the offending directory. **No package code has been changed.** This note records the defects
so the fixes can be applied to `coastal_sst_data` itself.

## Symptom

`flag_georef` / `correct_georef` ran on every scene of both AoIs and moved nothing. Every scene was
classified `insufficient_signal`, and the "corrected" channels were verbatim copies of their inputs:

| | north_sound | tillamook_bay |
|---|---|---|
| `eco_sst_aligned` finite cells | 0 | 0 |
| `eco_georef_flag == insufficient_signal (5)` | 107/107 | 107/107 |
| `eco_georef_n_edge` max | 0 | 0 |
| `eco_georef_applied.sum()` | 0 | 0 |
| `eco_sst_v003` vs `..._georef_corrected` identical where both finite | True | True |

Nothing failed and nothing warned. The step reported success, the diagnostic channels were written,
and every value in them was the "no signal" sentinel. After the workaround below, the same data
yields 4 corrected scenes at north_sound and 2 at tillamook_bay — including a genuine 5,126 m
displacement at tillamook_bay 2026-07-15 (z = 6.1, agreement 1% → 29%).

## Defect A — a directory that is not a version tag is loaded as one

`processes/datacube.py:571`, in `_contribute_stacked_sensor`:

```python
on_disk = {d.name for d in base.iterdir() if d.is_dir()}
ordered = [t for t in pref if t in on_disk] + sorted(on_disk - set(pref))
```

Every subdirectory of `ECOSTRESS/` becomes a version tag. The project had two:

```
data/ECOSTRESS/v003/aligned/<aoi>/    <- the versioned layout
data/ECOSTRESS/aligned/<aoi>/         <- a flat pre-versioning leftover, 34 granules
```

For the phantom tag `"aligned"` the loader resolves granules at `ECOSTRESS/aligned/aligned/<aoi>/`,
which does not exist. `_load_sensor` returns empty arrays and the tag is emitted anyway, producing a
complete but entirely-NaN channel set: `eco_sst_aligned`, `eco_valid_aligned`, `eco_hour_aligned`,
and — because later steps fan out over channels by prefix — `eco_sst_aligned_cloudfiltered`,
`eco_sst_aligned_georef_corrected`, `eco_sst_aligned_georef_corrected_clean`, and so on. Eight
phantom channels in the preprocessed cube, none of them holding a single value.

Note the legacy granules **do** contain real data. They are never read; only the directory name
propagates.

**Proposed fix.** Only accept a tag whose granule directory actually exists and is non-empty, so an
empty channel is never emitted:

```python
on_disk = {d.name for d in base.iterdir() if d.is_dir()}
on_disk = {t for t in on_disk if any(ctx.adir(s.product.value, t).glob("*.nc"))}
```

A tag dropped this way is worth an `INFO` line — silently ignoring a directory the user put on disk
is its own failure mode. An alternative, stricter reading is that a tag not listed in
`sensor_version_pref` should never be auto-discovered at all; that would also have prevented this,
but it changes the documented "acquired but dropped from the config still contributes" behaviour.

## Defect B — "primary version" is decided alphabetically

`processes/georef.py:383-386`, in `_step_flag_georef`:

```python
sst_names = ctx.channels_with_prefix(f"{pre}_sst")
if not sst_names:
    return
working = _working(ctx, sst_names[0])   # cloud-filtered SST of the primary version
```

`channels_with_prefix` returns `sorted(...)` (`processes/preprocess.py:165`). Sorting is not a
version ordering, and the comment's "primary version" is not what `[0]` selects. With
`['eco_sst_aligned', 'eco_sst_v003']`, the fit ran on the all-NaN channel: Canny found 0 edges,
`quality_reject` returned `insufficient_signal`, and `applied = (flag == FLAG["displaced"])` was
all-False for every scene.

This is **still latent** after the workaround. Any tag sorting before the real one reintroduces it —
a future `v002` alongside `v003` would fit on `v002` regardless of which carries the data.

**Proposed fix.** Prefer the populated channel, ideally honouring the config order that
`_contribute_stacked_sensor` already computes (`sensor_version_pref`) and falling back to occupancy:

```python
def _primary(ctx, names):
    """The version channel to fit on: config preference first, then most-populated.

    NOT `sorted(...)[0]` -- alphabetical order is not version order, and an empty channel that
    sorts first silently turns the whole step into a no-op (see docs/bug-empty-version-tag-channels).
    """
    populated = [n for n in names if np.isfinite(ctx.read(n)).any()]
    ...
```

The "most finite cells wins" rule is already proven in the project's
`src/plot_waterline.py:resolve_channel`, which was written specifically to dodge this empty channel.

Independently: a fit that sees **zero edges on every scene in the cube** should warn. A single
`WARNING` when `n_edge == 0` for 100% of scenes would have surfaced this immediately instead of
producing a plausible-looking set of all-sentinel diagnostics.

## Defect C — occupancy-based channel resolution now picks a derived channel

This one is in the analysis project (`src/plot_waterline.py:124-144`), but the package should not
adopt the same heuristic. `resolve_channel` picks among `<sensor>_sst*` float channels by "most
finite cells wins". That was correct when the only float siblings were version variants. The georef
step then introduced `..._georef_corrected`, which is seeded from the **raw** field and so by
construction has more finite cells than any cloud-masked channel:

```
north_sound:    resolve_channel(pre,'eco','sst') -> eco_sst_v003_georef_corrected
                finite=7,768,536   filters reported=[]
                (intended cloud-masked eco_sst_v003 finite=3,391,396)
tillamook_bay:  resolve_channel(pre,'eco','sst') -> eco_sst_v003_georef_corrected
                finite=4,944,833   filters reported=[]
                (intended cloud-masked eco_sst_v003 finite=1,197,157)
```

So `plot_waterline.py`, `plot_thermal_edges.py` and `georef_diagnose.py` all read the **unfiltered**
corrected field while setting `filtered=True` and captioning "none dropped".

**Lesson for the package:** measurement channels must be distinguished from derived channels by
**name**, not by population. `src/georef_compare.py:resolve_pair` does this — it restricts candidates
to `<sensor>_sst` / `<sensor>_sst_<tag>` (a single alphanumeric run), so every derived suffix, all of
which contain more than one underscore, is structurally excluded.

## Defect D — the corrected product is not filtered like the uncorrected one

Found while building the comparison figure. `correct_georef` seeds from the **raw** field rather than
the filtered working channel:

```
north_sound    raw= 7,768,536   _georef_corrected= 7,766,210   filtered v003= 3,391,396
tillamook_bay  raw= 4,944,833   _georef_corrected= 4,916,936   filtered v003= 1,197,157
```

(the small deficits are margin loss on the scenes actually shifted). The corrected branch therefore
rebuilds its cloud mask from scratch — but only from the steps the config re-runs. With
`filter_clouds_corrected` and `filter_land_clouds_corrected` enabled and no corrected counterpart for
`filter_cloud_cover`, the met cloud gate is never re-applied. The cube shows this directly:
`<sst>_georef_corrected_clean_metcloudfiltered` does not exist, while `..._clean_cloudfiltered` and
`..._clean_landcloudfiltered` do.

`filter_cloud_cover` is the only filter that can empty an entire scene (scene gate: AoI-water-mean
cloud > `scene_max_pct`; pixel gate: > `pixel_max_pct`), so the asymmetry is concentrated exactly
where it is most visible. Of the scenes empty in the uncorrected channel but populated in the
corrected one, **100% are whole-grid met-gate kills** — 26/26 at tillamook_bay, 14/14 at
north_sound, at 57–100% scene cloud cover.

The consequence is that the corrected product retains cloudy overpasses the uncorrected product
rejects, so the two are not like-for-like and a naive comparison reads cloud as recovered data.

**Proposed fix.** Either register a `filter_cloud_cover_corrected` partial alongside the other two
(`processes/cloud_filter.py` already builds these with `functools.partial(..., mode="corrected")`),
or have `correct_georef` seed from the filtered working channel so the corrected branch inherits the
mask instead of rebuilding a different one. The first keeps the intermediate inspectable, which is
the stated reason for the `_clean` split; the second removes the duplication that let the two chains
diverge. Worth deciding deliberately — the current behaviour looks like an omission rather than a
choice, since nothing documents it.

## Suggested regression tests

`tests/test_datacube.py`

- a fixture with a decoy directory under the product root containing no granules at
  `<tag>/aligned/<aoi>/`: assert no channel for that tag is emitted (Defect A)
- assert a tag whose granules *are* present at the expected path still is emitted, so the
  discovery fix does not over-reject

`tests/test_georef.py`

- two SST version channels where the alphabetically-first is all-NaN and the second carries a known
  displacement: assert the fit selects the populated channel and recovers the shift (Defect B)
- assert that a cube in which every scene yields zero edges produces a warning, not a silent set of
  `insufficient_signal` flags

`tests/test_preprocess.py`

- assert the corrected channel's filter-mask set matches the uncorrected one, so a filter present in
  one chain and absent from the other cannot pass unnoticed (Defect D)

## Workaround applied in the analysis project

```bash
mkdir -p _legacy && mv data/ECOSTRESS/aligned _legacy/ECOSTRESS_flat_aligned
coastal-sst-data assemble   --config configs/main_grids.yml --overwrite
coastal-sst-data preprocess --config configs/main_grids.yml --overwrite
```

Both commands are local-only; `assemble` re-knits already-downloaded aligned NetCDFs. This removes
the phantom channels, after which `sorted(...)[0]` lands on `eco_sst_v003` — by luck, not by design,
which is why Defect B still wants a real fix.

Unrelated environment note: the conda env's PROJ database is only found when the env is activated;
in a bare shell `PROJ_DATA` points at the base install and `pyproj` raises
`CRSError: Invalid projection: EPSG:4326`. Export `PROJ_DATA=$CONDA_PREFIX/share/proj` when invoking
the CLI without activation.
