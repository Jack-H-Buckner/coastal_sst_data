# Coastal SST data loader

This library is desinged to obtain data for coastal and nearshore ocean ecosystem and combine them in to a gridded data format for down stream modeling tasks. The primary goal of this code base is to load thermal remote sensing images and covarites that drive nearshore ocean temperatures to feed into high reolution sea surface temperature models. The package also loads in situ observations form monitoring networks like the IOOS network for model validation with ground truth measurments. 

## Project structure. 

This package is driven by a configuration file that defines the data products  acquired, the data range and areas where they are pulled from and  options for how they are compiled. 

A project has a few key components 

- name: a name for the project 

- time: A field that defines the start and end for the data aquisition
    - fields: start date, end_date   

- grid: paramters that define the grid that the data products are mapped ot in each area of interest.
    - fields: 
        - resolution_m: resolution of the grid.       
        - target_crs: the CRS used for the aois or the method for selecting it       
        - resampling_continuous: method for resampling continuous variables, e.g. bilinear.
        - resampling_categorical: method for resampling categoraicl variables, e.g. nearest.
        - snap_origin: Is the grid aligned with the origin, default = true

- products: A list of data types to load, when available for each area of interest. Global options for each prodct are listed here as well. The avaible processes and options are described in detail in the following section. 
-  regions: defines the locations to obatian data for using a hierarchical structure. The base unit is an area of interst which is defined by a bounding box. Data aproducts are obtained for each AOI and projected to a unique grid defined over the AOI. The next uity up is the region. Regions are a larger spatial unit which is intended to group aois that use common data sources. some data products like digital elevation models are only avaible for specific regions. If a project need to use multiple data sources to cover the full geographic extent of the study then the AOIs that have the same data sources should be grouped into a region. 


```
regions:
  - name: pnw_estuaries
    # Region-DEPENDENT source options (which source has coverage here, etc.).
    sources:
      bathymetry:
        dem_source: cudem
    # list aois in the region.
    areas:
      - name: tillamook_bay
        center_lat: 45.52
        center_lon: -123.925
        buffer_ns_km: 25
        buffer_ew_km: 15
```
## Installation and basic usage

### Requirements

Python **3.10+** (the environment pins 3.11). The package leans on a heavy geospatial stack — `rasterio`, `rioxarray`, `pyproj`, `xarray`, `zarr`, `earthaccess`, `pyresample` — so **conda/mamba is strongly recommended** over a bare `pip` install, which would leave you to source the compiled GDAL/PROJ libraries yourself.

### Install (recommended: conda / mamba)

```bash
git clone <repository-url> coastal_sst_data
cd coastal_sst_data

mamba env create -f environment.yml       # or: conda env create -f environment.yml
conda activate coastal_sst_data
```

The environment file installs the package itself in editable mode (its `pip:` section runs `pip install -e ".[dev]"`), so the `coastal-sst-data` command is on your path as soon as the env is active. Confirm the install with:

```bash
coastal-sst-data --help
pytest                                     # optional: run the test suite
```

To pick up dependency changes later (e.g. after a `git pull`):

```bash
mamba env update -f environment.yml --prune
```

Two optional extras are used only if present: **cartopy** (coastlines on the `grids --plot` maps) and a downloaded **eo-tides** model directory (the global tide backup — see the [Tides](#tides) section). Neither is required for a normal run.

### Install (pip only)

The package declares its runtime dependencies in `pyproject.toml`, so a plain `pip install` pulls everything the core needs (config, grids, MUR/ECOSTRESS/MODIS acquisition, and datacube assembly):

```bash
pip install -e .              # from a clone (editable)
pip install .                 # from a clone (copy)
```

**Optional feature backends** are split into extras so you install only what your config uses. Add them in brackets:

| Extra | Enables |
| --- | --- |
| `landsat`, `landcover` | Planetary Computer sources (`planetary-computer`, `pystac-client`) |
| `modis` | swath→grid resampling (`pyresample`) + server-side subsetting (`harmony-py`) |
| `met` | HRRR + ERA5 forcing (`herbie-data`, `gcsfs`, `pyresample`) |
| `tides` | NOAA CO-OPS backend (`pytides2`) |
| `cmems` | Copernicus Marine global physics (`copernicusmarine`) |
| `tides-global` | global tide-model backup (`eo-tides`) |
| `plot` | `grids --plot` maps (`matplotlib`, `cartopy`) |
| `all` | every feature backend at once |

```bash
pip install ".[landsat,met,plot]"    # just the backends you need
pip install ".[all]"                 # the lot
```

The caveat is unchanged: the compiled geospatial libraries (GDAL via `rasterio`/`rioxarray`/`pyproj`, and `cartopy`) install far more reliably from conda-forge than from PyPI. For a pip-first setup make sure those resolve on your platform (they ship manylinux/macOS wheels, but a conda base is still the smoother path).

### Use it in another project (as a library)

Once installed, `coastal_sst_data` is importable from any project or working directory — an editable install is on the environment's path globally, not tied to a directory. The CLI is a thin wrapper over a small public API you can call directly:

```python
import coastal_sst_data as csd

project = csd.load_config("config.yaml")     # validated Project
grids = csd.project_grids(project)           # {aoi_name: AoiGrid}, computed once
csd.run_pipeline(project, assemble=True)     # acquire everything, then build the cubes
```

#### Depending on it from another project's conda env

The common case is another project — with its **own conda environment, its own dependencies, and its own directory** — that wants to import `coastal_sst_data`. The install is directory-independent: nothing needs the repo cloned anywhere. The rule that keeps it working is:

> **conda-forge provides all of `coastal_sst_data`'s compiled deps; pip installs only the package itself from GitHub.**

This matters because the package pulls in GDAL/PROJ- and eccodes-backed libraries (`rasterio`, `rioxarray`, `pyproj`, `pyresample`, `herbie`, `cartopy`). If pip is left to resolve those, it installs them as **wheels**, which mix incompatible native runtimes (e.g. multiple OpenMP copies) and can **segfault** at import or during the met/regrid stages. Installing them from conda-forge first — so pip finds every dependency already satisfied and adds nothing of its own — avoids that entirely.

So in the **consuming project's** `environment.yml`, list the conda-forge deps you need, then add the package on the pip line:

```yaml
name: my_other_project
channels: [conda-forge]
dependencies:
  - python=3.11
  # ... that project's own deps ...

  # coastal_sst_data's compiled deps (add only the backends you use):
  - numpy
  - pandas
  - xarray
  - pyproj
  - shapely
  - affine
  - rasterio
  - rioxarray
  - zarr
  - numcodecs
  - scipy
  - h5netcdf
  - earthaccess
  - pyresample        # modis / met
  - harmony-py        # modis (server-side AoI subsetting)
  - herbie-data       # met (HRRR)
  - gcsfs             # met (ERA5 fallback)
  - copernicusmarine  # cmems (Copernicus Marine global physics)
  # ... pystac-client, planetary-computer, eo-tides, matplotlib, cartopy as needed

  - pip
  - pip:
      - git+https://github.com/Jack-H-Buckner/coastal_sst_data.git@main
```

A ready-to-copy version of this file lives at [`environment.consumer.yml`](environment.consumer.yml) — copy it into your project, rename it, add your own deps, and trim the backends you don't need.

**The CO-OPS tides backend needs a separate step.** Don't add `pytides2` to the `pip:` block above: it's not on conda-forge and declares a broken pin (`numpy>=1.19,<1.19.4`), so pip tries to downgrade numpy and fails building an ancient sdist. The package ships a compatibility shim, so `pytides2` runs fine on modern numpy when installed *without its dependencies*. If your config uses the default CO-OPS tide source, install it after the env is created:

```bash
conda env create -f environment.my_other_project.yml -n my_other_project
conda activate my_other_project
python -m pip install --no-build-isolation --no-deps pytides2 # ensure you are usign the correct python instance 
```

(Skip this if you don't use CO-OPS tides — the `eo-tides` global fallback is a normal conda-forge dep.)

**Pin the version for reproducibility.** `@main` is a moving target — two projects set up a week apart can get different code. Pin to a git tag or commit so each project's env is reproducible:

```yaml
- git+https://github.com/Jack-H-Buckner/coastal_sst_data.git@v0.2.7     # a git tag
- git+https://github.com/Jack-H-Buckner/coastal_sst_data.git@21c14a1    # a commit SHA
```

Releases are tagged `vX.Y.Z`; `git tag --list` in a clone, or the repo's tags page, shows what is available.

To bump a consumer to a newer version later, see [Updating an existing install](#updating-an-existing-install).

**Alternative — no per-project install.** If a project just needs to *run* the package occasionally, you can skip embedding it and instead `conda activate coastal_sst_data` (the standalone env from `environment.github.yml`) and work there. That's simplest for one-off use, but it doesn't let the package coexist with another project's own dependencies — for that, use the embedded pattern above.

### Updating an existing install

How you pull in a newer version depends on how the environment installed the package in the first place. Activate the environment first in every case.

**A clone you develop in** (`environment.yml`, which installs `-e ".[dev]"`). The install is **editable**, so a `git pull` is enough for code changes — there is nothing to reinstall to pick up a new function or a bug fix:

```bash
git pull
```

Re-run the install when the **version** changed, or when you want the version reported correctly:

```bash
pip install -e ".[dev]" --no-deps
```

This matters more than it looks. An editable install records its version in the *dist metadata at install time*, and `coastal_sst_data.__version__` reads that metadata — not `pyproject.toml`. So an env installed months ago keeps reporting the version it was created at however many times you pull, and since the assembler stamps that value into every cube as the `package_version` attribute, a stale env quietly mislabels its own output. Check what an environment actually has:

```bash
python -c "import importlib.metadata as m; print(m.version('coastal_sst_data'))"
```

If that disagrees with `version` in [`pyproject.toml`](pyproject.toml), reinstall as above.

When **dependencies** change (a new package in `environment.yml`, not just new code), update the env itself — see [Install (recommended: conda / mamba)](#install-recommended-conda--mamba):

```bash
mamba env update -f environment.yml --prune
```

**An environment that installed from GitHub** (`environment.github.yml`, or another project built on `environment.consumer.yml`). There is no working copy to pull, so reinstall the package — and only the package:

```bash
pip install --upgrade --force-reinstall --no-deps \
  "git+https://github.com/Jack-H-Buckner/coastal_sst_data.git@v0.2.7"
```

Use `@main` for the tip, or `@<tag>` / `@<commit-sha>` to pin (see [Pin the version for reproducibility](#use-it-in-another-project-as-a-library) above). **`--no-deps` is not optional here**: conda-forge supplies the compiled geospatial stack, and letting pip resolve dependencies pulls wheels that mix native runtimes and can segfault at import or during the met/regrid stages. `--force-reinstall` is what makes pip replace a same-version-numbered install that has moved underneath the tag.

### Credentials

Most data products stream from open archives that need **no login** (Landsat, land-cover, bathymetry, met, tides). Only **ECOSTRESS, MODIS, and MUR** require a free [NASA Earthdata](https://urs.earthdata.nasa.gov) account, and secrets never go in the config — they live in `~/.netrc` (or env vars). See [Authenticating to data services](#authenticating-to-data-services) for the details.

### Basic usage

**1. Write a config.** A project is one YAML file describing the time range, the areas of interest (grouped into regions), and which products to acquire. A minimal example:

```yaml
name: my_project
output_dir: ./data                 # where everything is written

time:
  start_date: "2023-07-01"
  end_date: "2023-07-31"

auth:
  earthdata:
    auth_strategy: netrc           # only needed because mur/ecostress are selected

products:
  mur:                             # gap-free SST backbone   (NASA Earthdata)
  ecostress:                       # high-res thermal scenes (NASA Earthdata)
  bathymetry:                      # static depth covariate  (no login)
  landcover:                       # static water mask       (no login)
  tides:                           # tide-height forcing     (no login)

regions:
  - name: oregon_coast
    areas:
      - name: tillamook_bay
        center_lat: 45.52
        center_lon: -123.925
        buffer_ns_km: 25
        buffer_ew_km: 15
```

A fuller, annotated example lives at [`examples/config.test.yaml`](examples/config.test.yaml). Every product's options (and which are set at the project vs. region level) are documented under [Data sources](#data-sources).

**2. Sanity-check, then run.** Build up from the cheap offline checks to the full run:

```bash
coastal-sst-data validate --config config.yaml           # config valid? products implemented?
coastal-sst-data grids    --config config.yaml --plot     # do the AOIs land where you expect?
coastal-sst-data verify   --config config.yaml            # do the Earthdata credentials connect?
coastal-sst-data run      --config config.yaml --assemble # acquire everything, then build the cubes
```

**3. Use the result.** Acquisition writes one aligned file per product under `output_dir` (e.g. `data/ECOSTRESS/aligned/<aoi>/…`), and `--assemble` knits them into one analysis-ready Zarr cube per AOI:

```python
import xarray as xr

cube = xr.open_zarr("data/datacube/tillamook_bay.zarr")
print(cube)                        # sensors, covariates, masks on a common daily grid
cube["eco_sst"].isel(time=0).plot()
```

See [Command line interface](#command-line-interface) below for every subcommand and flag.

## Command line interface

Installing the package provides the `coastal-sst-data` command (equivalently, `python -m coastal_sst_data.cli`). Every command is driven by a project config file passed with `--config`; add `-v` for debug logging.

There are nine subcommands:

| Command | What it does | Network |
| --- | --- | --- |
| `validate` | Load and validate the config; print a summary of the products, each one's auth backend, and whether it is implemented yet | no |
| `grids` | Show the target grid (CRS, size, bounding box) computed for each AOI | no |
| `verify` | Connect to every credentialed service the selected products need and confirm the credentials work | yes |
| `run` | Run the pipeline: compute the shared grid once, then acquire each selected product in order | yes |
| `assemble` | Knit the aligned per-product outputs into one analysis-ready datacube (`.zarr`) per AOI | no |
| `preprocess` | Post-assembly: add the derived channels (waterline, gap-filled level-4, screened SST) to the assembled cube | no |
| `extract` | Pull point time series out of the assembled cubes at a CSV of lat/lon sites, as one long-format parquet/CSV table (optional; needs an `extract:` block) | no |
| `provenance` | Print a built cube's provenance: the config that made it, each field's sources, access dates | no |
| `check` | Scan the output tree for truncated or incomplete files (reads each payload, not just its header); `--repair` deletes them so the next run re-fetches | no |

**A typical workflow** builds up from cheap, offline checks to the full run:

```bash
# 1. Is the config valid, and is everything I selected actually implemented?
coastal-sst-data validate --config config.yaml

# 2. Do the AOIs project to sensible grids?
coastal-sst-data grids --config config.yaml

# 3. Do my credentials actually connect?
coastal-sst-data verify --config config.yaml

# 4. Preview what would be acquired (searches, but downloads nothing)
coastal-sst-data run --config config.yaml --dry-run

# 5. Run it for real, then knit the outputs into per-AOI datacubes
coastal-sst-data run --config config.yaml --assemble

# 5b. …or the same thing with several products/AOIs downloading at once
coastal-sst-data run --config config.yaml --assemble --jobs 8

# 6. (optional) pull the cubes' values at a list of lat/lon sites into one table
coastal-sst-data extract --config config.yaml --points sites.csv
```

Steps 4 and 5 are the slow ones, and both can be overlapped — see [Running in parallel](#running-in-parallel).

### `run`

Computes the shared grid once and dispatches each selected product to its acquisition module in a fixed order — static covariates and the MUR backbone first, then Landsat *before* `modis_ref` (its coincidence filter reads the Landsat outputs). Before downloading anything it runs the credential preflight automatically, so a bad or missing credential fails in seconds rather than mid-run. A product whose module isn't implemented yet is skipped with a warning; a product that errors is logged and the run continues to the next, with a per-product outcome summary at the end.

- `--aoi <name> …` — restrict to specific areas of interest (default: all).
- `--products <name> …` — run only these products, e.g. `--products mur landsat modis` (each must be selected in the config).
- `--dry-run` — search each source but download/write nothing (also skips the preflight).
- `--overwrite` — reprocess and overwrite outputs that already exist.
- `--no-verify` — skip the credential preflight (not recommended for a real run).
- `--jobs <N>` — acquire N `(product, AOI)` pairs at once (default: `runtime.jobs`, or 1). See [Running in parallel](#running-in-parallel).

### `verify`

Runs the credential preflight on its own — handy before a long unattended run, or when setting up credentials for the first time. It actually *connects* to each backend (not just checks that a file exists), and exits non-zero listing **every** failing backend if any credential does not connect. Restrict to specific products' backends with `--products`. See [Authenticating to data services](#authenticating-to-data-services) for what each backend needs.

### `assemble`

The terminal stage: once the products are acquired, it knits their per-AOI aligned files into one analysis-ready **Zarr datacube per AOI** (`<output_dir>/datacube/<aoi>.zarr`), on a common **daily** time axis and the same shared grid every product was regridded onto. It reads only files already on disk, so it needs no network — run it after a `run`, or fold it into one with `run --assemble`.

Each cube keeps SST **separate per sensor** (`mur_sst`, `eco_sst_<version>`, `lst_sst`, `modis_sst_<platform>`, `modisref_sst`) so a downstream model can learn per-source offsets; each high-res sensor carries its own `valid` mask and overpass hour. When a sensor has several granules on one day, the **clearest** (most valid pixels) is the base; for Landsat, ECOSTRESS and MODIS the others then fill only the cells it left invalid, so an AoI straddling two Landsat path/rows — or seen twice a night by two overlapping MODIS orbits — keeps both parts instead of losing one. `modis_ref` keeps the clearest granule alone, because its `footprint_id` indices restart per granule. Either way the day reports **one** overpass — the base granule's — and `<sensor>_hour` with it. The cube ships **raw ingredients** on a common grid and daily axis — MUR ships its observed values with honest NaN gaps (no fill), and the raw land-cover water layer ships as `landcover_water` rather than an opinionated derived land mask.

`modis_ref` additionally ships `modisref_footprint_id` (`int32`, `-1` = no observation): the index of the native ~1 km swath pixel each grid cell was resampled from, so grouping by it gives exactly the fine-grid cells one MODIS reading covers — the grouping a Landsat-to-MODIS calibration needs. **The ids identify a pixel within one day's chosen scene only**: they restart at 0 in every granule, so group within a timestep, never across the time axis. The channel appears only when the aligned granules actually carried the layer (it is optional at acquisition via `products.modis_ref.footprint_id`); granules acquired before it, or with it off, are not re-fetched automatically — use `modis_ref --overwrite` to backfill.

- `--aoi <name> …` — assemble only specific AOIs (default: all).
- `--overwrite` — rebuild cubes that already exist.
- `--dry-run` — report what would be assembled; write nothing.
- `--memory-budget-gb <GB>` — override the memory budget for this run (default: detected, halved). Also what `runtime.assemble_jobs` divides between concurrent AOIs — see [Running in parallel](#running-in-parallel).

Storage is tuned by the optional `datacube:` config block — `chunks` (the `(time, y, x)` chunking), `met_time`, `block_days` / `memory_budget_gb` (see below), and a `compression` block (Blosc codec, level, shuffle). Compression is **lossless**: values are kept as float32 / uint8 and only entropy-coded, so smooth and interpolated fields still shrink substantially (byte-shuffle on continuous channels, bit-shuffle on the integer masks) without discarding any precision. (Met-at-overpass is configured on the `met_overpass` product, not here — see below.)

Memory is bounded by `block_days` and `memory_budget_gb`. A cube costs `channels × days × height × width × 4` bytes to build, which grows with the AOI **and** with the date range — a multi-year window on a large grid can want hundreds of GB and simply be killed. So the assembler builds and writes a **block of days at a time**, making peak memory a function of the block instead of the window. The defaults handle this on their own: `block_days: auto` sizes each AOI's block from its own grid and a memory budget, and an AOI that fits is still assembled in one pass exactly as before. Set `block_days` to an integer to pin it, or `memory_budget_gb` when the assembler shares a machine — or when the detected figure is not the allowance the job really has, which is the usual case on a scheduler, where the kernel kills the process at its cgroup limit rather than at the hardware's. The chosen block, the budget, and where the budget came from are logged for every AOI:

```
=== assembling Hobart (2404 days, grid=1218x1507) ===
  34 channel(s), 30 of them (t,y,x): 233 MB/day; budget 16.0 GiB (half of cgroup v2)
    -> 76 block(s) of 32 day(s), time chunk 32, peak ~14.6 GiB
```

If the budget cannot fit even one time chunk's worth of days, the cube's time chunk is reduced to match the block (loudly), so that every block boundary stays chunk-aligned — an append landing mid-chunk forces Zarr to rewrite every chunk it touches.

> **Note (raw-output simplification).** The cube ships **raw ingredients** on a common grid and daily axis; masking, water-filling, station snapping, and multi-input derivations are downstream modelling determinations. The `fill_mur_water`, `fill_cmems_water`, and `water_level` keys were **removed** — MUR/CMEMS ship observed values with honest NaN gaps, there is no derived `landmask`, and water level is reconstructed downstream from the raw per-source `elevation_<dem>` + `depth_<dem>` + `tide_<src>` channels plus each DEM's `datum_offset_m` / `datum_status` attributes. An old config that still sets any of these three keys now **fails validation** rather than being silently ignored. Those computations are not gone — they are the opt-in [`preprocess`](#preprocess) stage now, which runs *after* assembly and adds them to the cube under **names of their own**, so the assembled channels keep the values their sources delivered.

#### Provenance: what produced each field, and when

Every cube is **self-describing**. Assembly embeds, in the Zarr's own attributes, the config that built it and — for every field — the source(s) it came from and when those data were accessed. Nothing lives in a sidecar that can be separated from the data: copy the cube, keep the record.

| Attribute | Contents |
| --- | --- |
| `config_yaml` | the **full text** of the config that built the cube, verbatim (comments and all) |
| `config_sha256`, `config_path` | its hash and where it lived, so drift is detectable |
| `created_at`, `package_version` | when the cube was assembled, and by which version |
| `provenance` | per **field**: its inputs, their sources, when accessed, and on what basis |
| `provenance_products` | per **product**: source, file count, access window |

Read it with `coastal-sst-data provenance --config config.yaml` (add `--fields` to list every channel and the products behind it):

```
=== tillamook_bay ===
  built    : 2026-07-14T15:32:00Z  (coastal_sst_data 0.0.1)
  config   : /path/to/config.yaml
  sha256   : 0eadc006…
  sources:
    cmems        cmems_mod_glo_phy_my_0.083deg_P1D-m    2026-07-14T15:31:58  n=2
    insitu       IOOS Sensors ERDDAP                    2026-07-14T15:32:00  n=1
    mur          GHRSST MUR-JPL-L4-GLOB-v4.1            2026-07-14T15:32:00  n=2   [date from file mtime, not recorded]
```

Two things this is careful about:

**A guessed date is never passed off as a recorded one.** Acquisition stamps `acquired_at` into every file it writes. Data acquired *before* this existed has no stamp, so the access date falls back to the file's **mtime** — and every record says which basis it used. An mtime is wrong the moment a tree is rsynced or restored from backup, so that has to be legible rather than silent. Re-acquire (or `--overwrite`) to replace a guessed date with a recorded one.

**Derived fields list all of their inputs.** `eco_airtemp_hrrr` (the forcing at the ECOSTRESS overpass) is met_overpass *and* the ECOSTRESS overpass that set the instant; `insitu_sst` is the in-situ network *and* the met reference time it is sampled at. Picking one to report would be tidy and wrong. A channel with no mapping is logged loudly rather than shipping blank.

Because the config is embedded, `provenance` also **detects drift**: if the config on disk has changed since the cube was built, it says so and prints both hashes. And because every distinct-data product now ships **one channel per source** (`depth_cudem` vs `depth_gmrt`, `airtemp_hrrr` vs `airtemp_era5`), each channel attributes to exactly **one** source — provenance names it unambiguously, with no "which source produced this day" guesswork.

#### Met in the cube: forcing (per source) and per-overpass products

**A day gets one met value per channel, taken at a fixed *time of day* — not a daily mean.** A mean over `[0, 6, 12, 18]` UTC averages pre-dawn and mid-afternoon forcing together, which is the wrong thing to hand a model of a sensor that flew at one instant. The cube's forcing channels — `airtemp_<src>` / `wind_u_<src>` / `wind_speed_<src>` / `swrad_<src>` / `cloud_cover_<src>`, **one per stacked met source** (`airtemp_hrrr`, `airtemp_era5`, …) — therefore come from the **reference-time snapshot**: by default **10:30 local solar time**, Landsat's overpass.

The basis is *solar*, not UTC, because a fixed UTC hour is a different time of day in every AOI — mid-morning in Oregon, the middle of the night in Maine — so cross-AOI forcing would not be like-for-like. Each AOI's reference instant is derived from its own longitude (`UTC = local − lon/15`, rounded to the hour, rolling the date where it crosses midnight). Change it with `products.met.reference_time` / `reference_basis`, or set `datacube.met_time: daily_mean` to get the old averaging behavior back. If no reference files exist (an older MET tree), the assembler falls back to the daily mean rather than emitting an empty channel, and records which it used in the cube's `met_time` attribute.

**Met at each sensor's own overpass is a separate product, `met_overpass`.** A weather model's value at 14:32 is *not* reconstructable from a daily sample, so it is a real acquisition (not a downstream derivation). It snapshots the thermal scenes and the cube emits `<sensor>_<var>_<src>` — `lst_airtemp_hrrr`, `eco_wind_speed_era5`, and so on — so an ECOSTRESS scene at 03:00 and a Landsat scene at 19:00 on the same day see *different* forcing rather than sharing one value. You name exactly which pairings to produce with `products.met_overpass.combinations` (a list of `[sensor, source]`), so the channel count is what you opt into, not a sensor×source cross-product. These follow the **exact scene the cube kept** — when a sensor flies twice in a day, the forcing follows the *base* (clearest) scene's timestamp. On a mosaicked Landsat/ECOSTRESS day that means the forcing describes the base granule alone, even where a cell was filled from another granule of the same day; the `mosaic_time` attribute on those channels says so. Days with no scene stay NaN.

**Tide at each sensor's overpass is `tide_overpass`** (`<sensor>_tide_<src>`, e.g. `eco_tide_coops`), configured with `products.tides.overpass_combinations`. Unlike met, tide *is* reconstructable — it is the smooth tide series interpolated to the scene's hour — so it is a derived cube channel, not a separate acquisition. It is what a downstream process needs to put a scene's ground on the tide-adjusted waterline.

#### Water level: reconstructed downstream from raw ingredients

Water level (and the submerged/exposed classification of a tidal flat) is a **modelling determination**, so the cube no longer ships a pre-derived `<sensor>_water_elev` / `<sensor>_water_class`. It ships instead the **raw ingredients** to reconstruct any of them per-process — all *per source*, so a downstream model chooses which DEM and which tide source to combine:

| Ingredient | Dims | Role |
| --- | --- | --- |
| `elevation_<dem>` | (y, x) | that DEM's ground/seafloor elevation, in its native vertical datum (+ up); carries its own `datum_offset_m` / `datum_status` attributes |
| `depth_<dem>` (+ `depth_p25`/`depth_p75`) | (y, x) | mean water depth where the DEM is known |
| `tide_<src>`, `tide_range_<src>` | (time,) | the daily tide statistics from that source's series |
| `<sensor>_tide_<src>` | (time,) | *ready-made:* the tide already interpolated to that sensor's overpass instant (the `tide_overpass` product) |
| `<sensor>_hour` | (time,) | the exact overpass hour of the scene the cube kept (NaN where none) |

A downstream process references `elevation_<dem>` to MSL with **that DEM's** `datum_offset_m` attribute, takes the tide at the scene from `<sensor>_tide_<src>` (or interpolates `tide_<src>` to `<sensor>_hour` itself), and classifies each cell — exactly the computation the assembler used to hard-code, now made per-process. The built-in [`preprocess`](#preprocess) stage's `water_line` step is exactly this computation, materialised into the cube for you.

**Datum.** Tides are relative to **MSL** (a *tidal* datum — the 19-year mean of observed water level at a gauge), but a DEM need not be: CUDEM is **NAVD88** (a *geodetic* datum). The gap between the two surfaces is **local** and far from negligible — MSL sits roughly **1.0–1.4 m above NAVD88 in the Pacific Northwest** (vs. ~0.1–0.3 m on the Gulf coast), which is comparable to the entire intertidal range, so ignoring it misclassifies much of a tidal flat. Each DEM source's offset is **resolved automatically as it is acquired** (NOAA VDatum for CUDEM/NAVD88, cross-checked against the nearest CO-OPS gauge; 0 for GMRT, which is already ~MSL) and ships as attributes on that source's `elevation_<dem>` channel. It can't be a config constant: the right value depends on **which DEM** it belongs to (CUDEM and GMRT on the same AOI need different offsets), which is exactly why it rides per-source rather than as one cube-wide number. Re-run `bathymetry --overwrite` to re-resolve it.

How each source resolves (during the bathymetry acquisition of *that* source):

| DEM source | Vertical datum | Offset |
| --- | --- | --- |
| **CUDEM** | NAVD88 | looked up from [NOAA VDatum](https://vdatum.noaa.gov/) (NAVD88 → LMSL), **cross-checked** against the nearest CO-OPS gauge's published datums |
| **GMRT** | ~sea level | `0.0` — no network call; GMRT is already sea-level referenced |

VDatum is sampled at **several points spread across the AOI's waterline band**, not at its centroid: coverage is patchy at point scale (the Padilla Bay centroid falls in a hole while points a few km away resolve fine), and out-of-coverage comes back as a `-999999` sentinel rather than an error. The median is taken; if the samples span more than 0.5 m the AOI straddles tidal zones and a single scalar is refused rather than averaged into something wrong everywhere. The result (method, sample count, spread, uncertainty, the gauge it was cross-checked against) is stamped onto that DEM source's `elevation_<dem>` cube channel — there is no separate stage or sidecar, so `assemble` stays entirely offline and the offset can never go stale against the DEM it belongs to. Re-run `bathymetry --overwrite` to re-resolve.

**Regional fallback.** Set a region's best-estimate offset under `regions[].sources.bathymetry.datum_offset_m` (the elevation of MSL in the DEM's datum, in metres). It is a **fallback, not an override**: a value VDatum (or the CO-OPS cross-check) resolves is authoritative and **wins**, so the config value is used **only** when the offset would otherwise be unresolved. In that case the offset takes your `datum_offset_m` with `datum_status = "regional_default"` (instead of the biased `0.0` / `unresolved_assumed_zero`). This covers both **no coverage** (VDatum has no coverage and no NAVD88 gauge within 30 km) and a **wide spread** (the AOI straddles tidal zones, >0.5 m across it, so a single resolved scalar is refused). A **transient VDatum outage** (network/5xx) still saves the DEM and retries the offset alone on the next run (`datum_status = "pending_retry"`), but while pending it stamps your `datum_offset_m` rather than `0.0` — so the retry can still find the better local value, without a metre of bias in the meantime. With no `datum_offset_m` set, an unresolved offset is still `0.0` / `unresolved_assumed_zero`, loudly, so the bias is visible in the artifact. (The old "override always wins" behaviour, and its >0.5 m-on-an-MSL-DEM rejection, are gone: GMRT always resolves structurally, so a fallback never reaches it.)

### `preprocess`

An **opt-in** stage that runs *after* `assemble` and turns the raw cube's ingredients into the
**downstream determinations the cube deliberately does not bake in** (masking, water-filling, water
level — see the note above). It opens each `datacube/<aoi>.zarr`, adds its derived channels, and
rewrites **that same store** atomically — one cube per AOI holding the raw analysis-ready product and
the "cleaned" modelling product side by side, so nothing downstream has to join two stores.

**Assembled channels are never overwritten.** Every derived channel gets a name of its own, so
`eco_sst_v002` still holds what the sensor delivered while `eco_sst_v002_clean` holds the screened
product. That is also what makes the stage **idempotent**: each step seeds from the raw channel and
rewrites only its own outputs, so re-running with new thresholds needs no re-assembly and never
compounds the previous run's drops. Re-running with an unchanged step selection is a no-op; change a
threshold and it re-runs on its own.

Like acquisition it is driven by a registry: each step declares what it reads/writes, and adding one
is a single registration (see the
[developer guide](docs/DEVELOPMENT.md#5b-adding-a-post-assembly-preprocess-step)).

Ten steps ship today:

- **`water_line`** — the tide-adjusted waterline per thermal sensor, at that sensor's overpass. Emits
  `<sensor>_water_elev` (metres relative to the waterline: 0 at it, + exposed, − submerged) and
  `<sensor>_water_class` (submerged / exposed / unknown), reconstructed from `elevation_<dem>` (+ its
  `datum_offset_m`) and the overpass tide — exactly the computation [Water level](#water-level-reconstructed-downstream-from-raw-ingredients)
  describes, now materialised.
- **`fill_water`** — nearest-neighbour fill of the level-4 SST products' (`mur_sst`, `cmems_*`) NaN gaps
  over water (`landcover_water == 1`) into `<channel>_gapfilled`, beside the untouched source channel,
  with a `<channel>_filled` companion mask so a filled value never passes for an observed one.
- **`filter_clouds`** — screens cloud-contaminated ECOSTRESS pixels against a **gap-free L4 baseline**
  (`mur_sst` / a `cmems_*` analysis). Cloud biases thermal-IR SST **cold**, so a pixel far colder than
  the baseline is treated as cloud. Two `method`s: **`offset`** (default) drops where
  `baseline − eco > threshold_k` (default `5.0` K; `≤0` disables), using the co-located baseline —
  the `fill_water` gap-filled one if present, which is why this step runs after `fill_water`; **`sigma`**
  builds the baseline's **own climatology** (least-squares mean + residual σ) and drops any pixel colder
  than `mean − n_sigma·σ` (default `3.0`), with `stat_scope: pixel` (per-cell, default) or `pooled` and
  `seasonality: off` (default) or `harmonic` (a day-of-year annual cycle). Drops fold into a
  **screened product** — `<sensor>_valid_<ver>_clean`, and a NaN'd `<sensor>_sst_<ver>_clean` (unless
  `mask_sst: false`) — seeded from the raw channels, which keep their own values, plus a
  `<sensor>_sst_<ver>_clean_cloudfiltered` flag (`1 = dropped`) so every drop stays auditable.
- **`filter_cloud_cover`** — gates ECOSTRESS on the **met total-cloud-cover** field (HRRR/ERA5, in
  percent). Two independent gates — a pixel drops if **either** fires: **scene-level** (`scene_max_pct`,
  default `30`) drops the *whole* overpass when the AOI-mean cloud cover at the overpass exceeds it, and
  **per-pixel** (`pixel_max_pct`, default `80`) drops individual pixels above it (a `null` / `≥100`
  threshold disables that gate). Prefers the overpass-matched `<sensor>_cloud_cover_<src>`, falling back
  to the daily forcing `cloud_cover_<src>`; `source` (default auto: overpass over forcing, `hrrr` over
  `era5`) picks. Emits a `<sensor>_scene_cloud_pct_<src>` diagnostic and a
  `<sensor>_sst_<ver>_clean_metcloudfiltered` flag. It **composes** with `filter_clouds` — both write
  the same `_clean` product, so their drops union regardless of order.
- **`filter_land_clouds`** — screens cloud-contaminated pixels **over land** against the **near-surface
  air temperature** (HRRR/ERA5). Over land the thermal-IR surface reads *warmer* than the air by day, so
  a land pixel far colder than the air is likely cloud: it drops where `airtemp − sst > threshold_k`
  (default `5.0`; a K difference equals a °C one, so `5` means 5 K ≡ 5 °C regardless of the cube's unit;
  `≤0` disables). The drop is gated to **land** cells, chosen by `land_source`: **`landcover`** (default)
  uses the static `landcover_water` mask (land = water == 0), or **`water_line`** uses this sensor's
  `<sensor>_water_class == exposed` at its overpass (which requires the `water_line` step to be enabled).
  Air temperature prefers the overpass-matched `<sensor>_airtemp_<src>` over the daily forcing
  `airtemp_<src>`; `source` (default auto: overpass over forcing, `hrrr` over `era5`) picks. Drops fold
  into the same `<sensor>_sst_<ver>_clean` / `_valid_<ver>_clean` product plus a
  `<sensor>_sst_<ver>_clean_landcloudfiltered` flag, and compose with the other two filters. This is the **land**
  counterpart to `filter_clouds`, which screens **water** against an L4 SST baseline.
- **`flag_georef`** — diagnoses per-scene ECOSTRESS **georeferencing error**. Some granules are
  geolocated wrong, contributing wrong SST at every coastal pixel with nothing to flag it. This step
  registers the scene's **thermal coastline** (NaN-aware Canny edges) against the **static**
  `landcover_water` coastline by brute-force **whole-cell translation**, scoring every candidate shift,
  and classifies the scene. The fit runs on the **cloud-filtered** SST (cleaner edges → better fit), so
  it runs *after* the cloud filters — but it **moves no pixels**. It emits per-sensor 1-D `("time",)`
  diagnostics only: `<sensor>_georef_dy` / `_dx` (best-fit whole-cell offset), `_shift_m` (magnitude, m),
  `_z` (peak significance vs the search surface), `_agree` / `_agree0` (edge/coast agreement at the peak
  / at zero shift), `_n_edge`, `_coast_obs` (coastline cells observed), `_dt_k` (land–sea contrast,
  diagnostic), and `<sensor>_georef_flag` — `0 = ok, 1 = displaced, 2 = suspect, 3 = unstable,
  4 = unrecoverable, 5 = insufficient_signal`. **Quality gates run *before* the fit**: a scene observing
  too little coastline / too few edges is `insufficient_signal` and never fitted (this is where most of
  the discriminating power lives — see [`docs/plan-georef-preprocess-step.md`](docs/plan-georef-preprocess-step.md)).
- **`correct_georef`** — applies `flag_georef`'s fitted `(dy,dx)` to the **raw** ECOSTRESS geometry, for
  **`displaced` scenes only**, into `<field>_georef_corrected` channels (**SST + validity** by default).
  It shifts the raw granule-native fields read from the raw cube — **not** the preprocess-derived masks:
  the corrected geometry invalidates the cloud/water preprocessing, which must be **re-run** against the
  corrected images (a separate later step). The shift is slice-based (never `np.roll`, which would
  fabricate a coastline on the far edge); the vacated margin is **NaN-filled** for SST (`0` for masks).
  A `<sensor>_georef_applied` flag (`1 = shifted`) marks which scenes were moved, and non-displaced
  scenes are copied verbatim, so `<field>_georef_corrected` is a complete drop-in replacement.
- **`filter_clouds_corrected`** / **`filter_cloud_cover_corrected`** / **`filter_land_clouds_corrected`**
  — **re-run the cloud filters on the corrected geometry.** `correct_georef` shifts the *raw*
  (unfiltered) SST, so `<pre>_sst<ver>_georef_corrected` still contains clouds; and the pre-fit filter
  pass compared ECOSTRESS pixel-by-pixel against MUR/met/landcover while the scene was *misregistered*,
  so those comparisons were wrong at the coast. Each corrected step reuses its base filter's **exact
  math and config** (it **inherits** `filter_clouds` / `filter_cloud_cover` / `filter_land_clouds` —
  list it empty, `{}`, or override any key) against the corrected channels, writing a **separate clean
  product** `<pre>_sst<ver>_georef_corrected_clean` (+ `<pre>_valid<ver>_georef_corrected_clean` and
  `..._georef_corrected_clean_{cloud,metcloud,landcloud}filtered` audit flags). They compose with each
  other exactly like the raw filters, and leave `<pre>_sst<ver>_georef_corrected` intact so each stage
  stays inspectable. Re-running the masking on the corrected geometry (rather than shifting the stale
  drops) is the whole point — the clean channel is the correctly-georeferenced, cloud-screened product.

The three cloud filters require their inputs to be in the raw cube: `filter_clouds` needs a level-4
baseline (`mur` or `cmems`) and the ECOSTRESS SST channels; `filter_cloud_cover` needs a `met` or
`met_overpass` cloud-cover channel; `filter_land_clouds` needs a `met` / `met_overpass` **air-temp**
channel and a land source (`landcover`, or the `water_line` step). `flag_georef` needs a **non-degenerate
`landcover_water`** channel (`landcover` product) and the ECOSTRESS SST; `correct_georef` needs
`flag_georef` selected. Where an input is absent the step logs a warning and does nothing.

**Memory is bounded the same way `assemble` bounds it**, by `block_days` and `memory_budget_gb` — this stage reads a whole cube and writes a larger one, so it has the same problem. It processes and writes a **block of days at a time**, and both keys default to *inheriting the `datacube` value*, so configuring the assembler usually configures this too. Set `preprocess.block_days` / `preprocess.memory_budget_gb` when this stage needs a **smaller** block than assembly did, which it can: it holds the channels it *reads* as well as the ones it derives.

Two behaviours worth knowing:

- **The cube's existing chunking is preserved.** Preprocess inherits the store's on-disk time chunk rather than re-imposing `datacube.chunks` — it has to, because the assembler may have deliberately *reduced* that chunk to fit its own memory budget, and silently undoing that would make the cube unwritable again. (`preprocess:` therefore has no `chunks` or `compression` of its own; one cube, one encoding, taken from `datacube`.)
- **Blocking does not change any value.** Nine of the ten steps are day-local, so a block sees exactly what the whole cube would have shown it. The exception is `filter_clouds` with `method: sigma`, whose climatology is a least-squares fit over the whole time axis: that fit is *streamed* across the blocks and solved once, so a blocked run and an unblocked one produce the same cutoff. Nothing about the result depends on the block size.

Enable it in the config (nothing runs otherwise), then run it folded into a `run` or on its own:

```yaml
preprocess:
  enabled: true
  # Memory blocking, as datacube.block_days / memory_budget_gb. Unset -> inherit the datacube
  # value. Set them when preprocessing needs a smaller block than assembly did.
  # block_days: auto
  # memory_budget_gb: 16
  steps:
    water_line: { dem_source: cudem, tide_source: coops, sensors: [eco, lst] }
    fill_water: { sources: [mur, cmems] }
    # ECOSTRESS cloud screening vs the gap-free L4 baseline (cold-deviation).
    filter_clouds: { method: offset, baseline: mur_sst, threshold_k: 5.0, sensors: [eco] }
    #   ...or the distribution-based, seasonally-aware variant. Note this one builds its cutoff
    #   from the baseline's OWN climatology, so it wants a dense baseline: pairing it with
    #   `products.mur.overpass_sensors` leaves fewer samples per pixel (see the MUR section).
    # filter_clouds: { method: sigma, baseline: mur_sst, n_sigma: 3.0, stat_scope: pixel, seasonality: harmonic }
    # Meteorological cloud-cover gate (HRRR/ERA5, %): scene-level + per-pixel.
    filter_cloud_cover: { scene_max_pct: 30, pixel_max_pct: 80, sensors: [eco] }
    # Over-land screen vs the air temperature (drop where airtemp - sst > threshold_k).
    filter_land_clouds: { threshold_k: 5.0, land_source: landcover, sensors: [eco] }
    #   ...or gate on the tide-adjusted water line instead of the static landcover mask
    #   (needs the water_line step above):
    # filter_land_clouds: { threshold_k: 5.0, land_source: water_line, sensors: [eco] }
    # Diagnose ECOSTRESS georeferencing error vs the static landcover coastline (flags only).
    flag_georef: { sensors: [eco], tol_m: 200, max_shift_m: 10000, min_coast_obs: 500, min_edges: 300 }
    # Apply the fitted shift to the RAW ECOSTRESS SST + validity (displaced scenes only).
    correct_georef: { sensors: [eco], fields: [sst, valid] }
    # Re-run the cloud filters on the corrected geometry -> *_georef_corrected_clean.
    filter_clouds_corrected: {}          # inherits filter_clouds config (override any key here)
    filter_land_clouds_corrected: {}     # inherits filter_land_clouds config
```

**`flag_georef` parameters** (defaults shown; every key is optional):

| key | default | what it does |
|---|---|---|
| `sensors` | `[eco]` | sensor prefixes to diagnose |
| `tol_m` | `200` | how close (m) a thermal edge counts as *on* the coast (200 m ≈ 2 cells at 100 m) |
| `max_shift_m` | `10000` | half-width of the shift search (± metres) |
| `coarse_stride` | `4` | coarse-pass stride (cells) before the fine refinement |
| `n_refine` | `8` | number of top coarse peaks refined at ±`coarse_stride` (refining several, not just the winner, is required) |
| `sigma` | `1.5` | Canny Gaussian σ (cells) |
| `lo_pct` / `hi_pct` | `80` / `97` | Canny hysteresis low/high percentiles — **lower the LOW one** to recover faint coastal edges without admitting land clutter |
| `min_coast_obs` | `500` | **gate:** skip a scene observing fewer coastline cells (an **absolute count** — coastlines differ ~10× in length between AOIs; the strongest predictor of a spurious fit). **Region-overridable.** |
| `min_valid_pct` | `2.0` | **gate:** skip a scene with less than this % of the grid valid. Region-overridable. |
| `min_edges` | `300` | **gate:** skip a scene with fewer thermal edges. Region-overridable. |
| `z_min` | `4.5` | minimum peak significance (`z`) for an `ok` or `displaced` verdict |
| `lift_min` | `2.5` | the peak must exceed `lift_min × surface-median` agreement, else `unrecoverable` (chance level is AOI-specific, so score on **lift**, not absolute agreement) |
| `gain_min` | `0.10` | minimum agreement gain (peak − zero-shift) for `displaced` |
| `ok_shift_m` | `300` | a fit no larger than this (with `z ≥ z_min`) is `ok`, not `displaced` |
| `stability_windows_km` | `[5, 10, 20, 30]` | search half-widths swept for stability; a peak that drifts > 1–2 cells across them is `unstable` (never corrected) |

**`correct_georef` parameters:**

| key | default | what it does |
|---|---|---|
| `sensors` | `[eco]` | sensor prefixes to correct |
| `fields` | `[sst, valid]` | which **raw ECOSTRESS-native** fields to shift; add `cloud` to also shift the granule cloud band. Preprocess-derived masks are never shifted |
| `fill` | `NaN` | margin fill for float fields (SST); mask fields always fill `0` (unobserved) |

The gates `min_coast_obs` / `min_edges` / `min_valid_pct` are **per-region tunable** — a dense
archipelago coastline warrants a higher `min_coast_obs` than a short one. Override them per region
(not globally) under `regions[].preprocess_steps`:

```yaml
regions:
  - name: puget_sound
    preprocess_steps:
      flag_georef: { min_coast_obs: 1500 }     # denser coastline than the project default
    areas: [ ... ]
```

The corrected-pass filters shift only what `correct_georef` shifts. If `filter_clouds.use_cloud_raster`
is on (it also drops on the native `<pre>_cloud<ver>` band), that band must be shifted too, so add it
to `correct_georef`: `correct_georef: { fields: [sst, valid, cloud] }`.

```bash
coastal-sst-data run --config config.yaml --assemble --preprocess   # acquire, assemble, preprocess
coastal-sst-data preprocess --config config.yaml --aoi tillamook_bay # just this stage
```

- `--aoi <name> …` — only specific AOIs. `--overwrite` — re-derive channels the cube already carries
  (not normally needed: a changed step selection re-runs on its own). `--dry-run` — report only. An AOI
  whose cube hasn't been assembled yet is skipped with a warning (run `assemble` first). There is one
  store either way — open it with `xr.open_zarr("data/datacube/<aoi>.zarr")`. The cube's
  `preprocess_channels` attribute lists exactly which channels this stage owns.

### `extract`

**Optional, and only if you ask for it.** Nothing in `run`/`assemble`/`preprocess` reads
this; a project with no `extract:` block is completely unaffected, and the parquet backend
is an extra rather than a dependency.

`extract` is the **transpose** of everything else in the package: the pipeline turns
scattered granules into a dense `(time, y, x)` cube, and this pulls that cube's values back
out at a list of sites, as one long-format table.

```bash
# what would be extracted, and how many rows -- opens no cube, writes nothing
coastal-sst-data extract --config config.yaml --points sites.csv --dry-run

coastal-sst-data extract --config config.yaml --points sites.csv
# -> <output_dir>/extract/points.parquet
```

The **points file** is a CSV with a latitude, a longitude, and (ideally) an id. Column
names are matched case-insensitively against `lat`/`latitude`/`y`, `lon`/`longitude`/`x`,
and `id`/`point_id`/`station`/`name`/`site`; a file carrying two candidates for the same
field is rejected rather than guessed at, and every coordinate is range-checked against
WGS84 so a file of **projected metres** (which is what an `x`/`y` file usually is) fails
loudly instead of silently landing in the ocean. Extra columns are ignored — join them back
on `point_id`.

Each point is assigned to exactly **one AOI**: the one whose grid contains it, and if
several do, the one whose grid centre is nearest. A point inside no AOI is dropped with a
warning naming it, and if that leaves nothing the run fails rather than writing an empty
file. Points are **never** snapped to the nearest water pixel — a site is where you said it
is.

**The output** is one row per (point, AOI, time, variable, stat):

| column | |
| --- | --- |
| `point_id`, `lat`, `lon` | from your points file — the coordinates you gave, not the pixel centre |
| `aoi` | which cube the value came from |
| `time` | the cube's date; **empty** for a static `(y,x)` channel, which gets one row rather than a copy per date |
| `variable` | the cube channel name, unmodified |
| `stat` | which statistic this row is |
| `radius_m` | the neighbourhood this row actually **used** (a `nearest` row is always `0`, whatever the channel declares) |
| `value` | the number |

**Configuration** lives in an `extract:` block, and the channels are **explicit** — a
channel you do not list is not extracted, and a channel you list that the cube does not
have is a hard error naming it (with a spelling suggestion). A silently missing column in a
modelling table is indistinguishable from a channel that was genuinely all-NaN.

```yaml
extract:
  points: sites.csv          # or pass --points
  format: parquet            # or csv
  channels:
    depth_cudem:                                          # bare key = nearest pixel
    tide_coops:
    eco_sst_clean: { radius_m: 300, stat: [nanmean, nanstd, count_valid] }
    mur_sst:       { radius_m: 1000, stat: mean }
    lst_sst:       { radius_m: 300, stat: nanmean, mask: water }
```

`radius_m` is a **disc in metres on the ground**, in the AOI's projected CRS — never
degrees, never pixels — measured from the point's exact position to each pixel centre. (A
box of the same nominal size would reach `radius x 1.41` into its corners.) The pixel the
point falls in is always included, so a radius below the grid posting degenerates to that
one pixel; the run says so rather than returning a column of NaN. A disc clipped by the
grid edge still returns a value, over a partial disc — which is why `count` exists, and why
it is worth requesting alongside any radius.

The statistics:

| stat | |
| --- | --- |
| `nearest` (the default) | the value of the pixel the point falls in. **Not** the nearest *finite* value: if that pixel is cloudy the answer is NaN, because filling from just offshore is how a validation set quietly acquires a warm bias. Ignores `radius_m` and `mask`. |
| `mean` `median` `std` `min` `max` `sum` | plain NumPy — **NaN propagates**: one cloudy pixel in the disc gives NaN |
| `nanmean` `nanmedian` `nanstd` `nanmin` `nanmax` `nansum` | the same, **skipping** NaN |
| `count` | pixels in the disc after masking, finite or not — how edge-clipping and mask shrinkage become visible |
| `count_valid` | **finite** pixels in the disc — how many observations the value actually rests on |
| `p10` … `p97.5` | percentiles (NaN-skipping) |

`mean` and `nanmean` are both offered because they are different questions, and which one
you got should be visible in the output rather than decided for you. Ship `count_valid`
next to any reducing statistic.

`mask: water` restricts the disc to the cube's `landcover_water` cells (any other value
names a static `(y,x)` channel to use instead), so a coastal point's mean is not
contaminated by land. The mask wins over the include-the-containing-pixel rule: a point
whose own cell is masked out contributes nothing, and an entirely masked-out neighbourhood
gives NaN with `count == 0` rather than quietly falling back to the unmasked window.

Other flags: `--aoi` restricts which cubes are read (and gives the output its own filename,
so a one-AOI run cannot overwrite a full one), `--out` sets the path outright, `--format
parquet|csv` overrides the config, and `--overwrite` replaces an existing table.

**Parquet needs `pyarrow`**, which is *not* a core dependency — install it with
`pip install 'coastal_sst_data[extract]'` or `conda install pyarrow`. `--format csv` works
with no extra at all.

### `validate` and `grids`

Neither touches the network or credentials, so they are safe quick sanity checks anywhere. `validate` prints the project summary and flags any selected product that has **no implementation yet**, so you catch a silently-skipped product before running. `grids` prints the concrete grid each AOI resolves to (auto-selected UTM zone, pixel dimensions, and bounding box) — useful for spotting an AOI that is far larger or smaller than intended.

`grids` can also **draw the AOIs on a map** with `--plot`, which is often the fastest way to catch a mistyped coordinate or a box that lands in the wrong place:

```bash
coastal-sst-data grids --config config.yaml --plot
```

This writes an **overview map** of every AOI, coloured by region, plus **one zoomed map per region** with that region's AOIs highlighted (neighbouring regions faded in for context). Each AOI is drawn as its acquisition bounding box with a labelled centre point.

- `--plot` — produce the maps (PNG).
- `--plot-dir <path>` — where to write them (default `<output_dir>/figures/`).
- `--show` — also open the figures interactively.

Plotting is an **optional** capability: it needs `matplotlib` (install with `conda install matplotlib`). If `cartopy` is also installed the maps gain coastlines and land/ocean shading; without it they fall back to plain longitude/latitude axes.

## Running in parallel

A full run spends most of its wall clock **waiting on other people's servers** — a catalogue search here, a granule download there, one product at a time, one AOI at a time. Most of that waiting is independent, and the pipeline can overlap it.

Parallelism is **opt-in and off by default**. Nothing about it changes what a run produces: the same files land in the same places, and `--jobs 1` takes the original serial code path rather than a one-worker emulation of the new one.

```bash
# Acquire 8 (product, AOI) pairs at once
coastal-sst-data run --config config.yaml --jobs 8 --assemble
```

or set it once in the config:

```yaml
runtime:
  jobs: 8
  assemble_jobs: 2
```

### What runs at the same time

The unit of work is a **`(product, AOI)` pair**. Two things decide what may overlap:

**The product dependency graph.** Only three ordering constraints exist, and each is a case of one product reading another's aligned files: `modis_ref`'s coincidence filter reads Landsat's, and both MUR's `overpass_sensors` filter and `met_overpass` read the thermal sensors'. The other nine products — bathymetry, CMEMS, ECOSTRESS, Landsat, MODIS, met, tides, landcover, in-situ — depend on nothing and all start immediately.

**Those edges are per-AOI.** Every one of those reads is scoped to an AOI, so `mur(hobart)` waits for `landsat(hobart)` but **not** for `landsat(tamar)`. There is no barrier between AOIs: Hobart can be downloading MUR while Tamar is still working through Landsat.

If a stage fails for one AOI, the other AOIs carry on, and the products that depended on it **for that AOI** are reported as skipped rather than run against a directory that was never written — an important distinction, because a product that ran against missing inputs looks exactly like one that found no data.

### Per-service limits — how many calls one account may make

> *"Can I send multiple requests to the same API on one account?"*

Mostly yes, but the answer differs enough per service that a single worker count cannot express it. So there are **two** limits: `jobs` bounds the worker pool, and **gates** bound each service. Products that share a *server* share a gate — `mur`, `modis`, `modis_ref` and `ecostress` are four products behind one Earthdata account, so a cap on "Earthdata" is what actually protects the account.

| Gate | Products | Default | Why this number |
| --- | --- | --- | --- |
| `earthdata` | mur, modis, modis_ref, ecostress | 6 | Several granule reads at once are normal, and `earthaccess` already threads internally — so the real connection count is a multiple of this. CMR *search* is the throttled part. |
| `pc` | landsat, landcover | 4 | Planetary Computer is **anonymous** — there is no account and no per-account limit. Throttling is per-IP, and the STAC search endpoint is the sensitive half. |
| `herbie` | met, met_overpass | 4 | HRRR via Herbie. |
| `copernicus` | cmems | 1 | The lazy dataset handle carries its own client and is not safe to share; the toolbox parallelises internally already. |
| `noaa_small` | tides | 1 | Small public metadata API behind a shared HTTP session. |
| `erddap` | insitu | 1 | Same. |
| `dem` | bathymetry | 1 | One CUDEM tile-index cache file, shared by every AOI. |

Override any of them in the config:

```yaml
runtime:
  gates:
    earthdata: 8
```

Raise them when the run moves closer to the data — in-region on AWS (`us-west-2` for Earthdata Cloud), the sensible Earthdata number is far higher than it is over a home link.

**Credentials are shared, not multiplied.** All workers run in one process, so they share one login per backend, one refresh budget and one rate-limit window. Eight workers acquiring three Earthdata products across two AOIs produce **exactly one** login — the same as `--jobs 1`. This is the main reason the pipeline uses threads rather than separate processes: separate processes would each hold their own credential state, so eight workers would become eight independent login storms against one account.

### Assembly and preprocessing: bounded by memory, not by the network

`assemble` and `preprocess` get their own, much smaller knob:

```yaml
runtime:
  assemble_jobs: 2
```

Each AOI owns its own `<aoi>.zarr`, so AOIs are independent — but `assemble(aoi)` and `preprocess(aoi)` read and rewrite the *same* store, so those two are chained rather than overlapped. That chaining is also the win: one AOI preprocesses while the next is still assembling, instead of every AOI waiting at a barrier between the stages.

**The memory budget is divided between them.** The assembler normally detects the machine's memory and halves it, on the assumption that it is the only thing running — true for one AOI at a time, false the moment two run. With `assemble_jobs: 4` on a 200 GB machine each AOI gets roughly 25 GB rather than 100 GB, and `block_days: auto` simply produces smaller (still chunk-aligned) blocks. The division is logged:

```
=== terminal stages: 6 AoI(s), 2 at a time; memory budget 100.0 GiB (half of physical RAM) -> 50.0 GiB each ===
```

Keep `assemble_jobs` low. Acquisition failures cost a retry; an OOM here throws away all the downloading, which is the expensive part.

### Reading the output

Eight workers interleaving into one log would otherwise be unreadable, so in a parallel run every line is stamped with the task that produced it — including output from the underlying libraries:

```
16:05:02 INFO [tillamook_bay/landsat] === AOI: tillamook_bay (CRS=EPSG:32610 grid=308x505 @ 100m) ===
16:05:02 INFO [padilla_bay/ecostress] You're now authenticated with NASA Earthdata Login
16:05:02 INFO [tillamook_bay/tides]   [dry-run] would build tides (coops) for tillamook_bay
```

A serial run adds no prefix and prints exactly what it always did.

### Suggested rollout

```bash
# 1. Serial baseline — the reference run report
coastal-sst-data run --config config.yaml --jobs 1 --dry-run

# 2. Same thing in parallel: proves the graph and the gates, downloads nothing
coastal-sst-data run --config config.yaml --jobs 8 --dry-run

# 3. One AOI, for real. The written/skipped/failed tallies should match the serial run
coastal-sst-data run --config config.yaml --aoi <one> --jobs 8 --assemble --preprocess

# 4. Confirm nothing was half-written (reads each file's payload, not just its header)
coastal-sst-data check --config config.yaml
```

Step 4 is the one worth not skipping: it is what would catch a concurrent-write problem, and it reports a clean tree when there is none.

### What is *not* parallelised

- **Within a product's download loop.** Granules and scenes inside one `(product, AOI)` task are still fetched one at a time.
- **Across time.** One run covers one date range; splitting a long window into shards is a possible future addition, and would only be safe for the per-day/per-scene products (tides and in-situ each write a single file spanning the whole range).
- **Blocks within one cube.** The assembler's time blocks are order-dependent appends into one store.

For coarser parallelism than this, `--aoi` already shards cleanly, so a scheduler (Slurm array jobs, separate containers) can run whole AOIs as independent jobs. Note that separate *processes* do not share credential state, so give each one a smaller share of the gate budgets.

**Sharing one `output_dir` between concurrent runs is safe**, including when their AOI lists or date ranges overlap. Every write goes to scratch named for the host and process that made it and is renamed into place only when finished, so two runs writing one file both complete and the later one wins — each file on disk is always a whole file. A run will say so when it notices:

```
WARNING   leaving MH_20040327.nc.part-node07-41233-... alone -- node07:41233 may still be writing it
WARNING   MH_20040327.nc is being written by more than one run at once (node07:41233); both writes
          complete and whichever finishes LAST wins -- check for overlapping --aoi lists or date ranges
```

That is a report, not an error — but it usually means two jobs are doing the same work twice, which is worth fixing in the sharding.

Two consequences worth knowing:

- **`check --repair` refuses to run while another job is writing the tree.** It reports that job's in-flight scratch as `IN USE` and stops, because deleting it is precisely the failure atomic writes exist to prevent — and its verdict on everything else is a moving target while another run is still producing output. Wait for the run to finish, then repair. `--force` overrides it, for the one case the liveness check cannot get right on its own: a machine rebooted while scratch was open and the owning pid has since been reused, so nothing will ever call that file dead.
- **Scratch from a run that died clears on the next run**, immediately if the run was on this machine (`os.kill(pid, 0)` proves the writer is gone), and after six hours (`store.STALE_SCRATCH_S`) if it was on another node, where the clock is the only evidence available. A discard says which of those decided it.

What this does *not* make safe is two runs **assembling the same cube**: `preprocess` reads and rewrites `<aoi>.zarr` in place, so one run replacing the store under another's open reader is a hazard the write path cannot address. Shard cubes by AOI.

## Authenticating to data services

Most data products stream from **open archives that need no login** — Landsat and landcover from Microsoft Planetary Computer, bathymetry from NOAA CUDEM/GMRT, met from HRRR and Google's public ERA5, and tides from NOAA CO-OPS. Only a few products need credentials, and the package is built so that adding a new authenticated service later is a small, contained change.

**The golden rule: secrets never live in the config file (or the repo).** The config records only *how* to authenticate — a strategy name, or non-secret identifiers like a Google Cloud project. The actual usernames, passwords, and keys stay in the standard locations each service already uses (`~/.netrc`, a key file, environment variables), outside version control.

Auth requirements are declared **per product — and per source** where a product has several. A product served from an anonymous source needs nothing: Landsat via its default `pc` (Planetary Computer) source requires no credentials, whereas Landsat via a future `gee` source would require Google Earth Engine auth.

**Which products need what:**

| Backend | Required by | Credentials |
| --- | --- | --- |
| `earthdata` | ECOSTRESS, MODIS, MODIS_REF, MUR | free NASA Earthdata account |
| `copernicus` | CMEMS | free [Copernicus Marine](https://data.marine.copernicus.eu) account |
| `gee` | Landsat / landcover with `source: gee` | Google Earth Engine |
| *(none)* | Landsat (`pc`), landcover (`esa`), bathymetry, met, tides | — |

**Declaring auth in the config.** Add an `auth:` block with a sub-block per backend you use. These hold only non-secret settings:

```yaml
auth:
  earthdata:
    auth_strategy: netrc      # netrc | environment | interactive
  copernicus:
    auth_strategy: netrc      # only if you select the cmems product
  gee:
    project: my-gcp-project   # only if you use a gee source

  # Mid-run re-authentication (both optional; these are the defaults)
  max_age_s: 1800             # replace a credential older than this at the next safe boundary
  max_refreshes: 20           # per backend per run; exceeded -> a real failure, not a retry
```

The config is validated on load: if you select a product that needs a backend and its block is missing, loading fails immediately with a clear message (e.g. *"product 'ecostress' requires `auth.earthdata`"*). Products using anonymous sources need no `auth:` block at all.

**Credentials expire; long runs outlive them.** A multi-year AoI is hours of downloading, and an Earthdata token or a Planetary Computer signature minted at the start is dead well before the last granule. The failure is nastier than it sounds: granules are processed in **date order**, so an expiring token looks exactly like the record simply ending partway through — and the next AoI, starting with a fresh token, works for another hour.

So the pipeline treats "your credential" as a distinct answer from "the server is busy" and from "it isn't there":

- **Reactive** — a call rejected with a credential-shaped error triggers one re-authentication and one retry. If the *fresh* credential is rejected the same way, the log says so explicitly (`still rejected after a credential refresh — this is not an expiry`), because at that point the credentials themselves are wrong.
- **Proactive** — a credential older than `max_age_s` is replaced at a safe boundary (between AoIs, and every 50 items), so the usual case is replacing it *before* anything fails.
- **Bounded** — refreshes are rate-limited per backend, so hundreds of failing granules cannot cause hundreds of logins, and capped by `max_refreshes`, so a genuinely bad credential surfaces as a failure instead of an endless retry loop.

> **`auth_strategy: interactive` is never auto-refreshed.** Re-authenticating would block on a password prompt, which on an unattended overnight run is worse than failing. It fails immediately with a message naming the backend. Use `netrc` or `environment` for long runs.

**Verify credentials before a full run.** Before downloading anything, the pipeline runs a **preflight that actually connects** to every service the run needs — so a wrong password or a missing key fails in seconds, not hours in. You can run that check yourself at any time:

```bash
# verify the configured credentials connect, then exit
python -m coastal_sst_data.pipeline --config config.yaml --verify-only
# or the standalone checker
python -m coastal_sst_data.auth --config config.yaml
```

A normal run performs this preflight automatically first; pass `--no-verify` to skip it, or use `--dry-run` (a preview, which skips the preflight).

### NASA Earthdata

Used by **ECOSTRESS, MODIS, and MUR** (all streamed with `earthaccess`). You need a free Earthdata account: <https://urs.earthdata.nasa.gov>. Set the strategy in the config (`auth.earthdata.auth_strategy`) and put the credentials in the matching place:

- **`netrc`** (recommended) — add a line to `~/.netrc` and `chmod 600` it:
  ```
  machine urs.earthdata.nasa.gov login <username> password <password>
  ```
- **`environment`** — set the `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables.
- **`interactive`** — prompts for your username/password when the pipeline runs (not suitable for unattended runs).

### Google Earth Engine

Reserved for the **`gee` sources** of Landsat and landcover (a more robust water mask from JRC Global Surface Water + Landsat NDWI). Configure with `auth.gee`:

- `project` (required) — your Google Cloud project id with the Earth Engine API enabled.
- `service_account` + `key_file` (optional, **both or neither**) — for unattended/service use. `key_file` is the *path* to the service-account JSON key; the key itself stays outside the repo. Omit both to use the token created by `earthengine authenticate`.

## Defining data grids

All data products are mapped to a common grid within each AOI. The end product is a matching grid of observations for each variable, for each day. These matched data points can then be used to fit grid cell level models or fed into machine lenaring lagorithms like convolutional nerual networks that require a grid based structure. 

The grids are defined by a target resolution (defaults to 100m) and a cordinate reference system is chosen using the UTM zone assocaited with the longitude of the center of the AOI. 


## Data sources

Each data product is implemented as a process module in `src/coastal_sst_data/processes/`. Options are set at two levels:

- **Project level** — global options for a product, set once under the top-level `products:` block. These apply to every AOI.
- **Region level** — options that vary geographically, set per region under `regions[].sources:`. Products whose best source depends on location (currently bathymetry and tides) use these; they override the project-level default for the AOIs in that region.

Two options are shared by every product's project-level block:

- `output_format`: `netcdf` (default) or `geotiff`.
- `overwrite`: reprocess and overwrite outputs that already exist (default `false`).

Products that read from NASA Earthdata (ECOSTRESS, MODIS, MUR) also require an `auth.earthdata` block; see [Authenticating to data services](#authenticating-to-data-services).

### Source coverage & stacking

Several products draw the *same* variable from more than one provider, and **no single provider covers the whole globe**. A U.S.-only tide network, a North-America-only weather model, a CONUS-only DEM — each is authoritative where it reaches and absent everywhere else. Older versions of this package resolved that with a **source *chain***: a primary provider and a `fallback`, merged per day, so a channel might silently switch providers mid-record and a single `mur_sst` column could be two different models stitched together.

That is gone. Products that read one variable from several providers now **STACK** them: one channel *per source*, named for its source, side by side, **no fallback and no merging**. `airtemp_hrrr` and `airtemp_era5` ride in the cube together; where HRRR doesn't reach, `airtemp_hrrr` is simply NaN and `airtemp_era5` carries the value. The consumer chooses — the cube never chooses for you, and provenance for `airtemp_hrrr` names **only** HRRR. Each source writes to its own tree, `<PRODUCT>/<source>/aligned/<aoi>/`, and the channel suffix is the source tag.

The catch every consumer must know: **which sources actually cover which regions and windows.** A stacked channel is honest about its gaps (they read as NaN), but you have to expect them.

| Product | Source (tag) | Spatial coverage | Temporal window | Native resolution |
|---|---|---|---|---|
| **ecostress** | `v002` | global | 2018→ (older reprocessing) | ~70 m, per overpass |
| | `v003` | global | 2019→present (newer reprocessing) | ~70 m, per overpass |
| **bathymetry** | `cudem` | U.S. coasts (CONUS + territories) only | static | ~1/9″ (≈3 m) |
| | `gmrt` | global | static | ~coarser, variable (100 m–1 km near shore) |
| **met** / **met_overpass** | `hrrr` | North America only (CONUS + fringe) | 2014→present | 3 km, hourly |
| | `era5` | global | 1940→present (~5-day lag) | ~31 km, hourly |
| **cmems** | `my_global` (GLORYS12 reanalysis) | global ocean | 1993→~present (reanalysis lag) | 1/12° (~9 km), daily |
| | `anfc_global` (global forecast) | global ocean | ~recent → +10 day forecast | 1/12° (~9 km), daily |
| | regional tags (e.g. `anfc_med`, `anfc_nws`, `anfc_bal`) | one regional sea only | forecast window | ~2–4 km, daily |
| **tides** | `coops` (CO-OPS gauges) | U.S. tide stations only | station record | observed, sub-hourly |
| | `eo_tides` (global tide model) | global | any date (harmonic prediction) | model, arbitrary sampling |

Rule of thumb: **`cudem`, `hrrr`, and `coops` are the U.S./North-America high-resolution sources; `gmrt`, `era5`, `my_global`/`anfc_global`, and `eo_tides` are the global fallbacks you stack alongside them.** Outside their coverage the high-resolution channel is all-NaN by design — that is the gap the table documents, not a bug. Pick the `sources` list per region accordingly (see each product's **Region-level options**).

### ECOSTRESS
ECOSTRESS provides the core high resolution thermal images used in the analysis. ECOSTRESS has overpasses every 1 to 5 days with overpasses occuring at differnt times of day providing a unique data set for capturing ocean temperatures under a range of conditions.

- **Where it comes from**: the `ECO_L2T_LSTE` tiled Land Surface Temperature & Emissivity product from the NASA Earthdata catalog, streamed with `earthaccess` (windowed HTTP range reads — the full ~110 km tile is never downloaded).
- **What it measures**: ~70 m land surface temperature (LST, which approximates skin SST over water), plus the accompanying `cloud`, `water`, and `QC` masks. Outputs `sst` (K or °C), `cloud`, `water`, and a derived `valid` layer (water & clear & finite SST).
- **Collection versions are STACKED, not picked-one** (like CMEMS/bathymetry sources). ECOSTRESS ships in two overlapping collections with **asymmetric coverage**: `v002` starts earlier (back to 2018) while `v003` is the newer reprocessing that reaches the present. Neither alone spans the full record, so you list the `versions` you need and each is acquired into its own `ECOSTRESS/<ver>/aligned/` tree and shipped as its **own cube channel-set** — `eco_sst_v002`, `eco_sst_v003`, and likewise `eco_valid_<ver>`, `eco_cloud_<ver>`, `eco_hour_<ver>`. A day a version doesn't cover is an honest NaN slice in that version's channel, which you fill by stacking the other. The per-overpass **matchup** channels (met at overpass, tide at overpass, in-situ) stay unqualified (`eco_airtemp_hrrr`, `eco_tide_coops`, `eco_insitu_sst`): the versions describe the same physical overpasses, so they share one overpass identity, taking the first-listed version's scene each day and falling back to later ones.

**Project-level options** (`products.ecostress`):

- `short_name`: Earthdata collection short name (default `ECO_L2T_LSTE`).
- `versions`: list of collection version tags to stack (default `[v002]`; e.g. `[v003, v002]` to span 2018→present). The **first-listed** version wins the shared overpass identity per day. (The old scalar `version:` was removed — a config still setting it fails validation with a pointer to `versions`.)
- `layers`: mapping of output role → COG asset suffix (default `sst`/`lst`→`LST`, `cloud`→`cloud`, `water`→`water`, `quality`→`QC`).
- `categorical`: which layers are resampled with the categorical method (default `cloud`, `water`, `quality`).

**Region-level options**: none.

### Landsat
Landsat contributes additional high-resolution thermal scenes and the water/cloud masks derived from its surface-reflectance and QA bands.

- **Where it comes from**: the `landsat-c2-l2` (Collection 2, Level-2) collection served by Microsoft Planetary Computer via a STAC search + windowed Cloud-Optimized GeoTIFF reads. Planetary Computer signs asset URLs anonymously, so **no credentials are required**. The module (`landsat_pc.py`) is one interchangeable source behind a common contract; future `landsat_aws` / `landsat_gee` sources honour the same output schema.
- **What it measures**: Landsat 8/9 surface temperature from the thermal band (`sst`, scaled to K or °C). `water` is derived from NDWI (green vs. NIR surface reflectance) and `cloud` from the `QA_PIXEL` cloud/shadow/dilated bits, optionally buffered by the `ST_CDIST` cloud-distance band. Also emits the derived `valid` layer.

**Project-level options** (`products.landsat`):

- `source`: which Landsat backend to use (default `pc` = Planetary Computer).
- `collection`: STAC collection (default `landsat-c2-l2`).
- `stac_url`: STAC API endpoint (default the Planetary Computer catalog).
- `platforms`: platforms to include (default `landsat-8`, `landsat-9`).
- `cloud_cover_max`: scene-level cloud-cover cutoff as a fraction 0–1 (default `0.7`).
- `masking.ndwi_threshold`: NDWI cutoff for classifying water (default `0.0`).
- `masking.cloud_buffer_km`: distance to buffer cloud/shadow pixels using `ST_CDIST` (default `1.0`; set `0` to disable).

**Region-level options**: none.

### MODIS
MODIS is a ~1 km thermal sensor in its own right, alongside ECOSTRESS and Landsat. It is far coarser than either, but it is well calibrated, it images an AoI several times a day, and its record runs from 2000 — an order of magnitude more observations than the high-resolution sensors over the same window.

There are **two MODIS products**, because there are two different questions:

| | question | shape |
|---|---|---|
| `modis` | what was the SST over this AoI, per overpass | a thermal **sensor** — stacked per platform, no dependencies |
| `modis_ref` | what did MODIS see **at the moment Landsat did** | a calibration **reference** — Landsat-coincident, carries footprint ids |

- **Where it comes from**: the GHRSST MODIS L2P skin-SST products from NASA OB.DAAC via `earthaccess` — `MODIS_T-JPL-L2P-v2019.0` (Terra) and `MODIS_A-JPL-L2P-v2019.0` (Aqua). They are swath (2D curvilinear) products, so they are regridded with nearest-neighbour resampling (`pyresample`) to preserve the observed values rather than smoothing them.
- **What it measures**: skin sea-surface temperature (`sst`, K or °C), quality-filtered on the GHRSST `quality_level` band, plus a derived `valid` layer. MODIS arrives already quality-filtered with no water or cloud layer, so it publishes **no** `modis_cloud` channel — an all-zero one would read as "this scene was never cloudy", which is a claim the files do not make.

#### Overpasses, and why the time of day is a decision

MODIS flies on **two** spacecraft, both sun-synchronous at ~705 km with a 2330 km swath (1 km at nadir, ~5 × 2 km at the swath edge):

| Platform | Node | Nominal day | Nominal night |
|---|---|---|---|
| Terra | descending AM | 10:30 local | 22:30 local |
| Aqua | ascending PM | 13:30 local | 01:30 local |

Consecutive same-day orbits are spaced `40075·cos(φ)/14.56` km apart — about 2752 km at the equator (leaving gaps), but **overlapping above ~32° latitude**. So at mid-latitudes each platform sees an AoI roughly **twice per day and twice per night**: once near nadir and once near the swath edge, each pass covering a different part of it. Terra + Aqua therefore deliver on the order of **4–8 granules a day**, which is why this product mosaics a platform's same-day granules rather than keeping only the clearest.

**Two things about the record you have to plan around:**

1. **The overpass times drift.** NASA stopped maintaining both orbits — Terra's last inclination maneuver was Feb 2020, Aqua's Mar 2021. Terra's morning crossing has since moved 10:30 → 10:15 (2022) → ~09:00 (2025); Aqua's afternoon crossing 13:30 → ~15:50 (2026). The night crossings moved with them. **A 2000–2026 series therefore has hours of local-time drift baked into it**, so anything that assumes a fixed local hour (a diurnal correction, say) is wrong at the ends. This is why `time_of_day` is decided on **computed local mean solar time**, not on a fixed clock and not on the granule's `-D-`/`-N-` filename token alone — see `night_solar_hours`. Every aligned granule records its own `solar_hour` attribute so the drift stays auditable.
2. **The missions are over.** Terra MODIS ended production Dec 2025 and Aqua MODIS Aug 2026. MODIS is a **closed 2000–2026 archive**, not a forward-processing source. VIIRS is its successor; it is not implemented here.

**Which time of day to pick.** Night (`time_of_day: night`, the default for the standalone product) gives the higher-quality SST retrieval — no solar contamination and no diurnal skin warming — and roughly doubles the observation count relative to daytime-only. Day matches the illumination state of daytime Landsat and ECOSTRESS scenes, which is what a cross-sensor matchup wants. `modis_ref` therefore defaults to `day`, and `modis` to `night`.

**A note on day binning.** Like every other sensor, granules are binned by **UTC** day. At Puget Sound (UTC−8) Terra's ~22:30 and Aqua's ~01:30 local passes of the *same night* both land on the same UTC day (06:30 and 09:30 UTC), so the binning is coherent. Near UTC+0 one local night splits across two cube days.

#### `products.modis` — the standalone sensor

- `platforms`: which spacecraft to stack, e.g. `[terra, aqua]` (default: both). **Distinct data, stacked** — each writes its own `MODIS/<platform>/aligned/` tree and its own `modis_sst_<platform>` channel set, because Terra and Aqua never observe the same overpass. **Order matters**: the first listed wins the single overpass identity that the in-situ / met / tide matchups key off. With no list configured the preference falls back to alphabetical, i.e. `aqua`.
- `time_of_day`: `night` (default), `day`, or `both`. Judged on local mean solar time at the AoI.
- `night_solar_hours`: the local-solar window that means "night", wrapping midnight (default `[19, 5]`). Wide enough to hold both platforms' night crossings across the whole record including the drift.
- `short_name`: pin every platform to one Earthdata collection. Unset (default) means each platform uses its own.
- `variable`: SST variable to read (default `sea_surface_temperature`).
- `quality_min`: minimum GHRSST quality level to keep, 0–5 (default `4`; 5 is best).
- `regrid_radius_m`: nearest-neighbour search radius in metres (default `1500`).
- `access`: fetch backend, `harmony` (default) or `download`. **This matters at scale.** A full L2P granule is ~15–25 MB holding a global swath and ~15 variables, of which this module reads four; a night-time Terra+Aqua series over the full archive is tens of thousands of granules. `harmony` asks PO.DAAC's subsetter for the AoI bounding box and only the needed variables, so a few hundred KB crosses the wire instead of tens of GB, and the reader never materialises a full swath. `download` fetches whole granules and crops in memory — no extra dependency, and fine for short ranges. Harmony needs `harmony-py` (in the `modis` extra).
- `daytime_only`: **deprecated** — `true` maps to `time_of_day: day`, `false` to `both`.

#### `products.modis_ref` — the Landsat calibration reference

Same instrument and the same swath machinery, restricted to overpasses coincident with an acquired Landsat scene. Writes a flat `MODIS_REF/aligned/` tree and contributes `modisref_*` channels.

- `match_landsat`: only load granules within `max_time_diff_minutes` of an already-acquired Landsat scene (default `true`; requires Landsat to have run first — the pipeline enforces the ordering).
- `max_time_diff_minutes`: coincidence window (default `360`, i.e. ±6 h).
- `time_of_day`: default `day`, since Landsat flies in the morning.
- `short_name`: default `MODIS_T-JPL-L2P-v2019.0` — Terra, whose 10:30 crossing sits within minutes of Landsat's. Aqua's 13:30 is three hours off and would fold the diurnal warming cycle into the calibration.
- `footprint_id`: emit the swath-pixel-index layer (default `true`). A ~1 km observation covers many grid cells, so grouping by it recovers exactly the fine-grid cells one MODIS reading saw — the grouping a Landsat-to-MODIS calibration needs. The assembler carries it into the cube as `modisref_footprint_id` (see [`assemble`](#assemble)).
- `access`: `download` (default) or `harmony`. **`harmony` is refused while `footprint_id` is on**, and this is not a limitation to work around: the ids are indices into the *native* swath, and a server-side subset trims the swath and renumbers them, so the same id would mean different observations in different AoIs. Set `footprint_id: false` if you only need the SST.
- `variable`, `quality_min`, `regrid_radius_m`: as above.

**Region-level options**: none, for either product.

#### Migrating from the single `modis` product (≤ 0.3.2)

Before 0.4.0 there was one `modis` product: Terra only, daytime only, Landsat-coincident by default, writing a flat `MODIS/aligned/` tree and contributing `modis_sst` / `modis_footprint_id`. That product is now `modis_ref`, and `modis` is the standalone sensor.

- **Config**: rename your `products.modis` block to `products.modis_ref` to keep the old behaviour. Add a `products.modis` block only if you want the standalone sensor.
- **Channels**: `modis_sst` → `modisref_sst`, `modis_footprint_id` → `modisref_footprint_id`, and so on. The standalone sensor's channels are suffixed by platform: `modis_sst_terra`, `modis_sst_aqua`.
- **Sensor prefixes** in `met_overpass.combinations`, `tides.overpass_combinations`, `mur.overpass_sensors` and the `cloud_filter` / `georef` step `sensors` lists: `modis` now means the standalone sensor. Use `modisref` for the reference. An unknown prefix is rejected at config load with a suggestion, so a stale name fails loudly rather than silently producing no matchups.
- **Existing data**: an old `MODIS/aligned/` tree is orphaned. Move it to `MODIS_REF/aligned/` to keep it, or delete it. Left in place it is now *ignored with a log line* rather than loaded as a phantom platform tag — see `docs/bug-empty-version-tag-channels.md`.

### MUR
MUR is the always-present, gap-free SST backbone the high-resolution products add detail onto.

- **Where it comes from**: the `MUR-JPL-L4-GLOB-v4.1` GHRSST MUR L4 analysis (daily, ~1 km, global) from PO.DAAC via `earthaccess`. For each day the AOI window is subset out of the global granule — either client-side over HDF5 range reads or server-side via OPeNDAP (see `access` below) — then bilinearly upsampled onto the AOI grid. The global file is never fully downloaded either way.
- **What it measures**: `analysed_sst`, a gap-free (cloud-free) foundation SST analysis (`sst`, K or °C). Because it is an L4 analysis there is no cloud mask; `valid` is simply finite SST (i.e. water).

**Project-level options** (`products.mur`):

- `short_name`: Earthdata short name (default `MUR-JPL-L4-GLOB-v4.1`).
- `variable`: variable to read (default `analysed_sst`).
- `pad_deg`: degrees of padding added around the AOI lat/lon window before subsetting (default `0.05`).
- `access`: fetch backend, `download` (default) or `opendap`. See below.
- `overpass_sensors`: fetch only the days these sensors flew (default: unset — every day). See below.

**Region-level options**: `overpass_sensors`.

#### `access: opendap` — server-side subsetting

`download` (the default) streams the granule over HTTPS and subsets client-side. That works, but
the granule is remote HDF5 read through fsspec in **8 MB blocks with background prefetch**, so the
scattered header reads needed to locate one AOI window drag far more over the wire than the window
contains. `opendap` instead asks PO.DAAC's Hyrax server for a DAP4 hyperslab — the window itself.

```yaml
products:
  mur:
    access: opendap
```

Measured live on one granule over a 50 km AOI (Washington coast, 100 m grid):

| `access` | bytes on the wire | wall time |
|---|---|---|
| `download` | 26.83 MB | 9–14 s |
| `opendap` | 0.03 MB (+ a one-off 0.14 MB axis read per process) | ~4 s |

The gridded output is **bit-identical** — the backend selects exactly the index range
`.sel(lat=slice(...))` selects, so both paths reproject from the same source window.

Same credentials: Hyrax sits behind Earthdata Login and reuses the session `earthaccess` already
holds — there is nothing extra to configure. One caveat: if OPeNDAP returns 401/403 and a
credential refresh does not clear it, the Earthdata profile has probably never authorized the
OPeNDAP application. Approve it once under *Applications → Authorized Apps* at
`urs.earthdata.nasa.gov`; a fresh token cannot fix an unapproved application. The error says so.

`download` remains the default because it is the path with years of runs behind it. Switch per
config once `opendap` has proved itself on your own AOIs and date range.

#### MUR on overpass days only

MUR's main job downstream is the cloud-filter baseline, which can only ever read a day a thermal
sensor also imaged. Over a multi-year window most of the daily downloads therefore produce a file
nothing reads. `overpass_sensors` restricts the days fetched to the ones a named sensor actually
recorded an overpass over *this* AOI, discovered by reading the sensors' aligned dirs:

```yaml
products:
  mur:
    overpass_sensors: [eco]        # ...or [eco, lst, modis]
```

Name sensors by their **channel prefix** (`eco`, `lst`, `modis`, `modisref`), not the product name — a wrong
name is rejected at config load. It is region-overridable, because which sensors are worth
restricting to genuinely varies (an AOI may be ECOSTRESS-only). Because the filter reads what the
sensors wrote, MUR now runs **after** them in the process order; running `--products mur` alone
against an output dir where those sensors have never run fails loudly rather than silently
downloading nothing.

Three consequences worth knowing:

- **`fill_water` leaves non-overpass days unfilled.** It fills water pixels from `mur_sst`/`cmems_*`;
  where MUR was not fetched there is nothing to fill from.
- **`filter_clouds: {method: offset}` is unaffected** — it compares each day's sensor SST against
  that day's baseline, and only overpass days have sensor SST to filter. **`method: sigma` is
  affected**: it builds its cutoff from the baseline's *own* observed climatology, so a sparser MUR
  means fewer samples per pixel, and pixels with too few samples get a NaN cutoff (kept).
- **The cube's thin-coverage warning for `mur` is suppressed** when this option is set, and the
  `coverage` attr marks the entry `"sparse": true` — the fraction is still measured and reported.

The filter restricts what is **fetched**, not what exists: days written before you set the option
stay on disk and stay in the cube. Conversely, setting it does not backfill — widen or drop the
option and re-run to fetch the rest.

### CMEMS (Copernicus Marine global physics)
CMEMS supplies the **offshore ocean state at depth** — the water column the nearshore exchanges with. Where MUR gives one skin temperature at the surface, this gives temperature (and optionally salinity and currents) *through the water column*, so a model can see the stratification and the offshore water mass that upwelling and tidal exchange draw into an estuary.

- **Where it comes from** — two (or more) STACKED Copernicus Marine physics models (1/12°, daily means, ~50 depth levels), streamed with the `copernicusmarine` toolbox. `open_dataset` subsets the AOI window **server-side and lazily**, so the global model is never downloaded. One channel per source tag, no fallback (D10):
    - **`my_global`** — `cmems_mod_glo_phy_my_0.083deg_P1D-m`, the **GLORYS12 reanalysis** (hindcast). Best quality, but it stops a year or two behind the present; outside its window the channel is simply NaN.
    - **`anfc_global`** — `cmems_mod_glo_phy_anfc_0.083deg_P1D-m`, the **analysis/forecast** product, which reaches the present and out to a ~10-day forecast. It rides *alongside* the reanalysis rather than backfilling it, so `cmems_thetao_0m_my_global` and `cmems_thetao_0m_anfc_global` are never silently conflated — the consumer picks per day.
    - **regional tags** (register in `datasets`) — higher-resolution seas like `anfc_med` / `anfc_nws` / `anfc_bal` (~2–4 km) that a region stacks in place of, or alongside, the global tags.
- **What it measures**: whichever `variables` you select (default `thetao`, sea-water potential temperature; also `so` salinity, `uo`/`vo` currents, and the 2D `zos` / `mlotst`), emitted **once per requested depth**.

**Depths.** The model has ~50 *fixed* levels (0.494, 1.541, 2.646, 5.078 m …), so a requested depth is snapped to the **nearest level** — never interpolated. Every value is therefore one the model actually computed. The channel is named for what you asked for; the level actually used is recorded in the variable's `model_depth_m` attr:

```yaml
products:
  cmems:
    depths: [0, 10, 30]     # metres
```
gives `thetao_0m` (level 0.494 m), `thetao_10m` (level **9.573** m), `thetao_30m` (level **29.445** m). A requested depth more than 5 m from any level logs a warning.

**Project-level options** (`products.cmems`):

- `sources`: the source TAGS to STACK, one channel per tag (default `[my_global, anfc_global]` — GLORYS12 reanalysis + global forecast). There is no `source`/`fallback` chain any more.
- `datasets`: register extra tags → dataset ids for regional models, e.g. `{anfc_med: cmems_mod_med_phy-tem_anfc_4.2km_P1D-m}`; a region then lists that tag in its `sources`. The tag is the provenance identity, so a channel is self-describing.
- `variables`: which fields to acquire (default `[thetao]`).
- `depths`: depths in metres, each snapped to the nearest model level (default `[0]`).
- `pad_deg`: padding around the AOI window (default `0.15`, ≥ one 1/12° cell).

**Region-level options** (`regions[].sources.cmems`): `sources` (which tags to stack here) and `datasets` (register a regional tag).

**In the cube** the channels arrive prefixed and per-source — `cmems_thetao_0m_my_global`, `cmems_thetao_30m_anfc_global` — and ship the model's **observed values with honest NaN gaps**, like MUR. At ~9 km the model's land mask can swallow an entire estuary, so expect NaN over cells it never resolved; filling those from the nearest resolved water column is a downstream determination, not something the cube bakes in.

**Credentials**: a free [Copernicus Marine](https://data.marine.copernicus.eu) account, declared as `auth.copernicus`. The toolbox reads `~/.netrc` natively, so the secret lives there like every other one:

```
machine auth.marine.copernicus.eu login <username> password <password>
```

### In-situ (IOOS + your own CSVs)
In-situ observations are the cube's **only ground truth**. Every other channel is modelled (met, CMEMS, tides) or remotely sensed (ECOSTRESS, Landsat, MODIS, MUR); this is what a thermometer in the water actually read. The assembler writes each station's value into **the grid cell the station sits in**, and — the point of the exercise — **at the instant each satellite flew**, so a scene can be validated against a buoy pixel-for-pixel and minute-for-minute.

In-situ sources **STACK**: `sources: [ioos, csv]` acquires both, each into its own `INSITU/<source>/aligned/` tree, and the cube merges their platforms into **one** station table and one channel set. Public buoys and your own moorings are different *platforms*, not two routes to the same data, so a cube can carry both — with each station recording which source it came from.

- **`ioos`** — the [IOOS Sensors ERDDAP](https://erddap.sensors.ioos.us/erddap): one server aggregating **NDBC, NOAA CO-OPS, CDIP and the IOOS regional associations**, so most of North America is a single query. Stations are auto-discovered inside each AOI's bounding box. **No credentials.** It measures water temperature (`sea_water_temperature`, falling back per station to `sea_surface_temperature` — providers do not agree on the name), quality-flagged with QARTOD. The native sampling interval is kept (6 min for CO-OPS gauges), because matching an overpass needs the sub-hourly series.
- **`csv`** — your own thermometers, in **long format**: one row per observation. See below.

#### Your own observations (`sources: [csv]`)

```csv
station_id,time,latitude,longitude,value,z
mooring_a,2026-06-01T18:00:00Z,45.52,-123.92,11.8,1.0
mooring_a,2026-06-01T18:10:00Z,45.52,-123.92,11.9,1.0
mooring_b,2026-06-01T18:00:00Z,45.48,-123.90,12.4,0.5
```

Rows are **grouped by `station_id`**, so **one file may hold any number of stations** — each becomes its own platform, at its own position, in its own pixel. That is the normal case: a logger-fleet export is one file. The same id split across several files (per-year, per-deployment) merges into one platform, so you needn't pre-concatenate. `path` takes a file, a directory, or a glob, and one file can serve every AOI in the project — each AOI keeps the platforms that fall inside its grid.

```yaml
products:
  insitu:
    sources: [ioos, csv]
    path: ~/data/moorings/*.csv
    columns:                    # only the ones whose names differ from the defaults
      station_id: platform
      time: datetime_utc
      value: temp_c
      z: depth_m
    units: degC                 # degC | degF | K -> converted to degC
    time_zone: UTC              # what a stamp with no offset means
    max_sensor_depth_m: 2       # drop rows whose z is deeper than this
    qc_pass_values: [1, 2]      # values of your qc column to keep; omit to keep every row
```

Column names are configurable because rewriting files you already have is a good way to leave a feature unused; only the four required fields (`time`, `latitude`, `longitude`, `value`) must be present under *some* name. `z` (sensor depth in metres), `qc` and `station_name` are picked up when present — with no `qc` column the rows assert QARTOD `2` (not evaluated), which the default `qc_flags` accept.

**Timestamps.** No format is imposed — `time` is parsed by inference, so ISO 8601 (`2026-06-01T18:00:00Z`), `2026-06-01 11:00:00` and `2026-06-01T11:00:00-07:00` all read; ISO 8601 is the safe choice. A stamp **carrying an offset is honoured**, and `time_zone` is ignored for it. A **naive** stamp means `time_zone` (default `UTC`) — set it to e.g. `America/Los_Angeles` for a logger that wrote local time, and daylight-saving gaps and repeats become unparseable rather than a wrong hour. Rows whose stamp will not parse are dropped, with a count in the log. The project window includes the whole final **day**, so `end_date: 2026-06-30` keeps observations through 23:59 that date.

**Sensor depth.** Give a `z` column (or map your name onto it with `columns:`) and rows deeper than `max_sensor_depth_m` — default `5` — are dropped, so a profiling mooring contributes its surface sensor and not its thermocline. The comparison is on the absolute value, so either sign convention works: `2.0` and `-2.0` both mean 2 m down. A row with a blank `z` survives, since an unstated depth is not evidence of a deep one. Note that `z` only *filters* — the cube's in-situ model has no depth dimension, so **set the limit tight enough that one sensor per station survives**. If several depths remain at the same timestamp, one is kept arbitrarily and the rest are discarded; the merged-station check below won't catch it, because a mooring's depths share one position. Giving each depth its own `station_id` doesn't help either — they land in the same pixel and are averaged together.

Three failures this loader refuses to make quiet, because a *wrong* ground-truth value doesn't look like an error — it looks like the truth:

- **A mistyped `path` is an error**, not an empty channel. "No files matched" and "no thermometers here" must never be the same outcome.
- **A `units` mistake is range-checked.** 54 °F is a perfectly plausible °C sea temperature, so nothing downstream would ever catch it; values outside −5…45 °C after conversion warn loudly with a count.
- **Merged stations are rejected.** If a multi-station file's `station_id` column isn't mapped, every station collapses into one platform. Where they share a time base (synchronised loggers usually do) de-duplication would keep one station's rows and silently discard the rest, so a "platform" reporting one instant from two positions is a hard error naming `station_id`. Where they don't, the moving-platform guard below catches it.

**Moving platforms are not supported yet.** The cube's in-situ model is fixed-station: one position per platform for the whole window. A platform whose observations stray beyond `max_position_drift_m` (default: one grid cell, so GPS jitter passes) is **dropped and reported** — never collapsed onto its median position, which would place a whole track in a pixel it may never have visited. Drifters, gliders and ship transects need per-observation placement, a change to the shared data model and the cube schema; see [`docs/plan-user-provided-insitu-csv.md`](docs/plan-user-provided-insitu-csv.md).

**Quality control.** QARTOD flags are `1` pass, `2` not-evaluated, `3` suspect, `4` fail, `9` missing. The default keeps **`[1, 2]`**: flag 2 is what stations that don't run QARTOD emit, and demanding flag 1 would discard much of the network.

**Stations that report nothing are dropped — loudly.** A station can *advertise* a temperature variable and never report it (NDBC 46120 exposes both temperature names and returns all-NaN for a whole month; it is a wave buoy with no thermometer). Those are logged by name, because an empty in-situ channel that reads as "no buoys here" is the one failure this product cannot afford.

**Project-level options** (`products.insitu`):

- `sources`: which networks to **stack** (default `[ioos]`; `csv` must be opted into, since it needs a `path`).
- `qc_flags`: QARTOD flags to keep (default `[1, 2]`).
- `max_sensor_depth_m`: ignore sensors deeper than this on profiling moorings (default `5`) — the station's `z` variable for `ioos`, your `z` column for `csv`.
- `stations` / `exclude_stations`: an allow-list (else auto-discovery for `ioos`, every platform for `csv`) and a deny-list. Both match on station id, in **either** source — so a 50-platform CSV can be narrowed to the 5 that matter without editing the file.
- `max_position_drift_m`: how far a platform's observations may stray before it is rejected as moving (default: one grid cell).
- `variables` *(ioos)*: preference order of ERDDAP variable names (default `[sea_water_temperature]`; `sea_surface_temperature` is tried as a fallback).
- `pad_deg` *(ioos)*: extra search padding around the AOI bbox.
- `path`, `columns`, `units`, `time_zone`, `qc_pass_values`, `default_station_id` *(csv)*: see above.

A region may override `sources`, `path`, `stations`, `exclude_stations` and `variables` — which data reaches *this* region is a fact about the world. It may **not** override `columns`, `units` or `qc_pass_values`: those decide what the channel *means*, and letting a region change them would make two AOIs' cubes silently non-comparable.

**In the cube** (`datacube.insitu`, default `true`):

| Channel | Dims | Meaning |
| --- | --- | --- |
| `insitu_sst` | (time,y,x) | the observation nearest the **reference time** (10:30 local solar), so it is contemporaneous with the met channels |
| `{eco,lst,modis,modisref}_insitu_sst` | (time,y,x) | the observation nearest **that sensor's overpass** — the satellite-vs-buoy matchup |
| `{eco,lst,modis,modisref}_insitu_dt_min` | (time,y,x) | signed minutes between the observation and the overpass, so matchup quality is auditable |
| `insitu_n` | (time,y,x) | stations contributing to the cell — N stations sharing a cell are **averaged**, from any mix of sources |
| `insitu_station` | (y,x) | `0` = none, `k` = station #k, indexing the `insitu_stations` cube attribute (which records each station's `source`) |

All are sparse — NaN everywhere except station cells — which costs almost nothing under the cube's Blosc/zstd encoding. Beyond `datacube.insitu_max_dt_min` (default 60) a matchup is **NaN rather than a stale value**: a buoy reading two hours off an overpass is not truth for that scene, and pretending otherwise is how a validation set quietly acquires a bias.

**The land-pixel problem.** A station near shore can land in a cell the cube calls *land* (coarse water mask, or a gauge on a pier), where it would be masked out of every downstream loss. Such a station is **snapped to the nearest water pixel**, and the snap distance is recorded; a snap beyond 500 m warns, because that means the station is probably not where we think it is.

### Bathymetry
Bathymetry is a static (time-invariant) covariate: one file per AOI describing water depth, used both as a model input and to build the land mask.

- **Where it comes from**: the NOAA NCEI CUDEM 1/9 arc-second (~3 m) seamless topobathy DEM (`cudem`, U.S. coasts only), read straight from its `/vsicurl` tiles, and the GMRT GridServer (`gmrt`, ~100 m, global). **No credentials required.** These STACK — where CUDEM has no coverage (e.g. SE Alaska) its channel is NaN and GMRT's carries; the cube does not merge them. The fine CUDEM pixels are aggregated to depth statistics within each grid cell; CUDEM is referenced to NAVD88 rather than MSL, so its 0 contour is not the mean waterline — set `datum_offset_m` (below) to reconcile the two.
- **Stacked, one channel per DEM source** (D10): list the DEMs you want and each is acquired independently into `depth_<dem>` / `elevation_<dem>` — no fallback. **What it measures** (all in metres): `elevation_<dem>` (mean, negative below sea level; carries that DEM's `datum_offset_m` attributes), `depth_<dem>` (mean water depth over the cell), and `depth_p25_<dem>` / `depth_p75_<dem>` (sub-grid depth variability). For GMRT there is no sub-grid, so `depth_p25 = depth_p75 = depth`. A source with no coverage here simply contributes no channel — stack another to fill the gap.

**Project-level options** (`products.bathymetry`):

- `sources`: the DEMs to STACK (default `[cudem, gmrt]`); e.g. `[cudem]` for a CONUS-only project, or `[gmrt]` outside it. There is no `source`/`fallback`/`dem_source` any more.
- `stats_subgrid_m`: fine sub-grid resolution for CUDEM depth statistics (default `10.0`).
- `min_cudem_cover`: minimum fraction of the AOI CUDEM must cover before it declines (default `0.5`).
- `pad_deg`: padding around the AOI bbox in degrees (default `0.02`).
- `layer`: GMRT layer (default `topo`).
- `resolution`: GMRT resolution (default `max`).
- `cudem_urllist`: URL of the CUDEM tile index (defaults to the NCEI 2014 ninth-arc-second list).
- `cudem_index_cache`: local path to cache the tile index (defaults under the output directory).

**Region-level options** (`regions[].sources.bathymetry`):

- `sources`: the DEMs to stack for the AOIs in this region, overriding the project list (e.g. `[gmrt]` for a region outside CUDEM coverage). Each DEM's DEM→MSL datum offset is resolved automatically *as it is acquired* (see [Water level](#water-level-reconstructed-downstream-from-raw-ingredients)).
- `datum_offset_m`: **optional** regional **fallback** for the DEM→MSL offset (the elevation of MSL in the DEM's vertical datum, in metres), used only when VDatum and the CO-OPS cross-check cannot resolve it (no coverage, a wide spread, or while a transient outage retries) — replacing the biased `0.0` with `datum_status = "regional_default"`. A resolved VDatum/CO-OPS value always wins over it. Leave it unset unless VDatum has gaps in this region.

### Met (forcing) and Met overpass

Met provides the meteorological forcing that drives nearshore ocean temperatures — air temperature, wind, shortwave radiation and cloud cover. These are complete driver channels, so there is no `valid`/mask layer. Two products: **`met`** is the daily FORCING (no sensor dependency); **`met_overpass`** is the same fields snapshotted at each thermal sensor's overpass instant (see [met in the cube](#met-in-the-cube-forcing-per-source-and-per-overpass-products)).

- **Where it comes from** — two STACKED sources (D10), one channel per source, no fallback. Each field is unit-harmonized to a single convention regardless of source.
    - **HRRR** — NOAA's 3 km High-Resolution Rapid Refresh surface analysis, fetched with [Herbie](https://github.com/blaylockbk/Herbie). CONUS (`hrrr`) below 50°N, Alaska (`hrrrak`) at/above it. Curvilinear grid → nearest-neighbour resampling (`pyresample`). **North America only.** No credentials.
    - **ERA5** — ECMWF's 0.25° hourly reanalysis, streamed from Google's public [ARCO-ERA5](https://github.com/google-research/arco-era5) Zarr on GCS. **Global**, so it covers any AOI HRRR cannot (e.g. outside North America). Regular lat/lon grid → bilinear reproject. No credentials (anonymous GCS).
- **What it measures** (harmonized units): `airtemp_<src>` (2 m air temperature, K or °C), `wind_u_<src>` / `wind_v_<src>` + derived `wind_speed_<src>` (m s⁻¹), `swrad_<src>` (downward shortwave, W m⁻²), `cloud_cover_<src>` (%). `met` writes a reference-time (and optional daily-mean) file per day, per source; `met_overpass` writes an instantaneous snapshot at each configured sensor's overpass time.

**Project-level options** (`products.met` — the forcing):

- `sources`: the met sources to STACK (default `[hrrr, era5]`); outside North America a region stacks only `[era5]`. There is no `source`/`fallback` any more.
- `variables`: which fields to acquire (default `airtemp`, `wind`, `swrad`, `cloud`).
- `reference_time`: the **time of day** the daily snapshot is taken at (default `"10:30"`, Landsat's overpass); `null` to skip.
- `reference_basis`: how `reference_time` is interpreted — `solar` (default; **local** solar time, per AOI longitude) or `utc`.
- `daily_mean_hours`: UTC hours averaged into the daily-mean field (default `[0, 6, 12, 18]`; `[]` to skip).
- `regrid_radius_m` (default `6000`), `pad_deg` (ERA5 subset pad, default `0.25`), `fxx` (HRRR forecast hour, default `0`), `product` (HRRR level, default `sfc`), `model` (force an HRRR domain), `era5_zarr` (ARCO store URI).

**Project-level options** (`products.met_overpass` — the overpass documentation):

- `sources`: the met sources to stack for the snapshots (default `[hrrr, era5]`).
- `combinations`: the `[sensor, source]` pairings to produce, e.g. `[[lst, hrrr], [eco, era5]]` → cube channels `lst_<var>_hrrr`, `eco_<var>_era5`. You opt into exactly the pairings you want. A combo naming an unloaded sensor or a bad source fails config validation.
- `variables` and the HRRR/ERA5 fetch knobs, as for `met`. Run `met_overpass` *after* the sensor stages (it reads their overpass times).

**Region-level options** (`met` / `met_overpass`): `sources`, `model`; `met_overpass` also takes `combinations`.

### Tides
Tide height is a forcing channel for nearshore temperature (mixing, exchange with cooler/warmer offshore water). Because tide is essentially spatially uniform over a small AOI, this is a single 1D time series per AOI rather than a gridded product — the datacube assembler broadcasts it across the AOI grid and samples it at the daily/overpass times.

- **Where it comes from** — two STACKED sources (D10), one channel per source, no fallback:
    - **`coops`** — NOAA CO-OPS (Tides & Currents). Finds the nearest CO-OPS water-level station to the AOI centroid, fetches that station's published **harmonic constituents** (one small, fast metadata request), and synthesizes the series **locally** with `pytides2` (nodal corrections included). Public, **no credentials required**, any date range — but CO-OPS gauges only exist in **U.S. waters**, so where the nearest gauge is farther than `max_distance_km` it contributes no channel here. Needs `requests` + `pytides2`.
    - **`eo_tides`** — a **global** ocean-tide model (**EOT20** by default) sampled at the AOI centroid via the [eo-tides](https://geoscienceaustralia.github.io/eo-tides/) package (pyTMD under the hood). Works **anywhere**. Needs `eo-tides` plus a downloaded tide-model directory (see the eo-tides "Setting up tide models" docs), pointed at via `model_directory` or the `EO_TIDES_TIDE_MODELS` environment variable.
- **What it measures**: `tide_<src>` — tide height in metres, relative to mean sea level, on a `time` dimension — and `tide_range_<src>`. CO-OPS files record the station id/name, distance, and constituent count; global-model files record the model name and sample point. Always written as NetCDF (a 1D series). The cube's `tide_overpass` channels (`<sensor>_tide_<src>`) are derived from these by interpolating to each sensor's overpass hour.

**Project-level options** (`products.tides`):

- `sources`: the tide sources to STACK (default `[coops, eo_tides]`); a region outside U.S. waters stacks only `[eo_tides]`. There is no `source`/`default_source`/`fallback` any more.
- `overpass_combinations`: the `[sensor, source]` pairings for the derived `tide_overpass` channels (`<sensor>_tide_<src>`), e.g. `[[eco, coops], [lst, eo_tides]]`. Default none.
- `max_distance_km`: CO-OPS declines (no channel) when the nearest gauge is farther than this (default `150`).
- `interval`: prediction step (default `h`, i.e. hourly).
- `stations`: per-AOI CO-OPS gauge overrides as `{aoi_name: station id}`, for when the automatically chosen nearest gauge is not the right one (default none).
- `warn_distance_km`: log a warning when the nearest gauge is farther than this from the AOI centroid (default `75`).
- `model`: eo-tides global model to sample (default `EOT20`; e.g. `FES2022`, `TPXO10-atlas`).
- `model_directory`: path to the downloaded tide-model files (default none → the `EO_TIDES_TIDE_MODELS` environment variable).

**Region-level options** (`regions[].sources.tides`): `sources` (which to stack here), `overpass_combinations`, `model`, `model_directory`, `stations`.

### Landcover
Landcover is a static (time-invariant) covariate: one file per AOI giving a land-cover class and a binary water mask. The assembler uses `water` as an authoritative **land override** — pixels the classifier calls non-water are forced to land even when bathymetry is below sea level, which fixes diked/reclaimed farmland (negative elevation but not actually water).

- **Where it comes from**: like Landsat, landcover is one product behind a `source` selector, and each `landcover_<source>` module honours the same output contract. The default source is **`esa`** — ESA WorldCover 10 m, read from the `esa-worldcover` collection on Microsoft Planetary Computer via a STAC search + windowed Cloud-Optimized GeoTIFF reads. Planetary Computer signs asset URLs anonymously, so **no credentials required**. A future **`gee`** source (the JRC Global Surface Water + Landsat-NDWI water mask from the legacy `00_make_water_mask.py`) will honour the same contract for a more robust mask where WorldCover is weak; it will need `auth.gee`.
- **What it measures**: `landcover` — the WorldCover class code (10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 70 snow/ice, 80 permanent water, 90 wetland, 95 mangrove, 100 moss; `0` = no data) — and `water` (1 where the class is in `water_classes`, else 0). Reprojected onto the AOI grid with nearest-neighbour (categorical) and written as a single static NetCDF per AOI (no time dimension).

**Project-level options** (`products.landcover`):

- `source`: which landcover backend to use (default `esa` = ESA WorldCover via Planetary Computer; `gee` reserved for the future JRC+NDWI mask).
- `year`: WorldCover epoch to fetch (default `2021` = v200; `2020` = v100).
- `water_classes`: class codes counted as water (default `[80]`; add `90`/`95` to include wetland/mangrove near tidal flats).
- `collection` / `stac_url`: STAC collection and API endpoint (default the `esa-worldcover` collection on the Planetary Computer catalog).

**Region-level options**: none.

## Extending the package (for developers)

Everything above is for *using* the package. If you want to **add a data product, add a new source to an existing product, or change how the datacube is assembled**, see the **[developer & contributor guide](docs/DEVELOPMENT.md)**. It covers the architecture, the product registry (`ProductSpec`), the `acquire()` contract every process module implements, and the one asymmetry to watch for: adding a thermal *sensor* is a single registry declaration, but a new *non-sensor* covariate also needs to be wired into the datacube assembler by hand. An architecture diagram lives at [`wireframes/main_wire_frame_claude.drawio`](wireframes/main_wire_frame_claude.drawio).

