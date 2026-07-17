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
| `modis` | swath→grid resampling (`pyresample`) |
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
- git+https://github.com/Jack-H-Buckner/coastal_sst_data.git@v0.0.1     # a git tag
- git+https://github.com/Jack-H-Buckner/coastal_sst_data.git@21c14a1    # a commit SHA
```

To bump a consumer to a newer version later: `pip install --upgrade --force-reinstall --no-deps "git+https://github.com/Jack-H-Buckner/coastal_sst_data.git@v0.1.0"`.

**Alternative — no per-project install.** If a project just needs to *run* the package occasionally, you can skip embedding it and instead `conda activate coastal_sst_data` (the standalone env from `environment.github.yml`) and work there. That's simplest for one-off use, but it doesn't let the package coexist with another project's own dependencies — for that, use the embedded pattern above.

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

There are five subcommands:

| Command | What it does | Network |
| --- | --- | --- |
| `validate` | Load and validate the config; print a summary of the products, each one's auth backend, and whether it is implemented yet | no |
| `grids` | Show the target grid (CRS, size, bounding box) computed for each AOI | no |
| `verify` | Connect to every credentialed service the selected products need and confirm the credentials work | yes |
| `run` | Run the pipeline: compute the shared grid once, then acquire each selected product in order | yes |
| `assemble` | Knit the aligned per-product outputs into one analysis-ready datacube (`.zarr`) per AOI | no |
| `provenance` | Print a built cube's provenance: the config that made it, each field's sources, access dates | no |

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
```

### `run`

Computes the shared grid once and dispatches each selected product to its acquisition module in a fixed order — static covariates and the MUR backbone first, then Landsat *before* MODIS (MODIS coincidence reads the Landsat outputs). Before downloading anything it runs the credential preflight automatically, so a bad or missing credential fails in seconds rather than mid-run. A product whose module isn't implemented yet is skipped with a warning; a product that errors is logged and the run continues to the next, with a per-product outcome summary at the end.

- `--aoi <name> …` — restrict to specific areas of interest (default: all).
- `--products <name> …` — run only these products, e.g. `--products mur landsat modis` (each must be selected in the config).
- `--dry-run` — search each source but download/write nothing (also skips the preflight).
- `--overwrite` — reprocess and overwrite outputs that already exist.
- `--no-verify` — skip the credential preflight (not recommended for a real run).

### `verify`

Runs the credential preflight on its own — handy before a long unattended run, or when setting up credentials for the first time. It actually *connects* to each backend (not just checks that a file exists), and exits non-zero listing **every** failing backend if any credential does not connect. Restrict to specific products' backends with `--products`. See [Authenticating to data services](#authenticating-to-data-services) for what each backend needs.

### `assemble`

The terminal stage: once the products are acquired, it knits their per-AOI aligned files into one analysis-ready **Zarr datacube per AOI** (`<output_dir>/datacube/<aoi>.zarr`), on a common **daily** time axis and the same shared grid every product was regridded onto. It reads only files already on disk, so it needs no network — run it after a `run`, or fold it into one with `run --assemble`.

Each cube keeps SST **separate per sensor** (`mur_sst`, `eco_sst`, `lst_sst`, `modis_sst`) so a downstream model can learn per-source offsets; each high-res sensor carries its own `valid` mask and overpass hour, and multiple scenes on a day collapse to the **clearest** one. The cube ships **raw ingredients** on a common grid and daily axis — MUR ships its observed values with honest NaN gaps (no fill), and the raw land-cover water layer ships as `landcover_water` rather than an opinionated derived land mask.

- `--aoi <name> …` — assemble only specific AOIs (default: all).
- `--overwrite` — rebuild cubes that already exist.
- `--dry-run` — report what would be assembled; write nothing.

Storage is tuned by the optional `datacube:` config block — `chunks` (the `(time, y, x)` chunking), `met_time`, and a `compression` block (Blosc codec, level, shuffle). Compression is **lossless**: values are kept as float32 / uint8 and only entropy-coded, so smooth and interpolated fields still shrink substantially (byte-shuffle on continuous channels, bit-shuffle on the integer masks) without discarding any precision. (Met-at-overpass is configured on the `met_overpass` product, not here — see below.)

> **Note (raw-output simplification).** The cube ships **raw ingredients** on a common grid and daily axis; masking, water-filling, station snapping, and multi-input derivations are downstream modelling determinations. The `fill_mur_water`, `fill_cmems_water`, and `water_level` keys were **removed** — MUR/CMEMS ship observed values with honest NaN gaps, there is no derived `landmask`, and water level is reconstructed downstream from the raw per-source `elevation_<dem>` + `depth_<dem>` + `tide_<src>` channels plus each DEM's `datum_offset_m` / `datum_status` attributes. An old config that still sets any of these three keys now **fails validation** rather than being silently ignored.

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

**Met at each sensor's own overpass is a separate product, `met_overpass`.** A weather model's value at 14:32 is *not* reconstructable from a daily sample, so it is a real acquisition (not a downstream derivation). It snapshots the thermal scenes and the cube emits `<sensor>_<var>_<src>` — `lst_airtemp_hrrr`, `eco_wind_speed_era5`, and so on — so an ECOSTRESS scene at 03:00 and a Landsat scene at 19:00 on the same day see *different* forcing rather than sharing one value. You name exactly which pairings to produce with `products.met_overpass.combinations` (a list of `[sensor, source]`), so the channel count is what you opt into, not a sensor×source cross-product. These follow the **exact scene the cube kept** — when a sensor flies twice in a day, only the clearest scene survives, and the forcing follows *that* scene's timestamp. Days with no scene stay NaN.

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

A downstream process references `elevation_<dem>` to MSL with **that DEM's** `datum_offset_m` attribute, takes the tide at the scene from `<sensor>_tide_<src>` (or interpolates `tide_<src>` to `<sensor>_hour` itself), and classifies each cell — exactly the computation the assembler used to hard-code, now made per-process.

**Datum.** Tides are relative to **MSL** (a *tidal* datum — the 19-year mean of observed water level at a gauge), but a DEM need not be: CUDEM is **NAVD88** (a *geodetic* datum). The gap between the two surfaces is **local** and far from negligible — MSL sits roughly **1.0–1.4 m above NAVD88 in the Pacific Northwest** (vs. ~0.1–0.3 m on the Gulf coast), which is comparable to the entire intertidal range, so ignoring it misclassifies much of a tidal flat. Each DEM source's offset is **resolved automatically as it is acquired** (NOAA VDatum for CUDEM/NAVD88, cross-checked against the nearest CO-OPS gauge; 0 for GMRT, which is already ~MSL) and ships as attributes on that source's `elevation_<dem>` channel. It can't be a config constant: the right value depends on **which DEM** it belongs to (CUDEM and GMRT on the same AOI need different offsets), which is exactly why it rides per-source rather than as one cube-wide number. Re-run `bathymetry --overwrite` to re-resolve it.

How each source resolves (during the bathymetry acquisition of *that* source):

| DEM source | Vertical datum | Offset |
| --- | --- | --- |
| **CUDEM** | NAVD88 | looked up from [NOAA VDatum](https://vdatum.noaa.gov/) (NAVD88 → LMSL), **cross-checked** against the nearest CO-OPS gauge's published datums |
| **GMRT** | ~sea level | `0.0` — no network call; GMRT is already sea-level referenced |

VDatum is sampled at **several points spread across the AOI's waterline band**, not at its centroid: coverage is patchy at point scale (the Padilla Bay centroid falls in a hole while points a few km away resolve fine), and out-of-coverage comes back as a `-999999` sentinel rather than an error. The median is taken; if the samples span more than 0.5 m the AOI straddles tidal zones and a single scalar is refused rather than averaged into something wrong everywhere. The result (method, sample count, spread, uncertainty, the gauge it was cross-checked against) is stamped onto that DEM source's `elevation_<dem>` cube channel — there is no separate stage or sidecar, so `assemble` stays entirely offline and the offset can never go stale against the DEM it belongs to. Re-run `bathymetry --overwrite` to re-resolve.

**Override.** If you know better than VDatum for a region, assert it under `regions[].sources.bathymetry.datum_offset_m` (the elevation of MSL in the DEM's datum, in metres). It wins — but it is still validated: a >0.5 m offset on a DEM that is GMRT is rejected as a NAVD88 number applied to an MSL DEM, the silent-metre error this exists to prevent. **If it cannot be resolved** (VDatum has no coverage and no NAVD88 gauge within 30 km), the offset falls back to `0.0` with a loud error and `datum_status = "unresolved_assumed_zero"`, so the bias is visible in the artifact and not only in a log line that scrolls away.

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

## Authenticating to data services

Most data products stream from **open archives that need no login** — Landsat and landcover from Microsoft Planetary Computer, bathymetry from NOAA CUDEM/GMRT, met from HRRR and Google's public ERA5, and tides from NOAA CO-OPS. Only a few products need credentials, and the package is built so that adding a new authenticated service later is a small, contained change.

**The golden rule: secrets never live in the config file (or the repo).** The config records only *how* to authenticate — a strategy name, or non-secret identifiers like a Google Cloud project. The actual usernames, passwords, and keys stay in the standard locations each service already uses (`~/.netrc`, a key file, environment variables), outside version control.

Auth requirements are declared **per product — and per source** where a product has several. A product served from an anonymous source needs nothing: Landsat via its default `pc` (Planetary Computer) source requires no credentials, whereas Landsat via a future `gee` source would require Google Earth Engine auth.

**Which products need what:**

| Backend | Required by | Credentials |
| --- | --- | --- |
| `earthdata` | ECOSTRESS, MODIS, MUR | free NASA Earthdata account |
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
```

The config is validated on load: if you select a product that needs a backend and its block is missing, loading fails immediately with a clear message (e.g. *"product 'ecostress' requires `auth.earthdata`"*). Products using anonymous sources need no `auth:` block at all.

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

**Project-level options** (`products.ecostress`):

- `short_name`: Earthdata collection short name (default `ECO_L2T_LSTE`).
- `version`: collection version (default `002`).
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
MODIS provides a coarser, well-calibrated SST reference that downstream calibration can match Landsat/ECOSTRESS against, optionally restricted to overpasses coincident with a Landsat scene.

- **Where it comes from**: the `MODIS_T-JPL-L2P-v2019.0` GHRSST MODIS Terra L2P skin-SST product from NASA OB.DAAC via `earthaccess`. It is a swath (2D curvilinear) product, so it is regridded with nearest-neighbour resampling (`pyresample`) to preserve the observed values rather than smoothing them.
- **What it measures**: skin sea-surface temperature (`sst`, K or °C), quality-filtered on the GHRSST `quality_level` band, plus a derived `valid` layer. Optionally emits `footprint_id`, the MODIS swath pixel index each grid cell was drawn from, for exact footprint-median matchups.

**Project-level options** (`products.modis`):

- `short_name`: Earthdata short name (default `MODIS_T-JPL-L2P-v2019.0`).
- `variable`: SST variable to read (default `sea_surface_temperature`).
- `quality_min`: minimum GHRSST quality level to keep, 0–5 (default `4`; 5 is best).
- `regrid_radius_m`: nearest-neighbour search radius in metres (default `1500`).
- `access`: fetch backend, `download` (default) or `harmony` (not yet implemented).
- `match_landsat`: only load granules within `max_time_diff_minutes` of an already-acquired Landsat scene (default `true`; requires Landsat to have run first).
- `max_time_diff_minutes`: coincidence window for `match_landsat` (default `360`, i.e. ±6 h).
- `daytime_only`: drop night granules (default `true`).
- `footprint_id`: emit the swath-pixel-index layer (default `true`).

**Region-level options**: none.

### MUR
MUR is the always-present, gap-free SST backbone the high-resolution products add detail onto.

- **Where it comes from**: the `MUR-JPL-L4-GLOB-v4.1` GHRSST MUR L4 analysis (daily, ~1 km, global) from PO.DAAC via `earthaccess`. For each day the global granule is opened lazily and subset to the AOI window (HDF5 range reads — the global file is never fully downloaded), then bilinearly upsampled onto the AOI grid.
- **What it measures**: `analysed_sst`, a gap-free (cloud-free) foundation SST analysis (`sst`, K or °C). Because it is an L4 analysis there is no cloud mask; `valid` is simply finite SST (i.e. water).

**Project-level options** (`products.mur`):

- `short_name`: Earthdata short name (default `MUR-JPL-L4-GLOB-v4.1`).
- `variable`: variable to read (default `analysed_sst`).
- `pad_deg`: degrees of padding added around the AOI lat/lon window before subsetting (default `0.05`).

**Region-level options**: none.

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

### In-situ (IOOS)
In-situ observations are the cube's **only ground truth**. Every other channel is modelled (met, CMEMS, tides) or remotely sensed (ECOSTRESS, Landsat, MODIS, MUR); this is what a thermometer in the water actually read. The assembler writes each station's value into **the grid cell the station sits in**, and — the point of the exercise — **at the instant each satellite flew**, so a scene can be validated against a buoy pixel-for-pixel and minute-for-minute.

- **Where it comes from**: the [IOOS Sensors ERDDAP](https://erddap.sensors.ioos.us/erddap) — one server aggregating **NDBC, NOAA CO-OPS, CDIP and the IOOS regional associations**, so most of North America is a single query. Stations are auto-discovered inside each AOI's bounding box. **No credentials.** It is a `source`-selector product (`source: ioos`), so another network is a new module behind the same contract.
- **What it measures**: water temperature (`sea_water_temperature`, falling back per station to `sea_surface_temperature` — providers do not agree on the name), quality-flagged with QARTOD. The native sampling interval is kept (6 min for CO-OPS gauges), because matching an overpass needs the sub-hourly series.

**Quality control.** QARTOD flags are `1` pass, `2` not-evaluated, `3` suspect, `4` fail, `9` missing. The default keeps **`[1, 2]`**: flag 2 is what stations that don't run QARTOD emit, and demanding flag 1 would discard much of the network.

**Stations that report nothing are dropped — loudly.** A station can *advertise* a temperature variable and never report it (NDBC 46120 exposes both temperature names and returns all-NaN for a whole month; it is a wave buoy with no thermometer). Those are logged by name, because an empty in-situ channel that reads as "no buoys here" is the one failure this product cannot afford.

**Project-level options** (`products.insitu`):

- `source`: the network (default `ioos`).
- `variables`: preference order of ERDDAP variable names (default `[sea_water_temperature]`; `sea_surface_temperature` is tried as a fallback).
- `qc_flags`: QARTOD flags to keep (default `[1, 2]`).
- `max_sensor_depth_m`: ignore sensors deeper than this on profiling moorings (default `5`).
- `stations` / `exclude_stations`: an explicit allow-list (else auto-discovery) and a deny-list for a known-bad mooring.

**In the cube** (`datacube.insitu`, default `true`):

| Channel | Dims | Meaning |
| --- | --- | --- |
| `insitu_sst` | (time,y,x) | the observation nearest the **reference time** (10:30 local solar), so it is contemporaneous with the met channels |
| `{eco,lst,modis}_insitu_sst` | (time,y,x) | the observation nearest **that sensor's overpass** — the satellite-vs-buoy matchup |
| `{eco,lst,modis}_insitu_dt_min` | (time,y,x) | signed minutes between the observation and the overpass, so matchup quality is auditable |
| `insitu_n` | (time,y,x) | stations contributing to the cell (two buoys in one cell are averaged) |
| `insitu_station` | (y,x) | `0` = none, `k` = station #k, indexing the `insitu_stations` cube attribute |

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
- `datum_offset_m`: **optional** override of the automatically-resolved DEM→MSL offset (the elevation of MSL in the DEM's vertical datum, in metres). Leave it unset unless you know better than VDatum for this region.

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

