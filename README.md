# Coastal SST data loader

This library is desinged to obtain data for coastal and nearshore ocean ecosystem and combine them in to a gridded data format for down stream modeling tasks. The primary goal of this code base is to load thermal remote sensing images and covarites that drive nearshore ocean temperatures to feed into high reolution sea surface temperature models. 

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
| `datum` | Resolve each AOI's DEM→MSL vertical-datum offset (runs inside `run`; standalone for backfill) | yes (small) |
| `assemble` | Knit the aligned per-product outputs into one analysis-ready datacube (`.zarr`) per AOI | no |

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

Each cube keeps SST **separate per sensor** (`mur_sst`, `eco_sst`, `lst_sst`, `modis_sst`) so a downstream model can learn per-source offsets; each high-res sensor carries its own `valid` mask and overpass hour, and multiple scenes on a day collapse to the **clearest** one. The water/land mask is taken from **land-cover** (authoritative where known), and the gap-free MUR backbone is nearest-neighbour filled over land-cover water so it has no holes in narrow estuaries.

- `--aoi <name> …` — assemble only specific AOIs (default: all).
- `--overwrite` — rebuild cubes that already exist.
- `--dry-run` — report what would be assembled; write nothing.

Storage is tuned by the optional `datacube:` config block — `chunks` (the `(time, y, x)` chunking), `fill_mur_water`, `water_level`, `met_time`, `overpass_met`, and a `compression` block (Blosc codec, level, shuffle). Compression is **lossless**: values are kept as float32 / uint8 and only entropy-coded, so smooth and interpolated fields still shrink substantially (byte-shuffle on continuous channels, bit-shuffle on the integer masks) without discarding any precision.

#### Met in the cube: reference time, and per-overpass

**A day gets one met value per channel, taken at a fixed *time of day* — not a daily mean.** A mean over `[0, 6, 12, 18]` UTC averages pre-dawn and mid-afternoon forcing together, which is the wrong thing to hand a model of a sensor that flew at one instant. The cube's `airtemp` / `wind_u` / `wind_v` / `wind_speed` / `swrad` / `cloud_cover` therefore come from the **reference-time snapshot**: by default **10:30 local solar time**, Landsat's overpass.

The basis is *solar*, not UTC, because a fixed UTC hour is a different time of day in every AOI — mid-morning in Oregon, the middle of the night in Maine — so cross-AOI forcing would not be like-for-like. Each AOI's reference instant is derived from its own longitude (`UTC = local − lon/15`, rounded to the hour, rolling the date where it crosses midnight). Change it with `products.met.reference_time` / `reference_basis`, or set `datacube.met_time: daily_mean` to get the old averaging behavior back. If no reference files exist (an older MET tree), the assembler falls back to the daily mean rather than emitting an empty channel, and records which it used in the cube's `met_time` attribute.

**Each sensor additionally carries the forcing at its own overpass.** Where `products.met.overpass_from` made met snapshot the thermal scenes, the cube emits `<sensor>_<var>` — `eco_airtemp`, `lst_wind_speed`, `modis_swrad`, and so on — so an ECOSTRESS scene at 03:00 and a Landsat scene at 19:00 on the same day see *different* air temperature and wind, rather than sharing one value. Pick the variables with `datacube.overpass_met` (default `[airtemp, wind_speed, swrad, cloud_cover]`; `[]` disables). These are matched to the **exact scene the cube kept** — when a sensor flies twice in a day, only the clearest scene survives, and the forcing follows *that* scene's timestamp, not merely its date. Days with no scene from a sensor stay NaN rather than carrying a stale value.

#### Water level (derived at assembly)

The assembler also **derives** a water level from two products it already has: the static bathymetry DEM and the 1D tide series. Because a tidal flat's state depends on *when* the sensor flew, the water level is evaluated **at each sensor's overpass time** — the hourly tide series is linearly interpolated to the scene's hour — giving, for each of `eco` / `lst` / `modis`:

| Channel | Dims | Meaning |
| --- | --- | --- |
| `<sensor>_water_elev` | (time, y, x) | ground elevation relative to the tide-adjusted waterline, in metres: **0 at the waterline**, positive above it (exposed, height above water), negative below it (submerged — the magnitude is the water depth) |
| `<sensor>_water_class` | (time, y, x) | `0` = submerged, `1` = exposed, `255` = unknown |
| `<sensor>_tide` | (time,) | the interpolated tide height at that overpass (m, rel. MSL) |

On a day with no scene from that sensor there is no overpass time, so the fields are NaN / `255` rather than carrying a stale value; the same holds where the DEM has no coverage. This is a pure function of `bathymetry` + `tides`, so it needs both selected in the config; it is a derived stage rather than an acquisition product and is turned off with `datacube.water_level: false`.

Two things to know about the classification:

- **Datum.** Tides are relative to **MSL** (a *tidal* datum — the 19-year mean of observed water level at a gauge), but a DEM need not be: CUDEM is **NAVD88** (a *geodetic* datum). The gap between the two surfaces is **local** and far from negligible — MSL sits roughly **1.0–1.4 m above NAVD88 in the Pacific Northwest** (vs. ~0.1–0.3 m on the Gulf coast), which is comparable to the entire intertidal range, so ignoring it misclassifies much of a tidal flat. This offset is **resolved automatically** by the [`datum` stage](#datum) — you do not configure it.
- **It is purely geometric** — "is this cell's ground below the waterline?" — and deliberately does **not** consult land-cover. Land-cover routinely classifies tidal flats as non-water, so letting it override here would erase exactly the intertidal signal these fields exist to capture. The cost is that diked or reclaimed ground lying below the waterline reads as submerged; intersect with the cube's `landmask` / `landcover_water` if you need a hydrologically-connected water mask.

### `datum`

Resolves each AOI's **DEM→MSL vertical-datum offset** — the number the [water-level fields](#water-level-derived-at-assembly) need to put the DEM and the tide on one surface. It runs automatically inside `run` (after bathymetry, before the assembler) whenever bathymetry is selected and `datacube.water_level` is on; the subcommand exists to **backfill an existing data tree** without re-downloading a single DEM tile, since it reads the bathymetry output off disk rather than re-fetching it.

Why it can't be a config constant: the right offset depends on **which DEM actually ran**, and bathymetry silently falls back CUDEM→GMRT where CUDEM has no coverage. So two AOIs in one region can legitimately need different offsets, and a project-wide number would be wrong for one of them.

How each case resolves:

| DEM that ran | Vertical datum | Offset |
| --- | --- | --- |
| **CUDEM** | NAVD88 | looked up from [NOAA VDatum](https://vdatum.noaa.gov/) (NAVD88 → LMSL), **cross-checked** against the nearest CO-OPS gauge's published datums |
| **GMRT** | ~sea level | `0.0` — no network call; GMRT is already sea-level referenced |

VDatum is sampled at **several points spread across the AOI's waterline band**, not at its centroid: coverage is patchy at point scale (the Padilla Bay centroid falls in a hole while points a few km away resolve fine), and out-of-coverage comes back as a `-999999` sentinel rather than an error. The median is taken; if the samples span more than 0.5 m the AOI straddles tidal zones and a single scalar is refused rather than averaged into something wrong everywhere.

The result is written to `<output_dir>/DATUM/aligned/<aoi>/<aoi>_datum.json` with its full provenance (method, sample count, spread, uncertainty, the gauge it was cross-checked against, and a fingerprint of the DEM it was resolved for). The assembler reads that sidecar — so `assemble` stays entirely offline — and **refuses a stale one** whose fingerprint no longer matches the DEM on disk.

- `--aoi <name> …` — resolve only specific AOIs (default: all).
- `--overwrite` — re-resolve offsets already on disk (otherwise it's a no-op, zero requests).
- `--dry-run` — report what would be resolved; write nothing, call nothing.

**Override.** If you know better than VDatum for a region, assert it under `regions[].sources.bathymetry.datum_offset_m` (the elevation of MSL in the DEM's datum, in metres). It wins — but it is still validated: a >0.5 m offset on a DEM that turned out to be GMRT is rejected as a NAVD88 number applied to an MSL DEM, which is exactly the silent-metre error this stage exists to prevent.

**If it cannot be resolved** (VDatum has no coverage and no NAVD88 gauge is within 30 km), the offset falls back to `0.0` with a loud error, and the cube is stamped `datum_status = "unresolved_assumed_zero"` so the bias is visible in the artifact and not only in a log line that scrolls away.

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

- **Where it comes from**: the Copernicus Marine global physics models (1/12°, daily means, ~50 depth levels), streamed with the `copernicusmarine` toolbox. `open_dataset` subsets the AOI window **server-side and lazily**, so the global model is never downloaded. Two products form a **source chain**, like met's HRRR→ERA5:
    - **`my`** (default) — `cmems_mod_glo_phy_my_0.083deg_P1D-m`, the **GLORYS12 reanalysis** (hindcast). Best quality, but it stops a year or two behind the present.
    - **`anfc`** — `cmems_mod_glo_phy_anfc_0.083deg_P1D-m`, the **analysis/forecast** product, which reaches the present and backfills whatever days the reanalysis cannot cover. Each output file records which product produced it in its `source` / `cmems_source` attrs, so a reanalysis day and a forecast day are never silently conflated.
- **What it measures**: whichever `variables` you select (default `thetao`, sea-water potential temperature; also `so` salinity, `uo`/`vo` currents, and the 2D `zos` / `mlotst`), emitted **once per requested depth**.

**Depths.** The model has ~50 *fixed* levels (0.494, 1.541, 2.646, 5.078 m …), so a requested depth is snapped to the **nearest level** — never interpolated. Every value is therefore one the model actually computed. The channel is named for what you asked for; the level actually used is recorded in the variable's `model_depth_m` attr:

```yaml
products:
  cmems:
    depths: [0, 10, 30]     # metres
```
gives `thetao_0m` (level 0.494 m), `thetao_10m` (level **9.573** m), `thetao_30m` (level **29.445** m). A requested depth more than 5 m from any level logs a warning.

**Project-level options** (`products.cmems`):

- `source`: primary product, `my` (default) or `anfc`.
- `fallback`: product used for days the primary does not cover, `anfc` (default) or `none` to disable the chain.
- `dataset_id`: an explicit CMEMS dataset id, which **overrides the chain entirely** — the escape hatch for any other Copernicus physics product.
- `variables`: which fields to acquire (default `[thetao]`).
- `depths`: depths in metres, each snapped to the nearest model level (default `[0]`).
- `pad_deg`: padding around the AOI window (default `0.15`, ≥ one 1/12° cell).

**Region-level options**: none.

**In the cube** the channels arrive prefixed — `cmems_thetao_0m`, `cmems_thetao_30m` — and, like MUR, are **nearest-neighbour filled over land-cover water** (`datacube.fill_cmems_water`, default `true`). This matters more here than for MUR: at ~9 km the model's land mask can swallow an entire estuary, and the nearest resolved water column is the honest value for a cell the model never resolved. Real land is left NaN.

**Credentials**: a free [Copernicus Marine](https://data.marine.copernicus.eu) account, declared as `auth.copernicus`. The toolbox reads `~/.netrc` natively, so the secret lives there like every other one:

```
machine auth.marine.copernicus.eu login <username> password <password>
```

### Bathymetry
Bathymetry is a static (time-invariant) covariate: one file per AOI describing water depth, used both as a model input and to build the land mask.

- **Where it comes from**: the NOAA NCEI CUDEM 1/9 arc-second (~3 m) seamless topobathy DEM, read straight from its `/vsicurl` tiles, with the GMRT GridServer (~100 m, global) as a fallback where CUDEM has no coverage (e.g. SE Alaska). **No credentials required.** The fine CUDEM pixels are aggregated to depth statistics within each grid cell; CUDEM is referenced to NAVD88 rather than MSL, so its 0 contour is not the mean waterline — set `datum_offset_m` (below) to reconcile the two.
- **What it measures** (all in metres): `elevation` (mean, negative below sea level — drives the land mask), `depth` (mean water depth over the cell), and `depth_p25` / `depth_p75` (sub-grid depth variability). For GMRT there is no sub-grid, so `depth_p25 = depth_p75 = depth`.

**Project-level options** (`products.bathymetry`):

- `default_source`: DEM source used when a region does not override it (default `gmrt`).
- `fallback`: source tried when the chosen source lacks coverage (default `gmrt`).
*(There is no project-level datum option: the DEM→MSL offset is resolved per AOI by the [`datum` stage](#datum), because it follows the DEM that actually ran.)*
- `stats_subgrid_m`: fine sub-grid resolution for CUDEM depth statistics (default `10.0`).
- `min_cudem_cover`: minimum fraction of the AOI CUDEM must cover before falling back (default `0.5`).
- `pad_deg`: padding around the AOI bbox in degrees (default `0.02`).
- `layer`: GMRT layer (default `topo`).
- `resolution`: GMRT resolution (default `max`).
- `cudem_urllist`: URL of the CUDEM tile index (defaults to the NCEI 2014 ninth-arc-second list).
- `cudem_index_cache`: local path to cache the tile index (defaults under the output directory).

**Region-level options** (`regions[].sources.bathymetry`):

- `dem_source`: the DEM source (`cudem` or `gmrt`) to use for the AOIs in this region, overriding the project-level `default_source`. This is the option to set when different parts of the study area have different DEM coverage.
- `datum_offset_m`: **optional** override of the automatically-resolved DEM→MSL offset (the elevation of MSL in the DEM's vertical datum, in metres). Leave it unset unless you know better than VDatum for this region; see [`datum`](#datum).

### Met
Met provides the gap-free meteorological forcing that drives nearshore ocean temperatures — air temperature, wind, shortwave radiation and cloud cover — used both as model inputs and for the skin-to-bulk SST correction. Unlike the SST products these are complete driver channels, so there is no `valid`/mask layer.

- **Where it comes from**: two sources are tried in order (the "source chain"), so the output is gap-free everywhere. Each field is unit-harmonized to a single convention regardless of which source supplied it, and every output file records its provenance in the `source` attribute.
    - **HRRR** — NOAA's 3 km High-Resolution Rapid Refresh surface analysis, fetched with [Herbie](https://github.com/blaylockbk/Herbie). The CONUS domain (`hrrr`) is used below 50°N and the Alaska domain (`hrrrak`) at/above it. HRRR is a curvilinear grid, so it is regridded with nearest-neighbour resampling (`pyresample`). **No credentials required.**
    - **ERA5** — ECMWF's 0.25° hourly reanalysis, streamed from Google's public [ARCO-ERA5](https://github.com/google-research/arco-era5) Zarr store on GCS. It is global, so it backfills any AOI, date, or cycle HRRR cannot cover (e.g. AOIs outside North America). ERA5 is a regular lat/lon grid, so it is bilinearly reprojected. **No credentials required** (anonymous GCS access).
- **What it measures** (harmonized units): `airtemp` (2 m air temperature, K or °C), `wind_u` / `wind_v` (10 m wind components) and the derived `wind_speed` (m s⁻¹), `swrad` (downward shortwave, W m⁻²), and `cloud_cover` (%). For each AOI it writes a daily-mean file per day plus an instantaneous snapshot at each thermal-scene overpass time (discovered from the `overpass_from` products' aligned outputs), so forcing can be matched to each SST scene.

**Project-level options** (`products.met`):

- `source`: primary source, `auto` (default), `hrrr`, or `era5`. `auto` is equivalent to `hrrr` as the primary with the fallback appended.
- `fallback`: source used where the primary misses, `era5` (default) or `none` to disable. With the defaults (`source: auto`, `fallback: era5`) the chain is `hrrr → era5`.
- `variables`: which fields to acquire (default `airtemp`, `wind`, `swrad`, `cloud`).
- `reference_time`: the **time of day** the daily snapshot is taken at (default `"10:30"`, Landsat's overpass); `null` to skip it. This is the cube's default met channel — see [met in the cube](#met-in-the-cube-reference-time-and-per-overpass).
- `reference_basis`: how `reference_time` is interpreted — `solar` (default; **local** solar time, converted per AOI from its longitude) or `utc` (taken literally).
- `daily_mean_hours`: UTC hours averaged into the daily-mean field (default `[0, 6, 12, 18]`; set `[]` to skip it and save the four fetches per day).
- `overpass_from`: SST products whose aligned scenes set the instantaneous snapshot times, e.g. `[ecostress, landsat]` (default none — daily means only). Run met *after* these products so their outputs exist.
- `regrid_radius_m`: nearest-neighbour search radius for HRRR regridding in metres (default `6000`).
- `pad_deg`: degrees of padding around the AOI window for the ERA5 subset (default `0.25`, i.e. ≥ one ERA5 cell).
- `fxx`: HRRR forecast hour (default `0` = analysis; use `1` if `DSWRF` reads 0 at the analysis time).
- `product`: HRRR product/level (default `sfc`, the 2D surface fields).
- `model`: force an HRRR domain, `auto` (default), `hrrr`, or `hrrrak`, instead of choosing it by latitude.
- `era5_zarr`: override the ARCO-ERA5 store URI (defaults to the public GCS Zarr).

**Region-level options**: none.

### Tides
Tide height is a forcing channel for nearshore temperature (mixing, exchange with cooler/warmer offshore water). Because tide is essentially spatially uniform over a small AOI, this is a single 1D time series per AOI rather than a gridded product — the datacube assembler broadcasts it across the AOI grid and samples it at the daily/overpass times.

- **Where it comes from**: tides is a `source`-selector product with a `SOURCES` registry and fallback (like bathymetry), so an AOI is served from whichever source has coverage:
    - **`coops`** (default) — NOAA CO-OPS (Tides & Currents). Finds the nearest CO-OPS water-level station to the AOI centroid, fetches that station's published **harmonic constituents** (one small, fast metadata request), and synthesizes the series **locally** with `pytides2` (nodal corrections included). Public, **no credentials required**, works for any date range — but CO-OPS gauges only exist in **U.S. waters**. Needs `requests` + `pytides2`.
    - **`eo_tides`** — a **global** ocean-tide model (**EOT20** by default) sampled at the AOI centroid via the [eo-tides](https://geoscienceaustralia.github.io/eo-tides/) package (pyTMD under the hood). Works **anywhere**, so it is the natural **backup** for AOIs outside the U.S. Needs `eo-tides` plus a downloaded tide-model directory (see the eo-tides "Setting up tide models" docs), pointed at via `model_directory` or the `EO_TIDES_TIDE_MODELS` environment variable.
- **How the backup kicks in**: with the defaults (`default_source: coops`, `fallback: eo_tides`) an AOI falls through to the global model when the nearest CO-OPS gauge is farther than `fallback_distance_km`, **or** when the CO-OPS fetch fails. Set `fallback: none` to disable the backup, or select a source explicitly per region (below).
- **What it measures**: `tide` — tide height in metres, relative to mean sea level, on a `time` dimension. CO-OPS files record the station id/name, its distance from the AOI centroid, and the number of constituents; global-model files record the model name and sample point. The `source` attribute records which source actually produced the file. Always written as NetCDF (a 1D series, so `geotiff` does not apply).

**Project-level options** (`products.tides`):

- `default_source`: primary source, `coops` (default) or `eo_tides`.
- `fallback`: source used when the primary has no coverage/fails, `eo_tides` (default) or `none` to disable.
- `fallback_distance_km`: fall back to the global model when the nearest CO-OPS gauge is farther than this (default `150`).
- `interval`: prediction step (default `h`, i.e. hourly).
- `stations`: per-AOI CO-OPS gauge overrides as `{aoi_name: station id}`, for when the automatically chosen nearest gauge is not the right one (default none). An override is never distance-pre-empted to the backup.
- `warn_distance_km`: log a warning when the nearest gauge is farther than this from the AOI centroid (default `75`).
- `model`: eo-tides global model to sample (default `EOT20`; e.g. `FES2022`, `TPXO10-atlas`).
- `model_directory`: path to the downloaded tide-model files (default none → the `EO_TIDES_TIDE_MODELS` environment variable).

**Region-level options** (`regions[].sources.tides`):

- `source`: the tide source (`coops` or `eo_tides`) for the AOIs in this region, overriding the project-level `default_source`. Set `eo_tides` for a region outside CO-OPS coverage (e.g. an international study area) while U.S. regions keep `coops`.

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

