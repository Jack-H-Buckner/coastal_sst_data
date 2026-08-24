from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from coastal_sst_data.config import AreaOfInterest, GridSpec
from coastal_sst_data import auth, grid


@pytest.fixture(autouse=True)
def _clean_auth_sessions():
    """Isolate `auth`'s process-global credential state between tests.

    `auth.login` records a timestamp and `auth.refresh` counts against a budget, both in
    module dicts that outlive a test. Without this, a test that logs in leaves a credential
    that a later test sees as already-fresh -- so `ensure_fresh` silently does nothing and the
    test passes for the wrong reason. `configure()` likewise resets the policy to its defaults.
    """
    auth.reset()
    auth.configure(None)
    yield
    auth.reset()


@pytest.fixture
def base_project():
    """Smallest valid project dict; tests copy + mutate it for edge cases.

    Returned as a plain dict (not a Project) so tests can mutate it to build
    invalid variants -- e.g. drop a product, blank a section, duplicate a name --
    and check that validation fires. Function-scoped, so each test gets a fresh
    copy and in-place mutation is safe.
    """
    return {
        "name": "test",
        "output_dir": "path/to/data",
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-30"},
        "grid": {},
        "auth": {"earthdata": {"auth_strategy": "netrc"}},   # mur needs earthdata
        "products": {"bathymetry": None, 
                     "mur": {"variable": "analysed_sst"}},
        "regions": [{
            "name": "r1",
            "sources": {"bathymetry": {"sources": ["cudem"]}},
            "areas": [{"name": "a1", "center_lat": 45.0, "center_lon": -123.0,
                       "buffer_ns_km": 25, "buffer_ew_km": 15}],
        }],
    }


# ---------------------------------------------------------------------------
# Shared geospatial test objects.
# ---------------------------------------------------------------------------
@pytest.fixture
def aoi_grid():
    """A small AoI's shared grid (100 m, auto UTM) to build/reproject onto.

    Shared by every process test so products are checked against one grid.
    """
    area = AreaOfInterest(name="test_aoi", center_lat=45.52, center_lon=-123.925,
                          buffer_ns_km=8.0, buffer_ew_km=8.0)
    return grid.compute_aoi_grid(area, GridSpec())


# ---------------------------------------------------------------------------
# Synthetic Landsat C2 L2 scene, shared by ALL landsat_<source> modules
# (landsat_pc today; landsat_aws / future STAC+COG sources tomorrow). A minimal
# STAC-item stand-in points at local single-band COGs of raw DN, so every source
# module can be fed IDENTICAL input and asserted to produce the same output.
# ---------------------------------------------------------------------------
class FakeAsset:
    """Stand-in for a pystac Asset: only .href is read by the source modules."""
    def __init__(self, href):
        self.href = href


class FakeStacItem:
    """Minimal stand-in for a pystac Item (what scene_to_dataset / run touch).

    `geometry` defaults to ABSENT -- no attribute at all, as an item served without one -- so
    every test that does not care about the AoI-footprint filter keeps exercising the keep
    path, which is also the convention the filter itself follows (thin metadata means keep).
    """
    def __init__(self, assets, properties, dt, id="FAKE_LC09_L2SP", geometry=None):
        self.assets = {k: FakeAsset(h) for k, h in assets.items()}
        self.properties = properties
        self.datetime = dt
        self.id = id
        if geometry is not None:
            self.geometry = geometry


def write_landsat_cogs(dir_path, g, *, native_res=30.0, pad_m=1500.0,
                       thermal_key="lwir11", fill_rows=None):
    """Write synthetic Landsat C2 L2 single-band COGs (raw DN) for one scene.

    DN values are chosen so the DERIVED layers are known after a source module
    reprojects them onto grid `g`:
        lwir11 (thermal) DN 41252  -> 290 K everywhere
        green DN 9091 (0.05) / nir08 DN 8000 (0.02) -> NDWI ~0.43 -> water everywhere
        qa_pixel: cloud bit (1<<3) over the WEST half, 0 over the EAST half
        cdist DN 200 -> 2 km, so the 1 km cloud buffer never fires

    The reflectances are DARK ON PURPOSE, and were not always: they used to be green 0.20 /
    NIR 0.10, which has the right NDWI but is four times brighter than real water. Measured
    over 9.0M pixels across three AoIs and three missions, QA-confirmed water sits at green
    P50 0.017 and P99 0.075 -- 0.20 is cloud-top territory, and once `masking.brightness_max`
    existed the fixture's "water everywhere" stopped being water at all. 0.05/0.02 is squarely
    inside the real distribution while keeping NDWI comfortably positive.
    => expected valid mask: 1 in the EAST half, 0 in the WEST half.
    Source is native ~30 m in the SAME CRS as the target grid. Returns
    {asset_key: path-str}, ready to hang off a FakeStacItem.

    `thermal_key` names the thermal asset: `lwir11` is what OLI-TIRS (Landsat 8/9) serves and
    `lwir` is what TM/ETM+ (Landsat 4/5/7) serves. The SAME array either way -- that is the
    point of the parameter. C2 Level-2 uses one single-channel algorithm and one scale/offset
    for every mission, so a TM scene differs from an OLI-TIRS scene in the asset's NAME and
    nothing else, and a test that feeds both through one module proves exactly that.

    `fill_rows` (a row `slice`) marks a band of the scene as QA fill -- `qa_pixel` bit 0 set and
    thermal/SR at their nodata -- which is how a Landsat 7 SLC-off gap arrives. Verified against
    a real scene: every gap pixel reads `QA_PIXEL == 1` exactly, with no cloud bit set.
    """
    minx, miny, maxx, maxy = g.geom_proj.bounds
    minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
    W = int(round((maxx - minx) / native_res))
    H = int(round((maxy - miny) / native_res))
    transform = from_origin(minx, maxy, native_res, native_res)

    thermal = np.full((H, W), 41252, dtype="uint16")   # -> 290.0 K
    green = np.full((H, W), 9091, dtype="uint16")       # reflectance ~0.05
    nir = np.full((H, W), 8000, dtype="uint16")         # reflectance ~0.02 -> dark water
    qa = np.zeros((H, W), dtype="uint16")
    qa[:, : W // 2] = 1 << 3                             # west half: cloud bit
    cdist = np.full((H, W), 200, dtype="uint16")        # 2 km -> no buffer effect

    if fill_rows is not None:
        # An SLC-off gap: bit 0 alone, and nodata in every band that carries a value. Written
        # LAST so it overwrites the cloud bit -- a real fill pixel carries no other flag.
        qa[fill_rows, :] = 1
        thermal[fill_rows, :] = 0
        green[fill_rows, :] = 0
        nir[fill_rows, :] = 0

    layers = [(thermal_key, thermal, 0), ("green", green, 0), ("nir08", nir, 0),
              ("qa_pixel", qa, 1), ("cdist", cdist, 0)]
    paths = {}
    for key, arr, nodata in layers:
        p = dir_path / f"{key}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1,
                           dtype="uint16", crs=g.target_crs, transform=transform,
                           nodata=nodata) as dst:
            dst.write(arr, 1)
        paths[key] = str(p)
    return paths


@pytest.fixture
def landsat_scene(tmp_path, aoi_grid):
    """A FakeStacItem whose assets point at synthetic Landsat COGs (offline).

    Feed this to any landsat_<source>.scene_to_dataset. Known layout -> output:
    sst ~290 K, water=1 everywhere, cloud=1 in the WEST half, valid=1 in the EAST.
    """
    paths = write_landsat_cogs(tmp_path, aoi_grid)
    dt = datetime(2023, 8, 15, 19, 2, 0, tzinfo=timezone.utc)
    props = {"platform": "landsat-9", "datetime": "2023-08-15T19:02:00Z",
             "instruments": ["oli", "tirs"], "eo:cloud_cover": 10.0}
    return FakeStacItem(paths, props, dt)


@pytest.fixture
def landsat_scene_tm(tmp_path, aoi_grid):
    """`landsat_scene`'s pre-2013 twin: a Landsat 5 TM scene, thermal asset named `lwir`.

    The SAME synthetic pixels as `landsat_scene`, so the two fixtures are directly comparable:
    anything a source module produces differently for this one is a difference it invented,
    because C2 Level-2 gives TM and OLI-TIRS one algorithm and one scale/offset.

    Written to its own subdirectory so both fixtures can be requested by one test without the
    `lwir`/`lwir11` files colliding in a shared `tmp_path`.
    """
    d = tmp_path / "tm"
    d.mkdir()
    paths = write_landsat_cogs(d, aoi_grid, thermal_key="lwir")
    dt = datetime(1995, 7, 18, 18, 30, 0, tzinfo=timezone.utc)
    props = {"platform": "landsat-5", "datetime": "1995-07-18T18:30:00Z",
             "instruments": ["tm"], "eo:cloud_cover": 10.0}
    return FakeStacItem(paths, props, dt, id="FAKE_LT05_L2SP")


class UniformDs(dict):
    """A stand-in for `eff["ds"]`, which is keyed by AoI.

    Every product resolves its options PER AoI now (region override -> project default), so
    `run()` reads `eff["ds"][aoi_name]`. Most tests build an `eff` by hand and don't care
    about regions at all -- they want one uniform config for whatever AoIs they happen to
    pass -- so this hands back the same cfg for any key.

    A test that DOES exercise a region override builds a real `{aoi: cfg}` dict, which is
    the point: the difference between "regions are irrelevant here" and "regions are the
    thing under test" should be visible in the test's own setup.
    """

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    def __missing__(self, key):
        return self._cfg

    def values(self):                      # e.g. modis.acquire(full_series=True)
        return [self._cfg]


# ---------------------------------------------------------------------------
# Synthetic assembled cube + points file, for the `extract` stage.
# ---------------------------------------------------------------------------
# Every channel is an ANISOTROPIC ramp, and the 3-D and 2-D ones vary along
# DIFFERENT axes on purpose. A y-flip, a row/col transposition, or an off-by-one
# in the affine inversion then CHANGES THE NUMBER rather than returning something
# plausible -- which is the only way a point-extraction bug is ever caught.
CUBE_DAYS = 4
NAN_ROW, NAN_COL = 20, 20        # the one hole punched into `eco_sst_gappy`


def build_cube(g, days=CUBE_DAYS):
    """The synthetic cube as an xr.Dataset (see the fixture below for the layout)."""
    import pandas as pd
    import xarray as xr

    H, W = g.shape
    xs, ys = g.xy_centers()
    rows = np.arange(H)[:, None] * np.ones((1, W))
    cols = np.ones((H, 1)) * np.arange(W)[None, :]
    t = np.arange(days)[:, None, None]

    sst = (1000 * t + rows[None]).astype("float32")      # varies with ROW and time
    gappy = sst.copy()
    gappy[:, NAN_ROW, NAN_COL] = np.nan                  # the point's own pixel
    return xr.Dataset(
        {"eco_sst": (("time", "y", "x"), sst),
         "eco_sst_gappy": (("time", "y", "x"), gappy),
         "elevation_cudem": (("y", "x"), cols.astype("float32")),   # varies with COL
         "landcover_water": (("y", "x"), (cols < W // 2).astype("uint8")),  # west = water
         "tide_coops": (("time",), np.arange(days, dtype="float32")),
         # A second 1-D channel, and a NaN day: `<sensor>_hour` is the overpass time, which
         # is NaN whenever that sensor saw nothing -- the shape a real cube has.
         "lst_hour": (("time",), np.where(np.arange(days) == 1, np.nan,
                                          10.0 + np.arange(days)).astype("float32"))},
        coords={"time": pd.date_range("2023-06-01", periods=days, freq="D"),
                "y": ys, "x": xs},
        attrs={"crs": g.target_crs, "aoi_id": g.name})


@pytest.fixture
def cube_dir(tmp_path, aoi_grid):
    """A synthetic assembled cube at <tmp_path>/datacube/test_aoi.zarr.

        eco_sst        (t,y,x) = 1000*t + row   -> a y-flip or row error changes it
        eco_sst_gappy  (t,y,x) = eco_sst, NaN at (20, 20)
        elevation_cudem  (y,x) = col            -> a row/col swap changes it
        landcover_water  (y,x) = 1 over the WEST half
        tide_coops      (time,) = 0, 1, 2, 3
        lst_hour        (time,) = 10, NaN, 12, 13   (NaN = no overpass that day)
    """
    d = tmp_path / "datacube"
    d.mkdir(parents=True, exist_ok=True)
    build_cube(aoi_grid).to_zarr(d / f"{aoi_grid.name}.zarr", mode="w-", consolidated=True)
    return d


def pixel_lonlat(g, row, col):
    """The lon/lat of the CENTRE of pixel (row, col) on `g`.

    Derived by inverting the grid's own `xy_centers`, so a test asserting "this point is in
    pixel (20, 20)" is asserting against the grid's definition of a pixel centre rather than
    against a second, independent piece of arithmetic that could share the same bug.
    """
    from pyproj import Transformer
    xs, ys = g.xy_centers()
    inv = Transformer.from_crs(g.target_crs, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(float(xs[col]), float(ys[row]))
    return float(lon), float(lat)


@pytest.fixture
def points_csv(tmp_path, aoi_grid):
    """A points CSV with deliberately NON-canonical column names.

    `station`/`latitude`/`longitude` -- because the alias resolution is part of what has to
    work, and a fixture that used the canonical names would never exercise it.
    Sites: `centre` at pixel (20,20) (where the cube's NaN hole is), and `edge` one pixel
    in from the north-west corner.
    """
    import pandas as pd
    rows = []
    for name, (r, c) in {"centre": (20, 20), "edge": (1, 1)}.items():
        lon, lat = pixel_lonlat(aoi_grid, r, c)
        rows.append({"station": name, "latitude": lat, "longitude": lon})
    path = tmp_path / "sites.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
