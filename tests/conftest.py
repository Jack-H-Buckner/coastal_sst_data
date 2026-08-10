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
    """Minimal stand-in for a pystac Item (what scene_to_dataset / run touch)."""
    def __init__(self, assets, properties, dt, id="FAKE_LC09_L2SP"):
        self.assets = {k: FakeAsset(h) for k, h in assets.items()}
        self.properties = properties
        self.datetime = dt
        self.id = id


def write_landsat_cogs(dir_path, g, *, native_res=30.0, pad_m=1500.0):
    """Write synthetic Landsat C2 L2 single-band COGs (raw DN) for one scene.

    DN values are chosen so the DERIVED layers are known after a source module
    reprojects them onto grid `g`:
        lwir11 (thermal) DN 41252  -> 290 K everywhere
        green DN 14545 / nir08 DN 10909 -> NDWI ~0.33 -> water everywhere
        qa_pixel: cloud bit (1<<3) over the WEST half, 0 over the EAST half
        cdist DN 200 -> 2 km, so the 1 km cloud buffer never fires
    => expected valid mask: 1 in the EAST half, 0 in the WEST half.
    Source is native ~30 m in the SAME CRS as the target grid. Returns
    {asset_key: path-str}, ready to hang off a FakeStacItem.
    """
    minx, miny, maxx, maxy = g.geom_proj.bounds
    minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
    W = int(round((maxx - minx) / native_res))
    H = int(round((maxy - miny) / native_res))
    transform = from_origin(minx, maxy, native_res, native_res)

    thermal = np.full((H, W), 41252, dtype="uint16")   # -> 290.0 K
    green = np.full((H, W), 14545, dtype="uint16")      # reflectance ~0.20
    nir = np.full((H, W), 10909, dtype="uint16")        # reflectance ~0.10 -> water
    qa = np.zeros((H, W), dtype="uint16")
    qa[:, : W // 2] = 1 << 3                             # west half: cloud bit
    cdist = np.full((H, W), 200, dtype="uint16")        # 2 km -> no buffer effect

    layers = [("lwir11", thermal, 0), ("green", green, 0), ("nir08", nir, 0),
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
             "eo:cloud_cover": 10.0}
    return FakeStacItem(paths, props, dt)


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
