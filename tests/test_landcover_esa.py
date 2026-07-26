"""ESA WorldCover landcover/water mask via Planetary Computer. `_build_eff` maps
the config; `items_to_dataset` turns WorldCover map COGs into the static aligned
landcover/water Dataset on the shared grid. No network (synthetic COG)."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from coastal_sst_data.config import load_config, parse_config, DataProduct
from coastal_sst_data.processes import landcover_esa

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


def _ds(eff):
    """The settings ONE AoI runs with. `eff["ds"]` is keyed by AoI, because `collection` /
    `stac_url` travel with `source` and are region-overridable (a region served by a
    different catalogue names its own). No config below sets a region override, so every
    AoI resolves alike -- take any."""
    return next(iter(eff["ds"].values()))




# --- minimal STAC-item stand-ins (only .assets[k].href / .properties are read) - #
class _Asset:
    def __init__(self, href):
        self.href = href


class _Item:
    def __init__(self, assets, properties=None):
        self.assets = {k: _Asset(v) for k, v in assets.items()}
        self.properties = properties or {}


def write_worldcover_cog(path, g, *, native_res=30.0, pad_m=1500.0,
                         west_class=10, east_class=80):
    """Synthetic WorldCover `map` COG over grid `g`: west half class 10 (tree),
    east half class 80 (water). Same CRS as the target grid, native ~30 m."""
    minx, miny, maxx, maxy = g.geom_proj.bounds
    minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
    W = int(round((maxx - minx) / native_res))
    H = int(round((maxy - miny) / native_res))
    transform = from_origin(minx, maxy, native_res, native_res)
    arr = np.full((H, W), west_class, dtype="uint8")
    arr[:, W // 2:] = east_class
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="uint8", crs=g.target_crs, transform=transform, nodata=0) as dst:
        dst.write(arr, 1)
    return str(path)


# ---------------------------------------------------------------------------
# _build_eff: config -> acquisition params. No auth (PC is anonymous).
# ---------------------------------------------------------------------------
def test_build_eff_maps_example_config():
    eff = landcover_esa._build_eff(load_config(EXAMPLE))
    assert _ds(eff)["collection"] == "esa-worldcover"
    assert _ds(eff)["stac_url"] == landcover_esa.STAC_URL
    assert _ds(eff)["year"] == 2021
    assert _ds(eff)["water_classes"] == [80]
    assert eff["fmt"] == "netcdf"
    assert eff["out_dir"] == Path("path/to/data") / "LANDCOVER" / "aligned"
    assert "earthdata" not in eff and "gee_project" not in eff   # PC needs no auth


def test_build_eff_requires_landcover_selected(base_project):
    """base_project selects bathymetry + mur (no landcover)."""
    with pytest.raises(ValueError, match="landcover is not a selected product"):
        landcover_esa._build_eff(parse_config(base_project))


def test_build_eff_defaults_when_options_omitted(base_project):
    base_project["products"]["landcover"] = None      # bare -> defaults (source esa)
    eff = landcover_esa._build_eff(parse_config(base_project))
    assert _ds(eff)["collection"] == landcover_esa.COLLECTION
    assert _ds(eff)["year"] == landcover_esa.DEFAULT_YEAR
    assert _ds(eff)["water_classes"] == landcover_esa.DEFAULT_WATER_CLASSES


def test_build_eff_applies_option_overrides(base_project):
    base_project["products"]["landcover"] = {
        "year": 2020, "water_classes": [80, 90], "output_format": "geotiff",
        "overwrite": True,
    }
    eff = landcover_esa._build_eff(parse_config(base_project))
    assert _ds(eff)["year"] == 2020
    assert _ds(eff)["water_classes"] == [80, 90]
    assert eff["fmt"] == "geotiff"
    assert eff["overwrite"] is True


# ---------------------------------------------------------------------------
# _item_year
# ---------------------------------------------------------------------------
def test_item_year():
    assert landcover_esa._item_year(_Item({}, {"start_datetime": "2021-06-30T00:00:00Z"})) == 2021
    assert landcover_esa._item_year(_Item({}, {"datetime": "2020-01-01T00:00:00Z"})) == 2020
    assert landcover_esa._item_year(_Item({}, {})) is None


# ---------------------------------------------------------------------------
# items_to_dataset: WorldCover COG(s) -> static landcover/water on the grid.
# Known layout: west half class 10 (land), east half class 80 (water).
# ---------------------------------------------------------------------------
def test_items_to_dataset_schema_and_grid(aoi_grid, tmp_path):
    href = write_worldcover_cog(tmp_path / "wc.tif", aoi_grid)
    item = _Item({"map": href}, {"start_datetime": "2021-01-01T00:00:00Z"})
    ds = landcover_esa.items_to_dataset([item], aoi_grid, [80], "test_aoi")

    assert {"landcover", "water"} <= set(ds.data_vars)
    assert ds.sizes["y"] == aoi_grid.height and ds.sizes["x"] == aoi_grid.width
    assert str(ds["water"].rio.crs) == aoi_grid.target_crs
    assert ds["landcover"].dtype == "int16" and ds["water"].dtype == "uint8"
    assert "time" not in ds.dims                       # static layer
    assert ds.attrs["aoi_id"] == "test_aoi"


def test_items_to_dataset_water_east_land_west(aoi_grid, tmp_path):
    href = write_worldcover_cog(tmp_path / "wc.tif", aoi_grid)
    item = _Item({"map": href})
    ds = landcover_esa.items_to_dataset([item], aoi_grid, [80], "test_aoi")

    w = ds["water"].values
    assert w[:, 3 * w.shape[1] // 4:].sum() > 0        # deep east: class 80 -> water
    assert w[:, : w.shape[1] // 4].sum() == 0          # deep west: class 10 -> land
    assert set(np.unique(ds["landcover"].values)) <= {0, 10, 80}


def test_items_to_dataset_water_classes_configurable(aoi_grid, tmp_path):
    """Treating class 10 as 'water' flips which half is marked."""
    href = write_worldcover_cog(tmp_path / "wc.tif", aoi_grid)
    ds = landcover_esa.items_to_dataset([_Item({"map": href})], aoi_grid, [10], "test_aoi")
    w = ds["water"].values
    assert w[:, : w.shape[1] // 4].sum() > 0           # west (class 10) now water
    assert w[:, 3 * w.shape[1] // 4:].sum() == 0       # east (class 80) now "land"


# ---------------------------------------------------------------------------
# Round-trip: `water` must survive a netcdf write + default read intact. The COG read
# leaks _FillValue: 0 onto `water`, where 0 means LAND -- if it reaches disk, xarray's
# default mask_and_scale=True decodes every land cell to NaN, collapsing the mask to
# uniformly-water downstream. The tests above only see the in-memory array; this is the
# read path every consumer (load_landcover, the skip guard) actually uses.
# ---------------------------------------------------------------------------
def test_water_roundtrips_without_fillvalue_corruption(aoi_grid, tmp_path):
    from coastal_sst_data import store

    href = write_worldcover_cog(tmp_path / "wc.tif", aoi_grid)
    ds = landcover_esa.items_to_dataset([_Item({"map": href})], aoi_grid, [80], "test_aoi")
    encoding = {"water": dict(landcover_esa.WATER_ENCODING),
                "landcover": {"zlib": True, "complevel": 4}}
    path = store.write_output(ds, tmp_path / "out", "test_aoi", "netcdf", encoding=encoding)

    with xr.open_dataset(path) as re:                       # default read = mask_and_scale=True
        w = re["water"]
        assert w.dtype == np.dtype("uint8")                 # not promoted to float64
        assert not np.isnan(w.values.astype("float64")).any()   # land (0) stayed 0, not NaN
        assert set(np.unique(w.values)) <= {0, 1}
        assert (w.values[:, : w.shape[1] // 4] == 0).all()      # deep west = land = 0
        assert w.values[:, 3 * w.shape[1] // 4:].sum() > 0      # deep east = water = 1

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])