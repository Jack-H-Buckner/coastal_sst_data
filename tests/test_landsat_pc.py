"""Landsat via Planetary Computer. `_build_eff` maps the config; `scene_to_dataset`
turns a STAC item's COG assets into the aligned sst/cloud/water/valid Dataset.

The scene_to_dataset tests use the SHARED `landsat_scene` fixture (conftest), so
the same synthetic input can later be fed to landsat_aws / other source modules
and asserted to produce the same output. No network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coastal_sst_data.config import load_config, parse_config, DataProduct
from coastal_sst_data.processes import landsat_pc

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


# ---------------------------------------------------------------------------
# _build_eff: config -> acquisition params. No auth (PC is anonymous).
# ---------------------------------------------------------------------------
def test_build_eff_maps_example_config():
    eff = landsat_pc._build_eff(load_config(EXAMPLE))
    assert eff["ds"]["collection"] == "landsat-c2-l2"
    assert eff["ds"]["stac_url"] == landsat_pc.STAC_URL
    assert eff["ds"]["platforms"] == landsat_pc.DEFAULT_PLATFORMS
    assert eff["ds"]["cloud_cover_max"] == 0.7      # default
    assert eff["fmt"] == "netcdf"
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["out_dir"] == Path("path/to/data") / "LANDSAT" / "aligned"
    assert "earthdata" not in eff and "gee_project" not in eff   # PC needs no auth


def test_build_eff_requires_landsat_selected(base_project):
    """base_project selects bathymetry + mur (no landsat)."""
    with pytest.raises(ValueError, match="landsat is not a selected product"):
        landsat_pc._build_eff(parse_config(base_project))


def test_build_eff_defaults_when_options_omitted(base_project):
    base_project["products"]["landsat"] = None      # bare -> defaults
    eff = landsat_pc._build_eff(parse_config(base_project))
    assert eff["ds"]["platforms"] == landsat_pc.DEFAULT_PLATFORMS
    assert eff["ds"]["cloud_cover_max"] == 0.7
    assert eff["ds"]["collection"] == landsat_pc.COLLECTION
    assert eff["ds"]["masking"] == {}


def test_build_eff_applies_option_overrides(base_project):
    base_project["products"]["landsat"] = {
        "platforms": ["landsat-8"], "cloud_cover_max": 0.3,
        "output_format": "geotiff", "masking": {"ndwi_threshold": 0.1},
    }
    eff = landsat_pc._build_eff(parse_config(base_project))
    assert eff["ds"]["platforms"] == ["landsat-8"]
    assert eff["ds"]["cloud_cover_max"] == 0.3
    assert eff["fmt"] == "geotiff"
    assert eff["ds"]["masking"] == {"ndwi_threshold": 0.1}


# ---------------------------------------------------------------------------
# scene_to_dataset: STAC item (synthetic COGs) -> aligned Dataset on the grid.
# Uses the shared `landsat_scene` fixture (known layout: cloud west, water all).
# ---------------------------------------------------------------------------
_MASK = {"ndwi_threshold": 0.0, "cloud_buffer_km": 1.0}


def test_scene_to_dataset_schema_and_grid(aoi_grid, landsat_scene):
    ds = landsat_pc.scene_to_dataset(landsat_scene, aoi_grid, _MASK,
                                     to_celsius=False,
                                     acq_time=pd.Timestamp("2023-08-15T19:02:00"),
                                     aoi_id="test_aoi")
    # the shared contract schema, on the shared grid
    assert {"sst", "cloud", "water", "valid"} <= set(ds.data_vars)
    assert ds.sizes["y"] == aoi_grid.height and ds.sizes["x"] == aoi_grid.width
    assert str(ds["sst"].rio.crs) == aoi_grid.target_crs
    assert ds.attrs["aoi_id"] == "test_aoi"


def test_scene_to_dataset_sst_kelvin(aoi_grid, landsat_scene):
    ds = landsat_pc.scene_to_dataset(landsat_scene, aoi_grid, _MASK, False,
                                     pd.Timestamp("2023-08-15"), "test_aoi")
    finite = ds["sst"].values[np.isfinite(ds["sst"].values)]
    assert finite.min() == pytest.approx(290.0, abs=0.5)   # DN 41252 -> 290 K
    assert finite.max() == pytest.approx(290.0, abs=0.5)
    assert ds["sst"].attrs["units"] == "K"


def test_scene_to_dataset_to_celsius(aoi_grid, landsat_scene):
    ds = landsat_pc.scene_to_dataset(landsat_scene, aoi_grid, _MASK, True,
                                     pd.Timestamp("2023-08-15"), "test_aoi")
    finite = ds["sst"].values[np.isfinite(ds["sst"].values)]
    assert finite.min() == pytest.approx(290.0 - 273.15, abs=0.5)   # 16.85
    assert ds["sst"].attrs["units"] == "degC"


def test_scene_to_dataset_valid_mask_east_west(aoi_grid, landsat_scene):
    """Known layout: water everywhere, cloud in the WEST -> valid in the EAST."""
    ds = landsat_pc.scene_to_dataset(landsat_scene, aoi_grid, _MASK, False,
                                     pd.Timestamp("2023-08-15"), "test_aoi")
    assert ds["valid"].dtype == "uint8"
    assert np.nanmin(ds["water"].isel(time=0).values) == 1.0    # all water
    v = ds["valid"].isel(time=0).values
    assert v[:, : v.shape[1] // 4].sum() == 0      # deep west: cloudy -> invalid
    assert v[:, 3 * v.shape[1] // 4 :].sum() > 0   # deep east: clear water -> valid
