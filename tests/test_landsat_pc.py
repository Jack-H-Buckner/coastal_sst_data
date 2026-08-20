"""Landsat via Planetary Computer. `_build_eff` maps the config; `scene_to_dataset`
turns a STAC item's COG assets into the aligned sst/cloud/water/valid Dataset.

The scene_to_dataset tests use the SHARED `landsat_scene` fixture (conftest), so
the same synthetic input can later be fed to landsat_aws / other source modules
and asserted to produce the same output. No network."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

import xarray as xr

from coastal_sst_data.config import load_config, parse_config, DataProduct
from coastal_sst_data import products, store
from coastal_sst_data.processes import landsat_pc

from .conftest import FakeStacItem, UniformDs

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"

# A signed PC href, as GDAL quotes it back at you in every error message. The acquisition
# date `20240403` contains "403" -- see test_net.test_a_date_in_an_href_is_not_a_status_code.
_HREF = ("/vsicurl/https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/"
         "oli-tirs/2024/046/027/LC08_L2SP_046027_20240403_02_T1/"
         "LC08_L2SP_046027_20240403_02_T1_ST_B10.TIF?st=2024-04-03T00%3A00%3A00Z&sig=x")


def _forbidden():
    """What an expired Planetary Computer signature looks like coming out of GDAL."""
    return RasterioIOError(f"'{_HREF}': HTTP response code: 403")


@pytest.fixture
def no_sleep(monkeypatch):
    """net.retry's backoff, without the wall clock.

    `sleep=time.sleep` is a DEFAULT ARGUMENT, bound when net.py was imported, so patching
    time.sleep does nothing -- wrap the function instead. Wrapping (rather than stubbing
    net.retry out) keeps the real transient/permanent logic under test, which is the whole
    point of these tests.
    """
    real = landsat_pc.net.retry
    monkeypatch.setattr(landsat_pc.net, "retry",
                        lambda fn, **kw: real(fn, **{**kw, "sleep": lambda s: None}))


def _ds(eff):
    """The settings ONE AoI runs with. `eff["ds"]` is keyed by AoI, because `collection` /
    `stac_url` travel with `source` and are region-overridable (a region served by a
    different catalogue names its own). No config below sets a region override, so every
    AoI resolves alike -- take any."""
    return next(iter(eff["ds"].values()))




# ---------------------------------------------------------------------------
# _build_eff: config -> acquisition params. No auth (PC is anonymous).
# ---------------------------------------------------------------------------
def test_build_eff_maps_example_config():
    eff = landsat_pc._build_eff(load_config(EXAMPLE))
    assert _ds(eff)["collection"] == "landsat-c2-l2"
    assert _ds(eff)["stac_url"] == landsat_pc.STAC_URL
    assert _ds(eff)["platforms"] == landsat_pc.DEFAULT_PLATFORMS
    assert _ds(eff)["cloud_cover_max"] == 0.7      # default
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
    assert _ds(eff)["platforms"] == landsat_pc.DEFAULT_PLATFORMS
    assert _ds(eff)["cloud_cover_max"] == 0.7
    assert _ds(eff)["collection"] == landsat_pc.COLLECTION
    assert _ds(eff)["masking"] == {}


def test_build_eff_applies_option_overrides(base_project):
    base_project["products"]["landsat"] = {
        "platforms": ["landsat-8"], "cloud_cover_max": 0.3,
        "output_format": "geotiff", "masking": {"ndwi_threshold": 0.1},
    }
    eff = landsat_pc._build_eff(parse_config(base_project))
    assert _ds(eff)["platforms"] == ["landsat-8"]
    assert _ds(eff)["cloud_cover_max"] == 0.3
    assert eff["fmt"] == "geotiff"
    assert _ds(eff)["masking"] == {"ndwi_threshold": 0.1}


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


def test_the_mask_layers_survive_a_round_trip_to_disk(tmp_path, aoi_grid, landsat_scene):
    """The regression, end to end: write the scene the way acquisition does, reopen it, and
    recompute validity the way the ASSEMBLER does.

    `cloud` is derived from the QA read, which runs `masked=False`, so it arrived carrying
    that read's `_FillValue: 0`. On a 0/1 mask 0 is CLEAR, not absent -- so every clear cell
    decoded back as NaN, `datacube._read_granule` read a NaN cloud cell as CLOUDY, and
    `lst_valid` came out empty on every granule. In the cube that made all of a day's
    granules tie at zero valid pixels, so the mosaic kept only whichever was read first and
    a multi-scene AoI showed one scene's footprint.
    """
    from coastal_sst_data.processes import datacube

    ds = landsat_pc.scene_to_dataset(landsat_scene, aoi_grid, _MASK, False,
                                     pd.Timestamp("2023-08-15T19:02:00"), "test_aoi")
    in_memory = int(ds["valid"].isel(time=0).values.sum())
    assert in_memory > 0                                    # sane before it is written

    store.write_output(ds, tmp_path, "test_aoi_20230815T190200", "netcdf")
    out = tmp_path / "test_aoi_20230815T190200.nc"

    reopened = xr.open_dataset(out).isel(time=0)
    clear = np.asarray(reopened["cloud"].values) == 0
    assert clear.any(), "every CLEAR cell decoded back as NaN -- the fill value ate them"

    sensor = products.spec(DataProduct.landsat).sensor
    _s, _c, v, _fp = datacube._read_granule(
        out, aoi_grid.height, aoi_grid.width,
        water_is_land=sensor.water_is_land, use_cloud=sensor.use_cloud,
        qset=list(sensor.qc_levels) if sensor.qc_levels else None,
        trust_valid=sensor.trust_valid, read_fp=False)
    assert int(v.sum()) > 0, "the assembler recomputed an EMPTY validity mask"
    assert int(v.sum()) == pytest.approx(in_memory, rel=0.01)


# ---------------------------------------------------------------------------
# Network hardening. A multi-year AoI used to stop partway through its date
# range: assets were signed ONCE at search time, the Planetary Computer SAS
# token expired ~an hour in, and every remaining scene 403'd. Scenes are
# processed in DATE order, so an expiring token looked exactly like a date
# cutoff -- and the next AoI, with a fresh search, worked for another hour.
# ---------------------------------------------------------------------------
def test_read_asset_retries_a_transient_read(monkeypatch, aoi_grid, landsat_scene, no_sleep):
    """The blip that used to lose a whole scene. Five assets, and one 503 on any of them meant
    the caller logged FAILED and moved on -- and the skip guard never revisited it."""
    real, calls = landsat_pc.read_cog_window, []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("connection reset by peer")
        return real(*a, **kw)

    monkeypatch.setattr(landsat_pc, "read_cog_window", flaky)
    da = landsat_pc._read_asset(landsat_scene, "lwir11", aoi_grid,
                                resampling=Resampling.bilinear)
    assert len(calls) == 2
    assert float(np.nanmax(da.values)) == pytest.approx(41252)   # the synthetic thermal DN


def test_read_asset_does_not_retry_a_forbidden_read(monkeypatch, aoi_grid, landsat_scene,
                                                    no_sleep):
    """A dead signature is an ANSWER. Re-requesting the same expired URL three times just
    buys the same 403 three times; the re-sign belongs one level up, in read_scene."""
    calls = []

    def forbidden(*a, **kw):
        calls.append(1)
        raise _forbidden()

    monkeypatch.setattr(landsat_pc, "read_cog_window", forbidden)
    with pytest.raises(RasterioIOError):
        landsat_pc._read_asset(landsat_scene, "lwir11", aoi_grid,
                               resampling=Resampling.bilinear)
    assert len(calls) == 1


def test_read_asset_names_the_scene_and_the_asset(monkeypatch, aoi_grid, landsat_scene):
    """net.retry's warning has to say WHICH asset of WHICH scene is flaking -- five reads a
    scene, hundreds of scenes a run."""
    seen = {}

    def spy(fn, *, what, **kw):
        seen["what"] = what
        return fn()

    monkeypatch.setattr(landsat_pc.net, "retry", spy)
    landsat_pc._read_asset(landsat_scene, "qa_pixel", aoi_grid,
                           resampling=Resampling.nearest, masked=False)
    assert landsat_scene.id in seen["what"] and "qa_pixel" in seen["what"]


@pytest.fixture
def sign_spy(monkeypatch):
    """Record every sign_item call. The real one needs pystac + the PC client and a network
    round-trip; FakeStacItem is not a pystac Item, so this is also what makes it usable."""
    signs = []
    monkeypatch.setattr(landsat_pc, "sign_item", lambda it: signs.append(it.id) or it)
    return signs


def test_read_scene_resigns_once_when_the_signature_expires(monkeypatch, aoi_grid,
                                                            landsat_scene, sign_spy):
    """A token with a minute left can die BETWEEN the first asset and the last, so signing per
    scene is necessary but not sufficient. One re-sign, then a full second pass."""
    real, reads = landsat_pc.read_cog_window, []

    def flaky(*a, **kw):
        reads.append(1)
        if len(reads) == 1:
            raise _forbidden()
        return real(*a, **kw)

    monkeypatch.setattr(landsat_pc, "read_cog_window", flaky)
    ds = landsat_pc.read_scene(landsat_scene, aoi_grid, _MASK, False,
                               pd.Timestamp("2023-08-15T19:02:00"), "test_aoi")

    assert sign_spy == [landsat_scene.id] * 2      # signed, 403'd, re-signed -- exactly once
    assert len(reads) == 6                         # the failed read + a full 5-asset retry
    assert {"sst", "cloud", "water", "valid"} <= set(ds.data_vars)


def test_read_scene_does_not_resign_a_flaky_connection(monkeypatch, aoi_grid, landsat_scene,
                                                       sign_spy, no_sleep):
    """Minting a new token for a URL that was never the problem is wasted work -- net.retry
    already handled the retrying, and it failed on its own terms."""
    reads = []

    def dead(*a, **kw):
        reads.append(1)
        raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(landsat_pc, "read_cog_window", dead)
    with pytest.raises(ConnectionError):
        landsat_pc.read_scene(landsat_scene, aoi_grid, _MASK, False,
                              pd.Timestamp("2023-08-15"), "test_aoi")
    assert len(sign_spy) == 1
    assert len(reads) == landsat_pc.READ_ATTEMPTS


def _eff(tmp_path, **over):
    """A minimal `eff` dict for run(). One uniform AoI config -- regions are not the subject."""
    eff = {
        "config_sha256": "deadbeef",
        "ds": UniformDs({"collection": landsat_pc.COLLECTION,
                         "stac_url": landsat_pc.STAC_URL,
                         "platforms": landsat_pc.DEFAULT_PLATFORMS,
                         "cloud_cover_max": 0.7, "masking": _MASK}),
        "grid": {"to_celsius": False},
        "out_dir": tmp_path / "out" / "LANDSAT" / "aligned",
        "fmt": "netcdf",
        "overwrite": False,
        "time": {"start_date": "2023-08-01", "end_date": "2023-08-31"},
    }
    eff.update(over)
    return eff


def test_run_finishes_the_date_range_after_a_signature_expires(monkeypatch, tmp_path,
                                                               aoi_grid, landsat_scene,
                                                               sign_spy):
    """THE REPORTED BUG. Three date-sorted scenes; the token dies on the first asset of the
    second. Before per-scene signing that killed scene 2 AND scene 3 -- the AoI simply stopped
    partway through its range and the loader moved on to the next one."""
    hrefs = {k: a.href for k, a in landsat_scene.assets.items()}
    items = [FakeStacItem(hrefs,
                          {"platform": "landsat-9", "datetime": f"2023-08-{d}T19:02:00Z"},
                          datetime(2023, 8, d, 19, 2, tzinfo=timezone.utc),
                          id=f"LC09_{d}")
             for d in (15, 16, 17)]
    monkeypatch.setattr(landsat_pc, "search_scenes", lambda *a, **k: items)

    real, reads = landsat_pc.read_cog_window, []

    def flaky(*a, **kw):
        reads.append(1)
        if len(reads) == 6:            # first asset of scene 2
            raise _forbidden()
        return real(*a, **kw)

    monkeypatch.setattr(landsat_pc, "read_cog_window", flaky)
    rep = landsat_pc.run(_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, dry_run=False)

    assert (rep.written, rep.failed) == (3, 0)
    assert sign_spy == ["LC09_15", "LC09_16", "LC09_16", "LC09_17"]   # one re-sign, scene 2
    assert len(list((tmp_path / "out" / "LANDSAT" / "aligned" / "test_aoi").glob("*.nc"))) == 3


def test_search_scenes_does_not_sign_at_search_time(monkeypatch):
    """The tripwire on the whole fix. `planetary_computer.sign_url` returns an href that
    already carries st/se/sp UNCHANGED -- so signing here would make every later re-sign a
    silent no-op handing back the same dead token, and the diff would still look correct."""
    pystac_client = pytest.importorskip("pystac_client")
    seen = {}

    class _Cat:
        def search(self, **kw):
            return type("S", (), {"items": lambda self: []})()

    def _open(url, **kw):
        seen["kwargs"] = kw
        return _Cat()

    monkeypatch.setattr(pystac_client.Client, "open", _open)
    items = landsat_pc.search_scenes(landsat_pc.COLLECTION, landsat_pc.STAC_URL,
                                     (-124.0, 45.4, -123.8, 45.6), "2023-08-01", "2023-08-31",
                                     landsat_pc.DEFAULT_PLATFORMS, 0.7)
    assert items == []
    assert "modifier" not in seen["kwargs"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])