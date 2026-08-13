
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from pathlib import Path

from coastal_sst_data.config import load_config, parse_config, AreaOfInterest, GridSpec
from coastal_sst_data.processes import met
from coastal_sst_data import grid


EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


def _ds(eff):
    """The met settings ONE AoI runs with.

    `eff["ds"]` is keyed by AoI, because met's source chain is region-dependent: HRRR is
    North America only, so an AoI outside it must start at ERA5. None of the configs below
    set a region override, so every AoI resolves to the same settings -- take any.
    (test_region_sources.py covers the case where they genuinely differ.)
    """
    return next(iter(eff["ds"].values()))


# --------------------------------------------------------------------------- #
# _build_eff: config -> acquisition params
# --------------------------------------------------------------------------- #
def test_build_eff_maps_example_config():
    """The example config maps to the expected met FORCING acquisition parameters."""
    eff = met._build_eff(load_config(EXAMPLE))
    assert _ds(eff)["sources"] == ["hrrr", "era5"]        # stacked (D10)
    assert _ds(eff)["variables"] == ["airtemp", "wind", "swrad", "cloud"]
    assert _ds(eff)["daily_mean_hours"] == [0, 6, 12, 18]
    assert _ds(eff)["era5_zarr"] == met.ARCO_ERA5_URI     # default ARCO store
    # shared project settings flow through
    assert eff["fmt"] == "netcdf"
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["grid"]["resolution_m"] == 100.0
    assert eff["met_root"] == Path("path/to/data") / "MET"    # per-source dirs under here
    # overpass documentation is a SEPARATE product now (met_overpass); met carries no overpass.
    assert "overpass_dirs" not in eff
    # AoI geometry lives in the shared grid (grid.py), not in eff.
    assert "aois" not in eff


def test_build_eff_requires_met_selected(base_project):
    """Calling the adapter when met isn't a selected product is an error."""
    base_project["products"] = {"bathymetry": None}        # drop met
    with pytest.raises(ValueError, match="met is not a selected product"):
        met._build_eff(parse_config(base_project))


def test_build_eff_defaults_when_options_omitted(base_project):
    """A bare `met:` (no options) falls back to product defaults."""
    base_project["products"]["met"] = None                 # bare -> default options
    eff = met._build_eff(parse_config(base_project))
    assert _ds(eff)["sources"] == met.DEFAULT_SOURCES == ["hrrr", "era5"]
    assert _ds(eff)["variables"] == met.DEFAULT_VARIABLES
    assert _ds(eff)["daily_mean_hours"] == met.DEFAULT_MEAN_HOURS
    assert eff["overwrite"] is False


def test_build_eff_applies_option_overrides(base_project):
    """met options override the product defaults."""
    base_project["products"]["met"] = {
        "sources": ["era5"], "variables": ["airtemp", "wind"],
        "regrid_radius_m": 3000, "output_format": "geotiff", "overwrite": True,
    }
    eff = met._build_eff(parse_config(base_project))
    assert _ds(eff)["sources"] == ["era5"]                # era5-only stack
    assert _ds(eff)["variables"] == ["airtemp", "wind"]
    assert _ds(eff)["regrid_radius_m"] == 3000.0
    assert eff["fmt"] == "geotiff"
    assert eff["overwrite"] is True


def test_unknown_source_is_rejected_at_config_load(base_project):
    """A typo'd source in the stacked list fails at config validation, before any work."""
    base_project["products"]["met"] = {"sources": ["gfs"]}
    with pytest.raises(Exception, match="unknown source"):
        parse_config(base_project)


# --------------------------------------------------------------------------- #
# HRRR domain selection
# --------------------------------------------------------------------------- #
def _grid_at(lat, lon):
    area = AreaOfInterest(name="a", center_lat=lat, center_lon=lon,
                          buffer_ns_km=8.0, buffer_ew_km=8.0)
    return grid.compute_aoi_grid(area, GridSpec())


def test_hrrr_model_for_grid():
    assert met._hrrr_model_for_grid(_grid_at(45.5, -123.9), "auto") == "hrrr"    # PNW
    assert met._hrrr_model_for_grid(_grid_at(58.0, -135.0), "auto") == "hrrrak"  # SE Alaska
    assert met._hrrr_model_for_grid(_grid_at(50.0, 0.0), "auto") is None         # Europe -> ERA5
    # explicit model overrides the auto domain check
    assert met._hrrr_model_for_grid(_grid_at(50.0, 0.0), "hrrr") == "hrrr"


# --------------------------------------------------------------------------- #
# ERA5 unit harmonization
# --------------------------------------------------------------------------- #
def test_era5_normalize():
    fields = {
        "airtemp": np.array([290.0], dtype="float32"),
        "swrad": np.array([3.6e6], dtype="float32"),      # 3.6 MJ/m2 over 1 h
        "cloud_cover": np.array([0.5], dtype="float32"),  # fraction
    }
    out = met._era5_normalize(fields)
    assert out["airtemp"][0] == pytest.approx(290.0)      # K unchanged
    assert out["swrad"][0] == pytest.approx(1000.0)       # 3.6e6 / 3600 -> W/m2
    assert out["cloud_cover"][0] == pytest.approx(50.0)   # 0.5 -> 50 %


# --------------------------------------------------------------------------- #
# Dataset assembly (source-agnostic)
# --------------------------------------------------------------------------- #
def test_to_dataset_builds_wind_speed_and_units(aoi_grid):
    g = aoi_grid
    shp = g.shape
    grids = {
        "airtemp": np.full(shp, 290.0, dtype="float32"),
        "wind_u": np.full(shp, 3.0, dtype="float32"),
        "wind_v": np.full(shp, 4.0, dtype="float32"),
        "swrad": np.full(shp, 500.0, dtype="float32"),
        "cloud_cover": np.full(shp, 40.0, dtype="float32"),
    }
    ds = met.to_dataset(grids, g, "2026-06-15", to_celsius=False)
    assert ds["airtemp"].isel(time=0).shape == shp
    assert ds["airtemp"].attrs["units"] == "K"
    assert float(ds["wind_speed"].isel(time=0).values[0, 0]) == pytest.approx(5.0)  # 3-4-5
    assert ds["wind_speed"].attrs["units"] == "m s-1"
    assert ds["swrad"].attrs["units"] == "W m-2"
    assert ds["cloud_cover"].attrs["units"] == "%"
    assert "time" in ds.dims
    assert str(ds.rio.crs) == g.target_crs


def test_to_dataset_to_celsius(aoi_grid):
    g = aoi_grid
    grids = {"airtemp": np.full(g.shape, 290.0, dtype="float32")}
    ds = met.to_dataset(grids, g, "2026-06-15", to_celsius=True)
    assert float(ds["airtemp"].isel(time=0).values[0, 0]) == pytest.approx(290.0 - 273.15)
    assert ds["airtemp"].attrs["units"] == "degC"


# --------------------------------------------------------------------------- #
# Reference time of day (the cube's met channel; default 10:30 Landsat overpass)
# --------------------------------------------------------------------------- #
def test_parse_hhmm():
    assert met.parse_hhmm("10:30") == pytest.approx(10.5)
    assert met.parse_hhmm("6") == pytest.approx(6.0)
    assert met.parse_hhmm(None) is None          # None -> no reference snapshot
    assert met.parse_hhmm("") is None


def test_solar_reference_is_the_same_time_of_day_in_every_aoi():
    """10:30 LOCAL SOLAR is a different UTC hour per AoI -- that is the whole point.

    A fixed UTC hour would be mid-morning in Oregon and the middle of the night in
    Maine, so cross-AoI forcing would not be like-for-like.
    """
    day = pd.Timestamp("2026-06-15")
    # Tillamook (-123.9): UTC = 10.5 + 8.26 = 18.76 -> 19:00
    assert met.reference_time_utc(day, -123.925, 10.5, "solar").hour == 19
    # Chesapeake (-76.0): UTC = 10.5 + 5.07 = 15.57 -> 16:00
    assert met.reference_time_utc(day, -76.0, 10.5, "solar").hour == 16


def test_utc_basis_is_taken_literally():
    day = pd.Timestamp("2026-06-15")
    t = met.reference_time_utc(day, -123.925, 10.5, "utc")
    assert (t.hour, t.day) == (10, 15)           # longitude ignored


def test_solar_reference_rolls_the_date_across_the_dateline():
    day = pd.Timestamp("2026-06-15")
    t = met.reference_time_utc(day, 170.0, 10.5, "solar")   # 10.5 - 11.33 = -0.83 h
    assert t.day == 14 and t.hour == 23                     # previous UTC day


def test_reference_defaults_are_the_landsat_overpass():
    eff = met._build_eff(load_config(EXAMPLE))
    assert _ds(eff)["reference_time"] == "10:30"
    assert _ds(eff)["reference_basis"] == "solar"


# --------------------------------------------------------------------------- #
# Per-source fetch (no chain, no fallback -- sources are STACKED as separate channels)
# --------------------------------------------------------------------------- #
def _stub_sources(monkeypatch, **by_name):
    """Replace the source registry: {name: callable(g, dt, cfg) -> grids or None}."""
    monkeypatch.setattr(met, "_SOURCES", by_name)


def test_fetch_one_returns_a_sources_grids(monkeypatch, aoi_grid):
    _stub_sources(monkeypatch,
                  era5=lambda g, dt, cfg: {"airtemp": np.zeros((2, 2), "float32")})
    got = met._fetch_one("era5", aoi_grid, datetime(2023, 7, 15, 18), {})
    assert got is not None and "airtemp" in got


def test_fetch_one_no_data_returns_none_no_substitution(monkeypatch, aoi_grid):
    """No fallback (D10): a source with no data here yields None (a NaN slice in ITS channel),
    never the OTHER source's data -- HRRR and ERA5 are not interchangeable."""
    _stub_sources(monkeypatch, hrrr=lambda g, dt, cfg: None,
                  era5=lambda g, dt, cfg: {"airtemp": np.zeros((2, 2), "float32")})
    assert met._fetch_one("hrrr", aoi_grid, datetime(2023, 7, 15, 18), {}) is None


def test_fetch_one_raises_a_source_error_rather_than_swallowing_it(monkeypatch, aoi_grid):
    """A FAILED read and an ABSENCE of data are different answers, and `_fetch_one` used to
    give both of them as None. The caller then tallied that None as `"<src>: no data"`, so a
    lost download reached the run report wearing the words of a coverage gap -- exactly the
    "one blip becomes a permanent hole in the cube" failure net.py exists to remove."""
    def boom(g, dt, cfg):
        raise RuntimeError("herbie exploded")
    _stub_sources(monkeypatch, hrrr=boom)
    with pytest.raises(RuntimeError, match="herbie exploded"):
        met._fetch_one("hrrr", aoi_grid, datetime(2023, 7, 15, 18), {})


def test_daily_mean_records_the_hours_that_ACTUALLY_contributed(monkeypatch, tmp_path,
                                                                aoi_grid, caplog):
    """The attr used to echo the REQUESTED hours regardless, so a 'daily mean' built from
    1 of 4 hours claimed all 4 -- a mean over a quarter of the diurnal cycle, wearing the
    label of a full-day mean."""
    def only_noon(g, dt, cfg):                       # 00/06/18 UTC unavailable; 12 works
        if dt.hour != 12:
            return None
        return {"airtemp": np.full((aoi_grid.height, aoi_grid.width), 290.0, "float32")}

    _stub_sources(monkeypatch, hrrr=only_noon)

    eff = {
        "ds": {aoi_grid.name: {
            "sources": ["hrrr"], "daily_mean_hours": [0, 6, 12, 18],
            "reference_time": None, "reference_basis": "solar", "variables": ["airtemp"],
            "model": "auto", "fxx": 0, "product": "sfc", "regrid_radius_m": 6000.0}},
        "grid": {"to_celsius": False},
        "met_root": tmp_path, "fmt": "netcdf", "overwrite": False,
        "time": {"start_date": "2023-07-15", "end_date": "2023-07-15"},
        "config_sha256": "x",
    }
    with caplog.at_level("WARNING"):
        met.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    f = tmp_path / "hrrr" / "aligned" / aoi_grid.name / f"{aoi_grid.name}_20230715.nc"
    with xr.open_dataset(f) as ds:
        assert ds.attrs["daily_mean_hours"] == "[12]"                  # what we GOT
        assert ds.attrs["daily_mean_hours_requested"] == "[0, 6, 12, 18]"   # what we asked
    assert "built from 1 of 4 hours" in caplog.text
    assert "NOT a full-day mean" in caplog.text

# --------------------------------------------------------------------------- #
# Network hardening
#
# met was the last module reading the network without a timeout, a retry or a backoff --
# and `_fetch_one` swallowed every exception into None, which the caller then tallied with
# the SAME WORDS as a genuine coverage gap. A lost download was therefore indistinguishable
# from an AoI the model does not cover, in the log and in the run report alike.
# --------------------------------------------------------------------------- #
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def _met_eff(tmp_path, aoi, **ds):
    cfg = {"sources": ["hrrr"], "daily_mean_hours": [], "reference_time": "10:30",
           "reference_basis": "solar", "variables": ["airtemp"], "model": "auto",
           "fxx": 0, "product": "sfc", "regrid_radius_m": 6000.0}
    cfg.update(ds)
    return {
        "ds": {aoi.name: cfg}, "grid": {"to_celsius": False},
        "met_root": tmp_path, "fmt": "netcdf", "overwrite": False,
        "time": {"start_date": "2023-07-15", "end_date": "2023-07-15"},
        "config_sha256": "x",
    }


def test_hrrr_cycle_is_retried_on_a_transient_failure(monkeypatch, aoi_grid):
    """Herbie had no retry at all, so one 503 permanently lost that cycle -- and because the
    skip guard treats a written output as done, a later run never went back for it."""
    _no_sleep(monkeypatch)
    calls = []

    def flaky(model, dt, fxx, product, var_keys):
        calls.append(dt)
        if len(calls) < 3:
            raise TimeoutError("connection reset by peer")
        return {"airtemp": np.zeros((2, 2), "float32")}, np.zeros((2, 2)), np.zeros((2, 2))

    monkeypatch.setattr(met, "_hrrr_fetch_cycle", flaky)
    monkeypatch.setattr(met, "_regrid_nearest",
                        lambda fields, lon2d, lat2d, g, radius: fields)

    got = met._fetch_hrrr(aoi_grid, datetime(2023, 7, 15, 18),
                          {"model": "auto", "fxx": 0, "product": "sfc",
                           "variables": ["airtemp"], "regrid_radius_m": 6000.0})
    assert len(calls) == 3 and got is not None


def test_a_failed_reference_is_reported_as_a_loss_not_as_no_data(monkeypatch, tmp_path,
                                                                 aoi_grid):
    """THE bug this hardening exists for. Both outcomes used to reach the run report as
    `hrrr: no data`, so a flaky network read like a region the model does not cover."""
    _no_sleep(monkeypatch)

    def boom(g, dt, cfg):
        raise TimeoutError("herbie mirror stalled")
    _stub_sources(monkeypatch, hrrr=boom)

    rep = met.run(_met_eff(tmp_path, aoi_grid), {aoi_grid.name: aoi_grid}, None, False)

    assert rep.failed == 1 and rep.written == 0
    (_item, why), = rep.failures
    assert "herbie mirror stalled" in why
    assert "no data" not in why
    # And nothing on disk, so the next run retries the day instead of skipping it forever.
    assert not list(tmp_path.rglob("*.nc"))


def test_a_genuinely_absent_reference_still_reports_no_data(monkeypatch, tmp_path, aoi_grid):
    """The other half of the split: HRRR is North America only, and an AoI outside it is a
    fact about the DATA. That must keep reading as `no data`, not as a failure."""
    _stub_sources(monkeypatch, hrrr=lambda g, dt, cfg: None)

    rep = met.run(_met_eff(tmp_path, aoi_grid), {aoi_grid.name: aoi_grid}, None, False)

    assert rep.failed == 1
    (_item, why), = rep.failures
    assert why == "hrrr: no data"


def test_a_lost_hour_abandons_the_daily_mean_rather_than_writing_a_partial(
        monkeypatch, tmp_path, aoi_grid):
    """A partial mean written over a LOST hour would be permanent: the file exists, so the
    skip guard takes the day for done on every later run and the lost hour is never
    re-fetched. A transient blip must not become a permanent hole."""
    _no_sleep(monkeypatch)

    def dies_at_noon(g, dt, cfg):
        if dt.hour == 12:
            raise TimeoutError("GCS said 503")
        return {"airtemp": np.full((aoi_grid.height, aoi_grid.width), 290.0, "float32")}
    _stub_sources(monkeypatch, hrrr=dies_at_noon)

    eff = _met_eff(tmp_path, aoi_grid, reference_time=None, daily_mean_hours=[0, 6, 12, 18])
    rep = met.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.written == 0 and rep.failed == 1
    (_item, why), = rep.failures
    assert "GCS said 503" in why
    assert not list(tmp_path.rglob("*.nc"))


def test_a_no_data_hour_still_builds_a_partial_daily_mean(monkeypatch, tmp_path, aoi_grid):
    """Contrast with the test above: a MISSING hour is a fact about the data, so the partial
    mean is still built (and still labelled with the hours that actually contributed)."""
    def only_noon(g, dt, cfg):
        if dt.hour != 12:
            return None
        return {"airtemp": np.full((aoi_grid.height, aoi_grid.width), 290.0, "float32")}
    _stub_sources(monkeypatch, hrrr=only_noon)

    eff = _met_eff(tmp_path, aoi_grid, reference_time=None, daily_mean_hours=[0, 6, 12, 18])
    rep = met.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.written == 1 and rep.failed == 0
    f = tmp_path / "hrrr" / "aligned" / aoi_grid.name / f"{aoi_grid.name}_20230715.nc"
    with xr.open_dataset(f) as ds:
        assert ds.attrs["daily_mean_hours"] == "[12]"


def test_era5_store_is_opened_once_under_concurrency(monkeypatch):
    """The cache holds a live gcsfs handle and the check-then-open was not atomic, so two
    workers would both miss, both open the store, and one would silently discard the
    other's handle -- a doubled metadata fetch and an orphaned client."""
    import threading
    import time as _time

    met._ERA5_CACHE.clear()
    opens = []

    def slow_open(uri, **kw):
        # Wide enough that every other thread reaches the check while this one is inside it.
        # Unlocked, all four would miss the cache and open the store; locked, the three that
        # follow find it populated and open nothing.
        _time.sleep(0.05)
        opens.append(uri)
        return object()

    monkeypatch.setattr(met.xr, "open_zarr", slow_open)

    threads = [threading.Thread(target=met._era5_store, args=("gs://fake",))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    try:
        assert opens == ["gs://fake"]
    finally:
        met._ERA5_CACHE.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])

