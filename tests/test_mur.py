
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from coastal_sst_data.config import load_config, parse_config, AreaOfInterest, GridSpec
from coastal_sst_data.processes import mur
from coastal_sst_data import grid


def _ds(eff):
    """The settings ONE AoI runs with. `eff["ds"]` is keyed by AoI, because every product
    now resolves its options per AoI (region override -> project default). MUR's DATA options
    are global, so every AoI resolves alike for those -- take any. (`overpass_sensors` is the
    one that genuinely varies; the tests that exercise it index by name.)"""
    return next(iter(eff["ds"].values()))




EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


def test_build_eff_maps_example_config():
    """The example config maps to the expected MUR acquisition parameters."""
    eff = mur._build_eff(load_config(EXAMPLE))
    # product constants (module defaults) + config overrides
    assert _ds(eff)["short_name"] == "MUR-JPL-L4-GLOB-v4.1"
    assert _ds(eff)["variable"] == "analysed_sst"     # from the mur options
    assert _ds(eff)["pad_deg"] == 0.05                # default
    # shared project settings flow through
    assert eff["earthdata"]["auth_strategy"] == "netrc"
    assert eff["fmt"] == "netcdf"                       # default output format
    assert eff["time"] == {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    assert eff["grid"]["resolution_m"] == 100.0
    assert eff["out_dir"] == Path("path/to/data") / "MUR" / "aligned"
    # AoI geometry now lives in the shared grid (grid.py), not in eff.
    assert "aois" not in eff


def test_build_eff_requires_mur_selected(base_project):
    """Calling the adapter when mur isn't a selected product is an error."""
    base_project["products"] = {"bathymetry": None}    # drop mur (public source, no auth)
    with pytest.raises(ValueError, match="mur is not a selected product"):
        mur._build_eff(parse_config(base_project))


def test_build_eff_defaults_when_options_omitted(base_project):
    """A bare `mur:` (no options) falls back to product defaults."""
    base_project["products"]["mur"] = None             # bare -> default options
    eff = mur._build_eff(parse_config(base_project))
    assert _ds(eff)["variable"] == mur.DEFAULT_VARIABLE       # "analysed_sst"
    assert _ds(eff)["pad_deg"] == mur.DEFAULT_PAD_DEG         # 0.05
    assert _ds(eff)["short_name"] == mur.SHORT_NAME
    assert eff["fmt"] == "netcdf"
    assert eff["overwrite"] is False


def test_build_eff_applies_option_overrides(base_project):
    """mur options override the product defaults."""
    base_project["products"]["mur"] = {
        "variable": "analysed_sst_anomaly", "pad_deg": 0.2,
        "output_format": "geotiff", "overwrite": True,
    }
    eff = mur._build_eff(parse_config(base_project))
    assert _ds(eff)["variable"] == "analysed_sst_anomaly"
    assert _ds(eff)["pad_deg"] == 0.2
    assert eff["fmt"] == "geotiff"
    assert eff["overwrite"] is True


# --------------------------------------------------------------------------- #
# `overpass_sensors`: fetch only the days a sensor actually flew
# --------------------------------------------------------------------------- #
def test_build_eff_defaults_to_every_day(base_project):
    """Absent option = the old behaviour, unrestricted. This is the guard that keeps the
    feature opt-in: a default of "some sensor" would silently shrink every existing run."""
    eff = mur._build_eff(parse_config(base_project))
    assert _ds(eff)["overpass_sensors"] is None
    # The filter reads the SENSORS' trees, which hang off the output root, not MUR's own dir.
    assert eff["root"] == Path("path/to/data")


@pytest.mark.parametrize("raw,expect", [
    (["eco", "modis"], ("eco", "modis")),
    ("eco", ("eco",)),                      # a bare string is one sensor
])
def test_build_eff_reads_overpass_sensors(base_project, raw, expect):
    base_project["products"]["ecostress"] = {"versions": ["v002"]}
    base_project["products"]["modis"] = None
    base_project["products"]["mur"] = {"overpass_sensors": raw}
    eff = mur._build_eff(parse_config(base_project))
    assert _ds(eff)["overpass_sensors"] == expect


def test_build_eff_region_overrides_overpass_sensors(base_project):
    """Which sensors are worth restricting to is a REGION fact (an AoI may be ECOSTRESS-only),
    so it resolves region-then-global like every other per-AoI option."""
    base_project["products"]["ecostress"] = {"versions": ["v002"]}
    base_project["products"]["landsat"] = {"source": "pc"}
    base_project["products"]["mur"] = {"overpass_sensors": ["eco"]}
    base_project["regions"].append({
        "name": "r2",
        "sources": {"mur": {"overpass_sensors": ["lst"]}},
        "areas": [{"name": "a2", "center_lat": 46.0, "center_lon": -124.0,
                   "buffer_ns_km": 25, "buffer_ew_km": 15}],
    })
    eff = mur._build_eff(parse_config(base_project))
    assert eff["ds"]["a1"]["overpass_sensors"] == ("eco",)      # project default
    assert eff["ds"]["a2"]["overpass_sensors"] == ("lst",)      # region override


def _granule(*, ur=None, native_id=None, beg=None, end=None):
    """A stand-in for the CMR record earthaccess.search_data returns (only the two metadata
    paths `_granule_day_stamp` reads are populated)."""
    umm = {}
    if ur is not None:
        umm["GranuleUR"] = ur
    if beg or end:
        umm["TemporalExtent"] = {"RangeDateTime": {"BeginningDateTime": beg,
                                                   "EndingDateTime": end}}
    return {"umm": umm, "meta": {"native-id": native_id} if native_id else {}}


def test_granule_day_comes_from_the_nominal_stamp_not_the_range_start():
    """THE regression this feature turns on. A MUR L4 granule's temporal extent is the 24 h
    ANALYSIS WINDOW centred on 09:00Z, so `BeginningDateTime` falls on the PREVIOUS calendar
    day. Selecting on it (what modis._granule_time reads) would shift every chosen day by one
    -- and shift it silently, since a MUR file for the wrong day still opens and still writes.
    """
    g = _granule(ur="20230715090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1",
                 beg="2023-07-14T21:00:00.000Z", end="2023-07-15T21:00:00.000Z")
    assert mur._granule_day_stamp(g) == "20230715"


def test_granule_day_falls_back_to_the_range_midpoint():
    """No id to parse -> the MIDPOINT of the analysis window, which lands back on 09:00Z of
    the right day. (The start would not.)"""
    g = _granule(beg="2023-07-14T21:00:00.000Z", end="2023-07-15T21:00:00.000Z")
    assert mur._granule_day_stamp(g) == "20230715"


def test_granule_day_is_none_when_nothing_is_readable():
    assert mur._granule_day_stamp(_granule()) is None


def test_select_granules_without_a_day_set_is_a_no_op():
    """The default path must not depend on granule metadata being readable at all."""
    granules = [_granule(), _granule(ur="20260601090000-JPL")]
    assert mur._select_granules(granules, None) == granules


def test_select_granules_keeps_an_unidentifiable_granule(caplog):
    """Fail-OPEN on identification: the worst case is downloading a day we would have
    downloaded anyway, whereas dropping it would silently delete a day from the record."""
    good = _granule(ur="20260601090000-JPL")
    mystery = _granule()
    with caplog.at_level("WARNING"):
        kept = mur._select_granules([good, mystery], {"20260601"})
    assert kept == [good, mystery]
    assert "no readable day" in caplog.text


# --------------------------------------------------------------------------- #
# run() end to end (the granule loop had no test before this)
# --------------------------------------------------------------------------- #
DAYS = ["2026-06-01", "2026-06-02", "2026-06-03"]


def _mur_eff(tmp_path, g, **over):
    cfg = {"short_name": "MUR-JPL-L4-GLOB-v4.1", "variable": "analysed_sst",
           "pad_deg": 0.05, "overpass_sensors": None} | over.pop("ds", {})
    return {
        "config_sha256": "x",
        "ds": {g.name: cfg},
        "grid": {"resampling_continuous": "bilinear", "to_celsius": False},
        "root": tmp_path,
        "out_dir": tmp_path / "MUR" / "aligned",
        "fmt": "netcdf", "overwrite": False,
        "earthdata": {"auth_strategy": "netrc"},
        "time": {"start_date": DAYS[0], "end_date": DAYS[-1]},
    } | over


@pytest.fixture
def three_days(monkeypatch, tmp_path, aoi_grid):
    """Three daily MUR granules on disk + a CMR/earthaccess stand-in that serves them.

    `open` resolves the granule through the same metadata the filter selects on, so a test
    asserting on the FILES written is also asserting the two agree.
    """
    monkeypatch.setattr(mur.earthaccess, "login", lambda **kw: None)
    made = {d.replace("-", ""): make_mur_granule(tmp_path / f"src_{d}.nc", aoi_grid,
                                                 when=d, sst_kelvin=290.0)
            for d in DAYS}
    granules = [_granule(ur=f"{s}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1")
                for s in sorted(made)]
    monkeypatch.setattr(mur.earthaccess, "search_data", lambda **kw: list(granules))
    monkeypatch.setattr(mur.earthaccess, "open",
                        lambda gs: [made[mur._granule_day_stamp(gs[0])]])
    return made


def _written(tmp_path, aoi):
    return sorted(p.name for p in (tmp_path / "MUR" / "aligned" / aoi).glob("*.nc"))


def test_run_without_the_option_still_fetches_every_day(three_days, tmp_path, aoi_grid):
    """The default-unchanged guard: no `overpass_sensors`, no sensor trees, all three days."""
    rep = mur.run(_mur_eff(tmp_path, aoi_grid), {aoi_grid.name: aoi_grid}, None, False)

    assert _written(tmp_path, aoi_grid.name) == [f"{aoi_grid.name}_2026060{i}.nc"
                                                 for i in (1, 2, 3)]
    assert (rep.expected, rep.written) == (3, 3)


def test_run_fetches_only_the_days_a_sensor_flew(three_days, tmp_path, aoi_grid):
    scenes = tmp_path / "LANDSAT" / "aligned" / aoi_grid.name
    scenes.mkdir(parents=True)
    (scenes / f"{aoi_grid.name}_20260602T180000.nc").write_bytes(b"")

    eff = _mur_eff(tmp_path, aoi_grid, ds={"overpass_sensors": ("lst",)})
    rep = mur.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert _written(tmp_path, aoi_grid.name) == [f"{aoi_grid.name}_20260602.nc"]
    # `expected` is what the stage MEANT to produce -- so it counts the FILTERED days. The
    # unfiltered count would make every filtered run read as having lost two days.
    assert (rep.expected, rep.written) == (1, 1)


def test_run_raises_when_the_named_sensors_never_ran(three_days, tmp_path, aoi_grid):
    """Downloading nothing and reporting success is the failure mode this feature invites.
    pipeline.run_pipeline catches this and records `mur: failed: ...` without aborting the
    other products, so the raise is loud AND non-fatal."""
    eff = _mur_eff(tmp_path, aoi_grid, ds={"overpass_sensors": ("lst",)})
    with pytest.raises(ValueError, match="have not run in this output dir"):
        mur.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert not (tmp_path / "MUR").exists()


def test_dry_run_on_a_fresh_tree_warns_instead_of_raising(three_days, tmp_path, aoi_grid,
                                                          caplog):
    """The sensors' own dry run writes nothing, so on a fresh output dir EVERY --dry-run
    would otherwise report MUR failed."""
    eff = _mur_eff(tmp_path, aoi_grid, ds={"overpass_sensors": ("lst",)})
    with caplog.at_level("WARNING"):
        mur.run(eff, {aoi_grid.name: aoi_grid}, None, True)       # must not raise
    assert "have not run in this output dir" in caplog.text
    # The per-AoI warning that follows must not then claim the trees exist.
    assert "fact about the data" not in caplog.text


def test_dry_run_previews_the_filtered_count(three_days, tmp_path, aoi_grid, caplog):
    """A preview of a run that isn't going to happen is worse than no preview."""
    scenes = tmp_path / "LANDSAT" / "aligned" / aoi_grid.name
    scenes.mkdir(parents=True)
    (scenes / f"{aoi_grid.name}_20260602T180000.nc").write_bytes(b"")

    eff = _mur_eff(tmp_path, aoi_grid, ds={"overpass_sensors": ("lst",)})
    with caplog.at_level("INFO"):
        mur.run(eff, {aoi_grid.name: aoi_grid}, None, True)

    assert "would process 1 day(s) of 3" in caplog.text
    assert not (tmp_path / "MUR").exists()


def test_run_skips_an_aoi_whose_sensors_flew_on_no_day(three_days, tmp_path, aoi_grid,
                                                       caplog):
    """The tree exists but holds no scene for THIS AoI -- a fact about the data, not a stage
    that was skipped, so it is a warning and the run continues."""
    (tmp_path / "LANDSAT" / "aligned" / "somewhere_else").mkdir(parents=True)

    eff = _mur_eff(tmp_path, aoi_grid, ds={"overpass_sensors": ("lst",)})
    with caplog.at_level("WARNING"):
        rep = mur.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.written == 0
    assert "no overpass days" in rep.note
    assert "recorded no scene on ANY day" in caplog.text
    # ...and it says WHICH of the two situations this is: data, not a stage that never ran.
    assert "fact about the data" in caplog.text


@pytest.fixture
def aoi_grid():
    """A small AoI's shared grid to subset/reproject onto."""
    area = AreaOfInterest(name="test_aoi", center_lat=45.52, center_lon=-123.925,
                          buffer_ns_km=8.0, buffer_ew_km=8.0)
    return grid.compute_aoi_grid(area, GridSpec())


def make_mur_granule(path, g, *, sst_kelvin=290.0, when="2026-06-15",
                     res_deg=0.01, margin_deg=0.3, variable="analysed_sst"):
    """Write a synthetic daily MUR NetCDF for one 'day' over grid `g`.

    `variable` (Kelvin) on an ascending lat/lon grid at ~1 km (0.01 deg), like
    real MUR, spanning g.search_bbox + a margin. dims (time, lat, lon). Ascending
    lat/lon so xarray's .sel(lat=slice(...)) works; margin > the search pad so the
    subset has data on every edge. Returns the path (str).
    """
    w, s, e, n = g.search_bbox
    lat = np.arange(s - margin_deg, n + margin_deg, res_deg, dtype="float64")
    lon = np.arange(w - margin_deg, e + margin_deg, res_deg, dtype="float64")
    sst = np.full((1, lat.size, lon.size), sst_kelvin, dtype="float32")
    ds = xr.Dataset(
        {variable: (("time", "lat", "lon"), sst)},
        coords={"time": [np.datetime64(when)], "lat": lat, "lon": lon},
    )
    ds[variable].attrs["units"] = "kelvin"
    ds.to_netcdf(path)          # NETCDF4/HDF5 -> the h5netcdf reader opens it
    return str(path)


def test_subset_and_reproject(tmp_path, aoi_grid):
    g = aoi_grid
    path = make_mur_granule(tmp_path / "mur_20260615.nc", g, sst_kelvin=290.0)
    grid_cfg = {"resampling_continuous": "bilinear", "to_celsius": False}
    out, t = mur.subset_and_reproject(
        path, "analysed_sst", g.search_bbox, 0.05,
        g.target_crs, g.transform, g.width, g.height, g.geom_proj, grid_cfg)
    # lands on the shared grid
    assert out.rio.shape == (g.height, g.width)
    assert str(out.rio.crs) == g.target_crs
    assert out.rio.transform() == g.transform
    # data came through, constant preserved; timestamp parsed from the granule
    finite = out.values[np.isfinite(out.values)]
    assert finite.size and finite.min() == pytest.approx(290.0, abs=1e-3)
    assert t == pd.Timestamp("2026-06-15")


def test_subset_and_reproject_to_celsius(tmp_path, aoi_grid):
    g = aoi_grid
    path = make_mur_granule(tmp_path / "mur.nc", g, sst_kelvin=290.0)
    out, _ = mur.subset_and_reproject(
        path, "analysed_sst", g.search_bbox, 0.05,
        g.target_crs, g.transform, g.width, g.height, g.geom_proj,
        {"resampling_continuous": "bilinear", "to_celsius": True})
    finite = out.values[np.isfinite(out.values)]
    assert finite.min() == pytest.approx(290.0 - 273.15, abs=1e-3)   # 16.85

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])