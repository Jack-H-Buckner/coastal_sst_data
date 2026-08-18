"""MODIS (the standalone Terra+Aqua sensor): config -> params, the local-solar-time
day/night filter, the Harmony subsetting backend, and the swath read + pyresample nearest
regrid onto the shared AoiGrid. All offline (no earthaccess, no Harmony). The swath tests
write a synthetic MODIS L2P NetCDF (root-level sst/quality/lat/lon 2D swath) and run the
real read + resample.

The Landsat-coincident half of the old MODIS product now lives in `processes.modis_ref`
and is tested in test_modis_ref.py."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data.processes import modis
from coastal_sst_data import auth
from .conftest import UniformDs


def _ds(eff):
    """The settings ONE AoI runs with. `eff["ds"]` is keyed by AoI, because every product
    now resolves its options per AoI (region override -> project default). This is a global
    product with no region-varying options, so every AoI resolves alike -- take any."""
    return next(iter(eff["ds"].values()))


# --- fake earthaccess granule (dict-like: what the selectors read) ----------- #
def _granule(iso, night=False, tagged=True):
    nid = ("X-N-Y" if night else "X-D-Y") if tagged else "X-untagged-Y"
    return {"umm": {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": iso}}},
            "meta": {"native-id": nid,
                     "concept-id": "G123-POCLOUD",
                     "collection-concept-id": "C456-POCLOUD"}}


# --- synthetic MODIS L2P swath granule -------------------------------------- #
def write_modis_granule(path, bbox, *, sst_kelvin=290.0, quality=5, res_deg=0.01):
    """A minimal MODIS L2P NetCDF: sea_surface_temperature/quality_level on a 2D
    swath, with lat/lon covering `bbox` + margin. quality 5=best, 1=bad."""
    w, s, e, n = bbox
    lon2d, lat2d = np.meshgrid(np.arange(w - 0.1, e + 0.1, res_deg),
                               np.arange(s - 0.1, n + 0.1, res_deg))
    shape = (1,) + lon2d.shape
    ds = xr.Dataset(
        {"sea_surface_temperature": (("time", "nj", "ni"),
                                     np.full(shape, sst_kelvin, dtype="float32")),
         "quality_level": (("time", "nj", "ni"), np.full(shape, quality, dtype="int8"))},
        coords={"time": [np.datetime64("2023-08-15T18:56:19")],
                "lat": (("nj", "ni"), lat2d), "lon": (("nj", "ni"), lon2d)},
    )
    ds.to_netcdf(path, engine="netcdf4")
    return str(path)


# ---------------------------------------------------------------------------
# _build_eff: config -> acquisition params (modis needs earthdata auth).
# ---------------------------------------------------------------------------
def test_build_eff_maps_defaults(base_project):
    base_project["products"]["modis"] = None       # bare -> defaults
    eff = modis._build_eff(parse_config(base_project))
    assert _ds(eff)["platforms"] == ["terra", "aqua"]
    assert _ds(eff)["short_name"] is None          # unset -> each platform's own collection
    assert _ds(eff)["variable"] == modis.DEFAULT_VARIABLE
    assert _ds(eff)["quality_min"] == 4
    # Server-side subsetting is the default: a night Terra+Aqua series is tens of thousands
    # of ~20 MB granules, and this module reads four of their ~15 variables.
    assert _ds(eff)["access"] == "harmony"
    assert _ds(eff)["time_of_day"] == "night"
    assert _ds(eff)["night_solar_hours"] == (19.0, 5.0)
    assert eff["earthdata"]["auth_strategy"] == "netrc"
    # STACKED: the per-platform trees hang under this root (MODIS/<tag>/aligned/<aoi>).
    assert eff["modis_root"] == Path("path/to/data") / "MODIS"
    assert "landsat_dir" not in eff               # standalone -- no Landsat coupling


def test_build_eff_requires_modis_selected(base_project):
    with pytest.raises(ValueError, match="modis is not a selected product"):
        modis._build_eff(parse_config(base_project))


def test_build_eff_applies_overrides(base_project):
    base_project["products"]["modis"] = {
        "platforms": ["aqua"], "quality_min": 5, "regrid_radius_m": 1000,
        "access": "download", "time_of_day": "day", "night_solar_hours": [20, 4],
        "output_format": "geotiff",
    }
    eff = modis._build_eff(parse_config(base_project))
    assert _ds(eff)["platforms"] == ["aqua"]
    assert _ds(eff)["quality_min"] == 5
    assert _ds(eff)["regrid_radius_m"] == 1000
    assert _ds(eff)["access"] == "download"
    assert _ds(eff)["time_of_day"] == "day"
    assert _ds(eff)["night_solar_hours"] == (20.0, 4.0)
    assert eff["fmt"] == "geotiff"


def test_build_eff_rejects_unknown_access(base_project):
    base_project["products"]["modis"] = {"access": "ftp"}
    with pytest.raises(ValueError, match="not recognized"):
        modis._build_eff(parse_config(base_project))


def test_build_eff_rejects_unknown_time_of_day(base_project):
    base_project["products"]["modis"] = {"time_of_day": "dusk"}
    with pytest.raises(ValueError, match="time_of_day"):
        modis._build_eff(parse_config(base_project))


def test_build_eff_rejects_malformed_night_window(base_project):
    base_project["products"]["modis"] = {"night_solar_hours": [19, 19]}
    with pytest.raises(ValueError, match="empty window"):
        modis._build_eff(parse_config(base_project))
    base_project["products"]["modis"] = {"night_solar_hours": [19, 30]}
    with pytest.raises(ValueError, match=r"\[0, 24\)"):
        modis._build_eff(parse_config(base_project))


def test_unknown_platform_fails_loudly(base_project):
    """A platform tag no code recognises would silently drop a whole satellite. Because
    `platforms` is the spec's `sources_option`, config validation already rejects it at LOAD
    time -- before a multi-hour run starts, not after it quietly acquired half the data."""
    base_project["products"]["modis"] = {"platforms": ["terra", "seastar"]}
    with pytest.raises(Exception, match="unknown source 'seastar'"):
        parse_config(base_project)


def test_daytime_only_still_understood(base_project):
    """The option `time_of_day` replaced. Silently ignoring it would not error -- it would
    change WHICH granules an existing config selects, which looks like the archive changing."""
    base_project["products"]["modis"] = {"daytime_only": True}
    assert _ds(modis._build_eff(parse_config(base_project)))["time_of_day"] == "day"
    base_project["products"]["modis"] = {"daytime_only": False}
    assert _ds(modis._build_eff(parse_config(base_project)))["time_of_day"] == "both"


# ---------------------------------------------------------------------------
# solar_hour: the drift-proof basis for the day/night decision.
# ---------------------------------------------------------------------------
_PUGET_LON = -122.5


@pytest.mark.parametrize("utc,expected", [
    ("2015-07-15T06:30:00", 22.33),   # Terra night   (nominal 22:30 local)
    ("2015-07-15T09:30:00", 1.33),    # Aqua night    (nominal 01:30 local)
    ("2015-07-15T18:45:00", 10.58),   # Terra day     (nominal 10:30 local)
    ("2015-07-15T21:30:00", 13.33),   # Aqua day      (nominal 13:30 local)
])
def test_solar_hour_lands_on_the_known_crossings(utc, expected):
    got = modis.solar_hour(datetime.fromisoformat(utc), _PUGET_LON)
    assert got == pytest.approx(expected, abs=0.02)


def test_solar_hour_agrees_with_the_met_stage():
    """`met.reference_time_utc` converts solar -> UTC; this converts UTC -> solar. They are
    the same convention or the cube's "10:30 local" forcing and its sensor selection disagree."""
    from coastal_sst_data.processes import met
    # lon=-120 and 11:00 solar land on an exact UTC hour, so `reference_time_utc`'s round-to-
    # the-hour (both forcing models are hourly) does not blur the round-trip being checked.
    lon = -120.0
    utc = met.reference_time_utc(pd.Timestamp("2023-07-15"), lon, 11.0, "solar")
    assert utc.hour == 19                                     # 11:00 local = 19:00 UTC
    assert modis.solar_hour(utc.to_pydatetime(), lon) == pytest.approx(11.0, abs=1e-9)


def test_night_window_wraps_through_midnight():
    lo, hi = modis.DEFAULT_NIGHT_SOLAR_HOURS         # (19, 5)
    assert modis._in_window(22.5, lo, hi)            # before midnight
    assert modis._in_window(1.5, lo, hi)             # after midnight
    assert not modis._in_window(10.5, lo, hi)        # mid-morning
    assert not modis._in_window(13.5, lo, hi)


# ---------------------------------------------------------------------------
# select_by_time_of_day: what actually gets acquired.
# ---------------------------------------------------------------------------
_NIGHT = modis.DEFAULT_NIGHT_SOLAR_HOURS


def _sel(granules, tod):
    return modis.select_by_time_of_day(granules, time_of_day=tod,
                                       night_hours=_NIGHT, lon=_PUGET_LON)


def test_night_keeps_the_night_overpasses():
    g = [_granule("2015-07-15T06:30:00.000000Z", night=True),    # 22:20 local
         _granule("2015-07-15T09:30:00.000000Z", night=True)]    # 01:20 local
    assert len(_sel(g, "night")) == 2


def test_night_drops_the_day_overpasses():
    g = [_granule("2015-07-15T18:45:00.000000Z"),                # 10:35 local
         _granule("2015-07-15T21:30:00.000000Z")]                # 13:20 local
    assert _sel(g, "night") == []


def test_day_is_the_complement_of_the_night_window():
    g = [_granule("2015-07-15T18:45:00.000000Z"),
         _granule("2015-07-15T06:30:00.000000Z", night=True)]
    kept = _sel(g, "day")
    assert len(kept) == 1
    assert kept[0][1].hour == 18


def test_both_filters_nothing():
    g = [_granule("2015-07-15T18:45:00.000000Z"),
         _granule("2015-07-15T06:30:00.000000Z", night=True)]
    assert len(_sel(g, "both")) == 2


def test_drift_does_not_lose_a_late_aqua_night_overpass():
    """THE REASON THIS FILTER IS SOLAR-TIME BASED. Aqua's night crossing drifted from 01:30
    local at launch to ~03:50 by end of mission. A fixed clock around 01:30 -- or a UTC
    window fitted to the early record -- silently drops the last years of data."""
    early = _granule("2005-07-15T09:30:00.000000Z", night=True)   # 01:20 local
    late = _granule("2026-07-15T11:50:00.000000Z", night=True)    # 03:40 local
    assert len(_sel([early, late], "night")) == 2


def test_a_granule_tagged_against_the_solar_decision_is_dropped():
    """The `-D-`/`-N-` token is OBPG's own classification. Where it contradicts the window,
    the granule straddles the terminator and keeping it would make "night" mean two things."""
    g = [_granule("2015-07-15T06:30:00.000000Z", night=False)]    # 22:20 local, tagged DAY
    assert _sel(g, "night") == []


def test_an_untagged_granule_is_judged_on_solar_time_alone():
    g = [_granule("2015-07-15T06:30:00.000000Z", tagged=False)]   # 22:20 local
    assert len(_sel(g, "night")) == 1


def test_select_returns_sorted_by_time():
    g = [_granule("2015-07-16T06:30:00.000000Z", night=True),
         _granule("2015-07-15T06:30:00.000000Z", night=True)]
    kept = _sel(g, "night")
    assert kept[0][1] < kept[1][1]


# ---------------------------------------------------------------------------
# read_swath: quality masking + Kelvin/Celsius (real synthetic granule).
# ---------------------------------------------------------------------------
def test_read_swath_good_quality_kelvin(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, sst_kelvin=290.0, quality=5)
    sst, lat, lon = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    finite = sst[np.isfinite(sst)]
    assert finite.size and np.allclose(finite, 290.0)
    assert lat.ndim == 2 and lon.ndim == 2                  # 2D swath coords


def test_read_swath_masks_low_quality(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, quality=1)  # bad
    sst, _, _ = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    assert np.isnan(sst).all()                              # all below quality_min -> NaN


def test_read_swath_to_celsius(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, sst_kelvin=290.0)
    sst, _, _ = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, True)
    assert np.allclose(sst[np.isfinite(sst)], 290.0 - 273.15)


def test_harmony_variables_are_exactly_what_read_swath_opens():
    """The request and the reader must not drift: a subset missing `quality_level` would
    read as a granule with no quality information rather than as a bad request."""
    want = set(modis.harmony_variables(modis.DEFAULT_VARIABLE))
    assert want == {modis.DEFAULT_VARIABLE, modis.QUALITY_VAR, "lat", "lon"}


# ---------------------------------------------------------------------------
# resample_to_grid + _scene_dataset: swath -> shared grid, output schema.
# ---------------------------------------------------------------------------
def test_swath_resamples_onto_shared_grid(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox, sst_kelvin=290.0)
    sst, lat, lon = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    fp = np.arange(sst.size, dtype="int32").reshape(sst.shape)
    sst_g, fp_g = modis.resample_to_grid(sst, lat, lon, aoi_grid, 1500.0, fp)
    assert sst_g.shape == aoi_grid.shape
    finite = sst_g[np.isfinite(sst_g)]
    assert finite.size and np.allclose(finite, 290.0, atol=0.5)   # nearest preserves value
    assert (fp_g >= 0).any()                                       # footprint ids assigned


def test_scene_dataset_schema(tmp_path, aoi_grid):
    p = write_modis_granule(tmp_path / "m.nc", aoi_grid.search_bbox)
    sst, lat, lon = modis.read_swath(p, modis.DEFAULT_VARIABLE, 4, False)
    sst_g, _ = modis.resample_to_grid(sst, lat, lon, aoi_grid, 1500.0)
    ds = modis._scene_dataset(sst_g, None, aoi_grid,
                              pd.Timestamp("2023-08-15T18:56:19"), "test_aoi", False)
    assert {"sst", "valid"} <= set(ds.data_vars)
    assert "footprint_id" not in ds.data_vars      # standalone MODIS publishes none
    assert ds["sst"].attrs["units"] == "K"
    assert ds.sizes["y"] == aoi_grid.height and ds.sizes["x"] == aoi_grid.width
    assert "time" in ds.coords


def test_scene_records_the_local_solar_hour(aoi_grid):
    """The cube only carries the UTC hour, and the point of a 26-year MODIS series is that
    the local hour behind it MOVED. The drift has to be auditable from the granules."""
    g = aoi_grid
    sst = np.full((g.height, g.width), 290.0, "float32")
    t = datetime(2015, 7, 15, 6, 30)
    ds = modis._scene_dataset(sst, None, g, t, "aoi", False, short_name="X",
                              platform="terra", solar_h=modis.solar_hour(t, g.search_bbox[0]),
                              day_night="night")
    assert ds.attrs["platform"] == "terra"
    assert ds.attrs["day_night"] == "night"
    assert 19.0 <= ds.attrs["solar_hour"] < 24.0


def test_source_attr_names_the_configured_sensor(aoi_grid):
    """Configure Aqua and every file used to still claim Terra, because the attr was built
    from the module constant rather than the short_name the search actually used."""
    g = aoi_grid
    sst = np.full((g.height, g.width), 290.0, "float32")

    terra = modis._scene_dataset(sst, None, g, datetime(2023, 7, 15, 21, 0), "aoi", False,
                                 short_name=modis.PLATFORMS["terra"])
    assert terra.attrs["source"] == f"GHRSST {modis.PLATFORMS['terra']}"

    aqua = modis._scene_dataset(sst, None, g, datetime(2023, 7, 15, 21, 0), "aoi", False,
                                short_name=modis.PLATFORMS["aqua"])
    assert aqua.attrs["source"] == "GHRSST MODIS_A-JPL-L2P-v2019.0"       # follows config
    assert "MODIS_T" not in aqua.attrs["source"]                          # NOT Terra


# --------------------------------------------------------------------------- #
# The Harmony backend. Never contacts Harmony: the point is the REQUEST shape and the
# failure handling, both of which are what a live run would depend on.
# --------------------------------------------------------------------------- #
class _FakeFuture:
    def __init__(self, v):
        self._v = v

    def result(self):
        return self._v


def _install_fake_harmony(monkeypatch, *, files, capture=None):
    """A stand-in `harmony` module, injected as an import target for _fetch_harmony."""
    import sys
    import types

    mod = types.ModuleType("harmony")

    class BBox(tuple):
        def __new__(cls, w, s, e, n):
            return super().__new__(cls, (w, s, e, n))

    class Collection:
        def __init__(self, id):
            self.id = id

    class Request:
        def __init__(self, collection, **kw):
            self.collection, self.kw = collection, kw
            if capture is not None:
                capture.append(self)

        def is_valid(self):
            return True

        def error_messages(self):
            return []

    class Client:
        def __init__(self, **kw):
            self.kw = kw

        def submit(self, request):
            return "job-1"

        def download_all(self, job, directory=None, overwrite=False):
            for name in files:
                p = Path(directory) / name
                p.write_bytes(b"subset")
                yield _FakeFuture(str(p))

    mod.BBox, mod.Collection, mod.Request, mod.Client = BBox, Collection, Request, Client
    monkeypatch.setitem(sys.modules, "harmony", mod)
    return mod


def test_harmony_request_carries_the_granule_bbox_and_variables(monkeypatch, tmp_path):
    captured = []
    _install_fake_harmony(monkeypatch, files=["subset.nc"], capture=captured)
    monkeypatch.setattr(auth, "earthdata_client_auth", lambda: {"token": "t"})

    gr = _granule("2015-07-15T06:30:00.000000Z", night=True)
    want = modis.harmony_variables(modis.DEFAULT_VARIABLE)
    out = modis._fetch_harmony(gr, (-123.0, 47.0, -122.0, 48.0), tmp_path / "g",
                               variables=want)

    assert out.exists()
    req = captured[0]
    # Addressed by the concept-ids the earthaccess search ALREADY returned -- no second
    # catalogue round-trip to find the granule again.
    assert req.collection.id == "C456-POCLOUD"
    assert req.kw["granule_id"] == ["G123-POCLOUD"]
    assert tuple(req.kw["spatial"]) == (-123.0, 47.0, -122.0, 48.0)   # BBox is (w, s, e, n)
    assert list(req.kw["variables"]) == list(want)


def test_harmony_refuses_a_granule_with_no_concept_ids(monkeypatch, tmp_path):
    _install_fake_harmony(monkeypatch, files=["subset.nc"])
    gr = {"umm": {"TemporalExtent": {"RangeDateTime": {
              "BeginningDateTime": "2015-07-15T06:30:00.000000Z"}}},
          "meta": {"native-id": "X-N-Y"}}
    with pytest.raises(RuntimeError, match="concept-ids"):
        modis._fetch_harmony(gr, (-123.0, 47.0, -122.0, 48.0), tmp_path / "g")


def test_harmony_turns_a_failed_job_into_an_error(monkeypatch, tmp_path):
    """`download_all` SWALLOWS a failed job -- it prints to stderr and yields nothing. Without
    this check the granule would silently produce no file and be reported as a success."""
    _install_fake_harmony(monkeypatch, files=[])          # a job that yields nothing
    monkeypatch.setattr(auth, "earthdata_client_auth", lambda: {})
    gr = _granule("2015-07-15T06:30:00.000000Z", night=True)
    with pytest.raises(RuntimeError, match="no file"):
        modis._fetch_harmony(gr, (-123.0, 47.0, -122.0, 48.0), tmp_path / "g")


# --------------------------------------------------------------------------- #
# The tmp-granule lifecycle. `earthaccess.download` skips a file that already exists BY
# NAME, so a truncated granule left by a killed run was handed straight back to the reader.
# --------------------------------------------------------------------------- #
class _Gran:
    """A night granule over Puget Sound: 06:30 UTC is 22:20 local solar."""

    def __init__(self, t="2023-07-15T06:30:00.000Z"):
        self._t = t

    def __getitem__(self, k):
        if k == "umm":
            return {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": self._t}}}
        return {"native-id": "MODIS-N-granule", "concept-id": "G1",
                "collection-concept-id": "C1"}


def _modis_eff(tmp_path, **over):
    ds = {"platforms": ["terra"], "short_name": modis.PLATFORMS["terra"],
          "variable": modis.DEFAULT_VARIABLE, "quality_min": 4,
          "regrid_radius_m": 1500.0, "access": "download",
          "time_of_day": "night", "night_solar_hours": modis.DEFAULT_NIGHT_SOLAR_HOURS}
    ds.update(over)
    return {
        "ds": UniformDs(ds),
        "grid": {"to_celsius": False},
        "modis_root": tmp_path / "MODIS",
        "fmt": "netcdf", "overwrite": False,
        "earthdata": {"auth_strategy": "netrc"},
        "time": {"start_date": "2023-07-15", "end_date": "2023-07-15"},
        "config_sha256": "x",
    }


def test_a_failed_granule_leaves_no_tmp_file_behind(monkeypatch, tmp_path, aoi_grid):
    """A partial download must not survive the attempt: the next run would see the name
    already present, skip the download, and read the truncated file."""
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis.earthaccess, "search_data", lambda **kw: [_Gran()])

    def fetch_then_die(granule, bbox, tmp_dir, *, variables=None):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        p = tmp_dir / "granule.nc"
        p.write_bytes(b"half a granule")          # a partial download, as a kill would leave
        raise ConnectionError("connection reset mid-download")

    monkeypatch.setitem(modis._ACCESS, "download", fetch_then_die)

    eff = _modis_eff(tmp_path)
    rep = modis.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.failed == 1                        # counted as a LOSS, not a silent skip
    leftovers = list((tmp_path / "MODIS").rglob("*.nc"))
    assert leftovers == []                        # ...and nothing truncated survives


def test_a_successful_granule_also_cleans_up_its_scratch(monkeypatch, tmp_path, aoi_grid):
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis.earthaccess, "search_data", lambda **kw: [_Gran()])

    src = write_modis_granule(tmp_path / "src.nc", aoi_grid.search_bbox,
                              sst_kelvin=290.0, quality=5)

    def fetch(granule, bbox, tmp_dir, *, variables=None):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / "granule.nc"
        dst.write_bytes(Path(src).read_bytes())
        return dst

    monkeypatch.setitem(modis._ACCESS, "download", fetch)

    eff = _modis_eff(tmp_path)
    rep = modis.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.written == 1 and rep.failed == 0
    # The granule landed in its PLATFORM tree, which is what makes it a stacked cube channel.
    written = list((tmp_path / "MODIS" / "terra" / "aligned").rglob("*.nc"))
    assert len(written) == 1
    assert list((tmp_path / "MODIS" / "terra" / "_tmp").rglob("*.nc")) == []


def test_scratch_never_sits_beside_the_platform_tags(monkeypatch, tmp_path, aoi_grid):
    """A direct child of MODIS/ is a candidate SOURCE TAG to the assembler. A scratch dir
    there would be loaded as a platform and emit an entire all-NaN channel set
    (`modis_sst__tmp`) -- see docs/bug-empty-version-tag-channels.md, Defect A."""
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis.earthaccess, "search_data", lambda **kw: [_Gran()])

    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis.earthaccess, "search_data", lambda **kw: [_Gran()])

    src = write_modis_granule(tmp_path / "src.nc", aoi_grid.search_bbox, quality=5)
    seen = {}

    def fetch(granule, bbox, tmp_dir, *, variables=None):
        seen["tmp"] = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / "granule.nc"
        dst.write_bytes(Path(src).read_bytes())
        return dst

    monkeypatch.setitem(modis._ACCESS, "download", fetch)
    modis.run(_modis_eff(tmp_path), {aoi_grid.name: aoi_grid}, None, False)

    root = tmp_path / "MODIS"
    assert seen["tmp"].is_relative_to(root / "terra")
    # The ONLY direct child of MODIS/ is a real platform tag. This is the assertion the
    # assembler's tag discovery depends on.
    assert {d.name for d in root.iterdir() if d.is_dir()} == {"terra"}


def test_both_platforms_write_separate_trees(monkeypatch, tmp_path, aoi_grid):
    """Terra and Aqua never observe the same overpass, so they stack rather than merge."""
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: None)
    monkeypatch.setattr(modis.earthaccess, "search_data", lambda **kw: [_Gran()])

    src = write_modis_granule(tmp_path / "src.nc", aoi_grid.search_bbox, quality=5)

    def fetch(granule, bbox, tmp_dir, *, variables=None):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / "granule.nc"
        dst.write_bytes(Path(src).read_bytes())
        return dst

    monkeypatch.setitem(modis._ACCESS, "download", fetch)

    eff = _modis_eff(tmp_path, platforms=["terra", "aqua"], short_name=None)
    rep = modis.run(eff, {aoi_grid.name: aoi_grid}, None, False)

    assert rep.written == 2
    for tag in ("terra", "aqua"):
        files = list((tmp_path / "MODIS" / tag / "aligned").rglob("*.nc"))
        assert len(files) == 1, tag
        with xr.open_dataset(files[0]) as ds:
            # Each file names the collection ITS platform was searched with.
            assert ds.attrs["source"] == f"GHRSST {modis.PLATFORMS[tag]}"
            assert ds.attrs["platform"] == tag


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
