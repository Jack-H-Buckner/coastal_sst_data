"""Copernicus Marine In-Situ TAC as an in-situ source.

All network goes through the module's THREE seams -- `_catalogue_root`, `_fetch_index` and
`_download` -- monkeypatched with table-driven fakes, so these run offline and never import
`copernicusmarine`. That is the `test_cmems` pattern (replace the one function that talks to the
toolbox), not the `test_insitu` pattern (replace a low-level `_get`), because the toolbox is an
optional dependency and the format, not the transport, is what has traps in it.

The synthetic fixtures below mirror a REAL file, dumped from `GL_TS_MO_46211.nc`: scalar
LATITUDE/LONGITUDE, `DEPH` (not `DEPTH`) carrying two levels, `TEMP` on (TIME, DEPTH), and
`TEMP_QC` arriving as FLOAT32 rather than the int8 it is stored as.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data import grid, products
from coastal_sst_data.config import DataProduct, parse_config
from coastal_sst_data.processes import insitu_acquire, insitu_cmems

AOI = "aoi1"
LAT, LON = 46.92, -124.10

# The real catalog's five comment lines, then the `# `-prefixed header. Reproduced verbatim in
# shape because `parse_index` skips exactly five lines and strips exactly that prefix.
_INDEX_HEAD = (
    "# Title : in-situ files catalog\n"
    "# Description : catalog of available in-situ files\n"
    "# Project : Copernicus Marine In Situ TAC\n"
    "# Format version : 3.0\n"
    "# Date of update : 2026-08-24T20:34:16Z\n"
    "# product_id,file_name,geospatial_lat_min,geospatial_lat_max,geospatial_lon_min,"
    "geospatial_lon_max,time_coverage_start,time_coverage_end,institution,date_update,"
    "data_mode,parameters\n")

_PREFIX = "INSITU_GLO/cmems_x_202311/history"


def _codes(sel):
    """The platform codes of a selection -- the last `_`-separated field of each filename."""
    return {f.rsplit("/", 1)[-1].removesuffix(".nc").split("_")[-1] for f in sel["file_name"]}


def _row(name, dtype, lat0, lat1, lon0, lon1, t0, t1, params="TEMP PSAL"):
    return (f"COP-X,{_PREFIX}/{dtype}/GL_TS_{dtype}_{name}.nc,"
            f"{lat0},{lat1},{lon0},{lon1},{t0},{t1},Inst,2026-01-01T00:00:00Z,R,{params}\n")


def make_index(rows) -> bytes:
    return (_INDEX_HEAD + "".join(rows)).encode()


# A mooring squarely inside the AoI; a mooring whose bounds are far WIDER than the AoI (the
# containment-vs-intersection case); a glider and a ship track (mobile); a mooring with no TEMP;
# and a mooring outside the window.
DEFAULT_ROWS = [
    _row("MOORED", "MO", 46.90, 46.94, -124.12, -124.08,
         "2004-09-08T16:00:00Z", "2026-07-31T23:56:00Z"),
    _row("WIDE", "MO", 40.0, 50.0, -130.0, -120.0,
         "1987-05-06T00:00:00Z", "1987-11-30T00:00:00Z"),
    _row("GLIDER", "GL", 46.90, 46.94, -124.12, -124.08,
         "2010-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    _row("SHIP", "TS", 46.90, 46.94, -124.12, -124.08,
         "2010-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    _row("NOTEMP", "MO", 46.90, 46.94, -124.12, -124.08,
         "2010-01-01T00:00:00Z", "2020-01-01T00:00:00Z", params="PSAL VHM0"),
    _row("ELSEWHEN", "MO", 46.90, 46.94, -124.12, -124.08,
         "1950-01-01T00:00:00Z", "1955-01-01T00:00:00Z"),
]


def write_platform_nc(path: Path, *, n=48, lat=46.92, lon=-124.10, depths=(0.0, 0.46),
                      temp=12.0, qc=1.0, name="Grays Harbor", moving=False):
    """A synthetic In-Situ per-platform file with the REAL structure.

    `TEMP_QC` is written as float on purpose -- that is how xarray hands back the int8-with-fill
    the format actually stores, and comparing it wrongly is what silently empties a series.
    `depths` becomes `DEPH`, so a deep level can be given a different temperature and the
    shallowest-level rule checked. `moving=True` makes position (TIME)-dimensioned, as a drifter
    or glider file really is.
    """
    time = pd.date_range("2020-01-01", periods=n, freq="h")
    nd = len(depths)
    temps = np.full((n, nd), float(temp), dtype="float32")
    for j in range(1, nd):                       # deeper levels are colder, and distinguishable
        temps[:, j] = temp - 3.0 * j
    ds = xr.Dataset(
        {"TEMP": (("TIME", "DEPTH"), temps),
         "TEMP_QC": (("TIME", "DEPTH"), np.full((n, nd), float(qc), dtype="float32")),
         "DEPH": (("DEPTH",), np.asarray(depths, dtype="float32"))},
        coords={"TIME": time})
    if moving:
        ds["LATITUDE"] = (("TIME",), lat + np.linspace(0, 0.5, n))
        ds["LONGITUDE"] = (("TIME",), np.full(n, lon))
    else:
        ds["LATITUDE"] = float(lat)              # SCALAR, as a fixed platform really is
        ds["LONGITUDE"] = float(lon)
    ds.attrs.update(platform_code=path.stem.split("_")[-1], platform_name=name, data_mode="R")
    ds.to_netcdf(path)


@pytest.fixture
def seams(monkeypatch, tmp_path):
    """The three network seams, replaced. Returns a handle recording what was asked for."""
    class Seams:
        def __init__(self):
            self.rows = list(DEFAULT_ROWS)
            self.index_calls = 0
            self.root_calls = 0
            self.downloaded = []            # the file_name lists handed to _download
            self.written = {}               # stem -> kwargs for write_platform_nc

        def root(self, dataset_id, part):
            self.root_calls += 1
            return f"https://example.invalid/{dataset_id}"

        def index(self, url):
            self.index_calls += 1
            return make_index(self.rows)

        def download(self, dataset_id, part, file_names, out_dir):
            self.downloaded.append(list(file_names))
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in file_names:
                stem = f.rsplit("/", 1)[-1]
                write_platform_nc(out_dir / stem, **self.written.get(stem, {}))

    s = Seams()
    monkeypatch.setattr(insitu_cmems, "_catalogue_root", s.root)
    monkeypatch.setattr(insitu_cmems, "_fetch_index", s.index)
    monkeypatch.setattr(insitu_cmems, "_download", s.download)
    # The per-process catalog cache must not leak between tests.
    monkeypatch.setattr(insitu_cmems, "_CATALOG_CACHE", {})
    return s


def _project(tmp_path, **opts):
    return parse_config({
        "name": "i", "output_dir": str(tmp_path),
        "time": {"start_date": "2020-01-01", "end_date": "2020-01-02"},
        "auth": {"copernicus": {"auth_strategy": "netrc"}},
        "products": {"insitu": opts or {}},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": LAT, "center_lon": LON,
             "buffer_ns_km": 5, "buffer_ew_km": 5}]}],
    })


@pytest.fixture
def g(tmp_path):
    return grid.project_grids(_project(tmp_path))[AOI]


def cfg(tmp_path, **over):
    """A resolved options bag, as insitu_acquire._ds_cfg + run() would build it."""
    base = {"dataset_id": insitu_cmems.DEFAULT_DATASET_ID, "dataset_part": "history",
            "platform_types": list(insitu_cmems.DEFAULT_PLATFORM_TYPES), "pad_deg": 0.0,
            "stations": [], "exclude_stations": [], "qc_flags": [1, 2],
            "max_sensor_depth_m": 5.0, "cache_dir": tmp_path / "_cache"}
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# Index selection -- intersection, not containment
# --------------------------------------------------------------------------- #
def test_a_platform_wider_than_the_aoi_is_kept(g):
    """THE bug to guard. Copernicus's own published example filters with CONTAINMENT

        lon_min >= box_lon_min and lon_max <= box_lon_max

    which keeps only files whose bounds fall entirely inside the box, silently discarding
    every platform whose recorded extent is merely generous. `WIDE` spans 10 degrees around a
    5 km AoI and holds the oldest record in the catalog (1987); containment drops it.
    """
    idx = insitu_cmems.parse_index(make_index(DEFAULT_ROWS))
    sel = insitu_cmems.select_files(idx, g.search_bbox, "1980-01-01", "2026-01-01",
                                    ["MO", "TG"])
    assert "WIDE" in _codes(sel)


def test_selection_rejects_the_things_it_should(g):
    idx = insitu_cmems.parse_index(make_index(DEFAULT_ROWS))
    sel = insitu_cmems.select_files(idx, g.search_bbox, "1980-01-01", "2026-01-01",
                                    ["MO", "TG"])
    names = _codes(sel)
    assert "GLIDER" not in names and "SHIP" not in names   # mobile platform types
    assert "NOTEMP" not in names                            # carries no TEMP parameter
    assert "ELSEWHEN" not in names                          # window does not overlap


def test_temp_is_matched_as_a_whole_token(g):
    """`parameters` is a SPACE-separated list inside a COMMA-separated file. A substring test
    would match `ATEMP` (air temperature) and `TEMPX`, admitting platforms with no water
    temperature at all."""
    rows = [_row("AIRONLY", "MO", 46.90, 46.94, -124.12, -124.08,
                 "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z", params="ATEMP TEMPX")]
    idx = insitu_cmems.parse_index(make_index(rows))
    sel = insitu_cmems.select_files(idx, g.search_bbox, "2020-01-01", "2020-01-02", ["MO"])
    assert sel.empty


def test_platform_type_is_read_positionally():
    """The vocabularies COLLIDE across filename fields: `MO` is Mediterranean in position 1 and
    mooring in position 3. A substring search would call every Mediterranean file a mooring."""
    assert insitu_cmems.platform_type("x/GL_TS_MO_46211.nc") == "MO"
    assert insitu_cmems.platform_type("x/MO_TS_GL_12345.nc") == "GL"   # Med glider, not mooring


# --------------------------------------------------------------------------- #
# fetch_aoi -- the source seam
# --------------------------------------------------------------------------- #
def test_mobile_platforms_are_never_downloaded(g, tmp_path, seams, caplog):
    """Filtered at the INDEX, before any byte is fetched. Over a real AoI this was 419 mobile
    files to reach 2 moorings, so downloading-then-dropping would be most of the run."""
    with caplog.at_level("INFO"):
        insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    fetched = "".join(seams.downloaded[0])
    assert "GLIDER" not in fetched and "SHIP" not in fetched
    # ...and said out loud, or a run that skips 419 files looks like a hang.
    assert "mobile/one-off" in caplog.text


def test_a_platform_whose_index_bounds_lie_is_dropped(g, tmp_path, seams, caplog):
    """The archive's `geospatial_*` bounds are sometimes wrong by continents, and intersection
    selection duly believes them.

    Real case: `GL_TS_MO_31261` advertises lat -31.5..48.7, lon -123.4..-34.6 -- half the planet
    -- while the file sits at (-8.16, -34.56) off Brazil. It intersected a Puget Sound AoI and
    arrived in the station table as a plausible mooring reporting 0.0-38.9 degC. Only the file's
    OWN position can settle it, so the guard runs after the download, not in the index filter.
    """
    seams.rows = [_row("LIAR", "MO", -31.5, 48.7, -125.0, -34.6,
                       "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z")]
    seams.written = {"GL_TS_MO_LIAR.nc": {"lat": -8.156, "lon": -34.564}}
    with caplog.at_level("INFO"):
        recs = insitu_cmems.fetch_aoi(g, "2020-01-01", "2020-01-02", cfg(tmp_path))
    assert recs == []
    assert "the file sits at" in caplog.text
    # It WAS selected and downloaded -- the index is what lied, and only opening it can tell.
    assert seams.downloaded and "LIAR" in "".join(seams.downloaded[0])


def test_a_platform_whose_position_agrees_is_kept(g, tmp_path, seams):
    """The other half of the guard: it must not reject platforms for being merely wide."""
    seams.rows = [_row("WIDEBUTHERE", "MO", 40.0, 50.0, -130.0, -120.0,
                       "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z")]
    seams.written = {"GL_TS_MO_WIDEBUTHERE.nc": {"lat": LAT, "lon": LON}}
    recs = insitu_cmems.fetch_aoi(g, "2020-01-01", "2020-01-02", cfg(tmp_path))
    assert {r["id"] for r in recs} == {"WIDEBUTHERE"}


def test_the_records_honour_the_shared_contract(g, tmp_path, seams):
    recs = insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    assert {r["id"] for r in recs} == {"MOORED", "WIDE"}
    for r in recs:
        df = r["df"]
        assert list(df.columns)[:4] == ["time", "latitude", "longitude", "value"]
        assert df["time"].dt.tz is None                  # naive UTC, as build_dataset needs
        assert np.isfinite(r["lat"]) and np.isfinite(r["lon"])
        # A fixed platform must survive the drift guard, or the source delivers nothing.
        assert insitu_acquire.platform_drift_m(df) == pytest.approx(0.0)


def test_an_aoi_with_only_mobile_platforms_is_reported_not_failed(g, tmp_path, seams, caplog):
    """Under 5% of this archive is fixed platforms, so an empty coastal AoI is an ORDINARY
    outcome (measured: all of Tasmania). It must read as a fact about the ocean, not as a
    download that broke."""
    seams.rows = [r for r in DEFAULT_ROWS if "_MO_" not in r]
    with caplog.at_level("INFO"):
        recs = insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    assert recs == []
    assert seams.downloaded == []                        # nothing was fetched
    assert "no FIXED platforms" in caplog.text


def test_dry_run_lists_without_downloading(g, tmp_path, seams, caplog):
    with caplog.at_level("INFO"):
        recs = insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path),
                                      dry_run=True)
    assert recs == [] and seams.downloaded == []
    assert "[dry-run]" in caplog.text and "MOORED" in caplog.text


def test_the_index_is_fetched_once_across_aois(g, tmp_path, seams):
    """The catalog is ~28 MB and `fetch_aoi` runs once per AoI, so a ten-AoI project would
    otherwise pull 280 MB of identical bytes."""
    for _ in range(3):
        insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    assert seams.index_calls == 1
    assert seams.root_calls == 1


def test_the_index_is_cached_on_disk_across_runs(g, tmp_path, seams, monkeypatch):
    """In-memory caching helps one run; the index outlives runs. It is ~28 MB in a single
    unresumed GET, fetched once for EVERY AoI, so re-pulling it each invocation is both slow
    and a single point of failure for the whole run."""
    insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    assert (tmp_path / "_cache" / "index_history.txt").exists()
    assert seams.index_calls == 1

    monkeypatch.setattr(insitu_cmems, "_CATALOG_CACHE", {})     # a fresh process
    insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    assert seams.index_calls == 1, "a warm on-disk index was refetched"


def test_a_stale_index_beats_no_data_when_the_refresh_fails(g, tmp_path, seams, monkeypatch,
                                                            caplog):
    """The failure this was written after: `Connection broken: IncompleteRead(...)` exhausted
    every retry and failed EVERY AoI, on an endpoint that served the same file cleanly seconds
    later. A day-old catalog is a far better answer than none -- said out loud, not silently."""
    insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    cached = tmp_path / "_cache" / "index_history.txt"
    import os
    os.utime(cached, (0, 0))                                    # force it stale

    def boom(url):
        raise ConnectionError("Connection broken: IncompleteRead(12243941 bytes read)")

    monkeypatch.setattr(insitu_cmems, "_fetch_index", boom)
    monkeypatch.setattr(insitu_cmems, "_CATALOG_CACHE", {})
    with caplog.at_level("WARNING"):
        recs = insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))
    assert {r["id"] for r in recs} == {"MOORED", "WIDE"}         # the run survived
    assert "using the cached copy" in caplog.text


def test_a_refresh_failure_with_no_cache_still_raises(g, tmp_path, seams, monkeypatch):
    """The fallback must not turn a first-run outage into a silent empty channel."""
    monkeypatch.setattr(insitu_cmems, "_fetch_index",
                        lambda url: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(ConnectionError):
        insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path))


def test_the_station_allow_and_deny_lists_apply(g, tmp_path, seams):
    recs = insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01",
                                  cfg(tmp_path, stations=["MOORED"]))
    assert {r["id"] for r in recs} == {"MOORED"}
    recs = insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01",
                                  cfg(tmp_path, exclude_stations=["MOORED"]))
    assert {r["id"] for r in recs} == {"WIDE"}


def test_a_region_shorthand_resolves_to_a_dataset_id(g, tmp_path, seams, monkeypatch):
    """`dataset_id: nws` beats pasting an opaque `cmems_obs-ins_nws_...` string, and a regional
    product is worth naming: it ingests national networks the global stream never sees."""
    seen = {}

    def spy_root(dataset_id, part):
        seen["id"] = dataset_id
        return "https://example.invalid/x"

    monkeypatch.setattr(insitu_cmems, "_catalogue_root", spy_root)
    insitu_cmems.fetch_aoi(g, "1980-01-01", "2026-01-01", cfg(tmp_path, dataset_id="nws"))
    assert seen["id"] == insitu_cmems.REGIONAL_DATASETS["nws"]


# --------------------------------------------------------------------------- #
# read_platform_file -- the format traps
# --------------------------------------------------------------------------- #
def test_the_qc_flag_filters_despite_arriving_as_float(tmp_path):
    """`TEMP_QC` is stored int8 with a -127 fill and DECODED BY XARRAY TO FLOAT32. A comparison
    that assumes an integer dtype silently keeps nothing."""
    p = tmp_path / "GL_TS_MO_X.nc"
    write_platform_nc(p, qc=4.0)                          # 4 = bad data
    assert insitu_cmems.read_platform_file(p, [1, 2], 5.0) is None
    kept = insitu_cmems.read_platform_file(p, [1, 2, 4], 5.0)
    assert kept is not None and len(kept) > 0


def test_the_shallowest_level_wins_per_timestamp(tmp_path):
    """A mooring reports several depths; only the shallowest is comparable to a satellite's
    surface retrieval. The fixture makes the deep level 3 degC colder, so a wrong pick shows."""
    p = tmp_path / "GL_TS_MO_X.nc"
    write_platform_nc(p, depths=(0.0, 3.0), temp=12.0)
    df = insitu_cmems.read_platform_file(p, [1, 2], 5.0)
    assert df["value"].unique().tolist() == [pytest.approx(12.0)]
    assert len(df) == 48                                   # one row per timestamp, not two


def test_deep_sensors_are_dropped_by_max_sensor_depth(tmp_path):
    p = tmp_path / "GL_TS_MO_X.nc"
    write_platform_nc(p, depths=(40.0, 80.0))              # a profiling mooring, nothing shallow
    assert insitu_cmems.read_platform_file(p, [1, 2], 5.0) is None


def test_a_moving_platform_reports_its_real_drift(tmp_path):
    """The parser must not flatten a (TIME)-dimensioned position into one point -- that is what
    lets the shared drift guard SEE a track and drop it, instead of placing it confidently in
    the wrong pixel."""
    p = tmp_path / "GL_TS_GL_X.nc"
    write_platform_nc(p, moving=True)
    df = insitu_cmems.read_platform_file(p, [1, 2], 5.0)
    assert insitu_acquire.platform_drift_m(df) > 1000.0


def test_pressure_stands_in_for_depth(tmp_path):
    """Profiling platforms carry `PRES` (dbar) and no `DEPH`. In the top few metres that is
    within a few percent of depth in metres -- close enough for the gate, and better than
    dropping the platform for lacking a variable name."""
    p = tmp_path / "GL_TS_MO_X.nc"
    write_platform_nc(p, depths=(1.0,))
    with xr.open_dataset(p) as ds:
        renamed = ds.rename({"DEPH": "PRES"}).load()
    p2 = tmp_path / "GL_TS_MO_Y.nc"
    renamed.to_netcdf(p2)
    assert insitu_cmems.read_platform_file(p2, [1, 2], 5.0) is not None


# --------------------------------------------------------------------------- #
# Registry + the auth preflight
# --------------------------------------------------------------------------- #
def test_the_source_is_registered_and_stacks():
    spec = products.spec(DataProduct.insitu)
    assert "marineinsitu" in spec.sources
    assert spec.auth["marineinsitu"] == "copernicus"
    assert insitu_cmems.SOURCE in insitu_acquire.SOURCES


def test_it_is_not_on_by_default(tmp_path):
    """Like `csv`, it must be opted into: it needs a credential no other source needs."""
    assert "marineinsitu" not in insitu_acquire.DEFAULT_SOURCES
    eff = insitu_acquire._build_eff(_project(tmp_path))
    assert eff["ds"][AOI]["sources"] == ["ioos"]


def test_an_insitu_config_with_no_sources_needs_no_copernicus_block(tmp_path):
    """THE REGRESSION GUARD. The auth preflight used to read an unset `sources` as "every known
    source", so registering one credentialed source would have made every existing config with
    an `insitu:` block demand an `auth.copernicus` block for a source it never runs."""
    parse_config({
        "name": "i", "output_dir": str(tmp_path),
        "time": {"start_date": "2020-01-01", "end_date": "2020-01-02"},
        "products": {"insitu": {}},                       # no `sources:`, no `auth:`
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": LAT, "center_lon": LON,
             "buffer_ns_km": 5, "buffer_ew_km": 5}]}],
    })


def test_naming_the_source_does_require_the_credential(tmp_path):
    with pytest.raises(ValueError, match="copernicus"):
        parse_config({
            "name": "i", "output_dir": str(tmp_path),
            "time": {"start_date": "2020-01-01", "end_date": "2020-01-02"},
            "products": {"insitu": {"sources": ["marineinsitu"]}},   # no auth block
            "regions": [{"name": "r", "areas": [
                {"name": AOI, "center_lat": LAT, "center_lon": LON,
                 "buffer_ns_km": 5, "buffer_ew_km": 5}]}],
        })


def test_the_config_options_are_accepted(tmp_path):
    eff = insitu_acquire._build_eff(_project(
        tmp_path, sources=["marineinsitu"], dataset_id="ibi", dataset_part="history",
        platform_types=["MO"]))
    ds = eff["ds"][AOI]
    assert ds["dataset_id"] == "ibi" and ds["platform_types"] == ["MO"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
