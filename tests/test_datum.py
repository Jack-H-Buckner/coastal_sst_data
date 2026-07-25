"""DEM vertical-datum resolver (a LIBRARY, called inline by the bathymetry module). All
network goes through the single seam `datum._http_json`, which is monkeypatched with
table-driven fakes -- so every test here is offline. The emphasis is on the ways this can be
silently WRONG (a bad offset biases every water-level pixel by ~1 m while the cube still looks
healthy), not merely missing: the -999999 sentinel, the sign, region pinning, and the
region fallback (datum_offset_m, used only when VDatum/CO-OPS cannot resolve the offset)."""

import numpy as np
import pytest
import xarray as xr

from coastal_sst_data.config import parse_config
from coastal_sst_data import grid
from coastal_sst_data.processes import datacube, datum


AOI = "aoi1"

# A real VDatum success (Seattle): t_z = -1.278 -> offset +1.278.
VD_OK = {"t_z": "-1.278", "uncertainty": "0.076"}
VD_SENTINEL = {"t_z": "-999999", "uncertainty": ""}
VD_412_FRAME = {"errorCode": 412,
                "message": "For West Coast Region, Target Horizontal Frame should be "
                           "IGS14 for Tidal"}
# The real gauge 9447130 (metric): MSL - NAVD88 = 4.443 - 3.134 = 1.309.
COOPS_OK = {"units": "meters", "epoch": "1983-2001",
            "datums": [{"name": "MHHW", "value": 5.882}, {"name": "MSL", "value": 4.443},
                       {"name": "NAVD88", "value": 3.134}]}
COOPS_NO_NAVD88 = {"units": "meters", "datums": [{"name": "MSL", "value": 4.443}]}

STATIONS = [{"id": "9447130", "name": "Seattle", "lat": 45.52, "lon": -123.93}]


def _project(tmp_path, fallback=None):
    sources = {"bathymetry": {"sources": ["cudem"]}}
    if fallback is not None:
        sources["bathymetry"]["datum_offset_m"] = fallback
    return parse_config({
        "name": "d", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-02"},
        "products": {"bathymetry": None, "tides": None},
        "regions": [{"name": "r", "sources": sources, "areas": [
            {"name": AOI, "center_lat": 45.52, "center_lon": -123.925,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


@pytest.fixture
def project(tmp_path):
    return _project(tmp_path)


@pytest.fixture
def g(project):
    return grid.project_grids(project)[AOI]


class FakeHTTP:
    """Records every call and replies from a queue (VDatum) / table (CO-OPS)."""

    def __init__(self, vdatum=None, coops=None):
        self.vdatum = list(vdatum or [])
        self.coops = coops
        self.calls = []

    def __call__(self, url, params, **kw):
        self.calls.append((url, dict(params)))
        if "vdatum" in url:
            if not self.vdatum:
                return dict(VD_SENTINEL)
            nxt = self.vdatum.pop(0)
            return dict(nxt) if isinstance(nxt, dict) else nxt
        if self.coops is None:
            raise AssertionError("unexpected CO-OPS call")
        return dict(self.coops)

    @property
    def n_vdatum(self):
        return sum(1 for u, _ in self.calls if "vdatum" in u)


def _bathy_source(project, g, elev, src, *, datum_offset_m, datum_status="ok",
                  dem_vertical_datum="NAVD88"):
    """Write one DEM source's aligned output (BATHYMETRY/<src>/aligned/<aoi>) with its datum
    offset stamped on -- exactly the shape the bathymetry module now produces per source."""
    xs, ys = g.xy_centers()
    ds = xr.Dataset(
        {"elevation": (("y", "x"), elev),
         "depth": (("y", "x"), np.where(elev < 0, -elev, 0.0).astype("float32")),
         "depth_p25": (("y", "x"), np.where(elev < 0, -elev, 0.0).astype("float32")),
         "depth_p75": (("y", "x"), np.where(elev < 0, -elev, 0.0).astype("float32"))},
        coords={"y": ys, "x": xs})
    ds.attrs.update(aoi_id=AOI, bathy_source=src, datum_offset_m=datum_offset_m,
                    datum_status=datum_status, datum_method="vdatum_navd88_to_lmsl",
                    dem_vertical_datum=dem_vertical_datum)
    d = project.output_dir / "BATHYMETRY" / src / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{AOI}.nc")


def _shore(g):
    """A DEM with a waterline: west deep, a band around zero, east upland."""
    elev = np.full((g.height, g.width), -20.0, "float32")
    third = g.width // 3
    elev[:, third:2 * third] = -1.0          # the waterline band
    elev[:, 2 * third:] = 30.0               # upland
    return elev


# --------------------------------------------------------------------------- #
# Which DEM actually ran (never guessed)
# --------------------------------------------------------------------------- #
def test_dem_source_parsed_from_the_bathymetry_provenance_string():
    cudem = xr.Dataset(attrs={"source": 'NCEI CUDEM 1/9" (98% cover, 10x10 subgrid)'})
    gmrt = xr.Dataset(attrs={"source": "GMRT (topo, max)"})
    assert datum.dem_source_of(cudem) == "cudem"
    assert datum.dem_source_of(gmrt) == "gmrt"


def test_unknown_dem_provenance_raises_rather_than_guessing():
    # Guessing here would silently bias every water-level pixel, so it must fail loud.
    with pytest.raises(ValueError, match="cannot tell which DEM"):
        datum.dem_source_of(xr.Dataset(attrs={"source": "GEBCO 2024"}))
    with pytest.raises(ValueError):
        datum.dem_source_of(xr.Dataset(attrs={}))


# --------------------------------------------------------------------------- #
# Sampling (pure -- no network at all)
# --------------------------------------------------------------------------- #
def test_samples_are_drawn_from_the_waterline_band_and_spread_out(g):
    pts = datum.sample_points(_shore(g), g)
    assert 1 <= len(pts) <= datum.MAX_QUERIES
    lons = [p[0] for p in pts]
    assert len(set(lons)) > 1                       # not all the same point/column
    w, s, e, n = g.search_bbox
    assert all(w - 0.05 <= lon <= e + 0.05 and s - 0.05 <= lat <= n + 0.05
               for lon, lat in pts)


def test_sampling_falls_back_to_the_closest_to_zero_cells_when_nothing_is_in_band(g):
    deep = np.full((g.height, g.width), -50.0, "float32")   # no cell within the band
    assert datum.sample_points(deep, g)                     # still returns points

    assert datum.sample_points(np.full((g.height, g.width), np.nan, "float32"), g) == []


# --------------------------------------------------------------------------- #
# VDatum: sign, sentinel, frame repair, region pinning
# --------------------------------------------------------------------------- #
def test_sign_convention_offset_is_minus_t_z(monkeypatch):
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(vdatum=[VD_OK]))
    res = datum.vdatum_point(-122.34, 47.60, "westcoast")
    assert res["offset"] == pytest.approx(1.278)     # NOT -1.278
    assert res["uncertainty"] == pytest.approx(0.076)


def test_out_of_coverage_sentinel_is_rejected_not_negated(monkeypatch):
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(vdatum=[VD_SENTINEL]))
    res = datum.vdatum_point(-122.58, 48.45, "westcoast")
    assert res["offset"] is None                     # never +999999
    assert res["uncertainty"] is None                # empty string must not raise


def test_float_sentinel_and_all_sentinel_points_yield_no_offset(monkeypatch, g):
    fake = FakeHTTP(vdatum=[{"t_z": -999999.0, "uncertainty": ""}] * 40)
    monkeypatch.setattr(datum, "_http_json", fake)
    assert datum.vdatum_offset(datum.sample_points(_shore(g), g), -123.9) is None


def test_412_names_the_frame_and_is_retried_once_with_it(monkeypatch):
    fake = FakeHTTP(vdatum=[VD_412_FRAME] + [VD_OK] * 30)
    monkeypatch.setattr(datum, "_http_json", fake)
    out = datum.vdatum_offset([(-123.9, 45.5)] * 6, -123.9)
    assert out["offset_m"] == pytest.approx(1.278)
    assert out["vdatum_frame"] == "IGS14"
    # the retry carried the frame the error asked for -- and only the TARGET frame
    retry = fake.calls[1][1]
    assert retry["t_h_frame"] == "IGS14"
    assert "s_h_frame" not in retry


def test_region_is_pinned_after_the_first_success(monkeypatch, g):
    # A later sentinel must not re-route us into `contiguous`, whose tidal grid would
    # answer a west-coast point with a plausible but wrong number.
    fake = FakeHTTP(vdatum=[VD_OK] * 3 + [VD_SENTINEL] + [VD_OK] * 30)
    monkeypatch.setattr(datum, "_http_json", fake)
    out = datum.vdatum_offset(datum.sample_points(_shore(g), g), -123.9)
    assert out["vdatum_region"] == "westcoast"
    assert {p["region"] for _, p in fake.calls} == {"westcoast"}


def test_implausibly_large_offset_is_refused(monkeypatch):
    monkeypatch.setattr(datum, "_http_json",
                        FakeHTTP(vdatum=[{"t_z": "-9.5", "uncertainty": "0.1"}] * 10))
    assert datum.vdatum_offset([(-123.9, 45.5)] * 6, -123.9) is None


# --------------------------------------------------------------------------- #
# CO-OPS
# --------------------------------------------------------------------------- #
def test_coops_offset_is_msl_minus_navd88(monkeypatch):
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(coops=COOPS_OK))
    out = datum.coops_offset(-123.925, 45.52, STATIONS)
    assert out["offset_m"] == pytest.approx(1.309)
    assert out["station_id"] == "9447130"


def test_gauge_without_navd88_gives_none_not_zero(monkeypatch):
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(coops=COOPS_NO_NAVD88))
    assert datum.coops_offset(-123.925, 45.52, STATIONS) is None


def test_distant_gauge_is_refused(monkeypatch):
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(coops=COOPS_OK))
    far = [{"id": "x", "name": "far", "lat": 40.0, "lon": -123.9}]      # ~600 km
    assert datum.coops_offset(-123.925, 45.52, far) is None


def test_feet_are_never_read_as_metres(monkeypatch):
    feet = {"units": "feet", "datums": [{"name": "MSL", "value": 14.58},
                                        {"name": "NAVD88", "value": 10.28}]}
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(coops=feet))
    assert datum.coops_offset(-123.925, 45.52, STATIONS) is None


# --------------------------------------------------------------------------- #
# resolve_aoi: the whole chain
# --------------------------------------------------------------------------- #
def test_gmrt_resolves_to_zero_with_no_network_at_all(monkeypatch, g):
    fake = FakeHTTP()
    monkeypatch.setattr(datum, "_http_json", fake)
    rec = datum.resolve_aoi(g, _shore(g), "gmrt")
    assert rec["datum_offset_m"] == 0.0
    assert rec["method"] == "gmrt_msl_assumed"
    assert rec["status"] == "ok"
    assert fake.calls == []                       # GMRT is already sea-level referenced


def test_cudem_resolves_from_vdatum_and_crosschecks_the_gauge(monkeypatch, g, caplog):
    fake = FakeHTTP(vdatum=[VD_OK] * 30, coops=COOPS_OK)
    monkeypatch.setattr(datum, "_http_json", fake)
    rec = datum.resolve_aoi(g, _shore(g), "cudem", stations=STATIONS)
    assert rec["datum_offset_m"] == pytest.approx(1.278)
    assert rec["method"] == "vdatum_navd88_to_lmsl"
    assert rec["coops_offset_m"] == pytest.approx(1.309)
    assert rec["crosscheck_delta_m"] == pytest.approx(0.031, abs=1e-3)
    assert rec["status"] == "ok"


def test_crosscheck_shouts_when_the_two_methods_disagree(monkeypatch, g, caplog):
    # A sign flip is the classic failure: VDatum -1.278 vs the gauge's +1.309.
    flipped = {"t_z": "1.278", "uncertainty": "0.076"}
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(vdatum=[flipped] * 30, coops=COOPS_OK))
    with caplog.at_level("WARNING"):
        datum.resolve_aoi(g, _shore(g), "cudem", stations=STATIONS)
    assert "disagree" in caplog.text


def test_coops_is_the_fallback_when_vdatum_has_no_coverage(monkeypatch, g):
    monkeypatch.setattr(datum, "_http_json",
                        FakeHTTP(vdatum=[VD_SENTINEL] * 40, coops=COOPS_OK))
    rec = datum.resolve_aoi(g, _shore(g), "cudem", stations=STATIONS)
    assert rec["datum_offset_m"] == pytest.approx(1.309)
    assert rec["method"] == "coops_msl_minus_navd88"


def test_unresolvable_navd88_dem_assumes_zero_but_says_so_loudly(monkeypatch, g, caplog):
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(vdatum=[VD_SENTINEL] * 40,
                                                      coops=COOPS_NO_NAVD88))
    with caplog.at_level("ERROR"):
        rec = datum.resolve_aoi(g, _shore(g), "cudem", stations=STATIONS)
    assert rec["datum_offset_m"] == 0.0
    assert rec["status"] == "unresolved_assumed_zero"     # travels onto the cube
    assert "COULD NOT RESOLVE" in caplog.text


def test_a_wide_spread_refuses_a_single_scalar(monkeypatch, g, caplog):
    spread = [{"t_z": f"-{v}", "uncertainty": "0.08"} for v in
              (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)] * 5
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(vdatum=spread, coops=COOPS_OK))
    with caplog.at_level("ERROR"):
        rec = datum.resolve_aoi(g, _shore(g), "cudem", stations=STATIONS)
    assert rec["status"] == "unresolved_assumed_zero"
    assert "straddles tidal zones" in caplog.text


# --------------------------------------------------------------------------- #
# The region FALLBACK (datum_offset_m): used ONLY when VDatum/CO-OPS can't resolve.
# --------------------------------------------------------------------------- #
def test_a_resolved_vdatum_value_wins_over_the_config_fallback(monkeypatch, tmp_path):
    # datum_offset_m is a fallback, not an override: a value VDatum resolved is authoritative,
    # so the config value is IGNORED here (the semantic flip from the old override behaviour).
    p = _project(tmp_path, fallback=1.31)
    gg = grid.project_grids(p)[AOI]
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(vdatum=[VD_OK] * 30, coops=COOPS_OK))
    rec = datum.resolve_aoi(gg, _shore(gg), "cudem",
                            fallback=datum.resolve_fallback(p, AOI), stations=STATIONS)
    assert rec["datum_offset_m"] == pytest.approx(1.278)
    assert rec["method"] == "vdatum_navd88_to_lmsl"
    assert "config_offset_m" not in rec                   # the fallback was never applied


def test_config_fallback_is_used_when_nothing_resolves(monkeypatch, tmp_path, caplog):
    # VDatum out of coverage everywhere + no NAVD88 gauge -> the region's datum_offset_m is used.
    p = _project(tmp_path, fallback=1.3)
    gg = grid.project_grids(p)[AOI]
    monkeypatch.setattr(datum, "_http_json",
                        FakeHTTP(vdatum=[VD_SENTINEL] * 40, coops=COOPS_NO_NAVD88))
    with caplog.at_level("WARNING"):
        rec = datum.resolve_aoi(gg, _shore(gg), "cudem",
                                fallback=datum.resolve_fallback(p, AOI), stations=STATIONS)
    assert rec["datum_offset_m"] == pytest.approx(1.3)
    assert rec["status"] == "regional_default"
    assert rec["method"] == "config_fallback"
    assert rec["resolved_offset_m"] == 0.0                # the 0.0 it replaced, kept for provenance
    assert "falling back to the region's datum_offset_m" in caplog.text


def test_config_fallback_is_used_when_vdatum_straddles_tidal_zones(monkeypatch, tmp_path):
    # A wide VDatum spread refuses a single scalar (unresolved_assumed_zero) -> the fallback fills it.
    p = _project(tmp_path, fallback=0.9)
    gg = grid.project_grids(p)[AOI]
    spread = [{"t_z": f"-{v}", "uncertainty": "0.08"} for v in
              (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)] * 5
    monkeypatch.setattr(datum, "_http_json", FakeHTTP(vdatum=spread, coops=COOPS_OK))
    rec = datum.resolve_aoi(gg, _shore(gg), "cudem",
                            fallback=datum.resolve_fallback(p, AOI), stations=STATIONS)
    assert rec["datum_offset_m"] == pytest.approx(0.9)
    assert rec["status"] == "regional_default"


def test_config_fallback_is_ignored_when_gmrt_resolves_structurally(monkeypatch, tmp_path):
    # The old footgun (a NAVD88 number reaching an MSL DEM) is now avoided STRUCTURALLY: GMRT
    # always resolves to ok, so the fallback -- which only fires on unresolved -- never touches it.
    p = _project(tmp_path, fallback=1.3)
    gg = grid.project_grids(p)[AOI]
    monkeypatch.setattr(datum, "_http_json", FakeHTTP())     # no network expected for GMRT
    rec = datum.resolve_aoi(gg, _shore(gg), "gmrt",
                            fallback=datum.resolve_fallback(p, AOI))
    assert rec["datum_offset_m"] == 0.0                      # GMRT's structural 0.0 stands
    assert rec["status"] == "ok"
    assert "config_offset_m" not in rec


# --------------------------------------------------------------------------- #
# Hand-off to the cube: each DEM source's offset ships as attrs on its elevation_<src>
# channel (stamped by bathymetry when it acquired that DEM, surfaced by the assembler).
# --------------------------------------------------------------------------- #
def test_the_cube_carries_each_sources_datum_on_its_elevation_channel(project, g):
    import pandas as pd
    # Two DEMs stacked: CUDEM (NAVD88, offset 1.3) and GMRT (MSL, offset 0) -- and a single
    # global offset would be WRONG for one of them, which is why it is per source.
    elev = np.full((g.height, g.width), -0.5, "float32")
    _bathy_source(project, g, elev, "cudem", datum_offset_m=1.3, dem_vertical_datum="NAVD88")
    _bathy_source(project, g, elev, "gmrt", datum_offset_m=0.0,
                  dem_vertical_datum="MSL_APPROX")
    days = pd.date_range("2026-06-01", "2026-06-02", freq="D")
    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    assert ds["elevation_cudem"].attrs["datum_offset_m"] == pytest.approx(1.3)
    assert ds["elevation_cudem"].attrs["datum_status"] == "ok"
    assert ds["elevation_cudem"].attrs["dem_vertical_datum"] == "NAVD88"
    assert ds["elevation_gmrt"].attrs["datum_offset_m"] == pytest.approx(0.0)
    # ...and no single global datum attr any more (it would be wrong for one source).
    assert "datum_offset_m" not in ds.attrs

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])