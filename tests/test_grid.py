"""Per-AoI grid computation (coastal_sst_data.grid). Pure/offline: no network."""

from pathlib import Path

import pytest

from coastal_sst_data.config import AreaOfInterest, GridSpec, load_config
from coastal_sst_data import grid

EXAMPLE = Path(__file__).parents[1] / "examples" / "config.test.yaml"


def _area(**kw):
    """A valid AreaOfInterest with sensible defaults; override per test."""
    base = dict(name="a", center_lat=45.0, center_lon=-123.0,
                buffer_ns_km=10.0, buffer_ew_km=10.0)
    base.update(kw)
    return AreaOfInterest(**base)


# ---------------------------------------------------------------------------
# utm_epsg_from_lonlat: local UTM zone selection.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lon, lat, expected", [
    (-123.0, 45.0, "EPSG:32610"),    # Pacific NW -> zone 10 N
    (-123.925, 45.52, "EPSG:32610"),
    (0.5, 51.5, "EPSG:32631"),       # London -> zone 31 N
    (147.0, -43.0, "EPSG:32755"),    # Tasmania -> zone 55 S (327xx)
])
def test_utm_epsg_from_lonlat(lon, lat, expected):
    assert grid.utm_epsg_from_lonlat(lon, lat) == expected


# ---------------------------------------------------------------------------
# compute_aoi_grid: CRS resolution, snapping, pixel size, shape.
# ---------------------------------------------------------------------------
def test_compute_aoi_grid_auto_crs_and_snapped():
    area = _area(center_lat=45.52, center_lon=-123.925, buffer_ns_km=25, buffer_ew_km=15)
    g = grid.compute_aoi_grid(area, GridSpec())    # defaults: 100 m, auto CRS, snap
    assert g.target_crs == "EPSG:32610"            # auto local UTM from the center
    assert g.width > 0 and g.height > 0
    assert g.shape == (g.height, g.width)
    # north-up pixel size equals the requested resolution
    assert g.transform.a == 100.0
    assert g.transform.e == -100.0
    # snapped: the top-left origin lands on a 100 m multiple
    assert g.transform.c % 100 == 0
    assert g.transform.f % 100 == 0


def test_compute_aoi_grid_explicit_crs_passthrough():
    """An explicit target_crs is used verbatim (no auto override)."""
    g = grid.compute_aoi_grid(_area(), GridSpec(target_crs="EPSG:3857"))
    assert g.target_crs == "EPSG:3857"


def test_compute_aoi_grid_resolution_override():
    """resolution_m drives the pixel size."""
    g = grid.compute_aoi_grid(_area(), GridSpec(resolution_m=250))
    assert g.resolution_m == 250.0
    assert g.transform.a == 250.0 and g.transform.e == -250.0


def test_compute_aoi_grid_antimeridian_raises():
    """A box crossing 180 deg can't map to one UTM zone -> NotImplementedError."""
    area = _area(center_lon=-179.95, buffer_ew_km=50)   # bbox crosses the seam
    with pytest.raises(NotImplementedError, match="antimeridian"):
        grid.compute_aoi_grid(area, GridSpec())


# ---------------------------------------------------------------------------
# project_grids: one grid per AoI, keyed by name.
# ---------------------------------------------------------------------------
def test_project_grids_one_per_aoi():
    grids = grid.project_grids(load_config(EXAMPLE))
    assert set(grids) == {"tillamook_bay", "padilla_bay"}
    assert all(g.name == name for name, g in grids.items())
    # both AoIs share the same UTM zone here -> the "one grid per AoI" guarantee
    assert {g.target_crs for g in grids.values()} == {"EPSG:32610"}

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])