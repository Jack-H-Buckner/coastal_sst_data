"""THE ACID TEST: add a fourth thermal sensor with no edits outside its declaration.

The whole point of the ProductSpec registry is that adding a data product stops being a
dozen edits across five files -- the DataProduct enum, PRODUCT_OPTIONS, AUTH_REQUIREMENTS,
the pipeline's PROCESSES + source registry + PROCESS_ORDER, store.REQUIRED_VARS, the
assembler's PRODUCT_DIRS + DAILY_CHANNELS + three hardcoded ("eco","lst","modis") tuples +
the Dataset literal, and provenance's SENSORS + field map -- any one of which could be
forgotten SILENTLY.

So: register a synthetic VIIRS-shaped sensor, write nothing but its spec, and assert that
every one of those registries picks it up and that an assembled cube gains its full set of
channels with correct values.

The monkeypatching below re-derives the module-level tables from the patched registry. That
is not a design smell -- those tables are computed once at import, so this is simply
simulating "restart the process with the new spec in products.py". The point stands: the
only SOURCE edit is the spec.
"""

from enum import Enum

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data import grid, pipeline, products, provenance, store
from coastal_sst_data.config import parse_config
from coastal_sst_data.processes import datacube
from coastal_sst_data.products import Kind, ProductSpec, SensorSpec

AOI = "aoi1"


class NewProduct(str, Enum):
    """Stands in for a new member of the DataProduct enum (which is closed at runtime).

    In real use, adding a sensor means adding an enum member AND a ProductSpec -- both in
    products.py, the same file. Nothing else.
    """
    viirs = "viirs"


VIIRS = ProductSpec(
    product=NewProduct.viirs,
    dir="VIIRS",
    kind=Kind.OVERPASS_SENSOR,
    module="coastal_sst_data.processes.modis",   # stand-in: we only assemble, never acquire
    auth="earthdata",
    options=frozenset({"output_format", "overwrite", "short_name"}),
    required_vars=("sst", "water", "cloud", "valid"),
    # Deliberately a DIFFERENT validity rule from all three existing sensors: normal water
    # polarity like Landsat, but gated on QC like ECOSTRESS. If the assembler were still
    # branching on hardcoded prefixes rather than reading the spec, this could not work.
    sensor=SensorSpec(prefix="viirs", water_is_land=False, use_cloud=False,
                      qc_levels=(0, 1), has_cloud=True),
    provenance_inputs=("viirs",),
)


@pytest.fixture
def registered(monkeypatch):
    """Register VIIRS and re-derive every table that is computed from the registry."""
    reg = products.REGISTRY + (VIIRS,)
    monkeypatch.setattr(products, "REGISTRY", reg)
    monkeypatch.setattr(products, "BY_PRODUCT", {s.product: s for s in reg})

    # The tables each module derives at import time.
    monkeypatch.setattr(store, "REQUIRED_VARS", {s.dir: s.required_vars for s in reg})
    monkeypatch.setattr(datacube, "PRODUCT_DIRS", products.product_dirs())
    monkeypatch.setattr(datacube, "DAILY_CHANNELS",
                        {s.product.value: s.coverage_channel for s in reg if s.coverage_channel})
    sensors = {s.sensor.prefix: s.product.value for s in products.sensors()}
    monkeypatch.setattr(provenance, "SENSORS", sensors)
    monkeypatch.setattr(provenance, "_SENSOR_RE",
                        __import__("re").compile(rf"^({'|'.join(sensors)})_(.+)$"))
    monkeypatch.setattr(pipeline, "PROCESS_MODULES",
                        {s.product: s.module for s in reg if s.module})
    return reg


# --------------------------------------------------------------------------- #
# The registries pick it up
# --------------------------------------------------------------------------- #
def test_the_registry_invariants_accept_the_new_sensor(registered):
    products._check_registry()                    # must not raise
    assert VIIRS.sensor.prefix in {s.sensor.prefix for s in products.sensors()}


def test_store_learns_its_completeness_check(registered):
    """Without this, a truncated VIIRS download would be taken for done on every re-run."""
    assert store.REQUIRED_VARS["VIIRS"] == ("sst", "water", "cloud", "valid")


def test_the_assembler_learns_its_directory(registered):
    assert datacube.PRODUCT_DIRS["viirs"] == "VIIRS"


def test_it_is_not_coverage_checked(registered):
    """An overpass sensor with no scene on a day is normal, not a defect -- warning about it
    would train the user to ignore the warning."""
    assert "viirs" not in datacube.DAILY_CHANNELS


def test_provenance_maps_every_channel_it_will_produce(registered):
    """An unmapped field ships with a BLANK source record and only a log line to say so."""
    assert provenance.SENSORS["viirs"] == "viirs"
    assert provenance.field_inputs("viirs_sst") == ["viirs"]
    assert provenance.field_inputs("viirs_valid") == ["viirs"]
    assert provenance.field_inputs("viirs_insitu_sst") == ["insitu", "viirs"]
    assert provenance.field_inputs("viirs_airtemp_hrrr") == ["met_overpass", "viirs"]


def test_the_pipeline_orders_it(registered):
    order = [p.value for p in pipeline.process_order()]
    assert "viirs" in order


# --------------------------------------------------------------------------- #
# ...and an assembled cube actually gains its channels
# --------------------------------------------------------------------------- #
@pytest.fixture
def project(tmp_path):
    return parse_config({
        "name": "dc", "output_dir": str(tmp_path),
        "time": {"start_date": "2026-06-01", "end_date": "2026-06-02"},
        "products": {"bathymetry": None},
        "regions": [{"name": "r", "areas": [
            {"name": AOI, "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 2, "buffer_ew_km": 2}]}],
    })


def _write(project, sub, fname, ds):
    d = project.output_dir / sub / "aligned" / AOI
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / fname)


def test_an_assembled_cube_gains_the_new_sensors_channels(registered, project):
    """The payoff. Write VIIRS aligned files; the assembler -- untouched -- produces its
    sst / cloud / valid / hour channels, honouring the validity rule from its SPEC
    (QC-gated, normal water polarity)."""
    g = grid.project_grids(project)[AOI]
    days = pd.date_range("2026-06-01", "2026-06-02", freq="D")
    H, W = g.height, g.width
    xs, ys = g.xy_centers()

    def t3(v):
        return (("time", "y", "x"), np.full((1, H, W), v, "float32"))

    # statics, so the cube has a water mask + elevation to build water level from
    _write(project, "BATHYMETRY", f"{AOI}.nc", xr.Dataset(
        {"elevation": (("y", "x"), np.full((H, W), -10.0, "float32")),
         "depth": (("y", "x"), np.full((H, W), 10.0, "float32")),
         "depth_p25": (("y", "x"), np.full((H, W), 8.0, "float32")),
         "depth_p75": (("y", "x"), np.full((H, W), 12.0, "float32"))},
        coords={"y": ys, "x": xs}))
    _write(project, "LANDCOVER", f"{AOI}.nc", xr.Dataset(
        {"landcover": (("y", "x"), np.full((H, W), 80, "int16")),
         "water": (("y", "x"), np.ones((H, W), "float32"))}, coords={"y": ys, "x": xs}))

    # Day 0: QC good -> valid.   Day 1: QC bad (bits 0-1 == 0b11) -> NOT valid, even though
    # it is cloud-free and over water. That rule comes from the SPEC, nowhere else.
    for i, (day, qual) in enumerate(zip(days, [0.0, 3.0])):
        _write(project, "VIIRS", f"{AOI}_{day.strftime('%Y%m%d')}T2000{i}0.nc", xr.Dataset(
            {"sst": t3(291.0), "cloud": t3(0.0),
             "water": t3(1.0),                      # normal polarity: 1 == water
             "quality": t3(qual)},
            coords={"time": [day], "y": ys, "x": xs}))

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    # Every channel the sensor family generates, present for a sensor nobody hand-wired.
    for ch in ("viirs_sst", "viirs_cloud", "viirs_valid", "viirs_hour"):
        assert ch in ds.data_vars, f"{ch} missing"

    assert np.nanmean(ds["viirs_sst"].isel(time=0).values) == pytest.approx(291.0)
    assert ds["viirs_hour"].isel(time=0).item() == pytest.approx(20.0)

    # The spec's validity rule was honoured: QC-good day valid, QC-bad day not.
    assert int(ds["viirs_valid"].isel(time=0).values.sum()) == H * W
    assert int(ds["viirs_valid"].isel(time=1).values.sum()) == 0

    # ...and the three existing sensors are untouched by its arrival.
    for pre in ("eco", "lst", "modis"):
        assert f"{pre}_sst" in ds.data_vars


def test_the_new_sensor_is_provenanced_in_the_assembled_cube(registered, project):
    """A channel that ships with a blank provenance record is exactly what the registry
    exists to prevent -- so check it on the real cube, not just the mapping function."""
    import json

    g = grid.project_grids(project)[AOI]
    days = pd.date_range("2026-06-01", "2026-06-02", freq="D")
    H, W = g.height, g.width
    xs, ys = g.xy_centers()
    _write(project, "VIIRS", f"{AOI}_20260601T200000.nc", xr.Dataset(
        {"sst": (("time", "y", "x"), np.full((1, H, W), 291.0, "float32")),
         "cloud": (("time", "y", "x"), np.zeros((1, H, W), "float32")),
         "water": (("time", "y", "x"), np.ones((1, H, W), "float32")),
         "quality": (("time", "y", "x"), np.zeros((1, H, W), "float32"))},
        coords={"time": [days[0]], "y": ys, "x": xs}, attrs={"source": "VIIRS L2P"}))

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    prov = json.loads(ds.attrs["provenance"])
    assert prov["viirs_sst"]["inputs"] == ["viirs"]
    assert "VIIRS L2P" in prov["viirs_sst"]["sources"]
