"""THE ACID TEST for a NON-SENSOR covariate (mirrors test_add_a_sensor.py).

Before the contributor protocol, the per-overpass sensor family was registry-driven but every
OTHER product was a hand-written block in `assemble_aoi`. A brand-new non-sensor covariate
acquired to disk from its ProductSpec alone and then vanished from every cube -- SILENTLY --
until someone remembered to hand-wire it into the assembler. That is the exact class of silent
omission the registry exists to eliminate.

So: register a synthetic daily covariate, add ONE `Contributor`, and assert (1) an assembled
cube gains its channel with correct values, and -- the whole point -- (2) forgetting the
contributor is now a HARD ERROR at import, not a silent no-op.
"""

from enum import Enum

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data import grid, products, provenance
from coastal_sst_data.config import parse_config
from coastal_sst_data.processes import datacube

AOI = "aoi1"


class NewProduct(str, Enum):
    """Stands in for a new member of the DataProduct enum (closed at runtime). In real use,
    adding a covariate means an enum member AND a ProductSpec -- both in products.py."""
    chl = "chl"


# A daily-raster covariate (chlorophyll, say): non-sensor, so under the OLD design it would
# have needed a hand-written block in assemble_aoi and would have been dropped without one.
CHL = products.ProductSpec(
    product=NewProduct.chl,
    dir="CHL",
    kind=products.Kind.DAILY_RASTER,
    module="coastal_sst_data.processes.mur",     # stand-in: we only assemble, never acquire
    required_vars=("chl",),
    coverage_channel="chl",
    provenance_inputs=("chl",),
)


def _contribute_chl(ctx: datacube.AssemblyContext) -> None:
    """The ONE new wiring: read the daily CHL files and emit a `chl` channel."""
    d = ctx.adir("chl")
    ctx.emit("chl", ("time", "y", "x"),
             datacube.load_daily_sensor(d, ctx.aid, ctx.days, ctx.H, ctx.W, "chl"),
             long_name="chlorophyll-a (synthetic test covariate)")


CHL_CONTRIBUTOR = datacube.Contributor("chl", (), (), _contribute_chl)


@pytest.fixture
def with_chl_product(monkeypatch):
    """Register CHL in the registry and re-derive the tables computed from it -- but do NOT
    add its contributor. This is the state a forgetful contributor leaves behind."""
    reg = products.REGISTRY + (CHL,)
    monkeypatch.setattr(products, "REGISTRY", reg)
    monkeypatch.setattr(products, "BY_PRODUCT", {s.product: s for s in reg})
    monkeypatch.setattr(datacube, "PRODUCT_DIRS", products.product_dirs())
    monkeypatch.setattr(datacube, "DAILY_CHANNELS",
                        {s.product.value: s.coverage_channel for s in reg if s.coverage_channel})
    monkeypatch.setitem(provenance._EXACT, "chl", ["chl"])
    return reg


@pytest.fixture
def with_chl_contributor(with_chl_product, monkeypatch):
    """...and then add the ONE contributor. Now the cube should gain the channel."""
    monkeypatch.setattr(datacube, "CONTRIBUTORS", datacube.CONTRIBUTORS + (CHL_CONTRIBUTOR,))
    return with_chl_product


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


def _write_chl(project, g, days, *, base=0.5):
    """Synthetic daily CHL aligned files: a distinct constant per day, so a wrong day would
    show up as a wrong value rather than merely present/absent."""
    xs, ys = g.xy_centers()
    H, W = g.height, g.width
    for i, day in enumerate(days):
        ds = xr.Dataset({"chl": (("time", "y", "x"), np.full((1, H, W), base + i, "float32"))},
                        coords={"time": [day], "y": ys, "x": xs})
        d = project.output_dir / "CHL" / "aligned" / AOI
        d.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(d / f"{AOI}_{day.strftime('%Y%m%d')}.nc")


# --------------------------------------------------------------------------- #
# (1) the loud-omission invariant -- the whole reason for the protocol
# --------------------------------------------------------------------------- #
def test_a_covariate_without_a_contributor_is_a_HARD_ERROR(with_chl_product):
    """The trap this closes: CHL acquires to disk but has no cube contributor. Under the old
    hand-wired assembler it would be silently absent from every cube. Now it raises at import."""
    with pytest.raises(RuntimeError, match=r"chl: no cube contributor"):
        datacube._check_contributors()


def test_cube_opt_out_lets_a_product_have_no_channel_on_purpose(monkeypatch):
    """A product that acquires but deliberately ships no cube channel says so with
    cube_opt_out=True -- and then the invariant is satisfied without a contributor."""
    from dataclasses import replace
    opted_out = replace(CHL, cube_opt_out=True)
    monkeypatch.setattr(products, "REGISTRY", products.REGISTRY + (opted_out,))
    datacube._check_contributors()          # must NOT raise


def test_registering_the_contributor_satisfies_the_invariant(with_chl_contributor):
    datacube._check_contributors()          # must NOT raise once wired


# --------------------------------------------------------------------------- #
# (2) ...and an assembled cube actually gains its channel
# --------------------------------------------------------------------------- #
def test_an_assembled_cube_gains_the_covariate_channel(with_chl_contributor, project):
    """The payoff: write CHL aligned files, and the assembler -- with only the spec + one
    contributor added -- produces its channel with the right per-day values, plus it is
    coverage-checked like any other daily product."""
    g = grid.project_grids(project)[AOI]
    days = pd.date_range("2026-06-01", "2026-06-02", freq="D")
    _write_chl(project, g, days, base=0.5)

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)

    assert "chl" in ds.data_vars
    assert ds["chl"].dims == ("time", "y", "x")
    assert np.nanmean(ds["chl"].isel(time=0).values) == pytest.approx(0.5)
    assert np.nanmean(ds["chl"].isel(time=1).values) == pytest.approx(1.5)   # day 1, not stale
    assert ds["chl"].attrs["long_name"].startswith("chlorophyll")

    # It is judged on daily coverage (both days landed), like MUR/met/tides.
    import json
    assert json.loads(ds.attrs["coverage"])["chl"]["fraction"] == 1.0

    # And the existing flat sensors are untouched by its arrival. (ECOSTRESS is a stacked-data
    # sensor now -- its per-version channels appear only where a version tree exists, so it is
    # exercised in test_datacube, not here where no eco scene is written.)
    for pre in ("lst", "modis"):
        assert f"{pre}_sst" in ds.data_vars


def test_a_missing_day_is_a_nan_slice_not_a_dropped_channel(with_chl_contributor, project):
    """A lost day must be a NaN slice on a full-length axis (a countable coverage gap), not a
    silently shorter cube -- the same guarantee every daily product gets."""
    g = grid.project_grids(project)[AOI]
    days = pd.date_range("2026-06-01", "2026-06-02", freq="D")
    _write_chl(project, g, [days[0]], base=0.5)              # only day 0 written

    ds = datacube.assemble_aoi(g, datacube._build_eff(project), days)
    assert ds.sizes["time"] == 2
    assert np.isfinite(ds["chl"].isel(time=0).values).any()
    assert np.isnan(ds["chl"].isel(time=1).values).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
