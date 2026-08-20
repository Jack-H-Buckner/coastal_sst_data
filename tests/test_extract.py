"""processes/extract.py -- assembled cubes -> one long-format table of point time series.

Almost every bug this stage can have returns a finite, plausible number from the wrong
place, so each test below pins ONE such failure. The synthetic cube (see conftest.cube_dir)
is built from anisotropic ramps for exactly that reason: `eco_sst` varies with ROW and time,
`elevation_cudem` with COLUMN, so a y-flip and a row/col transposition each change a value
that a correct read would not.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_sst_data import points, store
from coastal_sst_data.config import ExtractChannel, parse_config
from coastal_sst_data.processes import extract

from .conftest import NAN_COL, NAN_ROW, build_cube, pixel_lonlat


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _project(tmp_path, points_path, channels, fmt="csv", **extra):
    """A Project whose single AoI is conftest's `aoi_grid`, plus an `extract` block."""
    return parse_config({
        "name": "t",
        "output_dir": str(tmp_path),
        "time": {"start_date": "2023-06-01", "end_date": "2023-06-04"},
        "auth": {"earthdata": {"auth_strategy": "netrc"}},
        "products": {"mur": None},
        "regions": [{"name": "r", "areas": [
            {"name": "test_aoi", "center_lat": 45.52, "center_lon": -123.925,
             "buffer_ns_km": 8.0, "buffer_ew_km": 8.0}]}],
        "extract": {"points": str(points_path), "format": fmt,
                    "channels": channels, **extra},
    })


def _sites(tmp_path, g, wanted):
    """A points CSV placing each named site at the centre of a given (row, col)."""
    rows = []
    for name, (r, c) in wanted.items():
        lon, lat = pixel_lonlat(g, r, c)
        rows.append({"id": name, "lat": lat, "lon": lon})
    path = tmp_path / "sites.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(tmp_path, g, wanted, channels, fmt="csv", out=None, **kw):
    """Extract and return the table as a DataFrame."""
    pcsv = _sites(tmp_path, g, wanted)
    project = _project(tmp_path, pcsv, channels, fmt=fmt)
    extract.extract(project, out=out, **kw)
    out = out or tmp_path / "extract" / f"points.{fmt}"
    return pd.read_csv(out) if str(out).endswith(".csv") else pd.read_parquet(out)


def _val(df, point_id, variable, stat, when=None):
    m = ((df["point_id"] == point_id) & (df["variable"] == variable) & (df["stat"] == stat))
    sub = df[m]
    if when is not None:
        sub = sub[pd.to_datetime(sub["time"]) == pd.Timestamp(when)]
    assert len(sub) == 1, f"expected one row, got {len(sub)}"
    return float(sub["value"].iloc[0])


# --------------------------------------------------------------------------- #
# Which pixel
# --------------------------------------------------------------------------- #
def test_nearest_reads_the_pixel_the_point_falls_in(tmp_path, aoi_grid, cube_dir):
    """`eco_sst` is 1000*t + row, so the value NAMES the cell it came from.

    An off-by-one in the affine inversion, or a pixel-edge/centre confusion, shifts this by
    exactly 1 -- a difference indistinguishable from real spatial variation in a real cube.
    """
    df = _run(tmp_path, aoi_grid, {"p": (37, 12)}, {"eco_sst": None})
    for i, day in enumerate(pd.date_range("2023-06-01", periods=4)):
        assert _val(df, "p", "eco_sst", "nearest", day) == 1000 * i + 37


def test_north_and_south_points_differ_as_y_descends(tmp_path, aoi_grid, cube_dir):
    """Row 5 is NORTH of row 150 -- a flipped y axis would swap these two values."""
    df = _run(tmp_path, aoi_grid, {"north": (5, 50), "south": (150, 50)}, {"eco_sst": None})
    day = "2023-06-01"
    assert _val(df, "north", "eco_sst", "nearest", day) == 5
    assert _val(df, "south", "eco_sst", "nearest", day) == 150


def test_static_channel_varies_along_x(tmp_path, aoi_grid, cube_dir):
    """`elevation_cudem` is the COLUMN index; a row/col transposition changes it."""
    df = _run(tmp_path, aoi_grid, {"p": (37, 12)}, {"elevation_cudem": None})
    assert _val(df, "p", "elevation_cudem", "nearest") == 12


# --------------------------------------------------------------------------- #
# The neighbourhood
# --------------------------------------------------------------------------- #
def test_radius_is_a_disc_in_metres(tmp_path, aoi_grid, cube_dir):
    """250 m at a 100 m posting selects 21 pixels.

    Three wrong answers this rules out at once: 25 would be a square box (whose corners
    reach 354 m), 1 would be `radius_m` read as pixels, and the whole grid would be it read
    as degrees.
    """
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)},
              {"eco_sst": {"radius_m": 250, "stat": ["count", "nanmean"]}})
    assert _val(df, "p", "eco_sst", "count", "2023-06-01") == 21


def test_disc_edge_is_symmetric_when_pixels_sit_exactly_on_it(tmp_path, aoi_grid, cube_dir):
    """A 300 m disc on a 100 m grid puts four pixel centres at EXACTLY the radius.

    Their computed distances come out as 300.0000000009 and 299.9999999991 because the
    point has been through a lon/lat round-trip, so a bare `<=` admits one and rejects its
    mirror image -- a neighbourhood made lopsided, systematically, by a nanometre of float
    noise. The count would read 27 where the geometry says 29, and the mean would carry the
    bias of whichever side won.
    """
    df = _run(tmp_path, aoi_grid, {"p": (77, 42)},
              {"eco_sst": {"radius_m": 300, "stat": ["count", "nanmean"]}})
    day = "2023-06-01"
    assert _val(df, "p", "eco_sst", "count", day) == 29
    # eco_sst is the row index, so a symmetric disc averages to the point's own row exactly
    assert _val(df, "p", "eco_sst", "nanmean", day) == pytest.approx(77.0)


def test_radius_smaller_than_a_pixel_returns_the_containing_pixel(tmp_path, aoi_grid,
                                                                  cube_dir, caplog):
    """A 50 m radius on a 100 m grid contains no pixel CENTRE.

    Without the always-include-the-containing-pixel rule the reduction runs over an empty
    set and the whole column is NaN -- which reads exactly like a channel that was cloudy
    for the entire record. This is the user's own documented example, so it also warns.
    """
    with caplog.at_level("WARNING"):
        df = _run(tmp_path, aoi_grid, {"p": (60, 60)},
                  {"eco_sst": {"radius_m": 50, "stat": ["nanmean", "count"]}})
    assert _val(df, "p", "eco_sst", "count", "2023-06-01") == 1
    assert _val(df, "p", "eco_sst", "nanmean", "2023-06-01") == 60
    assert "smaller than the 100 m grid posting" in caplog.text


def test_window_clipped_by_the_grid_edge_reduces_count(tmp_path, aoi_grid, cube_dir):
    """A clipped window still returns a mean -- of a partial disc. `count` is the only tell."""
    df = _run(tmp_path, aoi_grid, {"corner": (0, 0), "middle": (60, 60)},
              {"eco_sst": {"radius_m": 250, "stat": ["count", "nanmean"]}})
    assert _val(df, "corner", "eco_sst", "count", "2023-06-01") < 21
    assert _val(df, "middle", "eco_sst", "count", "2023-06-01") == 21


def test_disc_is_centred_on_the_point_not_its_pixel(tmp_path, aoi_grid, cube_dir):
    """A point at a pixel's centre and one near its corner see different neighbourhoods."""
    xs, ys = aoi_grid.xy_centers()
    from pyproj import Transformer
    inv = Transformer.from_crs(aoi_grid.target_crs, "EPSG:4326", always_xy=True)
    res = aoi_grid.resolution_m
    rows = []
    # Projected y increases NORTHWARD, so a negative offset puts `south` below the centre
    # of pixel row 60 -- while still inside it, so both points have the same `nearest`.
    for name, dy in [("centre", 0.0), ("south", -0.49 * res)]:
        lon, lat = inv.transform(float(xs[60]), float(ys[60]) + dy)
        rows.append({"id": name, "lat": lat, "lon": lon})
    pcsv = tmp_path / "sites.csv"
    pd.DataFrame(rows).to_csv(pcsv, index=False)
    project = _project(tmp_path, pcsv,
                       {"eco_sst": {"radius_m": 150, "stat": ["nearest", "nanmean"]}},
                       fmt="csv")
    extract.extract(project)
    df = pd.read_csv(tmp_path / "extract" / "points.csv")
    day = "2023-06-01"
    # Same cell...
    assert _val(df, "south", "eco_sst", "nearest", day) == 60
    assert _val(df, "centre", "eco_sst", "nearest", day) == 60
    # ...but the southern point's disc reaches further into the higher (southward) rows, so
    # its mean of `1000*t + row` is higher. Centring the disc on the PIXEL instead of the
    # POINT would make these two identical.
    assert (_val(df, "south", "eco_sst", "nanmean", day)
            > _val(df, "centre", "eco_sst", "nanmean", day))


# --------------------------------------------------------------------------- #
# NaN semantics
# --------------------------------------------------------------------------- #
def test_mean_propagates_nan_and_nanmean_does_not(tmp_path, aoi_grid, cube_dir):
    """The whole reason both are offered: they answer different questions."""
    df = _run(tmp_path, aoi_grid, {"p": (NAN_ROW, NAN_COL)},
              {"eco_sst_gappy": {"radius_m": 250,
                                 "stat": ["mean", "nanmean", "count", "count_valid"]}})
    day = "2023-06-01"
    assert np.isnan(_val(df, "p", "eco_sst_gappy", "mean", day))
    assert _val(df, "p", "eco_sst_gappy", "nanmean", day) == NAN_ROW
    assert _val(df, "p", "eco_sst_gappy", "count", day) == 21
    assert _val(df, "p", "eco_sst_gappy", "count_valid", day) == 20


def test_nearest_does_not_substitute_a_finite_neighbour(tmp_path, aoi_grid, cube_dir):
    """The point's own pixel is NaN and every neighbour is finite.

    Filling from just offshore is precisely how a validation set acquires a warm bias, so
    `nearest` must stay NaN.
    """
    df = _run(tmp_path, aoi_grid, {"p": (NAN_ROW, NAN_COL)},
              {"eco_sst_gappy": {"radius_m": 250, "stat": ["nearest", "nanmean"]}})
    assert np.isnan(_val(df, "p", "eco_sst_gappy", "nearest", "2023-06-01"))
    assert np.isfinite(_val(df, "p", "eco_sst_gappy", "nanmean", "2023-06-01"))


def test_all_nan_neighbourhood_gives_nan_and_zero_valid(tmp_path, aoi_grid, cube_dir):
    ds = build_cube(aoi_grid)
    ds["eco_sst_gappy"].values[:] = np.nan
    zpath = cube_dir / f"{aoi_grid.name}.zarr"
    import shutil
    shutil.rmtree(zpath)
    ds.to_zarr(zpath, mode="w-", consolidated=True)
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)},
              {"eco_sst_gappy": {"radius_m": 250, "stat": ["nanmean", "count_valid", "count"]}})
    day = "2023-06-01"
    assert np.isnan(_val(df, "p", "eco_sst_gappy", "nanmean", day))
    assert _val(df, "p", "eco_sst_gappy", "count_valid", day) == 0
    assert _val(df, "p", "eco_sst_gappy", "count", day) == 21   # pixels were there; data was not


def test_percentile_stat(tmp_path, aoi_grid, cube_dir):
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)},
              {"eco_sst": {"radius_m": 250, "stat": ["p50", "nanmedian"]}})
    day = "2023-06-01"
    assert _val(df, "p", "eco_sst", "p50", day) == _val(df, "p", "eco_sst", "nanmedian", day)


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def test_water_mask_excludes_land_pixels(tmp_path, aoi_grid, cube_dir):
    """`landcover_water` is 1 over the WEST half; a point on the boundary straddles it.

    The counts here are exact, not a comparison, so the test also pins the disc's shape:
    a 250 m disc at 100 m posting spans columns edge-2..edge+2 as 3/5/5/5/3 pixels, and
    masking to `col <= edge` keeps the first three of those -- 13, not 21.
    """
    edge = aoi_grid.width // 2 - 1                     # last water column
    masked = _run(tmp_path, aoi_grid, {"p": (60, edge)},
                  {"elevation_cudem": {"radius_m": 250, "stat": ["count", "nanmean"],
                                       "mask": "water"}})
    unmasked = _run(tmp_path, aoi_grid, {"p": (60, edge)},
                    {"elevation_cudem": {"radius_m": 250, "stat": ["count", "nanmean"]}},
                    out=tmp_path / "extract" / "unmasked.csv")

    assert _val(unmasked, "p", "elevation_cudem", "count") == 21
    assert _val(masked, "p", "elevation_cudem", "count") == 13
    # elevation == the column index, so dropping the eastern (land) half lowers the mean
    assert _val(masked, "p", "elevation_cudem", "nanmean") == pytest.approx(
        (3 * (edge - 2) + 5 * (edge - 1) + 5 * edge) / 13)
    assert _val(unmasked, "p", "elevation_cudem", "nanmean") == pytest.approx(float(edge))


def test_all_land_neighbourhood_is_nan_with_zero_count(tmp_path, aoi_grid, cube_dir):
    """The mask WINS over the always-include-the-containing-pixel rule.

    A point whose own cell is land must contribute nothing rather than smuggle a land value
    into a water-only statistic -- and that has to be visible, not substituted.
    """
    df = _run(tmp_path, aoi_grid, {"p": (60, 150)},         # deep in the land half
              {"elevation_cudem": {"radius_m": 250, "stat": ["nanmean", "count"],
                                   "mask": "water"}})
    assert np.isnan(_val(df, "p", "elevation_cudem", "nanmean"))
    assert _val(df, "p", "elevation_cudem", "count") == 0


def test_nearest_ignores_the_mask(tmp_path, aoi_grid, cube_dir):
    """`nearest` is one specific pixel by definition; the mask cannot apply to it."""
    df = _run(tmp_path, aoi_grid, {"p": (60, 150)},
              {"elevation_cudem": {"radius_m": 250, "stat": ["nearest", "nanmean"],
                                   "mask": "water"}})
    assert _val(df, "p", "elevation_cudem", "nearest") == 150


def test_missing_mask_channel_fails_loudly(tmp_path, aoi_grid, cube_dir):
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv,
                       {"eco_sst": {"radius_m": 250, "stat": "nanmean", "mask": "no_such"}})
    with pytest.raises(ValueError, match="mask 'no_such'"):
        extract.extract(project)


# --------------------------------------------------------------------------- #
# Table shape
# --------------------------------------------------------------------------- #
def test_multiple_stats_are_separate_rows(tmp_path, aoi_grid, cube_dir):
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)},
              {"eco_sst": {"radius_m": 250, "stat": ["nanmean", "nanstd", "count"]}})
    stats = set(df[df["variable"] == "eco_sst"]["stat"])
    assert stats == {"nanmean", "nanstd", "count"}
    assert list(df.columns) == extract.COLUMNS


def test_nearest_row_records_radius_zero(tmp_path, aoi_grid, cube_dir):
    """`radius_m` reports what the row USED, never what the config declared."""
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)},
              {"eco_sst": {"radius_m": 250, "stat": ["nearest", "nanmean"]}})
    r = df[df["stat"] == "nearest"]["radius_m"].unique()
    assert list(r) == [0.0]
    assert list(df[df["stat"] == "nanmean"]["radius_m"].unique()) == [250.0]


def test_static_channel_gets_one_row_with_null_time(tmp_path, aoi_grid, cube_dir):
    """Repeating a constant down the time axis would multiply the file by T for no gain."""
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)}, {"elevation_cudem": None})
    sub = df[df["variable"] == "elevation_cudem"]
    assert len(sub) == 1
    assert pd.isna(sub["time"].iloc[0])


def test_1d_channel_is_one_row_per_time_and_equal_for_every_point(tmp_path, aoi_grid,
                                                                  cube_dir):
    df = _run(tmp_path, aoi_grid, {"a": (30, 30), "b": (90, 90)}, {"tide_coops": None})
    sub = df[df["variable"] == "tide_coops"]
    assert len(sub) == 8                                     # 2 points x 4 days
    a = sub[sub["point_id"] == "a"].sort_values("time")["value"].tolist()
    b = sub[sub["point_id"] == "b"].sort_values("time")["value"].tolist()
    assert a == b == [0.0, 1.0, 2.0, 3.0]


def test_overpass_hour_extracts_with_nan_on_days_with_no_overpass(tmp_path, aoi_grid,
                                                                  cube_dir):
    """`<sensor>_hour` is the channel that prompted the 1-D error report.

    Bare, it must simply work -- and a day the sensor saw nothing must come through as NaN
    rather than as a filled-in hour, because a fabricated overpass time is worse than a gap.
    """
    df = _run(tmp_path, aoi_grid, {"p": (30, 30)}, {"lst_hour": None})
    sub = df[df["variable"] == "lst_hour"].sort_values("time")
    assert len(sub) == 4
    assert set(sub["stat"]) == {"nearest"} and set(sub["radius_m"]) == {0.0}
    vals = sub["value"].tolist()
    assert vals[0] == 10.0 and vals[2] == 12.0 and vals[3] == 13.0
    assert np.isnan(vals[1])                                 # no overpass on day 2


def test_lat_lon_are_the_input_coordinates(tmp_path, aoi_grid, cube_dir):
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    want = pd.read_csv(pcsv)
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)}, {"eco_sst": None})
    assert df["lat"].iloc[0] == pytest.approx(float(want["lat"].iloc[0]))
    assert df["lon"].iloc[0] == pytest.approx(float(want["lon"].iloc[0]))


def test_primary_key_is_unique_and_order_is_deterministic(tmp_path, aoi_grid, cube_dir):
    ch = {"eco_sst": {"radius_m": 250, "stat": ["nanmean", "count"]},
          "elevation_cudem": None, "tide_coops": None}
    a = _run(tmp_path, aoi_grid, {"p": (60, 60), "q": (30, 30)}, ch)
    b = _run(tmp_path, aoi_grid, {"p": (60, 60), "q": (30, 30)}, ch,
             overwrite=True)
    assert not a.duplicated(subset=extract.KEY).any()
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# Loud failures
# --------------------------------------------------------------------------- #
def test_missing_channel_fails_with_a_suggestion(tmp_path, aoi_grid, cube_dir):
    """A silently-absent column reads as 'that variable was all-NaN'."""
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv, {"eco_sstt": None})
    with pytest.raises(ValueError, match="did you mean eco_sst"):
        extract.extract(project)


def test_cube_grid_mismatch_fails_loudly(tmp_path, aoi_grid, cube_dir):
    """A cube written on a different grid than the config now computes."""
    ds = build_cube(aoi_grid).isel(y=slice(0, 40), x=slice(0, 40))
    zpath = cube_dir / f"{aoi_grid.name}.zarr"
    import shutil
    shutil.rmtree(zpath)
    ds.to_zarr(zpath, mode="w-", consolidated=True)
    pcsv = _sites(tmp_path, aoi_grid, {"p": (10, 10)})
    project = _project(tmp_path, pcsv, {"eco_sst": None})
    with pytest.raises(ValueError, match="does not match the grid this config computes"):
        extract.extract(project)


def test_ascending_y_cube_is_rejected(tmp_path, aoi_grid, cube_dir):
    """Every grid this package writes runs y DOWN; an ascending one flips north for south."""
    ds = build_cube(aoi_grid).isel(y=slice(None, None, -1))
    zpath = cube_dir / f"{aoi_grid.name}.zarr"
    import shutil
    shutil.rmtree(zpath)
    ds.to_zarr(zpath, mode="w-", consolidated=True)
    pcsv = _sites(tmp_path, aoi_grid, {"p": (10, 10)})
    project = _project(tmp_path, pcsv, {"eco_sst": None})
    with pytest.raises(ValueError, match="ASCENDS"):
        extract.extract(project)


def test_missing_crs_attr_warns_and_assumes_the_config(tmp_path, aoi_grid, cube_dir, caplog):
    ds = build_cube(aoi_grid)
    del ds.attrs["crs"]
    zpath = cube_dir / f"{aoi_grid.name}.zarr"
    import shutil
    shutil.rmtree(zpath)
    ds.to_zarr(zpath, mode="w-", consolidated=True)
    with caplog.at_level("WARNING"):
        df = _run(tmp_path, aoi_grid, {"p": (37, 12)}, {"eco_sst": None})
    assert "no `crs` attr" in caplog.text
    assert _val(df, "p", "eco_sst", "nearest", "2023-06-01") == 37


def test_no_points_inside_any_aoi_is_an_error(tmp_path, aoi_grid, cube_dir):
    """Not a zero-row file: that is indistinguishable from a mistyped coordinate column."""
    pcsv = tmp_path / "sites.csv"
    pd.DataFrame([{"id": "far", "lat": 0.0, "lon": 0.0}]).to_csv(pcsv, index=False)
    project = _project(tmp_path, pcsv, {"eco_sst": None})
    with pytest.raises(SystemExit, match="none of the 1 point"):
        extract.extract(project)


def test_no_channels_is_an_error(tmp_path, aoi_grid, cube_dir):
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv, {})
    with pytest.raises(SystemExit, match="no channels configured"):
        extract.extract(project)


def test_1d_channel_with_a_radius_fails_with_a_usable_message(tmp_path, aoi_grid, cube_dir):
    """The message has to say the channel IS extractable, and show the exact fix.

    Its predecessor said only that `radius_m`/`stat`/`mask` "have nothing to reduce over",
    and was read as "extraction does not support overpass times" -- so what is asserted here
    is the wording, not just that something was raised.
    """
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv,
                       {"tide_coops": {"radius_m": 250, "stat": "nanmean"}})
    with pytest.raises(ValueError) as exc:
        extract.extract(project)
    msg = str(exc.value)
    assert "ARE extracted" in msg                       # the capability, stated outright
    assert "radius_m: 250" in msg and "stat: nanmean" in msg   # the values to go and edit
    assert "    channels:\n      tide_coops:\n" in msg  # a copy-pasteable fix
    assert "one row per date" in msg                    # what you get if you take it


def test_every_1d_channel_is_named_in_one_error(tmp_path, aoi_grid, cube_dir):
    """A real cube has ~11 1-D channels, so one uniform config block trips all of them.

    Raising on the first would mean a dozen edit-and-re-run cycles, each paying for the
    whole point-assignment pass before failing again on the next name.
    """
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv, {
        "eco_sst": {"radius_m": 250, "stat": "nanmean"},          # legitimate, unaffected
        "tide_coops": {"radius_m": 250, "stat": "nanmean"},
        "lst_hour": {"radius_m": 250, "stat": "nanmean"},
    })
    with pytest.raises(ValueError) as exc:
        extract.extract(project)
    msg = str(exc.value)
    assert "2 channel(s)" in msg
    assert "tide_coops" in msg and "lst_hour" in msg
    assert "eco_sst" not in msg                          # the 3-D channel is not implicated


def test_missing_cube_skips_with_a_warning(tmp_path, aoi_grid, caplog):
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv, {"eco_sst": None})
    with caplog.at_level("WARNING"), pytest.raises(SystemExit, match="no cube produced"):
        extract.extract(project)
    assert "run `assemble` first" in caplog.text


def test_unknown_aoi_fails_before_any_cube_is_opened(tmp_path, aoi_grid, cube_dir):
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv, {"eco_sst": None})
    with pytest.raises(SystemExit, match="not found in config"):
        extract.extract(project, aois=["nope"])


# --------------------------------------------------------------------------- #
# Output, formats and re-runs
# --------------------------------------------------------------------------- #
def test_csv_and_parquet_agree(tmp_path, aoi_grid, cube_dir):
    pytest.importorskip("pyarrow")
    ch = {"eco_sst": {"radius_m": 250, "stat": ["nanmean", "count"]}, "tide_coops": None}
    a = _run(tmp_path, aoi_grid, {"p": (60, 60)}, ch, fmt="csv")
    b = _run(tmp_path, aoi_grid, {"p": (60, 60)}, ch, fmt="parquet")
    assert list(a.columns) == list(b.columns) == extract.COLUMNS
    np.testing.assert_allclose(a["value"].to_numpy(), b["value"].to_numpy())


def test_parquet_keeps_dtypes_and_leaves_no_scratch(tmp_path, aoi_grid, cube_dir):
    pytest.importorskip("pyarrow")
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)},
              {"eco_sst": None, "elevation_cudem": None}, fmt="parquet")
    assert str(df["time"].dtype).startswith("datetime64")
    assert df["value"].dtype == "float64"
    out = tmp_path / "extract" / "points.parquet"
    assert out.exists() and not list(out.parent.glob("*.part-*"))


def test_parquet_without_pyarrow_raises_with_advice(tmp_path, aoi_grid, cube_dir,
                                                    monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyarrow", None)     # import pyarrow -> ImportError
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv, {"eco_sst": None}, fmt="parquet")
    with pytest.raises(ImportError, match="pyarrow"):
        extract.extract(project)


def test_existing_output_is_skipped_without_overwrite(tmp_path, aoi_grid, cube_dir, caplog):
    df = _run(tmp_path, aoi_grid, {"p": (60, 60)}, {"eco_sst": None})
    out = tmp_path / "extract" / "points.csv"
    stamp = out.stat().st_mtime_ns
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    with caplog.at_level("INFO"):
        extract.extract(_project(tmp_path, pcsv, {"eco_sst": None}))
    assert out.stat().st_mtime_ns == stamp
    assert "use --overwrite" in caplog.text
    assert len(df) == 4


def test_dry_run_writes_nothing(tmp_path, aoi_grid, cube_dir, caplog):
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    project = _project(tmp_path, pcsv, {"eco_sst": None, "tide_coops": None})
    with caplog.at_level("INFO"):
        extract.extract(project, dry_run=True)
    assert not (tmp_path / "extract").exists()
    assert "dry run" in caplog.text


def test_aoi_subset_gets_its_own_filename(tmp_path, aoi_grid, cube_dir):
    """`--aoi one` must not overwrite the complete table from a full run."""
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    extract.extract(_project(tmp_path, pcsv, {"eco_sst": None}))
    extract.extract(_project(tmp_path, pcsv, {"eco_sst": None}), aois=["test_aoi"])
    assert (tmp_path / "extract" / "points.csv").exists()
    assert (tmp_path / "extract" / "points_test_aoi.csv").exists()


def test_out_path_overrides_everything(tmp_path, aoi_grid, cube_dir):
    pcsv = _sites(tmp_path, aoi_grid, {"p": (60, 60)})
    dest = tmp_path / "elsewhere" / "mine.csv"
    extract.extract(_project(tmp_path, pcsv, {"eco_sst": None}), out=dest)
    assert dest.exists()


def test_points_file_argument_overrides_the_config(tmp_path, aoi_grid, cube_dir):
    cfg_pts = _sites(tmp_path, aoi_grid, {"from_config": (10, 10)})
    other = tmp_path / "other.csv"
    lon, lat = pixel_lonlat(aoi_grid, 60, 60)
    pd.DataFrame([{"id": "from_flag", "lat": lat, "lon": lon}]).to_csv(other, index=False)
    extract.extract(_project(tmp_path, cfg_pts, {"eco_sst": None}), points_file=other)
    df = pd.read_csv(tmp_path / "extract" / "points.csv")
    assert set(df["point_id"]) == {"from_flag"}


# --------------------------------------------------------------------------- #
# Reading strategy: the union read is an optimisation, never a difference
# --------------------------------------------------------------------------- #
def test_union_and_per_point_reads_agree(tmp_path, aoi_grid, cube_dir):
    """The two strategies in `read_windows` must be indistinguishable in their output."""
    pcsv = _sites(tmp_path, aoi_grid, {"a": (20, 20), "b": (120, 130)})
    ch = {"eco_sst": ExtractChannel(radius_m=250, stat=["nanmean", "count"])}
    pts = points.assign_aois(points.read_points(pcsv), {aoi_grid.name: aoi_grid})
    with xr.open_zarr(cube_dir / f"{aoi_grid.name}.zarr") as ds:
        g = extract.grid_from_cube(ds, aoi_grid)
        big = extract.extract_aoi(ds, g, pts, ch, budget_bytes=1e12)     # union
        small = extract.extract_aoi(ds, g, pts, ch, budget_bytes=1)      # per point
    pd.testing.assert_frame_equal(big, small)


def test_scattered_points_fall_back_and_say_so(tmp_path, aoi_grid, cube_dir, caplog):
    pcsv = _sites(tmp_path, aoi_grid, {"a": (5, 5), "b": (155, 155)})
    ch = {"eco_sst": ExtractChannel(radius_m=250, stat=["nanmean"])}
    pts = points.assign_aois(points.read_points(pcsv), {aoi_grid.name: aoi_grid})
    with xr.open_zarr(cube_dir / f"{aoi_grid.name}.zarr") as ds:
        g = extract.grid_from_cube(ds, aoi_grid)
        with caplog.at_level("INFO"):
            extract.extract_aoi(ds, g, pts, ch, budget_bytes=1)
    assert "reading each point's window separately" in caplog.text


# --------------------------------------------------------------------------- #
# store.write_table
# --------------------------------------------------------------------------- #
def test_write_table_is_atomic_on_failure(tmp_path):
    """A writer that dies must leave the previous table, or none -- never half of one."""
    df = pd.DataFrame({"a": [1, 2]})
    dest = tmp_path / "out"
    store.write_table(df, dest, "t", "csv")
    before = (dest / "t.csv").read_text()

    class Boom(pd.DataFrame):
        def to_csv(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        store.write_table(Boom({"a": [9]}), dest, "t", "csv")
    assert (dest / "t.csv").read_text() == before
    assert not list(dest.glob("*.part-*"))


def test_write_table_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unknown table format"):
        store.write_table(pd.DataFrame({"a": [1]}), tmp_path, "t", "feather")
