#!/usr/bin/env python3
"""
coastal_sst_data -- ECOSTRESS cloud filtering (post-assembly preprocess steps).

Three composable `preprocess` steps that screen cloud-contaminated ECOSTRESS pixels into a
SCREENED product, `<sensor>_sst_<ver>_clean` / `<sensor>_valid_<ver>_clean`, seeded from the raw
channels. They live here rather than in `preprocess.py` to keep that module a thin registry +
orchestrator; `preprocess` imports the `_step_*` functions and registers each as one
`PreprocessStep` (the same contributor protocol as every other step). Nothing here imports
`preprocess` at runtime -- a step only calls methods on the `PreprocessContext` it is handed --
so there is no import cycle.

The screened product is a separate channel because these steps write into the ASSEMBLED cube
(see `processes.preprocess`): filtering `<sensor>_sst_<ver>` in place would destroy the values
the sensor delivered, leaving only the boolean audit flags to say what had been there.

  * filter_clouds       -- screen against a gap-free baseline L4 SST (MUR/CMEMS): a fixed cold
                          offset (`baseline - eco > threshold_k`, ported from oceanSR) OR a
                          distribution-based, seasonally-aware outlier floor
                          (`eco < mean - n_sigma*sigma` from the baseline's own climatology).
  * filter_cloud_cover  -- gate on the meteorological total-cloud-cover field (HRRR/ERA5, in
                          percent): a scene-level rejection AND a per-pixel cutoff.
  * filter_land_clouds  -- screen OVER-LAND pixels against the near-surface air temperature
                          (HRRR/ERA5): drop where `airtemp - sst > threshold_k` on cells the
                          land mask (landcover) or the tide-adjusted water line marks as land.

All three write the SAME screened channels, so they read the WORKING value (`_target`: this run's
partially-filtered product, seeded from the raw channel on the first write) and their drops UNION
regardless of run order; the emitted `*_cloudfiltered` / `*_metcloudfiltered` /
`*_landcloudfiltered` flags keep each drop auditable.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np

from .channels import CLEAN, CORRECTED, CORRECTED_CLEAN, GAPFILLED
from .water_level import EXPOSED         # land classification for the over-land air-temp filter

if TYPE_CHECKING:                          # avoid an import cycle: only a type hint needs it
    from .preprocess import PreprocessContext

log = logging.getLogger(__name__)

T3 = ("time", "y", "x")

_CLOUD_METHODS = ("offset", "sigma")
_STAT_SCOPES = ("pixel", "pooled")
_SEASONALITY = ("off", "harmonic")
_MET_SRC_PREF = ("hrrr", "era5")
_LAND_SOURCES = ("landcover", "water_line")

# The channel suffixes correct_georef writes and the corrected-pass filters produce (the shared
# vocabulary lives in `processes.channels`, which the input scans also key off).
_CORR = CORRECTED                     # correct_georef's shifted raw fields
_CLEAN = CORRECTED_CLEAN              # the re-filtered, corrected product (separate channel)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _as_list(v):
    """A config value that may be a scalar, a list, or absent -> a list (or None)."""
    if v is None:
        return None
    return [v] if isinstance(v, str) else list(v)


def _emitted(ctx: "PreprocessContext", name: str, *, dims=T3, dtype="float32"):
    """A channel emitted THIS run, or None. Never touches the cube on disk."""
    if name not in ctx.channels:
        return None
    d, arr = ctx.channels[name]
    if tuple(d) != tuple(dims):
        return None
    return np.asarray(arr).astype(dtype, copy=False)


def _working(ctx: "PreprocessContext", name: str, *, dims=T3, dtype="float32"):
    """The CURRENT value of an INPUT channel: prefer one already emitted this run (an earlier
    step's output) over the cube. None if absent / wrong-shaped."""
    got = _emitted(ctx, name, dims=dims, dtype=dtype)
    return got if got is not None else ctx.read(name, dims=dims, dtype=dtype)


def _target(ctx: "PreprocessContext", name: str, seed: str | None = None, *,
            dims=T3, dtype="float32"):
    """The partially-filtered value of a TARGET channel: this run's emission if a filter has
    already written it, else the SEED channel it starts from. None if neither is usable.

    This is what lets the filters compose -- whichever runs second sees the first's drops and
    unions onto them -- and it deliberately never falls back to `name` on disk. The stage rewrites
    the cube in place, so `<pre>_sst<ver>_clean` may already be there from an EARLIER run; reading
    it would union this run's drops onto that run's and make the stage non-idempotent, so a fresh
    target always restarts from the raw (or corrected) source.
    """
    got = _emitted(ctx, name, dims=dims, dtype=dtype)
    if got is not None:
        return got
    return None if seed is None else _working(ctx, seed, dims=dims, dtype=dtype)


def _resolve_opts(ctx: "PreprocessContext", key: str, base_key: str | None) -> dict:
    """A step's options, with a corrected-pass variant INHERITING its base filter's config: the
    base key's bag first, the step's own bag layered on top (so `filter_clouds_corrected: {}` reuses
    `filter_clouds`'s method/threshold, and any key it sets overrides)."""
    opts = dict(ctx.step_opts(base_key)) if base_key else {}
    opts.update(ctx.step_opts(key))
    return opts


# One ECOSTRESS version's channel names for a filter pass. `sst`/`valid` are the write targets;
# `cloud` is the native cloud raster (for use_cloud_raster); `flag_base` names the audit flag;
# `seed_*` are the SOURCE channels a fresh target starts from.
_ChannelSet = namedtuple("_ChannelSet", "sst valid cloud flag_base seed_sst seed_valid")


def _emitted_channels(ctx: "PreprocessContext", prefix: str, suffix: str) -> list[str]:
    """Names emitted THIS run (in `ctx.channels`) starting with `prefix` and ending with `suffix`.
    Needed for the corrected pass because `channels_with_prefix` only sees the RAW cube, and the
    `_georef_corrected` channels are produced this run by `correct_georef`."""
    return sorted(n for n in ctx.channels if n.startswith(prefix) and n.endswith(suffix))


def _channel_sets(ctx: "PreprocessContext", pre: str, mode: str) -> list[_ChannelSet]:
    """The channel sets a filter operates on for sensor `pre`.

    Both modes filter a SOURCE into a SEPARATE `_clean` product seeded from it -- only the source
    differs -- so the filters never overwrite an input:

      * "raw" (default): the assembler's `<pre>_sst<ver>` -> `<pre>_sst<ver>_clean`.
      * "corrected": the `<pre>_sst<ver>_georef_corrected` channels `correct_georef` emitted this
        run -> `<pre>_sst<ver>_georef_corrected_clean`.

    Every filter in a pass shares the pass's target, so their drops compose onto it (`_target`).
    """
    if mode == "corrected":
        sets = []
        for cor in _emitted_channels(ctx, f"{pre}_sst", _CORR):
            ver = cor[len(f"{pre}_sst"):-len(_CORR)]            # e.g. "_v002"
            sets.append(_ChannelSet(
                sst=f"{pre}_sst{ver}{_CLEAN}", valid=f"{pre}_valid{ver}{_CLEAN}",
                cloud=f"{pre}_cloud{ver}{_CORR}", flag_base=f"{pre}_sst{ver}{_CLEAN}",
                seed_sst=f"{pre}_sst{ver}{_CORR}", seed_valid=f"{pre}_valid{ver}{_CORR}"))
        return sets
    # `base_channels`, not a plain prefix scan: on a re-run the cube already holds this pass's
    # own `<pre>_sst<ver>_clean` output, and filtering THAT would nest a second `_clean`.
    sets = []
    for sst_name in ctx.base_channels(f"{pre}_sst"):
        ver = sst_name[len(f"{pre}_sst"):]                      # e.g. "_v002"
        sets.append(_ChannelSet(
            sst=f"{pre}_sst{ver}{CLEAN}", valid=f"{pre}_valid{ver}{CLEAN}",
            cloud=f"{pre}_cloud{ver}", flag_base=f"{pre}_sst{ver}{CLEAN}",
            seed_sst=sst_name, seed_valid=f"{pre}_valid{ver}"))
    return sets


def _fold_drop(ctx: "PreprocessContext", cs: _ChannelSet, dropped: np.ndarray, *,
               flag_suffix: str, mask_sst: bool, attrs: dict) -> None:
    """Fold a boolean drop mask (T,y,x) into one channel set's sst/valid, and emit the companion
    flag. Reads the TARGET sst/valid (seeding from the source on first write, so filters compose)
    and re-emits: `cs.valid` &= ~dropped, `cs.sst` NaN'd (if mask_sst), and
    `cs.flag_base<flag_suffix>` (uint8, 1 = dropped)."""
    valid = _target(ctx, cs.valid, cs.seed_valid, dtype="uint8")
    if valid is not None:
        ctx.emit(cs.valid, T3, (valid.astype(bool) & ~dropped).astype("uint8"), **attrs)
    if mask_sst:
        eco = _target(ctx, cs.sst, cs.seed_sst)
        if eco is not None:
            out = eco.copy()
            out[dropped] = np.nan
            ctx.emit(cs.sst, T3, out, **attrs)
    ctx.emit(f"{cs.flag_base}{flag_suffix}", T3, dropped.astype("uint8"),
             flag_values=[0, 1], flag_meanings="kept dropped", **attrs)


# --------------------------------------------------------------------------- #
# Step: filter_clouds -- baseline cold-deviation filter (ported from oceanSR).
# --------------------------------------------------------------------------- #
def _step_filter_clouds(ctx: "PreprocessContext", *, key: str = "filter_clouds",
                        base_key: str | None = None, mode: str = "raw") -> None:
    """Screen ECOSTRESS pixels against a gap-free baseline L4 SST (MUR/CMEMS).

    Cloud contamination biases thermal-IR SST COLD, so a pixel far COLDER than the baseline is
    likely cloud. Two methods (`method`):
      * "offset" (default): drop where `baseline - eco > threshold_k` (default 5.0 K; <=0
        disables). Uses the per-day co-located baseline (prefers fill_water's gap-filled one).
      * "sigma": build the baseline's own climatology mean/sigma and drop any eco pixel colder
        than `mean - n_sigma*sigma` (`n_sigma` default 3.0). `stat_scope` is per-`pixel` (default,
        a per-cell climatology) or `pooled`; `seasonality` is `off` (default) or a day-of-year
        `harmonic`. The stats come from the OBSERVED baseline (never fill_water's invented cells).
    NaN operands are kept. One-sided (cold only).
    """
    opts = _resolve_opts(ctx, key, base_key)
    method = str(opts.get("method", "offset")).lower()
    if method not in _CLOUD_METHODS:
        raise ValueError(f"filter_clouds.method {method!r} is invalid; "
                         f"choose one of {list(_CLOUD_METHODS)}")
    baseline_name = opts.get("baseline", "mur_sst")
    sensors = _as_list(opts.get("sensors")) or ["eco"]
    mask_sst = bool(opts.get("mask_sst", True))
    use_cloud = bool(opts.get("use_cloud_raster", False))

    cutoff = None
    base = None
    base_used = baseline_name
    if method == "offset":
        thr = float(opts.get("threshold_k", 5.0))
        if thr <= 0:
            log.info("  filter_clouds: threshold_k<=0 -> disabled; nothing filtered")
            return
        base, base_used = _baseline_field(ctx, baseline_name)
        if base is None:
            log.warning("  filter_clouds: baseline %r absent; nothing filtered", baseline_name)
            return
        note = f"cloud-filtered: {base_used} - eco > {thr} K"
    else:
        n_sigma = float(opts.get("n_sigma", 3.0))
        stat_scope = str(opts.get("stat_scope", "pixel")).lower()
        seasonality = str(opts.get("seasonality", "off")).lower()
        if stat_scope not in _STAT_SCOPES:
            raise ValueError(f"filter_clouds.stat_scope {stat_scope!r} is invalid; "
                             f"choose one of {list(_STAT_SCOPES)}")
        if seasonality not in _SEASONALITY:
            raise ValueError(f"filter_clouds.seasonality {seasonality!r} is invalid; "
                             f"choose one of {list(_SEASONALITY)}")
        cutoff = _baseline_cutoff(ctx, baseline_name, n_sigma, stat_scope, seasonality)
        if cutoff is None:
            log.warning("  filter_clouds: baseline %r absent; nothing filtered", baseline_name)
            return
        note = (f"cloud-filtered: eco < {baseline_name} climatology mean - {n_sigma} sigma "
                f"({stat_scope}, seasonality={seasonality})")

    emitted = False
    for pre in sensors:
        for cs in _channel_sets(ctx, pre, mode):
            eco = _target(ctx, cs.sst, cs.seed_sst)
            if eco is None:
                continue
            with np.errstate(invalid="ignore"):
                if method == "offset":
                    dropped = (base - eco) > thr
                else:
                    dropped = eco < cutoff
            dropped = np.asarray(dropped, bool) & np.isfinite(eco)
            if use_cloud:
                cloud = _working(ctx, cs.cloud, dims=T3)   # corrected pass: the shifted cloud band
                if cloud is not None:
                    dropped |= np.nan_to_num(cloud, nan=0.0) > 0
            _fold_drop(ctx, cs, dropped, flag_suffix="_cloudfiltered",
                       mask_sst=mask_sst, attrs={"preprocess": note,
                                                 "cloud_baseline": str(base_used)})
            emitted = True


def _baseline_field(ctx: "PreprocessContext", name: str):
    """The baseline L4 SST to compare against, and the channel it came from.

    Prefers the gap-filled `<name>_gapfilled` when `fill_water` produced one THIS run (a NaN
    baseline cell filters nothing, so the gaps are worth closing), else the raw `<name>`. It does
    not read a `_gapfilled` left on the cube by an earlier run: which baseline a filter used has
    to follow from the steps this run selected, not from what happens to be on disk.
    """
    got = _emitted(ctx, f"{name}{GAPFILLED}")
    if got is not None:
        return got, f"{name}{GAPFILLED}"
    return ctx.read(name, dims=T3), name


def _harmonic_ok(ctx: "PreprocessContext") -> bool:
    """Can a seasonal harmonic be constrained by this CUBE's day-of-year span?

    A WHOLE-WINDOW question, answered from `ctx.all_days` and cached, never from `ctx.days`.
    Asked of one time block it answers "no" for every block of any sensible size -- a ten-year
    cube would then fit a CONSTANT climatology everywhere while the config said `harmonic`,
    reported only by an INFO line repeated once per block.
    """
    if "harmonic_ok" in ctx.window:
        return ctx.window["harmonic_ok"]
    days = ctx.all_days
    s = ctx.read_all("doy_sin")
    c = ctx.read_all("doy_cos")
    if s is None or c is None:
        doy = days.dayofyear.values.astype("float64")
        s = np.sin(2 * np.pi * doy / 365.25)
        c = np.cos(2 * np.pi * doy / 365.25)
    s = np.asarray(s, "float64")
    c = np.asarray(c, "float64")
    ok = not (len(days) == 0
              or np.ptp(days.dayofyear.values) < 60
              or (np.std(s) < 1e-3 and np.std(c) < 1e-3))
    if not ok:
        log.info("  filter_clouds: DOY span too short for a seasonal harmonic; "
                 "using a constant climatology")
    ctx.window["harmonic_ok"] = ok
    return ok


def _harmonic_design(ctx: "PreprocessContext", seasonality: str) -> np.ndarray:
    """The climatology design matrix X (len(ctx.days), k) for THIS BLOCK's days: a constant
    column (k=1), plus -- for a day-of-year `harmonic` -- annual sin/cos columns (k=3, reusing
    the cube's doy_sin/doy_cos).

    The columns are the block's; the harmonic-vs-constant DECISION is the whole cube's
    (`_harmonic_ok`), so every block fits the same model.
    """
    ones = np.ones(len(ctx.days))
    if seasonality != "harmonic" or not _harmonic_ok(ctx):
        return ones[:, None]
    s = ctx.read("doy_sin", dims=("time",))
    c = ctx.read("doy_cos", dims=("time",))
    if s is None or c is None:
        doy = ctx.days.dayofyear.values.astype("float64")
        s = np.sin(2 * np.pi * doy / 365.25)
        c = np.cos(2 * np.pi * doy / 365.25)
    return np.column_stack([ones, np.asarray(s, "float64"), np.asarray(c, "float64")])


def _batched_solve(XtX: np.ndarray, Xtb: np.ndarray) -> np.ndarray:
    """Solve a per-cell k x k normal system, cell by cell, vectorised: (H,W,k,k) & (H,W,k) ->
    (H,W,k), NaN where the system is singular (a cell with < k finite baseline times)."""
    H, W, k, _ = XtX.shape
    A = XtX.reshape(-1, k, k)
    b = Xtb.reshape(-1, k)
    out = np.full((H * W, k), np.nan)
    dets = np.linalg.det(A)
    ok = np.isfinite(dets) & (np.abs(dets) > 1e-9)
    if ok.any():
        # RHS as an explicit (M,k,1) column so the k=1 case isn't read as a vector stack.
        out[ok] = np.linalg.solve(A[ok], b[ok][:, :, None])[:, :, 0]
    return out.reshape(H, W, k)


_SIGMA = "sigma_climatology"


def _sigma_key(opts: dict) -> tuple:
    """The window-statistic key for a sigma climatology.

    Keyed on the MATHS -- baseline, scope, seasonality -- not on the step. `filter_clouds` and
    `filter_clouds_corrected` fit the same climatology (the corrected geometry never touches the
    baseline), so keying this way lets them share one prepass instead of streaming the cube
    twice over. `n_sigma` is deliberately absent: it scales the finished sigma at evaluation
    time and does not change the fit.
    """
    return (_SIGMA, str(opts.get("baseline", "mur_sst")),
            str(opts.get("stat_scope", "pixel")).lower(),
            str(opts.get("seasonality", "off")).lower())


def sigma_accumulate(ctx: "PreprocessContext", pass_i: int, opts: dict) -> None:
    """Fold ONE block into the baseline's climatology accumulators.

    Pass 0 builds the masked normal equations; pass 1 the squared residuals about the fit those
    equations gave. Every term is a plain sum over `t`, so summing per block and adding is the
    same arithmetic in a different order -- the blocked climatology is exact, not approximate.

    TWO passes rather than one. The identity `ssr = sum(b^2) - beta.Xtb` would let a single pass
    do it, but the baseline is in KELVIN: `b^2` is ~8e4 per sample against a residual variance
    of order 1, so the subtraction cancels away the answer. Re-reading one channel is cheap;
    a wrong sigma is not.
    """
    base = ctx.read(opts["_name"], dims=T3)
    if base is None:
        return
    acc = ctx.window.setdefault(_sigma_key(opts), {})
    base = base.astype("float64", copy=False)
    X = _harmonic_design(ctx, opts["_seasonality"])
    k = X.shape[1]
    finite = np.isfinite(base)
    M = finite.astype("float64")
    pooled = opts["_scope"] == "pooled"

    if pass_i == 0:
        Bz = np.where(finite, base, 0.0)
        if pooled:
            acc["XtX"] = acc.get("XtX", 0.0) + np.einsum("ti,tj,tyx->ij", X, X, M)
            acc["Xtb"] = acc.get("Xtb", 0.0) + np.einsum("ti,tyx->i", X, Bz)
            acc["cnt"] = acc.get("cnt", 0.0) + float(M.sum())
        else:
            acc["XtX"] = acc.get("XtX", 0.0) + np.einsum("ti,tj,tyx->yxij", X, X, M)
            acc["Xtb"] = acc.get("Xtb", 0.0) + np.einsum("ti,tyx->yxi", X, Bz)
            acc["cnt"] = acc.get("cnt", 0.0) + M.sum(axis=0)
        acc["k"] = k
        return

    beta = acc.get("beta")
    if beta is None:
        return
    if pooled:
        mu_t = X @ beta                                        # (T,)
        acc["ssr"] = acc.get("ssr", 0.0) + float(
            np.sum(np.where(finite, (base - mu_t[:, None, None]) ** 2, 0.0)))
    else:
        mu = np.einsum("ti,yxi->tyx", X, beta)                 # (block,H,W)
        acc["ssr"] = acc.get("ssr", 0.0) + np.sum(
            np.where(finite, (base - mu) ** 2, 0.0), axis=0)


def sigma_reduce(ctx: "PreprocessContext", pass_i: int, opts: dict) -> None:
    """Turn one pass's accumulators into the statistic the next pass (or the step) needs."""
    acc = ctx.window.get(_sigma_key(opts))
    if not acc or "k" not in acc:
        return                                  # the baseline was absent: nothing accumulated
    k = acc["k"]
    if pass_i == 0:
        if opts["_scope"] == "pooled":
            det = np.linalg.det(acc["XtX"])
            acc["beta"] = (None if not np.isfinite(det) or abs(det) < 1e-9
                           else np.linalg.solve(acc["XtX"], acc["Xtb"]))
        else:
            acc["beta"] = _batched_solve(acc["XtX"], acc["Xtb"])
        # (H,W,k,k) is the largest thing here and pass 1 does not need it.
        acc.pop("XtX", None)
        acc.pop("Xtb", None)
        return

    cnt = acc["cnt"]
    if opts["_scope"] == "pooled":
        acc["sigma"] = np.sqrt(acc.get("ssr", 0.0) / max(float(cnt) - k, 1.0))
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma = np.sqrt(acc.get("ssr", 0.0) / np.maximum(cnt - k, 1.0))
        acc["sigma"] = np.where(cnt > k, sigma, np.nan)
    acc.pop("ssr", None)


def sigma_cutoff_block(ctx: "PreprocessContext", opts: dict, n_sigma: float) -> np.ndarray | None:
    """`mean - n_sigma*sigma` for THIS block's days, from the fitted climatology.

    Evaluated per block and never held as (T,H,W): the cutoff IS `X @ beta - n_sigma*sigma`, so
    materialising it for the whole cube -- as this used to -- was one more full-size float array
    for something two per-pixel parameters already describe.
    """
    acc = ctx.window.get(_sigma_key(opts))
    if acc is None or "sigma" not in acc:
        return None
    beta, sigma = acc.get("beta"), acc["sigma"]
    X = _harmonic_design(ctx, opts["_seasonality"])
    if opts["_scope"] == "pooled":
        if beta is None:                       # singular system: no cutoff, so nothing is cold
            return np.full((len(ctx.days), ctx.H, ctx.W), np.nan, "float32")
        mu_t = X @ beta
        cutoff = (mu_t - n_sigma * sigma)[:, None, None]
        return np.broadcast_to(cutoff, (len(ctx.days), ctx.H, ctx.W)).astype("float32")
    mu = np.einsum("ti,yxi->tyx", X, beta)
    return (mu - n_sigma * sigma[None, :, :]).astype("float32")


def _sigma_opts(ctx: "PreprocessContext", key: str, base_key: str | None) -> dict | None:
    """The climatology this filter's config asks for, or None when it is configured day-locally.

    `method: offset` needs no whole-window statistic at all, so an offset-configured filter
    costs nothing: `passes` returns 0 and the orchestrator never streams the cube for it.
    """
    o = _resolve_opts(ctx, key, base_key)
    if str(o.get("method", "offset")).lower() != "sigma":
        return None
    return {"_name": o.get("baseline", "mur_sst"),
            "_scope": str(o.get("stat_scope", "pixel")).lower(),
            "_seasonality": str(o.get("seasonality", "off")).lower()}


def sigma_passes(ctx: "PreprocessContext", *, key: str, base_key: str | None = None) -> int:
    return 2 if _sigma_opts(ctx, key, base_key) else 0


def sigma_accumulate_step(ctx: "PreprocessContext", pass_i: int, *, key: str,
                          base_key: str | None = None) -> None:
    o = _sigma_opts(ctx, key, base_key)
    if o is not None:
        sigma_accumulate(ctx, pass_i, o)


def sigma_reduce_step(ctx: "PreprocessContext", pass_i: int, *, key: str,
                      base_key: str | None = None) -> None:
    o = _sigma_opts(ctx, key, base_key)
    if o is not None:
        sigma_reduce(ctx, pass_i, o)


def _baseline_cutoff(ctx: "PreprocessContext", name: str, n_sigma: float,
                     scope: str, seasonality: str) -> np.ndarray | None:
    """The cold cutoff field `mean - n_sigma*sigma` (T,y,x) from the baseline's OWN climatology.

    The mean is fit by least squares against a (constant or seasonal-harmonic) design matrix;
    sigma is the std of the residuals about it. Stats come from the OBSERVED raw baseline (via
    `ctx.read`, never fill_water's invented cells, which would collapse sigma). `pixel` scope
    fits a per-cell climatology; `pooled` fits one over all cells & times. Cells/scenes with too
    few finite samples get a NaN cutoff (-> pixel kept). None if the baseline is absent.

    The WHOLE-WINDOW path: one block covering every day, fitted and evaluated here. When the
    cube is preprocessed in blocks the same three functions are driven by the orchestrator
    instead (`WindowStat`), which is why there is only one implementation of the maths.
    """
    if ctx.read(name, dims=T3) is None:
        return None
    opts = {"_name": name, "_scope": scope, "_seasonality": seasonality}
    if _sigma_key(opts) not in ctx.window:
        for p in (0, 1):
            sigma_accumulate(ctx, p, opts)
            sigma_reduce(ctx, p, opts)
    return sigma_cutoff_block(ctx, opts, n_sigma)


# --------------------------------------------------------------------------- #
# Step: filter_cloud_cover -- meteorological total-cloud-cover gate.
# --------------------------------------------------------------------------- #
def _step_filter_cloud_cover(ctx: "PreprocessContext", *, key: str = "filter_cloud_cover",
                             base_key: str | None = None, mode: str = "raw") -> None:
    """Gate ECOSTRESS on the met total-cloud-cover field (HRRR/ERA5), in PERCENT (0-100).

    Two independent gates (a pixel is dropped if EITHER fires):
      * scene-level (`scene_max_pct`, default 30): if the AOI-mean cloud cover at the overpass
        exceeds it, drop the WHOLE overpass;
      * per-pixel (`pixel_max_pct`, default 80): drop individual pixels above it.
    Either is disabled by a null / >=100 threshold. Prefers the overpass-matched
    `<sensor>_cloud_cover_<src>`, falls back to the daily forcing `cloud_cover_<src>`; `source`
    (default auto: overpass over forcing, hrrr over era5) picks. Emits an
    `<sensor>_scene_cloud_pct_<src>` (time,) diagnostic. Composes with filter_clouds.

    KNOWN LIMIT ON A MOSAICKED DAY. "The overpass" is one instant: the base granule's. But a
    sensor with `SensorSpec.mosaic_same_day` (eco -- this step's own default -- and lst) merges
    a day's granules, so the slice can hold pixels from another overpass whose cloud cover was
    never consulted. The scene-level gate then drops, or spares, those pixels on evidence about
    a different time of day. Correcting it needs a per-cell granule identity, which the cube
    does not carry; until then the `mosaic_time` attribute on `<sensor>_sst` is the disclosure.
    """
    opts = _resolve_opts(ctx, key, base_key)
    sensors = _as_list(opts.get("sensors")) or ["eco"]
    mask_sst = bool(opts.get("mask_sst", True))
    source = opts.get("source")
    scene_max = _gate_threshold(opts.get("scene_max_pct", 30.0))
    pixel_max = _gate_threshold(opts.get("pixel_max_pct", 80.0))
    if scene_max is None and pixel_max is None:
        log.info("  filter_cloud_cover: both gates disabled (null / >=100); nothing filtered")
        return

    mask_channel = opts.get("water_mask_channel", "landcover_water")
    water = ctx.read(mask_channel, dims=("y", "x"))
    water = (water > 0.5) if water is not None else None

    emitted = False
    for pre in sensors:
        tcc, src, _ = _met_cloud_channel(ctx, pre, source)
        if tcc is None:
            log.warning("  filter_cloud_cover: no met cloud-cover channel for sensor %r "
                        "(source=%s); skipping (met not acquired?)", pre, source or "auto")
            continue
        scene_pct = _scene_mean(tcc, water)                   # (T,)
        dropped = np.zeros(tcc.shape, bool)
        if scene_max is not None:
            dropped |= (scene_pct > scene_max)[:, None, None]
        if pixel_max is not None:
            with np.errstate(invalid="ignore"):
                dropped |= np.nan_to_num(tcc, nan=-1.0) > pixel_max

        if mode == "raw":                                     # met field isn't shifted -> identical
            ctx.emit(f"{pre}_scene_cloud_pct_{src}", ("time",), scene_pct.astype("float32"),
                     long_name=f"{pre} overpass AOI-mean cloud cover ({src})", units="%")
        attrs = {"preprocess": (f"met cloud gate ({src}): "
                                f"scene>{scene_max}% or pixel>{pixel_max}%"),
                 "cloud_cover_source": str(src)}
        for cs in _channel_sets(ctx, pre, mode):
            _fold_drop(ctx, cs, dropped, flag_suffix="_metcloudfiltered",
                       mask_sst=mask_sst, attrs=attrs)
        emitted = True
    if not emitted:
        log.info("  filter_cloud_cover: no sensor had a met cloud-cover channel; nothing filtered")


def _gate_threshold(v):
    """A gate threshold in percent, or None when the gate is disabled (null or >=100)."""
    if v is None:
        return None
    v = float(v)
    return None if v >= 100 else v


def _met_var_channel(ctx: "PreprocessContext", pre: str, source, var: str):
    """The met `<var>` array for a sensor, preferring the overpass channel `<pre>_<var>_<src>`
    over the daily forcing `<var>_<src>`, and (when `source` is unset) hrrr over era5. Returns
    (arr (T,y,x), src_tag, channel_name) or (None, None, None). `var` is a met field root,
    e.g. `cloud_cover` (percent) or `airtemp` (harmonized to the cube's K/degC)."""
    overpass = {n[len(f"{pre}_{var}_"):]: n
                for n in ctx.channels_with_prefix(f"{pre}_{var}_")}
    forcing = {n[len(f"{var}_"):]: n
               for n in ctx.channels_with_prefix(f"{var}_")}
    if source is not None:
        order = [source]
    else:
        others = sorted((set(overpass) | set(forcing)) - set(_MET_SRC_PREF))
        order = list(_MET_SRC_PREF) + others
    for src in order:
        for cand in (overpass.get(src), forcing.get(src)):
            if cand is None:
                continue
            arr = ctx.read(cand, dims=T3)
            if arr is not None:
                return arr, src, cand
    return None, None, None


def _met_cloud_channel(ctx: "PreprocessContext", pre: str, source):
    """The met total-cloud-cover array for a sensor (see `_met_var_channel`)."""
    return _met_var_channel(ctx, pre, source, "cloud_cover")


def _scene_mean(tcc: np.ndarray, water) -> np.ndarray:
    """AOI spatial mean of the cloud field per time slice, over water cells (else all finite).
    Slices with no finite cell -> NaN (a day with no overpass), which never trips a gate."""
    T = tcc.shape[0]
    sel = tcc[:, water] if (water is not None and water.any()) else tcc.reshape(T, -1)
    finite = np.isfinite(sel)
    cnt = finite.sum(axis=1)
    tot = np.where(finite, sel, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out


# --------------------------------------------------------------------------- #
# Step: filter_land_clouds -- over-land air-temperature deviation filter.
# --------------------------------------------------------------------------- #
def _land_mask(ctx: "PreprocessContext", pre: str, land_source: str, mask_channel: str):
    """A boolean LAND mask for a sensor -- `(y,x)` for `landcover`, `(T,y,x)` for `water_line`
    (both broadcast against a `(T,y,x)` drop mask). None if the chosen source is absent.

      * `landcover` -- the static land/water mask (land = `mask_channel` == 0). A NaN (unknown)
        cell stays NOT-land, so an unknown mask never licenses a drop.
      * `water_line` -- this sensor's `<pre>_water_class` == EXPOSED at its overpass (the
        `water_line` step's output; per-day, since it tracks the tide). SUBMERGED / UNKNOWN
        cells are not land.
    """
    if land_source == "water_line":
        wc = _working(ctx, f"{pre}_water_class", dtype="uint8")
        return None if wc is None else (wc == EXPOSED)
    water = ctx.read(mask_channel, dims=("y", "x"))
    if water is None:
        return None
    return np.nan_to_num(water, nan=1.0) < 0.5


def _step_filter_land_clouds(ctx: "PreprocessContext", *, key: str = "filter_land_clouds",
                             base_key: str | None = None, mode: str = "raw") -> None:
    """Screen cloud-contaminated OVER-LAND pixels against the near-surface air temperature.

    Over land the thermal-IR surface temperature runs WARMER than the air by day; cloud
    contamination biases it COLD, so a land pixel far colder than the air temperature is likely
    cloud. Drop where `airtemp - sst > threshold_k` (`threshold_k` default 5.0; both operands are
    the cube's harmonized temperature unit, and a K difference equals a degC one, so `5` means 5
    K == 5 degC regardless; `<=0` disables). The drop applies only where the land mask marks the
    cell as land (`land_source`):
      * "landcover" (default): the static `landcover_water` mask (land = water == 0);
      * "water_line": this sensor's `<pre>_water_class` == EXPOSED at its overpass -- needs the
        `water_line` step enabled.
    Air temperature prefers the overpass-matched `<pre>_airtemp_<src>` over the daily forcing
    `airtemp_<src>`; `source` (default auto: overpass over forcing, hrrr over era5) picks. Drops
    fold into `<sensor>_sst_<ver>_clean` / `<sensor>_valid_<ver>_clean` + a `*_landcloudfiltered`
    flag and compose with the other cloud filters (all share that target). NaN operands are kept.
    """
    opts = _resolve_opts(ctx, key, base_key)
    thr = float(opts.get("threshold_k", 5.0))
    sensors = _as_list(opts.get("sensors")) or ["eco"]
    mask_sst = bool(opts.get("mask_sst", True))
    source = opts.get("source")
    land_source = str(opts.get("land_source", "landcover")).lower()
    mask_channel = opts.get("mask_channel", "landcover_water")
    if land_source not in _LAND_SOURCES:
        raise ValueError(f"filter_land_clouds.land_source {land_source!r} is invalid; "
                         f"choose one of {list(_LAND_SOURCES)}")
    if thr <= 0:
        log.info("  filter_land_clouds: threshold_k<=0 -> disabled; nothing filtered")
        return

    emitted = False
    for pre in sensors:
        air, src, _ = _met_var_channel(ctx, pre, source, "airtemp")
        if air is None:
            log.warning("  filter_land_clouds: no air-temp channel for sensor %r (source=%s); "
                        "skipping (met not acquired?)", pre, source or "auto")
            continue
        land = _land_mask(ctx, pre, land_source, mask_channel)
        if land is None:
            log.warning("  filter_land_clouds: no land mask for sensor %r (land_source=%s); "
                        "skipping (enable the water_line step, or acquire landcover)",
                        pre, land_source)
            continue
        note = f"land cloud-filtered ({land_source}, {src}): airtemp - sst > {thr}"
        for cs in _channel_sets(ctx, pre, mode):
            sst = _target(ctx, cs.sst, cs.seed_sst)
            if sst is None:
                continue
            with np.errstate(invalid="ignore"):
                dropped = (air - sst) > thr
            dropped = (np.asarray(dropped, bool)
                       & np.isfinite(sst) & np.isfinite(air) & land)
            _fold_drop(ctx, cs, dropped, flag_suffix="_landcloudfiltered",
                       mask_sst=mask_sst, attrs={"preprocess": note,
                                                 "airtemp_source": str(src),
                                                 "land_source": land_source})
            emitted = True
    if not emitted:
        log.info("  filter_land_clouds: nothing filtered")
