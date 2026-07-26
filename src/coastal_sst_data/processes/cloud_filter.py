#!/usr/bin/env python3
"""
coastal_sst_data -- ECOSTRESS cloud filtering (post-assembly preprocess steps).

Two composable `preprocess` steps that screen cloud-contaminated ECOSTRESS pixels and fold the
drops into `<sensor>_valid_<ver>` / `<sensor>_sst_<ver>`. They live here rather than in
`preprocess.py` to keep that module a thin registry + orchestrator; `preprocess` imports the two
`_step_*` functions and registers each as one `PreprocessStep` (the same contributor protocol as
every other step). Nothing here imports `preprocess` at runtime -- a step only calls methods on
the `PreprocessContext` it is handed -- so there is no import cycle.

  * filter_clouds       -- screen against a gap-free baseline L4 SST (MUR/CMEMS): a fixed cold
                          offset (`baseline - eco > threshold_k`, ported from oceanSR) OR a
                          distribution-based, seasonally-aware outlier floor
                          (`eco < mean - n_sigma*sigma` from the baseline's own climatology).
  * filter_cloud_cover  -- gate on the meteorological total-cloud-cover field (HRRR/ERA5, in
                          percent): a scene-level rejection AND a per-pixel cutoff.

Both mutate the SAME sst/valid channels, so they read the WORKING value (`_working`: a channel
already emitted this run wins over the raw cube) and their drops UNION regardless of run order;
the emitted `*_cloudfiltered` / `*_metcloudfiltered` flags keep each drop auditable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                          # avoid an import cycle: only a type hint needs it
    from .preprocess import PreprocessContext

log = logging.getLogger(__name__)

T3 = ("time", "y", "x")

_CLOUD_METHODS = ("offset", "sigma")
_STAT_SCOPES = ("pixel", "pooled")
_SEASONALITY = ("off", "harmonic")
_MET_SRC_PREF = ("hrrr", "era5")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _as_list(v):
    """A config value that may be a scalar, a list, or absent -> a list (or None)."""
    if v is None:
        return None
    return [v] if isinstance(v, str) else list(v)


def _working(ctx: "PreprocessContext", name: str, *, dims=T3, dtype="float32"):
    """The CURRENT value of a channel this stage may already have modified: prefer a channel
    already emitted this run (an earlier step's output) over the raw cube. None if absent /
    wrong-shaped. This is what lets the two cloud steps compose -- whichever runs second sees the
    first's drops and unions onto them."""
    if name in ctx.channels:
        d, arr = ctx.channels[name]
        if tuple(d) != tuple(dims):
            return None
        return np.asarray(arr).astype(dtype, copy=False)
    return ctx.read(name, dims=dims, dtype=dtype)


def _fold_drop(ctx: "PreprocessContext", pre: str, suffix: str, dropped: np.ndarray, *,
               flag_suffix: str, mask_sst: bool, attrs: dict) -> None:
    """Fold a boolean drop mask (T,y,x) into ONE ECOSTRESS version's sst/valid, and emit the
    companion flag. Reads the WORKING sst/valid (so a second cloud step composes with the first)
    and re-emits: `<pre>_valid<suffix>` &= ~dropped, `<pre>_sst<suffix>` NaN'd (if mask_sst), and
    `<pre>_sst<suffix><flag_suffix>` (uint8, 1 = dropped)."""
    sst_name = f"{pre}_sst{suffix}"
    valid = _working(ctx, f"{pre}_valid{suffix}", dtype="uint8")
    if valid is not None:
        ctx.emit(f"{pre}_valid{suffix}", T3,
                 (valid.astype(bool) & ~dropped).astype("uint8"), **attrs)
    if mask_sst:
        eco = _working(ctx, sst_name)
        if eco is not None:
            out = eco.copy()
            out[dropped] = np.nan
            ctx.emit(sst_name, T3, out, **attrs)
    ctx.emit(f"{sst_name}{flag_suffix}", T3, dropped.astype("uint8"),
             flag_values=[0, 1], flag_meanings="kept dropped", **attrs)


# --------------------------------------------------------------------------- #
# Step: filter_clouds -- baseline cold-deviation filter (ported from oceanSR).
# --------------------------------------------------------------------------- #
def _step_filter_clouds(ctx: "PreprocessContext") -> None:
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
    opts = ctx.step_opts("filter_clouds")
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
    if method == "offset":
        thr = float(opts.get("threshold_k", 5.0))
        if thr <= 0:
            log.info("  filter_clouds: threshold_k<=0 -> disabled; nothing filtered")
            return
        base = _working(ctx, baseline_name)                  # prefers fill_water's filled baseline
        if base is None:
            log.warning("  filter_clouds: baseline %r absent; nothing filtered", baseline_name)
            return
        note = f"cloud-filtered: {baseline_name} - eco > {thr} K"
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
        for sst_name in ctx.channels_with_prefix(f"{pre}_sst"):
            suffix = sst_name[len(f"{pre}_sst"):]
            eco = _working(ctx, sst_name)
            if eco is None:
                continue
            with np.errstate(invalid="ignore"):
                if method == "offset":
                    dropped = (base - eco) > thr
                else:
                    dropped = eco < cutoff
            dropped = np.asarray(dropped, bool) & np.isfinite(eco)
            if use_cloud:
                cloud = ctx.read(f"{pre}_cloud{suffix}", dims=T3)
                if cloud is not None:
                    dropped |= np.nan_to_num(cloud, nan=0.0) > 0
            _fold_drop(ctx, pre, suffix, dropped, flag_suffix="_cloudfiltered",
                       mask_sst=mask_sst, attrs={"preprocess": note,
                                                 "cloud_baseline": str(baseline_name)})
            emitted = True
    if emitted:
        ctx.carry(baseline_name)


def _harmonic_design(ctx: "PreprocessContext", seasonality: str) -> np.ndarray:
    """The climatology design matrix X (T,k): a constant column (k=1), plus -- for a day-of-year
    `harmonic` -- annual sin/cos columns (k=3, reusing the cube's doy_sin/doy_cos). Falls back to
    the constant when the cube's DOY span is too short to constrain the seasonal cycle."""
    T = len(ctx.days)
    ones = np.ones(T)
    if seasonality != "harmonic":
        return ones[:, None]
    s = ctx.read("doy_sin", dims=("time",))
    c = ctx.read("doy_cos", dims=("time",))
    if s is None or c is None:
        doy = ctx.days.dayofyear.values.astype("float64")
        s = np.sin(2 * np.pi * doy / 365.25)
        c = np.cos(2 * np.pi * doy / 365.25)
    s = np.asarray(s, "float64")
    c = np.asarray(c, "float64")
    if np.ptp(ctx.days.dayofyear.values) < 60 or (np.std(s) < 1e-3 and np.std(c) < 1e-3):
        log.info("  filter_clouds: DOY span too short for a seasonal harmonic; "
                 "using a constant climatology")
        return ones[:, None]
    return np.column_stack([ones, s, c])


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


def _baseline_cutoff(ctx: "PreprocessContext", name: str, n_sigma: float,
                     scope: str, seasonality: str) -> np.ndarray | None:
    """The cold cutoff field `mean - n_sigma*sigma` (T,y,x) from the baseline's OWN climatology.

    The mean is fit by least squares against a (constant or seasonal-harmonic) design matrix;
    sigma is the std of the residuals about it. Stats come from the OBSERVED raw baseline (via
    `ctx.read`, never fill_water's invented cells, which would collapse sigma). `pixel` scope
    fits a per-cell climatology; `pooled` fits one over all cells & times. Cells/scenes with too
    few finite samples get a NaN cutoff (-> pixel kept). None if the baseline is absent.
    """
    base = ctx.read(name, dims=T3)
    if base is None:
        return None
    base = base.astype("float64", copy=False)
    X = _harmonic_design(ctx, seasonality)
    k = X.shape[1]
    finite = np.isfinite(base)
    M = finite.astype("float64")
    Bz = np.where(finite, base, 0.0)

    if scope == "pooled":
        XtX = np.einsum("ti,tj,tyx->ij", X, X, M)
        Xtb = np.einsum("ti,tyx->i", X, Bz)
        if not np.isfinite(np.linalg.det(XtX)) or abs(np.linalg.det(XtX)) < 1e-9:
            return np.full(base.shape, np.nan, "float32")
        beta = np.linalg.solve(XtX, Xtb)
        mu_t = X @ beta                                        # (T,)
        ssr = float(np.sum(np.where(finite, (base - mu_t[:, None, None]) ** 2, 0.0)))
        dof = max(float(M.sum()) - k, 1.0)
        sigma = np.sqrt(ssr / dof)
        cutoff = (mu_t - n_sigma * sigma)[:, None, None]
        return np.broadcast_to(cutoff, base.shape).astype("float32")

    # pixel scope: a per-cell climatology (masked normal equations, fully vectorised).
    XtX = np.einsum("ti,tj,tyx->yxij", X, X, M)                # (H,W,k,k)
    Xtb = np.einsum("ti,tyx->yxi", X, Bz)                      # (H,W,k)
    beta = _batched_solve(XtX, Xtb)                            # (H,W,k)
    mu = np.einsum("ti,yxi->tyx", X, beta)                     # (T,H,W)
    cnt = M.sum(axis=0)                                        # (H,W)
    ssr = np.sum(np.where(finite, (base - mu) ** 2, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt(ssr / np.maximum(cnt - k, 1.0))
    sigma = np.where(cnt > k, sigma, np.nan)
    cutoff = mu - n_sigma * sigma[None, :, :]
    return cutoff.astype("float32")


# --------------------------------------------------------------------------- #
# Step: filter_cloud_cover -- meteorological total-cloud-cover gate.
# --------------------------------------------------------------------------- #
def _step_filter_cloud_cover(ctx: "PreprocessContext") -> None:
    """Gate ECOSTRESS on the met total-cloud-cover field (HRRR/ERA5), in PERCENT (0-100).

    Two independent gates (a pixel is dropped if EITHER fires):
      * scene-level (`scene_max_pct`, default 30): if the AOI-mean cloud cover at the overpass
        exceeds it, drop the WHOLE overpass;
      * per-pixel (`pixel_max_pct`, default 80): drop individual pixels above it.
    Either is disabled by a null / >=100 threshold. Prefers the overpass-matched
    `<sensor>_cloud_cover_<src>`, falls back to the daily forcing `cloud_cover_<src>`; `source`
    (default auto: overpass over forcing, hrrr over era5) picks. Emits an
    `<sensor>_scene_cloud_pct_<src>` (time,) diagnostic. Composes with filter_clouds.
    """
    opts = ctx.step_opts("filter_cloud_cover")
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
        tcc, src, name = _met_cloud_channel(ctx, pre, source)
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

        ctx.emit(f"{pre}_scene_cloud_pct_{src}", ("time",), scene_pct.astype("float32"),
                 long_name=f"{pre} overpass AOI-mean cloud cover ({src})", units="%")
        attrs = {"preprocess": (f"met cloud gate ({src}): "
                                f"scene>{scene_max}% or pixel>{pixel_max}%"),
                 "cloud_cover_source": str(src)}
        for sst_name in ctx.channels_with_prefix(f"{pre}_sst"):
            suffix = sst_name[len(f"{pre}_sst"):]
            _fold_drop(ctx, pre, suffix, dropped, flag_suffix="_metcloudfiltered",
                       mask_sst=mask_sst, attrs=attrs)
        ctx.carry(name)
        emitted = True
    if not emitted:
        log.info("  filter_cloud_cover: no sensor had a met cloud-cover channel; nothing filtered")


def _gate_threshold(v):
    """A gate threshold in percent, or None when the gate is disabled (null or >=100)."""
    if v is None:
        return None
    v = float(v)
    return None if v >= 100 else v


def _met_cloud_channel(ctx: "PreprocessContext", pre: str, source):
    """The met cloud-cover array for a sensor, preferring the overpass channel
    `<pre>_cloud_cover_<src>` over the daily forcing `cloud_cover_<src>`, and (when `source` is
    unset) hrrr over era5. Returns (arr (T,y,x), src_tag, channel_name) or (None, None, None)."""
    overpass = {n[len(f"{pre}_cloud_cover_"):]: n
                for n in ctx.channels_with_prefix(f"{pre}_cloud_cover_")}
    forcing = {n[len("cloud_cover_"):]: n
               for n in ctx.channels_with_prefix("cloud_cover_")}
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
