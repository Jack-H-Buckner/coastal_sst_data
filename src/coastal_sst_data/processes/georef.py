#!/usr/bin/env python3
"""
coastal_sst_data -- ECOSTRESS georeferencing diagnosis + correction (post-assembly preprocess).

Some ECOSTRESS granules are geolocated wrong: a misregistered scene contributes wrong SST at every
coastal pixel, and nothing downstream flags it. These two composable `preprocess` steps register
each thermal scene's coastline against the STATIC land-cover coastline by brute-force whole-cell
translation, then (optionally) apply the fitted shift:

  * flag_georef     -- per-scene diagnosis. Detect thermal edges (NaN-aware Canny), score them
                      against the `landcover_water` coastline over every whole-cell (dy,dx) shift,
                      and classify the scene (ok / displaced / suspect / unstable / unrecoverable /
                      insufficient_signal). Emits per-sensor 1-D `("time",)` diagnostics only --
                      it moves NO pixels. The fit runs on the WORKING (cloud-filtered) SST, since
                      clean edges give a better registration.
  * correct_georef  -- apply the fitted (dy,dx) to the RAW ECOSTRESS geometry (SST + validity by
                      default), for `displaced` scenes only, into `<field>_georef_corrected`
                      channels. It shifts the raw granule-native fields read from the raw cube, NOT
                      the preprocess-derived masks: the corrected geometry invalidates the cloud /
                      water preprocessing, which must be RE-RUN against the corrected images (future
                      work, possibly dedicated georef-aware modules).

Lives here rather than in `preprocess.py` to keep that a thin registry + orchestrator; `preprocess`
imports the two `_step_*` functions and registers each as one `PreprocessStep`. Nothing here imports
`preprocess` at runtime (only a `TYPE_CHECKING` hint), so there is no import cycle.

Ported faithfully from the validated prototype
(`/Users/johnbuckner/Documents/test_coastal_sst_data/src/georef_diagnose.py` +
`plot_thermal_edges.py`); the load-bearing details are documented at each function. The six-way
classifier and the stability sweep (`classify`, `_stable`) are new: the prototype produced the
search surface and its statistics but did not classify.

Known limits (whole-cell translation only -- no sub-pixel, rotation, or scale): adequate for the
observed failure mode, not general. A straight coastline constrains the across-shore offset well and
the along-shore one barely (the search surface is a ridge, not a peak); both validated AoIs have
convoluted coastlines, so this is untested on a simple one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from .channels import CLEAN, CORRECTED

if TYPE_CHECKING:                          # avoid an import cycle: only a type hint needs it
    from .preprocess import PreprocessContext

log = logging.getLogger(__name__)

T3 = ("time", "y", "x")

# flag encoding for `<pre>_georef_flag` (uint8). Order fixed -- values are stored in cubes.
FLAG_MEANINGS = "ok displaced suspect unstable unrecoverable insufficient_signal"
FLAG = {name: i for i, name in enumerate(FLAG_MEANINGS.split())}

# Per-step option defaults. Search/edge defaults match the prototype CLI; the classifier
# thresholds are design §5. `min_valid_pct` is a percent (2.0), stored internally as a fraction.
_DEFAULTS = dict(
    sensors=["eco"], tol_m=200.0, max_shift_m=10000.0, coarse_stride=4, n_refine=8,
    sigma=1.5, lo_pct=80.0, hi_pct=97.0,
    min_coast_obs=500, min_valid_pct=2.0, min_edges=300,
    z_min=4.5, lift_min=2.5, gain_min=0.10, ok_shift_m=300.0,
    stability_windows_km=[5.0, 10.0, 20.0, 30.0],
)

# Which raw ECOSTRESS-native spatial fields `correct_georef` shifts, and how to fill the vacated
# margin. SST -> NaN (no observation); a mask -> 0 (unobserved). Fields not listed default to the
# SST treatment. The granule-native cloud band would go here as ("cloud", (0, "uint8")) if selected.
# `footprint_id` is an INDEX: the default NaN/float32 treatment would silently turn it into floats
# and invent a 0 id, so it is declared here even though no sensor lists it in `fields` today.
_FIELD_FILL = {"sst": (np.nan, "float32"), "valid": (0, "uint8"), "cloud": (0, "uint8"),
               "footprint_id": (-1, "int32")}


# --------------------------------------------------------------------------- #
# Image ops -- ported from plot_thermal_edges.py (they have no other package home)
# --------------------------------------------------------------------------- #
def nan_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN (normalised convolution).

    A plain gaussian_filter would spread every cloud hole across its whole neighbourhood and
    manufacture an edge around it. Smoothing value and validity separately, then dividing, keeps
    the blur inside the observed data.
    """
    valid = np.isfinite(arr).astype("float32")
    num = ndimage.gaussian_filter(np.where(valid > 0, arr, 0).astype("float32"), sigma,
                                  mode="nearest")
    den = ndimage.gaussian_filter(valid, sigma, mode="nearest")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan).astype("float32")


def _nbr_shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Neighbour lookup with a zero border (np.roll would wrap the far edge onto the near one)."""
    out = np.zeros_like(a)
    ys, yd = (slice(max(dy, 0), a.shape[0] + min(dy, 0)), slice(max(-dy, 0), a.shape[0] - max(dy, 0)))
    xs, xd = (slice(max(dx, 0), a.shape[1] + min(dx, 0)), slice(max(-dx, 0), a.shape[1] - max(dx, 0)))
    out[yd, xd] = a[ys, xs]
    return out


def canny(arr: np.ndarray, sigma: float, lo_pct: float, hi_pct: float) -> tuple:
    """Canny edges on a NaN-holed field -> (edge mask, gradient magnitude).

    Full Canny rather than a bare Sobel threshold: non-maximum suppression is what makes the result
    ONE cell wide, the only way it can be scored fairly against a one-cell reference coastline. A
    thick gradient ridge would score high recall for free. Sensitivity is tuned by the LOW threshold
    only -- lowering it extends the same strong seeds along weak coastal continuations without
    admitting isolated land clutter.
    """
    sm = nan_gaussian(arr, sigma)
    valid = np.isfinite(sm)
    gy = ndimage.sobel(np.where(valid, sm, 0.0), axis=0, mode="nearest")
    gx = ndimage.sobel(np.where(valid, sm, 0.0), axis=1, mode="nearest")
    mag = np.hypot(gx, gy)
    # A gradient across the rim of a data hole is an artefact of the hole, not a feature of the
    # scene: keep only cells whose full 3x3 neighbourhood was observed.
    solid = ndimage.binary_erosion(valid, np.ones((3, 3), bool))
    mag = np.where(solid, mag, np.nan)
    if not np.isfinite(mag).any():
        return np.zeros(arr.shape, bool), mag

    # Non-maximum suppression along the gradient direction, quantised to 4 orientations.
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    binned = (np.round(ang / 45.0).astype(int) % 4)
    m = np.nan_to_num(mag)
    neighbours = {0: ((0, 1), (0, -1)), 1: ((-1, 1), (1, -1)),
                  2: ((-1, 0), (1, 0)), 3: ((-1, -1), (1, 1))}
    thin = np.zeros(arr.shape, bool)
    for b, (p, q) in neighbours.items():
        thin |= (binned == b) & (m >= _nbr_shift(m, *p)) & (m >= _nbr_shift(m, *q))
    thin &= np.isfinite(mag)

    vals = mag[thin]
    if vals.size == 0:
        return np.zeros(arr.shape, bool), mag
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])

    # Hysteresis: keep a weak ridge only if it connects to a strong one, so a long faint coastline
    # survives while isolated speckle does not.
    weak, strong = thin & (mag >= lo), thin & (mag >= hi)
    lab, n = ndimage.label(weak, structure=np.ones((3, 3), bool))
    if n == 0:
        return strong, mag
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab], mag


def water_boundary(water: np.ndarray) -> np.ndarray:
    """The reference coastline: water cells touching land, one cell wide, on the WATER side
    (8-connected dilation of ~water intersected with water)."""
    return water & ndimage.binary_dilation(~water, structure=np.ones((3, 3), bool))


# --------------------------------------------------------------------------- #
# Geometry + shift search -- ported from georef_diagnose.py
# --------------------------------------------------------------------------- #
def shift_array(a: np.ndarray, dy: int, dx: int, fill=np.nan, dtype="float32") -> np.ndarray:
    """Move CONTENT by (dy, dx) whole cells: the value at row r lands at row r+dy.

    Deliberately not `np.roll`, which wraps the far edge onto the near one and would fabricate a
    coastline on the opposite side of the scene. Cells shifted in from outside are `fill`.
    """
    out = np.full(a.shape, fill, dtype=dtype)
    H, W = a.shape
    src = (slice(max(-dy, 0), H - max(dy, 0)), slice(max(-dx, 0), W - max(dx, 0)))
    dst = (slice(max(dy, 0), H - max(-dy, 0)), slice(max(dx, 0), W - max(-dx, 0)))
    out[dst] = a[src]
    return out


def coast_offsets(ys, xs, iy, ix, res_m: float):
    """Signed displacement from each edge cell to its NEAREST coast cell, in compass metres.

    `iy`/`ix` are the nearest-feature index arrays from `distance_transform_edt`. Cube rows run
    north->south (y descending), so a positive row offset points SOUTH -- hence the negation on
    `north`. Getting this backwards silently mirrors every result. Diagnostics only: the vector
    SATURATES (recovers 0-20% of a known displacement), so it must not be used as an estimator.
    """
    row_off = iy[ys, xs].astype("float32") - ys
    col_off = ix[ys, xs].astype("float32") - xs
    return col_off * res_m, -row_off * res_m          # east, north


def _score(ys, xs, dref, H, W, dy, dx, tol_cells, n):
    """Fraction of ALL edge cells that land within tolerance of the coastline after (dy,dx).

    Cells pushed off the grid are counted as NOT agreeing rather than voiding the candidate.
    Rejecting a shift outright when any single cell leaves the grid silently makes whole DIRECTIONS
    unsearchable (scene edges normally reach the margin). Keeping `n` (all edges) as the denominator
    also stops a huge shift from scoring well by retaining a handful of lucky cells.
    """
    yy, xx = ys + dy, xs + dx
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    if not ok.any():
        return 0.0
    return float((dref[yy[ok], xx[ok]] < tol_cells).sum()) / n


def search_shift(ys, xs, dref, shape, max_cells: int, tol_cells: float, stride: int = 4,
                 n_refine: int = 8):
    """Whole-cell (dy,dx) that best aligns the edges with the coastline, + surface statistics.

    Scores by gathering the precomputed distance transform at the shifted edge positions -- no image
    is resampled, so the whole surface costs one gather per candidate. Coarse pass at `stride`, then
    a fine pass around the best `n_refine` coarse peaks (refining around SEVERAL peaks matters: a
    stride-4 grid cannot land on (+3,-1), and refining only the coarse winner leaves the true
    optimum unreachable). A true registration is a sharp isolated peak; a chance alignment is a broad
    noisy surface, and only the surface (`med`, `sd`, `z`) separates them.
    """
    H, W = shape
    n = len(ys)
    if n == 0:
        return dict(dy=0, dx=0, agree=np.nan, agree0=np.nan, med=np.nan, sd=np.nan, z=np.nan)

    surf = [(_score(ys, xs, dref, H, W, dy, dx, tol_cells, n), dy, dx)
            for dy in range(-max_cells, max_cells + 1, stride)
            for dx in range(-max_cells, max_cells + 1, stride)]
    vals = np.array([s[0] for s in surf])

    best = surf[int(vals.argmax())]
    for k in np.argsort(vals)[-n_refine:]:
        cy, cx = surf[int(k)][1], surf[int(k)][2]
        for dy in range(cy - stride, cy + stride + 1):
            for dx in range(cx - stride, cx + stride + 1):
                f = _score(ys, xs, dref, H, W, dy, dx, tol_cells, n)
                if f > best[0]:
                    best = (f, dy, dx)

    med, sd = float(np.median(vals)), float(vals.std())
    return dict(dy=int(best[1]), dx=int(best[2]), agree=float(best[0]),
                agree0=_score(ys, xs, dref, H, W, 0, 0, tol_cells, n),
                med=med, sd=sd, z=(best[0] - med) / max(sd, 1e-9))


def scene_quality(sst: np.ndarray, ref: np.ndarray, water: np.ndarray) -> dict:
    """What a scene offers the fit BEFORE any fitting. `coast_obs` -- coastline cells under valid
    data -- is the strongest predictor of a spurious fit; use the absolute count (coastlines differ
    ~10x in length between AoIs). `dt_k` (land-sea contrast) is a DIAGNOSTIC, never a gate."""
    fin = np.isfinite(sst)
    wm, lm = water & fin, (~water) & fin
    return dict(
        valid_frac=float(fin.mean()),
        coast_obs=int((fin & ref).sum()),
        coast_obs_frac=float((fin & ref).sum() / max(int(ref.sum()), 1)),
        dt_k=float(abs(np.nanmean(sst[lm]) - np.nanmean(sst[wm])))
        if wm.any() and lm.any() else np.nan,
    )


def quality_reject(q: dict, n_edge: int, gates: dict) -> str | None:
    """Reason this scene should not be fitted, or None to proceed. Deliberately NOT gating on
    land-sea contrast: a confirmed real correction had dT = 0.47 K. `min_valid_frac` is a fraction."""
    if q["coast_obs"] < gates["min_coast_obs"]:
        return f"only {q['coast_obs']:,} coastline cells observed (<{gates['min_coast_obs']:,})"
    if q["valid_frac"] < gates["min_valid_frac"]:
        return f"only {100*q['valid_frac']:.1f}% of the grid valid (<{100*gates['min_valid_frac']:.0f}%)"
    if n_edge < gates["min_edges"]:
        return f"only {n_edge:,} thermal edges (<{gates['min_edges']:,})"
    return None


# --------------------------------------------------------------------------- #
# Classification -- NEW (design §5; not in the prototype)
# --------------------------------------------------------------------------- #
def _stable(ys, xs, dref, shape, windows_cells, tol_cells, ref_dy, ref_dx,
            stride, n_refine, max_move: int = 2) -> bool:
    """Does the best (dy,dx) hold across the window sweep? Real corrections are invariant; the
    runaways drift kilometres as the search half-width grows. Stable iff every window's peak sits
    within `max_move` cells of the main fit."""
    for mc in windows_cells:
        if mc <= 0:
            continue
        st = search_shift(ys, xs, dref, shape, mc, tol_cells, stride, n_refine)
        if abs(st["dy"] - ref_dy) > max_move or abs(st["dx"] - ref_dx) > max_move:
            return False
    return True


def classify(st: dict, shift_m: float, gates: dict, stable: bool) -> str:
    """Label a scene from its search surface (design §5). Order is load-bearing.

    No penalty on shift magnitude (it would reject the one confirmed large displacement); `z` alone
    is insufficient (spurious fits reach higher z than the real large one); absolute agreement is
    AoI-specific -- score on LIFT over the surface median.
    """
    z, agree, agree0, med = st["z"], st["agree"], st["agree0"], st["med"]
    gain = agree - agree0
    if not stable:
        return "unstable"
    if not (agree >= gates["lift_min"] * med):
        return "unrecoverable"
    if shift_m <= gates["ok_shift_m"] and z >= gates["z_min"]:
        return "ok"
    if z >= gates["z_min"] and gain >= gates["gain_min"] and stable:
        return "displaced"
    return "suspect"


# --------------------------------------------------------------------------- #
# Shared step helpers
# --------------------------------------------------------------------------- #
def _as_list(v):
    if v is None:
        return None
    return [v] if isinstance(v, str) else list(v)


def _opts(ctx: "PreprocessContext", key: str) -> dict:
    """This AoI's resolved options for `key`, over the step defaults."""
    return {**_DEFAULTS, **ctx.step_opts(key)}


def _working(ctx: "PreprocessContext", name: str, *, dims=T3, dtype="float32"):
    """The CURRENT value of a channel: a channel already emitted this run (an earlier step's output,
    e.g. the cloud-filtered SST) wins over the cube. None if absent / wrong-shaped."""
    if name in ctx.channels:
        d, arr = ctx.channels[name]
        if tuple(d) != tuple(dims):
            return None
        return np.asarray(arr).astype(dtype, copy=False)
    return ctx.read(name, dims=dims, dtype=dtype)


def _fit_sst(ctx: "PreprocessContext", name: str):
    """The SST the registration fits on for raw channel `name`: this run's cloud-filtered
    `<name>_clean` if a filter produced one, else the raw channel.

    The filters write a SEPARATE `_clean` product rather than screening `name` in place, so the
    fit has to ask for it by name -- cloud edges are the dominant noise source in the
    registration, and fitting the unscreened scene is a silent quality regression. Restricted to
    THIS run's output: a `_clean` left on the cube by an earlier run reflects that run's
    thresholds, not the ones selected now.
    """
    clean = f"{name}{CLEAN}"
    if clean in ctx.channels:
        d, arr = ctx.channels[clean]
        if tuple(d) == T3:
            return np.asarray(arr).astype("float32", copy=False)
    return ctx.read(name, dims=T3, dtype="float32")


def _res_m(ctx: "PreprocessContext") -> float:
    """Grid posting in metres from the raw cube's x coordinate."""
    x = np.asarray(ctx.ds_cube["x"].values, "float64")
    return float(abs(x[1] - x[0])) if x.size > 1 else float(ctx.g.resolution_m)


def _ref_water_mask(ctx: "PreprocessContext"):
    """The land-cover WATER mask (bool), or None (degrade) when it is absent / degenerate.

    A degenerate (all-water / all-land) `landcover_water` yields no coastline to score against; that
    channel was broken until recently, so a single-class mask is rejected with a warning rather than
    used. The caller derives the reference coastline via `water_boundary`."""
    water = ctx.read("landcover_water", dims=("y", "x"), dtype="float32")
    if water is None:
        log.warning("  flag_georef: no `landcover_water` channel; no reference coastline -- skipped.")
        return None
    w = water > 0.5
    if not w.any() or w.all():
        log.warning("  flag_georef: `landcover_water` is degenerate (%.0f%% water) -- skipped.",
                    100 * w.mean())
        return None
    return w


# --------------------------------------------------------------------------- #
# Step: flag_georef -- per-scene diagnosis (emits 1-D diagnostics; moves no pixels)
# --------------------------------------------------------------------------- #
def _step_flag_georef(ctx: "PreprocessContext") -> None:
    """Diagnose per-scene ECOSTRESS georeferencing error against the land-cover coastline.

    For each sensor and time: detect thermal edges on the WORKING (cloud-filtered) SST, gate on
    scene quality (skip -> `insufficient_signal`), search whole-cell shifts, sweep windows for
    stability, and classify. Emits per-sensor 1-D `("time",)` channels only. Never raises: a missing
    sensor / reference coastline degrades to emitting nothing for that sensor.
    """
    opts = _opts(ctx, "flag_georef")
    water = _ref_water_mask(ctx)
    if water is None:
        return
    ref = water_boundary(water)                        # one-cell reference coastline, water side

    # One distance transform per AoI, reused by every scene's scoring.
    dref = ndimage.distance_transform_edt(~ref)
    res_m = _res_m(ctx)
    tol_cells = float(opts["tol_m"]) / res_m
    max_cells = int(round(float(opts["max_shift_m"]) / res_m))
    windows_cells = [int(round(w * 1000.0 / res_m)) for w in opts["stability_windows_km"]]
    stride, n_refine = int(opts["coarse_stride"]), int(opts["n_refine"])
    sigma, lo_pct, hi_pct = float(opts["sigma"]), float(opts["lo_pct"]), float(opts["hi_pct"])
    gates = dict(min_coast_obs=int(opts["min_coast_obs"]), min_edges=int(opts["min_edges"]),
                 min_valid_frac=float(opts["min_valid_pct"]) / 100.0,
                 z_min=float(opts["z_min"]), lift_min=float(opts["lift_min"]),
                 gain_min=float(opts["gain_min"]), ok_shift_m=float(opts["ok_shift_m"]))

    H, W = int(ctx.H), int(ctx.W)
    T = len(ctx.days)

    for pre in (_as_list(opts["sensors"]) or ["eco"]):
        sst_names = ctx.base_channels(f"{pre}_sst")
        if not sst_names:
            continue                                   # sensor not acquired -> emit nothing
        working = _fit_sst(ctx, sst_names[0])          # cloud-filtered SST of the primary version
        if working is None or working.ndim != 3:
            continue

        dy = np.zeros(T, "int16"); dx = np.zeros(T, "int16")
        shift_m = np.zeros(T, "float32")
        z = np.full(T, np.nan, "float32")
        agree = np.full(T, np.nan, "float32"); agree0 = np.full(T, np.nan, "float32")
        n_edge_a = np.zeros(T, "int32"); coast_obs = np.zeros(T, "int32")
        dt_k = np.full(T, np.nan, "float32")
        flag = np.full(T, FLAG["insufficient_signal"], "uint8")

        for t in range(T):
            sst = working[t]
            edges, _ = canny(sst, sigma, lo_pct, hi_pct)
            n_edge = int(edges.sum())
            q = scene_quality(sst, ref, water)
            n_edge_a[t] = n_edge; coast_obs[t] = q["coast_obs"]; dt_k[t] = q["dt_k"]
            if quality_reject(q, n_edge, gates) is not None:
                continue                               # stays `insufficient_signal`
            ys, xs = np.nonzero(edges)
            st = search_shift(ys, xs, dref, (H, W), max_cells, tol_cells, stride, n_refine)
            sm = res_m * float(np.hypot(st["dy"], st["dx"]))
            stable = _stable(ys, xs, dref, (H, W), windows_cells, tol_cells,
                             st["dy"], st["dx"], stride, n_refine)
            label = classify(st, sm, gates, stable)
            dy[t] = st["dy"]; dx[t] = st["dx"]; shift_m[t] = sm
            z[t] = st["z"]; agree[t] = st["agree"]; agree0[t] = st["agree0"]
            flag[t] = FLAG[label]

        p = f"{pre}_georef"
        ctx.emit(f"{p}_dy", ("time",), dy, long_name=f"{pre} best-fit north-south cell offset")
        ctx.emit(f"{p}_dx", ("time",), dx, long_name=f"{pre} best-fit east-west cell offset")
        ctx.emit(f"{p}_shift_m", ("time",), shift_m, units="m",
                 long_name=f"{pre} georeferencing shift magnitude")
        ctx.emit(f"{p}_z", ("time",), z, long_name=f"{pre} peak significance vs search surface")
        ctx.emit(f"{p}_agree", ("time",), agree, long_name=f"{pre} edge/coast agreement at peak")
        ctx.emit(f"{p}_agree0", ("time",), agree0, long_name=f"{pre} agreement at zero shift")
        ctx.emit(f"{p}_n_edge", ("time",), n_edge_a, long_name=f"{pre} thermal edge cell count")
        ctx.emit(f"{p}_coast_obs", ("time",), coast_obs,
                 long_name=f"{pre} coastline cells observed")
        ctx.emit(f"{p}_dt_k", ("time",), dt_k, units="K",
                 long_name=f"{pre} land-sea contrast (diagnostic, not a gate)")
        ctx.emit(f"{p}_flag", ("time",), flag,
                 flag_values=list(range(len(FLAG))), flag_meanings=FLAG_MEANINGS,
                 long_name=f"{pre} georeferencing classification")


# --------------------------------------------------------------------------- #
# Step: correct_georef -- apply the fitted shift to RAW ECOSTRESS geometry
# --------------------------------------------------------------------------- #
def _step_correct_georef(ctx: "PreprocessContext") -> None:
    """Shift the raw ECOSTRESS granule-native fields by `flag_georef`'s (dy,dx), for `displaced`
    scenes only, into `<field>_georef_corrected` channels.

    The fit was estimated on the cloud-filtered SST; the shift is applied to the RAW fields read
    from the raw cube (SST + validity by default). Companion preprocess-derived masks are
    deliberately NOT shifted -- the corrected geometry invalidates them, so the cloud / water
    preprocessing must be RE-RUN against the corrected images (future work). `np.roll` is never used
    (it would fabricate a coastline on the opposite side); the vacated margin is filled per field
    (SST -> NaN, mask -> 0). Never raises: absent flag channels or fields degrade to emitting nothing.
    """
    opts = {**_DEFAULTS, "fields": ["sst", "valid"], "fill": np.nan, **ctx.step_opts("correct_georef")}
    fields = _as_list(opts["fields"]) or ["sst", "valid"]

    for pre in (_as_list(opts["sensors"]) or ["eco"]):
        dy = _emitted(ctx, f"{pre}_georef_dy")
        dx = _emitted(ctx, f"{pre}_georef_dx")
        flag = _emitted(ctx, f"{pre}_georef_flag")
        if dy is None or dx is None or flag is None:
            continue                                   # flag_georef emitted nothing for this sensor
        applied = (flag == FLAG["displaced"])

        # Version suffixes come from the RAW SST channels (shared granule geometry across versions).
        # `base_channels` skips this step's own `_georef_corrected` output and the filters'
        # `_clean` product, both of which sit in the cube after an earlier run.
        for sst_name in ctx.base_channels(f"{pre}_sst"):
            suffix = sst_name[len(f"{pre}_sst"):]
            for fld in fields:
                name = f"{pre}_{fld}{suffix}"
                fill, dtype = _FIELD_FILL.get(fld, (float(opts["fill"]), "float32"))
                raw = ctx.read(name, dims=T3, dtype=dtype)
                if raw is None:
                    continue
                out = raw.copy()
                for t in np.nonzero(applied)[0]:
                    out[t] = shift_array(raw[t], int(dy[t]), int(dx[t]), fill=fill, dtype=dtype)
                ctx.emit(f"{name}{CORRECTED}", T3, out,
                         long_name=f"{name} shifted onto corrected georeferencing "
                                   f"(displaced scenes only)")

        ctx.emit(f"{pre}_georef_applied", ("time",), applied.astype("uint8"),
                 flag_values=[0, 1], flag_meanings="unchanged shifted",
                 long_name=f"{pre} scene had its georeferencing correction applied")


def _emitted(ctx: "PreprocessContext", name: str):
    """A 1-D `("time",)` channel emitted earlier this run (e.g. by flag_georef), or None."""
    if name in ctx.channels:
        d, arr = ctx.channels[name]
        if tuple(d) == ("time",):
            return np.asarray(arr)
    return None
