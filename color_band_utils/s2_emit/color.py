from __future__ import annotations
import numpy as np
import ot  # POT
from typing import Optional

def robust_norm(x: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    lo, hi = np.nanpercentile(x, [pmin, pmax])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)

def robust_norm_rgb(img: np.ndarray, mask: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    """
    Per-channel percentile stretch within mask.
    img: (H,W,3)
    mask: (H,W) bool
    """
    y = np.zeros_like(img, dtype=float)
    for c in range(3):
        vals = img[..., c][mask]
        lo, hi = np.percentile(vals, [pmin, pmax])
        cc = (img[..., c] - lo) / (hi - lo + 1e-12)
        cc[~mask] = np.nan
        y[..., c] = np.clip(cc, 0, 1)
    return y

def apply_shared_percentile_stretch(img: np.ndarray, mask: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    """
    Same idea as above, but returns float32 and keeps values in [0,1].
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        vals = img[..., c][mask]
        lo, hi = np.percentile(vals, [pmin, pmax])
        out[..., c] = np.clip((img[..., c] - lo) / (hi - lo + 1e-12), 0, 1)
    return out

def _hist_match_channel(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_vals = src[mask].ravel()
    ref_vals = ref[mask].ravel()

    s_values, s_idx, s_counts = np.unique(src_vals, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(ref_vals, return_counts=True)

    s_quant = np.cumsum(s_counts).astype(np.float64); s_quant /= (s_quant[-1] + 1e-32)
    r_quant = np.cumsum(r_counts).astype(np.float64); r_quant /= (r_quant[-1] + 1e-32)

    interp_r_values = np.interp(s_quant, r_quant, r_values)
    matched = interp_r_values[s_idx].reshape(src_vals.shape)

    out = src.copy()
    out_masked = out[mask].ravel()
    out_masked[:] = matched
    out[mask] = out_masked.reshape(out[mask].shape)
    return out

def histogram_match_rgb(src_rgb: np.ndarray, ref_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Histogram-match each channel independently within mask.
    Inputs assumed in [0,1].
    """
    out = src_rgb.copy()
    for c in range(3):
        out[..., c] = _hist_match_channel(out[..., c], ref_rgb[..., c], mask)
    return np.clip(out, 0, 1)

def ot_match_rgb_sinkhorn_pot(
    src_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    mask: np.ndarray,
    n_samples: int = 5_000,
    reg: float = 0.05,
    numItermax: int = 300,
    stopThr: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    """
    3D color transfer using Sinkhorn OT on masked RGB samples + affine fit.
    src_rgb, ref_rgb: (H,W,3) in [0,1]
    mask: (H,W) boolean
    """
    rng = np.random.default_rng(seed)

    X_all = src_rgb[mask].reshape(-1, 3).astype(np.float64)
    Y_all = ref_rgb[mask].reshape(-1, 3).astype(np.float64)

    X_all = X_all[np.isfinite(X_all).all(axis=1)]
    Y_all = Y_all[np.isfinite(Y_all).all(axis=1)]

    if X_all.shape[0] < 2 or Y_all.shape[0] < 2:
        return src_rgb.copy()

    ns = min(n_samples, X_all.shape[0])
    nt = min(n_samples, Y_all.shape[0])

    X = X_all[rng.choice(X_all.shape[0], size=ns, replace=False)]
    Y = Y_all[rng.choice(Y_all.shape[0], size=nt, replace=False)]

    a = np.full(ns, 1.0 / ns, dtype=np.float64)
    b = np.full(nt, 1.0 / nt, dtype=np.float64)

    M = ot.dist(X, Y, metric="sqeuclidean")
    P = ot.sinkhorn(a, b, M, reg=reg, numItermax=numItermax, stopThr=stopThr)

    row_sum = P.sum(axis=1, keepdims=True) + 1e-32
    Ybar = (P @ Y) / row_sum

    X_aug = np.concatenate([X, np.ones((ns, 1))], axis=1)
    W, *_ = np.linalg.lstsq(X_aug, Ybar, rcond=None)
    A = W[:3, :]
    t = W[3, :]

    out = src_rgb.copy().astype(np.float32)
    Xm = out[mask].reshape(-1, 3).astype(np.float64)
    Xm2 = Xm @ A + t
    Xm2 = np.clip(Xm2, 0.0, 1.0)
    out[mask] = Xm2.reshape(out[mask].shape).astype(np.float32)
    return out
