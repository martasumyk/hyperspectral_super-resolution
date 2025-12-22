import numpy as np
import matplotlib.pyplot as plt
import rasterio
import ot 

from s2_emit import (
    load_s2_srf_from_xlsx,
    load_emit_envi_rfl, load_emit_wavelengths_from_nc,
    pseudo_s2_srf_integral
)

from s2_emit.viz import show_side_by_side
from s2_emit.color import apply_shared_percentile_stretch


def fit_ot_poly_rgb(
    src_rgb, ref_rgb, mask,
    deg=2,
    n_samples=5000,
    reg=0.05,
    numItermax=300,
    stopThr=1e-6,
    seed=0
):
    """
    Fit per-channel polynomial mapping y = poly(x) using OT barycentric targets.
    src_rgb, ref_rgb: (H,W,3) float in [0,1]
    mask: (H,W) boolean
    Returns coeffs: (3, deg+1) poly coefficients (highest power first) for R,G,B.
    """
    rng = np.random.default_rng(seed)

    X_all = src_rgb[mask].reshape(-1, 3).astype(np.float64)
    Y_all = ref_rgb[mask].reshape(-1, 3).astype(np.float64)
    X_all = X_all[np.isfinite(X_all).all(axis=1)]
    Y_all = Y_all[np.isfinite(Y_all).all(axis=1)]

    if X_all.shape[0] < 200 or Y_all.shape[0] < 200:
        coeffs = np.zeros((3, deg + 1), dtype=np.float64)
        coeffs[:, -2] = 1.0
        return coeffs

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

    coeffs = np.zeros((3, deg + 1), dtype=np.float64)
    for c in range(3):
        coeffs[c] = np.polyfit(X[:, c], Ybar[:, c], deg=deg)

    return coeffs


def apply_poly_rgb(rgb, coeffs, mask=None):
    """
    Apply per-channel polynomial mapping to RGB image in [0,1].
    coeffs: (3, deg+1) poly coefficients from np.polyfit (highest power first).
    """
    out = rgb.copy().astype(np.float32)

    if mask is None:
        for c in range(3):
            out[..., c] = np.polyval(coeffs[c], out[..., c])
        return np.clip(out, 0.0, 1.0)

    for c in range(3):
        x = out[..., c]
        y = np.polyval(coeffs[c], x)
        x2 = x.copy()
        x2[mask] = y[mask]
        out[..., c] = x2

    return np.clip(out, 0.0, 1.0)

hdr = "/content/hyperspectral_super-resolution/pairs_output/emit/2023-08-21_emit_overlap.hdr"
bin_path = "/content/hyperspectral_super-resolution/pairs_output/emit/2023-08-21_emit_overlap.bin"
nc_path  = "/content/hyperspectral_super-resolution/pairs_output/emit/EMIT_L2A_RFL_001_20230819T110126_2323107_023.nc"

s2_real_path = "/content/hyperspectral_super-resolution/pairs_output/s2/2023-08-21_s2_overlap.tif"

emit_grid_path = "/content/emit_grid_template.tif"
s2_grid_path   = s2_real_path


# srf
srf = load_s2_srf_from_xlsx(platform="S2A")
R = load_emit_envi_rfl(hdr, bin_path)
emit_w, good_mask = load_emit_wavelengths_from_nc(nc_path)

pseudo_s2 = pseudo_s2_srf_integral(R, emit_w, srf, good_mask=good_mask)

bands_rgb = ["B2", "B3", "B4"]  # [Blue, Green, Red]
emit_sim_60m = np.stack([pseudo_s2[b] for b in bands_rgb], axis=0).astype(np.float32)  # (3,H,W)

valid60 = np.isfinite(emit_sim_60m).all(axis=0) & (emit_sim_60m[0] > 0)


# downsampling
s2_real_60m = downsample_s2_to_grid(
    src_path=s2_real_path,
    dst_grid_path=emit_grid_path,
    band_indexes=[1, 2, 3],   
    src_scale=(1.0 / 255.0),
    resampling="average"
)

valid60 = valid60 & np.isfinite(s2_real_60m).all(axis=0)


# poly regression
emit_rgb_60m = np.transpose(emit_sim_60m[[2, 1, 0], ...], (1, 2, 0)) 

s2_rgb_60m   = np.transpose(s2_real_60m[[0, 1, 2], ...], (1, 2, 0)) 

emit_rgb_n = apply_shared_percentile_stretch(emit_rgb_60m, valid60)
s2_rgb_n   = apply_shared_percentile_stretch(s2_rgb_60m,   valid60)

coeffs = fit_ot_poly_rgb(
    src_rgb=emit_rgb_n,
    ref_rgb=s2_rgb_n,
    mask=valid60,
    deg=4,       
    n_samples=5000,
    reg=0.05,
    seed=0
)

emit_rgb_matched_60m = apply_poly_rgb(emit_rgb_n, coeffs, mask=valid60)

show_side_by_side(
    emit_rgb_matched_60m,
    s2_rgb_n,
    "EMIT_sim 60m (OT+poly)",
    "S2 real 60m (downsampled)"
)



emit_sim_10m = reproject_stack_to_grid(
    src_stack=emit_sim_60m,      
    src_grid_path=emit_grid_path, 
    dst_grid_path=s2_grid_path,  
    resampling="bilinear"
)

emit_rgb_10m = np.transpose(emit_sim_10m[[2, 1, 0], ...], (1, 2, 0))

mask10 = np.isfinite(emit_rgb_10m).all(axis=-1)

emit_rgb_10m_n = apply_shared_percentile_stretch(emit_rgb_10m, mask10)
emit_rgb_10m_matched = apply_poly_rgb(emit_rgb_10m_n, coeffs, mask=mask10)

with rasterio.open(s2_real_path) as ds:
    s2_rgb_u8 = np.transpose(np.stack([ds.read(1), ds.read(2), ds.read(3)], axis=0), (1, 2, 0))

show_side_by_side(
    emit_rgb_10m_matched,
    s2_rgb_u8.astype(np.float32) / 255.0,
    "EMIT_sim @10m + OT+poly",
    "S2 RGB (uint8)"
)
