from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

def pseudo_s2_srf_integral(
    R: np.ndarray,
    emit_w: np.ndarray,
    srf_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    good_mask: Optional[np.ndarray] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    SRF-weighted band synthesis.

    R: (H, W, B) reflectance
    emit_w: (B,) wavelengths (nm)
    good_mask: (B,) boolean, optional

    returns: band -> (H, W) float
    """
    out: Dict[str, Optional[np.ndarray]] = {}
    emit_w = emit_w.astype(float)

    if R.ndim != 3:
        raise ValueError(f"R must be (H,W,B). Got shape {R.shape}")
    if emit_w.ndim != 1 or emit_w.shape[0] != R.shape[-1]:
        raise ValueError(f"emit_w must be (B,) matching R bands. Got {emit_w.shape} vs {R.shape[-1]}")

    for band, (lam_srf, rsp_srf) in srf_dict.items():
        rsp_on_emit = np.interp(emit_w, lam_srf, rsp_srf, left=0.0, right=0.0)
        if good_mask is not None:
            rsp_on_emit = rsp_on_emit * good_mask.astype(float)

        if np.all(rsp_on_emit == 0):
            out[band] = None
            continue

        num = np.trapz(R * rsp_on_emit[None, None, :], x=emit_w, axis=-1)
        den = np.trapz(rsp_on_emit, x=emit_w)
        out[band] = num / (den + 1e-32)

    return out

def pseudo_s2_rgb(pseudo_s2: Dict[str, Optional[np.ndarray]], order=("B4","B3","B2")) -> np.ndarray:
    """
    Builds RGB stack from pseudo_s2 dict.
    Returns (H, W, 3)
    """
    chans = []
    for b in order:
        x = pseudo_s2.get(b, None)
        if x is None:
            raise ValueError(f"Band {b} is None/missing in pseudo_s2.")
        chans.append(x)
    return np.stack(chans, axis=-1)
