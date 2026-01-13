from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

from rasterio.windows import from_bounds
from rasterio.windows import transform as win_transform
import rasterio

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


def crop_to_overlap(s2_path, emit_path, out_s2_path, out_emit_path):
    with rasterio.open(s2_path) as s2_ds, rasterio.open(emit_path) as emit_ds:
        if s2_ds.crs != emit_ds.crs:
            raise ValueError(f"CRS mismatch: {s2_ds.crs} != {emit_ds.crs}.")

        s2_b = s2_ds.bounds
        e_b  = emit_ds.bounds

        left   = max(s2_b.left,   e_b.left)
        right  = min(s2_b.right,  e_b.right)
        bottom = max(s2_b.bottom, e_b.bottom)
        top    = min(s2_b.top,    e_b.top)

        if not (left < right and bottom < top):
            raise ValueError("No overlap between S2 and EMIT extents.")

        overlap_bounds = (left, bottom, right, top)

        s2_win   = from_bounds(*overlap_bounds, transform=s2_ds.transform).round_offsets().round_lengths()
        emit_win = from_bounds(*overlap_bounds, transform=emit_ds.transform).round_offsets().round_lengths()

        s2_data   = s2_ds.read(window=s2_win)
        emit_data = emit_ds.read(window=emit_win)

        s2_desc      = s2_ds.descriptions
        s2_ds_tags   = s2_ds.tags()
        s2_band_tags = [s2_ds.tags(i) for i in range(1, s2_ds.count + 1)]

        emit_ds_tags   = emit_ds.tags()
        emit_band_tags = [emit_ds.tags(i) for i in range(1, emit_ds.count + 1)]

        s2_profile = s2_ds.profile.copy()
        s2_profile.update(
            driver="GTiff",
            height=s2_data.shape[1],
            width=s2_data.shape[2],
            count=s2_data.shape[0],
            dtype=s2_data.dtype,
            transform=win_transform(s2_win, s2_ds.transform),
            compress="DEFLATE",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )

        emit_profile = emit_ds.profile.copy()
        emit_profile.update(
            driver="GTiff",
            height=emit_data.shape[1],
            width=emit_data.shape[2],
            count=emit_data.shape[0],
            dtype=emit_data.dtype,
            transform=win_transform(emit_win, emit_ds.transform),
            compress="DEFLATE",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )

    with rasterio.open(out_s2_path, "w", **s2_profile) as dst:
        dst.write(s2_data)
        # dataset-level tags
        if s2_ds_tags:
            dst.update_tags(**s2_ds_tags)
        # per-band descriptions
        for i, d in enumerate(s2_desc, start=1):
            if d:
                dst.set_band_description(i, d)
        # optional: per-band tags
        for i, bt in enumerate(s2_band_tags, start=1):
            if bt:
                dst.update_tags(i, **bt)

    with rasterio.open(out_emit_path, "w", **emit_profile) as dst:
        dst.write(emit_data)
        dst.update_tags(**emit_ds_tags)
        for i, bt in enumerate(emit_band_tags, start=1):
            if bt:
                dst.update_tags(i, **bt)

    return out_s2_path, out_emit_path
