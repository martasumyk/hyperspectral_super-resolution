from __future__ import annotations
import numpy as np
import h5py
import spectral.io.envi as envi
from typing import Tuple, Optional

def load_emit_envi_rfl(hdr_path: str, bin_path: str, as_float32: bool = True) -> np.ndarray:
    """
    Loads EMIT reflectance ENVI pair into memory.
    Returns R: (H, W, B)
    """
    img = envi.open(hdr_path, bin_path)
    R = np.asarray(img.load())
    if as_float32:
        R = R.astype(np.float32, copy=False)
    return R

def load_emit_wavelengths_from_nc(
    nc_path: str,
    wavelengths_key: str = "sensor_band_parameters/wavelengths",
    good_key: str = "sensor_band_parameters/good_wavelengths",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (emit_wavelengths_nm, good_mask_bool_or_None)
    """
    with h5py.File(nc_path, "r") as f:
        emit_w = f[wavelengths_key][:].astype(np.float32)
        good_mask = None
        if good_key in f:
            good_mask = f[good_key][:].astype(bool)
    return emit_w, good_mask
