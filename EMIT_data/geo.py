from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import xarray as xr

from .emit_tools import apply_glt as _et_apply_glt  # type: ignore
from .emit_tools import get_pixel_center_coords as _et_pxcoords  # type: ignore


def get_pixel_center_coords(ds: xr.Dataset, x_name='x', y_name='y') -> Tuple[np.ndarray, np.ndarray]:
    """Return meshgrid of pixel center coordinates from dataset coords.
    If emit_tools is available and provides this, use it.
    """
    return _et_pxcoords(ds)

def apply_glt(
    arr: xr.DataArray,
    glt_x: np.ndarray,
    glt_y: np.ndarray,
    *,
    out_x: Optional[np.ndarray] = None,
    out_y: Optional[np.ndarray] = None,
    fill: float = np.nan,
    method: str = 'nearest',
) -> xr.DataArray:
    """Apply a GLT mapping (image → map grid) to a 2D or 3D DataArray.

    This mirrors the notebook approach: use GLT index maps to place pixels on an output grid.
    If `emit_tools` is installed, its implementation is preferred.
    """
    return _et_apply_glt(arr, glt_x, glt_y, out_x=out_x, out_y=out_y, fill=fill, method=method)


def ortho_xr(ds: xr.Dataset, glt_ds: xr.Dataset, *, data_var: str='reflectance', method: str='nearest') -> xr.Dataset:
    """Orthorectify an EMIT dataset using a GLT dataset.
    Expects `glt_ds` to have arrays `x_map` and `y_map` (or similar names).
    """
    # Identify GLT arrays
    for xk in ('x_map','map_x','x','X'):
        if xk in glt_ds: glt_x = glt_ds[xk].values; break
    else:
        raise KeyError('GLT x map not found')
    for yk in ('y_map','map_y','y','Y'):
        if yk in glt_ds: glt_y = glt_ds[yk].values; break
    else:
        raise KeyError('GLT y map not found')

    arr = ds[data_var]
    ortho = apply_glt(arr, glt_x, glt_y, method=method)
    return xr.Dataset({data_var: ortho}).assign_attrs(ds.attrs)
