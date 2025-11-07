from __future__ import annotations
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _nearest_idx(wl: np.ndarray, nm: float) -> int:
    return int(np.nanargmin(np.abs(wl - float(nm))))

def plot_band_nm(ds: xr.Dataset, wavelength_nm: float, *, data_var='reflectance', p=(2,98), ax=None):
    wl = ds.coords.get('wavelength_nm')
    if wl is None:
        raise ValueError("wavelength_nm coord missing; call attach_wavelengths()")
    idx = _nearest_idx(wl.values, wavelength_nm)
    img = ds[data_var].isel({ds[data_var].dims[-1]: idx}).values
    lo, hi = np.nanpercentile(img, p)
    img = (img - lo) / (hi - lo + 1e-6)
    img = np.clip(img, 0, 1)
    ax = ax or plt.gca()
    ax.imshow(img)
    ax.set_title(f"~{wl.values[idx]:.0f} nm")
    ax.axis('off')
    return ax

def rgb_from_nm(ds: xr.Dataset, rgb_nm=(665.0,560.0,492.0), *, data_var='reflectance', p=(2,98), ax=None):
    wl = ds.coords.get('wavelength_nm')
    if wl is None:
        raise ValueError("wavelength_nm coord missing; call attach_wavelengths()")
    arr = ds[data_var]
    idx = [_nearest_idx(wl.values, nm) for nm in rgb_nm]
    ch = [arr.isel({arr.dims[-1]: i}).values for i in idx]
    out = []
    for c in ch:
        lo, hi = np.nanpercentile(c, p)
        c = (c - lo) / (hi - lo + 1e-6)
        out.append(np.clip(c, 0, 1))
    rgb = np.dstack(out)
    ax = ax or plt.gca()
    ax.imshow(rgb)
    ax.set_title(f"RGB ~{rgb_nm[0]:.0f}/{rgb_nm[1]:.0f}/{rgb_nm[2]:.0f} nm")
    ax.axis('off')
    return ax

def hv_quicklook(ds: xr.Dataset, wavelength_nm: float, *, data_var='reflectance', clim=(2,98)):
    """Interactive quicklook using hvplot (Bokeh), following the notebook vibe."""
    wl = ds.coords.get('wavelength_nm')
    if wl is None:
        raise ValueError("wavelength_nm coord missing")
    idx = _nearest_idx(wl.values, wavelength_nm)
    da = ds[data_var].isel({ds[data_var].dims[-1]: idx})
    return da.hvplot.image(rasterize=False, clim=clim, title=f"~{wl.values[idx]:.0f} nm")

def hv_quicklook_nearest_valid(
    ds,
    target_nm: float = 850.0,
    *,
    data_var: str = "reflectance",
    good_var: str = "good_wavelengths",
    wavelength_candidates = ("wavelength_nm", "wavelengths", "wavelength", "band_center", "wl"),
    rasterize: bool = False,
    frame_height: int = 600,
    cmap: str = "Viridis",
    verbose: bool = True,
):
    """
    Fast interactive quicklook:
      - Finds the nearest wavelength to `target_nm` among bands that are BOTH
        (a) flagged 'good' (if available) and (b) actually contain data in the scene.
      - Plots only that single 2-D slice with hvplot (no heavy multi-band ops).

    Returns:
        A Holoviews object (rendered by hvPlot).
    """
    if data_var not in ds:
        raise KeyError(f"{data_var!r} not found in dataset")

    R = ds[data_var]

    # ---- discover dims ----
    y_candidates = ("latitude", "y")
    x_candidates = ("longitude", "x")
    y = next((d for d in y_candidates if d in R.dims), None) or R.dims[0]
    x = next((d for d in x_candidates if d in R.dims), None) or R.dims[1]
    spec_dim = next((d for d in R.dims if d not in (y, x)), R.dims[-1])

    # ---- get wavelength vector (nm) ----
    w = None
    for cand in wavelength_candidates:
        if cand in ds:
            w = ds[cand]
            break
        if cand in ds.coords:
            w = ds.coords[cand]
            break
    if w is None:
        raise KeyError(
            "No wavelength coordinate/variable found. "
            "Expected one of: " + ", ".join(wavelength_candidates)
        )
    wv = np.asarray(w.values, dtype=float)
    if np.nanmax(wv) <= 10.0:  # micrometers -> nm
        wv = wv * 1000.0

    # ---- good-band mask (1D along spec_dim) ----
    if good_var in ds:
        gw = np.asarray(ds[good_var].values, dtype=bool)
    else:
        gw = np.ones_like(wv, dtype=bool)

    # ---- has-data mask across spatial dims (1D along spec_dim) ----
    has_data = np.isfinite(R).any(dim=(y, x)).values  # bool per band

    valid = gw & has_data
    if not np.any(valid):
        raise ValueError("No valid bands: good_wavelengths & has-data masks removed all bands.")

    # pick nearest valid band to target_nm
    idx_valid = np.where(valid)[0]
    pick_idx = int(idx_valid[np.nanargmin(np.abs(wv[idx_valid] - float(target_nm)))])
    picked_nm = float(wv[pick_idx])

    arr = R.isel({spec_dim: pick_idx}).squeeze()
    all_nan = bool(arr.isnull().all())

    if verbose:
        print(f"picked wavelength with data: {picked_nm:.1f} nm")
        print(f"all-NaN at picked band: {all_nan}")

    fig = arr.hvplot.image(
        x=x,
        y=y,
        cmap=cmap,
        aspect="equal",
        frame_height=frame_height,
        rasterize=rasterize,
        title=f"~{picked_nm:.0f} nm",
    ).opts(invert_yaxis=True)
    return fig

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def emit_quicklook_matplotlib(
    ds: xr.Dataset,
    *,
    rfl_var: str = "reflectance",
    wavelengths_var: str = "wavelengths",  
    good_var: str = "good_wavelengths",
    target_rgb_nm=(665.0, 560.0, 492.0),  
    grayscale_nm: float | None = None,    
    percentile=(2, 98),                       
    gamma: float = 1/2.2,                   
    white_balance: bool = True,             
    origin: str = "upper",                
    ax: plt.Axes | None = None,           
    return_arrays: bool = False            
):
    if rfl_var not in ds.data_vars:
        rfl_var = next(iter(ds.data_vars))

    if "wavelength_nm" in ds.coords:
        wl = np.asarray(ds["wavelength_nm"].values)
    elif wavelengths_var in ds:
        wl = np.asarray(ds[wavelengths_var].values)
    else:
        raise ValueError("No wavelengths found: need coord 'wavelength_nm' or variable "
                         f"'{wavelengths_var}' in the dataset.")

    nwl = wl.size

    spec_candidates = [d for d in ds[rfl_var].dims if ds.sizes[d] == nwl]
    if not spec_candidates:
        spec_candidates = [d for d, sz in ds.sizes.items() if sz == nwl]
    if len(spec_candidates) != 1:
        raise ValueError(f"Can't uniquely determine spectral dimension of length {nwl}. "
                         f"Candidates: {spec_candidates}")
    spec_dim = spec_candidates[0]

    if "wavelength_nm" not in ds.coords or ds.coords["wavelength_nm"].sizes.get(spec_dim, 0) != nwl:
        ds = ds.assign_coords({"wavelength_nm": (spec_dim, wl)})
    wl = np.asarray(ds["wavelength_nm"].values)  # refresh

    if good_var in ds:
        gw = ds[good_var]
        if gw.ndim != 1 or gw.sizes.get(spec_dim, None) != nwl:
            gw = xr.DataArray(np.asarray(gw.values), dims=(spec_dim,))
            ds = ds.assign({good_var: gw})
        good = np.asarray(ds[good_var].astype(bool).values)
    else:
        good = np.ones(nwl, dtype=bool)
    good_idx = np.flatnonzero(good) if good.any() else np.arange(nwl)

    def nearest_good_index(target_nm: float) -> tuple[int, float]:
        if good_idx.size == 0:
            raise ValueError("No good wavelengths available.")
        i = int(good_idx[np.argmin(np.abs(wl[good_idx] - target_nm))])
        return i, float(wl[i])

    def pct_stretch(img, p_low=2, p_high=98):
        out = np.empty_like(img, dtype=np.float32)
        if img.ndim == 3:  # RGB
            for c in range(3):
                ch = img[..., c]
                lo, hi = np.nanpercentile(ch, (p_low, p_high))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    out[..., c] = 0
                else:
                    out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:              # grayscale
            lo, hi = np.nanpercentile(img, (p_low, p_high))
            out = np.zeros_like(img, dtype=np.float32) if hi <= lo or not np.isfinite(hi) else \
                  np.clip((img - lo) / (hi - lo), 0, 1)
        return out

    if grayscale_nm is not None:
        idx, nm = nearest_good_index(grayscale_nm)
        band = ds[rfl_var].isel({spec_dim: idx}).values.astype(np.float32)
        band[~np.isfinite(band)] = np.nan
        band = np.clip(band, 0, 1)
        band = pct_stretch(band, *percentile)
        gray_disp = np.clip(band, 0, 1) ** gamma

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(gray_disp, cmap="gray", origin=origin)
        ax.set_title(f"EMIT ~{nm:.0f} nm")
        ax.set_axis_off()

        payload = {"picked_nm": (nm,), "band_idx": (idx,), "gray_disp": gray_disp, "spec_dim": spec_dim}
        return (ax, payload) if return_arrays else ax

    idx_r, nm_r = nearest_good_index(target_rgb_nm[0])
    idx_g, nm_g = nearest_good_index(target_rgb_nm[1])
    idx_b, nm_b = nearest_good_index(target_rgb_nm[2])

    R = ds[rfl_var].isel({spec_dim: idx_r}).values
    G = ds[rfl_var].isel({spec_dim: idx_g}).values
    B = ds[rfl_var].isel({spec_dim: idx_b}).values

    rgb = np.dstack([R, G, B]).astype(np.float32)
    rgb[~np.isfinite(rgb)] = np.nan
    rgb = np.clip(rgb, 0, 1)

    rgb = pct_stretch(rgb, *percentile)
    if white_balance:
        means = np.nanmean(rgb.reshape(-1, 3), axis=0)
        scale = np.nanmean(means) / np.maximum(means, 1e-6)
        rgb = np.clip(rgb * scale, 0, 1)

    rgb_disp = np.clip(rgb, 0, 1) ** gamma

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb_disp, origin=origin)
    ax.set_title(f"EMIT True Color ~ {nm_r:.0f}/{nm_g:.0f}/{nm_b:.0f} nm")
    ax.set_axis_off()

    payload = {
        "picked_nm": (nm_r, nm_g, nm_b),
        "band_idx": (idx_r, idx_g, idx_b),
        "rgb_disp": rgb_disp,
        "spec_dim": spec_dim,
    }
    return (ax, payload) if return_arrays else ax
