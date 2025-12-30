import matplotlib.pyplot as plt
import rasterio
import os
import glob
import numpy as np



def _nearest_band_indices(wavelengths, targets_nm):
    """Return indices in `wavelengths` nearest to each nm in `targets_nm`."""
    wl = np.asarray(wavelengths, dtype=float)
    idxs, picked = [], []
    for t in targets_nm:
        i = int(np.argmin(np.abs(wl - float(t))))
        idxs.append(i)
        picked.append(wl[i])
        print(f"Picked band {i} at {wl[i]:.1f} nm for target {t:.1f} nm")
    return idxs, picked


def _parse_wavelengths_from_tags(tags: dict):
    """
    ENVI stores wavelengths as a header key like:
      'wavelength': '{ 400.0, 401.0, ... }'
    This parses it into a list[float]. Returns None if absent.
    """
    w_txt = tags.get('wavelength') or tags.get('WAVELENGTH')
    if not w_txt:
        return None
    # normalize to Python list literal
    w_txt = w_txt.strip()
    if w_txt.startswith('{') and w_txt.endswith('}'):
        w_txt = '[' + w_txt[1:-1] + ']'
    # remove potential line breaks, duplicate spaces
    w_txt = re.sub(r'\s+', ' ', w_txt)
    try:
        vals = ast.literal_eval(w_txt)
        return [float(v) for v in vals]
    except Exception:
        return None
    
def _percentile_stretch(img, p_low=2, p_high=98):
    """
    Percentile-stretch to [0,1].
    - If img is HxWxC: stretch each channel independently (works for any C>=1)
    - If img is HxW: stretch as a single band
    Ignores non-finite values.
    """
    img = img.astype(np.float32, copy=False)
    out = np.zeros_like(img, dtype=np.float32)

    if img.ndim == 3:
        for c in range(img.shape[2]):
            ch = img[..., c]
            finite = np.isfinite(ch)
            if not np.any(finite):
                continue
            lo, hi = np.percentile(ch[finite], (p_low, p_high))
            if hi > lo:
                out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    else:
        finite = np.isfinite(img)
        if np.any(finite):
            lo, hi = np.percentile(img[finite], (p_low, p_high))
            if hi > lo:
                out = np.clip((img - lo) / (hi - lo), 0, 1)

    return out

def show_emit_rgb_from_envi(out_dir,
                            pattern="SISTER_EMIT_L2A_RFL_*_000.bin",
                            targets_nm=(630.0, 532.0, 465.0),
                            percentile=(2, 98),
                            gamma=1/2.2,
                            white_balance=True,
                            ax=None):
    """
    Find the EMIT RFL ENVI output in out_dir and display an RGB.

    If `ax` is provided, the image is drawn on that axes (e.g. for subplots).
    If `ax` is None, a new figure and axes are created.
    """
    # 1) pick file
    bins = sorted(glob.glob(os.path.join(out_dir, pattern)))
    if not bins:
        raise FileNotFoundError(f"No files matching {pattern} in {out_dir}")
    data_bin = bins[-1]  # latest
    print(f"Reading: {os.path.basename(data_bin)}")

    # 2) open with rasterio, read metadata + bands
    with rasterio.open(data_bin) as ds:
        # wavelengths from header tags (file-level)
        tags = ds.tags()
        wavelengths = _parse_wavelengths_from_tags(tags)
        if wavelengths is None:
            # sometimes wavelengths end up as per-band tags; try that
            wavelengths = []
            for b in range(1, ds.count+1):
                bt = ds.tags(b)
                w = bt.get('wavelength') or bt.get('WAVELENGTH')
                wavelengths.append(float(w) if w else np.nan)
            if not np.isfinite(wavelengths).any():
                raise ValueError("No wavelengths found in ENVI header tags.")

        idxs, picked = _nearest_band_indices(wavelengths, targets_nm)
        # rasterio bands are 1-based
        R = ds.read(idxs[0] + 1).astype(np.float32)
        G = ds.read(idxs[1] + 1).astype(np.float32)
        B = ds.read(idxs[2] + 1).astype(np.float32)

        # 3) handle nodata and reflectance scaling (EMIT reflectance should be 0..1)
        nodata = ds.nodata
        if nodata is not None:
            for arr in (R, G, B):
                arr[arr == nodata] = np.nan

        # clip extreme outliers (safety), then percentile-stretch
        rgb = np.dstack([np.clip(R, 0, 1),
                         np.clip(G, 0, 1),
                         np.clip(B, 0, 1)])

        rgb = _percentile_stretch(rgb, *percentile)

        if white_balance:
            means = np.nanmean(rgb.reshape(-1, 3), axis=0)
            scale = np.nanmean(means) / np.maximum(means, 1e-6)
            rgb = np.clip(rgb * scale, 0, 1)

        rgb_disp = np.clip(rgb, 0, 1) ** gamma

        # 4) plot
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            created_fig = True

        ax.imshow(rgb_disp, origin="upper")
        epsg = ds.crs.to_string() if ds.crs else "unknown CRS"
        ax.set_title(f"EMIT True Color ~ {picked[0]:.0f}/{picked[1]:.0f}/{picked[2]:.0f} nm\n{epsg}")
        ax.axis("off")

        if created_fig:
            plt.show()