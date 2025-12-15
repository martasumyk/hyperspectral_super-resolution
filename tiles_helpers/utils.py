import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.windows import from_bounds, transform as win_transform



def compute_invalid_mask(tile, nodata, zero_threshold=1e-6):
    """
    tile: (bands, H, W)

    Returns mask (H, W) where pixels are considered invalid:
      - all bands ~= nodata, OR
      - all bands are very close to 0
    """
    if nodata is not None:
        is_nodata = np.all(np.isclose(tile, nodata, atol=zero_threshold), axis=0)
    else:
        is_nodata = np.zeros(tile.shape[1:], dtype=bool)

    is_zero = np.all(np.abs(tile) < zero_threshold, axis=0)
    invalid = is_nodata | is_zero
    return invalid


def is_black_mask(arr, nodata=None):
    """
    arr: (bands, H, W)
    Returns a boolean (H, W) mask where pixels are 'black'.
    - If nodata is given: pixel is black if *all* bands == nodata
    - Else: pixel is black if *all* bands == 0
    """
    if nodata is not None:
        return np.all(arr == nodata, axis=0)
    else:
        return np.all(arr == 0, axis=0)



def make_paired_tiles(
    emit_path,
    s2_path,
    out_root,
    emit_tile_size=100,
    overlap_frac=0.1,
    max_invalid_frac=1,
    zero_threshold=1e-6,
):
    """
    Create paired tiles:
      - EMIT tile: emit_tile_size x emit_tile_size   (e.g. 100 x 100)
      - S2 tile:   (emit_tile_size * scale)²        (e.g. 600 x 600 if scale=6)

    overlap_frac: e.g., 0.1 -> 10% overlap → stride = 0.9 * tile size
    max_invalid_frac: max fraction of invalid (black/nodata) pixels allowed
                      separately for EMIT and S2. If either exceeds this threshold,
                      the tile is skipped.
    """
    out_root = Path(out_root)
    out_emit_dir = out_root / "emit_tiles"
    out_s2_dir   = out_root / "s2_tiles"
    out_emit_dir.mkdir(parents=True, exist_ok=True)
    out_s2_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(emit_path) as e_ds, rasterio.open(s2_path) as s_ds:
        assert e_ds.crs == s_ds.crs, "CRS mismatch between EMIT and S2"

        e_res = abs(e_ds.transform.a) 
        s_res = abs(s_ds.transform.a)

        scale_float = e_res / s_res
        scale = int(round(scale_float))
        print(f"Pixel sizes: EMIT={e_res}, S2={s_res}, scale≈{scale_float:.2f}")

        if scale <= 0 or abs(scale_float - scale) > 0.1:
            raise ValueError(
                f"EMIT/S2 resolutions not a near-integer factor (got {scale_float:.2f}). "
                "Check that EMIT is coarser than S2."
            )

        s2_tile_size = emit_tile_size * scale

        emit_stride = int(emit_tile_size * (1 - overlap_frac))
        s2_stride   = emit_stride * scale

        e_h, e_w = e_ds.height, e_ds.width
        s_h, s_w = s_ds.height, s_ds.width

        print(f"EMIT size: {e_w} x {e_h}")
        print(f"S2   size: {s_w} x {s_h}")
        print(f"Tile sizes: EMIT={emit_tile_size}, S2={s2_tile_size}")
        print(f"Strides:    EMIT={emit_stride}, S2={s2_stride}")
        print(f"max_invalid_frac={max_invalid_frac}")

        emit_nodata = e_ds.nodata
        s2_nodata   = s_ds.nodata

        emit_ds_tags   = e_ds.tags()
        emit_band_tags = [e_ds.tags(i) for i in range(1, e_ds.count + 1)]
        s2_ds_tags     = s_ds.tags()

        tile_id = 0
        kept = 0

        for er in range(0, e_h - emit_tile_size + 1, emit_stride):
            for ec in range(0, e_w - emit_tile_size + 1, emit_stride):
                sr = er * scale
                sc = ec * scale

                if sr + s2_tile_size > s_h or sc + s2_tile_size > s_w:
                    tile_id += 1
                    continue

                emit_win = Window(ec, er, emit_tile_size, emit_tile_size)
                s2_win   = Window(sc, sr, s2_tile_size, s2_tile_size)

                emit_tile = e_ds.read(window=emit_win)   # (bands, 100, 100)
                s2_tile   = s_ds.read(window=s2_win)     # (bands, 600, 600)

                inv_emit = compute_invalid_mask(emit_tile, emit_nodata, zero_threshold)
                inv_s2   = compute_invalid_mask(s2_tile, s2_nodata, zero_threshold)

                invalid_emit_frac = float(inv_emit.mean())
                invalid_s2_frac   = float(inv_s2.mean())

                if (invalid_emit_frac > max_invalid_frac) or (invalid_s2_frac > max_invalid_frac):
                    tile_id += 1
                    continue

                emit_profile = e_ds.profile.copy()
                emit_profile.update({
                    "height": emit_tile_size,
                    "width":  emit_tile_size,
                    "transform": win_transform(emit_win, e_ds.transform),
                })

                emit_fname = out_emit_dir / f"emit_tile_{tile_id:05d}.bin"
                with rasterio.open(emit_fname, "w", **emit_profile) as dst:
                    dst.write(emit_tile)
                    dst.update_tags(**emit_ds_tags)
                    for i, bt in enumerate(emit_band_tags, start=1):
                        if bt:
                            dst.update_tags(i, **bt)

                s2_profile = s_ds.profile.copy()
                s2_profile.update({
                    "height": s2_tile_size,
                    "width":  s2_tile_size,
                    "transform": win_transform(s2_win, s_ds.transform),
                })

                s2_fname = out_s2_dir / f"s2_tile_{tile_id:05d}.tif"
                with rasterio.open(s2_fname, "w", **s2_profile) as dst:
                    dst.write(s2_tile)
                    dst.update_tags(**s2_ds_tags)

                kept += 1
                tile_id += 1

        print(f"Done. Kept {kept} tiles (max_invalid_frac={max_invalid_frac}).")
        return out_emit_dir, out_s2_dir




def plot_tile_pair_simple(emit_tile_path, s2_tile_path, title_suffix=""):
    """
    Plot one pair of perfectly matching (geometrically) tiles.
    """
    emit_tile_path = Path(emit_tile_path)
    s2_tile_path   = Path(s2_tile_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    with rasterio.open(s2_tile_path) as ds_s2:
        rgb = ds_s2.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
        vmin = np.percentile(rgb, 2)
        vmax = np.percentile(rgb, 98)
        rgb = np.clip((rgb - vmin) / (vmax - vmin + 1e-6), 0, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"S2 tile {title_suffix}")
        ax1.axis("off")

    with rasterio.open(emit_tile_path) as ds_e:
        best_b, best_var = 1, -1.0
        for b in range(1, ds_e.count + 1):
            arr = ds_e.read(b).astype(np.float32)
            v = float(np.var(arr))
            if v > best_var:
                best_var = v
                best_b = b

        band = ds_e.read(best_b).astype(np.float32)
        vmin = np.percentile(band, 2)
        vmax = np.percentile(band, 98)
        img = (band - vmin) / (vmax - vmin + 1e-6)
        img = np.clip(img, 0, 1) ** 0.5

        print(f"[{title_suffix}] EMIT band {best_b}, var={best_var:.3e}, "
              f"min={band.min()}, max={band.max()}")

        ax2.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax2.set_title(f"EMIT tile {title_suffix}\n(best band {best_b})")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()