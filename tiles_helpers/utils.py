import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.windows import from_bounds, transform as window_transform


def plot_tile_pair_simple(emit_tile_path, s2_tile_path, title_suffix="", save_path=None, show=True):
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

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def is_black_mask(arr, nodata=None, masked_val=-0.01,
                  nodata_atol=1e-3, zero_atol=1e-6):
    """
    arr: (bands, H, W)

    Pixel is 'black/invalid' if:
      - all bands ≈ nodata          (e.g. -9999)
      - OR all bands ≈ masked_val   (EMIT masked reflectance, ~ -0.01)
      - OR all bands ≈ 0            (true black)
    """
    if nodata is not None:
        nodata_mask = np.all(np.isclose(arr, nodata, atol=nodata_atol), axis=0)
    else:
        nodata_mask = np.zeros(arr.shape[1:], dtype=bool)

    masked_mask = np.all(np.isclose(arr, masked_val, atol=nodata_atol), axis=0)

    zero_mask = np.all(np.abs(arr) < zero_atol, axis=0)

    return nodata_mask | masked_mask | zero_mask


def find_valid_paired_tiles(
    emit_path,
    s2_path,
    emit_tile_size=100,
    scale=6,
    max_black_frac=0.0,
    max_tiles=None
):
    """
    Iterate over EMIT and S2 in paired tiles.
    Assumes:
      - EMIT and S2 cover the same area
      - spatial resolution ratio = `scale` (S2 is finer)
      - S2 height/width ≈ scale * EMIT height/width
    Returns a list of tile descriptors:
      [ { 'emit_window': Window, 's2_window': Window, 'idx': k }, ... ]
    """

    tiles = []

    with rasterio.open(emit_path) as emit_ds, rasterio.open(s2_path) as s2_ds:
        h_e, w_e = emit_ds.height, emit_ds.width
        h_s, w_s = s2_ds.height, s2_ds.width

        ratio_h = h_s / h_e
        ratio_w = w_s / w_e
        print(f"EMIT shape: {h_e}x{w_e}, S2 shape: {h_s}x{w_s}")
        print(f"Pixel ratio (h, w): {ratio_h:.3f}, {ratio_w:.3f}")


        emit_nodata = emit_ds.nodata
        s2_nodata = s2_ds.nodata

        tile_h_e = emit_tile_size
        tile_w_e = emit_tile_size
        tile_h_s = tile_h_e * scale
        tile_w_s = tile_w_e * scale

        idx = 0

        step_y = tile_h_e
        step_x = tile_w_e

        for row_e in range(0, h_e - tile_h_e + 1, step_y):
            for col_e in range(0, w_e - tile_w_e + 1, step_x):


                row_s = row_e * scale
                col_s = col_e * scale

                if (row_s + tile_h_s > h_s) or (col_s + tile_w_s > w_s):
                    continue

                w_emit = Window(col_e, row_e, tile_w_e, tile_h_e)
                w_s2   = Window(col_s, row_s, tile_w_s, tile_h_s)

                emit_tile = emit_ds.read(window=w_emit)
                s2_tile   = s2_ds.read(window=w_s2)

                emit_black = is_black_mask(emit_tile, nodata=emit_nodata)
                s2_black   = is_black_mask(s2_tile, nodata=s2_nodata)

                emit_black_frac = emit_black.sum() / emit_black.size
                s2_black_frac   = s2_black.sum() / s2_black.size

                if (emit_black_frac <= max_black_frac) and (s2_black_frac <= max_black_frac):
                    tiles.append(
                        {
                            "idx": idx,
                            "emit_window": w_emit,
                            "s2_window": w_s2,
                            "emit_black_frac": emit_black_frac,
                            "s2_black_frac": s2_black_frac,
                        }
                    )
                    idx += 1

                    if max_tiles is not None and len(tiles) >= max_tiles:
                        print(f"Collected {len(tiles)} tiles, stopping.")
                        return tiles

        print(f"Total valid tiles found: {len(tiles)}")
        return tiles


def save_tile_pair(
    emit_path,
    s2_path,
    tile_info,
    out_dir,
    *,
    tiled=True,
    overwrite=True,
    # EMIT uint16 scaling
    emit_scale=10000.0,          
    emit_nodata_u16=65535,
    # compression
    compress="DEFLATE",
    zlevel=1,
    num_threads="ALL_CPUS",
):

    def _auto_block_size(width: int, height: int) -> int:
        # Simple rule: smaller tiles -> 64; medium/large -> 256.
        m = min(width, height)
        if m >= 256:
            return 256
        if m >= 64:
            return 64
        return 16  # fallback (won't happen for your 100/600 tiles)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(tile_info["idx"])

    emit_out = out_dir / f"tile_{k:03d}_emit.tif"
    s2_out   = out_dir / f"tile_{k:03d}_s2.tif"

    if overwrite:
        emit_out.unlink(missing_ok=True)
        s2_out.unlink(missing_ok=True)

    w_emit = tile_info["emit_window"]
    w_s2   = tile_info["s2_window"]

    with rasterio.open(emit_path) as emit_ds, rasterio.open(s2_path) as s2_ds:
        emit_tile = emit_ds.read(window=w_emit)   # (bands, H, W)
        s2_tile   = s2_ds.read(window=w_s2)

        if emit_tile.size == 0:
            raise ValueError(f"Empty EMIT tile idx={k}, window={w_emit}")
        if s2_tile.size == 0:
            raise ValueError(f"Empty S2 tile idx={k}, window={w_s2}")

        emit_transform = window_transform(w_emit, emit_ds.transform)
        s2_transform   = window_transform(w_s2,   s2_ds.transform)

        # Preserve EMIT tags
        emit_ds_tags   = emit_ds.tags()
        emit_band_tags = [emit_ds.tags(i) for i in range(1, emit_ds.count + 1)]

        # --- EMIT: float -> uint16 (scaled) ---
        emit = emit_tile.astype(np.float32, copy=False)

        valid = np.isfinite(emit)
        if emit_ds.nodata is not None:
            valid &= (emit != emit_ds.nodata)

        scaled_i32 = np.rint(emit * float(emit_scale)).astype(np.int32, copy=False)
        # keep within uint16 range, reserve 65535 as nodata
        scaled_i32 = np.clip(scaled_i32, 0, int(emit_nodata_u16) - 1)

        emit_u16 = np.full(emit.shape, int(emit_nodata_u16), dtype=np.uint16)
        emit_u16[valid] = scaled_i32[valid].astype(np.uint16, copy=False)

        eh, ew = emit_u16.shape[1], emit_u16.shape[2]
        sh, sw = s2_tile.shape[1], s2_tile.shape[2]

        # Auto block sizes (simple + sane)
        e_blk = _auto_block_size(ew, eh)
        s_blk = _auto_block_size(sw, sh)

        emit_profile = dict(
            driver="GTiff",
            height=eh, width=ew,
            count=emit_u16.shape[0],
            dtype="uint16",
            crs=emit_ds.crs,
            transform=emit_transform,
            nodata=int(emit_nodata_u16),
            compress=compress,
            predictor=2,                  # good for integers
            ZLEVEL=int(zlevel),
            BIGTIFF="IF_SAFER",
            NUM_THREADS=str(num_threads),
            tiled=bool(tiled),
        )

        s2_is_int = np.issubdtype(s2_tile.dtype, np.integer)
        s2_profile = dict(
            driver="GTiff",
            height=sh, width=sw,
            count=s2_tile.shape[0],
            dtype=str(s2_tile.dtype),
            crs=s2_ds.crs,
            transform=s2_transform,
            nodata=s2_ds.nodata,
            compress=compress,
            predictor=2 if s2_is_int else 3,
            ZLEVEL=int(zlevel),
            BIGTIFF="IF_SAFER",
            NUM_THREADS=str(num_threads),
            tiled=bool(tiled),
        )

        # If tiling, set block sizes (keep it dead simple)
        if tiled:
            emit_profile.update(blockxsize=min(e_blk, ew), blockysize=min(e_blk, eh))
            s2_profile.update(blockxsize=min(s_blk, sw), blockysize=min(s_blk, sh))

        with rasterio.open(emit_out, "w", **emit_profile) as dst_e:
            dst_e.write(emit_u16)
            if emit_ds_tags:
                dst_e.update_tags(**emit_ds_tags)
            for i, bt in enumerate(emit_band_tags, start=1):
                if bt:
                    dst_e.update_tags(i, **bt)

        with rasterio.open(s2_out, "w", **s2_profile) as dst_s:
            dst_s.write(s2_tile)

    return emit_out, s2_out



def _subsample_bands_evenly(num_bands_total, num_keep=32):
    """Pick evenly spaced band indices (needed for EMIT) [0..num_bands_total-1]."""
    idx = np.linspace(0, num_bands_total - 1, num_keep).round().astype(int)
    idx = np.unique(idx)

    while len(idx) < num_keep:
        missing = num_keep - len(idx)
        add = []
        for i in range(len(idx) - 1):
            if len(add) >= missing:
                break
            mid = (idx[i] + idx[i + 1]) // 2
            add.append(int(mid))
        idx = np.unique(np.concatenate([idx, np.array(add, dtype=int)]))
    return idx[:num_keep]

def write_emit_b32_tile(emit_tile_path: Path, *, num_keep=32, idx_0based=None, overwrite=True):
    emit_tile_path = Path(emit_tile_path)
    out = emit_tile_path.with_name(emit_tile_path.stem + f"_b{num_keep}.tif")

    with rasterio.open(emit_tile_path) as src:
        if idx_0based is None:
            if src.count < num_keep:
                raise ValueError(f"Tile has only {src.count} bands, can't keep {num_keep}.")
            idx_0based = _subsample_bands_evenly(src.count, num_keep=num_keep)
        idx_0based = np.asarray(idx_0based, dtype=int)

        # If file exists and we’re not overwriting, still return consistent idx
        if out.exists() and not overwrite:
            return out, idx_0based

        profile = src.profile.copy()
        profile.update(count=len(idx_0based))

        # Optional: enforce training-friendly GeoTIFF settings (only if you want to guarantee them)
        # profile.update(driver="GTiff", tiled=True, compress="DEFLATE", predictor=2, zlevel=1)

        data = src.read((idx_0based + 1).tolist())
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data)

            desc = list(src.descriptions)
            for out_i, src0 in enumerate(idx_0based, start=1):
                d = desc[src0] if src0 < len(desc) else None
                if d:
                    dst.set_band_description(out_i, d)

    return out, idx_0based
