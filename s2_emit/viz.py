from __future__ import annotations
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from typing import Tuple


from rasterio.warp import transform

def load_s2_rgb_u8(s2_path: str, bands=(1,2,3)) -> np.ndarray:
    """
    Loads a 3-band S2 RGB (uint8 or uint16 etc) and returns (H,W,3) array.
    """
    with rasterio.open(s2_path) as src:
        arr = np.stack([src.read(b) for b in bands], axis=-1)
    return arr

def resize_s2_rgb_to(s2_rgb: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    """
    target_hw: (H, W)
    """
    H, W = target_hw
    return cv2.resize(s2_rgb, (W, H), interpolation=cv2.INTER_AREA)

def show_side_by_side(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str, figsize=(12,5)):
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1); plt.imshow(left);  plt.title(left_title);  plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(right); plt.title(right_title); plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_raster_geometry(name, path):
    with rasterio.open(path) as ds:
        b = ds.bounds
        crs = ds.crs
        transform_aff = ds.transform

        # Corners in raster CRS (UTM for S2 & EMIT)
        corners_utm = {
            "UL": (b.left,  b.top),     # upper-left
            "UR": (b.right, b.top),     # upper-right
            "LR": (b.right, b.bottom),  # lower-right
            "LL": (b.left,  b.bottom)   # lower-left
        }

        # Center in raster CRS
        center_utm = ((b.left + b.right) / 2, (b.top + b.bottom) / 2)

        # Convert to WGS84
        xs = [p[0] for p in corners_utm.values()] + [center_utm[0]]
        ys = [p[1] for p in corners_utm.values()] + [center_utm[1]]

        lon, lat = transform(crs, CRS.from_epsg(4326), xs, ys)
        corners_wgs84 = {k: (lon[i], lat[i]) for i, k in enumerate(corners_utm.keys())}
        center_wgs84 = (lon[-1], lat[-1])

    print(f"\n==============================")
    print(f"{name} — Geometry")
    print(f"File: {path}")
    print(f"CRS:  {crs}")
    print("------------------------------")

    print("Corners (UTM):")
    for k, v in corners_utm.items():
        print(f"  {k}: {v}")

    print("\nCorners (WGS84):")
    for k, v in corners_wgs84.items():
        print(f"  {k}: (lon={v[0]:.6f}, lat={v[1]:.6f})")

    print("\nCenter (UTM):", center_utm)
    print("Center (WGS84): (lon=%.6f, lat=%.6f)" % center_wgs84)
    print("==============================\n")