from __future__ import annotations
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from typing import Tuple

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
