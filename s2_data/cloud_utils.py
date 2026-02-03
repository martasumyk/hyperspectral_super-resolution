import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pyproj
from shapely.ops import transform as shp_transform
from shapely.geometry import mapping
from shapely.ops import transform
import tqdm


SCL_NAMES = {
    0:"No data",1:"Saturated/Defective",2:"Dark features/shadows",3:"Cloud shadows",
    4:"Vegetation",5:"Bare soils",6:"Water",7:"Unclassified",8:"Cloud med",9:"Cloud high",10:"Thin cirrus",11:"Snow/Ice"
}

# CLOUD_CLASSES = {8, 9, 10, 11}

# def count_cloud_pixels(scl_path: str, roi_geom_wgs84):
#     """Return (#cloud_pixels, #total_valid_pixels) within ROI from an SCL raster."""
#     with rasterio.open(scl_path) as ds:
#         roi_proj = reproject_geom(roi_geom_wgs84, ds.crs)
#         data, _ = rio_mask(ds, [mapping(roi_proj)], crop=True)
#         scl = data[0]
#         valid = scl != 0
#         total = int(valid.sum())
#         clouds = int(np.isin(scl, list(CLOUD_CLASSES)).sum())
#     return clouds, total

CLOUD_CLASSES = {8, 9, 10, 11}

def count_cloud_pixels(scl_href: str, roi_geom_wgs84):
    """Return (#cloud_pixels, #total_valid_pixels) within ROI from an SCL raster (URL or local)."""
    vsi_href = scl_href
    if scl_href.startswith("http://") or scl_href.startswith("https://"):
        vsi_href = f"/vsicurl/{scl_href}"

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        VSI_CACHE=True,
        VSI_CACHE_SIZE=50_000_000,
    ):
        with rasterio.open(vsi_href) as ds:
            roi_proj = reproject_geom(roi_geom_wgs84, ds.crs)
            data, _ = rio_mask(ds, [mapping(roi_proj)], crop=True)

            scl = data[0]
            valid = scl != 0
            total = int(valid.sum())
            clouds = int(np.isin(scl[valid], list(CLOUD_CLASSES)).sum())  # <-- only valid pixels
            return clouds, total



def best_asset_key(assets, base):
    """
    Prefer GeoTIFF/COG asset (e.g., 'scl') and fall back to JP2 (e.g., 'scl-jp2').
    Returns the actual key in assets (case-preserving), or None.
    """
    aset = {k.lower(): k for k in assets.keys()}
    for cand in (base, f"{base}-jp2"):
        k = aset.get(cand.lower())
        if k is not None:
            return k
    return None

ALIASES = {
    "SCL":  ["SCL","scl","scl-jp2"],
    "QA60": ["QA60","qa60","qa60-jp2"], 
    "B02":  ["B02","blue","blue-jp2"],
    "B03":  ["B03","green","green-jp2"],
    "B04":  ["B04","red","red-jp2"],
    "B08":  ["B08","nir","nir-jp2","nir08","nir08-jp2"],
}

def reproject_geom(geom_wgs84, dst_crs):
    tfm = pyproj.Transformer.from_crs(4326, dst_crs, always_xy=True).transform
    return transform(tfm, geom_wgs84)

def scl_metrics(scl_tif, roi_geom_wgs84, include_shadows=False):
    with rasterio.open(scl_tif) as ds:
        roi_in_ds = reproject_geom(roi_geom_wgs84, ds.crs)
        data, _ = rio_mask(ds, [mapping(roi_in_ds)], crop=True, filled=True)
        scl = data[0]
    vals, counts = np.unique(scl, return_counts=True)
    total = int(counts.sum())
    by_class = {int(v): int(c) for v, c in zip(vals, counts)}
    valid_mask = scl != 0
    cloud_set = {8,9,10} | ({3} if include_shadows else set())
    cloud_px = int(np.isin(scl, list(cloud_set))[valid_mask].sum())
    valid_px = int(valid_mask.sum())
    return {
        "total_px": total,
        "valid_px": valid_px,
        "nodata_px": by_class.get(0, 0),
        "cloud_px": cloud_px,
        "cloud_frac_valid": (cloud_px/valid_px) if valid_px else np.nan,
        "class_counts": {SCL_NAMES.get(k, str(k)): v for k, v in by_class.items()}
    }


def _reproject_geom(geom_wgs84, dst_crs):
    tfm = pyproj.Transformer.from_crs(4326, dst_crs, always_xy=True).transform
    return shp_transform(tfm, geom_wgs84)


def plot_scl_map(scl_tif_path, roi_geom_wgs84=None, out_png=None, title=None):
    scl_colors = [
        "#000000", "#ff00ff", "#4d4d4d", "#ffa500", "#00a600", "#bdb76b",
        "#2c7fb8", "#aaaaaa", "#fdae61", "#d7191c", "#abd9e9", "#ffffff"
    ]
    bounds = np.arange(-0.5, 12.5, 1.0)
    cmap = ListedColormap(scl_colors)
    norm = BoundaryNorm(bounds, cmap.N)

    with rasterio.open(scl_tif_path) as ds:
        if roi_geom_wgs84 is not None:
            roi_in_ds = _reproject_geom(roi_geom_wgs84, ds.crs)
            data, _ = rio_mask(ds, [mapping(roi_in_ds)], crop=True, filled=True)
            scl = data[0]
        else:
            scl = ds.read(1)

    if out_png is None:
        out_png = str(scl_tif_path).replace(".tif", ".scl_preview.png")
    if title is None:
        import os
        title = f"S2 L2A SCL: {os.path.basename(scl_tif_path)}"

    plt.figure(figsize=(9, 9))
    im = plt.imshow(scl, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=range(12), shrink=0.85)
    cbar.ax.set_yticklabels([SCL_NAMES.get(i, str(i)) for i in range(12)])
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved to:", out_png)
