import pyproj
from shapely.geometry import Point, box, Polygon
from datetime import datetime, timezone
import requests, tqdm
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show
from pathlib import Path

import json
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from pystac_client import Client
from s2_data.cloud_utils import (
    find_asset_key, reproject_geom, scl_metrics, count_cloud_pixels
)


def find_best_s2_for_date(date_iso: str, lon: float, lat: float, s2_collection, search_buffer, s2_api, s2_dir):
    """For the SAME date, search S2 in ROI; return least-cloudy item + cloud fraction using SCL."""
    bbox = point_buffer_bbox(lon, lat, search_buffer)
    roi_geom = box(*bbox.bounds)
    time_range = f"{date_iso}T00:00:00Z/{date_iso}T23:59:59Z"
    client = Client.open(s2_api)
    search = client.search(collections=[s2_collection], datetime=time_range, bbox=bbox.bounds)
    items = list(search.get_items())
    if not items:
        return None, None

    best_item = None
    best_frac = None
    for item in tqdm(items, desc=f"S2 cloud check {date_iso}"):
        key = find_asset_key(item.assets, ["scl", "scl-jp2"])
        asset = item.assets[key]
        url = asset.href
        ext = Path(url).suffix or (".tif" if key == "scl" else ".jp2")
        scl_path = s2_dir / f"{item.id}_SCL{ext}"
        if not scl_path.exists():
            download_asset(url, scl_path)

        try:
            clouds, total = count_cloud_pixels(str(scl_path), roi_geom)
        except (rasterio.errors.RasterioIOError, ValueError):

            continue

        frac = (clouds / total) if total else 1.0
        if best_frac is None or frac < best_frac:
            best_frac = frac
            best_item = item

    return best_item, best_frac


def point_buffer_bbox(lon: float, lat: float, meters: float):
    """
    Build a WGS84 polygon (bbox) centered at (lon, lat) whose sides
    are tangent to a circle of radius `meters` in a local AEQD projection.
    """
    wgs84 = pyproj.CRS.from_epsg(4326)
    aeqd = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    )

    fwd = pyproj.Transformer.from_crs(wgs84, aeqd, always_xy=True)  # lon,lat -> x,y (m)
    inv = pyproj.Transformer.from_crs(aeqd, wgs84, always_xy=True)  # x,y (m) -> lon,lat

    # Project center to local meters
    x0, y0 = fwd.transform(lon, lat)

    p_local = Point(x0, y0)
    bbox_local = box(*p_local.buffer(meters).bounds) 

    xs, ys = bbox_local.exterior.coords.xy 
    lons, lats = inv.transform(xs, ys)

    return Polygon(zip(lons, lats))



def get_item_dt_utc(it):
    ts = it.datetime if hasattr(it, "datetime") else it["datetime"]

    if isinstance(ts, str):
        ts = ts.replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts)

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return ts.astimezone(timezone.utc)


def download_asset(href, out_path):
    r = requests.get(href, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(out_path, "wb") as f, \
         tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
        for chunk in r.iter_content(chunk_size=2**20):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return str(out_path)


def plot_s2_truecolor(paths, ax=None):
    if len(paths) == 1:
        p = paths[0]
        with rasterio.open(p) as ds:
            idx = [1, 2, 3] if ds.count >= 3 else [1, 1, 1]
            arr = ds.read(idx).astype("float32")
        rgb = np.moveaxis(arr, 0, -1)

        if np.nanmax(rgb) > 1.5:
            rgb = rgb / 10000.0

    else:
        paths = list(map(Path, paths))
        bands = {}
        for p in paths:
            n = p.name
            if "_B04" in n: bands["R"] = str(p)
            if "_B03" in n: bands["G"] = str(p)
            if "_B02" in n: bands["B"] = str(p)

        if not all(k in bands for k in ("R", "G", "B")):
            raise ValueError("Expected B04, B03, and B02 bands for RGB composite.")

        with rasterio.open(bands["R"]) as r, \
             rasterio.open(bands["G"]) as g, \
             rasterio.open(bands["B"]) as b:

            R = r.read(1).astype("float32")
            G = g.read(1, out_shape=(r.height, r.width), resampling=Resampling.bilinear).astype("float32")
            B = b.read(1, out_shape=(r.height, r.width), resampling=Resampling.bilinear).astype("float32")

        rgb = np.dstack([R, G, B])
        if np.nanmax(rgb) > 1.5:
            rgb = rgb / 10000.0

    valid = np.isfinite(rgb)
    p2, p98 = np.nanpercentile(rgb[valid], [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(rgb)
    ax.set_title("Sentinel-2 true color (B04, B03, B02)")
    ax.axis("off")

    return rgb


from shapely.geometry import mapping, shape
from rasterio.warp import transform_geom
from rasterio.crs import CRS


def reproject_geom(geom, dst_crs, src_crs: str = "EPSG:4326"):
    """
    Reproject a Shapely geometry from src_crs (default: WGS84) to dst_crs.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Geometry in src_crs coordinates (e.g. lon/lat in EPSG:4326).
    dst_crs : str or rasterio.crs.CRS
        Target CRS (e.g. ds.crs from a rasterio dataset).
    src_crs : str, optional
        Source CRS. Defaults to "EPSG:4326".

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Geometry reprojected into dst_crs.
    """
    # Allow passing rasterio CRS object directly
    if isinstance(dst_crs, CRS):
        dst_crs = dst_crs.to_string()

    geojson = mapping(geom)  # shapely → GeoJSON-like dict
    new_geojson = transform_geom(src_crs, dst_crs, geojson)
    return shape(new_geojson)

def _save_roi_from_asset(href: str, roi_geom_wgs84, out_path: Path) -> Path:
    """Read only ROI from a remote S2 asset and save to out_path."""
    with rasterio.open(href) as src:
        roi_proj = reproject_geom(roi_geom_wgs84, src.crs)
        out_image, out_transform = rio_mask(
            src,
            [mapping(roi_proj)],
            crop=True
        )

        meta = src.meta.copy()
        meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(out_image)

    return out_path


def download_s2_truecolor_roi(item, roi_geom_wgs84, out_dir: Path | str = None) -> Path:
    """
    Download ONLY the ROI part of a Sentinel-2 truecolor image.

    - If 'visual' asset is present: save cropped visual as {item.id}_visual_roi.tif
    - Else: crop B04/B03/B02 and stack into an RGB GeoTIFF {item.id}_RGB_roi.tif

    Returns path to the ROI GeoTIFF.
    """
    if out_dir is None:
        out_dir = S2_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = item.assets

    # 1) Prefer precomputed truecolor ('visual')
    if "visual" in assets:
        href = assets["visual"].href
        out = out_dir / f"{item.id}_visual_roi.tif"
        if not out.exists():
            _save_roi_from_asset(href, roi_geom_wgs84, out)
        return out

    # 2) Otherwise build ROI RGB from B04/B03/B02
    bands = ("B04", "B03", "B02")
    hrefs = []
    for b in bands:
        if b not in assets:
            raise RuntimeError(f"Item {item.id} missing band {b} for truecolor RGB.")
        hrefs.append(assets[b].href)

    out = out_dir / f"{item.id}_RGB_roi.tif"
    if out.exists():
        return out

    # Use first band to define ROI extent/grid
    first_href = hrefs[0]
    with rasterio.open(first_href) as src0:
        roi_proj = reproject_geom(roi_geom_wgs84, src0.crs)
        img0, out_transform = rio_mask(
            src0,
            [mapping(roi_proj)],
            crop=True
        )
        meta = src0.meta.copy()

    # Allocate array for all 3 bands
    h, w = img0.shape[1], img0.shape[2]
    data = np.empty((len(bands), h, w), dtype=img0.dtype)
    data[0] = img0[0]

    # Read remaining bands with same ROI
    for i, href in enumerate(hrefs[1:], start=1):
        with rasterio.open(href) as src:
            roi_proj = reproject_geom(roi_geom_wgs84, src.crs)
            img, _ = rio_mask(
                src,
                [mapping(roi_proj)],
                crop=True
            )
            data[i] = img[0]

    meta.update(
        {
            "count": len(bands),
            "height": h,
            "width": w,
            "transform": out_transform,
        }
    )

    with rasterio.open(out, "w", **meta) as dst:
        dst.write(data)

    return out