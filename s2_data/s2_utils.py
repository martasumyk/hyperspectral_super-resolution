import pyproj
from shapely.geometry import Point, box, Polygon
from datetime import datetime, timezone
import requests
from tqdm import tqdm
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
    best_asset_key, reproject_geom, scl_metrics, count_cloud_pixels
)


from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.windows import transform as win_transform



def find_best_s2_for_date(date_iso: str, lon: float, lat: float,
                          s2_collection, search_buffer, s2_api):
    """For the SAME date, search S2 in ROI; return least-cloudy item + cloud fraction using SCL."""
    bbox_poly = point_buffer_bbox(lon, lat, search_buffer)
    roi_geom = box(*bbox_poly.bounds)

    time_range = f"{date_iso}T00:00:00Z/{date_iso}T23:59:59Z"
    client = Client.open(s2_api)
    search = client.search(collections=[s2_collection], datetime=time_range, bbox=bbox_poly.bounds)
    items = list(search.get_items())
    if not items:
        return None, None

    best_item = None
    best_frac = None

    for item in tqdm(items, desc=f"S2 cloud check {date_iso}"):
        key = best_asset_key(item.assets, "scl")
        if key is None:
            continue

        url = item.assets[key].href

        # Strongly prefer GeoTIFF/Cog (usually 'scl'); JP2 often can't do efficient range reads.
        if key.lower() == "scl-jp2":
            continue  # or fallback to metadata-based cloud if you want

        try:
            clouds, total = count_cloud_pixels(url, roi_geom)   # <-- URL, not local file
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
        tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
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

def download_s2_truecolor(item, s2_dir) -> Path:
    """Download S2 visual (true color) if available; else save B04/B03/B02 and return JSON list."""
    assets = item.assets
    if "visual" in assets:
        href = assets["visual"].href
        out = s2_dir / f"{item.id}_visual.tif"
        if not out.exists():
            download_asset(href, out)
        return out
    band_paths = []
    for b in ("B04", "B03", "B02"):
        if b in assets:
            href = assets[b].href
            out = s2_dir / f"{item.id}_{b}.tif"
            if not out.exists():
                download_asset(href, out)
            band_paths.append(str(out))
    out_json = s2_dir / f"{item.id}_RGB_bands.json"
    out_json.write_text(json.dumps(band_paths, indent=2))
    return out_json


import numpy as np
from pathlib import Path
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

def _href_suffix(href: str) -> str:
    # strip query params if present
    base = href.split("?", 1)[0]
    suf = Path(base).suffix.lower()
    return suf if suf else ".tif"

def _download_band(item, key: str, out_dir: Path, stem: str) -> Path:
    href = item.assets[key].href
    ext = _href_suffix(href)
    out = out_dir / f"{stem}{ext}"
    if not out.exists():
        download_asset(href, out)
    return out

def download_s2_spectral_stack(item, s2_dir: Path) -> Path:
    """
    For STAC items with assets named like:
      blue, green, red, nir, rededge1/2/3, swir16, swir22, nir08 (optional)
    Create a single 10-band GeoTIFF on the 10m grid.
    """
    s2_dir = Path(s2_dir)
    s2_dir.mkdir(parents=True, exist_ok=True)

    assets = item.assets
    required = ["blue", "green", "red", "nir", "rededge1", "rededge2", "rededge3", "swir16", "swir22"]
    missing = [k for k in required if k not in assets]
    if missing:
        raise ValueError(f"Missing required assets: {missing}. Available: {list(assets.keys())}")

    paths = {}
    paths["blue"]     = _download_band(item, "blue",     s2_dir, f"{item.id}_blue")
    paths["green"]    = _download_band(item, "green",    s2_dir, f"{item.id}_green")
    paths["red"]      = _download_band(item, "red",      s2_dir, f"{item.id}_red")
    paths["nir"]      = _download_band(item, "nir",      s2_dir, f"{item.id}_nir")
    paths["rededge1"] = _download_band(item, "rededge1", s2_dir, f"{item.id}_rededge1")
    paths["rededge2"] = _download_band(item, "rededge2", s2_dir, f"{item.id}_rededge2")
    paths["rededge3"] = _download_band(item, "rededge3", s2_dir, f"{item.id}_rededge3")
    paths["swir16"]   = _download_band(item, "swir16",   s2_dir, f"{item.id}_swir16")
    paths["swir22"]   = _download_band(item, "swir22",   s2_dir, f"{item.id}_swir22")

    nir08_path = None
    if "nir08" in assets:
        nir08_path = _download_band(item, "nir08", s2_dir, f"{item.id}_nir08")

    out_stack = s2_dir / f"{item.id}_S2_10band_10m.tif"
    if out_stack.exists():
        return out_stack

    # Reference grid = blue (10m)
    with rasterio.open(paths["blue"]) as ref:
        H, W = ref.height, ref.width
        ref_transform = ref.transform
        ref_crs = ref.crs
        out_dtype = ref.dtypes[0]

    def warp_to_ref(src_path: Path, resampling):
        with rasterio.open(src_path) as src:
            dst = np.zeros((H, W), dtype=out_dtype)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling,
            )
            return dst

    include_nir08 = False
    if nir08_path is not None:
        with rasterio.open(paths["nir"]) as ds_nir, rasterio.open(nir08_path) as ds_nir08:
            nir_res = abs(ds_nir.transform.a)
            nir08_res = abs(ds_nir08.transform.a)
        include_nir08 = (nir08_res != nir_res)

    band_order = [
        ("B02_blue",     paths["blue"],     Resampling.nearest),
        ("B03_green",    paths["green"],    Resampling.nearest),
        ("B04_red",      paths["red"],      Resampling.nearest),
        ("B08_nir",      paths["nir"],      Resampling.nearest),
        ("B05_rededge1", paths["rededge1"], Resampling.bilinear),
        ("B06_rededge2", paths["rededge2"], Resampling.bilinear),
        ("B07_rededge3", paths["rededge3"], Resampling.bilinear),
    ]
    if include_nir08:
        band_order.append(("B8A_nir08", nir08_path, Resampling.bilinear))
    else:
        # If nir08 isn't usable, still keep 10 bands by duplicating nir as a placeholder is NOT recommended.
        # Better: only output 9 bands if nir08 isn't distinct.
        pass

    band_order += [
        ("B11_swir16", paths["swir16"], Resampling.bilinear),
        ("B12_swir22", paths["swir22"], Resampling.bilinear),
    ]

    # If nir08 isn't distinct, you'll end up with 9 bands; warn early
    if not include_nir08:
        print("WARNING: 'nir08' not included (missing or same resolution as 'nir'). Output will have 9 bands.")

    stack = np.stack([warp_to_ref(p, rs) for (_, p, rs) in band_order], axis=0)

    # Write GeoTIFF
    with rasterio.open(paths["blue"]) as ref:
        profile = ref.profile.copy()
    profile.update(
        driver="GTiff",
        height=H,
        width=W,
        count=stack.shape[0],
        dtype=stack.dtype,
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(out_stack, "w", **profile) as dst:
        dst.write(stack)
        for i, (name, _, _) in enumerate(band_order, start=1):
            dst.set_band_description(i, name)

    return out_stack


def crop_s2_stack_to_te(s2_stack_path, out_path, left, bottom, right, top, overwrite=False):
    s2_stack_path = Path(s2_stack_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print("Cropped S2 already exists.")
        return out_path

    with rasterio.open(s2_stack_path) as ds:
        w = from_bounds(left, bottom, right, top, transform=ds.transform)

        w_int = Window(
            col_off=int(round(w.col_off)),
            row_off=int(round(w.row_off)),
            width=int(round(w.width)),
            height=int(round(w.height)),
        )

        full = Window(0, 0, ds.width, ds.height)
        w_int = w_int.intersection(full)

        if w_int.width <= 0 or w_int.height <= 0:
            raise ValueError("Overlap window is empty after clipping. Check -te / CRS inputs.")

        profile = ds.profile.copy()
        profile.update(
            driver="GTiff",
            width=int(w_int.width),
            height=int(w_int.height),
            transform=win_transform(w_int, ds.transform),
            tiled=True,
            compress="DEFLATE",
            predictor=2,
            zlevel=1,
        )

        with rasterio.open(out_path, "w", **profile) as out:
            out.write(ds.read(window=w_int))

    print("Wrote:", out_path)
    return out_path
