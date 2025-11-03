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

def plot_s2_truecolor(paths):

    def stretch01(x):
        p2, p98 = np.nanpercentile(x, [2, 98])
        return np.clip((x - p2) / (p98 - p2 + 1e-6), 0, 1)

    if len(paths) == 1:
        p = paths[0]
        with rasterio.open(p) as ds:


            count = ds.count
            idx = [1,2,3] if count >= 3 else [1]*3
            arr = ds.read(idx, masked=True)          # (3, H, W)
            rgb = np.moveaxis(arr, 0, -1).astype("float32")  # (H, W, 3)
            rgb = np.where(np.ma.getmaskarray(rgb), np.nan, rgb)



            for i in range(3):
                rgb[..., i] = stretch01(rgb[..., i])
        plt.figure(figsize=(7,7))
        plt.imshow(rgb)
        plt.title("Sentinel-2 true color")
        plt.axis("off")
        plt.show()
        return

    bands = {}
    for p in paths:
        p = Path(p)
        if "_B04" in p.name: bands["R"] = p
        if "_B03" in p.name: bands["G"] = p
        if "_B02" in p.name: bands["B"] = p
    if not all(k in bands for k in ("R","G","B")):
        print("Missing RGB bands for visualization.")
        return

    with rasterio.open(bands["R"]) as r, \
         rasterio.open(bands["G"]) as g, \
         rasterio.open(bands["B"]) as b:

        R = r.read(1).astype("float32")
        G = g.read(1, out_shape=(r.height, r.width), resampling=Resampling.bilinear).astype("float32")
        B = b.read(1, out_shape=(r.height, r.width), resampling=Resampling.bilinear).astype("float32")

        rgb = np.dstack([stretch01(R), stretch01(G), stretch01(B)])

    plt.figure(figsize=(7,7))
    plt.imshow(rgb)
    plt.title("Sentinel-2 True Color (composed B04/B03/B02)")
    plt.axis("off")
    plt.show()

