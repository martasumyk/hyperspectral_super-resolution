from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Iterable, List
import pyproj
from shapely.geometry import Point, box, Polygon

import numpy as np
import xarray as xr
import earthaccess as ea
from datetime import datetime, date, timezone

EMIT_SHORT_NAME = "EMITL2ARFL"  # L2A Reflectance


def _emit_item_date(item) -> date:
    """Extract UTC date from an EMIT search item."""
    try:
        iso = item["umm"]["ProviderDates"][0]["Date"]
    except Exception:
        iso = item.get("datetime") or item.get("start_time")
    dt_utc = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt_utc.date()

def _emit_cloud_pct(item) -> float:
    try:
        return float(item["umm"].get("CloudCover"))
    except Exception:
        return float("inf")
    
def login(persist: bool = True) -> None:
    ea.login(persist=persist)

def find_emit_candidates(lon: float, lat: float, 
                         date_start = date(2022, 1, 1), 
                         date_end =   date(2022, 1, 1), 
                         seach_buffer_m = 20_000):
    """Search EMIT around the location and within DATE_START..DATE_END."""
    login(persist=True)
    roi_bbox = point_buffer_bbox(lon, lat, seach_buffer_m)
    items = search(
        point=(lon, lat),
        bbox=roi_bbox,
        start=date_start.isoformat(),
        end=seach_buffer_m.isoformat(),
    )
    return list(items)



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

def search(*, bbox=None, point=None, buffer_m=5000.0, start: dt.datetime, end: dt.datetime,
           short_name: str = EMIT_SHORT_NAME, cloud_cover = [0,100]) -> List:
    if bbox is None and point is None:
        raise ValueError("Provide either bbox or point")
    if bbox is None:
        poly = point_buffer_bbox(point[0], point[1], buffer_m)
        bbox = poly.bounds 
        
    result = ea.search_data(short_name=short_name, temporal=(start, end), bounding_box=bbox, cloud_cover=cloud_cover)
    if len(result) == 0:
        print("No granules found for the given search criteria.")
        return None
    print(f"Found {len(result)} granule(s).")
    return result


def choose_nearest(granules: Iterable, target_dt: dt.datetime):
    items = list(granules)
    if not items:
        return None
    def granule_date(g):
        return dt.datetime.fromisoformat(g["umm"]["ProviderDates"][0]["Date"])
    return min(items, key=lambda g: abs(granule_date(g) - target_dt))

def _filter_rfl_links(links: Iterable[str], desired_assets: List[str] = ['_RFL_', '_MASK_']) -> List[str]:
    filtered_asset_links = []
    for url in links:
        asset_name = url.split('/')[-1]
        if any(asset in asset_name for asset in desired_assets):
            filtered_asset_links.append(url)
    print(f"Filtered to {len(filtered_asset_links)} reflectance-related asset link(s).")
    return filtered_asset_links


def download_reflectance(pick , dest_dir: Path | str, assets: List[str] = ['_RFL_', '_MASK_']) -> List[Path]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    links = _filter_rfl_links(pick.data_links(), desired_assets=assets)
    if not links:
        raise RuntimeError("No EMIT L2A Reflectance .nc links for the selected granule")
    files = ea.download(links, str(dest))
    return [Path(p) for p in files]


def open_reflectance(nc_path: Path | str, engine: str | None = None) -> xr.Dataset:
    engine = engine or 'netcdf4'
    ds = xr.open_dataset(nc_path, engine=engine, decode_cf=True, mask_and_scale=True)

    var = None
    for c in ('reflectance','RFL','radiance'):
        if c in ds.data_vars:
            var = c; break
    if var is None:
        raise KeyError("Reflectance variable not found in dataset")
    fv = ds[var].attrs.get('_FillValue')
    if fv is not None:
        ds[var] = ds[var].where(ds[var] != fv)
    return ds

def attach_wavelengths(ds: xr.Dataset, data_var: str = 'reflectance') -> xr.Dataset:
    wl = None
    for c in ('wavelength','wavelengths','band_center','wl'):
        if c in ds.variables:
            wl = np.asarray(ds[c].values); break
    if wl is None:
        coord = ds[data_var].coords.get('wavelength')
        if coord is not None:
            wl = np.asarray(coord.values)
    if wl is None:
        return ds
    if wl.max() <= 10.0: 
        wl = wl * 1000.0

    dim = None
    for d in ('band','bands','wavelength'):
        if d in ds[data_var].dims:
            dim = d; break
    if dim is None:
        dim = ds[data_var].dims[-1]
    return ds.assign_coords({ 'wavelength_nm': (dim, wl) })

def _emit_item_date(item) -> date:
    """Extract UTC date from an EMIT search item."""
    try:
        iso = item["umm"]["ProviderDates"][0]["Date"]
    except Exception:
        iso = item.get("datetime") or item.get("start_time")
    dt_utc = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt_utc.date()

def _emit_cloud_pct(item) -> float:
    try:
        return float(item["umm"].get("CloudCover"))
    except Exception:
        return float("inf")


def choose_best_emit_per_date(items, max_cloud_pct = 50):
    """Group by date and keep least-cloudy EMIT per date; apply optional max cloud threshold."""
    by_date = {}
    for it in items:
        d = _emit_item_date(it)
        cur = by_date.get(d)
        if cur is None or _emit_cloud_pct(it) < _emit_cloud_pct(cur):
            by_date[d] = it
    if max_cloud_pct is not None:
        by_date = {d: it for d, it in by_date.items() if _emit_cloud_pct(it) <= max_cloud_pct}
    return {d.isoformat(): it for d, it in by_date.items()}


