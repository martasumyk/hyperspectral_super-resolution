from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import pyproj
from shapely.geometry import Point, box, Polygon

import numpy as np
import xarray as xr
import earthaccess as ea

EMIT_SHORT_NAME = "EMITL2ARFL"  # L2A Reflectance


def login(persist: bool = True) -> None:
    ea.login(persist=persist)


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
           short_name: str = EMIT_SHORT_NAME) -> List:
    if bbox is None and point is None:
        raise ValueError("Provide either bbox or point")
    if bbox is None:
        poly = point_buffer_bbox(point[0], point[1], buffer_m)
        bbox = poly.bounds 
        
    result = ea.search_data(short_name=short_name, temporal=(start, end), bounding_box=bbox)
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



