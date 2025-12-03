from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import pyproj
from shapely.geometry import Point, box, Polygon
from shapely.geometry import mapping

import numpy as np
import xarray as xr
import earthaccess as ea

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

def find_emit_candidates(lon: float, lat: float):
    """Search EMIT around the location and within DATE_START..DATE_END."""
    login(persist=True)
    roi_bbox = point_buffer_bbox(lon, lat, SEARCH_BUFFER_M)
    items = search(
        point=(lon, lat),
        bbox=roi_bbox,
        start=DATE_START.isoformat(),
        end=DATE_END.isoformat(),
    )
    return list(items)


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



def _emit_window_from_roi(url: str, roi_geom_wgs84) -> Tuple[int, int, int, int]:
    """
    Given an EMIT L2A .nc URL and an ROI polygon in WGS84,
    return (row_min, row_max, col_min, col_max) in downtrack/crosstrack space.
    """
    # Get authenticated HTTPS session (no full download)
    fs = ea.get_fsspec_https_session()
    
    # Open ONLY the location group to get lat/lon (much smaller than reflectance)
    loc_ds = xr.open_dataset(
        fs.open(url),
        group="location",
        engine="h5netcdf",
        chunks={}
    )
    lat = loc_ds["latitude"]   # dims: (downtrack, crosstrack)
    lon = loc_ds["longitude"]

    minx, miny, maxx, maxy = roi_geom_wgs84.bounds

    in_roi = ((lon >= minx) & (lon <= maxx) &
              (lat >= miny) & (lat <= maxy))

    # Find rows/cols that intersect ROI at least once
    rows = np.where(in_roi.any(dim="crosstrack"))[0]
    cols = np.where(in_roi.any(dim="downtrack"))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("ROI does not intersect this EMIT granule")

    row_min, row_max = int(rows[0]), int(rows[-1])
    col_min, col_max = int(cols[0]), int(cols[-1])

    return row_min, row_max, col_min, col_max


def download_reflectance_roi(
    pick,
    roi_geom_wgs84,
    dest_dir: Path | str,
    assets: List[str] = ["_RFL_", "_MASK_"],
    pad_px: int = 0,
) -> List[Path]:
    """
    For a single EMIT granule (`pick`), stream the L2A RFL/MASK files
    and save ONLY the ROI subset (downtrack/crosstrack window) to disk.

    Returns list of paths to ROI-subsetted .nc files.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Find URLs for RFL/MASK .nc files
    links = _filter_rfl_links(pick.data_links(), desired_assets=assets)
    if not links:
        raise RuntimeError("No EMIT L2A Reflectance .nc links for the selected granule")

    # Use the reflectance file to compute window (location group is the same)
    rfl_url = [u for u in links if "_RFL_" in u][0]

    row_min, row_max, col_min, col_max = _emit_window_from_roi(rfl_url, roi_geom_wgs84)

    # Optional pixel padding around ROI
    row_min = max(row_min - pad_px, 0)
    col_min = max(col_min - pad_px, 0)
    row_max = row_max + pad_px
    col_max = col_max + pad_px

    fs = ea.get_fsspec_https_session()
    out_paths: List[Path] = []

    for url in links:
        # Stream file, but only slice the small window
        with fs.open(url) as fp:
            ds = xr.open_dataset(fp, engine="h5netcdf", chunks={})

            # EMIT uses downtrack / crosstrack
            ds_roi = ds.isel(
                downtrack=slice(row_min, row_max + 1),
                crosstrack=slice(col_min, col_max + 1),
            )

        out_path = dest / (Path(url).stem + "_ROI.nc")
        ds_roi.to_netcdf(out_path)
        out_paths.append(out_path)

    return out_paths


