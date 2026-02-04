import datetime as dt
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import PosixPath


import netCDF4 as nc
import h5netcdf
import hytools as ht
import numpy as np
import rasterio
from pathlib import Path
from typing import Iterable, Optional, Union
import math

import json, shlex, subprocess
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


from pyproj import Transformer


NO_DATA_VALUE = -9999.0

def _read_obs_cube_and_names(obs_nc):
    """
    Returns (obs_arr, band_names):
      obs_arr: (H, W, 11) float32
      band_names: list[str] length 11
    Raises KeyError if cannot find expected OBS variables.
    """
    # Preferred canonical band-name order
    canonical = [
        ("path_length", ["path_length", "pathlength", "path_len", "plength"]),
        ("to-sensor azimuth", ["to_sensor_azimuth", "view_azimuth", "sensor_azimuth"]),
        ("to-sensor zenith", ["to_sensor_zenith", "view_zenith", "sensor_zenith"]),
        ("to-sun azimuth", ["to_sun_azimuth", "sun_azimuth"]),
        ("to-sun zenith", ["to_sun_zenith", "sun_zenith"]),
        ("phase", ["phase", "phase_angle"]),
        ("slope", ["slope"]),
        ("aspect", ["aspect"]),
        ("cosine i", ["cosine_i", "cos_i", "cosine_incidence"]),
        ("UTC time", ["utc_time", "utc_decimal_hours", "utc_hours"]),
        ("earth-sun distance", ["earth_sun_distance", "earth_sun_dist", "es_dist"]),
    ]

    # 1) Try a single 3D array in root or subgroup
    def find_3d_var(ds):
        # root variables
        for k, v in ds.variables.items():
            if hasattr(v, "ndim") and v.ndim == 3:
                return v
        # search first-level groups
        for gname, g in ds.groups.items():
            for k, v in g.variables.items():
                if hasattr(v, "ndim") and v.ndim == 3:
                    return v
        return None

    v3 = find_3d_var(obs_nc)
    if v3 is not None:
        arr = np.array(v3[:], dtype=np.float32)
        # Try to pull names from attributes if present; else fallback to canonical
        names = None
        for attr in ("band_names", "observation_bands", "bands", "names"):
            if hasattr(v3, attr):
                try:
                    bn = getattr(v3, attr)
                    names = [str(x) for x in (bn if isinstance(bn, (list, tuple)) else bn[:])]
                    break
                except Exception:
                    pass
        if names is None or len(names) != arr.shape[2]:
            # fall back
            names = [c[0] for c in canonical][:arr.shape[2]]
        return arr, names

    # 2) Assemble from per-band variables (root or first-level groups)
    def find_var_by_alias(ds, aliases):
        # search root
        for a in aliases:
            if a in ds.variables:
                return ds.variables[a]
        # search groups
        for g in ds.groups.values():
            for a in aliases:
                if a in g.variables:
                    return g.variables[a]
        return None

    bands = []
    names = []
    shape_hw = None
    for canonical_name, aliases in canonical:
        var = find_var_by_alias(obs_nc, aliases)
        if var is None:
            raise KeyError(f"OBS var not found for '{canonical_name}' (tried {aliases})")
        arr = np.array(var[:], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise KeyError(f"OBS var '{canonical_name}' has unexpected shape {arr.shape}")
        if shape_hw is None:
            shape_hw = arr.shape
        elif arr.shape != shape_hw:
            raise KeyError(f"OBS var '{canonical_name}' has shape {arr.shape}, expected {shape_hw}")
        bands.append(arr)
        names.append(canonical_name)

    obs_arr = np.stack(bands, axis=-1)  # (H, W, 11)
    return obs_arr.astype(np.float32), names



def _pretty_write_xml(root: ET.Element, out_path: str):
    """Write a pretty-printed XML to file."""
    # Minimal pretty print without external deps
    def _indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for e in elem:
                _indent(e, level + 1)
            if not e.tail or not e.tail.strip():
                e.tail = i
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    _indent(root)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def _write_xml_sidecar(
    out_bin_path: str,
    *,
    product: str,
    epsg_str: str,
    crs_wkt: str | None,
    pixel_size: tuple[float, float] | None,
    shape: tuple[int, int, int] | tuple[int, int],  # (lines, samples, bands?) or (lines, samples)
    start_time: dt.datetime,
    end_time: dt.datetime,
    bbox_lonlat: list[list[float]],
    wavelengths: list[float] | None = None,
    fwhm: list[float] | None = None,
    band_names: list[str] | None = None,
    description: str | None = None,
):
    """Create a compact XML metadata file next to the ENVI product."""
    lines = shape[0]
    samples = shape[1]
    bands = shape[2] if len(shape) == 3 else (len(band_names) if band_names else 1)

    root = ET.Element("EMITProduct")
    ET.SubElement(root, "ProductType").text = product
    if description:
        ET.SubElement(root, "Description").text = description

    # Time
    t = ET.SubElement(root, "AcquisitionTime")
    ET.SubElement(t, "StartUTC").text = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    ET.SubElement(t, "EndUTC").text = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Geometry
    g = ET.SubElement(root, "Geometry")
    ET.SubElement(g, "EPSG").text = epsg_str
    if crs_wkt:
        ET.SubElement(g, "CRS_WKT").text = crs_wkt
    if pixel_size:
        ps = ET.SubElement(g, "PixelSize")
        ET.SubElement(ps, "X").text = f"{float(pixel_size[0]):.10g}"
        ET.SubElement(ps, "Y").text = f"{float(pixel_size[1]):.10g}"

    bb = ET.SubElement(root, "BoundingBoxLonLat")
    # order: UL(corner_1), UR(corner_2), LR(corner_3), LL(corner_4)
    for i, (lon, lat) in enumerate(bbox_lonlat, start=1):
        c = ET.SubElement(bb, f"Corner{i}")
        ET.SubElement(c, "Lon").text = f"{float(lon):.10g}"
        ET.SubElement(c, "Lat").text = f"{float(lat):.10g}"

    # Raster shape
    s = ET.SubElement(root, "RasterShape")
    ET.SubElement(s, "Lines").text = str(int(lines))
    ET.SubElement(s, "Samples").text = str(int(samples))
    ET.SubElement(s, "Bands").text = str(int(bands))

    # Spectral info
    if wavelengths or fwhm or band_names:
        spec = ET.SubElement(root, "Spectral")
        if wavelengths:
            w = ET.SubElement(spec, "Wavelengths")
            w.set("units", "nanometers")
            for val in wavelengths:
                ET.SubElement(w, "Wavelength").text = f"{float(val):.10g}"
        if fwhm:
            f = ET.SubElement(spec, "FWHM")
            f.set("units", "nanometers")
            for val in fwhm:
                ET.SubElement(f, "Value").text = f"{float(val):.10g}"
        if band_names:
            bn = ET.SubElement(spec, "BandNames")
            for name in band_names:
                ET.SubElement(bn, "Band").text = str(name)

    out_xml = os.path.splitext(out_bin_path)[0] + ".xml"
    _pretty_write_xml(root, out_xml)

def get_attr(ds, name):
    if hasattr(ds, "ncattrs") and name in ds.ncattrs():
        v = ds.getncattr(name)
    elif hasattr(ds, "attrs") and name in ds.attrs:
        v = ds.attrs[name]
    else:
        raise KeyError(name)
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8")
    return v

def open_any_nc(path):
    path = str(Path(path).expanduser().resolve())
    try:
        import netCDF4 as nc
        return nc.Dataset(path, "r"), "netCDF4"
    except Exception:
        import h5netcdf
        return h5netcdf.File(path, "r"), "h5netcdf"
    


def run_cmd(cmd: list[str], check=True) -> dict:
    """Run a subprocess command and return a JSON-friendly record (with truncated stdout/stderr)."""
    res = subprocess.run(cmd, text=True)
    rec = {
        "cmd": cmd,
        "cmd_str": shlex.join(cmd),
        "returncode": res.returncode,
        "stdout_tail": (res.stdout[-5000:] if res.stdout else ""),
        "stderr_tail": (res.stderr[-5000:] if res.stderr else ""),
    }
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
    return rec

def export_uint16_deflate_geotiff(
    src_path: str,
    dst_tif: str,
    *,
    assign_epsg: str | None = None,
    scale_mode: str = "none",   # "none" | "emit_reflectance_0_1"
    nodata_uint16: int = 65535,
    zlevel: int = 1,
) -> dict:
    cmd = ["gdal_translate", "-of", "GTiff",
           "-ot", "UInt16",
           "-co", "COMPRESS=DEFLATE",
           "-co", f"ZLEVEL={int(zlevel)}",
           "-co", "PREDICTOR=2",
           "-co", "NUM_THREADS=ALL_CPUS",
           "-co", "BIGTIFF=IF_SAFER"]

    if scale_mode == "emit_reflectance_0_1":
        cmd += ["-scale", "0", "1", "0", "10000"]
        cmd += ["-a_nodata", str(int(nodata_uint16))]
        cmd += ["-mo", "scale_factor=0.0001"]
        cmd += ["-mo", "units=reflectance"]
        cmd += ["-mo", f"uint16_nodata={int(nodata_uint16)}"]

    if assign_epsg:
        cmd += ["-a_srs", assign_epsg]

    cmd += [src_path, dst_tif]
    return run_cmd(cmd, check=True)




def raster_meta(path: str) -> dict:
    """Reads CRS/bounds/shape/res from any GDAL-readable raster (GeoTIFF or ENVI)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}

    with rasterio.open(str(p)) as ds:
        b = ds.bounds
        crs = ds.crs
        out = {
            "path": str(p),
            "exists": True,
            "driver": ds.driver,
            "crs": crs.to_string() if crs else None,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "res": [float(ds.res[0]), float(ds.res[1])] if ds.res else None,
            "bounds": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
            "nodata": ds.nodata,
        }
        if crs:
            out["bounds_wgs84"] = list(transform_bounds(
                crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
            ))
        return out


def _bounds_to_out_crs(src_path: str, out_crs: str):
    with rasterio.open(src_path) as ds:
        b = ds.bounds
        src_crs = ds.crs
    if src_crs is None:
        raise ValueError(f"Source CRS is None for {src_path}")

    tf = Transformer.from_crs(src_crs, out_crs, always_xy=True)

    # transform all 4 corners (safe for rotation-ish cases)
    xs = [b.left, b.left, b.right, b.right]
    ys = [b.bottom, b.top, b.bottom, b.top]
    X, Y = tf.transform(xs, ys)

    return (min(X), min(Y), max(X), max(Y))  # left, bottom, right, top

def _intersect(a, b):
    # a,b: (l,b,r,t)
    l = max(a[0], b[0]); bb = max(a[1], b[1])
    r = min(a[2], b[2]); t = min(a[3], b[3])
    if (r <= l) or (t <= bb):
        return None
    return (l, bb, r, t)

def _snap_te_to_s2_grid(te, s2_bounds, xres=60.0, yres=60.0):
    # snap to a grid anchored at s2_left/top
    s2_left = float(s2_bounds.left)
    s2_top  = float(s2_bounds.top)

    l, b, r, t = te

    # snap left/right to columns
    col0 = math.floor((l - s2_left) / xres)
    col1 = math.ceil((r - s2_left) / xres)
    l2 = s2_left + col0 * xres
    r2 = s2_left + col1 * xres

    # snap top/bottom to rows (note y axis downward in row indexing)
    row0 = math.floor((s2_top - t) / yres)
    row1 = math.ceil((s2_top - b) / yres)
    t2 = s2_top - row0 * yres
    b2 = s2_top - row1 * yres

    return (l2, b2, r2, t2)

def _compute_te(src_path, s2_te_exact, s2_origin_xy, out_crs, xres=60.0, yres=60.0):
    # src bounds in out_crs (UTM)
    src_te = _bounds_to_out_crs(src_path, out_crs)  # (left,bottom,right,top) in out_crs

    # exact S2 extent in out_crs (already in out_crs because S2 is in out_crs)
    s2_te = tuple(map(float, s2_te_exact))  # (left,bottom,right,top)

    inter = _intersect(src_te, s2_te)
    if inter is None:
        raise ValueError("No overlap between EMIT source bounds and S2 extent in out_crs.")

    inter_left, inter_bottom, inter_right, inter_top = map(float, inter)

    x0, y0 = map(float, s2_origin_xy)
    step_x, step_y = float(xres), float(yres)

    eps = 1e-9

    left  = x0 + math.ceil(((inter_left  - x0) / step_x) - eps) * step_x
    right = x0 + math.floor(((inter_right - x0) / step_x) + eps) * step_x

    top    = y0 - math.ceil(((y0 - inter_top) / step_y) - eps) * step_y
    bottom = y0 - math.floor(((y0 - inter_bottom) / step_y) + eps) * step_y


    if right <= left or top <= bottom:
        raise ValueError(f"Snapped TE is invalid: {(left,bottom,right,top)}")

    return (left, bottom, right, top)



import math
import shutil
import numpy as np
import rasterio
from rasterio.enums import Resampling

def _which_gdal_edit() -> str | None:
    for c in ("gdal_edit", "gdal_edit.py"):
        if shutil.which(c):
            return c
    return None


def export_loc_uint16_deflate_geotiff(
    src_path: str,
    dst_tif: str,
    *,
    lon_range=(-180.0, 180.0),
    lat_range=(-90.0, 90.0),
    elev_range=(-1000.0, 12000.0),
    nodata_uint16: int = 0,
) -> dict:
    """
    Export EMIT LOC (lon,lat,elev) to UInt16 GeoTIFF with meaningful scaling.
    Writes per-band scale/offset metadata when gdal_edit is available.
    """
    # Per-band scaling using -scale_X (aka -scale_bn) :contentReference[oaicite:3]{index=3}
    cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-ot", "UInt16",
        "-a_nodata", str(nodata_uint16),
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        # clamp + scale each band
        "-scale_1", str(lon_range[0]), str(lon_range[1]), "0", "65535",
        "-exponent_1", "1",
        "-scale_2", str(lat_range[0]), str(lat_range[1]), "0", "65535",
        "-exponent_2", "1",
        "-scale_3", str(elev_range[0]), str(elev_range[1]), "0", "65535",
        "-exponent_3", "1",
        src_path, dst_tif,
    ]
    rec = run_cmd(cmd, check=True)

    # Store decode metadata: true = raw*scale + offset :contentReference[oaicite:4]{index=4}
    scales = [
        (lon_range[1] - lon_range[0]) / 65535.0,
        (lat_range[1] - lat_range[0]) / 65535.0,
        (elev_range[1] - elev_range[0]) / 65535.0,
    ]
    offsets = [lon_range[0], lat_range[0], elev_range[0]]

    gdal_edit = _which_gdal_edit()
    if gdal_edit:
        run_cmd(
            [gdal_edit, "-scale", *[f"{s:.16g}" for s in scales],
                       "-offset", *[f"{o:.16g}" for o in offsets],
             dst_tif],
            check=True,
        )

    rec["uint16_decode"] = {
        "scales": scales,
        "offsets": offsets,
        "ranges": [list(lon_range), list(lat_range), list(elev_range)],
        "nodata_uint16": nodata_uint16,
        "note": "Recover: true = raw*scale + offset",
    }
    return rec


def _sample_band_minmax(
    src_path: str,
    band_index_1based: int,
    nodata: float,
    *,
    stride: int = 64,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[float, float]:
    """
    Fast-ish robust min/max from a decimated read (percentiles).
    """
    with rasterio.open(src_path) as ds:
        out_h = max(1, ds.height // stride)
        out_w = max(1, ds.width // stride)
        arr = ds.read(
            band_index_1based,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        ).astype(np.float32)

    m = np.isfinite(arr) & (arr != float(nodata))
    if not np.any(m):
        # fallback: arbitrary small range to avoid gdal_translate failure
        return 0.0, 1.0

    vals = arr[m]
    lo, hi = np.percentile(vals, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if lo == hi:
            hi = lo + 1.0
    return float(lo), float(hi)


def export_obs_uint16_deflate_geotiff(
    src_path: str,
    dst_tif: str,
    *,
    nodata_float: float,
    nodata_uint16: int = 0,
    stride: int = 64,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> dict:
    """
    Export EMIT OBS cube to UInt16 GeoTIFF with per-band robust scaling.
    Writes per-band scale/offset metadata when gdal_edit is available.
    """
    with rasterio.open(src_path) as ds:
        nb = ds.count

    # build per-band scale args (-scale_X ... -exponent_X 1) :contentReference[oaicite:5]{index=5}
    cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-ot", "UInt16",
        "-a_nodata", str(nodata_uint16),
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
    ]

    src_mins, src_maxs = [], []
    for b in range(1, nb + 1):
        lo, hi = _sample_band_minmax(
            src_path, b, nodata_float,
            stride=stride, p_low=p_low, p_high=p_high
        )
        src_mins.append(lo)
        src_maxs.append(hi)
        cmd += [f"-scale_{b}", str(lo), str(hi), "0", "65535", f"-exponent_{b}", "1"]

    cmd += [src_path, dst_tif]
    rec = run_cmd(cmd, check=True)

    # Store decode metadata per band: true = raw*scale + offset :contentReference[oaicite:6]{index=6}
    scales = [(mx - mn) / 65535.0 for mn, mx in zip(src_mins, src_maxs)]
    offsets = list(src_mins)

    gdal_edit = _which_gdal_edit()
    if gdal_edit:
        run_cmd(
            [gdal_edit, "-scale", *[f"{s:.16g}" for s in scales],
                       "-offset", *[f"{o:.16g}" for o in offsets],
             dst_tif],
            check=True,
        )

    rec["uint16_decode"] = {
        "scales": scales,
        "offsets": offsets,
        "src_mins": src_mins,
        "src_maxs": src_maxs,
        "nodata_uint16": nodata_uint16,
        "note": "Recover: true = raw*scale + offset",
        "percentiles": [p_low, p_high],
        "stride": stride,
    }
    return rec



def nc_to_envi(
    img_file: str,
    out_dir: str,
    temp_dir: str,
    obs_file: str | None = None,
    export_loc: bool = False,
    s2_tif_path: str | None = None,
    match_res: bool = False,
    write_xml: bool = True,
    *,
    overwrite: bool = False,
    tag: str | None = None,
    return_info: bool = False,
    save_info_path: str | Path | None = None,
    save_geotiffs: bool = True,
) -> Path | tuple[Path, dict]:
    """
    Export EMIT L1B_RDN or L2A_RFL to ENVI and write XML sidecars.

    NEW:
      - overwrite: if False, skips any output that already exists (.bin + .hdr)
      - tag: optional string appended to filenames to avoid collisions across tiles
    Returns:
      Path to the main projected ENVI cube (.bin)
    """
    def _finalize_and_return(out_path: Path):
        if save_info_path is not None:
            p = Path(save_info_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(info, indent=2, default=str))
            info["saved_info_path"] = str(p)

        return (out_path, info) if return_info else out_path

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    print(img_file)

    img_file = str(Path(img_file).expanduser().resolve())
    out_dir_p = Path(out_dir)
    temp_dir_p = Path(temp_dir)

    try:
        import netCDF4 as nc
        img_nc = nc.Dataset(img_file, "r")
        backend = "netCDF4"
    except OSError:
        import h5netcdf
        img_nc = h5netcdf.File(img_file, "r")
        backend = "h5netcdf"

    print("Opened with:", backend)

    img_path = Path(img_file)
    if tag is None:
        stem = img_path.stem
        tag = stem.replace("EMIT_", "")

    try:
        # Disable auto mask/scale where possible
        for vname in ("radiance", "reflectance"):
            if vname in getattr(img_nc, "variables", {}):
                var = img_nc.variables[vname]
                if hasattr(var, "set_auto_maskandscale"):
                    print(f"Disabling auto mask/scale for variable '{vname}'")
                    var.set_auto_maskandscale(False)
        if hasattr(img_nc, "set_auto_mask"):
            img_nc.set_auto_mask(False)

        # Product & data
        if "radiance" in img_nc.variables.keys():
            data = img_nc.variables["radiance"]
            product = "L1B_RDN"
            description = "Radiance micro-watts/cm^2/nm/sr"
        elif "reflectance" in img_nc.variables.keys():
            data = img_nc.variables["reflectance"]
            product = "L2A_RFL"
            description = "Reflectance (unitless)"
        else:
            raise ValueError("Unrecognized input image dataset (expected 'radiance' or 'reflectance').")
        
        dims = getattr(data, "dimensions", None)

        transpose_raw_yx = False
        if dims is not None and len(dims) >= 2:
            d0, d1 = str(dims[0]).lower(), str(dims[1]).lower()

            if ("crosstrack" in d0 and "downtrack" in d1) or ("x" == d0 and "y" == d1):
                transpose_raw_yx = True

            if ("downtrack" not in d0 and "crosstrack" not in d0 and "along" not in d0 and "across" not in d0) or \
            ("downtrack" not in d1 and "crosstrack" not in d1 and "along" not in d1 and "across" not in d1):
                print(f"[WARN] Unclear EMIT raw dims order: {dims}. Assuming data[y,x,...].")
        else:
            print("[WARN] data.dimensions not available; assuming data[y,x,...].")

        print(f"[DEBUG] data.dimensions={dims} -> transpose_raw_yx={transpose_raw_yx}")


        sbp = img_nc.groups["sensor_band_parameters"]
        fwhm  = np.asarray(sbp.variables["fwhm"][:])
        waves = np.asarray(sbp.variables["wavelengths"][:])

        # Geotransform/GLT
        gt = np.array(get_attr(img_nc, "geotransform"))

        gt = np.asarray(gt, dtype=float)
        if len(gt) != 6:
            raise ValueError(f"Expected geotransform of length 6, got {len(gt)}: {gt}")

        if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
            raise ValueError(
                "Rotated/sheared geotransform detected (gt[2] or gt[4] non-zero). "
                "ENVI 'map info' cannot represent rotation. "
                f"gt={gt.tolist()}"
            )

        loc = img_nc.groups["location"]
        glt_x = np.asarray(loc.variables["glt_x"][:])
        glt_y = np.asarray(loc.variables["glt_y"][:])
        glt = np.zeros(list(glt_x.shape) + [2], dtype=np.int32)
        glt[..., 0] = np.nan_to_num(glt_x, nan=0).astype(np.int32)
        glt[..., 1] = np.nan_to_num(glt_y, nan=0).astype(np.int32)



        valid_glt = np.all(glt != 0, axis=-1)

        glt0 = glt.copy()
        glt0[valid_glt] -= 1

        raw_h, raw_w = (int(data.shape[1]), int(data.shape[0])) if transpose_raw_yx else (int(data.shape[0]), int(data.shape[1]))

        in_bounds = (
            (glt0[..., 1] >= 0) & (glt0[..., 1] < raw_h) &
            (glt0[..., 0] >= 0) & (glt0[..., 0] < raw_w)
        )

        valid_glt2 = valid_glt & in_bounds

        dropped = int(np.count_nonzero(valid_glt) - np.count_nonzero(valid_glt2))
        if dropped > 0:
            pct = 100.0 * dropped / max(1, int(np.count_nonzero(valid_glt)))
            print(f"[WARN] Dropping {dropped} GLT pixels ({pct:.4f}%) that are out-of-bounds for raw grid {raw_h}x{raw_w}")

            # optional: record diagnostics (only if info already exists at this point)
            try:
                info.setdefault("glt_diag", {})
                info["glt_diag"].update({
                    "raw_shape_yx": [raw_h, raw_w],
                    "valid_glt_count": int(np.count_nonzero(valid_glt)),
                    "valid_glt_inbounds_count": int(np.count_nonzero(valid_glt2)),
                    "valid_glt_dropped_oob": dropped,
                })
            except Exception:
                pass
        # ---------------------------------------------------------------


        x0 = float(gt[0] + 0.5 * gt[1])
        y0 = float(gt[3] + 0.5 * gt[5]) 

        H, W = glt.shape[:2]

        x_ul, x_res, x_rot, y_ul, y_rot, y_res = map(float, gt)

        def _xy(col: float, row: float) -> tuple[float, float]:
            X = x_ul + col * x_res + row * x_rot
            Y = y_ul + col * y_rot + row * y_res
            return (X, Y)

        c1 = _xy(0, 0)     # upper-left edge
        c2 = _xy(W, 0)     # upper-right edge
        c3 = _xy(W, H)     # lower-right edge
        c4 = _xy(0, H)     # lower-left edge

        corner_1 = [c1[0], c1[1]]
        corner_2 = [c2[0], c2[1]]
        corner_3 = [c3[0], c3[1]]
        corner_4 = [c4[0], c4[1]]

        map_info = [
            "Geographic Lat/Lon",
            1, 1,
            x0, y0,
            float(gt[1]), float(-gt[5]),
            "WGS-84",
            "units=degrees",
        ]


        t0 = get_attr(img_nc, "time_coverage_start")
        t1 = get_attr(img_nc, "time_coverage_end")
        start_time = dt.datetime.strptime(str(t0), "%Y-%m-%dT%H:%M:%S+0000")
        end_time   = dt.datetime.strptime(str(t1), "%Y-%m-%dT%H:%M:%S+0000")

        # Choose target CRS & resolution
        out_crs = None
        out_epsg_int = None
        out_ps = (60.0, 60.0)
        crs_wkt = None

        if s2_tif_path:
            with rasterio.open(s2_tif_path) as src:
                s2_crs = src.crs
                out_crs = s2_crs.to_string()
                out_epsg_int = s2_crs.to_epsg()
                crs_wkt = s2_crs.to_wkt()

                s2_bounds = src.bounds

                s2_transform = src.transform
                s2_width = int(src.width)
                s2_height = int(src.height)

                s2_dx = abs(float(s2_transform.a))
                s2_dy = abs(float(s2_transform.e))

                s2_x0 = float(s2_transform.c) 
                s2_y0 = float(s2_transform.f) 

                s2_left   = s2_x0
                s2_top    = s2_y0
                s2_right  = s2_left + s2_width * s2_dx
                s2_bottom = s2_top  - s2_height * s2_dy

                emit_step = 60.0
                s2_te_exact = (s2_left, s2_bottom, s2_right, s2_top)

                if abs((emit_step / s2_dx) - round(emit_step / s2_dx)) > 1e-9:
                    raise ValueError(f"emit_step={emit_step} must be integer multiple of S2 dx={s2_dx}")
                if abs((emit_step / s2_dy) - round(emit_step / s2_dy)) > 1e-9:
                    raise ValueError(f"emit_step={emit_step} must be integer multiple of S2 dy={s2_dy}")

        if not out_crs:
            raise ValueError("out_crs is None. Provide s2_tif_path or enable epsg/UTM fallback.")

        xres, yres = (60.0, 60.0)

        ts = start_time.strftime("%Y%m%dT%H%M%S")
        suffix = f"_{tag}" if tag else ""

        data_utm = out_dir_p / f"{tag}.bin"
        data_hdr = data_utm.with_suffix(".hdr")

        loc_utm  = out_dir_p / f"{tag}_LOC.bin"
        loc_hdr  = loc_utm.with_suffix(".hdr")

        obs_utm  = out_dir_p / f"{tag}_OBS.bin"
        obs_hdr  = obs_utm.with_suffix(".hdr")

        need_data = overwrite or not (data_utm.exists() and data_hdr.exists())
        need_loc  = export_loc and (overwrite or not (loc_utm.exists() and loc_hdr.exists()))
        need_obs  = (obs_file is not None) and (overwrite or not (obs_utm.exists() and obs_hdr.exists()))

        info = {
            "img_file": img_file,
            "obs_file": str(obs_file) if obs_file else None,
            "tag": tag,
            "backend": backend,
            "product": product,
            "description": description,
            "time": {
                "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "out": {
                "out_crs": out_crs,
                "out_epsg": out_epsg_int,
                "match_res": bool(match_res),
                "pixel_size_m": list(out_ps) if match_res else [60.0, 60.0],
                "nodata": NO_DATA_VALUE,
                "resampling": "cubic",
            },
            "s2_align": {
                "s2_tif_path": str(s2_tif_path) if s2_tif_path else None,
                "s2_bounds": [float(s2_bounds.left), float(s2_bounds.bottom), float(s2_bounds.right), float(s2_bounds.top)] if s2_tif_path else None,
            },
            "commands": {},
            "outputs": {},
            "rasters": {},
        }

        info["s2_align"].update({
            "s2_transform": [float(s2_transform.a), float(s2_transform.b), float(s2_transform.c),
                            float(s2_transform.d), float(s2_transform.e), float(s2_transform.f)],
            "s2_grid_extent": [float(s2_left), float(s2_bottom), float(s2_right), float(s2_top)],
            "s2_origin": [float(s2_x0), float(s2_y0)],
            "emit_target_ps": [60.0, 60.0],
            "emit_anchor_mode": "s2_origin",
        })

        if not (need_data or need_loc or need_obs):
            print(f"All requested outputs already exist; skipping. Returning: {data_utm}")

            # record what already exists (requested outputs)
            info["outputs"]["data_envi_bin"] = str(data_utm)
            info["outputs"]["data_envi_hdr"] = str(data_hdr)

            if export_loc:
                info["outputs"]["loc_envi_bin"] = str(loc_utm)
                info["outputs"]["loc_envi_hdr"] = str(loc_hdr)

            if obs_file is not None:
                info["outputs"]["obs_envi_bin"] = str(obs_utm)
                info["outputs"]["obs_envi_hdr"] = str(obs_hdr)

            return _finalize_and_return(data_utm)
        
        

        def _run_gdalwarp(
            src_path: str,
            dst_path: str,
            *,
            xres: float = 60.0,
            yres: float = 60.0,
            src_srs: str | None = None,
        ) -> dict:

            if s2_bounds is None:
                raise ValueError("s2_bounds is None. Need s2_tif_path to align output grid.")

            xres = float(xres)
            yres = float(yres)

            x0 = s2_x0
            y0 = s2_y0
            step_x = xres
            step_y = yres

            left, bottom, right, top = _compute_te(
                src_path,
                s2_te_exact=s2_te_exact,
                s2_origin_xy=(s2_x0, s2_y0),
                out_crs=out_crs,
                xres=xres,
                yres=yres,
            )
            cols = int(round((right - left) / step_x))
            rows = int(round((top - bottom) / step_y))

            if cols <= 0 or rows <= 0:
                raise ValueError(f"Bad target shape cols={cols}, rows={rows} from snapped extent.")

            cmd = ["gdalwarp"]
            if overwrite:
                cmd.append("-overwrite")
            cmd += ["--config", "GDAL_CACHEMAX", "2048"]

            if src_srs is not None:
                cmd += ["-s_srs", src_srs]

            cmd += ["-t_srs", out_crs]

            cmd += ["-te", str(left), str(bottom), str(right), str(top)]
            cmd += ["-ts", str(cols), str(rows)]
            cmd += ["-srcnodata", str(NO_DATA_VALUE), "-dstnodata", str(NO_DATA_VALUE)]
            cmd += ["-multi", "-wo", "NUM_THREADS=ALL_CPUS", "-wm", "16384"]
            cmd += ["-r", "cubic", "-of", "ENVI", src_path, dst_path]

            rec = run_cmd(cmd, check=True)
            rec["aligned_extent"] = {
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
                "cols": cols,
                "rows": rows,
                "xres": step_x,
                "yres": step_y,
                "anchor_x0": x0,
                "anchor_y0": y0,
            }

            return rec



        # ----------------------------------------------------------------

        # Precompute gather indices once (used by all exports)
        gy = glt0[..., 1][valid_glt2]
        gx = glt0[..., 0][valid_glt2]

        # ===== DATA export (only if needed) =====
        if need_data:
            print(f"Exporting EMIT {product} dataset")

            data_header = ht.io.envi_header_dict()
            data_header["lines"] = glt.shape[0]
            data_header["samples"] = glt.shape[1]
            data_header["bands"] = data.shape[2]
            data_header["byte order"] = 0
            data_header["data ignore value"] = NO_DATA_VALUE
            data_header["data type"] = 4
            data_header["interleave"] = "bil"
            data_header["map info"] = map_info

            data_gcs = str(temp_dir_p / f"data_gcs_{tag}")
            writer = ht.io.WriteENVI(data_gcs, data_header)
            

            B = data.shape[2]
            chunk = 32  # tune 16/32/64 depending on RAM

            for b0 in range(0, B, chunk):
                b1 = min(b0 + chunk, B)

                # raw block: (raw_y, raw_x, nb)
                raw_blk = np.asarray(data[:, :, b0:b1], dtype=np.float32)
                if transpose_raw_yx:
                    raw_blk = np.transpose(raw_blk, (1, 0, 2))


                # ortho block: (H, W, nb)
                out_blk = np.full((H, W, b1 - b0), NO_DATA_VALUE, dtype=np.float32)
                out_blk[valid_glt2, :] = raw_blk[gy, gx, :]


                # write each band (because WriteENVI is band-oriented)
                for i, band_num in enumerate(range(b0, b1)):
                    writer.write_band(out_blk[:, :, i], band_num)

            # ---- Diagnostic quicklook (single band) for alignment checks ----
            # Pick a stable band index (e.g., middle band) so it always exists
            diag_band = int(data.shape[2] // 2)  # 0-based
            diag_src = f"{data_gcs}"

            diag_dir = out_dir_p / "diag"
            diag_dir.mkdir(parents=True, exist_ok=True)

            diag_ortho_tif = diag_dir / f"{tag}_DATA_diag_band{diag_band:03d}_ortho_wgs84.tif"
            diag_utm_tif   = diag_dir / f"{tag}_DATA_diag_band{diag_band:03d}_warp_utm.tif"

            # Export only one band from the ortho ENVI to GeoTIFF (still uint16 reflectance scaling OK for DATA)
            if save_geotiffs:
                cmd = [
                    "gdal_translate",
                    "-b", str(diag_band + 1),  # GDAL is 1-based bands
                    "-a_srs", "EPSG:4326",
                    "-ot", "UInt16",
                    "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2", "-co", "TILED=YES",
                    "-scale", "0", "1", "0", "10000",  # reflectance scaling (keep your convention)
                    diag_src, str(diag_ortho_tif),
                ]
                info["commands"]["gdal_translate_data_diag_ortho"] = run_cmd(cmd, check=True)
                info["outputs"]["data_diag_ortho_tif"] = str(diag_ortho_tif)
            # ---------------------------------------------------------------



            geotiff_dir = out_dir_p / "geotiff"
            geotiff_dir.mkdir(parents=True, exist_ok=True)

            ortho_tif = geotiff_dir / f"{tag}_DATA_ortho_wgs84.tif"
            utm_tif   = geotiff_dir / f"{tag}_DATA_warp_utm.tif"

            # Save orthorectified GeoTIFF (this is your GLT-projected cube in Geographic Lat/Lon)
            if save_geotiffs:
                info["commands"]["gdal_translate_data_ortho"] = export_uint16_deflate_geotiff(
                data_gcs, str(ortho_tif),
                assign_epsg="EPSG:4326",
                scale_mode="emit_reflectance_0_1"
)
                info["outputs"]["data_ortho_tif"] = str(ortho_tif)
                info["rasters"]["data_ortho_tif"] = raster_meta(str(ortho_tif))


            print(f"Projecting data to {out_crs} {'(match S2 res)' if match_res else '(60 m)'} -> {data_utm}")
            warp_rec = _run_gdalwarp(data_gcs, str(data_utm), xres=xres, yres=yres)

            info["commands"]["gdalwarp_data"] = warp_rec
            info["outputs"]["data_envi_bin"] = str(data_utm)
            info["outputs"]["data_envi_hdr"] = str(data_hdr)
            info["rasters"]["data_envi"] = raster_meta(str(data_utm))

            if save_geotiffs:
                info["commands"]["gdal_translate_data_utm"] = export_uint16_deflate_geotiff(
                str(data_utm), str(utm_tif),
                scale_mode="emit_reflectance_0_1"
)
                info["outputs"]["data_utm_tif"] = str(utm_tif)
                info["rasters"]["data_utm_tif"] = raster_meta(str(utm_tif))
                cmd = [
                    "gdal_translate",
                    "-b", str(diag_band + 1),
                    "-ot", "UInt16",
                    "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2", "-co", "TILED=YES",
                    "-scale", "0", "1", "0", "10000",
                    str(data_utm), str(diag_utm_tif),
                ]
                info["commands"]["gdal_translate_data_diag_utm"] = run_cmd(cmd, check=True)
                info["outputs"]["data_diag_utm_tif"] = str(diag_utm_tif)


            # Fix header
            data_header2 = ht.io.envi.parse_envi_header(str(data_hdr))
            data_header2["description"] = description
            data_header2["band names"] = []
            data_header2["fwhm"] = fwhm.tolist()
            data_header2["wavelength"] = waves.tolist()
            data_header2["wavelength units"] = "nanometers"
            data_header2["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            data_header2["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            data_header2["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
            if crs_wkt:
                data_header2["coordinate system string"] = [crs_wkt]

            data_header2["sensor type"] = "EMIT"
            if "map info" in data_header2 and isinstance(data_header2["map info"], list):
                if any(isinstance(v, str) and v.lower().startswith("units=") for v in data_header2["map info"]):
                    data_header2["map info"] = [
                        ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
                        for v in data_header2["map info"]
                    ]
                else:
                    data_header2["map info"].append("units=Meters")
            ht.io.envi.write_envi_header(str(data_hdr), data_header2)

            if write_xml:
                lines = int(data_header2.get("lines", 0))
                samples = int(data_header2.get("samples", 0))
                _write_xml_sidecar(
                    str(data_utm),
                    product=product,
                    epsg_str=f"EPSG:{out_epsg_int}" if out_epsg_int else out_crs,
                    crs_wkt=crs_wkt,
                    pixel_size=out_ps if match_res else (60.0, 60.0),
                    shape=(lines, samples, data.shape[2]),
                    start_time=start_time,
                    end_time=end_time,
                    bbox_lonlat=[corner_1, corner_2, corner_3, corner_4],
                    wavelengths=waves.tolist() if hasattr(waves, "tolist") else list(waves),
                    fwhm=fwhm.tolist() if hasattr(fwhm, "tolist") else list(fwhm),
                    band_names=None,
                    description=description,
                )
        else:
            print(f"DATA exists; skipping regeneration: {data_utm}")

        # ===== LOC export (only if needed) =====
        if need_loc:
            print("Exporting EMIT location dataset")

            loc_header0 = ht.io.envi_header_dict()
            loc_header0["lines"] = glt.shape[0]
            loc_header0["samples"] = glt.shape[1]
            loc_header0["bands"] = 3
            loc_header0["byte order"] = 0
            loc_header0["data ignore value"] = NO_DATA_VALUE
            loc_header0["data type"] = 4
            loc_header0["interleave"] = "bil"
            loc_header0["map info"] = map_info

            loc_gcs  = str(temp_dir_p / f"loc_gcs_{tag}")
            writer = ht.io.WriteENVI(loc_gcs, loc_header0)

            loc_vars = img_nc.groups["location"].variables
            loc_band = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)
            for band_num, name in enumerate(("lon", "lat", "elev")):
                band = np.asarray(loc_vars[name][:], dtype=np.float32)
                if transpose_raw_yx:
                    band = band.T
                loc_band[valid_glt2] = band[gy, gx]
                writer.write_band(loc_band, band_num)
                loc_band.fill(NO_DATA_VALUE)

            print(f"Projecting location datacube to {out_crs} {'(match S2 res)' if match_res else '(60 m)'} -> {loc_utm}")
            
            warp_rec = _run_gdalwarp(loc_gcs, str(loc_utm), xres=xres, yres=yres)

            info["commands"]["gdalwarp_loc"] = warp_rec
            info["outputs"]["loc_envi_bin"] = str(loc_utm)
            info["outputs"]["loc_envi_hdr"] = str(loc_hdr)
            info["rasters"]["loc_envi"] = raster_meta(str(loc_utm))

            if save_geotiffs:
                loc_tif = (out_dir_p / "geotiff" / f"{tag}_LOC_warp_utm.tif")
                info["commands"]["gdal_translate_loc_utm"] = export_loc_uint16_deflate_geotiff(
                    str(loc_utm), str(loc_tif),
                    # optional: tune elev_range if you want tighter precision
                    elev_range=(-1000.0, 12000.0),
                )

                info["outputs"]["loc_utm_tif"] = str(loc_tif)
                info["rasters"]["loc_utm_tif"] = raster_meta(str(loc_tif))


            loc_header2 = ht.io.envi.parse_envi_header(str(loc_hdr))
            loc_header2["band names"] = ["longitude", "latitude", "elevation"]
            loc_header2["description"] = "Location datacube"
            loc_header2["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            loc_header2["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            loc_header2["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
            loc_header2["sensor type"] = "EMIT"
            if "map info" in loc_header2 and isinstance(loc_header2["map info"], list):
                if any(isinstance(v, str) and v.lower().startswith("units=") for v in loc_header2["map info"]):
                    loc_header2["map info"] = [
                        ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
                        for v in loc_header2["map info"]
                    ]
                else:
                    loc_header2["map info"].append("units=Meters")
            ht.io.envi.write_envi_header(str(loc_hdr), loc_header2)

            if write_xml:
                lines = int(loc_header2.get("lines", 0))
                samples = int(loc_header2.get("samples", 0))
                _write_xml_sidecar(
                    str(loc_utm),
                    product=f"{product}_LOC",
                    epsg_str=f"EPSG:{out_epsg_int}" if out_epsg_int else out_crs,
                    crs_wkt=crs_wkt,
                    pixel_size=out_ps if match_res else (60.0, 60.0),
                    shape=(lines, samples, 3),
                    start_time=start_time,
                    end_time=end_time,
                    bbox_lonlat=[corner_1, corner_2, corner_3, corner_4],
                    band_names=["longitude", "latitude", "elevation"],
                    description="Location datacube",
                )
        elif export_loc:
            print(f"LOC exists; skipping regeneration: {loc_utm}")

        # ===== OBS export (only if needed) =====
        if need_obs:
            print("Exporting EMIT observation dataset")
            obs_nc, obs_backend = open_any_nc(obs_file)

            try:
                obs_cube, obs_bandnames = _read_obs_cube_and_names(obs_nc)
            except Exception as e:
                print(f"[WARN] Could not parse OBS file '{obs_file}': {e}")
                print("Skipping OBS export.")
                obs_nc.close()
                obs_nc = None

            if obs_nc is not None:
                obs_header0 = ht.io.envi_header_dict()
                obs_header0["lines"] = glt.shape[0]
                obs_header0["samples"] = glt.shape[1]
                obs_header0["bands"] = obs_cube.shape[2]
                obs_header0["byte order"] = 0
                obs_header0["data ignore value"] = NO_DATA_VALUE
                obs_header0["data type"] = 4
                obs_header0["interleave"] = "bil"
                obs_header0["map info"] = map_info

                obs_gcs  = str(temp_dir_p / f"obs_gcs_{tag}")
                writer = ht.io.WriteENVI(obs_gcs, obs_header0)

                obs_band = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)
                for band_num in range(obs_cube.shape[2]):
                    band = obs_cube[..., band_num]
                    if transpose_raw_yx:
                        band = band.T
                    obs_band[valid_glt2] = band[gy, gx]
                    writer.write_band(obs_band, band_num)
                    obs_band.fill(NO_DATA_VALUE)

                print(f"Projecting observation datacube to {out_crs} {'(match S2 res)' if match_res else '(60 m)'} -> {obs_utm}")
                warp_rec = _run_gdalwarp(obs_gcs, str(obs_utm), xres=xres, yres=yres)

                info["commands"]["gdalwarp_obs"] = warp_rec
                info["outputs"]["obs_envi_bin"] = str(obs_utm)
                info["outputs"]["obs_envi_hdr"] = str(obs_hdr)
                info["rasters"]["obs_envi"] = raster_meta(str(obs_utm))

                if save_geotiffs:
                    obs_tif = (out_dir_p / "geotiff" / f"{tag}_OBS_warp_utm.tif")
                    info["commands"]["gdal_translate_obs_utm"] = export_obs_uint16_deflate_geotiff(
                        str(obs_utm), str(obs_tif),
                        nodata_float=NO_DATA_VALUE,
                        stride=64,     # speed/quality tradeoff
                        p_low=1.0,
                        p_high=99.0,
                    )

                    info["outputs"]["obs_utm_tif"] = str(obs_tif)
                    info["rasters"]["obs_utm_tif"] = raster_meta(str(obs_tif))


                obs_header2 = ht.io.envi.parse_envi_header(str(obs_hdr))
                obs_header2["band names"] = obs_bandnames
                obs_header2["description"] = "Observation datacube"
                obs_header2["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                obs_header2["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                obs_header2["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
                obs_header2["sensor type"] = "EMIT"
                if "map info" in obs_header2 and isinstance(obs_header2["map info"], list):
                    if any(isinstance(v, str) and v.lower().startswith("units=") for v in obs_header2["map info"]):
                        obs_header2["map info"] = [
                            ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
                            for v in obs_header2["map info"]
                        ]
                    else:
                        obs_header2["map info"].append("units=Meters")
                ht.io.envi.write_envi_header(str(obs_hdr), obs_header2)

                if write_xml:
                    lines = int(obs_header2.get("lines", 0))
                    samples = int(obs_header2.get("samples", 0))
                    _write_xml_sidecar(
                        str(obs_utm),
                        product=f"{product}_OBS",
                        epsg_str=f"EPSG:{out_epsg_int}" if out_epsg_int else out_crs,
                        crs_wkt=crs_wkt,
                        pixel_size=out_ps if match_res else (60.0, 60.0),
                        shape=(lines, samples, obs_cube.shape[2]),
                        start_time=start_time,
                        end_time=end_time,
                        bbox_lonlat=[corner_1, corner_2, corner_3, corner_4],
                        band_names=obs_bandnames,
                        description="Observation datacube",
                    )

                obs_nc.close()
        elif obs_file is not None:
            print(f"OBS exists; skipping regeneration: {obs_utm}")

        # Clean GDAL aux XMLs (not the sidecars we wrote):
        for xml in glob.glob(f"{out_dir}/*.xml"):
            if xml.endswith(".aux.xml"):
                try:
                    os.remove(xml)
                except Exception:
                    pass

        return _finalize_and_return(data_utm)

    finally:
        try:
            img_nc.close()
        except Exception:
            pass


def convert_emit_nc_to_envi(
    emit_nc_paths: Iterable[Union[str, Path]],
    s2_visual_path: Union[str, Path],
    out_dir: Union[str, Path],
    emit_obs_nc: Optional[Union[str, Path]] = None,
    *,
    export_loc: bool = True,
    overwrite: bool = False,
    return_info: bool = False,
    save_info_path: str | Path | None = None,
    save_geotiffs: bool = True,
) -> Path | tuple[Path, dict]:
    """
    Convert EMIT netCDF to ENVI using nc_to_envi and return the main ENVI cube path (.bin).

    - Uses nc_to_envi's deterministic naming + skip logic (overwrite=False by default)
    - Adds a per-input tag to avoid collisions across tiles
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    emit_nc_paths = [Path(p) for p in emit_nc_paths]
    if not emit_nc_paths:
        raise ValueError("emit_nc_paths is empty")

    img_path = emit_nc_paths[0]


    result = nc_to_envi(
        img_file=str(img_path),
        out_dir=str(out_dir),
        temp_dir=str(tmp_dir),
        obs_file=str(emit_obs_nc) if emit_obs_nc else None,
        export_loc=export_loc,
        s2_tif_path=str(s2_visual_path),
        match_res=False,
        write_xml=False,
        overwrite=overwrite,

        return_info=return_info,
        save_info_path=save_info_path,
        save_geotiffs=save_geotiffs,
    )

    out_bin, info = result if return_info else (Path(result), None)

    if not out_bin.exists():
        raise FileNotFoundError(f"nc_to_envi returned {out_bin}, but it does not exist")
    if not out_bin.with_suffix(".hdr").exists():
        raise FileNotFoundError(f"Missing ENVI header for {out_bin} (expected {out_bin.with_suffix('.hdr')})")

    return (out_bin, info) if return_info else out_bin
