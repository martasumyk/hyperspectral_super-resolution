#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTER
Space-based Imaging Spectroscopy and Thermal PathfindER

"""

import datetime as dt
import os
import glob
import xml.etree.ElementTree as ET

import netCDF4 as nc
import hytools as ht
import numpy as np
import rasterio

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


# def nc_to_envi(
#     img_file: str,
#     out_dir: str,
#     temp_dir: str,
#     obs_file: str | None = None,
#     export_loc: bool = False,
#     crid: str = "000",
#     s2_tif_path: str | None = None,  # derive CRS/res from Sentinel-2 GeoTIFF
#     epsg: int | str | None = None,   # (kept unused to match current behavior)
#     match_res: bool = False,         # copy pixel size from S2
#     write_xml: bool = True           # write XML sidecar(s)
# ):
#     """
#     Export EMIT L1B_RDN or L2A_RFL to ENVI and write XML sidecars.
#     (Speed-tuned version: precomputed GLT indices, BSQ interleave, GDAL skip-nosource, threading/cache.)
#     """
#     import os, glob, subprocess
#     import numpy as np
#     import rasterio
#     import netCDF4 as nc
#     import datetime as dt

#     os.makedirs(out_dir, exist_ok=True)
#     os.makedirs(temp_dir, exist_ok=True)

#     img_nc = nc.Dataset(img_file)
#     try:
#         # --- speed: disable netCDF masking/scaling (we handle NoData ourselves) ---
#         if hasattr(img_nc, "set_auto_mask"):
#             img_nc.set_auto_mask(False)
#         for vname in ("radiance", "reflectance"):
#             if vname in img_nc.variables and hasattr(img_nc.variables[vname], "set_auto_maskandscale"):
#                 img_nc.variables[vname].set_auto_maskandscale(False)

#         # Product & data
#         if "radiance" in img_nc.variables.keys():
#             data = img_nc["radiance"]
#             product = "L1B_RDN"
#             description = "Radiance micro-watts/cm^2/nm/sr"
#         elif "reflectance" in img_nc.variables.keys():
#             data = img_nc["reflectance"]
#             product = "L2A_RFL"
#             description = "Reflectance (unitless)"
#         else:
#             print("Unrecognized input image dataset (expected 'radiance' or 'reflectance').")
#             return

#         # Bands metadata
#         fwhm = img_nc["sensor_band_parameters"]["fwhm"][:].data
#         waves = img_nc["sensor_band_parameters"]["wavelengths"][:].data

#         # Geotransform/GLT
#         gt = np.array(img_nc.__dict__["geotransform"])
#         glt = np.zeros(list(img_nc.groups["location"]["glt_x"].shape) + [2], dtype=np.int32)
#         glt[..., 0] = np.array(img_nc.groups["location"]["glt_x"])
#         glt[..., 1] = np.array(img_nc.groups["location"]["glt_y"])
#         valid_glt = np.all(glt != 0, axis=-1)
#         glt[valid_glt] -= 1  # 1-based -> 0-based

#         # --- speed: precompute gather indices once ---
#         gy = glt[..., 1][valid_glt]
#         gx = glt[..., 0][valid_glt]

#         # ENVI header 'map info' as list (Geographic, degrees)
#         map_info = [
#             "Geographic Lat/Lon",
#             1, 1,
#             float(gt[0]), float(gt[3]),
#             float(gt[1]), float(-gt[5]),
#             "WGS-84",
#             "units=degrees",
#         ]

#         # Extents & time
#         latitude = np.array(img_nc.groups["location"]["lat"])
#         longitude = np.array(img_nc.groups["location"]["lon"])

#         corner_1 = [float(longitude[0, 0]), float(latitude[0, 0])]
#         corner_2 = [float(longitude[0, -1]), float(latitude[0, -1])]
#         corner_3 = [float(longitude[-1, -1]), float(latitude[-1, -1])]
#         corner_4 = [float(longitude[-1, 0]), float(latitude[-1, 0])]

#         # Robust time parse (keeps original format if present)
#         try:
#             start_time = dt.datetime.strptime(img_nc.time_coverage_start, "%Y-%m-%dT%H:%M:%S+0000")
#             end_time = dt.datetime.strptime(img_nc.time_coverage_end, "%Y-%m-%dT%H:%M:%S+0000")
#         except ValueError:
#             start_time = dt.datetime.strptime(img_nc.time_coverage_start.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S+0000")
#             end_time = dt.datetime.strptime(img_nc.time_coverage_end.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S+0000")

#         # Prepare unprojected ENVI (GCS) container to warp from
#         data_header = ht.io.envi_header_dict()
#         data_header["lines"] = glt.shape[0]
#         data_header["samples"] = glt.shape[1]
#         data_header["bands"] = data.shape[2]
#         data_header["byte order"] = 0
#         data_header["data ignore value"] = NO_DATA_VALUE
#         data_header["data type"] = 4
#         # --- speed: BSQ is friendlier for GDAL warps than BIL ---
#         data_header["interleave"] = "bsq"
#         data_header["map info"] = map_info

#         data_gcs = f"{temp_dir}/data_gcs"
#         writer = ht.io.WriteENVI(data_gcs, data_header)
#         data_prj = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)

#         print(f"Exporting EMIT {product} dataset")
#         # --- speed: precomputed gy/gx + single fill() per iteration ---
#         for band_num in range(data.shape[2]):
#             band = data[:, :, band_num]
#             data_prj[valid_glt] = band[gy, gx]
#             writer.write_band(data_prj, band_num)
#             data_prj.fill(NO_DATA_VALUE)

#         # Choose target CRS & resolution (keep original behavior: derive from S2 only)
#         tr_opt = "-tr 60 60"
#         out_crs = None
#         out_epsg_int = None
#         out_ps = (60.0, 60.0)
#         crs_wkt = None

#         if s2_tif_path:
#             with rasterio.open(s2_tif_path) as src:
#                 s2_crs = src.crs
#                 out_crs = s2_crs.to_string()
#                 out_epsg_int = s2_crs.to_epsg()
#                 crs_wkt = s2_crs.to_wkt()
#                 if match_res and src.res is not None:
#                     out_ps = (abs(float(src.res[0])), abs(float(src.res[1])))
#                     tr_opt = f"-tr {out_ps[0]} {out_ps[1]}"

#         # Warp helper with safe args & faster options
#         def _run_warp(src_path: str, dst_path: str):
#             # thread/cache env (don’t override if user already set them)
#             os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
#             os.environ.setdefault("GDAL_CACHEMAX", "1024")  # MB

#             nodata = [ "-srcnodata", str(NO_DATA_VALUE), "-dstnodata", str(NO_DATA_VALUE) ]
#             # --- speed: skip all-empty source blocks, unified nodata, init dest as NoData ---
#             skip = [ "-wo", "SKIP_NOSOURCE=YES", "-wo", "UNIFIED_SRC_NODATA=YES", "-wo", "INIT_DEST=NO_DATA" ]
#             multi = [ "-multi", "-wo", "NUM_THREADS=ALL_CPUS", "-wm", "1024" ]

#             args = [
#                 "gdalwarp", "-overwrite",
#                 "-t_srs", out_crs if out_crs is not None else "",
#                 *tr_opt.split(),
#                 "-r", "near",
#                 *nodata, *skip, *multi,
#                 "-of", "ENVI",
#                 src_path, dst_path
#             ]
#             # remove empty items if out_crs is None to avoid passing a blank
#             args = [a for a in args if a != ""]
#             subprocess.run(args, check=True)

#         # Warp data
#         print(f"Projecting data to {out_crs} {'(match S2 res)' if match_res else '(60 m)'}")
#         data_utm = f'{out_dir}/SISTER_EMIT_{product}_{start_time.strftime("%Y%m%dT%H%M%S")}_{crid}.bin'
#         _run_warp(data_gcs, data_utm)

#         # Fix header
#         data_hdr = data_utm.replace(".bin", ".hdr")
#         data_header = ht.io.envi.parse_envi_header(data_hdr)
#         data_header["description"] = description
#         data_header["band names"] = []  # keep as original behavior
#         data_header["fwhm"] = fwhm.tolist()
#         data_header["wavelength"] = waves.tolist()
#         data_header["wavelength units"] = "nanometers"
#         data_header["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
#         data_header["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
#         data_header["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
#         data_header["coordinate system string"] = []
#         data_header["sensor type"] = "EMIT"
#         if "map info" in data_header and isinstance(data_header["map info"], list):
#             if any(isinstance(v, str) and v.lower().startswith("units=") for v in data_header["map info"]):
#                 data_header["map info"] = [
#                     ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
#                     for v in data_header["map info"]
#                 ]
#             else:
#                 data_header["map info"].append("units=Meters")
#         ht.io.envi.write_envi_header(data_hdr, data_header)

#         # Optional XML for DATA cube
#         if write_xml:
#             # We don't know the warped lines/samples without reading; parse from header fields:
#             lines = int(data_header.get("lines", 0))
#             samples = int(data_header.get("samples", 0))
#             _write_xml_sidecar(
#                 data_utm,
#                 product=product,
#                 epsg_str=f"EPSG:{out_epsg_int}" if out_epsg_int else out_crs,
#                 crs_wkt=crs_wkt,
#                 pixel_size=out_ps if match_res else (60.0, 60.0),
#                 shape=(lines, samples, data.shape[2]),
#                 start_time=start_time,
#                 end_time=end_time,
#                 bbox_lonlat=[corner_1, corner_2, corner_3, corner_4],
#                 wavelengths=waves.tolist() if hasattr(waves, "tolist") else list(waves),
#                 fwhm=fwhm.tolist() if hasattr(fwhm, "tolist") else list(fwhm),
#                 band_names=None,
#                 description=description,
#             )

#         # Location export
#         if export_loc:
#             print("Exporting EMIT location dataset")
#             loc_header = ht.io.envi_header_dict()
#             loc_header["lines"] = glt.shape[0]
#             loc_header["samples"] = glt.shape[1]
#             loc_header["bands"] = 3
#             loc_header["byte order"] = 0
#             loc_header["data ignore value"] = NO_DATA_VALUE
#             loc_header["data type"] = 4
#             loc_header["interleave"] = "bsq"  # faster for GDAL
#             loc_header["map info"] = map_info

#             loc_gcs = f"{temp_dir}/loc_gcs"
#             writer = ht.io.WriteENVI(loc_gcs, loc_header)

#             # turn off autoscale for location vars (if present)
#             loc_vars = img_nc.groups["location"].variables
#             for name in ("lon", "lat", "elev"):
#                 v = loc_vars.get(name)
#                 if v is not None and hasattr(v, "set_auto_maskandscale"):
#                     v.set_auto_maskandscale(False)

#             loc_band = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)
#             for band_num, band_name in enumerate(("lon", "lat", "elev")):
#                 band = loc_vars[band_name][:]
#                 loc_band[valid_glt] = band[gy, gx]
#                 writer.write_band(loc_band, band_num)
#                 loc_band.fill(NO_DATA_VALUE)

#             loc_utm = f'{out_dir}/SISTER_EMIT_{product}_{start_time.strftime("%Y%m%dT%H%M%S")}_{crid}_LOC.bin'
#             print(f"Projecting location datacube to {out_crs} {'(match S2 res)' if match_res else '(60 m)'}")
#             _run_warp(loc_gcs, loc_utm)

#             loc_hdr = loc_utm.replace(".bin", ".hdr")
#             loc_header = ht.io.envi.parse_envi_header(loc_hdr)
#             loc_header["band names"] = ["longitude", "latitude", "elevation"]
#             loc_header["description"] = "Location datacube"
#             loc_header["coordinate system string"] = []
#             loc_header["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
#             loc_header["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
#             loc_header["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
#             loc_header["sensor type"] = "EMIT"
#             if "map info" in loc_header and isinstance(loc_header["map info"], list):
#                 if any(isinstance(v, str) and v.lower().startswith("units=") for v in loc_header["map info"]):
#                     loc_header["map info"] = [
#                         ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
#                         for v in loc_header["map info"]
#                     ]
#                 else:
#                     loc_header["map info"].append("units=Meters")
#             ht.io.envi.write_envi_header(loc_hdr, loc_header)

#             if write_xml:
#                 lines = int(loc_header.get("lines", 0))
#                 samples = int(loc_header.get("samples", 0))
#                 _write_xml_sidecar(
#                     loc_utm,
#                     product=f"{product}_LOC",
#                     epsg_str=f"EPSG:{out_epsg_int}" if out_epsg_int else out_crs,
#                     crs_wkt=crs_wkt,
#                     pixel_size=out_ps if match_res else (60.0, 60.0),
#                     shape=(lines, samples, 3),
#                     start_time=start_time,
#                     end_time=end_time,
#                     bbox_lonlat=[corner_1, corner_2, corner_3, corner_4],
#                     band_names=["longitude", "latitude", "elevation"],
#                     description="Location datacube",
#                 )

#         # OBS export
#         if obs_file:
#             print("Exporting EMIT observation dataset")
#             obs_nc = nc.Dataset(obs_file)
#             try:
#                 # ---- robust read of OBS cube ----
#                 try:
#                     obs_cube, obs_bandnames = _read_obs_cube_and_names(obs_nc)  # (H, W, K)
#                 except Exception as e:
#                     print(f"[WARN] Could not parse OBS file '{obs_file}': {e}")
#                     print("Skipping OBS export. (Pass obs_file=None to silence this.)")
#                     obs_nc.close()
#                     obs_nc = None

#                 if obs_nc is not None:
#                     obs_header = ht.io.envi_header_dict()
#                     obs_header["lines"] = glt.shape[0]
#                     obs_header["samples"] = glt.shape[1]
#                     obs_header["bands"] = obs_cube.shape[2]
#                     obs_header["byte order"] = 0
#                     obs_header["data ignore value"] = NO_DATA_VALUE
#                     obs_header["data type"] = 4
#                     obs_header["interleave"] = "bsq"  # faster for GDAL
#                     obs_header["map info"] = map_info

#                     obs_gcs = f"{temp_dir}/obs_gcs"
#                     writer = ht.io.WriteENVI(obs_gcs, obs_header)

#                     obs_band = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)
#                     for band_num in range(obs_cube.shape[2]):
#                         band = obs_cube[..., band_num]
#                         obs_band[valid_glt] = band[gy, gx]
#                         writer.write_band(obs_band, band_num)
#                         obs_band.fill(NO_DATA_VALUE)

#                     obs_utm = f'{out_dir}/SISTER_EMIT_{product}_{start_time.strftime("%Y%m%dT%H%M%S")}_{crid}_OBS.bin'
#                     print(f'Projecting observation datacube to {out_crs} {("(match S2 res)" if match_res else "(60 m)")}')
#                     _run_warp(obs_gcs, obs_utm)

#                     # Update header
#                     obs_header_file = obs_utm.replace(".bin", ".hdr")
#                     obs_header = ht.io.envi.parse_envi_header(obs_header_file)
#                     obs_header["band names"] = obs_bandnames
#                     obs_header["description"] = "Observation datacube"
#                     obs_header["coordinate system string"] = []
#                     obs_header["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
#                     obs_header["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
#                     obs_header["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
#                     obs_header["sensor type"] = "EMIT"
#                     if "map info" in obs_header and isinstance(obs_header["map info"], list):
#                         if any(isinstance(v, str) and v.lower().startswith("units=") for v in obs_header["map info"]):
#                             obs_header["map info"] = [
#                                 ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
#                                 for v in obs_header["map info"]
#                             ]
#                         else:
#                             obs_header["map info"].append("units=Meters")
#                     ht.io.envi.write_envi_header(obs_header_file, obs_header)

#                     if write_xml:
#                         lines = int(obs_header.get("lines", 0))
#                         samples = int(obs_header.get("samples", 0))
#                         _write_xml_sidecar(
#                             obs_utm,
#                             product=f"{product}_OBS",
#                             epsg_str=f"EPSG:{out_epsg_int}" if out_epsg_int else out_crs,
#                             crs_wkt=crs_wkt,
#                             pixel_size=out_ps if match_res else (60.0, 60.0),
#                             shape=(lines, samples, obs_cube.shape[2]),
#                             start_time=start_time,
#                             end_time=end_time,
#                             bbox_lonlat=[corner_1, corner_2, corner_3, corner_4],
#                             band_names=obs_bandnames,
#                             description="Observation datacube",
#                         )
#             finally:
#                 try:
#                     if obs_nc is not None:
#                         obs_nc.close()
#                 except Exception:
#                     pass

#         # Clean GDAL aux XMLs (not the sidecars we wrote):
#         for xml in glob.glob(f"{out_dir}/*.xml"):
#             if xml.endswith(".aux.xml"):
#                 try:
#                     os.remove(xml)
#                 except Exception:
#                     pass
#     finally:
#         try:
#             img_nc.close()
#         except Exception:
#             pass

from pathlib import PosixPath

def nc_to_envi(
    img_file: str,
    out_dir: str,
    temp_dir: str,
    obs_file: str | None = None,
    export_loc: bool = False,
    crid: str = "000",
    s2_tif_path: str | None = None,  # NEW: derive CRS/res from Sentinel-2 GeoTIFF
    epsg: int | str | None = None,   # NEW: explicit EPSG if no TIF
    match_res: bool = False,         # NEW: copy pixel size from S2
    write_xml: bool = True           # NEW: write XML sidecar(s)
):
    """
    Export EMIT L1B_RDN or L2A_RFL to ENVI and write XML sidecars.

    Parameters
    ----------
    img_file : str
        EMIT L1B_RDN or L2A_RFL netCDF (with 'geotransform' and 'location' groups)
    out_dir : str
        Output directory for ENVI datasets
    temp_dir : str
        Temporary working directory
    obs_file : str | None
        EMIT L1B_OBS netCDF; if provided, exports *_OBS
    export_loc : bool
        Export *_LOC datacube (lon, lat, elev)
    crid : str
        Collection/reprocessing id for filenames
    s2_tif_path : str | None
        Path to a Sentinel-2 GeoTIFF to copy CRS (and pixel size if match_res=True)
    epsg : int | str | None
        EPSG code if s2_tif_path is None
    match_res : bool
        If True, copy pixel size from Sentinel-2; else use 60 m
    write_xml : bool
        If True, write an XML sidecar with rich metadata next to each ENVI output
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    print(img_file)

    img_nc = nc.Dataset(PosixPath(img_file))
    print(f"Opened EMIT image dataset: {img_file}")

    for vname in ("radiance", "reflectance"):
        if vname in img_nc.variables:
            var = img_nc.variables[vname]
            if hasattr(var, "set_auto_maskandscale"):
                print(f"Disabling auto mask/scale for variable '{vname}'")
                var.set_auto_maskandscale(False)

    # optional: disable dataset-level auto masking for everything (faster, no masked arrays)
    if hasattr(img_nc, "set_auto_mask"):
        img_nc.set_auto_mask(False)

    # Product & data
    if "radiance" in img_nc.variables.keys():
        data = img_nc["radiance"]
        product = "L1B_RDN"
        description = "Radiance micro-watts/cm^2/nm/sr"
    elif "reflectance" in img_nc.variables.keys():
        data = img_nc["reflectance"]
        product = "L2A_RFL"
        description = "Reflectance (unitless)"
    else:
        print("Unrecognized input image dataset (expected 'radiance' or 'reflectance').")
        return

    fwhm = img_nc["sensor_band_parameters"]["fwhm"][:].data
    waves = img_nc["sensor_band_parameters"]["wavelengths"][:].data

    # Geotransform/GLT
    gt = np.array(img_nc.__dict__["geotransform"])
    glt = np.zeros(list(img_nc.groups["location"]["glt_x"].shape) + [2], dtype=np.int32)
    glt[..., 0] = np.array(img_nc.groups["location"]["glt_x"])
    glt[..., 1] = np.array(img_nc.groups["location"]["glt_y"])
    valid_glt = np.all(glt != 0, axis=-1)
    glt[valid_glt] -= 1

    # ENVI header 'map info' as list (Geographic, degrees)
    map_info = [
        "Geographic Lat/Lon",
        1, 1,
        float(gt[0]), float(gt[3]),
        float(gt[1]), float(-gt[5]),
        "WGS-84",
        "units=degrees",
    ]

    # Extents & time
    latitude = np.array(img_nc.groups["location"]["lat"])
    longitude = np.array(img_nc.groups["location"]["lon"])

    corner_1 = [float(longitude[0, 0]), float(latitude[0, 0])]
    corner_2 = [float(longitude[0, -1]), float(latitude[0, -1])]
    corner_3 = [float(longitude[-1, -1]), float(latitude[-1, -1])]
    corner_4 = [float(longitude[-1, 0]), float(latitude[-1, 0])]

    start_time = dt.datetime.strptime(img_nc.time_coverage_start, "%Y-%m-%dT%H:%M:%S+0000")
    end_time = dt.datetime.strptime(img_nc.time_coverage_end, "%Y-%m-%dT%H:%M:%S+0000")

    # Prepare unprojected ENVI (GCS) container to warp from
    data_header = ht.io.envi_header_dict()
    data_header["lines"] = glt.shape[0]
    data_header["samples"] = glt.shape[1]
    data_header["bands"] = data.shape[2]
    data_header["byte order"] = 0
    data_header["data ignore value"] = NO_DATA_VALUE
    data_header["data type"] = 4
    data_header["interleave"] = "bil"
    data_header["map info"] = map_info

    data_gcs = f"{temp_dir}/data_gcs"
    writer = ht.io.WriteENVI(data_gcs, data_header)
    data_prj = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)

    print(f"Exporting EMIT {product} dataset")
    gy = glt[..., 1][valid_glt]
    gx = glt[..., 0][valid_glt]
    for band_num in range(data.shape[2]):
        band = data[:, :, band_num]          # netCDF slice -> ndarray
        data_prj[valid_glt] = band[gy, gx]   # gather via precomputed indices
        writer.write_band(data_prj, band_num)
        data_prj.fill(NO_DATA_VALUE)

    # Choose target CRS & resolution
    tr_opt = "-tr 60 60"
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
            if match_res and src.res is not None:
                out_ps = (abs(float(src.res[0])), abs(float(src.res[1])))
                tr_opt = f"-tr {out_ps[0]} {out_ps[1]}"
    # elif epsg is not None:
    #     out_epsg_int = int(epsg)
    #     s2_crs = rasterio.crs.CRS.from_epsg(out_epsg_int)
    #     out_crs = s2_crs.to_string()
    #     crs_wkt = s2_crs.to_wkt()
    # else:
    #     zone, direction = utm_zone(longitude, latitude)
    #     epsg_dir = 6 if direction == "North" else 7
    #     out_crs = "EPSG:32%s%02d" % (epsg_dir, zone)
    #     out_epsg_int = int(out_crs.split(":")[1])
    #     crs_wkt = rasterio.crs.CRS.from_epsg(out_epsg_int).to_wkt()

    # Warp data
    print(f"Projecting data to {out_crs} {'(match S2 res)' if match_res else '(60 m)'}")
    data_utm = f'{out_dir}/SISTER_EMIT_{product}_{start_time.strftime("%Y%m%dT%H%M%S")}_{crid}.bin'
    os.system(f"gdalwarp -overwrite -t_srs {out_crs} {tr_opt} -r near -of ENVI {data_gcs} {data_utm}")

    # Fix header
    data_hdr = data_utm.replace(".bin", ".hdr")
    data_header = ht.io.envi.parse_envi_header(data_hdr)
    data_header["description"] = description
    data_header["band names"] = []
    data_header["fwhm"] = fwhm.tolist()
    data_header["wavelength"] = waves.tolist()
    data_header["wavelength units"] = "nanometers"
    data_header["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    data_header["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    data_header["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
    data_header["coordinate system string"] = []
    data_header["sensor type"] = "EMIT"
    if "map info" in data_header and isinstance(data_header["map info"], list):
        if any(isinstance(v, str) and v.lower().startswith("units=") for v in data_header["map info"]):
            data_header["map info"] = [
                ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
                for v in data_header["map info"]
            ]
        else:
            data_header["map info"].append("units=Meters")
    ht.io.envi.write_envi_header(data_hdr, data_header)

    # Optional XML for DATA cube
    if write_xml:
        # We don't know the warped lines/samples without reading; parse from header fields:
        lines = int(data_header.get("lines", 0))
        samples = int(data_header.get("samples", 0))
        _write_xml_sidecar(
            data_utm,
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

    # Location export
    if export_loc:
        print("Exporting EMIT location dataset")
        loc_header = ht.io.envi_header_dict()
        loc_header["lines"] = glt.shape[0]
        loc_header["samples"] = glt.shape[1]
        loc_header["bands"] = 3
        loc_header["byte order"] = 0
        loc_header["data ignore value"] = NO_DATA_VALUE
        loc_header["data type"] = 4
        loc_header["interleave"] = "bil"
        loc_header["map info"] = map_info

        loc_gcs = f"{temp_dir}/loc_gcs"
        writer = ht.io.WriteENVI(loc_gcs, loc_header)

        loc_vars = img_nc.groups["location"].variables
        # for name in ("lon", "lat", "elev"):
        #     if hasattr(loc_vars[name], "set_auto_maskandscale"):
        #         loc_vars[name].set_auto_maskandscale(False)

        loc_band = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)
        for band_num, name in enumerate(("lon", "lat", "elev")):
            band = loc_vars[name][:]              # read once; no np.copy()
            loc_band[valid_glt] = band[gy, gx]
            writer.write_band(loc_band, band_num)
            loc_band.fill(NO_DATA_VALUE)

        loc_utm = f'{out_dir}/SISTER_EMIT_{product}_{start_time.strftime("%Y%m%dT%H%M%S")}_{crid}_LOC.bin'
        print(f"Projecting location datacube to {out_crs} {'(match S2 res)' if match_res else '(60 m)'}")
        os.system(f"gdalwarp -overwrite -t_srs {out_crs} {tr_opt} -r near -of ENVI {loc_gcs} {loc_utm}")

        loc_hdr = loc_utm.replace(".bin", ".hdr")
        loc_header = ht.io.envi.parse_envi_header(loc_hdr)
        loc_header["band names"] = ["longitude", "latitude", "elevation"]
        loc_header["description"] = "Location datacube"
        loc_header["coordinate system string"] = []
        loc_header["start acquisition time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        loc_header["end acquisition time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        loc_header["bounding box"] = [corner_1, corner_2, corner_3, corner_4]
        loc_header["sensor type"] = "EMIT"
        if "map info" in loc_header and isinstance(loc_header["map info"], list):
            if any(isinstance(v, str) and v.lower().startswith("units=") for v in loc_header["map info"]):
                loc_header["map info"] = [
                    ("units=Meters" if (isinstance(v, str) and v.lower().startswith("units=")) else v)
                    for v in loc_header["map info"]
                ]
            else:
                loc_header["map info"].append("units=Meters")
        ht.io.envi.write_envi_header(loc_hdr, loc_header)

        if write_xml:
            lines = int(loc_header.get("lines", 0))
            samples = int(loc_header.get("samples", 0))
            _write_xml_sidecar(
                loc_utm,
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

    # OBS export
    if obs_file:
        print("Exporting EMIT observation dataset")
        obs_nc = nc.Dataset(obs_file)

        # ---- NEW: robust read of OBS cube ----
        try:
            obs_cube, obs_bandnames = _read_obs_cube_and_names(obs_nc)  # (H, W, 11)
        except Exception as e:
            print(f"[WARN] Could not parse OBS file '{obs_file}': {e}")
            print("Skipping OBS export. (Pass obs_file=None to silence this.)")
            obs_nc.close()
            obs_nc = None
        # --------------------------------------

        if obs_nc is not None:
            obs_header = ht.io.envi_header_dict()
            obs_header['lines'] = glt.shape[0]
            obs_header['samples'] = glt.shape[1]
            obs_header['bands'] = obs_cube.shape[2]
            obs_header['byte order'] = 0
            obs_header['data ignore value'] = NO_DATA_VALUE
            obs_header['data type'] = 4
            obs_header['interleave'] = 'bil'
            obs_header['map info'] = map_info

            obs_gcs = f'{temp_dir}/obs_gcs'
            writer = ht.io.WriteENVI(obs_gcs, obs_header)

            obs_band = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)
            for band_num in range(obs_cube.shape[2]):
                band = obs_cube[..., band_num]
                obs_band[valid_glt] = band[gy, gx]
                writer.write_band(obs_band, band_num)
                obs_band.fill(NO_DATA_VALUE)

            obs_utm = f'{out_dir}/SISTER_EMIT_{product}_{start_time.strftime("%Y%m%dT%H%M%S")}_{crid}_OBS.bin'
            print(f'Projecting observation datacube to {out_crs} {("(match S2 res)" if match_res else "(60 m)")}')
            os.system(f'gdalwarp -overwrite -t_srs {out_crs} {tr_opt} -r near -of ENVI {obs_gcs} {obs_utm}')

            # Update header
            obs_header_file = obs_utm.replace('.bin', '.hdr')
            obs_header = ht.io.envi.parse_envi_header(obs_header_file)
            obs_header['band names'] = obs_bandnames
            obs_header['description'] = 'Observation datacube'
            obs_header['coordinate system string'] = []
            obs_header['start acquisition time'] = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            obs_header['end acquisition time'] = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            obs_header['bounding box'] = [corner_1, corner_2, corner_3, corner_4]
            obs_header['sensor type'] = 'EMIT'
            if 'map info' in obs_header and isinstance(obs_header['map info'], list):
                if any(isinstance(v, str) and v.lower().startswith('units=') for v in obs_header['map info']):
                    obs_header['map info'] = [('units=Meters' if (isinstance(v, str) and v.lower().startswith('units=')) else v)
                                            for v in obs_header['map info']]
                else:
                    obs_header['map info'].append('units=Meters')
            ht.io.envi.write_envi_header(obs_header_file, obs_header)

            if write_xml:
                lines = int(obs_header.get("lines", 0))
                samples = int(obs_header.get("samples", 0))
                _write_xml_sidecar(
                    obs_utm,
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

    # Clean GDAL aux XMLs (not the sidecars we wrote):
    for xml in glob.glob(f"{out_dir}/*.xml"):
        # keep our sidecars (which match our outputs); remove GDAL's auto *.img.aux.xml if created
        if xml.endswith(".aux.xml"):
            try:
                os.remove(xml)
            except Exception:
                pass
