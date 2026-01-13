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


import subprocess


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
    overwrite: bool = False,      # NEW: if False, don't regenerate existing outputs
    tag: str | None = None,       # NEW: disambiguate outputs for multiple tiles
) -> Path:
    """
    Export EMIT L1B_RDN or L2A_RFL to ENVI and write XML sidecars.

    NEW:
      - overwrite: if False, skips any output that already exists (.bin + .hdr)
      - tag: optional string appended to filenames to avoid collisions across tiles
    Returns:
      Path to the main projected ENVI cube (.bin)
    """

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

        sbp = img_nc.groups["sensor_band_parameters"]
        fwhm  = np.asarray(sbp.variables["fwhm"][:])
        waves = np.asarray(sbp.variables["wavelengths"][:])

        # Geotransform/GLT
        gt = np.array(get_attr(img_nc, "geotransform"))
        loc = img_nc.groups["location"]
        glt_x = np.asarray(loc.variables["glt_x"][:])
        glt_y = np.asarray(loc.variables["glt_y"][:])
        glt = np.zeros(list(glt_x.shape) + [2], dtype=np.int32)
        glt[..., 0] = glt_x
        glt[..., 1] = glt_y
        valid_glt = np.all(glt != 0, axis=-1)
        glt[valid_glt] -= 1

        # ENVI header 'map info' as list
        map_info = [
            "Geographic Lat/Lon",
            1, 1,
            float(gt[0]), float(gt[3]),
            float(gt[1]), float(-gt[5]),
            "WGS-84",
            "units=degrees",
        ]

        # Extents & time
        latitude  = np.asarray(loc.variables["lat"][:])
        longitude = np.asarray(loc.variables["lon"][:])

        corner_1 = [float(longitude[0, 0]), float(latitude[0, 0])]
        corner_2 = [float(longitude[0, -1]), float(latitude[0, -1])]
        corner_3 = [float(longitude[-1, -1]), float(latitude[-1, -1])]
        corner_4 = [float(longitude[-1, 0]), float(latitude[-1, 0])]

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

                s2_bounds = src.bounds  # left, bottom, right, top
                s2_res = (abs(float(src.res[0])), abs(float(src.res[1])))

                if match_res and src.res is not None:
                    out_ps = s2_res
        if not out_crs:
            raise ValueError("out_crs is None. Provide s2_tif_path or enable epsg/UTM fallback.")

        tr_args = ["-tr", str(out_ps[0]), str(out_ps[1])] if (match_res and out_ps) else ["-tr", "60", "60"]

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

        if not (need_data or need_loc or need_obs):
            print(f"All requested outputs already exist; skipping. Returning: {data_utm}")
            return data_utm

    def _run_gdalwarp(src_path: str, dst_path: str):
        cmd = ["gdalwarp"]

        if overwrite:
            cmd.append("-overwrite")

        # Target CRS
        cmd += ["-t_srs", out_crs]

        # Target resolution (you already control this via tr_args)
        cmd += tr_args

        # Force target extent to match S2 grid (this is what improves alignment)
        if s2_bounds is not None:
            cmd += [
                "-te",
                str(s2_bounds.left),
                str(s2_bounds.bottom),
                str(s2_bounds.right),
                str(s2_bounds.top),
                "-tap",  # snap extent to the -tr grid
            ]

        # Nodata handling (prevents nodata from being treated as real values)
        cmd += ["-srcnodata", str(NO_DATA_VALUE), "-dstnodata", str(NO_DATA_VALUE)]

        # Resampling + output format
        cmd += ["-r", "near", "-of", "ENVI", src_path, dst_path]

        subprocess.run(cmd, check=True)

        # ----------------------------------------------------------------

        # Precompute gather indices once (used by all exports)
        gy = glt[..., 1][valid_glt]
        gx = glt[..., 0][valid_glt]

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
            data_prj = np.full((glt.shape[0], glt.shape[1]), NO_DATA_VALUE, dtype=np.float32)

            for band_num in range(data.shape[2]):
                band = data[:, :, band_num]
                data_prj[valid_glt] = band[gy, gx]
                writer.write_band(data_prj, band_num)
                data_prj.fill(NO_DATA_VALUE)

            print(f"Projecting data to {out_crs} {'(match S2 res)' if match_res else '(60 m)'} -> {data_utm}")
            _run_gdalwarp(data_gcs, str(data_utm))

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
            data_header2["coordinate system string"] = []
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
                band = loc_vars[name][:]
                loc_band[valid_glt] = band[gy, gx]
                writer.write_band(loc_band, band_num)
                loc_band.fill(NO_DATA_VALUE)

            print(f"Projecting location datacube to {out_crs} {'(match S2 res)' if match_res else '(60 m)'} -> {loc_utm}")
            _run_gdalwarp(loc_gcs, str(loc_utm))

            loc_header2 = ht.io.envi.parse_envi_header(str(loc_hdr))
            loc_header2["band names"] = ["longitude", "latitude", "elevation"]
            loc_header2["description"] = "Location datacube"
            loc_header2["coordinate system string"] = []
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
                    obs_band[valid_glt] = band[gy, gx]
                    writer.write_band(obs_band, band_num)
                    obs_band.fill(NO_DATA_VALUE)

                print(f"Projecting observation datacube to {out_crs} {'(match S2 res)' if match_res else '(60 m)'} -> {obs_utm}")
                _run_gdalwarp(obs_gcs, str(obs_utm))

                obs_header2 = ht.io.envi.parse_envi_header(str(obs_hdr))
                obs_header2["band names"] = obs_bandnames
                obs_header2["description"] = "Observation datacube"
                obs_header2["coordinate system string"] = []
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

        return data_utm

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
) -> Path:
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


    out_bin = nc_to_envi(
        img_file = str(img_path),
        out_dir=str(out_dir),
        temp_dir=str(tmp_dir),
        obs_file=str(emit_obs_nc) if emit_obs_nc else None,
        export_loc=export_loc,
        s2_tif_path=str(s2_visual_path),
        match_res=False,
        write_xml=False,
        overwrite=overwrite,
    )

    out_bin = Path(out_bin)
    if not out_bin.exists():
        raise FileNotFoundError(f"nc_to_envi returned {out_bin}, but it does not exist")
    if not out_bin.with_suffix(".hdr").exists():
        raise FileNotFoundError(f"Missing ENVI header for {out_bin} (expected {out_bin.with_suffix('.hdr')})")

    return out_bin