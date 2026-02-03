from pathlib import Path
import re
import json
import numpy as np
import rasterio
from arosics import COREG_LOCAL

def s2_bandmap_from_template(s2_template_tif: str) -> dict[str, int]:
    with rasterio.open(s2_template_tif) as src:
        descs = src.descriptions or ()
    if not descs or all(d is None for d in descs):
        raise ValueError(f"S2 template has no band descriptions: {s2_template_tif}")

    bandmap = {}
    for i, d in enumerate(descs, start=1):  # 1-based
        if not d:
            continue
        band_code = str(d).split("_", 1)[0].upper()  # "B04_red" -> "B04"
        bandmap[band_code] = i
    return bandmap


def closest_band_1based(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wavelengths_nm, float) - float(target_nm)))) + 1


def load_emit_wavelengths_nm_from_nc(emit_nc_path: str) -> np.ndarray:
    emit_nc_path = str(emit_nc_path)

    try:
        import xarray as xr
        ds = xr.open_dataset(emit_nc_path, group="sensor_band_parameters")
        candidates = ["wavelengths", "wavelength", "wavelength_center", "band_center_wavelength"]
        var = None
        for c in candidates:
            if c in ds.variables:
                var = c
                break
        if var is None:
            for c in candidates:
                if c in ds.coords:
                    var = c
                    break
        if var is None:
            raise KeyError(f"Could not find wavelength variable in sensor_band_parameters. Vars: {list(ds.variables)}")

        wl = ds[var].values.astype(np.float64)
        units = (ds[var].attrs.get("units") or "").lower()

    except Exception:
        from netCDF4 import Dataset
        with Dataset(emit_nc_path, "r") as nc:
            grp = nc.groups.get("sensor_band_parameters", None)
            if grp is None:
                raise KeyError("Group 'sensor_band_parameters' not found in EMIT netCDF.")
            candidates = ["wavelengths", "wavelength", "wavelength_center", "band_center_wavelength"]
            var = None
            for c in candidates:
                if c in grp.variables:
                    var = c
                    break
            if var is None:
                raise KeyError(f"Could not find wavelength variable. Vars: {list(grp.variables.keys())}")
            v = grp.variables[var]
            wl = np.array(v[:], dtype=np.float64)
            units = (getattr(v, "units", "") or "").lower()

    if units in ("nanometers", "nm", ""):
        wl_nm = wl
    elif units in ("micrometers", "um", "µm"):
        wl_nm = wl * 1000.0
    else:
        wl_nm = wl

    return wl_nm


def cache_wavelengths_json(wavelengths_nm: np.ndarray, out_path: str):
    out = {
        "wavelength_units": "nm",
        "wavelengths_nm": [float(x) for x in np.asarray(wavelengths_nm).ravel()],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))


def _norm_code(x: str) -> str:
    return str(x).split("_", 1)[0].upper()



def coregister_s2_granule_to_emit_granule(
    *,
    emit_ref_tif: str,
    s2_tgt_tif: str,
    emit_nc_path: str,
    s2_template_tif: str,
    out_s2_tif: str,
    prefer=("B08", "B04"),
    window_size=(512, 512),   
    grid_res=600,             
    max_points=500,           
    max_shift=50,
    min_reliability=60,  
    tieP_filter_level=3,
    nodata_emit=65535,
    nodata_s2=None,
    out_gsd=[10, 10],
    resamp_calc="cubic",
    resamp_deshift="cubic",
    cliptoextent=True
):
    s2_map = s2_bandmap_from_template(str(s2_template_tif))
    emit_wl_nm = load_emit_wavelengths_nm_from_nc(str(emit_nc_path))

    target_nm = {"B08": 842.0, "B04": 665.0}
    emit_match = {k: closest_band_1based(emit_wl_nm, target_nm[k]) for k in target_nm}

    if nodata_s2 is None:
        with rasterio.open(s2_tgt_tif) as src:
            nodata_s2 = src.nodata

    attempts = []
    last_err = None
 
    for code_raw in prefer:
        code = _norm_code(code_raw)
        if code not in target_nm:
            continue
        if code not in s2_map:
            attempts.append({"s2_code": code, "success": False, "error": f"{code} not in S2 template descriptions"})
            continue

        try:
            CRL = COREG_LOCAL(
                im_ref=str(emit_ref_tif),
                im_tgt=str(s2_tgt_tif),
                grid_res=float(grid_res),
                max_points=int(max_points) if max_points is not None else None,
                window_size=tuple(window_size),

                path_out=str(out_s2_tif),
                fmt_out="GTIFF",
                out_crea_options=["TILED=YES", "COMPRESS=DEFLATE"],

                r_b4match=int(emit_match[code]),
                s_b4match=int(s2_map[code]),

                max_shift=int(max_shift),
                min_reliability=int(min_reliability),
                tieP_filter_level=int(tieP_filter_level),

                nodata=(nodata_emit, nodata_s2),
                resamp_alg_calc=resamp_calc,
                resamp_alg_deshift=resamp_deshift,

                match_gsd=False,
                align_grids=True,
                out_gsd=list(out_gsd),

            )

            result = CRL.correct_shifts(cliptoextent=cliptoextent) 
            ok = bool(getattr(CRL, "success", True)) 
            info = {
                "success": ok,
                "s2_code": code,
                "s2_match_band_1b": int(s2_map[code]),
                "emit_match_band_1b": int(emit_match[code]),
                "emit_match_wl_nm": float(emit_wl_nm[int(emit_match[code]) - 1]),
                "grid_res": grid_res,
                "max_points": max_points,
                "result_keys": list(result.keys()) if isinstance(result, dict) else None,
            }
            attempts.append(info)

            if ok:
                return {"final": info, "attempts": attempts, "out_s2_tif": str(out_s2_tif)}
            
        except Exception as e:
            last_err = str(e)
            attempts.append({"s2_code": code, "success": False, "error": last_err})

    return {"final": {"success": False, "error": last_err or "All attempts failed"}, "attempts": attempts, "out_s2_tif": str(out_s2_tif)}
