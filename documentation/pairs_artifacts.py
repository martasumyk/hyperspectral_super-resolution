from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Iterable
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

import rasterio
from rasterio.warp import transform_bounds

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=indent, default=str))
    return path



@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for a single EMIT and S-2 pair"""

    # local outputs
    local_root: Path
    local_emit: Path
    local_s2: Path
    local_emit_utm: Path
    local_plots: Path
    local_tiles: Path

    # drive outputs
    drive_root: Optional[Path] = None
    drive_emit: Optional[Path] = None
    drive_s2: Optional[Path] = None
    drive_emit_utm: Optional[Path] = None
    drive_plots: Optional[Path] = None
    drive_tiles: Optional[Path] = None
    drive_meta: Optional[Path] = None

    drive_report_md: Optional[Path] = None
    drive_manifest_csv: Optional[Path] = None

    @staticmethod
    def _emit_id_from_nc(emit_nc: str | Path) -> str:
        p = Path(emit_nc)
        stem = p.stem
        # notebook convention: EMIT_L2A_RFL_<id>
        return stem.replace("EMIT_L2A_RFL_", "", 1)

    @classmethod
    def build(
        cls,
        *,
        emit_nc: str | Path,
        local_root: str | Path,
        drive_base: str | Path | None = None,
    ) -> "RunPaths":
        """Create output folders. If drive_base is provided, it will create a subfolder per granule id."""
        local_root = Path(local_root)
        ensure_dir(local_root)

        local_emit = ensure_dir(local_root / "emit")
        local_s2 = ensure_dir(local_root / "s2")
        local_emit_utm = ensure_dir(local_root / "emit_utm")
        local_plots = ensure_dir(local_root / "plots")
        local_tiles = ensure_dir(local_root / "tiles")

        if drive_base is None:
            return cls(
                local_root=local_root,
                local_emit=local_emit,
                local_s2=local_s2,
                local_emit_utm=local_emit_utm,
                local_plots=local_plots,
                local_tiles=local_tiles,
            )

        emit_id = cls._emit_id_from_nc(emit_nc)
        drive_root = ensure_dir(Path(drive_base) / emit_id)

        drive_emit = ensure_dir(drive_root / "emit")
        drive_s2 = ensure_dir(drive_root / "s2")
        drive_emit_utm = ensure_dir(drive_root / "emit_utm")
        drive_plots = ensure_dir(drive_root / "plots")
        drive_tiles = ensure_dir(drive_root / "tiles")
        drive_meta = ensure_dir(drive_root / "metadata")

        return cls(
            local_root=local_root,
            local_emit=local_emit,
            local_s2=local_s2,
            local_emit_utm=local_emit_utm,
            local_plots=local_plots,
            local_tiles=local_tiles,
            drive_root=drive_root,
            drive_emit=drive_emit,
            drive_s2=drive_s2,
            drive_emit_utm=drive_emit_utm,
            drive_plots=drive_plots,
            drive_tiles=drive_tiles,
            drive_meta=drive_meta,
            drive_report_md=drive_root / "report.md",
            drive_manifest_csv=drive_root / "manifest.csv",
        )


class RunReport:
    """
    Create markdown report for tiling process.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def start(self, *, overwrite: bool = True, 
              title: str = "EMIT and Sentinel-2 pairing report", 
              extra_header_lines: Iterable[str] | None = None) -> None:
        mode = "w" if overwrite else "a"
        with self.path.open(mode, encoding="utf-8") as f:
            if overwrite:
                f.write(f"# {title}\n")
                f.write(f"\n- Generated: {utc_now_iso()}\n")
                if extra_header_lines:
                    for ln in extra_header_lines:
                        f.write(f"- {ln}\n")
                f.write("\n")

    def section(self, heading: str, lines: Iterable[str]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {heading}\n")
            for ln in lines:
                if ln is None:
                    continue
                f.write(f"- {ln}\n")

    def raw(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)


def bounds_from_bbox(bbox: Any) -> Optional[list[float]]:
    """
    Calculate Sentinel-2 bounds from its bbox.
    """
    if not bbox or len(bbox) != 4:
        return None
    xmin, ymin, xmax, ymax = map(float, bbox)
    return [xmin, ymin, xmax, ymax]


def centroid_from_bounds(bounds: Optional[list[float]]) -> Optional[dict[str, float]]:
    """
    Calculate centroid from bounds.
    """
    if not bounds:
        return None
    xmin, ymin, xmax, ymax = bounds
    return {"lon": (xmin + xmax) / 2.0, "lat": (ymin + ymax) / 2.0}


def write_emit_metadata(
    emit_item: dict,
    out_dir: str | Path,
    *,
    report: RunReport | None = None,
) -> dict:
    """
    Write raw and summarized EMIT metadata

    """
    out_dir = ensure_dir(Path(out_dir))

    meta_raw_path = out_dir / "emit_meta_raw.json"
    umm_raw_path = out_dir / "emit_umm_raw.json"

    write_json(meta_raw_path, emit_item.get("meta", {}) or {})
    write_json(umm_raw_path, emit_item.get("umm", {}) or {})

    umm = emit_item.get("umm", {}) or {}
    begin = (((umm.get("TemporalExtent") or {}).get("RangeDateTime") or {}).get("BeginningDateTime"))
    end = (((umm.get("TemporalExtent") or {}).get("RangeDateTime") or {}).get("EndingDateTime"))

    add_attrs = {a.get("Name"): a.get("Values") for a in (umm.get("AdditionalAttributes") or []) if isinstance(a, dict)}

    summary = {
        "granule_ur": umm.get("GranuleUR"),
        "native_id": (emit_item.get("meta", {}) or {}).get("native-id"),
        "concept_id": (emit_item.get("meta", {}) or {}).get("concept-id"),
        "time": {"begin": begin, "end": end},
        "cloud_cover_umm": umm.get("CloudCover"),
        "orbit_scene": {
            "ORBIT": add_attrs.get("ORBIT"),
            "ORBIT_SEGMENT": add_attrs.get("ORBIT_SEGMENT"),
            "SCENE": add_attrs.get("SCENE"),
        },
        "software": {
            "SOFTWARE_BUILD_VERSION": add_attrs.get("SOFTWARE_BUILD_VERSION"),
            "SOFTWARE_DELIVERY_VERSION": add_attrs.get("SOFTWARE_DELIVERY_VERSION"),
        },
        "size_mb_from_item": emit_item.get("size"),
    }

    summary_path = out_dir / "emit_summary.json"
    write_json(summary_path, summary)

    if report is not None:
        report.section(
            "EMIT (from CMR UMM)",
            [
                f"GranuleUR: {summary['granule_ur']}",
                f"Native ID: {summary['native_id']}",
                f"Time begin/end: {begin} → {end}",
                f"CloudCover (UMM): {summary['cloud_cover_umm']}",
                f"Orbit/Scene: ORBIT={summary['orbit_scene']['ORBIT']} SCENE={summary['orbit_scene']['SCENE']}",
                f"Raw metadata saved: {meta_raw_path.name}, {umm_raw_path.name}",
            ],
        )

    return summary


def pick_s2_assets_minimal(s2_dict: dict) -> dict:
    """
    Pick a subset of S-2's assets for documentation.
    """
    assets = s2_dict.get("assets", {}) or {}
    keep_keys = [
        "visual",
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
        "B12",
        "SCL",
    ]
    out = {}
    for k in keep_keys:
        a = assets.get(k)
        if isinstance(a, dict):
            out[k] = {"href": a.get("href"), "type": a.get("type")}
    return out


def write_s2_metadata(
    s2_item: Any,
    out_dir: str | Path,
    *,
    report: RunReport | None = None,
) -> dict:
    """
    Write raw and summarized Sentinel-2 metadata.
    """
    out_dir = ensure_dir(Path(out_dir))

    s2_dict = s2_item if isinstance(s2_item, dict) else (s2_item.to_dict() if hasattr(s2_item, "to_dict") else {})

    raw_path = out_dir / "s2_item_raw.json"
    write_json(raw_path, s2_dict)

    props = s2_dict.get("properties", {}) or {}
    bbox = s2_dict.get("bbox")
    bounds = bounds_from_bbox(bbox)

    summary = {
        "id": s2_dict.get("id"),
        "datetime": props.get("datetime"),
        "created": props.get("created"),
        "updated": props.get("updated"),
        "platform": props.get("platform"),
        "product_uri": props.get("s2:product_uri"),
        "mgrs": {
            "grid_code": props.get("grid:code"),
            "utm_zone": props.get("mgrs:utm_zone"),
            "latitude_band": props.get("mgrs:latitude_band"),
            "grid_square": props.get("mgrs:grid_square"),
        },
        "projection": {"proj:code": props.get("proj:code")},
        "spatial": {
            "bbox_wgs84": bounds,
            "centroid_wgs84": centroid_from_bounds(bounds),
            "geometry_type": (s2_dict.get("geometry") or {}).get("type"),
        },
        "clouds": {
            "eo:cloud_cover": props.get("eo:cloud_cover"),
            "s2:cloud_shadow_percentage": props.get("s2:cloud_shadow_percentage"),
            "s2:medium_proba_clouds_percentage": props.get("s2:medium_proba_clouds_percentage"),
            "s2:high_proba_clouds_percentage": props.get("s2:high_proba_clouds_percentage"),
            "s2:thin_cirrus_percentage": props.get("s2:thin_cirrus_percentage"),
        },
        "scene_percentages": {
            "s2:nodata_pixel_percentage": props.get("s2:nodata_pixel_percentage"),
            "s2:dark_features_percentage": props.get("s2:dark_features_percentage"),
            "s2:vegetation_percentage": props.get("s2:vegetation_percentage"),
            "s2:not_vegetated_percentage": props.get("s2:not_vegetated_percentage"),
            "s2:water_percentage": props.get("s2:water_percentage"),
            "s2:unclassified_percentage": props.get("s2:unclassified_percentage"),
            "s2:snow_ice_percentage": props.get("s2:snow_ice_percentage"),
        },
        "sun": {"view:sun_azimuth": props.get("view:sun_azimuth"), "view:sun_elevation": props.get("view:sun_elevation")},
        "processing": {
            "s2:processing_baseline": props.get("s2:processing_baseline"),
            "s2:generation_time": props.get("s2:generation_time"),
            "processing:software": props.get("processing:software"),
            "earthsearch:s3_path": props.get("earthsearch:s3_path"),
            "earthsearch:boa_offset_applied": props.get("earthsearch:boa_offset_applied"),
        },
        "assets_minimal": pick_s2_assets_minimal(s2_dict),
    }

    summary_path = out_dir / "s2_summary.json"
    write_json(summary_path, summary)

    if report is not None:
        report.section(
            "Sentinel-2 (from STAC)",
            [
                f"ID: {summary['id']}",
                f"Datetime: {summary['datetime']}",
                f"Platform: {summary['platform']}",
                f"Product URI: {summary['product_uri']}",
                f"proj:code: {summary['projection']['proj:code']}",
                f"MGRS: {summary['mgrs']}",
                f"BBox WGS84: {summary['spatial']['bbox_wgs84']}",
                f"Centroid WGS84: {summary['spatial']['centroid_wgs84']}",
                f"eo:cloud_cover (%): {summary['clouds']['eo:cloud_cover']}",
                f"Raw metadata saved: {raw_path.name}",
            ],
        )

    return summary


# --------------------------------------------------------------------------------------
# GeoTIFF summary (for tiles)
# --------------------------------------------------------------------------------------


def tif_geo_summary(path: str | Path) -> dict:
    """
    Return basic geospatial metadata for a GeoTIFF.
    """

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "error": "not found"}

    with rasterio.open(p) as ds:
        b = ds.bounds
        crs = ds.crs
        bounds_crs = [float(b.left), float(b.bottom), float(b.right), float(b.top)]

        bounds_wgs84 = None
        centroid_wgs84 = None
        try:
            if crs is not None:
                wb = transform_bounds(crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
                bounds_wgs84 = [float(wb[0]), float(wb[1]), float(wb[2]), float(wb[3])]
                centroid_wgs84 = centroid_from_bounds(bounds_wgs84)
        except Exception as e:
            bounds_wgs84 = None

        return {
            "path": str(p),
            "crs": str(crs) if crs is not None else None,
            "size": {"width": int(ds.width), "height": int(ds.height), "count": int(ds.count)},
            "bounds_crs": bounds_crs,
            "bounds_wgs84": bounds_wgs84,
            "centroid_wgs84": centroid_wgs84,
            "transform": tuple(ds.transform) if ds.transform is not None else None,
        }


# --------------------------------------------------------------------------------------
# Tile metadata + manifest
# --------------------------------------------------------------------------------------

@dataclass
class TileRecord:
    idx: int
    emit_tif: str
    s2_tif: str
    plot_png: str | None = None

    emit_geo: dict | None = None
    s2_geo: dict | None = None

    emit_black_frac: float | None = None
    s2_black_frac: float | None = None

    emit_window: dict | None = None
    s2_window: dict | None = None

    def to_manifest_row(self) -> dict:
        row = {
            "idx": int(self.idx),
            "emit_tif": self.emit_tif,
            "s2_tif": self.s2_tif,
            "plot_png": self.plot_png,
            "emit_black_frac": self.emit_black_frac,
            "s2_black_frac": self.s2_black_frac,
        }

        def _pull(prefix: str, g: dict | None):
            if not isinstance(g, dict):
                return
            row[f"{prefix}_crs"] = g.get("crs")
            row[f"{prefix}_bounds_crs"] = g.get("bounds_crs")
            row[f"{prefix}_bounds_wgs84"] = g.get("bounds_wgs84")
            row[f"{prefix}_centroid_wgs84"] = g.get("centroid_wgs84")

        _pull("emit", self.emit_geo)
        _pull("s2", self.s2_geo)
        return row


def write_tile_doc(
    *,
    out_dir: str | Path,
    record: TileRecord,
    emit_granule: str | None = None,
    emit_datetime: Any = None,
    s2_id: str | None = None,
    s2_datetime: str | None = None,
    params: dict | None = None,
) -> Path:
    out_dir = ensure_dir(Path(out_dir))

    doc = {
        "tile_id": int(record.idx),
        "created_utc": utc_now_iso(),
        "pair": {
            "emit_granule": emit_granule,
            "emit_time": emit_datetime,
            "s2_id": s2_id,
            "s2_datetime": s2_datetime,
        },
        "geometry": {
            "emit_tile": record.emit_geo,
            "s2_tile": record.s2_geo,
        },
        "windows": {
            "emit_window": record.emit_window,
            "s2_window": record.s2_window,
        },
        "params": params or {},
        "quality": {
            "emit_black_frac": record.emit_black_frac,
            "s2_black_frac": record.s2_black_frac,
        },
        "files": {
            "emit_tif": record.emit_tif,
            "s2_tif": record.s2_tif,
            "plot_png": record.plot_png,
        },
    }

    path = out_dir / f"tile_{record.idx:03d}.json"
    write_json(path, doc)
    return path


def write_manifest_csv(path: str | Path, records: list[TileRecord] | list[dict]) -> Path:
    """Write a manifest.csv.

    Accepts either:
      - list[TileRecord]
      - list[dict] rows
    """

    path = Path(path)
    ensure_dir(path.parent)

    if len(records) == 0:
        df = pd.DataFrame([])
    else:
        first = records[0]
        if isinstance(first, TileRecord):
            rows = [r.to_manifest_row() for r in records]  # type: ignore
        else:
            rows = records  # type: ignore
        df = pd.DataFrame(rows)

    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------------------
# Drive copy
# --------------------------------------------------------------------------------------


def copy_any(src: str | Path, dst: str | Path, *, overwrite: bool = False, use_rsync: bool = True, exclude: list[str] | None = None) -> None:
    """
    Copy file or directory src -> dst
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")

    exclude = exclude or []

    if src.is_file():
        ensure_dir(dst.parent)
    else:
        ensure_dir(dst)

    if use_rsync:
        try:
            if src.is_dir():
                cmd = ["rsync", "-a"]
                if not overwrite:
                    cmd += ["--ignore-existing"]
                for pat in exclude:
                    cmd += ["--exclude", pat]
                cmd += [str(src) + "/", str(dst) + "/"]
            else:
                cmd = ["rsync", "-a"]
                if not overwrite:
                    cmd += ["--ignore-existing"]
                cmd += [str(src), str(dst)]
            subprocess.run(cmd, check=True)
            return
        except Exception as e:
            print("[WARN] rsync failed, falling back to shutil:", e)

    if src.is_dir():
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                if target.exists() and overwrite:
                    shutil.rmtree(target)
                if not target.exists():
                    shutil.copytree(item, target)
            else:
                if target.exists() and (not overwrite):
                    continue
                shutil.copy2(item, target)
    else:
        if dst.is_dir():
            target = dst / src.name
        else:
            target = dst
        if target.exists() and (not overwrite):
            return
        ensure_dir(target.parent)
        shutil.copy2(src, target)


def write_archive_map(path: str | Path, mapping: dict[str, Any], *, report: RunReport | None = None) -> Path:
    path = Path(path)
    write_json(path, mapping)

    if report is not None:
        report.section(
            "Drive archival",
            [
                f"Raw EMIT copied to: {mapping.get('drive_raw_emit')}",
                f"Raw S2 copied to: {mapping.get('drive_raw_s2')}",
                f"EMIT products copied to: {mapping.get('drive_emit_reprojections')}",
            ],
        )

    return path
