from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import json
import shutil
import subprocess

import rasterio
from rasterio.warp import transform_bounds

import pandas as pd


# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=indent, default=str))
    return path


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    """
    Folder layout for one pair.
    """

    run_id: str

    # local
    local_root: Path
    local_emit: Path
    local_s2: Path
    local_emit_utm: Path
    local_plots: Path
    local_tiles: Path
    local_meta: Path
    local_tile_meta: Path
    local_report_md: Path
    local_manifest_csv: Path

    # drive
    drive_root: Optional[Path] = None
    drive_emit: Optional[Path] = None
    drive_s2: Optional[Path] = None
    drive_emit_utm: Optional[Path] = None
    drive_plots: Optional[Path] = None
    drive_tiles: Optional[Path] = None
    drive_meta: Optional[Path] = None
    drive_tile_meta: Optional[Path] = None
    drive_report_md: Optional[Path] = None
    drive_manifest_csv: Optional[Path] = None

    @staticmethod
    def emit_id_from_nc(emit_nc: str | Path) -> str:
        stem = Path(emit_nc).stem
        return stem.replace("EMIT_L2A_RFL_", "", 1)

    @classmethod
    def build(
        cls,
        *,
        emit_nc: str | Path,
        local_root: str | Path,
        drive_base: str | Path | None = None,
    ) -> "RunPaths":
        run_id = cls.emit_id_from_nc(emit_nc)

        # local
        local_root = ensure_dir(local_root)
        local_emit = ensure_dir(local_root / "emit")
        local_s2 = ensure_dir(local_root / "s2")
        local_emit_utm = ensure_dir(local_root / "emit_utm")
        local_plots = ensure_dir(local_root / "plots")
        local_tiles = ensure_dir(local_root / "tiles")
        local_meta = ensure_dir(local_root / "metadata")
        local_tile_meta = ensure_dir(local_meta / "tiles")
        local_report_md = local_root / "report.md"
        local_manifest_csv = local_root / "manifest.csv"

        if drive_base is None:
            return cls(
                run_id=run_id,
                local_root=local_root,
                local_emit=local_emit,
                local_s2=local_s2,
                local_emit_utm=local_emit_utm,
                local_plots=local_plots,
                local_tiles=local_tiles,
                local_meta=local_meta,
                local_tile_meta=local_tile_meta,
                local_report_md=local_report_md,
                local_manifest_csv=local_manifest_csv,
            )

        drive_root = ensure_dir(Path(drive_base) / run_id)
        drive_emit = ensure_dir(drive_root / "emit")
        drive_s2 = ensure_dir(drive_root / "s2")
        drive_emit_utm = ensure_dir(drive_root / "emit_utm")
        drive_plots = ensure_dir(drive_root / "plots")
        drive_tiles = ensure_dir(drive_root / "tiles")
        drive_meta = ensure_dir(drive_root / "metadata")
        drive_tile_meta = ensure_dir(drive_meta / "tiles")

        return cls(
            run_id=run_id,
            local_root=local_root,
            local_emit=local_emit,
            local_s2=local_s2,
            local_emit_utm=local_emit_utm,
            local_plots=local_plots,
            local_tiles=local_tiles,
            local_meta=local_meta,
            local_tile_meta=local_tile_meta,
            local_report_md=local_report_md,
            local_manifest_csv=local_manifest_csv,
            drive_root=drive_root,
            drive_emit=drive_emit,
            drive_s2=drive_s2,
            drive_emit_utm=drive_emit_utm,
            drive_plots=drive_plots,
            drive_tiles=drive_tiles,
            drive_meta=drive_meta,
            drive_tile_meta=drive_tile_meta,
            drive_report_md=drive_root / "report.md",
            drive_manifest_csv=drive_root / "manifest.csv",
        )

    @classmethod
    def from_emit_nc(
        cls,
        emit_nc: str | Path,
        local_root: str | Path,
        drive_root_base: str | Path | None = None,
    ) -> "RunPaths":
        return cls.build(emit_nc=emit_nc, local_root=local_root, drive_base=drive_root_base)


class ReportWriter:
    """
    Report
    """

    def __init__(self, path: str | Path, *, mode: str = "overwrite"):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.mode = mode
        self._started = False

    def start(self, *, title: str = "EMIT and Sentinel-2 pairs report") -> "ReportWriter":
        if self._started:
            return self

        overwrite = self.mode.lower() in {"overwrite", "w", "write"}
        if overwrite:
            self.path.write_text(f"# {title}\n\n- Generated: {utc_now_iso()}\n")
        else:
            if not self.path.exists():
                self.path.write_text(f"# {title}\n\n- Generated: {utc_now_iso()}\n")
        self._started = True
        return self

    def section(self, heading: str, lines: Iterable[str]) -> None:
        if not self._started:
            self.start()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {heading}\n")
            for ln in lines:
                if ln is None:
                    continue
                f.write(f"- {ln}\n")

    def raw(self, text: str) -> None:
        if not self._started:
            self.start()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)


# -----------------------------------------------------------------------------
# EMIT helpers
# -----------------------------------------------------------------------------


def emit_polygon_bounds_wgs84(umm: dict):
    """
    Return bounds from first UMM Polygon
    """
    polys = (
        umm.get("SpatialExtent", {})
        .get("HorizontalSpatialDomain", {})
        .get("Geometry", {})
        .get("GPolygons", [])
    )
    if not polys:
        return None, None

    pts = polys[0].get("Boundary", {}).get("Points", [])
    if not pts:
        return None, None

    lons = [p["Longitude"] for p in pts if "Longitude" in p]
    lats = [p["Latitude"] for p in pts if "Latitude" in p]
    if not lons or not lats:
        return None, None

    bounds = [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))]
    centroid = {"lon": (bounds[0] + bounds[2]) / 2.0, "lat": (bounds[1] + bounds[3]) / 2.0}
    return bounds, centroid


def emit_file_records(umm: dict):
    """
    EMIT UMM sizes and stuff
    """
    recs = umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
    out = []
    for r in recs:
        out.append(
            {
                "name": r.get("Name"),
                "size_bytes": r.get("SizeInBytes"),
                "format": r.get("Format"),
                "checksum": r.get("Checksum", {}),
            }
        )
    return out


def emit_related_urls(umm: dict):
    """
    URLs for report
    """
    urls = umm.get("RelatedUrls", []) or []
    keep = []
    for u in urls:
        url = u.get("URL", "")
        typ = u.get("Type")
        desc = u.get("Description")
        sub = u.get("Subtype")
        if typ in (
            "GET DATA",
            "GET DATA VIA DIRECT ACCESS",
            "EXTENDED METADATA",
            "USE SERVICE API",
        ):
            keep.append({"url": url, "type": typ, "subtype": sub, "description": desc})
    return keep


def write_emit_metadata(
    emit_item: dict,
    out_dir: str | Path,
    *,
    report: ReportWriter | None = None,
) -> dict:
    """
    Metadata and summary for EMIT
    """
    out_dir = ensure_dir(out_dir)

    meta_raw_path = out_dir / "emit_meta_raw.json"
    umm_raw_path = out_dir / "emit_umm_raw.json"
    summary_path = out_dir / "emit_summary.json"

    write_json(meta_raw_path, emit_item.get("meta", {}) or {})
    write_json(umm_raw_path, emit_item.get("umm", {}) or {})

    umm = emit_item.get("umm", {}) or {}
    begin = (umm.get("TemporalExtent") or {}).get("RangeDateTime", {}).get("BeginningDateTime")
    end = (umm.get("TemporalExtent") or {}).get("RangeDateTime", {}).get("EndingDateTime")

    bounds_wgs84, centroid_wgs84 = emit_polygon_bounds_wgs84(umm)

    add_attrs = {
        a["Name"]: a.get("Values")
        for a in (umm.get("AdditionalAttributes") or [])
        if isinstance(a, dict) and "Name" in a
    }

    summary = {
        "granule_ur": umm.get("GranuleUR"),
        "native_id": (emit_item.get("meta", {}) or {}).get("native-id"),
        "concept_id": (emit_item.get("meta", {}) or {}).get("concept-id"),
        "collection": umm.get("CollectionReference"),
        "time": {"begin": begin, "end": end},
        "cloud_cover_umm": umm.get("CloudCover"),
        "spatial": {
            "bounds_wgs84": bounds_wgs84,
            "centroid_wgs84": centroid_wgs84,
        },
        "orbit_scene": {
            "ORBIT": add_attrs.get("ORBIT"),
            "ORBIT_SEGMENT": add_attrs.get("ORBIT_SEGMENT"),
            "SCENE": add_attrs.get("SCENE"),
        },
        "pge": umm.get("PGEVersionClass"),
        "software": {
            "SOFTWARE_BUILD_VERSION": add_attrs.get("SOFTWARE_BUILD_VERSION"),
            "SOFTWARE_DELIVERY_VERSION": add_attrs.get("SOFTWARE_DELIVERY_VERSION"),
        },
        "files": emit_file_records(umm),
        "related_urls": emit_related_urls(umm),
        "size_mb_from_item": emit_item.get("size"),
    }

    write_json(summary_path, summary)

    if report is not None:
        report.section(
            "EMIT (from CMR UMM)",
            [
                f"GranuleUR: {summary['granule_ur']}",
                f"Native ID: {summary['native_id']}",
                f"Time begin/end: {begin} → {end}",
                f"CloudCover (UMM): {summary['cloud_cover_umm']}",
                f"Bounds WGS84 (UMM polygon): {bounds_wgs84}",
                f"Centroid WGS84: {centroid_wgs84}",
                f"Orbit/Scene: ORBIT={summary['orbit_scene']['ORBIT']} SCENE={summary['orbit_scene']['SCENE']}",
                f"Raw metadata: {umm_raw_path.name}, {meta_raw_path.name}",
            ],
        )

    return summary


# -----------------------------------------------------------------------------
# S2 helpers
# -----------------------------------------------------------------------------


def bounds_from_bbox(bbox: Any) -> Optional[list[float]]:
    if not bbox or len(bbox) != 4:
        return None
    xmin, ymin, xmax, ymax = map(float, bbox)
    return [xmin, ymin, xmax, ymax]


def centroid_from_bounds(bounds: Optional[list[float]]) -> Optional[dict[str, float]]:
    if not bounds:
        return None
    xmin, ymin, xmax, ymax = bounds
    return {"lon": (xmin + xmax) / 2.0, "lat": (ymin + ymax) / 2.0}


def pick_s2_assets_minimal(s2_dict: dict) -> dict:
    assets = s2_dict.get("assets", {}) or {}
    keep_keys = ["visual", "B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
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
    report: ReportWriter | None = None,
) -> dict:
    """
    Metadata and summary
    """
    out_dir = ensure_dir(out_dir)

    s2_dict = s2_item if isinstance(s2_item, dict) else (s2_item.to_dict() if hasattr(s2_item, "to_dict") else {})

    raw_path = out_dir / "s2_item_raw.json"
    summary_path = out_dir / "s2_summary.json"

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
                f"Raw metadata: {raw_path.name}",
            ],
        )

    return summary


# -----------------------------------------------------------------------------
# this one is for tiles
# -----------------------------------------------------------------------------


def tif_geo_summary(path: str | Path) -> dict:
    """
    get geoTIFF spatial summary
    """

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "error": "not found"}

    with rasterio.open(p) as ds:
        b = ds.bounds
        crs = ds.crs
        out = {
            "path": str(p),
            "crs": crs.to_string() if crs else None,
            "bounds_crs": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
            "shape": [int(ds.height), int(ds.width)],
            "res": [float(ds.res[0]), float(ds.res[1])] if ds.res else None,
            "nodata": ds.nodata,
        }

        if crs:
            wb = transform_bounds(crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
            out["bounds_wgs84"] = [float(wb[0]), float(wb[1]), float(wb[2]), float(wb[3])]
            xmin, ymin, xmax, ymax = out["bounds_wgs84"]
            out["centroid_wgs84"] = {"lon": (xmin + xmax) / 2.0, "lat": (ymin + ymax) / 2.0}

        return out


@dataclass
class TileRecord:
    idx: int
    emit_tif: str
    s2_tif: str
    plot_png: Optional[str] = None

    emit_black_frac: Optional[float] = None
    s2_black_frac: Optional[float] = None

    emit_geo: Optional[dict] = None
    s2_geo: Optional[dict] = None

    emit_window: Optional[dict] = None
    s2_window: Optional[dict] = None

    def to_manifest_row(self) -> dict:
        row = {
            "idx": int(self.idx),
            "emit_tif": self.emit_tif,
            "s2_tif": self.s2_tif,
            "plot_png": self.plot_png,
            "emit_black_frac": self.emit_black_frac,
            "s2_black_frac": self.s2_black_frac,
        }

        def _pull(prefix: str, g: Optional[dict]):
            if not isinstance(g, dict):
                return
            row[f"{prefix}_crs"] = g.get("crs")
            row[f"{prefix}_bounds_crs"] = g.get("bounds_crs")
            row[f"{prefix}_bounds_wgs84"] = g.get("bounds_wgs84")
            row[f"{prefix}_centroid_wgs84"] = g.get("centroid_wgs84")

        _pull("emit", self.emit_geo)
        _pull("s2", self.s2_geo)
        return row


def write_tile_metadata(
    record: TileRecord,
    tile_info: dict,
    out_dir: str | Path,
    *,
    emit_granule: str | None = None,
    emit_time: Any = None,
    s2_id: str | None = None,
    s2_datetime: str | None = None,
    params: dict | None = None,
) -> tuple[Path, dict]:
    """
    write summary for tiles 
    """
    out_dir = ensure_dir(out_dir)

    doc = {
        "tile_id": int(record.idx),
        "created_utc": utc_now_iso(),
        "pair": {
            "emit_granule": emit_granule,
            "emit_time": emit_time,
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
        "tile_info": tile_info or {},
    }

    path = out_dir / f"tile_{record.idx:03d}.json"
    write_json(path, doc)
    return path, record.to_manifest_row()


def write_manifest_csv(path: str | Path, rows: list[dict] | list[TileRecord]) -> Path:
    """
    Write manifest.csv
    """

    path = Path(path)
    ensure_dir(path.parent)

    if not rows:
        pd.DataFrame([]).to_csv(path, index=False)
        return path

    if isinstance(rows[0], TileRecord):
        data = [r.to_manifest_row() for r in rows] 
    else:
        data = rows

    pd.DataFrame(data).to_csv(path, index=False)
    return path


# -----------------------------------------------------------------------------
# Drive stuff
# -----------------------------------------------------------------------------


def copy_any(
    src: str | Path,
    dst: str | Path,
    *,
    overwrite: bool = False,
    use_rsync: bool = True,
    exclude: list[str] | None = None,
) -> None:
    """
    copy file/dir
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
            cmd = ["rsync", "-a"]
            if not overwrite:
                cmd += ["--ignore-existing"]
            for pat in exclude:
                cmd += ["--exclude", pat]
            if src.is_dir():
                cmd += [str(src) + "/", str(dst) + "/"]
            else:
                cmd += [str(src), str(dst)]
            subprocess.run(cmd, check=True)
            return
        except Exception:
            pass

    if src.is_dir():
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                if target.exists() and overwrite:
                    shutil.rmtree(target)
                if not target.exists():
                    shutil.copytree(item, target)
            else:
                if target.exists() and not overwrite:
                    continue
                shutil.copy2(item, target)
    else:
        target = (dst / src.name) if dst.is_dir() else dst
        if target.exists() and not overwrite:
            return
        ensure_dir(target.parent)
        shutil.copy2(src, target)


def write_archive_map(path: str | Path, mapping: dict[str, Any], *, report: ReportWriter | None = None) -> Path:
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
