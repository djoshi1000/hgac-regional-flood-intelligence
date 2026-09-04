from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import zipfile

import geopandas as gpd
import requests
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import box

from .dem_usgs import houston_dem_prefix, list_s3_keys, download_tiles

HGAC_DATASET_ID = "5e9698743aa0425bae0fc97653b9bbc1_0"
HGAC_SHAPEFILE_URL = (
    "https://hub.arcgis.com/api/v3/datasets/"
    f"{HGAC_DATASET_ID}/downloads/data?format=shp&spatialRefId=4326&where=1%3D1"
)


def _normalize(value):
    s = str(value).strip().lower()
    s = Path(s).name
    s = re.sub(r"\.(tif|tiff|las|laz|shp)$", "", s)
    s = re.sub(r"preliminary|final|draft", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def ensure_hgac_lidar_grid(cache_root: str | Path):
    root = Path(cache_root)
    out_dir = root / "lidar_grid"
    shp_candidates = list(out_dir.rglob("*.shp")) if out_dir.exists() else []
    if shp_candidates:
        return shp_candidates[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    z = root / "lidar_grid.zip"
    with requests.get(HGAC_SHAPEFILE_URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with z.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if not zipfile.is_zipfile(z):
        raise RuntimeError("H-GAC LiDAR grid download did not return a valid ZIP.")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out_dir)
    shp = list(out_dir.rglob("*.shp"))
    if not shp:
        raise RuntimeError("H-GAC LiDAR grid ZIP contained no shapefile.")
    return shp[0]


def build_s3_index(dem_cfg: dict, cache_root: str | Path, force=False):
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    idx_path = root / "houston_b24_s3_index.json"
    if idx_path.exists() and not force:
        return json.loads(idx_path.read_text(encoding="utf-8"))
    records = []
    for work_unit in dem_cfg.get("work_units", []):
        prefix = houston_dem_prefix(dem_cfg["project"], work_unit)
        for key in list_s3_keys(prefix):
            records.append({"work_unit": work_unit, "key": key, "name": Path(key).name})
    idx_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _candidate_values(row):
    vals = []
    for col, val in row.items():
        if col == "geometry" or val is None:
            continue
        s = str(val).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            vals.append((col, s))
    # Prefer the known H-GAC tile-name field when available.
    vals.sort(key=lambda kv: 0 if kv[0].lower() in {"name", "tile", "tilename", "tile_name"} else 1)
    return vals


def match_grid_to_s3(selected: gpd.GeoDataFrame, index_records):
    basename = {r["name"].lower(): r["key"] for r in index_records}
    token_map = {}
    for r in index_records:
        token_map.setdefault(_normalize(r["name"]), []).append(r["key"])
    all_rows = [(r["key"], _normalize(r["name"])) for r in index_records]
    matched, unmatched = [], []
    for idx, row in selected.iterrows():
        found = None
        vals = _candidate_values(row)
        for _, v in vals:
            b = Path(v).name.lower()
            for candidate in (b, b + ".tif", b + ".tiff"):
                if candidate in basename:
                    found = basename[candidate]
                    break
            if found:
                break
        if not found:
            for _, v in vals:
                t = _normalize(v)
                if len(t) < 5:
                    continue
                mm = token_map.get(t, [])
                if len(mm) == 1:
                    found = mm[0]
                    break
        if not found:
            for _, v in vals:
                t = _normalize(v)
                if len(t) < 6:
                    continue
                mm = [k for k, kt in all_rows if t in kt or kt in t]
                mm = sorted(set(mm))
                if len(mm) == 1:
                    found = mm[0]
                    break
        if found:
            matched.append(found)
        else:
            unmatched.append(int(idx) if isinstance(idx, int) else str(idx))
    return sorted(set(matched)), unmatched


def _clip_mosaic(tile_paths, aoi_wgs84, out_path, resolution_m):
    datasets = [rasterio.open(str(p)) for p in tile_paths]
    try:
        crss = {str(ds.crs) for ds in datasets}
        if len(crss) != 1:
            raise RuntimeError(f"Cached DEM tiles span multiple CRSs: {sorted(crss)}")
        crs = datasets[0].crs
        aoi = aoi_wgs84.to_crs(crs)
        mosaic, transform = merge(
            datasets,
            bounds=tuple(aoi.total_bounds),
            res=(float(resolution_m), float(resolution_m)),
            nodata=datasets[0].nodata,
        )
        meta = datasets[0].meta.copy()
        meta.update(
            height=mosaic.shape[1], width=mosaic.shape[2], transform=transform,
            count=mosaic.shape[0], compress="deflate", tiled=True, BIGTIFF="IF_SAFER"
        )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix(".bbox.tif")
        with rasterio.open(temp, "w", **meta) as dst:
            dst.write(mosaic)
        with rasterio.open(temp) as src:
            clipped, tr = mask(src, [g.__geo_interface__ for g in aoi.geometry], crop=True)
            meta2 = src.meta.copy()
            meta2.update(height=clipped.shape[1], width=clipped.shape[2], transform=tr,
                         compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
        with rasterio.open(out, "w", **meta2) as dst:
            dst.write(clipped)
        temp.unlink(missing_ok=True)
        return out
    finally:
        for ds in datasets:
            ds.close()


def clip_from_regional_raster(regional_path, aoi_wgs84, out_path, resolution_m=None):
    """Clip a prebuilt GeoTIFF/COG/VRT cache on the fly.

    A VRT or ArcGIS-exported regional raster cache is much faster than physically
    merging and rewriting a huge 1-m H-GAC DEM.  When ``resolution_m`` is given,
    the output is resampled directly to that resolution during the crop.
    """
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import calculate_default_transform, Resampling

    regional_path = Path(regional_path)
    if not regional_path.exists():
        raise FileNotFoundError(regional_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(regional_path) as src:
        aoi = aoi_wgs84.to_crs(src.crs)
        source = src
        vrt = None
        if resolution_m is not None:
            res = float(resolution_m)
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height, *src.bounds,
                resolution=(res, res),
            )
            vrt = WarpedVRT(
                src, crs=src.crs, transform=transform, width=width, height=height,
                resampling=Resampling.bilinear,
            )
            source = vrt
        try:
            clipped, tr = mask(
                source, [g.__geo_interface__ for g in aoi.geometry], crop=True
            )
            meta = source.meta.copy()
            meta.update(
                height=clipped.shape[1], width=clipped.shape[2], transform=tr,
                compress="deflate", tiled=True, BIGTIFF="IF_SAFER",
            )
            with rasterio.open(out, "w", **meta) as dst:
                dst.write(clipped)
        finally:
            if vrt is not None:
                vrt.close()
    return out


def _cached_tiles_for_aoi(aoi_wgs84: gpd.GeoDataFrame, dem_cfg: dict, dem_cache_cfg: dict):
    """Return local cached tile paths intersecting an AOI, downloading only missing tiles."""
    cache_root = Path(dem_cache_cfg.get("cache_root", "data/regional_cache/dem"))
    cache_root.mkdir(parents=True, exist_ok=True)
    grid_path = ensure_hgac_lidar_grid(cache_root)
    tiles = gpd.read_file(grid_path)
    aa = aoi_wgs84.to_crs(tiles.crs)
    geom = aa.geometry.union_all()
    selected = tiles[tiles.geometry.intersects(geom)].copy()
    if selected.empty:
        raise RuntimeError(
            "The selected model domain does not intersect the configured 2024 Houston LiDAR grid. "
            "The current Houston B24 source is not a complete H-GAC DEM. For other counties/bayous, "
            "configure dem_cache.regional_dem_path to a regional 3DEP/agency GeoTIFF/COG/VRT or "
            "extend the DEM provider catalog."
        )
    index_records = build_s3_index(dem_cfg, cache_root)
    keys, unmatched = match_grid_to_s3(selected, index_records)
    if not keys:
        raise RuntimeError(
            f"No USGS Houston B24 DEM names matched the selected H-GAC grid polygons. "
            f"Unmatched={unmatched[:10]}"
        )
    tile_dir = cache_root / "tiles"
    local = download_tiles(keys, tile_dir, workers=int(dem_cfg.get("download_workers", 4)))
    if not local:
        raise RuntimeError("DEM tile download/cache returned no files.")
    return selected, local, unmatched


def ensure_dem_for_aoi(aoi_wgs84: gpd.GeoDataFrame, out_path: str | Path,
                       resolution_m: float, dem_cfg: dict, dem_cache_cfg: dict):
    """Create one DEM crop using the fastest available regional cache mode."""
    out = Path(out_path)
    if out.exists():
        return {"mode": "workspace_cache", "path": str(out), "resolution_m": float(resolution_m)}

    regional = str(dem_cache_cfg.get("regional_dem_path", "") or "").strip()
    if regional:
        clip_from_regional_raster(regional, aoi_wgs84, out, resolution_m=resolution_m)
        return {"mode": "regional_raster", "path": str(out), "resolution_m": float(resolution_m)}

    selected, local, unmatched = _cached_tiles_for_aoi(aoi_wgs84, dem_cfg, dem_cache_cfg)
    _clip_mosaic(local, aoi_wgs84, out, float(resolution_m))
    return {
        "mode": "tile_cache", "path": str(out), "resolution_m": float(resolution_m),
        "tile_count": len(local), "unmatched_grid_rows": unmatched,
        "selected_tiles": selected,
    }



def _adaptive_analysis_resolution(aoi_wgs84: gpd.GeoDataFrame, base_resolution_m: float, dem_cache_cfg: dict):
    """Choose a routing/HAND resolution that keeps large basins memory-safe.

    The expensive pyflwdir terrain step holds several full-size arrays at once.
    For a large basin, a 2 m raster can require hundreds of millions of cells.
    This helper keeps the routing raster below a configurable cell budget while
    preserving the requested 2 m resolution for small basins.
    """
    max_cells = int(dem_cache_cfg.get("max_analysis_cells", 12_000_000))
    steps = dem_cache_cfg.get(
        "adaptive_analysis_resolutions_m",
        [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
    )
    steps = sorted({float(x) for x in steps if float(x) > 0})
    base = float(base_resolution_m)
    if base not in steps:
        steps.append(base)
        steps = sorted(set(steps))

    # Houston-area datasets are projected safely into UTM 15N for estimating
    # the raster bounding-box size. Raster memory depends on bbox cells, not
    # polygon area alone, so use total bounds rather than geometry.area.
    aa = aoi_wgs84.to_crs(26915)
    minx, miny, maxx, maxy = map(float, aa.total_bounds)
    width_m = max(1.0, maxx - minx)
    height_m = max(1.0, maxy - miny)
    bbox_area_m2 = width_m * height_m
    required = max(base, (bbox_area_m2 / max(max_cells, 1)) ** 0.5)
    chosen = next((r for r in steps if r >= required), steps[-1])
    est_cells = int((width_m / chosen) * (height_m / chosen))
    return {
        "base_resolution_m": base,
        "chosen_resolution_m": float(chosen),
        "estimated_bbox_cells": int(est_cells),
        "max_analysis_cells": int(max_cells),
        "bbox_width_km": width_m / 1000.0,
        "bbox_height_km": height_m / 1000.0,
        "adaptive": bool(chosen > base + 1e-9),
    }


def _resample_dem_in_place(path: str | Path, target_resolution_m: float):
    """Stream-resample a DEM in place without loading the source raster fully."""
    path = Path(path)
    tmp = path.with_name(path.stem + f"__{float(target_resolution_m):g}m_tmp.tif")
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        width = max(1, int(round((right - left) / float(target_resolution_m))))
        height = max(1, int(round((top - bottom) / float(target_resolution_m))))
        transform = from_bounds(left, bottom, right, top, width, height)
        profile = src.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=transform,
            dtype="float32",
            count=1,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(tmp, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=src.crs,
                dst_nodata=src.nodata,
                resampling=Resampling.bilinear,
            )
    tmp.replace(path)
    return path


def coarsen_analysis_dem(path: str | Path, factor: float = 1.5, max_resolution_m: float = 30.0):
    """Emergency memory fallback used when pyflwdir still reports allocation pressure."""
    path = Path(path)
    with rasterio.open(path) as src:
        current = max(abs(float(src.transform.a)), abs(float(src.transform.e)))
    target = min(float(max_resolution_m), max(current + 1.0, current * float(factor)))
    # Round up to a practical whole-metre working grid.
    import math
    target = float(math.ceil(target))
    if target <= current + 1e-9:
        raise RuntimeError(f"Cannot coarsen routing DEM beyond {current:.1f} m safely.")
    _resample_dem_in_place(path, target)
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "resolution_m": target,
            "width": int(src.width),
            "height": int(src.height),
            "cells": int(src.width * src.height),
            "mode": "memory_fallback_resample",
        }

def ensure_analysis_dem(aoi_wgs84: gpd.GeoDataFrame, workspace_root: str | Path,
                        dem_cfg: dict, dem_cache_cfg: dict):
    """Prepare a memory-safe routing/HAND DEM.

    Small basins retain the requested analysis resolution (normally 2 m).
    Large basins automatically use a coarser working grid so the routing step
    cannot exhaust RAM. The raw 1 m LiDAR tile cache is preserved and the later
    bank/cross-section crop still uses the configured 1 m hydraulic DEM.
    """
    ws = Path(workspace_root)
    out = ws / "data" / "processed" / "dem_analysis_2m.tif"
    base = float(dem_cfg.get("analysis_resolution_m", 2.0))
    adaptive = _adaptive_analysis_resolution(aoi_wgs84, base, dem_cache_cfg)
    target = float(adaptive["chosen_resolution_m"])

    # Existing failed-workspace crops can be reused without downloading again.
    # If the old 2 m crop is too large, downsample it in a streaming operation.
    if out.exists():
        with rasterio.open(out) as src:
            current_res = max(abs(float(src.transform.a)), abs(float(src.transform.e)))
            current_cells = int(src.width * src.height)
        max_cells = int(adaptive["max_analysis_cells"])
        if current_cells > max_cells or current_res + 1e-9 < target:
            _resample_dem_in_place(out, max(target, current_res))
            with rasterio.open(out) as src:
                info = {
                    "mode": "workspace_cache_adaptive_resample",
                    "path": str(out),
                    "resolution_m": max(abs(float(src.transform.a)), abs(float(src.transform.e))),
                    "width": int(src.width),
                    "height": int(src.height),
                    "cells": int(src.width * src.height),
                    **adaptive,
                }
            return info
        with rasterio.open(out) as src2:
            return {
                "mode": "workspace_cache",
                "path": str(out),
                "resolution_m": current_res,
                "width": int(src2.width),
                "height": int(src2.height),
                "cells": current_cells,
                **adaptive,
            }

    info = ensure_dem_for_aoi(aoi_wgs84, out, target, dem_cfg, dem_cache_cfg)
    selected = info.pop("selected_tiles", None)
    if selected is not None and not selected.empty:
        selected.to_crs(4326).to_file(
            ws / "data" / "processed" / "selected_dem_tiles.geojson", driver="GeoJSON"
        )
    with rasterio.open(out) as src:
        info.update({
            "width": int(src.width),
            "height": int(src.height),
            "cells": int(src.width * src.height),
        })
    info.update(adaptive)
    info["resolution_m"] = target
    return info

def ensure_hydraulic_dem(aoi_wgs84: gpd.GeoDataFrame, workspace_root: str | Path,
                         dem_cfg: dict, dem_cache_cfg: dict):
    ws = Path(workspace_root)
    out = ws / "data" / "processed" / "dem_hydraulic_1m.tif"
    info = ensure_dem_for_aoi(
        aoi_wgs84, out, float(dem_cfg.get("hydraulic_resolution_m", 1.0)),
        dem_cfg, dem_cache_cfg,
    )
    info.pop("selected_tiles", None)
    return info


def ensure_dem_products(aoi_wgs84: gpd.GeoDataFrame, workspace_root: str | Path,
                        dem_cfg: dict, dem_cache_cfg: dict):
    """Backward-compatible full-AOI 1-m + 2-m DEM preparation."""
    a = ensure_analysis_dem(aoi_wgs84, workspace_root, dem_cfg, dem_cache_cfg)
    h = ensure_hydraulic_dem(aoi_wgs84, workspace_root, dem_cfg, dem_cache_cfg)
    return {
        "mode": h.get("mode") if h.get("mode") == a.get("mode") else f"{a.get('mode')}+{h.get('mode')}",
        "hydraulic": h["path"], "analysis": a["path"],
        "analysis_info": a, "hydraulic_info": h,
    }

def build_virtual_mosaic(tile_dir: str | Path, out_vrt: str | Path):
    """Build a GDAL VRT from cached DEM tiles if gdalbuildvrt is available."""
    import subprocess
    tile_dir = Path(tile_dir); out_vrt = Path(out_vrt)
    tiles = sorted([*tile_dir.glob("*.tif"), *tile_dir.glob("*.tiff")])
    if not tiles:
        raise RuntimeError("No cached TIFFs found for VRT creation.")
    exe = shutil.which("gdalbuildvrt")
    if not exe:
        raise RuntimeError("gdalbuildvrt was not found on PATH. ArcGIS Pro/OSGeo environments often provide it.")
    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, str(out_vrt), *map(str, tiles)], check=True)
    return out_vrt


def build_arcpy_mosaic_dataset(tile_dir: str | Path, gdb_path: str | Path, mosaic_name="HGAC_DEM_CACHE"):
    """Optional ArcPy-backed mosaic dataset for organizations standardizing on ArcGIS Pro."""
    try:
        import arcpy
    except Exception as exc:
        raise RuntimeError("ArcPy is not available in this Python environment.") from exc
    tile_dir = Path(tile_dir); gdb_path = Path(gdb_path)
    if not gdb_path.exists():
        arcpy.management.CreateFileGDB(str(gdb_path.parent), gdb_path.name)
    md = str(gdb_path / mosaic_name)
    if not arcpy.Exists(md):
        sample = next(iter(tile_dir.glob("*.tif")), None)
        if sample is None:
            raise RuntimeError("No DEM TIFFs in tile cache.")
        sr = arcpy.Describe(str(sample)).spatialReference
        arcpy.management.CreateMosaicDataset(str(gdb_path), mosaic_name, sr)
    arcpy.management.AddRastersToMosaicDataset(md, "Raster Dataset", str(tile_dir), update_cellsize_ranges="UPDATE_CELL_SIZES")
    arcpy.management.BuildOverviews(md)
    return md
