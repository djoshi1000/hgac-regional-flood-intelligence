from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import Point, mapping, box
from shapely.ops import unary_union

FLOAT_NODATA = -9999.0
M2_PER_SQMI = 2_589_988.110336


def _read_vector(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise RuntimeError(f"Vector file contains no features: {path}")
    if gdf.crs is None:
        raise RuntimeError(f"Vector file has no CRS: {path}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise RuntimeError(f"Vector file contains no usable geometry: {path}")
    return gdf


def _union_geometry(gdf: gpd.GeoDataFrame):
    return unary_union(list(gdf.geometry))


def prepare_aoi_context(
    aoi_path: str | Path,
    dem_path: str | Path,
    gauge_lon: float,
    gauge_lat: float,
    basin_path: str | Path | None = None,
    mainstem_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    context_buffer_m: float = 1000.0,
    mainstem_corridor_buffer_m: float = 250.0,
):
    """Prepare an AOI/display-domain QA package.

    The hydraulic model remains gauge/reach based. The AOI is a *display and
    evaluation clip*, so the gauge does not need to fall inside it. The DEM/model
    domain still has to include the hydraulic control and the relevant connected
    river reach.
    """
    aoi_path = Path(aoi_path)
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_bounds = src.bounds
        dem_geom = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)

    aoi = _read_vector(aoi_path).to_crs(dem_crs)
    aoi_geom = _union_geometry(aoi)
    if aoi_geom.is_empty:
        raise RuntimeError("AOI geometry is empty after reprojection.")

    to_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    gx, gy = to_dem.transform(float(gauge_lon), float(gauge_lat))
    gauge_point = Point(gx, gy)

    gauge_inside_aoi = bool(aoi_geom.covers(gauge_point))
    gauge_distance_m = float(aoi_geom.distance(gauge_point))
    gauge_inside_dem = bool(dem_geom.covers(gauge_point))
    aoi_inside_dem_fraction = (
        float(aoi_geom.intersection(dem_geom).area / aoi_geom.area)
        if aoi_geom.area > 0 else 0.0
    )

    basin_overlap_fraction = None
    basin_geom = None
    if basin_path and Path(basin_path).exists():
        basin = _read_vector(basin_path).to_crs(dem_crs)
        basin_geom = _union_geometry(basin)
        if aoi_geom.area > 0:
            basin_overlap_fraction = float(aoi_geom.intersection(basin_geom).area / aoi_geom.area)

    mainstem_intersects_aoi = None
    mainstem_geom = None
    if mainstem_path and Path(mainstem_path).exists():
        mainstem = _read_vector(mainstem_path).to_crs(dem_crs)
        mainstem_geom = _union_geometry(mainstem)
        mainstem_intersects_aoi = bool(mainstem_geom.intersects(aoi_geom))

    context_parts = [aoi_geom.buffer(max(float(context_buffer_m), 0.0)), gauge_point.buffer(max(float(context_buffer_m), 0.0))]
    if mainstem_geom is not None:
        context_parts.append(mainstem_geom.buffer(max(float(mainstem_corridor_buffer_m), 0.0)))
    context_geom = unary_union(context_parts)

    warnings: list[str] = []
    if not gauge_inside_dem:
        warnings.append("gauge_outside_dem_model_domain")
    if aoi_inside_dem_fraction < 0.999:
        warnings.append("aoi_not_fully_covered_by_dem")
    if basin_overlap_fraction is not None and basin_overlap_fraction < 0.50:
        warnings.append("aoi_has_less_than_50pct_overlap_with_gauge_upstream_basin")
    if mainstem_intersects_aoi is False:
        warnings.append("aoi_does_not_intersect_current_mainstem")
    if gauge_distance_m > 0:
        warnings.append("gauge_outside_display_aoi_but_can_be_used_if_same_hydraulic_network")

    status = "supported"
    if not gauge_inside_dem or aoi_inside_dem_fraction < 0.95:
        status = "model_domain_needs_expansion"
    elif basin_overlap_fraction is not None and basin_overlap_fraction < 0.10:
        status = "current_gauge_basin_probably_not_applicable"

    result = {
        "aoi_path": str(aoi_path),
        "dem_path": str(dem_path),
        "status": status,
        "gauge_inside_aoi": gauge_inside_aoi,
        "gauge_distance_to_aoi_m": gauge_distance_m,
        "gauge_inside_dem": gauge_inside_dem,
        "aoi_dem_coverage_fraction": aoi_inside_dem_fraction,
        "aoi_overlap_with_current_upstream_basin_fraction": basin_overlap_fraction,
        "aoi_intersects_current_mainstem": mainstem_intersects_aoi,
        "context_buffer_m": float(context_buffer_m),
        "mainstem_corridor_buffer_m": float(mainstem_corridor_buffer_m),
        "warnings": warnings,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({"name": ["Display AOI"]}, geometry=[aoi_geom], crs=dem_crs).to_crs(4326).to_file(
            output_dir / "aoi_display.geojson", driver="GeoJSON"
        )
        gpd.GeoDataFrame({"name": ["Suggested hydraulic context domain"]}, geometry=[context_geom], crs=dem_crs).to_crs(4326).to_file(
            output_dir / "model_context_domain.geojson", driver="GeoJSON"
        )
        gpd.GeoDataFrame(
            {"name": ["Configured gauge"], "inside_aoi": [gauge_inside_aoi], "distance_to_aoi_m": [gauge_distance_m]},
            geometry=[gauge_point], crs=dem_crs,
        ).to_crs(4326).to_file(output_dir / "configured_gauge.geojson", driver="GeoJSON")
        (output_dir / "aoi_qa.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result.update(
            {
                "aoi_display_geojson": str(output_dir / "aoi_display.geojson"),
                "model_context_domain_geojson": str(output_dir / "model_context_domain.geojson"),
                "configured_gauge_geojson": str(output_dir / "configured_gauge.geojson"),
                "aoi_qa_json": str(output_dir / "aoi_qa.json"),
            }
        )

    return result


def raster_aoi_mask(raster_path: str | Path, aoi_path: str | Path):
    with rasterio.open(raster_path) as src:
        aoi = _read_vector(aoi_path).to_crs(src.crs)
        geom = _union_geometry(aoi)
        mask = geometry_mask([mapping(geom)], transform=src.transform, invert=True, out_shape=(src.height, src.width))
        return mask, src.profile.copy(), src.transform, src.crs


def clip_raster_to_aoi(input_raster: str | Path, aoi_path: str | Path, output_raster: str | Path):
    input_raster = Path(input_raster)
    output_raster = Path(output_raster)
    output_raster.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(input_raster) as src:
        arr = src.read(1)
        aoi = _read_vector(aoi_path).to_crs(src.crs)
        geom = _union_geometry(aoi)
        inside = geometry_mask([mapping(geom)], transform=src.transform, invert=True, out_shape=(src.height, src.width))
        nodata = src.nodata
        if nodata is None:
            nodata = FLOAT_NODATA if np.issubdtype(arr.dtype, np.floating) else 0
        out = arr.copy()
        out[~inside] = nodata
        profile = src.profile.copy()
        # Preserve compression while choosing TIFF block sizes that are legal
        # for both small test rasters and large production DEMs.
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        if src.width >= 16 and src.height >= 16:
            bx = max(16, min(512, (src.width // 16) * 16))
            by = max(16, min(512, (src.height // 16) * 16))
            profile.update(nodata=nodata, compress="deflate", tiled=True, blockxsize=bx, blockysize=by, BIGTIFF="IF_SAFER")
        else:
            profile.update(nodata=nodata, compress="deflate", tiled=False, BIGTIFF="IF_SAFER")
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(out, 1)
    return output_raster


def clip_vector_to_aoi(input_vector: str | Path, aoi_path: str | Path, output_vector: str | Path):
    input_vector = Path(input_vector)
    output_vector = Path(output_vector)
    output_vector.parent.mkdir(parents=True, exist_ok=True)
    if not input_vector.exists():
        return None
    src = _read_vector(input_vector)
    aoi = _read_vector(aoi_path).to_crs(src.crs)
    geom = _union_geometry(aoi)
    clipped = src.copy()
    clipped["geometry"] = clipped.geometry.intersection(geom)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty]
    if clipped.empty:
        output_vector.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    else:
        clipped.to_crs(4326).to_file(output_vector, driver="GeoJSON")
    return output_vector


def summarize_depth_within_aoi(depth_raster: str | Path, aoi_path: str | Path, min_depth_m: float = 0.05):
    with rasterio.open(depth_raster) as src:
        arr = src.read(1).astype("float64")
        aoi = _read_vector(aoi_path).to_crs(src.crs)
        geom = _union_geometry(aoi)
        inside = geometry_mask([mapping(geom)], transform=src.transform, invert=True, out_shape=(src.height, src.width))
        valid = inside & np.isfinite(arr) & (arr >= float(min_depth_m))
        if src.nodata is not None:
            valid &= arr != src.nodata
        pixel_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
        vals = arr[valid]
        return {
            "aoi_wet_pixels": int(valid.sum()),
            "aoi_inundated_area_m2": float(valid.sum() * pixel_area),
            "aoi_inundated_area_sqmi": float(valid.sum() * pixel_area / M2_PER_SQMI),
            "aoi_storage_volume_m3": float(np.sum(vals * pixel_area)) if vals.size else 0.0,
            "aoi_max_depth_m": float(vals.max()) if vals.size else 0.0,
            "aoi_mean_depth_m": float(vals.mean()) if vals.size else 0.0,
        }


def postprocess_scenario_to_aoi(
    aoi_path: str | Path,
    depth_raster: str | Path,
    extent_geojson: str | Path,
    potential_depth_raster: str | Path | None,
    potential_extent_geojson: str | Path | None,
    outputs_dir: str | Path,
    min_depth_m: float = 0.05,
):
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    depth_aoi = outputs_dir / "latest_depth_aoi.tif"
    extent_aoi = outputs_dir / "latest_inundation_aoi.geojson"
    clip_raster_to_aoi(depth_raster, aoi_path, depth_aoi)
    clip_vector_to_aoi(extent_geojson, aoi_path, extent_aoi)

    result = summarize_depth_within_aoi(depth_raster, aoi_path, min_depth_m=min_depth_m)
    result.update({"aoi_depth_raster": str(depth_aoi), "aoi_extent_geojson": str(extent_aoi)})

    if potential_depth_raster and Path(potential_depth_raster).exists():
        pdepth = outputs_dir / "latest_potential_depth_aoi.tif"
        clip_raster_to_aoi(potential_depth_raster, aoi_path, pdepth)
        result["aoi_potential_depth_raster"] = str(pdepth)
    if potential_extent_geojson and Path(potential_extent_geojson).exists():
        pext = outputs_dir / "latest_potential_inundation_aoi.geojson"
        clip_vector_to_aoi(potential_extent_geojson, aoi_path, pext)
        result["aoi_potential_extent_geojson"] = str(pext)

    return result
