from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

M2_PER_SQMI = 2_589_988.110336
FT_TO_M = 0.3048


def _read_vector(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise RuntimeError(f"Reference vector contains no features: {path}")
    if gdf.crs is None:
        raise RuntimeError(f"Reference vector has no CRS: {path}")
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def _vector_mask(path, crs, transform, shape):
    gdf = _read_vector(path).to_crs(crs)
    geom = unary_union(list(gdf.geometry))
    return geometry_mask([mapping(geom)], transform=transform, invert=True, out_shape=shape)


def _reference_mask(reference_path, model_src, reference_depth_threshold_m=0.05):
    reference_path = Path(reference_path)
    if reference_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(reference_path) as ref:
            src_arr = ref.read(1).astype("float32")
            src_valid = np.isfinite(src_arr)
            if ref.nodata is not None:
                src_valid &= src_arr != ref.nodata
            src_binary = (src_valid & (src_arr >= float(reference_depth_threshold_m))).astype("uint8")
            dst = np.zeros((model_src.height, model_src.width), dtype="uint8")
            reproject(
                source=src_binary,
                destination=dst,
                src_transform=ref.transform,
                src_crs=ref.crs,
                src_nodata=0,
                dst_transform=model_src.transform,
                dst_crs=model_src.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            return dst > 0
    gdf = _read_vector(reference_path).to_crs(model_src.crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize(shapes, out_shape=(model_src.height, model_src.width), transform=model_src.transform, fill=0, dtype="uint8") > 0


def validate_extent(
    model_depth_raster: str | Path,
    reference_extent: str | Path,
    output_dir: str | Path,
    event_name: str = "historical_event",
    model_depth_threshold_m: float = 0.05,
    reference_depth_threshold_m: float = 0.05,
    evaluation_aoi: str | Path | None = None,
    exclude_mask_raster: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(model_depth_raster) as src:
        depth = src.read(1).astype("float64")
        model_valid = np.isfinite(depth)
        if src.nodata is not None:
            model_valid &= depth != src.nodata
        model = model_valid & (depth >= float(model_depth_threshold_m))
        ref = _reference_mask(reference_extent, src, reference_depth_threshold_m=reference_depth_threshold_m)
        eval_mask = model_valid.copy()

        if evaluation_aoi:
            eval_mask &= _vector_mask(evaluation_aoi, src.crs, src.transform, (src.height, src.width))

        if exclude_mask_raster and Path(exclude_mask_raster).exists():
            with rasterio.open(exclude_mask_raster) as ex:
                if ex.crs == src.crs and ex.transform.almost_equals(src.transform) and ex.shape == src.shape:
                    excl = ex.read(1) > 0
                else:
                    excl = np.zeros((src.height, src.width), dtype="uint8")
                    reproject(
                        source=(ex.read(1) > 0).astype("uint8"),
                        destination=excl,
                        src_transform=ex.transform,
                        src_crs=ex.crs,
                        dst_transform=src.transform,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest,
                    )
                    excl = excl > 0
                eval_mask &= ~excl

        tp_mask = eval_mask & model & ref
        fp_mask = eval_mask & model & ~ref
        fn_mask = eval_mask & ~model & ref
        tn_mask = eval_mask & ~model & ~ref

        tp, fp, fn, tn = map(int, [tp_mask.sum(), fp_mask.sum(), fn_mask.sum(), tn_mask.sum()])
        pixel_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        csi = tp / (tp + fp + fn + eps)
        specificity = tn / (tn + fp + eps)
        model_area_m2 = float((tp + fp) * pixel_area)
        ref_area_m2 = float((tp + fn) * pixel_area)
        area_bias_ratio = model_area_m2 / ref_area_m2 if ref_area_m2 > 0 else None

        confusion = np.full((src.height, src.width), 255, dtype="uint8")
        confusion[tn_mask] = 0
        confusion[tp_mask] = 1
        confusion[fp_mask] = 2
        confusion[fn_mask] = 3
        conf_path = output_dir / f"{event_name}_extent_confusion.tif"
        profile = src.profile.copy()
        profile.update(dtype="uint8", nodata=255, count=1, compress="deflate")
        with rasterio.open(conf_path, "w", **profile) as dst:
            dst.write(confusion, 1)

    summary = {
        "event_name": event_name,
        "model_depth_raster": str(model_depth_raster),
        "reference_extent": str(reference_extent),
        "evaluation_aoi": str(evaluation_aoi) if evaluation_aoi else None,
        "exclude_mask_raster": str(exclude_mask_raster) if exclude_mask_raster else None,
        "model_depth_threshold_m": float(model_depth_threshold_m),
        "reference_depth_threshold_m": float(reference_depth_threshold_m),
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "critical_success_index_iou": float(csi),
        "specificity": float(specificity),
        "model_area_m2": model_area_m2,
        "reference_area_m2": ref_area_m2,
        "model_area_sqmi": model_area_m2 / M2_PER_SQMI,
        "reference_area_sqmi": ref_area_m2 / M2_PER_SQMI,
        "area_bias_ratio": area_bias_ratio,
        "confusion_raster": str(conf_path),
    }
    (output_dir / f"{event_name}_extent_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_points(path, lat_field="latitude", lon_field="longitude"):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if lat_field not in df.columns or lon_field not in df.columns:
            raise RuntimeError(f"CSV point file needs {lat_field!r} and {lon_field!r} columns.")
        geom = [Point(float(x), float(y)) for x, y in zip(df[lon_field], df[lat_field])]
        return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    return _read_vector(path)


def validate_high_water_marks(
    model_wse_raster: str | Path,
    observed_points: str | Path,
    observed_wse_field: str,
    output_dir: str | Path,
    event_name: str = "historical_event",
    observed_unit: str = "ft",
    lat_field: str = "latitude",
    lon_field: str = "longitude",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pts = _load_points(observed_points, lat_field=lat_field, lon_field=lon_field)
    if observed_wse_field not in pts.columns:
        raise RuntimeError(f"Observed WSE field not found: {observed_wse_field}")

    with rasterio.open(model_wse_raster) as src:
        pts_proj = pts.to_crs(src.crs)
        coords = [(float(g.x), float(g.y)) for g in pts_proj.geometry]
        sampled = np.array([v[0] for v in src.sample(coords)], dtype="float64")
        if src.nodata is not None:
            sampled[sampled == src.nodata] = np.nan

    observed = pd.to_numeric(pts[observed_wse_field], errors="coerce").to_numpy(dtype="float64", copy=True)
    unit = observed_unit.strip().lower()
    if unit in {"ft", "feet", "foot"}:
        observed_m = observed * FT_TO_M
    elif unit in {"m", "meter", "meters"}:
        observed_m = observed
    else:
        raise ValueError("observed_unit must be ft or m")

    valid = np.isfinite(sampled) & np.isfinite(observed_m)
    error_m = sampled - observed_m
    valid_error = error_m[valid]
    mae = float(np.mean(np.abs(valid_error))) if valid_error.size else None
    rmse = float(np.sqrt(np.mean(valid_error ** 2))) if valid_error.size else None
    bias = float(np.mean(valid_error)) if valid_error.size else None

    out = pts.copy()
    out["observed_wse_m_navd88"] = observed_m
    out["modeled_wse_m_navd88"] = sampled
    out["error_m_model_minus_obs"] = error_m
    out.to_crs(4326).to_file(output_dir / f"{event_name}_hwm_validation.geojson", driver="GeoJSON")
    pd.DataFrame(out.drop(columns="geometry")).to_csv(output_dir / f"{event_name}_hwm_validation.csv", index=False)

    summary = {
        "event_name": event_name,
        "model_wse_raster": str(model_wse_raster),
        "observed_points": str(observed_points),
        "observed_wse_field": observed_wse_field,
        "observed_unit": observed_unit,
        "point_count": int(len(out)),
        "valid_comparison_count": int(valid.sum()),
        "mae_m": mae,
        "rmse_m": rmse,
        "bias_m_model_minus_observed": bias,
    }
    (output_dir / f"{event_name}_hwm_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _hydro_df(path, time_col, flow_col, unit):
    df = pd.read_csv(path)
    if time_col not in df.columns or flow_col not in df.columns:
        raise RuntimeError(f"{path} must contain {time_col!r} and {flow_col!r}")
    out = pd.DataFrame({
        "time": pd.to_datetime(df[time_col], utc=True, errors="coerce"),
        "flow": pd.to_numeric(df[flow_col], errors="coerce"),
    }).dropna()
    u = unit.lower()
    if u in {"cms", "m3/s", "m^3/s"}:
        out["flow"] *= 35.3146667215
    elif u not in {"cfs", "ft3/s", "ft^3/s"}:
        raise ValueError(f"Unsupported flow unit: {unit}")
    return out.sort_values("time").drop_duplicates("time")


def validate_hydrograph(
    modeled_csv: str | Path,
    observed_csv: str | Path,
    output_dir: str | Path,
    event_name: str = "historical_event",
    modeled_time_col: str = "time",
    modeled_flow_col: str = "flow_cfs",
    observed_time_col: str = "time",
    observed_flow_col: str = "flow_cfs",
    modeled_unit: str = "cfs",
    observed_unit: str = "cfs",
    tolerance_minutes: float = 30.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mod = _hydro_df(modeled_csv, modeled_time_col, modeled_flow_col, modeled_unit).rename(columns={"flow": "modeled_cfs"})
    obs = _hydro_df(observed_csv, observed_time_col, observed_flow_col, observed_unit).rename(columns={"flow": "observed_cfs"})
    merged = pd.merge_asof(
        obs.sort_values("time"), mod.sort_values("time"), on="time", direction="nearest",
        tolerance=pd.Timedelta(minutes=float(tolerance_minutes)),
    ).dropna(subset=["modeled_cfs", "observed_cfs"])
    if merged.empty:
        raise RuntimeError("No modeled/observed hydrograph pairs matched within the requested time tolerance.")

    o = merged["observed_cfs"].to_numpy(dtype=float, copy=True)
    m = merged["modeled_cfs"].to_numpy(dtype=float, copy=True)
    err = m - o
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    pbias = float(100.0 * np.sum(err) / np.sum(o)) if abs(np.sum(o)) > 1e-9 else None
    corr = float(np.corrcoef(m, o)[0, 1]) if len(m) > 1 and np.std(m) > 0 and np.std(o) > 0 else None
    nse_den = float(np.sum((o - np.mean(o)) ** 2))
    nse = float(1.0 - np.sum((m - o) ** 2) / nse_den) if nse_den > 0 else None
    if corr is not None and np.mean(o) != 0 and np.std(o) != 0:
        alpha = float(np.std(m) / np.std(o))
        beta = float(np.mean(m) / np.mean(o))
        kge = float(1.0 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        kge = None

    obs_peak_i = int(np.argmax(o))
    mod_peak_i = int(np.argmax(m))
    obs_peak_time = pd.to_datetime(merged.iloc[obs_peak_i]["time"], utc=True)
    mod_peak_time = pd.to_datetime(merged.iloc[mod_peak_i]["time"], utc=True)
    peak_timing_error_hours = float((mod_peak_time - obs_peak_time).total_seconds() / 3600.0)

    merged["error_cfs_model_minus_obs"] = err
    merged.to_csv(output_dir / f"{event_name}_hydrograph_pairs.csv", index=False)
    summary = {
        "event_name": event_name,
        "matched_points": int(len(merged)),
        "rmse_cfs": rmse,
        "mae_cfs": mae,
        "bias_cfs_model_minus_observed": bias,
        "percent_bias": pbias,
        "pearson_r": corr,
        "nash_sutcliffe_efficiency": nse,
        "kling_gupta_efficiency": kge,
        "observed_peak_cfs": float(np.max(o)),
        "modeled_peak_cfs": float(np.max(m)),
        "peak_ratio_modeled_to_observed": float(np.max(m) / np.max(o)) if np.max(o) > 0 else None,
        "observed_peak_time_utc": obs_peak_time.isoformat(),
        "modeled_peak_time_utc": mod_peak_time.isoformat(),
        "peak_timing_error_hours_model_minus_observed": peak_timing_error_hours,
    }
    (output_dir / f"{event_name}_hydrograph_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
