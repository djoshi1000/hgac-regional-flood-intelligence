from __future__ import annotations

from pathlib import Path
import json
import shutil
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import load_config, ensure_directories
from .nldi import get_basin, get_comid
from .dem_cache import ensure_analysis_dem, ensure_hydraulic_dem, coarsen_analysis_dem
from .terrain import build_corrected_terrain
from .bank_profile import build_bank_profile, write_gauge_cross_section_products
from .usgs import fetch_iv, latest_observation
from .usgs_rating import fetch_usgs_rating, make_absolute_rating, datum_candidate_from_monitoring_metadata
from .rating_curve import RatingCurve
from .nwps import get_reach_streamflow, normalize_streamflow
from .inundation import make_v4_discharge_capacity_inundation
from .aoi import prepare_aoi_context, postprocess_scenario_to_aoi


def _emit(cb, step, message, status="running", extra=None):
    payload = {"step": step, "message": message, "status": status, "time": time.time()}
    if extra:
        payload.update(extra)
    if cb:
        cb(payload)
    return payload


def _state(ws):
    p = Path(ws) / "workspace_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_pipeline_state(ws, data):
    p = Path(ws) / "pipeline_state.json"
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return p


def static_ready(ws):
    p = Path(ws) / "data" / "processed"
    required = [
        p / "dem_analysis_2m.tif", p / "dem_hydraulic_1m.tif",
        p / "terrain_qa.json", p / "bank_profile.csv", p / "rating_curve.csv",
    ]
    return all(x.exists() for x in required)


def _rating_curve_for_workspace(cfg, ws, project_root, terrain_result, gauge, progress=None):
    processed = Path(ws) / "data" / "processed"
    out = processed / "rating_curve.csv"
    qa_path = processed / "rating_curve_qa.json"
    if out.exists() and qa_path.exists():
        return {"rating_curve_csv": str(out), **json.loads(qa_path.read_text(encoding="utf-8"))}

    site = str(cfg["pilot"]["usgs_site"])
    # Preserve the already-vetted Hunting Bayou rating curve when the regional
    # app selects the same pilot gauge. This avoids degrading a known-good pilot.
    try:
        root_cfg = load_config(Path(project_root) / "config" / "config.yaml")
        root_site = str(root_cfg.get("pilot", {}).get("usgs_site", ""))
        root_rating = Path(root_cfg.get("paths", {}).get("rating_curve_csv", ""))
        if root_site == site and root_rating.exists():
            shutil.copy2(root_rating, out)
            qa = {"source": "existing_vetted_project_rating", "absolute_navd88": True, "datum_status": "vetted_existing_pilot"}
            qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
            return {"rating_curve_csv": str(out), **qa}
    except Exception:
        pass

    raw_rating = fetch_usgs_rating(site, processed / "rating_curve_usgs_gage_height.csv")
    override = cfg.get("rating", {}).get("gage_datum_navd88_ft_override")
    if override not in (None, ""):
        datum = float(override)
        datum_status = "manual_navd88_override"
    else:
        datum, datum_status = datum_candidate_from_monitoring_metadata(gauge)
    if datum is None:
        raise RuntimeError(
            "A stage-discharge rating was found, but the gage-height datum cannot be automatically tied to NAVD88. "
            f"Status: {datum_status}. Set rating.gage_datum_navd88_ft_override in this workspace config after verifying the gage datum."
        )

    # Independent sanity check: current absolute WSE should be reasonably close
    # to the LiDAR channel-surface proxy. This catches land-surface altitude being
    # mistaken for a gage datum.
    iv = fetch_iv(site, cfg.get("usgs", {}).get("iv_period", "P2D"),
                  cache_path=processed / "usgs_iv_cache.csv", allow_legacy_fallback=True)
    obs = latest_observation(iv) or {}
    current_stage = obs.get("stage_ft")
    channel_ft = float(terrain_result["gauge_channel_elevation_ft_navd88"])
    consistency = None
    if current_stage is not None:
        current_wse = float(datum) + float(current_stage)
        consistency = current_wse - channel_ft
        tol = float(cfg.get("rating", {}).get("datum_consistency_tolerance_ft", 10.0))
        if abs(consistency) > tol:
            raise RuntimeError(
                "Automatic gage-datum QA failed. The metadata-based NAVD88 datum plus current USGS gage height "
                f"places water {consistency:+.1f} ft relative to the LiDAR channel-surface proxy. "
                "Verify the gage datum and set rating.gage_datum_navd88_ft_override before inundation modeling."
            )
    make_absolute_rating(raw_rating, datum, out_csv=out)
    qa = {
        "source": "USGS_Water_Data_STAC_rating",
        "absolute_navd88": True,
        "gage_datum_navd88_ft": float(datum),
        "datum_status": datum_status,
        "current_stage_vs_lidar_channel_ft": consistency,
    }
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"rating_curve_csv": str(out), **qa}


def prepare_static_workspace(project_root, workspace_root, config_path, regional_cfg, force=False, progress=None):
    ws = Path(workspace_root)
    cfg = load_config(config_path)
    ensure_directories(cfg)
    processed = ws / "data" / "processed"
    outputs = ws / "outputs"
    state = _state(ws)
    gauge = state.get("gauge", {})
    logs = []

    if static_ready(ws) and not force:
        logs.append(_emit(progress, "CACHE", "Static bayou model already prepared; reusing DEM/terrain/banks/rating.", "cached"))
        return {"status": "cached", "workspace": str(ws), "config": str(config_path)}

    # 00 — basin/domain
    logs.append(_emit(progress, "00", "Resolving hydraulically relevant upstream basin from USGS NLDI."))
    basin_path = Path(cfg["paths"]["basin_geojson"])
    if force or not basin_path.exists():
        basin = get_basin(cfg["pilot"]["usgs_site"], simplified=False, split_catchment=False).to_crs(4326)
        basin_path.parent.mkdir(parents=True, exist_ok=True)
        basin.to_file(basin_path, driver="GeoJSON")
    else:
        basin = gpd.read_file(basin_path).to_crs(4326)
    logs.append(_emit(progress, "00", "Basin/domain ready.", "done"))

    # 01 — DEM cache/crop
    logs.append(_emit(progress, "01", "Preparing DEM from regional/tile cache. Missing tiles are downloaded once and retained."))
    dem_cache_cfg = dict(regional_cfg.get("dem_cache", {}))
    cache_root = dem_cache_cfg.get("cache_root", "data/regional_cache/dem")
    cache_root = Path(cache_root)
    if not cache_root.is_absolute():
        cache_root = Path(project_root) / cache_root
    dem_cache_cfg["cache_root"] = str(cache_root)
    regional_dem = str(dem_cache_cfg.get("regional_dem_path", "") or "").strip()
    if regional_dem and not Path(regional_dem).is_absolute():
        dem_cache_cfg["regional_dem_path"] = str((Path(project_root) / regional_dem).resolve())
    analysis_info = ensure_analysis_dem(basin, ws, cfg.get("dem", {}), dem_cache_cfg)
    
    if analysis_info.get("adaptive"):
        logs.append(_emit(
            progress, "01",
            f"Large basin detected. Routing/HAND DEM automatically set to {analysis_info.get('resolution_m', analysis_info.get('chosen_resolution_m')):.1f} m to protect memory; cached 1 m LiDAR is still used later for bank cross-sections.",
            "done", analysis_info,
        ))
    else:
        logs.append(_emit(progress, "01", f"Analysis DEM ready using {analysis_info.get('mode')} at {analysis_info.get('resolution_m', 2.0):.1f} m.", "done", analysis_info))

    # 02 — terrain
    logs.append(_emit(progress, "02", "Conditioning routing DEM and deriving basin, upstream area, mainstem, HAND and QA."))
    terrain_qa_path = processed / "terrain_qa.json"
    terrain_cached = (
        terrain_qa_path.exists() and (processed / "mainstem_mask.tif").exists()
        and (processed / "hand_2m.tif").exists() and not force
    )
    if terrain_cached:
        terrain_result = json.loads(terrain_qa_path.read_text(encoding="utf-8"))
        logs.append(_emit(progress, "02", "Terrain/HAND/mainstem cache found; reusing static terrain products.", "cached"))
    else:
        terrain_kwargs = dict(
            dem_path=cfg["paths"]["dem_analysis"],
            gauge_lon=float(cfg["pilot"]["gauge_lon"]),
            gauge_lat=float(cfg["pilot"]["gauge_lat"]),
            processed_dir=processed,
            stream_threshold_km2=float(cfg["terrain"].get("stream_threshold_km2", 5.0)),
            mainstem_threshold_km2=float(cfg["terrain"].get("mainstem_threshold_km2", 8.0)),
            expected_drainage_area_sqmi=(
                float(cfg["pilot"].get("expected_drainage_area_sqmi"))
                if float(cfg["pilot"].get("expected_drainage_area_sqmi", 0.0) or 0.0) > 0
                else None
            ),
            gauge_snap_max_m=float(cfg["terrain"].get("gauge_snap_max_m", 250.0)),
            selected_lon=cfg.get("selection", {}).get("click_lon"),
            selected_lat=cfg.get("selection", {}).get("click_lat"),
            selected_snap_max_m=float(cfg.get("selection", {}).get("selected_stream_snap_max_m", 2000.0)),
        )
        terrain_result = None
        for attempt in range(3):
            try:
                terrain_result = build_corrected_terrain(**terrain_kwargs)
                break
            except Exception as exc:
                msg = str(exc).lower()
                memory_pressure = isinstance(exc, MemoryError) or (
                    "allocation failed" in msg
                    or "unable to allocate" in msg
                    or "out of memory" in msg
                    or "memoryerror" in msg
                )
                if not memory_pressure or attempt >= 2:
                    raise
                import gc
                gc.collect()
                fallback = coarsen_analysis_dem(
                    cfg["paths"]["dem_analysis"],
                    factor=1.6,
                    max_resolution_m=float(dem_cache_cfg.get("max_fallback_resolution_m", 30.0)),
                )
                terrain_kwargs["dem_path"] = cfg["paths"]["dem_analysis"]
                logs.append(_emit(
                    progress, "02",
                    f"Memory pressure detected during terrain routing. No DEM re-download is needed; automatically retrying at {fallback['resolution_m']:.0f} m working resolution ({fallback['cells']:,} cells).",
                    "running", fallback,
                ))
        if terrain_result is None:
            raise RuntimeError("Terrain preparation did not complete after memory-safe retries.")
    # V5.2: terrain routing now traces the branch the user actually selected.
    # The selected point is snapped to the DEM stream network and connected
    # downstream to the chosen gauge before HAND/bank work is created.
    click_to_mainstem_m = terrain_result.get("selected_click_to_mainstem_m")
    mainstem_raster_path = processed / "mainstem_mask.tif"
    import rasterio
    from rasterio.features import shapes as raster_shapes
    from shapely.geometry import shape as shp_shape

    with rasterio.open(mainstem_raster_path) as ms_src:
        ms = ms_src.read(1) > 0
        rr, cc = np.where(ms)
        if len(rr) == 0:
            raise RuntimeError("Derived selected-bayou mainstem raster is empty.")

        # Polygonize one-cell mainstem pixels, dissolve, then buffer.  This gives
        # a lightweight 1-m LiDAR crop corridor without first building the bank
        # profile or keeping the whole basin at 1 m.
        geoms = [
            shp_shape(g) for g, value in raster_shapes(
                ms.astype("uint8"), mask=ms, transform=ms_src.transform
            ) if int(value) == 1
        ]
        if not geoms:
            raise RuntimeError("Could not vectorize the derived mainstem for the hydraulic DEM corridor.")
        mainstem_pixels = gpd.GeoDataFrame(geometry=geoms, crs=ms_src.crs)

    buffer_m = float(
        regional_cfg.get("dem_cache", {}).get(
            "hydraulic_corridor_buffer_m",
            float(cfg.get("bank_profile", {}).get("half_width_m", 150.0)) + 75.0,
        )
    )
    corridor_geom = mainstem_pixels.geometry.union_all().buffer(buffer_m)
    hydraulic_aoi = gpd.GeoDataFrame(geometry=[corridor_geom], crs=mainstem_pixels.crs).to_crs(4326)
    hydraulic_info = ensure_hydraulic_dem(
        hydraulic_aoi, ws, cfg.get("dem", {}), dem_cache_cfg
    )
    dem_info = {
        "analysis": analysis_info.get("path"),
        "hydraulic": hydraulic_info.get("path"),
        "analysis_info": analysis_info,
        "hydraulic_info": hydraulic_info,
        "hydraulic_corridor_buffer_m": buffer_m,
    }
    logs.append(_emit(progress, "02", "Terrain/HAND/mainstem and 1-m hydraulic corridor DEM ready.", "done", {"click_to_mainstem_m": click_to_mainstem_m, **dem_info}))

    # 03 — NWM reach discovery
    logs.append(_emit(progress, "03", "Resolving NHDPlus/NWM reach from selected USGS control gauge."))
    comid = get_comid(cfg["pilot"]["usgs_site"])
    if not comid:
        raise RuntimeError("Could not resolve an NHDPlus COMID/NWM reach for the selected control gauge.")
    logs.append(_emit(progress, "03", f"NWM reach resolved: {comid}.", "done"))

    # 04 — rating/datum
    logs.append(_emit(progress, "04", "Preparing stage-discharge rating and vertical-datum QA."))
    rating_info = _rating_curve_for_workspace(cfg, ws, project_root, terrain_result, gauge, progress)
    logs.append(_emit(progress, "04", f"Rating curve ready ({rating_info.get('source')}).", "done"))

    # 05 — bank/hydraulic profile
    logs.append(_emit(progress, "05", "Extracting LiDAR cross sections, local bank profile and relative hydraulic capacity."))
    bp = cfg.get("bank_profile", {})
    bank_cached = (processed / "bank_profile.csv").exists() and (processed / "bank_profile_summary.json").exists() and not force
    if bank_cached:
        bank_summary_cached = json.loads((processed / "bank_profile_summary.json").read_text(encoding="utf-8"))
        bank_result = {
            "section_count": int(bank_summary_cached.get("section_count", bank_summary_cached.get("cross_sections", 0))),
            "cached": True,
        }
        logs.append(_emit(progress, "05", "LiDAR bank/hydraulic profile cache found; reusing cross sections.", "cached"))
    else:
        bank_result = build_bank_profile(
            hydraulic_dem_path=cfg["paths"]["dem_hydraulic"],
            conditioned_dem_path=str(processed / "dem_conditioned_2m.tif"),
            flowdir_path=str(processed / "flowdir_d8.tif"),
            mainstem_path=str(processed / "mainstem_mask.tif"),
            basin_mask_path=str(processed / "dem_upstream_basin_mask.tif"),
            terrain_qa_path=str(processed / "terrain_qa.json"),
            upstream_area_path=str(processed / "upstream_area_km2.tif"),
            processed_dir=processed,
            section_spacing_m=float(bp.get("section_spacing_m", 150.0)),
            half_width_m=float(bp.get("half_width_m", 150.0)),
            sample_spacing_m=float(bp.get("sample_spacing_m", 1.0)),
            tangent_window_m=float(bp.get("tangent_window_m", 30.0)),
            bank_search_min_m=float(bp.get("bank_search_min_m", 8.0)),
            bank_search_max_m=float(bp.get("bank_search_max_m", 65.0)),
            min_relief_ft=float(bp.get("bank_min_relief_ft", 6.5)),
            peak_prominence_ft=float(bp.get("peak_prominence_ft", 0.35)),
            outlier_threshold_ft=float(bp.get("outlier_threshold_ft", 4.0)),
            seed_offset_m=float(bp.get("landward_seed_offset_m", 4.0)),
            min_channel_slope=float(bp.get("min_channel_slope", 1.0e-5)),
            max_channel_slope=float(bp.get("max_channel_slope", 0.02)),
        )
        write_gauge_cross_section_products(
            hydraulic_dem_path=cfg["paths"]["dem_hydraulic"],
            bank_profile_csv=bank_result["paths"]["bank_profile_csv"],
            processed_dir=processed,
            outputs_dir=outputs,
        )
        logs.append(_emit(progress, "05", f"Bank profile ready: {bank_result['section_count']} sections.", "done"))

    result = {
        "status": "prepared",
        "workspace": str(ws), "config": str(config_path), "nwm_reach": str(comid),
        "dem": dem_info, "terrain": terrain_result, "rating": rating_info,
        "bank_profile": {k: v for k, v in bank_result.items() if k != "paths"},
        "logs": logs,
    }
    _save_pipeline_state(ws, result)
    return result


def run_live_workspace(workspace_root, config_path, progress=None):
    ws = Path(workspace_root)
    cfg = load_config(config_path)
    processed = ws / "data" / "processed"
    outputs = ws / "outputs"; outputs.mkdir(parents=True, exist_ok=True)
    if not static_ready(ws):
        raise RuntimeError("Static workspace is not prepared. Run the full preparation pipeline first.")

    _emit(progress, "06", "Fetching latest USGS observations.")
    usgs = fetch_iv(
        cfg["pilot"]["usgs_site"], cfg.get("usgs", {}).get("iv_period", "P7D"),
        cache_path=processed / "usgs_iv_cache.csv", allow_legacy_fallback=True,
    )
    obs = latest_observation(usgs) or {}
    _emit(progress, "06", "USGS observation refresh complete.", "done")

    _emit(progress, "07", "Fetching NOAA National Water Model short-range hydrograph.")
    reach = cfg.get("nwm", {}).get("reach_id", "auto")
    if reach == "auto":
        reach = get_comid(cfg["pilot"]["usgs_site"])
    nwm = normalize_streamflow(get_reach_streamflow(reach, cfg.get("nwm", {}).get("preferred_series", "short_range")))
    if nwm.empty:
        raise RuntimeError("NOAA NWM returned no usable streamflow rows.")
    unit = str(cfg.get("nwm", {}).get("value_unit", "cfs")).lower()
    hg = nwm[["time", "streamflow"]].copy()
    if unit == "cms":
        hg["streamflow"] *= 35.3146667215
    elif unit != "cfs":
        raise RuntimeError(f"Unsupported NWM flow unit: {unit}")
    hg.to_csv(outputs / "latest_nwm_hydrograph.csv", index=False)
    if not usgs.empty:
        usgs.to_csv(outputs / "latest_usgs_observations.csv", index=False)
    peak_idx = hg["streamflow"].idxmax()
    peak_q = float(hg.loc[peak_idx, "streamflow"])
    peak_time = hg.loc[peak_idx, "time"]
    _emit(progress, "07", f"NWM peak {peak_q:.1f} cfs at {peak_time}.", "done")

    _emit(progress, "08", "Converting forecast discharge to WSE and running volume-limited floodplain screening.")
    curve = RatingCurve.from_csv(processed / "rating_curve.csv")
    stage_ft = float(curve.stage_from_flow(peak_q))
    terrain_qa = json.loads((processed / "terrain_qa.json").read_text(encoding="utf-8"))
    bank_summary = json.loads((processed / "bank_profile_summary.json").read_text(encoding="utf-8"))
    gauge_bank_ft = float(bank_summary["gauge_bank_stage_ft_navd88"])
    gauge_bankfull_q = float(curve.flow_from_stage(gauge_bank_ft))
    hy = cfg.get("hydraulics", {}); incfg = cfg.get("inundation", {}); bp = cfg.get("bank_profile", {})

    result = make_v4_discharge_capacity_inundation(
        dem_path=cfg["paths"]["dem_analysis"],
        hand_path=str(processed / "hand_2m.tif"),
        mainstem_path=str(processed / "mainstem_mask.tif"),
        bank_profile_csv=str(processed / "bank_profile.csv"),
        scenario_stage_ft=stage_ft, scenario_flow_cfs=peak_q,
        gauge_bankfull_flow_cfs=gauge_bankfull_q,
        gauge_channel_elevation_m=float(terrain_qa["gauge_channel_elevation_m_navd88"]),
        gauge_upstream_area_km2=float(terrain_qa["outlet_upstream_area_km2"]),
        depth_out=cfg["paths"]["latest_inundation"], polygon_out=cfg["paths"]["latest_inundation_geojson"],
        potential_depth_out=str(outputs / "latest_potential_depth.tif"),
        potential_polygon_out=str(outputs / "latest_potential_inundation.geojson"),
        channel_corridor_out=str(outputs / "latest_channel_corridor.geojson"),
        channel_mask_out=str(outputs / "latest_channel_corridor_mask.tif"),
        channel_exclusion_buffer_m=float(hy.get("channel_exclusion_buffer_m", 2.0)),
        hydrograph=hg, hydrograph_flow_unit="cfs",
        hydrograph_max_interval_hours=float(hy.get("hydrograph_max_interval_hours", 3.0)),
        min_depth_m=float(cfg["terrain"].get("min_depth_m", 0.05)),
        max_hand_m=float(cfg["terrain"].get("max_hand_m", 12.0)),
        fill_depth_path=str(processed / "dem_fill_depth_2m.tif"),
        wse_out=str(outputs / "latest_wse.tif"),
        overtopped_sections_out=str(outputs / "latest_overtopped_sections.geojson"),
        seed_points_out=str(outputs / "latest_floodplain_seeds.geojson"),
        max_depth_point_out=str(outputs / "latest_max_depth_point.geojson"),
        scenario_profile_out=str(outputs / "latest_bank_wse_profile.csv"),
        seed_mask_out=str(outputs / "latest_seed_mask.tif"),
        connectivity=int(incfg.get("connectivity", 4)),
        min_component_pixels=int(incfg.get("min_component_pixels", 20)),
        min_overtop_ft=float(incfg.get("min_overtop_ft", 0.02)),
        landward_seed_offset_m=float(bp.get("landward_seed_offset_m", 4.0)),
        landward_seed_search_m=float(bp.get("landward_seed_search_m", 20.0)),
        landward_seed_step_m=float(bp.get("landward_seed_step_m", 2.0)),
        fill_depth_warning_m=float(cfg["terrain"].get("fill_hotspot_threshold_m", 1.0)),
        flow_area_exponent=float(hy.get("flow_area_exponent", 0.85)),
        capacity_area_exponent=float(hy.get("capacity_area_exponent", 0.70)),
        contained_stage_exponent=float(hy.get("contained_stage_exponent", 0.60)),
        overbank_excess_exponent=float(hy.get("overbank_excess_exponent", 0.70)),
        capacity_outlier_factor=float(hy.get("capacity_outlier_factor", 3.0)),
        capacity_floor_factor=float(hy.get("capacity_floor_factor", 1.0)),
        capacity_ceiling_factor=float(hy.get("capacity_ceiling_factor", 1.5)),
        max_local_overbank_ft=float(hy.get("max_local_overbank_ft", 12.0)),
        max_local_overbank_multiplier=float(hy.get("max_local_overbank_multiplier", 1.5)),
        max_section_influence_m=float(hy.get("max_section_influence_m", 1200.0)),
        screening_duration_hours=float(hy.get("screening_duration_hours", 6.0)),
        floodplain_volume_fraction=float(hy.get("floodplain_volume_fraction", 1.0)),
        volume_uphill_penalty=float(hy.get("volume_uphill_penalty", 8.0)),
        volume_warning_ratio=float(hy.get("volume_warning_ratio", 1.5)),
        deep_depth_threshold_m=float(hy.get("deep_depth_threshold_m", 1.5)),
        additional_stage_controls=cfg.get("gauge_network", {}).get("stage_controls", []) or [],
    )
    _emit(progress, "08", "Inundation screening complete.", "done")

    # 09 optional display-AOI clip
    aoi_cfg = cfg.get("aoi", {})
    if aoi_cfg.get("enabled") and aoi_cfg.get("path") and Path(aoi_cfg["path"]).exists():
        _emit(progress, "09", "Clipping finished full-domain result to the selected display AOI.")
        qa = prepare_aoi_context(
            aoi_path=aoi_cfg["path"], dem_path=cfg["paths"]["dem_analysis"],
            gauge_lon=float(cfg["pilot"]["gauge_lon"]), gauge_lat=float(cfg["pilot"]["gauge_lat"]),
            basin_path=processed / "dem_upstream_basin.geojson", mainstem_path=processed / "mainstem.geojson",
            output_dir=processed / "aoi",
            context_buffer_m=float(aoi_cfg.get("hydraulic_context_buffer_m", 1000.0)),
            mainstem_corridor_buffer_m=float(aoi_cfg.get("mainstem_context_buffer_m", 250.0)),
        )
        if qa.get("aoi_dem_coverage_fraction", 0) >= 0.95:
            result.update(postprocess_scenario_to_aoi(
                aoi_path=aoi_cfg["path"], depth_raster=result["depth_raster"], extent_geojson=result["extent_geojson"],
                potential_depth_raster=result.get("potential_depth_raster"),
                potential_extent_geojson=result.get("potential_extent_geojson"), outputs_dir=outputs,
                min_depth_m=float(cfg["terrain"].get("min_depth_m", 0.05)),
            ))
        result["aoi_qa"] = qa
        _emit(progress, "09", "Display-AOI postprocessing complete.", "done")

    metadata = {
        "scenario_label": f"Live NWM peak {peak_q:.2f} cfs",
        "nwm_reach": str(reach), "peak_time": str(peak_time),
        "usgs_latest": obs, **result,
    }
    (outputs / "dashboard_scenario.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    _emit(progress, "10", "Dashboard products refreshed.", "done")
    return metadata


def run_full_pipeline(project_root, workspace_root, config_path, regional_cfg, force_static=False, progress=None):
    prep = prepare_static_workspace(project_root, workspace_root, config_path, regional_cfg, force=force_static, progress=progress)
    live = run_live_workspace(workspace_root, config_path, progress=progress)
    return {"static": prep, "live": live}
