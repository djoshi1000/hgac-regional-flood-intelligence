from __future__ import annotations

from pathlib import Path
import copy
import json
import re
import yaml
import geopandas as gpd


def slugify(text: str):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return s[:80] or "selected_bayou"


def workspace_path(project_root: str | Path, channel_name: str, site_no: str):
    return Path(project_root) / "workspaces" / f"{slugify(channel_name)}__usgs_{site_no}"


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_v5_config():
    return {
        "project": {"name": "H-GAC Regional Flood Intelligence", "timezone": "America/Chicago"},
        "pilot": {},
        "usgs": {"iv_period": "P7D"},
        "nldi": {"split_catchment": False, "simplified": False},
        "nwm": {"reach_id": "auto", "preferred_series": "short_range", "value_unit": "cfs"},
        "floodhub": {
            "gauge_id": "", "search_padding_deg": 0.10,
            "include_non_quality_verified": True, "include_gauges_without_hydro_model": False,
        },
        "dem": {
            "project": "TX_Houston_B24",
            "work_units": ["TX_Houston_1_B24", "TX_Houston_2_B24", "TX_Houston_3_B24", "TX_Houston_4_B24"],
            "hydraulic_resolution_m": 1.0, "analysis_resolution_m": 2.0, "download_workers": 4,
        },
        "terrain": {
            "stream_threshold_km2": 5.0, "mainstem_threshold_km2": 8.0,
            "gauge_snap_max_m": 250.0, "min_depth_m": 0.05, "max_hand_m": 12.0,
            "min_connected_pixels": 20, "screening_overbank_stage_ft": 35.0,
            "fill_hotspot_threshold_m": 1.0,
        },
        "bank_profile": {
            "section_spacing_m": 150.0, "half_width_m": 150.0, "sample_spacing_m": 1.0,
            "tangent_window_m": 30.0, "bank_search_min_m": 8.0, "bank_search_max_m": 65.0,
            "bank_min_relief_ft": 6.5, "peak_prominence_ft": 0.35, "outlier_threshold_ft": 4.0,
            "landward_seed_offset_m": 4.0, "landward_seed_search_m": 20.0,
            "landward_seed_step_m": 2.0, "min_channel_slope": 1.0e-5, "max_channel_slope": 0.02,
        },
        "hydraulics": {
            "flow_area_exponent": 0.85, "capacity_area_exponent": 0.70,
            "contained_stage_exponent": 0.60, "overbank_excess_exponent": 0.70,
            "capacity_outlier_factor": 3.0, "capacity_floor_factor": 1.00,
            "capacity_ceiling_factor": 1.50, "max_local_overbank_ft": 12.0,
            "max_local_overbank_multiplier": 1.5, "max_section_influence_m": 1200.0,
            "channel_exclusion_buffer_m": 2.0, "hydrograph_max_interval_hours": 3.0,
            "screening_duration_hours": 6.0, "floodplain_volume_fraction": 1.0,
            "volume_uphill_penalty": 8.0, "volume_warning_ratio": 1.5,
            "deep_depth_threshold_m": 1.5,
        },
        "inundation": {"connectivity": 4, "min_component_pixels": 20, "min_overtop_ft": 0.02},
        "gauge_network": {"stage_controls": []},
        "aoi": {
            "enabled": False, "path": "", "display_clip": True,
            "hydraulic_context_buffer_m": 1000.0, "mainstem_context_buffer_m": 250.0,
            "map_include_gauge": True,
        },
        "paths": {
            "raw": "data/raw", "processed": "data/processed", "outputs": "outputs",
            "basin_geojson": "data/processed/model_basin.geojson",
            "dem_hydraulic": "data/processed/dem_hydraulic_1m.tif",
            "dem_analysis": "data/processed/dem_analysis_2m.tif",
            "hand": "data/processed/hand_2m.tif",
            "upstream_area": "data/processed/upstream_area_km2.tif",
            "stream_mask": "data/processed/stream_mask.tif",
            "rating_curve_csv": "data/processed/rating_curve.csv",
            "latest_inundation": "outputs/latest_depth.tif",
            "latest_inundation_geojson": "outputs/latest_inundation.geojson",
        },
    }


def create_workspace(project_root, selection: dict, gauge: dict, base_cfg=None, watershed=None, overwrite_config=True):
    ws = workspace_path(project_root, selection["name"], gauge["site_no"])
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ws / "outputs").mkdir(parents=True, exist_ok=True)

    cfg = _deep_merge(default_v5_config(), base_cfg or {})
    cfg["project"]["name"] = f"H-GAC Flood Intelligence — {selection['name']}"
    cfg["pilot"] = {
        "usgs_site": gauge["site_no"],
        "usgs_site_name": gauge.get("name", gauge["site_no"]),
        "nws_gauge": "",
        "gauge_lat": float(gauge["latitude"]),
        "gauge_lon": float(gauge["longitude"]),
        "expected_drainage_area_sqmi": float(
            gauge.get("contributing_drainage_area_sqmi")
            or gauge.get("drainage_area_sqmi")
            or 0.0
        ),
    }
    cfg["selection"] = {
        "channel_name": selection["name"],
        "channel_source": selection.get("source"),
        "channel_unit_no": selection.get("unit_no", ""),
        "click_lon": float(selection.get("click_lon", selection["geometry"].centroid.x)),
        "click_lat": float(selection.get("click_lat", selection["geometry"].centroid.y)),
        "distance_to_clicked_channel_m": float(selection.get("distance_m", 0.0)),
        "mainstem_match_tolerance_m": float(selection.get("mainstem_match_tolerance_m", 500.0)),
    }
    cfg.setdefault("rating", {})
    cfg["rating"].update({
        "source": "auto_usgs_stac",
        "gage_datum_navd88_ft_override": None,
        "datum_consistency_tolerance_ft": 10.0,
    })

    if watershed is not None:
        aoi_path = ws / "data" / "processed" / "selected_display_aoi.geojson"
        gpd.GeoDataFrame({"name": [watershed["name"]]}, geometry=[watershed["geometry"]], crs=4326).to_file(aoi_path, driver="GeoJSON")
        cfg["aoi"]["enabled"] = True
        cfg["aoi"]["path"] = "data/processed/selected_display_aoi.geojson"

    config_path = ws / "config" / "config.yaml"
    if overwrite_config or not config_path.exists():
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Save selection geometry/provenance for map display and reproducibility.
    gpd.GeoDataFrame(
        {"name": [selection["name"]], "source": [selection.get("source", "")]},
        geometry=[selection["geometry"]], crs=4326,
    ).to_file(ws / "data" / "processed" / "selected_channel.geojson", driver="GeoJSON")
    state = {
        "channel": {k: v for k, v in selection.items() if k not in {"geometry", "properties"}},
        "gauge": gauge,
        "workspace": str(ws),
        "config": str(config_path),
    }
    (ws / "workspace_state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return ws, config_path, cfg
