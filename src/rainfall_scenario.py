from __future__ import annotations

"""Transparent rainfall-to-flood screening scenarios for V5.3.

This module is intentionally separate from Google Flood Hub. Flood Hub's public
API returns Google's forecast/status products; it does not accept arbitrary user
rainfall totals as a forcing input. For user-defined storms, V5.3 converts a
storm depth/duration to a conservative runoff hydrograph with an NRCS Curve
Number + SCS triangular hydrograph approximation, then passes that hydrograph
through the same local LiDAR/HAND/channel-excluded inundation screen used by the
live NWM workflow.

This is a planning/sensitivity screen, not a calibrated rainfall-runoff or 2-D
hydraulic simulation.
"""

from pathlib import Path
import json
import math
import shutil

import numpy as np
import pandas as pd

from .config import load_config
from .rating_curve import RatingCurve
from .inundation import make_v4_discharge_capacity_inundation

FT3_TO_M3 = 0.028316846592
SQMI_INCH_TO_FT3 = 5280.0 * 5280.0 / 12.0
KM2_TO_SQMI = 0.3861021585424458


def scs_runoff_depth_in(rainfall_in: float, curve_number: float) -> float:
    p = max(float(rainfall_in), 0.0)
    cn = float(np.clip(curve_number, 30.0, 98.0))
    s = 1000.0 / cn - 10.0
    ia = 0.2 * s
    if p <= ia:
        return 0.0
    return float((p - ia) ** 2 / (p - ia + s))


def _profile_geometry(processed: Path):
    p = processed / "bank_profile.csv"
    if not p.exists():
        raise FileNotFoundError("bank_profile.csv is required before rainfall scenarios can run.")
    df = pd.read_csv(p).sort_values("station_m").reset_index(drop=True)
    length_m = float(pd.to_numeric(df["station_m"], errors="coerce").max())
    z = pd.to_numeric(df.get("routing_channel_elevation_ft_navd88"), errors="coerce")
    valid = z.notna() & pd.to_numeric(df["station_m"], errors="coerce").notna()
    if valid.sum() >= 2 and length_m > 0:
        a = df.loc[valid].iloc[0]
        b = df.loc[valid].iloc[-1]
        dz_ft = abs(float(b["routing_channel_elevation_ft_navd88"]) - float(a["routing_channel_elevation_ft_navd88"]))
        slope = dz_ft / max(length_m * 3.280839895, 1.0)
    else:
        slope = 0.0005
    # Very small LiDAR-derived slopes can make empirical Tc explode; keep the
    # screening estimate within a physically interpretable range and report it.
    slope = float(np.clip(slope, 0.0001, 0.02))
    return df, length_m, slope


def estimate_time_of_concentration_hours(mainstem_length_m: float, slope: float) -> float:
    """Kirpich screening estimate; L in metres, slope dimensionless, result h."""
    L = max(float(mainstem_length_m), 100.0)
    S = float(np.clip(slope, 0.0001, 0.02))
    tc_min = 0.01947 * (L ** 0.77) * (S ** -0.385)
    return float(np.clip(tc_min / 60.0, 0.25, 72.0))


def build_scs_triangular_hydrograph(
    rainfall_in: float,
    duration_hours: float,
    basin_area_sqmi: float,
    curve_number: float,
    baseline_cfs: float = 0.0,
    timestep_minutes: float = 15.0,
):
    """Create a volume-conserving SCS triangular direct-runoff hydrograph.

    The SCS peak relationship q_p = 484 A Q / T_p and triangular base time
    T_b = 2.67 T_p approximately preserve A*Q runoff volume.
    """
    runoff_in = scs_runoff_depth_in(rainfall_in, curve_number)
    if runoff_in <= 0 or basin_area_sqmi <= 0:
        now = pd.Timestamp.now(tz="UTC")
        return pd.DataFrame({"time": [now, now + pd.Timedelta(hours=max(duration_hours, 1.0))], "streamflow": [baseline_cfs, baseline_cfs]}), {
            "runoff_depth_in": 0.0, "runoff_coefficient": 0.0, "peak_direct_cfs": 0.0,
            "peak_total_cfs": float(baseline_cfs), "runoff_volume_m3": 0.0,
        }
    raise RuntimeError("Geometry-aware wrapper must be used.")


def build_geometry_aware_hydrograph(
    rainfall_in: float,
    duration_hours: float,
    basin_area_sqmi: float,
    curve_number: float,
    mainstem_length_m: float,
    mainstem_slope: float,
    baseline_cfs: float = 0.0,
    timestep_minutes: float = 15.0,
):
    runoff_in = scs_runoff_depth_in(rainfall_in, curve_number)
    tc_hr = estimate_time_of_concentration_hours(mainstem_length_m, mainstem_slope)
    lag_hr = 0.60 * tc_hr
    dhr = max(float(duration_hours), 0.05)
    tp_hr = dhr / 2.0 + lag_hr
    tb_hr = max(2.67 * tp_hr, tp_hr + 0.25)
    qp = 0.0 if runoff_in <= 0 else 484.0 * float(basin_area_sqmi) * runoff_in / max(tp_hr, 1e-6)
    baseline = max(float(baseline_cfs), 0.0)

    dt_hr = max(float(timestep_minutes) / 60.0, 1.0 / 60.0)
    times_hr = np.arange(0.0, tb_hr + dt_hr * 0.5, dt_hr)
    if times_hr[-1] < tb_hr:
        times_hr = np.append(times_hr, tb_hr)
    direct = np.zeros_like(times_hr, dtype=float)
    rise = times_hr <= tp_hr
    direct[rise] = qp * times_hr[rise] / max(tp_hr, 1e-9)
    fall = ~rise
    direct[fall] = qp * np.maximum(tb_hr - times_hr[fall], 0.0) / max(tb_hr - tp_hr, 1e-9)
    flow = baseline + np.maximum(direct, 0.0)
    start = pd.Timestamp.now(tz="UTC")
    hydro = pd.DataFrame({
        "time": [start + pd.Timedelta(hours=float(x)) for x in times_hr],
        "streamflow": flow,
        "direct_runoff_cfs": direct,
    })

    runoff_volume_ft3 = runoff_in * float(basin_area_sqmi) * SQMI_INCH_TO_FT3
    return hydro, {
        "rainfall_in": float(rainfall_in),
        "duration_hours": dhr,
        "curve_number": float(curve_number),
        "runoff_depth_in": float(runoff_in),
        "runoff_coefficient": float(runoff_in / rainfall_in) if rainfall_in > 0 else 0.0,
        "basin_area_sqmi": float(basin_area_sqmi),
        "mainstem_length_km": float(mainstem_length_m / 1000.0),
        "mainstem_slope": float(mainstem_slope),
        "time_of_concentration_hours": float(tc_hr),
        "lag_hours": float(lag_hr),
        "time_to_peak_hours": float(tp_hr),
        "base_time_hours": float(tb_hr),
        "baseline_cfs": float(baseline),
        "peak_direct_cfs": float(qp),
        "peak_total_cfs": float(baseline + qp),
        "runoff_volume_m3": float(runoff_volume_ft3 * FT3_TO_M3),
        "method": "NRCS Curve Number + SCS triangular unit-hydrograph screening",
    }


def _latest_baseline_cfs(outputs: Path) -> float:
    p = outputs / "latest_usgs_observations.csv"
    if not p.exists():
        return 0.0
    try:
        df = pd.read_csv(p)
        for col in ["00060", "discharge_cfs", "flow_cfs", "streamflow"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals):
                    return float(vals.iloc[-1])
    except Exception:
        pass
    return 0.0


def run_rainfall_scenario(
    workspace_root: str | Path,
    config_path: str | Path,
    rainfall_in: float,
    duration_hours: float,
    wetness: str = "normal",
    curve_number: float | None = None,
    progress=None,
):
    ws = Path(workspace_root)
    cfg = load_config(config_path)
    processed = ws / "data" / "processed"
    outputs = ws / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    preset_cn = {"dry": 75.0, "normal": 85.0, "wet": 92.0}
    cn = float(curve_number) if curve_number not in (None, "") else preset_cn.get(str(wetness).lower(), 85.0)
    cn = float(np.clip(cn, 30.0, 98.0))

    terrain_qa = json.loads((processed / "terrain_qa.json").read_text(encoding="utf-8"))
    bank_summary = json.loads((processed / "bank_profile_summary.json").read_text(encoding="utf-8"))
    bank_df, length_m, slope = _profile_geometry(processed)
    area_km2 = float(terrain_qa["outlet_upstream_area_km2"])
    area_sqmi = area_km2 * KM2_TO_SQMI
    baseline = _latest_baseline_cfs(outputs)

    if progress:
        progress({"step": "S1", "status": "running", "message": "Converting the rainfall scenario to basin runoff and a screening hydrograph."})
    hydro, rain_meta = build_geometry_aware_hydrograph(
        rainfall_in=float(rainfall_in), duration_hours=float(duration_hours),
        basin_area_sqmi=area_sqmi, curve_number=cn,
        mainstem_length_m=length_m, mainstem_slope=slope,
        baseline_cfs=baseline,
    )
    hydro.to_csv(outputs / "scenario_rainfall_hydrograph.csv", index=False)

    curve = RatingCurve.from_csv(processed / "rating_curve.csv")
    gauge_bank_ft = float(bank_summary["gauge_bank_stage_ft_navd88"])
    gauge_bankfull_q = float(curve.flow_from_stage(gauge_bank_ft))
    peak_q = float(hydro["streamflow"].max())
    rating_max_q = float(curve.max_flow_cfs)
    rating_exceeded = bool(peak_q > rating_max_q)
    scenario_stage_ft = float(curve.stage_from_flow(peak_q))

    if progress:
        progress({"step": "S2", "status": "running", "message": f"Scenario peak {peak_q:,.0f} cfs; converting to local WSE and floodplain screening."})

    hy = cfg.get("hydraulics", {}); incfg = cfg.get("inundation", {}); bp = cfg.get("bank_profile", {})
    result = make_v4_discharge_capacity_inundation(
        dem_path=cfg["paths"]["dem_analysis"],
        hand_path=str(processed / "hand_2m.tif"),
        mainstem_path=str(processed / "mainstem_mask.tif"),
        bank_profile_csv=str(processed / "bank_profile.csv"),
        scenario_stage_ft=scenario_stage_ft,
        scenario_flow_cfs=peak_q,
        gauge_bankfull_flow_cfs=gauge_bankfull_q,
        gauge_channel_elevation_m=float(terrain_qa["gauge_channel_elevation_m_navd88"]),
        gauge_upstream_area_km2=area_km2,
        depth_out=str(outputs / "scenario_latest_depth.tif"),
        polygon_out=str(outputs / "scenario_latest_inundation.geojson"),
        potential_depth_out=str(outputs / "scenario_latest_potential_depth.tif"),
        potential_polygon_out=str(outputs / "scenario_latest_potential_inundation.geojson"),
        channel_corridor_out=str(outputs / "scenario_latest_channel_corridor.geojson"),
        channel_mask_out=str(outputs / "scenario_latest_channel_corridor_mask.tif"),
        channel_exclusion_buffer_m=float(hy.get("channel_exclusion_buffer_m", 2.0)),
        hydrograph=hydro[["time", "streamflow"]], hydrograph_flow_unit="cfs",
        hydrograph_max_interval_hours=float(hy.get("hydrograph_max_interval_hours", 3.0)),
        min_depth_m=float(cfg["terrain"].get("min_depth_m", 0.05)),
        max_hand_m=float(cfg["terrain"].get("max_hand_m", 12.0)),
        fill_depth_path=str(processed / "dem_fill_depth_2m.tif"),
        wse_out=str(outputs / "scenario_latest_wse.tif"),
        overtopped_sections_out=str(outputs / "scenario_latest_overtopped_sections.geojson"),
        seed_points_out=str(outputs / "scenario_latest_floodplain_seeds.geojson"),
        max_depth_point_out=str(outputs / "scenario_latest_max_depth_point.geojson"),
        scenario_profile_out=str(outputs / "scenario_latest_bank_wse_profile.csv"),
        seed_mask_out=str(outputs / "scenario_latest_seed_mask.tif"),
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

    metadata = {
        "scenario_label": f"Rainfall scenario: {float(rainfall_in):g} in / {float(duration_hours):g} h",
        "scenario_type": "user_rainfall_screening",
        "rainfall": rain_meta,
        "gauge_bankfull_flow_cfs": gauge_bankfull_q,
        "scenario_flow_cfs": peak_q,
        "scenario_stage_ft_navd88": scenario_stage_ft,
        "rating_curve_max_flow_cfs": rating_max_q,
        "rating_curve_exceeded": rating_exceeded,
        "rating_curve_exceedance_ratio": float(peak_q / max(rating_max_q, 1e-6)),
        "method_note": (
            "User-defined rainfall is NOT sent to Google Flood Hub. Flood Hub is a live AI forecast/"
            "inundation reference. This hypothetical storm uses NRCS-CN/SCS runoff plus the local LiDAR/HAND screen."
        ),
        "accuracy_note": (
            "Planning/sensitivity screening only. Validate against historical observed inundation/high-water marks "
            "before operational use; HEC-RAS 2D is recommended for engineering-grade scenario mapping."
        ),
        "scenario_warnings": ([
            "Scenario peak exceeds the published/local rating-curve flow range; WSE is clamped at the rating maximum, so extreme-storm depth/extent is highly uncertain."
        ] if rating_exceeded else []),
        **result,
    }
    (outputs / "scenario_dashboard.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if progress:
        progress({"step": "S3", "status": "done", "message": "Rainfall scenario flood map and hydrograph are ready."})
    return metadata
