from pathlib import Path
import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ensure_directories
from src.rating_curve import RatingCurve
from src.inundation import make_v4_discharge_capacity_inundation
from src.aoi import prepare_aoi_context, postprocess_scenario_to_aoi
from src.nldi import get_comid
from src.nwps import get_reach_streamflow, normalize_streamflow


parser = argparse.ArgumentParser(
    description=(
        "Generate V4.3 gauge/reach-based screening inundation using stabilized local "
        "capacity, bank-to-bank channel exclusion, hydrograph-aware volume limitation, "
        "historical replay, and optional AOI display clipping."
    )
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--flow-cfs", type=float, help="Gauge scenario discharge in cfs.")
group.add_argument(
    "--nwm-live", action="store_true",
    help="Use the live NOAA NWM forecast peak and integrate the actual forecast hydrograph above bankfull."
)
group.add_argument(
    "--stage-ft", type=float,
    help="Absolute gauge scenario water-surface elevation in ft NAVD88."
)
group.add_argument(
    "--stage-above-bank-ft", type=float,
    help="Manual gauge water level above the local bank elevation."
)
group.add_argument(
    "--hydrograph-csv", type=str,
    help="Historical/replay hydrograph CSV. Expected time + flow_cfs/streamflow/flow column."
)
parser.add_argument(
    "--duration-hours", type=float, default=None,
    help="Duration used for manual single-flow/stage excess-flow volume budget."
)
parser.add_argument("--hydrograph-unit", default="cfs", choices=["cfs", "cms"], help="Unit for --hydrograph-csv.")
parser.add_argument("--scenario-label", default=None, help="Optional label for a historical replay.")
args = parser.parse_args()

cfg = load_config()
ensure_directories(cfg)
processed = Path(cfg["paths"]["processed"])
outputs = Path(cfg["paths"]["outputs"])
outputs.mkdir(parents=True, exist_ok=True)

terrain_qa_path = processed / "terrain_qa.json"
bank_summary_path = processed / "bank_profile_summary.json"
bank_csv = processed / "bank_profile.csv"

if not terrain_qa_path.exists():
    raise RuntimeError("Run first: python scripts\\02_build_terrain.py")
if not bank_summary_path.exists() or not bank_csv.exists():
    raise RuntimeError(
        "Bank/hydraulic profile is missing. Run: "
        "python scripts\\08_extract_gauge_cross_section.py"
    )

terrain_qa = json.loads(terrain_qa_path.read_text(encoding="utf-8"))
bank_summary = json.loads(bank_summary_path.read_text(encoding="utf-8"))
gauge_channel_m = float(terrain_qa["gauge_channel_elevation_m_navd88"])
gauge_area_km2 = float(terrain_qa["outlet_upstream_area_km2"])
gauge_bank_ft = float(bank_summary["gauge_bank_stage_ft_navd88"])

curve = RatingCurve.from_csv(cfg["paths"]["rating_curve_csv"])
gauge_bankfull_q_cfs = float(curve.flow_from_stage(gauge_bank_ft))

hydrograph_for_volume = None
hydrograph_flow_unit = "cfs"

if args.nwm_live:
    reach = cfg.get("nwm", {}).get("reach_id", "auto")
    if reach == "auto":
        reach = get_comid(cfg["pilot"]["usgs_site"])
    nwm = normalize_streamflow(
        get_reach_streamflow(reach, cfg.get("nwm", {}).get("preferred_series", "short_range"))
    )
    if nwm.empty:
        raise RuntimeError("NOAA NWM returned no usable streamflow records.")
    nwm_unit = str(cfg.get("nwm", {}).get("value_unit", "cfs")).lower()
    nwm_cfs = nwm[["time", "streamflow"]].copy()
    if nwm_unit == "cms":
        nwm_cfs["streamflow"] = nwm_cfs["streamflow"] * 35.3146667215
    elif nwm_unit != "cfs":
        raise RuntimeError(f"Unsupported NWM value_unit for V4.2: {nwm_unit}")
    peak_idx = nwm_cfs["streamflow"].idxmax()
    scenario_flow_cfs = float(nwm_cfs.loc[peak_idx, "streamflow"])
    scenario_stage_ft = float(curve.stage_from_flow(scenario_flow_cfs))
    peak_time = nwm_cfs.loc[peak_idx, "time"]
    scenario_label = f"Live NWM peak {scenario_flow_cfs:.2f} cfs"
    hydrograph_for_volume = nwm_cfs
elif args.hydrograph_csv is not None:
    hpath = Path(args.hydrograph_csv)
    if not hpath.exists():
        raise FileNotFoundError(hpath)
    h = pd.read_csv(hpath)
    time_col = next((c for c in ["time", "valid_time", "datetime", "forecast_time"] if c in h.columns), None)
    flow_col = next((c for c in ["flow_cfs", "streamflow", "flow", "discharge", "value"] if c in h.columns), None)
    if time_col is None or flow_col is None:
        raise RuntimeError("Historical hydrograph CSV needs a time column and a flow_cfs/streamflow/flow/discharge/value column.")
    hg = pd.DataFrame({
        "time": pd.to_datetime(h[time_col], utc=True, errors="coerce"),
        "streamflow": pd.to_numeric(h[flow_col], errors="coerce"),
    }).dropna()
    if args.hydrograph_unit == "cms":
        hg["streamflow"] = hg["streamflow"] * 35.3146667215
    hg = hg.sort_values("time").drop_duplicates("time")
    if hg.empty:
        raise RuntimeError("Historical hydrograph CSV contains no usable rows.")
    scenario_flow_cfs = float(hg["streamflow"].max())
    scenario_stage_ft = float(curve.stage_from_flow(scenario_flow_cfs))
    hydrograph_for_volume = hg
    scenario_label = args.scenario_label or f"Historical replay {hpath.stem} peak {scenario_flow_cfs:.1f} cfs"
elif args.flow_cfs is not None:
    scenario_flow_cfs = max(0.0, float(args.flow_cfs))
    scenario_stage_ft = float(curve.stage_from_flow(scenario_flow_cfs))
    scenario_label = f"Flow {scenario_flow_cfs:.2f} cfs"
elif args.stage_ft is not None:
    scenario_stage_ft = float(args.stage_ft)
    scenario_flow_cfs = float(curve.flow_from_stage(scenario_stage_ft))
    scenario_label = f"Stage {scenario_stage_ft:.2f} ft NAVD88"
else:
    scenario_stage_ft = gauge_bank_ft + max(0.0, float(args.stage_above_bank_ft))
    scenario_flow_cfs = float(curve.flow_from_stage(scenario_stage_ft))
    scenario_label = f"Gauge bank + {float(args.stage_above_bank_ft):.2f} ft"

in_cfg = cfg.get("inundation", {})
bp_cfg = cfg.get("bank_profile", {})
hy_cfg = cfg.get("hydraulics", {})
additional_controls = cfg.get("gauge_network", {}).get("stage_controls", []) or []

duration_hours = (
    float(args.duration_hours)
    if args.duration_hours is not None
    else float(hy_cfg.get("screening_duration_hours", 6.0))
)

result = make_v4_discharge_capacity_inundation(
    dem_path=cfg["paths"].get("dem_analysis", str(processed / "dem_analysis_2m.tif")),
    hand_path=str(processed / "hand_2m.tif"),
    mainstem_path=str(processed / "mainstem_mask.tif"),
    bank_profile_csv=str(bank_csv),
    scenario_stage_ft=scenario_stage_ft,
    scenario_flow_cfs=scenario_flow_cfs,
    gauge_bankfull_flow_cfs=gauge_bankfull_q_cfs,
    gauge_channel_elevation_m=gauge_channel_m,
    gauge_upstream_area_km2=gauge_area_km2,
    depth_out=cfg["paths"]["latest_inundation"],
    polygon_out=cfg["paths"]["latest_inundation_geojson"],
    potential_depth_out=str(outputs / "latest_potential_depth.tif"),
    potential_polygon_out=str(outputs / "latest_potential_inundation.geojson"),
    channel_corridor_out=str(outputs / "latest_channel_corridor.geojson"),
    channel_mask_out=str(outputs / "latest_channel_corridor_mask.tif"),
    channel_exclusion_buffer_m=float(hy_cfg.get("channel_exclusion_buffer_m", 2.0)),
    hydrograph=hydrograph_for_volume,
    hydrograph_flow_unit=hydrograph_flow_unit,
    hydrograph_max_interval_hours=float(hy_cfg.get("hydrograph_max_interval_hours", 3.0)),
    min_depth_m=float(cfg["terrain"].get("min_depth_m", 0.05)),
    max_hand_m=float(cfg["terrain"].get("max_hand_m", 12.0)),
    fill_depth_path=str(processed / "dem_fill_depth_2m.tif"),
    wse_out=str(outputs / "latest_wse.tif"),
    overtopped_sections_out=str(outputs / "latest_overtopped_sections.geojson"),
    seed_points_out=str(outputs / "latest_floodplain_seeds.geojson"),
    max_depth_point_out=str(outputs / "latest_max_depth_point.geojson"),
    scenario_profile_out=str(outputs / "latest_bank_wse_profile.csv"),
    seed_mask_out=str(outputs / "latest_seed_mask.tif"),
    connectivity=int(in_cfg.get("connectivity", 4)),
    min_component_pixels=int(in_cfg.get("min_component_pixels", 20)),
    min_overtop_ft=float(in_cfg.get("min_overtop_ft", 0.02)),
    landward_seed_offset_m=float(bp_cfg.get("landward_seed_offset_m", 4.0)),
    landward_seed_search_m=float(bp_cfg.get("landward_seed_search_m", 20.0)),
    landward_seed_step_m=float(bp_cfg.get("landward_seed_step_m", 2.0)),
    fill_depth_warning_m=float(cfg["terrain"].get("fill_hotspot_threshold_m", 1.0)),
    flow_area_exponent=float(hy_cfg.get("flow_area_exponent", 0.85)),
    capacity_area_exponent=float(hy_cfg.get("capacity_area_exponent", 0.70)),
    contained_stage_exponent=float(hy_cfg.get("contained_stage_exponent", 0.60)),
    overbank_excess_exponent=float(hy_cfg.get("overbank_excess_exponent", 0.70)),
    capacity_outlier_factor=float(hy_cfg.get("capacity_outlier_factor", 3.0)),
    capacity_floor_factor=float(hy_cfg.get("capacity_floor_factor", 1.00)),
    capacity_ceiling_factor=float(hy_cfg.get("capacity_ceiling_factor", 1.50)),
    max_local_overbank_ft=float(hy_cfg.get("max_local_overbank_ft", 12.0)),
    max_local_overbank_multiplier=float(hy_cfg.get("max_local_overbank_multiplier", 1.5)),
    max_section_influence_m=float(hy_cfg.get("max_section_influence_m", 1200.0)),
    screening_duration_hours=duration_hours,
    floodplain_volume_fraction=float(hy_cfg.get("floodplain_volume_fraction", 1.0)),
    volume_uphill_penalty=float(hy_cfg.get("volume_uphill_penalty", 8.0)),
    volume_warning_ratio=float(hy_cfg.get("volume_warning_ratio", 1.5)),
    deep_depth_threshold_m=float(hy_cfg.get("deep_depth_threshold_m", 1.5)),
    additional_stage_controls=additional_controls,
)

# Optional AOI is a display/evaluation clip only. The hydraulic calculation
# remains on the full gauge/reach model domain so a gauge may sit outside AOI.
aoi_cfg = cfg.get("aoi", {})
aoi_path = aoi_cfg.get("path", "")
if aoi_cfg.get("enabled", False) and aoi_path and Path(aoi_path).exists():
    aoi_qa = prepare_aoi_context(
        aoi_path=aoi_path,
        dem_path=cfg["paths"].get("dem_analysis", str(processed / "dem_analysis_2m.tif")),
        gauge_lon=float(cfg["pilot"]["gauge_lon"]),
        gauge_lat=float(cfg["pilot"]["gauge_lat"]),
        basin_path=processed / "dem_upstream_basin.geojson",
        mainstem_path=processed / "mainstem.geojson",
        output_dir=processed / "aoi",
        context_buffer_m=float(aoi_cfg.get("hydraulic_context_buffer_m", 1000.0)),
        mainstem_corridor_buffer_m=float(aoi_cfg.get("mainstem_context_buffer_m", 250.0)),
    )
    result["aoi_qa"] = aoi_qa
    if aoi_qa["aoi_dem_coverage_fraction"] >= 0.95:
        result.update(postprocess_scenario_to_aoi(
            aoi_path=aoi_path,
            depth_raster=result["depth_raster"],
            extent_geojson=result["extent_geojson"],
            potential_depth_raster=result.get("potential_depth_raster"),
            potential_extent_geojson=result.get("potential_extent_geojson"),
            outputs_dir=outputs,
            min_depth_m=float(cfg["terrain"].get("min_depth_m", 0.05)),
        ))

metadata = {"scenario_label": scenario_label, **result}
(outputs / "dashboard_scenario.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\n========================================")
print("V4.3 HISTORICAL-VALIDATION + FLEXIBLE-AOI SCREENING")
print("========================================")
print(f"Scenario:                         {scenario_label}")
print(f"Gauge scenario flow:              {scenario_flow_cfs:.2f} cfs")
print(f"Gauge scenario stage:             {scenario_stage_ft:.2f} ft NAVD88")
print(f"Gauge local-bank elevation:       {gauge_bank_ft:.2f} ft NAVD88")
print(f"Gauge bankfull flow proxy:        {gauge_bankfull_q_cfs:.1f} cfs")
print(f"Gauge excess over bank:           {result['gauge_overbank_excess_ft']:.2f} ft")
print(f"Bank sections evaluated:          {result['bank_sections']:,}")
print(f"Hydraulic-valid sections:         {result['hydraulic_valid_sections']:,}")
print(f"Capacity-profile outliers:        {result['capacity_profile_outliers']:,}")
print(f"Capacity constrained low:         {result['capacity_constrained_low_sections']:,}")
print(f"Capacity constrained high:        {result['capacity_constrained_high_sections']:,}")
print(f"Locally overtopped sections:      {result['overtopped_sections']:,}")
print(f"Sections that produced seed:      {result['seeded_sections']:,}")
print(f"Channel corridor area excluded:   {result['channel_corridor_area_sqmi']:.4f} mi²")
print(f"Channel bank pairs raw/interp:    {result['channel_corridor_raw_bank_pairs']} / {result['channel_corridor_interpolated_sections']}")

print("\nPotential terrain-connected upper bound:")
print(f"  Candidate terrain pixels:       {result['candidate_pixels_before_connectivity']:,}")
print(f"  Connected wet pixels:           {result['potential_connected_pixels']:,}")
print(f"  Area:                           {result['potential_inundated_area_sqmi']:.4f} mi²")
print(f"  Storage:                        {result['potential_storage_volume_m3']:,.0f} m³")
print(f"  Maximum depth:                  {result['potential_max_depth_m']:.3f} m")

print("\nPreferred volume-limited screening:")
print(f"  Floodplain seed pixels:         {result['seed_pixels']:,}")
print(f"  Volume-limited wet pixels:      {result['volume_limited_pixels']:,}")
print(f"  Inundated area:                 {result['inundated_area_sqmi']:.4f} mi²")
print(f"  Flood storage volume:           {result['flood_storage_volume_m3']:,.0f} m³")
print(f"  Maximum depth:                  {result['max_depth_m']:.3f} m")
print(f"  Mean positive depth:            {result['mean_depth_m']:.3f} m")
print(f"  Deep area >= {result['deep_depth_threshold_m']:.2f} m:          {result['deep_area_sqmi']:.4f} mi²")

print("\nVolume budget:")
print(f"  Source:                         {result['volume_budget_source']}")
if result['volume_budget_source'] == 'integrated_hydrograph':
    hqa = result.get('hydrograph_volume_qa', {})
    print(f"  Hydrograph points:              {hqa.get('point_count', 0)}")
    print(f"  Forecast horizon:               {hqa.get('horizon_hours', 0.0):.2f} h")
    print(f"  Exceedance intervals:           {hqa.get('exceedance_interval_hours', 0.0):.2f} h")
    print(f"  Large-gap intervals skipped:    {hqa.get('intervals_skipped_large_gap', 0)}")
else:
    print(f"  Assumed duration:               {result['screening_duration_hours']:.2f} h")
print(f"  Gauge excess flow:              {result['gauge_excess_flow_cfs']:.1f} cfs")
print(f"  Excess-flow volume proxy:       {result['excess_flow_volume_proxy_m3']:,.0f} m³")
print(f"  Floodplain fraction:            {result['floodplain_volume_fraction']:.2f}")
print(f"  Floodplain volume budget:       {result['volume_budget_m3']:,.0f} m³")
print(f"  Volume used:                    {result['volume_used_m3']:,.0f} m³")
print(f"  Volume remaining:               {result['volume_remaining_m3']:,.0f} m³")
pot_ratio = result.get("potential_storage_to_excess_volume_ratio")
print(
    f"  Potential storage / excess:     {pot_ratio:.2f}"
    if pot_ratio is not None else
    "  Potential storage / excess:     N/A"
)

hq = result.get("hydraulic_profile_qa", {})
print("\nHydraulic-profile QA:")
print(f"  Gauge Q / capacity ratio:       {hq.get('gauge_flow_to_capacity_ratio', float('nan')):.3f}")
print(f"  Capacity lower factor:          {hq.get('capacity_floor_factor', float('nan')):.2f}")
print(f"  Capacity upper factor:          {hq.get('capacity_ceiling_factor', float('nan')):.2f}")
print(f"  Local overbank cap this run:    {hq.get('scenario_local_overbank_cap_ft', float('nan')):.2f} ft")
print(f"  Capacity calibration section:   {hq.get('calibration_section_id', 'N/A')}")

if result.get("max_depth_location"):
    q = result["max_depth_location"]
    print("\nMaximum-depth QA (preferred map):")
    print(f"  Lat/lon:                       {q['latitude']:.6f}, {q['longitude']:.6f}")
    print(f"  Ground elevation:              {q['ground_elevation_ft_navd88']:.2f} ft NAVD88")
    print(f"  WSE:                           {q['wse_ft_navd88']:.2f} ft NAVD88")
    print(f"  HAND:                          {q['hand_m']:.2f} m")
    print(f"  Distance to mainstem:          {q['distance_to_mainstem_m']:.1f} m")
    print(f"  Distance outside channel:      {q.get('distance_to_channel_corridor_m', float('nan')):.1f} m")
    print(f"  DEM fill depth:                {q['fill_depth_m'] if q['fill_depth_m'] is not None else 'N/A'} m")
    if q.get("nearest_section_id") is not None:
        print(f"  Nearest hydraulic section:      {q['nearest_section_id']} at {q.get('nearest_section_distance_m', float('nan')):.1f} m")
        print(f"  Nearest section bank/WSE:       {q.get('nearest_section_bank_ft_navd88', float('nan')):.2f} / {q.get('nearest_section_wse_ft_navd88', float('nan')):.2f} ft NAVD88")
        print(f"  Nearest section overtopping:    {q.get('nearest_section_overtop_ft', float('nan')):.2f} ft")
        print(f"  Bank-to-ground drop:            {q.get('bank_to_ground_drop_ft', float('nan')):.2f} ft")

if result.get("qa_flags"):
    print("\nQA FLAGS:")
    for flag in result["qa_flags"]:
        print(f"  - {flag}")

if "aoi_inundated_area_sqmi" in result:
    print("\nAOI display/evaluation clip:")
    print(f"  Gauge inside AOI:                {result['aoi_qa']['gauge_inside_aoi']}")
    print(f"  Gauge distance to AOI:           {result['aoi_qa']['gauge_distance_to_aoi_m']:.1f} m")
    print(f"  AOI flooded area:                {result['aoi_inundated_area_sqmi']:.4f} mi²")
    print(f"  AOI flood storage:               {result['aoi_storage_volume_m3']:,.0f} m³")
    print(f"  AOI maximum depth:               {result['aoi_max_depth_m']:.3f} m")

print("\nOutputs:")
for key in [
    "depth_raster", "extent_geojson", "potential_depth_raster", "potential_extent_geojson",
    "channel_corridor_geojson", "channel_corridor_mask_raster",
    "wse_raster", "overtopped_sections_geojson", "seed_points_geojson",
    "max_depth_point_geojson", "scenario_profile_csv", "seed_mask_raster",
]:
    print(f"  {key}: {result.get(key)}")

print("\nIMPORTANT:")
print("The primary V4.3 depth/extent excludes the LiDAR-derived bank-to-bank channel corridor.")
print("For --nwm-live or --hydrograph-csv, floodplain volume is integrated from the supplied hydrograph above bankfull.")
print("Single-flow/stage manual scenarios still use excess flow x the supplied/fallback duration.")
print("The potential terrain-connected extent is an unconstrained upper-bound QA layer only.")
print("Neither output is an engineering hydraulic model or a regulatory floodplain.")
