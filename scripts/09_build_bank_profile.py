from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ensure_directories
from src.bank_profile import build_bank_profile, write_gauge_cross_section_products


cfg = load_config()
ensure_directories(cfg)

processed = Path(cfg["paths"]["processed"])
outputs = Path(cfg["paths"]["outputs"])

bp = cfg.get("bank_profile", {})

result = build_bank_profile(
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

gauge_products = write_gauge_cross_section_products(
    hydraulic_dem_path=cfg["paths"]["dem_hydraulic"],
    bank_profile_csv=result["paths"]["bank_profile_csv"],
    processed_dir=processed,
    outputs_dir=outputs,
)

print("\n========================================")
print("V4 LOCAL BANK + HYDRAULIC PROFILE COMPLETE")
print("========================================")
print(f"Mainstem length:          {result['mainstem_length_m'] / 1000.0:.2f} km")
print(f"Cross sections:           {result['section_count']}")
print(f"Raw valid sections:       {result['raw_valid_sections']}")
print(f"Raw sections used:        {result['raw_sections_used_after_outlier_filter']}")
print(f"Profile outliers:         {result['profile_outlier_sections']}")
print(f"Gauge bank elevation:     {result['gauge_bank_stage_ft_navd88']:.2f} ft NAVD88")
print(f"Median bank freeboard:    {result['median_final_freeboard_ft']:.2f} ft")
print(f"Profile bank range:       {result['min_final_bank_ft_navd88']:.2f} to {result['max_final_bank_ft_navd88']:.2f} ft NAVD88")
print(f"Hydraulic-valid sections: {result.get('hydraulic_valid_sections', 0)}")
print(f"Median channel slope:     {result.get('median_local_channel_slope', float('nan')):.6f}")

print("\nCreated:")
for name, path in result["paths"].items():
    print(f"  {name}: {path}")
for name, path in gauge_products.items():
    print(f"  {name}: {path}")

print("\nIMPORTANT:")
print("The local bank profile is an automated LiDAR screening product.")
print("The inundation model will now release floodplain water only at sections")
print("where the scenario water surface locally exceeds the local bank profile.")
