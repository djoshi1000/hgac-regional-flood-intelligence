from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.validation import validate_extent, validate_high_water_marks

p = argparse.ArgumentParser(description="Validate a historical inundation replay against observed flood extent and/or high-water marks.")
p.add_argument("--event-name", required=True)
p.add_argument("--reference-extent", default=None, help="Observed flood polygon or depth raster.")
p.add_argument("--model-depth", default=None, help="Defaults to outputs/latest_depth.tif")
p.add_argument("--model-wse", default=None, help="Defaults to outputs/latest_wse.tif")
p.add_argument("--aoi", default=None, help="Optional evaluation AOI; defaults to config aoi.path when enabled.")
p.add_argument("--hwm", default=None, help="High-water-mark points: GeoJSON/Shapefile/GPKG/CSV.")
p.add_argument("--hwm-wse-field", default=None, help="Observed NAVD88 water-surface elevation field.")
p.add_argument("--hwm-unit", default="ft", choices=["ft", "m"])
p.add_argument("--hwm-lat-field", default="latitude")
p.add_argument("--hwm-lon-field", default="longitude")
p.add_argument("--model-depth-threshold-m", type=float, default=0.05)
p.add_argument("--reference-depth-threshold-m", type=float, default=0.05)
args = p.parse_args()

cfg = load_config()
outputs = Path(cfg["paths"]["outputs"])
val_dir = outputs / "validation" / args.event_name
val_dir.mkdir(parents=True, exist_ok=True)
model_depth = Path(args.model_depth or cfg["paths"].get("latest_inundation", outputs / "latest_depth.tif"))
model_wse = Path(args.model_wse or (outputs / "latest_wse.tif"))
channel_mask = outputs / "latest_channel_corridor_mask.tif"
aoi_cfg = cfg.get("aoi", {})
evaluation_aoi = args.aoi or (aoi_cfg.get("path") if aoi_cfg.get("enabled", False) else None)

combined = {"event_name": args.event_name}
if args.reference_extent:
    combined["extent"] = validate_extent(
        model_depth_raster=model_depth,
        reference_extent=args.reference_extent,
        output_dir=val_dir,
        event_name=args.event_name,
        model_depth_threshold_m=args.model_depth_threshold_m,
        reference_depth_threshold_m=args.reference_depth_threshold_m,
        evaluation_aoi=evaluation_aoi,
        exclude_mask_raster=channel_mask if channel_mask.exists() else None,
    )
if args.hwm:
    if not args.hwm_wse_field:
        raise RuntimeError("--hwm-wse-field is required when --hwm is supplied")
    combined["high_water_marks"] = validate_high_water_marks(
        model_wse_raster=model_wse,
        observed_points=args.hwm,
        observed_wse_field=args.hwm_wse_field,
        output_dir=val_dir,
        event_name=args.event_name,
        observed_unit=args.hwm_unit,
        lat_field=args.hwm_lat_field,
        lon_field=args.hwm_lon_field,
    )
if len(combined) == 1:
    raise RuntimeError("Provide --reference-extent and/or --hwm")

summary = val_dir / "validation_summary.json"
summary.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print("\n========================================")
print("V4.3 HISTORICAL INUNDATION VALIDATION")
print("========================================")
print(f"Event: {args.event_name}")
if "extent" in combined:
    e = combined["extent"]
    print(f"Extent CSI / IoU:          {e['critical_success_index_iou']:.3f}")
    print(f"Extent precision:          {e['precision']:.3f}")
    print(f"Extent recall:             {e['recall']:.3f}")
    print(f"Extent F1:                 {e['f1']:.3f}")
    print(f"Model/reference area:      {e['model_area_sqmi']:.3f} / {e['reference_area_sqmi']:.3f} mi²")
    print(f"Area bias ratio:           {e['area_bias_ratio'] if e['area_bias_ratio'] is not None else 'N/A'}")
if "high_water_marks" in combined:
    h = combined["high_water_marks"]
    print(f"HWM valid comparisons:     {h['valid_comparison_count']}")
    print(f"HWM MAE:                   {h['mae_m'] if h['mae_m'] is not None else 'N/A'} m")
    print(f"HWM RMSE:                  {h['rmse_m'] if h['rmse_m'] is not None else 'N/A'} m")
    print(f"HWM bias model-observed:   {h['bias_m_model_minus_observed'] if h['bias_m_model_minus_observed'] is not None else 'N/A'} m")
print(f"Summary: {summary}")
