from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ensure_directories
from src.aoi import prepare_aoi_context

parser = argparse.ArgumentParser(description="Prepare/QA a display AOI relative to the configured gauge and current model domain.")
parser.add_argument("--aoi", default=None, help="AOI polygon path (GeoJSON/Shapefile/GPKG). Defaults to config aoi.path.")
parser.add_argument("--context-buffer-m", type=float, default=None)
parser.add_argument("--mainstem-buffer-m", type=float, default=None)
args = parser.parse_args()

cfg = load_config()
ensure_directories(cfg)
processed = Path(cfg["paths"]["processed"])
outputs = Path(cfg["paths"]["outputs"])
aoi_cfg = cfg.get("aoi", {})
aoi_path = args.aoi or aoi_cfg.get("path", "")
if not aoi_path:
    raise RuntimeError("Provide --aoi or set aoi.path in config/config.yaml")

result = prepare_aoi_context(
    aoi_path=aoi_path,
    dem_path=cfg["paths"].get("dem_analysis", processed / "dem_analysis_2m.tif"),
    gauge_lon=float(cfg["pilot"]["gauge_lon"]),
    gauge_lat=float(cfg["pilot"]["gauge_lat"]),
    basin_path=processed / "dem_upstream_basin.geojson",
    mainstem_path=processed / "mainstem.geojson",
    output_dir=processed / "aoi",
    context_buffer_m=float(args.context_buffer_m if args.context_buffer_m is not None else aoi_cfg.get("hydraulic_context_buffer_m", 1000.0)),
    mainstem_corridor_buffer_m=float(args.mainstem_buffer_m if args.mainstem_buffer_m is not None else aoi_cfg.get("mainstem_context_buffer_m", 250.0)),
)

print("\n========================================")
print("V4.3 AOI / GAUGE DOMAIN QA")
print("========================================")
print(f"Status:                         {result['status']}")
print(f"Gauge inside display AOI:       {result['gauge_inside_aoi']}")
print(f"Gauge distance to AOI:          {result['gauge_distance_to_aoi_m']:.1f} m")
print(f"Gauge inside DEM/model domain:  {result['gauge_inside_dem']}")
print(f"AOI covered by DEM:             {100*result['aoi_dem_coverage_fraction']:.1f}%")
if result['aoi_overlap_with_current_upstream_basin_fraction'] is not None:
    print(f"AOI in gauge upstream basin:    {100*result['aoi_overlap_with_current_upstream_basin_fraction']:.1f}%")
print(f"AOI intersects current mainstem:{result['aoi_intersects_current_mainstem']}")
if result['warnings']:
    print("Warnings:")
    for w in result['warnings']:
        print(f"  - {w}")
print("\nCreated:")
for key in ["aoi_display_geojson", "model_context_domain_geojson", "configured_gauge_geojson", "aoi_qa_json"]:
    if key in result:
        print(f"  {key}: {result[key]}")
print("\nINTERPRETATION:")
print("The gauge does NOT need to be inside the display AOI. It must, however, be hydraulically relevant")
print("and the DEM/model domain must include the gauge plus the connected river reach needed by the model.")
print("If the AOI is mostly outside the gauge's DEM-derived upstream basin, use a more appropriate downstream")
print("control gauge/reach rather than choosing the geographically nearest gauge from another watershed.")
