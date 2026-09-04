from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ensure_directories
from src.terrain import build_corrected_terrain


cfg = load_config()
ensure_directories(cfg)

processed_dir = Path(cfg["paths"]["processed"])

dem_path = Path(
    cfg["paths"].get(
        "dem_analysis",
        processed_dir / "dem_analysis_2m.tif",
    )
)

result = build_corrected_terrain(
    dem_path=dem_path,
    gauge_lon=float(cfg["pilot"]["gauge_lon"]),
    gauge_lat=float(cfg["pilot"]["gauge_lat"]),
    processed_dir=processed_dir,
    stream_threshold_km2=float(cfg["terrain"].get("stream_threshold_km2", 5.0)),
    mainstem_threshold_km2=float(cfg["terrain"].get("mainstem_threshold_km2", 8.0)),
    expected_drainage_area_sqmi=float(
        cfg["pilot"].get("expected_drainage_area_sqmi", 16.1)
    ),
    gauge_snap_max_m=float(cfg["terrain"].get("gauge_snap_max_m", 250.0)),
)

print("\n========================================")
print("UPSTREAM-BASIN TERRAIN PRODUCTS COMPLETE")
print("========================================")
print(f"Input gauge DEM elev:    {result['input_gauge_dem_elevation_ft_navd88']:.2f} ft NAVD88")
print(f"Snapped channel elev:    {result['gauge_channel_elevation_ft_navd88']:.2f} ft NAVD88")
print(f"Gauge snap distance:     {result['gauge_snap_euclidean_distance_m']:.1f} m")
print(f"DEM upstream basin:      {result['dem_upstream_basin_area_km2']:.2f} km²")
if result['published_drainage_area_km2'] is not None:
    print(f"USGS published basin:    {result['published_drainage_area_km2']:.2f} km²")
    print(f"DEM / USGS area ratio:   {result['dem_to_published_area_ratio']:.1%}")
print(f"Outlet upstream area:    {result['outlet_upstream_area_km2']:.2f} km²")
print(f"Mainstem cells:          {result['mainstem_cells']:,}")
print(f"HAND minimum:            {result['hand_min_m']:.4f} m")
print(f"HAND median:             {result['hand_median_m']:.3f} m")
print(f"HAND 95th percentile:    {result['hand_p95_m']:.3f} m")
print(f"DEM fill p95:            {result['fill_depth_p95_m']:.3f} m")
print(f"DEM fill p99:            {result['fill_depth_p99_m']:.3f} m")
print(f"DEM fill maximum:        {result['fill_depth_max_m']:.3f} m")

print("\nCreated/replaced:")
for name, path in result["paths"].items():
    print(f"  {name}: {path}")

print("\nIMPORTANT:")
print("The DEM is conditioned only for routing.")
print("The final inundation depth still uses the original LiDAR DEM.")
print("The HAND/drain products are now limited to the DEM-derived basin upstream of the gauge.")
