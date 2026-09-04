from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import yaml

from src.dem_cache import build_s3_index, download_tiles, build_virtual_mosaic, build_arcpy_mosaic_dataset

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description="Manage the shared V5 DEM cache.")
p.add_argument("--prefetch-houston-b24", action="store_true", help="Download all indexed Houston B24 DEM tiles once. This may be very large.")
p.add_argument("--build-vrt", action="store_true", help="Build a GDAL VRT from cached TIFF tiles.")
p.add_argument("--build-arcpy-mosaic", action="store_true", help="Build an ArcGIS Pro Mosaic Dataset from cached tiles (requires ArcPy environment).")
p.add_argument("--force-index", action="store_true")
args = p.parse_args()

regional = yaml.safe_load((ROOT / "config" / "v5_regional.yaml").read_text(encoding="utf-8")) or {}
base = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8")) if (ROOT / "config" / "config.yaml").exists() else {}
dem_cfg = base.get("dem") or {
    "project": "TX_Houston_B24",
    "work_units": ["TX_Houston_1_B24", "TX_Houston_2_B24", "TX_Houston_3_B24", "TX_Houston_4_B24"],
    "download_workers": 4,
}
dc = regional.get("dem_cache", {})
cache = Path(dc.get("cache_root", "data/regional_cache/dem"))
if not cache.is_absolute(): cache = ROOT / cache
cache.mkdir(parents=True, exist_ok=True)
tile_dir = cache / "tiles"

idx = build_s3_index(dem_cfg, cache, force=args.force_index)
print(f"Indexed Houston B24 DEM objects: {len(idx):,}")

if args.prefetch_houston_b24:
    print("WARNING: prefetching the entire Houston B24 project can consume substantial disk space/time.")
    keys = [r["key"] for r in idx]
    local = download_tiles(keys, tile_dir, workers=int(dem_cfg.get("download_workers", 4)))
    print(f"Cached tiles: {len(local):,}")

if args.build_vrt:
    vrt = build_virtual_mosaic(tile_dir, cache / "hgac_dem_cache.vrt")
    print(f"VRT: {vrt}")

if args.build_arcpy_mosaic:
    gdb = Path(dc.get("arcpy_mosaic_gdb", "data/regional_cache/HGAC_DEM_Cache.gdb"))
    if not gdb.is_absolute(): gdb = ROOT / gdb
    md = build_arcpy_mosaic_dataset(tile_dir, gdb, dc.get("arcpy_mosaic_name", "HGAC_DEM_CACHE"))
    print(f"ArcPy Mosaic Dataset: {md}")

if not any([args.prefetch_houston_b24, args.build_vrt, args.build_arcpy_mosaic]):
    print("No build action requested. Use --prefetch-houston-b24, --build-vrt, or --build-arcpy-mosaic.")
