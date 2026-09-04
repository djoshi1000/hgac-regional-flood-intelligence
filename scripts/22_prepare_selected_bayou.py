from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import yaml

from src.regional_catalog import load_regional_catalog, select_channel_from_click, watershed_for_click
from src.gauge_discovery import choose_best_gauge
from src.workspace import create_workspace
from src.pipeline_v5 import run_full_pipeline

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description="CLI equivalent of V5 map click → gauge → full pipeline.")
p.add_argument("--lon", type=float, required=True)
p.add_argument("--lat", type=float, required=True)
p.add_argument("--force-static", action="store_true")
args = p.parse_args()

regional = yaml.safe_load((ROOT / "config" / "v5_regional.yaml").read_text(encoding="utf-8")) or {}
r = regional.get("regional", {})
cache = Path(r.get("catalog_cache", "data/regional_cache/catalog")); cache = cache if cache.is_absolute() else ROOT / cache
boundary, channels, watersheds, _ = load_regional_catalog(cache)
sel = select_channel_from_click(boundary, channels, args.lon, args.lat,
                                hcfcd_max_distance_m=float(r.get("hcfcd_click_tolerance_m", 1500)),
                                nhd_search_radius_m=float(r.get("nhd_search_radius_m", 6000)))
sel["mainstem_match_tolerance_m"] = float(r.get("mainstem_match_tolerance_m", 500))
print(f"Selected: {sel['name']} ({sel['source']})")

best, candidates = choose_best_gauge(args.lon, args.lat,
    radius_km=float(r.get("gauge_search_radius_km", 30)),
    max_candidates_to_test=int(r.get("gauge_candidates_to_test", 8)),
    cache_root=ROOT / "data" / "regional_cache" / "gauges")
if not best or not best.get("basin_contains_click") or not best.get("recent_data_ok"):
    raise SystemExit("No hydraulically supported live USGS control gauge was found. Review candidates in the Streamlit app.")
print(f"Gauge: USGS {best['site_no']} — {best['name']}")

raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8")) if (ROOT / "config" / "config.yaml").exists() else {}
keep = ["usgs","nldi","nwm","floodhub","dem","terrain","bank_profile","hydraulics","inundation","gauge_network"]
base = {k: copy.deepcopy(raw[k]) for k in keep if k in raw}
ws, cp, _ = create_workspace(ROOT, sel, best, base_cfg=base, watershed=watershed_for_click(watersheds, args.lon, args.lat), overwrite_config=False)

def progress(e):
    print(f"[{e.get('step')}] {e.get('status')}: {e.get('message')}")

result = run_full_pipeline(ROOT, ws, cp, regional, force_static=args.force_static, progress=progress)
print(f"\nWorkspace: {ws}")
print(f"Current preferred inundated area: {result['live'].get('inundated_area_sqmi', 0):.4f} mi²")
