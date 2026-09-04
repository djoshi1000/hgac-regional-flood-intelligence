from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import yaml

from src.regional_catalog import build_regional_catalog

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser(description="Build/cache the H-GAC regional waterway catalog.")
p.add_argument("--force", action="store_true")
args = p.parse_args()

cfg_path = ROOT / "config" / "v5_regional.yaml"
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
cache = Path(cfg.get("regional", {}).get("catalog_cache", "data/regional_cache/catalog"))
if not cache.is_absolute():
    cache = ROOT / cache

result = build_regional_catalog(cache, force=args.force)
print("\nH-GAC REGIONAL CATALOG READY")
for k, v in result.items():
    print(f"  {k}: {v}")
