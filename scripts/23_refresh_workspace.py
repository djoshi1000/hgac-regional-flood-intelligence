from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from src.pipeline_v5 import run_live_workspace

p = argparse.ArgumentParser(description="Refresh USGS + NWM + inundation for an already-prepared V5 workspace.")
p.add_argument("workspace", help="Path to workspaces/<bayou>__usgs_<site>")
args = p.parse_args()
ws = Path(args.workspace).resolve()
cp = ws / "config" / "config.yaml"
if not cp.exists(): raise SystemExit(f"Missing workspace config: {cp}")

def progress(e): print(f"[{e.get('step')}] {e.get('message')}")
meta = run_live_workspace(ws, cp, progress=progress)
print(f"Preferred inundated area: {meta.get('inundated_area_sqmi', 0):.4f} mi²")
