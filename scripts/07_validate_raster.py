from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation import raster_flood_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--obs", required=True)
parser.add_argument("--threshold", type=float, default=0.05)
args = parser.parse_args()

print(raster_flood_metrics(args.pred, args.obs, args.threshold))
