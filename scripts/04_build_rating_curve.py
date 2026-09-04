from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.nwps import get_gauge_ratings, normalize_rating_points

cfg = load_config()
payload = get_gauge_ratings(cfg["pilot"]["nws_gauge"])
df = normalize_rating_points(payload)

if df.empty:
    raise RuntimeError(
        "Automatic rating extraction failed. Create a CSV with columns "
        "flow_cfs,stage_ft at: " + cfg["paths"]["rating_curve_csv"]
    )

out = Path(cfg["paths"]["rating_curve_csv"])
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("Saved rating curve:", out)
print(df.head())
print(df.tail())
