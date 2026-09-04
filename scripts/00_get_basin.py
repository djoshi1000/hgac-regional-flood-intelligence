from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ensure_directories
from src.nldi import get_basin, get_comid

cfg = load_config()
ensure_directories(cfg)
site = cfg["pilot"]["usgs_site"]

basin = get_basin(
    site,
    simplified=cfg["nldi"]["simplified"],
    split_catchment=cfg["nldi"]["split_catchment"],
)

out = Path(cfg["paths"]["basin_geojson"])
basin.to_file(out, driver="GeoJSON")

print("\nSUCCESS")
print("Saved basin:", out)

comid = get_comid(site)
if comid:
    print("NHDPlus COMID:", comid)
else:
    print(
        "NHDPlus COMID could not be retrieved right now. "
        "That does NOT block DEM processing; we can resolve the NWM reach later."
    )
