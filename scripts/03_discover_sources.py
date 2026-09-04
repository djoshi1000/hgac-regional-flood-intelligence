from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.nldi import get_comid
from src.nwps import get_reach
from src.floodhub import FloodHubClient

cfg = load_config()
site = cfg["pilot"]["usgs_site"]

comid = get_comid(site)
print("NLDI COMID:", comid)

if comid:
    try:
        print("NOAA NWM reach lookup:")
        print(get_reach(comid))
    except Exception as exc:
        print("COMID did not resolve as an NWM reach:", exc)
        print("Set nwm.reach_id manually if needed.")

key = cfg["secrets"]["floodhub_api_key"]
if key:
    lat = cfg["pilot"]["gauge_lat"]
    lon = cfg["pilot"]["gauge_lon"]
    p = cfg["floodhub"]["search_padding_deg"]
    client = FloodHubClient(key)
    gauges = client.search_gauges_bbox(
        lon-p, lat-p, lon+p, lat+p,
        cfg["floodhub"]["include_non_quality_verified"],
        cfg["floodhub"]["include_gauges_without_hydro_model"],
    )
    print("Flood Hub gauges found:", len(gauges))
    for g in gauges[:30]:
        print(
            g.get("gaugeId"), "|", g.get("siteName"), "|",
            g.get("river"), "| verified:", g.get("qualityVerified"),
            "| hasModel:", g.get("hasModel"), "|", g.get("location")
        )
else:
    print("No Flood Hub API key yet; Google discovery skipped.")
