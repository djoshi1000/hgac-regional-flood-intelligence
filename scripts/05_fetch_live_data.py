from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.usgs import fetch_iv, latest_observation
from src.nldi import get_comid
from src.nwps import get_reach_streamflow, normalize_streamflow
from src.floodhub import FloodHubClient

cfg = load_config()

usgs = fetch_iv(cfg["pilot"]["usgs_site"], cfg["usgs"]["iv_period"])
print("USGS latest:", latest_observation(usgs))

reach = cfg["nwm"]["reach_id"]
if reach == "auto":
    reach = get_comid(cfg["pilot"]["usgs_site"])

if reach:
    try:
        nwm = normalize_streamflow(
            get_reach_streamflow(reach, cfg["nwm"]["preferred_series"])
        )
        print("\nNWM latest rows:")
        print(nwm.tail(20).to_string(index=False))
    except Exception as exc:
        print("NWM failed:", exc)

key = cfg["secrets"]["floodhub_api_key"]
gid = cfg["floodhub"]["gauge_id"]
if key and gid:
    client = FloodHubClient(key)
    try:
        fh = client.normalize_forecasts(client.query_forecasts([gid]))
        print("\nFlood Hub latest rows:")
        print(fh.tail(20).to_string(index=False))
    except Exception as exc:
        print("Flood Hub failed:", exc)
else:
    print("\nFlood Hub not configured yet.")
