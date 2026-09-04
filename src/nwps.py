import pandas as pd
import requests

BASE = "https://api.water.noaa.gov/nwps/v1"


def _get(path, params=None):
    r = requests.get(f"{BASE}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def get_gauge(identifier):
    return _get(f"/gauges/{identifier}")


def get_gauge_ratings(identifier):
    return _get(f"/gauges/{identifier}/ratings", {"all": "true", "limit": 10000})


def get_reach(reach_id):
    return _get(f"/reaches/{reach_id}")


def get_reach_streamflow(reach_id, series="short_range"):
    return _get(f"/reaches/{reach_id}/streamflow", {"series": series})


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def normalize_streamflow(payload):
    """Defensively find time/value records in the experimental NOAA response."""
    rows = []
    time_keys = ("validTime", "valid_time", "time", "forecastTime", "forecast_time", "datetime")
    flow_keys = ("streamflow", "flow", "value", "discharge")
    for rec in _walk(payload):
        t = next((rec.get(k) for k in time_keys if rec.get(k) is not None), None)
        q = next((rec.get(k) for k in flow_keys if rec.get(k) is not None), None)
        if t is None or q is None:
            continue
        try:
            q = float(q)
        except (TypeError, ValueError):
            continue
        ts = pd.to_datetime(t, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append({"time": ts, "streamflow": q})

    if not rows:
        return pd.DataFrame(columns=["time", "streamflow"])
    return pd.DataFrame(rows).drop_duplicates().sort_values("time").reset_index(drop=True)


def normalize_rating_points(payload):
    """Extract stage/flow pairs from NOAA's nested rating response."""
    rows = []
    stage_keys = ("stage", "stageValue", "gageHeight", "height")
    flow_keys = ("flow", "flowValue", "discharge", "streamflow")
    for rec in _walk(payload):
        h = next((rec.get(k) for k in stage_keys if rec.get(k) is not None), None)
        q = next((rec.get(k) for k in flow_keys if rec.get(k) is not None), None)
        if h is None or q is None:
            continue
        try:
            h, q = float(h), float(q)
        except (TypeError, ValueError):
            continue
        if q >= 0:
            rows.append({"stage_ft": h, "flow_cfs": q})

    if not rows:
        return pd.DataFrame(columns=["stage_ft", "flow_cfs"])
    return pd.DataFrame(rows).drop_duplicates().sort_values("flow_cfs").reset_index(drop=True)
