from __future__ import annotations

"""Supplemental data collection for the simple V5.2 regional workflow.

These products are deliberately non-blocking.  The hydraulic screen can still
finish if historical daily flow, rainfall, or Flood Hub is temporarily
unavailable.  Each source writes its own CSV/JSON plus a summary explaining
what succeeded and what did not.
"""

from pathlib import Path
import json
import math
import re
from datetime import timedelta

import pandas as pd
import requests

from .floodhub import FloodHubClient

USGS_DAILY = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
NWS_POINTS = "https://api.weather.gov/points/{lat:.5f},{lon:.5f}"
HEADERS = {
    "User-Agent": "HGAC-Regional-Flood-Intelligence-V5.2 (government research prototype)",
    "Accept": "application/geo+json, application/json",
}


def _next_link(payload):
    for link in payload.get("links", []) or []:
        if str(link.get("rel", "")).lower() == "next" and link.get("href"):
            return link["href"]
    return None


def _get_json(url, params=None, timeout=60):
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_usgs_daily(site_no: str, parameter_code="00060", statistic_id="00003", max_pages=20):
    """Fetch the available daily-value record for a USGS site.

    For discharge, statistic 00003 is daily mean.  The daily endpoint is used
    because decades of 15-minute continuous values are unnecessarily large for
    a trend panel.
    """
    params = {
        "f": "json",
        "monitoring_location_id": f"USGS-{site_no}",
        "parameter_code": str(parameter_code),
        "limit": 50000,
    }
    if statistic_id:
        params["statistic_id"] = str(statistic_id)
    url = USGS_DAILY
    rows = []
    page_params = params
    pages = 0
    while url and pages < int(max_pages):
        payload = _get_json(url, page_params, timeout=90)
        for feat in payload.get("features", []) or []:
            p = feat.get("properties", {}) or {}
            value = pd.to_numeric(p.get("value"), errors="coerce")
            if pd.isna(value):
                continue
            when = p.get("time") or p.get("date") or p.get("datetime")
            rows.append({
                "time": when,
                "value": float(value),
                "parameter_code": p.get("parameter_code"),
                "statistic_id": p.get("statistic_id"),
                "unit": p.get("unit_of_measure") or p.get("unit"),
            })
        url = _next_link(payload)
        page_params = None
        pages += 1
    if not rows:
        return pd.DataFrame(columns=["time", "value", "parameter_code", "statistic_id", "unit"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df


def _duration_seconds(text: str) -> float:
    """Small ISO-8601 duration parser for NWS validTime strings."""
    if not text:
        return 0.0
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", str(text))
    if not m:
        return 0.0
    d, h, mi, s = m.groups()
    return (float(d or 0) * 86400 + float(h or 0) * 3600 + float(mi or 0) * 60 + float(s or 0))


def fetch_nws_qpf(lat: float, lon: float):
    """Fetch point-based NWS quantitative precipitation forecast grid values."""
    point = _get_json(NWS_POINTS.format(lat=float(lat), lon=float(lon)), timeout=45)
    props = point.get("properties", {}) or {}
    grid_url = props.get("forecastGridData")
    if not grid_url:
        raise RuntimeError("NWS point metadata did not provide forecastGridData.")
    grid = _get_json(grid_url, timeout=60)
    gp = grid.get("properties", {}) or {}
    qpf = (gp.get("quantitativePrecipitation") or {}).get("values", []) or []
    pop = (gp.get("probabilityOfPrecipitation") or {}).get("values", []) or []

    rows = []
    for item in qpf:
        vt = str(item.get("validTime") or "")
        if "/" in vt:
            start_s, dur_s = vt.split("/", 1)
        else:
            start_s, dur_s = vt, ""
        start = pd.to_datetime(start_s, utc=True, errors="coerce")
        val = pd.to_numeric(item.get("value"), errors="coerce")
        if pd.isna(start) or pd.isna(val):
            continue
        sec = _duration_seconds(dur_s)
        rows.append({
            "time": start,
            "duration_hours": sec / 3600.0 if sec else None,
            "precip_mm": float(val),
            "precip_in": float(val) / 25.4,
        })
    out = pd.DataFrame(rows)

    pop_rows = []
    for item in pop:
        vt = str(item.get("validTime") or "")
        start_s = vt.split("/", 1)[0]
        start = pd.to_datetime(start_s, utc=True, errors="coerce")
        val = pd.to_numeric(item.get("value"), errors="coerce")
        if pd.isna(start) or pd.isna(val):
            continue
        pop_rows.append({"time": start, "probability_percent": float(val)})
    pop_df = pd.DataFrame(pop_rows)
    return out, pop_df, {"grid_url": grid_url, "forecast_office": props.get("forecastOffice")}


def _fh_location(g):
    loc = g.get("location") or g.get("latLng") or g.get("latlng") or {}
    lat = loc.get("latitude") if isinstance(loc, dict) else None
    lon = loc.get("longitude") if isinstance(loc, dict) else None
    if lat is None:
        lat = g.get("latitude")
    if lon is None:
        lon = g.get("longitude")
    try:
        return float(lat), float(lon)
    except Exception:
        return None, None


def fetch_floodhub_near(api_key: str, lat: float, lon: float, selected_name: str = "", padding_deg=0.15):
    """Find a nearby Flood Hub gauge and retrieve forecast/status when available.

    Flood Hub remains supplemental; inability to match a gauge never stops the
    local USGS/NWM/LiDAR workflow.
    """
    if not str(api_key or "").strip():
        return {"configured": False, "matched": False, "message": "FLOODHUB_API_KEY not configured"}, pd.DataFrame()
    client = FloodHubClient(str(api_key).strip())
    p = float(padding_deg)
    vertices = [
        (lat-p, lon-p), (lat-p, lon+p), (lat+p, lon+p), (lat+p, lon-p), (lat-p, lon-p)
    ]
    payload = client.search_gauges_by_loop(vertices, include_non_quality_verified=True, page_size=5000)
    gauges = payload.get("gauges") or payload.get("results") or payload.get("items") or []
    if isinstance(gauges, dict):
        gauges = list(gauges.values())
    if not gauges:
        return {"configured": True, "matched": False, "message": "No Flood Hub gauge returned near selected bayou"}, pd.DataFrame()

    name_tokens = {t for t in re.sub(r"[^a-z0-9]+", " ", selected_name.lower()).split() if len(t) > 2}
    ranked = []
    for g in gauges:
        glat, glon = _fh_location(g)
        if glat is None:
            dist = 1e9
        else:
            dist = math.hypot((glat-lat)*111.32, (glon-lon)*111.32*max(math.cos(math.radians(lat)), .2))
        gname = " ".join(str(g.get(k) or "") for k in ["siteName", "river", "displayName", "name"]).lower()
        match = sum(1 for t in name_tokens if t in gname)
        ranked.append((0 if match else 1, dist, g))
    ranked.sort(key=lambda x: (x[0], x[1]))
    gauge = ranked[0][2]
    gid = gauge.get("gaugeId") or gauge.get("id") or gauge.get("name")
    if not gid:
        return {"configured": True, "matched": False, "message": "Flood Hub gauge response had no gauge ID"}, pd.DataFrame()

    info = {"configured": True, "matched": True, "gauge_id": str(gid), "gauge": gauge}
    try:
        info["flood_status"] = client.query_latest_flood_status([gid])
    except Exception as exc:
        info["flood_status_error"] = str(exc)
    try:
        fdf = client.normalize_forecasts(client.query_forecasts([gid]))
    except Exception as exc:
        info["forecast_error"] = str(exc)
        fdf = pd.DataFrame()
    return info, fdf


def collect_intelligence_bundle(workspace: str | Path, cfg: dict, selection: dict, gauge: dict):
    ws = Path(workspace)
    out = ws / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"gauge": {}, "rainfall": {}, "floodhub": {}}

    # Full period-of-record daily mean streamflow: compact enough for long-term trends.
    try:
        flow = fetch_usgs_daily(str(gauge["site_no"]), "00060", "00003")
        flow.to_csv(out / "historical_daily_streamflow.csv", index=False)
        summary["gauge"]["daily_streamflow_records"] = int(len(flow))
        if not flow.empty:
            summary["gauge"]["record_start"] = flow["time"].min().isoformat()
            summary["gauge"]["record_end"] = flow["time"].max().isoformat()
            summary["gauge"]["mean_daily_cfs"] = float(flow["value"].mean())
            summary["gauge"]["max_daily_cfs"] = float(flow["value"].max())
    except Exception as exc:
        summary["gauge"]["history_error"] = str(exc)

    # Historical rainfall when that same USGS site actually measures precipitation.
    try:
        precip = fetch_usgs_daily(str(gauge["site_no"]), "00045", None)
        precip.to_csv(out / "historical_daily_precipitation.csv", index=False)
        summary["rainfall"]["historical_records"] = int(len(precip))
    except Exception as exc:
        summary["rainfall"]["history_error"] = str(exc)

    # Future rainfall at the selected bayou point from NWS forecast grid data.
    try:
        qpf, pop, nws_meta = fetch_nws_qpf(float(selection["click_lat"]), float(selection["click_lon"]))
        qpf.to_csv(out / "nws_forecast_precipitation.csv", index=False)
        pop.to_csv(out / "nws_forecast_precip_probability.csv", index=False)
        summary["rainfall"].update({
            "forecast_records": int(len(qpf)),
            "forecast_total_in": float(qpf["precip_in"].sum()) if not qpf.empty else 0.0,
            **nws_meta,
        })
    except Exception as exc:
        summary["rainfall"]["forecast_error"] = str(exc)

    # Supplemental Google Flood Hub forecast/status, only when key is configured.
    try:
        fh, fdf = fetch_floodhub_near(
            cfg.get("secrets", {}).get("floodhub_api_key", ""),
            float(selection["click_lat"]), float(selection["click_lon"]),
            selected_name=str(selection.get("name") or ""),
            padding_deg=float(cfg.get("floodhub", {}).get("search_padding_deg", 0.10)),
        )
        summary["floodhub"] = fh
        # Keep raw summary human-readable but avoid embedding a very large gauge object twice.
        if not fdf.empty:
            fdf.to_csv(out / "floodhub_forecast.csv", index=False)
            summary["floodhub"]["forecast_rows"] = int(len(fdf))
        (out / "floodhub_summary.json").write_text(json.dumps(fh, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        summary["floodhub"] = {"configured": bool(cfg.get("secrets", {}).get("floodhub_api_key")), "error": str(exc)}

    (out / "data_sources_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary

# ---------------------------------------------------------------------------
# V5.3 Flood Hub enrichment
# ---------------------------------------------------------------------------

def _find_flood_status_obj(obj, gauge_id=None):
    """Recursively find a FloodStatus-like dict in variable API wrappers."""
    if isinstance(obj, dict):
        if "severity" in obj and ("inundationMapSet" in obj or "gaugeId" in obj):
            if gauge_id is None or str(obj.get("gaugeId", "")) in {"", str(gauge_id)}:
                return obj
        for v in obj.values():
            hit = _find_flood_status_obj(v, gauge_id)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_flood_status_obj(v, gauge_id)
            if hit is not None:
                return hit
    return None


def _kml_to_geojson_features(kml: str, properties: dict | None = None):
    """Parse simple KML Polygon coordinate rings from Flood Hub serialized polygons."""
    import xml.etree.ElementTree as ET
    props = dict(properties or {})
    feats = []
    try:
        root = ET.fromstring(kml)
    except Exception:
        return feats
    for elem in root.iter():
        if str(elem.tag).lower().endswith("coordinates") and (elem.text or "").strip():
            coords = []
            for token in elem.text.strip().replace("\n", " ").split():
                parts = token.split(",")
                if len(parts) < 2:
                    continue
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except Exception:
                    pass
            if len(coords) >= 4:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                feats.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                })
    return feats


def fetch_floodhub_near(api_key: str, lat: float, lon: float, selected_name: str = "", padding_deg=0.15):
    """V5.3: retrieve Flood Hub gauge, model metadata, forecast/status and inundation polygons."""
    if not str(api_key or "").strip():
        return {"configured": False, "matched": False, "message": "FLOODHUB_API_KEY not configured"}, pd.DataFrame(), []
    client = FloodHubClient(str(api_key).strip())
    p = float(padding_deg)
    vertices = [(lat-p, lon-p), (lat-p, lon+p), (lat+p, lon+p), (lat+p, lon-p), (lat-p, lon-p)]
    payload = client.search_gauges_by_loop(vertices, include_non_quality_verified=True, page_size=5000)
    gauges = payload.get("gauges") or payload.get("results") or payload.get("items") or []
    if isinstance(gauges, dict):
        gauges = list(gauges.values())
    if not gauges:
        return {"configured": True, "matched": False, "message": "No Flood Hub gauge returned near selected bayou"}, pd.DataFrame(), []

    name_tokens = {t for t in re.sub(r"[^a-z0-9]+", " ", selected_name.lower()).split() if len(t) > 2}
    ranked = []
    for g in gauges:
        glat, glon = _fh_location(g)
        if glat is None:
            dist = 1e9
        else:
            dist = math.hypot((glat-lat)*111.32, (glon-lon)*111.32*max(math.cos(math.radians(lat)), .2))
        gname = " ".join(str(g.get(k) or "") for k in ["siteName", "river", "displayName", "name"]).lower()
        match = sum(1 for t in name_tokens if t in gname)
        ranked.append((0 if match else 1, dist, g))
    ranked.sort(key=lambda x: (x[0], x[1]))
    gauge = ranked[0][2]
    gid = gauge.get("gaugeId") or gauge.get("id") or gauge.get("name")
    if not gid:
        return {"configured": True, "matched": False, "message": "Flood Hub gauge response had no gauge ID"}, pd.DataFrame(), []

    info = {
        "configured": True,
        "matched": True,
        "gauge_id": str(gid),
        "gauge_name": gauge.get("siteName") or gauge.get("name"),
        "river": gauge.get("river"),
        "gauge_quality_verified": gauge.get("qualityVerified"),
        "has_model": gauge.get("hasModel"),
    }
    try:
        model = client.get_gauge_model(gid)
        info["model"] = model
        info["model_quality_verified"] = model.get("qualityVerified")
        info["gauge_value_unit"] = model.get("gaugeValueUnit")
        info["thresholds"] = model.get("thresholds") or {}
    except Exception as exc:
        info["model_error"] = str(exc)
        model = {}

    try:
        status_payload = client.query_latest_flood_status([gid])
        status = _find_flood_status_obj(status_payload, gid)
        info["flood_status"] = status or status_payload
        if status:
            info["severity"] = status.get("severity")
            info["forecast_trend"] = status.get("forecastTrend")
            info["status_quality_verified"] = status.get("qualityVerified")
            info["map_inference_type"] = status.get("mapInferenceType")
    except Exception as exc:
        info["flood_status_error"] = str(exc)
        status = None

    try:
        fdf = client.normalize_forecasts(client.query_forecasts([gid]))
        if not fdf.empty:
            unit = str(info.get("gauge_value_unit") or "")
            if unit == "CUBIC_METERS_PER_SECOND":
                fdf["value_cfs"] = pd.to_numeric(fdf["value"], errors="coerce") * 35.3146667215
                info["forecast_peak_cfs"] = float(fdf["value_cfs"].max())
            else:
                info["forecast_peak_value"] = float(pd.to_numeric(fdf["value"], errors="coerce").max())
            info["forecast_rows"] = int(len(fdf))
    except Exception as exc:
        info["forecast_error"] = str(exc)
        fdf = pd.DataFrame()

    # Google may return probability/depth inundation polygons with a FloodStatus.
    map_features = []
    if status:
        imset = status.get("inundationMapSet") or {}
        map_type = imset.get("inundationMapType")
        info["inundation_map_type"] = map_type
        for item in imset.get("inundationMaps", []) or []:
            pid = item.get("serializedPolygonId")
            if not pid:
                continue
            try:
                poly = client.get_serialized_polygon(pid)
                kml = poly.get("kml") or ""
                map_features.extend(_kml_to_geojson_features(kml, {
                    "source": "Google Flood Hub",
                    "gauge_id": str(gid),
                    "severity": info.get("severity"),
                    "map_type": map_type,
                    "level": item.get("level"),
                    "quality_verified": info.get("status_quality_verified"),
                }))
            except Exception as exc:
                info.setdefault("polygon_errors", []).append(str(exc))
        info["inundation_polygon_features"] = int(len(map_features))

    return info, fdf, map_features


def collect_intelligence_bundle(workspace: str | Path, cfg: dict, selection: dict, gauge: dict):
    """V5.3 supplemental bundle with explicit Flood Hub comparison products."""
    ws = Path(workspace)
    out = ws / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"gauge": {}, "rainfall": {}, "floodhub": {}}

    try:
        flow = fetch_usgs_daily(str(gauge["site_no"]), "00060", "00003")
        flow.to_csv(out / "historical_daily_streamflow.csv", index=False)
        summary["gauge"]["daily_streamflow_records"] = int(len(flow))
        if not flow.empty:
            summary["gauge"]["record_start"] = flow["time"].min().isoformat()
            summary["gauge"]["record_end"] = flow["time"].max().isoformat()
            summary["gauge"]["mean_daily_cfs"] = float(flow["value"].mean())
            summary["gauge"]["max_daily_cfs"] = float(flow["value"].max())
    except Exception as exc:
        summary["gauge"]["history_error"] = str(exc)

    try:
        precip = fetch_usgs_daily(str(gauge["site_no"]), "00045", None)
        precip.to_csv(out / "historical_daily_precipitation.csv", index=False)
        summary["rainfall"]["historical_records"] = int(len(precip))
    except Exception as exc:
        summary["rainfall"]["history_error"] = str(exc)

    try:
        qpf, pop, nws_meta = fetch_nws_qpf(float(selection["click_lat"]), float(selection["click_lon"]))
        qpf.to_csv(out / "nws_forecast_precipitation.csv", index=False)
        pop.to_csv(out / "nws_forecast_precip_probability.csv", index=False)
        summary["rainfall"].update({
            "forecast_records": int(len(qpf)),
            "forecast_total_in": float(qpf["precip_in"].sum()) if not qpf.empty else 0.0,
            **nws_meta,
        })
    except Exception as exc:
        summary["rainfall"]["forecast_error"] = str(exc)

    try:
        fh, fdf, map_features = fetch_floodhub_near(
            cfg.get("secrets", {}).get("floodhub_api_key", ""),
            float(selection["click_lat"]), float(selection["click_lon"]),
            selected_name=str(selection.get("name") or ""),
            padding_deg=float(cfg.get("floodhub", {}).get("search_padding_deg", 0.10)),
        )
        summary["floodhub"] = fh
        if not fdf.empty:
            fdf.to_csv(out / "floodhub_forecast.csv", index=False)
        else:
            (out / "floodhub_forecast.csv").unlink(missing_ok=True)
        if map_features:
            (out / "floodhub_inundation.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": map_features}, indent=2), encoding="utf-8")
        else:
            (out / "floodhub_inundation.geojson").unlink(missing_ok=True)

        # Simple independent-model comparison when Google's model value is discharge.
        nwm_path = out / "latest_nwm_hydrograph.csv"
        if nwm_path.exists() and fh.get("forecast_peak_cfs") is not None:
            nwm = pd.read_csv(nwm_path)
            nwm_peak = float(pd.to_numeric(nwm.get("streamflow"), errors="coerce").max())
            google_peak = float(fh["forecast_peak_cfs"])
            if np.isfinite(nwm_peak) and np.isfinite(google_peak) and max(nwm_peak, google_peak) > 0:
                fh["nwm_peak_cfs"] = nwm_peak
                fh["peak_difference_percent"] = 100.0 * (google_peak - nwm_peak) / max(nwm_peak, 1e-6)
                fh["forecast_agreement"] = "close" if abs(fh["peak_difference_percent"]) <= 25 else "divergent"
        (out / "floodhub_summary.json").write_text(json.dumps(fh, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        summary["floodhub"] = {"configured": bool(cfg.get("secrets", {}).get("floodhub_api_key")), "error": str(exc)}

    (out / "data_sources_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def refresh_live_supplemental(workspace: str | Path, cfg: dict, selection: dict, gauge: dict):
    """Refresh only fast-changing NWS/Flood Hub layers; preserve historical files."""
    ws = Path(workspace); out = ws / "outputs"; out.mkdir(parents=True, exist_ok=True)
    existing = {}
    psummary = out / "data_sources_summary.json"
    if psummary.exists():
        try: existing = json.loads(psummary.read_text(encoding="utf-8"))
        except Exception: existing = {}
    existing.setdefault("gauge", {}); existing.setdefault("rainfall", {}); existing.setdefault("floodhub", {})
    try:
        qpf, pop, nws_meta = fetch_nws_qpf(float(selection["click_lat"]), float(selection["click_lon"]))
        qpf.to_csv(out / "nws_forecast_precipitation.csv", index=False)
        pop.to_csv(out / "nws_forecast_precip_probability.csv", index=False)
        existing["rainfall"].update({"forecast_records": int(len(qpf)), "forecast_total_in": float(qpf["precip_in"].sum()) if not qpf.empty else 0.0, **nws_meta})
    except Exception as exc:
        existing["rainfall"]["forecast_error"] = str(exc)
    try:
        fh, fdf, map_features = fetch_floodhub_near(
            cfg.get("secrets", {}).get("floodhub_api_key", ""), float(selection["click_lat"]), float(selection["click_lon"]),
            selected_name=str(selection.get("name") or ""), padding_deg=float(cfg.get("floodhub", {}).get("search_padding_deg", 0.10)),
        )
        nwm_path = out / "latest_nwm_hydrograph.csv"
        if nwm_path.exists() and fh.get("forecast_peak_cfs") is not None:
            nwm = pd.read_csv(nwm_path); nwm_peak=float(pd.to_numeric(nwm.get("streamflow"),errors="coerce").max()); google_peak=float(fh["forecast_peak_cfs"])
            if np.isfinite(nwm_peak) and np.isfinite(google_peak) and max(nwm_peak,google_peak)>0:
                fh["nwm_peak_cfs"]=nwm_peak; fh["peak_difference_percent"]=100.0*(google_peak-nwm_peak)/max(nwm_peak,1e-6); fh["forecast_agreement"]="close" if abs(fh["peak_difference_percent"])<=25 else "divergent"
        existing["floodhub"] = fh
        if not fdf.empty: fdf.to_csv(out / "floodhub_forecast.csv", index=False)
        if map_features: (out / "floodhub_inundation.geojson").write_text(json.dumps({"type":"FeatureCollection","features":map_features},indent=2),encoding="utf-8")
        else: (out / "floodhub_inundation.geojson").unlink(missing_ok=True)
        (out / "floodhub_summary.json").write_text(json.dumps(fh,indent=2,default=str),encoding="utf-8")
    except Exception as exc:
        existing["floodhub"]={"configured":bool(cfg.get("secrets",{}).get("floodhub_api_key")),"error":str(exc)}
    psummary.write_text(json.dumps(existing,indent=2,default=str),encoding="utf-8")
    return existing
