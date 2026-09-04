from __future__ import annotations

from pathlib import Path
import math
import time

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from .nldi import get_basin
from .usgs import fetch_iv, latest_observation

MONITORING_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items"
HEADERS = {"User-Agent": "HGAC-Flood-Intelligence-V5/1.0"}


def discover_usgs_stream_gauges(lon: float, lat: float, radius_km=30.0, limit=250):
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
    bbox = f"{lon-lon_delta},{lat-lat_delta},{lon+lon_delta},{lat+lat_delta}"
    params = {
        "bbox": bbox,
        "f": "json",
        "limit": int(limit),
        "agency_code": "USGS",
    }
    r = requests.get(MONITORING_URL, params=params, headers=HEADERS, timeout=90)
    r.raise_for_status()
    payload = r.json()
    feats = payload.get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    gdf = gpd.GeoDataFrame.from_features(feats, crs=4326)
    # Stream, stream-stage, and streamflow sites normally use ST* codes.
    if "site_type_code" in gdf.columns:
        gdf = gdf[gdf["site_type_code"].astype(str).str.startswith("ST")].copy()
    if gdf.empty:
        return gdf
    local = gdf.to_crs(32615)
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32615).iloc[0]
    gdf["distance_to_click_m"] = local.geometry.distance(pt).to_numpy()
    gdf = gdf.sort_values("distance_to_click_m").reset_index(drop=True)
    return gdf


def _site_number(row):
    val = row.get("monitoring_location_number") or row.get("id") or ""
    val = str(val)
    return val.split("-")[-1]


def _norm_waterway_name(value):
    import re
    text = str(value or "").lower()
    text = text.replace("creek", "ck").replace("river", "rv")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stop = {"at", "near", "nr", "above", "below", "houston", "tx", "texas", "street", "st", "road", "rd", "drive", "dr"}
    return " ".join(tok for tok in text.split() if tok not in stop)


def _same_waterway(selected_name, gauge_name):
    a = _norm_waterway_name(selected_name)
    b = _norm_waterway_name(gauge_name)
    if not a or not b:
        return False
    # Strong exact phrase match first (e.g., Buffalo Bayou).
    if a in b or b in a:
        return True
    at = set(a.split())
    bt = set(b.split())
    return len(at & bt) >= max(1, min(2, len(at)))


def rank_hydraulically_relevant_gauges(
    lon: float,
    lat: float,
    selected_channel_geom=None,
    selected_channel_name=None,
    radius_km=30.0,
    max_candidates_to_test=20,
    cache_root: str | Path | None = None,
):
    """Return gauge candidates ranked by whether their upstream basin contains the click.

    The crucial rule is hydraulic relevance, not geographic proximity. For the
    nearest few USGS streamgages, V5 asks NLDI for each upstream basin and favors
    gauges whose basin contains the selected bayou click. It then checks that
    recent discharge/stage are actually available.
    """
    gauges = discover_usgs_stream_gauges(lon, lat, radius_km=radius_km)
    if gauges.empty:
        return []

    # Prefer gauges explicitly named for the selected waterway before nearby
    # tributary/water-quality-only sites. Distance alone is a poor ranking in
    # dense urban gauge networks such as Buffalo/Whiteoak Bayou.
    gauges = gauges.copy()
    gauges["same_waterway_name"] = gauges.apply(
        lambda r: _same_waterway(
            selected_channel_name,
            r.get("monitoring_location_name") or r.get("name") or "",
        ),
        axis=1,
    )
    gauges = gauges.sort_values(
        ["same_waterway_name", "distance_to_click_m"],
        ascending=[False, True],
    ).reset_index(drop=True)

    click = Point(lon, lat)
    results = []
    cache_root = Path(cache_root) if cache_root else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    for i, row in gauges.head(int(max_candidates_to_test)).iterrows():
        site = _site_number(row)
        item = {
            "site_no": site,
            "name": str(row.get("monitoring_location_name") or site),
            "longitude": float(row.geometry.x),
            "latitude": float(row.geometry.y),
            "distance_to_click_m": float(row.get("distance_to_click_m", math.nan)),
            "drainage_area_sqmi": float(row.get("drainage_area")) if pd.notna(row.get("drainage_area")) else None,
            "contributing_drainage_area_sqmi": float(row.get("contributing_drainage_area")) if pd.notna(row.get("contributing_drainage_area")) else None,
            "altitude_ft": float(row.get("altitude")) if pd.notna(row.get("altitude")) else None,
            "altitude_accuracy_ft": float(row.get("altitude_accuracy")) if pd.notna(row.get("altitude_accuracy")) else None,
            "vertical_datum": str(row.get("vertical_datum") or ""),
            "vertical_datum_name": str(row.get("vertical_datum_name") or ""),
            "same_waterway_name": bool(row.get("same_waterway_name", False)),
            "selected_waterway_named": bool(_norm_waterway_name(selected_channel_name)),
            "basin_contains_click": False,
            "recent_data_ok": False,
            "auto_supported": False,
            "basin_path": None,
            "error": None,
        }
        try:
            basin = get_basin(site, simplified=False, split_catchment=False).to_crs(4326)
            item["basin_contains_click"] = bool(basin.geometry.intersects(click).any())
            if cache_root is not None:
                bp = cache_root / f"usgs_{site}_basin.geojson"
                basin.to_file(bp, driver="GeoJSON")
                item["basin_path"] = str(bp)
        except Exception as exc:
            item["error"] = f"NLDI basin: {exc}"

        try:
            cp = str(cache_root / f"usgs_{site}_iv.csv") if cache_root is not None else None
            iv = fetch_iv(site, "P2D", cache_path=cp, allow_legacy_fallback=True)
            obs = latest_observation(iv)
            item["recent_data_ok"] = bool(obs and obs.get("discharge_cfs") is not None)
            item["latest_observation"] = obs
        except Exception as exc:
            item["recent_data_ok"] = False
            if item["error"]:
                item["error"] += f"; IV: {exc}"
            else:
                item["error"] = f"IV: {exc}"

        # A basin-scale connection alone is not enough in dense urban networks.
        # When the selected channel has a usable name, automatic control MUST be
        # on the same named waterway. This prevents Buffalo Bayou selections from
        # silently accepting a Brays/Whiteoak tributary gauge merely because the
        # larger NLDI basin relationship happens to pass.
        same_name_required = bool(_norm_waterway_name(selected_channel_name))
        item["auto_supported"] = bool(
            item["basin_contains_click"]
            and item["recent_data_ok"]
            and (item["same_waterway_name"] or not same_name_required)
        )

        # Strongly favor downstream gauges whose basin contains the selected point.
        item["score"] = (
            (0 if item["basin_contains_click"] else 1_000_000)
            + (0 if item["recent_data_ok"] else 250_000)
            + (0 if item["same_waterway_name"] else 5_000_000)
            + item["distance_to_click_m"]
        )
        results.append(item)

    results.sort(key=lambda x: x["score"])
    return results


def choose_best_gauge(*args, **kwargs):
    rows = rank_hydraulically_relevant_gauges(*args, **kwargs)
    if not rows:
        return None, []
    supported = [r for r in rows if r.get("auto_supported")]
    return (supported[0] if supported else None), rows
