from __future__ import annotations

from pathlib import Path
import json
import math
import time
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, box

HGAC_BOUNDARY_URL = (
    "https://services1.arcgis.com/Z6SBWLWGRRejblAA/ArcGIS/rest/services/"
    "HGAC_Counties_Sea_Sever/FeatureServer/0/query"
)
HCFCD_CHANNELS_URL = (
    "https://services2.arcgis.com/nLl0k0Mja5hnSeSl/ArcGIS/rest/services/"
    "Drainage_Network/FeatureServer/0/query"
)
HCFCD_WATERSHEDS_URL = (
    "https://services2.arcgis.com/nLl0k0Mja5hnSeSl/arcgis/rest/services/"
    "Regional_Watersheds/FeatureServer/0/query"
)
NHDPLUS_FLOWLINE_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/3/query"
)

HEADERS = {"User-Agent": "HGAC-Flood-Intelligence-V5/1.0"}


def _request_json(url, params, retries=4, timeout=90):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"{r.status_code} from {r.url}")
                time.sleep(min(2 ** attempt, 8))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise
    raise last


def _query_geojson(url, where="1=1", out_fields="*", geometry=None, geometry_type=None,
                   result_record_count=2000, max_pages=100):
    """Page through an ArcGIS REST query returning one WGS84 GeoDataFrame."""
    features = []
    offset = 0
    for _ in range(max_pages):
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": result_record_count,
        }
        if geometry is not None:
            params["geometry"] = geometry
            params["geometryType"] = geometry_type or "esriGeometryEnvelope"
            params["spatialRel"] = "esriSpatialRelIntersects"
            params["inSR"] = 4326
        payload = _request_json(url, params)
        chunk = payload.get("features", [])
        features.extend(chunk)
        if len(chunk) < result_record_count:
            break
        offset += len(chunk)
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def build_regional_catalog(cache_dir: str | Path, force=False):
    """Cache H-GAC boundary, HCFCD major channels, and HCFCD watersheds.

    HCFCD drainage-network coverage is primarily Harris County. V5 therefore
    uses USGS NHDPlus HR as an on-demand click fallback elsewhere in H-GAC.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    boundary_path = cache / "hgac_counties.geojson"
    channels_path = cache / "hcfcd_major_channels.geojson"
    watersheds_path = cache / "hcfcd_regional_watersheds.geojson"
    nhd_path = cache / "nhdplus_major_named_streams.geojson"

    if force or not boundary_path.exists():
        boundary = _query_geojson(HGAC_BOUNDARY_URL)
        if boundary.empty:
            raise RuntimeError("H-GAC regional boundary service returned no features.")
        boundary.to_file(boundary_path, driver="GeoJSON")
    else:
        boundary = gpd.read_file(boundary_path).to_crs(4326)

    if force or not channels_path.exists():
        channels = _query_geojson(
            HCFCD_CHANNELS_URL,
            where="ChanType = 'Major' AND CHAN_NAME IS NOT NULL",
            out_fields="OBJECTID,CHAN_NAME,UNIT_NO,FNAME,TNAME,ChanType",
        )
        if not channels.empty:
            channels["display_name"] = channels["CHAN_NAME"].fillna(channels.get("FNAME"))
            channels["source"] = "HCFCD"
            channels.to_file(channels_path, driver="GeoJSON")
    else:
        channels = gpd.read_file(channels_path).to_crs(4326)

    if force or not watersheds_path.exists():
        watersheds = _query_geojson(HCFCD_WATERSHEDS_URL)
        if not watersheds.empty:
            watersheds.to_file(watersheds_path, driver="GeoJSON")
    else:
        watersheds = gpd.read_file(watersheds_path).to_crs(4326)

    # Regional visual catalog: cache only named, larger NHDPlus HR flowlines.
    # This keeps the first H-GAC map useful outside Harris County without
    # trying to load every minor ditch/flowline in the 13-county region.
    if force or not nhd_path.exists():
        try:
            minx, miny, maxx, maxy = boundary.total_bounds
            env = f"{minx},{miny},{maxx},{maxy}"
            nhd = _query_geojson(
                NHDPLUS_FLOWLINE_URL,
                where="gnis_name IS NOT NULL AND streamorde >= 3 AND ftype IN (460,558,336,468)",
                out_fields=(
                    "OBJECTID,gnis_name,reachcode,nhdplusid,streamorde,streamleve,"
                    "totdasqkm,qema,qama,slope,ftype,fcode"
                ),
                geometry=env, geometry_type="esriGeometryEnvelope",
                result_record_count=2000, max_pages=100,
            )
            if not nhd.empty:
                # Keep only segments intersecting the actual H-GAC counties.
                region_geom = boundary.geometry.union_all()
                nhd = nhd[nhd.geometry.intersects(region_geom)].copy()
                nhd["display_name"] = nhd["gnis_name"].astype(str)
                nhd["source"] = "USGS NHDPlus HR"
                nhd.to_file(nhd_path, driver="GeoJSON")
        except Exception:
            # The on-click NHD query remains available even if the optional
            # regional visualization cache cannot be built.
            nhd = gpd.GeoDataFrame(geometry=[], crs=4326)
    else:
        nhd = gpd.read_file(nhd_path).to_crs(4326)

    meta = {
        "boundary": str(boundary_path),
        "channels": str(channels_path),
        "watersheds": str(watersheds_path),
        "nhd_major_streams": str(nhd_path),
        "boundary_count": int(len(boundary)),
        "channel_count": int(len(channels)),
        "watershed_count": int(len(watersheds)),
        "nhd_major_stream_count": int(len(nhd)) if "nhd" in locals() else 0,
        "note": "HCFCD channels are primary where available; named NHDPlus HR major streams provide regional context and on-click fallback.",
    }
    (cache / "regional_catalog.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_regional_catalog(cache_dir: str | Path):
    cache = Path(cache_dir)
    build_regional_catalog(cache, force=False)
    boundary = gpd.read_file(cache / "hgac_counties.geojson").to_crs(4326)
    chp = cache / "hcfcd_major_channels.geojson"
    wsp = cache / "hcfcd_regional_watersheds.geojson"
    nhp = cache / "nhdplus_major_named_streams.geojson"
    channels = gpd.read_file(chp).to_crs(4326) if chp.exists() else gpd.GeoDataFrame(geometry=[], crs=4326)
    watersheds = gpd.read_file(wsp).to_crs(4326) if wsp.exists() else gpd.GeoDataFrame(geometry=[], crs=4326)
    nhd = gpd.read_file(nhp).to_crs(4326) if nhp.exists() else gpd.GeoDataFrame(geometry=[], crs=4326)
    return boundary, channels, watersheds, nhd


def point_in_hgac(lon: float, lat: float, boundary: gpd.GeoDataFrame) -> bool:
    if boundary.empty:
        return False
    p = Point(float(lon), float(lat))
    return bool(boundary.geometry.intersects(p).any())


def _nearest(gdf: gpd.GeoDataFrame, lon: float, lat: float):
    if gdf.empty:
        return None, math.inf
    local_crs = 32615  # appropriate for the Houston-Galveston region
    gg = gdf.to_crs(local_crs)
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(local_crs).iloc[0]
    distances = gg.geometry.distance(pt)
    idx = distances.idxmin()
    return gdf.loc[idx], float(distances.loc[idx])


def nearest_hcfcd_channel(channels, lon, lat, max_distance_m=1500.0):
    row, dist = _nearest(channels, lon, lat)
    if row is None or dist > float(max_distance_m):
        return None
    name = str(row.get("CHAN_NAME") or row.get("display_name") or row.get("FNAME") or "Unnamed channel")
    geom = row.geometry
    if "CHAN_NAME" in channels.columns and row.get("CHAN_NAME") is not None:
        same = channels[channels["CHAN_NAME"].astype(str) == str(row.get("CHAN_NAME"))]
        if not same.empty:
            geom = same.geometry.union_all()
    return {
        "source": "HCFCD",
        "name": name,
        "unit_no": str(row.get("UNIT_NO") or ""),
        "distance_m": dist,
        "geometry": geom,
        "properties": {k: row.get(k) for k in row.index if k != "geometry"},
    }


def query_nhdplus_near_click(lon: float, lat: float, radius_m=5000.0, min_stream_order=1):
    # Approximate lon/lat envelope, then compute true metric distance locally.
    lat_delta = float(radius_m) / 111_320.0
    lon_delta = float(radius_m) / (111_320.0 * max(math.cos(math.radians(lat)), 0.2))
    env = f"{lon-lon_delta},{lat-lat_delta},{lon+lon_delta},{lat+lat_delta}"
    where = f"gnis_name IS NOT NULL AND streamorde >= {int(min_stream_order)} AND ftype IN (460,558,336,468)"
    nhd = _query_geojson(
        NHDPLUS_FLOWLINE_URL,
        where=where,
        out_fields=(
            "OBJECTID,gnis_name,reachcode,nhdplusid,streamorde,streamleve,"
            "totdasqkm,qema,qama,slope,ftype,fcode"
        ),
        geometry=env,
        geometry_type="esriGeometryEnvelope",
    )
    if nhd.empty:
        return nhd
    nhd["source"] = "USGS NHDPlus HR"
    return nhd


def select_channel_from_click(boundary, hcfcd_channels, lon, lat,
                              hcfcd_max_distance_m=1500.0,
                              nhd_search_radius_m=5000.0):
    if not point_in_hgac(lon, lat, boundary):
        raise ValueError("The click is outside the configured H-GAC service-region boundary.")

    primary = nearest_hcfcd_channel(hcfcd_channels, lon, lat, hcfcd_max_distance_m)
    if primary is not None:
        primary["click_lon"] = float(lon)
        primary["click_lat"] = float(lat)
        return primary

    nhd = query_nhdplus_near_click(lon, lat, radius_m=nhd_search_radius_m, min_stream_order=1)
    row, dist = _nearest(nhd, lon, lat)
    if row is None or dist > nhd_search_radius_m:
        raise RuntimeError("No named HCFCD/NHDPlus channel was found near that click.")
    selected_name = str(row.get("gnis_name") or "Unnamed stream")
    same = nhd[nhd["gnis_name"].astype(str) == selected_name] if "gnis_name" in nhd.columns else nhd.iloc[0:0]
    selected_geom = same.geometry.union_all() if not same.empty else row.geometry
    return {
        "source": "USGS NHDPlus HR",
        "name": selected_name,
        "unit_no": "",
        "distance_m": dist,
        "reachcode": str(row.get("reachcode") or ""),
        "nhdplus_hr_id": str(row.get("nhdplusid") or ""),
        "stream_order": int(row.get("streamorde")) if pd.notna(row.get("streamorde")) else None,
        "total_drainage_area_km2": float(row.get("totdasqkm")) if pd.notna(row.get("totdasqkm")) else None,
        "geometry": selected_geom,
        "click_lon": float(lon),
        "click_lat": float(lat),
        "properties": {k: row.get(k) for k in row.index if k != "geometry"},
    }


def watershed_for_click(watersheds: gpd.GeoDataFrame, lon: float, lat: float):
    if watersheds.empty:
        return None
    p = Point(float(lon), float(lat))
    hit = watersheds[watersheds.geometry.intersects(p)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    name = row.get("WTSHNAME") or row.get("WTSHUNIT") or "Selected watershed"
    return {"name": str(name), "geometry": row.geometry, "properties": row.drop(labels=["geometry"]).to_dict()}
