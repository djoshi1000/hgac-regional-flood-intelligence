from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import yaml

import streamlit as st
import folium
from folium.plugins import Fullscreen, MeasureControl, MousePosition, MiniMap
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from src.regional_catalog import (
    build_regional_catalog,
    load_regional_catalog,
    select_channel_from_click,
    watershed_for_click,
)
from src.gauge_discovery import choose_best_gauge
from src.workspace import create_workspace, workspace_path
from src.pipeline_v5 import run_full_pipeline, run_live_workspace, static_ready
from src.dem_cache import build_virtual_mosaic, build_arcpy_mosaic_dataset
from src.config import load_config


# ============================================================================
# PAGE + THEME
# ============================================================================

st.set_page_config(
    page_title="H-GAC Regional Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --panel: rgba(11, 20, 32, 0.86);
  --panel2: rgba(17, 31, 48, 0.92);
  --line: rgba(78, 205, 196, .25);
  --accent: #49d6c7;
  --accent2: #5aa9ff;
  --muted: #9fb0c3;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 15% 0%, rgba(29,93,128,.25), transparent 28%),
    radial-gradient(circle at 95% 10%, rgba(26,128,119,.18), transparent 24%),
    #07111c;
}
[data-testid="stSidebar"] { background: #081420; border-right: 1px solid rgba(255,255,255,.06); }
.block-container { padding-top: 1.15rem; padding-bottom: 2.5rem; max-width: 1700px; }
.hero {
  padding: 1.25rem 1.35rem; border: 1px solid var(--line); border-radius: 18px;
  background: linear-gradient(120deg, rgba(12,27,42,.96), rgba(11,44,54,.88));
  box-shadow: 0 18px 50px rgba(0,0,0,.22); margin-bottom: 1rem;
}
.hero h1 { margin:0; font-size:2.15rem; letter-spacing:-.04em; }
.hero p { color:var(--muted); margin:.45rem 0 0; }
.metric-card {
  padding: .85rem 1rem; border-radius: 15px; background: var(--panel);
  border: 1px solid rgba(255,255,255,.07); min-height: 94px;
}
.metric-card .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-card .value { font-size:1.55rem; font-weight:750; margin-top:.18rem; }
.metric-card .sub { color:#7f93a8; font-size:.78rem; margin-top:.08rem; }
.stage-pill { display:inline-block; padding:.28rem .6rem; border-radius:999px; margin-right:.35rem;
  background:rgba(73,214,199,.11); border:1px solid rgba(73,214,199,.3); color:#b6fff7; font-size:.78rem; }
.small-note { color:#91a4b8; font-size:.82rem; }
div[data-testid="stStatusWidget"] { border-radius: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REGIONAL_CONFIG_PATH = PROJECT_ROOT / "config" / "v5_regional.yaml"
ROOT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ============================================================================
# CONFIG HELPERS
# ============================================================================


def _read_yaml(path: Path):
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _abs_project_path(value: str | Path):
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _regional_cfg():
    cfg = _read_yaml(REGIONAL_CONFIG_PATH)
    cfg.setdefault("regional", {})
    cfg.setdefault("dem_cache", {})
    cfg.setdefault("live", {})
    cfg.setdefault("ui", {})
    return cfg


def _base_workspace_settings():
    """Copy model settings, not pilot-specific paths, into a new bayou workspace."""
    raw = _read_yaml(ROOT_CONFIG_PATH)
    keep = [
        "usgs", "nldi", "nwm", "floodhub", "dem", "terrain", "bank_profile",
        "hydraulics", "inundation", "gauge_network",
    ]
    return {k: copy.deepcopy(raw[k]) for k in keep if k in raw}


def _patch_workspace_datum(config_path: Path, override_text: str):
    cfg = _read_yaml(config_path)
    cfg.setdefault("rating", {})
    text = str(override_text or "").strip()
    cfg["rating"]["gage_datum_navd88_ft_override"] = float(text) if text else None
    _write_yaml(config_path, cfg)


def _json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v, nd=2, suffix=""):
    try:
        if v is None or not np.isfinite(float(v)):
            return "—"
        return f"{float(v):,.{nd}f}{suffix}"
    except Exception:
        return "—"


def _dir_size(path: Path):
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _human_bytes(n):
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


RCFG = _regional_cfg()
CATALOG_DIR = _abs_project_path(RCFG["regional"].get("catalog_cache", "data/regional_cache/catalog"))
DEM_CACHE_DIR = _abs_project_path(RCFG["dem_cache"].get("cache_root", "data/regional_cache/dem"))

# Resolve the optional regional DEM path once for all workspaces.
regional_dem_path = str(RCFG["dem_cache"].get("regional_dem_path", "") or "").strip()
if regional_dem_path:
    RCFG["dem_cache"]["regional_dem_path"] = str(_abs_project_path(regional_dem_path))
RCFG["dem_cache"]["cache_root"] = str(DEM_CACHE_DIR)


# ============================================================================
# REGIONAL CATALOG
# ============================================================================

@st.cache_resource(show_spinner=False)
def _catalog(cache_dir_str: str):
    return load_regional_catalog(cache_dir_str)


try:
    with st.spinner("Loading H-GAC regional waterway catalog …"):
        boundary, hcfcd_channels, hcfcd_watersheds, nhd_major = _catalog(str(CATALOG_DIR))
except Exception as exc:
    st.error(f"Regional catalog could not be initialized: {exc}")
    st.info("Run `python scripts\\20_build_regional_catalog.py` once with internet access, then relaunch the app.")
    st.stop()


# ============================================================================
# SESSION STATE
# ============================================================================

for key, default in {
    "selected_channel": None,
    "selection_click": None,
    "gauge_candidates": [],
    "selected_gauge_site": None,
    "workspace": None,
    "workspace_config": None,
    "live_metadata": None,
    "last_pipeline_error": None,
    "last_auto_refresh_counter": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================================
# MAP HELPERS
# ============================================================================


def _add_geojson(m, path, name, style, show=True, tooltip_fields=None, tooltip_aliases=None):
    p = Path(path)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        tooltip = None
        if tooltip_fields:
            tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases or tooltip_fields)
        folium.GeoJson(
            data=data, name=name, show=show,
            style_function=lambda _f, s=style: s,
            tooltip=tooltip,
        ).add_to(m)
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def _raster_overlay_cached(path_str: str, mtime: float, min_value=0.02, max_size=1100, alpha=.76):
    del mtime
    path = Path(path_str)
    if not path.exists():
        return None
    import matplotlib
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        valid = np.isfinite(arr)
        if src.nodata is not None:
            valid &= arr != src.nodata
        valid &= arr >= float(min_value)
        if not valid.any():
            return None
        transform, width, height = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        scale = max(width / max_size, height / max_size, 1.0)
        dw, dh = max(1, int(width / scale)), max(1, int(height / scale))
        dtr = transform * rasterio.Affine.scale(width / dw, height / dh)
        dst = np.full((dh, dw), np.nan, dtype="float32")
        reproject(
            arr, dst, src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=dtr, dst_crs="EPSG:4326", dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    valid = np.isfinite(dst) & (dst >= float(min_value))
    if not valid.any():
        return None
    vals = dst[valid]
    vmax = max(float(np.nanpercentile(vals, 99)), float(min_value) * 1.1)
    norm = np.clip((dst - min_value) / max(vmax - min_value, 1e-9), 0, 1)
    rgba = matplotlib.colormaps["turbo"](norm)
    rgba[~valid, 3] = 0
    rgba[valid, 3] = alpha
    rgba = (rgba * 255).astype("uint8")
    west, north = dtr.c, dtr.f
    east = west + dtr.a * dw
    south = north + dtr.e * dh
    return {"image": rgba, "bounds": [[south, west], [north, east]], "vmax": vmax}


def _metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def _discovery_map():
    minx, miny, maxx, maxy = boundary.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=8, control_scale=True, tiles=None, prefer_canvas=True)
    folium.TileLayer("CartoDB dark_matter", name="Carto Dark", control=True, show=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map", control=True, show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Imagery", control=True, show=False,
    ).add_to(m)

    folium.GeoJson(
        boundary.__geo_interface__, name="H-GAC 13-county region",
        style_function=lambda _f: {"color": "#7ee7dc", "weight": 1.1, "fillColor": "#12384a", "fillOpacity": .11},
        tooltip=folium.GeoJsonTooltip(fields=[c for c in ["NAME", "name", "County"] if c in boundary.columns][:1]) if any(c in boundary.columns for c in ["NAME", "name", "County"]) else None,
    ).add_to(m)

    if not nhd_major.empty:
        folium.GeoJson(
            nhd_major.__geo_interface__, name="Regional named streams (NHDPlus HR)", show=True,
            style_function=lambda _f: {"color": "#4a8fd8", "weight": 1.25, "opacity": .55},
            tooltip=folium.GeoJsonTooltip(fields=["gnis_name"], aliases=["Waterway"]),
        ).add_to(m)

    if not hcfcd_channels.empty:
        folium.GeoJson(
            hcfcd_channels.__geo_interface__, name="HCFCD major channels", show=True,
            style_function=lambda _f: {"color": "#35e0cc", "weight": 2.1, "opacity": .78},
            tooltip=folium.GeoJsonTooltip(fields=["CHAN_NAME"], aliases=["Channel"]),
        ).add_to(m)

    sel = st.session_state.selected_channel
    if sel is not None:
        folium.GeoJson(
            gpd.GeoDataFrame({"name": [sel["name"]]}, geometry=[sel["geometry"]], crs=4326).__geo_interface__,
            name="Selected bayou", style_function=lambda _f: {"color": "#ffe26b", "weight": 6, "opacity": 1.0},
            tooltip=sel["name"],
        ).add_to(m)
        folium.CircleMarker(
            [sel["click_lat"], sel["click_lon"]], radius=6, color="#ffe26b", fill=True,
            fill_opacity=1, tooltip="Selection point",
        ).add_to(m)

    for g in st.session_state.gauge_candidates or []:
        ok = bool(g.get("basin_contains_click") and g.get("recent_data_ok"))
        folium.CircleMarker(
            [g["latitude"], g["longitude"]], radius=6 if ok else 4,
            color="#ff7d7d" if not ok else "#77ffb1", fill=True, fill_opacity=.9,
            tooltip=f"USGS {g['site_no']} — {g['name']}",
        ).add_to(m)

    Fullscreen(position="topright").add_to(m)
    MeasureControl(position="topright", primary_length_unit="meters").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Cursor").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return m


def _result_map(ws: Path, cfg: dict, meta: dict):
    p = ws / "data" / "processed"
    o = ws / "outputs"
    basin = p / "dem_upstream_basin.geojson"
    mainstem = p / "mainstem.geojson"
    channel = p / "selected_channel.geojson"
    extent = Path(meta.get("aoi_extent_geojson") or meta.get("extent_geojson") or o / "latest_inundation.geojson")
    depth = Path(meta.get("aoi_depth_raster") or meta.get("depth_raster") or o / "latest_depth.tif")

    center = [float(cfg["selection"]["click_lat"]), float(cfg["selection"]["click_lon"])]
    m = folium.Map(location=center, zoom_start=12, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("CartoDB dark_matter", name="Carto Dark", show=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map", show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Imagery", show=False,
    ).add_to(m)

    _add_geojson(m, basin, "DEM-derived model basin", {"color": "#7395b8", "weight": 1.2, "fillOpacity": 0.03}, False)
    _add_geojson(m, channel, "Selected bayou", {"color": "#ffe36d", "weight": 5, "opacity": .95}, True)
    _add_geojson(m, mainstem, "Modeled dominant mainstem", {"color": "#68d8ff", "weight": 3, "opacity": .9}, True)
    _add_geojson(m, o / "latest_channel_corridor.geojson", "Bank-to-bank excluded channel", {"color": "#a879ff", "weight": 1.2, "fillColor": "#6c41ac", "fillOpacity": .15}, False)
    _add_geojson(m, o / "latest_overtopped_sections.geojson", "Locally overtopped sections", {"color": "#ff5d73", "weight": 3, "opacity": .95}, True)
    _add_geojson(m, o / "latest_floodplain_seeds.geojson", "Floodplain entry seeds", {"color": "#ffba49", "weight": 2, "fillColor": "#ffba49", "fillOpacity": .9}, True)
    _add_geojson(m, extent, "Preferred volume-limited flood extent", {"color": "#3de6f4", "weight": 1.2, "fillColor": "#1aabc2", "fillOpacity": .10}, True)
    _add_geojson(m, o / "latest_potential_inundation.geojson", "Potential connected upper bound (QA)", {"color": "#f27aa8", "weight": 1, "dashArray": "5 5", "fillOpacity": .03}, False)
    _add_geojson(m, o / "latest_max_depth_point.geojson", "Maximum-depth QA point", {"color": "#ffffff", "weight": 2, "fillColor": "#ffffff", "fillOpacity": 1}, True)

    if depth.exists() and depth.stat().st_size:
        ov = _raster_overlay_cached(str(depth), depth.stat().st_mtime)
        if ov:
            ImageOverlay(ov["image"], bounds=ov["bounds"], opacity=.82, name=f"Flood depth (0–{ov['vmax']:.2f} m)").add_to(m)

    folium.Marker(
        [float(cfg["pilot"]["gauge_lat"]), float(cfg["pilot"]["gauge_lon"])],
        tooltip=f"USGS {cfg['pilot']['usgs_site']} — control gauge",
        icon=folium.Icon(color="red", icon="tint", prefix="fa"),
    ).add_to(m)

    # Fit to display extent if possible, otherwise basin.
    try:
        fit = gpd.read_file(extent if extent.exists() else basin).to_crs(4326)
        if not fit.empty:
            minx, miny, maxx, maxy = fit.total_bounds
            if maxx > minx and maxy > miny:
                m.fit_bounds([[miny, minx], [maxy, maxx]])
    except Exception:
        pass
    Fullscreen(position="topright").add_to(m)
    MeasureControl(position="topright", primary_length_unit="meters").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Cursor").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ============================================================================
# SIDEBAR / SYSTEM STATUS
# ============================================================================

with st.sidebar:
    st.markdown("### 🌊 Regional Control Tower")
    st.caption("Select → control gauge → static model cache → live refresh")
    st.markdown("---")
    st.markdown("**Regional catalog**")
    st.write(f"HCFCD major-channel features: **{len(hcfcd_channels):,}**")
    st.write(f"NHDPlus regional stream features: **{len(nhd_major):,}**")
    st.write(f"DEM cache: **{_human_bytes(_dir_size(DEM_CACHE_DIR))}**")
    if regional_dem_path:
        st.success("Regional DEM/VRT configured")
    else:
        st.info("On-demand Houston B24 tile cache mode")
    st.markdown("---")
    if st.button("Refresh regional catalog", use_container_width=True):
        try:
            build_regional_catalog(CATALOG_DIR, force=True)
            _catalog.clear()
            st.success("Regional catalog refreshed. Reloading …")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.caption("Flood Hub remains intentionally inactive unless you later configure credentials.")


# ============================================================================
# HERO
# ============================================================================

st.markdown(
    """
<div class="hero">
  <h1>H-GAC Regional Flood Intelligence</h1>
  <p>Interactive bayou selection, hydraulically relevant gauge discovery, cached LiDAR terrain preparation, NOAA NWM + USGS live screening, and event validation in one app.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="stage-pill">01 Select Bayou</span>'
    '<span class="stage-pill">02 Find Gauge</span>'
    '<span class="stage-pill">03 Prepare Static Cache</span>'
    '<span class="stage-pill">04 Live NWM + USGS</span>'
    '<span class="stage-pill">05 Volume-Limited Flood Screen</span>',
    unsafe_allow_html=True,
)

st.warning(
    "Screening system only — not a HEC-RAS, FEMA, engineering, emergency-warning, or regulatory floodplain product. "
    "For a new bayou, the app requires a hydraulically relevant gauge/rating and NAVD88 vertical-datum QA before it will generate an inundation screen."
)


# ============================================================================
# 1 — REGIONAL BAYOU SELECTION
# ============================================================================

st.subheader("1 · Select a bayou from the H-GAC regional map")
st.caption(
    "Click on or very near a waterway. HCFCD channels are used where available; elsewhere the app uses the national NHDPlus HR network. "
    "The click is only the selection point — the hydraulic model domain is determined later from the chosen control gauge and DEM routing."
)

map_out = st_folium(
    _discovery_map(),
    width=None,
    height=int(RCFG.get("ui", {}).get("map_height_px", 610)),
    returned_objects=["last_clicked"],
    key="regional_discovery_map",
)

click = (map_out or {}).get("last_clicked")
if click and click.get("lng") is not None:
    lon, lat = float(click["lng"]), float(click["lat"])
    prev = st.session_state.selection_click
    changed = prev is None or math.hypot(lon - prev[0], lat - prev[1]) > 1e-7
    if changed:
        try:
            with st.spinner("Resolving the clicked bayou/channel …"):
                sel = select_channel_from_click(
                    boundary, hcfcd_channels, lon, lat,
                    hcfcd_max_distance_m=float(RCFG["regional"].get("hcfcd_click_tolerance_m", 1500.0)),
                    nhd_search_radius_m=float(RCFG["regional"].get("nhd_search_radius_m", 6000.0)),
                )
                sel["mainstem_match_tolerance_m"] = float(RCFG["regional"].get("mainstem_match_tolerance_m", 500.0))
            st.session_state.selected_channel = sel
            st.session_state.selection_click = (lon, lat)
            st.session_state.gauge_candidates = []
            st.session_state.selected_gauge_site = None
            st.session_state.workspace = None
            st.session_state.workspace_config = None
            st.session_state.live_metadata = None
            st.rerun()
        except Exception as exc:
            st.error(f"Selection could not be resolved: {exc}")

sel = st.session_state.selected_channel
if sel:
    a, b, c, d = st.columns([2.2, 1, 1, 1])
    with a:
        _metric_card("Selected waterway", sel["name"], sel.get("source", ""))
    with b:
        _metric_card("Click → waterway", _fmt(sel.get("distance_m"), 0, " m"), "snap distance")
    with c:
        _metric_card("Stream order", str(sel.get("stream_order") or "—"), "NHDPlus if available")
    with d:
        _metric_card("Drainage area", _fmt(sel.get("total_drainage_area_km2"), 1, " km²"), "NHDPlus if available")


# ============================================================================
# 2 — GAUGE DISCOVERY
# ============================================================================

if sel:
    st.subheader("2 · Resolve the hydraulic control gauge")
    st.caption(
        "The app does not simply choose the nearest gauge. It tests nearby USGS streamgages and favors a downstream gauge whose NLDI upstream basin contains your bayou click and that has recent discharge data."
    )

    if st.button("Find hydraulically relevant USGS gauges", type="primary", use_container_width=False):
        try:
            cache = PROJECT_ROOT / "data" / "regional_cache" / "gauges" / f"{sel['name'].replace(' ', '_')[:50]}"
            with st.spinner("Checking nearby gauges, upstream basins, and live availability …"):
                best, candidates = choose_best_gauge(
                    float(sel["click_lon"]), float(sel["click_lat"]),
                    selected_channel_geom=sel["geometry"],
                    radius_km=float(RCFG["regional"].get("gauge_search_radius_km", 30.0)),
                    max_candidates_to_test=int(RCFG["regional"].get("gauge_candidates_to_test", 8)),
                    cache_root=cache,
                )
            st.session_state.gauge_candidates = candidates
            st.session_state.selected_gauge_site = best["site_no"] if best else None
            st.rerun()
        except Exception as exc:
            st.error(f"Gauge discovery failed: {exc}")

    candidates = st.session_state.gauge_candidates or []
    selected_gauge = None
    if candidates:
        display = []
        for g in candidates:
            display.append({
                "site_no": g["site_no"],
                "name": g["name"],
                "distance_km": round(float(g["distance_to_click_m"]) / 1000, 2),
                "basin_contains_click": bool(g["basin_contains_click"]),
                "recent_flow": bool(g["recent_data_ok"]),
                "drainage_sqmi": g.get("contributing_drainage_area_sqmi") or g.get("drainage_area_sqmi"),
                "vertical_datum": g.get("vertical_datum_name") or g.get("vertical_datum"),
            })
        st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)
        options = [g["site_no"] for g in candidates]
        default_site = st.session_state.selected_gauge_site if st.session_state.selected_gauge_site in options else options[0]
        selected_site = st.selectbox(
            "Control gauge",
            options,
            index=options.index(default_site),
            format_func=lambda site: next(f"USGS {g['site_no']} · {g['name']}" for g in candidates if g["site_no"] == site),
        )
        st.session_state.selected_gauge_site = selected_site
        selected_gauge = next(g for g in candidates if g["site_no"] == selected_site)
        if selected_gauge.get("basin_contains_click") and selected_gauge.get("recent_data_ok"):
            st.success("Gauge passes the initial hydraulic-relevance + recent-data screen.")
        elif not selected_gauge.get("basin_contains_click"):
            st.error("This gauge's upstream basin does not contain the selected bayou click. Do not run the single-mainstem model with this control.")
        else:
            st.warning("Gauge is hydraulically relevant, but recent live discharge could not be confirmed.")


# ============================================================================
# 3 — STATIC WORKSPACE + FULL PIPELINE
# ============================================================================

if sel and st.session_state.gauge_candidates:
    selected_gauge = next(
        (g for g in st.session_state.gauge_candidates if g["site_no"] == st.session_state.selected_gauge_site),
        None,
    )
    if selected_gauge:
        st.subheader("3 · Prepare the bayou model and run the full near-real-time workflow")
        prospective_ws = workspace_path(PROJECT_ROOT, sel["name"], selected_gauge["site_no"])
        is_cached = static_ready(prospective_ws)

        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            _metric_card("Workspace", prospective_ws.name, "persistent bayou-specific cache")
        with c2:
            _metric_card("Static model", "READY" if is_cached else "NOT PREPARED", "DEM · HAND · banks · rating")
        with c3:
            _metric_card("Control", f"USGS {selected_gauge['site_no']}", f"{float(selected_gauge['distance_to_click_m'])/1000:.1f} km from click")

        with st.expander("Advanced gauge / vertical datum controls", expanded=False):
            st.caption(
                "For a new gauge, the USGS rating is in gage-height coordinates. The app performs a NAVD88 consistency check before combining it with LiDAR. "
                "Only enter an override if you have independently verified the USGS gage datum in NAVD88 — do not use station ground altitude as a substitute."
            )
            existing_override = ""
            existing_cfg_path = prospective_ws / "config" / "config.yaml"
            if existing_cfg_path.exists():
                ev = _read_yaml(existing_cfg_path).get("rating", {}).get("gage_datum_navd88_ft_override")
                if ev not in (None, ""):
                    existing_override = str(ev)
            datum_override = st.text_input("Verified gage datum NAVD88 (ft), optional", value=existing_override)
            force_static = st.checkbox("Force rebuild static terrain/bank products", value=False)

        supported = bool(selected_gauge.get("basin_contains_click") and selected_gauge.get("recent_data_ok"))
        run_col, refresh_col = st.columns([1, 1])
        run_clicked = run_col.button(
            "Run 00 → 10 for selected bayou",
            type="primary",
            disabled=not supported,
            use_container_width=True,
        )
        refresh_clicked = refresh_col.button(
            "Refresh live USGS + NWM + inundation only",
            disabled=not is_cached,
            use_container_width=True,
        )

        if run_clicked:
            watershed = watershed_for_click(hcfcd_watersheds, sel["click_lon"], sel["click_lat"])
            base_cfg = _base_workspace_settings()
            ws_exists = prospective_ws.exists() and (prospective_ws / "config" / "config.yaml").exists()
            ws, config_path, _ = create_workspace(
                PROJECT_ROOT, sel, selected_gauge,
                base_cfg=base_cfg, watershed=watershed,
                overwrite_config=not ws_exists,
            )
            _patch_workspace_datum(config_path, datum_override)
            st.session_state.workspace = str(ws)
            st.session_state.workspace_config = str(config_path)

            progress_bar = st.progress(0, text="Initializing workflow …")
            status = st.status("Running regional flood pipeline", expanded=True)
            steps = {"00": 8, "01": 18, "02": 38, "03": 46, "04": 56, "05": 68, "06": 76, "07": 83, "08": 94, "09": 97, "10": 100, "CACHE": 70}
            log_slot = status.empty()

            def cb(evt):
                step = str(evt.get("step", ""))
                pct = steps.get(step, 0)
                progress_bar.progress(min(max(pct, 0), 100), text=f"Step {step}: {evt.get('message', '')}")
                log_slot.markdown(f"**{evt.get('status','').upper()} · {step}** — {evt.get('message','')}")

            try:
                result = run_full_pipeline(
                    PROJECT_ROOT, ws, config_path, RCFG,
                    force_static=force_static, progress=cb,
                )
                st.session_state.live_metadata = result["live"]
                st.session_state.last_pipeline_error = None
                progress_bar.progress(100, text="Workflow complete")
                status.update(label="Pipeline complete", state="complete", expanded=False)
                st.success("Selected bayou is prepared and the live flood screen has been refreshed.")
                st.rerun()
            except Exception as exc:
                st.session_state.last_pipeline_error = str(exc)
                status.update(label="Pipeline stopped by QA/error", state="error", expanded=True)
                st.error(str(exc))

        if refresh_clicked:
            config_path = prospective_ws / "config" / "config.yaml"
            try:
                with st.status("Refreshing live USGS + NWM + flood screen …", expanded=True) as live_status:
                    line = st.empty()
                    def cb2(evt):
                        line.write(f"Step {evt.get('step')}: {evt.get('message')}")
                    meta = run_live_workspace(prospective_ws, config_path, progress=cb2)
                    st.session_state.workspace = str(prospective_ws)
                    st.session_state.workspace_config = str(config_path)
                    st.session_state.live_metadata = meta
                    live_status.update(label="Live refresh complete", state="complete", expanded=False)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if st.session_state.last_pipeline_error:
            st.warning(f"Last pipeline stop: {st.session_state.last_pipeline_error}")

        if is_cached and st.session_state.workspace is None:
            st.session_state.workspace = str(prospective_ws)
            st.session_state.workspace_config = str(prospective_ws / "config" / "config.yaml")


# ============================================================================
# 4 — LIVE AUTO REFRESH + ADVANCED DASHBOARD
# ============================================================================

workspace = Path(st.session_state.workspace) if st.session_state.workspace else None
workspace_cfg_path = Path(st.session_state.workspace_config) if st.session_state.workspace_config else None

if workspace and workspace_cfg_path and workspace_cfg_path.exists() and static_ready(workspace):
    cfg = load_config(workspace_cfg_path)
    saved_meta = _json(workspace / "outputs" / "dashboard_scenario.json")
    if saved_meta:
        st.session_state.live_metadata = saved_meta

    st.subheader("4 · Near-real-time operational dashboard")
    auto_c1, auto_c2, auto_c3 = st.columns([1.2, .8, 2])
    auto_default = bool(RCFG.get("live", {}).get("auto_refresh_enabled_default", False))
    auto_live = auto_c1.toggle("Automatic live refresh", value=auto_default, key=f"auto_{workspace.name}")
    min_refresh = int(RCFG.get("live", {}).get("minimum_refresh_minutes", 3))
    default_refresh = int(RCFG.get("live", {}).get("auto_refresh_minutes", 5))
    refresh_minutes = auto_c2.selectbox("Interval (min)", [3, 5, 10, 15, 30], index=[3,5,10,15,30].index(default_refresh) if default_refresh in [3,5,10,15,30] else 1)
    auto_c3.caption(
        "Static DEM/HAND/bank/rating products are reused. Auto refresh only re-fetches USGS + NWM and regenerates the volume-limited flood screen."
    )

    if auto_live:
        if st_autorefresh is None:
            st.warning("Automatic refresh requires `streamlit-autorefresh`. Install `requirements_v5.txt`; manual live refresh still works.")
        else:
            counter = st_autorefresh(interval=max(refresh_minutes, min_refresh) * 60 * 1000, key=f"refresh_{workspace.name}")
            prev_counter = st.session_state.last_auto_refresh_counter
            if prev_counter is None:
                st.session_state.last_auto_refresh_counter = counter
            elif counter != prev_counter:
                st.session_state.last_auto_refresh_counter = counter
                try:
                    st.session_state.live_metadata = run_live_workspace(workspace, workspace_cfg_path)
                    st.toast("Live USGS/NWM flood screen refreshed", icon="🌊")
                except Exception as exc:
                    st.warning(f"Automatic refresh failed; previous saved products remain visible: {exc}")

    meta = st.session_state.live_metadata or {}
    obs = meta.get("usgs_latest") or {}
    bank_stage = meta.get("gauge_bank_stage_ft_navd88")
    scenario_stage = meta.get("scenario_stage_ft_navd88")
    scenario_q = meta.get("scenario_flow_cfs")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: _metric_card("USGS discharge", _fmt(obs.get("discharge_cfs"), 1, " cfs"), f"USGS {cfg['pilot']['usgs_site']}")
    with m2: _metric_card("USGS stage", _fmt(obs.get("stage_ft"), 2, " ft"), "reported gage height")
    with m3: _metric_card("NWM peak", _fmt(scenario_q, 1, " cfs"), str(meta.get("peak_time", ""))[:19])
    with m4: _metric_card("Modeled WSE", _fmt(scenario_stage, 2, " ft"), "NAVD88")
    with m5: _metric_card("Flooded area", _fmt(meta.get("inundated_area_sqmi"), 3, " mi²"), "preferred volume-limited")
    with m6: _metric_card("Max depth", _fmt(meta.get("max_depth_m"), 2, " m"), "outside bank-to-bank channel")

    tab_map, tab_hydro, tab_qa, tab_validation, tab_cache = st.tabs([
        "🗺️ Live Flood Map", "📈 Hydrographs", "🧪 Terrain / Hydraulic QA", "✅ Historical Validation", "⚙️ Cache & System"
    ])

    with tab_map:
        st_folium(_result_map(workspace, cfg, meta), width=None, height=720, key=f"result_map_{workspace.name}")
        qa_flags = meta.get("qa_flags") or []
        if qa_flags:
            st.warning("QA flags: " + " · ".join(qa_flags))
        else:
            st.success("No model QA flags were emitted for the current saved scenario.")
        st.caption(
            "Preferred flood depth is volume-limited and excludes the LiDAR-derived bank-to-bank channel corridor. "
            "The potential connected extent is retained only as an unconstrained QA upper bound."
        )

    with tab_hydro:
        nwm_csv = workspace / "outputs" / "latest_nwm_hydrograph.csv"
        usgs_csv = workspace / "outputs" / "latest_usgs_observations.csv"
        if go is not None and nwm_csv.exists():
            nwm_df = pd.read_csv(nwm_csv)
            nwm_df["time"] = pd.to_datetime(nwm_df["time"], utc=True, errors="coerce")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nwm_df["time"], y=nwm_df["streamflow"], mode="lines+markers", name="NOAA NWM discharge"))
            qbf = meta.get("gauge_bankfull_flow_cfs")
            if qbf is not None:
                fig.add_hline(y=float(qbf), line_dash="dash", annotation_text="Bankfull-flow proxy")
            fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20,r=20,t=45,b=20), title="Forecast hydrograph at control reach", yaxis_title="Discharge (cfs)")
            st.plotly_chart(fig, use_container_width=True)
        elif nwm_csv.exists():
            df = pd.read_csv(nwm_csv).set_index("time")
            st.line_chart(df["streamflow"])
        else:
            st.info("Run a live refresh to save the NWM hydrograph.")

        if usgs_csv.exists():
            u = pd.read_csv(usgs_csv)
            u["time"] = pd.to_datetime(u["time"], utc=True, errors="coerce")
            if go is not None:
                fig2 = go.Figure()
                if "00060" in u:
                    fig2.add_trace(go.Scatter(x=u["time"], y=u["00060"], name="USGS discharge", mode="lines"))
                fig2.update_layout(template="plotly_dark", height=330, margin=dict(l=20,r=20,t=45,b=20), title="Recent USGS observation history", yaxis_title="Discharge (cfs)")
                st.plotly_chart(fig2, use_container_width=True)

    with tab_qa:
        tqa = _json(workspace / "data" / "processed" / "terrain_qa.json") or {}
        bqa = _json(workspace / "data" / "processed" / "bank_profile_summary.json") or {}
        q1, q2, q3, q4 = st.columns(4)
        with q1: _metric_card("DEM basin", _fmt(tqa.get("dem_upstream_basin_area_km2"), 2, " km²"), "surface-topographic")
        with q2: _metric_card("Mainstem cells", _fmt(tqa.get("mainstem_cells"), 0), "derived drainage path")
        with q3: _metric_card("Cross sections", _fmt(meta.get("bank_sections"), 0), "LiDAR screening sections")
        with q4: _metric_card("Overtopped", _fmt(meta.get("overtopped_sections"), 0), "current scenario")

        bp_csv = workspace / "data" / "processed" / "bank_profile.csv"
        if bp_csv.exists() and go is not None:
            bp = pd.read_csv(bp_csv)
            fig3 = go.Figure()
            if "lower_bank_final_ft_navd88" in bp:
                fig3.add_trace(go.Scatter(x=bp["station_m"], y=bp["lower_bank_final_ft_navd88"], name="Local lower bank", mode="lines+markers"))
            if "routing_channel_elevation_ft_navd88" in bp:
                fig3.add_trace(go.Scatter(x=bp["station_m"], y=bp["routing_channel_elevation_ft_navd88"], name="LiDAR channel-surface proxy", mode="lines"))
            fig3.update_layout(template="plotly_dark", height=390, margin=dict(l=20,r=20,t=45,b=20), title="Longitudinal bank / channel-surface screening profile", xaxis_title="Station upstream (m)", yaxis_title="Elevation (ft NAVD88)")
            st.plotly_chart(fig3, use_container_width=True)

        max_info = meta.get("max_depth_location") or {}
        if max_info:
            st.markdown("**Maximum-depth QA**")
            st.json(max_info, expanded=False)
        st.caption("LiDAR channel elevation is a channel-surface proxy, not bathymetric bed elevation.")

    with tab_validation:
        st.markdown("#### Historical validation workflow")
        st.write(
            "Use observed historical gauge hydrographs to isolate inundation-model performance, then compare the modeled event extent/WSE against event-specific inundation mapping and high-water marks. Separately compare historical NWM/reanalysis flow against observed flow."
        )
        st.code(
            f'$env:HGAC_CONFIG_PATH = "{workspace_cfg_path}"\n'
            f'python scripts\\06_make_inundation.py --hydrograph-csv data\\validation\\event_flow.csv --hydrograph-unit cfs --scenario-label "Historical Event"\n\n'
            f'python scripts\\11_validate_historical_event.py --event-name event_name --reference-extent data\\validation\\observed_extent.geojson\n\n'
            f'python scripts\\12_validate_hydrograph.py --event-name event_name --modeled-csv data\\validation\\nwm_event.csv --observed-csv data\\validation\\usgs_event.csv',
            language="powershell",
        )
        st.info("For older events, document terrain-date mismatch if the available LiDAR/DEM post-dates the flood event.")

    with tab_cache:
        st.markdown("#### Regional DEM cache architecture")
        st.write(
            "V5 does not recommend physically merging a single 1-m raster for all H-GAC. Instead, use one persistent tile library plus a VRT (or ArcGIS Mosaic Dataset) and crop each selected bayou on demand. The full gauge basin is processed at the analysis resolution; the 1-m hydraulic DEM is cropped only to a corridor around the modeled mainstem."
        )
        cc1, cc2, cc3 = st.columns(3)
        with cc1: _metric_card("Regional DEM cache", _human_bytes(_dir_size(DEM_CACHE_DIR)), str(DEM_CACHE_DIR))
        with cc2: _metric_card("Bayou workspace", _human_bytes(_dir_size(workspace)), workspace.name)
        with cc3: _metric_card("Static cache", "READY" if static_ready(workspace) else "INCOMPLETE", "reused on every live refresh")

        tile_dir = DEM_CACHE_DIR / "tiles"
        vrt_default = DEM_CACHE_DIR / "hgac_dem_cache.vrt"
        a, b = st.columns(2)
        if a.button("Build GDAL VRT from cached tiles", disabled=not tile_dir.exists(), use_container_width=True):
            try:
                vrt = build_virtual_mosaic(tile_dir, vrt_default)
                st.success(f"VRT built: {vrt}. Put this path in config/v5_regional.yaml → dem_cache.regional_dem_path for fastest reuse.")
            except Exception as exc:
                st.error(str(exc))
        if b.button("Build ArcPy Mosaic Dataset", disabled=not tile_dir.exists(), use_container_width=True):
            try:
                gdb = _abs_project_path(RCFG["dem_cache"].get("arcpy_mosaic_gdb", "data/regional_cache/HGAC_DEM_Cache.gdb"))
                md = build_arcpy_mosaic_dataset(tile_dir, gdb, RCFG["dem_cache"].get("arcpy_mosaic_name", "HGAC_DEM_CACHE"))
                st.success(f"ArcGIS Mosaic Dataset ready: {md}")
            except Exception as exc:
                st.error(f"ArcPy mosaic could not be built in this Python environment: {exc}")

        st.warning(
            "Coverage note: the built-in automatic downloader currently knows the 2024 USGS Houston B24 LiDAR project. That is not the entire 13-county H-GAC region. "
            "For bayous outside that source grid, configure a regional 3DEP/agency VRT/COG/GeoTIFF or extend the DEM provider catalog."
        )

else:
    st.info("Select a bayou, find a supported USGS control gauge, then run the full workflow to unlock the live dashboard.")

st.markdown("---")
st.caption(
    "H-GAC Regional Flood Intelligence V5 · static terrain products are cached by bayou/gauge; live products refresh from USGS + NOAA NWM. "
    "Google Flood Hub integration remains coded but inactive until explicitly configured."
)
