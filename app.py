from __future__ import annotations

"""H-GAC Regional Flood Intelligence V5.3 guided local/public web application.

This is intentionally NOT a Streamlit app.  It uses Flask + a small browser UI
so long-running DEM/terrain jobs can continue in the background while the UI
shows explicit progress, elapsed time, step-by-step status, errors, and cached
workspace state.
"""

from pathlib import Path
from threading import RLock, Thread, Timer
from uuid import uuid4
import copy
import io
import json
import os
import time
import traceback
import webbrowser

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import yaml
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from src.regional_catalog import (
    load_regional_catalog,
    select_channel_from_click,
    watershed_for_click,
)
from src.gauge_discovery import choose_best_gauge
from src.workspace import create_workspace, workspace_path
from src.pipeline_v5 import run_full_pipeline, run_live_workspace, static_ready
from src.dem_cache import build_virtual_mosaic, build_arcpy_mosaic_dataset
from src.intelligence_data import collect_intelligence_bundle, refresh_live_supplemental
from src.rainfall_scenario import run_rainfall_scenario
from src.config import load_config


APP_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = Path(
    os.getenv("HGAC_RUNTIME_ROOT", str(APP_ROOT))
).expanduser().resolve()
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
PUBLIC_MODE = os.getenv("HGAC_PUBLIC_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
REGIONAL_CONFIG_PATH = APP_ROOT / "config" / "v5_regional.yaml"
ROOT_CONFIG_PATH = APP_ROOT / "config" / "config.yaml"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

LOCK = RLock()
JOBS: dict[str, dict] = {}
STATE = {
    "catalog_status": "starting",
    "catalog_error": None,
    "boundary": None,
    "channels": None,
    "watersheds": None,
    "nhd_major": None,
    "catalog_json": None,
    "selected_channel": None,
    "selection_click": None,
    "gauge_candidates": [],
    "selected_gauge_site": None,
    "workspace": None,
    "workspace_config": None,
    "last_error": None,
}

STEP_PROGRESS = {
    "00": 8,
    "01": 25,
    "02": 52,
    "03": 60,
    "04": 70,
    "05": 80,
    "06": 86,
    "07": 91,
    "08": 97,
    "09": 99,
    "10": 100,
    "CACHE": 75,
}

STEP_HELP = {
    "00": "Resolving the gauge-controlled basin. Usually quick.",
    "01": "Preparing DEM. On the first run this can take several minutes because missing source tiles may be downloaded and cached. Later runs reuse them.",
    "02": "Conditioning terrain, deriving drainage, HAND and mainstem. This is CPU/disk intensive; please keep the app open.",
    "03": "Linking the gauge to its NHDPlus / NOAA NWM reach.",
    "04": "Preparing the rating curve and checking the vertical datum against LiDAR.",
    "05": "Building LiDAR bank cross-sections and the local bank profile.",
    "06": "Refreshing current USGS observations.",
    "07": "Fetching the NOAA National Water Model hydrograph.",
    "08": "Generating the channel-excluded, volume-limited floodplain screen.",
    "09": "Clipping finished outputs to the selected display AOI when applicable.",
    "10": "Finalizing dashboard products.",
    "CACHE": "Static terrain products already exist; reusing the cached model.",
}


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def abs_project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (APP_ROOT / p).resolve()


def regional_cfg() -> dict:
    cfg = read_yaml(REGIONAL_CONFIG_PATH)
    cfg.setdefault("regional", {})
    cfg.setdefault("dem_cache", {})
    cfg.setdefault("live", {})
    cfg.setdefault("ui", {})

    # cloud_runtime_cache_override:
    # When HGAC_RUNTIME_ROOT is set (e.g. Render persistent disk), keep all
    # expensive cache products outside the ephemeral application filesystem.
    if os.getenv("HGAC_RUNTIME_ROOT"):
        cfg["regional"]["catalog_cache"] = str(
            RUNTIME_ROOT / "data" / "regional_cache" / "catalog"
        )
        cfg["dem_cache"]["cache_root"] = str(
            RUNTIME_ROOT / "data" / "regional_cache" / "dem"
        )
        cfg["dem_cache"]["arcpy_mosaic_gdb"] = str(
            RUNTIME_ROOT / "data" / "regional_cache" / "HGAC_DEM_Cache.gdb"
        )
    cache_root = abs_project_path(cfg["dem_cache"].get("cache_root", "data/regional_cache/dem"))
    cfg["dem_cache"]["cache_root"] = str(cache_root)
    rdem = str(cfg["dem_cache"].get("regional_dem_path", "") or "").strip()
    if rdem:
        cfg["dem_cache"]["regional_dem_path"] = str(abs_project_path(rdem))
    return cfg


def base_workspace_settings() -> dict:
    raw = read_yaml(ROOT_CONFIG_PATH)
    keep = [
        "usgs", "nldi", "nwm", "floodhub", "dem", "terrain", "bank_profile",
        "hydraulics", "inundation", "gauge_network",
    ]
    return {k: copy.deepcopy(raw[k]) for k in keep if k in raw}


def patch_workspace_datum(config_path: Path, override_text: str | None):
    cfg = read_yaml(config_path)
    cfg.setdefault("rating", {})
    text = str(override_text or "").strip()
    cfg["rating"]["gage_datum_navd88_ft_override"] = float(text) if text else None
    write_yaml(config_path, cfg)


def json_file(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def serialize_selection(sel: dict | None):
    if not sel:
        return None
    out = {k: v for k, v in sel.items() if k not in {"geometry", "properties"}}
    out["properties"] = {k: _json_safe(v) for k, v in (sel.get("properties") or {}).items()}
    if sel.get("geometry") is not None:
        from shapely.geometry import mapping
        out["geometry"] = mapping(sel["geometry"])
    return out


def _json_safe(v):
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        try:
            if isinstance(v, float) and not np.isfinite(v):
                return None
        except Exception:
            pass
        return v
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return str(v)


def safe_gauge(g: dict):
    return {k: _json_safe(v) for k, v in g.items() if k not in {"geometry"}}


def new_job(kind: str, title: str, worker):
    jid = uuid4().hex[:12]
    now = time.time()
    with LOCK:
        JOBS[jid] = {
            "id": jid,
            "kind": kind,
            "title": title,
            "status": "queued",
            "progress": 0,
            "step": "",
            "message": "Queued",
            "help": "",
            "started_at": None,
            "finished_at": None,
            "created_at": now,
            "events": [],
            "result": None,
            "error": None,
        }

    def runner():
        with LOCK:
            j = JOBS[jid]
            j["status"] = "running"
            j["started_at"] = time.time()
            j["message"] = "Startingâ€¦"
        try:
            result = worker(jid)
            with LOCK:
                j = JOBS[jid]
                j["status"] = "done"
                j["progress"] = 100
                j["message"] = "Completed"
                j["finished_at"] = time.time()
                j["result"] = result
        except Exception as exc:
            with LOCK:
                j = JOBS[jid]
                j["status"] = "error"
                j["finished_at"] = time.time()
                j["error"] = str(exc)
                j["message"] = str(exc)
                j["events"].append({
                    "time": time.time(), "status": "error", "step": j.get("step", ""),
                    "message": str(exc), "details": traceback.format_exc(limit=8),
                })
                STATE["last_error"] = str(exc)

    Thread(target=runner, daemon=True, name=f"hgac-{kind}-{jid}").start()
    return jid


def update_job(jid: str, *, step=None, message=None, status=None, progress=None, help_text=None, event=True):
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            return
        if step is not None:
            j["step"] = str(step)
        if message is not None:
            j["message"] = str(message)
        if status is not None:
            j["status_detail"] = str(status)
        if progress is not None:
            j["progress"] = max(0, min(100, int(progress)))
        if help_text is not None:
            j["help"] = str(help_text)
        if event:
            j["events"].append({
                "time": time.time(),
                "step": str(step or j.get("step", "")),
                "status": str(status or "running"),
                "message": str(message or j.get("message", "")),
            })
            j["events"] = j["events"][-120:]


def pipeline_callback(jid: str):
    def cb(evt):
        step = str(evt.get("step", ""))
        p = STEP_PROGRESS.get(step, 0)
        status = str(evt.get("status", "running"))
        if status in {"done", "cached"}:
            p = max(p, STEP_PROGRESS.get(step, p))
        update_job(
            jid,
            step=step,
            message=evt.get("message", ""),
            status=status,
            progress=p,
            help_text=STEP_HELP.get(step, "The operation is still running. Keep this window open."),
        )
    return cb


def simplify_gdf(gdf: gpd.GeoDataFrame, tolerance_m: float, keep_cols: list[str]):
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    cols = [c for c in keep_cols if c in gdf.columns]
    gg = gdf[cols + ["geometry"]].copy()
    try:
        local = gg.to_crs(32615)
        local["geometry"] = local.geometry.simplify(float(tolerance_m), preserve_topology=True)
        gg = local.to_crs(4326)
    except Exception:
        gg = gg.to_crs(4326)
    return json.loads(gg.to_json(drop_id=True))


def load_catalog_worker():
    with LOCK:
        STATE["catalog_status"] = "running"
        STATE["catalog_error"] = None
    try:
        rcfg = regional_cfg()
        cache = abs_project_path(rcfg["regional"].get("catalog_cache", "data/regional_cache/catalog"))
        boundary, channels, watersheds, nhd = load_regional_catalog(cache)
        cat_json = {
            "boundary": simplify_gdf(boundary, 60, ["NAME", "County", "county", "display_name"]),
            "channels": simplify_gdf(channels, 20, ["CHAN_NAME", "UNIT_NO", "display_name", "source"]),
            "nhd": simplify_gdf(nhd, 35, ["gnis_name", "streamorde", "display_name", "source"]),
        }
        with LOCK:
            STATE.update({
                "catalog_status": "ready", "boundary": boundary, "channels": channels,
                "watersheds": watersheds, "nhd_major": nhd, "catalog_json": cat_json,
            })
    except Exception as exc:
        with LOCK:
            STATE["catalog_status"] = "error"
            STATE["catalog_error"] = str(exc)


def ensure_catalog_async():
    with LOCK:
        if STATE["catalog_status"] in {"running", "ready"}:
            return
        STATE["catalog_status"] = "running"
    Thread(target=load_catalog_worker, daemon=True, name="hgac-catalog").start()


def workspace_paths():
    ws = STATE.get("workspace")
    if not ws:
        return None, None
    w = Path(ws)
    return w, w / "config" / "config.yaml"


def recent_workspaces():
    root = RUNTIME_ROOT / "workspaces"
    rows = []
    if not root.exists():
        return rows
    for p in sorted(root.iterdir(), key=lambda q: q.stat().st_mtime if q.exists() else 0, reverse=True):
        sp = p / "workspace_state.json"
        if not sp.exists():
            continue
        st = json_file(sp) or {}
        rows.append({
            "name": p.name,
            "path": str(p),
            "channel": (st.get("channel") or {}).get("name") or (st.get("channel") or {}).get("channel_name") or p.name,
            "gauge": (st.get("gauge") or {}).get("site_no"),
            "gauge_name": (st.get("gauge") or {}).get("name"),
            "static_ready": static_ready(p),
            "has_live": (p / "outputs" / "dashboard_scenario.json").exists(),
            "modified": p.stat().st_mtime,
        })
    return rows[:20]


def load_workspace_into_state(path: Path):
    state = json_file(path / "workspace_state.json") or {}
    channel_geo = path / "data" / "processed" / "selected_channel.geojson"
    sel = state.get("channel") or {}
    if channel_geo.exists():
        gg = gpd.read_file(channel_geo).to_crs(4326)
        if not gg.empty:
            sel = dict(sel)
            sel["geometry"] = gg.geometry.iloc[0]
            sel.setdefault("name", str(gg.iloc[0].get("name") or path.name))
            sel.setdefault("source", str(gg.iloc[0].get("source") or "cached"))
    gauge = state.get("gauge") or {}
    with LOCK:
        STATE["selected_channel"] = sel
        STATE["selection_click"] = (sel.get("click_lon"), sel.get("click_lat")) if sel else None
        STATE["gauge_candidates"] = [gauge] if gauge else []
        STATE["selected_gauge_site"] = gauge.get("site_no")
        STATE["workspace"] = str(path)
        STATE["workspace_config"] = str(path / "config" / "config.yaml")
        STATE["last_error"] = None


def file_feature_collection(path: Path):
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    try:
        return json.loads(gpd.read_file(path).to_crs(4326).to_json(drop_id=True))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"type": "FeatureCollection", "features": []}


def depth_overlay(workspace: Path):
    tif = workspace / "outputs" / "latest_depth.tif"
    png = workspace / "outputs" / "web_depth_overlay.png"
    meta = workspace / "outputs" / "web_depth_overlay.json"
    if not tif.exists():
        return None
    if png.exists() and meta.exists() and png.stat().st_mtime >= tif.stat().st_mtime:
        obj = json_file(meta)
        if obj:
            return obj

    with rasterio.open(tif) as src:
        left, bottom, right, top = src.bounds
        dst_transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, left, bottom, right, top
        )
        max_dim = 1400
        scale = min(1.0, max_dim / max(width, height))
        width = max(1, int(width * scale)); height = max(1, int(height * scale))
        if scale < 1:
            sx = (right - left) / max(width, 1)
            # Recompute default transform using requested dimensions for correct bounds.
            dst_transform, _, _ = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, left, bottom, right, top,
                dst_width=width, dst_height=height,
            )
        dst = np.zeros((height, width), dtype="float32")
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=0,
        )

    valid = np.isfinite(dst) & (dst > 0.02)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if valid.any():
        vmax = float(np.nanpercentile(dst[valid], 98))
        vmax = max(vmax, 0.25)
        x = np.clip(dst / vmax, 0, 1)
        # Professional blue-cyan depth ramp with transparent dry pixels.
        stops = np.array([0.0, 0.2, 0.5, 0.78, 1.0])
        colors = np.array([
            [185, 236, 255], [86, 199, 244], [34, 126, 208], [37, 64, 177], [71, 33, 130]
        ], dtype=float)
        flat = x.ravel()
        rgb = np.zeros((flat.size, 3), dtype=float)
        for c in range(3):
            rgb[:, c] = np.interp(flat, stops, colors[:, c])
        rgba[..., :3] = rgb.reshape(height, width, 3).astype(np.uint8)
        rgba[..., 3][valid] = 185
    Image.fromarray(rgba, mode="RGBA").save(png)
    west, south, east, north = rasterio.transform.array_bounds(height, width, dst_transform)
    obj = {
        "image_url": "/api/depth/image",
        "bounds": [[float(south), float(west)], [float(north), float(east)]],
        "mtime": tif.stat().st_mtime,
    }
    meta.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return obj


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/reset")
def api_reset():
    with LOCK:
        STATE["selected_channel"] = None
        STATE["selection_click"] = None
        STATE["gauge_candidates"] = []
        STATE["selected_gauge_site"] = None
        STATE["workspace"] = None
        STATE["workspace_config"] = None
        STATE["last_error"] = None
    return jsonify({"ok": True})


@app.get("/api/state")
def api_state():
    ws, _ = workspace_paths()
    with LOCK:
        selection = serialize_selection(STATE.get("selected_channel"))
        candidates = [safe_gauge(x) for x in STATE.get("gauge_candidates", [])]
        site = STATE.get("selected_gauge_site")
        catalog_status = STATE.get("catalog_status")
        catalog_error = STATE.get("catalog_error")
        last_error = STATE.get("last_error")
    current_gauge = next((g for g in candidates if g.get("site_no") == site), None)
    return jsonify({
        "catalog_status": catalog_status,
        "catalog_error": catalog_error,
        "selection": selection,
        "gauge_candidates": candidates,
        "selected_gauge_site": site,
        "selected_gauge": current_gauge,
        "workspace": str(ws) if ws else None,
        "static_ready": bool(ws and static_ready(ws)),
        "has_live": bool(ws and (ws / "outputs" / "dashboard_scenario.json").exists()),
        "has_scenario": bool(ws and (ws / "outputs" / "scenario_dashboard.json").exists()),
        "last_error": last_error,
        "recent_workspaces": recent_workspaces(),
    })


@app.get("/api/catalog")
def api_catalog():
    with LOCK:
        status = STATE.get("catalog_status")
        data = STATE.get("catalog_json")
        error = STATE.get("catalog_error")
    if status != "ready":
        return jsonify({"status": status, "error": error}), 202
    return jsonify({"status": "ready", **data})


@app.post("/api/catalog/rebuild")
def api_catalog_rebuild():
    with LOCK:
        STATE["catalog_status"] = "starting"
    Thread(target=load_catalog_worker, daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/select")
def api_select():
    payload = request.get_json(force=True) or {}
    lon = float(payload["lon"]); lat = float(payload["lat"])
    with LOCK:
        if STATE.get("catalog_status") != "ready":
            return jsonify({"error": "Regional map catalog is still loading. Please wait for initialization to finish."}), 409
        boundary = STATE["boundary"]; channels = STATE["channels"]
    rcfg = regional_cfg()
    try:
        sel = select_channel_from_click(
            boundary, channels, lon, lat,
            hcfcd_max_distance_m=float(rcfg["regional"].get("hcfcd_click_tolerance_m", 1500.0)),
            nhd_search_radius_m=float(rcfg["regional"].get("nhd_search_radius_m", 6000.0)),
        )
        sel["mainstem_match_tolerance_m"] = float(rcfg["regional"].get("mainstem_match_tolerance_m", 500.0))
        with LOCK:
            STATE["selected_channel"] = sel
            STATE["selection_click"] = (lon, lat)
            STATE["gauge_candidates"] = []
            STATE["selected_gauge_site"] = None
            STATE["workspace"] = None
            STATE["workspace_config"] = None
            STATE["last_error"] = None
        return jsonify({"ok": True, "selection": serialize_selection(sel)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/analysis/start")
def api_analysis_start():
    """V5.2 simple mode: user selects a waterway; the app does the rest.

    Internally this still performs careful gauge screening, caching, terrain,
    rating, NWM, inundation, history, rainfall, and optional Flood Hub work.
    The user is not asked to understand or manually operate those substeps.
    """
    payload = request.get_json(force=True) or {}
    force_static = bool(payload.get("force_static", False))
    with LOCK:
        sel = STATE.get("selected_channel")
        watersheds = STATE.get("watersheds")
    if not sel:
        return jsonify({"error": "Click a bayou/stream on the map first."}), 409
    rcfg = regional_cfg()

    def worker(jid):
        cache = RUNTIME_ROOT / "data" / "regional_cache" / "gauges" / str(sel["name"]).replace(" ", "_")[:50]
        update_job(
            jid, step="A", progress=3,
            message=f"Selected {sel['name']}. Finding the best downstream USGS control automaticallyâ€¦",
            help_text="You do not need to choose a gauge. The app checks same-waterway naming, watershed connectivity, and recent discharge.",
        )
        base_radius = float(rcfg["regional"].get("gauge_search_radius_km", 30.0))
        base_n = max(20, int(rcfg["regional"].get("gauge_candidates_to_test", 8)))
        best, candidates = choose_best_gauge(
            float(sel["click_lon"]), float(sel["click_lat"]),
            selected_channel_geom=sel.get("geometry"),
            selected_channel_name=sel.get("name"),
            radius_km=base_radius, max_candidates_to_test=base_n, cache_root=cache,
        )
        if best is None:
            update_job(jid, step="A", progress=10, message="No usable control in the first search; looking farther downstream on the same waterwayâ€¦")
            best, candidates2 = choose_best_gauge(
                float(sel["click_lon"]), float(sel["click_lat"]),
                selected_channel_geom=sel.get("geometry"),
                selected_channel_name=sel.get("name"),
                radius_km=max(60.0, base_radius * 2.0), max_candidates_to_test=max(35, base_n), cache_root=cache,
            )
            if candidates2:
                candidates = candidates2
        if best is None:
            with LOCK:
                STATE["gauge_candidates"] = candidates or []
                STATE["selected_gauge_site"] = None
            raise RuntimeError(
                "I could not find a same-waterway USGS discharge gauge that can safely control this selected reach. "
                "The app stopped instead of modeling a different bayou. Try another point on this waterway; an ungauged NWM/Flood-Hub mode can be added later."
            )

        with LOCK:
            STATE["gauge_candidates"] = candidates or [best]
            STATE["selected_gauge_site"] = best["site_no"]
        update_job(
            jid, step="B", progress=15,
            message=f"Using USGS {best['site_no']} â€” {best['name']}. Preparing/reusing the bayou workspaceâ€¦",
            help_text="This gauge was selected automatically because it is on the same named waterway, downstream of the selected point, and has recent discharge.",
        )

        watershed = watershed_for_click(watersheds, sel["click_lon"], sel["click_lat"]) if watersheds is not None else None
        prospective = workspace_path(RUNTIME_ROOT, sel["name"], best["site_no"])
        ws_exists = prospective.exists() and (prospective / "config" / "config.yaml").exists()
        ws, config_path, _ = create_workspace(
            RUNTIME_ROOT, sel, best, base_cfg=base_workspace_settings(), watershed=watershed,
            overwrite_config=not ws_exists,
        )
        with LOCK:
            STATE["workspace"] = str(ws)
            STATE["workspace_config"] = str(config_path)

        # Old V5 workspaces may contain the dominant-branch terrain. Rebuild it
        # once so V5.2 honors the selected polyline branch.
        terrain_qa = json_file(ws / "data" / "processed" / "terrain_qa.json") or {}
        rebuild_for_branch = terrain_qa.get("mainstem_mode") != "selected_branch"
        do_force = bool(force_static or rebuild_for_branch)

        def simple_progress(evt):
            step = str(evt.get("step", ""))
            mapping = {"00":20,"01":32,"02":50,"03":56,"04":63,"05":72,"06":78,"07":84,"08":91,"09":94,"10":96,"CACHE":70}
            labels = {
                "00":"Building the drainage area", "01":"Preparing cached DEM/topography",
                "02":"Tracing your selected bayou and deriving HAND/terrain", "03":"Connecting NOAA NWM",
                "04":"Preparing flow-stage rating curve", "05":"Extracting local banks/cross-sections",
                "06":"Reading current USGS flow", "07":"Reading future NWM streamflow",
                "08":"Creating flood depth and extent", "09":"Preparing display outputs", "10":"Finalizing model",
                "CACHE":"Reusing cached terrain model",
            }
            update_job(
                jid, step=step, progress=mapping.get(step, 20),
                message=labels.get(step, evt.get("message", "Workingâ€¦")),
                status=evt.get("status", "running"),
                help_text=evt.get("message", "The app is working in the background. Keep this page open."),
            )

        result = run_full_pipeline(
            APP_ROOT, ws, config_path, rcfg, force_static=do_force, progress=simple_progress,
        )

        update_job(
            jid, step="DATA", progress=97,
            message="Collecting the gauge history, rainfall forecast, and optional Flood Hub informationâ€¦",
            help_text="These are supplemental dashboard datasets. A temporary rainfall/Flood Hub failure will not invalidate the completed USGS/NWM/LiDAR flood screen.",
        )
        cfg_loaded = load_config(config_path)
        bundle = collect_intelligence_bundle(ws, cfg_loaded, sel, best)
        update_job(
            jid, step="DONE", progress=100, status="done",
            message="Flood intelligence package is ready.",
            help_text="The dashboard now contains current and forecast streamflow, long-term gauge trend when available, rainfall forecast, terrain outputs, rating curve, flood depth/extent, and optional Flood Hub information.",
        )
        return {
            "workspace": str(ws),
            "gauge": safe_gauge(best),
            "scenario": (result.get("live") or {}).get("scenario_label"),
            "data_sources": bundle,
        }

    jid = new_job("analysis", f"Run complete flood intelligence â€” {sel['name']}", worker)
    return jsonify({"job_id": jid})


@app.post("/api/gauges/start")
def api_gauges_start():
    with LOCK:
        sel = STATE.get("selected_channel")
    if not sel:
        return jsonify({"error": "Select a bayou first."}), 409
    rcfg = regional_cfg()

    def worker(jid):
        update_job(jid, step="G1", progress=5, message="Searching nearby USGS streamgagesâ€¦", help_text="This step checks multiple nearby gauges rather than blindly using the closest one.")
        cache = RUNTIME_ROOT / "data" / "regional_cache" / "gauges" / str(sel["name"]).replace(" ", "_")[:50]
        update_job(jid, step="G2", progress=25, message="Testing upstream-basin connectivity and recent gauge dataâ€¦", help_text="Each candidate may require an NLDI basin request and a recent USGS observation request, so this can take a minute.")
        base_radius = float(rcfg["regional"].get("gauge_search_radius_km", 30.0))
        base_n = max(20, int(rcfg["regional"].get("gauge_candidates_to_test", 8)))
        best, candidates = choose_best_gauge(
            float(sel["click_lon"]), float(sel["click_lat"]),
            selected_channel_geom=sel.get("geometry"),
            selected_channel_name=sel.get("name"),
            radius_km=base_radius,
            max_candidates_to_test=base_n,
            cache_root=cache,
        )
        if not candidates:
            raise RuntimeError("No USGS streamgage candidates were found near this selected bayou location.")

        # Dense urban networks often have many inactive or water-quality-only
        # sites closer than the useful live-discharge control. If the first pass
        # finds no supported gauge, automatically widen the search rather than
        # leaving the user stuck on Step 2.
        if best is None:
            update_job(
                jid, step="G2B", progress=62,
                message="No same-bayou live control passed the first screen; expanding the searchâ€¦",
                help_text="The app is looking farther along the same bayou for an active discharge gauge. Do not choose an unrelated tributary gauge just because it is closer.",
            )
            best2, candidates2 = choose_best_gauge(
                float(sel["click_lon"]), float(sel["click_lat"]),
                selected_channel_geom=sel.get("geometry"),
                selected_channel_name=sel.get("name"),
                radius_km=max(base_radius * 2.0, 60.0),
                max_candidates_to_test=max(base_n, 35),
                cache_root=cache,
            )
            if candidates2:
                candidates = candidates2
            if best2 is not None:
                best = best2

        with LOCK:
            STATE["gauge_candidates"] = candidates
            # Never auto-select a gauge that failed the hydraulic + live-flow
            # screen. This avoids the confusing V5.1 state where Step 2 looked
            # complete but Step 3 remained permanently locked.
            STATE["selected_gauge_site"] = best["site_no"] if best else None

        if best is not None:
            update_job(
                jid, step="G3", progress=100, message="Same-bayou supported control gauge found.", status="done",
                help_text=f"Recommended USGS {best['site_no']} ({best['name']}). It matches the selected bayou name, contains the click in its upstream basin, and has recent discharge. Step 3 is now unlocked.",
            )
        else:
            update_job(
                jid, step="G3", progress=100, message="Gauge screening finished, but no same-bayou live-discharge control was found.", status="done",
                help_text="Do not force an unrelated gauge. Try another point on the same bayou or add an HCFCD/NWS gauge fallback for this reach.",
            )
        return {"recommended": safe_gauge(best) if best else None, "count": len(candidates)}

    jid = new_job("gauges", "Find hydraulically relevant gauges", worker)
    return jsonify({"job_id": jid})


@app.post("/api/gauges/select")
def api_gauge_select():
    payload = request.get_json(force=True) or {}
    site = str(payload.get("site_no", ""))
    with LOCK:
        candidate = next(
            (x for x in STATE.get("gauge_candidates", []) if str(x.get("site_no")) == site),
            None,
        )
        if candidate is None:
            return jsonify({"error": "Gauge is not in the current candidate list."}), 400
        if not candidate.get("auto_supported"):
            return jsonify({
                "error": "This gauge cannot unlock Step 3 automatically. It must match the selected bayou name, contain the selected point in its upstream basin, and have recent discharge."
            }), 409
        STATE["selected_gauge_site"] = site
        STATE["workspace"] = None
        STATE["workspace_config"] = None
    return jsonify({"ok": True})


@app.post("/api/pipeline/start")
def api_pipeline_start():
    payload = request.get_json(force=True) or {}
    force_static = bool(payload.get("force_static", False))
    datum_override = payload.get("datum_override", "")
    with LOCK:
        sel = STATE.get("selected_channel")
        site = STATE.get("selected_gauge_site")
        candidates = list(STATE.get("gauge_candidates", []))
        watersheds = STATE.get("watersheds")
    if not sel or not site:
        return jsonify({"error": "Select a bayou and confirm a control gauge first."}), 409
    gauge = next((g for g in candidates if str(g.get("site_no")) == str(site)), None)
    if not gauge:
        return jsonify({"error": "Selected gauge is unavailable."}), 409
    if not gauge.get("auto_supported"):
        return jsonify({"error": "The selected gauge is not an automatic same-bayou control. Step 3 requires: same named waterway + upstream-basin connectivity + recent discharge."}), 409
    rcfg = regional_cfg()

    def worker(jid):
        update_job(jid, step="SETUP", progress=2, message="Creating or reopening the persistent bayou workspaceâ€¦", help_text="Static terrain products are saved by bayou/gauge and reused on future refreshes.")
        watershed = watershed_for_click(watersheds, sel["click_lon"], sel["click_lat"]) if watersheds is not None else None
        prospective = workspace_path(RUNTIME_ROOT, sel["name"], gauge["site_no"])
        ws_exists = prospective.exists() and (prospective / "config" / "config.yaml").exists()
        ws, config_path, _ = create_workspace(
            APP_ROOT, sel, gauge, base_cfg=base_workspace_settings(), watershed=watershed,
            overwrite_config=not ws_exists,
        )
        patch_workspace_datum(config_path, datum_override)
        with LOCK:
            STATE["workspace"] = str(ws)
            STATE["workspace_config"] = str(config_path)
        result = run_full_pipeline(
            APP_ROOT, ws, config_path, rcfg, force_static=force_static,
            progress=pipeline_callback(jid),
        )
        update_job(jid, step="10", progress=100, message="Bayou model prepared and live flood screen refreshed.", status="done", help_text="You can now use Live Dashboard. Future refreshes reuse the static cache.")
        return {"workspace": str(ws), "static_status": result.get("static", {}).get("status"), "scenario": result.get("live", {}).get("scenario_label")}

    jid = new_job("pipeline", "Prepare selected bayou and run 00 â†’ 10", worker)
    return jsonify({"job_id": jid})


@app.post("/api/live/start")
def api_live_start():
    ws, cfg = workspace_paths()
    if not ws or not cfg or not static_ready(ws):
        return jsonify({"error": "Prepare the selected bayou model first."}), 409

    def worker(jid):
        update_job(jid, step="06", progress=10, message="Starting near-real-time refreshâ€¦", help_text=STEP_HELP["06"])
        meta = run_live_workspace(ws, cfg, progress=pipeline_callback(jid))
        try:
            with LOCK:
                sel = STATE.get("selected_channel")
                gauges = list(STATE.get("gauge_candidates", []))
                site = STATE.get("selected_gauge_site")
            gauge = next((g for g in gauges if str(g.get("site_no")) == str(site)), None)
            if sel and gauge:
                update_job(jid, step="DATA", progress=96, message="Refreshing NWS rainfall and Google Flood Hub comparisonâ€¦", help_text="Flood Hub is an independent AI forecast/status layer; it does not replace the local NWM/LiDAR screen.")
                refresh_live_supplemental(ws, load_config(cfg), sel, gauge)
        except Exception as exc:
            update_job(jid, step="DATA", progress=97, message=f"Supplemental Google/NWS refresh warning: {exc}", help_text="The core USGS/NWM/LiDAR flood screen is still valid for this refresh.")
        update_job(jid, step="10", progress=100, message="Near-real-time refresh complete.", status="done", help_text="USGS, NWM, Flood Hub comparison and flood-screen outputs are current for this refresh.")
        return {"scenario": meta.get("scenario_label")}

    jid = new_job("live", "Refresh USGS + NWM + inundation", worker)
    return jsonify({"job_id": jid})


@app.get("/api/jobs/<jid>")
def api_job(jid):
    with LOCK:
        j = copy.deepcopy(JOBS.get(jid))
    if not j:
        return jsonify({"error": "Unknown job"}), 404
    now = time.time()
    start = j.get("started_at") or j.get("created_at") or now
    j["elapsed_seconds"] = max(0, (j.get("finished_at") or now) - start)
    return jsonify(j)


@app.get("/api/dashboard")
def api_dashboard():
    ws, _ = workspace_paths()
    if not ws:
        return jsonify({"ready": False})
    meta = json_file(ws / "outputs" / "dashboard_scenario.json") or {}
    nwm_path = ws / "outputs" / "latest_nwm_hydrograph.csv"
    usgs_path = ws / "outputs" / "latest_usgs_observations.csv"
    profile_path = ws / "outputs" / "latest_bank_wse_profile.csv"
    nwm = []
    usgs = []
    profile = []
    try:
        if nwm_path.exists():
            df = pd.read_csv(nwm_path).tail(250)
            nwm = df.where(pd.notna(df), None).to_dict("records")
    except Exception:
        pass
    try:
        if usgs_path.exists():
            df = pd.read_csv(usgs_path).tail(500)
            usgs = df.where(pd.notna(df), None).to_dict("records")
    except Exception:
        pass
    try:
        if profile_path.exists():
            df = pd.read_csv(profile_path)
            keep = [c for c in ["station_m", "local_flow_cfs", "bankfull_capacity_cfs", "scenario_wse_ft_navd88", "lower_bank_final_ft_navd88", "overtopped"] if c in df.columns]
            profile = df[keep].where(pd.notna(df[keep]), None).to_dict("records")
    except Exception:
        pass
    historical_flow = []
    rainfall_history = []
    rainfall_forecast = []
    floodhub = json_file(ws / "outputs" / "floodhub_summary.json") or {}
    source_summary = json_file(ws / "outputs" / "data_sources_summary.json") or {}
    try:
        hp = ws / "outputs" / "historical_daily_streamflow.csv"
        if hp.exists():
            df = pd.read_csv(hp)
            # Long records are downsampled only for browser rendering; the full CSV stays on disk.
            if len(df) > 4000:
                stride = max(1, len(df) // 4000)
                df = df.iloc[::stride].copy()
            historical_flow = df.where(pd.notna(df), None).to_dict("records")
    except Exception:
        pass
    try:
        rp = ws / "outputs" / "historical_daily_precipitation.csv"
        if rp.exists():
            df = pd.read_csv(rp)
            if len(df) > 4000:
                stride = max(1, len(df) // 4000)
                df = df.iloc[::stride].copy()
            rainfall_history = df.where(pd.notna(df), None).to_dict("records")
    except Exception:
        pass
    try:
        fp = ws / "outputs" / "nws_forecast_precipitation.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            rainfall_forecast = df.where(pd.notna(df), None).to_dict("records")
    except Exception:
        pass
    floodhub_forecast = []
    try:
        fp = ws / "outputs" / "floodhub_forecast.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            # Prefer the newest issued forecast in the browser.
            if "issued_time" in df.columns and len(df):
                df["issued_time"] = pd.to_datetime(df["issued_time"], utc=True, errors="coerce")
                newest = df["issued_time"].max()
                if pd.notna(newest):
                    df = df[df["issued_time"] == newest].copy()
            floodhub_forecast = df.where(pd.notna(df), None).to_dict("records")
    except Exception:
        pass

    overlay = depth_overlay(ws) if (ws / "outputs" / "latest_depth.tif").exists() else None
    return jsonify({
        "ready": bool(meta),
        "metadata": meta,
        "nwm": nwm,
        "usgs": usgs,
        "profile": profile,
        "historical_flow": historical_flow,
        "rainfall_history": rainfall_history,
        "rainfall_forecast": rainfall_forecast,
        "floodhub": floodhub,
        "floodhub_forecast": floodhub_forecast,
        "data_sources": source_summary,
        "depth_overlay": overlay,
    })


LAYER_MAP = {
    "selected_channel": ("data/processed/selected_channel.geojson",),
    "basin": ("data/processed/model_basin.geojson", "data/processed/hunting_bayou_basin.geojson"),
    "mainstem": ("data/processed/mainstem.geojson",),
    "extent": ("outputs/latest_inundation.geojson",),
    "potential": ("outputs/latest_potential_inundation.geojson",),
    "channel_corridor": ("outputs/latest_channel_corridor.geojson",),
    "overtopped": ("outputs/latest_overtopped_sections.geojson",),
    "seeds": ("outputs/latest_floodplain_seeds.geojson",),
    "max_depth": ("outputs/latest_max_depth_point.geojson",),
    "bank_points": ("data/processed/bank_profile_points.geojson",),
}


@app.get("/api/layer/<name>")
def api_layer(name):
    ws, _ = workspace_paths()
    if not ws or name not in LAYER_MAP:
        return jsonify({"type": "FeatureCollection", "features": []})
    for rel in LAYER_MAP[name]:
        p = ws / rel
        if p.exists():
            return jsonify(file_feature_collection(p))
    return jsonify({"type": "FeatureCollection", "features": []})


@app.get("/api/gauge-layer")
def api_gauge_layer():
    with LOCK:
        site = STATE.get("selected_gauge_site")
        gauges = STATE.get("gauge_candidates", [])
    g = next((x for x in gauges if str(x.get("site_no")) == str(site)), None)
    if not g:
        return jsonify({"type": "FeatureCollection", "features": []})
    return jsonify({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"site_no": g.get("site_no"), "name": g.get("name")},
            "geometry": {"type": "Point", "coordinates": [float(g["longitude"]), float(g["latitude"])]},
        }],
    })


@app.get("/api/depth/image")
def api_depth_image():
    ws, _ = workspace_paths()
    if not ws:
        return jsonify({"error": "No active workspace"}), 404
    info = depth_overlay(ws)
    if not info:
        return jsonify({"error": "No depth raster"}), 404
    p = ws / "outputs" / "web_depth_overlay.png"
    return send_file(p, mimetype="image/png", max_age=0)


@app.get("/api/workspaces")
def api_workspaces():
    return jsonify(recent_workspaces())


@app.post("/api/workspaces/open")
def api_workspace_open():
    payload = request.get_json(force=True) or {}
    p = Path(str(payload.get("path", ""))).resolve()
    root = (RUNTIME_ROOT / "workspaces").resolve()
    if not p.exists() or root not in p.parents:
        return jsonify({"error": "Invalid workspace path."}), 400
    load_workspace_into_state(p)
    return jsonify({"ok": True})


@app.get("/api/cache")
def api_cache():
    rcfg = regional_cfg()
    cache = Path(rcfg["dem_cache"]["cache_root"])
    tiles = list(cache.glob("*.tif")) + list(cache.glob("*.tiff"))
    size = sum(p.stat().st_size for p in tiles if p.exists())
    vrt = cache / "hgac_dem_cache.vrt"
    gdb = abs_project_path(rcfg["dem_cache"].get("arcpy_mosaic_gdb", "data/regional_cache/HGAC_DEM_Cache.gdb"))
    return jsonify({
        "cache_root": str(cache), "tile_count": len(tiles), "bytes": size,
        "vrt_exists": vrt.exists(), "vrt": str(vrt),
        "arcpy_gdb_exists": gdb.exists(), "arcpy_gdb": str(gdb),
        "regional_dem_path": rcfg["dem_cache"].get("regional_dem_path", ""),
    })


@app.post("/api/cache/vrt/start")
def api_cache_vrt_start():
    rcfg = regional_cfg(); cache = Path(rcfg["dem_cache"]["cache_root"]); out = cache / "hgac_dem_cache.vrt"
    def worker(jid):
        update_job(jid, step="DEM", progress=10, message="Building GDAL virtual mosaic from cached DEM tilesâ€¦", help_text="This does not duplicate raster data; it creates a lightweight virtual mosaic index.")
        p = build_virtual_mosaic(cache, out)
        update_job(jid, step="DEM", progress=100, message="Regional VRT cache is ready.", status="done", help_text="Point dem_cache.regional_dem_path to this VRT for fastest bayou cropping.")
        return {"path": str(p)}
    return jsonify({"job_id": new_job("cache_vrt", "Build regional DEM VRT", worker)})


@app.post("/api/cache/arcpy/start")
def api_cache_arcpy_start():
    rcfg = regional_cfg(); cache = Path(rcfg["dem_cache"]["cache_root"])
    gdb = abs_project_path(rcfg["dem_cache"].get("arcpy_mosaic_gdb", "data/regional_cache/HGAC_DEM_Cache.gdb"))
    name = rcfg["dem_cache"].get("arcpy_mosaic_name", "HGAC_DEM_CACHE")
    def worker(jid):
        update_job(jid, step="ARCPY", progress=5, message="Creating/updating ArcGIS Mosaic Datasetâ€¦", help_text="Run the app with an ArcGIS Pro Python environment if ArcPy is not visible in the current .venv.")
        p = build_arcpy_mosaic_dataset(cache, gdb, name)
        update_job(jid, step="ARCPY", progress=100, message="ArcGIS Mosaic Dataset is ready.", status="done")
        return {"path": str(p)}
    return jsonify({"job_id": new_job("cache_arcpy", "Build ArcPy DEM mosaic", worker)})


def scenario_depth_overlay(workspace: Path):
    tif = workspace / "outputs" / "scenario_latest_depth.tif"
    png = workspace / "outputs" / "web_scenario_depth_overlay.png"
    meta = workspace / "outputs" / "web_scenario_depth_overlay.json"
    if not tif.exists():
        return None
    if png.exists() and meta.exists() and png.stat().st_mtime >= tif.stat().st_mtime:
        obj = json_file(meta)
        if obj:
            return obj

    with rasterio.open(tif) as src:
        left, bottom, right, top = src.bounds
        dst_transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, left, bottom, right, top
        )
        max_dim = 1400
        scale = min(1.0, max_dim / max(width, height))
        width = max(1, int(width * scale)); height = max(1, int(height * scale))
        if scale < 1:
            dst_transform, _, _ = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, left, bottom, right, top,
                dst_width=width, dst_height=height,
            )
        dst = np.zeros((height, width), dtype="float32")
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=0,
        )
    valid = np.isfinite(dst) & (dst > 0.02)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if valid.any():
        vmax = max(float(np.nanpercentile(dst[valid], 98)), 0.25)
        x = np.clip(dst / vmax, 0, 1)
        stops = np.array([0.0, 0.2, 0.5, 0.78, 1.0])
        colors = np.array([[255,235,170],[255,183,77],[244,109,67],[194,43,84],[115,26,96]], dtype=float)
        flat=x.ravel(); rgb=np.zeros((flat.size,3),dtype=float)
        for c in range(3): rgb[:,c]=np.interp(flat,stops,colors[:,c])
        rgba[...,:3]=rgb.reshape(height,width,3).astype(np.uint8); rgba[...,3][valid]=190
    Image.fromarray(rgba, mode="RGBA").save(png)
    west, south, east, north = rasterio.transform.array_bounds(height, width, dst_transform)
    obj={"image_url":"/api/scenario/depth/image","bounds":[[float(south),float(west)],[float(north),float(east)]],"mtime":tif.stat().st_mtime}
    meta.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return obj


@app.post("/api/scenario/start")
def api_scenario_start():
    ws, cfg_path = workspace_paths()
    if not ws or not cfg_path or not static_ready(ws):
        return jsonify({"error": "Run the complete bayou analysis first so terrain, rating and bank products are cached."}), 409
    payload = request.get_json(force=True) or {}
    try:
        rainfall_in = float(payload.get("rainfall_in", 0))
        duration_hours = float(payload.get("duration_hours", 0))
    except Exception:
        return jsonify({"error": "Rainfall and duration must be numbers."}), 400
    if rainfall_in <= 0 or duration_hours <= 0:
        return jsonify({"error": "Rainfall depth and duration must both be greater than zero."}), 400
    wetness = str(payload.get("wetness", "normal"))
    cn = payload.get("curve_number")

    def worker(jid):
        def cb(evt):
            step = str(evt.get("step", "S"))
            progress_map = {"S1": 25, "S2": 55, "S3": 100}
            update_job(
                jid, step=step, progress=progress_map.get(step, 30),
                message=evt.get("message", "Running rainfall scenarioâ€¦"),
                status=evt.get("status", "running"),
                help_text=(
                    "This is a hypothetical rainfall sensitivity screen. Google Flood Hub is not forced with custom rainfall; "
                    "the scenario uses NRCS-CN/SCS runoff plus the cached local LiDAR/HAND flood screen."
                ),
            )
        meta = run_rainfall_scenario(
            ws, cfg_path, rainfall_in=rainfall_in, duration_hours=duration_hours,
            wetness=wetness, curve_number=cn, progress=cb,
        )
        return {"scenario": meta.get("scenario_label"), "area_sqmi": meta.get("inundated_area_sqmi"), "max_depth_m": meta.get("max_depth_m")}

    jid = new_job("scenario", f"Rainfall scenario â€” {rainfall_in:g} in over {duration_hours:g} h", worker)
    return jsonify({"job_id": jid})


@app.get("/api/scenario/dashboard")
def api_scenario_dashboard():
    ws, _ = workspace_paths()
    if not ws:
        return jsonify({"ready": False})
    meta = json_file(ws / "outputs" / "scenario_dashboard.json") or {}
    hydro=[]
    hp=ws / "outputs" / "scenario_rainfall_hydrograph.csv"
    try:
        if hp.exists():
            df=pd.read_csv(hp)
            hydro=df.where(pd.notna(df),None).to_dict("records")
    except Exception:
        pass
    return jsonify({"ready": bool(meta), "metadata": meta, "hydrograph": hydro, "depth_overlay": scenario_depth_overlay(ws) if meta else None})


@app.get("/api/scenario/layer/<name>")
def api_scenario_layer(name):
    ws, _ = workspace_paths()
    if not ws:
        return jsonify({"type":"FeatureCollection","features":[]})
    mapping={
        "extent":"outputs/scenario_latest_inundation.geojson",
        "potential":"outputs/scenario_latest_potential_inundation.geojson",
        "overtopped":"outputs/scenario_latest_overtopped_sections.geojson",
        "seeds":"outputs/scenario_latest_floodplain_seeds.geojson",
        "max_depth":"outputs/scenario_latest_max_depth_point.geojson",
    }
    rel=mapping.get(name)
    if not rel:
        return jsonify({"type":"FeatureCollection","features":[]})
    p=ws / rel
    return jsonify(file_feature_collection(p)) if p.exists() else jsonify({"type":"FeatureCollection","features":[]})


@app.get("/api/scenario/depth/image")
def api_scenario_depth_image():
    ws, _ = workspace_paths()
    if not ws:
        return jsonify({"error":"No active workspace"}),404
    info=scenario_depth_overlay(ws)
    if not info:
        return jsonify({"error":"No scenario depth raster"}),404
    return send_file(ws / "outputs" / "web_scenario_depth_overlay.png", mimetype="image/png", max_age=0)


@app.get("/api/floodhub/layer")
def api_floodhub_layer():
    ws, _ = workspace_paths()
    if not ws:
        return jsonify({"type":"FeatureCollection","features":[]})
    p=ws / "outputs" / "floodhub_inundation.geojson"
    return jsonify(file_feature_collection(p)) if p.exists() else jsonify({"type":"FeatureCollection","features":[]})


@app.get("/health")
def health():
    return jsonify({"ok": True, "version": "5.3"})


def start_catalog_on_boot():
    with LOCK:
        STATE["catalog_status"] = "running"
    Thread(target=load_catalog_worker, daemon=True, name="hgac-catalog").start()


def main():
    start_catalog_on_boot()

    # Render and similar platforms provide PORT. Local Windows runs continue
    # to use HGAC_APP_PORT/8765.
    port = int(os.getenv("PORT", os.getenv("HGAC_APP_PORT", "8765")))
    host = os.getenv("HOST", "0.0.0.0" if PUBLIC_MODE else "127.0.0.1")

    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{port}"

    print("\nH-GAC Regional Flood Intelligence V5.3")
    print("Regional flood-intelligence web application")
    print(f"Listening on {host}:{port}")
    print(f"Runtime data root: {RUNTIME_ROOT}")

    # A cloud container must not try to launch a browser.
    if not PUBLIC_MODE:
        print(f"Opening: {url}")
        print("Press Ctrl+C in this PowerShell window to stop the app.\n")
        Timer(1.4, lambda: webbrowser.open(url)).start()

    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

