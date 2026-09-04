# H-GAC Regional Flood Intelligence V5.1

## Why V5.1

V5.1 replaces the default Streamlit interface with a guided Flask/Leaflet local web application. The flood-modeling engine remains the V5/V4.3 channel-excluded, volume-limited workflow; the change is primarily operational UX.

### Key UX changes

- Light professional theme by default; no black-on-black sidebar issue.
- The app opens directly to an H-GAC regional map.
- A four-step workflow makes the required next action explicit:
  1. Select bayou.
  2. Find/confirm a hydraulically relevant USGS gauge.
  3. Prepare the static bayou model and perform the first live run.
  4. Refresh near-real-time USGS/NWM/inundation products.
- Gauge discovery and the 00→10 pipeline run in background worker threads.
- A persistent **Process Activity** drawer shows:
  - current step,
  - current operation,
  - percentage,
  - elapsed time,
  - explanation of long-running steps,
  - recent pipeline messages,
  - success/error status.
- Static model cache is clearly distinguished from live refresh.
- Cached bayou workspaces can be reopened from the UI.
- Operational map layers include the selected bayou, control gauge, basin, derived mainstem, channel corridor, preferred flood extent, potential QA extent, overtopped sections, floodplain entry seeds and maximum-depth QA point.
- Depth raster is rendered as a semi-transparent map overlay.
- NOAA NWM, USGS, hydraulic-profile and QA panels are built into the same application.
- DEM cache management supports GDAL VRT and optional ArcPy Mosaic Dataset creation.

## Install

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

.\INSTALL_V5_1.ps1 `
  -ProjectRoot "C:\Users\Durga\Downloads\HGAC_Hunting_Bayou_Flood_Prototype\hgac_hunting_bayou_flood"
```

Then:

```powershell
cd "C:\Users\Durga\Downloads\HGAC_Hunting_Bayou_Flood_Prototype\hgac_hunting_bayou_flood"
.\.venv\Scripts\Activate.ps1
pip install -r requirements_v5_1.txt
.\RUN_V5_1.ps1
```

The browser opens automatically at `http://127.0.0.1:8765`.

Keep the PowerShell window open while using the application. Ctrl+C stops the local server.

## Regional catalog

V5.1 initializes the H-GAC boundary/waterway catalog automatically in the background. The map shows a clear loading overlay until it is ready. No separate Streamlit command is required.

## First bayou run versus later refresh

The first preparation may be slow because it can include DEM download/crop, terrain conditioning, flow routing, HAND/mainstem derivation, rating/datum QA and LiDAR bank cross-sections. Those products are stored in the bayou/gauge workspace.

Later **Refresh live now** operations reuse the static cache and update USGS observations, NOAA NWM hydrograph, WSE, overtopping and inundation only.

## DEM cache strategy

The regional cache remains shared across workspaces. Do not create a giant duplicate 1-m TIFF unless operationally required. The preferred fast approach is:

1. Cache source DEM tiles once.
2. Build a lightweight GDAL VRT over cached coverage.
3. Set `dem_cache.regional_dem_path` in `config/v5_regional.yaml` to that VRT.
4. Crop the selected gauge basin at analysis resolution and the mainstem corridor at hydraulic resolution on demand.

The app's **DEM cache** tab can build/update the VRT. ArcPy Mosaic Dataset creation is also exposed for ArcGIS Pro environments.

## Legacy UI

The former Streamlit interface is retained as `legacy_streamlit_app.py` for comparison/troubleshooting, but it is no longer the recommended launcher.
