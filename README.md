# HGAC Hunting Bayou Flood Intelligence Prototype

This is an end-to-end **research/decision-support prototype** for the 16.1-mi²
Hunting Bayou drainage area upstream of USGS 08075770 at IH-610.

## Core architecture

Weather/hydrology
→ USGS observations
+ NOAA National Water Model guidance
+ Google Flood Hub forecasts when API access is approved
→ discharge/stage relationship
→ Houston 2024 LiDAR DEM
→ HAND terrain-screening inundation
→ Streamlit dashboard

The HAND module is intentionally a first prototype. Replace it with calibrated
HEC-RAS 2D for engineering/operational flood depth and extent.

## Install on Windows

```powershell
cd hgac_hunting_bayou_flood
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Or double-click / run:

```bat
setup_windows.bat
```

## Step 1 — Get the Hunting Bayou basin

```powershell
python scripts\00_get_basin.py
```

Uses the USGS NLDI basin endpoint for `USGS-08075770`, saves
`data/processed/hunting_bayou_basin.geojson`, and prints the NHDPlus COMID.

## Step 2 — Download the 2024 Houston LiDAR DEM

```powershell
python scripts\01_download_dem.py
```

The downloader scans all four public `TX_Houston_B24` USGS work units, inspects
GeoTIFF tile bounds, downloads only tiles intersecting the NLDI basin, and builds:

- `dem_hydraulic_1m.tif`
- `dem_analysis_2m.tif`

If your corporate network/GDAL cannot range-read remote GeoTIFF headers, manually
download the basin tiles into `data/raw/usgs_dem_tiles` and modify the script to
use those local TIFFs.

## Step 3 — Build HAND

```powershell
python scripts\02_build_terrain.py
```

Produces:

- `hand_2m.tif`
- `upstream_area_km2.tif`
- `stream_mask.tif`

The stream-area threshold is configurable in `config/config.yaml`.

## Step 4 — Discover NWM + Flood Hub IDs

```powershell
python scripts\03_discover_sources.py
```

The code gets the NLDI COMID and tries it as the NOAA NWPS/NWM reach ID.

For Google, after API approval put the key in `.env`:

```text
FLOODHUB_API_KEY=YOUR_KEY
```

Run the discovery script again. It uses the official Google
`gauges:searchGaugesByArea` endpoint around Hunting Bayou. Put the preferred
Google gauge ID into `config/config.yaml` under `floodhub.gauge_id`.

## Step 5 — Build/obtain a rating curve

Try:

```powershell
python scripts\04_build_rating_curve.py
```

The configured NWS identifier is `HTGT2` (Hunting Bayou at Loop 610 East).
The script requests NOAA NWPS rating information and tries to write:

```text
data/processed/rating_curve.csv
```

with:

```csv
flow_cfs,stage_ft
```

If automatic parsing does not work, populate this CSV manually from an
authoritative current rating table. Do not mix old stage data across datum
changes without reconciliation.

## Step 6 — Fetch live data

```powershell
python scripts\05_fetch_live_data.py
```

This retrieves USGS observations, NOAA NWM output if the reach resolves, and
Google forecasts once the Google key/gauge are configured.

NOAA NWM is model guidance, not an official NWS forecast product.

**Important unit check:** the project intentionally defaults `nwm.value_unit` to
`unknown`. Inspect the live NWPS response/documentation for the selected reach,
then set it to `cfs` or `cms` in `config/config.yaml`. The dashboard will not
feed NWM values into a cfs rating curve until this is explicitly confirmed.

## Step 7 — Generate a first screening flood map

Without a rating curve:

```powershell
python scripts\06_make_inundation.py --stage-rise-ft 3
```

With a forecast discharge and rating curve:

```powershell
python scripts\06_make_inundation.py --flow-cfs 3000
```

The prototype uses:

`screening depth = max(stage rise - HAND, 0)`

Outputs:

- `outputs/latest_depth.tif`
- `outputs/latest_inundation.geojson`

## Step 8 — Launch the dashboard

```powershell
streamlit run app.py
```

## Validation

If you obtain an observed flood raster on the same grid:

```powershell
python scripts\07_validate_raster.py --pred outputs\latest_depth.tif --obs data\validation\observed.tif
```

It returns precision, recall, F1 and IoU.

## Earth Engine

`gee/hunting_bayou_layers.js` is the companion GEE script. The recommended
architecture is:

Python live APIs → GeoTIFF/GeoJSON → GCS/EE Asset → GEE analysis.

Use GEE for Sentinel-1 flood validation, JRC surface water, land cover,
imperviousness and regional overlays. Do not use GEE as the live REST API
backend for Flood Hub/NWM.

## Phase 2 — engineering-grade hydraulic model

After the end-to-end prototype works:

1. Build HEC-RAS 2D terrain from the 1 m LiDAR DEM.
2. Add HCFCD channels, bridges, culverts, detention features, levees and controls.
3. Add calibrated Manning roughness.
4. Use USGS/HCFCD events and high-water marks for calibration.
5. Run a library of discharge/stage scenarios.
6. Export depth/extent rasters.
7. Let the dashboard select/interpolate the precomputed result based on the
   NWM/Google/local ensemble forecast.

This is preferable to running HEC-RAS live on every dashboard refresh.

## Scientific cautions

- Flood Hub does not expose a documented endpoint for uploading your DEM into
  Google's production forecast model.
- Verify Google gauge/model units before combining Google values numerically
  with NWM discharge.
- Verify the NWM reach crosswalk.
- HAND does not represent storm sewers, culverts, pumps, reservoir operations,
  tidal backwater or full 2D hydraulics.
- Houston pluvial/urban flash flooding and coastal/tidal flooding should be
  modeled as additional components.
