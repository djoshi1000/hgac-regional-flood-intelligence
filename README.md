<div align="center">

# H-GAC Regional Flood Intelligence

### Bayou-scale flood forecasting, scenario screening, and geospatial decision support

**USGS observations · NOAA National Water Model · Google Flood Hub · LiDAR/DEM terrain analysis**

> **Research prototype** — not an official flood warning, regulatory floodplain product, or engineering hydraulic model.

</div>

---

## Overview

**H-GAC Regional Flood Intelligence** is an experimental geospatial flood-analysis platform for rivers, bayous, and streams in the Houston–Galveston region.

The workflow is designed to make flood information easier to explore from one interface. A user selects a waterway, and the application can automatically combine:

- observed streamflow and stage from **USGS**,
- forecast streamflow guidance from the **NOAA National Water Model (NWM)**,
- independent AI-based forecast/status information from **Google Flood Hub**,
- high-resolution **LiDAR / DEM** terrain information,
- local channel, bank, HAND, and cross-section products,
- historical streamflow trends, and
- user-defined rainfall scenarios.

The dashboard then presents hydrographs, forecast comparisons, terrain-screening flood extents, depth products, and QA information in one place.

---

## What problem does it address?

Flood information is often spread across separate systems. This project explores whether those sources can be brought together into a single regional workflow:

```text
                     SELECTED WATERWAY
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
            USGS         NOAA NWM    Google Flood Hub
        observations      forecast      AI forecast
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                     Local LiDAR / DEM
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
      HAND / terrain    bank geometry     cross sections
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                local flood-screening model
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
      hydrographs      flood extent       depth / QA
                            │
                            ▼
                    interactive dashboard
```

The intent is **not** to treat any one source as ground truth. Instead, the dashboard allows an analyst to compare observations, operational forecast guidance, AI forecast information, and local terrain response.

---

## Main capabilities

### 1. Select a river, bayou, or stream

The regional workflow is built around the waterway selected by the user.

The application can then:

- identify a hydraulically relevant USGS control,
- associate the selected waterway with its drainage area,
- resolve the corresponding NHDPlus / NWM reach where possible, and
- prepare or reuse a cached terrain workspace.

---

### 2. Current and forecast conditions

The dashboard can display:

- current USGS discharge and stage,
- recent hydrograph behavior,
- NOAA NWM forecast hydrograph,
- forecast peak flow,
- Google Flood Hub forecast/status when API access is available,
- model threshold comparisons, and
- screening-level local inundation results.

---

### 3. Google Flood Hub integration

Google Flood Hub is used as an **independent AI forecast and flood-status source**.

Where supported by the API, the application can retrieve:

- Flood Hub gauges,
- forecast values,
- flood status,
- model thresholds,
- model quality information, and
- inundation polygons when available.

The purpose is comparison:

```text
USGS observation  ←→  NOAA NWM  ←→  Google Flood Hub
```

Agreement between independent sources may strengthen situational awareness. Large disagreement is also useful information and should be investigated rather than automatically averaged away.

> Flood Hub is **not** used as a hidden replacement for USGS, NWM, or the local terrain model.

---

## Rainfall Scenario Lab

The **Scenario Lab** is for hypothetical questions such as:

> **What area could be flooded if 11 inches of rain falls over the selected basin in 5 hours?**

Example inputs:

```text
Storm total:          11 in
Storm duration:        5 h
Antecedent condition:  Dry / Normal / Wet
```

The current scenario workflow uses:

```text
rainfall
   ↓
NRCS Curve Number runoff transform
   ↓
effective runoff depth
   ↓
synthetic runoff hydrograph
   ↓
estimated discharge / peak flow
   ↓
rating / stage relationship
   ↓
local LiDAR / HAND terrain screen
   ↓
flood extent + depth + QA
```

Typical outputs include:

- estimated runoff depth,
- scenario hydrograph,
- estimated peak discharge,
- comparison with bankfull/threshold flow,
- flood extent,
- depth raster,
- flooded area,
- storage estimates, and
- uncertainty warnings.

### Important distinction

The public Google Flood Forecasting API retrieves Google's operational forecast/status products. It does **not** provide a documented endpoint for submitting an arbitrary custom storm such as *11 inches in 5 hours*.

Therefore:

- **Flood Hub** = independent real-world AI forecast/reference
- **Scenario Lab** = local hypothetical rainfall-runoff and terrain-screening workflow

---

## Terrain and flood-screening workflow

For a selected waterway, the application can:

1. obtain or delineate the gauge-controlled drainage area,
2. download or reuse required elevation tiles,
3. build a memory-safe routing DEM,
4. derive flow routing and upstream area,
5. derive a stream network and HAND,
6. trace the selected bayou branch,
7. use higher-resolution terrain near the channel,
8. extract LiDAR-derived cross sections and bank profiles,
9. relate discharge to stage/WSE where rating information is available,
10. generate channel-excluded and volume-limited flood-screening outputs.

For large basins, routing can use an adaptive coarser resolution while preserving higher-resolution terrain near the channel.

---

## Key outputs

Depending on the selected location and available data, the workflow may generate:

- flood-depth GeoTIFF,
- inundation extent GeoJSON,
- HAND raster,
- upstream-area raster,
- stream/mainstem layers,
- rating-curve products,
- channel-bank profiles,
- cross-section products,
- historical hydrographs,
- NOAA NWM forecast hydrographs,
- Google Flood Hub forecast/status information,
- rainfall-scenario hydrographs,
- flooded-area and storage summaries,
- QA diagnostics, and
- historical validation metrics.

Large generated datasets, DEM caches, workspaces, and outputs are intentionally excluded from GitHub.

---

## Accuracy and scientific status

The application combines real observations and operational forecast sources, but the **local inundation model is still a screening approximation**.

There are three separate accuracy questions:

1. **Observation accuracy** — how well the monitoring data represent actual conditions.
2. **Forecast accuracy** — how well NWM / Flood Hub predict future flow.
3. **Inundation accuracy** — how well the local model translates flow and WSE into flood depth and extent.

Historical-event validation is required before the local inundation output should be described as accurate.

Available/planned validation measures include:

- CSI / IoU,
- precision,
- recall,
- F1,
- flood-area bias,
- WSE/depth MAE and RMSE,
- hydrograph RMSE / MAE / bias,
- NSE,
- KGE,
- peak-flow error, and
- peak-timing error.

For engineering-grade flood mapping, a calibrated **HEC-RAS 2D** or equivalent hydraulic model should ultimately replace or validate the screening-level inundation component.

---

## Repository structure

```text
hgac-regional-flood-intelligence/
│
├── app.py                 # Flask dashboard and API
├── RUN_WINDOWS.ps1        # Windows launcher
├── requirements.txt       # Python dependencies
├── .env.example           # Environment-variable template
├── .gitignore
│
├── config/                # Regional and hydraulic configuration
├── src/                   # Core hydrology / terrain / API modules
├── scripts/               # Processing and validation utilities
├── static/                # Browser CSS and JavaScript
├── templates/             # Flask HTML templates
├── data/
│   └── validation/        # Small validation templates
└── gee/                   # Companion Earth Engine example(s)
```

Not stored in GitHub:

```text
.env
.venv/
data/raw/
data/processed/
data/regional_cache/
workspaces/
outputs/
large DEM / LiDAR files
```

---

## Quick start — Windows

### 1. Clone

```powershell
git clone https://github.com/djoshi1000/hgac-regional-flood-intelligence.git
cd hgac-regional-flood-intelligence
```

### 2. Create a Python environment

Python **3.11** is recommended for the current Windows workflow.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure optional Flood Hub access

```powershell
Copy-Item .env.example .env
notepad .env
```

Add your key locally:

```text
FLOODHUB_API_KEY=YOUR_API_KEY
FLOODHUB_GAUGE_ID=
```

**Never commit `.env` or an API key to GitHub.**

### 4. Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\RUN_WINDOWS.ps1
```

The current launcher starts the Flask application locally and opens the dashboard in a browser.

---

## First-run behavior

The first analysis of a new bayou can take longer because the application may need to:

- discover the control gauge and reach,
- download missing DEM tiles,
- condition terrain,
- derive routing/HAND,
- trace the selected channel,
- build cross sections and bank profiles, and
- create a reusable workspace.

Later runs reuse cached static products whenever possible.

---

## External data and services

The project may use data or APIs from:

- **U.S. Geological Survey (USGS)**
- **NOAA / National Water Model**
- **Google Flood Forecasting / Flood Hub API**
- public elevation and hydrography sources
- optional **Google Earth Engine** workflows

Coverage, update frequency, units, datum, and API availability should be verified for each location.

---

## Current limitations

The prototype does not yet fully represent all processes important to Houston-area flooding, including:

- storm-sewer hydraulics,
- bridge and culvert hydraulics,
- pumps and gates,
- reservoir and detention operations,
- detailed channel bathymetry,
- tidal/backwater effects,
- full 2-D momentum routing,
- spatially variable rainfall,
- complete uncertainty propagation.

Urban/pluvial, riverine, and coastal flooding should eventually be treated as interacting but distinct modeling components.

---

## Public deployment roadmap

The repository currently targets local development. A scalable public version should separate the web interface from long-running geospatial processing:

```text
Public dashboard
      │
      ▼
job queue / worker
      │
      ├── USGS / NWM / Flood Hub APIs
      ├── persistent DEM cache
      ├── precomputed bayou terrain models
      └── result / object storage
```

Major next steps include:

- historical-event validation,
- broader H-GAC terrain coverage,
- HCFCD/local gauge integration,
- precomputed regional bayou workspaces,
- improved rainfall-runoff modeling,
- HEC-RAS 2D coupling,
- infrastructure/exposure analysis,
- uncertainty and model-agreement reporting, and
- scalable public deployment.

---

## Disclaimer

This repository contains an **experimental research and decision-support prototype**.

It is intended for research, screening, visualization, and method development. It is **not an official flood-warning system** and should not be used as a substitute for information from emergency-management agencies, the National Weather Service, local flood-control authorities, FEMA products, or engineering-grade hydraulic studies.

Unless formally adopted and published by the relevant agency, this repository should not be interpreted as an official H-GAC operational product.
