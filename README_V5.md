# H-GAC Regional Flood Intelligence V5

V5 converts the Hunting Bayou prototype into a regional, workspace-based Streamlit application.

## What changes in V5

1. The opening screen is the H-GAC regional map.
2. Click on/near a bayou or named stream.
3. The app resolves the waterway using HCFCD channels where available and NHDPlus HR elsewhere.
4. Nearby USGS streamgages are tested for **hydraulic relevance**: the selected click should fall inside the gauge's NLDI upstream basin and recent flow should exist.
5. Each selected bayou/gauge gets a persistent `workspaces/<bayou>__usgs_<site>` folder.
6. The first run prepares the static products (basin, DEM, terrain/HAND, mainstem, rating/datum QA, LiDAR bank profile).
7. Later refreshes reuse all static products and only fetch USGS/NWM plus regenerate the volume-limited inundation screen.
8. Optional automatic refresh provides a near-real-time dashboard.

## DEM caching strategy

Do **not** physically merge a single 1-m raster for the full 13-county H-GAC region unless you have a very specific storage/serving reason. A physical merge duplicates enormous data and is slow to rebuild.

V5 supports the faster pattern:

- permanent tile cache: `data/regional_cache/dem/tiles`
- optional GDAL VRT: one tiny virtual mosaic referencing those tiles
- optional ArcGIS Pro Mosaic Dataset
- 2-m crop for the full selected gauge basin
- 1-m crop only along a corridor around the derived mainstem for bank cross sections
- persistent bayou workspaces so a previously prepared bayou does not rerun static processing

The built-in downloader currently knows the **2024 USGS Houston B24** project. This is not complete H-GAC coverage. For bayous outside that project, point `config/v5_regional.yaml -> dem_cache.regional_dem_path` to a regional 3DEP/agency GeoTIFF, COG, or VRT, or add another DEM provider.

### VRT

After tiles are cached:

```powershell
python scripts\21_build_dem_cache.py --build-vrt
```

Then set:

```yaml
dem_cache:
  regional_dem_path: "data/regional_cache/dem/hgac_dem_cache.vrt"
```

### ArcPy mosaic (optional)

Run from a Python environment that can import ArcPy:

```powershell
python scripts\21_build_dem_cache.py --build-arcpy-mosaic
```

The rasterio flood pipeline directly consumes GeoTIFF/COG/VRT. The ArcPy Mosaic Dataset is useful for organizational ArcGIS management/visualization and can also be exported/served as a regional raster source.

## Install

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\INSTALL_V5.ps1 -ProjectRoot "C:\path\to\hgac_hunting_bayou_flood"
cd "C:\path\to\hgac_hunting_bayou_flood"
.\.venv\Scripts\Activate.ps1
pip install -r requirements_v5.txt
python scripts\20_build_regional_catalog.py
streamlit run app.py
```

## Important scientific safeguards

- The nearest gauge is not automatically assumed to be the correct control gauge.
- The selected bayou click must also match the DEM-derived dominant mainstem for the chosen gauge; otherwise V5 stops rather than model a different branch.
- A new USGS rating cannot be combined with LiDAR until the gage-height datum is tied to NAVD88 and passes consistency QA.
- LiDAR channel-surface elevation is not bathymetric bed elevation.
- The preferred inundation is a terrain/hydraulic screening product with channel exclusion and a flood-volume budget. It is not HEC-RAS, FEMA, engineering, emergency-warning, or regulatory mapping.
- Flood Hub remains inactive until intentionally configured.

## Validation

Historical replay and validation from V4.3 remain available:

For a V5 bayou workspace, point the scripts at that workspace config first:

```powershell
$env:HGAC_CONFIG_PATH = "workspaces\<bayou>__usgs_<site>\config\config.yaml"
python scripts\06_make_inundation.py --hydrograph-csv data\validation\event_flow.csv --hydrograph-unit cfs --scenario-label "Historical Event"
python scripts\11_validate_historical_event.py --event-name event --reference-extent data\validation\observed_extent.geojson
python scripts\12_validate_hydrograph.py --event-name event --modeled-csv data\validation\nwm_event.csv --observed-csv data\validation\usgs_event.csv
```
