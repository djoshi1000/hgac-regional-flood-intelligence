# H-GAC Regional Flood Intelligence V5.3

V5.3 clarifies the roles of USGS, NOAA NWM, Google Flood Hub and the local LiDAR terrain screen, and adds a **Rainfall Scenario Lab**.

## What each source does

- **USGS:** observed river discharge/stage and historical record.
- **NOAA NWM:** operational U.S. streamflow forecast used by the local screening model.
- **Google Flood Hub:** independent AI forecast/status comparator. When the API returns inundation polygons, V5.3 downloads and overlays them as a separate Google layer. It is not silently treated as ground truth.
- **Local LiDAR/HAND model:** translates flow/WSE to a high-resolution local riverine terrain-screening extent/depth.

## Rainfall Scenario Lab

After a bayou static model has been prepared, open **Scenario Lab** and enter, for example:

- Storm total: `11 in`
- Duration: `5 h`
- Antecedent wetness: Dry / Normal / Wet

V5.3 uses an NRCS Curve Number runoff transform and SCS triangular hydrograph to convert the hypothetical rainfall to a discharge hydrograph, then runs that hydrograph through the cached local flood-screening engine.

**Important:** the public Google Flood Forecasting API returns Google's forecasts/status products. It does not accept a custom `11 inches in 5 hours` forcing. The hypothetical scenario therefore does not pretend to be a Flood Hub scenario. Flood Hub stays visible as an independent current AI forecast/reference.

For extreme scenarios, if the generated peak flow exceeds the available USGS/local rating-curve range, the dashboard explicitly warns that the WSE is clamped and the result is highly uncertain.

## Accuracy

The local inundation result is still a screening approximation, not a calibrated 2-D hydraulic model. Do not label it "accurate" until historical events have been replayed and evaluated against observed extents/high-water marks. For engineering-grade extreme-storm mapping, use a calibrated HEC-RAS 2D / equivalent dynamic hydraulic model.

## Install

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

.\INSTALL_V5_3.ps1 `
  -ProjectRoot "C:\Users\Durga\Downloads\HGAC_Hunting_Bayou_Flood_Prototype\hgac_hunting_bayou_flood"
```

Then:

```powershell
cd "C:\Users\Durga\Downloads\HGAC_Hunting_Bayou_Flood_Prototype\hgac_hunting_bayou_flood"
.\.venv\Scripts\Activate.ps1
pip install -r requirements_v5_3.txt
.\RUN_V5_3.ps1
```

Your `.env`, `config.yaml`, DEM cache, workspaces and existing processed data are preserved.
