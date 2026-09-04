# H-GAC Regional Flood Intelligence V5.2

V5.2 intentionally simplifies the user experience.

## User workflow

1. Open the app.
2. Click one bayou, river, or stream polyline.
3. Click **Run complete analysis**.
4. Wait while the Process Activity panel shows what is happening.
5. Review maps, curves, trends, rainfall, forecast and QA outputs.

The user no longer manually selects a gauge or runs numbered scripts.

## What happens automatically

- Resolves the clicked HCFCD/NHDPlus waterway.
- Finds a same-waterway downstream USGS discharge gauge and verifies watershed connectivity.
- Uses the selected waterway branch itself when tracing the DEM mainstem, rather than silently switching to the basin's largest tributary.
- Retrieves current USGS discharge/stage.
- Downloads the period-of-record USGS daily mean streamflow for long-term trend plots.
- Retrieves historical precipitation from the same USGS site when that parameter exists.
- Retrieves NWS quantitative precipitation forecast at the clicked waterway.
- Resolves NOAA National Water Model streamflow forecast.
- Uses the persistent DEM tile/VRT cache, crops the selected basin, and builds/reuses HAND, routing, cross sections and bank profile.
- Builds/uses the rating curve and NAVD88 water-surface framework.
- Runs the channel-excluded, volume-limited terrain flood screen.
- If `FLOODHUB_API_KEY` is present in `.env`, attempts to retrieve nearby Google Flood Hub forecast/status as a supplemental source. Flood Hub failure does not stop the local model.

## Flood Hub key

Place the key in the project root `.env`:

```text
FLOODHUB_API_KEY=YOUR_KEY
FLOODHUB_GAUGE_ID=
```

V5.2 attempts automatic nearby Flood Hub discovery; leave the gauge ID blank unless you have a specific reason to force one.

## DEM caching

The first run for a new area can be slow. Downloaded source tiles are retained in the regional cache. Static model products are also saved by waterway/gauge workspace. Future live refreshes reuse these products.

For broader H-GAC coverage, a regional VRT or ArcPy Mosaic Dataset can point to a larger cached DEM collection without physically merging it into one giant TIFF.

## Important limitation

This remains a screening/decision-support prototype, not HEC-RAS, FEMA, regulatory, or engineering floodplain modeling. The application deliberately stops if the selected waterway cannot be safely connected to a same-waterway control gauge in the current gauged workflow.
