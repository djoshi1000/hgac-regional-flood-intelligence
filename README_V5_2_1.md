# H-GAC Regional Flood Intelligence V5.2.1

Memory-safe hotfix for large basins such as Buffalo Bayou.

## What changed

- The app no longer attempts to route every large basin at 2 m.
- It estimates the working raster size before Stage 02 and automatically chooses
  2/3/4/5/6/8/10/12/15/20 m so the routing/HAND grid stays below ~12 million cells.
- Existing downloaded 2024 Houston LiDAR tiles are reused. A failed workspace's
  existing 2 m analysis crop can be stream-resampled locally; no new download is required.
- 1 m LiDAR remains the source for local bank/cross-section extraction after the
  selected branch is derived.
- If pyflwdir still reports memory pressure, the app automatically coarsens the
  working DEM and retries up to two times instead of ending with "Allocation failed".
- Terrain arrays use float32 where appropriate to reduce peak RAM.

## Install

Run INSTALL_V5_2_1.ps1 against your existing project, then restart with RUN_V5_2.ps1.
You do not need to delete the DEM tile cache.

## Important

For a large watershed, the full-basin HAND/routing screen may be 6-15 m rather
than 2 m. This is intentional for a regional screening app. The high-resolution
LiDAR tile cache is retained and local 1 m bank geometry is still used.
