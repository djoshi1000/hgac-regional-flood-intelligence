# Public deployment — Render beta

This deployment profile is for the H-GAC Regional Flood Intelligence research prototype.

## Important

The current Flask application uses shared in-memory application state. This deployment is suitable as a **public beta / demonstration** with limited simultaneous use, but it is not yet a fully isolated multi-user production architecture.

For a production public release, the next architecture should move each browser/user to an independent session/job state and place long-running model work in a queue/background worker.

## Why Render

The app is a dynamic Python/Flask application, so GitHub Pages cannot run it. Render provides a public web-service URL and can build directly from the GitHub repository.

The supplied `render.yaml` uses:

- Docker runtime
- 2 CPU / 4 GB RAM
- a 20 GB persistent disk at `/var/data`
- `/health` as the health-check endpoint
- `FLOODHUB_API_KEY` as a secret entered in the Render dashboard

If terrain jobs still hit memory pressure, change the Render plan from `2c-4g` to `2c-8g`.

## Persistent data

The cloud patch redirects these runtime products to `/var/data`:

- regional catalog cache
- DEM cache
- gauge discovery cache
- bayou workspaces
- generated outputs contained inside each workspace

This means Git deployments do not erase the DEM/workspace cache.

## Deploy

1. Run `PREPARE_PUBLIC_RENDER.ps1` from the local repository.
2. Confirm it commits and pushes the new deployment files to GitHub.
3. Sign in to Render.
4. Choose **New > Blueprint**.
5. Connect `djoshi1000/hgac-regional-flood-intelligence`.
6. Render detects `render.yaml`.
7. When prompted for `FLOODHUB_API_KEY`, paste the API key there.
8. Create/deploy the service.
9. Wait for the Docker build and health check.
10. Open the generated `https://...onrender.com` URL.

## Cost/resource caution

The geospatial terrain pipeline is not appropriate for Render's 512 MB free web service. The supplied Blueprint therefore uses a paid 2 CPU / 4 GB configuration and persistent disk.

For an inexpensive first test, deploy for a short period, verify the application, and then decide whether to keep the service running continuously.

## Public-safety notice

Keep the dashboard's experimental/research disclaimer visible. The local inundation product remains a terrain-screening model until historical validation/calibration is completed.
