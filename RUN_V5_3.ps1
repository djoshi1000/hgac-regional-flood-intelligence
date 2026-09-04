$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Virtual environment not found: $Py" }
Write-Host "Starting H-GAC Regional Flood Intelligence V5.3..." -ForegroundColor Cyan
Write-Host "Live: USGS + NWM + Google Flood Hub comparison + local LiDAR screen." -ForegroundColor Yellow
Write-Host "What-if: open Scenario Lab for rainfall depth/duration screening." -ForegroundColor Yellow
& $Py app.py
