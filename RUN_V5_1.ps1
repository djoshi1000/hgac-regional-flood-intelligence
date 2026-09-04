$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Virtual environment not found: $Py" }

Write-Host "Starting H-GAC Regional Flood Intelligence V5.1..." -ForegroundColor Cyan
Write-Host "A browser window will open automatically. Keep this PowerShell window open." -ForegroundColor Yellow
& $Py app.py
