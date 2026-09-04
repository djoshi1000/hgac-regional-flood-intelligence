$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Virtual environment not found: $Py" }
Write-Host "Starting H-GAC Regional Flood Intelligence V5.2..." -ForegroundColor Cyan
Write-Host "Click one waterway, then click Run complete analysis." -ForegroundColor Yellow
& $Py app.py
