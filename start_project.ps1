# Start UAV Simulation Project (Backend + Frontend)
$backendPath = Join-Path $PSScriptRoot "uav_project\backend"
$frontendPath = Join-Path $PSScriptRoot "uav_project\frontend"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting UAV Stability Simulation System" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backendPath'; python -m uvicorn api:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$frontendPath'; python -m http.server 5500"

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5500"

Write-Host "Backend running on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Frontend running on http://127.0.0.1:5500" -ForegroundColor Green

