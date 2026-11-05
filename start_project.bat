@echo off
echo ========================================================
echo Starting UAV Stability Simulation System
echo ========================================================
echo.

cd /d "%~dp0uav_project\backend"
start "UAV Backend API (Port 8000)" cmd /k "python -m uvicorn api:app --reload --port 8000"

cd /d "%~dp0uav_project\frontend"
start "UAV Frontend Server (Port 5500)" cmd /k "python -m http.server 5500"

timeout /t 3 >nul
start http://127.0.0.1:5500

echo Backend running on: http://127.0.0.1:8000 (API Docs: http://127.0.0.1:8000/docs)
echo Frontend running on: http://127.0.0.1:5500
echo.
echo You can keep the backend and frontend terminal windows open.
echo To stop the servers, close their command windows.

