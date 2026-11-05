@echo off
echo ========================================================
echo Deploying UAV Simulation to Hugging Face Spaces
echo ========================================================
echo.
set /p HF_USERNAME="Enter your Hugging Face username: "
set /p SPACE_NAME="Enter your Space name (default: uav-simulation): "
if "%SPACE_NAME%"=="" set SPACE_NAME=uav-simulation

echo.
echo Uploading project files to Hugging Face Space: %HF_USERNAME%/%SPACE_NAME% ...
echo.

python -m huggingface_hub.cli.hf upload "%HF_USERNAME%/%SPACE_NAME%" . . --repo-type=space --exclude=".venv/*" "results/*" "diagnostic_logs/*" "report_builder/node_modules/*"

echo.
echo Done! Your Space will build and be live at:
echo https://huggingface.co/spaces/%HF_USERNAME%/%SPACE_NAME%
echo.
pause

