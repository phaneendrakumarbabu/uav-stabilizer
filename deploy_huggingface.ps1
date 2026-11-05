# Hugging Face Space Deployment Script
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Deploying UAV Simulation to Hugging Face Spaces" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$hfUser = Read-Host "Enter your Hugging Face username"
$spaceName = Read-Host "Enter your Space name (Press Enter for 'uav-simulation')"
if ([string]::IsNullOrWhiteSpace($spaceName)) {
    $spaceName = "uav-simulation"
}

$repoId = "$hfUser/$spaceName"
Write-Host "`nTarget Space: $repoId" -ForegroundColor Green
Write-Host "Uploading project files to Hugging Face Spaces..." -ForegroundColor Yellow

& python -m huggingface_hub.cli.hf upload $repoId . . --repo-type=space --exclude=".venv/*" "results/*" "diagnostic_logs/*" "report_builder/node_modules/*"

Write-Host "`nDone! View your live Space at:" -ForegroundColor Cyan
Write-Host "https://huggingface.co/spaces/$repoId" -ForegroundColor Green

