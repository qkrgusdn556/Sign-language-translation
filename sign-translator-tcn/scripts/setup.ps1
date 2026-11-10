$projectRoot = Split-Path -Parent $PSScriptRoot
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit) with the Windows Python launcher first.' }
}
& $python -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check Internet access and retry.' }
& $python -c "import torch, torchvision, ultralytics, onnxruntime, cv2; print('Runtime ready:', ultralytics.__version__, 'CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed.' }
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot '.build') | Out-Null
Write-Output 'Setup complete. Run run_tcn.bat. Restore data before using trained models.'
