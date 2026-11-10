$projectRoot = Split-Path -Parent $PSScriptRoot
$ErrorActionPreference = 'Stop'
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = Join-Path $projectRoot '../slrt-evaluation/.venv/Scripts/python.exe' }
$existing = Get-NetTCPConnection -State Listen -LocalPort 5196 -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output 'Port 5196 is already in use. Check http://127.0.0.1:5196/'
    exit 0
}
if (-not (Test-Path -LiteralPath $python)) { throw 'Training Python environment was not found.' }
$logs = Join-Path $projectRoot 'data/yolo-studio'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Start-Process -FilePath $python -ArgumentList @('backend/yolo_studio.py', '--port', '5196') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'server.log') -RedirectStandardError (Join-Path $logs 'server-error.log')
Write-Output 'YOLO studio: http://127.0.0.1:5196/'
