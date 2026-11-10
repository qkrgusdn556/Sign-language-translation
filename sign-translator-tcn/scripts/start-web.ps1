$projectRoot = Split-Path -Parent $PSScriptRoot
$ErrorActionPreference = 'Stop'
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = Join-Path $projectRoot '../slrt-evaluation/.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup.ps1 first.' }
if (Get-NetTCPConnection -State Listen -LocalPort 5194 -ErrorAction SilentlyContinue) {
    Write-Output 'Port 5194 is already in use. Check http://127.0.0.1:5194/ before starting another server.'
    exit 0
}
$logs = Join-Path $projectRoot '.build'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
foreach ($name in @('GEMINI_API_KEY','GEMINI_MODEL','SLRT_PROJECT_PATH','SLRT_PYTHON')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if ($value) { [Environment]::SetEnvironmentVariable($name, $value, 'Process') }
}
Start-Process -FilePath $python -ArgumentList @('backend/web_server.py','--port','5194') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'server-launch.log') -RedirectStandardError (Join-Path $logs 'server-launch-error.log')
for ($i=0; $i -lt 20; $i++) {
    try { $response = Invoke-WebRequest 'http://127.0.0.1:5194/api/gemini/status' -UseBasicParsing -TimeoutSec 1; if ($response.StatusCode -eq 200) { Write-Output 'Open http://127.0.0.1:5194/'; exit 0 } } catch {}
    Start-Sleep -Milliseconds 500
}
throw 'Server did not become ready. Check .build/server-launch-error.log.'
