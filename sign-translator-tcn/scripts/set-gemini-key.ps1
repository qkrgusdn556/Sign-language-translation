$projectRoot = Split-Path -Parent $PSScriptRoot
$ErrorActionPreference = 'Stop'
$secret = Read-Host 'Gemini API key (stored in your Windows user environment, not encrypted)' -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    if ([string]::IsNullOrWhiteSpace($key) -or $key.Length -lt 10 -or $key -match '\s') { throw 'Invalid key.' }
    [Environment]::SetEnvironmentVariable('GEMINI_API_KEY', $key, 'User')
    Write-Output 'Key saved locally. Restart the web server to apply it. Never upload the key.'
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    $key = $null
    $secret.Dispose()
}
