# Starts ZESTOLES in manual live-assistant mode. ASCII-only for Windows
# PowerShell 5.1 compatibility on Turkish systems.

param([switch]$Surekli)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

if (-not (Test-Path (Join-Path $root 'run.py'))) {
    Write-Host "ZESTOLES bulunamadi: $root icinde run.py yok." -ForegroundColor Red
    Read-Host "Kapatmak icin Enter"
    exit 1
}

function Resolve-ZestolesPython {
    if ($env:ZESTOLES_PYTHON -and (Test-Path $env:ZESTOLES_PYTHON)) {
        return $env:ZESTOLES_PYTHON
    }
    $projectPython = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path $projectPython) { return $projectPython }
    $legacyPython = $env:JARVIS_PYTHON
    if ($legacyPython -and (Test-Path $legacyPython)) { return $legacyPython }
    $found = Get-Command python.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -and (Get-Item $_.Source).Length -gt 0 } |
        Select-Object -First 1
    if ($found) { return $found.Source }
    return 'python'
}

$python = Resolve-ZestolesPython
$host.UI.RawUI.WindowTitle = 'ZESTOLES'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

$mode = if ($Surekli) { '--surekli' } else { '--arayuz' }
& $python run.py $mode
$code = $LASTEXITCODE

if ($code -ne 0) {
    Write-Host ""
    Write-Host "ZESTOLES beklenmedik sekilde kapandi (cikis kodu $code)." -ForegroundColor Yellow
    Write-Host "Ayrinti icin: logs\jarvis.log" -ForegroundColor DarkGray
    Read-Host "Kapatmak icin Enter"
}

exit $code
