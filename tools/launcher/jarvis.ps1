# Starts JARVIS for a person, not for a scheduler.
#
# Manual start is the default: nothing here registers a task. With -Surekli the
# same launcher serves the 7/24 backbone instead -- run.py --surekli keeps
# running after the last browser tab closes.
#
# The project root is derived from this script's own location rather than being
# written down, so moving or renaming the project folder does not break the
# shortcut that points at it.
#
# ASCII only. Windows PowerShell 5.1 reads a BOM-less UTF-8 file as the ANSI
# codepage, and on a Turkish system the second byte of an em dash decodes to a
# closing quote, ends the string it sits in, and turns the rest of the file into
# parse errors. tools/autostart/stop-jarvis.ps1 failed exactly that way once.

param(
    [switch]$Surekli
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

if (-not (Test-Path (Join-Path $root 'run.py'))) {
    Write-Host "JARVIS bulunamadi: $root icinde run.py yok." -ForegroundColor Red
    Write-Host "Kisayol tasinmis olabilir. Yeniden olusturun:" -ForegroundColor Yellow
    Write-Host "  tools\launcher\install-shortcut.ps1" -ForegroundColor Yellow
    Read-Host "Kapatmak icin Enter"
    exit 1
}

# The Store alias at the front of PATH launches a packaged shim that re-launches
# the real interpreter, which puts two extra processes and an AppX container in
# the chain. Preferring a real python.exe keeps the chain short. JARVIS_PYTHON
# still wins when it is set, so this stays a default rather than a decision.
function Resolve-Python {
    if ($env:JARVIS_PYTHON -and (Test-Path $env:JARVIS_PYTHON)) {
        return $env:JARVIS_PYTHON
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.14-64\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    $found = Get-Command python.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -and (Get-Item $_.Source).Length -gt 0 } |
        Select-Object -First 1
    if ($found) { return $found.Source }
    return 'python'
}

$python = Resolve-Python

$host.UI.RawUI.WindowTitle = 'JARVIS'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

$mode = if ($Surekli) { '--surekli' } else { '--arayuz' }
& $python run.py $mode
$code = $LASTEXITCODE

if ($code -ne 0) {
    Write-Host ""
    Write-Host "JARVIS beklenmedik sekilde kapandi (cikis kodu $code)." -ForegroundColor Yellow
    Write-Host "Ayrinti icin: logs\jarvis.log" -ForegroundColor DarkGray
    Read-Host "Kapatmak icin Enter"
}

exit $code
