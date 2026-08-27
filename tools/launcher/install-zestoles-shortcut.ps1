# Creates a manual ZESTOLES desktop shortcut. ASCII-only for PowerShell 5.1.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$launcher = Join-Path $PSScriptRoot 'zestoles-tray.ps1'
if (-not (Test-Path $launcher)) { throw "Baslatici bulunamadi: $launcher" }

$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'ZESTOLES.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = 'powershell.exe'
$shortcut.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $launcher + '"'
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'ZESTOLES - Turkce canli yapay zeka asistani (Ctrl+Alt+J)'
$shortcut.WindowStyle = 7
$icon = Join-Path $root 'ui\zestoles.ico'
if (Test-Path $icon) { $shortcut.IconLocation = $icon }
$shortcut.Save()

Write-Host "Kisayol olusturuldu: $link" -ForegroundColor Green
Write-Host "ZESTOLES kendiliginden baslamaz; kisayol veya Ctrl+Alt+J ile acilir."
