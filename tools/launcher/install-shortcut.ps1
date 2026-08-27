# Puts a JARVIS shortcut on the desktop. Run once.
#
# Creates nothing that starts by itself: this is a shortcut the user clicks, not
# a scheduled task. Removing it is deleting the file.
#
# ASCII only, for the reason written down in jarvis.ps1.

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$launcher = Join-Path $PSScriptRoot 'JARVIS.bat'

if (-not (Test-Path $launcher)) {
    throw "Baslatici bulunamadi: $launcher"
}

$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'JARVIS.lnk'

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $launcher
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'JARVIS - kisisel yapay zeka asistani'
# 7 = minimised. The console is how JARVIS is started, not what the user is
# meant to look at; the interface opens in the browser. Minimised rather than
# hidden so a failed startup still has somewhere to say so.
$shortcut.WindowStyle = 7

# Use the project icon when there is one; otherwise leave the default rather
# than pointing at a file that does not exist.
$icon = Join-Path $root 'ui\jarvis.ico'
if (Test-Path $icon) {
    $shortcut.IconLocation = $icon
}

$shortcut.Save()

Write-Host "Kisayol olusturuldu: $link" -ForegroundColor Green
Write-Host "JARVIS'i acmak icin masaustundeki JARVIS ikonuna cift tiklayin."
Write-Host ""
Write-Host "Not: JARVIS Windows acilisinda kendiliginden baslamaz." -ForegroundColor DarkGray
Write-Host "Kapatmak icin JARVIS penceresindeki Kapat dugmesi." -ForegroundColor DarkGray
