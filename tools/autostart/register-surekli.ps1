# Registers the JARVIS 7/24 backbone to start at logon.
#
# This is the continuous-mode twin of register.ps1. Instead of the headless
# autonomous loop it starts the full interface-serving process:
#
#     tools\launcher\jarvis.ps1 -Surekli   ->   run.py --surekli
#
# which keeps running after every browser tab closes; the user talks to JARVIS
# by opening http://127.0.0.1:8797/ whenever they like.
#
# Before enabling this one, make sure the old "JARVIS Otonom" task stays
# DISABLED: two always-on processes would fight over the instance lock, and
# whichever loses serves nothing.
#
# The 0xC000013A mystery (see docs/CURRENT-STATE.md section 7) belongs to the
# old task chain; Task Scheduler Operational log is open now, so if it recurs
# it will finally leave a record. RestartOnFailure below is the seatbelt.
#
# Run once, from an ordinary (non-elevated) PowerShell:
#   powershell -ExecutionPolicy Bypass -File C:\JARVIS\tools\autostart\register-surekli.ps1
#
# ASCII only -- see jarvis.ps1 for why.

$ErrorActionPreference = 'Stop'

$taskName = 'JARVIS Surekli'
$root     = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$launcher = Join-Path $root 'tools\launcher\jarvis.ps1'

if (-not (Test-Path $launcher)) { throw "baslatici bulunamadi: $launcher" }

$other = Get-ScheduledTask -TaskName 'JARVIS Otonom' -ErrorAction SilentlyContinue
if ($other -and $other.State -ne 'Disabled') {
    Write-Host "UYARI: eski 'JARVIS Otonom' gorevi hala Enabled." -ForegroundColor Yellow
    Write-Host "Iki surec ayni kilidi paylasmak zorunda kalir. Kapatmak icin:" -ForegroundColor Yellow
    Write-Host "  Disable-ScheduledTask -TaskName 'JARVIS Otonom'" -ForegroundColor Yellow
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcher`" -Surekli" `
    -WorkingDirectory $root

# One minute of settling, same reasoning as register.ps1: a machine that just
# reached the desktop is busy, and the warm-up work JARVIS does at startup is
# better spent after the storm.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = 'PT1M'

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

# ExecutionTimeLimit 0 = no limit: this is meant to run for months.
# RestartOnFailure is the seatbelt for whatever killed the old daemon.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew `
    -Hidden

$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries     = $false
$settings.IdleSettings.StopOnIdleEnd = $false

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force `
    -Description 'JARVIS 7/24 omurgasi (run.py --surekli). Oturum acilisinda baslar, sekmeler kapansa da yasar.' | Out-Null

Write-Host "kayit tamam: $taskName"
Write-Host ""
Write-Host "simden denemek icin : Start-ScheduledTask -TaskName '$taskName'"
Write-Host "arayuz              : http://127.0.0.1:8797/"
Write-Host "kaldirmak icin      : Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
Write-Host "durdurmak icin      : arayuzdeki Kapat dugmesi (sureci dusurur)"
