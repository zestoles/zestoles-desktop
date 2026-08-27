# Registers the JARVIS autonomous loop to start at logon.
#
# Why logon and not "run whether the user is logged on or not": the policy layer
# decides whether autonomous work is polite by asking GetLastInputInfo whether
# anyone is at the keyboard. In session 0 there is no keyboard to ask about, the
# reading comes back unknown, and the policy correctly refuses to do anything.
# An always-on service would be a loop that never runs.
#
# Run this once, from an ordinary (non-elevated) PowerShell:
#   powershell -ExecutionPolicy Bypass -File C:\JARVIS\tools\autostart\register.ps1

$ErrorActionPreference = 'Stop'

$taskName = 'JARVIS Otonom'
$root     = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$launcher = Join-Path $PSScriptRoot 'start-jarvis.ps1'

if (-not (Test-Path $launcher)) { throw "baslatici bulunamadi: $launcher" }

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcher`"" `
    -WorkingDirectory $root

# One minute of settling. A machine that just reached the desktop is busy with
# its own startup, and the first thing JARVIS would do is measure it and decide
# the machine is busy anyway.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = 'PT1M'

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

# ExecutionTimeLimit 0 = no limit: this is meant to run for months.
# MultipleInstances IgnoreNew is a second belt next to the daemon's own lock file.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew `
    -Hidden

# Idle and battery must not stop it; the policy layer already decides when it is
# polite to work, and it is better at it than Task Scheduler is.
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries     = $false
$settings.IdleSettings.StopOnIdleEnd = $false

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force `
    -Description 'JARVIS otonom dongusu (run.py --otonom --yayin). Oturum acilisinda baslar.' | Out-Null

Write-Host "kayit tamam: $taskName"
Get-ScheduledTask -TaskName $taskName |
    Select-Object TaskName, State, @{n = 'Trigger'; e = { $_.Triggers[0].CimClass.CimClassName } } |
    Format-List
Write-Host "elle calistirmak icin : Start-ScheduledTask -TaskName '$taskName'"
Write-Host "durdurmak icin        : tools\autostart\stop-jarvis.ps1  (Stop-ScheduledTask yetmez)"
Write-Host "kaldirmak icin        : tools\autostart\unregister.ps1"
