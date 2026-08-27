# Removes the autostart entry. Does not stop a loop that is already running.
$ErrorActionPreference = 'Stop'
$taskName = 'JARVIS Otonom'

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "kaldirildi: $taskName"
} else {
    Write-Host "zaten kayitli degil: $taskName"
}
