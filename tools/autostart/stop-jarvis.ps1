# Stops the autonomous loop, for real.
#
# Stop-ScheduledTask alone is not enough and this was measured: the action is a
# PowerShell launcher which runs cmd which runs the `python` app-execution alias
# which runs the actual interpreter. Task Scheduler ends what it started and the
# interpreter at the end of that chain keeps running, holding the port and the
# lock, invisible to anyone looking at the task state.
#
# So the lock file is the authority on which process is the loop.
#
# There is no graceful stop over the wire on purpose. The websocket has no
# authentication, and a shutdown command on an unauthenticated loopback port
# would be a kill switch for any local process. A task caught mid-run is left
# RUNNING and returned to the queue with its attempt counted on the next start;
# that recovery path exists precisely because this is how JARVIS gets stopped.
#
# Everything in this directory stays pure ASCII. Windows PowerShell 5.1 reads a
# UTF-8 file with no BOM as the ANSI codepage, and on a Turkish system byte 0x94
# of an em dash decodes to a closing quote, which ends a string early and turns
# the rest of the file into parse errors. This script failed exactly that way on
# its first run.

$ErrorActionPreference = 'Stop'

$taskName = 'JARVIS Otonom'
$root     = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$lockFile = Join-Path $root 'data\daemon.lock'

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Write-Host "gorev durduruldu: $taskName"
}

if (-not (Test-Path $lockFile)) {
    Write-Host "kilit dosyasi yok - calisan bir dongu gorunmuyor"
    exit 0
}

$daemonPid = (Get-Content $lockFile -Raw).Trim()
$process = Get-Process -Id $daemonPid -ErrorAction SilentlyContinue
if (-not $process) {
    Write-Host "kilitteki surec ($daemonPid) zaten yok; bayat kilit siliniyor"
    Remove-Item $lockFile -Force
    exit 0
}

Write-Host "dongu sonlandiriliyor: PID $daemonPid"
Stop-Process -Id $daemonPid -Force
Start-Sleep -Seconds 2

if (Get-Process -Id $daemonPid -ErrorAction SilentlyContinue) {
    Write-Warning "PID $daemonPid hala calisiyor"
    exit 1
}

Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
Write-Host "durdu. Yarim kalan bir gorev varsa sonraki baslangicta kuyruga geri konur."
