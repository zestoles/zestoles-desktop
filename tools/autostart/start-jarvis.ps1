# Starts the autonomous loop and stays attached to it.
#
# Foreground on purpose: Task Scheduler treats "the action is still running" as
# "the task is running", which is what makes its restart-on-failure setting mean
# anything. A launcher that fires the process and exits would report success one
# second after a crash loop began.
#
# Output goes to logs\daemon.out.log. Events are already in SQLite, so this file
# is a convenience rather than the record; it is rotated at 5 MB so an unattended
# machine cannot fill a disk with it.

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$logDir = Join-Path $root 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$log = Join-Path $logDir 'daemon.out.log'
if ((Test-Path $log) -and ((Get-Item $log).Length -gt 5MB)) {
    Move-Item $log (Join-Path $logDir 'daemon.out.1.log') -Force
}

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

$python = if ($env:JARVIS_PYTHON) { $env:JARVIS_PYTHON } else { 'python' }

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') otonom baslatiliyor ($python) ===" |
    Out-File -FilePath $log -Append -Encoding ascii

# The redirect goes through cmd rather than PowerShell's `*>>`. Windows
# PowerShell 5.1 writes redirected output as UTF-16, which turned the first real
# run of this launcher into a log file no tool could read; cmd appends bytes as
# the process produced them, and PYTHONIOENCODING above makes those bytes UTF-8.
& cmd.exe /c "`"$python`" run.py --otonom --yayin >> `"$log`" 2>&1"
exit $LASTEXITCODE
