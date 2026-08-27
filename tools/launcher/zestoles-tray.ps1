# ZESTOLES Windows tray controller. ASCII-only for Windows PowerShell 5.1.
# No pystray, keyboard module, package install, or administrator rights needed.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$controlPath = Join-Path $root 'data\control.json'
$launcher = Join-Path $PSScriptRoot 'zestoles.ps1'
$iconPath = Join-Path $root 'ui\zestoles.ico'

function Get-Control {
    if (-not (Test-Path -LiteralPath $controlPath)) { return $null }
    try {
        $control = Get-Content -LiteralPath $controlPath -Raw | ConvertFrom-Json
        $process = Get-Process -Id ([int]$control.pid) -ErrorAction SilentlyContinue
        if (-not $process) { return $null }
        return $control
    }
    catch { return $null }
}

function Open-Zestoles {
    $control = Get-Control
    if ($control) {
        $edge = Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
        if (-not (Test-Path -LiteralPath $edge)) {
            $edge = Join-Path $env:ProgramFiles 'Microsoft\Edge\Application\msedge.exe'
        }
        if (Test-Path -LiteralPath $edge) {
            Start-Process -FilePath $edge -ArgumentList @("--app=$($control.url)", '--start-maximized')
        }
        else { Start-Process $control.url }
        return
    }
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
        '-File', ('"' + $launcher + '"')) -WorkingDirectory $root -WindowStyle Hidden
}

function Stop-Zestoles {
    $control = Get-Control
    if (-not $control) { return $false }
    try {
        $headers = @{ 'X-Jarvis-Token' = [string]$control.token }
        $body = '{"op":"kapat"}'
        Invoke-RestMethod -Uri ($control.url.TrimEnd('/') + '/istek') -Method Post `
            -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 10 | Out-Null
        return $true
    }
    catch { return $false }
}

# If the controller already exists, a second shortcut click simply opens or
# starts ZESTOLES and exits. The mutex is per user session, not machine-wide.
$created = $false
$mutex = New-Object Threading.Mutex($true, 'Local\ZESTOLES_TRAY_CONTROLLER', [ref]$created)
if (-not $created) {
    Open-Zestoles
    $mutex.Dispose()
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Windows.Forms;
using System.Runtime.InteropServices;
public sealed class ZestolesHotKeyFilter : IMessageFilter {
    public event EventHandler Pressed;
    public bool PreFilterMessage(ref Message message) {
        if (message.Msg == 0x0312 && message.WParam.ToInt32() == 9187) {
            if (Pressed != null) Pressed(this, EventArgs.Empty);
            return true;
        }
        return false;
    }
}
public static class ZestolesNative {
    [DllImport("user32.dll")] public static extern bool RegisterHotKey(IntPtr hWnd, int id, uint mods, uint key);
    [DllImport("user32.dll")] public static extern bool UnregisterHotKey(IntPtr hWnd, int id);
}
'@ -ReferencedAssemblies 'System.Windows.Forms.dll'

$menu = New-Object System.Windows.Forms.ContextMenuStrip
$openItem = $menu.Items.Add('Ac / One getir    Ctrl+Alt+J')
$stopItem = $menu.Items.Add('ZESTOLES kapat')
$menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator)) | Out-Null
$exitItem = $menu.Items.Add('Kontrol simgesinden cik')

$tray = New-Object System.Windows.Forms.NotifyIcon
$tray.Text = 'ZESTOLES Kontrol'
$tray.ContextMenuStrip = $menu
if (Test-Path -LiteralPath $iconPath) { $tray.Icon = New-Object System.Drawing.Icon($iconPath) }
$tray.Visible = $true

$openAction = { Open-Zestoles }
$openItem.add_Click($openAction)
$tray.add_DoubleClick($openAction)
$stopItem.add_Click({
    if (Stop-Zestoles) {
        $tray.ShowBalloonTip(1500, 'ZESTOLES', 'Guvenli kapatma istendi.', [System.Windows.Forms.ToolTipIcon]::Info)
    }
    else {
        $tray.ShowBalloonTip(1500, 'ZESTOLES', 'Calisan bir oturum bulunamadi.', [System.Windows.Forms.ToolTipIcon]::Warning)
    }
})

$context = New-Object System.Windows.Forms.ApplicationContext
$exitItem.add_Click({ $context.ExitThread() })

$filter = New-Object ZestolesHotKeyFilter
$filter.add_Pressed($openAction)
[System.Windows.Forms.Application]::AddMessageFilter($filter)
$hotKeyOk = [ZestolesNative]::RegisterHotKey([IntPtr]::Zero, 9187, 3, 0x4A)

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 2000
$timer.add_Tick({
    $running = $null -ne (Get-Control)
    $tray.Text = if ($running) { 'ZESTOLES - CALISIYOR' } else { 'ZESTOLES - KAPALI' }
    $stopItem.Enabled = $running
})
$timer.Start()

try {
    Open-Zestoles
    [System.Windows.Forms.Application]::Run($context)
}
finally {
    $timer.Stop()
    if ($hotKeyOk) { [ZestolesNative]::UnregisterHotKey([IntPtr]::Zero, 9187) | Out-Null }
    [System.Windows.Forms.Application]::RemoveMessageFilter($filter)
    $tray.Visible = $false
    $tray.Dispose()
    $menu.Dispose()
    $context.Dispose()
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
