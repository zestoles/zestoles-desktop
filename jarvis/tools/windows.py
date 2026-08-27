"""Windows desktop tools exposed through the same risk gate as every action.

These are deliberately small adapters over operating-system facilities.  No
tool interpolates model text into a shell command: application names are mapped
or resolved as executables, clipboard content travels on stdin, and trash paths
travel through an environment variable.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import shutil
import subprocess
from pathlib import Path

from . import LOW, MEDIUM, ToolError, ToolResult, Workspace, tool


APP_ALIASES = {
    "not defteri": "notepad.exe",
    "notepad": "notepad.exe",
    "hesap makinesi": "calc.exe",
    "hesap makinası": "calc.exe",
    "calculator": "calc.exe",
    "dosya gezgini": "explorer.exe",
    "explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "powershell": "powershell.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "visual studio code": "code.cmd",
    "vscode": "code.cmd",
    "spotify": "spotify.exe",
    "steam": "steam.exe",
}

PROTECTED_PROCESSES = frozenset({
    "csrss.exe", "dwm.exe", "explorer.exe", "lsass.exe", "services.exe",
    "smss.exe", "svchost.exe", "system", "system.exe", "wininit.exe", "winlogon.exe",
})


def resolve_application(name: str) -> str:
    """Resolve a friendly name without invoking a shell."""
    requested = " ".join(str(name or "").strip().casefold().split())
    if not requested:
        raise ToolError("uygulama adı boş")
    candidate = APP_ALIASES.get(requested, str(name).strip())
    if Path(candidate).is_absolute():
        if not Path(candidate).is_file():
            raise ToolError(f"uygulama bulunamadı: {candidate}")
        return candidate
    found = shutil.which(candidate)
    if found:
        return found
    # Windows App Paths and Start-menu registrations are understood by
    # startfile even when PATH is not.  Returning a plain executable name lets
    # app.open try that route without handing the string to cmd.exe.
    if os.name == "nt" and candidate.lower().endswith((".exe", ".cmd")):
        return candidate
    raise ToolError(f"uygulama bulunamadı: {name}")


@tool("system.time", risk=LOW, summary="Tarih, saat ve gün bilgisini verir")
def system_time(*, workspace: Workspace) -> ToolResult:
    now = dt.datetime.now().astimezone()
    days = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma",
            "Cumartesi", "Pazar")
    return ToolResult(True, output=f"{now:%d.%m.%Y %H:%M:%S} · {days[now.weekday()]} · {now.tzname()}")


@tool("system.processes", risk=LOW, summary="Çalışan uygulama ve süreçleri listeler")
def system_processes(*, workspace: Workspace, filter: str = "",
                     limit: int = 80) -> ToolResult:
    if os.name != "nt":
        raise ToolError("süreç listesi bu sürümde yalnızca Windows'ta")
    result = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=15, check=False)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "süreç listesi okunamadı")
    needle = str(filter or "").casefold()
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) < 5 or (needle and needle not in row[0].casefold()):
            continue
        rows.append(f"{row[0]} · PID {row[1]} · {row[4]}")
        if len(rows) >= max(1, min(int(limit), 200)):
            break
    return ToolResult(True, output="\n".join(rows) or "eşleşen süreç yok",
                      detail={"count": len(rows)})


@tool("app.open", risk=MEDIUM, summary="Windows'ta bir uygulama açar")
def app_open(*, workspace: Workspace, application: str,
             arguments: list[str] | None = None) -> ToolResult:
    executable = resolve_application(application)
    args = [str(value) for value in (arguments or [])]
    if any("\x00" in value for value in args):
        raise ToolError("uygulama argümanında NUL olamaz")
    try:
        process = subprocess.Popen([executable, *args], cwd=workspace.root,
                                   shell=False, close_fds=True)
    except OSError as exc:
        raise ToolError(f"uygulama açılamadı: {exc}") from exc
    return ToolResult(True, output=f"{application} açıldı",
                      detail={"pid": process.pid, "executable": executable})


@tool("app.close", risk=MEDIUM, summary="Bir uygulamayı normal kapatma isteğiyle sonlandırır")
def app_close(*, workspace: Workspace, process: str) -> ToolResult:
    name = Path(str(process or "").strip()).name.casefold()
    if not name:
        raise ToolError("süreç adı boş")
    if not name.endswith(".exe"):
        name += ".exe"
    if name in PROTECTED_PROCESSES:
        raise ToolError(f"korunan Windows süreci kapatılamaz: {name}")
    result = subprocess.run(["taskkill", "/IM", name, "/T"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=20, check=False)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or f"{name} kapatılamadı")
    return ToolResult(True, output=f"{name} kapatma isteği gönderildi")


@tool("clipboard.read", risk=LOW, summary="Windows panosundaki metni okur")
def clipboard_read(*, workspace: Workspace) -> ToolResult:
    if os.name != "nt":
        raise ToolError("pano bu sürümde yalnızca Windows'ta")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=15, check=False)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "pano okunamadı")
    return ToolResult(True, output=result.stdout[:100_000])


@tool("clipboard.write", risk=MEDIUM, summary="Windows panosuna metin yazar")
def clipboard_write(*, workspace: Workspace, text: str) -> ToolResult:
    if os.name != "nt":
        raise ToolError("pano bu sürümde yalnızca Windows'ta")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "$input | Set-Clipboard"], input=str(text), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=15, check=False)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "panoya yazılamadı")
    return ToolResult(True, output="metin panoya kopyalandı",
                      detail={"characters": len(str(text))})


@tool("fs.trash", risk=MEDIUM, summary="Dosya veya klasörü Geri Dönüşüm Kutusu'na taşır")
def fs_trash(*, workspace: Workspace, path: str) -> ToolResult:
    target = workspace.resolve(path, must_exist=True)
    if target == workspace.root:
        raise ToolError("çalışma alanının tamamı çöpe taşınamaz")
    script = (
        "Add-Type -AssemblyName Microsoft.VisualBasic; "
        "$p=$env:ZESTOLES_TRASH_TARGET; "
        "if ([IO.Directory]::Exists($p)) { "
        "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($p,'OnlyErrorDialogs','SendToRecycleBin') "
        "} elseif ([IO.File]::Exists($p)) { "
        "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p,'OnlyErrorDialogs','SendToRecycleBin') "
        "} else { exit 2 }"
    )
    env = {**os.environ, "ZESTOLES_TRASH_TARGET": str(target)}
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False, env=env)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "çöpe taşınamadı")
    return ToolResult(True, output=f"{workspace.show(target)} Geri Dönüşüm Kutusu'na taşındı",
                      detail={"path": str(target), "recoverable": True})


__all__ = ["APP_ALIASES", "PROTECTED_PROCESSES", "resolve_application"]
