"""What the machine is doing right now, and whether the user is at it.

Autonomous work is only acceptable when it is invisible. A background research
task that makes a game stutter is worse than no background task at all, so the
scheduler asks this module before every decision.

Everything here is ctypes against Win32 plus one nvidia-smi call — no psutil, no
GPUtil. The dependency would be small but this is a handful of documented calls
and the system is meant to keep working without a package index.

On anything other than Windows the readings come back unknown, and the policy
treats unknown as "assume busy" rather than "assume free".
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

log = logging.getLogger("jarvis.autonomy.resources")

IS_WINDOWS = os.name == "nt"


@dataclass(slots=True)
class Snapshot:
    taken: float
    idle_seconds: float | None
    cpu_percent: float | None
    ram_percent: float | None
    gpu_percent: float | None
    vram_used_mb: int | None
    vram_total_mb: int | None
    #: reading name → why it is missing. A measurement that failed without saying
    #: why cannot be diagnosed later, and "kaynak okunamadı" five times in a night
    #: with no reason attached is exactly the log line that teaches nothing.
    unknown: dict[str, str] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.idle_seconds is not None and self.cpu_percent is not None

    def why_unknown(self) -> str:
        """Short, stable explanation of what could not be read and why."""
        missing = {name: reason for name, reason in self.unknown.items()
                   if name in ("idle_seconds", "cpu_percent")}
        if not missing:
            return "sebep kaydedilmedi"
        return " · ".join(f"{name}: {reason}" for name, reason in sorted(missing.items()))

    def summary(self) -> str:
        parts = []
        if self.idle_seconds is not None:
            parts.append(f"boşta {int(self.idle_seconds)}s")
        if self.cpu_percent is not None:
            parts.append(f"CPU %{self.cpu_percent:.0f}")
        if self.ram_percent is not None:
            parts.append(f"RAM %{self.ram_percent:.0f}")
        if self.gpu_percent is not None:
            parts.append(f"GPU %{self.gpu_percent:.0f}")
        if self.vram_used_mb is not None and self.vram_total_mb:
            parts.append(f"VRAM {self.vram_used_mb}/{self.vram_total_mb}MB")
        return " · ".join(parts) or "ölçüm yok"


# --------------------------------------------------------------------- Win32
class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint), ("dwHighDateTime", ctypes.c_uint)]


def _filetime_to_int(ft: _FILETIME) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def idle_seconds() -> float | None:
    """Seconds since the last keyboard or mouse input, system-wide."""
    if not IS_WINDOWS:
        return None
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    # GetTickCount wraps every ~49.7 days; masking keeps the difference sane.
    elapsed = (ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF
    return elapsed / 1000.0


def ram_percent() -> float | None:
    if not IS_WINDOWS:
        return None
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return float(status.dwMemoryLoad)


#: Two callers share one meter — the scheduler tick and the telemetry pump. Inside
#: this window the last real measurement is reused instead of taking a new one.
CPU_MIN_INTERVAL_S = 0.75


class CpuMeter:
    """CPU load between two samples.

    GetSystemTimes reports cumulative totals, so a single reading says nothing;
    the meter keeps the previous one and reports the load over the interval.

    The minimum interval exists because of a measured failure. The telemetry pump
    polls every 3s and the scheduler every 5s through the *same* meter, so twice a
    minute one call lands milliseconds after the other. GetSystemTimes advances in
    15.6ms ticks, the delta comes back zero, and the reading is reported unknown —
    which the policy correctly treats as "assume busy". Five stances of "kaynak
    okunamadı" in one night came from that, not from a machine that could not be
    measured. Reusing the last real value inside the window is honest; it is a
    measurement taken slightly earlier, not a guess.
    """

    def __init__(self) -> None:
        self._previous: tuple[int, int, int] | None = None
        self._value: float | None = None
        self._value_at: float = 0.0
        self.last_reason: str = "ilk ölçüm — karşılaştırılacak önceki değer yok"

    def sample(self) -> float | None:
        if not IS_WINDOWS:
            self.last_reason = "Windows değil — GetSystemTimes yok"
            return None
        now = time.monotonic()
        if self._value is not None and now - self._value_at < CPU_MIN_INTERVAL_S:
            return self._value
        idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            self.last_reason = "GetSystemTimes başarısız döndü"
            return None
        current = (_filetime_to_int(idle), _filetime_to_int(kernel), _filetime_to_int(user))
        previous, self._previous = self._previous, current
        if previous is None:
            self.last_reason = "ilk ölçüm — karşılaştırılacak önceki değer yok"
            return None
        idle_delta = current[0] - previous[0]
        total_delta = (current[1] - previous[1]) + (current[2] - previous[2])
        if total_delta <= 0:
            self.last_reason = (f"ölçüm aralığı çok kısa — sayaç ilerlemedi "
                                f"(Δ={total_delta})")
            return None
        self._value = max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))
        self._value_at = now
        self.last_reason = ""
        return self._value


# ----------------------------------------------------------------------- GPU
_NVIDIA_SMI = shutil.which("nvidia-smi")


def gpu_reading() -> tuple[float | None, int | None, int | None, str]:
    """Utilisation, used and total VRAM, plus why they are missing when they are."""
    if not _NVIDIA_SMI:
        return None, None, None, "nvidia-smi bulunamadı"
    try:
        proc = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        log.debug("nvidia-smi zaman aşımı")
        return None, None, None, "nvidia-smi 10s içinde cevap vermedi"
    except OSError as exc:
        log.debug("nvidia-smi failed: %s", exc)
        return None, None, None, f"nvidia-smi çalıştırılamadı: {exc}"
    if proc.returncode != 0:
        return None, None, None, f"nvidia-smi çıkış kodu {proc.returncode}"
    if not proc.stdout.strip():
        return None, None, None, "nvidia-smi boş çıktı verdi"
    first = proc.stdout.strip().splitlines()[0]
    try:
        util, used, total = (part.strip() for part in first.split(","))
        return float(util), int(used), int(total), ""
    except ValueError:
        return None, None, None, f"nvidia-smi çıktısı çözümlenemedi: {first[:60]!r}"


class ResourceMonitor:
    def __init__(self) -> None:
        self._cpu = CpuMeter()
        self._cpu.sample()  # prime; the first reading is always None

    def snapshot(self) -> Snapshot:
        util, used, total, gpu_reason = gpu_reading()
        idle = idle_seconds()
        cpu = self._cpu.sample()
        ram = ram_percent()

        unknown: dict[str, str] = {}
        if idle is None:
            unknown["idle_seconds"] = ("Windows değil — GetLastInputInfo yok" if not IS_WINDOWS
                                       else "GetLastInputInfo başarısız döndü")
        if cpu is None:
            unknown["cpu_percent"] = self._cpu.last_reason or "sebep kaydedilmedi"
        if ram is None:
            unknown["ram_percent"] = ("Windows değil — GlobalMemoryStatusEx yok"
                                      if not IS_WINDOWS
                                      else "GlobalMemoryStatusEx başarısız döndü")
        if util is None:
            unknown["gpu_percent"] = gpu_reason or "sebep kaydedilmedi"

        return Snapshot(
            taken=time.time(),
            idle_seconds=idle,
            cpu_percent=cpu,
            ram_percent=ram,
            gpu_percent=util,
            vram_used_mb=used,
            vram_total_mb=total,
            unknown=unknown,
        )
