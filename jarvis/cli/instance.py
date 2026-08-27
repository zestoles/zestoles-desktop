"""One autonomous loop per machine.

Autostart makes a second copy easy to create by accident: the scheduled task
starts JARVIS at logon, then the user opens a terminal and starts it again. Two
schedulers against one SQLite queue do not corrupt anything — `claim()` takes a
task under BEGIN IMMEDIATE, so no task runs twice — but two improvement loops
would spend the same nightly budget twice over and two websocket servers would
fight for one port. Neither failure announces itself.

The lock is a file holding a pid and the creation time of that process. It is
not a kernel mutex and does not try to be: the question it answers is "is the
process that wrote this file still running", which is enough for the accident it
prevents. A stale file left by a process the OS killed is detected and taken
over rather than treated as a running instance — after a Windows Update restart
the file is always stale, and refusing to start then would defeat the autostart
it exists to protect.

The creation time is there because a pid alone is not an identity. Windows
reuses pids, and this system recorded it happening: the loop that died at 17:15
held 14408, and by 23:53 an unrelated process had inherited that number. Had
autostart run in between, the lock would have seen a live 14408 and refused —
JARVIS would have stayed down for the same reason it exists to prevent, and
nothing would have said so.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

log = logging.getLogger("jarvis.cli.instance")

_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 0x00000102
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint), ("dwHighDateTime", ctypes.c_uint)]


def process_started_at(pid: int) -> int | None:
    """When the OS says this process began, or None when that cannot be read.

    A pid on its own is not an identity. Windows reuses them, and this system
    watched it happen: 14408 was the autonomous loop at 17:15 and belonged to
    something unrelated by 23:53. A lock that remembers only the number can
    refuse to start because a stranger inherited it — the failure mode being
    that autostart silently never comes up.

    The value is the raw creation FILETIME. It is never compared across
    machines or reboots, only against the process the lock file names, so its
    only requirement is that two different processes do not share one.
    """
    if pid <= 0 or os.name != "nt":
        return None
    handle = ctypes.windll.kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None
    try:
        creation, exited, kernel, user = (_FILETIME() for _ in range(4))
        ok = ctypes.windll.kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exited),
            ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            return None
        return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def process_alive(pid: int) -> bool:
    """Is a process with this id running?

    Deliberately not `os.kill(pid, 0)`: on Windows that maps to TerminateProcess
    and would kill the very instance it is asking about.
    """
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    handle = ctypes.windll.kernel32.OpenProcess(_SYNCHRONIZE, False, int(pid))
    if not handle:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


class InstanceLock:
    """Refuses to hand out the lock while the recorded process is alive."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.held = False
        self.holder: int | None = None

    def acquire(self) -> bool:
        holder, started = self._read()
        self.holder = holder
        if holder is not None and holder != os.getpid() \
                and self._still_running(holder, started):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.write_text(self._stamp(), encoding="ascii")
        except OSError as exc:
            # A lock that cannot be written must not stop the daemon; the risk it
            # guards against is smaller than not running at all.
            log.warning("kilit dosyası yazılamadı (%s): %s", self.path, exc)
            return True
        self.held = True
        self.holder = os.getpid()
        return True

    def release(self) -> None:
        if not self.held:
            return
        try:
            if self._read()[0] == os.getpid():
                self.path.unlink(missing_ok=True)
        except OSError as exc:
            log.debug("kilit dosyası silinemedi: %s", exc)
        self.held = False

    # ------------------------------------------------------------- internals
    def _still_running(self, pid: int, started: int | None) -> bool:
        """Is the process that wrote this lock the one holding that pid now?

        Fails closed on purpose. When the two cannot be told apart — an older
        lock file with no timestamp, or a process whose creation time cannot be
        read — the answer is "assume it is still the holder", because starting a
        second loop is worse than not starting one.
        """
        if not process_alive(pid):
            return False
        if started is None:
            return True
        live = process_started_at(pid)
        if live is None:
            return True
        return live == started

    def _stamp(self) -> str:
        return f"{os.getpid()} {process_started_at(os.getpid()) or 0}"

    def _read(self) -> tuple[int | None, int | None]:
        """The pid in the lock file and, when present, its creation time.

        Tolerates the older single-number format so an upgrade over a lock left
        behind by the previous build reads as "pid only" rather than as garbage.
        """
        try:
            parts = self.path.read_text(encoding="ascii").split()
            pid = int(parts[0])
        except (OSError, ValueError, IndexError):
            return None, None
        started: int | None = None
        if len(parts) > 1:
            try:
                started = int(parts[1]) or None
            except ValueError:
                started = None
        return pid, started

    def __enter__(self) -> InstanceLock:
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
