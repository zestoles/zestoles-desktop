"""A confined workspace, and the only place JARVIS may write files or run commands.

Until now `shell` and `fs.write` were refused to every agent unconditionally,
because there was nowhere to contain them. This module is that place, and the ban
lifts only for code holding a Sandbox — never by an agent's own declaration.

## What "confined" means here, precisely

Every path is resolved and then checked to be inside the sandbox root. Resolution
happens first because the interesting attacks are all about making a path look
local while pointing elsewhere:

  traversal        ../../../Windows/System32
  absolute         C:\\Windows\\System32
  drive-relative   C:config.sys — no separator, still leaves the sandbox
  UNC              \\\\server\\share and \\\\?\\C:\\
  symlink          a link inside the sandbox pointing out of it
  junction         the Windows directory equivalent, same effect
  hard link        the one path resolution cannot see — see below
  device names     CON, NUL, COM1 — these bypass the filesystem entirely
  ADS              notes.txt:hidden — a second stream on an allowed file
  NUL byte         path\\x00.txt — truncates in some APIs, not in others

Each has a test in tests/test_sandbox.py. A rule with no test is a rule that will
be quietly broken by the next refactor.

## Hard links, and why path checks alone were not enough

Resolution defeats symlinks and junctions because both are links that resolution
follows. A hard link is not a link — it is a second name for the same file, with
no pointer to follow, so a hard link inside the sandbox naming a file outside it
resolves to a path that is genuinely inside. Every path rule above says yes, and
writing through it edits the outside file.

It was verified here, not theorised: a canary written outside the sandbox was
overwritten through a hard link created inside it, and `resolve()` reported the
path as contained. Unlike symlinks, creating one needs no elevation at all, which
makes it the more available attack of the two.

The defence cannot be a path rule, so it is a file rule: a regular file in a fresh
sandbox has exactly one name, and `st_nlink > 1` means it has another one
somewhere. Such files are refused for both reading and writing. Legitimate hard
links inside a sandbox are refused too — a workspace has no need of them, and that
is a cheap price for closing this.

## Honest limits

Resolve-then-open has a time-of-check-to-time-of-use window: a symlink created
between the check and the open would not be caught. Closing it properly needs
O_NOFOLLOW, which Windows does not offer. The window is microseconds and the
attacker would have to already be running code on the machine, at which point the
sandbox is not the weakest thing there — but it is a real gap and it is not hidden.

This is a workspace boundary, not a security boundary against hostile native code.
A command allowed to run can do whatever that program can do; confinement is of
paths and of which programs may start, not of what a running program may then ask
the kernel for. Real isolation needs a container or a VM, and this is deliberately
neither.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PureWindowsPath

log = logging.getLogger("jarvis.lab.sandbox")

#: Windows device names. Opening one of these succeeds and talks to a device
#: rather than a file, wherever the sandbox happens to be.
_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})

#: Environment variables never passed to a sandboxed command. A lab process has no
#: business holding the keys to anything.
_SECRET_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|SESSION|COOKIE|AUTH)",
    re.IGNORECASE)

DEFAULT_COMMANDS = ("python", "python3", "py", "git", "pip")
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_OUTPUT = 200_000
DEFAULT_MAX_FILE_BYTES = 5_000_000
DEFAULT_MAX_FILES = 2_000


class SandboxViolation(RuntimeError):
    """An attempt to act outside the sandbox. Always a bug or an attack."""


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def summary(self) -> str:
        state = "zaman aşımı" if self.timed_out else f"çıkış {self.returncode}"
        return f"{' '.join(self.args[:3])} → {state} ({self.duration_ms} ms)"


@dataclass(slots=True)
class SandboxLimits:
    timeout_s: int = DEFAULT_TIMEOUT_S
    max_output: int = DEFAULT_MAX_OUTPUT
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_files: int = DEFAULT_MAX_FILES
    allowed_commands: tuple[str, ...] = DEFAULT_COMMANDS
    allow_network: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)


class Sandbox:
    def __init__(self, root: Path, *, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        # The resolved root is the comparison basis. Comparing against the
        # unresolved one would let a symlinked sandbox root defeat every check.
        self.root = root.resolve()
        if self._is_dangerous_root(self.root):
            raise SandboxViolation(f"sandbox kökü olamaz: {self.root}")
        self._allowed = frozenset(c.casefold() for c in self.limits.allowed_commands)

    # ----------------------------------------------------------------- paths
    @staticmethod
    def _is_dangerous_root(root: Path) -> bool:
        """Refuse a root that would make confinement meaningless."""
        if root.parent == root:  # a drive root such as C:\
            return True
        system = os.environ.get("SystemRoot", "C:\\Windows")
        profile = os.environ.get("USERPROFILE", "")
        for forbidden in (system, profile, os.environ.get("ProgramFiles", "")):
            if forbidden and Path(forbidden).resolve() == root:
                return True
        return False

    def _reject(self, relative: str, reason: str) -> None:
        log.warning("sandbox ihlali reddedildi (%s): %s", reason, relative)
        raise SandboxViolation(f"{reason}: {relative!r}")

    def resolve(self, relative: str) -> Path:
        """Turn a sandbox-relative path into a real one, or refuse.

        Everything that is not plainly inside the sandbox is refused rather than
        sanitised. Silently rewriting a hostile path leaves the caller believing
        it did what was asked.
        """
        if not isinstance(relative, str) or not relative.strip():
            self._reject(str(relative), "boş yol")
        if "\x00" in relative:
            self._reject(relative, "yolda NUL baytı")

        # Parse with Windows semantics regardless of host so the rules are the same
        # everywhere and a POSIX test run still exercises them.
        pure = PureWindowsPath(relative)
        if pure.drive:
            self._reject(relative, "sürücü belirten yol")
        if pure.root:
            self._reject(relative, "mutlak yol")
        if relative.startswith(("\\\\", "//")):
            self._reject(relative, "UNC yolu")

        for part in pure.parts:
            if part == "..":
                self._reject(relative, "üst dizine çıkış")
            if ":" in part:
                self._reject(relative, "alternatif veri akışı")
            if part.split(".")[0].upper() in _RESERVED_NAMES:
                self._reject(relative, "ayrılmış aygıt adı")

        target = self.root / PurePath(*pure.parts)
        # resolve() follows symlinks and junctions, so a link pointing out of the
        # sandbox lands outside the root here and is caught by the check below.
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as exc:
            self._reject(relative, f"yol çözümlenemedi ({exc})")

        if not self._within(resolved):
            self._reject(relative, "sandbox dışına çıkıyor")
        return resolved

    def _within(self, path: Path) -> bool:
        try:
            return path == self.root or path.is_relative_to(self.root)
        except (ValueError, OSError):
            return False

    def _reject_hard_link(self, path: Path, relative: str) -> None:
        """Refuse a file that has more than one name.

        The second name may be anywhere on the volume, including outside the
        sandbox, and no amount of path checking can see it. One name is what a
        file created inside a sandbox has.
        """
        try:
            links = path.stat().st_nlink
        except OSError:
            return
        if links > 1:
            self._reject(relative, f"sabit bağ ({links} isim) — dosya sandbox dışında da olabilir")

    # ------------------------------------------------------------------ files
    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise SandboxViolation(f"dosya yok: {relative}")
        self._reject_hard_link(path, relative)
        if path.stat().st_size > self.limits.max_file_bytes:
            raise SandboxViolation(f"dosya çok büyük: {relative}")
        return path.read_text(encoding="utf-8", errors="replace")

    def write(self, relative: str, content: str) -> Path:
        data = content.encode("utf-8")
        if len(data) > self.limits.max_file_bytes:
            raise SandboxViolation(
                f"içerik sınırı aşıldı ({len(data)} > {self.limits.max_file_bytes})")
        if self.file_count() >= self.limits.max_files:
            raise SandboxViolation(f"dosya sayısı sınırı aşıldı ({self.limits.max_files})")

        path = self.resolve(relative)
        # Refuse to write through a link even though resolve() already proved the
        # destination is inside: a link inside the sandbox pointing at another file
        # inside it is still not the file the caller named.
        unresolved = self.root / PurePath(*PureWindowsPath(relative).parts)
        if unresolved.is_symlink():
            self._reject(relative, "sembolik bağ üzerinden yazma")
        if path.exists():
            self._reject_hard_link(path, relative)

        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._within(path.parent.resolve()):
            self._reject(relative, "üst dizin sandbox dışında")
        path.write_bytes(data)
        return path

    def mkdir(self, relative: str) -> Path:
        path = self.resolve(relative)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def remove(self, relative: str) -> bool:
        path = self.resolve(relative)
        if path == self.root:
            raise SandboxViolation("sandbox kökü silinemez")
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            return True
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, relative: str) -> bool:
        try:
            return self.resolve(relative).exists()
        except SandboxViolation:
            return False

    def listdir(self, relative: str = ".") -> list[str]:
        path = self.resolve(relative) if relative not in (".", "") else self.root
        if not path.is_dir():
            return []
        return sorted(entry.name for entry in path.iterdir())

    def file_count(self) -> int:
        return sum(1 for entry in self.root.rglob("*") if entry.is_file())

    def total_bytes(self) -> int:
        return sum(entry.stat().st_size for entry in self.root.rglob("*") if entry.is_file())

    # --------------------------------------------------------------- commands
    def _environment(self) -> dict[str, str]:
        """A scrubbed copy of the environment.

        Anything that looks like a credential is dropped. A lab process running an
        experiment has no business holding an API key, and a leaked one would go
        out through whatever the experiment happens to do.
        """
        env = {k: v for k, v in os.environ.items() if not _SECRET_PATTERN.search(k)}
        env["JARVIS_SANDBOX"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("PYTHONPATH", None)
        if not self.limits.allow_network:
            # Advisory only — respected by requests/urllib-based tools that read
            # proxy settings, ignored by anything that opens a socket directly.
            env["NO_PROXY"] = "*"
            env["HTTP_PROXY"] = env["HTTPS_PROXY"] = "http://127.0.0.1:9"
        env.update(self.limits.extra_env)
        return env

    def run(self, args: list[str], *, timeout: int | None = None) -> CommandResult:
        """Run an allowed command with the sandbox as its working directory."""
        if not args:
            raise SandboxViolation("boş komut")
        if not all(isinstance(arg, str) for arg in args):
            raise SandboxViolation("komut argümanları metin olmalı")

        program = PurePath(args[0]).name.casefold()
        program = program.removesuffix(".exe")
        if program not in self._allowed:
            raise SandboxViolation(
                f"izin verilmeyen komut: {args[0]} "
                f"(izinliler: {', '.join(sorted(self._allowed))})")

        limit = timeout or self.limits.timeout_s
        started = time.monotonic()
        timed_out = False

        try:
            # shell=False throughout: a shell would reintroduce quoting, globbing
            # and chaining, and the allowlist would only be checking the first word
            # of something that can contain three more commands.
            process = subprocess.Popen(
                args, cwd=self.root, env=self._environment(), shell=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except (OSError, ValueError) as exc:
            raise SandboxViolation(f"komut başlatılamadı: {exc}") from exc

        try:
            stdout, stderr = process.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(process)
            stdout, stderr = process.communicate()
            stderr = (stderr or "") + f"\n[sandbox] {limit}s sonra durduruldu"

        return CommandResult(
            args=list(args),
            returncode=process.returncode if not timed_out else -1,
            stdout=(stdout or "")[: self.limits.max_output],
            stderr=(stderr or "")[: self.limits.max_output],
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
        )

    @staticmethod
    def _kill_tree(process: subprocess.Popen) -> None:
        """Kill the process and its children.

        Popen.kill() ends only the direct child; a test runner that spawned
        workers would leave them running and holding the sandbox open.
        """
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                               capture_output=True, timeout=15, check=False)
                return
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("taskkill başarısız, doğrudan öldürülüyor: %s", exc)
        try:
            process.kill()
        except OSError:
            pass

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        """Empty the sandbox without destroying it."""
        for entry in self.root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    def dispose(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def status(self) -> dict[str, object]:
        return {
            "kok": str(self.root),
            "dosya": self.file_count(),
            "bayt": self.total_bytes(),
            "izinli_komutlar": sorted(self._allowed),
            "zaman_asimi_s": self.limits.timeout_s,
            "ag": self.limits.allow_network,
        }
