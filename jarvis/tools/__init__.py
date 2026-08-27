"""Tools JARVIS runs on the user's real machine, at the user's request.

## Why this is not the agent permission model

`agents/permissions.py` refuses `shell` and `fs.write` to anything that is not
holding a Sandbox, and that stays exactly as it is. It guards *autonomous* work:
the improvement engine deciding by itself to write a file is the failure mode the
whole lab exists to prevent, and nothing here relaxes it.

This module answers a different question. When the user types "create test.txt on my
desktop", the trust basis is not "an agent inferred this was a good idea" — it is
a person asking for a specific thing in a live session. That is a different
authority, so it gets a different layer rather than a hole in the existing one.

The two never mix: nothing in this package imports the agent grant system, and
nothing in the agent path imports this.

## Risk, not blanket confirmation

Asking permission for everything trains a user to click yes without reading,
which is worse than not asking. So each tool declares what it is:

    LOW     reads and lists. Runs immediately.
    MEDIUM  changes something recoverable. Needs an explicit confirmed=True.
    HIGH    destructive or outside the workspace. Needs confirmed=True, and the
            worst cases are refused outright rather than confirmable.

`run()` will not perform a MEDIUM or HIGH tool without `confirmed=True`. It
returns a result saying so instead of raising, because the caller that has to ask
the user is the UI, and an exception is not a question.

## The workspace

Every path is resolved against a workspace root and must stay inside it. That is
a containment boundary, not a security boundary — the same honest limit the
sandbox notes carry. It stops accidents and mistakes, not a determined attacker
who already has the ability to run code as this user.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.tools")

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
RISKS = (LOW, MEDIUM, HIGH)

#: Windows device names. A path component equal to one of these is not a file.
_DEVICE_NAMES = frozenset({
    "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})

#: Commands that have no legitimate place in an assistant session. Refused
#: outright rather than offered for confirmation: there is no wording of "are you
#: sure" that makes `format c:` a reasonable thing to have asked an assistant.
_NEVER_RUN = (
    re.compile(r"\bformat\s+[a-z]:", re.I),
    re.compile(r"\bdiskpart\b", re.I),
    re.compile(r"\bmkfs(\.\w+)?\b", re.I),
    re.compile(r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+/(\s|$)", re.I),
    re.compile(r"\bdel\s+/[sq]\b.*\\\*", re.I),
    re.compile(r"\bshutdown\b|\bReset-Computer\b|\bRestart-Computer\b", re.I),
    re.compile(r"\bcipher\s+/w\b", re.I),
    re.compile(r"\breg\s+delete\b", re.I),
    re.compile(r"\bvssadmin\s+delete\b", re.I),
    re.compile(r"\bbcdedit\b", re.I),
    re.compile(r"\bnetsh\s+advfirewall\s+set\b", re.I),
    re.compile(r"Set-MpPreference|Add-MpPreference", re.I),
)

#: Commands that change something but are ordinary requests. Confirmable.
_NEEDS_CONFIRMATION = (
    re.compile(r"\b(del|erase|rmdir|rd|rm|remove-item)\b", re.I),
    re.compile(r"\b(move|mv|ren|rename|rename-item)\b", re.I),
    re.compile(r"\b(pip|npm|winget|choco)\s+(install|uninstall)\b", re.I),
    re.compile(r"\bgit\s+(push|reset|clean)\b", re.I),
    re.compile(r"\b(taskkill|stop-process|kill)\b", re.I),
)

DEFAULT_TIMEOUT_S = 60
MAX_OUTPUT = 100_000
MAX_READ_BYTES = 2_000_000


class ToolError(RuntimeError):
    """A tool refused to act. The message is meant to be shown to the user."""


class WorkspaceViolation(ToolError):
    pass


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    needs_confirmation: bool = False
    #: Empty means "the registry's risk for this tool". A tool sets it only when
    #: this particular call was riskier than the tool generally is — `shell.run`
    #: is MEDIUM, but refusing `format c:` is a HIGH-risk answer.
    risk: str = ""
    tool: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def summary(self) -> str:
        if self.needs_confirmation:
            return f"onay bekliyor ({self.risk}): {self.tool}"
        if self.ok:
            return self.output[:200] or "tamam"
        return f"başarısız: {self.error}"


class Workspace:
    """Where tools are allowed to touch, and the rules for getting there."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str, *, must_exist: bool = False) -> Path:
        text = str(relative or "").strip()
        if not text:
            raise WorkspaceViolation("yol boş")
        if "\x00" in text:
            raise WorkspaceViolation("yolda NUL baytı var")

        candidate = Path(text)
        target = candidate if candidate.is_absolute() else self.root / candidate
        try:
            target = target.resolve()
        except (OSError, ValueError) as exc:
            raise WorkspaceViolation(f"yol çözümlenemedi: {text}") from exc

        if target != self.root and not target.is_relative_to(self.root):
            raise WorkspaceViolation(
                f"çalışma alanı dışında: {target} (izinli kök: {self.root})")
        for part in target.parts:
            stem = part.split(".")[0].lower()
            if stem in _DEVICE_NAMES:
                raise WorkspaceViolation(f"aygıt adı kullanılamaz: {part}")
        if must_exist and not target.exists():
            raise ToolError(f"bulunamadı: {self.show(target)}")
        return target

    def show(self, path: Path) -> str:
        """A path as the user should read it: relative to the workspace."""
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)


@dataclass(slots=True)
class Tool:
    name: str
    risk: str
    summary: str
    run: Callable[..., ToolResult]


REGISTRY: dict[str, Tool] = {}

#: Live subsystems some tools need and none of them may assume. A tool signature
#: carries only its own arguments plus the workspace, so anything that has to
#: reach a running system — the research pipeline, for one — looks it up here and
#: says so plainly when it is absent. Wired once at session build; absent in a
#: build where that subsystem failed to come up, which is a refusal rather than
#: a crash.
SERVICES: dict[str, Any] = {}


def provide(name: str, service: Any) -> None:
    SERVICES[name] = service


def tool(name: str, *, risk: str = LOW, summary: str = "") -> Callable:
    if risk not in RISKS:
        raise ValueError(f"bilinmeyen risk seviyesi: {risk}")

    def register(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        if name in REGISTRY:
            raise ValueError(f"araç adı zaten kayıtlı: {name}")
        REGISTRY[name] = Tool(name, risk, summary or (func.__doc__ or "").strip(), func)
        return func

    return register


def get(name: str) -> Tool | None:
    return REGISTRY.get(name)


def names() -> list[str]:
    return sorted(REGISTRY)


def catalogue() -> list[dict[str, str]]:
    """What the assistant may call, for building a model-facing tool list.

    Includes the argument names, read from the same signature `check_arguments`
    validates against -- so what the model is told and what the tool accepts
    cannot drift. Leaving them out was measured to produce exactly the failure
    it sounds like: the right tool called with no arguments, because the names
    were never stated and could only be guessed.
    """
    return [{"name": t.name, "risk": t.risk, "summary": t.summary,
             "arguments": signature_of(t)}
            for t in sorted(REGISTRY.values(), key=lambda t: t.name)]


def signature_of(entry: Tool) -> str:
    """The call shape, as a model should read it: `konu, ayrinti, [onay]`.

    Optional arguments are bracketed rather than omitted; a tool whose useful
    behaviour is behind a default nobody mentions is a tool nobody uses well.
    """
    try:
        signature = inspect.signature(entry.run)
    except (TypeError, ValueError):  # pragma: no cover - builtins have none
        return ""
    parts = []
    for name, parameter in signature.parameters.items():
        if name.startswith("_") or parameter.kind in (
                inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if name in ("workspace", "confirmed"):
            continue          # supplied by the loop, never by the model
        parts.append(name if parameter.default is inspect.Parameter.empty
                     else f"[{name}]")
    return ", ".join(parts)


def check_arguments(entry: Tool, arguments: dict[str, Any]) -> str:
    """Why this call does not fit the tool, or empty when it does.

    A language model picks these arguments, so "close enough" arrives regularly:
    an extra key, a missing required one, a plausible synonym. Reading the
    signature turns that into a sentence the model can act on, instead of a
    TypeError that reaches the interface as a spinner that never stops.
    """
    try:
        signature = inspect.signature(entry.run)
    except (TypeError, ValueError):  # pragma: no cover - builtins have none
        return ""

    accepted, required = set(), set()
    for parameter in signature.parameters.values():
        if parameter.name == "workspace":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return ""  # takes anything; nothing to check
        accepted.add(parameter.name)
        if parameter.default is inspect.Parameter.empty:
            required.add(parameter.name)

    unknown = sorted(set(arguments) - accepted)
    if unknown:
        return (f"{entry.name} şu argümanları almıyor: {', '.join(unknown)} "
                f"(alabildikleri: {', '.join(sorted(accepted)) or 'yok'})")
    missing = sorted(required - set(arguments))
    if missing:
        return f"{entry.name} için eksik argüman: {', '.join(missing)}"
    return ""


def run(name: str, *, workspace: Workspace, confirmed: bool = False,
        **kwargs: Any) -> ToolResult:
    """Run a registered tool, refusing anything that needs an answer first."""
    entry = get(name)
    if entry is None:
        return ToolResult(False, error=f"bilinmeyen araç: {name}", tool=name)

    # Checked before the risk gate so a malformed call is refused outright
    # rather than put to the user for confirmation.
    problem = check_arguments(entry, kwargs)
    if problem:
        return ToolResult(False, error=problem, tool=name, risk=entry.risk,
                          detail=dict(kwargs))

    if entry.risk in (MEDIUM, HIGH) and not confirmed:
        return ToolResult(
            False, needs_confirmation=True, risk=entry.risk, tool=name,
            error=f"{entry.summary or name} — onay gerekiyor",
            detail=dict(kwargs))

    started = time.monotonic()
    try:
        result = entry.run(workspace=workspace, **kwargs)
    except ToolError as exc:
        result = ToolResult(False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a tool must not take the turn down
        # Measured: the model called system.info with an argument the tool does
        # not take, and the TypeError travelled all the way to the interface as
        # a dead spinner. Anything a tool raises is a failed call, and the loop
        # is told so in words it can act on.
        log.warning("araç hatası (%s): %s", name, exc)
        result = ToolResult(False, error=f"{type(exc).__name__}: {exc}")
    result.tool = name
    result.risk = result.risk or entry.risk
    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result


# ------------------------------------------------------------------ filesystem
@tool("fs.list", risk=LOW, summary="Bir klasördeki dosyaları listeler")
def _list(*, workspace: Workspace, path: str = ".", pattern: str = "*") -> ToolResult:
    target = workspace.resolve(path, must_exist=True)
    if not target.is_dir():
        raise ToolError(f"klasör değil: {workspace.show(target)}")
    entries = sorted(target.glob(pattern), key=lambda p: (p.is_file(), p.name.lower()))
    lines = [f"{'[D] ' if e.is_dir() else '    '}{e.name}"
             + ("" if e.is_dir() else f"  ({e.stat().st_size} B)")
             for e in entries[:500]]
    return ToolResult(True, output="\n".join(lines) or "(boş)",
                      detail={"count": len(entries), "path": str(target)})


@tool("fs.read", risk=LOW, summary="Bir dosyanın içeriğini okur")
def _read(*, workspace: Workspace, path: str, max_bytes: int = MAX_READ_BYTES) -> ToolResult:
    target = workspace.resolve(path, must_exist=True)
    if not target.is_file():
        raise ToolError(f"dosya değil: {workspace.show(target)}")
    size = target.stat().st_size
    if size > max_bytes:
        raise ToolError(
            f"dosya çok büyük ({size} B > {max_bytes} B): {workspace.show(target)}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"metin dosyası değil: {workspace.show(target)}") from None
    return ToolResult(True, output=text, detail={"bytes": size, "path": str(target)})


@tool("fs.search", risk=LOW, summary="Ada göre dosya arar")
def _search(*, workspace: Workspace, pattern: str, path: str = ".",
            limit: int = 200) -> ToolResult:
    target = workspace.resolve(path, must_exist=True)
    found = []
    for entry in target.rglob(pattern):
        found.append(workspace.show(entry))
        if len(found) >= limit:
            break
    return ToolResult(True, output="\n".join(found) or "(eşleşme yok)",
                      detail={"count": len(found)})


@tool("fs.stat", risk=LOW, summary="Dosya bilgilerini gösterir")
def _stat(*, workspace: Workspace, path: str) -> ToolResult:
    target = workspace.resolve(path, must_exist=True)
    info = target.stat()
    kind = "klasör" if target.is_dir() else "dosya"
    return ToolResult(
        True,
        output=f"{workspace.show(target)} · {kind} · {info.st_size} B",
        detail={"size": info.st_size, "modified": info.st_mtime, "is_dir": target.is_dir()})


@tool("fs.write", risk=MEDIUM, summary="Dosya oluşturur veya içeriğini değiştirir")
def _write(*, workspace: Workspace, path: str, content: str,
           append: bool = False) -> ToolResult:
    target = workspace.resolve(path)
    if target.is_dir():
        raise ToolError(f"klasörün üzerine yazılamaz: {workspace.show(target)}")
    existed = target.is_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a" if append else "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    action = "eklendi" if append else ("güncellendi" if existed else "oluşturuldu")
    return ToolResult(True, output=f"{workspace.show(target)} {action}",
                      detail={"path": str(target), "existed": existed,
                              "bytes": len(content.encode('utf-8'))})


@tool("fs.mkdir", risk=MEDIUM, summary="Klasör oluşturur")
def _mkdir(*, workspace: Workspace, path: str) -> ToolResult:
    target = workspace.resolve(path)
    existed = target.is_dir()
    target.mkdir(parents=True, exist_ok=True)
    return ToolResult(True,
                      output=f"{workspace.show(target)} "
                             + ("zaten vardı" if existed else "oluşturuldu"),
                      detail={"path": str(target), "existed": existed})


@tool("fs.move", risk=MEDIUM, summary="Dosya taşır veya adını değiştirir")
def _move(*, workspace: Workspace, source: str, destination: str) -> ToolResult:
    src = workspace.resolve(source, must_exist=True)
    dst = workspace.resolve(destination)
    if dst.exists():
        raise ToolError(f"hedef zaten var: {workspace.show(dst)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return ToolResult(True, output=f"{workspace.show(src)} → {workspace.show(dst)}")


@tool("fs.copy", risk=MEDIUM, summary="Dosya kopyalar")
def _copy(*, workspace: Workspace, source: str, destination: str) -> ToolResult:
    src = workspace.resolve(source, must_exist=True)
    dst = workspace.resolve(destination)
    if dst.exists():
        raise ToolError(f"hedef zaten var: {workspace.show(dst)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return ToolResult(True, output=f"{workspace.show(src)} → {workspace.show(dst)}")


# -------------------------------------------------------------------- terminal
def classify_command(command: str) -> str:
    """What running this would be: LOW, MEDIUM, or HIGH.

    Pure, so the risk of a command can be shown to the user before anything is
    started, and so the classification can be tested without running anything.
    """
    text = str(command or "")
    for pattern in _NEVER_RUN:
        if pattern.search(text):
            return HIGH
    for pattern in _NEEDS_CONFIRMATION:
        if pattern.search(text):
            return MEDIUM
    return LOW


def refuses(command: str) -> str:
    """Why this command will not be run at all, or empty if it may be."""
    for pattern in _NEVER_RUN:
        if pattern.search(str(command or "")):
            return ("Bu komut sistemi kalıcı olarak etkileyebilir; "
                    "ZESTOLES bunu çalıştırmıyor. Gerekiyorsa kendiniz çalıştırın.")
    return ""


@tool("shell.run", risk=MEDIUM, summary="Komut çalıştırır")
def _shell(*, workspace: Workspace, command: str,
           timeout_s: int = DEFAULT_TIMEOUT_S) -> ToolResult:
    reason = refuses(command)
    if reason:
        return ToolResult(False, error=reason, risk=HIGH,
                          detail={"command": command, "refused": True})

    started = time.monotonic()
    timeout = max(1, int(timeout_s))
    try:
        # Intentional shell capability. The registry marks this MEDIUM risk and
        # run() cannot reach it without a separate user confirmation.
        process = subprocess.Popen(  # noqa: S602  # nosec B602
            command, cwd=workspace.root, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, shell=True, encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return ToolResult(False, error=f"komut çalıştırılamadı: {exc}",
                          detail={"command": command})

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # `shell=True` means the child is a shell and the real work is its
        # child. Killing only the process we hold leaves that grandchild
        # running after JARVIS has been closed, which is exactly the thing V1
        # promises does not happen. Same tree kill lab/sandbox.py does.
        killed = _kill_tree(process)
        process.communicate()
        return ToolResult(
            False, error=f"komut {timeout} saniyede bitmedi ve durduruldu",
            detail={"command": command, "timed_out": True, "tree_killed": killed})

    stdout = (stdout or "")[:MAX_OUTPUT]
    stderr = (stderr or "")[:MAX_OUTPUT]
    elapsed = int((time.monotonic() - started) * 1000)
    ok = process.returncode == 0
    body = stdout if ok else (stdout + ("\n" + stderr if stderr else ""))
    return ToolResult(
        ok, output=body.strip(),
        error="" if ok else (stderr.strip() or f"çıkış kodu {process.returncode}"),
        detail={"command": command, "exit_code": process.returncode,
                "stdout": stdout, "stderr": stderr, "duration_ms": elapsed})


def _kill_tree(process: subprocess.Popen) -> bool:
    """End a process and everything it started. True when the tree was reached."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                           capture_output=True, timeout=15, check=False)
            return True
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("süreç ağacı öldürülemedi (%s): %s", process.pid, exc)
    try:
        process.kill()
    except OSError:
        pass
    return False


@tool("system.info", risk=LOW, summary="Sistem bilgilerini gösterir")
def _system(*, workspace: Workspace) -> ToolResult:
    import platform

    usage = shutil.disk_usage(workspace.root)
    lines = [
        f"İşletim sistemi : {platform.system()} {platform.release()}",
        f"Makine          : {platform.machine()}",
        f"Python          : {platform.python_version()}",
        f"Çalışma alanı   : {workspace.root}",
        f"Disk            : {usage.used // 2**30} / {usage.total // 2**30} GB kullanılıyor",
        f"CPU çekirdek    : {os.cpu_count()}",
    ]
    return ToolResult(True, output="\n".join(lines))


__all__ = [
    "LOW", "MEDIUM", "HIGH", "RISKS", "Tool", "ToolError", "ToolResult",
    "Workspace", "WorkspaceViolation", "REGISTRY", "SERVICES", "provide",
    "tool", "get", "names", "catalogue", "run", "classify_command", "refuses",
]

# Imported last: web.py needs the decorator defined above, and importing it here
# is what puts its tools in the registry.
from . import web  # noqa: E402,F401  isort:skip
from . import hafiza  # noqa: E402,F401  isort:skip
from . import windows  # noqa: E402,F401  isort:skip
from . import documents  # noqa: E402,F401  isort:skip
from . import reminders  # noqa: E402,F401  isort:skip
