"""Tier 2 brain: Claude, reached through the local `claude` CLI in headless mode.

This runs on the user's existing Claude subscription — no API key, no per-call
invoice — but the subscription has a usage cap, so every call is metered by
jarvis.brain.budget before it is made.

The flags below are not cosmetic. Without them the CLI loads CLAUDE.md, the skill
catalogue, MCP tool definitions and machine context on every call: measured at
25,662 input tokens for a two-word prompt. With them, the same prompt costs 1,989.
The empty --tools value turns off the agent loop entirely, leaving pure text
generation; --setting-sources project combined with an empty working directory
means there is no project config to find.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("jarvis.brain.cloud")


@dataclass(slots=True)
class CloudResult:
    text: str
    ok: bool
    model: str
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None


class CloudBrain:
    def __init__(
        self,
        *,
        cli: str = "claude",
        model: str = "sonnet",
        model_light: str = "haiku",
        workdir: Path,
        timeout_s: int = 300,
    ) -> None:
        self.cli = cli
        self.model = model
        self.model_light = model_light
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return shutil.which(self.cli) is not None

    def ask(self, prompt: str, *, system: str, model: str | None = None) -> CloudResult:
        model = model or self.model
        args = [
            self.cli,
            "-p",
            "--model", model,
            "--tools", "",
            "--system-prompt", system,
            "--setting-sources", "project",
            "--disable-slash-commands",
            "--exclude-dynamic-system-prompt-sections",
            "--no-session-persistence",
            "--output-format", "json",
        ]

        started = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                input=prompt,
                cwd=self.workdir,
                timeout=self.timeout_s,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.monotonic() - started) * 1000)
            log.warning("cloud call timed out after %ss", self.timeout_s)
            return CloudResult("", False, model, duration_ms=elapsed,
                               error=f"zaman aşımı ({self.timeout_s}s)")
        except OSError as exc:
            return CloudResult("", False, model, error=str(exc))

        elapsed = int((time.monotonic() - started) * 1000)

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:500]
            log.warning("cloud call failed (exit %s): %s", proc.returncode, detail)
            return CloudResult("", False, model, duration_ms=elapsed,
                               error=detail or f"exit {proc.returncode}")

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return CloudResult("", False, model, duration_ms=elapsed,
                               error="çözümlenemeyen CLI çıktısı")

        if data.get("is_error"):
            return CloudResult("", False, model, duration_ms=elapsed,
                               error=str(data.get("result", "bilinmeyen hata"))[:500])

        usage = data.get("usage", {}) or {}
        return CloudResult(
            text=(data.get("result") or "").strip(),
            ok=True,
            model=model,
            cost_usd=float(data.get("total_cost_usd") or 0.0),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_tokens=int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0),
            duration_ms=int(data.get("duration_ms") or elapsed),
        )
