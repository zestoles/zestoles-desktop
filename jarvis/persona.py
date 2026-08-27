"""Builds the system prompt: the persona file plus live runtime context.

persona/core.md is the part that defines who ZESTOLES is and is meant to be edited
by hand. Everything appended here is situational — the time, which brain is
answering, what that brain can and cannot do well. Telling the local model that it
is the local model matters: it should hand off rather than bluff its way through
work it is not good at.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .identity import PRODUCT_NAME

log = logging.getLogger("jarvis.persona")

FALLBACK = (
    f"You are {PRODUCT_NAME}, a personal AI system. Reason in English, always answer in "
    "Turkish. Semi-formal, calm, concise. No filler enthusiasm. Say when you do "
    "not know something."
)

_LOCAL_NOTE = """\
## Current runtime

You are answering from the local model ({model}) running on this machine. It is
fast and free but noticeably weaker at long planning chains, precise code and
careful multi-step reasoning.

Work within that. Answer directly what you can answer. When a request clearly
needs deeper reasoning than you can give reliably, say so plainly in one sentence
and suggest routing it to the Claude tier — do not produce a confident-sounding
answer you cannot stand behind."""

_CLOUD_NOTE = """\
## Current runtime

You are answering from the Claude tier ({model}). This tier is metered against the
user's subscription, so it is used for work that genuinely needs it. Be
substantive and complete — a shallow answer here wastes the allowance."""

_DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def load_core(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        log.warning("persona file unreadable (%s), using fallback", exc)
        return FALLBACK


def build(core: str, *, tier: str, model: str, user_name: str = "") -> str:
    now = datetime.now()
    parts = [core]

    context = [
        "## Situation",
        f"Local date and time: {now:%Y-%m-%d %H:%M} ({_DAYS[now.weekday()]})",
    ]
    if user_name:
        context.append(f"You are speaking with {user_name}.")
    parts.append("\n".join(context))

    note = _CLOUD_NOTE if tier == "cloud" else _LOCAL_NOTE
    parts.append(note.format(model=model))

    return "\n\n".join(parts)
