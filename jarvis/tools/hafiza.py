"""Remembering things about the user, as tools the model can reach.

JARVIS learns about the person by being told, not by concluding. These three
tools are how that happens inside an ordinary conversation:

    memory.remember   keep a preference the user agreed to keep
    memory.recall     say what is known, and where each item came from
    memory.forget     delete something

The interesting one is `remember`, and the interesting part of it is `onay`.

## Why consent is an argument rather than a policy

A model asked to "only save things the user agreed to" will save things the user
did not agree to, because it is predicting text rather than following a rule.
Making consent an explicit argument moves the decision somewhere it can be
checked: `Profile.accept()` refuses anything without it, and the refusal is
returned as a tool failure the model has to read. So the sequence that works is
the honest one -- ask, hear yes, then call with `onay=true` -- and the sequence
that skips the asking simply fails.

It is LOW risk on purpose. Writing "geceleri çalışıyor" into a preferences table
is not a dangerous operation and putting a confirmation card in front of it
would train the user to click through confirmations, which is how the ones that
matter stop being read. The protection here is provenance, not a dialog.

## What cannot be stored

Inference. `kaynak="cikarim"` is accepted as an argument and always refused as a
write, with a message telling the model to ask instead. "Sen stresli birisin" is
not a fact about a person however fluently it was produced.
"""

from __future__ import annotations

import logging
from typing import Any

from . import LOW, ToolResult, tool

log = logging.getLogger("jarvis.tools.hafiza")

#: Set by the runtime when a profile exists. Absent in a bare tool test, and the
#: tools say so rather than failing oddly.
_PROFILE: Any = None


def provide_profile(profile: Any) -> None:
    global _PROFILE  # noqa: PLW0603 - the registry is process-wide by design
    _PROFILE = profile


def _profile():
    return _PROFILE


@tool("memory.remember", risk=LOW,
      summary="Kullanıcının açıkça kaydedilmesini istediği bir tercihi hatırla")
def remember(*, konu: str, ayrinti: str, onay: bool = False,
             kaynak: str = "kullanici", guven: str = "yuksek",
             **_extra) -> ToolResult:
    """Keep something the user asked to be kept.

    `onay` must be true, and it means they said yes -- not that it seemed like a
    good idea. Without it the write is refused and the refusal says to ask.
    """
    from ..memory.profile import Confidence, Consent, Source

    profile = _profile()
    if profile is None:
        return ToolResult(False, error="hafıza bu süreçte açık değil",
                          tool="memory.remember", risk=LOW)

    consent = Consent.GRANTED if onay else Consent.UNKNOWN
    source = str(kaynak or Source.USER).strip().lower()
    confidence = str(guven or Confidence.HIGH).strip().lower()

    item, why = profile.remember(konu, ayrinti, source=source,
                                 confidence=confidence, consent=consent)
    if item is None:
        return ToolResult(False, error=why, tool="memory.remember", risk=LOW)
    return ToolResult(True, output=f"kaydedildi — {item.as_line()}",
                      tool="memory.remember", risk=LOW)


@tool("memory.recall", risk=LOW,
      summary="Kullanıcı hakkında kayıtlı olan şeyleri say")
def recall(**_extra) -> ToolResult:
    """List what is known. Only what may be stated: inference is not included."""
    profile = _profile()
    if profile is None:
        return ToolResult(False, error="hafıza bu süreçte açık değil",
                          tool="memory.recall", risk=LOW)

    items = profile.facts()
    if not items:
        return ToolResult(True, output="Kullanıcı hakkında kayıtlı bir şey yok.",
                          tool="memory.recall", risk=LOW)
    body = "\n".join(f"- {item.as_line()}" for item in items)
    return ToolResult(True, output=f"{len(items)} kayıt:\n{body}",
                      tool="memory.recall", risk=LOW)


@tool("memory.forget", risk=LOW,
      summary="Kullanıcı hakkında kayıtlı bir şeyi unut")
def forget(*, konu: str, **_extra) -> ToolResult:
    """Delete. The row is removed, not hidden."""
    profile = _profile()
    if profile is None:
        return ToolResult(False, error="hafıza bu süreçte açık değil",
                          tool="memory.forget", risk=LOW)

    removed = profile.forget(konu)
    if not removed:
        return ToolResult(False, error=f"böyle bir kayıt yok: {konu}",
                          tool="memory.forget", risk=LOW)
    return ToolResult(True, output=f"{removed} kayıt unutuldu: {konu}",
                      tool="memory.forget", risk=LOW)


__all__ = ["provide_profile", "remember", "recall", "forget"]
