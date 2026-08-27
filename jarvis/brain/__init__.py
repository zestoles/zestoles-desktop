"""The two-tier brain, assembled.

Everything above this layer asks `Brain` a question and gets an answer; which model
produced it, whether the allowance permitted it, and what it cost is handled here.

Fallback is always downward: if the metered tier is unavailable, exhausted or
errors out, the local model answers instead and the reply is marked degraded so the
caller can say so honestly rather than pretending nothing happened.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from ..config import Config
from ..identity import PRODUCT_NAME
from ..persona import build as build_prompt
from ..persona import load_core
from .budget import Budget
from .cloud import CloudBrain
from .local import LocalBrain
from .router import CLOUD, LOCAL, Decision, Router

log = logging.getLogger("jarvis.brain")


@dataclass(slots=True)
class Answer:
    text: str
    tier: str
    model: str
    reason: str
    duration_ms: int = 0
    cost_usd: float = 0.0
    degraded: bool = False
    degraded_reason: str = ""
    error: str | None = None


class Brain:
    def __init__(self, config: Config, memory: object | None = None) -> None:
        self.config = config
        self.memory = memory
        self.core = load_core(config.path("paths.persona", "persona/core.md"))
        self.user_name = config.get("user.name", "")

        self.local = LocalBrain(
            host=config.get("local.host"),
            model=config.get("local.model"),
            timeout_s=config.get("local.timeout_s", 300),
            keep_alive=config.get("local.keep_alive", "30m"),
            think=config.get("local.think", False),
            temperature=config.get("local.temperature", 0.7),
            context_window=config.get("local.context_window", 8192),
        )
        self.cloud = CloudBrain(
            cli=config.get("cloud.cli", "claude"),
            model=config.get("cloud.model", "sonnet"),
            model_light=config.get("cloud.model_light", "haiku"),
            workdir=config.path("cloud.workdir", "data/cleanroom"),
            timeout_s=config.get("cloud.timeout_s", 300),
        )
        self.budget = Budget(
            config.path("paths.db", "data/jarvis.db"),
            per_hour=config.get("budget.cloud_calls_per_hour", 20),
            per_day=config.get("budget.cloud_calls_per_day", 120),
            per_night=config.get("budget.cloud_calls_per_night", 0),
            night_hours=tuple(config.get("budget.night_hours", [1, 8])),
            allow_at_night=config.get("budget.cloud_at_night", False),
        )
        # Every local call is counted at the client, not at this layer, because
        # not every local call comes through this layer.
        self.local.usage = self.budget.record
        self.router = Router(
            min_chars=config.get("router.cloud_min_chars", 240),
            threshold=config.get("router.cloud_score_threshold", 3),
            mode=config.get("router.mode", "auto"),
        )
        self.cloud_enabled = bool(config.get("cloud.enabled", True))
        self.last: Answer | None = None
        self.last_decision: Decision | None = None

    # ------------------------------------------------------------- selection
    def plan(self, text: str, *, forced: str | None = None) -> tuple[str, str]:
        """Return the tier that will answer and the reason, without calling anything."""
        decision: Decision = self.router.decide(text)
        self.last_decision = decision

        if forced in (LOCAL, CLOUD):
            return forced, "elle yönlendirildi"

        if decision.tier == LOCAL:
            return LOCAL, decision.reason

        if not self.cloud_enabled:
            return LOCAL, "Claude katmanı yapılandırmada kapalı"
        if not self.cloud.available():
            return LOCAL, "claude CLI bulunamadı"
        verdict = self.budget.check_cloud()
        if not verdict.allowed:
            return LOCAL, verdict.reason
        return CLOUD, decision.reason

    def system_prompt(self, tier: str, context: str = "") -> str:
        model = self.cloud.model if tier == CLOUD else self.local.model
        prompt = build_prompt(self.core, tier=tier, model=model, user_name=self.user_name)
        return f"{prompt}\n\n{context}" if context else prompt

    def recall(self, text: str) -> str:
        """Memory relevant to this message, or nothing if memory is unavailable."""
        if self.memory is None:
            return ""
        try:
            return self.memory.context_for(text)
        except Exception as exc:  # noqa: BLE001 - never let recall break a reply
            log.warning("hafıza çağrılamadı: %s", exc)
            return ""

    # ---------------------------------------------------------------- asking
    def ask(
        self,
        messages: list[dict[str, str]],
        *,
        forced: str | None = None,
        purpose: str = "chat",
    ) -> Answer:
        """Blocking answer. `messages` is the conversation without a system entry."""
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        tier, reason = self.plan(user_text, forced=forced)
        context = self.recall(user_text)

        if tier == CLOUD:
            answer = self._ask_cloud(messages, reason, purpose, context)
            if answer is not None:
                return self._remember(answer)
            reason = "Claude katmanı hata verdi, yerel modele düşüldü"
            degraded = True
        else:
            degraded = forced is None and self.router.decide(user_text).tier == CLOUD

        return self._remember(self._ask_local(messages, reason, purpose, degraded, context))

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        forced: str | None = None,
        purpose: str = "chat",
    ) -> Iterator[str]:
        """Token stream for the local tier; the cloud tier yields once, at the end.

        After the iterator is exhausted, `self.last` holds the Answer metadata.
        """
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        tier, reason = self.plan(user_text, forced=forced)
        context = self.recall(user_text)

        if tier == CLOUD:
            answer = self._ask_cloud(messages, reason, purpose, context)
            if answer is not None:
                self._remember(answer)
                yield answer.text
                return
            reason = "Claude katmanı hata verdi, yerel modele düşüldü"
            degraded = True
        else:
            degraded = forced is None and self.router.decide(user_text).tier == CLOUD

        started = time.monotonic()
        collected: list[str] = []
        prompt = self.system_prompt(LOCAL, context)
        payload = [{"role": "system", "content": prompt}, *messages]
        try:
            # record=False: bu satırı sarmalayıcı kendi amacıyla ("chat")
            # aşağıda yazıyor; iki kez yazmak defterde hayalet çağrı olur.
            for piece in self.local.stream(payload, record=False):
                collected.append(piece)
                yield piece
        except OSError as exc:
            log.warning("local stream failed: %s", exc)
            # A model that is down is a fact about the run, not an absence of
            # one. `chat()` records its failures at the client; `stream()` does
            # not go through `chat()`, so the failure has to be booked here or
            # the ledger shows a quiet night instead of a broken one.
            self.budget.record(tier=LOCAL, model=self.local.model, purpose=purpose,
                               duration_ms=int((time.monotonic() - started) * 1000),
                               ok=False, error=str(exc))
            self._remember(Answer("", LOCAL, self.local.model, reason, error=str(exc)))
            yield f"\n[yerel model yanıt vermedi: {exc}]"
            return

        elapsed = int((time.monotonic() - started) * 1000)
        self.budget.record(tier=LOCAL, model=self.local.model, purpose=purpose,
                           duration_ms=elapsed, ok=True)
        self._remember(
            Answer("".join(collected), LOCAL, self.local.model, reason,
                   duration_ms=elapsed, degraded=degraded,
                   degraded_reason=reason if degraded else "")
        )

    # -------------------------------------------------------------- internals
    def _ask_cloud(
        self, messages: list[dict[str, str]], reason: str, purpose: str, context: str = ""
    ) -> Answer | None:
        prompt = _flatten(messages)
        result = self.cloud.ask(prompt, system=self.system_prompt(CLOUD, context))
        self.budget.record(
            tier=CLOUD, model=result.model, purpose=purpose,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            cache_tokens=result.cache_tokens, cost_usd=result.cost_usd,
            duration_ms=result.duration_ms, ok=result.ok, error=result.error,
        )
        if not result.ok:
            log.warning("cloud tier failed: %s", result.error)
            return None
        return Answer(result.text, CLOUD, result.model, reason,
                      duration_ms=result.duration_ms, cost_usd=result.cost_usd)

    def _ask_local(self, messages, reason: str, purpose: str, degraded: bool,
                   context: str = "") -> Answer:
        started = time.monotonic()
        payload = [{"role": "system", "content": self.system_prompt(LOCAL, context)}, *messages]
        try:
            # The client records this call, success or failure — recording it here
            # too would count it twice.
            text = self.local.chat(payload, purpose=purpose)
        except OSError as exc:
            log.warning("local tier failed: %s", exc)
            return Answer("", LOCAL, self.local.model, reason, error=str(exc))

        elapsed = int((time.monotonic() - started) * 1000)
        return Answer(text, LOCAL, self.local.model, reason, duration_ms=elapsed,
                      degraded=degraded, degraded_reason=reason if degraded else "")

    def _remember(self, answer: Answer) -> Answer:
        self.last = answer
        return answer

    # ----------------------------------------------------------------- status
    def status(self) -> dict[str, object]:
        return {
            "local_up": self.local.available(),
            "local_model": self.local.model,
            "local_model_present": self.local.has_model(),
            "cloud_up": self.cloud.available() and self.cloud_enabled,
            "cloud_enabled": self.cloud_enabled,
            "cloud_model": self.cloud.model,
            "budget_verdict": self.budget.check_cloud(),
            "usage": self.budget.usage(),
            "router_mode": self.router.mode,
        }


def _flatten(messages: list[dict[str, str]]) -> str:
    """The CLI takes one prompt string, so prior turns become labelled context."""
    if len(messages) == 1:
        return messages[0]["content"]
    lines = []
    for msg in messages[:-1]:
        speaker = "Kullanıcı" if msg["role"] == "user" else PRODUCT_NAME
        lines.append(f"{speaker}: {msg['content']}")
    history = "\n".join(lines)
    return f"[Önceki konuşma]\n{history}\n\n[Şimdiki mesaj]\n{messages[-1]['content']}"


__all__ = ["Brain", "Answer", "Budget", "CloudBrain", "LocalBrain", "Router", "CLOUD", "LOCAL"]
