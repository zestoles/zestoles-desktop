"""What an agent is, and how one run happens.

An agent here is a role, not a process: a system prompt, an output shape, a model
tier and a permission set. Nothing runs continuously and nothing holds a model
resident of its own accord — a run borrows the local model, produces one result,
and is gone. That keeps the cost of adding a specialist near zero and means a
misbehaving role can never leak into the next task.

Two rules the rest of the package depends on:

  Every result carries provenance. An agent's output is a claim by JARVIS, not a
  fact, and it is labelled AGENT_SOURCED from the moment it exists so that nothing
  downstream has to remember to be suspicious.

  Nothing here decides the model. The orchestrator picks one model for the whole
  run and passes it down, because on this machine loading a second model evicts
  the first and costs about seventy seconds — measured, not assumed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..autonomy.events import EventLog
from .permissions import MEMORY_READ, Grant

log = logging.getLogger("jarvis.agents")

#: Provenance for anything an agent produced. Belongs to the unverified family:
#: an agent asserting something is exactly the case S1's gate exists to catch.
AGENT_SOURCED = "ajan"

FAST = "fast"
HEAVY = "heavy"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    title: str
    purpose: str          # one line; the planner reads these to choose a role
    system: str
    capabilities: frozenset[str] = frozenset()
    schema: dict[str, Any] | None = None
    temperature: float = 0.4
    tier: str = FAST
    max_output_chars: int = 6000


@dataclass(slots=True)
class AgentResult:
    agent: str
    ok: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0
    model: str = ""
    source: str = AGENT_SOURCED
    verified: bool = False
    verdict: Any = None

    @property
    def trustworthy(self) -> bool:
        """Passed verification. Still an agent claim — just one that survived review."""
        return self.ok and self.verified


class AgentContext:
    """Everything a single run is allowed to reach.

    Constructed per run by the orchestrator. Capabilities the grant does not carry
    are absent rather than disabled: recall() raises instead of quietly returning
    nothing, so a missing permission surfaces as a failure to fix, not as an agent
    that mysteriously knows less than it should.
    """

    def __init__(
        self,
        *,
        brain,
        events: EventLog,
        grant: Grant,
        model: str,
        should_stop=lambda: False,
        memory=None,
        run_id: str = "",
    ) -> None:
        self.brain = brain
        self.events = events
        self.grant = grant
        self.model = model
        self.should_stop = should_stop
        self._memory = memory
        self.run_id = run_id

    def recall(self, query: str) -> str:
        self.grant.require(MEMORY_READ)
        if self._memory is None:
            return ""
        try:
            return self._memory.context_for(query)
        except Exception as exc:  # noqa: BLE001 - recall is help, not a dependency
            log.warning("ajan hafızayı çağıramadı: %s", exc)
            return ""

    def emit(self, kind: str, message: str, *, level: str = "info", **data: Any) -> None:
        self.events.publish("agent", kind, message,
                            level=level, data={"run": self.run_id, **data})


def run_agent(spec: AgentSpec, instruction: str, ctx: AgentContext, *,
              context_text: str = "") -> AgentResult:
    """Execute one agent once. Never raises — failure comes back as a result."""
    started = time.monotonic()
    ctx.emit("start", f"{spec.title} çalışıyor", agent=spec.name)

    system = spec.system
    if context_text:
        system = f"{system}\n\n## Bağlam\n\n{context_text}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]

    try:
        raw = ctx.brain.local.chat(
            messages,
            temperature=spec.temperature,
            schema=spec.schema,
            model=ctx.model,
        )
    except Exception as exc:  # noqa: BLE001 - one agent failing is not a crash
        elapsed = int((time.monotonic() - started) * 1000)
        detail = f"{type(exc).__name__}: {exc}"
        log.warning("ajan hata verdi (%s): %s", spec.name, detail)
        ctx.emit("error", f"{spec.title} hata verdi: {detail}", level="error", agent=spec.name)
        return AgentResult(spec.name, False, error=detail, duration_ms=elapsed, model=ctx.model)

    elapsed = int((time.monotonic() - started) * 1000)
    output = (raw or "").strip()

    if not output:
        ctx.emit("error", f"{spec.title} boş cevap verdi", level="warn", agent=spec.name)
        return AgentResult(spec.name, False, error="boş cevap",
                           duration_ms=elapsed, model=ctx.model)

    if len(output) > spec.max_output_chars:
        output = output[: spec.max_output_chars] + "\n…[kısaltıldı]"

    ctx.emit("done", f"{spec.title} bitirdi ({elapsed} ms)", agent=spec.name, ms=elapsed)
    return AgentResult(spec.name, True, output=output, duration_ms=elapsed, model=ctx.model)
