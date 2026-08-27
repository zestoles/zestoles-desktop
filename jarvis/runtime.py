"""Assembling JARVIS: the one place that knows how the layers fit together.

Everything below this module is a domain layer that knows only what it needs.
Everything above it — the terminal today, a websocket and an interface later —
talks to `Runtime` and to `SharedState`, and to nothing else.

That is the boundary this module exists to create. Before it, the terminal reached
directly into `AgentSystem`, `ResearchSystem`, `AutonomyCore` and the rest, which
meant a second front end would have had to reach into all of them too, and would
have grown its own copy of the assembly order, the failure handling and the
teardown. One composition root, two consumers.

## The projector

Domain layers publish events. They do not know what a UI is, and adding one must
not change them. So state is derived here: `EventProjector` subscribes to the event
log and translates what happened into what is currently true.

    domain → EventLog → projector → SharedState → (S7B) websocket

The arrow only goes one way. Nothing downstream of the projector can make a domain
module import it, which is what keeps the graph acyclic as the top grows.

## Degradation is normal

Ollama may be down, the vault may be unreadable, autonomy may be switched off. Each
subsystem is built inside its own guard and a failure is recorded as a warning
rather than raised, because a JARVIS that cannot research is still a JARVIS that
can talk, and refusing to start at all would be the worse answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from . import state as state_module
from .config import Config
from .state import SharedState

log = logging.getLogger("jarvis.runtime")


@dataclass(slots=True)
class Runtime:
    """Every subsystem, plus the state view over them."""

    config: Config
    state: SharedState
    memory: Any = None
    brain: Any = None
    core: Any = None
    agents: Any = None
    research: Any = None
    lab: Any = None
    improve: Any = None
    voice: Any = None
    telegram: Any = None
    documents: Any = None
    reminders: Any = None
    warnings: list[str] = field(default_factory=list)
    _unsubscribe: Any = None

    @property
    def events(self):
        """The event log, when autonomy came up. Otherwise nothing to subscribe to."""
        return self.core.events if self.core is not None else None

    # ------------------------------------------------------------- lifecycle
    def start_autonomy(self) -> bool:
        if self.reminders is not None:
            self.reminders.start()
        if self.core is None:
            return self.reminders is not None
        self.core.start()
        self.refresh()
        return bool(self.core.scheduler.running)

    def shutdown(self, *, timeout: float = 15.0) -> None:
        if self.reminders is not None:
            try:
                self.reminders.stop()
            except Exception as exc:  # noqa: BLE001
                log.debug("hatırlatıcılar kapatılamadı: %s", exc)
        if self.telegram is not None:
            try:
                self.telegram.stop()
            except Exception as exc:  # noqa: BLE001 - shutdown continues
                log.debug("Telegram kapatılamadı: %s", exc)
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception as exc:  # noqa: BLE001
                log.debug("projeksiyon aboneliği kapatılamadı: %s", exc)
            self._unsubscribe = None
        if self.core is not None and self.core.scheduler.running:
            self.core.stop(timeout=timeout)
        if self.voice is not None:
            try:
                self.voice.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown continues
                log.debug("ses modeli bırakılamadı: %s", exc)
        seen: set[int] = set()
        clients = [getattr(getattr(self, "brain", None), "local", None),
                   getattr(getattr(self, "memory", None), "local", None),
                   getattr(getattr(self, "memory", None), "embedder", None)]
        for client in clients:
            unload = getattr(client, "unload", None)
            if client is None or id(client) in seen or not callable(unload):
                continue
            seen.add(id(client))
            try:
                unload()
            except Exception as exc:  # noqa: BLE001 - resource release is best effort
                log.debug("yerel model bırakılamadı: %s", exc)
        self.state.set_activity(state_module.IDLE, "kapandı")

    # ---------------------------------------------------------------- state
    def refresh(self) -> dict[str, Any]:
        """Pull current values from each subsystem into shared state.

        Events carry what happened; some values only exist as a subsystem's own
        idea of itself. Those are polled here rather than invented by the
        projector, so a reader never sees a number nobody actually reported.
        """
        self.state.update(state_module.SYSTEM,
                          uptime_s=round(self.state.uptime_s, 1),
                          warnings=list(self.warnings))

        if self.core is not None:
            self._safely(state_module.SYSTEM, self._resources)
        if self.brain is not None:
            self._safely(state_module.BRAIN, self._brain_status)
        if self.memory is not None:
            self._safely(state_module.MEMORY, self.memory.stats)
        if self.documents is not None:
            try:
                self.state.update(state_module.MEMORY,
                                  documents=self.documents.status())
            except Exception as exc:  # noqa: BLE001
                log.debug("belge durumu okunamadı: %s", exc)
        if self.core is not None:
            self._safely(state_module.AUTONOMY, self.core.status)
        if self.reminders is not None:
            try:
                self.state.update(state_module.AUTONOMY,
                                  reminders=self.reminders.status())
            except Exception as exc:  # noqa: BLE001
                log.debug("hatırlatıcı durumu okunamadı: %s", exc)
        if self.agents is not None:
            self._safely(state_module.AGENTS, self.agents.status)
        if self.research is not None:
            self._safely(state_module.RESEARCH, self.research.status)
        if self.lab is not None:
            self._safely(state_module.LAB, self.lab.status)
        if self.improve is not None:
            self._safely(state_module.IMPROVE, self.improve.status)
        return self.state.snapshot()

    def _resources(self) -> dict[str, Any]:
        """CPU, RAM, GPU and idle time, for anything watching the machine.

        Lives on the monitor rather than in any status() call, so it is polled
        here instead of being invented by the projector. Unknown readings stay
        None: a gauge showing zero because nothing was measured is a lie a bar
        chart tells very convincingly.
        """
        snapshot = self.core.snapshot()
        return {"resources": {
            "cpu": snapshot.cpu_percent,
            "ram": snapshot.ram_percent,
            "gpu": snapshot.gpu_percent,
            "vram_used_mb": snapshot.vram_used_mb,
            "vram_total_mb": snapshot.vram_total_mb,
            "idle_s": snapshot.idle_seconds,
            "taken": snapshot.taken,
        }}

    def _brain_status(self) -> dict[str, Any]:
        status = self.brain.status()
        verdict = status.pop("budget_verdict", None)
        if verdict is not None:
            status["budget_allowed"] = verdict.allowed
            status["budget_reason"] = verdict.reason
        return status

    def _safely(self, section: str, getter) -> None:
        try:
            payload = getter()
        except Exception as exc:  # noqa: BLE001 - a broken status must not stop the rest
            log.debug("%s durumu okunamadı: %s", section, exc)
            self.state.update(section, error=str(exc)[:200])
            return
        self.state.update(section, **_plain(payload))


def _plain(value: Any) -> dict[str, Any]:
    """Flatten a subsystem's status into something JSON can carry."""
    if not isinstance(value, dict):
        return {"value": str(value)}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[key] = item
        elif isinstance(item, dict):
            out[key] = {k: (v if isinstance(v, (str, int, float, bool)) or v is None
                            else str(v)) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            out[key] = [v if isinstance(v, (str, int, float, bool)) else str(v)
                        for v in item]
        else:
            out[key] = str(item)
    return out


class EventProjector:
    """Turns "what happened" into "what is happening".

    Deliberately a lookup table rather than logic. A projector that reasons about
    events is a second place where the system decides what it is doing, and the two
    will disagree.
    """

    #: (source, kind prefix) → activity while that is in progress.
    STARTS = {
        ("agent", "run.start"): state_module.THINKING,
        ("agent", "start"): state_module.THINKING,
        ("research", "start"): state_module.RESEARCHING,
        ("task", "start"): state_module.AUTONOMOUS,
        ("lab", "experiment.opened"): state_module.EXPERIMENTING,
        ("improve", "experiment.started"): state_module.EXPERIMENTING,
        ("improve", "hypothesis.new"): state_module.THINKING,
    }
    ENDS = {
        ("agent", "run.done"), ("agent", "done"), ("research", "done"),
        ("research", "error"), ("research", "empty"), ("task", "done"),
        ("task", "error"), ("improve", "improvement.promoted"),
        ("improve", "improvement.failed"), ("lab", "experiment.discarded"),
    }

    def __init__(self, state: SharedState) -> None:
        self.state = state
        self._depth = 0

    def __call__(self, event) -> None:
        try:
            self._project(event)
        except Exception as exc:  # noqa: BLE001 - projection must not break publishing
            log.debug("olay yansıtılamadı: %s", exc)

    def _project(self, event) -> None:
        key = (event.source, event.kind)
        section = _SECTION_BY_SOURCE.get(event.source)

        if section is not None:
            self.state.update(section, last_event=event.kind,
                              last_message=event.message[:300], last_event_at=event.ts)

        if event.level == "error":
            self.state.update(state_module.SYSTEM, last_error=event.message[:300],
                              last_error_at=event.ts)

        if key in self.STARTS:
            self._depth += 1
            self.state.set_activity(self.STARTS[key], event.message[:120])
        elif key in self.ENDS:
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                self.state.set_activity(state_module.IDLE)

        if event.source == "policy" and event.kind == "stance":
            self.state.update(state_module.AUTONOMY,
                              mode=event.data.get("mode", ""), reason=event.message)


_SECTION_BY_SOURCE = {
    "scheduler": state_module.AUTONOMY,
    "task": state_module.AUTONOMY,
    "policy": state_module.AUTONOMY,
    "system": state_module.SYSTEM,
    "agent": state_module.AGENTS,
    "research": state_module.RESEARCH,
    "lab": state_module.LAB,
    "improve": state_module.IMPROVE,
}


def build(
    config: Config | None = None,
    *,
    with_memory: bool = True,
    with_autonomy: bool = True,
    with_agents: bool = True,
    with_research: bool = True,
    with_lab: bool = True,
    with_improve: bool = True,
) -> Runtime:
    """Assemble the system. Never raises: what fails is recorded as a warning."""
    from .brain import Brain

    config = config or Config.load()
    runtime = Runtime(config=config, state=SharedState())
    runtime.state.set_activity(state_module.IDLE, "başlatılıyor")

    from .documents import DocumentLibrary
    runtime.documents = _try(runtime, "belge kütüphanesi", lambda: DocumentLibrary(
        config.path("paths.documents", "data/belgeler")))
    from .reminders import ReminderService
    runtime.reminders = _try(runtime, "hatırlatma servisi", lambda: ReminderService(
        config.path("paths.db", "data/jarvis.db")))

    if with_memory and config.get("memory.enabled", True):
        runtime.memory = _try(runtime, "hafıza", lambda: _build_memory(config))

    runtime.brain = Brain(config, memory=runtime.memory)

    # Memory keeps its own model clients so distilling survives whatever the
    # conversation is doing, but there is only one ledger and both belong in it.
    if runtime.memory is not None:
        for client in (getattr(runtime.memory, "local", None),
                       getattr(runtime.memory, "embedder", None)):
            if client is not None:
                client.usage = runtime.brain.budget.record

    if with_autonomy:
        from .autonomy import AutonomyCore
        runtime.core = _try(runtime, "otonomi", lambda: AutonomyCore(
            config, memory=runtime.memory, brain=runtime.brain))

    # Agents, research and the lab all report through the event log, which lives
    # with autonomy. Without it there is nowhere for a run to say what it is doing.
    if runtime.core is not None:
        if with_agents:
            from .agents import AgentSystem
            runtime.agents = _try(runtime, "ajan sistemi", lambda: AgentSystem(
                config, runtime.brain, runtime.core.events, memory=runtime.memory))
        if with_research:
            from .research import ResearchSystem
            runtime.research = _try(runtime, "araştırma", lambda: ResearchSystem(
                config, runtime.brain, runtime.core.events, memory=runtime.memory))
            if runtime.agents is not None and runtime.research is not None:
                runtime.agents.provide_research(runtime.research)
        if with_lab:
            from .lab import Lab, register_runner as register_lab_runner
            runtime.lab = _try(runtime, "laboratuvar", lambda: Lab(
                config, events=runtime.core.events))
            if runtime.lab is not None:
                register_lab_runner(runtime.lab)
        if with_improve and runtime.lab is not None:
            from .improve import build as build_improve
            runtime.improve = _try(runtime, "iyileştirme motoru", lambda: build_improve(
                config, runtime.brain, lab=runtime.lab, memory=runtime.memory,
                events=runtime.core.events))

        runtime._unsubscribe = runtime.core.events.subscribe(EventProjector(runtime.state))

    runtime.refresh()
    return runtime


def _build_memory(config: Config):
    from .brain.local import LocalBrain
    from .memory import Memory

    # Memory keeps its own model client: distilling and summarising must keep
    # working when the conversation is routed elsewhere, and must never reach the
    # metered tier.
    local = LocalBrain(
        host=config.get("local.host"),
        model=config.get("memory.model") or config.get("local.model"),
        timeout_s=config.get("local.timeout_s", 300),
        keep_alive=config.get("local.keep_alive", "30m"),
        think=False,
        context_window=config.get("local.context_window", 8192),
    )
    return Memory(config, local)


def _try(runtime: Runtime, label: str, factory):
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - one dead subsystem is not a dead system
        message = f"{label} açılamadı: {exc}"
        log.warning(message)
        runtime.warnings.append(message)
        return None


__all__ = ["Runtime", "build", "EventProjector", "SharedState"]
