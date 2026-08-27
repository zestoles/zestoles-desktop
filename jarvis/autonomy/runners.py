"""What a task actually does.

A runner is a plain function from a context to a result string. Registration is by
name, and the name is what the queue stores, so tasks queued by one build are still
runnable by the next — and a runner that has been removed leaves its tasks failing
loudly rather than vanishing.

Every runner must honour `ctx.should_stop()`. Threads cannot be killed safely in
Python, so stopping is cooperative: a runner that never checks will keep the
scheduler from shutting down until it finishes on its own. Long loops check between
units of work.

S2 ships only what the system can already do by itself. Research, tool discovery
and self-improvement runners arrive with the phases that build them.
"""

from __future__ import annotations

import logging
import platform
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .events import EventLog
from .tasks import Task

log = logging.getLogger("jarvis.autonomy.runners")


@dataclass(slots=True)
class RunContext:
    task: Task
    events: EventLog
    should_stop: Callable[[], bool]
    memory: Any = None
    brain: Any = None
    config: Any = None
    queue: Any = None
    #: The user-facing assistant service, when this process has one. Opaque on
    #: purpose: autonomy sits below the assistant and must not import it. Only
    #: a task the user queued may use it -- see jarvis/assistant/background.py.
    assistant: Any = None

    def progress(self, message: str, **data: Any) -> None:
        self.events.publish("task", "progress", message,
                            data={"task_id": self.task.id, **data})


Runner = Callable[[RunContext], str]
REGISTRY: dict[str, Runner] = {}

#: Runners that exist to exercise the machinery rather than to do work. They
#: behave the way they do by design, so their outcomes are not evidence about
#: the system and must not be read as symptoms of one.
#:
#: `fail` is why this exists. It raises every time on purpose, so the task queued
#: to test the retry path quarantined itself and stayed quarantined — and
#: GapDetector._from_tasks reads quarantined kinds as "this keeps being given up
#: on". FROM_TASKS is in COMPARABLE_SOURCES, so that gap was experiment-shaped
#: and could spend a hypothesis and one of four daily experiment slots on a
#: fixture that is working exactly as written.
FIXTURE_KINDS = frozenset({"fail"})


def runner(name: str) -> Callable[[Runner], Runner]:
    def register(func: Runner) -> Runner:
        if name in REGISTRY:
            raise ValueError(f"runner adı zaten kayıtlı: {name}")
        REGISTRY[name] = func
        return func

    return register


def get(name: str) -> Runner | None:
    return REGISTRY.get(name)


def names() -> list[str]:
    return sorted(REGISTRY)


# --------------------------------------------------------------- built-ins
@runner("noop")
def _noop(ctx: RunContext) -> str:
    """Does nothing successfully. Used to prove the pipeline end to end."""
    return ctx.task.payload.get("message", "tamam")


@runner("fail")
def _fail(ctx: RunContext) -> str:
    """Fails on purpose, so retry and quarantine can be exercised for real."""
    raise RuntimeError(ctx.task.payload.get("message", "kasıtlı hata"))


@runner("memory.reindex")
def _reindex(ctx: RunContext) -> str:
    """Bring the memory index in line with the vault.

    Worth doing unattended: the vault is meant to be edited by hand, and an edit
    made in Obsidian is invisible to recall until the index catches up.
    """
    if ctx.memory is None:
        raise RuntimeError("hafıza bu süreçte açık değil")
    report = ctx.memory.reindex()
    if not report:
        return "indeks güncellenemedi"
    changed = report.get("eklendi", 0) + report.get("guncellendi", 0) + report.get("silindi", 0)
    detail = " · ".join(f"{k}: {v}" for k, v in report.items())
    return f"{'değişiklik yok' if not changed else detail}"


@runner("system.snapshot")
def _snapshot(ctx: RunContext) -> str:
    """Record what the machine looked like. The raw material for later analysis."""
    from .resources import ResourceMonitor

    snapshot = ResourceMonitor().snapshot()
    ctx.events.publish("system", "snapshot", snapshot.summary(),
                       data={"cpu": snapshot.cpu_percent, "ram": snapshot.ram_percent,
                             "gpu": snapshot.gpu_percent, "idle": snapshot.idle_seconds})
    return snapshot.summary()


@runner("system.selftest")
def _selftest(ctx: RunContext) -> str:
    """Check that the pieces JARVIS depends on are still answering."""
    lines = [f"python {platform.python_version()} · {platform.system()}"]

    if ctx.brain is not None:
        local_up = ctx.brain.local.available()
        lines.append(f"yerel model {'ayakta' if local_up else 'ERİŞİLEMİYOR'}"
                     f" ({ctx.brain.local.model})")
        if not local_up:
            raise RuntimeError("yerel model yanıt vermiyor — otonom çalışma anlamsız")

    if ctx.memory is not None:
        stats = ctx.memory.stats()
        lines.append(f"hafıza {stats.get('notlar', 0)} not · "
                     f"{stats.get('vektorlu', 0)} vektör")
        if not ctx.memory.embedder.available():
            lines.append("UYARI: embedding modeli yanıt vermiyor")

    return " | ".join(lines)


@runner("tasks.purge")
def _purge(ctx: RunContext) -> str:
    """Drop long-finished tasks so the queue table does not grow without bound."""
    if ctx.queue is None:
        raise RuntimeError("kuyruk referansı verilmedi")
    days = float(ctx.task.payload.get("older_than_days", 30))
    return f"{ctx.queue.purge(older_than_days=days)} eski görev silindi"


@runner("events.purge")
def _purge_events(ctx: RunContext) -> str:
    """Age out the activity log.

    The queue had a purge and the event log did not, so the only table that grew
    unattended was the one nothing pruned. Warnings and errors keep a longer
    cutoff than routine activity — see EventLog.purge.
    """
    days = float(ctx.task.payload.get("older_than_days", 30))
    problem_days = float(ctx.task.payload.get("problem_older_than_days", 90))
    deleted = ctx.events.purge(older_than_days=days, problem_older_than_days=problem_days)
    return f"{deleted} eski olay silindi (rutin >{days:.0f}g, sorun >{problem_days:.0f}g)"
