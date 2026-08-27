"""Autonomy, assembled: queue + policy + scheduler + activity log behind one object.

What S2 delivers is the machinery, not the ambition. JARVIS can now hold work,
decide when it is polite to do it, do it, survive failing at it, and remember that
it happened — across restarts. What it *does* with that machinery arrives in the
phases that follow: agents in S3, research in S4, self-improvement in S5.

Building it in this order is deliberate. A research loop without a scheduler is a
process that hammers the GPU while its owner is trying to work, and a
self-improvement loop without persistent task state cannot be interrupted safely.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from ..config import Config
from . import runners
from .events import EventLog, Event
from .policy import Policy, Stance
from .resources import ResourceMonitor, Snapshot
from .runners import RunContext
from .scheduler import Scheduler
from .tasks import Priority, State, Task, TaskQueue

log = logging.getLogger("jarvis.autonomy")

HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class Routine:
    """Work that should happen again, not work that happened once.

    The S9 soak measured what the previous version of this actually did: three
    upkeep tasks in the first minute and then five and a half hours of an empty
    queue. Queueing at startup answers "what should run when JARVIS comes up"; it
    was never an answer to "what should run tonight".

    Due-ness is derived from the task table rather than tracked in a new one. The
    queue already records when each kind last ran and already refuses a duplicate
    while one is pending, so a restart, a crash mid-task or a machine that stayed
    busy for six hours all resolve to the same question: has it been long enough
    since the last one.
    """

    kind: str
    title: str
    priority: int
    interval_s: float
    #: First run is delayed by this much after startup, so a machine that just
    #: booted is not met with a queue of work while it is still settling.
    initial_delay_s: float = 0.0
    payload: dict | None = None
    max_attempts: int = 3

    @property
    def dedupe_key(self) -> str:
        return f"routine:{self.kind}"


#: Recurring upkeep and, at night, real work.
#:
#: `improve.cycle` sits at IDLE_ONLY on purpose: that priority is admitted only by
#: the NIGHT stance, so a full self-improvement pass — research, hypothesis,
#: experiment, benchmark, promotion gate — can only start while the machine is
#: both idle and inside the night window. It goes through the same engine, the
#: same budget and the same gates as a cycle started by hand; being queued by a
#: routine grants it nothing.
ROUTINES = (
    Routine("system.selftest", "Sistem kendini kontrol ediyor",
            Priority.BACKGROUND, 6 * HOUR),
    Routine("memory.reindex", "Hafıza indeksi tazeleniyor",
            Priority.BACKGROUND, 6 * HOUR, initial_delay_s=60),
    Routine("tasks.purge", "Eski görevler temizleniyor",
            Priority.IDLE_ONLY, 24 * HOUR, initial_delay_s=300),
    Routine("events.purge", "Eski olaylar temizleniyor",
            Priority.IDLE_ONLY, 24 * HOUR, initial_delay_s=300),
    Routine("lab.cleanup", "Deney alanı temizleniyor",
            Priority.BACKGROUND, 24 * HOUR, initial_delay_s=300),
    Routine("improve.cycle", "Gece iyileştirme turu",
            Priority.IDLE_ONLY, 1.5 * HOUR, initial_delay_s=120),
)


def _configured_routines(config: Config) -> tuple[Routine, ...]:
    """Apply configuration to the built-in routines.

    Intervals and retention are the two things worth changing without editing
    code; the set of routines is not, because each one is a runner that has to
    exist. A malformed override is ignored with a warning rather than silently
    turning a nightly job into one that runs every second.
    """
    overrides = config.get("autonomy.routine_intervals", {}) or {}
    disabled = set(config.get("autonomy.routines_disabled", []) or [])
    payloads = {
        "tasks.purge": {"older_than_days": config.get("autonomy.task_retention_days", 30)},
        "events.purge": {
            "older_than_days": config.get("autonomy.event_retention_days", 30),
            "problem_older_than_days":
                config.get("autonomy.event_problem_retention_days", 90),
        },
        "lab.cleanup": {
            "keep_sandboxes": config.get("lab.keep_sandboxes", 5),
            "keep_snapshots": config.get("lab.keep_snapshots", 10),
        },
    }

    out: list[Routine] = []
    for routine in ROUTINES:
        if routine.kind in disabled:
            log.info("rutin yapılandırmada kapalı: %s", routine.kind)
            continue
        interval = routine.interval_s
        if routine.kind in overrides:
            try:
                candidate = float(overrides[routine.kind])
            except (TypeError, ValueError):
                log.warning("rutin aralığı sayı değil, yok sayıldı: %s=%r",
                            routine.kind, overrides[routine.kind])
            else:
                if candidate < 60:
                    log.warning("rutin aralığı 60 saniyenin altında olamaz: %s=%s",
                                routine.kind, candidate)
                else:
                    interval = candidate
        out.append(replace(routine, interval_s=interval,
                           payload=payloads.get(routine.kind, routine.payload)))
    return tuple(out)


class AutonomyCore:
    def __init__(self, config: Config, *, memory=None, brain=None) -> None:
        self.config = config
        self.enabled = bool(config.get("autonomy.enabled", True))
        db = config.path("paths.db", "data/jarvis.db")

        self.queue = TaskQueue(db)
        self.events = EventLog(db)
        self.policy = Policy(
            idle_after_s=config.get("autonomy.idle_after_s", 300),
            night_hours=tuple(config.get("budget.night_hours", [1, 8])),
            cpu_ceiling=config.get("autonomy.cpu_ceiling", 65.0),
            gpu_ceiling=config.get("autonomy.gpu_ceiling", 55.0),
            ram_ceiling=config.get("autonomy.ram_ceiling", 88.0),
            night_concurrency=config.get("autonomy.night_concurrency", 2),
            idle_concurrency=config.get("autonomy.idle_concurrency", 1),
        )
        self.routines = _configured_routines(config)
        self._started_at = time.time()
        self.scheduler = Scheduler(
            self.queue, self.events, self.policy,
            memory=memory, brain=brain, config=config,
            tick_s=config.get("autonomy.tick_s", 5.0),
            before_tick=self.queue_due_routines,
        )

    # ------------------------------------------------------------ lifecycle
    def start(self, *, queue_routine: bool = True) -> None:
        if not self.enabled:
            log.info("otonomi yapılandırmada kapalı")
            return
        self._started_at = time.time()
        if queue_routine:
            self.queue_due_routines()
        self.scheduler.start()

    def stop(self, *, timeout: float = 30.0) -> bool:
        return self.scheduler.stop(timeout=timeout)

    def pause(self) -> None:
        self.scheduler.pause()

    def resume(self) -> None:
        self.scheduler.resume()

    # ---------------------------------------------------------------- work
    def queue_due_routines(self, *, now: float | None = None) -> int:
        """Queue every routine whose interval has elapsed. Returns how many.

        Called from the scheduler tick, so this runs every few seconds and must
        stay cheap and quiet: it asks the queue when each kind last ran and adds
        nothing in the usual case.

        A routine whose runner is not registered is skipped rather than queued.
        Without that, a build with the improvement engine switched off would fill
        the queue with tasks that fail three times each and quarantine themselves
        — noise that looks exactly like a broken system.
        """
        now = time.time() if now is None else now
        queued = 0
        waiting = self.queue.waiting_dedupe_keys()
        for routine in self.routines:
            if runners.get(routine.kind) is None:
                continue
            if routine.dedupe_key in waiting:
                # Already queued or running: the dedupe index would refuse it.
                # last_run() falls back to `created` for a task that never ran,
                # which is older than any interval, so a routine waiting out a
                # busy machine took the insert-and-be-rejected path every tick —
                # 2,365 of 2,513 log lines on the 17.08 run. Same queue outcome,
                # the refusal just happens before the insert instead of behind it.
                continue
            last = self.queue.last_run(routine.kind, dedupe_key=routine.dedupe_key)
            if last:
                if now - last < routine.interval_s:
                    continue
            elif now - self._started_at < routine.initial_delay_s:
                # Never run before, and still inside the settling delay.
                continue
            task_id = self.queue.add(
                routine.kind, routine.title, payload=dict(routine.payload or {}),
                priority=routine.priority, origin="routine",
                max_attempts=routine.max_attempts, dedupe_key=routine.dedupe_key,
            )
            if task_id is not None:
                queued += 1
                log.debug("rutin kuyruğa alındı: %s (#%s)", routine.kind, task_id)
        if queued:
            self.scheduler.nudge()
        return queued

    def submit(
        self,
        kind: str,
        title: str,
        *,
        payload: dict | None = None,
        priority: int = Priority.NORMAL,
        origin: str = "user",
    ) -> int | None:
        if runners.get(kind) is None:
            raise ValueError(f"bilinmeyen görev türü: {kind} (mevcut: {', '.join(runners.names())})")
        task_id = self.queue.add(kind, title, payload=payload, priority=priority, origin=origin)
        if task_id is not None:
            self.events.publish("task", "queued", f"{title} kuyruğa alındı",
                                data={"task_id": task_id, "kind": kind})
            self.scheduler.nudge()
        return task_id

    def cancel(self, task_id: int) -> bool:
        cancelled = self.queue.cancel(task_id)
        if cancelled:
            self.events.publish("task", "cancelled", f"#{task_id} iptal edildi")
        return cancelled

    # -------------------------------------------------------------- status
    def snapshot(self) -> Snapshot:
        return self.scheduler.monitor.snapshot()

    def status(self) -> dict[str, object]:
        data = self.scheduler.status()
        data["acik"] = self.enabled
        data["kosucular"] = runners.names()
        data["rutinler"] = self.routine_status()
        return data

    def routine_status(self) -> dict[str, str]:
        """When each routine last ran and when it is next due.

        A routine nobody can see is a routine nobody notices has stopped.
        """
        now = time.time()
        out: dict[str, str] = {}
        for routine in self.routines:
            if runners.get(routine.kind) is None:
                out[routine.kind] = "çalıştırıcı kayıtlı değil"
                continue
            last = self.queue.last_run(routine.kind, dedupe_key=routine.dedupe_key)
            if not last:
                out[routine.kind] = f"hiç koşmadı · her {routine.interval_s / 3600:.1f} sa"
                continue
            due_in = (last + routine.interval_s) - now
            out[routine.kind] = (
                f"{(now - last) / 3600:.1f} sa önce · "
                + ("şimdi hazır" if due_in <= 0 else f"{due_in / 3600:.1f} sa sonra")
            )
        return out


__all__ = [
    "AutonomyCore", "TaskQueue", "Task", "Priority", "State", "Policy", "Stance",
    "Scheduler", "EventLog", "Event", "ResourceMonitor", "Snapshot", "RunContext",
    "runners", "Routine", "ROUTINES",
]
