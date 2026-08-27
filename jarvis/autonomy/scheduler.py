"""The loop that runs autonomous work.

One background thread, one task at a time by default. That is a deliberate choice
rather than a limitation: the whole system shares one GPU and one Ollama instance,
so running three "background" tasks at once would compete with itself and with the
user for the same resource the policy is trying to protect.

Shutdown is cooperative. Python cannot safely kill a thread, so stop() sets a flag,
the runner sees it through ctx.should_stop(), and the thread is joined with a
timeout. A runner that ignores the flag delays shutdown but cannot corrupt state —
the task simply stays RUNNING and is recovered on the next start.

Failure is contained per task. A runner that raises marks its own task failed and
the loop continues; a runner that raises every time exhausts its attempts and is
quarantined. Nothing a task does can stop the scheduler itself.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback

from . import runners
from .events import ERROR, EventLog, INFO, SUCCESS, WARN
from .policy import Policy, Stance
from .resources import ResourceMonitor
from .runners import RunContext
from .tasks import Priority, State, TaskQueue

log = logging.getLogger("jarvis.autonomy.scheduler")

TICK_SECONDS = 5.0
STANCE_LOG_INTERVAL = 300.0


class Scheduler:
    def __init__(
        self,
        queue: TaskQueue,
        events: EventLog,
        policy: Policy,
        *,
        memory=None,
        brain=None,
        config=None,
        assistant=None,
        tick_s: float = TICK_SECONDS,
        before_tick=None,
    ) -> None:
        self.queue = queue
        self.events = events
        self.policy = policy
        self.memory = memory
        self.brain = brain
        self.config = config
        #: Set once the interface is up; None in a headless daemon. The queue
        #: reaches it only through a task the user queued themselves.
        self.assistant = assistant
        self.tick_s = tick_s
        #: Called at the top of every tick, before anything is claimed. This is
        #: how recurring work gets queued: the scheduler stays a thing that runs
        #: what is in the queue, and what belongs in the queue stays with the
        #: layer that owns the routines.
        self.before_tick = before_tick

        self.monitor = ResourceMonitor()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._wake = threading.Event()
        self._current: str | None = None
        self._stance: Stance | None = None
        self._stance_key: tuple[str, str] | None = None
        self._last_stance_log = 0.0
        self._ran = 0
        self._failed = 0

    # ------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def current(self) -> str | None:
        return self._current

    @property
    def stance(self) -> Stance | None:
        return self._stance

    def start(self) -> None:
        if self.running:
            return
        recovered = self.queue.recover_orphans()
        if recovered:
            self.events.publish("scheduler", "recover",
                                f"{recovered} yarım kalmış görev kuyruğa geri kondu",
                                level=WARN)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="jarvis-autonomy", daemon=True)
        self._thread.start()
        self.events.publish("scheduler", "start", "otonom döngü başladı")

    def stop(self, *, timeout: float = 30.0) -> bool:
        """Ask the loop to finish. Returns False if a runner outlived the timeout."""
        if not self.running:
            return True
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)
        stopped = not self._thread.is_alive()
        self.events.publish(
            "scheduler", "stop",
            "otonom döngü durdu" if stopped else "döngü durmadı — bir görev hâlâ çalışıyor",
            level=INFO if stopped else WARN,
        )
        return stopped

    def pause(self) -> None:
        self._paused.set()
        self.events.publish("scheduler", "pause", "otonom çalışma duraklatıldı")

    def resume(self) -> None:
        self._paused.clear()
        self._wake.set()
        self.events.publish("scheduler", "resume", "otonom çalışma sürüyor")

    def nudge(self) -> None:
        """Wake the loop early — used when a task is queued by hand."""
        self._wake.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    # ---------------------------------------------------------------- loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - the loop outlives every failure in it
                log.exception("zamanlayıcı döngüsünde beklenmeyen hata")
                self.events.publish("scheduler", "error",
                                    "döngüde beklenmeyen hata — devam ediliyor",
                                    level=ERROR, data={"trace": traceback.format_exc()[-800:]})
            self._wake.wait(self.tick_s)
            self._wake.clear()
        self._current = None

    def _tick(self) -> None:
        if self._paused.is_set():
            return

        if self.before_tick is not None:
            try:
                self.before_tick()
            except Exception as exc:  # noqa: BLE001 - queueing must not kill the loop
                log.warning("tick öncesi kanca hata verdi: %s", exc)
                self.events.publish("scheduler", "error",
                                    f"rutin kuyruklama başarısız: {exc}", level=ERROR)

        stance = self.policy.evaluate(self.monitor.snapshot())
        self._stance = stance
        self._log_stance(stance)

        if not stance.autonomous_allowed:
            # User-priority work still runs; the machine being busy is not a reason
            # to ignore something the user asked for directly.
            self._drain(Priority.USER)
            return

        self._drain(stance.max_priority, limit=stance.concurrency)

    def _drain(self, max_priority: int, *, limit: int = 1) -> None:
        for _ in range(max(1, limit)):
            if self._stop.is_set():
                return
            task = self.queue.claim(max_priority=max_priority)
            if task is None:
                return
            self._execute(task)

    def _execute(self, task) -> None:
        self._current = task.label
        started = time.monotonic()
        self.events.publish("task", "start", f"{task.title} başladı",
                            data={"task_id": task.id, "kind": task.kind,
                                  "attempt": task.attempts})

        handler = runners.get(task.kind)
        if handler is None:
            self.queue.fail(task.id, f"bilinmeyen görev türü: {task.kind}")
            self.events.publish("task", "error", f"{task.title}: bilinmeyen tür {task.kind}",
                                level=ERROR, data={"task_id": task.id})
            self._current = None
            return

        context = RunContext(
            task=task, events=self.events, should_stop=self.should_stop,
            memory=self.memory, brain=self.brain, config=self.config, queue=self.queue,
            assistant=self.assistant,
        )

        try:
            result = handler(context) or ""
        except Exception as exc:  # noqa: BLE001 - one task must not stop the loop
            self._failed += 1
            detail = f"{type(exc).__name__}: {exc}"
            outcome = self.queue.fail(task.id, detail)
            log.warning("görev başarısız (%s): %s", task.label, detail)
            self.events.publish(
                "task",
                "quarantine" if outcome == State.QUARANTINED else "retry",
                f"{task.title} başarısız — "
                + ("karantinaya alındı" if outcome == State.QUARANTINED else "tekrar denenecek")
                + f": {detail}",
                level=ERROR if outcome == State.QUARANTINED else WARN,
                data={"task_id": task.id, "attempt": task.attempts,
                      "trace": traceback.format_exc()[-800:]},
            )
        else:
            self._ran += 1
            elapsed = time.monotonic() - started
            self.queue.complete(task.id, result)
            self.events.publish("task", "done", f"{task.title}: {result}"[:300],
                                level=SUCCESS,
                                data={"task_id": task.id, "seconds": round(elapsed, 2)})
        finally:
            self._current = None

    def _log_stance(self, stance: Stance) -> None:
        """Publish the stance on change, or occasionally while it holds.

        Without the interval a quiet night writes one identical event per tick;
        without the change detection the moment the user sits down goes unrecorded.

        The comparison is on `stance.key`, not on the message. It used to be on
        `(mode, reason)`, and the reason carries live numbers — "kullanıcı aktif
        (25s önce girdi)" differs every tick — so the interval never applied and
        an unattended night wrote one row every five seconds. Measured in the S9
        soak: 522 of 529 events were stance records that said nothing new.
        """
        now = time.monotonic()
        changed = self._stance_key != stance.key
        if changed or now - self._last_stance_log > STANCE_LOG_INTERVAL:
            self._last_stance_log = now
            self._stance_key = stance.key
            self.events.publish("policy", "stance", f"{stance.mode}: {stance.reason}",
                                data={"mode": stance.mode, "cause": stance.cause,
                                      "max_priority": stance.max_priority})

    # -------------------------------------------------------------- status
    def status(self) -> dict[str, object]:
        counts = self.queue.counts()
        return {
            "calisiyor": self.running,
            "duraklatildi": self.paused,
            "mod": self._stance.mode if self._stance else "—",
            "gerekce": self._stance.reason if self._stance else "henüz ölçüm yok",
            "aktif_gorev": self._current,
            "tamamlanan": self._ran,
            "basarisiz": self._failed,
            "kuyruk": counts,
        }
