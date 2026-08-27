"""Periodic refresh, so a watcher sees a machine that is alive rather than a
snapshot frozen at connect time.

Most of what the interface shows arrives as events, because most of it changes
when something happens. Resource readings do not: CPU is a number that is always
true and always different, and nothing publishes an event when it moves.

So one thread polls. Two rules keep it from being a cost:

**It sleeps when nobody is watching.** With no subscribers the pump does not poll,
does not measure and does not publish. A JARVIS running headless overnight pays
nothing for an interface that is not open.

**It never blocks anyone.** A refresh that fails is logged and skipped; the next
tick tries again. The pump has no way to affect the scheduler, an agent run, or
anything else the system is doing — it reads and publishes, and that is all.
"""

from __future__ import annotations

import logging
import threading
import time

from .bus import EventBus
from .types import SYSTEM_STATE_CHANGED

log = logging.getLogger("jarvis.bus.telemetry")

DEFAULT_INTERVAL_S = 3.0


class TelemetryPump:
    def __init__(self, runtime, bus: EventBus, *,
                 interval_s: float = DEFAULT_INTERVAL_S) -> None:
        self.runtime = runtime
        self.bus = bus
        self.interval_s = max(1.0, float(interval_s))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.ticks = 0
        self.skipped = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="jarvis-telemetry")
        self._thread.start()
        log.debug("telemetri döngüsü başladı (%.1fs)", self.interval_s)

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def tick(self) -> bool:
        """One measurement. Returns False when it was skipped or failed."""
        if self.bus.subscriber_count == 0:
            self.skipped += 1
            return False
        try:
            self.runtime.refresh()
            system = self.runtime.state.get("system")
        except Exception as exc:  # noqa: BLE001 - a failed tick is not a failed system
            log.debug("telemetri yenilenemedi: %s", exc)
            self.skipped += 1
            return False

        self.bus.publish(SYSTEM_STATE_CHANGED, {
            "activity": self.runtime.state.activity,
            "resources": system.get("resources", {}),
            "uptime_s": system.get("uptime_s"),
            "telemetry": True,
        })
        self.ticks += 1
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self.interval_s)

    def status(self) -> dict[str, object]:
        return {"calisiyor": self.running, "aralik_s": self.interval_s,
                "olcum": self.ticks, "atlanan": self.skipped}
