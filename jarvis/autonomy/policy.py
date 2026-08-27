"""When JARVIS is allowed to work on its own, and how hard.

The rule the whole file exists to serve: the user's machine belongs to the user. Anything
autonomous yields — to his input, to his games, to his builds. A background task
that is merely delayed costs nothing; a background task that makes the machine
stutter costs trust, and trust is what buys the system permission to run at all.

Four stances, from most to least restrictive:

  ACTIVE   he is at the keyboard — user work only
  BUSY     the machine is loaded even if he is away — nothing autonomous
  IDLE     no input for a while, machine quiet — ordinary background work
  NIGHT    night hours and idle — long work, still capped, never the paid tier

Unknown readings are treated as BUSY. A monitor that cannot see the machine is not
evidence that the machine is free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .resources import Snapshot
from .tasks import Priority

ACTIVE = "active"
BUSY = "busy"
IDLE = "idle"
NIGHT = "night"


#: Why a stance was taken, as a token that does not change while the situation
#: does not. `reason` is written for a human and carries live numbers — "kullanıcı
#: aktif (25s önce girdi)" is a different string every tick — so anything that
#: wants to know "has the situation changed?" must compare this instead.
USER_ACTIVE = "user_active"
UNREADABLE = "unreadable"
CPU = "cpu"
GPU = "gpu"
RAM = "ram"
QUIET = "quiet"


@dataclass(slots=True)
class Stance:
    mode: str
    max_priority: int
    concurrency: int
    reason: str
    cause: str = QUIET

    @property
    def autonomous_allowed(self) -> bool:
        return self.max_priority >= Priority.BACKGROUND

    def admits(self, priority: int) -> bool:
        return priority <= self.max_priority

    @property
    def key(self) -> tuple[str, str]:
        """What must change before the stance is worth recording again."""
        return (self.mode, self.cause)


class Policy:
    def __init__(
        self,
        *,
        idle_after_s: int = 300,
        night_hours: tuple[int, int] = (1, 8),
        cpu_ceiling: float = 65.0,
        gpu_ceiling: float = 55.0,
        ram_ceiling: float = 88.0,
        night_concurrency: int = 2,
        idle_concurrency: int = 1,
    ) -> None:
        self.idle_after_s = idle_after_s
        self.night_hours = tuple(night_hours)
        self.cpu_ceiling = cpu_ceiling
        self.gpu_ceiling = gpu_ceiling
        self.ram_ceiling = ram_ceiling
        self.night_concurrency = night_concurrency
        self.idle_concurrency = idle_concurrency

    def is_night(self, at: datetime | None = None) -> bool:
        hour = (at or datetime.now()).hour
        start, end = self.night_hours
        return start <= hour < end if start < end else (hour >= start or hour < end)

    def evaluate(self, snapshot: Snapshot, *, at: datetime | None = None) -> Stance:
        # User work always runs; only autonomous work is ever held back.
        if not snapshot.known:
            return Stance(BUSY, Priority.USER, 1,
                          f"kaynak okunamadı ({snapshot.why_unknown()}) — "
                          f"otonom iş güvenli sayılmıyor", UNREADABLE)

        idle = snapshot.idle_seconds or 0.0
        if idle < self.idle_after_s:
            return Stance(ACTIVE, Priority.USER, 1,
                          f"kullanıcı aktif ({int(idle)}s önce girdi)", USER_ACTIVE)

        cause, pressure = self._pressure(snapshot)
        if pressure:
            return Stance(BUSY, Priority.USER, 1, f"makine meşgul — {pressure}", cause)

        if self.is_night(at):
            return Stance(NIGHT, Priority.IDLE_ONLY, self.night_concurrency,
                          f"gece ve boşta ({int(idle // 60)} dk)", QUIET)

        return Stance(IDLE, Priority.BACKGROUND, self.idle_concurrency,
                      f"boşta ({int(idle // 60)} dk)", QUIET)

    def _pressure(self, snapshot: Snapshot) -> tuple[str, str]:
        """Load coming from something other than us. Empty reason means clear."""
        if snapshot.cpu_percent is not None and snapshot.cpu_percent > self.cpu_ceiling:
            return CPU, f"CPU %{snapshot.cpu_percent:.0f} > %{self.cpu_ceiling:.0f}"
        if snapshot.gpu_percent is not None and snapshot.gpu_percent > self.gpu_ceiling:
            return GPU, f"GPU %{snapshot.gpu_percent:.0f} > %{self.gpu_ceiling:.0f}"
        if snapshot.ram_percent is not None and snapshot.ram_percent > self.ram_ceiling:
            return RAM, f"RAM %{snapshot.ram_percent:.0f} > %{self.ram_ceiling:.0f}"
        return QUIET, ""
