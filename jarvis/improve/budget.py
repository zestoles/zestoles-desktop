"""Limits on how much the system may spend improving itself.

An improvement loop with no ceiling is a loop that runs until something breaks.
Left alone it will research the same topic nightly, queue an experiment for every
gap it can name, and fill the disk with sandboxes — not through malice but because
nothing told it to stop.

Three ceilings, all counted from what actually happened rather than from intent:

  per day        how many of each kind of activity may start in 24 hours
  per night      a tighter ceiling while the owner is asleep and cannot intervene
  concurrent     one improvement experiment at a time; they share one GPU

Every allowance is checked against the recorded history, so a restart does not
reset the count. That matters: a crash loop would otherwise be a way around
every limit here.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger("jarvis.improve.budget")

RESEARCH = "arastirma"
HYPOTHESIS = "hipotez"
EXPERIMENT = "deney"

ACTIVITIES = (RESEARCH, HYPOTHESIS, EXPERIMENT)

SCHEMA = """
CREATE TABLE IF NOT EXISTS improvement_activity (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    activity TEXT NOT NULL,
    subject  TEXT NOT NULL DEFAULT '',
    detail   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_improvement_activity ON improvement_activity(activity, ts);
"""

DEFAULT_DAILY = {RESEARCH: 6, HYPOTHESIS: 8, EXPERIMENT: 4}
DEFAULT_NIGHTLY = {RESEARCH: 4, HYPOTHESIS: 5, EXPERIMENT: 3}


@dataclass(slots=True)
class Allowance:
    allowed: bool
    reason: str
    used: int = 0
    limit: int = 0

    def summary(self) -> str:
        return f"{self.reason} ({self.used}/{self.limit})"


class ImprovementBudget:
    def __init__(self, db_path: Path, *, daily: dict[str, int] | None = None,
                 nightly: dict[str, int] | None = None,
                 night_hours: tuple[int, int] = (1, 8)) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily = {**DEFAULT_DAILY, **(daily or {})}
        self.nightly = {**DEFAULT_NIGHTLY, **(nightly or {})}
        self.night_hours = tuple(night_hours)
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def is_night(self, at: datetime | None = None) -> bool:
        hour = (at or datetime.now()).hour
        start, end = self.night_hours
        return start <= hour < end if start < end else (hour >= start or hour < end)

    def used(self, activity: str, *, since_seconds: float = 86400) -> int:
        cutoff = time.time() - since_seconds
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) c FROM improvement_activity WHERE activity=? AND ts>=?",
                (activity, cutoff)).fetchone()
        return int(row["c"]) if row else 0

    def check(self, activity: str, *, at: datetime | None = None) -> Allowance:
        if activity not in ACTIVITIES:
            return Allowance(False, f"bilinmeyen etkinlik: {activity}")

        daily_used = self.used(activity)
        daily_limit = self.daily.get(activity, 0)
        if daily_used >= daily_limit:
            return Allowance(False, "günlük bütçe dolu", daily_used, daily_limit)

        if self.is_night(at):
            # The night window is tighter because nobody is watching. A loop that
            # goes wrong at 03:00 has five hours to go wrong repeatedly.
            night_used = self.used(activity, since_seconds=8 * 3600)
            night_limit = self.nightly.get(activity, 0)
            if night_used >= night_limit:
                return Allowance(False, "gece bütçesi dolu", night_used, night_limit)
            return Allowance(True, "gece bütçesi uygun", night_used, night_limit)

        return Allowance(True, "bütçe uygun", daily_used, daily_limit)

    def record(self, activity: str, *, subject: str = "", detail: str = "") -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO improvement_activity (ts, activity, subject, detail)"
                " VALUES (?,?,?,?)", (time.time(), activity, subject[:200], detail[:500]))

    def spend(self, activity: str, *, subject: str = "", at: datetime | None = None
              ) -> Allowance:
        """Check and record in one step. Nothing is recorded when refused."""
        allowance = self.check(activity, at=at)
        if allowance.allowed:
            self.record(activity, subject=subject)
        else:
            log.info("iyileştirme bütçesi reddetti (%s): %s", activity, allowance.summary())
        return allowance

    def snapshot(self) -> dict[str, str]:
        return {
            activity: f"{self.used(activity)}/{self.daily.get(activity, 0)}"
            for activity in ACTIVITIES
        }
