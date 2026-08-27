"""Metering for the paid tier.

The Claude subscription has a hard usage cap. JARVIS must never be the reason the
user hits it — least of all at 04:00 while they are asleep and cannot intervene.
Every Tier 2 call passes this gate first, and every call that happens is recorded
so that /durum can show where the allowance went.

Night hours get their own, tighter ceiling on top of the daily one.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("jarvis.brain.budget")

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    tier          TEXT    NOT NULL,
    model         TEXT,
    purpose       TEXT,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_tokens  INTEGER DEFAULT 0,
    cost_usd      REAL    DEFAULT 0,
    duration_ms   INTEGER DEFAULT 0,
    ok            INTEGER DEFAULT 1,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls(ts);
"""

HOUR = 3600.0
DAY = 86400.0


@dataclass(slots=True)
class Verdict:
    allowed: bool
    reason: str


class Budget:
    def __init__(
        self,
        db_path: Path,
        *,
        per_hour: int = 10,
        per_day: int = 40,
        per_night: int = 0,
        night_hours: tuple[int, int] = (1, 8),
        allow_at_night: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.per_hour = per_hour
        self.per_day = per_day
        self.per_night = per_night
        self.night_hours = tuple(night_hours)
        self.allow_at_night = allow_at_night
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------ gate
    def is_night(self, at: datetime | None = None) -> bool:
        hour = (at or datetime.now()).hour
        start, end = self.night_hours
        return start <= hour < end if start < end else (hour >= start or hour < end)

    def check_cloud(self) -> Verdict:
        now = time.time()
        if self.is_night() and not self.allow_at_night:
            return Verdict(False, "gece modu — otonom çalışma tamamen yerel")
        with closing(self._conn()) as conn:
            in_hour = self._count_since(conn, now - HOUR)
            in_day = self._count_since(conn, now - DAY)
            if self.is_night():
                in_night = self._count_since(conn, self._night_window_start(now))
                if in_night >= self.per_night:
                    return Verdict(False, f"gece kotası dolu ({in_night}/{self.per_night})")
        if in_hour >= self.per_hour:
            return Verdict(False, f"saatlik kota dolu ({in_hour}/{self.per_hour})")
        if in_day >= self.per_day:
            return Verdict(False, f"günlük kota dolu ({in_day}/{self.per_day})")
        return Verdict(True, f"kota uygun (saat {in_hour}/{self.per_hour}, gün {in_day}/{self.per_day})")

    def _night_window_start(self, now: float) -> float:
        """Timestamp at which the current night window began.

        Night usually straddles midnight, so the anchor may belong to yesterday.
        """
        moment = datetime.fromtimestamp(now)
        anchor = moment.replace(hour=self.night_hours[0], minute=0, second=0, microsecond=0)
        if anchor > moment:
            anchor -= timedelta(days=1)
        return anchor.timestamp()

    @staticmethod
    def _count_since(conn: sqlite3.Connection, since: float) -> int:
        row = conn.execute(
            "SELECT COUNT(*) FROM llm_calls WHERE tier='cloud' AND ok=1 AND ts >= ?", (since,)
        ).fetchone()
        return int(row[0]) if row else 0

    # --------------------------------------------------------------- writing
    def record(
        self,
        *,
        tier: str,
        model: str | None,
        purpose: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        ok: bool = True,
        error: str | None = None,
    ) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO llm_calls (ts, tier, model, purpose, input_tokens, output_tokens,"
                " cache_tokens, cost_usd, duration_ms, ok, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), tier, model, purpose, input_tokens, output_tokens,
                 cache_tokens, cost_usd, duration_ms, int(ok), error),
            )

    # --------------------------------------------------------------- reading
    def usage(self) -> dict[str, float | int]:
        now = time.time()
        with closing(self._conn()) as conn:
            hour = self._count_since(conn, now - HOUR)
            day = self._count_since(conn, now - DAY)
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0), COALESCE(SUM(input_tokens+output_tokens+cache_tokens),0)"
                " FROM llm_calls WHERE tier='cloud' AND ts >= ?",
                (now - DAY,),
            ).fetchone()
            local_day = conn.execute(
                "SELECT COUNT(*) FROM llm_calls WHERE tier='local' AND ts >= ?", (now - DAY,)
            ).fetchone()[0]
        return {
            "cloud_hour": hour,
            "cloud_day": day,
            "cloud_day_cost": round(float(row[0]), 4),
            "cloud_day_tokens": int(row[1]),
            "local_day": int(local_day),
            "limit_hour": self.per_hour,
            "limit_day": self.per_day,
            "limit_night": self.per_night,
        }
