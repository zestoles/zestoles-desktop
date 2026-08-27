"""The activity log, and the seed of the real-time event bus.

Everything the system does autonomously is recorded here and fanned out to
subscribers. Right now the only subscriber is the terminal; in S7 a websocket
joins and the Control Center reads the same stream, which is why publishing is
already decoupled from display.

Two properties matter more than features:

  A subscriber must never break the publisher. A UI callback that raises during a
  night run cannot be allowed to kill the task that was reporting progress, so
  every callback is isolated.

  The log survives restarts. "What did it do last night?" is answered from SQLite,
  not from a buffer that died with the process.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.autonomy.events")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    source  TEXT NOT NULL,
    kind    TEXT NOT NULL,
    message TEXT NOT NULL,
    data    TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""

INFO = "info"
WARN = "warn"
ERROR = "error"
SUCCESS = "success"


@dataclass(slots=True)
class Event:
    ts: float
    level: str
    source: str
    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%H:%M:%S")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts, "level": self.level, "source": self.source,
            "kind": self.kind, "message": self.message, "data": self.data,
        }


Subscriber = Callable[[Event], None]


class EventLog:
    def __init__(self, db_path: Path, *, buffer_size: int = 200) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._recent: deque[Event] = deque(maxlen=buffer_size)
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(
        self,
        source: str,
        kind: str,
        message: str,
        *,
        level: str = INFO,
        data: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(time.time(), level, source, kind, message, data or {})
        try:
            with closing(self._conn()) as conn, conn:
                conn.execute(
                    "INSERT INTO events (ts, level, source, kind, message, data)"
                    " VALUES (?,?,?,?,?,?)",
                    (event.ts, level, source, kind, message,
                     json.dumps(event.data, ensure_ascii=False)),
                )
        except sqlite3.Error as exc:
            log.warning("olay yazılamadı: %s", exc)

        with self._lock:
            self._recent.append(event)
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 - a bad listener must not stop work
                log.warning("olay abonesi hata verdi: %s", exc)
        return event

    def purge(self, *, older_than_days: float = 30.0,
              problem_older_than_days: float = 90.0) -> int:
        """Drop old activity, keeping problems for longer. Returns rows deleted.

        The log is append-only and nothing used to remove from it, so it grew for
        as long as the system ran. Two cutoffs rather than one because the two
        kinds of row age differently: "what was it doing on a Tuesday in March"
        stops being useful quickly, while a warning or an error is the record of
        something that went wrong and may still be going wrong.
        """
        now = time.time()
        routine_cutoff = now - max(0.0, older_than_days) * 86400
        problem_cutoff = now - max(0.0, problem_older_than_days) * 86400
        try:
            with closing(self._conn()) as conn, conn:
                cursor = conn.execute(
                    "DELETE FROM events WHERE (level IN (?, ?) AND ts < ?)"
                    " OR (level NOT IN (?, ?) AND ts < ?)",
                    (WARN, ERROR, problem_cutoff, WARN, ERROR, routine_cutoff),
                )
                return int(cursor.rowcount)
        except sqlite3.Error as exc:
            log.warning("olay kaydı temizlenemedi: %s", exc)
            return 0

    def recent(self, limit: int = 20) -> list[Event]:
        with self._lock:
            return list(self._recent)[-limit:]

    def since(self, seconds: float, *, limit: int = 200) -> list[Event]:
        cutoff = time.time() - seconds
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        out = []
        for row in rows:
            try:
                data = json.loads(row["data"] or "{}")
            except json.JSONDecodeError:
                data = {}
            out.append(Event(float(row["ts"]), row["level"], row["source"],
                             row["kind"], row["message"], data))
        return out
