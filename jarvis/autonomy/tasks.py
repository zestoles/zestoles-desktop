"""The persistent task queue.

State lives in SQLite, not in memory, for one reason: JARVIS is meant to run for
months and will be killed mid-task — by a reboot, a crash, a closed terminal. A
queue that forgets what it was doing turns every restart into silent data loss.

Crash recovery is explicit rather than clever. Anything found RUNNING at startup
was orphaned by a process that is no longer alive, so it goes back to PENDING with
its attempt counted. A task that repeatedly kills the process therefore exhausts
its attempts and is quarantined instead of crash-looping forever.

Priority is ascending: lower number wins. USER work outranks everything because
the machine belongs to the user.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.autonomy.tasks")


class Priority(IntEnum):
    USER = 0          # asked for directly; runs whatever the machine is doing
    CRITICAL = 10     # system integrity: recovery, rollback
    NORMAL = 50       # ordinary queued work
    BACKGROUND = 80   # runs when the machine is idle
    IDLE_ONLY = 100   # long work; night only


class State:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUARANTINED = "quarantined"

    TERMINAL = frozenset({DONE, CANCELLED, QUARANTINED})


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL,
    title        TEXT    NOT NULL,
    payload      TEXT    NOT NULL DEFAULT '{}',
    priority     INTEGER NOT NULL DEFAULT 50,
    state        TEXT    NOT NULL DEFAULT 'pending',
    origin       TEXT    NOT NULL DEFAULT 'auto',
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    created      REAL    NOT NULL,
    not_before   REAL    NOT NULL DEFAULT 0,
    started      REAL,
    finished     REAL,
    result       TEXT,
    error        TEXT,
    dedupe_key   TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_ready ON tasks(state, priority, not_before);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_dedupe
    ON tasks(dedupe_key) WHERE dedupe_key IS NOT NULL AND state IN ('pending','running');
"""

BACKOFF_BASE_S = 30
BACKOFF_CAP_S = 3600


@dataclass(slots=True)
class Task:
    id: int
    kind: str
    title: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = Priority.NORMAL
    state: str = State.PENDING
    origin: str = "auto"
    attempts: int = 0
    max_attempts: int = 3
    created: float = 0.0
    not_before: float = 0.0
    started: float | None = None
    finished: float | None = None
    result: str | None = None
    error: str | None = None
    dedupe_key: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Task:
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        return cls(
            id=int(row["id"]), kind=row["kind"], title=row["title"], payload=payload,
            priority=int(row["priority"]), state=row["state"], origin=row["origin"],
            attempts=int(row["attempts"]), max_attempts=int(row["max_attempts"]),
            created=float(row["created"]), not_before=float(row["not_before"]),
            started=row["started"], finished=row["finished"],
            result=row["result"], error=row["error"], dedupe_key=row["dedupe_key"],
        )

    @property
    def label(self) -> str:
        return f"#{self.id} {self.title}"


class TaskQueue:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------- writing
    def add(
        self,
        kind: str,
        title: str,
        *,
        payload: dict[str, Any] | None = None,
        priority: int = Priority.NORMAL,
        origin: str = "auto",
        max_attempts: int = 3,
        delay_s: float = 0.0,
        dedupe_key: str | None = None,
    ) -> int | None:
        """Queue a task. Returns None when an identical one is already waiting."""
        now = time.time()
        try:
            with closing(self._conn()) as conn, conn:
                cursor = conn.execute(
                    "INSERT INTO tasks (kind, title, payload, priority, origin,"
                    " max_attempts, created, not_before, dedupe_key)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (kind, title, json.dumps(payload or {}, ensure_ascii=False), int(priority),
                     origin, max_attempts, now, now + delay_s, dedupe_key),
                )
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError:
            log.debug("zaten kuyrukta, atlandı: %s", dedupe_key)
            return None

    def last_run(self, kind: str, *, dedupe_key: str | None = None) -> float:
        """When work of this kind last happened. 0.0 when it never has.

        Finished time when there is one, otherwise started, otherwise created —
        a task queued an hour ago and still waiting for the machine to go quiet
        has already had its turn scheduled, and asking for a second one would
        queue the same work twice the moment the first one finally runs.
        """
        sql = ("SELECT MAX(COALESCE(finished, started, created)) AS last FROM tasks"
               " WHERE kind = ?")
        params: list[Any] = [kind]
        if dedupe_key is not None:
            sql += " AND dedupe_key = ?"
            params.append(dedupe_key)
        with closing(self._conn()) as conn:
            row = conn.execute(sql, params).fetchone()
        return float(row["last"]) if row and row["last"] is not None else 0.0

    def waiting_dedupe_keys(self) -> set[str]:
        """Dedupe keys that an add() would currently be refused for.

        Deliberately mirrors idx_tasks_dedupe: pending or running, key not null.
        A caller that already knows the answer can stop asking the database to
        reject it — see AutonomyCore.queue_due_routines for why that mattered.
        """
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT dedupe_key FROM tasks"
                " WHERE dedupe_key IS NOT NULL AND state IN (?, ?)",
                (State.PENDING, State.RUNNING),
            ).fetchall()
        return {row["dedupe_key"] for row in rows}

    def claim(self, *, max_priority: int) -> Task | None:
        """Atomically take the most important runnable task at or above a priority."""
        now = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM tasks WHERE state=? AND priority<=? AND not_before<=?"
                " ORDER BY priority ASC, created ASC LIMIT 1",
                (State.PENDING, int(max_priority), now),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE tasks SET state=?, started=?, attempts=attempts+1 WHERE id=?",
                (State.RUNNING, now, row["id"]),
            )
            task = Task.from_row(row)
            task.state = State.RUNNING
            task.started = now
            task.attempts += 1
            return task

    def complete(self, task_id: int, result: str = "") -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE tasks SET state=?, finished=?, result=?, error=NULL WHERE id=?",
                (State.DONE, time.time(), result[:4000], task_id),
            )

    def fail(self, task_id: int, error: str) -> str:
        """Record a failure; retry with backoff or quarantine when attempts run out."""
        now = time.time()
        with closing(self._conn()) as conn, conn:
            row = conn.execute(
                "SELECT attempts, max_attempts FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                return State.FAILED
            attempts, limit = int(row["attempts"]), int(row["max_attempts"])
            if attempts >= limit:
                conn.execute(
                    "UPDATE tasks SET state=?, finished=?, error=? WHERE id=?",
                    (State.QUARANTINED, now, error[:4000], task_id),
                )
                return State.QUARANTINED
            delay = min(BACKOFF_BASE_S * (2 ** (attempts - 1)), BACKOFF_CAP_S)
            conn.execute(
                "UPDATE tasks SET state=?, error=?, not_before=?, finished=NULL WHERE id=?",
                (State.PENDING, error[:4000], now + delay, task_id),
            )
            return State.PENDING

    def cancel(self, task_id: int) -> bool:
        with closing(self._conn()) as conn, conn:
            cursor = conn.execute(
                "UPDATE tasks SET state=?, finished=? WHERE id=? AND state IN (?,?)",
                (State.CANCELLED, time.time(), task_id, State.PENDING, State.RUNNING),
            )
            return cursor.rowcount > 0

    def recover_orphans(self) -> int:
        """Return tasks abandoned by a dead process to the queue.

        Their attempt already counted, so a task that reliably kills the process
        quarantines itself instead of restarting forever.
        """
        with closing(self._conn()) as conn, conn:
            cursor = conn.execute(
                "UPDATE tasks SET state=?, not_before=? WHERE state=?",
                (State.PENDING, time.time() + BACKOFF_BASE_S, State.RUNNING),
            )
            if cursor.rowcount:
                log.info("%s yarım kalmış görev kuyruğa geri kondu", cursor.rowcount)
            return cursor.rowcount

    # ------------------------------------------------------------- reading
    def get(self, task_id: int) -> Task | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return Task.from_row(row) if row else None

    def list(self, *, states: tuple[str, ...] | None = None, limit: int = 25) -> list[Task]:
        query = "SELECT * FROM tasks"
        params: list[Any] = []
        if states:
            query += f" WHERE state IN ({','.join('?' * len(states))})"
            params.extend(states)
        query += " ORDER BY CASE state WHEN 'running' THEN 0 ELSE 1 END, priority, created DESC LIMIT ?"
        params.append(limit)
        with closing(self._conn()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [Task.from_row(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT state, COUNT(*) c FROM tasks GROUP BY state").fetchall()
        return {row["state"]: int(row["c"]) for row in rows}

    def purge(self, *, older_than_days: float = 30.0) -> int:
        cutoff = time.time() - older_than_days * 86400
        with closing(self._conn()) as conn, conn:
            cursor = conn.execute(
                "DELETE FROM tasks WHERE state IN (?,?) AND finished < ?",
                (State.DONE, State.CANCELLED, cutoff),
            )
            return cursor.rowcount
