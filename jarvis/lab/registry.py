"""The record of every experiment, and the state machine it must walk.

An experiment that cannot be reconstructed afterwards is not an experiment, it is
a change of unknown origin. So each one carries what it was for, which commit it
started from, which model produced it, what it touched, what the measurements
said, and when each of those happened.

## Why the states are a machine and not a field

The dangerous transition is the last one. If PROMOTED were merely a value that
could be written, then any code path — or any future agent holding a database
handle — could set it, and everything the benchmark gate does would become
advisory. Transitions are therefore checked against a fixed table, and the only
route to PROMOTED runs through CANDIDATE, which is only reachable from PASSED,
which is only set by a benchmark comparison.

    EXPERIMENT ──▶ PASSED ──▶ CANDIDATE ──▶ PROMOTED
        │            │            │
        └────────────┴────────────┴──────▶ FAILED ──▶ DISCARDED

Every state may fail. Nothing may skip forward. There is no edge into PROMOTED
that does not pass the gate.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.lab.registry")

EXPERIMENT = "experiment"
PASSED = "passed"
CANDIDATE = "candidate"
PROMOTED = "promoted"
FAILED = "failed"
DISCARDED = "discarded"

STATES = (EXPERIMENT, PASSED, CANDIDATE, PROMOTED, FAILED, DISCARDED)

#: The only permitted moves. Anything absent here is refused, including any
#: shortcut into PROMOTED.
TRANSITIONS: dict[str, frozenset[str]] = {
    EXPERIMENT: frozenset({PASSED, FAILED}),
    PASSED: frozenset({CANDIDATE, FAILED}),
    CANDIDATE: frozenset({PROMOTED, FAILED}),
    PROMOTED: frozenset(),
    FAILED: frozenset({DISCARDED}),
    DISCARDED: frozenset(),
}

TERMINAL = frozenset({PROMOTED, DISCARDED})

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            TEXT PRIMARY KEY,
    purpose       TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'experiment',
    base_commit   TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    sandbox_path  TEXT NOT NULL DEFAULT '',
    changed_files TEXT NOT NULL DEFAULT '[]',
    baseline      TEXT NOT NULL DEFAULT '{}',
    result        TEXT NOT NULL DEFAULT '{}',
    comparison    TEXT NOT NULL DEFAULT '{}',
    notes         TEXT NOT NULL DEFAULT '',
    created       REAL NOT NULL,
    updated       REAL NOT NULL,
    promoted      REAL
);
CREATE TABLE IF NOT EXISTS experiment_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    ts            REAL NOT NULL,
    from_state    TEXT,
    to_state      TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_experiment_events ON experiment_events(experiment_id, ts);
"""


class TransitionRefused(RuntimeError):
    """An attempt to move an experiment somewhere the machine does not allow."""


@dataclass(slots=True)
class Experiment:
    id: str
    purpose: str
    state: str = EXPERIMENT
    base_commit: str = ""
    model: str = ""
    sandbox_path: str = ""
    changed_files: list[str] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created: float = 0.0
    updated: float = 0.0
    promoted: float | None = None

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL

    def summary(self) -> str:
        return (f"{self.id[:8]} · {self.state} · {len(self.changed_files)} dosya "
                f"· {self.purpose[:60]}")


def current_commit(repo: Path) -> str:
    """HEAD of the repository, or empty when there is not one to read."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
            text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("commit okunamadı: %s", exc)
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


class ExperimentRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------- writing
    def open(self, purpose: str, *, base_commit: str = "", model: str = "",
             sandbox_path: str = "") -> Experiment:
        now = time.time()
        experiment = Experiment(
            id=uuid.uuid4().hex, purpose=purpose.strip() or "isimsiz deney",
            base_commit=base_commit, model=model, sandbox_path=sandbox_path,
            created=now, updated=now,
        )
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO experiments (id, purpose, state, base_commit, model,"
                " sandbox_path, created, updated) VALUES (?,?,?,?,?,?,?,?)",
                (experiment.id, experiment.purpose, EXPERIMENT, base_commit, model,
                 sandbox_path, now, now),
            )
            conn.execute(
                "INSERT INTO experiment_events (experiment_id, ts, from_state, to_state, reason)"
                " VALUES (?,?,?,?,?)",
                (experiment.id, now, None, EXPERIMENT, "açıldı"),
            )
        log.info("deney açıldı: %s", experiment.summary())
        return experiment

    def transition(self, experiment_id: str, to_state: str, *, reason: str = "") -> Experiment:
        """Move an experiment. Refuses anything the table does not permit."""
        if to_state not in STATES:
            raise TransitionRefused(f"bilinmeyen durum: {to_state}")

        experiment = self.get(experiment_id)
        if experiment is None:
            raise TransitionRefused(f"deney bulunamadı: {experiment_id}")

        allowed = TRANSITIONS.get(experiment.state, frozenset())
        if to_state not in allowed:
            raise TransitionRefused(
                f"{experiment.state} → {to_state} geçişine izin yok "
                f"(izinliler: {', '.join(sorted(allowed)) or 'yok'})")

        now = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE experiments SET state=?, updated=?, promoted=?, notes=? WHERE id=?",
                (to_state, now, now if to_state == PROMOTED else experiment.promoted,
                 reason or experiment.notes, experiment_id),
            )
            conn.execute(
                "INSERT INTO experiment_events (experiment_id, ts, from_state, to_state, reason)"
                " VALUES (?,?,?,?,?)",
                (experiment_id, now, experiment.state, to_state, reason),
            )
        log.info("deney %s: %s → %s (%s)", experiment_id[:8], experiment.state, to_state, reason)
        return self.get(experiment_id)

    def record_files(self, experiment_id: str, files: list[str]) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute("UPDATE experiments SET changed_files=?, updated=? WHERE id=?",
                         (json.dumps(sorted(set(files)), ensure_ascii=False),
                          time.time(), experiment_id))

    def record_measurements(self, experiment_id: str, *, baseline: dict | None = None,
                            result: dict | None = None, comparison: dict | None = None) -> None:
        sets, params = [], []
        for column, value in (("baseline", baseline), ("result", result),
                              ("comparison", comparison)):
            if value is not None:
                sets.append(f"{column}=?")
                params.append(json.dumps(value, ensure_ascii=False, default=str))
        if not sets:
            return
        params.extend([time.time(), experiment_id])
        with closing(self._conn()) as conn, conn:
            conn.execute(f"UPDATE experiments SET {', '.join(sets)}, updated=? WHERE id=?",
                         params)

    # ------------------------------------------------------------- reading
    def get(self, experiment_id: str) -> Experiment | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?",
                               (experiment_id,)).fetchone()
        return _to_experiment(row) if row else None

    def list(self, *, state: str | None = None, limit: int = 50) -> list[Experiment]:
        query = "SELECT * FROM experiments"
        params: list[Any] = []
        if state:
            query += " WHERE state=?"
            params.append(state)
        query += " ORDER BY created DESC LIMIT ?"
        params.append(limit)
        with closing(self._conn()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_to_experiment(row) for row in rows]

    def history(self, experiment_id: str) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT ts, from_state, to_state, reason FROM experiment_events"
                " WHERE experiment_id=? ORDER BY ts", (experiment_id,)).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) c FROM experiments GROUP BY state").fetchall()
        return {row["state"]: int(row["c"]) for row in rows}


def _to_experiment(row: sqlite3.Row) -> Experiment:
    def load(column: str, fallback):
        try:
            return json.loads(row[column])
        except (json.JSONDecodeError, TypeError):
            return fallback

    return Experiment(
        id=row["id"], purpose=row["purpose"], state=row["state"],
        base_commit=row["base_commit"], model=row["model"],
        sandbox_path=row["sandbox_path"], changed_files=load("changed_files", []),
        baseline=load("baseline", {}), result=load("result", {}),
        comparison=load("comparison", {}), notes=row["notes"],
        created=float(row["created"]), updated=float(row["updated"]),
        promoted=row["promoted"],
    )
