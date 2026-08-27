"""What the user has said they want, separate from system measurements.

Preferences change how opportunities are ranked. "Önce oyuncu sayısı, gelir ikinci
sırada" is not a fact about the world and must never be stored as one — but it is
binding on what the system chooses to work on, which makes it more important than
most facts.

Two rules keep the distinction:

  A preference records who said it and when. It is owner-sourced by definition;
  nothing inferred about what the user probably wants may be written here.

  A preference adjusts weights, never evidence. It can make revenue matter less in
  the ranking; it cannot make a revenue estimate more or less true.

Weights are multipliers on the scoring dimensions, clamped so that no single stated
preference can collapse the others to nothing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .opportunity import DEFAULT_WEIGHTS, DIMENSIONS

log = logging.getLogger("jarvis.improve.preferences")

#: Clamp on any stated preference's effect. Beyond this a single preference would
#: silence every other dimension, which is not what stating one means.
MIN_MULTIPLIER = 0.25
MAX_MULTIPLIER = 3.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    key       TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    weights   TEXT NOT NULL DEFAULT '{}',
    active    INTEGER NOT NULL DEFAULT 1,
    stated_at REAL NOT NULL,
    source    TEXT NOT NULL DEFAULT 'kullanici'
);
"""


@dataclass(slots=True)
class Preference:
    key: str
    statement: str
    weights: dict[str, float]
    active: bool = True
    stated_at: float = 0.0
    source: str = "kullanici"

    def summary(self) -> str:
        when = datetime.fromtimestamp(self.stated_at).strftime("%d.%m.%Y") if self.stated_at else "?"
        adjustments = ", ".join(f"{k}×{v:g}" for k, v in sorted(self.weights.items()))
        return f"{self.statement} ({when}) → {adjustments or 'ağırlık değişikliği yok'}"

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "statement": self.statement, "weights": self.weights,
                "active": self.active, "stated_at": self.stated_at, "source": self.source}


class PreferenceStore:
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

    def state(self, key: str, statement: str, weights: dict[str, float] | None = None,
              *, source: str = "kullanici") -> Preference:
        """Record something the owner said. Unknown dimensions are refused."""
        clean: dict[str, float] = {}
        for name, value in (weights or {}).items():
            if name not in DIMENSIONS:
                raise ValueError(f"bilinmeyen boyut: {name}")
            clean[name] = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, float(value)))

        preference = Preference(key=key, statement=statement.strip(), weights=clean,
                                active=True, stated_at=time.time(), source=source)
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO preferences (key, statement, weights, active, stated_at, source)"
                " VALUES (?,?,?,1,?,?)"
                " ON CONFLICT(key) DO UPDATE SET statement=excluded.statement,"
                " weights=excluded.weights, active=1, stated_at=excluded.stated_at,"
                " source=excluded.source",
                (key, preference.statement, json.dumps(clean, ensure_ascii=False),
                 preference.stated_at, source),
            )
        log.info("tercih kaydedildi: %s", preference.summary())
        return preference

    def retract(self, key: str) -> bool:
        with closing(self._conn()) as conn, conn:
            return conn.execute("UPDATE preferences SET active=0 WHERE key=?",
                                (key,)).rowcount > 0

    def get(self, key: str) -> Preference | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM preferences WHERE key=?", (key,)).fetchone()
        return _to_preference(row) if row else None

    def list(self, *, active_only: bool = True) -> list[Preference]:
        query = "SELECT * FROM preferences"
        if active_only:
            query += " WHERE active=1"
        query += " ORDER BY stated_at DESC"
        with closing(self._conn()) as conn:
            rows = conn.execute(query).fetchall()
        return [_to_preference(row) for row in rows]

    def weights(self) -> dict[str, float]:
        """Scoring weights with every active preference applied, newest last."""
        weights = dict(DEFAULT_WEIGHTS)
        for preference in sorted(self.list(), key=lambda p: p.stated_at):
            for name, multiplier in preference.weights.items():
                weights[name] = max(0.05, weights.get(name, 1.0) * multiplier)
        return weights


def _to_preference(row: sqlite3.Row) -> Preference:
    try:
        weights = json.loads(row["weights"])
    except (json.JSONDecodeError, TypeError):
        weights = {}
    return Preference(
        key=row["key"], statement=row["statement"], weights=weights,
        active=bool(row["active"]), stated_at=float(row["stated_at"]),
        source=row["source"],
    )
