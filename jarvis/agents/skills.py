"""Plans that worked, kept so they need not be re-derived.

When an orchestration is verified, the shape that produced it — which roles, in
which order, doing what — is worth more than the answer itself. Saving it turns a
one-off success into a capability: the next similar goal skips planning entirely,
which removes both the slowest step and the one most able to go wrong.

Matching is a token-overlap heuristic over folded Turkish, and it is deliberately
conservative. A skill applied to the wrong goal produces a confidently structured
wrong answer, so the threshold is set where a near miss falls through to planning
rather than forcing a fit.

Skills carry their own record. One that keeps failing after being promoted is
demoted automatically — a saved plan is a hypothesis that stays under test, not a
decision that was made once.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..text import fold, slugify

log = logging.getLogger("jarvis.agents.skills")

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name      TEXT PRIMARY KEY,
    title     TEXT NOT NULL,
    goal      TEXT NOT NULL,
    keywords  TEXT NOT NULL DEFAULT '',
    steps     TEXT NOT NULL,
    criteria  TEXT NOT NULL DEFAULT '[]',
    created   REAL NOT NULL,
    last_used REAL,
    runs      INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    retired   INTEGER NOT NULL DEFAULT 0
);
"""

#: Fraction of a stored skill's keywords that a goal must contain to reuse it.
MATCH_THRESHOLD = 0.6
#: Below this success rate, after enough runs to mean something, a skill retires.
MIN_SUCCESS_RATE = 0.5
RETIRE_AFTER_RUNS = 3

_STOPWORDS = frozenset({
    "bir", "bu", "su", "ve", "veya", "ile", "icin", "gibi", "ama", "de", "da",
    "ne", "nasil", "neden", "mi", "mu", "the", "a", "an", "of", "to", "for",
    "yap", "yapalim", "olustur", "lutfen",
})


def keywords_of(goal: str) -> set[str]:
    return {
        token for token in fold(goal).split()
        if len(token) > 2 and token not in _STOPWORDS
    }


@dataclass(slots=True)
class Skill:
    name: str
    title: str
    goal: str
    steps: list[dict[str, Any]]
    criteria: list[str] = field(default_factory=list)
    keywords: set[str] = field(default_factory=set)
    runs: int = 0
    successes: int = 0
    created: float = 0.0
    last_used: float | None = None
    retired: bool = False

    @property
    def success_rate(self) -> float:
        return self.successes / self.runs if self.runs else 0.0

    def summary(self) -> str:
        rate = f"{self.success_rate:.0%}" if self.runs else "—"
        return f"{self.title} ({len(self.steps)} adım, {self.runs} çalışma, başarı {rate})"


class SkillLibrary:
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

    # -------------------------------------------------------------- writing
    def save(self, title: str, goal: str, steps: list[dict[str, Any]],
             criteria: list[str] | None = None) -> Skill:
        name = slugify(title)
        words = keywords_of(goal)
        now = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO skills (name, title, goal, keywords, steps, criteria, created)"
                " VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET steps=excluded.steps,"
                " criteria=excluded.criteria, keywords=excluded.keywords, retired=0",
                (name, title, goal, " ".join(sorted(words)),
                 json.dumps(steps, ensure_ascii=False),
                 json.dumps(criteria or [], ensure_ascii=False), now),
            )
        log.info("beceri kaydedildi: %s", name)
        return Skill(name, title, goal, steps, criteria or [], words, created=now)

    def record_run(self, name: str, *, ok: bool) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE skills SET runs=runs+1, successes=successes+?, last_used=? WHERE name=?",
                (1 if ok else 0, time.time(), name),
            )
            row = conn.execute(
                "SELECT runs, successes FROM skills WHERE name=?", (name,)
            ).fetchone()
            if row and int(row["runs"]) >= RETIRE_AFTER_RUNS:
                rate = int(row["successes"]) / int(row["runs"])
                if rate < MIN_SUCCESS_RATE:
                    conn.execute("UPDATE skills SET retired=1 WHERE name=?", (name,))
                    log.info("beceri emekliye ayrıldı (başarı %.0f%%): %s", rate * 100, name)

    def retire(self, name: str) -> bool:
        with closing(self._conn()) as conn, conn:
            return conn.execute(
                "UPDATE skills SET retired=1 WHERE name=?", (name,)
            ).rowcount > 0

    # -------------------------------------------------------------- reading
    def find(self, goal: str) -> Skill | None:
        """The best active skill whose keywords the goal mostly contains."""
        target = keywords_of(goal)
        if not target:
            return None
        best: tuple[float, Skill] | None = None
        for skill in self.list(include_retired=False):
            if not skill.keywords:
                continue
            overlap = len(skill.keywords & target) / len(skill.keywords)
            if overlap >= MATCH_THRESHOLD and (best is None or overlap > best[0]):
                best = (overlap, skill)
        if best:
            log.info("beceri eşleşti (%.0f%%): %s", best[0] * 100, best[1].name)
        return best[1] if best else None

    def get(self, name: str) -> Skill | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
        return _to_skill(row) if row else None

    def list(self, *, include_retired: bool = True) -> list[Skill]:
        query = "SELECT * FROM skills"
        if not include_retired:
            query += " WHERE retired=0"
        query += " ORDER BY successes DESC, created DESC"
        with closing(self._conn()) as conn:
            rows = conn.execute(query).fetchall()
        return [_to_skill(row) for row in rows]


def _to_skill(row: sqlite3.Row) -> Skill:
    try:
        steps = json.loads(row["steps"])
    except json.JSONDecodeError:
        steps = []
    try:
        criteria = json.loads(row["criteria"] or "[]")
    except json.JSONDecodeError:
        criteria = []
    return Skill(
        name=row["name"], title=row["title"], goal=row["goal"], steps=steps,
        criteria=criteria, keywords=set((row["keywords"] or "").split()),
        runs=int(row["runs"]), successes=int(row["successes"]),
        created=float(row["created"]), last_used=row["last_used"],
        retired=bool(row["retired"]),
    )
