"""Ideas the system has had, what happened to them, and why it will not have them again.

A self-improving system that forgets its failures does not improve — it cycles. It
proposes the same optimisation every night, watches it fail the same way, records
the same disappointment, and starts over. The cost is not just wasted electricity:
the log fills with noise until a genuinely new failure is invisible in it.

So every hypothesis is fingerprinted on its meaning rather than its wording, and a
fingerprint that has been tried before is not tried again. Failure sets a cooldown
that doubles each time, and a hypothesis refuted outright is shelved unless the
lesson recorded against it says a specific change would make it worth retrying.

That last part is the difference between remembering a failure and learning from
one. "It failed" stops a repeat. "It failed because the benchmark had no warm-up
and the first run was always slower" tells the next attempt what to do differently.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..text import fold
from .opportunity import Opportunity

log = logging.getLogger("jarvis.improve.hypotheses")

PROPOSED = "onerildi"
TESTING = "deneniyor"
CONFIRMED = "dogrulandi"
REFUTED = "curutuldu"
SHELVED = "rafta"

STATES = (PROPOSED, TESTING, CONFIRMED, REFUTED, SHELVED)

#: First cooldown after a failure; doubles with each further attempt.
BASE_COOLDOWN_S = 6 * 3600
MAX_COOLDOWN_S = 30 * 86400
#: Attempts before a hypothesis is shelved regardless of what the lesson says.
MAX_ATTEMPTS = 3

_NOISE = frozenset({
    "bir", "bu", "su", "ve", "veya", "ile", "icin", "gibi", "daha", "cok",
    "olarak", "yapmak", "etmek", "olabilir", "gerekir", "sistem", "jarvis",
    "the", "a", "an", "of", "to", "for", "and", "or", "is", "be", "can", "we",
})

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id             TEXT PRIMARY KEY,
    fingerprint    TEXT NOT NULL UNIQUE,
    title          TEXT NOT NULL,
    statement      TEXT NOT NULL DEFAULT '',
    capability     TEXT NOT NULL DEFAULT '',
    gap_key        TEXT NOT NULL DEFAULT '',
    state          TEXT NOT NULL DEFAULT 'onerildi',
    attempts       INTEGER NOT NULL DEFAULT 0,
    last_attempt   REAL,
    cooldown_until REAL NOT NULL DEFAULT 0,
    opportunity    TEXT NOT NULL DEFAULT '{}',
    experiments    TEXT NOT NULL DEFAULT '[]',
    lesson         TEXT NOT NULL DEFAULT '{}',
    created        REAL NOT NULL,
    updated        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hypotheses_state ON hypotheses(state, cooldown_until);
"""


#: Tokens kept in the fingerprint, longest first. The connective words a
#: restatement adds or drops are short; the words carrying the idea are not.
CORE_TOKENS = 4
#: Prefix length. Turkish inflects endlessly — "hızlandır" and "hızlandırmak" are
#: the same word and must produce the same fingerprint.
STEM = 6


def fingerprint(text: str) -> str:
    """A hypothesis's identity is its meaning, not its phrasing.

    Reworded restatements must collide, or deduplication is defeated by a model
    that never says anything the same way twice. Two real examples that have to
    map together:

        "Yol çözümlemesini önbelleğe alarak sandbox'ı hızlandır"
        "Sandbox'ı hızlandırmak için yol çözümlemesini önbelleğe al"

    Exact token sets differ ("alarak" vs "al", "hızlandır" vs "hızlandırmak"), so
    the fingerprint keeps only the few longest words, stemmed. Erring toward
    collision is the safe direction: a false match costs one idea not being
    retried, a false miss costs the same experiment running every night forever.
    """
    import hashlib

    tokens = [token for token in fold(text).split()
              if len(token) > 3 and token not in _NOISE]
    tokens.sort(key=len, reverse=True)
    core = sorted({token[:STEM] for token in tokens[:CORE_TOKENS]})
    return hashlib.sha256(" ".join(core).encode("utf-8")).hexdigest()[:20]


@dataclass(slots=True)
class Lesson:
    """What a failure taught. Empty fields mean nobody looked."""

    why: str = ""
    wrong_assumption: str = ""
    conditions: str = ""
    retry_worth: bool = False
    needed_change: str = ""
    experiment_id: str = ""
    measured: dict[str, Any] = field(default_factory=dict)

    @property
    def useful(self) -> bool:
        return bool(self.why or self.wrong_assumption)

    def as_dict(self) -> dict[str, Any]:
        return {
            "why": self.why, "wrong_assumption": self.wrong_assumption,
            "conditions": self.conditions, "retry_worth": self.retry_worth,
            "needed_change": self.needed_change, "experiment_id": self.experiment_id,
            "measured": self.measured,
        }

    def summary(self) -> str:
        if not self.useful:
            return "ders çıkarılmadı"
        parts = [self.why or self.wrong_assumption]
        if self.needed_change:
            parts.append(f"gereken değişiklik: {self.needed_change}")
        return " · ".join(parts)


@dataclass(slots=True)
class Hypothesis:
    id: str
    fingerprint: str
    title: str
    statement: str = ""
    capability: str = ""
    gap_key: str = ""
    state: str = PROPOSED
    attempts: int = 0
    last_attempt: float | None = None
    cooldown_until: float = 0.0
    opportunity: dict[str, Any] = field(default_factory=dict)
    experiments: list[str] = field(default_factory=list)
    lesson: Lesson = field(default_factory=Lesson)
    created: float = 0.0
    updated: float = 0.0

    @property
    def cooling(self) -> bool:
        return self.cooldown_until > time.time()

    @property
    def runnable(self) -> bool:
        return (self.state in (PROPOSED, TESTING)
                and not self.cooling
                and self.attempts < MAX_ATTEMPTS)

    def summary(self) -> str:
        state = self.state
        if self.cooling:
            remaining = (self.cooldown_until - time.time()) / 3600
            state += f" (soğuma {remaining:.0f}s)"
        return f"{self.title[:70]} [{state}] deneme {self.attempts}"


class HypothesisStore:
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
    def propose(self, title: str, *, statement: str = "", capability: str = "",
                gap_key: str = "", opportunity: Opportunity | None = None
                ) -> tuple[Hypothesis, bool]:
        """Record a hypothesis. Returns (hypothesis, is_new).

        A fingerprint collision returns the existing record instead of a duplicate,
        so the caller can see it has been here before — and how that went.
        """
        print_id = fingerprint(f"{title} {statement}")
        existing = self.by_fingerprint(print_id)
        if existing is not None:
            return existing, False

        now = time.time()
        hypothesis = Hypothesis(
            id=uuid.uuid4().hex, fingerprint=print_id, title=title.strip(),
            statement=statement.strip(), capability=capability, gap_key=gap_key,
            opportunity=opportunity.as_dict() if opportunity else {},
            created=now, updated=now,
        )
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO hypotheses (id, fingerprint, title, statement, capability,"
                " gap_key, state, opportunity, created, updated)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (hypothesis.id, print_id, hypothesis.title, hypothesis.statement,
                 capability, gap_key, PROPOSED,
                 json.dumps(hypothesis.opportunity, ensure_ascii=False), now, now),
            )
        log.info("hipotez kaydedildi: %s", hypothesis.title[:60])
        return hypothesis, True

    def start_attempt(self, hypothesis_id: str, experiment_id: str) -> Hypothesis | None:
        hypothesis = self.get(hypothesis_id)
        if hypothesis is None:
            return None
        experiments = [*hypothesis.experiments, experiment_id]
        now = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE hypotheses SET state=?, attempts=attempts+1, last_attempt=?,"
                " experiments=?, updated=? WHERE id=?",
                (TESTING, now, json.dumps(experiments, ensure_ascii=False),
                 now, hypothesis_id),
            )
        return self.get(hypothesis_id)

    def confirm(self, hypothesis_id: str, *, note: str = "") -> Hypothesis | None:
        now = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE hypotheses SET state=?, cooldown_until=0, updated=? WHERE id=?",
                (CONFIRMED, now, hypothesis_id))
        log.info("hipotez doğrulandı: %s", hypothesis_id[:8])
        return self.get(hypothesis_id)

    def refute(self, hypothesis_id: str, lesson: Lesson) -> Hypothesis | None:
        """Record a failure with what it taught, and set the cooldown.

        A hypothesis whose lesson says a specific change would help stays open on a
        doubling cooldown. One with nothing learned, or out of attempts, is shelved
        — retrying it would only reproduce the same failure more slowly.
        """
        hypothesis = self.get(hypothesis_id)
        if hypothesis is None:
            return None

        attempts = hypothesis.attempts
        retryable = lesson.retry_worth and bool(lesson.needed_change) and attempts < MAX_ATTEMPTS
        state = REFUTED if retryable else SHELVED
        cooldown = (min(BASE_COOLDOWN_S * (2 ** max(0, attempts - 1)), MAX_COOLDOWN_S)
                    if retryable else MAX_COOLDOWN_S)

        now = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE hypotheses SET state=?, lesson=?, cooldown_until=?, updated=?"
                " WHERE id=?",
                (state, json.dumps(lesson.as_dict(), ensure_ascii=False),
                 now + cooldown, now, hypothesis_id),
            )
        log.info("hipotez %s: %s (soğuma %.0f saat)", hypothesis_id[:8], state, cooldown / 3600)
        return self.get(hypothesis_id)

    def shelve(self, hypothesis_id: str, *, reason: str = "") -> Hypothesis | None:
        now = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE hypotheses SET state=?, cooldown_until=?, updated=? WHERE id=?",
                (SHELVED, now + MAX_COOLDOWN_S, now, hypothesis_id))
        return self.get(hypothesis_id)

    # ------------------------------------------------------------- reading
    def get(self, hypothesis_id: str) -> Hypothesis | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM hypotheses WHERE id=?",
                               (hypothesis_id,)).fetchone()
        return _to_hypothesis(row) if row else None

    def by_fingerprint(self, print_id: str) -> Hypothesis | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM hypotheses WHERE fingerprint=?",
                               (print_id,)).fetchone()
        return _to_hypothesis(row) if row else None

    def seen(self, title: str, statement: str = "") -> Hypothesis | None:
        return self.by_fingerprint(fingerprint(f"{title} {statement}"))

    def open_for_gap(self, gap_key: str) -> Hypothesis | None:
        """Any hypothesis already addressing this gap and not yet finished with.

        Wording-based deduplication is not enough on its own. In a live run the
        model proposed the same weakness twice in materially different words, and
        the fingerprints did not collide — so a gap that already has an open,
        running or cooling hypothesis produces no second one. One idea per
        weakness at a time; the next arrives when this one is settled.
        """
        if not gap_key:
            return None
        now = time.time()
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT * FROM hypotheses WHERE gap_key=?"
                " AND (state IN (?,?) OR cooldown_until > ?)"
                " ORDER BY updated DESC LIMIT 1",
                (gap_key, PROPOSED, TESTING, now)).fetchone()
        return _to_hypothesis(row) if row else None

    def runnable(self, *, limit: int = 5) -> list[Hypothesis]:
        now = time.time()
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE state IN (?,?) AND cooldown_until<=?"
                " AND attempts<? ORDER BY created LIMIT ?",
                (PROPOSED, TESTING, now, MAX_ATTEMPTS, limit)).fetchall()
        return [_to_hypothesis(row) for row in rows]

    def list(self, *, state: str | None = None, limit: int = 50) -> list[Hypothesis]:
        query = "SELECT * FROM hypotheses"
        params: list[Any] = []
        if state:
            query += " WHERE state=?"
            params.append(state)
        query += " ORDER BY updated DESC LIMIT ?"
        params.append(limit)
        with closing(self._conn()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_to_hypothesis(row) for row in rows]

    def lessons(self, *, limit: int = 20) -> list[tuple[Hypothesis, Lesson]]:
        return [(h, h.lesson) for h in self.list(limit=limit) if h.lesson.useful]

    def counts(self) -> dict[str, int]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) c FROM hypotheses GROUP BY state").fetchall()
        return {row["state"]: int(row["c"]) for row in rows}


def _to_hypothesis(row: sqlite3.Row) -> Hypothesis:
    def load(column, fallback):
        try:
            return json.loads(row[column])
        except (json.JSONDecodeError, TypeError):
            return fallback

    lesson_data = load("lesson", {})
    return Hypothesis(
        id=row["id"], fingerprint=row["fingerprint"], title=row["title"],
        statement=row["statement"], capability=row["capability"],
        gap_key=row["gap_key"], state=row["state"], attempts=int(row["attempts"]),
        last_attempt=row["last_attempt"], cooldown_until=float(row["cooldown_until"]),
        opportunity=load("opportunity", {}), experiments=load("experiments", []),
        lesson=Lesson(**{k: v for k, v in lesson_data.items()
                         if k in Lesson.__slots__}) if lesson_data else Lesson(),
        created=float(row["created"]), updated=float(row["updated"]),
    )
