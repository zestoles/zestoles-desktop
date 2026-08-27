"""What JARVIS knows about the person using it, and how it is allowed to know it.

An assistant that remembers you is more useful than one that does not. It is
also the part of the system where being wrong is least forgivable: a made-up
"fact" about someone is not a bug they can see, it is a thing the machine will
quietly act on for months.

So this layer keeps three things about every remembered item, and refuses to
lose any of them.

## source -- where it came from

    kullanici   The user said it. The only source that needs no further defence.
    arac        a tool measured it. True about the machine, not about him.
    cikarim     JARVIS inferred it. A guess, however good it sounds.

Inference is stored, never promoted. "Sen stresli birisin" is not a fact about a
person no matter how confidently a model produces it, and the separation is
structural rather than a matter of prompting: `Profile.facts()` will not return
an inference, and `accept()` will not store one as anything else.

## consent -- whether they agreed to it being kept

Explicit consent is the difference between an assistant and surveillance. A
preference is only durable when the user said yes to keeping it; anything else
lives for the session and is gone. `Consent.UNKNOWN` exists because "not asked
yet" and "declined" are different states and collapsing them loses the ability
to ask later.

## confidence -- how sure

Not a probability, a category. "Geceleri çalışıyorum" said outright is HIGH;
the same thing concluded from three late timestamps is LOW, and staying LOW is
what stops it from being repeated back as though he had said it.

## Forgetting is real

`forget()` deletes rows. There is no tombstone, no "hidden" flag, no archive
the model can still read. When someone says "bunu unut" the only honest
implementation is the one where the thing is gone.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.memory.profile")


class Source:
    """Where a remembered thing came from. Never inferred from the text itself."""

    USER = "kullanici"
    TOOL = "arac"
    INFERRED = "cikarim"

    ALL = frozenset({USER, TOOL, INFERRED})
    #: Sources that may be stated back to the user as fact.
    FACTUAL = frozenset({USER, TOOL})


class Confidence:
    HIGH = "yuksek"
    MEDIUM = "orta"
    LOW = "dusuk"

    ALL = frozenset({HIGH, MEDIUM, LOW})


class Consent:
    GRANTED = "verildi"
    REFUSED = "reddedildi"
    UNKNOWN = "sorulmadi"

    ALL = frozenset({GRANTED, REFUSED, UNKNOWN})


SCHEMA = """
CREATE TABLE IF NOT EXISTS user_preferences (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject    TEXT    NOT NULL,
    detail     TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    confidence TEXT    NOT NULL,
    consent    TEXT    NOT NULL,
    created    REAL    NOT NULL,
    updated    REAL    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_pref_subject ON user_preferences(subject);
"""

#: A subject longer than this is a sentence, not a label, and a label is what
#: makes an item findable when the user asks to forget it.
SUBJECT_MAX = 60
DETAIL_MAX = 400


@dataclass(slots=True)
class Preference:
    """One thing JARVIS knows, with everything needed to judge it."""

    subject: str
    detail: str
    source: str = Source.USER
    confidence: str = Confidence.HIGH
    consent: str = Consent.UNKNOWN
    created: float = 0.0
    updated: float = 0.0
    id: int = 0

    @property
    def durable(self) -> bool:
        """Whether this may be kept between sessions.

        Consent is required and inference is excluded even with consent: a guess
        the user agreed to keep is still a guess, and keeping it as a preference
        would make it indistinguishable from something they said.
        """
        return self.consent == Consent.GRANTED and self.source in Source.FACTUAL

    @property
    def statable(self) -> bool:
        """Whether JARVIS may state this back as something it knows."""
        return self.source in Source.FACTUAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "konu": self.subject, "ayrinti": self.detail,
            "kaynak": self.source, "guven": self.confidence,
            "izin": self.consent, "kalici": self.durable,
        }

    def as_line(self) -> str:
        """One readable line, with the provenance visible rather than implied."""
        mark = {Source.USER: "senin söylediğin",
                Source.TOOL: "ölçülen",
                Source.INFERRED: "benim çıkarımım"}.get(self.source, self.source)
        return f"{self.subject}: {self.detail} ({mark})"


def accept(item: Preference) -> tuple[bool, str]:
    """Whether this may be stored durably, and why not when it may not.

    Pure, so the rule can be tested without a database. This is the gate that
    keeps a model's guess about a person out of long-term memory, and it must
    not depend on the model behaving well.
    """
    if not item.subject.strip():
        return False, "konu boş"
    if not item.detail.strip():
        return False, "ayrıntı boş"
    if item.source not in Source.ALL:
        return False, f"bilinmeyen kaynak: {item.source}"
    if item.confidence not in Confidence.ALL:
        return False, f"bilinmeyen güven düzeyi: {item.confidence}"
    if item.consent not in Consent.ALL:
        return False, f"bilinmeyen izin durumu: {item.consent}"
    if item.source == Source.INFERRED:
        return False, "çıkarım kalıcı hafızaya girmez — kullanıcıya sor"
    if item.consent != Consent.GRANTED:
        return False, "kullanıcı kaydedilmesini onaylamadı"
    return True, ""


class Profile:
    """The durable things JARVIS knows about the person using it."""

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
    def remember(self, subject: str, detail: str, *,
                 source: str = Source.USER,
                 confidence: str = Confidence.HIGH,
                 consent: str = Consent.UNKNOWN) -> tuple[Preference | None, str]:
        """Store one thing. Returns (stored, reason it was refused)."""
        item = Preference(
            subject=" ".join(str(subject or "").split())[:SUBJECT_MAX],
            detail=" ".join(str(detail or "").split())[:DETAIL_MAX],
            source=source, confidence=confidence, consent=consent)
        ok, why = accept(item)
        if not ok:
            log.info("tercih kaydedilmedi (%s): %s", why, item.subject)
            return None, why

        now = time.time()
        item.created = item.updated = now
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO user_preferences (subject, detail, source, confidence,"
                " consent, created, updated) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(subject) DO UPDATE SET detail=excluded.detail,"
                " source=excluded.source, confidence=excluded.confidence,"
                " consent=excluded.consent, updated=excluded.updated",
                (item.subject, item.detail, item.source, item.confidence,
                 item.consent, now, now))
        log.info("tercih kaydedildi: %s", item.subject)
        return item, ""

    def forget(self, subject: str) -> int:
        """Delete. Not hide, not archive -- the row is gone."""
        needle = " ".join(str(subject or "").split())
        if not needle:
            return 0
        with closing(self._conn()) as conn, conn:
            cursor = conn.execute(
                "DELETE FROM user_preferences WHERE subject = ? COLLATE NOCASE"
                " OR subject LIKE ? COLLATE NOCASE", (needle, f"%{needle}%"))
            removed = cursor.rowcount
        if removed:
            log.info("%s tercih unutuldu: %s", removed, needle)
        return removed

    def forget_all(self) -> int:
        with closing(self._conn()) as conn, conn:
            return conn.execute("DELETE FROM user_preferences").rowcount

    # -------------------------------------------------------------- reading
    def recall(self, limit: int = 50) -> list[Preference]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT * FROM user_preferences ORDER BY updated DESC LIMIT ?",
                (limit,)).fetchall()
        return [_from_row(row) for row in rows]

    def facts(self) -> list[Preference]:
        """Only what may be stated back as known. Inference never appears here."""
        return [item for item in self.recall() if item.statable]

    def summary(self, limit: int = 12) -> str:
        """What JARVIS knows, as the model should be told it.

        Given to the model as context rather than as the user's own words, and
        each line carries its provenance -- so a preference cannot be repeated
        back as though it had just been said.
        """
        items = self.facts()[:limit]
        if not items:
            return ""
        lines = "\n".join(f"- {item.as_line()}" for item in items)
        return ("Kullanıcı hakkında daha önce kaydedilenler (senin bilgin, "
                f"onun şimdi söylediği değil):\n{lines}")

    def count(self) -> int:
        with closing(self._conn()) as conn:
            return int(conn.execute(
                "SELECT COUNT(*) c FROM user_preferences").fetchone()["c"])


def _from_row(row: sqlite3.Row) -> Preference:
    return Preference(
        id=int(row["id"]), subject=row["subject"], detail=row["detail"],
        source=row["source"], confidence=row["confidence"],
        consent=row["consent"], created=float(row["created"]),
        updated=float(row["updated"]))


__all__ = ["Profile", "Preference", "Source", "Confidence", "Consent", "accept",
           "SUBJECT_MAX", "DETAIL_MAX"]
