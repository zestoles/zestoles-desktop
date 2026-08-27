"""Finding what is actually wrong, from evidence rather than introspection.

Asking a model "what are you bad at?" produces fluent, plausible self-criticism
that correlates with nothing. It will name whatever sounds like a reasonable
weakness for a system of this kind, and it will name it again tomorrow whether or
not anything changed.

So gaps are derived from things that happened and were recorded:

  capability status      what the registry says is missing, broken or unverified
  quarantined tasks      work that failed enough times to be given up on
  repeated errors        the same failure appearing again and again in the log
  failed experiments     hypotheses that were tried and did not survive
  stale verification     a capability nobody has measured in a month

Each gap carries the evidence that produced it, so a proposal built on it can be
checked against the same facts later. Severity is arithmetic on counts — a gap
cannot argue itself into being important.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..autonomy.runners import FIXTURE_KINDS
from ..text import fold
from .capabilities import BROKEN, MISSING, PARTIAL, WORKING, CapabilityRegistry

log = logging.getLogger("jarvis.improve.gaps")

FROM_CAPABILITY = "yetenek"
FROM_TASKS = "gorev"
FROM_ERRORS = "hata"
FROM_EXPERIMENTS = "deney"
FROM_STALE = "eskimis"
FROM_FEEDBACK = "geri-bildirim"

#: A failure seen fewer times than this is noise, not a pattern.
REPEAT_THRESHOLD = 3
#: Window for counting recent failures.
LOOKBACK_DAYS = 14

#: Sources where something already exists and already misbehaves. The current
#: behaviour is the baseline, so the gap can be measured as a comparison.
COMPARABLE_SOURCES = frozenset({FROM_ERRORS, FROM_EXPERIMENTS, FROM_TASKS})


def shape_of(source: str, *, status: str = "",
             has_benchmark: bool = False) -> tuple[bool, str]:
    """Can this gap become a baseline-versus-candidate comparison?

    One structural question, asked of recorded facts: is there something to
    compare against. An experiment measures a change against what came before,
    so a gap with no "before" cannot be turned into one.

    This is why S6b produced four plans and no passing experiment. Three of the
    open hypotheses came from capabilities recorded as *missing* — "tarayıcı
    otomasyonu yok" — and a missing capability has no current behaviour to be the
    baseline. Asked to write a comparison anyway, the model invented one, and
    invented code is what it invented.

    Nothing here asks the model. A gap cannot argue its way through this any more
    than it can argue its way into being severe.
    """
    if source in COMPARABLE_SOURCES:
        return True, "mevcut davranış baseline olarak ölçülebilir"
    if source == FROM_STALE:
        # The remedy is a measurement, not a change. There is no candidate.
        return False, "eksik bir ölçüm istiyor, bir değişiklik değil"
    if source == FROM_CAPABILITY:
        if status in (BROKEN, PARTIAL):
            return True, f"yetenek {status} — mevcut davranış baseline"
        if has_benchmark:
            return True, "kayıtlı ölçüm baseline sağlıyor"
        return False, "yetenek yok — karşılaştırılacak baseline yok"
    return False, f"bilinmeyen kaynak: {source}"


@dataclass(slots=True)
class Gap:
    key: str
    source: str
    title: str
    severity: float
    capability: str = ""
    evidence: list[str] = field(default_factory=list)
    #: Whether planning may be attempted for this gap, and why. Set where the
    #: gap is built, because that is the only place that knows the facts the
    #: rule needs. Defaults closed: a gap nobody classified is not a candidate.
    experiment_shaped: bool = False
    shape_reason: str = "sınıflandırılmadı"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "source": self.source, "title": self.title,
            "severity": round(self.severity, 3), "capability": self.capability,
            "evidence": self.evidence,
            "experiment_shaped": self.experiment_shaped,
            "shape_reason": self.shape_reason,
        }

    def summary(self) -> str:
        return f"[{self.severity:.2f}] {self.title} ({self.source})"


def _key(source: str, subject: str) -> str:
    return f"{source}:{fold(subject).replace(' ', '-')[:60]}"


class GapDetector:
    def __init__(self, db_path: Path, capabilities: CapabilityRegistry) -> None:
        self.db_path = Path(db_path)
        self.capabilities = capabilities

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def detect(self, *, limit: int = 12) -> list[Gap]:
        gaps: list[Gap] = []
        gaps.extend(self._from_capabilities())
        gaps.extend(self._from_tasks())
        gaps.extend(self._from_errors())
        gaps.extend(self._from_experiments())
        gaps.sort(key=lambda gap: -gap.severity)
        return gaps[:limit]

    # ------------------------------------------------------------ signals
    def _from_capabilities(self) -> list[Gap]:
        gaps: list[Gap] = []
        for capability in self.capabilities.list():
            weight = capability.gap_weight
            if weight <= 0.0:
                continue

            if capability.status in (MISSING, BROKEN, PARTIAL):
                source = FROM_CAPABILITY
                title = f"{capability.title} yeteneği {capability.status}"
            elif capability.status == WORKING and capability.stale:
                source = FROM_STALE
                age = capability.age_days
                title = (f"{capability.title} uzun süredir doğrulanmadı"
                         if age else f"{capability.title} hiç ölçülmedi")
            else:
                continue

            evidence = [f"kayıtlı durum: {capability.status}"]
            evidence.extend(f"bilinen sınır: {limit}" for limit in capability.limits[:3])
            has_benchmark = capability.benchmark_score is not None
            if has_benchmark:
                evidence.append(f"son ölçüm: {capability.benchmark_score:.2f}")
            shaped, reason = shape_of(source, status=capability.status,
                                      has_benchmark=has_benchmark)
            gaps.append(Gap(
                key=_key(source, capability.name), source=source, title=title,
                severity=weight, capability=capability.name, evidence=evidence,
                experiment_shaped=shaped, shape_reason=reason))
        return gaps

    def _from_tasks(self) -> list[Gap]:
        """Task kinds that keep being given up on."""
        try:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT kind, COUNT(*) c FROM tasks WHERE state='quarantined'"
                    " GROUP BY kind ORDER BY c DESC LIMIT 5").fetchall()
        except sqlite3.Error as exc:
            log.debug("görev sinyali okunamadı: %s", exc)
            return []

        shaped, reason = shape_of(FROM_TASKS)
        gaps = []
        for row in rows:
            if row["kind"] in FIXTURE_KINDS:
                # A runner that raises by design is not the system giving up.
                continue
            count = int(row["c"])
            gaps.append(Gap(
                key=_key(FROM_TASKS, row["kind"]), source=FROM_TASKS,
                title=f"'{row['kind']}' görevleri karantinaya düşüyor",
                severity=min(1.0, 0.4 + count * 0.15),
                evidence=[f"{count} karantina kaydı"],
                experiment_shaped=shaped, shape_reason=reason))
        return gaps

    def _from_errors(self) -> list[Gap]:
        """The same error text appearing repeatedly is a pattern, not bad luck."""
        cutoff = time.time() - LOOKBACK_DAYS * 86400
        try:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT source, kind, message FROM events"
                    " WHERE level='error' AND ts >= ? LIMIT 500", (cutoff,)).fetchall()
        except sqlite3.Error as exc:
            log.debug("hata sinyali okunamadı: %s", exc)
            return []

        counter = Counter(f"{row['source']}·{row['kind']}" for row in rows)
        shaped, reason = shape_of(FROM_ERRORS)
        gaps = []
        for signature, count in counter.most_common(5):
            if count < REPEAT_THRESHOLD:
                continue
            gaps.append(Gap(
                key=_key(FROM_ERRORS, signature), source=FROM_ERRORS,
                title=f"'{signature}' hatası tekrar ediyor",
                severity=min(1.0, 0.3 + count * 0.08),
                evidence=[f"son {LOOKBACK_DAYS} günde {count} kez"],
                experiment_shaped=shaped, shape_reason=reason))
        return gaps

    def _from_experiments(self) -> list[Gap]:
        try:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT purpose, notes FROM experiments WHERE state IN ('failed','discarded')"
                    " ORDER BY updated DESC LIMIT 5").fetchall()
        except sqlite3.Error as exc:
            log.debug("deney sinyali okunamadı: %s", exc)
            return []

        shaped, reason = shape_of(FROM_EXPERIMENTS)
        gaps = []
        for row in rows:
            gaps.append(Gap(
                key=_key(FROM_EXPERIMENTS, row["purpose"]), source=FROM_EXPERIMENTS,
                title=f"Başarısız deney: {row['purpose'][:70]}",
                severity=0.35,
                evidence=[row["notes"][:200] or "gerekçe kaydedilmedi"],
                experiment_shaped=shaped, shape_reason=reason))
        return gaps
