"""What JARVIS can actually do, tracked as data rather than believed.

A system that improves itself has to know what it is improving from, and asking a
model "what are you good at?" produces an answer shaped by the question. So
capabilities are a table: each one has a status somebody set for a reason, a
benchmark score if it has ever been measured, the date that measurement was taken,
and an explicit list of what it cannot do.

The honest part is `limits`. A capability with no recorded limits is not a
capability without limits — it is one nobody has examined, and `last_verified`
being old is itself a signal the gap detector reads.

The seed below describes this codebase as it actually stands after S5, including
the things that do not exist. Recording `voice: missing` is what lets the system
notice the absence later; a registry containing only what works can never find a
gap.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.improve.capabilities")

WORKING = "calisiyor"
PARTIAL = "kismi"
MISSING = "yok"
BROKEN = "bozuk"
UNKNOWN = "bilinmiyor"

STATUSES = (WORKING, PARTIAL, MISSING, BROKEN, UNKNOWN)

#: How much each status contributes to being worth improving. Missing beats broken
#: only because a broken capability may still be doing part of its job.
GAP_WEIGHT = {MISSING: 1.0, BROKEN: 0.9, PARTIAL: 0.6, UNKNOWN: 0.5, WORKING: 0.0}

#: Beyond this, a working capability counts as unverified rather than working.
STALE_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS capabilities (
    name            TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'bilinmiyor',
    version         TEXT NOT NULL DEFAULT '0.1',
    benchmark_name  TEXT NOT NULL DEFAULT '',
    benchmark_score REAL,
    last_verified   REAL,
    limits          TEXT NOT NULL DEFAULT '[]',
    dependencies    TEXT NOT NULL DEFAULT '[]',
    skills          TEXT NOT NULL DEFAULT '[]',
    notes           TEXT NOT NULL DEFAULT '',
    updated         REAL NOT NULL
);
"""


@dataclass(slots=True)
class Capability:
    name: str
    title: str
    status: str = UNKNOWN
    version: str = "0.1"
    benchmark_name: str = ""
    benchmark_score: float | None = None
    last_verified: float | None = None
    limits: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    notes: str = ""
    updated: float = 0.0

    @property
    def age_days(self) -> float | None:
        if not self.last_verified:
            return None
        return (time.time() - self.last_verified) / 86400

    @property
    def stale(self) -> bool:
        age = self.age_days
        return age is None or age > STALE_DAYS

    @property
    def gap_weight(self) -> float:
        """How much this capability's state argues for working on it."""
        weight = GAP_WEIGHT.get(self.status, 0.5)
        if self.status == WORKING and self.stale:
            # Working-but-unverified is not the same as working.
            weight = 0.25
        return weight

    def summary(self) -> str:
        verified = (datetime.fromtimestamp(self.last_verified).strftime("%d.%m.%Y")
                    if self.last_verified else "hiç")
        score = f" · {self.benchmark_score:.2f}" if self.benchmark_score is not None else ""
        return f"{self.title} [{self.status}]{score} · son doğrulama: {verified}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "title": self.title, "status": self.status,
            "version": self.version, "benchmark_name": self.benchmark_name,
            "benchmark_score": self.benchmark_score, "last_verified": self.last_verified,
            "limits": self.limits, "dependencies": self.dependencies,
            "skills": self.skills, "notes": self.notes,
        }


#: The system as it actually is after S5. Absences are listed on purpose — a
#: registry containing only what works can never notice what is missing.
SEED: tuple[dict[str, Any], ...] = (
    {"name": "memory.retrieval", "title": "Hafıza geri çağırma", "status": WORKING,
     "dependencies": ["ollama", "bge-m3"],
     "limits": ["embedding modeli kapalıysa yalnızca anahtar kelime araması kalır",
                "kasa elle düzenlenirse indeks tazelenene kadar eski kalır"]},
    {"name": "research.web", "title": "Web araştırması", "status": WORKING,
     "dependencies": ["github", "stackexchange", "discourse", "wikipedia"],
     "limits": ["SearXNG kapalıyken genel web araması yok",
                "JavaScript ile yüklenen içerik okunamıyor",
                "kaynak başına bir model çağrısı — 5 kaynak yaklaşık 60 saniye"]},
    {"name": "agents.orchestration", "title": "Ajan orkestrasyonu", "status": WORKING,
     "dependencies": ["ollama"],
     "limits": ["doğrulayıcı üreticiyle aynı model — paylaşılan kör nokta görülmez",
                "koşu başına tek model; katman değiştirmek yaklaşık 70 saniye"]},
    {"name": "task.planning", "title": "Görev planlama", "status": WORKING,
     "limits": ["plan en fazla beş adım", "adımlar sıralı, paralellik yok"]},
    {"name": "lab.sandbox", "title": "Sandbox ve deney", "status": WORKING,
     "limits": ["OS seviyesinde izolasyon değil",
                "resolve ile open arasında TOCTOU penceresi var"]},
    {"name": "code.writing", "title": "Kod yazma", "status": PARTIAL,
     "dependencies": ["ollama"],
     "limits": ["yerel model var olmayan API adı uydurabiliyor",
                "büyük dosyalarda bağlam sınırına takılıyor"]},
    {"name": "file.operations", "title": "Dosya işlemleri", "status": PARTIAL,
     "limits": ["yalnızca sandbox içinde", "üretim dosyalarına yazma kapalı"]},
    {"name": "data.analysis", "title": "Veri analizi", "status": PARTIAL,
     "limits": ["yapısal veri için özel araç yok", "grafik üretemiyor"]},
    {"name": "roblox.development", "title": "Roblox geliştirme", "status": PARTIAL,
     "limits": ["Studio ile doğrudan bağlantı yok", "Luau kodu test edilemiyor"]},
    {"name": "browser.automation", "title": "Tarayıcı otomasyonu", "status": MISSING,
     "limits": ["hiç kurulmadı"]},
    {"name": "voice.io", "title": "Ses girişi ve çıkışı", "status": MISSING,
     "limits": ["hiç kurulmadı", "Python 3.14'te ses kütüphaneleri geride kalabilir"]},
    {"name": "image.understanding", "title": "Görüntü anlama", "status": MISSING,
     "limits": ["hiç kurulmadı", "gemma4 görme destekliyor ama bağlanmadı"]},
    {"name": "ui.control_center", "title": "Kontrol merkezi arayüzü", "status": MISSING,
     "limits": ["hiç kurulmadı"]},
)


class CapabilityRegistry:
    def __init__(self, db_path: Path, *, seed: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)
        if seed:
            self.seed()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def seed(self) -> int:
        """Insert the known capabilities. Never overwrites a record already there."""
        added = 0
        for entry in SEED:
            if self.get(entry["name"]) is None:
                self.upsert(Capability(
                    name=entry["name"], title=entry["title"],
                    status=entry.get("status", UNKNOWN),
                    limits=list(entry.get("limits", [])),
                    dependencies=list(entry.get("dependencies", [])),
                ))
                added += 1
        if added:
            log.info("%s yetenek kaydedildi", added)
        return added

    def upsert(self, capability: Capability) -> Capability:
        capability.updated = time.time()
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO capabilities (name, title, status, version, benchmark_name,"
                " benchmark_score, last_verified, limits, dependencies, skills, notes, updated)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET title=excluded.title,"
                " status=excluded.status, version=excluded.version,"
                " benchmark_name=excluded.benchmark_name,"
                " benchmark_score=excluded.benchmark_score,"
                " last_verified=excluded.last_verified, limits=excluded.limits,"
                " dependencies=excluded.dependencies, skills=excluded.skills,"
                " notes=excluded.notes, updated=excluded.updated",
                (capability.name, capability.title, capability.status, capability.version,
                 capability.benchmark_name, capability.benchmark_score,
                 capability.last_verified,
                 json.dumps(capability.limits, ensure_ascii=False),
                 json.dumps(capability.dependencies, ensure_ascii=False),
                 json.dumps(capability.skills, ensure_ascii=False),
                 capability.notes, capability.updated),
            )
        return capability

    def record_benchmark(self, name: str, *, benchmark: str, score: float,
                         status: str | None = None) -> Capability | None:
        """A measurement is the only thing that refreshes last_verified."""
        capability = self.get(name)
        if capability is None:
            return None
        capability.benchmark_name = benchmark
        capability.benchmark_score = score
        capability.last_verified = time.time()
        if status:
            capability.status = status
        return self.upsert(capability)

    def set_status(self, name: str, status: str, *, note: str = "") -> Capability | None:
        if status not in STATUSES:
            raise ValueError(f"bilinmeyen durum: {status}")
        capability = self.get(name)
        if capability is None:
            return None
        capability.status = status
        if note:
            capability.notes = note
        return self.upsert(capability)

    def add_limit(self, name: str, limit: str) -> Capability | None:
        capability = self.get(name)
        if capability is None or limit in capability.limits:
            return capability
        capability.limits.append(limit)
        return self.upsert(capability)

    def get(self, name: str) -> Capability | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM capabilities WHERE name=?", (name,)).fetchone()
        return _to_capability(row) if row else None

    def list(self, *, status: str | None = None) -> list[Capability]:
        query = "SELECT * FROM capabilities"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY name"
        with closing(self._conn()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_to_capability(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) c FROM capabilities GROUP BY status").fetchall()
        return {row["status"]: int(row["c"]) for row in rows}


def _to_capability(row: sqlite3.Row) -> Capability:
    def load(column):
        try:
            return json.loads(row[column])
        except (json.JSONDecodeError, TypeError):
            return []

    return Capability(
        name=row["name"], title=row["title"], status=row["status"],
        version=row["version"], benchmark_name=row["benchmark_name"],
        benchmark_score=row["benchmark_score"], last_verified=row["last_verified"],
        limits=load("limits"), dependencies=load("dependencies"),
        skills=load("skills"), notes=row["notes"], updated=float(row["updated"]),
    )
