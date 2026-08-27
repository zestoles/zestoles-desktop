"""Decides which brain answers a given message.

Tier 1 is free and unlimited but weaker at planning, code and multi-step reasoning.
Tier 2 is strong but metered against the subscription cap. The router spends Tier 2
only where it would change the answer — a greeting does not need Claude, a design
decision does.

Deliberately a cheap heuristic rather than a model call: routing must not itself
cost a call. Every verdict carries its reasoning so /durum can explain it.

Three things learned from real messages:

  - Turkish is typed without diacritics as often as not ("gerekce", "nasil"), so
    all matching happens on a folded form.
  - Turkish is agglutinative: "mimari" arrives as "mimariyi", "dezavantaj" as
    "dezavantajlariyla". Exact word matching misses almost everything, so markers
    are stems matched by prefix. Short stems that collide with common words
    ("git" in "gitti", "cok" in "çok") are listed as exact matches instead.
  - "X mi, Y mi?" comparisons carry no keyword at all, yet are among the clearest
    signals that an answer needs real reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import fold  # re-exported: routing and memory must fold identically

LOCAL = "local"
CLOUD = "cloud"


class Lexicon:
    """Marker set matched by stem prefix, with an exact-only escape hatch."""

    __slots__ = ("stems", "exact")

    def __init__(self, stems: set[str], exact: set[str] | None = None) -> None:
        self.stems = frozenset(stems)
        self.exact = frozenset(exact or ())

    def hits(self, tokens: list[str]) -> list[str]:
        found: set[str] = set()
        for token in tokens:
            if token in self.exact:
                found.add(token)
                continue
            for stem in self.stems:
                if token.startswith(stem):
                    found.add(stem)
                    break
        return sorted(found)


# Technical subject matter — the answer needs real domain competence.
DOMAIN = Lexicon(
    stems={
        "kod", "script", "fonksiyon", "modul", "class", "refactor", "algoritma",
        "veritaban", "database", "sorgu", "query", "regex", "roblox", "luau",
        "python", "javascript", "typescript", "html", "docker", "server",
        "sunucu", "istemci", "client", "datastore", "profileservice", "mimari",
        "architect", "framework", "kutuphane", "library", "performans",
        "guvenlik", "security", "optimiz", "benchmark", "deploy", "pipeline",
        "async", "thread", "cache", "embedding", "agent", "ajan", "olceklen",
        "scalab", "endpoint", "webhook", "protokol", "protocol", "sifrele",
        "encrypt", "latency", "gecikme", "bellek", "memory",
    },
    exact={"api", "sql", "css", "git", "lua", "ui", "ux", "llm", "gpu", "cpu",
           "model", "prompt", "token"},
)

# Verbs asking for produced work rather than a recalled fact.
WORK = Lexicon(
    stems={
        "yaz", "olustur", "tasarla", "planla", "duzelt", "onar", "gelistir",
        "optimiz", "iyilestir", "analiz", "incele", "degerlendir", "arastir",
        "kurul", "entegre", "uygula", "kodla", "implement", "design", "build",
        "review", "explain", "refactor", "otomatiklestir", "donustur",
    },
    exact={"kur", "kurar", "kuralim", "kurmak", "kurmali", "kuracagiz", "plan",
           "yap", "yapalim", "yapmali", "yapmaliyim", "coz", "cozelim", "fix",
           "test", "kodu"},
)

# Debugging and failure reports.
TROUBLE = Lexicon(
    stems={
        "hata", "bug", "debug", "calismi", "patladi", "bozuldu", "exception",
        "traceback", "sorun", "neden", "nicin", "basarisiz", "yavas", "takiliy",
        "donuyor", "kilitleni",
    },
    exact={"coktu", "cokme", "cokuyor", "error", "failed", "crash"},
)

# Comparison and recommendation language.
DECISION = Lexicon(
    stems={
        "hangi", "karsilastir", "fark", "avantaj", "dezavantaj", "tavsiye",
        "oner", "gerekce", "mantikli", "secmeli", "kullanmal", "yapmal",
        "tercih", "uygun", "alternatif", "yerine",
    },
    exact={"yoksa", "daha", "iyi", "vs", "veya", "mi", "mu"},
)

_CODEY = re.compile(
    r"```|\b\w+\.(py|js|ts|lua|luau|json|md|ps1|cmd|sql|html|css|yaml|toml)\b"
    r"|[A-Za-z]:\\|\bdef \w+|\bfunction \w+|\bimport \w+"
)

# Turkish question particles in folded form (mı→mi, mü→mu). Two in one sentence
# means "X mi, Y mi?". Listed explicitly — a prefix rule would read "mimari" as one.
PARTICLES = frozenset({
    "mi", "mu", "misin", "musun", "miyim", "muyum", "miyiz", "muyuz",
    "midir", "mudur", "miydi", "muydu",
})

SMALL_TALK = frozenset({
    "selam", "merhaba", "naber", "gunaydin", "iyi", "geceler", "aksamlar",
    "nasilsin", "tesekkur", "tesekkurler", "sag", "sagol", "tamam", "peki",
    "evet", "hayir", "ok", "okey", "eyvallah", "gorusuruz", "hosca", "kal",
    "devam", "et", "hadi", "olur", "super", "harika", "jarvis",
})


@dataclass(slots=True)
class Decision:
    tier: str
    reason: str
    score: int
    threshold: int

    @property
    def near_miss(self) -> bool:
        """Local answered, but only just — worth offering the stronger tier."""
        return self.tier == LOCAL and self.score >= self.threshold - 1


class Router:
    def __init__(self, *, min_chars: int = 240, threshold: int = 3, mode: str = "auto") -> None:
        self.min_chars = min_chars
        self.threshold = threshold
        self.mode = mode

    def decide(self, text: str) -> Decision:
        if self.mode == LOCAL:
            return Decision(LOCAL, "yapılandırma yerel katmana sabitlenmiş", 0, self.threshold)
        if self.mode == CLOUD:
            return Decision(CLOUD, "yapılandırma Claude katmanına sabitlenmiş", 99, self.threshold)

        stripped = text.strip()
        folded = fold(stripped)
        tokens = re.findall(r"\w+", folded)
        words = set(tokens)

        score = 0
        reasons: list[str] = []

        def add(points: int, label: str) -> None:
            nonlocal score
            score += points
            reasons.append(label)

        domain = DOMAIN.hits(tokens)
        if domain:
            add(2 + min(len(domain) - 1, 2), "teknik konu: " + ", ".join(domain[:3]))

        work = WORK.hits(tokens)
        if work:
            add(2 + min(len(work) - 1, 2), "iş üretimi: " + ", ".join(work[:2]))

        trouble = TROUBLE.hits(tokens)
        if trouble:
            add(2, "hata/teşhis: " + trouble[0])

        # Comparison and recommendation stack: "hangisi daha iyi, X mi Y mi" is
        # both, and together they clear the threshold on their own.
        particles = sum(1 for token in tokens if token in PARTICLES)
        decisions = DECISION.hits(tokens)
        if particles >= 2 or "yoksa" in words:
            add(2, "karşılaştırmalı soru")
            if len(decisions) >= 2:
                add(1, "karar/öneri dili")
        elif len(decisions) >= 2:
            add(2, "karar/öneri isteniyor: " + ", ".join(decisions[:2]))

        if _CODEY.search(stripped):
            add(2, "kod veya dosya yolu içeriyor")

        if len(stripped) >= self.min_chars:
            add(2, "uzun mesaj")

        if stripped.count("?") >= 2 or stripped.count("\n") >= 2:
            add(1, "çok parçalı istek")

        if len(stripped) < 40 and score <= 0:
            if words & SMALL_TALK:
                add(-5, "kısa sohbet")
            else:
                add(-2, "kısa ve sinyalsiz")

        tier = CLOUD if score >= self.threshold else LOCAL
        why = "; ".join(reasons) or "belirgin sinyal yok"
        return Decision(tier, f"{why} (skor {score}/{self.threshold})", score, self.threshold)
