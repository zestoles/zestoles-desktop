"""Scoring what is worth doing, without inventing certainty about it.

## The failure this module exists to prevent

"Bu Roblox oyun fikri aylık $5000 kazandırabilir" is the most dangerous sentence
the system can produce. It is specific, it is actionable, it sounds researched, and
there is nothing behind it. Stored as a fact it becomes a plan; repeated a few
times it becomes a belief.

So a number never travels alone. Every estimate carries where it came from, how
sure the system is, what it assumed, and what would have to be true for it to be
wrong — and an estimate whose basis is a guess is marked speculative and refused
entry to memory, however confident the prose around it sounds.

    basis=measured   we ran it and recorded the number
    basis=sourced    an outside source said it, with a citation
    basis=estimated  derived from something sourced, by stated arithmetic
    basis=guess      nobody knows; this is a placeholder for a real answer

## Scoring is arithmetic

Seven dimensions, each in 0..1, each with its own basis. The composite is a
weighted sum whose weights come from the owner's stated preferences. No model is
asked for an overall verdict, because "this seems like a good opportunity" is
exactly the judgement that cannot be checked.

## Some money is not worth making

The revenue objective is bounded before it is scored. Fraud, spam, manipulation,
platform-rule violations and anything illegal are not low-scoring opportunities —
they are not opportunities, and they are refused before a number is attached.
Making that a filter rather than a penalty matters: a penalty can be outweighed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..text import fold

log = logging.getLogger("jarvis.improve.opportunity")

MEASURED = "olculdu"
SOURCED = "kaynakli"
ESTIMATED = "turetildi"
GUESS = "tahmin"

BASES = (MEASURED, SOURCED, ESTIMATED, GUESS)

#: Bases weak enough that the number must never be stored as knowledge.
SPECULATIVE_BASES = frozenset({GUESS, ESTIMATED})

DIMENSIONS = ("revenue", "feasibility", "time", "resource_fit",
              "competition", "risk", "confidence")

DIMENSION_LABEL = {
    "revenue": "gelir potansiyeli",
    "feasibility": "uygulanabilirlik",
    "time": "zaman maliyeti (düşük iyi)",
    "resource_fit": "kaynak uyumu",
    "competition": "rekabet (düşük iyi)",
    "risk": "risk (düşük iyi)",
    "confidence": "güven",
}

#: Dimensions where a low number is the good outcome.
INVERTED = frozenset({"time", "competition", "risk"})

DEFAULT_WEIGHTS = {
    "revenue": 1.0, "feasibility": 1.2, "time": 0.8, "resource_fit": 1.0,
    "competition": 0.6, "risk": 1.0, "confidence": 1.2,
}

#: Categories that are refused outright rather than scored badly. A penalty can be
#: outweighed by a large enough revenue number; a filter cannot.
FORBIDDEN_CATEGORIES = frozenset({
    "yasadisi", "dolandiricilik", "spam", "manipulasyon",
    "kural-ihlali", "gizlilik-ihlali", "guvenlik-ihlali",
})

_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("yasadisi", re.compile(
        r"\b(yasa\s*disi|illegal|korsan|piracy|crack|warez|kacak)\b")),
    ("dolandiricilik", re.compile(
        r"\b(dolandir|sahte\s*(hesap|yorum|kullanici)|fake\s*(review|account)"
        r"|scam|phishing|oltalama)\b")),
    ("spam", re.compile(r"\b(spam|toplu\s*mesaj|mass\s*dm|bot\s*yorum|kitlesel\s*e-?posta)\b")),
    ("manipulasyon", re.compile(
        r"\b(manipul|yaniltici|misleading|astroturf|pump\s*and\s*dump|sahte\s*etkilesim)\b")),
    # No trailing \b on the Turkish stems: "ihlal" is nearly always inflected
    # ("ihlali", "ihlalini"), and a word boundary after it matches none of them.
    ("kural-ihlali", re.compile(
        r"\b(tos\s*ihlal|kural\s*ihlal|bypass\s*(the\s*)?(rules|tos|ban)"
        r"|ban\s*atlat|exploit\s*(the\s*)?platform|hile\s*ile\s*siralama)")),
    ("gizlilik-ihlali", re.compile(
        r"\b(kisisel\s*veri\s*(topla|sat)|scrape\s*personal|izinsiz\s*veri"
        r"|kullanici\s*verisini\s*sat)\b")),
    ("guvenlik-ihlali", re.compile(
        r"\b(sifre\s*kir|password\s*crack|ddos|zafiyet\s*somur|keylog|malware)\b")),
)

#: Wording that turns a possibility into a promise. No trailing \b: Turkish verbs
#: are always inflected, and "kazan" followed by "dırır" fails a word boundary —
#: which is how "kesinlikle kazandırır" passed this filter in a live run.
_GUARANTEE = re.compile(
    r"\b(garanti|kesinlikle\s*kazan|kesin\s*gelir|mutlaka\s*kazan"
    r"|guaranteed|risk-?free|risksiz\s*kazan|kesin\s*kar"
    r"|kazandirir|kazandiracak|kazandirmasi\s*kesin)")


@dataclass(slots=True)
class Estimate:
    """A number and everything needed to judge whether to believe it."""

    value: float | None
    unit: str = ""
    basis: str = GUESS
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    uncertainty: str = ""

    @property
    def speculative(self) -> bool:
        """True when this must not be recorded as fact."""
        return (self.basis in SPECULATIVE_BASES
                or not self.evidence
                or self.confidence < 0.5)

    @property
    def label(self) -> str:
        if self.value is None:
            return "bilinmiyor"
        number = f"{self.value:,.0f} {self.unit}".strip()
        if self.speculative:
            return f"~{number} (spekülatif, {self.basis})"
        return f"{number} ({self.basis}, güven {self.confidence:.0%})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value, "unit": self.unit, "basis": self.basis,
            "confidence": round(self.confidence, 3), "evidence": self.evidence,
            "assumptions": self.assumptions, "uncertainty": self.uncertainty,
            "speculative": self.speculative,
        }


@dataclass(slots=True)
class Dimension:
    name: str
    value: float
    basis: str = GUESS
    rationale: str = ""

    @property
    def effective(self) -> float:
        """Normalised so that higher is always better."""
        clamped = max(0.0, min(1.0, self.value))
        return 1.0 - clamped if self.name in INVERTED else clamped


@dataclass(slots=True)
class Opportunity:
    title: str
    description: str = ""
    category: str = "urun"
    capability: str = ""
    dimensions: dict[str, Dimension] = field(default_factory=dict)
    estimates: dict[str, Estimate] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def dimension(self, name: str) -> Dimension:
        return self.dimensions.get(name, Dimension(name, 0.0, GUESS, "değer verilmedi"))

    @property
    def unmeasured(self) -> list[str]:
        return [name for name in DIMENSIONS if self.dimension(name).basis == GUESS]

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "description": self.description,
            "category": self.category, "capability": self.capability,
            "dimensions": {n: {"value": d.value, "basis": d.basis,
                               "rationale": d.rationale}
                           for n, d in self.dimensions.items()},
            "estimates": {n: e.as_dict() for n, e in self.estimates.items()},
            "notes": self.notes,
        }


@dataclass(slots=True)
class Screening:
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "uygun" if self.allowed else "; ".join(self.reasons)


def screen(opportunity: Opportunity) -> Screening:
    """Refuse whole categories before any number is attached to them."""
    reasons: list[str] = []

    if fold(opportunity.category) in FORBIDDEN_CATEGORIES:
        reasons.append(f"yasaklı kategori: {opportunity.category}")

    haystack = fold(f"{opportunity.title} {opportunity.description}")
    for label, pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(haystack):
            reasons.append(f"yasaklı içerik işareti: {label}")

    if _GUARANTEE.search(haystack):
        # Not refused for being wrong, refused for being a promise. Rewriting it
        # as a possibility costs nothing; leaving it costs the distinction between
        # "may earn" and "will earn".
        reasons.append("gelir garantisi gibi ifade edilmiş — olasılık olarak yazılmalı")

    return Screening(not reasons, reasons)


@dataclass(slots=True)
class Verdict:
    opportunity: Opportunity
    screening: Screening
    composite: float = 0.0
    ranked: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def worth_pursuing(self) -> bool:
        return self.ranked and self.screening.allowed

    def summary(self) -> str:
        if not self.screening.allowed:
            return f"elendi: {self.screening.summary()}"
        speculative = sum(1 for e in self.opportunity.estimates.values() if e.speculative)
        note = f" · {speculative} spekülatif tahmin" if speculative else ""
        return f"puan {self.composite:.2f}{note}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "opportunity": self.opportunity.as_dict(),
            "allowed": self.screening.allowed,
            "screening": self.screening.reasons,
            "composite": round(self.composite, 3),
            "ranked": self.ranked,
            "warnings": self.warnings,
        }


def score(opportunity: Opportunity, *, weights: dict[str, float] | None = None,
          minimum: float = 0.45) -> Verdict:
    """Weighted arithmetic over the dimensions. No model is consulted."""
    screening = screen(opportunity)
    verdict = Verdict(opportunity, screening)
    if not screening.allowed:
        return verdict

    active = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_weight = sum(active.get(name, 0.0) for name in DIMENSIONS)
    if total_weight <= 0:
        verdict.warnings.append("ağırlık toplamı sıfır")
        return verdict

    running = 0.0
    for name in DIMENSIONS:
        dimension = opportunity.dimension(name)
        running += dimension.effective * active.get(name, 0.0)
    verdict.composite = running / total_weight

    unmeasured = opportunity.unmeasured
    if unmeasured:
        verdict.warnings.append(
            "dayanaksız boyutlar: " + ", ".join(DIMENSION_LABEL[n] for n in unmeasured))
    speculative = [name for name, est in opportunity.estimates.items() if est.speculative]
    if speculative:
        verdict.warnings.append("spekülatif tahminler: " + ", ".join(speculative))

    verdict.ranked = verdict.composite >= minimum
    return verdict


def rank(opportunities: list[Opportunity], *, weights: dict[str, float] | None = None,
         minimum: float = 0.45) -> list[Verdict]:
    verdicts = [score(o, weights=weights, minimum=minimum) for o in opportunities]
    verdicts.sort(key=lambda v: (-v.composite if v.screening.allowed else 1.0))
    return verdicts


def recordable_estimates(opportunity: Opportunity) -> dict[str, Estimate]:
    """The estimates solid enough to be written down as more than a hypothesis."""
    return {name: est for name, est in opportunity.estimates.items()
            if not est.speculative}
