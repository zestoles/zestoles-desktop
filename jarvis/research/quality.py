"""How much weight a source gets.

Not all sources are equal and pretending otherwise is how a forum comment ends up
carrying the same weight as vendor documentation. But a hardcoded list of "good
sites" ages badly and encodes whoever wrote it, so scoring leans on structural
signals that stay true: who publishes the page, whether the project is alive, how
much scrutiny it has had, and how old it is.

The score is never used to decide truth on its own. It decides how much
independent corroboration a claim needs before it may enter memory — a vendor's
own documentation can stand closer to alone than a blog post can.

Independence is tracked separately and matters more than any single score. Three
articles that all rephrase the same announcement are one source, and treating them
as three is the most common way a research system convinces itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .providers import SearchHit

OFFICIAL = "resmi"
REPUTABLE = "guvenilir"
COMMUNITY = "topluluk"
UNKNOWN = "bilinmeyen"

TIER_WEIGHT = {OFFICIAL: 1.0, REPUTABLE: 0.75, COMMUNITY: 0.5, UNKNOWN: 0.3}

#: Reference and standards hosts. Short on purpose — structural signals below do
#: most of the work, and a long list is a maintenance burden that silently rots.
_REPUTABLE_DOMAINS = frozenset({
    "wikipedia.org", "arxiv.org", "stackoverflow.com", "stackexchange.com",
    "github.com", "gitlab.com", "python.org", "mozilla.org", "w3.org",
    "ietf.org", "rfc-editor.org", "kernel.org", "iso.org",
})

# First-party roots used by the bundled local stack. User-generated hosting
# platforms are intentionally excluded even when they host official projects.
_OFFICIAL_DOMAINS = frozenset({"ollama.com", "qwenlm.ai"})

#: Host shapes that indicate first-party documentation for whatever they document.
#: Deliberately excludes forum hosts. devforum.roblox.com sits on the vendor's
#: domain but its content is written by users, and treating it as first-party
#: documentation let a single forum thread verify a claim on its own.
_OFFICIAL_PATTERNS = (
    re.compile(r"^docs?\."), re.compile(r"^developer\."),
    re.compile(r"^create\."), re.compile(r"^api\."), re.compile(r"\.readthedocs\.io$"),
    re.compile(r"^documentation\."), re.compile(r"^learn\."),
)

_FORUM_PATTERNS = (re.compile(r"^devforum\."), re.compile(r"^forum"),
                   re.compile(r"^community\."), re.compile(r"^discuss"))

_LOW_SIGNAL = re.compile(
    r"(pinterest|quora|medium\.com/@|answers\.|ehow|geeksforgeeks|w3schools"
    r"|blogspot|wordpress\.com|\.blogspot\.)", re.IGNORECASE)


@dataclass(slots=True)
class Quality:
    tier: str
    score: float
    reasons: list[str]

    @property
    def weight(self) -> float:
        return self.score


def classify_domain(domain: str) -> tuple[str, list[str]]:
    if not domain:
        return UNKNOWN, ["alan adı yok"]
    reasons: list[str] = []

    if domain in _OFFICIAL_DOMAINS or any(
            domain.endswith("." + known) for known in _OFFICIAL_DOMAINS):
        return OFFICIAL, [f"birinci taraf ürün alan adı ({domain})"]

    for pattern in _FORUM_PATTERNS:
        if pattern.search(domain):
            return COMMUNITY, [f"resmi alan adında topluluk forumu ({domain})"]

    for pattern in _OFFICIAL_PATTERNS:
        if pattern.search(domain):
            return OFFICIAL, [f"birinci taraf dokümantasyon görünümü ({domain})"]

    for known in _REPUTABLE_DOMAINS:
        if domain == known or domain.endswith("." + known):
            return REPUTABLE, [f"bilinen referans kaynağı ({known})"]

    if domain.endswith((".edu", ".gov", ".edu.tr", ".gov.tr")):
        return REPUTABLE, ["akademik veya resmi alan adı"]

    if _LOW_SIGNAL.search(domain):
        reasons.append("düşük sinyalli içerik çiftliği görünümü")
        return UNKNOWN, reasons

    return COMMUNITY, [f"topluluk kaynağı ({domain})"]


def score_hit(hit: SearchHit, *, now: datetime | None = None) -> Quality:
    tier, reasons = classify_domain(hit.domain)
    score = TIER_WEIGHT[tier]

    if hit.provider == "github":
        stars = int(hit.extra.get("stars", 0) or 0)
        if stars >= 1000:
            score += 0.15
            reasons.append(f"{stars} yıldız")
        elif stars >= 100:
            score += 0.08
            reasons.append(f"{stars} yıldız")
        elif stars < 10:
            score -= 0.1
            reasons.append(f"az ilgi görmüş ({stars} yıldız)")
        if hit.extra.get("archived"):
            score -= 0.2
            reasons.append("depo arşivlenmiş")

    if hit.provider == "hackernews":
        points = int(hit.extra.get("points", 0) or 0)
        if points >= 100:
            score += 0.1
            reasons.append(f"{points} puan tartışma")
        elif points < 5:
            score -= 0.05
            reasons.append("neredeyse hiç tartışılmamış")

    age_penalty, age_reason = _age_adjustment(hit.published, now)
    score += age_penalty
    if age_reason:
        reasons.append(age_reason)

    return Quality(tier, max(0.05, min(1.0, score)), reasons)


def _age_adjustment(published: str, now: datetime | None) -> tuple[float, str]:
    if not published:
        return 0.0, ""
    moment = _parse_date(published)
    if moment is None:
        return 0.0, ""
    reference = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    years = (reference - moment).days / 365.25
    if years > 5:
        return -0.2, f"{years:.0f} yıllık kaynak"
    if years > 2:
        return -0.1, f"{years:.0f} yıllık kaynak"
    if years < 1:
        return 0.05, "güncel"
    return 0.0, ""


def _parse_date(value: str) -> datetime | None:
    text = value.strip().replace("Z", "+00:00")
    for parse in (datetime.fromisoformat,):
        try:
            return parse(text)
        except ValueError:
            continue
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        try:
            return datetime(*(int(part) for part in match.groups()), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def independent_domains(domains: list[str]) -> int:
    """Distinct publishers, with mirrors of the same site folded together.

    Three pages under docs.example.com are one voice. Counting them as three is the
    most common way a research system talks itself into certainty.
    """
    roots = set()
    for domain in domains:
        if not domain:
            continue
        parts = domain.split(".")
        roots.add(".".join(parts[-3:]) if parts[-2] in ("co", "com", "org", "gov", "edu")
                  and len(parts) > 2 else ".".join(parts[-2:]))
    return len(roots)
