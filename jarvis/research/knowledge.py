"""The gate between research and permanent memory.

This is the first path in the whole system by which something JARVIS learned on its
own may become long-term knowledge. Everything before it — agent output, session
summaries, its own confident assertions — is stored, if at all, as unverified.

A claim gets through only if all of these hold:

  the orchestration that produced it passed S3's verification gate
  independent publishers actually support it, each with a quote from their page
  no source contradicts it
  it is not merely important-sounding: importance affects nothing here

What is written keeps its evidence. The note carries every supporting URL, the
timestamp it was read at, and the sentence that source was relied on for. That is
what makes the entry auditable later — and auditable by a person, since the vault
is markdown and opens in any editor.

A claim that fails is not discarded silently. It is written into the run's report
with the reason, because "three sources disagree about this" is a finding, and
losing it would leave the same question to be researched again next month.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from ..text import slugify
from .crossverify import CONTRADICTED, Claim, VERIFIED

log = logging.getLogger("jarvis.research.knowledge")

#: Provenance for a claim that survived cross-source verification. Deliberately
#: not in memory.distill.UNVERIFIED_SOURCES — this is the one label that means the
#: system checked, rather than the system believes.
VERIFIED_SOURCED = "dogrulanmis"


@dataclass(slots=True)
class Admission:
    admitted: list[Claim] = field(default_factory=list)
    refused: list[tuple[Claim, str]] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {"kabul": len(self.admitted), "red": len(self.refused)}


def admit(claims: list[Claim], *, run_verified: bool) -> Admission:
    """Decide which claims may become knowledge. Pure — testable without a model."""
    result = Admission()
    for claim in claims:
        if not run_verified:
            result.refused.append((claim, "orkestrasyon doğrulama kapısını geçmedi"))
            continue
        if claim.status == CONTRADICTED:
            result.refused.append((claim, f"kaynaklar çelişiyor — {claim.note}"))
            continue
        if claim.status != VERIFIED:
            result.refused.append((claim, claim.note or "yeterli bağımsız destek yok"))
            continue
        if not any(ref.quote for ref in claim.supported_by):
            # Support with no quoted passage is an assertion wearing a citation.
            result.refused.append((claim, "destekleyen kaynaklarda alıntı yok"))
            continue
        result.admitted.append(claim)
    return result


def render_note(topic: str, claims: list[Claim], *, question: str = "") -> str:
    """The markdown body written to the vault, evidence included."""
    lines: list[str] = []
    if question:
        lines += [f"**Araştırma sorusu:** {question}", ""]

    for claim in claims:
        lines.append(f"## {claim.text}")
        lines.append("")
        lines.append(f"*{claim.note}*")
        lines.append("")
        lines.append("Kaynaklar:")
        for ref in claim.supported_by:
            lines.append(f"- [{ref.title or ref.domain}]({ref.url}) "
                         f"· {ref.tier} · okundu {ref.fetched_at}")
            if ref.quote:
                lines.append(f"  > {ref.quote}")
        lines.append("")

    lines += [
        "---",
        "",
        f"Bu not {datetime.now():%Y-%m-%d %H:%M} tarihinde otonom araştırmayla",
        "oluşturuldu. Her iddia en az iki bağımsız yayıncı tarafından desteklendi",
        "ve hiçbiri çelişmedi. Yine de kaynaklar yanılıyor olabilir — bağlantılar",
        "yukarıda, kontrol edilebilir.",
    ]
    return "\n".join(lines)


def write_knowledge(memory, topic: str, claims: list[Claim], *, question: str = "") -> str | None:
    """Write admitted claims to the vault with their provenance chain intact."""
    if memory is None or not claims:
        return None
    try:
        note = memory.vault.append(
            "bilgi",
            topic,
            render_note(topic, claims, question=question),
            slug=slugify(topic),
            source=VERIFIED_SOURCED,
        )
    except (OSError, ValueError) as exc:
        log.warning("doğrulanmış bilgi yazılamadı: %s", exc)
        return None

    try:
        memory.reindex()
    except Exception as exc:  # noqa: BLE001 - the note is written; the index can lag
        log.warning("indeks güncellenemedi: %s", exc)
    log.info("doğrulanmış bilgi kaydedildi: %s (%s iddia)", note.title, len(claims))
    return note.title


def refusal_report(admission: Admission) -> str:
    """What did not make it in, and why. Kept because a rejection is a finding."""
    if not admission.refused:
        return ""
    lines = ["Hafızaya alınmayan iddialar:"]
    for claim, reason in admission.refused:
        lines.append(f"  · {claim.text[:110]} — {reason}")
    return "\n".join(lines)
