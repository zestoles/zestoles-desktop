"""Checking whether a result is actually any good.

This is the layer the project exists to have. A multi-agent system without it is a
machine for producing confident text at scale, and confident text is exactly what
poisoned memory the first time.

Verification runs in two passes with different failure modes, deliberately:

  Mechanical checks are code. They catch empty output, refusals dressed as answers,
  and truncation. They cannot be talked out of a verdict, and they cost nothing.

  The model pass is adversarial. It receives the criteria, the work, and a list of
  identifiers extracted from the work, and is asked to assume the work is wrong
  until each criterion is shown satisfied.

An honest limitation, stated because it would otherwise be invisible: by default
the verifier runs on the same model that produced the work. That catches
inconsistency, omission and overreach, but not a blind spot the two share — a model
confident that a service exists will be equally confident when asked to check.
Cross-model verification is available via config and costs a full model swap
(~70s on this machine, measured), which is why it is off by default and worth
turning on for night runs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..memory.distill import USER_SOURCED
from .base import AgentContext, AgentResult, run_agent
from .roles import VERIFIER

log = logging.getLogger("jarvis.agents.verify")

# CamelCase names, dotted calls, and import targets — the shapes an invented API
# takes. Deliberately over-inclusive: a false positive costs the verifier one line
# of attention, a false negative costs a poisoned memory.
_IDENTIFIER = re.compile(
    r"\b(?:[A-Z][a-z0-9]+){2,}\b"
    r"|\b[a-z_][\w]*\.[A-Za-z_]\w*\s*\("
    r"|\bimport\s+([\w.]+)"
)

_REFUSAL = re.compile(
    r"^\s*(üzgünüm|maalesef|yapamam|bilmiyorum|bu konuda yardımcı olamam"
    r"|i'?m sorry|i can'?t|as an ai)\b",
    re.IGNORECASE,
)

MIN_USEFUL_CHARS = 40


#: Identifier judgements that mean "do not act on this name yet".
DOUBTFUL = frozenset({"emin degil", "yok"})


@dataclass(slots=True)
class Verdict:
    ok: bool
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    note: str = ""
    mechanical: list[str] = field(default_factory=list)
    checked_by: str = ""
    doubtful_names: list[str] = field(default_factory=list)
    confirmed_names: list[str] = field(default_factory=list)
    unmet: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[str]:
        """Why the verdict failed. Structured findings only.

        `issues` is deliberately absent. Asked for problems in free text, the
        verifier reliably also wrote there the criteria it had confirmed and the
        identifiers it had vouched for — "bu ölçüt sağlanmıştır" filed as a
        complaint. Every entry blocked the verdict, so being thorough failed the
        run. The gate is now the structured fields, which cannot be phrased into
        meaning their opposite; free text is kept as commentary.
        """
        names = [f"'{n}' isminin gerçekliği doğrulanamadı" for n in self.doubtful_names]
        return self.mechanical + self.unmet + names

    def summary(self) -> str:
        if self.ok:
            return f"doğrulandı (güven {self.confidence:.0%})"
        reasons = "; ".join(self.blocking[:3]) or self.note or "gerekçe verilmedi"
        return f"doğrulanmadı: {reasons}"


def suspicious_identifiers(text: str, *, limit: int = 25) -> list[str]:
    """Names in the text that would be worth confirming actually exist."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _IDENTIFIER.finditer(text):
        name = (match.group(1) or match.group(0)).strip().rstrip("(").strip()
        if name and name not in seen:
            seen.add(name)
            found.append(name)
        if len(found) >= limit:
            break
    return found


def mechanical_checks(result: AgentResult) -> list[str]:
    """Failures that need no judgement — and so cannot be argued away."""
    problems: list[str] = []
    text = (result.output or "").strip()

    if not result.ok:
        problems.append(f"adım başarısız: {result.error}")
        return problems
    if len(text) < MIN_USEFUL_CHARS:
        problems.append(f"çıktı fazla kısa ({len(text)} karakter)")
    if _REFUSAL.match(text):
        problems.append("çıktı bir reddetme cümlesiyle başlıyor")
    if text.endswith("…[kısaltıldı]"):
        problems.append("çıktı uzunluk sınırında kesildi")
    return problems


def known_from_user(memory, name: str) -> bool:
    """Whether the user has used this name in something they stated.

    A 9B model is not a reliable oracle on whether a given library exists, so it
    correctly refuses to vouch for real ones — ProfileService gets flagged
    alongside inventions. But a name that appears in a user-sourced memory note is
    a name the user brought to the system, and the user is not hallucinating his
    own tech stack. Only USER_SOURCED notes count: clearing a name against
    something an agent wrote would let an invention confirm itself.
    """
    if memory is None or not name:
        return False
    try:
        hits = memory.search(name, limit=4)
    except Exception as exc:  # noqa: BLE001 - a failed lookup just means unknown
        log.debug("isim hafızada aranamadı (%s): %s", name, exc)
        return False
    needle = name.casefold()
    return any(
        hit.source == USER_SOURCED and needle in hit.text.casefold()
        for hit in hits
    )


def verify(
    ctx: AgentContext,
    *,
    goal: str,
    criteria: list[str],
    result: AgentResult,
    model: str | None = None,
    memory=None,
) -> Verdict:
    """Judge a finished result. Never raises: a broken verifier means unverified."""
    mechanical = mechanical_checks(result)
    if not result.ok:
        return Verdict(False, 0.0, [], "adım başarısız", mechanical, checked_by="mekanik")

    identifiers = suspicious_identifiers(result.output)
    criteria_text = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(criteria)) or "  (yok)"
    identifier_text = ", ".join(identifiers) if identifiers else "(hiçbiri bulunmadı)"

    instruction = (
        f"[Hedef]\n{goal}\n\n"
        f"[Karşılanması gereken ölçütler]\n{criteria_text}\n\n"
        f"[İncelenecek çalışma]\n{result.output}\n\n"
        f"[Çalışmadan çıkarılan isimler — her biri gerçekten var mı?]\n{identifier_text}"
    )

    verify_ctx = ctx
    if model and model != ctx.model:
        verify_ctx = AgentContext(
            brain=ctx.brain, events=ctx.events, grant=ctx.grant, model=model,
            should_stop=ctx.should_stop, run_id=ctx.run_id,
        )

    judged = run_agent(VERIFIER, instruction, verify_ctx)
    if not judged.ok:
        log.warning("doğrulayıcı çalışmadı: %s", judged.error)
        return Verdict(False, 0.0, [], f"doğrulayıcı çalışmadı: {judged.error}",
                       mechanical, checked_by="—")

    try:
        payload = json.loads(judged.output)
    except json.JSONDecodeError:
        return Verdict(False, 0.0, [], "doğrulayıcı çözümlenemeyen cevap verdi",
                       mechanical, checked_by=verify_ctx.model)

    issues = [str(i).strip() for i in payload.get("issues", []) if str(i).strip()]
    doubtful, confirmed = _split_identifiers(payload.get("identifiers", []))
    if doubtful and memory is not None:
        cleared = [name for name in doubtful if known_from_user(memory, name)]
        if cleared:
            log.info("kullanıcının kendi kullandığı isimler temize çıktı: %s",
                     ", ".join(cleared))
            doubtful = [name for name in doubtful if name not in cleared]
            confirmed.extend(cleared)
    unmet = _unmet_criteria(payload.get("criteria", []), criteria)
    confidence = _clamp(payload.get("confidence", 0.0))

    # A name the verifier could not vouch for blocks on its own — that is the whole
    # point of the check, since the run that invented AsyncResultStorage met every
    # other criterion it was given. The verifier's own ok flag still counts, so a
    # concern it can only express in prose is not lost.
    verdict = Verdict(
        ok=bool(payload.get("ok")) and not mechanical and not unmet and not doubtful,
        confidence=confidence,
        issues=issues,
        note=str(payload.get("note", "")).strip(),
        mechanical=mechanical,
        checked_by=verify_ctx.model,
        doubtful_names=doubtful,
        confirmed_names=confirmed,
        unmet=unmet,
    )
    return verdict


def _unmet_criteria(raw: object, criteria: list[str]) -> list[str]:
    """Criteria the verifier judged unsatisfied, named rather than numbered."""
    unmet: list[str] = []
    if not isinstance(raw, list):
        return unmet
    for entry in raw:
        if not isinstance(entry, dict) or entry.get("met", True):
            continue
        try:
            index = int(entry.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        label = criteria[index - 1] if 1 <= index <= len(criteria) else f"ölçüt {index}"
        why = str(entry.get("why", "")).strip()
        unmet.append(f"{label}" + (f" — {why}" if why else ""))
    return unmet


def _split_identifiers(raw: object) -> tuple[list[str], list[str]]:
    doubtful: list[str] = []
    confirmed: list[str] = []
    if not isinstance(raw, list):
        return doubtful, confirmed
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        verdict = str(entry.get("verdict", "")).strip().casefold()
        (doubtful if verdict in DOUBTFUL else confirmed).append(name)
    return doubtful, confirmed


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
