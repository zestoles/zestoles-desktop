"""Turning a conversation into memory worth keeping.

Storing whole transcripts is not memory, it is a recording: it grows without bound
and buries the few durable facts under thousands of forgettable ones. So after a
session the local model reads the conversation and extracts what should still be
true next month, and those become notes in the vault.

The local model does this rather than Claude, deliberately — it runs after every
session, including at 04:00, and must cost nothing.

## Why provenance is enforced here

The first real run of this module recorded, as knowledge, that Roblox offers a
service called AsyncResultStorage which serialises datastore writes. No such
service exists; the local model invented it mid-answer and the distiller filed the
invention in the vault, where it would have been recalled later as an established
fact and repeated with growing confidence.

That is the failure mode that makes a self-learning system get worse over time
instead of better, and it cannot be fixed by asking the model to be careful. So
every extracted fact must say where it came from, and anything JARVIS asserted on
its own is refused. What the user states about their own work is theirs to state.
What JARVIS believes about the world has to be verified against a source before it
becomes memory — that path arrives with the research loop, carrying a citation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ..brain.local import LocalBrain
from ..text import slugify

log = logging.getLogger("jarvis.memory.distill")

USER_SOURCED = "kullanici"
SELF_SOURCED = "jarvis"
# A session summary is a record of what was said, not a claim that any of it was
# true. It bypasses accept() by design — the daily log must stay complete — so it
# carries its own provenance class and is surfaced as unverified on recall.
SUMMARY_SOURCED = "oturum-ozeti"
# Produced by a specialist agent (S3). Passing verification means another agent
# found no fault in it — not that it was checked against the world. Declared here
# rather than imported from jarvis.agents so memory stays the lower layer.
AGENT_SOURCED = "ajan"

# Survived S4's cross-source verification: independent publishers supported it,
# each with a quote, and none contradicted it. The only label in the system that
# means "checked" rather than "believed" — hence its deliberate absence below.
VERIFIED_SOURCED = "dogrulanmis"

#: Sources whose content must never be presented to the model as established fact.
UNVERIFIED_SOURCES = frozenset({SELF_SOURCED, SUMMARY_SOURCED, AGENT_SOURCED})

#: The role the assistant writes its tool records under. Spelled out here rather
#: than imported from jarvis.assistant, so memory stays the lower layer -- the
#: same reason AGENT_SOURCED is declared above. A test pins the two spellings.
TOOL_ROLE = "arac"

KINDS = ("kisi", "proje", "deneyim", "bilgi")

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "kaynak": {"type": "string", "enum": [USER_SOURCED, SELF_SOURCED]},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "kaynak", "title", "content"],
            },
        }
    },
    "required": ["facts"],
}

SYSTEM = """\
You extract durable memory from a conversation between a user and ZESTOLES.

For every fact you must record who it came from, in the "kaynak" field:

  kullanici  The user stated it, decided it, or explicitly confirmed it. Facts about
             his projects, preferences, constraints and choices are his.
  jarvis     ZESTOLES asserted it on its own — a technical claim, a recommendation,
             an explanation. Mark these honestly even when they sound correct.

Getting "kaynak" right matters more than extracting many facts. A ZESTOLES claim
filed as a user fact becomes an unverified belief that the system will later treat
as established. When you are unsure who a statement came from, mark it jarvis.

Categories:
  kisi     lasting facts about the user: preferences, working style, constraints
  proje    projects, their goals, their decisions, their current state
  deneyim  what was actually tried and what actually happened
  bilgi    technical knowledge

Rules:
  - Resolve relative dates. "yarın" becomes the actual date.
  - One fact per entry, in Turkish, specific enough to stand alone.
  - Titles are short noun phrases, not sentences.
  - Skip greetings, small talk, anything true only during this conversation, and
    anything you are unsure of.
  - An empty list is the correct answer for a conversation that taught nothing.
  - Never invent detail that was not said.

Lines beginning "[arac kaydi]" are not speech. They are the recorded result of
tools that really ran -- what happened, not what anyone claimed about it. When
they disagree with the surrounding prose, the record is what happened.

They are still not the user. A fact drawn from a tool record is "jarvis", never
"kullanici", however reliable the record is: the user did not state it, and marking
it as his would file a machine measurement as something he vouched for. Only
what the user says is "kullanici".
"""


TITLE_MAX = 60


def tidy_title(raw: str) -> str:
    """Small models write titles as full sentences however firmly asked not to.

    A title becomes a filename and a note's identity across sessions, so trailing
    punctuation and runaway length are trimmed here rather than hoped away in the
    prompt.
    """
    title = " ".join(raw.strip().split()).rstrip(" .!?:;,")
    if len(title) > TITLE_MAX:
        cut = title[:TITLE_MAX].rsplit(" ", 1)[0]
        title = (cut or title[:TITLE_MAX]).rstrip(" .!?:;,")
    return title


@dataclass(slots=True)
class Fact:
    kind: str
    title: str
    content: str
    tags: list[str]
    source: str = USER_SOURCED

    @property
    def slug(self) -> str:
        return slugify(self.title)


def accept(fact: Fact) -> tuple[bool, str]:
    """Whether a fact may enter long-term memory, and why not when it may not.

    Kept as a pure function so the rule can be tested without a model: this gate
    is the only thing standing between a confident hallucination and permanent
    memory, and it must not depend on the model behaving well.
    """
    if fact.kind not in KINDS:
        return False, f"bilinmeyen tür: {fact.kind}"
    if not fact.title.strip() or not fact.content.strip():
        return False, "boş başlık veya içerik"
    if fact.source == SELF_SOURCED:
        return False, "ZESTOLES'in kendi iddiası — doğrulanmadan hafızaya girmez"
    return True, ""


def _speaker(role: str) -> str:
    """Who said a line -- and whether anybody did.

    A tool record is not speech: it is what the machine measured. Labelling it
    "JARVIS" would file a measurement as a JARVIS claim, and `accept()` refuses
    those precisely because they are the model's own word. Getting this wrong
    would not leak one bad fact into the vault; it would teach the distiller that
    the most reliable lines in a session are the least believable ones.
    """
    if role == "user":
        return "Kullanıcı"
    if role == TOOL_ROLE:
        return "[arac kaydi]"
    return "ZESTOLES"


def _transcript(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{_speaker(m['role'])}: {m['content']}" for m in messages
    )


def distill(
    brain: LocalBrain, messages: list[dict[str, str]], *, today: str
) -> tuple[list[Fact], list[str]]:
    """Returns the facts that may be stored, and the reasons others were refused."""
    if not messages:
        return [], []

    prompt = f"Bugünün tarihi: {today}\n\n[Konuşma]\n{_transcript(messages)}"
    try:
        raw = brain.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            schema=FACT_SCHEMA,
            temperature=0.2,
        )
    except OSError as exc:
        log.warning("distillation failed: %s", exc)
        return [], []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("distillation returned non-JSON: %s", raw[:200])
        return [], []

    kept: list[Fact] = []
    refused: list[str] = []
    for item in payload.get("facts", []):
        fact = Fact(
            kind=str(item.get("kind", "")).strip(),
            title=tidy_title(str(item.get("title", ""))),
            content=str(item.get("content", "")).strip(),
            tags=[str(t).strip() for t in item.get("tags", []) if str(t).strip()],
            source=str(item.get("kaynak", SELF_SOURCED)).strip() or SELF_SOURCED,
        )
        ok, why = accept(fact)
        if ok:
            kept.append(fact)
        else:
            refused.append(f"{fact.title or '(başlıksız)'}: {why}")
            log.info("hafızaya alınmadı — %s (%s)", fact.title, why)
    return kept, refused


SUMMARY_SYSTEM = """\
Summarise this conversation in Turkish, in at most three sentences. State what was
discussed and what was decided. No preamble, no bullet points, no closing line.
Do not repeat technical claims as if they were established fact.
"""


def summarise(brain: LocalBrain, messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""
    try:
        return brain.chat(
            [{"role": "system", "content": SUMMARY_SYSTEM},
             {"role": "user", "content": _transcript(messages)}],
            temperature=0.3,
        ).strip()
    except OSError as exc:
        log.warning("summary failed: %s", exc)
        return ""
