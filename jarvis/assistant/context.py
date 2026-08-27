"""Keeping the conversation inside what the model can actually read.

The rolling window used to be a message count: keep the last N. That holds until
someone pastes a file into the chat, and then twelve "messages" is a prompt the
model quietly truncates from the front. What sits at the front is the system
prompt -- where every rule about not claiming a tool succeeded lives. Losing the
conversation is a nuisance; losing that is the model being told nothing about
how it is supposed to behave, at exactly the moment the context is confusing.

So the budget is characters, and what happens when it is exceeded is decided
here rather than by whatever the model does when it overflows.

## The rules, and why each one

**Whole turns.** Dropping a question and keeping its answer leaves the model
reading a reply to something it cannot see. That is worse than dropping both:
it invites confident nonsense about what was being discussed.

**The newest turns always survive**, budget or not. A context that dropped the
thing just said would have JARVIS answering the previous question, which reads
as the assistant losing its mind rather than running out of room.

**What was dropped is admitted**, in a line marked as system. Never as the user:
the user did not say the context was trimmed, and a note about missing context
filed as their words is exactly the attribution bug the memory layer already had
once. `NOTICE_ROLE` exists so that mistake cannot be made by accident here.

## What is deliberately not here

No model-written summary. Dropped turns have already been written through to
memory -- the durable copy lives in the layer built for it, with its provenance
intact -- so paying for a second model call to compress what was just said would
add seconds to every long conversation and a new way to be wrong. The notice
says where the rest went instead of pretending to carry it.
"""

from __future__ import annotations

import logging

log = logging.getLogger("jarvis.assistant.context")

#: Roughly what a local model can read without the front of the prompt starting
#: to disappear, with room left for the system prompt and the new request.
DEFAULT_BUDGET_CHARS = 12000

#: Kept whatever the budget says. Two exchanges is the least that still lets a
#: follow-up question ("peki ya onu sil") mean anything.
KEEP_RECENT_TURNS = 2

#: How the trimming announces itself. Not "user" and not "assistant", so a note
#: about missing context can never be read back as something somebody said.
NOTICE_ROLE = "system"

#: A single message longer than this is trimmed rather than dropped: a pasted
#: file is usually the subject of the conversation, and losing it entirely is
#: worse than losing its middle.
SINGLE_MESSAGE_CAP = 6000

TRIM_MARK = "\n[… kısaltıldı]"


def prune(history: list[dict[str, str]], *,
          budget_chars: int = DEFAULT_BUDGET_CHARS,
          keep_recent_turns: int = KEEP_RECENT_TURNS,
          ) -> tuple[list[dict[str, str]], int]:
    """Fit a conversation into a character budget. Returns (kept, dropped).

    `dropped` counts messages, not turns, because that is what the notice
    reports and what a caller would log. The input is never modified.
    """
    if not history:
        return [], 0

    pairs = _as_turns(history)
    floor = max(0, int(keep_recent_turns))
    budget = max(0, int(budget_chars))

    kept: list[list[dict[str, str]]] = []
    used = 0
    for index in range(len(pairs) - 1, -1, -1):
        turn = [_capped(message) for message in pairs[index]]
        size = sum(len(m["content"]) for m in turn)
        guaranteed = len(pairs) - index <= floor
        if not guaranteed and kept and used + size > budget:
            break
        kept.append(turn)
        used += size
    kept.reverse()

    result = [message for turn in kept for message in turn]
    dropped = len(history) - sum(len(turn) for turn in kept)
    if dropped > 0:
        result.insert(0, {
            "role": NOTICE_ROLE,
            "content": (
                f"[bağlam] Bu konuşmanın önceki {dropped} mesajı yer açmak için "
                "çıkarıldı. İçerikleri hafızada; gerekiyorsa kullanıcıya sor ya "
                "da hatırladığını varsayma."),
        })
        log.debug("bağlam budandı: %s mesaj çıkarıldı", dropped)
    return result, dropped


# ------------------------------------------------------------------ internals
def _as_turns(history: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Group messages into turns so a question and its answer move together.

    Deliberately tolerant: a history that starts with an assistant message, or
    has two user messages in a row, is grouped as best it can rather than
    raising. Real conversations get there -- a refused confirmation writes a
    user line with no reply -- and pruning is not the place to be strict about it.
    """
    turns: list[list[dict[str, str]]] = []
    for message in history:
        if message.get("role") == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def _capped(message: dict[str, str]) -> dict[str, str]:
    content = message.get("content", "")
    if len(content) <= SINGLE_MESSAGE_CAP:
        return dict(message)
    head = SINGLE_MESSAGE_CAP - len(TRIM_MARK)
    return {**message, "content": content[:head] + TRIM_MARK}


__all__ = ["prune", "DEFAULT_BUDGET_CHARS", "KEEP_RECENT_TURNS", "NOTICE_ROLE",
           "SINGLE_MESSAGE_CAP"]
