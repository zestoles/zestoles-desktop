"""Running an assistant turn from the task queue.

A turn is synchronous and holds the HTTP request that started it. That is fine
for "what is in this folder" and wrong for anything that takes minutes: the page
waits, the socket stays open, and closing the tab loses the work. The queue
already solves that problem for autonomous work -- it survives restarts, retries
with backoff and quarantines what keeps failing -- so the fix is to let the same
turn run there instead of teaching the interface to wait longer.

## Why this lives here and not in `autonomy/runners.py`

The dependency graph runs `assistant -> ... -> autonomy`, one way. A runner
defined next to the others would make the autonomy layer import the assistant
and close that loop, so registration happens from this side: the queue stores a
name, and this module is what makes the name mean something.

## The gate

Anything reachable from the queue is reachable by whatever can put a row in the
queue -- and an improvement task, a routine, or a future agent can all do that.
Without a check, `assistant.ask` would be a way for the autonomous side to run
shell commands and write files under the user's own trust, which is precisely
the boundary `test_tools.TestSeparationFromTheAgentGate` exists to keep.

So the origin is checked first, before the payload is read and before the model
is asked anything: only a task the user queued themselves may drive a turn.
Everything else raises, and raising is honest -- the task fails loudly, gets its
attempts counted and ends up quarantined, rather than quietly doing nothing.

## Nobody is there to say yes

`AssistantService` gives the loop no `approve` callback, so a MEDIUM or HIGH risk
call stops the turn and hands back the pending step instead of running. In the
interface a person answers it. Here there is nobody, so the turn ends and the
task reports what it did not do. Approving on the user's behalf because the
request came from them earlier would make the risk tier decorative -- the thing
the whole tool layer is arranged to prevent.
"""

from __future__ import annotations

import logging

from ..autonomy import runners
from ..autonomy.tasks import Priority

log = logging.getLogger("jarvis.assistant.background")

#: The task kind. Stored in the queue, so it outlives this build.
KIND = "assistant.ask"

#: The only origin allowed to drive a turn. Matches what `enqueue` writes.
USER_ORIGIN = "user"

#: A queued turn competes with the live conversation for one lock, and losing
#: that race is normal rather than a fault. The default of three attempts would
#: quarantine a perfectly good task because the user happened to be talking, so
#: it gets a longer rope; the backoff still caps at an hour.
MAX_ATTEMPTS = 12


def enqueue(queue, message: str, *, priority: int = Priority.USER) -> int | None:
    """Put a request in the queue. Returns the task id, or None if refused.

    USER priority by default: the machine belongs to the user, and work they
    asked for out loud outranks anything JARVIS decided to do by itself.
    """
    text = message.strip()
    if not text:
        raise ValueError("bos gorev mesaji")
    title = text if len(text) <= 60 else text[:57] + "..."
    return queue.add(KIND, title, payload={"mesaj": text}, priority=priority,
                     origin=USER_ORIGIN, max_attempts=MAX_ATTEMPTS)


def summarise(turn) -> str:
    """What the queue records as the result. Read from the steps, never the prose."""
    if turn.pending is not None:
        return (f"onay gerekiyor ({turn.pending.tool}) - arka planda onay "
                "alinamaz, ayni isi arayuzden isteyin")
    if turn.stopped:
        return f"durduruldu: {turn.stopped}"
    used = ", ".join(turn.used_tools)
    reply = (turn.reply or "(cevap yok)").strip()
    if not turn.succeeded:
        failed = ", ".join(step.tool for step in turn.failures) or "?"
        return f"basarisiz ({failed}): {reply}"
    return f"{reply} [{used}]" if used else reply


@runners.runner(KIND)
def _ask(ctx) -> str:
    """Answer a request the user queued instead of waiting for."""
    if ctx.task.origin != USER_ORIGIN:
        # Before the payload is read, before the model is asked. See the note.
        raise PermissionError(
            f"asistan turu yalnizca kullanicinin kuyrukladigi gorevde calisir "
            f"(gelen kaynak: {ctx.task.origin!r})")

    service = ctx.assistant
    if service is None:
        raise RuntimeError("asistan bu surecte acik degil - gorev calistirilamaz")

    message = str(ctx.task.payload.get("mesaj", ""))
    turn = service.run_queued(message, should_stop=ctx.should_stop)
    result = summarise(turn)
    log.info("kuyruktan tur bitti (#%s): %s", ctx.task.id, turn.summary())
    return result


__all__ = ["KIND", "USER_ORIGIN", "MAX_ATTEMPTS", "enqueue", "summarise"]
