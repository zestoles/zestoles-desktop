"""The wire vocabulary: what a watcher can be told, and in what shape.

## Why these are named types and not log lines

The event log speaks in free text because a human reads it. A UI cannot: it needs
to know that *this* is an agent starting so it can light the right indicator, and
"Orkestrasyon başladı: ..." is not something to pattern-match on. Renaming a log
message must not break an interface.

So the log's `(source, kind)` pairs are translated once, here, into a closed set of
types with structured payloads. The table below is the whole contract. A pair with
no entry does not reach the wire — silence is better than an event a client cannot
interpret, and an unmapped pair is a gap to fill deliberately rather than a
surprise to handle defensively at the far end.

## Envelope

Every message carries a sequence number, and sequence numbers are monotonic across
the whole stream. That single field answers three questions a client would
otherwise have to guess at: are these in order, did I miss any, and is what I hold
still current.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------ event types
SYSTEM_STATE_CHANGED = "system_state_changed"
AGENT_STARTED = "agent_started"
AGENT_FINISHED = "agent_finished"
TASK_STARTED = "task_started"
TASK_FINISHED = "task_finished"
RESEARCH_STARTED = "research_started"
RESEARCH_SOURCE_FOUND = "research_source_found"
RESEARCH_FINISHED = "research_finished"
EXPERIMENT_STARTED = "experiment_started"
EXPERIMENT_FINISHED = "experiment_finished"
BENCHMARK_STARTED = "benchmark_started"
BENCHMARK_FINISHED = "benchmark_finished"
PROMOTION_CANDIDATE = "promotion_candidate"
PROMOTION_COMPLETED = "promotion_completed"
ROLLBACK = "rollback"
MEMORY_CREATED = "memory_created"
MEMORY_RECALLED = "memory_recalled"
SELF_IMPROVEMENT_STARTED = "self_improvement_started"
SELF_IMPROVEMENT_FINISHED = "self_improvement_finished"
ERROR = "error"
WAITING_FOR_USER = "waiting_for_user"

#: The assistant loop: one user request, and the tools it runs to answer it.
#: Separate from TASK_* because those are queued background work — these are
#: what the person watching asked for a moment ago, and an interface shows
#: the two in different places.
ASSISTANT_TURN_STARTED = "assistant_turn_started"
ASSISTANT_TURN_FINISHED = "assistant_turn_finished"
TOOL_STARTED = "tool_started"
TOOL_FINISHED = "tool_finished"

#: A spoken answer that keeps synthesising after the response went out. Each
#: frame carries the turn token the first clip travelled with, so an interface
#: can drop clips of a turn the user already interrupted.
VOICE_CLIPS = "voice_clips"

#: Protocol frames, distinct from domain events so a client can switch on `type`
#: without a second field.
SNAPSHOT = "snapshot"
HEARTBEAT = "heartbeat"
DROPPED = "dropped"

EVENT_TYPES = frozenset({
    SYSTEM_STATE_CHANGED, AGENT_STARTED, AGENT_FINISHED, TASK_STARTED,
    TASK_FINISHED, RESEARCH_STARTED, RESEARCH_SOURCE_FOUND, RESEARCH_FINISHED,
    EXPERIMENT_STARTED, EXPERIMENT_FINISHED, BENCHMARK_STARTED,
    BENCHMARK_FINISHED, PROMOTION_CANDIDATE, PROMOTION_COMPLETED, ROLLBACK,
    MEMORY_CREATED, MEMORY_RECALLED, SELF_IMPROVEMENT_STARTED,
    SELF_IMPROVEMENT_FINISHED, ERROR, WAITING_FOR_USER,
    ASSISTANT_TURN_STARTED, ASSISTANT_TURN_FINISHED, TOOL_STARTED,
    TOOL_FINISHED, VOICE_CLIPS,
})

PROTOCOL_TYPES = frozenset({SNAPSHOT, HEARTBEAT, DROPPED})
ALL_TYPES = EVENT_TYPES | PROTOCOL_TYPES

#: (event source, event kind) → wire type. Anything absent stays off the wire.
TRANSLATION: dict[tuple[str, str], str] = {
    ("agent", "run.start"): AGENT_STARTED,
    ("agent", "run.done"): AGENT_FINISHED,
    ("agent", "run.error"): ERROR,
    ("agent", "start"): AGENT_STARTED,
    ("agent", "done"): AGENT_FINISHED,
    ("agent", "error"): ERROR,

    ("task", "start"): TASK_STARTED,
    ("task", "done"): TASK_FINISHED,
    ("task", "error"): ERROR,
    ("task", "retry"): TASK_FINISHED,
    ("task", "quarantine"): TASK_FINISHED,
    ("task", "queued"): TASK_STARTED,

    ("research", "start"): RESEARCH_STARTED,
    ("research", "sources"): RESEARCH_SOURCE_FOUND,
    ("research", "claims"): RESEARCH_SOURCE_FOUND,
    ("research", "done"): RESEARCH_FINISHED,
    ("research", "error"): ERROR,
    ("research", "empty"): RESEARCH_FINISHED,
    ("research", "injection"): ERROR,
    ("research", "offtopic"): RESEARCH_FINISHED,

    ("lab", "experiment.opened"): EXPERIMENT_STARTED,
    ("lab", "experiment.settled"): BENCHMARK_FINISHED,
    ("lab", "experiment.failed"): EXPERIMENT_FINISHED,
    ("lab", "experiment.discarded"): EXPERIMENT_FINISHED,
    ("lab", "promotion.refused"): EXPERIMENT_FINISHED,
    ("lab", "promotion.done"): PROMOTION_COMPLETED,
    ("lab", "promotion.failed"): ERROR,
    ("lab", "rollback"): ROLLBACK,
    ("lab", "recover"): ROLLBACK,

    ("improve", "experiment.started"): SELF_IMPROVEMENT_STARTED,
    ("improve", "hypothesis.new"): SELF_IMPROVEMENT_STARTED,
    ("improve", "hypothesis.duplicate"): SELF_IMPROVEMENT_FINISHED,
    ("improve", "hypothesis.open"): SELF_IMPROVEMENT_FINISHED,
    ("improve", "improvement.promoted"): PROMOTION_COMPLETED,
    ("improve", "improvement.failed"): SELF_IMPROVEMENT_FINISHED,

    ("scheduler", "start"): SYSTEM_STATE_CHANGED,
    ("scheduler", "stop"): SYSTEM_STATE_CHANGED,
    ("scheduler", "pause"): SYSTEM_STATE_CHANGED,
    ("scheduler", "resume"): SYSTEM_STATE_CHANGED,
    ("scheduler", "recover"): SYSTEM_STATE_CHANGED,
    ("scheduler", "error"): ERROR,
    ("policy", "stance"): SYSTEM_STATE_CHANGED,
    ("system", "snapshot"): SYSTEM_STATE_CHANGED,

    ("assistant", "turn.start"): ASSISTANT_TURN_STARTED,
    ("assistant", "turn.done"): ASSISTANT_TURN_FINISHED,
    ("assistant", "turn.cancelled"): ASSISTANT_TURN_FINISHED,
    ("assistant", "turn.stalled"): ASSISTANT_TURN_FINISHED,
    ("assistant", "turn.failed"): ERROR,
    ("assistant", "tool.start"): TOOL_STARTED,
    ("assistant", "tool.done"): TOOL_FINISHED,
    ("assistant", "tool.failed"): ERROR,
    ("assistant", "tool.denied"): TOOL_FINISHED,
    ("assistant", "tool.confirm"): WAITING_FOR_USER,
    ("assistant", "decision.rejected"): TOOL_FINISHED,

    ("ses", "parcalar"): VOICE_CLIPS,
}


def translate(source: str, kind: str) -> str | None:
    """The wire type for a log event, or None when it does not belong on the wire."""
    return TRANSLATION.get((source, kind))


@dataclass(frozen=True, slots=True)
class Envelope:
    seq: int
    type: str
    ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "type": self.type,
                "ts": round(self.ts, 3), "payload": self.payload}

    def to_json(self) -> str:
        # default=str so an unexpected object degrades to its repr rather than
        # killing the send loop and taking the connection with it.
        return json.dumps(self.as_dict(), ensure_ascii=False, default=str)


def payload_from_event(event) -> dict[str, Any]:
    """Structured fields from a log event, message kept as a human-readable label."""
    data = getattr(event, "data", None)
    return {
        "source": event.source,
        "kind": event.kind,
        "level": event.level,
        "label": event.message,
        "data": dict(data) if isinstance(data, dict) else {},
    }
