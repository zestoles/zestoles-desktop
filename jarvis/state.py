"""What JARVIS is doing *right now*, in one structured place.

## Why this module imports nothing

It is the leaf of the import graph. It knows about no subsystem, no config, no
event log — only the standard library. Everything else may import it; it may import
nothing of ours.

That constraint is the entire point. A shared-state module that reaches back into
the layers above it becomes the hub of a cycle, and the first symptom is an
`ImportError` in a module that has not been touched for weeks. Keeping it a leaf
means any layer can report into it and any consumer can read from it without either
knowing the other exists.

## State is not history

There are two questions a watcher asks, and conflating them makes both worse:

    "what happened?"        EventLog — append-only, ordered, persistent, queryable
    "what is happening?"    SharedState — current values, overwritten, in memory

An event is a fact about a moment that stays true forever. A state field is the
answer to a question asked now, and yesterday's answer is worthless. Storing state
in the event log means replaying thousands of rows to learn one current value;
storing history in state means losing it on the next update.

The bridge runs one way: events are projected into state. State never writes
events, because a value that has just been overwritten has nothing to say about
what happened.

## Concurrency

The autonomy scheduler writes from its own thread while the terminal reads from
the main one, and later a websocket will read from a third. Every access takes a
lock, and readers get copies rather than the live dictionaries — a consumer that
mutates what it was handed cannot corrupt what everyone else sees.

`version` increments on every change, so a client that reconnects can tell whether
the snapshot it holds is stale without comparing the whole tree.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------- sections
SYSTEM = "system"
BRAIN = "brain"
MEMORY = "memory"
AUTONOMY = "autonomy"
AGENTS = "agents"
RESEARCH = "research"
LAB = "lab"
IMPROVE = "improve"
SESSION = "session"

SECTIONS = (SYSTEM, BRAIN, MEMORY, AUTONOMY, AGENTS, RESEARCH, LAB, IMPROVE, SESSION)

# -------------------------------------------------------------------- activities
#: What the system is visibly doing. Named here rather than in the UI so both ends
#: agree on the vocabulary without importing each other.
IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
RESEARCHING = "researching"
CODING = "coding"
AUTONOMOUS = "autonomous"
EXPERIMENTING = "experimenting"
WAITING_USER = "waiting_for_user"
ERROR = "error"

ACTIVITIES = (IDLE, LISTENING, THINKING, RESEARCHING, CODING, AUTONOMOUS,
              EXPERIMENTING, WAITING_USER, ERROR)


@dataclass(slots=True)
class Section:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    updated: float = 0.0
    version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "data": dict(self.data),
                "updated": self.updated, "version": self.version}


Watcher = Callable[[str, dict[str, Any]], None]


class SharedState:
    """Current runtime state. Thread-safe, in memory, deliberately not persistent.

    Not persistent because it is not supposed to survive: after a restart the only
    honest answer to "what is JARVIS doing" is "starting up". A state file restored
    from disk would confidently describe work that is no longer running.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sections: dict[str, Section] = {
            name: Section(name, {}, time.time(), 0) for name in SECTIONS
        }
        self._version = 0
        self._watchers: list[Watcher] = []
        self._started = time.time()
        self._activity = IDLE
        self._detail = ""

    # ------------------------------------------------------------- mutation
    def update(self, section: str, **fields: Any) -> Section:
        """Merge fields into a section. Unknown sections are refused."""
        return self._write(section, fields, replace=False)

    def replace(self, section: str, data: dict[str, Any]) -> Section:
        return self._write(section, data, replace=True)

    def _write(self, section: str, data: dict[str, Any], *, replace: bool) -> Section:
        if section not in SECTIONS:
            raise KeyError(f"bilinmeyen bölüm: {section}")
        with self._lock:
            current = self._sections[section]
            merged = dict(data) if replace else {**current.data, **data}
            self._version += 1
            current.data = merged
            current.updated = time.time()
            current.version = self._version
            payload = dict(merged)
        self._notify(section, payload)
        return Section(section, payload, current.updated, current.version)

    def set_activity(self, activity: str, detail: str = "") -> None:
        """The single field a UI reads most. Unknown values become ERROR, loudly.

        Silently accepting an unknown activity would leave the interface showing a
        state it has no rendering for, which looks like a hang.
        """
        if activity not in ACTIVITIES:
            activity, detail = ERROR, f"bilinmeyen etkinlik: {activity}"
        with self._lock:
            self._activity = activity
            self._detail = detail
        self.update(SYSTEM, activity=activity, activity_detail=detail)

    # -------------------------------------------------------------- reading
    def get(self, section: str) -> dict[str, Any]:
        with self._lock:
            if section not in self._sections:
                raise KeyError(f"bilinmeyen bölüm: {section}")
            return dict(self._sections[section].data)

    @property
    def activity(self) -> str:
        with self._lock:
            return self._activity

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def uptime_s(self) -> float:
        return time.time() - self._started

    def snapshot(self) -> dict[str, Any]:
        """Everything, as a plain tree. What a client receives on connecting."""
        with self._lock:
            return {
                "version": self._version,
                "taken": time.time(),
                "started": self._started,
                "uptime_s": round(time.time() - self._started, 1),
                "activity": self._activity,
                "activity_detail": self._detail,
                "sections": {name: section.as_dict()
                             for name, section in self._sections.items()},
            }

    def is_stale(self, known_version: int) -> bool:
        return known_version != self.version

    # ------------------------------------------------------------- watching
    def watch(self, callback: Watcher) -> Callable[[], None]:
        """Be told when a section changes. Returns an unsubscribe function."""
        with self._lock:
            self._watchers.append(callback)

        def unwatch() -> None:
            with self._lock:
                if callback in self._watchers:
                    self._watchers.remove(callback)

        return unwatch

    def _notify(self, section: str, data: dict[str, Any]) -> None:
        with self._lock:
            watchers = list(self._watchers)
        for callback in watchers:
            try:
                callback(section, dict(data))
            except Exception:  # noqa: BLE001 - a watcher must not break a writer
                # No logging import here on purpose: this module stays a leaf, and
                # a broken watcher is the watcher's problem to notice.
                continue

    def reset(self) -> None:
        with self._lock:
            for name in SECTIONS:
                self._sections[name] = Section(name, {}, time.time(), 0)
            self._version = 0
            self._activity = IDLE
            self._detail = ""
