"""The conversation the terminal is holding.

Owns two things and no more: the rolling window sent to the model, and a reference
to the runtime everything else lives on. Subsystems are reached through properties
rather than copied in, so a session cannot drift out of step with what is actually
running.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..state import SESSION

log = logging.getLogger("jarvis.cli.session")


#: Roles that are dialogue. Anything else is a record of what happened.
CONVERSATION_ROLES = frozenset({"user", "assistant"})


def _provide_profile(config) -> None:
    """Hand the memory tools a profile, once per process.

    Failure is not fatal: an assistant that cannot remember preferences is
    still an assistant, and the tools report the absence themselves.
    """
    try:
        from ..memory.profile import Profile
        from ..tools import hafiza

        if hafiza._PROFILE is not None:  # noqa: SLF001 - module-level registry
            return
        hafiza.provide_profile(Profile(config.path("paths.db", "data/jarvis.db")))
    except Exception as exc:  # noqa: BLE001
        log.warning("kullanıcı profili açılamadı: %s", exc)


class Session:
    """The rolling window sent to the model, plus a full copy handed to memory.

    The window is trimmed so prompts stay affordable; memory keeps everything, so
    trimming here loses nothing permanently.
    """

    def __init__(self, runtime, *, history_turns: int) -> None:
        self.runtime = runtime
        self.history: list[dict[str, str]] = []
        self.max_messages = max(2, history_turns * 2)
        self._assistant: object | None = None
        self._assistant_built = False

    def assistant(self, *, approve=None, events=None):
        """The tool-using loop, or None when it could not be built.

        Built lazily and once: the workspace directory is created on first use,
        and a terminal that never asks for real work should not create it. A
        failure here costs tool use, not the session — plain conversation still
        works, which is why this returns None rather than raising.
        """
        if self._assistant_built:
            assistant = self._assistant
            if assistant is not None:
                assistant.approve = approve
                if events is not None:
                    assistant.events = events
            return assistant

        self._assistant_built = True
        try:
            from ..assistant import Assistant
            from ..tools import Workspace, provide

            config = self.runtime.config
            # Web tools reach the S4 pipeline through this rather than rebuilding
            # search, extraction and injection defence. Absent when research did
            # not come up, and those tools say so instead of failing oddly.
            if self.research is not None:
                provide("research", self.research)
            if getattr(self.runtime, "documents", None) is not None:
                provide("documents", self.runtime.documents)
            if getattr(self.runtime, "reminders", None) is not None:
                provide("reminders", self.runtime.reminders)
            # What JARVIS has been told about the user, reachable as tools.
            # Absent when memory is off, and the tools say so rather than
            # failing oddly.
            _provide_profile(config)
            root = config.get("assistant.workspace", "") or str(Path.home())
            self._assistant = Assistant(
                self.brain, Workspace(Path(root)),
                events=events if events is not None else self.runtime.events,
                approve=approve,
                max_steps=int(config.get("assistant.max_steps", 8)),
                model=config.get("assistant.model", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - tools are optional, chat is not
            log.warning("araç katmanı kurulamadı: %s", exc)
            self.runtime.warnings.append(f"araç katmanı kurulamadı: {exc}")
            self._assistant = None
        return self._assistant

    # Subsystems are read through the runtime so there is one source of truth
    # about what exists and what failed to start.
    @property
    def brain(self):
        return self.runtime.brain

    @property
    def memory(self):
        return self.runtime.memory

    @property
    def core(self):
        return self.runtime.core

    @property
    def agents(self):
        return self.runtime.agents

    @property
    def research(self):
        return self.runtime.research

    @property
    def state(self):
        return self.runtime.state

    def add(self, role: str, content: str) -> None:
        """Record a turn. Conversation goes to the model too; records only to memory.

        The assistant writes what its tools actually did under its own role, and
        that belongs in memory but not in the transcript handed back to `chat` --
        which takes system/user/assistant and nothing else.
        """
        if role in CONVERSATION_ROLES:
            self.history.append({"role": role, "content": content})
            if len(self.history) > self.max_messages:
                self.history = self.history[-self.max_messages :]
        if self.memory is not None:
            self.memory.remember(role, content)
        self.state.update(SESSION, turns=len(self.history), last_role=role)

    def clear(self) -> None:
        self.history.clear()
        self.state.update(SESSION, turns=0)
