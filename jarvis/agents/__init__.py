"""Agent orchestration, assembled.

What S3 adds is the ability to do work that needs more than one pass: split a goal
into steps, hand each to a role that suits it, put the pieces back together, and
then refuse to call the result finished until something has checked it.

What S3 deliberately does not add is trust. An orchestration produces a claim by
JARVIS — labelled AGENT_SOURCED, which S1's gate already treats as unverified — and
nothing here writes to long-term memory. Verified-by-another-agent is not
verified-against-reality, and the path for real knowledge is S4's research loop,
which will carry a citation. Wiring agent output into memory now would undo the
one protection this project has already paid to learn.
"""

from __future__ import annotations

import logging

from ..autonomy.events import EventLog
from ..config import Config
from . import roles
from .base import AGENT_SOURCED, AgentContext, AgentResult, AgentSpec, run_agent
from .orchestrator import Orchestration, Orchestrator, Step
from .permissions import Grant, PermissionDenied
from .runner import RUNNER_NAME, register as _register_runner
from .skills import Skill, SkillLibrary
from .verify import Verdict, mechanical_checks, suspicious_identifiers, verify

log = logging.getLogger("jarvis.agents")


class AgentSystem:
    def __init__(self, config: Config, brain, events: EventLog, *, memory=None) -> None:
        self.config = config
        self.brain = brain
        self.events = events
        self.memory = memory
        self.enabled = bool(config.get("agents.enabled", True))

        self.skills = SkillLibrary(config.path("paths.db", "data/jarvis.db"))
        self.orchestrator = Orchestrator(
            brain, events, memory=memory, config=config, skills=self.skills
        )
        _register_runner(self)

    def run(self, goal: str, *, should_stop=lambda: False, origin: str = "user") -> Orchestration:
        if not self.enabled:
            raise RuntimeError("ajan sistemi yapılandırmada kapalı")
        return self.orchestrator.run(goal, should_stop=should_stop, origin=origin)

    def provide_research(self, research) -> None:
        """Connect S4 after runtime assembly without introducing import cycles."""
        self.orchestrator.research = research

    def status(self) -> dict[str, object]:
        active = self.skills.list(include_retired=False)
        return {
            "acik": self.enabled,
            "roller": list(roles.ASSIGNABLE),
            "model": self.orchestrator.model,
            "dogrulama_modeli": self.orchestrator.verify_model,
            "ayri_model_dogrulama": self.orchestrator.verify_model != self.orchestrator.model,
            "max_adim": self.orchestrator.max_steps,
            "beceriler": len(active),
            "kaynakli_arastirma": self.orchestrator.research is not None,
        }


__all__ = [
    "AgentSystem", "Orchestrator", "Orchestration", "Step", "AgentSpec", "AgentResult",
    "AgentContext", "Grant", "PermissionDenied", "Skill", "SkillLibrary", "Verdict",
    "verify", "mechanical_checks", "suspicious_identifiers", "run_agent", "roles",
    "AGENT_SOURCED", "RUNNER_NAME",
]
