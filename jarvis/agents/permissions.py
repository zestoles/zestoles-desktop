"""What an agent is allowed to touch.

Permissions here are structural, not advisory. An agent without MEMORY_READ is not
asked politely to avoid memory — it is handed a context whose recall() refuses.
The distinction matters because the thing being restricted is a language model,
and a language model will do whatever a sufficiently confident instruction tells
it to. Only the code around it can actually say no.

Two capabilities — shell execution and filesystem writes — are refused unless the
caller holds a sandbox to contain them. They were refused outright through S3
because there was nowhere to put them; S5 built that place, so the ban is now
conditional rather than absolute. The condition is held by the code constructing
the grant, never by the agent asking for the capability: an agent declaring that
it needs a shell is a request, not an authorisation.
"""

from __future__ import annotations

from dataclasses import dataclass

MEMORY_READ = "memory.read"
MEMORY_WRITE = "memory.write"
WEB_SEARCH = "web.search"
FS_READ = "fs.read"
FS_WRITE = "fs.write"
SHELL = "shell"
CLOUD_BRAIN = "brain.cloud"
TASK_SUBMIT = "task.submit"

ALL = frozenset({
    MEMORY_READ, MEMORY_WRITE, WEB_SEARCH, FS_READ, FS_WRITE,
    SHELL, CLOUD_BRAIN, TASK_SUBMIT,
})

#: Granted only to a grant built with sandboxed=True. Irreversible outside a
#: sandbox, and the sandbox is what makes them reversible.
SANDBOX_ONLY = frozenset({SHELL, FS_WRITE})

#: Kept as the old name so nothing that imported it breaks silently.
HARD_DENIED = SANDBOX_ONLY

#: Granted only when the caller explicitly opts in — never by an agent's own
#: declaration. The system is designed to work without the metered tier.
OPT_IN_ONLY = frozenset({CLOUD_BRAIN})


class PermissionDenied(RuntimeError):
    def __init__(self, agent: str, capability: str) -> None:
        super().__init__(f"{agent} ajanının '{capability}' yetkisi yok")
        self.agent = agent
        self.capability = capability


@dataclass(frozen=True, slots=True)
class Grant:
    agent: str
    capabilities: frozenset[str]
    sandboxed: bool = False

    @classmethod
    def build(
        cls,
        agent: str,
        requested: frozenset[str],
        *,
        allow_cloud: bool = False,
        sandboxed: bool = False,
    ) -> Grant:
        """Build a grant. Dangerous capabilities need the caller to opt in.

        `sandboxed` says a Sandbox exists to contain shell and write access. It is
        a statement by the calling code, not by the agent — which is the whole
        point, since the agent is the part that can be talked into anything.
        """
        granted = set(requested)
        if not sandboxed:
            granted -= SANDBOX_ONLY
        if not allow_cloud:
            granted -= OPT_IN_ONLY
        unknown = granted - ALL
        if unknown:
            raise ValueError(f"bilinmeyen yetki: {', '.join(sorted(unknown))}")
        return cls(agent, frozenset(granted), sandboxed=sandboxed)

    def allows(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise PermissionDenied(self.agent, capability)

    def summary(self) -> str:
        return ", ".join(sorted(self.capabilities)) or "yetkisiz"
