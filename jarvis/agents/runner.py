"""Bridges an orchestration into the S2 task queue.

Registering here rather than in autonomy/runners.py keeps the dependency pointing
one way: agents know about autonomy, autonomy knows nothing about agents. Reversing
it would make the scheduler unable to start without the whole agent stack loaded.

Because it is a normal task, an orchestration inherits everything S2 already
guarantees — it waits for the machine to be idle, survives a restart, retries with
backoff, and yields to anything the user asked for directly.
"""

from __future__ import annotations

import logging

from ..autonomy.runners import RunContext, runner

log = logging.getLogger("jarvis.agents.runner")

RUNNER_NAME = "agents.orchestrate"


def register(system) -> None:
    """Attach an AgentSystem to the queue. Idempotent across reconstructions."""
    from ..autonomy import runners as registry

    def _orchestrate(ctx: RunContext) -> str:
        goal = str(ctx.task.payload.get("goal", "")).strip()
        if not goal:
            raise ValueError("görev yükünde 'goal' yok")

        run = system.run(goal, should_stop=ctx.should_stop, origin=ctx.task.origin)

        if not run.ok:
            # Raising hands the task back to S2's retry and quarantine machinery
            # instead of recording a failure as if it were a result.
            raise RuntimeError(run.final.error if run.final else "orkestrasyon başarısız")

        verdict = run.verdict.summary() if run.verdict else "doğrulanmadı"
        return f"{run.summary()} · {verdict}"

    registry.REGISTRY[RUNNER_NAME] = _orchestrate
    log.debug("orkestrasyon çalıştırıcısı kaydedildi")
