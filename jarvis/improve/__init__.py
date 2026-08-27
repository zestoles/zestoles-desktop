"""Self-improvement, assembled.

What S6 adds is the loop that decides what to work on. Everything it uses to do
that work already existed: S4 researches, S3 orchestrates and verifies, S5 contains
and measures, S2 schedules and yields, S1 remembers. This layer supplies the only
things missing — an honest account of what the system can do, a way to notice what
it cannot, and a bounded process for turning that into a tested change.

The boundary is unchanged and deliberate. JARVIS can find a weakness, propose a
fix, write it, run it, measure it against a baseline and record what happened. The
change lands in the promotion target, never in the source tree, because
ALLOW_SELF_MODIFICATION is still False. The loop that would let it edit itself is
now fully built and still switched off — which is the correct order to build those
two things in.
"""

from __future__ import annotations

import logging

from ..config import Config
from .budget import EXPERIMENT, HYPOTHESIS, RESEARCH, Allowance, ImprovementBudget
from .capabilities import (
    BROKEN,
    MISSING,
    PARTIAL,
    UNKNOWN,
    WORKING,
    Capability,
    CapabilityRegistry,
)
from .engine import (
    ENGINE_SOURCED,
    CycleResult,
    ImprovementEngine,
    ImprovementPlan,
    Observation,
)
from .gaps import Gap, GapDetector
from .hypotheses import (
    CONFIRMED,
    PROPOSED,
    REFUTED,
    SHELVED,
    TESTING,
    Hypothesis,
    HypothesisStore,
    Lesson,
    fingerprint,
)
from .opportunity import (
    DIMENSIONS,
    ESTIMATED,
    GUESS,
    MEASURED,
    SOURCED,
    Dimension,
    Estimate,
    Opportunity,
    Screening,
    Verdict,
    rank,
    recordable_estimates,
    score,
    screen,
)
from .planner import (
    ALLOWED_IMPORTS,
    FORBIDDEN_NAMES,
    PLAN_SCHEMA,
    ExperimentPlanner,
    PlanReview,
    validate as validate_plan,
)
from .preferences import Preference, PreferenceStore

log = logging.getLogger("jarvis.improve")

RUNNER_NAME = "improve.cycle"


def register_runner(engine: ImprovementEngine, *, planner=None) -> None:
    """Expose one improvement cycle as an S2 task.

    Registered from here so the dependency points one way: improvement knows about
    autonomy, autonomy knows nothing about improvement. A cycle queued this way
    inherits idle-waiting, restart survival and backoff without reimplementing any
    of them.
    """
    from ..autonomy import runners as registry

    def _cycle(ctx) -> str:
        if ctx.should_stop():
            return "durdurma istendi"
        result = engine.cycle(planner=planner)
        return result.summary()

    registry.REGISTRY[RUNNER_NAME] = _cycle
    log.debug("iyileştirme çalıştırıcısı kaydedildi")


def build(config: Config, brain, *, lab, memory=None, events=None,
          planner=None) -> ImprovementEngine:
    """Assemble the engine and, unless told otherwise, give it a planner.

    Before S6b this defaulted to None and every cycle stopped at "deney
    planlayıcı verilmedi" — the loop could think but not test. The planner is
    still an argument rather than a fixture, because a caller that wants the
    cycle without experiments (a dry run, a test) should be able to say so.
    """
    engine = ImprovementEngine(config, brain, lab=lab, memory=memory, events=events)
    if planner is None and config.get("improve.planner.enabled", True):
        from .planner import ExperimentPlanner

        planner = ExperimentPlanner(
            brain, model=config.get("improve.planner.model", "") or engine.model,
            events=events,
            temperature=float(config.get("improve.planner.temperature", 0.2)))
    register_runner(engine, planner=planner)
    return engine


__all__ = [
    "ImprovementEngine", "build", "register_runner", "RUNNER_NAME",
    "Observation", "ImprovementPlan", "CycleResult", "ENGINE_SOURCED",
    "CapabilityRegistry", "Capability", "WORKING", "PARTIAL", "MISSING",
    "BROKEN", "UNKNOWN",
    "GapDetector", "Gap",
    "HypothesisStore", "Hypothesis", "Lesson", "fingerprint",
    "PROPOSED", "TESTING", "CONFIRMED", "REFUTED", "SHELVED",
    "Opportunity", "Dimension", "Estimate", "Verdict", "Screening",
    "score", "rank", "screen", "recordable_estimates", "DIMENSIONS",
    "MEASURED", "SOURCED", "ESTIMATED", "GUESS",
    "PreferenceStore", "Preference",
    "ImprovementBudget", "Allowance", "RESEARCH", "HYPOTHESIS", "EXPERIMENT",
    "ExperimentPlanner", "PlanReview", "validate_plan", "PLAN_SCHEMA",
    "ALLOWED_IMPORTS", "FORBIDDEN_NAMES",
]
