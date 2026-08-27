"""The improvement loop: observe, propose, measure, decide, remember.

    OBSERVE → IDENTIFY GAP → RESEARCH → HYPOTHESIS → SCORE → DESIGN
            → SANDBOX → TEST → BENCHMARK → VERIFY → CANDIDATE
            → PROMOTION GATE → PROMOTE / DISCARD → RECORD → LEARN

Each stage is a separate method that can be run and tested on its own, because a
pipeline that only works end to end is a pipeline nobody can debug at four in the
morning when it has been failing quietly for a week.

Three things are load-bearing:

**The model proposes, arithmetic decides.** A local model is asked what might help
and how to word it. Whether the result is worth trying is scored by
`opportunity.score`, whether it worked is decided by `benchmark.compare`, and
whether it may be installed is decided by S5's promotion gate. None of those reads
prose.

**Nothing the engine concludes becomes a fact.** Lessons are written to memory as
agent-sourced, which S1 already treats as unverified. Measurements live in the
experiment registry with an id, so a claim about a measurement can be checked
against the measurement.

**The machine belongs to the user.** Every cycle asks the S2 policy first. Improvement
work is the lowest-value thing the system does — it yields to his input, to a
loaded machine, and to the budget.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..autonomy.policy import Policy, Stance
from ..autonomy.resources import ResourceMonitor, Snapshot
from ..lab import Lab
from ..lab.benchmark import Comparison
from ..lab.registry import PASSED
from . import budget as budget_module
from .budget import ImprovementBudget
from .capabilities import CapabilityRegistry
from .gaps import Gap, GapDetector
from .hypotheses import Hypothesis, HypothesisStore, Lesson
from .opportunity import (
    DIMENSIONS,
    ESTIMATED,
    GUESS,
    MEASURED,
    SOURCED,
    Dimension,
    Estimate,
    Opportunity,
    Verdict,
    score,
)
from .preferences import PreferenceStore

log = logging.getLogger("jarvis.improve.engine")

#: Provenance for anything the engine concluded on its own. Already in S1's
#: unverified family — stated here so the choice is visible at the use site.
ENGINE_SOURCED = "ajan"

HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "statement": {"type": "string"},
        "expected_effect": {"type": "string"},
        "how_to_measure": {"type": "string"},
        "risk": {"type": "string"},
        "dimensions": {
            "type": "object",
            "properties": {name: {"type": "number"} for name in DIMENSIONS},
        },
    },
    "required": ["title", "statement", "how_to_measure"],
}

PROPOSER_SYSTEM = """\
Reason in English. Write in Turkish.

You are given one concrete weakness in a system, with the evidence that produced
it. Propose a single change that would reduce that weakness, and say how anyone
would measure whether it worked.

Rules that matter more than being helpful:

  Propose something testable in an isolated sandbox with a test suite. "Daha iyi
  bir mimari kur" is not testable. "Yol çözümlemesini önbelleğe al ve aynı test
  setiyle süreyi karşılaştır" is.

  Scope it small. One change, one measurement. A proposal that touches five
  subsystems cannot be attributed to anything when it fails.

  Do not invent library, service or API names. If a change needs one you are not
  certain exists, describe the capability and say the name must be checked.

  "dimensions" are your rough sense of each factor from 0 to 1, and they are only
  a starting point — the system scores the proposal itself. Guessing high does not
  make a proposal more likely to be tried; it makes your estimate less useful.

  If the evidence does not support any specific change, say so in "statement"
  rather than producing a plausible-sounding one.
"""


@dataclass(slots=True)
class Observation:
    snapshot: Snapshot
    stance: Stance
    gaps: list[Gap] = field(default_factory=list)
    capability_counts: dict[str, int] = field(default_factory=dict)
    budget: dict[str, str] = field(default_factory=dict)

    @property
    def may_work(self) -> bool:
        return self.stance.autonomous_allowed

    def summary(self) -> str:
        return (f"{self.stance.mode} · {len(self.gaps)} eksik · "
                f"{self.snapshot.summary()}")


@dataclass(slots=True)
class ImprovementPlan:
    """A hypothesis turned into something that can actually be run and measured."""

    hypothesis_id: str
    setup_files: dict[str, str] = field(default_factory=dict)
    changed_files: dict[str, str] = field(default_factory=dict)
    promote: list[str] = field(default_factory=list)
    test_target: str = "tests"

    @property
    def valid(self) -> bool:
        return bool(self.setup_files) and bool(self.changed_files)


@dataclass(slots=True)
class CycleResult:
    observation: Observation | None = None
    hypothesis: Hypothesis | None = None
    verdict: Verdict | None = None
    comparison: Comparison | None = None
    experiment_id: str = ""
    promoted: bool = False
    lesson: Lesson | None = None
    note_title: str | None = None
    stopped: str = ""

    def summary(self) -> str:
        if self.stopped:
            return f"durdu: {self.stopped}"
        if self.hypothesis is None:
            return "hipotez üretilmedi"
        state = "terfi etti" if self.promoted else "terfi etmedi"
        measured = self.comparison.summary() if self.comparison else "ölçülmedi"
        return f"{self.hypothesis.title[:50]} · {measured} · {state}"


class ImprovementEngine:
    def __init__(self, config, brain, *, lab: Lab, memory=None, events=None) -> None:
        self.config = config
        self.brain = brain
        self.lab = lab
        self.memory = memory
        self.events = events

        db = config.path("paths.db", "data/jarvis.db")
        self.capabilities = CapabilityRegistry(db)
        self.gaps = GapDetector(db, self.capabilities)
        self.hypotheses = HypothesisStore(db)
        self.preferences = PreferenceStore(db)
        self.budget = ImprovementBudget(
            db,
            daily=config.get("improve.daily", None),
            nightly=config.get("improve.nightly", None),
            night_hours=tuple(config.get("budget.night_hours", [1, 8])),
        )
        self.policy = Policy(
            idle_after_s=config.get("autonomy.idle_after_s", 300),
            night_hours=tuple(config.get("budget.night_hours", [1, 8])),
            cpu_ceiling=config.get("autonomy.cpu_ceiling", 65.0),
            gpu_ceiling=config.get("autonomy.gpu_ceiling", 55.0),
            ram_ceiling=config.get("autonomy.ram_ceiling", 88.0),
        )
        self.monitor = ResourceMonitor()
        self.model = config.get("local.model") or brain.local.model
        self.minimum_score = float(config.get("improve.minimum_score", 0.45))

    def emit(self, kind: str, message: str, level: str = "info", **data) -> None:
        if self.events is not None:
            self.events.publish("improve", kind, message, level=level, data=data)

    # ------------------------------------------------------------- OBSERVE
    def observe(self) -> Observation:
        snapshot = self.monitor.snapshot()
        return Observation(
            snapshot=snapshot,
            stance=self.policy.evaluate(snapshot),
            gaps=self.gaps.detect(),
            capability_counts=self.capabilities.counts(),
            budget=self.budget.snapshot(),
        )

    # ------------------------------------------------------------ PROPOSE
    def propose(self, gap: Gap) -> tuple[Hypothesis | None, bool]:
        """Ask the model for a change, then record it if it is genuinely new."""
        # Checked before spending anything: a gap that already has an open idea
        # does not need another, and asking the model again would produce one in
        # different words that the fingerprint cannot recognise as the same.
        open_already = self.hypotheses.open_for_gap(gap.key)
        if open_already is not None:
            self.emit("hypothesis.open",
                      f"Bu eksik için zaten açık bir hipotez var: {open_already.summary()}",
                      hypothesis=open_already.id, gap=gap.key)
            return open_already, False

        allowance = self.budget.spend(budget_module.HYPOTHESIS, subject=gap.key)
        if not allowance.allowed:
            return None, False

        evidence = "\n".join(f"  - {item}" for item in gap.evidence) or "  (kayıt yok)"
        instruction = (
            f"[Tespit edilen eksik]\n{gap.title}\n\n"
            f"[Kanıt]\n{evidence}\n\n"
            f"[İlgili yetenek]\n{gap.capability or 'belirtilmemiş'}"
        )
        try:
            raw = self.brain.local.chat(
                [{"role": "system", "content": PROPOSER_SYSTEM},
                 {"role": "user", "content": instruction}],
                schema=HYPOTHESIS_SCHEMA, temperature=0.4, model=self.model)
        except OSError as exc:
            log.warning("hipotez üretilemedi: %s", exc)
            return None, False

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("hipotez üretici çözümlenemeyen cevap verdi")
            return None, False

        title = str(payload.get("title", "")).strip()
        statement = str(payload.get("statement", "")).strip()
        if not title or not statement:
            return None, False

        existing = self.hypotheses.seen(title, statement)
        if existing is not None:
            self.emit("hypothesis.duplicate",
                      f"Aynı fikir daha önce denenmiş: {existing.summary()}",
                      hypothesis=existing.id)
            return existing, False

        opportunity = self._to_opportunity(gap, payload)
        hypothesis, is_new = self.hypotheses.propose(
            title, statement=statement, capability=gap.capability,
            gap_key=gap.key, opportunity=opportunity)
        if is_new:
            self.emit("hypothesis.new", f"Yeni hipotez: {title[:80]}",
                      hypothesis=hypothesis.id, gap=gap.key)
        return hypothesis, is_new

    def _to_opportunity(self, gap: Gap, payload: dict[str, Any]) -> Opportunity:
        """Wrap a model's proposal in structure that records how weak it is.

        Everything the model supplied is marked GUESS, because that is what it is:
        a plausible number with nothing behind it. Only a measurement moves a
        dimension off that basis.
        """
        raw_dimensions = payload.get("dimensions", {}) or {}
        dimensions: dict[str, Dimension] = {}
        for name in DIMENSIONS:
            try:
                value = float(raw_dimensions.get(name, 0.4))
            except (TypeError, ValueError):
                value = 0.4
            dimensions[name] = Dimension(
                name=name, value=max(0.0, min(1.0, value)), basis=GUESS,
                rationale="model tahmini, ölçülmedi")

        # Evidence for the gap is real; it was derived from recorded facts.
        dimensions["feasibility"] = Dimension(
            "feasibility", dimensions["feasibility"].value, ESTIMATED,
            "eksik yapısal sinyallerden türetildi")

        return Opportunity(
            title=payload.get("title", gap.title),
            description=payload.get("statement", ""),
            category="ic-gelistirme",
            capability=gap.capability,
            dimensions=dimensions,
            estimates={
                "expected_effect": Estimate(
                    value=None, unit="", basis=GUESS, confidence=0.2,
                    assumptions=[str(payload.get("expected_effect", ""))[:300]],
                    uncertainty="deney yapılmadan bilinemez"),
            },
            notes=[f"kaynak eksik: {gap.key}",
                   f"ölçüm önerisi: {payload.get('how_to_measure', '')}"[:300]],
        )

    def evaluate(self, opportunity: Opportunity) -> Verdict:
        return score(opportunity, weights=self.preferences.weights(),
                     minimum=self.minimum_score)

    # ------------------------------------------------------------ EXPERIMENT
    def run_plan(self, hypothesis: Hypothesis, plan: ImprovementPlan, *,
                 python: str = "python") -> CycleResult:
        """Seed, measure, change, measure, gate. The S5 machinery does the deciding."""
        result = CycleResult(hypothesis=hypothesis)

        allowance = self.budget.spend(budget_module.EXPERIMENT, subject=hypothesis.id)
        if not allowance.allowed:
            result.stopped = allowance.summary()
            return result
        if not plan.valid:
            result.stopped = "plan eksik: kurulum veya değişiklik dosyası yok"
            return result

        session = self.lab.experiment(hypothesis.title[:60], model=self.model)
        result.experiment_id = session.id
        self.hypotheses.start_attempt(hypothesis.id, session.id)
        self.emit("experiment.started", f"Deney başladı: {hypothesis.title[:60]}",
                  hypothesis=hypothesis.id, experiment=session.id)

        try:
            for relative, content in plan.setup_files.items():
                session.sandbox.write(relative, content)
            # Before the measurement, not after: the candidate is written over
            # these same paths below, so once that happens the sandbox no longer
            # holds the code the baseline number describes.
            session.record_baseline_source(dict(plan.setup_files),
                                           test_target=plan.test_target)
            baseline = session.measure_baseline(python=python, target=plan.test_target)

            for relative, content in plan.changed_files.items():
                session.sandbox.write(relative, content)
            session.record_candidate_source(dict(plan.changed_files))
            candidate = session.measure_candidate(python=python, target=plan.test_target)
        except Exception as exc:  # noqa: BLE001 - a broken plan is a failed experiment
            log.warning("deney çalıştırılamadı: %s", exc)
            result.stopped = f"deney hatası: {exc}"
            result.lesson = self.learn(hypothesis, session.id,
                                       why=f"deney çalıştırılamadı: {exc}")
            session.discard(str(exc)[:120])
            return result

        result.comparison = session.settle()
        log.info("deney %s: baseline %s / aday %s", session.id[:8],
                 baseline.summary(), candidate.summary())

        if session.state != PASSED:
            result.lesson = self.learn(
                hypothesis, session.id,
                why=result.comparison.summary(),
                measured={"baseline": baseline.as_dict(), "candidate": candidate.as_dict()})
            session.discard("ölçüm kabul edilebilir değil")
            return result

        promotion = session.promote(plan.promote or list(plan.changed_files))
        result.promoted = promotion.ok
        if promotion.ok:
            self.hypotheses.confirm(hypothesis.id)
            self.capabilities.record_benchmark(
                hypothesis.capability or "code.writing",
                benchmark=plan.test_target,
                score=float(candidate.effective))
            result.note_title = self._record_success(hypothesis, session.id, result.comparison)
            self.emit("improvement.promoted",
                      f"İyileştirme terfi etti: {hypothesis.title[:60]}",
                      level="success", hypothesis=hypothesis.id, experiment=session.id)
        else:
            result.lesson = self.learn(hypothesis, session.id,
                                       why=f"terfi kapısı reddetti: {promotion.error}")
        return result

    # ----------------------------------------------------------------- LEARN
    def learn(self, hypothesis: Hypothesis, experiment_id: str, *, why: str,
              measured: dict[str, Any] | None = None) -> Lesson:
        """Turn a failure into something that changes what happens next.

        The model is asked what went wrong, but only after the structural facts are
        already recorded — so its answer is an interpretation attached to evidence
        rather than a replacement for it.
        """
        lesson = Lesson(why=why, experiment_id=experiment_id, measured=measured or {})

        interpretation = self._interpret_failure(hypothesis, why, measured)
        if interpretation:
            lesson.wrong_assumption = interpretation.get("wrong_assumption", "")
            lesson.conditions = interpretation.get("conditions", "")
            lesson.needed_change = interpretation.get("needed_change", "")
            lesson.retry_worth = bool(interpretation.get("retry_worth"))

        self.hypotheses.refute(hypothesis.id, lesson)
        self._record_lesson(hypothesis, lesson)
        self.emit("improvement.failed",
                  f"Deney başarısız: {hypothesis.title[:50]} — {lesson.summary()[:90]}",
                  level="warn", hypothesis=hypothesis.id, experiment=experiment_id)
        return lesson

    def _interpret_failure(self, hypothesis: Hypothesis, why: str,
                           measured: dict[str, Any] | None) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "wrong_assumption": {"type": "string"},
                "conditions": {"type": "string"},
                "needed_change": {"type": "string"},
                "retry_worth": {"type": "boolean"},
            },
            "required": ["wrong_assumption", "retry_worth"],
        }
        system = (
            "Reason in English, write in Turkish. An experiment failed. From the "
            "hypothesis and the recorded numbers, say which assumption turned out "
            "to be wrong and what would have to change for a retry to be worth "
            "running. If nothing specific would change the outcome, retry_worth is "
            "false — saying 'try again' without naming a change is how a system "
            "repeats itself forever."
        )
        instruction = (
            f"[Hipotez]\n{hypothesis.title}\n{hypothesis.statement}\n\n"
            f"[Sonuç]\n{why}\n\n"
            f"[Ölçümler]\n{json.dumps(measured or {}, ensure_ascii=False)[:1200]}"
        )
        try:
            raw = self.brain.local.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": instruction}],
                schema=schema, temperature=0.2, model=self.model)
            return json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("başarısızlık yorumlanamadı: %s", exc)
            return {}

    # ---------------------------------------------------------------- MEMORY
    def _record_lesson(self, hypothesis: Hypothesis, lesson: Lesson) -> str | None:
        """Write the lesson as an agent claim, never as established fact."""
        if self.memory is None:
            return None
        body = [
            f"**Hipotez:** {hypothesis.statement or hypothesis.title}",
            "",
            f"**Sonuç:** {lesson.why}",
        ]
        if lesson.wrong_assumption:
            body.append(f"**Yanlış çıkan varsayım:** {lesson.wrong_assumption}")
        if lesson.conditions:
            body.append(f"**Hangi koşullarda:** {lesson.conditions}")
        if lesson.needed_change:
            body.append(f"**Gereken değişiklik:** {lesson.needed_change}")
        body.append(f"**Tekrar denemeye değer mi:** {'evet' if lesson.retry_worth else 'hayır'}")
        if lesson.experiment_id:
            body.append("")
            body.append(f"Ölçümler deney kaydında: `{lesson.experiment_id}`")
        body.append("")
        body.append("Bu not JARVIS'in kendi yorumudur; ölçümler dışındaki kısmı "
                    "doğrulanmış bilgi değildir.")

        try:
            note = self.memory.vault.append(
                "deneyim", f"Başarısız deney: {hypothesis.title[:50]}",
                "\n".join(body), source=ENGINE_SOURCED)
            self.memory.reindex()
            return note.title
        except Exception as exc:  # noqa: BLE001
            log.warning("ders hafızaya yazılamadı: %s", exc)
            return None

    def _record_success(self, hypothesis: Hypothesis, experiment_id: str,
                        comparison: Comparison) -> str | None:
        if self.memory is None:
            return None
        body = [
            f"**Hipotez:** {hypothesis.statement or hypothesis.title}",
            "",
            f"**Ölçüm:** {comparison.summary()}",
            f"- baseline: {comparison.baseline.summary()}",
            f"- aday: {comparison.candidate.summary()}",
            "",
            f"Deney kaydı: `{experiment_id}`",
            "",
            "Ölçümler bu makinede alındı ve deney kaydından doğrulanabilir. "
            "Genel bir sonuç değil — yalnızca bu değişikliğin bu testlerdeki etkisi.",
        ]
        try:
            note = self.memory.vault.append(
                "deneyim", f"Başarılı iyileştirme: {hypothesis.title[:50]}",
                "\n".join(body), source=ENGINE_SOURCED)
            self.memory.reindex()
            return note.title
        except Exception as exc:  # noqa: BLE001
            log.warning("sonuç hafızaya yazılamadı: %s", exc)
            return None

    # ----------------------------------------------------------------- CYCLE
    def cycle(self, *, planner=None, python: str = "python") -> CycleResult:
        """One pass. Stops at the first gate that says no, and says which."""
        result = CycleResult()
        result.observation = self.observe()

        if not result.observation.may_work:
            result.stopped = f"kaynak politikası: {result.observation.stance.reason}"
            return result
        if not result.observation.gaps:
            result.stopped = "tespit edilen eksik yok"
            return result

        unshaped: list[str] = []
        for gap in result.observation.gaps:
            hypothesis, is_new = self.propose(gap)
            if hypothesis is None:
                continue
            if not is_new and not hypothesis.runnable:
                continue

            verdict = self.evaluate(_opportunity_from(hypothesis))
            result.hypothesis = hypothesis
            result.verdict = verdict
            if not verdict.worth_pursuing:
                self.hypotheses.shelve(hypothesis.id, reason=verdict.summary())
                continue

            # A gap with nothing to compare against cannot become an experiment,
            # and asking the model to write one anyway is how S6b got four plans
            # and no passing run. The hypothesis is still recorded — knowing what
            # is missing is worth keeping; it just does not go to the planner.
            if not gap.experiment_shaped:
                log.debug("planlamaya alınmadı (%s): %s", gap.key, gap.shape_reason)
                unshaped.append(f"{gap.key}: {gap.shape_reason}")
                continue

            if planner is None:
                result.stopped = "deney planlayıcı verilmedi"
                return result

            # Checked, not spent. Planning costs a model call, and burning one to
            # design an experiment the budget will refuse to run is waste the
            # ledger would never show.
            allowance = self.budget.check(budget_module.EXPERIMENT)
            if not allowance.allowed:
                result.stopped = allowance.summary()
                return result

            plan = planner(hypothesis, gap)
            if plan is None:
                result.stopped = "plan üretilemedi"
                return result
            return self.run_plan(hypothesis, plan, python=python)

        if unshaped:
            result.stopped = (f"deney-şekilli eksik yok — {len(unshaped)} eksik "
                              f"planlamaya alınmadı ({unshaped[0]})")
        else:
            result.stopped = "çalıştırılabilir hipotez bulunamadı"
        return result

    def status(self) -> dict[str, Any]:
        return {
            "yetenekler": self.capabilities.counts(),
            "hipotezler": self.hypotheses.counts(),
            "butce": self.budget.snapshot(),
            "tercihler": len(self.preferences.list()),
            "agirliklar": {k: round(v, 2) for k, v in self.preferences.weights().items()},
        }


def _opportunity_from(hypothesis: Hypothesis) -> Opportunity:
    data = hypothesis.opportunity or {}
    dimensions = {
        name: Dimension(name, float(entry.get("value", 0.4)),
                        entry.get("basis", GUESS), entry.get("rationale", ""))
        for name, entry in (data.get("dimensions") or {}).items()
    }
    estimates = {
        name: Estimate(
            value=entry.get("value"), unit=entry.get("unit", ""),
            basis=entry.get("basis", GUESS), confidence=float(entry.get("confidence", 0.0)),
            evidence=list(entry.get("evidence", [])),
            assumptions=list(entry.get("assumptions", [])),
            uncertainty=entry.get("uncertainty", ""))
        for name, entry in (data.get("estimates") or {}).items()
    }
    return Opportunity(
        title=data.get("title", hypothesis.title),
        description=data.get("description", hypothesis.statement),
        category=data.get("category", "ic-gelistirme"),
        capability=data.get("capability", hypothesis.capability),
        dimensions=dimensions, estimates=estimates,
        notes=list(data.get("notes", [])),
    )


__all__ = ["ImprovementEngine", "Observation", "ImprovementPlan", "CycleResult",
           "ENGINE_SOURCED", "MEASURED", "SOURCED", "ESTIMATED", "GUESS"]
