"""The coordinator: decompose, delegate, aggregate, verify.

The shape of a run is deliberately boring — plan, do the steps, combine, check —
because the interesting failures are all in what happens when a piece of it goes
wrong, and those need to be handled the same way every time.

Three properties are load-bearing:

  A step failing is not the run failing. Steps are isolated; a dead step is
  recorded, its instruction is carried into the aggregate as a stated gap, and the
  rest continues. A run that half-worked and says so is far more useful than one
  that collapses.

  The plan is validated before it is trusted. It comes from a language model, so
  it can name roles that do not exist, ask for forty steps, or return nothing
  usable. Every plan is checked and clamped against the roster before a single
  agent runs.

  Nothing is trusted at the end. The aggregate is verified against criteria that
  were written *before* the work was done — by the planner, as part of planning —
  which is what stops criteria from being quietly reshaped to fit whatever came out.

One model per run. Loading a second evicts the first on this machine and costs
about seventy seconds, so mixing tiers inside a run would spend more time swapping
than thinking.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..autonomy.events import ERROR, EventLog, SUCCESS, WARN
from . import roles
from .base import AGENT_SOURCED, AgentContext, AgentResult
from .base import run_agent
from .permissions import Grant
from .roles import PLANNER, SUMMARIZER
from .skills import Skill, SkillLibrary
from .verify import Verdict, verify

log = logging.getLogger("jarvis.agents.orchestrator")

MAX_STEPS_CEILING = 8
STEP_RETRIES = 1

PLAN_FROM_SKILL = "beceri"
PLAN_FROM_PLANNER = "planlayıcı"
PLAN_FROM_FALLBACK = "yedek"

SOURCED_CONTEXT_POLICY = (
    "### Kaynak disiplini\n\n"
    "Kaynaklı web araştırması mevcut. Dış dünya, güncel sürüm, sayı, özellik "
    "ve ürün iddialarında yalnız [Kaynaklı Web Araştırmacısı] bölümündeki "
    "denetlenmiş bulguları kullan. Yeni bir gerçek, alıntı veya kaynak uydurma; "
    "desteklenmeyen noktayı açıkça 'kaynaklarda doğrulanmadı' diye belirt."
)


@dataclass(slots=True)
class Step:
    role: str
    instruction: str
    why: str = ""
    result: AgentResult | None = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok

    def as_plan_entry(self) -> dict[str, str]:
        return {"role": self.role, "instruction": self.instruction, "why": self.why}


@dataclass(slots=True)
class Orchestration:
    goal: str
    steps: list[Step] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    final: AgentResult | None = None
    verdict: Verdict | None = None
    model: str = ""
    duration_ms: int = 0
    plan_source: str = PLAN_FROM_PLANNER
    skill_name: str | None = None
    source: str = AGENT_SOURCED

    @property
    def ok(self) -> bool:
        return self.final is not None and self.final.ok

    @property
    def verified(self) -> bool:
        return self.verdict is not None and self.verdict.ok

    @property
    def output(self) -> str:
        if self.final is None:
            return ""
        if self.verified:
            return self.final.output
        # Never present a rejected synthesis as the answer.  A sourced research
        # step already renders only claim-level evidence that survived its own
        # gate, so it is the one useful fallback we may show safely.
        sourced = next((step.result for step in self.steps
                        if step.role == "web_researcher" and step.ok), None)
        if sourced is not None:
            return (sourced.output +
                    "\n\nNot: Ajanların sonraki sentezi doğrulama kapısını geçmedi; "
                    "bu yüzden yalnız kaynak denetimli araştırma çıktısı gösterildi.")
        return ("Ajan çalışması doğrulama kapısını geçmedi. Sonuç gerçek veya "
                "tamamlanmış kabul edilmedi.")

    def summary(self) -> str:
        done = sum(1 for s in self.steps if s.ok)
        state = "doğrulandı" if self.verified else (
            "doğrulanmadı" if self.ok else "başarısız")
        return (f"{done}/{len(self.steps)} adım · {state} · "
                f"{self.duration_ms / 1000:.1f}s · {self.model}")


class Orchestrator:
    def __init__(
        self,
        brain,
        events: EventLog,
        *,
        memory=None,
        config=None,
        skills: SkillLibrary | None = None,
        research=None,
    ) -> None:
        self.brain = brain
        self.events = events
        self.memory = memory
        self.config = config
        self.skills = skills
        self.research = research

        get = config.get if config is not None else (lambda _k, d=None: d)
        self.max_steps = min(int(get("agents.max_steps", 5)), MAX_STEPS_CEILING)
        self.total_timeout_s = float(get("agents.total_timeout_s", 900))
        self.save_skills = bool(get("agents.save_skills", True))
        self.model = self._resolve_model(get("agents.model", "fast"))
        self.verify_model = self._resolve_model(get("agents.verify_model", "fast"))

    def _resolve_model(self, tier: str) -> str:
        get = self.config.get if self.config is not None else (lambda _k, d=None: d)
        if tier == "heavy":
            return get("local.model_heavy") or get("local.model") or self.brain.local.model
        return get("local.model") or self.brain.local.model

    # ------------------------------------------------------------------ run
    def run(self, goal: str, *, should_stop=lambda: False, origin: str = "user") -> Orchestration:
        run_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        deadline = started + self.total_timeout_s
        run = Orchestration(goal=goal, model=self.model)

        self.events.publish("agent", "run.start", f"Orkestrasyon başladı: {goal[:120]}",
                            data={"run": run_id, "model": self.model, "origin": origin})

        ctx = AgentContext(
            brain=self.brain, events=self.events,
            grant=Grant.build("orchestrator", frozenset(roles.PLANNER.capabilities)),
            model=self.model, should_stop=should_stop, memory=self.memory, run_id=run_id,
        )

        skill = self.skills.find(goal) if self.skills else None
        if skill is not None:
            run.steps = [Step(**entry) for entry in _as_steps(skill.steps)]
            run.criteria = list(skill.criteria)
            run.plan_source = PLAN_FROM_SKILL
            run.skill_name = skill.name
            self.events.publish("agent", "plan.skill",
                                f"Kayıtlı beceri kullanılıyor: {skill.title}",
                                data={"run": run_id, "skill": skill.name})
        else:
            run.steps, run.criteria, run.plan_source = self._plan(goal, ctx)

        if not run.steps:
            run.final = AgentResult("orchestrator", False, error="plan üretilemedi")
            run.duration_ms = int((time.monotonic() - started) * 1000)
            self.events.publish("agent", "run.error", "Plan üretilemedi",
                                level=ERROR, data={"run": run_id})
            return run

        self._execute(run, ctx, deadline=deadline, should_stop=should_stop)
        run.final = self._aggregate(run, ctx)

        if run.final.ok and not should_stop():
            run.verdict = verify(ctx, goal=goal, criteria=run.criteria,
                                 result=run.final, model=self.verify_model,
                                 memory=self.memory)
            run.final.verified = run.verdict.ok
            run.final.verdict = run.verdict

        run.duration_ms = int((time.monotonic() - started) * 1000)
        self._record(run, run_id)
        return run

    # ----------------------------------------------------------- planning
    def _plan(self, goal: str, ctx: AgentContext) -> tuple[list[Step], list[str], str]:
        instruction = (
            f"[Hedef]\n{goal}\n\n"
            f"[Kullanabileceğin roller]\n{roles.roster_text()}\n\n"
            f"En fazla {self.max_steps} adım kullan."
        )
        context = ctx.recall(goal) if ctx.grant.allows("memory.read") else ""
        result = run_agent(PLANNER, instruction, ctx, context_text=context)

        if not result.ok:
            return self._fallback(goal, f"planlayıcı çalışmadı: {result.error}")

        try:
            payload = json.loads(result.output)
        except json.JSONDecodeError:
            return self._fallback(goal, "planlayıcı çözümlenemeyen plan verdi")

        steps = self._validate_steps(payload.get("steps", []))
        if not steps:
            return self._fallback(goal, "planda kullanılabilir adım yok")

        criteria = [str(c).strip() for c in payload.get("criteria", []) if str(c).strip()]
        if not criteria:
            # Verification without criteria degenerates into "does this look fine",
            # which is precisely the judgement that cannot be trusted here.
            criteria = [f"Sonuç şu hedefi karşılamalı: {goal}"]
        return steps, criteria, PLAN_FROM_PLANNER

    def _validate_steps(self, raw: Any) -> list[Step]:
        """Clamp a model-authored plan to something the system can actually run."""
        steps: list[Step] = []
        if not isinstance(raw, list):
            return steps
        for entry in raw[: self.max_steps]:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role", "")).strip().lower()
            instruction = str(entry.get("instruction", "")).strip()
            if not instruction:
                continue
            if role not in roles.ASSIGNABLE:
                log.info("plan bilinmeyen rol istedi (%s) — generalist'e çevrildi", role)
                role = "generalist"
            steps.append(Step(role=role, instruction=instruction,
                              why=str(entry.get("why", "")).strip()))
        return steps

    def _fallback(self, goal: str, reason: str) -> tuple[list[Step], list[str], str]:
        """Planning failed. One generalist step beats refusing to try."""
        log.info("plan yedeğe düştü: %s", reason)
        self.events.publish("agent", "plan.fallback", f"Plan yedeğe düştü — {reason}",
                            level=WARN)
        return (
            [Step(role="generalist", instruction=goal, why=reason)],
            [f"Sonuç şu hedefi karşılamalı: {goal}"],
            PLAN_FROM_FALLBACK,
        )

    # ---------------------------------------------------------- execution
    def _execute(self, run: Orchestration, ctx: AgentContext, *, deadline: float,
                 should_stop) -> None:
        transcript: list[str] = []

        for index, step in enumerate(run.steps, start=1):
            if should_stop():
                step.result = AgentResult(step.role, False, error="durdurma istendi")
                continue
            if time.monotonic() > deadline:
                step.result = AgentResult(step.role, False, error="süre bütçesi doldu")
                self.events.publish("agent", "budget", "Süre bütçesi doldu — kalan adımlar atlandı",
                                    level=WARN, data={"run": ctx.run_id})
                continue

            spec = roles.get(step.role) or roles.GENERALIST
            step_ctx = AgentContext(
                brain=self.brain, events=self.events,
                grant=Grant.build(spec.name, spec.capabilities),
                model=self.model, should_stop=should_stop,
                memory=self.memory, run_id=ctx.run_id,
            )

            context_parts = []
            if step_ctx.grant.allows("memory.read"):
                recalled = step_ctx.recall(step.instruction)
                if recalled:
                    context_parts.append(recalled)
            if transcript:
                context_parts.append("### Önceki adımların çıktısı\n\n" + "\n\n".join(transcript))
            if any(previous.role == "web_researcher" and previous.ok
                   for previous in run.steps[:index - 1]):
                context_parts.append(SOURCED_CONTEXT_POLICY)

            header = f"[Ana hedef]\n{run.goal}\n\n[Bu adımın görevi ({index}/{len(run.steps)})]\n"
            if step.role == "web_researcher":
                step.attempts = 1
                step.result = self._research_step(
                    step.instruction, should_stop=should_stop, run_id=ctx.run_id)
            else:
                for attempt in range(STEP_RETRIES + 1):
                    step.attempts = attempt + 1
                    result = run_agent(spec, header + step.instruction, step_ctx,
                                       context_text="\n\n".join(context_parts))
                    step.result = result
                    if result.ok or should_stop():
                        break
                    if attempt < STEP_RETRIES:
                        self.events.publish("agent", "step.retry",
                                            f"{spec.title} tekrar deneniyor ({result.error})",
                                            level=WARN, data={"run": ctx.run_id, "step": index})

            if step.ok:
                transcript.append(f"[{spec.title}]\n{step.result.output}")
            else:
                # A failed step becomes a stated gap rather than a silent omission.
                transcript.append(
                    f"[{spec.title}] BU ADIM BAŞARISIZ OLDU ({step.result.error}). "
                    f"Yapılamayan iş: {step.instruction}"
                )

    def _research_step(self, question: str, *, should_stop, run_id: str) -> AgentResult:
        """Run the citation-bearing research pipeline as a real agent step."""
        started = time.monotonic()
        if self.research is None:
            return AgentResult("web_researcher", False,
                               error="kaynaklı araştırma sistemi bağlı değil")
        self.events.publish("agent", "research.start",
                            "Kaynaklı araştırma adımı başladı",
                            data={"run": run_id, "question": question[:160]})
        try:
            report = self.research.investigate(question, should_stop=should_stop)
        except Exception as exc:  # noqa: BLE001 - one step must stay isolated
            return AgentResult("web_researcher", False,
                               error=f"{type(exc).__name__}: {exc}",
                               duration_ms=int((time.monotonic() - started) * 1000))
        elapsed = int((time.monotonic() - started) * 1000)
        if not report.ok:
            return AgentResult("web_researcher", False,
                               error=report.error or "araştırma sonuç vermedi",
                               duration_ms=elapsed)
        sources = [
            {"title": page.title or page.url, "url": page.url}
            for page in report.pages
        ]
        source_lines = "\n".join(
            f"- [{item['title']}]({item['url']})" for item in sources)
        output = getattr(report, "answer", "") or report.synthesis
        if source_lines:
            output += "\n\nKaynaklar:\n" + source_lines
        return AgentResult(
            "web_researcher", True, output=output,
            data={"sources": sources,
                  "claims": len(report.claims),
                  "research_verified": report.verified,
                  "summary": report.summary()},
            duration_ms=elapsed, model=self.model,
            source="dogrulanmis-kaynak", verified=report.verified,
        )

    # ---------------------------------------------------------- aggregate
    def _aggregate(self, run: Orchestration, ctx: AgentContext) -> AgentResult:
        done = [step for step in run.steps if step.ok]
        if not done:
            errors = "; ".join(
                (step.result.error or "bilinmeyen") for step in run.steps if step.result
            )
            return AgentResult("orchestrator", False, error=f"tüm adımlar başarısız: {errors}")

        if len(run.steps) == 1:
            return done[0].result

        failed = [step for step in run.steps if not step.ok]
        pieces = [f"[{roles.get(s.role).title if roles.get(s.role) else s.role}]\n"
                  f"{s.result.output}" for s in done]
        if failed:
            pieces.append(
                "[Yapılamayanlar]\n"
                + "\n".join(f"- {s.instruction} ({s.result.error})" for s in failed)
            )

        instruction = (
            f"[Ana hedef]\n{run.goal}\n\n"
            + (f"[{SOURCED_CONTEXT_POLICY}]\n\n"
               if any(step.role == "web_researcher" and step.ok
                      for step in run.steps) else "")
            + f"[Birleştirilecek çıktılar]\n\n" + "\n\n".join(pieces)
        )
        summary_ctx = AgentContext(
            brain=self.brain, events=self.events,
            grant=Grant.build(SUMMARIZER.name, SUMMARIZER.capabilities),
            model=self.model, should_stop=ctx.should_stop, run_id=ctx.run_id,
        )
        merged = run_agent(SUMMARIZER, instruction, summary_ctx)
        # If merging itself fails, the last successful step is a worse answer than
        # a real synthesis but a far better one than nothing.
        return merged if merged.ok else done[-1].result

    # ------------------------------------------------------------ record
    def _record(self, run: Orchestration, run_id: str) -> None:
        if run.skill_name and self.skills:
            self.skills.record_run(run.skill_name, ok=run.verified)

        if (self.save_skills and self.skills and run.verified
                and run.plan_source == PLAN_FROM_PLANNER and len(run.steps) > 1
                and all(step.ok for step in run.steps)):
            skill = self.skills.save(
                title=_skill_title(run.goal),
                goal=run.goal,
                steps=[step.as_plan_entry() for step in run.steps],
                criteria=run.criteria,
            )
            self.events.publish("agent", "skill.saved",
                                f"Yeni beceri kaydedildi: {skill.title}",
                                data={"run": run_id, "skill": skill.name})

        level = SUCCESS if run.verified else (WARN if run.ok else ERROR)
        note = run.verdict.summary() if run.verdict else "doğrulanmadı"
        self.events.publish("agent", "run.done", f"Orkestrasyon bitti — {run.summary()} · {note}",
                            level=level,
                            data={"run": run_id, "verified": run.verified,
                                  "steps": len(run.steps), "ms": run.duration_ms})


def _as_steps(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for entry in entries:
        out.append({
            "role": str(entry.get("role", "generalist")),
            "instruction": str(entry.get("instruction", "")),
            "why": str(entry.get("why", "")),
        })
    return [e for e in out if e["instruction"]]


def _skill_title(goal: str) -> str:
    words = goal.strip().split()
    return " ".join(words[:8]) + ("…" if len(words) > 8 else "")
