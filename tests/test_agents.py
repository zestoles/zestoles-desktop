"""Agent orchestration tests.

These run against a fake brain rather than the local model. That is the point: the
behaviour worth pinning is what the orchestrator does when a model returns
something wrong — an unknown role, forty steps, unparseable JSON, an empty answer,
a failure halfway through. A real model would produce sensible output most of the
time and test almost none of it.

The provenance and identifier-extraction tests guard the specific failure this
project has already had: a confident invention reaching permanent storage.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.agents import roles  # noqa: E402
from jarvis.agents.base import AGENT_SOURCED, AgentContext, AgentResult, run_agent  # noqa: E402
from jarvis.agents.orchestrator import (  # noqa: E402
    PLAN_FROM_FALLBACK,
    PLAN_FROM_PLANNER,
    PLAN_FROM_SKILL,
    Orchestrator,
)
from jarvis.agents.permissions import (  # noqa: E402
    CLOUD_BRAIN,
    FS_WRITE,
    MEMORY_READ,
    SHELL,
    Grant,
    PermissionDenied,
)
from jarvis.agents.skills import SkillLibrary, keywords_of  # noqa: E402
from jarvis.agents.verify import (  # noqa: E402
    known_from_user,
    mechanical_checks,
    suspicious_identifiers,
)
from jarvis.autonomy.events import EventLog  # noqa: E402
from jarvis.memory.distill import UNVERIFIED_SOURCES  # noqa: E402


class FakeLocal:
    """Returns queued replies in order; falls back to a default once exhausted."""

    def __init__(self, replies=None, default="varsayılan cevap yeterince uzun olmalı ki geçsin"):
        self.replies = list(replies or [])
        self.default = default
        self.calls: list[dict] = []

    def chat(self, messages, *, temperature=None, schema=None, model=None, think=None):
        self.calls.append({"messages": messages, "schema": schema, "model": model})
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return self.default


class FakeBrain:
    def __init__(self, local):
        self.local = local


class FakeConfig:
    def __init__(self, values=None):
        self.values = {
            "agents.max_steps": 5,
            "agents.total_timeout_s": 60,
            "agents.save_skills": True,
            "agents.model": "fast",
            "agents.verify_model": "fast",
            "local.model": "test-model",
            "local.model_heavy": "test-heavy",
        }
        self.values.update(values or {})

    def get(self, key, default=None):
        return self.values.get(key, default)

    def path(self, key, default=""):
        return Path(self.values.get(key, default))


def plan_json(steps, criteria=("ölçüt bir",)):
    return json.dumps({"steps": steps, "criteria": list(criteria)})


def verdict_json(ok=True, confidence=0.9, issues=(), identifiers=(), criteria=()):
    return json.dumps({
        "ok": ok, "confidence": confidence, "issues": list(issues),
        "identifiers": list(identifiers), "criteria": list(criteria), "note": "",
    })


LONG = "Bu yeterince uzun bir çıktı; mekanik kontrollerden geçmesi gerekiyor."


class Harness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "t.db"
        self.events = EventLog(self.db)

    def tearDown(self):
        self._tmp.cleanup()

    def orchestrator(self, replies, **config):
        local = FakeLocal(replies)
        self.local = local
        return Orchestrator(
            FakeBrain(local), self.events,
            config=FakeConfig(config), skills=SkillLibrary(self.db),
        )


class TestPermissions(unittest.TestCase):
    def test_shell_is_refused_even_when_requested(self):
        grant = Grant.build("x", frozenset({SHELL, MEMORY_READ}))
        self.assertFalse(grant.allows(SHELL))
        self.assertTrue(grant.allows(MEMORY_READ))

    def test_filesystem_writes_are_refused(self):
        self.assertFalse(Grant.build("x", frozenset({FS_WRITE})).allows(FS_WRITE))

    def test_cloud_needs_an_explicit_opt_in(self):
        self.assertFalse(Grant.build("x", frozenset({CLOUD_BRAIN})).allows(CLOUD_BRAIN))
        self.assertTrue(
            Grant.build("x", frozenset({CLOUD_BRAIN}), allow_cloud=True).allows(CLOUD_BRAIN)
        )

    def test_require_raises_with_the_agent_named(self):
        with self.assertRaises(PermissionDenied) as caught:
            Grant.build("araştırmacı", frozenset()).require(MEMORY_READ)
        self.assertIn("araştırmacı", str(caught.exception))

    def test_unknown_capability_is_rejected(self):
        with self.assertRaises(ValueError):
            Grant.build("x", frozenset({"uydurma.yetki"}))


class TestContextEnforcement(Harness):
    def test_recall_refuses_without_permission(self):
        ctx = AgentContext(brain=FakeBrain(FakeLocal()), events=self.events,
                           grant=Grant.build("x", frozenset()), model="m")
        with self.assertRaises(PermissionDenied):
            ctx.recall("herhangi bir şey")

    def test_recall_returns_empty_when_memory_is_absent(self):
        ctx = AgentContext(brain=FakeBrain(FakeLocal()), events=self.events,
                           grant=Grant.build("x", frozenset({MEMORY_READ})), model="m")
        self.assertEqual(ctx.recall("soru"), "")


class TestAgentRun(Harness):
    def _ctx(self, local):
        return AgentContext(brain=FakeBrain(local), events=self.events,
                            grant=Grant.build("x", frozenset()), model="test-model")

    def test_output_is_agent_sourced(self):
        result = run_agent(roles.GENERALIST, "soru", self._ctx(FakeLocal([LONG])))
        self.assertEqual(result.source, AGENT_SOURCED)
        self.assertIn(AGENT_SOURCED, UNVERIFIED_SOURCES)

    def test_a_raising_model_becomes_a_failed_result_not_an_exception(self):
        result = run_agent(roles.GENERALIST, "soru",
                           self._ctx(FakeLocal([OSError("bağlantı yok")])))
        self.assertFalse(result.ok)
        self.assertIn("bağlantı yok", result.error)

    def test_empty_output_fails(self):
        self.assertFalse(run_agent(roles.GENERALIST, "s", self._ctx(FakeLocal(["   "]))).ok)

    def test_output_is_truncated_at_the_declared_limit(self):
        local = FakeLocal(["x" * 50000])
        result = run_agent(roles.GENERALIST, "s", self._ctx(local))
        self.assertLessEqual(len(result.output), roles.GENERALIST.max_output_chars + 20)
        self.assertTrue(result.output.endswith("[kısaltıldı]"))

    def test_the_run_model_is_passed_to_the_call(self):
        local = FakeLocal([LONG])
        run_agent(roles.GENERALIST, "s", self._ctx(local))
        self.assertEqual(local.calls[0]["model"], "test-model")


class TestVerifyHelpers(unittest.TestCase):
    def test_camelcase_identifiers_are_extracted(self):
        names = suspicious_identifiers("ProfileService yerine AsyncResultStorage kullan.")
        self.assertIn("AsyncResultStorage", names)
        self.assertIn("ProfileService", names)

    def test_dotted_calls_are_extracted(self):
        self.assertTrue(any("store.get" in n for n in suspicious_identifiers("store.getAsync(x)")))

    def test_plain_prose_yields_nothing(self):
        self.assertEqual(suspicious_identifiers("bu cümlede özel bir isim yok"), [])

    def test_short_output_is_flagged(self):
        problems = mechanical_checks(AgentResult("a", True, output="kısa"))
        self.assertTrue(any("kısa" in p for p in problems))

    def test_refusal_is_flagged(self):
        problems = mechanical_checks(
            AgentResult("a", True, output="Üzgünüm, bu konuda yardımcı olamam." + " x" * 40)
        )
        self.assertTrue(any("reddetme" in p for p in problems))

    def test_truncation_is_flagged(self):
        problems = mechanical_checks(
            AgentResult("a", True, output="uzun bir metin " * 5 + "…[kısaltıldı]")
        )
        self.assertTrue(any("kesildi" in p for p in problems))

    def test_failed_step_short_circuits(self):
        problems = mechanical_checks(AgentResult("a", False, error="patladı"))
        self.assertEqual(len(problems), 1)


class TestPlanValidation(Harness):
    def test_unknown_role_falls_back_to_generalist(self):
        orch = self.orchestrator([
            plan_json([{"role": "uydurma_rol", "instruction": "bir şey yap"}]),
            LONG, verdict_json(),
        ])
        run = orch.run("hedef")
        self.assertEqual(run.steps[0].role, "generalist")

    def test_step_count_is_clamped(self):
        many = [{"role": "generalist", "instruction": f"adım {i}"} for i in range(30)]
        orch = self.orchestrator([plan_json(many)] + [LONG] * 40 + [verdict_json()],
                                 **{"agents.max_steps": 3})
        run = orch.run("hedef")
        self.assertLessEqual(len(run.steps), 3)

    def test_unparseable_plan_falls_back_to_one_step(self):
        orch = self.orchestrator(["bu JSON değil", LONG, verdict_json()])
        run = orch.run("hedef")
        self.assertEqual(run.plan_source, PLAN_FROM_FALLBACK)
        self.assertEqual(len(run.steps), 1)

    def test_planner_failure_falls_back(self):
        orch = self.orchestrator([OSError("model kapalı"), LONG, verdict_json()])
        run = orch.run("hedef")
        self.assertEqual(run.plan_source, PLAN_FROM_FALLBACK)

    def test_missing_criteria_get_a_default(self):
        """Verification with no criteria degenerates into 'looks fine to me'."""
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}], criteria=[]),
            LONG, verdict_json(),
        ])
        run = orch.run("hedef")
        self.assertTrue(run.criteria)

    def test_steps_without_instructions_are_dropped(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": ""},
                       {"role": "analyst", "instruction": "gerçek adım"}]),
            LONG, verdict_json(),
        ])
        run = orch.run("hedef")
        self.assertEqual(len(run.steps), 1)


class TestExecution(Harness):
    @staticmethod
    def _research():
        class Page:
            title = "Resmî belge"
            url = "https://example.com/official"

        class Report:
            ok = True
            error = ""
            synthesis = LONG
            pages = [Page()]
            claims = []
            verified = True

            @staticmethod
            def summary():
                return "1 kaynak doğrulandı"

        class Research:
            def __init__(self):
                self.questions = []

            def investigate(self, question, *, should_stop):
                self.questions.append(question)
                return Report()

        return Research()

    def test_web_research_role_uses_the_sourced_pipeline(self):
        orch = self.orchestrator([
            plan_json([{"role": "web_researcher",
                        "instruction": "güncel sürümü doğrula"}]),
            verdict_json(),
        ])
        research = self._research()
        orch.research = research

        run = orch.run("güncel bilgi gerekli")

        self.assertTrue(run.steps[0].ok)
        self.assertEqual(research.questions, ["güncel sürümü doğrula"])
        self.assertIn("https://example.com/official", run.steps[0].result.output)
        self.assertTrue(run.steps[0].result.data["research_verified"])

    def test_later_agents_must_stay_inside_sourced_findings(self):
        orch = self.orchestrator([
            plan_json([{"role": "web_researcher", "instruction": "araştır"},
                       {"role": "analyst", "instruction": "yorumla"}]),
            LONG, LONG, verdict_json(),
        ])
        orch.research = self._research()

        orch.run("güncel ürün bilgisi")

        analyst_prompt = json.dumps(self.local.calls[1]["messages"], ensure_ascii=False)
        summary_prompt = json.dumps(self.local.calls[2]["messages"], ensure_ascii=False)
        for prompt in (analyst_prompt, summary_prompt):
            self.assertIn("Kaynak disiplini", prompt)
            self.assertIn("Yeni bir gerçek", prompt)
            self.assertIn("kaynaklarda doğrulanmadı", prompt)

    def test_web_research_role_fails_honestly_when_not_connected(self):
        orch = self.orchestrator([
            plan_json([{"role": "web_researcher", "instruction": "araştır"}]),
        ])
        run = orch.run("güncel bilgi gerekli")
        self.assertFalse(run.ok)
        self.assertIn("bağlı değil", run.steps[0].result.error)

    def test_a_failing_step_does_not_fail_the_run(self):
        orch = self.orchestrator([
            plan_json([{"role": "researcher", "instruction": "topla"},
                       {"role": "analyst", "instruction": "incele"}]),
            OSError("birinci adım öldü"), OSError("tekrar da öldü"),  # step 1 + its retry
            LONG,                                                     # step 2
            LONG,                                                     # summariser
            verdict_json(),
        ])
        run = orch.run("hedef")
        self.assertFalse(run.steps[0].ok)
        self.assertTrue(run.steps[1].ok)
        self.assertTrue(run.ok)

    def test_a_failed_step_is_retried_once(self):
        orch = self.orchestrator([
            plan_json([{"role": "analyst", "instruction": "incele"}]),
            OSError("ilk deneme"), LONG, verdict_json(),
        ])
        run = orch.run("hedef")
        self.assertEqual(run.steps[0].attempts, 2)
        self.assertTrue(run.steps[0].ok)

    def test_all_steps_failing_fails_the_run(self):
        orch = self.orchestrator([
            plan_json([{"role": "analyst", "instruction": "incele"}]),
            OSError("bir"), OSError("iki"),
        ])
        run = orch.run("hedef")
        self.assertFalse(run.ok)
        self.assertFalse(run.verified)

    def test_stop_request_is_honoured(self):
        orch = self.orchestrator([
            plan_json([{"role": "analyst", "instruction": "a"},
                       {"role": "coder", "instruction": "b"}]),
        ])
        run = orch.run("hedef", should_stop=lambda: True)
        self.assertFalse(any(step.ok for step in run.steps))

    def test_gap_from_a_failed_step_reaches_the_summariser(self):
        """A step that died must remain visible, not vanish from the aggregate."""
        orch = self.orchestrator([
            plan_json([{"role": "researcher", "instruction": "kritik veri topla"},
                       {"role": "analyst", "instruction": "incele"}]),
            OSError("öldü"), OSError("yine öldü"),
            LONG, LONG, verdict_json(),
        ])
        orch.run("hedef")
        summariser_call = self.local.calls[-2]
        sent = json.dumps(summariser_call["messages"], ensure_ascii=False)
        self.assertIn("Yapılamayanlar", sent)


class TestVerification(Harness):
    def test_rejected_synthesis_is_not_exposed_as_the_output(self):
        raw = "Bu yanlış sentez kullanıcıya gösterilmemeli. " * 4
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            raw, verdict_json(ok=False, issues=["dayanaksız"]),
        ])
        run = orch.run("hedef")
        self.assertFalse(run.verified)
        self.assertNotIn("yanlış sentez", run.output)
        self.assertIn("doğrulama kapısını geçmedi", run.output)

    def test_an_unmet_criterion_blocks_the_verdict(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}],
                      criteria=["session-locking açıklanmalı"]),
            LONG,
            verdict_json(ok=True, criteria=[{"index": 1, "met": False, "why": "hiç değinilmemiş"}]),
        ])
        run = orch.run("hedef")
        self.assertTrue(run.ok)
        self.assertFalse(run.verified)
        self.assertIn("session-locking", run.verdict.unmet[0])

    def test_met_criteria_do_not_block(self):
        """Confirming a criterion used to be filed as a complaint and fail the run."""
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}],
                      criteria=["bir ölçüt"]),
            LONG,
            verdict_json(ok=True, criteria=[{"index": 1, "met": True, "why": "sağlanmıştır"}]),
        ])
        self.assertTrue(orch.run("hedef").verified)

    def test_free_text_issues_are_kept_but_do_not_gate(self):
        """Prose is commentary; the gate is structural. The ok flag still counts."""
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG,
            verdict_json(ok=True, issues=["bu ölçüt sağlanmıştır"]),
        ])
        run = orch.run("hedef")
        self.assertTrue(run.verified)
        self.assertEqual(run.verdict.issues, ["bu ölçüt sağlanmıştır"])

    def test_the_verifier_ok_flag_still_blocks(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG,
            verdict_json(ok=False, issues=["dayanaksız iddia var"]),
        ])
        self.assertFalse(orch.run("hedef").verified)

    def test_mechanical_failure_blocks_even_when_the_model_approves(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            "kısa",  # too short; the model would still bless it
            verdict_json(ok=True),
        ])
        run = orch.run("hedef")
        self.assertFalse(run.verified)

    def test_a_doubtful_identifier_blocks_on_its_own(self):
        """The AsyncResultStorage case: every criterion met, one invented name."""
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG,
            verdict_json(ok=True, identifiers=[{"name": "AsyncResultStorage",
                                                "verdict": "emin degil"}]),
        ])
        run = orch.run("hedef")
        self.assertFalse(run.verified)
        self.assertIn("AsyncResultStorage", run.verdict.doubtful_names)

    def test_a_confirmed_identifier_does_not_block(self):
        """Confirming a real service used to land in issues and fail the run."""
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG,
            verdict_json(ok=True, identifiers=[{"name": "DataStoreService",
                                                "verdict": "var"}]),
        ])
        run = orch.run("hedef")
        self.assertTrue(run.verified)
        self.assertIn("DataStoreService", run.verdict.confirmed_names)

    def test_a_nonexistent_identifier_blocks(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG,
            verdict_json(ok=True, identifiers=[{"name": "UyduruServis", "verdict": "yok"}]),
        ])
        self.assertFalse(orch.run("hedef").verified)

    def test_unparseable_verdict_means_unverified(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG, "JSON değil",
        ])
        run = orch.run("hedef")
        self.assertFalse(run.verified)

    def test_verified_run_reports_verified(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG, verdict_json(ok=True, confidence=0.8),
        ])
        run = orch.run("hedef")
        self.assertTrue(run.verified)
        self.assertTrue(run.final.verified)


class FakeHit:
    def __init__(self, text, source):
        self.text = text
        self.source = source
        self.note_title = "not"
        self.note_kind = "proje"


class FakeMemory:
    def __init__(self, hits=()):
        self.hits = list(hits)

    def search(self, query, limit=5):
        return [h for h in self.hits if query.casefold() in h.text.casefold()][:limit]

    def context_for(self, query):
        return ""


class TestIdentifierClearing(Harness):
    """A name the user uses is not an invention, whatever a 9B model thinks."""

    def test_user_sourced_name_is_cleared(self):
        memory = FakeMemory([FakeHit("Veri kaydı için ProfileService kullanıyorum.", "kullanici")])
        self.assertTrue(known_from_user(memory, "ProfileService"))

    def test_agent_sourced_name_is_not_cleared(self):
        """Otherwise an invention could confirm itself through its own note."""
        memory = FakeMemory([FakeHit("AsyncResultStorage önerilir.", "ajan")])
        self.assertFalse(known_from_user(memory, "AsyncResultStorage"))

    def test_summary_sourced_name_is_not_cleared(self):
        memory = FakeMemory([FakeHit("AsyncResultStorage tartışıldı.", "oturum-ozeti")])
        self.assertFalse(known_from_user(memory, "AsyncResultStorage"))

    def test_absent_name_is_not_cleared(self):
        self.assertFalse(known_from_user(FakeMemory(), "UyduruServis"))

    def test_clearing_unblocks_a_run(self):
        local = FakeLocal([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG,
            verdict_json(ok=True, identifiers=[{"name": "ProfileService",
                                                "verdict": "emin degil"}]),
        ])
        memory = FakeMemory([FakeHit("ProfileService kullanıyorum.", "kullanici")])
        orch = Orchestrator(FakeBrain(local), self.events, memory=memory,
                            config=FakeConfig(), skills=SkillLibrary(self.db))
        run = orch.run("hedef")
        self.assertTrue(run.verified)
        self.assertIn("ProfileService", run.verdict.confirmed_names)

    def test_an_invented_name_still_blocks_with_memory_present(self):
        local = FakeLocal([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG,
            verdict_json(ok=True, identifiers=[{"name": "AsyncResultStorage",
                                                "verdict": "emin degil"}]),
        ])
        memory = FakeMemory([FakeHit("ProfileService kullanıyorum.", "kullanici")])
        orch = Orchestrator(FakeBrain(local), self.events, memory=memory,
                            config=FakeConfig(), skills=SkillLibrary(self.db))
        self.assertFalse(orch.run("hedef").verified)


class TestSkills(Harness):
    def test_keywords_ignore_stopwords(self):
        self.assertNotIn("bir", keywords_of("bir roblox tycoon oyunu"))
        self.assertIn("roblox", keywords_of("bir roblox tycoon oyunu"))

    def test_verified_multistep_run_is_saved_as_a_skill(self):
        orch = self.orchestrator([
            plan_json([{"role": "researcher", "instruction": "topla"},
                       {"role": "analyst", "instruction": "incele"}]),
            LONG, LONG, LONG, verdict_json(ok=True),
        ])
        orch.run("roblox tycoon veri saklama stratejisi")
        self.assertEqual(len(orch.skills.list()), 1)

    def test_unverified_run_is_not_saved(self):
        orch = self.orchestrator([
            plan_json([{"role": "researcher", "instruction": "topla"},
                       {"role": "analyst", "instruction": "incele"}]),
            LONG, LONG, LONG, verdict_json(ok=False, issues=["eksik"]),
        ])
        orch.run("roblox tycoon veri saklama stratejisi")
        self.assertEqual(orch.skills.list(), [])

    def test_single_step_run_is_not_saved(self):
        """A one-step plan is not a workflow worth remembering."""
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG, verdict_json(ok=True),
        ])
        orch.run("tek adımlık iş")
        self.assertEqual(orch.skills.list(), [])

    def test_a_saved_skill_is_reused_and_skips_planning(self):
        library = SkillLibrary(self.db)
        library.save("Roblox veri stratejisi", "roblox tycoon veri saklama stratejisi",
                     [{"role": "analyst", "instruction": "incele", "why": ""}],
                     ["ölçüt"])
        local = FakeLocal([LONG, verdict_json(ok=True)])
        orch = Orchestrator(FakeBrain(local), self.events,
                            config=FakeConfig(), skills=library)
        run = orch.run("roblox tycoon veri saklama stratejisi nasıl olmalı")
        self.assertEqual(run.plan_source, PLAN_FROM_SKILL)
        self.assertEqual(run.skill_name, "roblox-veri-stratejisi")

    def test_unrelated_goal_does_not_match_a_skill(self):
        library = SkillLibrary(self.db)
        library.save("Roblox veri", "roblox tycoon datastore profileservice",
                     [{"role": "analyst", "instruction": "x", "why": ""}], [])
        self.assertIsNone(library.find("akşam yemeği için tarif öner"))

    def test_a_repeatedly_failing_skill_retires_itself(self):
        library = SkillLibrary(self.db)
        library.save("Kötü beceri", "hedef kelimeleri burada",
                     [{"role": "analyst", "instruction": "x", "why": ""}], [])
        for _ in range(3):
            library.record_run("kotu-beceri", ok=False)
        self.assertTrue(library.get("kotu-beceri").retired)
        self.assertIsNone(library.find("hedef kelimeleri burada"))

    def test_success_keeps_a_skill_active(self):
        library = SkillLibrary(self.db)
        library.save("İyi beceri", "hedef kelimeleri burada",
                     [{"role": "analyst", "instruction": "x", "why": ""}], [])
        for _ in range(3):
            library.record_run("iyi-beceri", ok=True)
        self.assertFalse(library.get("iyi-beceri").retired)


class TestEvents(Harness):
    def test_a_run_emits_start_and_done(self):
        orch = self.orchestrator([
            plan_json([{"role": "generalist", "instruction": "yap"}]),
            LONG, verdict_json(),
        ])
        orch.run("hedef")
        kinds = {event.kind for event in self.events.since(60)}
        self.assertIn("run.start", kinds)
        self.assertIn("run.done", kinds)


class TestRoster(unittest.TestCase):
    def test_planner_and_verifier_are_not_assignable(self):
        """A plan that could schedule its verifier could also schedule it away."""
        self.assertNotIn("planner", roles.ASSIGNABLE)
        self.assertNotIn("verifier", roles.ASSIGNABLE)

    def test_every_assignable_role_exists(self):
        for name in roles.ASSIGNABLE:
            self.assertIsNotNone(roles.get(name))

    def test_roster_text_lists_assignable_roles(self):
        text = roles.roster_text()
        self.assertIn("analyst", text)
        self.assertIn("web_researcher", text)
        self.assertNotIn("verifier —", text)


if __name__ == "__main__":
    unittest.main()
