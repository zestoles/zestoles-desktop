"""Self-improvement engine tests.

The questions that decide whether this layer is safe to leave running unattended:

  does it notice what is actually wrong, from evidence rather than introspection
  can a revenue guess be stored as if it were known
  does a refused category stay refused however good its numbers look
  does the same idea get tried twice
  does a failure teach anything, and does the lesson change what happens next
  do the budgets hold when nobody is awake
  does a failed experiment leave production alone

The scoring and screening tests matter most. Everything else in the system can be
wrong and produce a bad suggestion; this is the part that decides what the machine
spends its nights doing, and its failures are quiet.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.improve.budget import EXPERIMENT, HYPOTHESIS, RESEARCH, ImprovementBudget  # noqa: E402
from jarvis.improve.capabilities import (  # noqa: E402
    BROKEN,
    MISSING,
    PARTIAL,
    WORKING,
    Capability,
    CapabilityRegistry,
)
from jarvis.improve.gaps import (  # noqa: E402
    FROM_CAPABILITY,
    FROM_ERRORS,
    FROM_EXPERIMENTS,
    FROM_STALE,
    FROM_TASKS,
    Gap,
    GapDetector,
    shape_of,
)
from jarvis.improve.hypotheses import (  # noqa: E402
    CONFIRMED,
    MAX_ATTEMPTS,
    PROPOSED,
    REFUTED,
    SHELVED,
    HypothesisStore,
    Lesson,
    fingerprint,
)
from jarvis.improve.opportunity import (  # noqa: E402
    ESTIMATED,
    GUESS,
    MEASURED,
    SOURCED,
    Dimension,
    Estimate,
    Opportunity,
    rank,
    recordable_estimates,
    score,
    screen,
)
from jarvis.improve.preferences import PreferenceStore  # noqa: E402
from jarvis.memory.distill import UNVERIFIED_SOURCES  # noqa: E402


def dims(**overrides):
    base = {name: 0.7 for name in
            ("revenue", "feasibility", "time", "resource_fit",
             "competition", "risk", "confidence")}
    base.update(overrides)
    return {name: Dimension(name, value, MEASURED, "test") for name, value in base.items()}


class ImproveCase(unittest.TestCase):
    def setUp(self):
        # SQLite in WAL mode keeps -wal and -shm files open briefly after the last
        # connection closes; on Windows that makes directory removal fail. The
        # temporary directory is disposable either way.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self._tmp.name) / "jarvis.db"

    def tearDown(self):
        self._tmp.cleanup()


# ------------------------------------------------------------------ capabilities
class TestCapabilityRegistry(ImproveCase):
    def setUp(self):
        super().setUp()
        self.registry = CapabilityRegistry(self.db)

    def test_seed_records_what_is_missing_too(self):
        """A registry of only what works can never notice an absence."""
        missing = {c.name for c in self.registry.list(status=MISSING)}
        self.assertIn("voice.io", missing)
        self.assertIn("browser.automation", missing)

    def test_seed_is_idempotent(self):
        before = len(self.registry.list())
        self.assertEqual(CapabilityRegistry(self.db).seed(), 0)
        self.assertEqual(len(self.registry.list()), before)

    def test_known_limits_are_recorded(self):
        research = self.registry.get("research.web")
        self.assertTrue(research.limits)
        self.assertTrue(any("SearXNG" in limit for limit in research.limits))

    def test_only_a_measurement_refreshes_verification(self):
        self.registry.set_status("code.writing", PARTIAL)
        self.assertIsNone(self.registry.get("code.writing").last_verified)

        self.registry.record_benchmark("code.writing", benchmark="tests", score=0.8)
        self.assertIsNotNone(self.registry.get("code.writing").last_verified)

    def test_unmeasured_capability_is_stale(self):
        self.assertTrue(self.registry.get("code.writing").stale)

    def test_working_but_stale_still_carries_weight(self):
        """Working-and-unverified is not the same as working."""
        capability = self.registry.get("memory.retrieval")
        self.assertEqual(capability.status, WORKING)
        self.assertGreater(capability.gap_weight, 0.0)

    def test_freshly_measured_working_capability_has_no_gap(self):
        self.registry.record_benchmark("memory.retrieval", benchmark="t", score=1.0)
        self.assertEqual(self.registry.get("memory.retrieval").gap_weight, 0.0)

    def test_missing_outranks_partial(self):
        self.assertGreater(
            self.registry.get("voice.io").gap_weight,
            self.registry.get("code.writing").gap_weight)

    def test_unknown_status_is_refused(self):
        with self.assertRaises(ValueError):
            self.registry.set_status("voice.io", "harika")

    def test_add_limit_does_not_duplicate(self):
        self.registry.add_limit("voice.io", "test sınırı")
        self.registry.add_limit("voice.io", "test sınırı")
        self.assertEqual(self.registry.get("voice.io").limits.count("test sınırı"), 1)


# ------------------------------------------------------------------- gap finding
class TestGapDetection(ImproveCase):
    def setUp(self):
        super().setUp()
        self.capabilities = CapabilityRegistry(self.db)
        self.detector = GapDetector(self.db, self.capabilities)

    def _make_tables(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS tasks (kind TEXT, state TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS events "
                         "(ts REAL, level TEXT, source TEXT, kind TEXT, message TEXT)")

    def test_missing_capability_becomes_a_gap(self):
        gaps = self.detector.detect(limit=50)
        self.assertTrue(any(g.capability == "voice.io" and g.source == FROM_CAPABILITY
                            for g in gaps))

    def test_gaps_carry_their_evidence(self):
        gap = next(g for g in self.detector.detect(limit=50) if g.capability == "research.web")
        self.assertTrue(gap.evidence)

    def test_gaps_are_ordered_by_severity(self):
        gaps = self.detector.detect(limit=50)
        self.assertEqual(gaps, sorted(gaps, key=lambda g: -g.severity))

    def test_quarantined_tasks_become_a_gap(self):
        self._make_tables()
        with sqlite3.connect(self.db) as conn:
            for _ in range(4):
                conn.execute("INSERT INTO tasks (kind, state) VALUES ('research.investigate',"
                             " 'quarantined')")
        gaps = self.detector.detect(limit=50)
        self.assertTrue(any(g.source == FROM_TASKS for g in gaps))

    def test_a_fixture_runner_is_not_a_gap(self):
        """`fail` raises on purpose so retry and quarantine can be exercised.
        The task queued to test that path stayed quarantined, and the detector
        read it as "this kind keeps being given up on" — an experiment-shaped
        gap, because FROM_TASKS is in COMPARABLE_SOURCES. So a fixture working
        exactly as written could spend a hypothesis and one of four daily
        experiment slots on itself.
        """
        self._make_tables()
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO tasks (kind, state) VALUES ('fail', 'quarantined')")
        gaps = self.detector.detect(limit=50)
        self.assertFalse([g for g in gaps if g.source == FROM_TASKS], gaps)

    def test_a_real_quarantined_kind_is_still_a_gap_alongside_a_fixture(self):
        """The exclusion is one name, not the signal."""
        self._make_tables()
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO tasks (kind, state) VALUES ('fail', 'quarantined')")
            for _ in range(3):
                conn.execute("INSERT INTO tasks (kind, state)"
                             " VALUES ('research.investigate', 'quarantined')")
        titles = [g.title for g in self.detector.detect(limit=50) if g.source == FROM_TASKS]
        self.assertTrue(any("research.investigate" in t for t in titles), titles)
        self.assertFalse(any("'fail'" in t for t in titles), titles)

    def test_repeated_errors_become_a_gap(self):
        self._make_tables()
        now = time.time()
        with sqlite3.connect(self.db) as conn:
            for _ in range(5):
                conn.execute("INSERT INTO events (ts, level, source, kind, message)"
                             " VALUES (?, 'error', 'agent', 'error', 'patladı')", (now,))
        self.assertTrue(any(g.source == FROM_ERRORS for g in self.detector.detect(limit=50)))

    def test_a_single_error_is_not_a_pattern(self):
        self._make_tables()
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO events (ts, level, source, kind, message)"
                         " VALUES (?, 'error', 'agent', 'error', 'bir kez')", (time.time(),))
        self.assertFalse(any(g.source == FROM_ERRORS for g in self.detector.detect(limit=50)))

    def test_missing_tables_do_not_break_detection(self):
        """A fresh install has no tasks or events yet."""
        self.assertTrue(self.detector.detect(limit=50))


# ---------------------------------------------------------------- gap shape
class TestExperimentShape(unittest.TestCase):
    """Which gaps may reach the planner, decided from recorded facts.

    An experiment measures a change against what came before. A gap with no
    "before" has nothing to compare, and S6b measured what happens when one is
    sent to the planner anyway: three of the four live plans came from
    capabilities recorded as *missing*, and the model invented a baseline
    because it had been asked for one.
    """

    def test_a_missing_capability_has_nothing_to_compare(self):
        shaped, reason = shape_of(FROM_CAPABILITY, status=MISSING)
        self.assertFalse(shaped)
        self.assertIn("baseline", reason)

    def test_a_broken_capability_is_its_own_baseline(self):
        self.assertTrue(shape_of(FROM_CAPABILITY, status=BROKEN)[0])

    def test_a_partial_capability_is_its_own_baseline(self):
        self.assertTrue(shape_of(FROM_CAPABILITY, status=PARTIAL)[0])

    def test_a_missing_capability_with_a_measurement_can_be_compared(self):
        """A recorded benchmark is a "before" even when the status says missing."""
        self.assertTrue(shape_of(FROM_CAPABILITY, status=MISSING, has_benchmark=True)[0])

    def test_stale_verification_asks_for_a_measurement_not_a_change(self):
        shaped, reason = shape_of(FROM_STALE)
        self.assertFalse(shaped)
        self.assertIn("ölçüm", reason)

    def test_things_that_already_misbehave_are_comparable(self):
        for source in (FROM_ERRORS, FROM_EXPERIMENTS, FROM_TASKS):
            with self.subTest(source=source):
                self.assertTrue(shape_of(source)[0])

    def test_an_unknown_source_fails_closed(self):
        shaped, reason = shape_of("yeni-bir-sinyal")
        self.assertFalse(shaped)
        self.assertIn("bilinmeyen", reason)

    def test_the_rule_consults_no_model(self):
        """Arithmetic on recorded facts. There is nothing here to persuade."""
        import inspect

        body = inspect.getsource(shape_of).lower()
        for forbidden in ("self.brain", "chat(", "llm", "prompt"):
            self.assertNotIn(forbidden, body)


class TestDetectedGapsCarryTheirShape(ImproveCase):
    def setUp(self):
        super().setUp()
        self.capabilities = CapabilityRegistry(self.db)
        self.capabilities.seed()
        self.detector = GapDetector(self.db, self.capabilities)

    def test_a_seeded_missing_capability_is_not_experiment_shaped(self):
        gap = next(g for g in self.detector.detect(limit=50)
                   if g.capability == "voice.io")
        self.assertFalse(gap.experiment_shaped)
        self.assertTrue(gap.shape_reason)

    def test_a_measurement_turns_it_into_a_candidate(self):
        self.capabilities.record_benchmark("voice.io", benchmark="ses", score=0.4)
        gap = next(g for g in self.detector.detect(limit=50)
                   if g.capability == "voice.io")
        self.assertTrue(gap.experiment_shaped, gap.shape_reason)

    def test_quarantined_task_gaps_are_experiment_shaped(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS tasks (kind TEXT, state TEXT)")
            for _ in range(4):
                conn.execute("INSERT INTO tasks (kind, state)"
                             " VALUES ('research.investigate', 'quarantined')")
        gap = next(g for g in self.detector.detect(limit=50) if g.source == FROM_TASKS)
        self.assertTrue(gap.experiment_shaped)

    def test_the_shape_survives_serialisation(self):
        payload = next(iter(self.detector.detect(limit=50))).as_dict()
        self.assertIn("experiment_shaped", payload)
        self.assertIn("shape_reason", payload)

    def test_a_hand_built_gap_defaults_to_refused(self):
        """Anything nobody classified is not a planning candidate."""
        self.assertFalse(Gap(key="k", source="x", title="t", severity=0.5).experiment_shaped)


# -------------------------------------------------------------------- estimates
class TestEstimateProvenance(unittest.TestCase):
    def test_a_guess_is_speculative(self):
        self.assertTrue(Estimate(5000, "USD/ay", GUESS, 0.9,
                                 evidence=["içime doğdu"]).speculative)

    def test_evidence_free_measurement_is_speculative(self):
        self.assertTrue(Estimate(5000, "USD/ay", SOURCED, 0.9).speculative)

    def test_low_confidence_is_speculative(self):
        self.assertTrue(Estimate(5000, "USD/ay", SOURCED, 0.3, evidence=["kaynak"]).speculative)

    def test_a_sourced_confident_estimate_is_not(self):
        self.assertFalse(Estimate(120, "oyuncu/gün", SOURCED, 0.8,
                                  evidence=["https://kaynak"]).speculative)

    def test_speculative_label_says_so(self):
        self.assertIn("spekülatif", Estimate(5000, "USD", GUESS, 0.9).label)

    def test_revenue_guess_is_not_recordable(self):
        """The sentence this whole module exists to stop being stored as a fact."""
        opportunity = Opportunity(
            "Roblox oyun fikri", "aylık gelir beklentisi",
            estimates={"revenue": Estimate(5000, "USD/ay", GUESS, 0.9,
                                           assumptions=["10k oyuncu olur"])})
        self.assertEqual(recordable_estimates(opportunity), {})

    def test_measured_estimate_is_recordable(self):
        opportunity = Opportunity(
            "Test süresi", "", estimates={"duration": Estimate(
                62, "ms", MEASURED, 0.95, evidence=["deney abc123"])})
        self.assertIn("duration", recordable_estimates(opportunity))


# --------------------------------------------------------------------- screening
class TestScreening(unittest.TestCase):
    def test_ordinary_opportunity_passes(self):
        self.assertTrue(screen(Opportunity("Oyuncu tutundurmayı artır",
                                           "yeni oyunculara başlangıç bonusu")).allowed)

    def test_forbidden_category_is_refused(self):
        self.assertFalse(screen(Opportunity("bir şey", "", category="dolandiricilik")).allowed)

    def test_fraud_wording_is_refused(self):
        self.assertFalse(screen(Opportunity(
            "Sahte yorum ile sıralama yükselt", "fake review kampanyası")).allowed)

    def test_spam_is_refused(self):
        self.assertFalse(screen(Opportunity(
            "Toplu mesaj kampanyası", "mass dm ile oyuncu çek")).allowed)

    def test_rule_breaking_is_refused(self):
        self.assertFalse(screen(Opportunity(
            "Ban atlatma yöntemi", "platform kural ihlali ile hesap aç")).allowed)

    def test_privacy_violation_is_refused(self):
        self.assertFalse(screen(Opportunity(
            "Kullanıcı verisini sat", "kişisel veri topla ve sat")).allowed)

    def test_security_violation_is_refused(self):
        self.assertFalse(screen(Opportunity("Şifre kır", "password crack aracı")).allowed)

    def test_guarantee_wording_is_refused(self):
        """'May earn' and 'will earn' are different claims and must stay different."""
        screening = screen(Opportunity(
            "Garanti gelir", "bu yöntem kesinlikle kazandırır"))
        self.assertFalse(screening.allowed)
        self.assertTrue(any("garanti" in r for r in screening.reasons))

    def test_an_inflected_promise_is_refused(self):
        """This one passed in a live run: 'kazan' + 'dırır' defeated a word boundary."""
        self.assertFalse(screen(Opportunity(
            "Roblox oyunu aylık 5000 dolar kazandırır", "gelir modeli")).allowed)

    def test_a_hedged_revenue_claim_passes(self):
        """The distinction is the point: possibility is allowed, promise is not."""
        self.assertTrue(screen(Opportunity(
            "Roblox oyunu aylık gelir getirebilir",
            "benzer oyunların verilerine göre bir olasılık")).allowed)

    def test_a_refused_category_cannot_be_outweighed(self):
        """A filter, not a penalty: perfect numbers do not rescue it."""
        opportunity = Opportunity("Sahte hesap satışı", "fake account üret ve sat",
                                  dimensions=dims(revenue=1.0, feasibility=1.0,
                                                  risk=0.0, competition=0.0))
        verdict = score(opportunity)
        self.assertFalse(verdict.worth_pursuing)
        self.assertEqual(verdict.composite, 0.0)


# ----------------------------------------------------------------------- scoring
class TestScoring(unittest.TestCase):
    def test_good_opportunity_ranks(self):
        verdict = score(Opportunity("iyi fikir", dimensions=dims(
            revenue=0.8, feasibility=0.9, time=0.2, resource_fit=0.9,
            competition=0.2, risk=0.2, confidence=0.8)))
        self.assertTrue(verdict.ranked)

    def test_inverted_dimensions_are_inverted(self):
        cheap = score(Opportunity("ucuz", dimensions=dims(time=0.1, risk=0.1)))
        costly = score(Opportunity("pahalı", dimensions=dims(time=0.9, risk=0.9)))
        self.assertGreater(cheap.composite, costly.composite)

    def test_weak_opportunity_does_not_rank(self):
        verdict = score(Opportunity("zayıf", dimensions=dims(
            revenue=0.1, feasibility=0.1, time=0.9, resource_fit=0.1,
            competition=0.9, risk=0.9, confidence=0.1)))
        self.assertFalse(verdict.ranked)

    def test_unmeasured_dimensions_are_warned_about(self):
        opportunity = Opportunity("belirsiz", dimensions={
            name: Dimension(name, 0.8, GUESS, "tahmin") for name in
            ("revenue", "feasibility", "time", "resource_fit",
             "competition", "risk", "confidence")})
        self.assertTrue(any("dayanaksız" in w for w in score(opportunity).warnings))

    def test_speculative_estimates_are_warned_about(self):
        opportunity = Opportunity("fikir", dimensions=dims(),
                                  estimates={"revenue": Estimate(5000, "USD", GUESS, 0.9)})
        self.assertTrue(any("spekülatif" in w for w in score(opportunity).warnings))

    def test_ranking_puts_refused_opportunities_last(self):
        verdicts = rank([
            Opportunity("dolandırıcılık", "fake review", dimensions=dims(revenue=1.0)),
            Opportunity("dürüst", "oyuncu deneyimini iyileştir", dimensions=dims()),
        ])
        self.assertTrue(verdicts[0].screening.allowed)
        self.assertFalse(verdicts[-1].screening.allowed)


# ------------------------------------------------------------------- preferences
class TestPreferences(ImproveCase):
    def setUp(self):
        super().setUp()
        self.store = PreferenceStore(self.db)

    def test_owner_preference_changes_ranking(self):
        """'Önce oyuncu sayısı, gelir ikinci' must actually reorder the list."""
        growth = Opportunity("büyüme", dimensions=dims(revenue=0.2, resource_fit=0.9))
        money = Opportunity("gelir", dimensions=dims(revenue=0.9, resource_fit=0.2))

        before = score(money).composite - score(growth).composite
        self.store.state("objective.growth_first",
                         "Gelirden önce oyuncu sayısını önemse",
                         {"revenue": 0.3, "resource_fit": 2.0})
        weights = self.store.weights()
        after = (score(money, weights=weights).composite
                 - score(growth, weights=weights).composite)
        self.assertGreater(before, after)

    def test_a_preference_is_owner_sourced_and_dated(self):
        preference = self.store.state("x", "bir tercih", {"revenue": 0.5})
        self.assertEqual(preference.source, "kullanici")
        self.assertGreater(preference.stated_at, 0)

    def test_unknown_dimension_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.state("x", "tercih", {"uydurma_boyut": 2.0})

    def test_multiplier_is_clamped(self):
        preference = self.store.state("x", "tercih", {"revenue": 999.0})
        self.assertLessEqual(preference.weights["revenue"], 3.0)

    def test_no_single_preference_silences_the_others(self):
        self.store.state("x", "tercih", {"revenue": 0.0001})
        self.assertGreater(self.store.weights()["revenue"], 0.0)

    def test_retracted_preference_stops_counting(self):
        self.store.state("x", "tercih", {"revenue": 0.25})
        lowered = self.store.weights()["revenue"]
        self.store.retract("x")
        self.assertGreater(self.store.weights()["revenue"], lowered)


# -------------------------------------------------------------------- hypotheses
class TestHypotheses(ImproveCase):
    def setUp(self):
        super().setUp()
        self.store = HypothesisStore(self.db)

    def test_the_same_idea_reworded_collides(self):
        """Deduplication must survive a model that never repeats its phrasing."""
        a = fingerprint("Yol çözümlemesini önbelleğe alarak sandbox'ı hızlandır")
        b = fingerprint("Sandbox'ı hızlandırmak için yol çözümlemesini önbelleğe al")
        self.assertEqual(a, b)

    def test_different_ideas_do_not_collide(self):
        self.assertNotEqual(fingerprint("ses sistemini kur"),
                            fingerprint("tarayıcı otomasyonu ekle"))

    def test_inflection_does_not_change_the_fingerprint(self):
        """Turkish inflects endlessly; the same word must stem to the same thing."""
        self.assertEqual(fingerprint("önbellekleme performansı iyileştirme"),
                         fingerprint("önbelleklemeyi performansına iyileştirmek"))

    def test_unrelated_long_words_still_separate(self):
        self.assertNotEqual(
            fingerprint("tarayıcı otomasyonu playwright ile kurulsun"),
            fingerprint("konuşma tanıma whisper modeliyle eklensin"))

    def test_duplicate_proposal_returns_the_original(self):
        first, is_new = self.store.propose("Önbellek ekle", statement="hızlandırır")
        second, is_new_again = self.store.propose("Ekle önbellek", statement="hızlandırır")
        self.assertTrue(is_new)
        self.assertFalse(is_new_again)
        self.assertEqual(first.id, second.id)

    def test_a_new_hypothesis_is_runnable(self):
        hypothesis, _ = self.store.propose("Bir fikir", statement="ayrıntı")
        self.assertTrue(hypothesis.runnable)
        self.assertEqual(hypothesis.state, PROPOSED)

    def test_failure_sets_a_cooldown(self):
        hypothesis, _ = self.store.propose("Fikir", statement="ayrıntı")
        self.store.start_attempt(hypothesis.id, "exp1")
        refuted = self.store.refute(hypothesis.id, Lesson(
            why="testler geçmedi", retry_worth=True, needed_change="ısınma turu ekle"))
        self.assertTrue(refuted.cooling)
        self.assertFalse(refuted.runnable)

    def test_a_lesson_with_no_change_shelves_the_idea(self):
        """'Try again' without naming a change is how a system repeats forever."""
        hypothesis, _ = self.store.propose("Fikir", statement="ayrıntı")
        self.store.start_attempt(hypothesis.id, "exp1")
        refuted = self.store.refute(hypothesis.id, Lesson(why="olmadı", retry_worth=False))
        self.assertEqual(refuted.state, SHELVED)

    def test_a_retryable_failure_stays_open(self):
        hypothesis, _ = self.store.propose("Fikir", statement="ayrıntı")
        self.store.start_attempt(hypothesis.id, "exp1")
        refuted = self.store.refute(hypothesis.id, Lesson(
            why="olmadı", retry_worth=True, needed_change="farklı ölçüm kullan"))
        self.assertEqual(refuted.state, REFUTED)

    def test_attempts_run_out(self):
        hypothesis, _ = self.store.propose("Fikir", statement="ayrıntı")
        for index in range(MAX_ATTEMPTS):
            self.store.start_attempt(hypothesis.id, f"exp{index}")
            self.store.refute(hypothesis.id, Lesson(
                why="olmadı", retry_worth=True, needed_change="başka bir şey dene"))
        self.assertEqual(self.store.get(hypothesis.id).state, SHELVED)

    def test_cooling_hypotheses_are_not_offered(self):
        hypothesis, _ = self.store.propose("Fikir", statement="ayrıntı")
        self.store.start_attempt(hypothesis.id, "exp1")
        self.store.refute(hypothesis.id, Lesson(why="olmadı", retry_worth=True,
                                                needed_change="değişiklik"))
        self.assertEqual(self.store.runnable(), [])

    def test_lesson_survives_a_round_trip(self):
        hypothesis, _ = self.store.propose("Fikir", statement="ayrıntı")
        self.store.start_attempt(hypothesis.id, "exp1")
        self.store.refute(hypothesis.id, Lesson(
            why="ölçüm gürültülü", wrong_assumption="tek koşu yeterli sanıldı",
            retry_worth=True, needed_change="beş koşunun ortancasını al",
            experiment_id="exp1"))
        stored = self.store.get(hypothesis.id)
        self.assertEqual(stored.lesson.wrong_assumption, "tek koşu yeterli sanıldı")
        self.assertEqual(stored.lesson.experiment_id, "exp1")

    def test_confirmation_clears_the_cooldown(self):
        hypothesis, _ = self.store.propose("Fikir", statement="ayrıntı")
        self.store.start_attempt(hypothesis.id, "exp1")
        self.assertEqual(self.store.confirm(hypothesis.id).state, CONFIRMED)

    def test_seen_finds_a_prior_attempt(self):
        self.store.propose("Ses sistemini kur", statement="piper ile")
        self.assertIsNotNone(self.store.seen("Kur ses sistemini", "piper ile"))

    def test_an_open_hypothesis_blocks_a_second_one_for_the_same_gap(self):
        """Wording dedupe alone failed live: the model reworded and slipped past."""
        self.store.propose("Bir yaklaşım", statement="ayrıntı", gap_key="yetenek:kod")
        self.assertIsNotNone(self.store.open_for_gap("yetenek:kod"))

    def test_a_cooling_hypothesis_still_blocks_its_gap(self):
        hypothesis, _ = self.store.propose("Yaklaşım", statement="x", gap_key="yetenek:kod")
        self.store.start_attempt(hypothesis.id, "exp1")
        self.store.refute(hypothesis.id, Lesson(why="olmadı", retry_worth=True,
                                                needed_change="başka yol"))
        self.assertIsNotNone(self.store.open_for_gap("yetenek:kod"))

    def test_a_confirmed_hypothesis_frees_its_gap(self):
        """Once an idea has landed, the gap may produce the next one."""
        hypothesis, _ = self.store.propose("Yaklaşım", statement="x", gap_key="yetenek:kod")
        self.store.start_attempt(hypothesis.id, "exp1")
        self.store.confirm(hypothesis.id)
        self.assertIsNone(self.store.open_for_gap("yetenek:kod"))

    def test_a_different_gap_is_unaffected(self):
        self.store.propose("Yaklaşım", statement="x", gap_key="yetenek:kod")
        self.assertIsNone(self.store.open_for_gap("yetenek:ses"))


# ----------------------------------------------------------------------- budget
class TestBudget(ImproveCase):
    def setUp(self):
        super().setUp()
        self.budget = ImprovementBudget(
            self.db, daily={RESEARCH: 2, HYPOTHESIS: 2, EXPERIMENT: 1},
            nightly={RESEARCH: 1, HYPOTHESIS: 1, EXPERIMENT: 1},
            night_hours=(1, 8))

    def test_spending_within_budget_is_allowed(self):
        self.assertTrue(self.budget.spend(RESEARCH, at=datetime(2026, 8, 11, 14)).allowed)

    def test_the_daily_ceiling_holds(self):
        day = datetime(2026, 8, 11, 14)
        self.budget.spend(RESEARCH, at=day)
        self.budget.spend(RESEARCH, at=day)
        self.assertFalse(self.budget.spend(RESEARCH, at=day).allowed)

    def test_the_night_ceiling_is_tighter(self):
        night = datetime(2026, 8, 11, 3)
        self.assertTrue(self.budget.spend(RESEARCH, at=night).allowed)
        self.assertFalse(self.budget.spend(RESEARCH, at=night).allowed)

    def test_a_refusal_is_not_recorded(self):
        day = datetime(2026, 8, 11, 14)
        self.budget.spend(EXPERIMENT, at=day)
        self.budget.spend(EXPERIMENT, at=day)
        self.budget.spend(EXPERIMENT, at=day)
        self.assertEqual(self.budget.used(EXPERIMENT), 1)

    def test_counts_survive_a_restart(self):
        """Otherwise a crash loop is a way around every limit here."""
        day = datetime(2026, 8, 11, 14)
        self.budget.spend(RESEARCH, at=day)
        reopened = ImprovementBudget(self.db, daily={RESEARCH: 2})
        self.assertEqual(reopened.used(RESEARCH), 1)

    def test_activities_have_separate_budgets(self):
        day = datetime(2026, 8, 11, 14)
        self.budget.spend(EXPERIMENT, at=day)
        self.assertTrue(self.budget.check(RESEARCH, at=day).allowed)

    def test_unknown_activity_is_refused(self):
        self.assertFalse(self.budget.check("uydurma").allowed)


class TestProvenanceBoundary(unittest.TestCase):
    def test_engine_conclusions_are_unverified(self):
        from jarvis.improve.engine import ENGINE_SOURCED

        self.assertIn(ENGINE_SOURCED, UNVERIFIED_SOURCES)


if __name__ == "__main__":
    unittest.main()
