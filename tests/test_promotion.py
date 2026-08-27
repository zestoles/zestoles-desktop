"""Experiment registry, benchmark gate, promotion gate, snapshot and rollback.

The questions this file has to answer, because they are the ones that decide
whether the machinery is worth having:

  does a failing benchmark stop a promotion
  does a regression stop one, including the cheap tricks — deleted tests, skipped
    tests, a suite that got slower
  does rollback actually restore the previous version, byte for byte
  is the system left broken when a promotion dies halfway through
  can a failed experiment change production
  can anything reach outside the sandbox

Every state transition is pinned too. PROMOTED is the one value in the system that
must be unreachable except through the gate, so every path that is not the gate is
tested to fail.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.lab.benchmark import BenchmarkResult, compare, parse_unittest  # noqa: E402
from jarvis.lab.promotion import (  # noqa: E402
    JOURNAL_SUFFIX,
    Promoter,
    PromotionRefused,
    Snapshot,
    evaluate,
    recover_interrupted,
    write_journal,
)
from jarvis.lab.registry import (  # noqa: E402
    CANDIDATE,
    DISCARDED,
    EXPERIMENT,
    FAILED,
    PASSED,
    PROMOTED,
    ExperimentRegistry,
    TransitionRefused,
)
from jarvis.lab.sandbox import CommandResult, Sandbox, SandboxLimits  # noqa: E402


def bench(tests=10, failures=0, errors=0, skipped=0, ms=1000, ran=True):
    return BenchmarkResult(ran=ran, tests=tests, failures=failures, errors=errors,
                           skipped=skipped, duration_ms=ms)


class LabCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.registry = ExperimentRegistry(self.base / "lab.db")
        self.target = self.base / "uretim"
        self.target.mkdir()
        self.project = self.base / "proje"
        (self.project / "jarvis").mkdir(parents=True)
        self.promoter = Promoter(
            self.registry, snapshots_dir=self.base / "snapshots",
            journal_dir=self.base / "journal", project_root=self.project)
        self.box = Sandbox(self.base / "kutu", limits=SandboxLimits(timeout_s=60))

    def tearDown(self):
        self._tmp.cleanup()

    def passed_experiment(self, purpose="deney"):
        experiment = self.registry.open(purpose)
        return self.registry.transition(experiment.id, PASSED, reason="ölçüldü")


class TestStateMachine(LabCase):
    def test_new_experiment_starts_in_experiment(self):
        self.assertEqual(self.registry.open("x").state, EXPERIMENT)

    def test_the_happy_path_is_allowed(self):
        experiment = self.registry.open("x")
        for state in (PASSED, CANDIDATE, PROMOTED):
            experiment = self.registry.transition(experiment.id, state)
        self.assertEqual(experiment.state, PROMOTED)
        self.assertIsNotNone(experiment.promoted)

    def test_experiment_cannot_jump_straight_to_promoted(self):
        """The one transition the whole gate exists to make unreachable."""
        experiment = self.registry.open("x")
        with self.assertRaises(TransitionRefused):
            self.registry.transition(experiment.id, PROMOTED)

    def test_passed_cannot_jump_to_promoted(self):
        experiment = self.passed_experiment()
        with self.assertRaises(TransitionRefused):
            self.registry.transition(experiment.id, PROMOTED)

    def test_failed_cannot_become_promoted(self):
        experiment = self.registry.open("x")
        self.registry.transition(experiment.id, FAILED)
        for state in (PASSED, CANDIDATE, PROMOTED):
            with self.subTest(state=state):
                with self.assertRaises(TransitionRefused):
                    self.registry.transition(experiment.id, state)

    def test_promoted_is_terminal(self):
        experiment = self.registry.open("x")
        for state in (PASSED, CANDIDATE, PROMOTED):
            self.registry.transition(experiment.id, state)
        with self.assertRaises(TransitionRefused):
            self.registry.transition(experiment.id, FAILED)

    def test_failed_may_be_discarded(self):
        experiment = self.registry.open("x")
        self.registry.transition(experiment.id, FAILED)
        self.assertEqual(self.registry.transition(experiment.id, DISCARDED).state, DISCARDED)

    def test_unknown_state_is_refused(self):
        experiment = self.registry.open("x")
        with self.assertRaises(TransitionRefused):
            self.registry.transition(experiment.id, "harika")

    def test_history_records_every_move(self):
        experiment = self.registry.open("x")
        self.registry.transition(experiment.id, PASSED, reason="ölçüldü")
        self.registry.transition(experiment.id, CANDIDATE, reason="kapı geçildi")
        history = self.registry.history(experiment.id)
        self.assertEqual([h["to_state"] for h in history], [EXPERIMENT, PASSED, CANDIDATE])


class TestRegistryRecords(LabCase):
    def test_purpose_and_provenance_are_kept(self):
        experiment = self.registry.open("hız denemesi", base_commit="abc123",
                                        model="qwen3.5:9b", sandbox_path="/kutu")
        stored = self.registry.get(experiment.id)
        self.assertEqual(stored.purpose, "hız denemesi")
        self.assertEqual(stored.base_commit, "abc123")
        self.assertEqual(stored.model, "qwen3.5:9b")

    def test_changed_files_round_trip(self):
        experiment = self.registry.open("x")
        self.registry.record_files(experiment.id, ["b.py", "a.py", "a.py"])
        self.assertEqual(self.registry.get(experiment.id).changed_files, ["a.py", "b.py"])

    def test_measurements_round_trip(self):
        experiment = self.registry.open("x")
        self.registry.record_measurements(experiment.id, baseline=bench().as_dict())
        self.assertEqual(self.registry.get(experiment.id).baseline["tests"], 10)

    def test_ids_are_unique(self):
        self.assertNotEqual(self.registry.open("a").id, self.registry.open("b").id)


class TestBenchmarkParsing(unittest.TestCase):
    def test_passing_run_is_read(self):
        result = parse_unittest(CommandResult(
            ["python"], 0, "", "Ran 219 tests in 7.8s\n\nOK (skipped=4)\n", 7800))
        self.assertTrue(result.ran)
        self.assertEqual(result.tests, 219)
        self.assertEqual(result.skipped, 4)
        self.assertEqual(result.effective, 215)
        self.assertTrue(result.passed)

    def test_failing_run_is_read(self):
        result = parse_unittest(CommandResult(
            ["python"], 1, "", "Ran 10 tests in 1.0s\n\nFAILED (failures=2, errors=1)\n", 1000))
        self.assertEqual(result.failures, 2)
        self.assertEqual(result.errors, 1)
        self.assertFalse(result.passed)

    def test_unparseable_output_is_not_a_run(self):
        result = parse_unittest(CommandResult(["python"], 1, "", "boom", 10))
        self.assertFalse(result.ran)
        self.assertFalse(result.passed)

    def test_timeout_is_not_passing(self):
        result = parse_unittest(CommandResult(
            ["python"], -1, "", "Ran 5 tests in 1.0s\nOK", 60000, timed_out=True))
        self.assertFalse(result.passed)


class TestRegressionDetection(unittest.TestCase):
    def test_identical_runs_are_acceptable(self):
        self.assertTrue(compare(bench(), bench()).acceptable)

    def test_new_failure_is_a_regression(self):
        result = compare(bench(), bench(failures=1))
        self.assertFalse(result.acceptable)
        self.assertTrue(any("başarısız" in r for r in result.regressions))

    def test_new_error_is_a_regression(self):
        self.assertFalse(compare(bench(), bench(errors=1)).acceptable)

    def test_deleting_tests_is_a_regression(self):
        """The cheapest way to make a suite green, and it must not work."""
        result = compare(bench(tests=10), bench(tests=6))
        self.assertFalse(result.acceptable)
        self.assertTrue(any("etkin test" in r for r in result.regressions))

    def test_skipping_tests_is_a_regression(self):
        """The second cheapest: same count, but four of them stopped asserting."""
        result = compare(bench(tests=10, skipped=0), bench(tests=10, skipped=4))
        self.assertFalse(result.acceptable)
        self.assertTrue(any("etkin test" in r for r in result.regressions))

    def test_material_slowdown_is_a_regression(self):
        result = compare(bench(ms=1000), bench(ms=3000))
        self.assertFalse(result.acceptable)
        self.assertTrue(any("yavaşlama" in r for r in result.regressions))

    def test_small_timing_noise_is_tolerated(self):
        self.assertTrue(compare(bench(ms=1000), bench(ms=1200)).acceptable)

    def test_speedup_is_an_improvement(self):
        result = compare(bench(ms=2000), bench(ms=1000))
        self.assertTrue(result.acceptable)
        self.assertTrue(any("hızlandı" in i for i in result.improvements))

    def test_adding_tests_is_an_improvement(self):
        result = compare(bench(tests=10), bench(tests=14))
        self.assertTrue(result.acceptable)
        self.assertTrue(any("arttı" in i for i in result.improvements))

    def test_fixing_a_failure_is_acceptable(self):
        self.assertTrue(compare(bench(failures=2), bench(failures=0)).acceptable)

    def test_candidate_must_be_green_even_with_no_regression(self):
        """Baseline red and candidate red is not a regression, but it is not done."""
        self.assertFalse(compare(bench(failures=2), bench(failures=2)).acceptable)

    def test_missing_candidate_measurement_blocks(self):
        self.assertFalse(compare(bench(), bench(ran=False)).acceptable)

    def test_missing_baseline_still_requires_a_green_candidate(self):
        self.assertTrue(compare(bench(ran=False), bench()).acceptable)
        self.assertFalse(compare(bench(ran=False), bench(failures=1)).acceptable)


class TestPromotionGate(LabCase):
    def test_a_clean_experiment_is_promotable(self):
        decision = evaluate(
            self.passed_experiment(), compare(bench(), bench()),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertTrue(decision.promotable, decision.blocking)

    def test_an_unsettled_experiment_is_refused(self):
        decision = evaluate(
            self.registry.open("x"), compare(bench(), bench()),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("durum" in b for b in decision.blocking))

    def test_a_failing_benchmark_is_refused(self):
        decision = evaluate(
            self.passed_experiment(), compare(bench(), bench(failures=1)),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)

    def test_a_regression_is_refused(self):
        decision = evaluate(
            self.passed_experiment(), compare(bench(tests=10), bench(tests=5)),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("regresyon" in b or "kapsam" in b for b in decision.blocking))

    def test_no_measurement_is_refused(self):
        decision = evaluate(
            self.passed_experiment(), None, target_root=self.target,
            changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)

    def test_no_changed_files_is_refused(self):
        decision = evaluate(
            self.passed_experiment(), compare(bench(), bench()),
            target_root=self.target, changed_files=[], project_root=self.project)
        self.assertFalse(decision.promotable)

    def test_targeting_the_source_tree_is_refused(self):
        """Self-modification is off; the gate is where that is enforced."""
        decision = evaluate(
            self.passed_experiment(), compare(bench(), bench()),
            target_root=self.project / "jarvis", changed_files=["brain/local.py"],
            project_root=self.project)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("korumalı" in b for b in decision.blocking))

    def test_escaping_the_target_via_traversal_is_refused(self):
        decision = evaluate(
            self.passed_experiment(), compare(bench(), bench()),
            target_root=self.target, changed_files=["../../jarvis/cli.py"],
            project_root=self.project)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("hedef içinde" in b for b in decision.blocking))

    def test_every_check_is_reported(self):
        decision = evaluate(
            self.passed_experiment(), compare(bench(), bench()),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertGreaterEqual(len(decision.checks), 6)

    # -- the baseline has to be a real basis for comparison --------------------
    #
    # PLANNER_SYSTEM tells the model: "The tests must pass against BOTH
    # baseline_code and candidate_code ... A test that only passes on the
    # candidate is a broken experiment, not a successful one." Nothing enforced
    # it. Measured on the 17.08 autonomous run: cb36e50d promoted on a baseline
    # with errors=1, exit_code=1, passed=False.

    def test_a_broken_baseline_is_refused(self):
        """A baseline failing its own test makes "not worse" vacuous: any
        candidate clears it, so the comparison carries no information."""
        decision = evaluate(
            self.passed_experiment(), compare(bench(errors=1), bench()),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("baseline" in b for b in decision.blocking), decision.blocking)

    def test_a_failing_baseline_is_refused_even_when_the_candidate_looks_better(self):
        """"Fixed the failure" is the shape a broken experiment takes."""
        decision = evaluate(
            self.passed_experiment(), compare(bench(failures=2), bench(failures=0)),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("baseline" in b for b in decision.blocking), decision.blocking)

    def test_a_missing_baseline_is_refused(self):
        """No basis for comparison is not the same as a comparison that passed.
        Without this, an unmeasurable baseline also cleared "kapsam korundu",
        because an absent run reports zero effective tests."""
        decision = evaluate(
            self.passed_experiment(), compare(bench(ran=False), bench()),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("baseline" in b for b in decision.blocking), decision.blocking)

    def test_a_timed_out_baseline_is_refused(self):
        timed_out = BenchmarkResult(ran=True, tests=10, duration_ms=99, timed_out=True)
        decision = evaluate(
            self.passed_experiment(), compare(timed_out, bench()),
            target_root=self.target, changed_files=["a.py"], project_root=self.project)
        self.assertFalse(decision.promotable)

    def test_the_comparison_itself_still_calls_a_fix_acceptable(self):
        """The gate got stricter; Comparison did not, on purpose. compare()
        answers "is the candidate worse than the baseline", which is not the
        same question as "may this be installed"."""
        self.assertTrue(compare(bench(failures=2), bench(failures=0)).acceptable)
        self.assertTrue(compare(bench(ran=False), bench()).acceptable)


class TestSourceTreeIsNeverModified(LabCase):
    """Refusal at the gate is a claim; the bytes on disk are the evidence.

    Every other test here checks what `evaluate()` decides. These check what is
    actually on the filesystem afterwards, because a gate that says no and a
    promoter that writes anyway would pass all of them.
    """

    def source_file(self, relative: str = "brain/local.py") -> Path:
        path = self.project / "jarvis" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ORIJINAL\n", encoding="utf-8")
        return path

    def test_promoting_into_the_source_tree_leaves_it_byte_identical(self):
        original = self.source_file()
        self.box.write("brain/local.py", "DEGISTIRILDI\n")
        result = self.promoter.promote(
            self.passed_experiment(), compare(bench(), bench()),
            sandbox=self.box, target_root=self.project / "jarvis",
            files=["brain/local.py"])
        self.assertFalse(result.ok)
        self.assertEqual(result.applied, [])
        self.assertEqual(original.read_text(encoding="utf-8"), "ORIJINAL\n")

    def test_escaping_the_target_by_traversal_writes_nothing(self):
        original = self.source_file()
        self.box.write("kacak.py", "DEGISTIRILDI\n")
        result = self.promoter.promote(
            self.passed_experiment(), compare(bench(), bench()),
            sandbox=self.box, target_root=self.target,
            files=["../proje/jarvis/brain/local.py"])
        self.assertFalse(result.ok)
        self.assertEqual(original.read_text(encoding="utf-8"), "ORIJINAL\n")

    def test_a_refused_promotion_creates_no_journal_and_no_snapshot(self):
        """Nothing half-done: a refusal must not leave recovery state behind
        that a later startup would try to roll back."""
        self.source_file()
        self.box.write("brain/local.py", "DEGISTIRILDI\n")
        self.promoter.promote(
            self.passed_experiment(), compare(bench(), bench()),
            sandbox=self.box, target_root=self.project / "jarvis",
            files=["brain/local.py"])
        self.assertEqual(list(self.promoter.journal_dir.glob(f"*{JOURNAL_SUFFIX}")), [])
        self.assertEqual(self.promoter.list_snapshots(), [])

    def test_a_broken_baseline_does_not_reach_the_filesystem(self):
        """The 17.08 gap, checked at the level that matters."""
        original = self.source_file()
        self.box.write("brain/local.py", "DEGISTIRILDI\n")
        result = self.promoter.promote(
            self.passed_experiment(), compare(bench(errors=1), bench()),
            sandbox=self.box, target_root=self.project / "jarvis",
            files=["brain/local.py"])
        self.assertFalse(result.ok)
        self.assertEqual(original.read_text(encoding="utf-8"), "ORIJINAL\n")

    def test_the_flag_is_off_and_the_gate_is_the_only_thing_that_reads_it(self):
        from jarvis.lab import promotion as promotion_module

        self.assertFalse(promotion_module.ALLOW_SELF_MODIFICATION)

    def test_turning_the_flag_on_is_what_it_would_take(self):
        """Pins that the refusal above comes from the flag and not from an
        accident of path handling — otherwise this suite could keep passing
        after the protection stopped working."""
        decision = evaluate(
            self.passed_experiment(), compare(bench(), bench()),
            target_root=self.project / "jarvis", changed_files=["brain/local.py"],
            project_root=self.project, allow_self_modification=True)
        self.assertTrue(decision.promotable, decision.blocking)


class TestSnapshotAndRollback(LabCase):
    def test_snapshot_restores_previous_content(self):
        existing = self.target / "a.txt"
        existing.write_text("ESKİ", encoding="utf-8")
        snapshot = Snapshot.create(self.target, ["a.txt"], self.base / "snapshots")

        existing.write_text("YENİ", encoding="utf-8")
        snapshot.restore()
        self.assertEqual(existing.read_text(encoding="utf-8"), "ESKİ")

    def test_snapshot_deletes_files_that_did_not_exist(self):
        snapshot = Snapshot.create(self.target, ["yeni.txt"], self.base / "snapshots")
        (self.target / "yeni.txt").write_text("sonradan", encoding="utf-8")
        snapshot.restore()
        self.assertFalse((self.target / "yeni.txt").exists())

    def test_snapshot_verifies_its_own_integrity(self):
        (self.target / "a.txt").write_text("içerik", encoding="utf-8")
        snapshot = Snapshot.create(self.target, ["a.txt"], self.base / "snapshots")
        self.assertTrue(snapshot.verify())

        (Path(snapshot.store) / "files" / "a.txt").write_text("bozuldu", encoding="utf-8")
        self.assertFalse(snapshot.verify())

    def test_corrupted_snapshot_refuses_to_restore(self):
        (self.target / "a.txt").write_text("içerik", encoding="utf-8")
        snapshot = Snapshot.create(self.target, ["a.txt"], self.base / "snapshots")
        (Path(snapshot.store) / "files" / "a.txt").write_text("bozuldu", encoding="utf-8")
        with self.assertRaises(PromotionRefused):
            snapshot.restore()

    def test_snapshot_survives_reload_from_disk(self):
        (self.target / "a.txt").write_text("ESKİ", encoding="utf-8")
        snapshot = Snapshot.create(self.target, ["a.txt"], self.base / "snapshots")
        (self.target / "a.txt").write_text("YENİ", encoding="utf-8")

        Snapshot.load(Path(snapshot.store)).restore()
        self.assertEqual((self.target / "a.txt").read_text(encoding="utf-8"), "ESKİ")


class TestInterruptedPromotion(LabCase):
    def test_a_leftover_journal_is_rolled_back_at_startup(self):
        """A promotion that died halfway must not be carried forward."""
        original = self.target / "a.txt"
        original.write_text("ÇALIŞAN SÜRÜM", encoding="utf-8")
        snapshot = Snapshot.create(self.target, ["a.txt"], self.base / "snapshots")
        write_journal(self.base / "journal", "yarim-deney", snapshot, ["a.txt"])

        # The process dies here, having written the new version.
        original.write_text("YARIM UYGULANMIŞ", encoding="utf-8")

        recovered = recover_interrupted(self.base / "journal")
        self.assertEqual(len(recovered), 1)
        self.assertTrue(recovered[0]["ok"])
        self.assertEqual(original.read_text(encoding="utf-8"), "ÇALIŞAN SÜRÜM")

    def test_the_journal_is_cleared_after_recovery(self):
        (self.target / "a.txt").write_text("x", encoding="utf-8")
        snapshot = Snapshot.create(self.target, ["a.txt"], self.base / "snapshots")
        write_journal(self.base / "journal", "yarim", snapshot, ["a.txt"])

        recover_interrupted(self.base / "journal")
        self.assertEqual(list((self.base / "journal").glob(f"*{JOURNAL_SUFFIX}")), [])

    def test_nothing_to_recover_is_not_an_error(self):
        self.assertEqual(recover_interrupted(self.base / "journal"), [])

    def test_a_broken_journal_is_reported_not_swallowed(self):
        directory = self.base / "journal"
        directory.mkdir(exist_ok=True)
        (directory / f"bozuk{JOURNAL_SUFFIX}").write_text("bu JSON değil", encoding="utf-8")
        recovered = recover_interrupted(directory)
        self.assertEqual(len(recovered), 1)
        self.assertFalse(recovered[0]["ok"])

    def test_no_journal_remains_after_a_successful_promotion(self):
        experiment = self.passed_experiment()
        self.box.write("a.txt", "yeni sürüm")
        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(), bench()),
            sandbox=self.box, target_root=self.target, files=["a.txt"])
        self.assertTrue(result.ok, result.error)
        self.assertEqual(list((self.base / "journal").glob(f"*{JOURNAL_SUFFIX}")), [])


class TestPromotionFlow(LabCase):
    def test_a_clean_promotion_applies_and_records(self):
        experiment = self.passed_experiment("iyileştirme")
        self.box.write("a.txt", "YENİ SÜRÜM")

        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(), bench()),
            sandbox=self.box, target_root=self.target, files=["a.txt"])

        self.assertTrue(result.ok, result.error)
        self.assertEqual((self.target / "a.txt").read_text(encoding="utf-8"), "YENİ SÜRÜM")
        self.assertEqual(self.registry.get(experiment.id).state, PROMOTED)

    def test_a_failed_experiment_cannot_change_production(self):
        """The headline guarantee: nothing that did not pass may touch the target."""
        experiment = self.registry.open("bozuk")
        self.registry.transition(experiment.id, FAILED, reason="testler geçmedi")
        (self.target / "a.txt").write_text("DOKUNULMAMIŞ", encoding="utf-8")
        self.box.write("a.txt", "ELE GEÇİRİLDİ")

        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(), bench(failures=3)),
            sandbox=self.box, target_root=self.target, files=["a.txt"])

        self.assertFalse(result.ok)
        self.assertEqual((self.target / "a.txt").read_text(encoding="utf-8"), "DOKUNULMAMIŞ")
        # Refusing must not raise on the way out: the experiment was already
        # FAILED, and FAILED → FAILED is not a legal move.
        self.assertEqual(self.registry.get(experiment.id).state, FAILED)

    def test_a_regressing_experiment_cannot_change_production(self):
        experiment = self.passed_experiment()
        (self.target / "a.txt").write_text("DOKUNULMAMIŞ", encoding="utf-8")
        self.box.write("a.txt", "ELE GEÇİRİLDİ")

        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(tests=20), bench(tests=8)),
            sandbox=self.box, target_root=self.target, files=["a.txt"])

        self.assertFalse(result.ok)
        self.assertEqual((self.target / "a.txt").read_text(encoding="utf-8"), "DOKUNULMAMIŞ")
        self.assertEqual(self.registry.get(experiment.id).state, FAILED)

    def test_promotion_into_the_source_tree_is_refused(self):
        experiment = self.passed_experiment()
        self.box.write("cli.py", "# ele geçirildi")
        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(), bench()),
            sandbox=self.box, target_root=self.project / "jarvis", files=["cli.py"])
        self.assertFalse(result.ok)
        self.assertFalse((self.project / "jarvis" / "cli.py").exists())

    def test_a_missing_sandbox_file_fails_before_touching_the_target(self):
        experiment = self.passed_experiment()
        (self.target / "a.txt").write_text("DOKUNULMAMIŞ", encoding="utf-8")
        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(), bench()),
            sandbox=self.box, target_root=self.target, files=["yok.txt"])
        self.assertFalse(result.ok)
        self.assertEqual((self.target / "a.txt").read_text(encoding="utf-8"), "DOKUNULMAMIŞ")

    def test_rollback_restores_the_previous_version(self):
        (self.target / "a.txt").write_text("ESKİ SÜRÜM", encoding="utf-8")
        experiment = self.passed_experiment()
        self.box.write("a.txt", "YENİ SÜRÜM")

        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(), bench()),
            sandbox=self.box, target_root=self.target, files=["a.txt"])
        self.assertEqual((self.target / "a.txt").read_text(encoding="utf-8"), "YENİ SÜRÜM")

        self.promoter.rollback(result.snapshot_id)
        self.assertEqual((self.target / "a.txt").read_text(encoding="utf-8"), "ESKİ SÜRÜM")

    def test_rollback_of_an_unknown_snapshot_is_refused(self):
        with self.assertRaises(PromotionRefused):
            self.promoter.rollback("olmayan-goruntu")

    def test_multi_file_promotion_applies_all(self):
        experiment = self.passed_experiment()
        for name in ("a.txt", "alt/b.txt", "alt/derin/c.txt"):
            self.box.write(name, f"içerik {name}")

        result = self.promoter.promote(
            self.registry.get(experiment.id), compare(bench(), bench()),
            sandbox=self.box, target_root=self.target,
            files=["a.txt", "alt/b.txt", "alt/derin/c.txt"])

        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.applied), 3)
        self.assertTrue((self.target / "alt" / "derin" / "c.txt").is_file())


class TestSandboxStillContained(LabCase):
    """S5's earlier guarantees must survive the promotion layer being added."""

    def test_writing_outside_the_sandbox_is_still_refused(self):
        from jarvis.lab.sandbox import SandboxViolation

        with self.assertRaises(SandboxViolation):
            self.box.write("../../uretim/a.txt", "ele geçirildi")

    def test_promotion_reads_only_from_inside_the_sandbox(self):
        from jarvis.lab.sandbox import SandboxViolation

        with self.assertRaises(SandboxViolation):
            self.box.read("../../uretim/a.txt")


class TestLabIntegration(unittest.TestCase):
    """A real experiment: write code and tests, measure, change, measure, promote."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()

        class FakeConfig:
            def __init__(self, root):
                self.root = root

            def get(self, key, default=None):
                return {
                    "lab.timeout_s": 90,
                    "lab.allowed_commands": ["python", "python3", "py"],
                }.get(key, default)

            def path(self, key, default=""):
                mapping = {"paths.lab": self.root / "lab",
                           "paths.db": self.root / "lab.db",
                           "lab.promotion_target": self.root / "lab" / "promoted"}
                return Path(mapping.get(key, self.root / str(default)))

        from jarvis.lab import Lab

        self.lab = Lab(FakeConfig(self.base))

    def tearDown(self):
        self._tmp.cleanup()

    def test_experiment_records_its_sandbox_and_purpose(self):
        session = self.lab.experiment("toplama fonksiyonunu hızlandır")
        stored = self.lab.registry.get(session.id)
        self.assertEqual(stored.purpose, "toplama fonksiyonunu hızlandır")
        self.assertTrue(Path(stored.sandbox_path).is_dir())

    def test_settle_without_measurement_is_refused(self):
        session = self.lab.experiment("x")
        with self.assertRaises(PromotionRefused):
            session.settle()

    def test_discard_disposes_the_sandbox(self):
        session = self.lab.experiment("x")
        root = session.sandbox.root
        session.discard("artık gerek yok")
        self.assertFalse(root.exists())
        self.assertEqual(self.lab.registry.get(session.id).state, DISCARDED)

    def test_status_reports_self_modification_is_off(self):
        self.assertFalse(self.lab.status()["kendi_kodunu_degistirme"])


if __name__ == "__main__":
    unittest.main()
