"""What happens when an experiment goes wrong, run against the real machinery.

The unit tests elsewhere pin each piece. This file drives the actual Lab — real
sandbox, real subprocess, real registry, real gate — because the failures worth
worrying about live in the seams: an experiment that stops halfway, a
measurement that never happened, a record that outlived the sandbox it came
from.

The rule every test here exists to defend: no path may end in a promotion
without a complete, verifiable measurement behind it.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.lab import Lab  # noqa: E402
from jarvis.lab.benchmark import BenchmarkResult, compare  # noqa: E402
from jarvis.lab.promotion import PromotionRefused, evaluate  # noqa: E402
from jarvis.lab.provenance import BASELINE, CANDIDATE, MANIFEST, Provenance  # noqa: E402
from jarvis.lab.registry import DISCARDED, FAILED, PASSED, PROMOTED  # noqa: E402

MODULE_BASELINE = """\
def benzersiz(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out
"""

MODULE_CANDIDATE = """\
def benzersiz(items):
    return list(dict.fromkeys(items))
"""

MODULE_BROKEN = """\
def benzersiz(items):
    return None
"""

TESTS = """\
import unittest

import benzersiz_modul


class TestBenzersiz(unittest.TestCase):
    def test_tekrarlari_atar(self):
        self.assertEqual(benzersiz_modul.benzersiz([1, 2, 2, 3]), [1, 2, 3])

    def test_sirayi_korur(self):
        self.assertEqual(benzersiz_modul.benzersiz([3, 1, 3]), [3, 1])
"""

MODULE_PATH = "tests/benzersiz_modul.py"
TEST_PATH = "tests/test_benzersiz_modul.py"


class _Config:
    def __init__(self, root: Path):
        self.root = root

    def get(self, key, default=None):
        return {"lab.timeout_s": 90,
                "lab.allowed_commands": ["python", "python3", "py"]}.get(key, default)

    def path(self, key, default=""):
        mapping = {"paths.lab": self.root / "lab",
                   "paths.db": self.root / "lab.db",
                   "lab.promotion_target": self.root / "lab" / "promoted"}
        return Path(mapping.get(key, self.root / str(default)))


class LiveExperimentCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.lab = Lab(_Config(self.base))
        # A stand-in for the source tree, so "did anything reach it" is checkable.
        self.canary = self.base / "kaynak_agaci.py"
        self.canary.write_text("DOKUNULMADI\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def setup_files(self):
        return {MODULE_PATH: MODULE_BASELINE, TEST_PATH: TESTS}

    def seed(self, session, files=None):
        for relative, content in (files or self.setup_files()).items():
            session.sandbox.write(relative, content)
        session.record_baseline_source(files or self.setup_files(), test_target="tests")

    def swap_in(self, session, content):
        session.sandbox.write(MODULE_PATH, content)
        session.record_candidate_source({MODULE_PATH: content})

    def canary_untouched(self):
        self.assertEqual(self.canary.read_text(encoding="utf-8"), "DOKUNULMADI\n")


class TestProvenanceEndToEnd(LiveExperimentCase):
    """The chain sprint 2 built, driven for real rather than in pieces."""

    def run_full(self):
        session = self.lab.experiment("benzersizlestirme")
        self.seed(session)
        session.measure_baseline(python=sys.executable, target="tests")
        self.swap_in(session, MODULE_CANDIDATE)
        session.measure_candidate(python=sys.executable, target="tests")
        session.settle()
        return session

    def test_a_real_run_produces_a_verifiable_record(self):
        session = self.run_full()
        store = self.lab.provenance_dir / session.id
        self.assertTrue((store / MANIFEST).is_file(), "manifest yazılmadı")

        loaded = Provenance.load(self.lab.provenance_dir, session.id)
        self.assertEqual(loaded.verify(), [], loaded.verify())
        self.assertEqual(loaded.read(BASELINE, MODULE_PATH), MODULE_BASELINE)
        self.assertEqual(loaded.read(BASELINE, TEST_PATH), TESTS)
        self.assertEqual(loaded.read(CANDIDATE, MODULE_PATH), MODULE_CANDIDATE)

    def test_the_recorded_hash_is_of_the_code_that_ran(self):
        session = self.run_full()
        loaded = Provenance.load(self.lab.provenance_dir, session.id)
        expected = hashlib.sha256(MODULE_BASELINE.encode("utf-8")).hexdigest()
        self.assertEqual(loaded.baseline[MODULE_PATH].sha256, expected)

    def test_the_baseline_survives_the_candidate_overwriting_it(self):
        """The original failure in one assertion: the sandbox holds the
        candidate at this point, and the baseline is still readable anyway."""
        session = self.run_full()
        self.assertEqual(session.sandbox.read(MODULE_PATH), MODULE_CANDIDATE)
        loaded = Provenance.load(self.lab.provenance_dir, session.id)
        self.assertEqual(loaded.read(BASELINE, MODULE_PATH), MODULE_BASELINE)

    def test_a_real_promotion_passes_the_provenance_check(self):
        session = self.run_full()
        self.assertEqual(session.state, PASSED, session.comparison.summary())
        result = session.promote([MODULE_PATH])
        self.assertTrue(result.ok, result.error)
        self.assertEqual(self.lab.registry.get(session.id).state, PROMOTED)
        promoted = self.lab.promotion_target / MODULE_PATH
        self.assertEqual(promoted.read_text(encoding="utf-8"), MODULE_CANDIDATE)
        self.canary_untouched()

    def test_a_tampered_record_stops_a_real_promotion(self):
        session = self.run_full()
        (self.lab.provenance_dir / session.id / BASELINE / MODULE_PATH).write_text(
            "sonradan degistirildi\n", encoding="utf-8")
        result = session.promote([MODULE_PATH])
        self.assertFalse(result.ok)
        self.assertIn("kaynak kaydı", result.error)
        self.assertFalse((self.lab.promotion_target / MODULE_PATH).exists())
        self.canary_untouched()

    def test_the_record_outlives_the_sandbox(self):
        """Provenance lives outside the sandboxes precisely so that discarding a
        failed experiment does not destroy the evidence about it."""
        session = self.run_full()
        sandbox_root = Path(session.sandbox.root)
        session.discard("test")
        self.assertFalse(sandbox_root.exists(), "sandbox silinmedi")
        loaded = Provenance.load(self.lab.provenance_dir, session.id)
        self.assertEqual(loaded.read(BASELINE, MODULE_PATH), MODULE_BASELINE)
        self.assertEqual(loaded.verify(), [])


class TestInterruptedExperiments(LiveExperimentCase):
    """Every way of stopping partway must land somewhere that is not success."""

    def test_a_run_that_never_measured_a_candidate_cannot_settle(self):
        session = self.lab.experiment("yarida kalan")
        self.seed(session)
        session.measure_baseline(python=sys.executable, target="tests")
        with self.assertRaises(PromotionRefused):
            session.settle()
        self.assertNotEqual(self.lab.registry.get(session.id).state, PASSED)

    def test_a_run_that_never_settled_cannot_promote(self):
        session = self.lab.experiment("olcumsuz")
        self.seed(session)
        session.measure_baseline(python=sys.executable, target="tests")
        with self.assertRaises(PromotionRefused):
            session.promote([MODULE_PATH])
        self.canary_untouched()

    def test_an_interrupted_run_still_left_its_baseline_behind(self):
        """Stopping early is exactly when the inputs matter most."""
        session = self.lab.experiment("kesildi")
        self.seed(session)
        session.measure_baseline(python=sys.executable, target="tests")
        session.discard("kesildi")
        self.assertEqual(self.lab.registry.get(session.id).state, DISCARDED)
        loaded = Provenance.load(self.lab.provenance_dir, session.id)
        self.assertEqual(loaded.read(BASELINE, MODULE_PATH), MODULE_BASELINE)
        self.assertEqual(loaded.candidate, {}, "aday hiç ölçülmedi, kaydı da olmamalı")

    def test_a_discarded_experiment_cannot_be_walked_back_to_promoted(self):
        session = self.lab.experiment("atildi")
        self.seed(session)
        session.measure_baseline(python=sys.executable, target="tests")
        session.discard("atildi")
        with self.assertRaises(PromotionRefused):
            session.promote([MODULE_PATH])
        self.canary_untouched()

    def test_a_candidate_that_breaks_the_tests_fails_the_experiment(self):
        session = self.lab.experiment("bozuk aday")
        self.seed(session)
        session.measure_baseline(python=sys.executable, target="tests")
        self.swap_in(session, MODULE_BROKEN)
        session.measure_candidate(python=sys.executable, target="tests")
        session.settle()
        self.assertEqual(self.lab.registry.get(session.id).state, FAILED)
        result = session.promote([MODULE_PATH])
        self.assertFalse(result.ok)
        self.canary_untouched()

    def test_a_broken_baseline_fails_at_the_gate_not_at_the_filesystem(self):
        """The 17.08 hole, driven through the real machinery."""
        session = self.lab.experiment("kirik baseline")
        broken = {MODULE_PATH: MODULE_BROKEN, TEST_PATH: TESTS}
        for relative, content in broken.items():
            session.sandbox.write(relative, content)
        session.record_baseline_source(broken, test_target="tests")
        baseline = session.measure_baseline(python=sys.executable, target="tests")
        self.assertFalse(baseline.passed, "bu deney kırık bir baseline istiyordu")

        self.swap_in(session, MODULE_CANDIDATE)
        session.measure_candidate(python=sys.executable, target="tests")
        session.settle()
        result = session.promote([MODULE_PATH])
        self.assertFalse(result.ok)
        self.assertIn("baseline", result.error)
        self.assertFalse((self.lab.promotion_target / MODULE_PATH).exists())
        self.canary_untouched()


class TestMalformedMeasurements(unittest.TestCase):
    """Results the gate must not read as success, whatever they claim."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.lab = Lab(_Config(self.base))
        self.target = self.base / "hedef"
        self.target.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def passed_experiment(self):
        experiment = self.lab.registry.open("olcum")
        return self.lab.registry.transition(experiment.id, PASSED, reason="x")

    def decide(self, baseline, candidate):
        return evaluate(self.passed_experiment(), compare(baseline, candidate),
                        target_root=self.target, changed_files=["a.py"],
                        project_root=self.base)

    def ok(self, **kw):
        base = dict(ran=True, tests=10, failures=0, errors=0, skipped=0,
                    duration_ms=1000)
        base.update(kw)
        return BenchmarkResult(**base)

    def test_a_run_that_never_happened_is_not_a_pass(self):
        self.assertFalse(self.ok(ran=False).passed)
        self.assertFalse(self.decide(self.ok(), self.ok(ran=False)).promotable)

    def test_a_timeout_is_not_a_pass_however_clean_the_counts(self):
        self.assertFalse(self.ok(timed_out=True).passed)
        self.assertFalse(self.decide(self.ok(), self.ok(timed_out=True)).promotable)

    def test_passing_is_derived_not_declared(self):
        """`passed` is computed from the counts, so a result cannot claim to be
        green while carrying failures — there is no field to lie in."""
        self.assertFalse(self.ok(errors=5).passed)
        self.assertFalse(self.ok(failures=1).passed)

    def test_more_skips_than_tests_cannot_manufacture_coverage(self):
        self.assertEqual(self.ok(tests=3, skipped=99).effective, 0)

    def test_a_suite_that_skipped_everything_does_not_hold_coverage(self):
        decision = self.decide(self.ok(tests=10, skipped=0),
                               self.ok(tests=10, skipped=10))
        self.assertFalse(decision.promotable)

    def test_a_zero_length_baseline_does_not_divide_by_zero(self):
        self.assertEqual(compare(self.ok(duration_ms=0),
                                 self.ok(duration_ms=500)).speed_ratio, 1.0)

    def test_no_comparison_at_all_is_refused(self):
        decision = evaluate(self.passed_experiment(), None, target_root=self.target,
                            changed_files=["a.py"], project_root=self.base)
        self.assertFalse(decision.promotable)


if __name__ == "__main__":
    unittest.main()
