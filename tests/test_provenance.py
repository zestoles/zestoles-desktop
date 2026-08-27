"""Can a finished experiment still answer "which baseline did I measure?"

Until this existed the answer was no. The registry kept what the measurements
said and which paths were touched; the sandbox kept one version at a time,
and the candidate is written over the baseline before the second measurement.
So the four experiments promoted on 17.08 recorded candidates 1.6x-1.9x faster
than their baselines, and the baseline code those numbers described was gone
before anyone could ask why.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.lab.benchmark import compare  # noqa: E402
from jarvis.lab.promotion import evaluate  # noqa: E402
from jarvis.lab.provenance import (  # noqa: E402
    BASELINE,
    CANDIDATE,
    MANIFEST,
    Provenance,
    ProvenanceRefused,
    digest,
)
from jarvis.lab.registry import PASSED, ExperimentRegistry  # noqa: E402
from tests.test_promotion import bench  # noqa: E402

BASE_SOURCE = "def f(items):\n    return [x for x in items]\n"
TEST_SOURCE = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
CAND_SOURCE = "def f(items):\n    return list(items)\n"


class ProvenanceCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "provenance"

    def tearDown(self):
        self._tmp.cleanup()

    def record(self, experiment_id="deney1", *, candidate=True):
        return Provenance.record(
            self.root, experiment_id,
            baseline={"tests/m.py": BASE_SOURCE, "tests/test_m.py": TEST_SOURCE},
            candidate={"tests/m.py": CAND_SOURCE} if candidate else None,
            test_target="tests")


class TestRecording(ProvenanceCase):
    def test_the_baseline_bytes_are_kept(self):
        provenance = self.record()
        self.assertEqual(provenance.read(BASELINE, "tests/m.py"), BASE_SOURCE)
        self.assertEqual(provenance.read(BASELINE, "tests/test_m.py"), TEST_SOURCE)

    def test_the_candidate_is_kept_separately(self):
        """The whole failure was one path holding two versions in turn."""
        provenance = self.record()
        self.assertEqual(provenance.read(BASELINE, "tests/m.py"), BASE_SOURCE)
        self.assertEqual(provenance.read(CANDIDATE, "tests/m.py"), CAND_SOURCE)
        self.assertNotEqual(provenance.read(BASELINE, "tests/m.py"),
                            provenance.read(CANDIDATE, "tests/m.py"))

    def test_the_hash_is_of_the_content_that_was_written(self):
        provenance = self.record()
        expected, size = digest(BASE_SOURCE)
        self.assertEqual(provenance.baseline["tests/m.py"].sha256, expected)
        self.assertEqual(provenance.baseline["tests/m.py"].size, size)

    def test_a_fresh_record_is_intact(self):
        self.assertTrue(self.record().intact)

    def test_the_manifest_is_readable_json(self):
        self.record()
        data = json.loads((self.root / "deney1" / MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], "deney1")
        self.assertEqual(data["test_target"], "tests")
        self.assertIn("tests/m.py", data[BASELINE])

    def test_the_candidate_can_arrive_after_the_baseline(self):
        """Which is the real order: the baseline is recorded before it is
        measured, the candidate only exists afterwards."""
        provenance = Provenance.record(self.root, "d2", baseline={"a.py": BASE_SOURCE})
        self.assertEqual(provenance.candidate, {})
        provenance.add_candidate({"a.py": CAND_SOURCE})
        self.assertEqual(Provenance.load(self.root, "d2").read(CANDIDATE, "a.py"),
                         CAND_SOURCE)


class TestIntegrity(ProvenanceCase):
    def test_an_edited_artifact_is_detected(self):
        provenance = self.record()
        (self.root / "deney1" / BASELINE / "tests" / "m.py").write_text(
            "def f(items):\n    return []\n", encoding="utf-8")
        problems = provenance.verify()
        self.assertTrue(problems)
        self.assertIn("içerik değişmiş", problems[0])
        self.assertFalse(provenance.intact)

    def test_a_missing_artifact_is_detected(self):
        provenance = self.record()
        (self.root / "deney1" / BASELINE / "tests" / "m.py").unlink()
        self.assertTrue(any("dosya yok" in p for p in provenance.verify()))

    def test_a_truncated_artifact_is_detected(self):
        provenance = self.record()
        (self.root / "deney1" / CANDIDATE / "tests" / "m.py").write_text(
            "", encoding="utf-8")
        self.assertFalse(provenance.intact)

    def test_an_empty_baseline_is_not_intact(self):
        """Nothing recorded is not the same as nothing changed."""
        provenance = Provenance.record(self.root, "bos", baseline={})
        self.assertTrue(any("baseline" in p for p in provenance.verify()))

    def test_verification_survives_a_reload(self):
        self.record()
        self.assertTrue(Provenance.load(self.root, "deney1").intact)

    def test_a_reload_sees_tampering_done_after_the_run(self):
        self.record()
        (self.root / "deney1" / BASELINE / "tests" / "m.py").write_text(
            "bozuldu\n", encoding="utf-8")
        self.assertFalse(Provenance.load(self.root, "deney1").intact)


class TestLoadingRefusals(ProvenanceCase):
    def test_a_missing_record_is_refused(self):
        with self.assertRaises(ProvenanceRefused):
            Provenance.load(self.root, "hicyok")

    def test_a_corrupt_manifest_is_refused(self):
        self.record()
        (self.root / "deney1" / MANIFEST).write_text("{bu json degil", encoding="utf-8")
        with self.assertRaises(ProvenanceRefused):
            Provenance.load(self.root, "deney1")

    def test_a_manifest_entry_without_a_hash_is_refused(self):
        self.record()
        path = self.root / "deney1" / MANIFEST
        data = json.loads(path.read_text(encoding="utf-8"))
        data[BASELINE]["tests/m.py"] = {"size": 10}
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ProvenanceRefused):
            Provenance.load(self.root, "deney1")


class TestPathSafety(ProvenanceCase):
    """The planner builds these paths rather than taking them from the model,
    but this module is downstream of that and does not get to assume it."""

    def bad(self, relative):
        with self.assertRaises(ProvenanceRefused, msg=relative):
            Provenance.record(self.root, "kotu", baseline={relative: "x"})

    def test_traversal_is_refused(self):
        self.bad("../kacak.py")

    def test_a_deep_traversal_is_refused(self):
        self.bad("tests/../../kacak.py")

    def test_an_absolute_path_is_refused(self):
        self.bad("C:/Windows/System32/kacak.py")

    def test_an_empty_path_is_refused(self):
        self.bad("")

    def test_a_nested_relative_path_is_fine(self):
        provenance = Provenance.record(
            self.root, "iyi", baseline={"tests/alt/derin/m.py": BASE_SOURCE})
        self.assertEqual(provenance.read(BASELINE, "tests/alt/derin/m.py"), BASE_SOURCE)


class TestTheGateReadsProvenance(ProvenanceCase):
    """A promotion decision that rests on a measurement nobody can re-check is
    the thing this whole module exists to prevent."""

    def setUp(self):
        super().setUp()
        self.registry = ExperimentRegistry(Path(self._tmp.name) / "lab.db")
        self.target = Path(self._tmp.name) / "uretim"
        self.target.mkdir()
        self.project = Path(self._tmp.name) / "proje"
        (self.project / "jarvis").mkdir(parents=True)

    def decide(self, provenance):
        experiment = self.registry.open("deney")
        return evaluate(
            self.registry.transition(experiment.id, PASSED, reason="ölçüldü"),
            compare(bench(), bench()), target_root=self.target,
            changed_files=["a.py"], project_root=self.project,
            provenance=provenance)

    def test_an_intact_record_promotes(self):
        self.assertTrue(self.decide(self.record()).promotable)

    def test_a_tampered_record_is_refused(self):
        provenance = self.record()
        (self.root / "deney1" / BASELINE / "tests" / "m.py").write_text(
            "bozuldu\n", encoding="utf-8")
        decision = self.decide(provenance)
        self.assertFalse(decision.promotable)
        self.assertTrue(any("kaynak kaydı" in b for b in decision.blocking),
                        decision.blocking)

    def test_a_missing_artifact_is_refused(self):
        provenance = self.record()
        (self.root / "deney1" / BASELINE / "tests" / "m.py").unlink()
        self.assertFalse(self.decide(provenance).promotable)

    def test_no_record_keeps_the_old_behaviour(self):
        """S5 by hand and the existing tests never recorded one, and this gate
        is not the place to break them."""
        self.assertTrue(self.decide(None).promotable)


class TestTheEngineRecordsBeforeItMeasures(unittest.TestCase):
    def test_the_baseline_source_is_kept_before_the_baseline_is_measured(self):
        """Ordering is the whole point. Recording after measure_baseline would
        still be recording the baseline, but recording after the candidate is
        written would silently record the candidate twice."""
        from jarvis.improve.engine import ImprovementEngine

        source = inspect.getsource(ImprovementEngine.run_plan)
        self.assertIn("record_baseline_source", source)
        self.assertLess(source.index("record_baseline_source"),
                        source.index("measure_baseline"))

    def test_the_candidate_source_is_kept_before_the_candidate_is_measured(self):
        from jarvis.improve.engine import ImprovementEngine

        source = inspect.getsource(ImprovementEngine.run_plan)
        self.assertLess(source.index("record_candidate_source"),
                        source.index("measure_candidate"))

    def test_the_session_hands_its_record_to_the_gate(self):
        from jarvis.lab import ExperimentSession

        self.assertIn("provenance=self.provenance",
                      inspect.getsource(ExperimentSession.promote))


if __name__ == "__main__":
    unittest.main()
