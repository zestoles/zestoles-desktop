"""S6b: the experiment planner, and everything it refuses.

The planner is the first component that takes text a language model wrote and
turns it into code a subprocess runs. Every test below is a case where that goes
wrong, because the interesting question is not "can it produce a plan" — it is
"what does it do with a plan it should not accept".

Three of these were found live rather than imagined: qwen3.5:9b produced a test
file containing `exec` and `globals` on its third attempt, and two plans with
broken indentation before that. The validator refused all three without running
anything.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.improve.planner import (  # noqa: E402
    ALLOWED_DUNDERS,
    ALLOWED_IMPORTS,
    FORBIDDEN_NAMES,
    MAX_FILE_BYTES,
    ExperimentPlanner,
    inspect_code,
    inspect_test,
    public_names,
    validate,
)
from jarvis.lab.benchmark import run_suite  # noqa: E402
from jarvis.lab.sandbox import Sandbox, SandboxLimits  # noqa: E402

BASELINE = "def topla(sayilar):\n    return sum(sayilar)\n"
CANDIDATE = "def topla(sayilar):\n    toplam = 0\n    for s in sayilar:\n        toplam += s\n    return toplam\n"
TEST = (
    "import unittest\n"
    "import hesap\n"
    "class T(unittest.TestCase):\n"
    "    def test_bir(self):\n"
    "        self.assertEqual(hesap.topla([1, 2]), 3)\n"
    "    def test_iki(self):\n"
    "        self.assertEqual(hesap.topla([]), 0)\n"
)


def payload(**overrides):
    base = {"module_name": "hesap", "summary": "deneme",
            "baseline_code": BASELINE, "candidate_code": CANDIDATE,
            "test_code": TEST}
    base.update(overrides)
    return base


class TestAcceptance(unittest.TestCase):
    def test_a_sound_plan_is_accepted(self):
        review = validate(payload(), "h1")
        self.assertTrue(review.ok, review.problems)

    def test_the_validator_builds_the_paths_not_the_model(self):
        """The model names a module. Every path is constructed here.

        This is what removes the traversal class entirely: `..`, absolute paths,
        drive-relative paths, UNC and alternate data streams cannot be expressed
        in the output format, so they cannot be attempted.
        """
        review = validate(payload(), "h1")
        self.assertEqual(sorted(review.plan.setup_files),
                         ["tests/hesap.py", "tests/test_hesap.py"])
        self.assertEqual(list(review.plan.changed_files), ["tests/hesap.py"])
        self.assertEqual(review.plan.test_target, "tests")

    def test_only_files_it_created_may_change(self):
        review = validate(payload(), "h1")
        self.assertLessEqual(set(review.plan.changed_files),
                             set(review.plan.setup_files))

    def test_only_changed_files_may_be_promoted(self):
        review = validate(payload(), "h1")
        self.assertLessEqual(set(review.plan.promote),
                             set(review.plan.changed_files))

    def test_the_hypothesis_is_carried_through(self):
        self.assertEqual(validate(payload(), "abc123").plan.hypothesis_id, "abc123")


class TestModuleName(unittest.TestCase):
    def test_a_path_cannot_be_smuggled_through_the_name(self):
        for name in ("../../etc/passwd", "C:/Windows/System32", "..\\gizli",
                     "a/b", "\\\\sunucu\\pay", "con", "hesap.py", "hesap:gizli",
                     "hesap\x00", "-rf", ".hesap"):
            with self.subTest(name=name):
                self.assertFalse(validate(payload(module_name=name), "h").ok)

    def test_reserved_names_are_refused(self):
        for name in ("unittest", "json", "os", "sys", "typing"):
            with self.subTest(name=name):
                self.assertFalse(validate(payload(module_name=name), "h").ok)

    def test_shape_rules(self):
        self.assertFalse(validate(payload(module_name="ab"), "h").ok)       # too short
        self.assertFalse(validate(payload(module_name="9abc"), "h").ok)     # leading digit
        self.assertFalse(validate(payload(module_name="Hesap"), "h").ok)    # uppercase
        self.assertFalse(validate(payload(module_name="hesap-1"), "h").ok)  # punctuation
        self.assertFalse(validate(payload(module_name="ö" * 5), "h").ok)    # non-ascii
        self.assertFalse(validate(payload(module_name="a" * 40), "h").ok)   # too long


class TestImportAllowlist(unittest.TestCase):
    def test_the_ways_out_of_a_sandbox_are_refused(self):
        for module in ("os", "sys", "subprocess", "pathlib", "shutil", "socket",
                       "ctypes", "importlib", "urllib.request", "http.client",
                       "multiprocessing", "threading", "pickle", "webbrowser",
                       "tempfile", "glob", "sqlite3"):
            with self.subTest(module=module):
                code = f"import {module}\ndef topla(x):\n    return sum(x)\n"
                self.assertFalse(validate(payload(baseline_code=code), "h").ok)

    def test_from_imports_are_checked_too(self):
        code = "from os import system\ndef topla(x):\n    return sum(x)\n"
        self.assertFalse(validate(payload(baseline_code=code), "h").ok)

    def test_relative_imports_are_refused(self):
        self.assertTrue(inspect_code("from . import yan", "x"))

    def test_the_useful_ones_are_allowed(self):
        for module in sorted(ALLOWED_IMPORTS):
            with self.subTest(module=module):
                self.assertEqual(inspect_code(f"import {module}\n", "x"), [])

    def test_the_module_may_be_imported_only_where_it_is_needed(self):
        """The test imports the module under test; the module does not."""
        self.assertEqual(
            inspect_code("import hesap\n", "test",
                         extra_allowed=frozenset({"hesap"})), [])
        self.assertTrue(inspect_code("import hesap\n", "baseline"))


class TestForbiddenNames(unittest.TestCase):
    def test_the_names_that_make_a_sandbox_pointless(self):
        for name in sorted(FORBIDDEN_NAMES):
            with self.subTest(name=name):
                code = f"def topla(x):\n    return {name}\n"
                self.assertTrue(inspect_code(code, "x"), name)

    def test_exec_and_globals_in_a_test_file(self):
        """Exactly what the model produced on its third live attempt."""
        hostile = (
            "import unittest\n"
            "import hesap\n"
            "class T(unittest.TestCase):\n"
            "    def test_bir(self):\n"
            "        exec('x = 1', globals())\n"
            "    def test_iki(self):\n"
            "        self.assertTrue(True)\n"
        )
        review = validate(payload(test_code=hostile), "h")
        self.assertFalse(review.ok)
        self.assertTrue(any("exec" in p for p in review.problems), review.problems)

    def test_dangerous_dunders_are_refused(self):
        for attribute in ("__class__", "__subclasses__", "__globals__", "__mro__",
                          "__builtins__", "__code__", "__bases__", "__reduce__"):
            with self.subTest(attribute=attribute):
                code = f"def topla(x):\n    return x.{attribute}\n"
                self.assertTrue(inspect_code(code, "x"), attribute)

    def test_ordinary_dunders_are_allowed(self):
        for attribute in sorted(ALLOWED_DUNDERS):
            with self.subTest(attribute=attribute):
                self.assertEqual(inspect_code(f"def f(x):\n    return x.{attribute}\n",
                                              "x"), [])

    def test_a_forbidden_call_through_an_attribute(self):
        self.assertTrue(inspect_code("def f(m):\n    return m.eval('1')\n", "x"))


class TestCodeQuality(unittest.TestCase):
    def test_broken_code_is_a_rejected_plan_not_a_failed_experiment(self):
        """Cheaper, and it keeps "the model wrote nonsense" separate from
        "the idea was wrong". Both live rejections were this."""
        review = validate(payload(candidate_code="def topla(x)\n    return 1\n"), "h")
        self.assertFalse(review.ok)
        self.assertTrue(any("sözdizimi" in p for p in review.problems))

    def test_indentation_damage_is_caught(self):
        broken = "def topla(x):\n    a = 1\n  return a\n"
        self.assertFalse(validate(payload(candidate_code=broken), "h").ok)

    def test_an_identical_candidate_has_nothing_to_measure(self):
        self.assertFalse(validate(payload(candidate_code=BASELINE), "h").ok)

    def test_a_candidate_may_not_drop_public_names(self):
        """Otherwise the same tests would be measuring something else."""
        review = validate(payload(candidate_code="def baska(x):\n    return 1\n"), "h")
        self.assertFalse(review.ok)
        self.assertTrue(any("topla" in p for p in review.problems))

    def test_private_helpers_may_come_and_go(self):
        candidate = ("def _yardim(x):\n    return x\n"
                     "def topla(sayilar):\n    return sum(_yardim(sayilar))\n")
        self.assertTrue(validate(payload(candidate_code=candidate), "h").ok)

    def test_public_names_reads_functions_and_classes(self):
        self.assertEqual(public_names("def a():\n    pass\nclass B:\n    pass\n"
                                      "def _c():\n    pass\n"), {"a", "B"})

    def test_empty_files_are_refused(self):
        for field in ("baseline_code", "candidate_code", "test_code"):
            with self.subTest(field=field):
                self.assertFalse(validate(payload(**{field: "   "}), "h").ok)

    def test_size_limits(self):
        huge = "x = 1\n" * (MAX_FILE_BYTES // 3)
        self.assertFalse(validate(payload(baseline_code=huge), "h").ok)


class TestGeneratedTests(unittest.TestCase):
    def test_the_test_must_import_the_module(self):
        orphan = ("import unittest\n"
                  "class T(unittest.TestCase):\n"
                  "    def test_a(self):\n        self.assertTrue(True)\n"
                  "    def test_b(self):\n        self.assertTrue(True)\n")
        self.assertTrue(any("import" in p for p in inspect_test(orphan, "hesap")))

    def test_one_test_method_is_not_a_suite(self):
        thin = ("import unittest\nimport hesap\n"
                "class T(unittest.TestCase):\n"
                "    def test_a(self):\n        self.assertTrue(hesap)\n")
        self.assertTrue(any("iki test" in p for p in inspect_test(thin, "hesap")))

    def test_tests_without_assertions_are_not_tests(self):
        empty = ("import unittest\nimport hesap\n"
                 "class T(unittest.TestCase):\n"
                 "    def test_a(self):\n        hesap.topla([])\n"
                 "    def test_b(self):\n        hesap.topla([1])\n")
        self.assertTrue(any("assert" in p for p in inspect_test(empty, "hesap")))

    def test_a_file_without_a_testcase_is_refused(self):
        self.assertTrue(any("TestCase" in p for p in
                            inspect_test("import hesap\nx = 1\n", "hesap")))


class TestPlannerBehaviour(unittest.TestCase):
    """The model call itself: what happens when the reply is not usable."""

    class Recorder:
        def __init__(self, reply=None, error=None):
            self.reply = reply
            self.error = error
            self.calls = []
            self.local = self

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if self.error is not None:
                raise self.error
            return self.reply

    class Events:
        def __init__(self):
            self.published = []

        def publish(self, source, kind, message, level="info", data=None):
            self.published.append((source, kind, message, level))

    def plan_from(self, reply, **kwargs):
        brain = self.Recorder(reply=reply)
        events = self.Events()
        planner = ExperimentPlanner(brain, events=events, **kwargs)
        return planner, planner(_Hypothesis(), None), events

    def test_a_usable_reply_becomes_a_plan(self):
        import json

        _, plan, events = self.plan_from(json.dumps(payload()))
        self.assertIsNotNone(plan)
        self.assertIn("plan.ready", [kind for _, kind, _, _ in events.published])

    def test_invalid_json_is_refused_quietly(self):
        planner, plan, events = self.plan_from("bu json degil")
        self.assertIsNone(plan)
        self.assertIn("plan.rejected", [kind for _, kind, _, _ in events.published])

    def test_a_reply_that_is_not_an_object(self):
        _, plan, _ = self.plan_from("[1, 2, 3]")
        self.assertIsNone(plan)

    def test_a_dead_model_is_not_a_crash(self):
        brain = self.Recorder(error=OSError("ollama yok"))
        planner = ExperimentPlanner(brain, events=self.Events())
        self.assertIsNone(planner(_Hypothesis(), None))

    def test_the_call_is_booked_against_a_purpose(self):
        """S9b found local calls going unrecorded. Planning is a local call."""
        import json

        planner, _, _ = self.plan_from(json.dumps(payload()))
        _, kwargs = planner.brain.calls[0]
        self.assertEqual(kwargs["purpose"], "deney-plani")

    def test_the_reply_is_shape_constrained_at_the_server_too(self):
        import json

        planner, _, _ = self.plan_from(json.dumps(payload()))
        _, kwargs = planner.brain.calls[0]
        self.assertIn("schema", kwargs)

    def test_the_rejection_reasons_survive_for_the_report(self):
        import json

        planner, plan, _ = self.plan_from(json.dumps(payload(module_name="../x")))
        self.assertIsNone(plan)
        self.assertTrue(planner.last_review.problems)


class _Hypothesis:
    id = "h-test"
    title = "başlık"
    statement = "iddia"


class TestPlanRunsForReal(unittest.TestCase):
    """No mocks: a validated plan, a real sandbox, a real unittest run.

    A planner whose output has never been executed is a planner nobody has
    tested — the schema says the shape is right, and says nothing about whether
    the thing runs.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.box = Sandbox(Path(self._tmp.name) / "kutu",
                           limits=SandboxLimits(timeout_s=120))

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_validated_plan_produces_a_green_baseline_and_candidate(self):
        plan = validate(payload(), "h").plan
        for relative, content in plan.setup_files.items():
            self.box.write(relative, content)
        baseline = run_suite(self.box, target=plan.test_target)
        self.assertTrue(baseline.passed, baseline.summary())
        self.assertEqual(baseline.tests, 2)

        for relative, content in plan.changed_files.items():
            self.box.write(relative, content)
        candidate = run_suite(self.box, target=plan.test_target)
        self.assertTrue(candidate.passed, candidate.summary())
        self.assertEqual(candidate.tests, 2)

    def test_a_candidate_that_breaks_behaviour_is_measured_as_broken(self):
        """The failure path has to be real too, or only half of it is tested."""
        broken = "def topla(sayilar):\n    return 999\n"
        plan = validate(payload(candidate_code=broken), "h").plan
        for relative, content in plan.setup_files.items():
            self.box.write(relative, content)
        self.assertTrue(run_suite(self.box, target=plan.test_target).passed)

        for relative, content in plan.changed_files.items():
            self.box.write(relative, content)
        self.assertFalse(run_suite(self.box, target=plan.test_target).passed)

    def test_every_plan_path_is_writable_inside_the_sandbox(self):
        plan = validate(payload(), "h").plan
        for relative in list(plan.setup_files) + list(plan.changed_files):
            with self.subTest(path=relative):
                written = self.box.write(relative, "# x\n")
                self.assertTrue(written.resolve().is_relative_to(self.box.root.resolve()))


class TestShapeGateBeforePlanning(unittest.TestCase):
    """The cycle must not hand an unshapeable gap to the planner.

    Built on the real engine with a real database rather than by reading the
    source: the question is whether the planner gets called, and only running it
    answers that.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.calls = []

        from jarvis.config import Config
        from jarvis.improve.engine import ImprovementEngine
        from jarvis.lab import Lab

        class Cfg(Config):
            def __init__(self, root):
                self._data = {}
                self._root = root

            def get(self, dotted, default=None):
                return {"improve.minimum_score": 0.0,
                        "lab.promotion_target": str(self._root / "promoted")
                        }.get(dotted, default)

            def path(self, dotted, default=""):
                return self._root / Path(default).name if default else self._root

            @property
            def root(self):
                return self._root

        class Local:
            model = "test-model"

            def chat(self, messages, **kwargs):
                import json as _json

                return _json.dumps({"title": "bir fikir", "statement": "bir iddia",
                                    "how_to_measure": "ölçülür"})

        class Brain:
            local = Local()

        config = Cfg(self.root)
        self.engine = ImprovementEngine(config, Brain(), lab=Lab(config))
        self.engine.capabilities.seed()

        # A quiet, idle machine, so the cycle reaches the gate under test rather
        # than stopping at the resource policy on whatever this desktop is doing.
        import time as _time
        import types as _types

        from jarvis.autonomy.resources import Snapshot

        self.engine.monitor = _types.SimpleNamespace(
            snapshot=lambda: Snapshot(_time.time(), 99_999.0, 1.0, 10.0, 1.0, 100, 16_000))

    def tearDown(self):
        self._tmp.cleanup()

    def planner(self, hypothesis, gap):
        self.calls.append(gap.key)
        return None

    def all_missing(self):
        """Leave the registry with nothing that has a baseline."""
        from jarvis.improve.capabilities import MISSING

        for capability in self.engine.capabilities.list():
            self.engine.capabilities.set_status(capability.name, MISSING)

    def test_only_shaped_gaps_reach_the_planner(self):
        shape = {gap.key: gap.experiment_shaped
                 for gap in self.engine.gaps.detect(limit=50)}
        self.engine.cycle(planner=self.planner)
        self.assertTrue(self.calls, "hiç plan denenmedi — kurulum yanlış")
        for key in self.calls:
            with self.subTest(gap=key):
                self.assertTrue(shape.get(key), f"{key} deney-şekilli değildi")

    def test_a_missing_capability_is_never_planned(self):
        """browser.automation is the gap that produced S6b's invented baseline."""
        self.engine.cycle(planner=self.planner)
        self.assertNotIn("yetenek:browser.automation", self.calls)
        self.assertNotIn("yetenek:voice.io", self.calls)

    def test_with_no_baseline_anywhere_the_planner_is_never_called(self):
        self.all_missing()
        result = self.engine.cycle(planner=self.planner)
        self.assertEqual(self.calls, [])
        self.assertIn("deney-şekilli", result.stopped)

    def test_the_reason_is_reported_not_swallowed(self):
        self.all_missing()
        self.assertIn("baseline", self.engine.cycle(planner=self.planner).stopped)

    def test_hypotheses_are_still_proposed_for_unshaped_gaps(self):
        """Knowing what is missing is worth keeping; it just is not an experiment."""
        self.all_missing()
        before = len(self.engine.hypotheses.list())
        self.engine.cycle(planner=self.planner)
        self.assertGreater(len(self.engine.hypotheses.list()), before)

    def test_no_experiment_budget_is_spent_on_a_refused_gap(self):
        self.all_missing()
        used = self.engine.budget.used("deney")
        self.engine.cycle(planner=self.planner)
        self.assertEqual(self.engine.budget.used("deney"), used)


class TestGatesUntouched(unittest.TestCase):
    """S6b adds a way to design experiments, not a way around the gates."""

    def test_self_modification_is_still_off(self):
        from jarvis.lab.promotion import ALLOW_SELF_MODIFICATION

        self.assertFalse(ALLOW_SELF_MODIFICATION)

    def test_the_planner_cannot_name_the_source_tree(self):
        for name in ("jarvis", "tests", "persona", "run", "config"):
            review = validate(payload(module_name=name), "h")
            if review.ok:
                # A legal module name still lands under tests/ inside the sandbox
                # and can never address the project root.
                for path in review.plan.setup_files:
                    self.assertTrue(path.startswith("tests/"), path)
                    self.assertNotIn("..", path)

    def test_the_engine_still_refuses_to_run_without_a_planner(self):
        import inspect

        from jarvis.improve.engine import ImprovementEngine

        source = inspect.getsource(ImprovementEngine.cycle)
        self.assertIn("deney planlayıcı verilmedi", source)

    def test_planning_checks_the_experiment_budget_before_spending_a_call(self):
        import inspect

        from jarvis.improve.engine import ImprovementEngine

        source = inspect.getsource(ImprovementEngine.cycle)
        self.assertIn("budget.check", source)


if __name__ == "__main__":
    unittest.main()
