"""Shared state, the composition root, and the import-graph rules they enforce.

The structural tests matter as much as the behavioural ones. "state.py imports
nothing of ours" is a rule that holds only until somebody adds a convenient import,
and by the time it fails the symptom is an ImportError in a module nobody touched.
Reading the AST makes the rule enforceable instead of aspirational.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import ast
import sys
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jarvis import state as state_module  # noqa: E402
from jarvis.runtime import EventProjector, Runtime, _plain  # noqa: E402
from jarvis.state import (  # noqa: E402
    AGENTS,
    AUTONOMY,
    IDLE,
    RESEARCH,
    SECTIONS,
    SYSTEM,
    THINKING,
    SharedState,
)


@dataclass
class FakeEvent:
    source: str
    kind: str
    message: str = "mesaj"
    level: str = "info"
    ts: float = 1000.0
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class TestSharedState(unittest.TestCase):
    def setUp(self):
        self.state = SharedState()

    def test_unknown_section_is_refused(self):
        with self.assertRaises(KeyError):
            self.state.update("uydurma", x=1)
        with self.assertRaises(KeyError):
            self.state.get("uydurma")

    def test_update_merges(self):
        self.state.update(SYSTEM, a=1)
        self.state.update(SYSTEM, b=2)
        self.assertEqual(self.state.get(SYSTEM)["a"], 1)
        self.assertEqual(self.state.get(SYSTEM)["b"], 2)

    def test_replace_discards(self):
        self.state.update(SYSTEM, a=1)
        self.state.replace(SYSTEM, {"b": 2})
        self.assertNotIn("a", self.state.get(SYSTEM))

    def test_version_increments_on_every_write(self):
        before = self.state.version
        self.state.update(SYSTEM, a=1)
        self.state.update(AGENTS, b=2)
        self.assertEqual(self.state.version, before + 2)

    def test_readers_get_a_copy(self):
        """A consumer that mutates what it was handed must not corrupt the source."""
        self.state.update(SYSTEM, items=[1, 2])
        snapshot = self.state.get(SYSTEM)
        snapshot["items"] = "ele geçirildi"
        self.assertEqual(self.state.get(SYSTEM)["items"], [1, 2])

    def test_snapshot_carries_every_section(self):
        snapshot = self.state.snapshot()
        self.assertEqual(set(snapshot["sections"]), set(SECTIONS))
        self.assertIn("version", snapshot)
        self.assertIn("activity", snapshot)

    def test_staleness_is_detected_by_version(self):
        known = self.state.version
        self.assertFalse(self.state.is_stale(known))
        self.state.update(SYSTEM, a=1)
        self.assertTrue(self.state.is_stale(known))

    def test_activity_is_recorded_in_the_system_section(self):
        self.state.set_activity(THINKING, "planlıyor")
        self.assertEqual(self.state.activity, THINKING)
        self.assertEqual(self.state.get(SYSTEM)["activity_detail"], "planlıyor")

    def test_unknown_activity_becomes_an_error_loudly(self):
        """Silently accepting one leaves the UI showing a state it cannot render."""
        self.state.set_activity("dans_ediyor")
        self.assertEqual(self.state.activity, state_module.ERROR)
        self.assertIn("bilinmeyen", self.state.get(SYSTEM)["activity_detail"])

    def test_watchers_are_told(self):
        seen = []
        self.state.watch(lambda section, data: seen.append((section, data)))
        self.state.update(AGENTS, running=True)
        self.assertEqual(seen[0][0], AGENTS)

    def test_a_broken_watcher_cannot_stop_a_writer(self):
        def explode(section, data):
            raise RuntimeError("izleyici bozuk")

        seen = []
        self.state.watch(explode)
        self.state.watch(lambda s, d: seen.append(s))
        self.state.update(AGENTS, running=True)
        self.assertEqual(seen, [AGENTS])

    def test_unwatch_stops_delivery(self):
        seen = []
        stop = self.state.watch(lambda s, d: seen.append(s))
        stop()
        self.state.update(AGENTS, running=True)
        self.assertEqual(seen, [])

    def test_concurrent_writers_do_not_lose_updates(self):
        """The scheduler writes from its own thread while the terminal reads."""
        def writer(index: int) -> None:
            for step in range(50):
                self.state.update(AUTONOMY, **{f"w{index}_{step}": step})

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(self.state.get(AUTONOMY)), 200)
        self.assertEqual(self.state.version, 200)

    def test_reset_clears_everything(self):
        self.state.update(SYSTEM, a=1)
        self.state.reset()
        self.assertEqual(self.state.get(SYSTEM), {})
        self.assertEqual(self.state.activity, IDLE)


class TestImportGraph(unittest.TestCase):
    """Structural rules that stop being true the moment nobody checks them."""

    def _imports(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # a relative import is one of ours by definition
                    found.add("." * node.level + (node.module or ""))
                elif node.module:
                    found.add(node.module)
        return found

    def test_state_is_a_leaf(self):
        """It may be imported by anything; it may import nothing of ours."""
        ours = [name for name in self._imports(ROOT / "jarvis" / "state.py")
                if name.startswith(".") or name.startswith("jarvis")]
        self.assertEqual(ours, [], f"state.py bizden import ediyor: {ours}")

    def test_no_domain_module_imports_the_cli(self):
        """The terminal is a consumer. Nothing below it may reach back up."""
        offenders = []
        for path in (ROOT / "jarvis").rglob("*.py"):
            if "cli" in path.parts or path.name == "runtime.py":
                continue
            for name in self._imports(path):
                if "cli" in name.split(".") or name.endswith(".cli"):
                    offenders.append(f"{path.name} → {name}")
        self.assertEqual(offenders, [], f"domain katmanı CLI'a bağlı: {offenders}")

    def test_no_domain_module_imports_the_runtime(self):
        """Composition happens above; importing it downward creates the cycle."""
        offenders = []
        for path in (ROOT / "jarvis").rglob("*.py"):
            if path.name in ("runtime.py",) or "cli" in path.parts:
                continue
            for name in self._imports(path):
                if name.endswith("runtime") or name == ".runtime":
                    offenders.append(f"{path.name} → {name}")
        self.assertEqual(offenders, [], f"domain katmanı runtime'a bağlı: {offenders}")

    def test_every_package_imports_without_side_effects(self):
        import importlib

        for module in ("jarvis.state", "jarvis.runtime", "jarvis.cli",
                       "jarvis.autonomy", "jarvis.agents", "jarvis.research",
                       "jarvis.lab", "jarvis.improve", "jarvis.memory", "jarvis.brain"):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))


class TestEventProjection(unittest.TestCase):
    """Events say what happened; the projector turns that into what is happening."""

    def setUp(self):
        self.state = SharedState()
        self.project = EventProjector(self.state)

    def test_a_run_starting_sets_the_activity(self):
        self.project(FakeEvent("agent", "run.start", "orkestrasyon başladı"))
        self.assertEqual(self.state.activity, THINKING)

    def test_a_run_finishing_returns_to_idle(self):
        self.project(FakeEvent("agent", "run.start"))
        self.project(FakeEvent("agent", "run.done"))
        self.assertEqual(self.state.activity, IDLE)

    def test_nested_runs_only_clear_at_the_outermost_end(self):
        """An inner step finishing does not mean the system went idle."""
        self.project(FakeEvent("agent", "run.start"))
        self.project(FakeEvent("agent", "start"))
        self.project(FakeEvent("agent", "done"))
        self.assertEqual(self.state.activity, THINKING)
        self.project(FakeEvent("agent", "run.done"))
        self.assertEqual(self.state.activity, IDLE)

    def test_research_sets_its_own_activity(self):
        self.project(FakeEvent("research", "start", "araştırma başladı"))
        self.assertEqual(self.state.activity, state_module.RESEARCHING)

    def test_events_land_in_the_matching_section(self):
        self.project(FakeEvent("research", "sources", "5 kaynak seçildi"))
        self.assertEqual(self.state.get(RESEARCH)["last_event"], "sources")

    def test_errors_are_surfaced_on_the_system_section(self):
        self.project(FakeEvent("task", "error", "patladı", level="error"))
        self.assertIn("patladı", self.state.get(SYSTEM)["last_error"])

    def test_a_policy_stance_updates_the_mode(self):
        self.project(FakeEvent("policy", "stance", "idle: boşta", data={"mode": "idle"}))
        self.assertEqual(self.state.get(AUTONOMY)["mode"], "idle")

    def test_an_unknown_source_is_ignored_quietly(self):
        self.project(FakeEvent("uydurma", "sey"))
        self.assertEqual(self.state.activity, IDLE)

    def test_a_malformed_event_cannot_break_projection(self):
        class Broken:
            source = "agent"
            kind = "run.start"

            @property
            def message(self):
                raise RuntimeError("bozuk olay")

        self.project(Broken())  # must not raise
        self.assertIsNotNone(self.state.snapshot())

    def test_depth_never_goes_negative(self):
        """An end without a matching start must not leave the counter broken."""
        self.project(FakeEvent("agent", "run.done"))
        self.project(FakeEvent("agent", "run.start"))
        self.assertEqual(self.state.activity, THINKING)


class TestPlainFlattening(unittest.TestCase):
    def test_scalars_survive(self):
        self.assertEqual(_plain({"a": 1, "b": "x", "c": True, "d": None}),
                         {"a": 1, "b": "x", "c": True, "d": None})

    def test_nested_objects_become_strings(self):
        result = _plain({"verdict": object()})
        self.assertIsInstance(result["verdict"], str)

    def test_lists_are_flattened(self):
        self.assertEqual(_plain({"names": ["a", "b"]})["names"], ["a", "b"])

    def test_a_non_dict_status_still_flattens(self):
        self.assertIn("value", _plain("bir metin"))


class TestRuntimeAssembly(unittest.TestCase):
    def test_a_bare_runtime_still_has_state(self):
        from jarvis.config import Config

        runtime = Runtime(config=Config.load(), state=SharedState())
        self.assertIsNone(runtime.events)
        self.assertEqual(runtime.warnings, [])

    def test_refresh_without_subsystems_does_not_raise(self):
        from jarvis.config import Config

        runtime = Runtime(config=Config.load(), state=SharedState())
        snapshot = runtime.refresh()
        self.assertIn("sections", snapshot)

    def test_shutdown_without_subsystems_is_safe(self):
        from jarvis.config import Config

        runtime = Runtime(config=Config.load(), state=SharedState())
        runtime.shutdown()
        self.assertEqual(runtime.state.activity, IDLE)


if __name__ == "__main__":
    unittest.main()
