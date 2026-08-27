"""Wiring the tool loop into the thing the user actually types at.

The loop existed and the terminal did not use it, which meant JARVIS could
create a file in a test and not when asked. These pin the join: the session
builds an assistant, the terminal shows what it really did, and a failure to
build tools costs tools rather than the conversation.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import builtins
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.cli.repl import ToolActivity, ask_permission  # noqa: E402
from jarvis.cli.session import Session  # noqa: E402
from jarvis.state import SharedState  # noqa: E402


class StubConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, dotted, default=None):
        return self._values.get(dotted, default)


class StubRuntime:
    def __init__(self, config, *, brain=None, events=None):
        self.config = config
        self.brain = brain
        self.memory = None
        self.core = None
        self.agents = None
        self.research = None
        self.lab = None
        self.improve = None
        self.state = SharedState()
        self.warnings: list[str] = []
        self._events = events

    @property
    def events(self):
        return self._events


class StubBrain:
    class _Local:
        model = "test-model"

        def chat(self, messages, **kwargs):
            return '{"action": "reply", "message": "tamam"}'

    def __init__(self):
        self.local = self._Local()


class SessionCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self._tmp.name) / "alan"
        self.runtime = StubRuntime(
            StubConfig({"assistant.workspace": str(self.workspace_root),
                        "assistant.max_steps": 4}),
            brain=StubBrain())
        self.session = Session(self.runtime, history_turns=5)

    def tearDown(self):
        self._tmp.cleanup()


class TestSessionBuildsTheAssistant(SessionCase):
    def test_the_assistant_uses_the_configured_workspace(self):
        assistant = self.session.assistant()
        self.assertIsNotNone(assistant)
        self.assertEqual(assistant.workspace.root, self.workspace_root.resolve())

    def test_the_configured_step_budget_is_honoured(self):
        self.assertEqual(self.session.assistant().max_steps, 4)

    def test_it_is_built_once_and_reused(self):
        first = self.session.assistant()
        self.assertIs(self.session.assistant(), first)

    def test_the_approval_callback_is_refreshed_each_time(self):
        def approve(*_a):
            return True

        self.session.assistant()
        self.assertIs(self.session.assistant(approve=approve).approve, approve)

    def test_a_workspace_that_cannot_be_created_costs_tools_not_the_session(self):
        """Plain conversation has to keep working when tools do not."""
        runtime = StubRuntime(StubConfig({"assistant.workspace": "\x00gecersiz"}),
                              brain=StubBrain())
        session = Session(runtime, history_turns=5)
        self.assertIsNone(session.assistant())
        self.assertTrue(runtime.warnings, "kurulum hatasi kaydedilmeli")

    def test_a_failed_build_is_not_retried_on_every_message(self):
        runtime = StubRuntime(StubConfig({"assistant.workspace": "\x00gecersiz"}),
                              brain=StubBrain())
        session = Session(runtime, history_turns=5)
        session.assistant()
        before = len(runtime.warnings)
        session.assistant()
        self.assertEqual(len(runtime.warnings), before)

    def test_the_default_workspace_is_the_users_home(self):
        runtime = StubRuntime(StubConfig(), brain=StubBrain())
        assistant = Session(runtime, history_turns=5).assistant()
        self.assertEqual(assistant.workspace.root, Path.home().resolve())


class TestToolActivity(unittest.TestCase):
    class Recorder:
        def __init__(self):
            self.rows = []

        def publish(self, source, kind, message, level="info", data=None):
            self.rows.append(kind)

    def test_events_reach_the_real_log(self):
        recorder = self.Recorder()
        ToolActivity(recorder).publish("assistant", "tool.done", "bitti")
        self.assertEqual(recorder.rows, ["tool.done"])

    def test_a_broken_log_does_not_break_the_display(self):
        class Broken:
            def publish(self, *a, **k):
                raise RuntimeError("olay yolu çöktü")

        ToolActivity(Broken()).publish("assistant", "tool.start", "başladı")

    def test_no_downstream_is_fine(self):
        ToolActivity().publish("assistant", "tool.start", "başladı")

    def test_unknown_kinds_are_still_forwarded(self):
        recorder = self.Recorder()
        ToolActivity(recorder).publish("assistant", "turn.start", "istek")
        self.assertEqual(recorder.rows, ["turn.start"])


class TestAskPermission(unittest.TestCase):
    """The default has to be no. A prompt that defaults to yes is not a gate."""

    def answer(self, text):
        original = builtins.input
        builtins.input = lambda *_a: text
        try:
            return ask_permission("fs.write", "medium", {"path": "a.txt"})
        finally:
            builtins.input = original

    def test_an_explicit_yes_approves(self):
        for text in ("e", "E", "evet", "y", "yes", " Evet "):
            with self.subTest(text=text):
                self.assertTrue(self.answer(text))

    def test_anything_else_refuses(self):
        for text in ("", "h", "hayır", "n", "no", "belki", "asdf"):
            with self.subTest(text=text):
                self.assertFalse(self.answer(text))

    def test_an_interrupted_prompt_refuses(self):
        original = builtins.input

        def interrupt(*_a):
            raise KeyboardInterrupt

        builtins.input = interrupt
        try:
            self.assertFalse(ask_permission("fs.write", "medium", {}))
        finally:
            builtins.input = original

    def test_an_closed_stdin_refuses(self):
        original = builtins.input

        def eof(*_a):
            raise EOFError

        builtins.input = eof
        try:
            self.assertFalse(ask_permission("shell.run", "medium", {}))
        finally:
            builtins.input = original


class TestTheTerminalPathIsWiredIn(unittest.TestCase):
    def test_the_repl_tries_tools_before_plain_chat(self):
        import inspect

        from jarvis.cli import repl

        source = inspect.getsource(repl.run_repl)
        self.assertIn("_run_with_tools", source)
        self.assertLess(source.index("_run_with_tools"), source.index("brain.plan"))

    def test_the_terminal_reports_real_failures_itself(self):
        """Not left to the model's prose: the recorded results are printed."""
        import inspect

        from jarvis.cli import repl

        self.assertIn("turn.failures", inspect.getsource(repl._run_with_tools))


if __name__ == "__main__":
    unittest.main()
