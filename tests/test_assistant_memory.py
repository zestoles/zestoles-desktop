"""What the conversation leaves behind when tools were involved.

The conversation was already written through to memory; what actually happened
was not. A session where JARVIS read three files and ran a command was recorded
as two sentences of prose about it -- and prose is the one part of a turn that
can be wrong, because a model may write "tamamdır" over a failed tool.

So the record of the steps is written too, built from the real `ToolResult`s.
These pin that, and the two boundaries it must not cross:

- **the record is not conversation.** It goes to memory, never into the history
  handed back to the model. `chat` takes system/user/assistant; a fourth role in
  that list is a request the local model was never promised it could parse.
- **the distiller must not read it as something JARVIS said.** `_transcript`
  labels every non-user role "JARVIS", and a tool measurement filed as a JARVIS
  claim is a category error in the one place the hallucination gate lives.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.assistant import (  # noqa: E402
    REPLY,
    TOOL,
    TOOL_ROLE,
    Assistant,
    Step,
    Turn,
)
from jarvis.assistant.service import AssistantService  # noqa: E402
from jarvis.tools import ToolResult, Workspace  # noqa: E402


def call(tool_name, **arguments):
    return json.dumps({"action": TOOL, "tool": tool_name, "arguments": arguments})


def answer(message="tamam"):
    return json.dumps({"action": REPLY, "message": message})


class ScriptedBrain:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []
        self.local = self

    def chat(self, messages, **_kwargs):
        self.seen.append(list(messages))
        if not self.replies:
            return answer("başka söyleyecek bir şey yok")
        return self.replies.pop(0)


class MemoryCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.workspace = Workspace(self.root / "alan")
        self.written = []

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, *replies):
        self.brain = ScriptedBrain(*replies)
        assistant = Assistant(self.brain, self.workspace)
        return AssistantService(
            assistant, remember=lambda role, text: self.written.append((role, text)))

    def roles(self):
        return [role for role, _ in self.written]

    def records(self):
        return [text for role, text in self.written if role == TOOL_ROLE]


class TestTheRecordIsBuiltFromResults(unittest.TestCase):
    def step(self, tool, ok, output="", error=""):
        return Step(tool, {}, ToolResult(ok, output=output, error=error, tool=tool))

    def test_a_turn_with_no_tools_has_nothing_to_record(self):
        self.assertEqual(Turn(reply="merhaba").tool_record(), "")

    def test_a_successful_call_is_recorded_with_its_output(self):
        turn = Turn(steps=[self.step("fs.read", True, output="merhaba dünya")])
        record = turn.tool_record()
        self.assertIn("fs.read", record)
        self.assertIn("merhaba dünya", record)

    def test_a_failure_is_recorded_as_a_failure(self):
        """The point. The reply may claim success; this cannot."""
        turn = Turn(reply="Tamamdır, dosyayı yazdım!",
                    steps=[self.step("fs.write", False, error="izin yok")])
        record = turn.tool_record()
        self.assertIn("izin yok", record)
        self.assertNotIn("Tamamdır", record)
        self.assertRegex(record.lower(), r"basarisiz|başarısız")

    def test_a_long_output_does_not_become_the_note(self):
        turn = Turn(steps=[self.step("fs.read", True, output="x" * 9000)])
        self.assertLess(len(turn.tool_record()), 2000)

    def test_every_step_is_named(self):
        turn = Turn(steps=[self.step("fs.list", True, output="a"),
                           self.step("shell.run", False, error="yok"),
                           self.step("system.info", True, output="b")])
        record = turn.tool_record()
        for tool in ("fs.list", "shell.run", "system.info"):
            self.assertIn(tool, record)


class TestTheRecordReachesMemory(MemoryCase):
    def test_a_turn_that_used_a_tool_leaves_a_record(self):
        service = self.build(call("fs.list", path="."), answer("bitti"))
        service.handle({"op": "sor", "mesaj": "neler var"})
        self.assertEqual(len(self.records()), 1)
        self.assertIn("fs.list", self.records()[0])

    def test_a_plain_answer_leaves_none(self):
        service = self.build(answer("Python bir dildir."))
        service.handle({"op": "sor", "mesaj": "python nedir"})
        self.assertEqual(self.records(), [])
        self.assertEqual(self.roles(), ["user", "assistant"])

    def test_the_record_is_written_before_the_reply(self):
        """What happened, then what was said about it -- the order they occurred."""
        service = self.build(call("fs.list", path="."), answer("bitti"))
        service.handle({"op": "sor", "mesaj": "neler var"})
        self.assertEqual(self.roles(), ["user", TOOL_ROLE, "assistant"])

    def test_an_approved_call_is_recorded_too(self):
        service = self.build(call("fs.write", path="not.txt", content="merhaba"),
                             answer("yazdım"))
        service.handle({"op": "sor", "mesaj": "not yaz"})
        service.handle({"op": "onay", "evet": True})
        self.assertTrue(any("fs.write" in text for text in self.records()))

    def test_a_refused_call_leaves_no_record_of_work(self):
        service = self.build(call("fs.write", path="not.txt", content="merhaba"))
        service.handle({"op": "sor", "mesaj": "not yaz"})
        service.handle({"op": "onay", "evet": False})
        self.assertEqual(self.records(), [])

    def test_a_queued_turn_records_the_same_way(self):
        from jarvis.autonomy.events import EventLog
        from jarvis.autonomy.runners import RunContext
        from jarvis.autonomy.tasks import Task
        from jarvis.assistant.background import KIND, USER_ORIGIN
        from jarvis.autonomy import runners

        service = self.build(call("fs.list", path="."), answer("bitti"))
        task = Task(id=1, kind=KIND, title="neler var", origin=USER_ORIGIN,
                    payload={"mesaj": "neler var"})
        runners.get(KIND)(RunContext(task=task, events=EventLog(self.root / "o.db"),
                                     should_stop=lambda: False, assistant=service))
        self.assertTrue(any("fs.list" in text for text in self.records()))


class TestTheRecordIsNotConversation(MemoryCase):
    def test_it_never_enters_the_history_the_model_sees(self):
        service = self.build(call("fs.list", path="."), answer("bitti"))
        service.handle({"op": "sor", "mesaj": "neler var"})
        self.assertEqual([turn["role"] for turn in service.history],
                         ["user", "assistant"])

    def test_a_later_turn_sends_only_conversation_roles(self):
        service = self.build(call("fs.list", path="."), answer("bitti"),
                             answer("ikinci"))
        service.handle({"op": "sor", "mesaj": "neler var"})
        service.handle({"op": "sor", "mesaj": "peki ya simdi"})
        sent = self.brain.seen[-1]
        for message in sent:
            with self.subTest(role=message["role"]):
                self.assertIn(message["role"], ("system", "user", "assistant"))

    def test_the_session_keeps_it_out_of_history_too(self):
        """`remember` is wired to `Session.add`, which also feeds the terminal."""
        from jarvis.cli.session import Session

        class FakeMemory:
            def __init__(self):
                self.seen = []

            def remember(self, role, content):
                self.seen.append((role, content))

        class FakeState:
            def update(self, *args, **kwargs):
                pass

        class FakeRuntime:
            def __init__(self, memory):
                self.memory = memory
                self.state = FakeState()
                self.brain = None
                self.config = None

        memory = FakeMemory()
        session = Session(FakeRuntime(memory), history_turns=12)
        session.add("user", "selam")
        session.add(TOOL_ROLE, "fs.list — basarili: a.txt")
        session.add("assistant", "merhaba")

        self.assertEqual([turn["role"] for turn in session.history],
                         ["user", "assistant"])
        self.assertEqual([role for role, _ in memory.seen],
                         ["user", TOOL_ROLE, "assistant"])


class TestTheTerminalRecordsItToo(MemoryCase):
    """Two front doors, one memory. They must write the same thing.

    The interface once held a conversation and forgot all of it because only the
    terminal went through `Session.add`. Same shape of bug, other direction: a
    tool record written by the page and not by the terminal would make what JARVIS
    remembers depend on which window the work was done in.
    """

    def repl_session(self, *replies):
        from jarvis.state import SharedState

        outer = self

        class FakeState(SharedState):
            def set_activity(self, *args, **kwargs):
                pass

        class FakeRuntime:
            events = None

        class FakeSession:
            def __init__(self):
                self.runtime = FakeRuntime()
                self.state = FakeState()
                self.history = [{"role": "user", "content": "neler var"}]

            def assistant(self, **_kwargs):
                return Assistant(outer.brain, outer.workspace)

            def add(self, role, content):
                outer.written.append((role, content))

        self.brain = ScriptedBrain(*replies)
        return FakeSession()

    def test_the_repl_writes_the_record_the_page_writes(self):
        from jarvis.cli.repl import _run_with_tools

        session = self.repl_session(call("fs.list", path="."), answer("bitti"))
        self.assertTrue(_run_with_tools(session, "neler var"))
        self.assertTrue(any("fs.list" in text for text in self.records()),
                        f"araç kaydı yazılmadı: {self.written}")

    def test_the_record_comes_before_the_reply_there_too(self):
        from jarvis.cli.repl import _run_with_tools

        session = self.repl_session(call("fs.list", path="."), answer("bitti"))
        _run_with_tools(session, "neler var")
        self.assertEqual(self.roles(), [TOOL_ROLE, "assistant"])

    def test_a_plain_answer_still_writes_nothing_extra(self):
        from jarvis.cli.repl import _run_with_tools

        session = self.repl_session(answer("Python bir dildir."))
        _run_with_tools(session, "python nedir")
        self.assertEqual(self.roles(), ["assistant"])


class TestTheDistillerCanTellThemApart(unittest.TestCase):
    def test_a_tool_record_is_not_labelled_as_a_jarvis_claim(self):
        """`accept()` refuses what JARVIS asserted. A measurement is not that,
        and presenting it as one teaches the distiller the wrong lesson about
        which parts of a session are believable."""
        from jarvis.memory.distill import _transcript

        text = _transcript([
            {"role": "user", "content": "neler var"},
            {"role": TOOL_ROLE, "content": "fs.list — basarili: a.txt"},
            {"role": "assistant", "content": "iki dosya var"},
        ])
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("Kullanıcı:"))
        self.assertFalse(lines[1].startswith("ZESTOLES:"), lines[1])
        self.assertTrue(lines[2].startswith("ZESTOLES:"))

    def test_the_two_layers_spell_the_role_the_same_way(self):
        """Memory declares it rather than importing it, so nothing else keeps
        them equal. A silent divergence would send tool records back to being
        labelled as JARVIS claims, with every test still passing."""
        import importlib

        # `jarvis.memory` re-exports the function under the module's own
        # name, so plain `import ... as` hands back the function.
        distill_module = importlib.import_module('jarvis.memory.distill')

        self.assertEqual(distill_module.TOOL_ROLE, TOOL_ROLE)

    def test_the_distiller_is_told_what_that_line_is(self):
        import inspect

        import importlib

        # `jarvis.memory` re-exports the function under the module's own
        # name, so plain `import ... as` hands back the function.
        distill_module = importlib.import_module('jarvis.memory.distill')

        self.assertIn("[arac kaydi]", inspect.getsource(distill_module))


if __name__ == "__main__":
    unittest.main()
