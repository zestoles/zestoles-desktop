"""The loop between "do this" and it actually being done.

The thing these tests defend is narrow and important: what the model *says*
never decides whether work happened. Every claim about success is checked
against the recorded `ToolResult`s, because a language model will happily write
"done!" over a failure and no prompt reliably stops it.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import tools  # noqa: E402
from jarvis.assistant import (  # noqa: E402
    REPLY,
    TOOL,
    Assistant,
    Step,
    read_decision,
)
from jarvis.tools import ToolResult, Workspace  # noqa: E402


def call(tool_name, **arguments):
    return json.dumps({"action": TOOL, "tool": tool_name, "arguments": arguments})


def answer(message="tamam"):
    return json.dumps({"action": REPLY, "message": message})


class ScriptedBrain:
    """A model that says exactly what a test needs it to say."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []
        self.local = self
        self.raises: Exception | None = None

    def chat(self, messages, **_kwargs):
        self.seen.append(list(messages))
        if self.raises is not None:
            raise self.raises
        if not self.replies:
            return answer("başka söyleyecek bir şey yok")
        return self.replies.pop(0)

    @property
    def observations(self) -> list[str]:
        """Everything the loop fed back as a tool observation."""
        return [m["content"] for turn in self.seen for m in turn
                if m["role"] == "user" and m["content"].startswith("[gözlem]")]


class Recorder:
    def __init__(self):
        self.rows = []

    def publish(self, source, kind, message, level="info", data=None):
        self.rows.append({"source": source, "kind": kind, "message": message,
                          "level": level, "data": data or {}})

    def kinds(self):
        return [row["kind"] for row in self.rows]


class AssistantCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "alan")
        self.events = Recorder()
        self.approvals: list[tuple] = []

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, *replies, approve=True, **kwargs):
        def approver(tool_name, risk, arguments):
            self.approvals.append((tool_name, risk, arguments))
            return approve

        return Assistant(
            ScriptedBrain(*replies), self.workspace, events=self.events,
            approve=(approver if approve is not None else None), **kwargs)

    def path(self, relative):
        return self.workspace.root / relative


# ------------------------------------------------------------------ pure part
class TestTheSystemPrompt(unittest.TestCase):
    """What the model is told about the difference between knowing and checking.

    Measured, not assumed: with the older wording, four runs of "Python
    surumunu kontrol et" produced "3.10.12" three times with no tool call at
    all. The interpreter is 3.14.6. The rule that invited it said a question of
    knowledge needs no tool, and a version number reads exactly like one.
    After narrowing it, 16/16 machine questions reached for a tool.
    """

    def prompt(self):
        from jarvis.tools import Workspace
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            return Assistant(None, Workspace(Path(tmp) / "a")).system_prompt()

    def test_it_says_the_model_knows_nothing_about_this_machine(self):
        self.assertIn("Bu bilgisayar hakkında hiçbir şey bilmiyorsun", self.prompt())

    def test_the_exception_is_limited_to_questions_not_about_this_machine(self):
        prompt = self.prompt()
        self.assertIn("genel bilgi", prompt.lower())
        self.assertNotIn("Bilgi sorusu için araca gerek yoksa", prompt)

    def test_it_still_forbids_claiming_work_that_did_not_happen(self):
        self.assertIn("Bir araç çağırmadan bir işi yaptığını söyleme", self.prompt())

    def test_live_voice_asks_for_a_short_natural_first_sentence(self):
        from jarvis.tools import Workspace
        with tempfile.TemporaryDirectory() as tmp:
            assistant = Assistant(None, Workspace(Path(tmp) / "a"))
            spoken = assistant.system_prompt(live_voice=True)
            typed = assistant.system_prompt()
        self.assertIn("en fazla sekiz kelime", spoken)
        self.assertIn("ilk cümlede başlık", spoken)
        self.assertNotIn("[Canlı ses modu]", typed)

    def test_prompt_understands_colloquial_turkish_in_context(self):
        prompt = self.prompt()
        self.assertIn("eksiltili Türkçe", prompt)
        self.assertIn("en son düzeltmesi", prompt)
        self.assertIn("kelime kelime değil niyet", prompt)

    def test_prompt_does_not_invent_dates_or_ask_generic_followups(self):
        prompt = self.prompt()
        self.assertIn("Tarih ve saat uydurma", prompt)
        self.assertIn("başka ne yapabilirim", prompt)
        self.assertIn("tek, kısa ve somut soru", prompt)


class TestReadDecision(unittest.TestCase):
    """The one thing standing between generated text and a subprocess."""

    def read(self, raw, available=("fs.read", "shell.run")):
        return read_decision(raw, available=set(available))

    def test_a_valid_tool_call_is_accepted(self):
        decision = self.read(call("fs.read", path="a.txt"))
        self.assertTrue(decision.ok, decision.problems)
        self.assertEqual(decision.tool, "fs.read")
        self.assertEqual(decision.arguments, {"path": "a.txt"})

    def test_a_valid_reply_is_accepted(self):
        decision = self.read(answer("merhaba"))
        self.assertTrue(decision.ok, decision.problems)
        self.assertEqual(decision.kind, REPLY)
        self.assertEqual(decision.message, "merhaba")

    def test_an_unregistered_tool_is_refused(self):
        decision = self.read(call("os.system", command="format c:"))
        self.assertFalse(decision.ok)
        self.assertTrue(any("kayıtlı olmayan" in p for p in decision.problems))

    def test_text_that_is_not_json_is_refused(self):
        self.assertFalse(self.read("Tabii, hemen yapıyorum!").ok)

    def test_a_json_array_is_refused(self):
        self.assertFalse(self.read('["fs.read"]').ok)

    def test_an_unknown_action_is_refused(self):
        self.assertFalse(self.read('{"action": "delete_everything"}').ok)

    def test_arguments_that_are_not_an_object_are_refused(self):
        self.assertFalse(self.read(
            '{"action": "tool", "tool": "fs.read", "arguments": "a.txt"}').ok)

    def test_argument_keys_that_are_not_identifiers_are_refused(self):
        self.assertFalse(self.read(
            '{"action": "tool", "tool": "fs.read", "arguments": {"a b": 1}}').ok)

    def test_missing_arguments_means_no_arguments(self):
        decision = self.read('{"action": "tool", "tool": "fs.read"}')
        self.assertTrue(decision.ok, decision.problems)
        self.assertEqual(decision.arguments, {})

    def test_an_empty_reply_is_refused(self):
        self.assertFalse(self.read('{"action": "reply", "message": "   "}').ok)


# ------------------------------------------------------------------ real work
class TestSingleToolTurn(AssistantCase):
    def test_a_file_the_user_asked_for_is_really_created(self):
        assistant = self.build(
            call("fs.write", path="test.txt", content="Merhaba"),
            answer("test.txt oluşturuldu."))
        turn = assistant.run("masaüstümde test.txt oluştur ve içine Merhaba yaz")

        self.assertTrue(turn.succeeded, turn.summary())
        self.assertEqual(turn.used_tools, ["fs.write"])
        self.assertEqual(self.path("test.txt").read_text(encoding="utf-8"), "Merhaba")

    def test_a_question_that_needs_no_tool_uses_none(self):
        assistant = self.build(answer("Python bir programlama dilidir."))
        turn = assistant.run("python nedir")
        self.assertEqual(turn.steps, [])
        self.assertTrue(turn.succeeded)
        self.assertIn("Python", turn.reply)

    def test_the_real_output_reaches_the_model(self):
        self.path("rapor.txt").parent.mkdir(parents=True, exist_ok=True)
        self.path("rapor.txt").write_text("satır bir\n", encoding="utf-8")
        assistant = self.build(call("fs.read", path="rapor.txt"), answer("okudum"))
        assistant.run("rapor.txt'yi oku")
        self.assertTrue(any("satır bir" in o for o in assistant.brain.observations),
                        assistant.brain.observations)


class TestMultiToolTurn(AssistantCase):
    def test_state_is_kept_across_several_calls(self):
        assistant = self.build(
            call("fs.mkdir", path="proje"),
            call("fs.write", path="proje/not.txt", content="ilk"),
            call("fs.read", path="proje/not.txt"),
            answer("Klasörü oluşturdum, dosyayı yazdım ve içeriğini doğruladım."))
        turn = assistant.run("proje klasörü aç, not.txt yaz, sonra doğrula")

        self.assertTrue(turn.succeeded, turn.summary())
        self.assertEqual(turn.used_tools, ["fs.mkdir", "fs.write", "fs.read"])
        self.assertEqual(self.path("proje/not.txt").read_text(encoding="utf-8"), "ilk")

    def test_a_later_step_sees_what_an_earlier_one_produced(self):
        assistant = self.build(
            call("fs.write", path="a.txt", content="içerik"),
            call("fs.search", pattern="*.txt"),
            answer("bulundu"))
        turn = assistant.run("dosya yaz ve sonra ara")
        self.assertTrue(turn.succeeded, turn.summary())
        self.assertTrue(any("a.txt" in o for o in assistant.brain.observations))


class TestFailureIsNotHidden(AssistantCase):
    """The property that matters most: prose cannot overrule results."""

    def test_a_failing_tool_makes_the_turn_unsuccessful(self):
        assistant = self.build(call("fs.read", path="yok.txt"), answer("okudum"))
        turn = assistant.run("yok.txt'yi oku")
        self.assertFalse(turn.succeeded)
        self.assertEqual(len(turn.failures), 1)

    def test_a_model_claiming_success_does_not_make_it_so(self):
        """This is the anti-fabrication guarantee, and it is structural: the
        reply says it worked, `succeeded` is computed from the results."""
        assistant = self.build(
            call("fs.read", path="hicyok.txt"),
            answer("Dosyayı başarıyla okudum, her şey yolunda!"))
        turn = assistant.run("oku")
        self.assertIn("başarıyla", turn.reply)
        self.assertFalse(turn.succeeded, "model iddiası sonucu ezmemeli")
        self.assertFalse(turn.failures[0].ok)

    def test_the_failure_is_reported_to_the_model_as_a_failure(self):
        assistant = self.build(call("fs.read", path="yok.txt"), answer("peki"))
        assistant.run("oku")
        self.assertTrue(any("BAŞARISIZ" in o for o in assistant.brain.observations),
                        assistant.brain.observations)

    def test_a_command_that_hangs_is_stopped_and_reported(self):
        assistant = self.build(
            call("shell.run",
                 command=f'"{sys.executable}" -c "import time; time.sleep(30)"',
                 timeout_s=2),
            answer("komut bitmedi"))
        turn = assistant.run("uzun komut çalıştır")
        self.assertFalse(turn.succeeded)
        self.assertTrue(turn.failures[0].result.detail.get("timed_out"))

    def test_a_model_that_cannot_be_reached_stops_the_turn_cleanly(self):
        assistant = self.build(answer("x"))
        assistant.brain.raises = OSError("ollama kapalı")
        turn = assistant.run("merhaba")
        self.assertFalse(turn.succeeded)
        self.assertIn("model yanıt vermedi", turn.stopped)
        self.assertEqual(turn.steps, [])


class TestRiskGateIsNotBypassed(AssistantCase):
    def test_a_denied_write_does_not_happen(self):
        assistant = self.build(
            call("fs.write", path="olmaz.txt", content="x"),
            answer("yazamadım"), approve=False)
        turn = assistant.run("dosya yaz")
        self.assertFalse(turn.succeeded)
        self.assertFalse(self.path("olmaz.txt").exists())
        self.assertIn("onaylamadı", turn.failures[0].result.error)

    def test_the_user_is_asked_before_a_write(self):
        self.build(call("fs.write", path="a.txt", content="x"),
                   answer("tamam")).run("yaz")
        self.assertEqual(len(self.approvals), 1)
        self.assertEqual(self.approvals[0][0], "fs.write")
        self.assertEqual(self.approvals[0][1], tools.MEDIUM)

    def test_a_read_is_not_put_to_the_user(self):
        self.path("v.txt").write_text("x", encoding="utf-8")
        self.build(call("fs.read", path="v.txt"), answer("okundu")).run("oku")
        self.assertEqual(self.approvals, [])

    def test_without_a_way_to_ask_the_turn_stops_and_nothing_happens(self):
        assistant = self.build(call("fs.write", path="a.txt", content="x"),
                               approve=None)
        turn = assistant.run("yaz")
        self.assertIsNotNone(turn.pending)
        self.assertEqual(turn.pending.tool, "fs.write")
        self.assertFalse(turn.succeeded)
        self.assertFalse(self.path("a.txt").exists())

    def test_the_pending_call_runs_once_the_user_approves_it(self):
        assistant = self.build(call("fs.write", path="a.txt", content="onaylandi"),
                               approve=None)
        turn = assistant.run("yaz")
        done = assistant.confirm(turn.pending)
        self.assertTrue(done.ok, done.result.error)
        self.assertEqual(self.path("a.txt").read_text(encoding="utf-8"), "onaylandi")

    def test_a_broken_approval_callback_is_a_refusal(self):
        def explode(*_a):
            raise RuntimeError("UI çöktü")

        assistant = Assistant(
            ScriptedBrain(call("fs.write", path="a.txt", content="x"), answer("olmadı")),
            self.workspace, events=self.events, approve=explode)
        turn = assistant.run("yaz")
        self.assertFalse(turn.succeeded)
        self.assertFalse(self.path("a.txt").exists())

    def test_a_catastrophic_command_is_refused_even_if_approved(self):
        assistant = self.build(call("shell.run", command="format c: /q"),
                               answer("yapmadım"))
        turn = assistant.run("diski formatla")
        self.assertFalse(turn.succeeded)
        self.assertTrue(turn.failures[0].result.detail.get("refused"))

    def test_a_write_outside_the_workspace_is_refused(self):
        outside = Path(self._tmp.name).resolve() / "disarida.txt"
        outside.write_text("DOKUNULMADI\n", encoding="utf-8")
        assistant = self.build(call("fs.write", path=str(outside), content="EZILDI"),
                               answer("olmadı"))
        turn = assistant.run("dışarı yaz")
        self.assertFalse(turn.succeeded)
        self.assertEqual(outside.read_text(encoding="utf-8"), "DOKUNULMADI\n")


class TestMalformedModelOutput(AssistantCase):
    def test_an_invented_tool_is_rejected_and_the_turn_continues(self):
        assistant = self.build(
            call("os.system", command="rm -rf /"),
            answer("öyle bir aracım yok"))
        turn = assistant.run("her şeyi sil")
        self.assertEqual(turn.steps, [])
        self.assertTrue(turn.succeeded)
        self.assertIn("decision.rejected", self.events.kinds())

    def test_prose_instead_of_json_is_rejected_and_retried(self):
        assistant = self.build("Tabii efendim, hemen hallediyorum.",
                               answer("tamam"))
        turn = assistant.run("selam")
        self.assertTrue(turn.succeeded)
        self.assertIn("decision.rejected", self.events.kinds())

    def test_the_model_is_told_what_was_wrong(self):
        assistant = self.build("json degil", answer("tamam"))
        assistant.run("selam")
        corrections = [m["content"] for turn in assistant.brain.seen for m in turn
                       if m["role"] == "user" and m["content"].startswith("[sistem]")]
        self.assertTrue(corrections)


class TestLoopLimits(AssistantCase):
    def test_a_repeated_identical_call_stops_the_turn(self):
        repeat = call("fs.list", path=".")
        assistant = self.build(*[repeat] * 6)
        turn = assistant.run("listele")
        self.assertIn("tekrar ediyor", turn.stopped)
        self.assertFalse(turn.succeeded)

    def test_the_step_budget_ends_a_runaway_turn(self):
        self.path("a.txt").write_text("x", encoding="utf-8")
        varying = [call("fs.read", path=f"a.txt", max_bytes=1000 + i) for i in range(10)]
        assistant = self.build(*varying, max_steps=3)
        turn = assistant.run("oku")
        self.assertIn("adım sınırına", turn.stopped)
        self.assertEqual(len(turn.steps), 3)


class TestCancellation(AssistantCase):
    """Work the user stopped must never be reported as done."""

    def cancel_once(self, relative: str):
        """Cancel as soon as a given file exists.

        Deliberately not a call counter: how many times the loop asks is an
        implementation detail, and a test written against it breaks when the
        loop starts asking in one more place — which is exactly what happened.
        """
        return lambda: self.path(relative).exists()

    def test_a_cancelled_turn_is_not_a_success(self):
        assistant = self.build(call("fs.list", path="."), answer("bitti"))
        assistant.should_stop = lambda: True
        turn = assistant.run("listele")
        self.assertTrue(turn.cancelled)
        self.assertFalse(turn.succeeded)
        self.assertEqual(turn.steps, [])

    def test_cancelling_stops_before_the_next_tool_runs(self):
        assistant = self.build(
            call("fs.write", path="bir.txt", content="1"),
            call("fs.write", path="iki.txt", content="2"),
            answer("bitti"))
        assistant.should_stop = self.cancel_once("bir.txt")
        turn = assistant.run("iki dosya yaz")

        self.assertTrue(turn.cancelled)
        self.assertTrue(self.path("bir.txt").exists(), "ilk adim tamamlanmis olmali")
        self.assertFalse(self.path("iki.txt").exists(), "iptalden sonra yazilmamali")

    def test_a_finished_step_is_kept_in_the_record(self):
        """Cancelling does not erase what really happened before it."""
        assistant = self.build(call("fs.write", path="bir.txt", content="1"),
                               answer("bitti"))
        assistant.should_stop = self.cancel_once("bir.txt")
        turn = assistant.run("yaz")
        self.assertEqual(turn.used_tools, ["fs.write"])
        self.assertTrue(turn.steps[0].ok)

    def test_cancellation_is_published(self):
        assistant = self.build(answer("x"))
        assistant.should_stop = lambda: True
        assistant.run("selam")
        self.assertIn("turn.cancelled", self.events.kinds())

    def test_a_broken_cancel_check_does_not_stop_the_turn(self):
        def explode():
            raise RuntimeError("kontrol çöktü")

        assistant = self.build(answer("tamam"))
        assistant.should_stop = explode
        turn = assistant.run("selam")
        self.assertTrue(turn.succeeded, turn.summary())

    def test_no_cancel_check_means_nothing_changes(self):
        assistant = self.build(answer("tamam"))
        self.assertIsNone(assistant.should_stop)
        self.assertTrue(assistant.run("selam").succeeded)


class TestObservability(AssistantCase):
    def test_tool_activity_is_published_for_the_ui(self):
        self.build(call("fs.write", path="a.txt", content="x"),
                   answer("tamam")).run("yaz")
        kinds = self.events.kinds()
        for expected in ("turn.start", "tool.start", "tool.done", "turn.done"):
            self.assertIn(expected, kinds)

    def test_a_failure_is_published_as_a_failure(self):
        self.build(call("fs.read", path="yok.txt"), answer("olmadı")).run("oku")
        self.assertIn("tool.failed", self.events.kinds())

    def test_a_broken_event_log_does_not_break_the_turn(self):
        class Broken:
            def publish(self, *a, **k):
                raise RuntimeError("olay yolu çöktü")

        assistant = Assistant(
            ScriptedBrain(call("fs.write", path="a.txt", content="x"), answer("tamam")),
            self.workspace, events=Broken(), approve=lambda *_: True)
        turn = assistant.run("yaz")
        self.assertTrue(turn.succeeded, turn.summary())


class TestBoundariesUnchanged(unittest.TestCase):
    def test_self_modification_is_still_off(self):
        from jarvis.lab.promotion import ALLOW_SELF_MODIFICATION

        self.assertFalse(ALLOW_SELF_MODIFICATION)

    def test_the_agent_sandbox_gate_is_untouched(self):
        from jarvis.agents.permissions import FS_WRITE, SHELL, Grant

        asked = frozenset({SHELL, FS_WRITE})
        self.assertFalse(Grant.build("gece", asked).capabilities)

    def test_the_loop_never_confirms_on_its_own(self):
        """A grep, because this is the line whose absence would make the whole
        risk tier decorative."""
        import inspect

        import jarvis.assistant as module

        source = inspect.getsource(module.Assistant._perform)
        self.assertNotIn("confirmed=True", source)


if __name__ == "__main__":
    unittest.main()
