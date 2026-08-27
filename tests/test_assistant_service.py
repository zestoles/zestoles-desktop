"""One request in, one answer out — and what happens when the answer is a question.

The interface cannot block a turn on a confirmation that arrives over a second
connection, so the service keeps the pending call instead. These pin that
handover, the "one turn at a time" rule, and the property everything else here
exists to protect: what the interface is told about success comes from the
recorded tool results, never from the model's prose.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.assistant import REPLY, TOOL, Assistant  # noqa: E402
from jarvis.assistant.service import (  # noqa: E402
    IDLE,
    WAITING,
    WORKING,
    AssistantService,
)
from jarvis.tools import Workspace  # noqa: E402


def call(tool_name, **arguments):
    return json.dumps({"action": TOOL, "tool": tool_name, "arguments": arguments})


def answer(message="tamam"):
    return json.dumps({"action": REPLY, "message": message})


class ScriptedBrain:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []
        self.local = self
        self.delay = 0.0

    def chat(self, messages, **_kwargs):
        self.seen.append(list(messages))
        if self.delay:
            time.sleep(self.delay)
        if not self.replies:
            return answer("başka söyleyecek bir şey yok")
        return self.replies.pop(0)


class ServiceCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "alan")

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, *replies):
        assistant = Assistant(ScriptedBrain(*replies), self.workspace)
        return AssistantService(assistant)

    def path(self, relative):
        return self.workspace.root / relative


class TestPlainConversation(ServiceCase):
    def test_a_question_gets_an_answer(self):
        service = self.build(answer("Python bir dildir."))
        result = service.handle({"op": "sor", "mesaj": "python nedir"})
        self.assertEqual(result["durum"], IDLE)
        self.assertTrue(result["basarili"])
        self.assertIn("Python", result["cevap"])

    def test_an_empty_message_is_refused(self):
        service = self.build(answer("x"))
        self.assertIn("hata", service.handle({"op": "sor", "mesaj": "   "}))

    def test_an_unknown_operation_is_refused(self):
        service = self.build(answer("x"))
        self.assertIn("hata", service.handle({"op": "ucmak"}))

    def test_history_is_kept_between_turns(self):
        service = self.build(answer("bir"), answer("iki"))
        service.handle({"op": "sor", "mesaj": "ilk"})
        service.handle({"op": "sor", "mesaj": "ikinci"})
        self.assertEqual(len(service.history), 4)
        second = service.assistant.brain.seen[-1]
        self.assertTrue(any("ilk" in m["content"] for m in second))

    def test_history_does_not_grow_without_end(self):
        service = self.build(*[answer(f"c{i}") for i in range(20)])
        service.history_limit = 4
        for i in range(6):
            service.handle({"op": "sor", "mesaj": f"m{i}"})
        self.assertLessEqual(len(service.history), 4)


class TestConfirmationHandover(ServiceCase):
    """The loop is given no approve callback on purpose. It stops, hands back
    the pending call, and the page answers over a second request."""

    def ask_to_write(self):
        service = self.build(
            call("fs.write", path="rapor.txt", content="içerik"),
            answer("rapor.txt yazıldı."))
        return service, service.handle({"op": "sor", "mesaj": "rapor.txt yaz"})

    def test_a_write_comes_back_as_a_question(self):
        service, result = self.ask_to_write()
        self.assertEqual(result["durum"], WAITING)
        self.assertEqual(result["bekleyen"]["arac"], "fs.write")
        self.assertFalse(result["basarili"])
        self.assertFalse(self.path("rapor.txt").exists(), "onaysiz yazilmis")

    def test_the_arguments_are_shown_so_the_user_can_read_them(self):
        _, result = self.ask_to_write()
        self.assertEqual(result["bekleyen"]["argumanlar"]["path"], "rapor.txt")

    def test_approving_runs_it_and_answers_from_the_result(self):
        service, _ = self.ask_to_write()
        result = service.handle({"op": "onay", "evet": True})
        self.assertEqual(result["durum"], IDLE)
        self.assertEqual(self.path("rapor.txt").read_text(encoding="utf-8"), "içerik")
        self.assertEqual(result["adimlar"][0]["arac"], "fs.write")
        self.assertTrue(result["adimlar"][0]["ok"])

    def test_the_model_is_shown_what_actually_happened(self):
        service, _ = self.ask_to_write()
        service.handle({"op": "onay", "evet": True})
        last = service.assistant.brain.seen[-1]
        self.assertTrue(any("gözlem" in m["content"] for m in last), last)

    def test_refusing_does_nothing_and_says_so(self):
        service, _ = self.ask_to_write()
        result = service.handle({"op": "onay", "evet": False})
        self.assertFalse(result["basarili"])
        self.assertTrue(result["reddedildi"])
        self.assertFalse(self.path("rapor.txt").exists())

    def test_a_confirmation_with_nothing_waiting_is_refused(self):
        service = self.build(answer("x"))
        self.assertIn("hata", service.handle({"op": "onay", "evet": True}))

    def test_a_new_message_while_waiting_is_refused(self):
        service, _ = self.ask_to_write()
        result = service.handle({"op": "sor", "mesaj": "başka bir şey"})
        self.assertEqual(result["durum"], WAITING)
        self.assertIn("hata", result)

    def test_the_pending_call_is_cleared_after_an_answer(self):
        service, _ = self.ask_to_write()
        service.handle({"op": "onay", "evet": False})
        self.assertIsNone(service.pending)
        self.assertEqual(service.status()["durum"], IDLE)


class TestSuccessComesFromResults(ServiceCase):
    def test_a_model_claiming_success_over_a_failure_is_not_believed(self):
        service = self.build(
            call("fs.read", path="hicyok.txt"),
            answer("Dosyayı başarıyla okudum!"))
        result = service.handle({"op": "sor", "mesaj": "oku"})
        self.assertIn("başarıyla", result["cevap"])
        self.assertFalse(result["basarili"])
        self.assertEqual(result["basarisiz"], ["fs.read"])

    def test_each_step_carries_its_own_verdict(self):
        service = self.build(call("fs.read", path="yok.txt"), answer("olmadı"))
        result = service.handle({"op": "sor", "mesaj": "oku"})
        self.assertFalse(result["adimlar"][0]["ok"])


class TestOneTurnAtATime(ServiceCase):
    def test_a_second_request_mid_turn_is_told_it_is_busy(self):
        service = self.build(answer("bitti"))
        service.assistant.brain.delay = 0.6

        answers = {}

        def first():
            answers["first"] = service.handle({"op": "sor", "mesaj": "uzun"})

        thread = threading.Thread(target=first)
        thread.start()
        time.sleep(0.2)
        second = service.handle({"op": "sor", "mesaj": "araya girdi"})
        thread.join(timeout=10)

        self.assertEqual(second["durum"], WORKING)
        self.assertIn("hata", second)
        self.assertTrue(answers["first"]["basarili"])


class TestCancelling(ServiceCase):
    def test_cancel_is_answered_without_waiting_for_the_lock(self):
        """A cancel that queued behind the running turn would arrive after the
        work it meant to stop."""
        service = self.build(answer("bitti"))
        service.assistant.brain.delay = 0.6

        thread = threading.Thread(
            target=lambda: service.handle({"op": "sor", "mesaj": "uzun"}))
        thread.start()
        time.sleep(0.2)

        started = time.monotonic()
        result = service.handle({"op": "iptal"})
        elapsed = time.monotonic() - started
        thread.join(timeout=10)

        self.assertTrue(result["iptal"])
        self.assertLess(elapsed, 0.3, "iptal kilidi beklemis")

    def test_cancelling_mid_turn_stops_the_next_step(self):
        """Cancelling has to reach a turn that is already running — that is the
        only moment it is worth anything."""
        # Both LOW, so neither stops for confirmation and the loop keeps going
        # long enough for a cancel to land between them.
        service = self.build(
            call("fs.list", path="."),
            call("fs.search", pattern="*"),
            answer("bitti"))
        service.assistant.brain.delay = 0.4

        answers = {}
        thread = threading.Thread(
            target=lambda: answers.update(
                turn=service.handle({"op": "sor", "mesaj": "listele ve ara"})))
        thread.start()
        time.sleep(0.6)
        service.handle({"op": "iptal"})
        thread.join(timeout=10)

        turn = answers["turn"]
        self.assertTrue(turn["iptal"], turn)
        self.assertFalse(turn["basarili"])
        self.assertLess(len(turn["adimlar"]), 2, "iptalden sonra adim kosmus")

    def test_a_new_message_clears_the_previous_cancel(self):
        service = self.build(answer("bir"), answer("iki"))
        service.handle({"op": "iptal"})
        service.handle({"op": "sor", "mesaj": "ilk"})
        result = service.handle({"op": "sor", "mesaj": "ikinci"})
        self.assertTrue(result["basarili"], result)


class TestStatus(ServiceCase):
    def test_an_idle_service_says_so(self):
        self.assertEqual(self.build(answer("x")).status()["durum"], IDLE)

    def test_status_needs_no_lock(self):
        service = self.build(answer("bitti"))
        service.assistant.brain.delay = 0.5
        thread = threading.Thread(
            target=lambda: service.handle({"op": "sor", "mesaj": "uzun"}))
        thread.start()
        time.sleep(0.15)
        self.assertEqual(service.handle({"op": "durum"})["durum"], WORKING)
        thread.join(timeout=10)

    def test_reset_clears_everything(self):
        service = self.build(answer("bir"))
        service.handle({"op": "sor", "mesaj": "x"})
        service.reset()
        self.assertEqual(service.history, [])
        self.assertIsNone(service.pending)


class TestTheLoopIsNotGivenAnApprover(ServiceCase):
    def test_the_service_leaves_approve_unset(self):
        """Setting it would deadlock: the callback needs a second request, and
        that request needs the lock the turn is holding."""
        service = self.build(answer("x"))
        self.assertIsNone(service.assistant.approve)

    def test_the_service_owns_the_cancel_check(self):
        service = self.build(answer("x"))
        self.assertIsNotNone(service.assistant.should_stop)


if __name__ == "__main__":
    unittest.main()


class TestTurnsReachMemory(ServiceCase):
    """The terminal writes each turn through `Session.add`. A second front end
    that quietly did not would come up blank next session."""

    def build_with_memory(self, *replies):
        service = self.build(*replies)
        self.written = []
        service.remember = lambda role, content: self.written.append((role, content))
        return service

    def test_both_sides_of_a_turn_are_written(self):
        service = self.build_with_memory(answer("merhaba"))
        service.handle({"op": "sor", "mesaj": "selam"})
        self.assertEqual(self.written, [("user", "selam"), ("assistant", "merhaba")])

    def test_a_confirmed_turn_is_written(self):
        service = self.build_with_memory(
            call("fs.write", path="a.txt", content="x"), answer("yazdım"))
        service.handle({"op": "sor", "mesaj": "yaz"})
        self.written.clear()
        service.handle({"op": "onay", "evet": True})
        self.assertIn(("user", "yaz"), self.written)

    def test_a_refusal_is_written_too(self):
        """What was asked and refused is part of the conversation."""
        service = self.build_with_memory(
            call("fs.write", path="a.txt", content="x"), answer("olmadı"))
        service.handle({"op": "sor", "mesaj": "yaz"})
        self.written.clear()
        service.handle({"op": "onay", "evet": False})
        self.assertTrue(any(role == "user" for role, _ in self.written))

    def test_a_broken_memory_does_not_cost_the_reply(self):
        def explode(role, content):
            raise RuntimeError("hafıza çöktü")

        service = self.build(answer("cevap"))
        service.remember = explode
        result = service.handle({"op": "sor", "mesaj": "selam"})
        self.assertTrue(result["basarili"], result)
        self.assertEqual(result["cevap"], "cevap")

    def test_no_callback_is_fine(self):
        service = self.build(answer("cevap"))
        self.assertIsNone(service.remember)
        self.assertTrue(service.handle({"op": "sor", "mesaj": "selam"})["basarili"])

    def test_the_interface_wires_it(self):
        import inspect

        from jarvis.cli import interface

        self.assertIn("remember=session.add",
                      inspect.getsource(interface.run_interface))
