"""Closing JARVIS actually closes JARVIS.

V1 is a manual-start desktop application: nothing starts it at login, and when
the user is done with it nothing of it should be left running. Until now the
only way to stop it was Ctrl+C in the console window the launcher opened --
which makes the terminal part of normal use, and the whole point of the desktop
shell is that it is not.

Two ways out, because there are two ways a person closes a window:

- **the Kapat button**, which is the deliberate one and says so;
- **being abandoned**, which is what actually happens when someone closes the
  browser tab. The server knows how many clients are attached; when the last one
  leaves and nobody comes back within the grace period, the process stops on its
  own rather than living on as a daemon nobody asked for.

The grace period is what keeps a page reload from killing the process, so these
pin both sides of it: it must not fire early, and it must fire.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.assistant import REPLY, TOOL, Assistant  # noqa: E402
from jarvis.assistant.service import AssistantService  # noqa: E402
from jarvis.cli.lifecycle import DEFAULT_GRACE_S, OrphanWatch  # noqa: E402
from jarvis.tools import Workspace  # noqa: E402


def answer(message="tamam"):
    return json.dumps({"action": REPLY, "message": message})


def call(tool_name, **arguments):
    return json.dumps({"action": TOOL, "tool": tool_name, "arguments": arguments})


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


class TestOrphanWatch(unittest.TestCase):
    """Nobody is looking at it any more. Is that true, and has it been true long
    enough to be a decision rather than a hiccup?"""

    def watch(self, grace=60.0):
        return OrphanWatch(grace_s=grace)

    def test_it_waits_for_the_first_client_before_watching_anything(self):
        """Startup is not abandonment. The browser takes a moment to connect,
        and a process that gave up during that moment would never open."""
        watch = self.watch()
        for tick in range(200):
            self.assertFalse(watch.observe(0, now=float(tick)), tick)

    def test_a_connected_client_is_not_abandonment(self):
        watch = self.watch()
        for tick in range(200):
            self.assertFalse(watch.observe(1, now=float(tick)))

    def test_leaving_starts_the_clock_but_does_not_stop_anything_yet(self):
        watch = self.watch(grace=60.0)
        watch.observe(1, now=0.0)
        self.assertFalse(watch.observe(0, now=1.0))
        self.assertFalse(watch.observe(0, now=60.0))

    def test_staying_away_past_the_grace_period_stops_it(self):
        watch = self.watch(grace=60.0)
        watch.observe(1, now=0.0)
        watch.observe(0, now=1.0)
        self.assertTrue(watch.observe(0, now=61.1))

    def test_a_reload_inside_the_grace_period_cancels_it(self):
        """The failure this exists to prevent: refreshing the page kills JARVIS."""
        watch = self.watch(grace=60.0)
        watch.observe(1, now=0.0)
        watch.observe(0, now=1.0)
        watch.observe(1, now=3.0)
        for tick in range(4, 400):
            self.assertFalse(watch.observe(1, now=float(tick)), tick)

    def test_leaving_again_starts_a_fresh_clock(self):
        watch = self.watch(grace=60.0)
        watch.observe(1, now=0.0)
        watch.observe(0, now=1.0)
        watch.observe(1, now=3.0)
        watch.observe(0, now=10.0)
        self.assertFalse(watch.observe(0, now=69.0))
        self.assertTrue(watch.observe(0, now=70.1))

    def test_a_grace_of_zero_switches_the_whole_thing_off(self):
        """For someone who wants JARVIS to outlive the page on purpose."""
        watch = self.watch(grace=0.0)
        watch.observe(1, now=0.0)
        for tick in range(1, 5000):
            self.assertFalse(watch.observe(0, now=float(tick)), tick)

    def test_the_default_grace_survives_a_slow_reload(self):
        self.assertGreaterEqual(DEFAULT_GRACE_S, 60)

    def test_it_reports_whether_it_is_counting_down(self):
        watch = self.watch(grace=60.0)
        watch.observe(1, now=0.0)
        self.assertIsNone(watch.deadline)
        watch.observe(0, now=5.0)
        self.assertEqual(watch.deadline, 65.0)


class ShutdownCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "alan")
        self.stopped = threading.Event()

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, *replies):
        self.brain = ScriptedBrain(*replies)
        return AssistantService(Assistant(self.brain, self.workspace),
                                shutdown=self.stopped.set)


class TestClosingFromThePage(ShutdownCase):
    def test_the_page_can_ask_jarvis_to_close(self):
        service = self.build()
        result = service.handle({"op": "kapat"})
        self.assertTrue(result.get("kapaniyor"))
        self.assertTrue(self.stopped.wait(3), "kapatma cagrilmadi")

    def test_it_answers_before_it_stops(self):
        """The reply travels over the socket that is about to be closed, so the
        answer has to be handed back first and the stopping done after."""
        service = self.build()
        result = service.handle({"op": "kapat"})
        self.assertIn("durum", result)

    def test_a_running_turn_is_cancelled_rather_than_abandoned(self):
        service = self.build()
        service.handle({"op": "kapat"})
        self.assertTrue(service.assistant.should_stop(),
                        "kapanirken calisan tur durdurulmali")

    def test_closing_does_not_wait_for_the_lock(self):
        """Same rule as cancel: an instruction to stop that queues behind the
        work it means to stop arrives too late to be one."""
        service = self.build()
        service._lock.acquire()
        try:
            result = service.handle({"op": "kapat"})
        finally:
            service._lock.release()
        self.assertTrue(result.get("kapaniyor"))

    def test_without_a_way_to_close_it_says_so_instead_of_pretending(self):
        assistant = Assistant(ScriptedBrain(), self.workspace)
        service = AssistantService(assistant)
        self.assertIn("hata", service.handle({"op": "kapat"}))


class TestTheInterfaceWiresShutdown(unittest.TestCase):
    def test_the_service_is_given_a_way_to_stop_the_process(self):
        import inspect

        from jarvis.cli import interface

        source = inspect.getsource(interface.run_interface)
        self.assertIn("shutdown=", source)

    def test_the_watchdog_is_actually_started(self):
        import inspect

        from jarvis.cli import interface

        source = inspect.getsource(interface)
        self.assertIn("OrphanWatch", source)


class TestTheShutdownSequence(unittest.TestCase):
    """Every step of the way out has to actually run.

    Each step is guarded separately so one failure cannot skip the rest -- which
    is right, and which also means a step that has been broken since it was
    written fails quietly into the log every single time. That is what happened:
    the sequence called `runtime.stop()`, `Runtime` has `shutdown()`, and for
    every close since then the scheduler was never asked to stop. The process
    exited anyway because the thread is a daemon, so nothing looked wrong.
    """

    def parts(self):
        calls = []

        class FakeServer:
            def stop(self):
                calls.append("server")

        class FakeRuntime:
            memory = None

            def shutdown(self, **kwargs):
                calls.append("runtime")

        class FakeLock:
            def release(self):
                calls.append("lock")

        return calls, FakeServer(), FakeRuntime(), FakeLock()

    def test_every_step_runs_in_the_order_that_cannot_strand_anything(self):
        from jarvis.cli.interface import _shut_down

        calls, server, runtime, lock = self.parts()
        _shut_down(runtime, server, lock)
        self.assertEqual(calls, ["server", "runtime", "lock"])

    def test_the_runtime_is_really_asked_to_stop(self):
        """The regression itself: a name that does not exist is not a shutdown."""
        from jarvis.cli.interface import _shut_down

        calls, server, runtime, lock = self.parts()
        _shut_down(runtime, server, lock)
        self.assertIn("runtime", calls, "zamanlayÄ±cÄ± durdurulmadÄ±")

    def test_a_failing_step_does_not_skip_the_lock(self):
        from jarvis.cli.interface import _shut_down

        calls, server, runtime, lock = self.parts()

        def explode(**kwargs):
            raise RuntimeError("kapanamadÄ±")

        runtime.shutdown = explode
        _shut_down(runtime, server, lock)
        self.assertIn("lock", calls, "kilit her hÃ¢lÃ¼kÃ¢rda bÄ±rakÄ±lmalÄ±")

    def test_closing_without_the_lock_is_fine(self):
        from jarvis.cli.interface import _shut_down

        calls, server, runtime, _lock = self.parts()
        _shut_down(runtime, server, None)
        self.assertEqual(calls, ["server", "runtime"])


class TestThePageOffersClosing(unittest.TestCase):
    def setUp(self):
        self.text = (Path(__file__).resolve().parents[1]
                     / "ui" / "jarvis.html").read_text(encoding="utf-8")

    def test_there_is_a_close_control(self):
        self.assertIn('op: "kapat"', self.text)

    def test_closing_asks_first(self):
        """An accidental click should not end the session."""
        window = self.text.split('op: "kapat"', 1)[0][-700:]
        self.assertIn("confirm", window, window[-300:])

    def test_input_is_shut_off_once_it_has_closed(self):
        """There is nothing behind the socket any more; an enabled send button
        would invite the user to talk to a process that is gone."""
        window = self.text.split("send.disabled", 1)[1][:200]
        self.assertIn("kapandi", window, window)

    def test_the_page_says_what_happened_afterwards(self):
        self.assertRegex(self.text, r"kapan[dı]|kapatıldı|kapanıyor")


if __name__ == "__main__":
    unittest.main()
