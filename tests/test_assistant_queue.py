"""Running a turn from the task queue instead of from a held connection.

A request that takes minutes cannot be answered by an HTTP handler that has to
keep the socket open for it, so the same turn has to be runnable in the
background. These pin what that costs and what it must never buy:

- the queue can drive a turn, and the honest record still comes from the steps;
- **only the user can queue one**. A runner reachable from the queue is a runner
  an autonomous task could queue, and a background task that could put words in
  the user's mouth would hand the agent side the tool layer that was deliberately
  kept away from it (see `test_tools.TestSeparationFromTheAgentGate`);
- nothing above LOW risk runs while nobody is there to approve it;
- one turn at a time still holds, across both front doors.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.assistant import REPLY, TOOL, Assistant  # noqa: E402
from jarvis.assistant.background import (  # noqa: E402
    KIND,
    USER_ORIGIN,
    enqueue,
)
from jarvis.assistant.service import AssistantService  # noqa: E402
from jarvis.autonomy import runners  # noqa: E402
from jarvis.autonomy.events import EventLog  # noqa: E402
from jarvis.autonomy.runners import RunContext  # noqa: E402
from jarvis.autonomy.tasks import Priority, Task  # noqa: E402
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

    def chat(self, messages, **_kwargs):
        self.seen.append(list(messages))
        if not self.replies:
            return answer("başka söyleyecek bir şey yok")
        return self.replies.pop(0)


class QueueCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.workspace = Workspace(self.root / "alan")
        self.events = EventLog(self.root / "olaylar.db")
        self.written = []

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, *replies):
        self.brain = ScriptedBrain(*replies)
        assistant = Assistant(self.brain, self.workspace)
        return AssistantService(
            assistant, remember=lambda role, text: self.written.append((role, text)))

    def context(self, service, *, message="dosyaları listele", origin=USER_ORIGIN,
                should_stop=None):
        task = Task(id=1, kind=KIND, title=message[:40], origin=origin,
                    payload={"mesaj": message})
        return RunContext(task=task, events=self.events,
                          should_stop=should_stop or (lambda: False),
                          assistant=service)

    def run_task(self, service, **kwargs):
        handler = runners.get(KIND)
        self.assertIsNotNone(handler, "assistant.ask runner kaydi yok")
        return handler(self.context(service, **kwargs))


class TestTheRunnerExists(QueueCase):
    def test_the_queue_can_reach_a_turn_by_name(self):
        self.assertIn(KIND, runners.names())

    def test_a_queued_request_runs_a_real_turn(self):
        service = self.build(answer("Python bir dildir."))
        result = self.run_task(service, message="python nedir")
        self.assertIn("Python bir dildir.", result)

    def test_the_conversation_survives_the_turn(self):
        service = self.build(answer("oldu"))
        self.run_task(service, message="bir şey yap")
        roles = [role for role, _ in self.written]
        self.assertEqual(roles, ["user", "assistant"])

    def test_an_empty_request_is_refused_before_the_model_is_asked(self):
        service = self.build(answer("olmaz"))
        with self.assertRaises(ValueError):
            self.run_task(service, message="   ")
        self.assertEqual(self.brain.seen, [])

    def test_without_an_assistant_the_task_fails_loudly(self):
        handler = runners.get(KIND)
        context = self.context(None)
        context.assistant = None
        with self.assertRaises(RuntimeError):
            handler(context)


class TestOnlyTheUserCanQueueOne(QueueCase):
    """The gate. An autonomous task must not be able to drive the tool layer."""

    def test_a_task_the_user_did_not_ask_for_is_refused(self):
        service = self.build(answer("çalışmamalıydım"))
        with self.assertRaises(PermissionError):
            self.run_task(service, origin="auto")

    def test_the_refused_task_never_reached_the_model(self):
        service = self.build(answer("çalışmamalıydım"))
        with self.assertRaises(PermissionError):
            self.run_task(service, origin="auto")
        self.assertEqual(self.brain.seen, [])
        self.assertEqual(self.written, [])

    def test_every_origin_but_the_user_is_refused(self):
        for origin in ("auto", "agent", "routine", "", "USER "):
            with self.subTest(origin=origin):
                service = self.build(answer("çalışmamalıydım"))
                with self.assertRaises(PermissionError):
                    self.run_task(service, origin=origin)


class TestNobodyIsThereToApprove(QueueCase):
    def test_a_write_is_not_performed_unattended(self):
        service = self.build(call("fs.write", path="not.txt", content="merhaba"))
        result = self.run_task(service, message="not yaz")
        self.assertFalse((self.workspace.root / "not.txt").exists())
        self.assertIn("onay", result.lower())

    def test_the_queued_turn_does_not_steal_the_live_confirmation_slot(self):
        service = self.build(call("fs.write", path="not.txt", content="merhaba"))
        self.run_task(service, message="not yaz")
        self.assertIsNone(service.pending)


class TestOneTurnAtATime(QueueCase):
    def test_a_queued_turn_waits_for_a_live_one(self):
        service = self.build(answer("olmaz"))
        service._lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.run_task(service)
        finally:
            service._lock.release()
        self.assertEqual(self.brain.seen, [])

    def test_a_pending_confirmation_blocks_a_queued_turn(self):
        service = self.build(call("fs.write", path="not.txt", content="x"),
                             answer("tamam"))
        service.handle({"op": "sor", "mesaj": "not yaz"})
        self.assertIsNotNone(service.pending)
        with self.assertRaises(RuntimeError):
            self.run_task(service)

    def test_the_lock_is_given_back_when_a_turn_raises(self):
        class Exploding:
            def __init__(self):
                self.local = self

            def chat(self, messages, **kwargs):
                raise MemoryError("model patladı")

        service = self.build()
        service.assistant.brain = Exploding()
        with self.assertRaises(MemoryError):
            self.run_task(service)
        self.assertFalse(service._lock.locked())


class TestCancellation(QueueCase):
    def test_the_scheduler_stopping_stops_the_turn(self):
        service = self.build(answer("çalışmamalıydım"))
        result = self.run_task(service, should_stop=lambda: True)
        self.assertIn("iptal", result.lower())
        self.assertEqual(self.brain.seen, [])

    def test_the_live_cancel_still_works_after_a_queued_turn(self):
        service = self.build(answer("tamam"), answer("tamam"))
        self.run_task(service, should_stop=lambda: False)
        service.handle({"op": "iptal"})
        self.assertTrue(service.assistant.should_stop())


class TestTheRunnerIsRegisteredInTheProduct(unittest.TestCase):
    """A registration that only happens when a test imports it is not one.

    Runners register on import. These tests import `background` directly, so
    they would keep passing on a build where nothing else ever does and the
    queue answers `assistant.ask` with "bilinmeyen gorev turu". Asked in a fresh
    interpreter, without that import, so the answer is about the product.
    """

    def test_importing_the_assistant_is_enough(self):
        import subprocess

        script = ("import jarvis.assistant; from jarvis.autonomy import runners; print('assistant.ask' in runners.names())")
        done = subprocess.run([sys.executable, "-c", script],
                              cwd=str(Path(__file__).resolve().parents[1]),
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertEqual(done.stdout.strip(), "True", done.stderr[-800:])

    def test_the_autonomy_layer_still_does_not_import_the_assistant(self):
        """The one-way graph. Registration happens from above, never below."""
        import inspect

        from jarvis import autonomy
        from jarvis.autonomy import runners as runners_module
        from jarvis.autonomy import scheduler as scheduler_module

        for module in (autonomy, runners_module, scheduler_module):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("from ..assistant", source)
                self.assertNotIn("import assistant", source)


class TestQueueing(QueueCase):
    """What `enqueue` writes and what the gate reads must be the same thing."""

    def queue(self):
        from jarvis.autonomy.tasks import TaskQueue

        return TaskQueue(self.root / "kuyruk.db")

    def test_a_queued_request_is_stored_as_user_work(self):
        queue = self.queue()
        task_id = enqueue(queue, "uzun bir is yap")
        task = queue.get(task_id)
        self.assertEqual(task.kind, KIND)
        self.assertEqual(task.origin, USER_ORIGIN)
        self.assertEqual(task.priority, Priority.USER)
        self.assertEqual(task.payload["mesaj"], "uzun bir is yap")

    def test_a_queued_turn_gets_more_than_three_tries(self):
        queue = self.queue()
        task = queue.get(enqueue(queue, "bir sey"))
        self.assertGreater(task.max_attempts, 3)

    def test_an_empty_request_is_never_queued(self):
        queue = self.queue()
        with self.assertRaises(ValueError):
            enqueue(queue, "   ")
        self.assertEqual(queue.counts(), {})

    def test_a_long_request_still_has_a_readable_title(self):
        queue = self.queue()
        task = queue.get(enqueue(queue, "a" * 200))
        self.assertLessEqual(len(task.title), 60)

    def test_what_the_queue_hands_back_passes_the_gate(self):
        """The end to end that a mismatched origin string would break silently."""
        queue = self.queue()
        enqueue(queue, "python nedir")
        claimed = queue.claim(max_priority=Priority.USER)
        self.assertIsNotNone(claimed)

        service = self.build(answer("Python bir dildir."))
        context = RunContext(task=claimed, events=self.events,
                             should_stop=lambda: False, assistant=service)
        result = runners.get(KIND)(context)
        self.assertIn("Python bir dildir.", result)


class ServiceDoorCase(QueueCase):
    def build_with_queue(self, *replies):
        from jarvis.autonomy.tasks import TaskQueue

        self.queue = TaskQueue(self.root / "kuyruk.db")
        self.nudged = []
        self.brain = ScriptedBrain(*replies)
        assistant = Assistant(self.brain, self.workspace)
        return AssistantService(assistant, queue=self.queue,
                                nudge=lambda: self.nudged.append(1))


class TestQueueingThroughTheService(ServiceDoorCase):
    def test_a_request_can_be_queued_instead_of_waited_for(self):
        service = self.build_with_queue()
        result = service.handle({"op": "kuyruk", "mesaj": "uzun is"})
        self.assertIn("gorev", result)
        task = self.queue.get(result["gorev"])
        self.assertEqual(task.origin, USER_ORIGIN)
        self.assertEqual(self.brain.seen, [], "kuyruklamak turu baslatmamali")

    def test_the_scheduler_is_woken_rather_than_waited_on(self):
        service = self.build_with_queue()
        service.handle({"op": "kuyruk", "mesaj": "uzun is"})
        self.assertEqual(len(self.nudged), 1)

    def test_work_can_be_queued_while_a_turn_is_running(self):
        """The point of the door: being busy is when you most want it."""
        service = self.build_with_queue()
        service._lock.acquire()
        try:
            result = service.handle({"op": "kuyruk", "mesaj": "sonra yap"})
        finally:
            service._lock.release()
        self.assertIn("gorev", result)

    def test_an_empty_request_is_refused(self):
        service = self.build_with_queue()
        self.assertIn("hata", service.handle({"op": "kuyruk", "mesaj": "  "}))

    def test_without_a_queue_the_door_says_so(self):
        service = self.build(answer("tamam"))
        self.assertIn("hata", service.handle({"op": "kuyruk", "mesaj": "is"}))


class TestReadingTheQueue(ServiceDoorCase):
    def test_queued_work_can_be_listed(self):
        service = self.build_with_queue()
        service.handle({"op": "kuyruk", "mesaj": "birinci is"})
        listed = service.handle({"op": "gorevler"})["gorevler"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["baslik"], "birinci is")
        self.assertEqual(listed[0]["durum"], "pending")

    def test_the_machines_own_housekeeping_is_not_listed_as_mine(self):
        service = self.build_with_queue()
        self.queue.add("memory.reindex", "hafiza indeksi")
        service.handle({"op": "kuyruk", "mesaj": "benim isim"})
        listed = service.handle({"op": "gorevler"})["gorevler"]
        self.assertEqual([task["baslik"] for task in listed], ["benim isim"])

    def test_the_list_answers_while_a_turn_is_running(self):
        service = self.build_with_queue()
        service.handle({"op": "kuyruk", "mesaj": "is"})
        service._lock.acquire()
        try:
            listed = service.handle({"op": "gorevler"})
        finally:
            service._lock.release()
        self.assertEqual(len(listed["gorevler"]), 1)

    def test_a_finished_task_reports_what_actually_happened(self):
        service = self.build_with_queue(answer("bitti"))
        task_id = service.handle({"op": "kuyruk", "mesaj": "is"})["gorev"]
        claimed = self.queue.claim(max_priority=Priority.USER)
        result = runners.get(KIND)(RunContext(
            task=claimed, events=self.events, should_stop=lambda: False,
            assistant=service))
        self.queue.complete(task_id, result)
        listed = service.handle({"op": "gorevler"})["gorevler"]
        self.assertEqual(listed[0]["durum"], "done")
        self.assertIn("bitti", listed[0]["sonuc"])


class TestCancellingQueuedWork(ServiceDoorCase):
    def test_a_waiting_task_can_be_cancelled(self):
        service = self.build_with_queue()
        task_id = service.handle({"op": "kuyruk", "mesaj": "vazgectim"})["gorev"]
        service.handle({"op": "gorev_iptal", "id": task_id})
        self.assertEqual(self.queue.get(task_id).state, "cancelled")

    def test_cancelling_a_waiting_task_does_not_stop_the_live_turn(self):
        """A queued item the user changed their mind about is not a cancel."""
        service = self.build_with_queue()
        task_id = service.handle({"op": "kuyruk", "mesaj": "vazgectim"})["gorev"]
        service.handle({"op": "gorev_iptal", "id": task_id})
        self.assertFalse(service._cancel.is_set())

    def test_cancelling_the_running_task_stops_the_turn_too(self):
        service = self.build_with_queue()
        task_id = service.handle({"op": "kuyruk", "mesaj": "calisan is"})["gorev"]
        self.queue.claim(max_priority=Priority.USER)
        service.handle({"op": "gorev_iptal", "id": task_id})
        self.assertTrue(service._cancel.is_set())

    def test_cancelling_something_that_is_not_there_is_not_a_crash(self):
        service = self.build_with_queue()
        self.assertIn("hata", service.handle({"op": "gorev_iptal", "id": 9999}))


class TestTheInterfaceWiresItUp(QueueCase):
    """The queue door opens only in the process that actually runs the loop."""

    def runtime(self):
        from jarvis.autonomy.tasks import TaskQueue

        class FakeScheduler:
            def __init__(self):
                self.assistant = None
                self.woken = 0

            def nudge(self):
                self.woken += 1

        class FakeCore:
            def __init__(self, queue):
                self.queue = queue
                self.scheduler = FakeScheduler()

        class FakeRuntime:
            config = None

            def __init__(self, core):
                self.core = core

        queue = TaskQueue(self.root / "kuyruk.db")
        return FakeRuntime(FakeCore(queue))

    def build_service(self, runtime, *, queued):
        from jarvis.cli.interface import build_service

        assistant = Assistant(ScriptedBrain(), self.workspace)
        return build_service(runtime, assistant, remember=None, queued=queued)

    def test_the_owning_process_gets_the_queue(self):
        runtime = self.runtime()
        service = self.build_service(runtime, queued=True)
        self.assertIs(service.queue, runtime.core.queue)

    def test_the_scheduler_is_given_the_same_service_the_page_talks_to(self):
        runtime = self.runtime()
        service = self.build_service(runtime, queued=True)
        self.assertIs(runtime.core.scheduler.assistant, service)

    def test_a_second_window_is_told_the_door_is_shut(self):
        runtime = self.runtime()
        service = self.build_service(runtime, queued=False)
        self.assertIsNone(service.queue)
        self.assertIsNone(runtime.core.scheduler.assistant)
        self.assertIn("hata", service.handle({"op": "kuyruk", "mesaj": "is"}))

    def test_autonomy_switched_off_is_not_a_crash(self):
        class NoCore:
            core = None
            config = None

        service = self.build_service(NoCore(), queued=True)
        self.assertIsNone(service.queue)

    def test_queueing_wakes_that_scheduler(self):
        runtime = self.runtime()
        service = self.build_service(runtime, queued=True)
        service.handle({"op": "kuyruk", "mesaj": "is"})
        self.assertEqual(runtime.core.scheduler.woken, 1)


class TestEndToEndThroughTheScheduler(QueueCase):
    """The whole chain, with the real scheduler doing the claiming.

    Everything above tests a piece. This is the one that would have caught a
    scheduler that never passed the assistant along, or a priority the policy
    refuses to run while the machine is busy -- which is exactly when a user
    leaves work behind.
    """

    def test_a_queued_request_actually_runs(self):
        import time

        from jarvis.autonomy.policy import Policy
        from jarvis.autonomy.scheduler import Scheduler
        from jarvis.autonomy.tasks import State, TaskQueue

        queue = TaskQueue(self.root / "kuyruk.db")
        self.brain = ScriptedBrain(answer("Python bir dildir."))
        service = AssistantService(Assistant(self.brain, self.workspace), queue=queue)
        scheduler = Scheduler(queue, self.events, Policy(), assistant=service,
                              tick_s=0.05)
        service.nudge = scheduler.nudge

        scheduler.start()
        try:
            task_id = service.handle({"op": "kuyruk", "mesaj": "python nedir"})["gorev"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                task = queue.get(task_id)
                if task.state != State.PENDING and task.state != State.RUNNING:
                    break
                time.sleep(0.05)
        finally:
            scheduler.stop(timeout=10)

        self.assertEqual(task.state, State.DONE, task.error or task.result)
        self.assertIn("Python bir dildir.", task.result)

    def test_the_answer_reaches_the_conversation(self):
        import time

        from jarvis.autonomy.policy import Policy
        from jarvis.autonomy.scheduler import Scheduler
        from jarvis.autonomy.tasks import State, TaskQueue

        queue = TaskQueue(self.root / "kuyruk.db")
        self.brain = ScriptedBrain(answer("bitti"))
        service = AssistantService(
            Assistant(self.brain, self.workspace), queue=queue,
            remember=lambda role, text: self.written.append((role, text)))
        scheduler = Scheduler(queue, self.events, Policy(), assistant=service,
                              tick_s=0.05)
        service.nudge = scheduler.nudge

        scheduler.start()
        try:
            task_id = service.handle({"op": "kuyruk", "mesaj": "bir sey yap"})["gorev"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if queue.get(task_id).state == State.DONE:
                    break
                time.sleep(0.05)
        finally:
            scheduler.stop(timeout=10)

        self.assertEqual([role for role, _ in self.written], ["user", "assistant"])


class TestThePageOffersIt(unittest.TestCase):
    """A feature nothing on the page can reach is not shipped."""

    def setUp(self):
        self.text = (Path(__file__).resolve().parents[1]
                     / "ui" / "jarvis.html").read_text(encoding="utf-8")

    def test_work_can_be_left_for_later(self):
        self.assertIn('op: "kuyruk"', self.text)

    def test_the_queue_can_be_read(self):
        self.assertIn('op: "gorevler"', self.text)

    def test_a_queued_task_can_be_dropped(self):
        self.assertIn('op: "gorev_iptal"', self.text)

    def test_the_list_updates_from_the_event_stream(self):
        for wire in ("task_started", "task_finished"):
            self.assertIn(wire, self.text, wire)

    def test_the_state_shown_is_the_state_reported(self):
        """No invented progress: the label comes from the queue's own word."""
        self.assertIn("TASK_STATE[task.durum] || task.durum", self.text)

    def test_every_state_the_queue_can_report_has_a_label(self):
        """The same rule the panel's wire table follows, one table further out.

        A state with no entry falls back to the raw English word in the middle
        of a Turkish page -- which is honest, and still looks like a bug.
        """
        from jarvis.autonomy.tasks import State

        labels = self.text.split("var TASK_STATE = {", 1)[1].split("};", 1)[0]
        for name in dir(State):
            value = getattr(State, name)
            if name.isupper() and isinstance(value, str):
                with self.subTest(state=value):
                    self.assertIn(f"{value}:", labels)


class TestTheContextCarriesTheAssistant(unittest.TestCase):
    def test_a_run_context_can_hold_one(self):
        self.assertIn("assistant", RunContext.__slots__)

    def test_the_scheduler_hands_it_to_the_runner(self):
        import inspect

        from jarvis.autonomy.scheduler import Scheduler

        self.assertIn("assistant", inspect.signature(Scheduler.__init__).parameters)
        self.assertIn("assistant=self.assistant", inspect.getsource(Scheduler._execute))


if __name__ == "__main__":
    unittest.main()
