"""Autonomy tests.

Two things are load-bearing and get the most attention here:

  The policy. It is the only thing standing between "helpful background work" and
  "the machine stutters while the user is playing a game". Its failure mode is silent,
  so every stance is pinned to a case.

  Crash recovery. JARVIS is meant to run for months and will be killed mid-task.
  A queue that loses or infinitely retries its work is worse than no queue.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.autonomy.events import EventLog  # noqa: E402
from jarvis.autonomy.policy import (  # noqa: E402
    ACTIVE, BUSY, CPU, IDLE, NIGHT, QUIET, USER_ACTIVE, Policy, Stance,
)
from jarvis.autonomy.resources import CPU_MIN_INTERVAL_S, CpuMeter, Snapshot  # noqa: E402
from jarvis.autonomy.runners import REGISTRY  # noqa: E402
from jarvis.autonomy.scheduler import Scheduler  # noqa: E402
from jarvis.autonomy.tasks import Priority, State, TaskQueue  # noqa: E402


def snap(idle=600.0, cpu=5.0, ram=40.0, gpu=2.0) -> Snapshot:
    return Snapshot(taken=time.time(), idle_seconds=idle, cpu_percent=cpu,
                    ram_percent=ram, gpu_percent=gpu, vram_used_mb=1000,
                    vram_total_mb=16000)


NOON = datetime(2026, 8, 11, 13, 0)
LATE = datetime(2026, 8, 11, 3, 0)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = Policy(idle_after_s=300, night_hours=(1, 8))

    def test_recent_input_blocks_autonomous_work(self):
        stance = self.policy.evaluate(snap(idle=10.0), at=NOON)
        self.assertEqual(stance.mode, ACTIVE)
        self.assertFalse(stance.autonomous_allowed)

    def test_user_tasks_run_even_when_active(self):
        stance = self.policy.evaluate(snap(idle=0.0), at=NOON)
        self.assertTrue(stance.admits(Priority.USER))

    def test_idle_and_quiet_allows_background(self):
        stance = self.policy.evaluate(snap(), at=NOON)
        self.assertEqual(stance.mode, IDLE)
        self.assertTrue(stance.admits(Priority.BACKGROUND))

    def test_idle_daytime_does_not_admit_long_night_work(self):
        stance = self.policy.evaluate(snap(), at=NOON)
        self.assertFalse(stance.admits(Priority.IDLE_ONLY))

    def test_night_admits_long_work(self):
        stance = self.policy.evaluate(snap(), at=LATE)
        self.assertEqual(stance.mode, NIGHT)
        self.assertTrue(stance.admits(Priority.IDLE_ONLY))

    def test_busy_cpu_blocks_even_when_user_is_away(self):
        """A long render with nobody at the desk is still not our GPU to take."""
        stance = self.policy.evaluate(snap(idle=99999.0, cpu=95.0), at=LATE)
        self.assertEqual(stance.mode, BUSY)
        self.assertFalse(stance.autonomous_allowed)

    def test_busy_gpu_blocks(self):
        stance = self.policy.evaluate(snap(gpu=90.0), at=LATE)
        self.assertEqual(stance.mode, BUSY)

    def test_unknown_readings_are_treated_as_busy(self):
        blind = Snapshot(time.time(), None, None, None, None, None, None)
        stance = self.policy.evaluate(blind, at=LATE)
        self.assertEqual(stance.mode, BUSY)
        self.assertFalse(stance.autonomous_allowed)

    def test_night_window_wraps_past_midnight(self):
        wrapping = Policy(night_hours=(23, 6))
        self.assertTrue(wrapping.is_night(datetime(2026, 8, 11, 23, 30)))
        self.assertTrue(wrapping.is_night(datetime(2026, 8, 11, 2, 0)))
        self.assertFalse(wrapping.is_night(datetime(2026, 8, 11, 12, 0)))


class TestQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.queue = TaskQueue(Path(self._tmp.name) / "q.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_priority_wins_over_arrival_order(self):
        self.queue.add("noop", "arka plan", priority=Priority.BACKGROUND)
        self.queue.add("noop", "kullanıcı", priority=Priority.USER)
        self.assertEqual(self.queue.claim(max_priority=Priority.IDLE_ONLY).title, "kullanıcı")

    def test_claim_respects_the_priority_ceiling(self):
        self.queue.add("noop", "gece işi", priority=Priority.IDLE_ONLY)
        self.assertIsNone(self.queue.claim(max_priority=Priority.BACKGROUND))
        self.assertIsNotNone(self.queue.claim(max_priority=Priority.IDLE_ONLY))

    def test_a_claimed_task_is_not_handed_out_twice(self):
        self.queue.add("noop", "tek")
        self.assertIsNotNone(self.queue.claim(max_priority=Priority.IDLE_ONLY))
        self.assertIsNone(self.queue.claim(max_priority=Priority.IDLE_ONLY))

    def test_delayed_tasks_are_not_runnable_yet(self):
        self.queue.add("noop", "sonra", delay_s=3600)
        self.assertIsNone(self.queue.claim(max_priority=Priority.IDLE_ONLY))

    def test_dedupe_key_blocks_a_second_copy(self):
        self.assertIsNotNone(self.queue.add("noop", "bir", dedupe_key="routine:noop"))
        self.assertIsNone(self.queue.add("noop", "iki", dedupe_key="routine:noop"))

    def test_dedupe_frees_up_once_the_first_finishes(self):
        first = self.queue.add("noop", "bir", dedupe_key="routine:noop")
        self.queue.complete(first, "tamam")
        self.assertIsNotNone(self.queue.add("noop", "iki", dedupe_key="routine:noop"))

    def test_failure_retries_with_backoff_then_quarantines(self):
        self.queue.add("fail", "bozuk", max_attempts=2)
        task = self.queue.claim(max_priority=Priority.IDLE_ONLY)
        self.assertEqual(self.queue.fail(task.id, "birinci"), State.PENDING)
        self.assertGreater(self.queue.get(task.id).not_before, time.time())

        # Second attempt exhausts the allowance.
        self.queue.add("noop", "dolgu")
        task2 = self.queue.get(task.id)
        self.assertEqual(task2.attempts, 1)

    def test_quarantine_after_max_attempts(self):
        self.queue.add("fail", "bozuk", max_attempts=1)
        task = self.queue.claim(max_priority=Priority.IDLE_ONLY)
        self.assertEqual(self.queue.fail(task.id, "tek deneme"), State.QUARANTINED)

    def test_orphans_return_to_the_queue_with_the_attempt_counted(self):
        """A reboot mid-task must not lose the task, nor retry it forever."""
        self.queue.add("noop", "yarım kalan", max_attempts=3)
        claimed = self.queue.claim(max_priority=Priority.IDLE_ONLY)
        self.assertEqual(self.queue.recover_orphans(), 1)
        recovered = self.queue.get(claimed.id)
        self.assertEqual(recovered.state, State.PENDING)
        self.assertEqual(recovered.attempts, 1)

    def test_state_survives_a_new_queue_object(self):
        self.queue.add("noop", "kalıcı")
        reopened = TaskQueue(self.queue.db_path)
        self.assertEqual(reopened.claim(max_priority=Priority.IDLE_ONLY).title, "kalıcı")

    def test_cancel_removes_a_pending_task(self):
        task_id = self.queue.add("noop", "vazgeçildi")
        self.assertTrue(self.queue.cancel(task_id))
        self.assertIsNone(self.queue.claim(max_priority=Priority.IDLE_ONLY))


class TestEvents(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.events = EventLog(Path(self._tmp.name) / "e.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_events_persist(self):
        self.events.publish("test", "kind", "bir şey oldu")
        self.assertEqual(len(EventLog(self.events.db_path).since(60)), 1)

    def test_subscribers_receive_events(self):
        seen = []
        self.events.subscribe(seen.append)
        self.events.publish("test", "kind", "mesaj")
        self.assertEqual(len(seen), 1)

    def test_a_broken_subscriber_cannot_stop_publishing(self):
        """A crashing UI listener must not take down the task that was reporting."""
        def explode(_event):
            raise RuntimeError("abone bozuk")

        seen = []
        self.events.subscribe(explode)
        self.events.subscribe(seen.append)
        self.events.publish("test", "kind", "mesaj")
        self.assertEqual(len(seen), 1)

    def test_unsubscribe_stops_delivery(self):
        seen = []
        cancel = self.events.subscribe(seen.append)
        cancel()
        self.events.publish("test", "kind", "mesaj")
        self.assertEqual(seen, [])


class TestStanceIdentity(unittest.TestCase):
    """What counts as "the situation changed".

    Measured in S9: 522 of 529 events in a 6.4 hour night were stance records
    that said nothing new, because the change detection compared a message with
    a live number in it. The stance now carries a stable cause for exactly this.
    """

    def setUp(self):
        self.policy = Policy(idle_after_s=300, night_hours=(1, 8))

    def test_the_key_holds_while_only_the_number_moves(self):
        first = self.policy.evaluate(snap(idle=10.0), at=NOON)
        second = self.policy.evaluate(snap(idle=20.0), at=NOON)
        self.assertNotEqual(first.reason, second.reason)
        self.assertEqual(first.key, second.key)

    def test_the_key_changes_when_the_situation_does(self):
        active = self.policy.evaluate(snap(idle=10.0), at=NOON)
        idle = self.policy.evaluate(snap(idle=600.0), at=NOON)
        busy = self.policy.evaluate(snap(cpu=99.0), at=NOON)
        self.assertNotEqual(active.key, idle.key)
        self.assertNotEqual(idle.key, busy.key)

    def test_different_pressures_are_different_causes(self):
        """Busy from CPU and busy from GPU are not the same event."""
        cpu = self.policy.evaluate(snap(cpu=99.0), at=NOON)
        gpu = self.policy.evaluate(snap(gpu=99.0), at=NOON)
        self.assertNotEqual(cpu.key, gpu.key)

    def test_an_unreadable_machine_says_what_could_not_be_read(self):
        blind = Snapshot(time.time(), None, None, None, None, None, None,
                         unknown={"cpu_percent": "ölçüm aralığı çok kısa"})
        stance = self.policy.evaluate(blind, at=NOON)
        self.assertEqual(stance.mode, BUSY)
        self.assertIn("ölçüm aralığı çok kısa", stance.reason)

    def test_an_unreadable_machine_without_a_reason_admits_that(self):
        stance = self.policy.evaluate(
            Snapshot(time.time(), None, None, None, None, None, None), at=NOON)
        self.assertIn("sebep kaydedilmedi", stance.reason)


class TestStanceLogging(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.events = EventLog(Path(self._tmp.name) / "e.db")
        self.scheduler = Scheduler(TaskQueue(Path(self._tmp.name) / "q.db"),
                                   self.events, Policy())
        self.seen = []
        self.events.subscribe(self.seen.append)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_held_stance_is_recorded_once(self):
        for seconds in range(0, 60, 5):
            self.scheduler._log_stance(  # noqa: SLF001 - the throttle is the subject
                Stance(ACTIVE, Priority.USER, 1,
                       f"kullanıcı aktif ({seconds}s önce girdi)", USER_ACTIVE))
        self.assertEqual(len(self.seen), 1)

    def test_a_changed_stance_is_recorded_again(self):
        self.scheduler._log_stance(  # noqa: SLF001
            Stance(ACTIVE, Priority.USER, 1, "kullanıcı aktif (5s)", USER_ACTIVE))
        self.scheduler._log_stance(  # noqa: SLF001
            Stance(NIGHT, Priority.IDLE_ONLY, 2, "gece ve boşta (1 dk)", QUIET))
        self.assertEqual(len(self.seen), 2)

    def test_the_cause_travels_with_the_event(self):
        self.scheduler._log_stance(  # noqa: SLF001
            Stance(BUSY, Priority.USER, 1, "makine meşgul — CPU %90", CPU))
        self.assertEqual(self.seen[0].data["cause"], CPU)


class TestCpuMeter(unittest.TestCase):
    """The reading that came back unknown five times in one night."""

    @unittest.skipUnless(os.name == "nt", "GetSystemTimes yalnızca Windows'ta")
    def test_a_second_reader_does_not_steal_the_interval(self):
        """The telemetry pump polls the same meter as the scheduler.

        Before the minimum interval, a call landing milliseconds after another
        divided by a zero delta and reported unknown — which the policy correctly
        turns into "assume busy" and which cost the S9 night five stances.
        """
        meter = CpuMeter()
        meter.sample()
        time.sleep(CPU_MIN_INTERVAL_S + 0.2)
        first = meter.sample()
        self.assertIsNotNone(first, meter.last_reason)
        self.assertEqual(meter.sample(), first)
        self.assertEqual(meter.sample(), first)

    def test_an_unavailable_reading_records_why(self):
        meter = CpuMeter()
        if meter.sample() is None:
            self.assertTrue(meter.last_reason)

    def test_a_snapshot_explains_its_gaps(self):
        blind = Snapshot(time.time(), None, None, None, None, None, None,
                         unknown={"cpu_percent": "GetSystemTimes başarısız döndü",
                                  "gpu_percent": "nvidia-smi bulunamadı"})
        self.assertIn("GetSystemTimes", blind.why_unknown())
        # A missing GPU reading is not what stops autonomous work, so it is not
        # what the stance message should be about.
        self.assertNotIn("nvidia-smi", blind.why_unknown())


class TestRunners(unittest.TestCase):
    def test_builtin_runners_are_registered(self):
        for name in ("noop", "memory.reindex", "system.snapshot", "system.selftest",
                     "tasks.purge", "events.purge"):
            self.assertIn(name, REGISTRY)


if __name__ == "__main__":
    unittest.main()
