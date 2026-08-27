"""S9b: recurring work, retention, one-instance-only, and autostart.

What the S9 soak measured is the reason this file exists. The system stayed up
for 6h23m without a single error and did almost nothing, because queueing upkeep
at startup is not the same as having work to do tonight. These tests pin the
difference, and pin the part that matters more than any of it: a routine gets no
privileges. It is queued the same way, admitted by the same policy, and passes
through the same budget, provenance, sandbox and promotion gates as work asked
for by hand.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.autonomy import ROUTINES, AutonomyCore, Routine, _configured_routines  # noqa: E402
from jarvis.autonomy import runners  # noqa: E402
from jarvis.autonomy.events import ERROR, EventLog, INFO, SUCCESS, WARN  # noqa: E402
from jarvis.autonomy.tasks import Priority, State, TaskQueue  # noqa: E402
from jarvis.cli.instance import InstanceLock  # noqa: E402
from jarvis.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400.0


class FakeConfig(Config):
    """A Config with nothing in it but what a test puts there."""

    def __init__(self, data=None):
        self._data = data or {}

    def get(self, dotted, default=None):
        node = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted, default=""):
        return Path(self.get(dotted, default))


class TestRoutineDefinitions(unittest.TestCase):
    def test_the_night_worker_is_night_only(self):
        """IDLE_ONLY is the whole gate: only the NIGHT stance admits it."""
        cycle = next(r for r in ROUTINES if r.kind == "improve.cycle")
        self.assertEqual(cycle.priority, Priority.IDLE_ONLY)

    def test_no_routine_claims_user_priority(self):
        """Routine work must never outrank what the user asked for."""
        for routine in ROUTINES:
            with self.subTest(kind=routine.kind):
                self.assertGreater(routine.priority, Priority.CRITICAL)

    def test_every_routine_repeats(self):
        for routine in ROUTINES:
            with self.subTest(kind=routine.kind):
                self.assertGreaterEqual(routine.interval_s, 60)

    def test_config_can_change_an_interval(self):
        config = FakeConfig({"autonomy": {"routine_intervals": {"improve.cycle": 7200}}})
        cycle = next(r for r in _configured_routines(config) if r.kind == "improve.cycle")
        self.assertEqual(cycle.interval_s, 7200)

    def test_a_nonsense_interval_is_ignored_not_obeyed(self):
        """A typo must not turn a nightly job into a spin loop."""
        config = FakeConfig({"autonomy": {"routine_intervals":
                                          {"improve.cycle": 5, "tasks.purge": "yarım saat"}}})
        routines = {r.kind: r for r in _configured_routines(config)}
        self.assertEqual(routines["improve.cycle"].interval_s,
                         next(r for r in ROUTINES if r.kind == "improve.cycle").interval_s)
        self.assertEqual(routines["tasks.purge"].interval_s,
                         next(r for r in ROUTINES if r.kind == "tasks.purge").interval_s)

    def test_a_routine_can_be_switched_off(self):
        config = FakeConfig({"autonomy": {"routines_disabled": ["improve.cycle"]}})
        kinds = [r.kind for r in _configured_routines(config)]
        self.assertNotIn("improve.cycle", kinds)

    def test_retention_reaches_the_purge_routines(self):
        config = FakeConfig({"autonomy": {"event_retention_days": 7,
                                          "event_problem_retention_days": 14}})
        purge = next(r for r in _configured_routines(config) if r.kind == "events.purge")
        self.assertEqual(purge.payload["older_than_days"], 7)
        self.assertEqual(purge.payload["problem_older_than_days"], 14)


class TestDueness(unittest.TestCase):
    """The queue is the only record of when something last ran."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        self.queue = TaskQueue(self.db)
        self.registered = []

    def tearDown(self):
        for name in self.registered:
            REGISTRY_pop(name)
        self.tmp.cleanup()

    def register(self, name):
        runners.REGISTRY[name] = lambda ctx: "tamam"
        self.registered.append(name)

    def core(self, routines):
        core = AutonomyCore.__new__(AutonomyCore)
        core.queue = self.queue
        core.routines = tuple(routines)
        core._started_at = time.time() - 10_000
        core.scheduler = type("S", (), {"nudge": lambda self: None})()
        return core

    def test_a_routine_that_never_ran_is_queued(self):
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.BACKGROUND, DAY)])
        self.assertEqual(core.queue_due_routines(), 1)

    def test_it_is_not_queued_twice_while_it_waits(self):
        """The machine can stay busy for hours; that is not a reason to pile up."""
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.BACKGROUND, DAY)])
        core.queue_due_routines()
        self.assertEqual(core.queue_due_routines(), 0)
        self.assertEqual(core.queue_due_routines(), 0)
        self.assertEqual(len(self.queue.list(limit=50)), 1)

    def test_it_is_not_queued_again_until_the_interval_passes(self):
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.BACKGROUND, DAY)])
        core.queue_due_routines()
        task = self.queue.claim(max_priority=Priority.IDLE_ONLY)
        self.queue.complete(task.id, "bitti")

        self.assertEqual(core.queue_due_routines(), 0)
        self.assertEqual(core.queue_due_routines(now=time.time() + DAY + 1), 1)

    def test_a_quarantined_routine_comes_back_next_interval(self):
        """One bad night must not retire a nightly job forever."""
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.BACKGROUND, DAY, max_attempts=1)])
        core.queue_due_routines()
        task = self.queue.claim(max_priority=Priority.IDLE_ONLY)
        self.assertEqual(self.queue.fail(task.id, "patladı"), State.QUARANTINED)

        self.assertEqual(core.queue_due_routines(), 0)
        self.assertEqual(core.queue_due_routines(now=time.time() + DAY + 1), 1)

    def test_a_routine_without_a_runner_is_skipped(self):
        """A build without the improvement engine must not quarantine itself."""
        core = self.core([Routine("t.missing", "yok", Priority.BACKGROUND, DAY)])
        self.assertEqual(core.queue_due_routines(), 0)
        self.assertEqual(self.queue.list(limit=10), [])

    def test_the_first_run_waits_out_the_settling_delay(self):
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.BACKGROUND, DAY,
                                  initial_delay_s=600)])
        core._started_at = time.time()
        self.assertEqual(core.queue_due_routines(), 0)
        self.assertEqual(core.queue_due_routines(now=time.time() + 601), 1)

    def test_routine_work_is_marked_as_such(self):
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.BACKGROUND, DAY)])
        core.queue_due_routines()
        self.assertEqual(self.queue.list(limit=1)[0].origin, "routine")

    def test_last_run_prefers_finished_then_started_then_created(self):
        self.queue.add("t.b", "B")
        self.assertGreater(self.queue.last_run("t.b"), 0.0)
        self.assertEqual(self.queue.last_run("t.nothing"), 0.0)

    # -- a routine that waits a long time -------------------------------------
    #
    # The tests above queue a task and ask again immediately, so the interval
    # check answers first and the insert path is never reached. Production does
    # not look like that: improve.cycle is IDLE_ONLY and waited five days for a
    # night window. These age the row to reach the path that actually ran.

    def age_created(self, task_id: int, seconds: float) -> None:
        with closing(self.queue._conn()) as conn, conn:  # noqa: SLF001 - test ages a row
            conn.execute("UPDATE tasks SET created = created - ? WHERE id = ?",
                         (seconds, task_id))

    def test_waiting_keys_mirror_the_dedupe_index(self):
        self.queue.add("t.a", "A", dedupe_key="routine:t.a")
        done = self.queue.add("t.b", "B", dedupe_key="routine:t.b")
        self.queue.complete(done, "bitti")
        self.queue.add("t.c", "C")
        self.assertEqual(self.queue.waiting_dedupe_keys(), {"routine:t.a"})

    def test_a_long_waiting_routine_stops_asking_to_be_queued(self):
        """last_run() falls back to `created` for a task that never ran, so a
        routine waiting for a quiet machine looked overdue on every tick and went
        through insert-and-be-rejected. Measured: 94% of the log file."""
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.IDLE_ONLY, DAY)])
        core.queue_due_routines()
        self.age_created(self.queue.list(limit=1)[0].id, 10 * DAY)

        attempts = []
        original = self.queue.add

        def spy(*args, **kwargs):
            attempts.append(args)
            return original(*args, **kwargs)

        self.queue.add = spy
        try:
            self.assertEqual(core.queue_due_routines(), 0)
            self.assertEqual(core.queue_due_routines(), 0)
        finally:
            self.queue.add = original
        self.assertEqual(attempts, [], "bekleyen rutin icin insert denenmemeli")
        self.assertEqual(len(self.queue.list(limit=50)), 1)

    def test_a_long_waiting_routine_is_quiet_in_the_log(self):
        """The damage was never disk; it was the log losing its use as a record."""
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.IDLE_ONLY, DAY)])
        core.queue_due_routines()
        self.age_created(self.queue.list(limit=1)[0].id, 10 * DAY)
        with self.assertNoLogs("jarvis.autonomy.tasks", level="DEBUG"):
            core.queue_due_routines()

    def test_a_running_routine_also_blocks_a_second_copy(self):
        """The dedupe index covers running as well; the skip has to match it."""
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.BACKGROUND, DAY)])
        core.queue_due_routines()
        task = self.queue.claim(max_priority=Priority.IDLE_ONLY)
        self.assertEqual(task.state, State.RUNNING)
        self.assertIn("routine:t.a", self.queue.waiting_dedupe_keys())
        self.assertEqual(core.queue_due_routines(now=time.time() + 10 * DAY), 0)

    def test_the_routine_still_returns_after_the_waiting_one_finishes(self):
        """The skip must not retire a routine: once it completes, the interval
        governs again. A quiet log that never runs anything is worse than spam."""
        self.register("t.a")
        core = self.core([Routine("t.a", "A", Priority.IDLE_ONLY, DAY)])
        core.queue_due_routines()
        self.age_created(self.queue.list(limit=1)[0].id, 10 * DAY)
        self.assertEqual(core.queue_due_routines(), 0)

        task = self.queue.claim(max_priority=Priority.IDLE_ONLY)
        self.queue.complete(task.id, "bitti")
        self.assertEqual(core.queue_due_routines(), 0)
        self.assertEqual(core.queue_due_routines(now=time.time() + DAY + 1), 1)


def REGISTRY_pop(name):
    runners.REGISTRY.pop(name, None)


class TestEventRetention(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = EventLog(Path(self.tmp.name) / "events.db")

    def tearDown(self):
        self.tmp.cleanup()

    def age(self, seconds: float, level: str) -> None:
        event = self.log.publish("test", "x", "eski", level=level)
        with closing(self.log._conn()) as conn, conn:  # noqa: SLF001 - test ages a row
            conn.execute("UPDATE events SET ts = ? WHERE message = 'eski' AND ts = ?",
                         (event.ts - seconds, event.ts))

    def count(self) -> int:
        with closing(self.log._conn()) as conn:  # noqa: SLF001
            return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def test_old_routine_activity_is_dropped(self):
        self.age(40 * DAY, INFO)
        self.assertEqual(self.log.purge(older_than_days=30), 1)
        self.assertEqual(self.count(), 0)

    def test_recent_activity_is_kept(self):
        self.log.publish("test", "x", "yeni")
        self.assertEqual(self.log.purge(older_than_days=30), 0)
        self.assertEqual(self.count(), 1)

    def test_problems_are_kept_longer_than_chatter(self):
        """A warning is the record of something going wrong; chatter is not."""
        self.age(40 * DAY, WARN)
        self.age(40 * DAY, ERROR)
        self.age(40 * DAY, SUCCESS)
        self.assertEqual(self.log.purge(older_than_days=30,
                                        problem_older_than_days=90), 1)
        self.assertEqual(self.count(), 2)

    def test_problems_go_eventually(self):
        self.age(100 * DAY, ERROR)
        self.assertEqual(self.log.purge(older_than_days=30,
                                        problem_older_than_days=90), 1)

    def test_the_runner_reports_what_it_removed(self):
        self.age(40 * DAY, INFO)
        task = type("T", (), {"payload": {"older_than_days": 30}})()
        context = runners.RunContext(task=task, events=self.log,
                                     should_stop=lambda: False)
        self.assertIn("1 eski olay", runners.get("events.purge")(context))


class TestInstanceLock(unittest.TestCase):
    """Autostart plus a terminal is two copies unless something says no."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "daemon.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_free_lock_is_taken(self):
        lock = InstanceLock(self.path)
        self.assertTrue(lock.acquire())
        # pid first, creation time second — see TestLockIdentity for why.
        self.assertEqual(self.path.read_text(encoding="ascii").split()[0],
                         str(os.getpid()))
        lock.release()
        self.assertFalse(self.path.exists())

    def test_a_live_holder_blocks_a_second_copy(self):
        self.path.write_text("999999", encoding="ascii")
        lock = InstanceLock(self.path)
        lock_module = sys.modules["jarvis.cli.instance"]
        original = lock_module.process_alive
        lock_module.process_alive = lambda pid: True
        try:
            self.assertFalse(lock.acquire())
            self.assertEqual(lock.holder, 999999)
        finally:
            lock_module.process_alive = original

    def test_a_stale_lock_is_taken_over(self):
        """After a Windows Update restart the file is always stale."""
        self.path.write_text("999999", encoding="ascii")
        lock_module = sys.modules["jarvis.cli.instance"]
        original = lock_module.process_alive
        lock_module.process_alive = lambda pid: False
        try:
            self.assertTrue(InstanceLock(self.path).acquire())
        finally:
            lock_module.process_alive = original

    def test_a_corrupt_lock_is_not_fatal(self):
        self.path.write_text("bu bir sayi degil", encoding="ascii")
        self.assertTrue(InstanceLock(self.path).acquire())

    def test_this_process_is_not_its_own_rival(self):
        lock = InstanceLock(self.path)
        self.assertTrue(lock.acquire())
        self.assertTrue(InstanceLock(self.path).acquire())

    def test_process_alive_never_signals_the_target_on_windows(self):
        """os.kill(pid, 0) terminates the process on Windows. Only OpenProcess asks.

        Scoped to the function rather than the file: reading the whole module
        made this assertion depend on what else happened to live above it.
        """
        import inspect

        from jarvis.cli.instance import process_alive

        source = inspect.getsource(process_alive)
        windows_branch = source.split('if os.name != "nt":')[1].split("handle = ")[1]
        self.assertIn("OpenProcess", windows_branch)
        self.assertNotIn("os.kill", windows_branch)


class TestLockIdentity(unittest.TestCase):
    """A pid is not an identity. Windows reuses them, and it did.

    Measured: the autonomous loop held pid 14408 until 17:15; by 23:53 an
    unrelated process had the same number. A lock that compares only the pid
    would have refused to start the loop that pid used to belong to, and the
    silence would have looked exactly like a machine nobody turned on.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "daemon.lock"
        self.module = sys.modules["jarvis.cli.instance"]
        self._alive = self.module.process_alive
        self._started = self.module.process_started_at

    def tearDown(self):
        self.module.process_alive = self._alive
        self.module.process_started_at = self._started
        self.tmp.cleanup()

    def fake(self, *, alive: bool, started):
        self.module.process_alive = lambda pid: alive
        self.module.process_started_at = lambda pid: started

    def test_the_lock_records_both_facts(self):
        lock = InstanceLock(self.path)
        self.assertTrue(lock.acquire())
        parts = self.path.read_text(encoding="ascii").split()
        self.assertEqual(int(parts[0]), os.getpid())
        self.assertEqual(len(parts), 2, "olusturulma zamani yazilmali")

    @unittest.skipUnless(os.name == "nt", "GetProcessTimes yalnızca Windows'ta")
    def test_a_real_creation_time_is_readable_and_stable(self):
        from jarvis.cli.instance import process_started_at

        first = process_started_at(os.getpid())
        self.assertIsNotNone(first)
        self.assertGreater(first, 0)
        self.assertEqual(first, process_started_at(os.getpid()))

    def test_a_recycled_pid_does_not_hold_the_lock(self):
        """Live pid, different creation time: a stranger inherited the number."""
        self.path.write_text("999999 111111111", encoding="ascii")
        self.fake(alive=True, started=999999999)
        self.assertTrue(InstanceLock(self.path).acquire())

    def test_the_same_process_still_holds_the_lock(self):
        self.path.write_text("999999 111111111", encoding="ascii")
        self.fake(alive=True, started=111111111)
        self.assertFalse(InstanceLock(self.path).acquire())

    def test_a_dead_pid_is_stale_whatever_the_timestamp(self):
        self.path.write_text("999999 111111111", encoding="ascii")
        self.fake(alive=False, started=111111111)
        self.assertTrue(InstanceLock(self.path).acquire())

    def test_an_old_format_lock_still_blocks_a_live_pid(self):
        """Upgrading over a lock written by the previous build must not double-start."""
        self.path.write_text("999999", encoding="ascii")
        self.fake(alive=True, started=None)
        self.assertFalse(InstanceLock(self.path).acquire())

    def test_an_unreadable_creation_time_fails_closed(self):
        """Cannot tell them apart: assume the holder is real and refuse."""
        self.path.write_text("999999 111111111", encoding="ascii")
        self.fake(alive=True, started=None)
        self.assertFalse(InstanceLock(self.path).acquire())

    def test_a_zero_timestamp_is_treated_as_absent(self):
        """Non-Windows writes 0; that must read as "no timestamp", not as one."""
        self.path.write_text("999999 0", encoding="ascii")
        self.fake(alive=False, started=None)
        self.assertTrue(InstanceLock(self.path).acquire())

    def test_a_garbled_timestamp_does_not_crash(self):
        self.path.write_text("999999 saat-yok", encoding="ascii")
        self.fake(alive=False, started=None)
        self.assertTrue(InstanceLock(self.path).acquire())

    def test_release_only_removes_its_own_lock(self):
        lock = InstanceLock(self.path)
        lock.acquire()
        self.path.write_text("999999 111111111", encoding="ascii")
        lock.release()
        self.assertTrue(self.path.exists(), "baskasinin kilidi silinmemeli")


class TestRoutinesGrantNoPrivileges(unittest.TestCase):
    """The point of S9b is more work at night, not looser rules at night.

    Every gate below existed before routines did. These tests are here so that a
    later change which quietly routes routine work around one of them fails.
    """

    def test_self_modification_is_still_off(self):
        from jarvis.lab.promotion import ALLOW_SELF_MODIFICATION

        self.assertFalse(ALLOW_SELF_MODIFICATION)

    def test_the_source_tree_is_still_protected(self):
        from jarvis.lab.promotion import protected_roots

        protected = {p.name for p in protected_roots(ROOT)}
        self.assertLessEqual({"jarvis", "tests", "persona", "run.py", "config.json"},
                             protected)

    def test_shell_still_needs_a_sandbox_not_a_request(self):
        from jarvis.agents.permissions import FS_WRITE, SHELL, Grant

        asked = frozenset({SHELL, FS_WRITE})
        self.assertFalse(Grant.build("gece", asked).capabilities)
        self.assertTrue(Grant.build("gece", asked, sandboxed=True).capabilities)

    def test_the_night_routine_runs_the_ordinary_cycle(self):
        """No separate path: the routine calls the same engine.cycle()."""
        from jarvis import improve

        calls = []

        class StubResult:
            def summary(self):
                return "tur bitti"

        class StubEngine:
            def cycle(self, *, planner=None):
                calls.append(planner)
                return StubResult()

        original = runners.REGISTRY.get(improve.RUNNER_NAME)
        try:
            improve.register_runner(StubEngine())
            runner = runners.get(improve.RUNNER_NAME)
            context = runners.RunContext(task=type("T", (), {"payload": {}})(),
                                         events=None, should_stop=lambda: False)
            self.assertEqual(runner(context), "tur bitti")
            self.assertEqual(len(calls), 1)
        finally:
            if original is None:
                runners.REGISTRY.pop(improve.RUNNER_NAME, None)
            else:
                runners.REGISTRY[improve.RUNNER_NAME] = original

    def test_the_night_routine_stops_when_asked(self):
        from jarvis import improve

        class Exploding:
            def cycle(self, *, planner=None):
                raise AssertionError("durdurma istendiğinde tur başlamamalıydı")

        original = runners.REGISTRY.get(improve.RUNNER_NAME)
        try:
            improve.register_runner(Exploding())
            context = runners.RunContext(task=type("T", (), {"payload": {}})(),
                                         events=None, should_stop=lambda: True)
            self.assertEqual(runners.get(improve.RUNNER_NAME)(context), "durdurma istendi")
        finally:
            if original is None:
                runners.REGISTRY.pop(improve.RUNNER_NAME, None)
            else:
                runners.REGISTRY[improve.RUNNER_NAME] = original

    def test_the_night_budget_still_refuses(self):
        """More cycles queued does not mean more experiments allowed."""
        from datetime import datetime

        from jarvis.improve.budget import EXPERIMENT, ImprovementBudget

        with tempfile.TemporaryDirectory() as tmp:
            budget = ImprovementBudget(Path(tmp) / "b.db",
                                       daily={EXPERIMENT: 10}, nightly={EXPERIMENT: 2},
                                       night_hours=(1, 8))
            night = datetime(2026, 8, 12, 3, 0)
            self.assertTrue(budget.spend(EXPERIMENT, at=night).allowed)
            self.assertTrue(budget.spend(EXPERIMENT, at=night).allowed)
            refused = budget.spend(EXPERIMENT, at=night)
            self.assertFalse(refused.allowed)
            self.assertIn("gece bütçesi dolu", refused.reason)

    def test_a_refused_activity_is_not_recorded_as_spent(self):
        from datetime import datetime

        from jarvis.improve.budget import RESEARCH, ImprovementBudget

        with tempfile.TemporaryDirectory() as tmp:
            budget = ImprovementBudget(Path(tmp) / "b.db",
                                       daily={RESEARCH: 1}, nightly={RESEARCH: 1})
            budget.spend(RESEARCH, at=datetime(2026, 8, 12, 13, 0))
            budget.spend(RESEARCH, at=datetime(2026, 8, 12, 13, 0))
            self.assertEqual(budget.used(RESEARCH), 1)


class TestAutostart(unittest.TestCase):
    """The scripts are the mechanism; a renamed file is a machine that stays down."""

    def setUp(self):
        self.dir = ROOT / "tools" / "autostart"

    def test_the_scripts_exist(self):
        for name in ("start-jarvis.ps1", "stop-jarvis.ps1", "register.ps1",
                     "unregister.ps1", "README.md"):
            with self.subTest(name=name):
                self.assertTrue((self.dir / name).exists(), name)

    def test_stopping_goes_by_the_lock_file_not_the_task(self):
        """Measured: Stop-ScheduledTask leaves the interpreter at the end of the
        launcher chain running, holding the port and the lock."""
        text = (self.dir / "stop-jarvis.ps1").read_text(encoding="utf-8")
        self.assertIn("daemon.lock", text)
        self.assertIn("Stop-Process", text)

    def test_the_scripts_are_pure_ascii(self):
        """Windows PowerShell 5.1 reads a BOM-less UTF-8 file as the ANSI
        codepage. On a Turkish system the second byte of an em dash decodes to a
        closing quote, ends the string it sits in, and the rest of the file
        becomes parse errors. stop-jarvis.ps1 failed exactly that way once."""
        for script in sorted(self.dir.glob("*.ps1")):
            with self.subTest(script=script.name):
                raw = script.read_bytes()
                offenders = [(i, b) for i, b in enumerate(raw) if b > 0x7F]
                self.assertEqual(offenders[:3], [], f"{script.name}: ASCII disi bayt")

    def test_the_launcher_does_not_redirect_through_powershell(self):
        """PowerShell 5.1 writes redirected output as UTF-16; the first real run
        of this launcher produced a log file no tool could read."""
        code = [line for line in
                (self.dir / "start-jarvis.ps1").read_text(encoding="utf-8").splitlines()
                if not line.lstrip().startswith("#")]
        self.assertNotIn("*>>", "\n".join(code))

    def test_the_launcher_starts_the_autonomous_loop(self):
        text = (self.dir / "start-jarvis.ps1").read_text(encoding="utf-8")
        self.assertIn("run.py --otonom --yayin", text)

    def test_the_launcher_stays_in_the_foreground(self):
        """Task Scheduler can only restart a task it still considers running."""
        text = (self.dir / "start-jarvis.ps1").read_text(encoding="utf-8")
        self.assertNotIn("Start-Process", text)

    def test_registration_uses_the_launcher_and_no_time_limit(self):
        text = (self.dir / "register.ps1").read_text(encoding="utf-8")
        self.assertIn("start-jarvis.ps1", text)
        self.assertIn("ExecutionTimeLimit", text)
        self.assertIn("AtLogOn", text)

    def test_registration_stays_in_the_users_session(self):
        """Session 0 has no keyboard, so the idle policy would never allow work."""
        text = (self.dir / "register.ps1").read_text(encoding="utf-8")
        self.assertIn("LogonType Interactive", text)
        self.assertNotIn("-RunLevel Highest", text)


if __name__ == "__main__":
    unittest.main()
