"""Persistent reminder service and its Turkish time parser."""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.reminders import ReminderService, parse_when  # noqa: E402
from jarvis.tools import Workspace, provide, run  # noqa: E402


NOW = dt.datetime(2026, 8, 26, 14, 20, 0)


class TestTurkishTimeParser(unittest.TestCase):
    def test_minutes_from_now(self):
        self.assertEqual(parse_when("30 dakika sonra", now=NOW),
                         dt.datetime(2026, 8, 26, 14, 50))

    def test_hours_from_now(self):
        self.assertEqual(parse_when("2 saat sonra", now=NOW),
                         dt.datetime(2026, 8, 26, 16, 20))

    def test_tomorrow_clock(self):
        self.assertEqual(parse_when("yarın 09:15", now=NOW),
                         dt.datetime(2026, 8, 27, 9, 15))

    def test_past_clock_rolls_to_tomorrow(self):
        self.assertEqual(parse_when("saat 10:00", now=NOW),
                         dt.datetime(2026, 8, 27, 10, 0))

    def test_iso_date(self):
        self.assertEqual(parse_when("2026-09-03 18:40", now=NOW),
                         dt.datetime(2026, 9, 3, 18, 40))

    def test_turkish_date(self):
        self.assertEqual(parse_when("03.09.2026 18:40", now=NOW),
                         dt.datetime(2026, 9, 3, 18, 40))

    def test_impossible_time_is_refused(self):
        self.assertIsNone(parse_when("yarın 29:99", now=NOW))


class ReminderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fired = []
        self.service = ReminderService(self.root / "jarvis.db",
                                       notifier=self.fired.append, poll_s=.1)

    def tearDown(self):
        self.service.stop()
        self.tmp.cleanup()


class TestReminderPersistence(ReminderCase):
    def test_add_list_and_cancel(self):
        item = self.service.add(dt.datetime.now() + dt.timedelta(hours=1), "Raporu gönder")
        self.assertEqual(self.service.list()[0]["metin"], "Raporu gönder")
        self.assertTrue(self.service.cancel(item["id"]))
        self.assertEqual(self.service.list(), [])

    def test_reminder_survives_a_new_service_instance(self):
        self.service.add(dt.datetime.now() + dt.timedelta(hours=1), "Kalıcı kayıt")
        other = ReminderService(self.root / "jarvis.db", notifier=self.fired.append)
        self.assertEqual(other.list()[0]["metin"], "Kalıcı kayıt")

    def test_due_item_fires_only_once(self):
        self.service.add(dt.datetime.now() + dt.timedelta(seconds=1), "Şimdi")
        future = time.time() + 2
        self.assertEqual(self.service.tick(now=future), 1)
        self.assertEqual(self.service.tick(now=future + 5), 0)
        self.assertEqual(self.fired, ["Şimdi"])

    def test_background_loop_fires(self):
        self.service.add(dt.datetime.now() + dt.timedelta(milliseconds=120), "Döngü")
        self.service.start()
        deadline = time.monotonic() + 3
        while not self.fired and time.monotonic() < deadline:
            time.sleep(.03)
        self.assertEqual(self.fired, ["Döngü"])

    def test_empty_text_is_refused(self):
        with self.assertRaises(ValueError):
            self.service.add(dt.datetime.now() + dt.timedelta(hours=1), "  ")


class TestReminderTools(ReminderCase):
    def setUp(self):
        super().setUp()
        self.workspace = Workspace(self.root)
        provide("reminders", self.service)

    def test_add_requires_confirmation(self):
        result = run("reminder.add", workspace=self.workspace,
                     when="30 dakika sonra", text="Su iç")
        self.assertTrue(result.needs_confirmation)
        self.assertEqual(self.service.list(), [])

    def test_confirmed_add_and_list(self):
        created = run("reminder.add", workspace=self.workspace, confirmed=True,
                      when="30 dakika sonra", text="Su iç")
        self.assertTrue(created.ok, created.error)
        listed = run("reminder.list", workspace=self.workspace)
        self.assertIn("Su iç", listed.output)

    def test_cancel_requires_confirmation(self):
        item = self.service.add(dt.datetime.now() + dt.timedelta(hours=1), "Toplantı")
        result = run("reminder.cancel", workspace=self.workspace, reminder_id=item["id"])
        self.assertTrue(result.needs_confirmation)


if __name__ == "__main__":
    unittest.main()
