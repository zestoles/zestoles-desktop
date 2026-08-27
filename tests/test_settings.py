"""Settings the user can change, and the ones they must not be handed.

A settings screen is a write path into the configuration of a program that runs
shell commands, so the interesting part is not the form -- it is the allow-list.
Anything not named here is refused, which means a new dangerous key cannot
become editable by accident: it has to be added deliberately, next to its own
bounds and its own test.

The rest is ordinary care with a file the program needs at startup: a value out
of range is refused rather than clamped silently, a broken config file falls
back to defaults instead of stopping JARVIS from opening, and a half-written
file is never left behind.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.config import Config  # noqa: E402
from jarvis.settings import (  # noqa: E402
    EDITABLE,
    describe,
    read_all,
    write_one,
)


class SettingsCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "config.json"
        self.config = Config.load(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))


class TestTheAllowList(SettingsCase):
    def test_the_dangerous_switches_are_not_in_it(self):
        """None of these are configuration at all -- and the list is the reason
        a future one cannot quietly become so."""
        for forbidden in ("allow_self_modification", "sandbox", "permissions",
                          "promotion", "self_modification", "lab"):
            for key in EDITABLE:
                self.assertNotIn(forbidden, key.lower(), key)

    def test_an_unknown_key_is_refused(self):
        ok, why = write_one(self.config, "lab.allow_self_modification", True,
                            path=self.path)
        self.assertFalse(ok)
        self.assertTrue(why)

    def test_a_refused_key_writes_nothing_at_all(self):
        write_one(self.config, "brain.local.host", "http://evil", path=self.path)
        self.assertFalse(self.path.exists(), "reddedilen ayar dosyaya yazilmamali")

    def test_every_editable_key_has_a_default_to_show(self):
        """A settings row with no current value renders an empty box, and an
        empty box is a control that will not say what it does. It happened:
        `assistant.max_steps` was read by the code and missing from DEFAULTS."""
        for row in read_all(self.config):
            with self.subTest(key=row["anahtar"]):
                self.assertIsNotNone(row["deger"], row["anahtar"])
                self.assertNotEqual(row["deger"], "", row["anahtar"])

    def test_every_editable_key_has_bounds_and_a_label(self):
        for key, rule in EDITABLE.items():
            with self.subTest(key=key):
                self.assertTrue(rule.label, key)
                self.assertIsNotNone(rule.kind, key)


class TestWriting(SettingsCase):
    def test_a_valid_change_is_saved_and_visible(self):
        ok, why = write_one(self.config, "ui.orphan_grace_s", 300, path=self.path)
        self.assertTrue(ok, why)
        self.assertEqual(self.config.get("ui.orphan_grace_s"), 300)
        self.assertEqual(self.saved()["ui"]["orphan_grace_s"], 300)

    def test_it_survives_a_reload(self):
        write_one(self.config, "chat.history_turns", 20, path=self.path)
        self.assertEqual(Config.load(self.path).get("chat.history_turns"), 20)

    def test_only_the_changed_key_is_written(self):
        """Saving the whole resolved config would freeze today's defaults into
        the file, and a later default change would never reach this user."""
        write_one(self.config, "ui.orphan_grace_s", 300, path=self.path)
        saved = self.saved()
        self.assertEqual(list(saved), ["ui"])
        self.assertEqual(list(saved["ui"]), ["orphan_grace_s"])

    def test_a_second_change_keeps_the_first(self):
        write_one(self.config, "ui.orphan_grace_s", 300, path=self.path)
        write_one(self.config, "chat.history_turns", 8, path=self.path)
        saved = self.saved()
        self.assertEqual(saved["ui"]["orphan_grace_s"], 300)
        self.assertEqual(saved["chat"]["history_turns"], 8)

    def test_saving_into_a_real_config_leaves_everything_else_alone(self):
        """The live config.json is a file the user has edited by hand. Writing
        one setting into it must change exactly that setting -- measured against
        a file with the shape of the real one, not an empty document."""
        self.path.write_text(json.dumps({
            "user": {"name": "Ada"},
            "local": {"model": "qwen3.5:9b", "temperature": 0.7},
            "chat": {"history_turns": 12},
            "autonomy": {"routine_intervals": {"memory.reindex": 21600}},
        }, ensure_ascii=False), encoding="utf-8")
        config = Config.load(self.path)

        ok, why = write_one(config, "chat.history_turns", 20, path=self.path)
        self.assertTrue(ok, why)

        saved = self.saved()
        self.assertEqual(saved["chat"]["history_turns"], 20)
        self.assertEqual(saved["user"]["name"], "Ada")
        self.assertEqual(saved["local"]["model"], "qwen3.5:9b")
        self.assertEqual(saved["local"]["temperature"], 0.7)
        self.assertEqual(saved["autonomy"]["routine_intervals"]["memory.reindex"],
                         21600)

    def test_a_new_section_does_not_disturb_the_existing_ones(self):
        self.path.write_text(json.dumps({"chat": {"history_turns": 12}}),
                             encoding="utf-8")
        config = Config.load(self.path)
        write_one(config, "ui.orphan_grace_s", 300, path=self.path)
        saved = self.saved()
        self.assertEqual(saved["chat"]["history_turns"], 12)
        self.assertEqual(saved["ui"]["orphan_grace_s"], 300)

    def test_a_value_below_the_floor_is_refused(self):
        ok, why = write_one(self.config, "chat.history_turns", 0, path=self.path)
        self.assertFalse(ok)
        self.assertIn("2", why)

    def test_a_value_above_the_ceiling_is_refused(self):
        ok, _why = write_one(self.config, "ui.orphan_grace_s", 999999, path=self.path)
        self.assertFalse(ok)

    def test_a_refused_value_does_not_change_the_live_config(self):
        before = self.config.get("chat.history_turns")
        write_one(self.config, "chat.history_turns", 0, path=self.path)
        self.assertEqual(self.config.get("chat.history_turns"), before)

    def test_text_that_is_not_a_number_is_refused(self):
        ok, _why = write_one(self.config, "ui.orphan_grace_s", "epeyce", path=self.path)
        self.assertFalse(ok)

    def test_a_number_written_as_text_is_accepted(self):
        """The form sends strings; refusing them would make the UI lie about
        which values are valid."""
        ok, why = write_one(self.config, "ui.orphan_grace_s", "300", path=self.path)
        self.assertTrue(ok, why)
        self.assertEqual(self.config.get("ui.orphan_grace_s"), 300)

    def test_zero_is_allowed_where_zero_means_off(self):
        ok, why = write_one(self.config, "ui.orphan_grace_s", 0, path=self.path)
        self.assertTrue(ok, why)


class TestReading(SettingsCase):
    def test_it_reports_every_editable_setting_with_its_value(self):
        rows = read_all(self.config)
        self.assertEqual({row["anahtar"] for row in rows}, set(EDITABLE))
        for row in rows:
            self.assertIn("deger", row)
            self.assertIn("etiket", row)

    def test_it_describes_the_system_without_offering_to_change_it(self):
        """Model, workspace and where the logs are: useful to see, not to edit
        from a chat window."""
        facts = describe(self.config)
        self.assertTrue(facts)
        for row in facts:
            self.assertIn("etiket", row)
            self.assertIn("deger", row)
            self.assertNotIn("anahtar", row, "salt okunur satir duzenlenebilir gorunmemeli")


class TestBrokenFiles(SettingsCase):
    def test_a_corrupt_config_falls_back_to_defaults(self):
        """A settings file that stops JARVIS from opening is worse than one
        whose contents were lost."""
        self.path.write_text("{ bu json degil", encoding="utf-8")
        config = Config.load(self.path)
        self.assertEqual(config.get("chat.history_turns"), 24)

    def test_a_corrupt_file_is_kept_rather_than_overwritten_silently(self):
        self.path.write_text("{ bu json degil", encoding="utf-8")
        Config.load(self.path)
        ok, _why = write_one(Config.load(self.path), "ui.orphan_grace_s", 60,
                             path=self.path)
        self.assertTrue(ok)
        backup = self.path.with_suffix(".json.bozuk")
        self.assertTrue(backup.exists(), "bozuk dosya yedeklenmeli")

    def test_nothing_partial_is_left_behind_when_writing_fails(self):
        ok, _why = write_one(self.config, "ui.orphan_grace_s", 60, path=self.path)
        self.assertTrue(ok)
        self.assertFalse(list(self.path.parent.glob("*.tmp")), "gecici dosya kalmis")


if __name__ == "__main__":
    unittest.main()
