"""Per-run control record used by the dependency-free Windows tray."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.cli.interface import _clear_control, _publish_control  # noqa: E402


class Config:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, _key, default=""):
        return self.root / default


class Server:
    token = "per-run-secret"


class TestControlRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_contains_the_bound_url_and_current_pid(self):
        path = _publish_control(Config(self.root), Server(), "http://127.0.0.1:4321/")
        data = json.loads(path.read_text(encoding="ascii"))
        self.assertEqual(data["url"], "http://127.0.0.1:4321/")
        self.assertEqual(data["token"], Server.token)
        self.assertGreater(data["pid"], 0)

    def test_own_record_is_removed_on_shutdown(self):
        path = _publish_control(Config(self.root), Server(), "http://127.0.0.1:1/")
        _clear_control(path, Server.token)
        self.assertFalse(path.exists())

    def test_an_old_process_cannot_remove_a_newer_record(self):
        path = _publish_control(Config(self.root), Server(), "http://127.0.0.1:1/")
        _clear_control(path, "older-token")
        self.assertTrue(path.exists())

    def test_corrupt_record_is_harmless(self):
        path = self.root / "data" / "control.json"
        path.parent.mkdir(parents=True)
        path.write_text("{broken", encoding="ascii")
        _clear_control(path, Server.token)
        self.assertTrue(path.exists(), "unowned corrupt evidence should be preserved")


if __name__ == "__main__":
    unittest.main()
