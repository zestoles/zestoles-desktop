"""The out-of-box ZESTOLES contract must survive a missing config file."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis.config import Config


class TestSafeFreeDefaults(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config.load(Path(self._tmp.name) / "missing.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_local_models_and_memory_are_the_default(self):
        self.assertEqual(self.config.get("local.model"), "qwen3.5:9b")
        self.assertEqual(self.config.get("local.model_heavy"), "qwen3:14b")
        self.assertEqual(self.config.get("router.mode"), "local")
        self.assertTrue(self.config.get("memory.enabled"))

    def test_cloud_cannot_spend_when_config_is_missing(self):
        self.assertFalse(self.config.get("cloud.enabled"))
        self.assertEqual(self.config.get("budget.cloud_calls_per_hour"), 0)
        self.assertEqual(self.config.get("budget.cloud_calls_per_day"), 0)
        self.assertEqual(self.config.get("budget.cloud_calls_per_night"), 0)

    def test_natural_voice_is_the_default(self):
        self.assertEqual(self.config.get("voice.tts.engine"), "chatterbox")

    def test_interface_avoids_the_known_local_mcp_port(self):
        self.assertEqual(self.config.get("bus.port"), 8797)


if __name__ == "__main__":
    unittest.main()
