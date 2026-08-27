from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "launcher"


class ZestolesLauncherTest(unittest.TestCase):
    def test_public_launchers_exist(self):
        for path in (ROOT / "ZESTOLES.cmd", LAUNCHER / "ZESTOLES.bat",
                     ROOT / "KURULUM.cmd",
                     LAUNCHER / "zestoles.ps1",
                     LAUNCHER / "install-zestoles.ps1",
                     LAUNCHER / "install-zestoles-shortcut.ps1"):
            self.assertTrue(path.is_file(), str(path))

    def test_manual_mode_opens_the_interface_and_prefers_project_python(self):
        text = (LAUNCHER / "zestoles.ps1").read_text(encoding="ascii")
        self.assertIn("--arayuz", text)
        self.assertIn(".venv\\Scripts\\python.exe", text)
        self.assertNotIn("register-scheduledtask", text.casefold())

    def test_shortcut_uses_the_zestoles_name(self):
        text = (LAUNCHER / "install-zestoles-shortcut.ps1").read_text(
            encoding="ascii")
        self.assertIn("ZESTOLES.lnk", text)
        self.assertIn("ui\\zestoles.ico", text)

    def test_installer_pins_local_models_and_voice_revision(self):
        text = (LAUNCHER / "install-zestoles.ps1").read_text(encoding="ascii")
        self.assertIn("qwen3.5:9b", text)
        self.assertIn("qwen3:14b", text)
        self.assertIn("bge-m3", text)
        self.assertIn("large-v3-turbo", text)
        self.assertIn("varsayilan.wav", text)
        self.assertIn("System.Speech", text)
        self.assertIn("5de7a54aa4e5e2baadb0182dde554908b48b85c2", text)


if __name__ == "__main__":
    unittest.main()
