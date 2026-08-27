"""The desktop entry point, and the things that would silently break it.

V1 is manual-start: the user double-clicks an icon. That makes these scripts
the first thing that runs and the first thing that can fail, usually in ways a
test catches and a person does not notice until the icon does nothing.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "launcher"


class TestLauncherExists(unittest.TestCase):
    def test_the_pieces_are_all_there(self):
        for name in ("jarvis.ps1", "JARVIS.bat", "install-shortcut.ps1"):
            with self.subTest(name=name):
                self.assertTrue((LAUNCHER / name).is_file(), name)

    def test_the_batch_file_calls_the_launcher(self):
        text = (LAUNCHER / "JARVIS.bat").read_text(encoding="ascii")
        self.assertIn("jarvis.ps1", text)
        self.assertIn("-NoProfile", text)

    def test_the_batch_file_finds_the_script_beside_itself(self):
        """%~dp0, not a written-down path: the folder can move."""
        self.assertIn("%~dp0", (LAUNCHER / "JARVIS.bat").read_text(encoding="ascii"))


class TestScriptsSurviveATurkishCodepage(unittest.TestCase):
    """Windows PowerShell 5.1 reads a BOM-less UTF-8 file as the ANSI codepage.
    On a Turkish system the second byte of an em dash decodes to a closing
    quote, ends the string it sits in, and the rest of the file becomes parse
    errors. stop-jarvis.ps1 failed exactly that way once."""

    def test_every_launcher_script_is_pure_ascii(self):
        for script in sorted(LAUNCHER.glob("*.ps1")) + sorted(LAUNCHER.glob("*.bat")):
            with self.subTest(script=script.name):
                offenders = [(i, b) for i, b in enumerate(script.read_bytes())
                             if b > 0x7F]
                self.assertEqual(offenders[:3], [], f"{script.name}: ASCII disi bayt")


class TestNoHardcodedPaths(unittest.TestCase):
    """AT-63: moving the project must not break the shortcut."""

    def scripts(self):
        return sorted(LAUNCHER.glob("*.ps1"))

    def test_the_project_root_is_derived_not_written_down(self):
        text = (LAUNCHER / "jarvis.ps1").read_text(encoding="ascii")
        self.assertIn("$PSScriptRoot", text)

    def test_no_script_hardcodes_the_current_install_location(self):
        for script in self.scripts():
            with self.subTest(script=script.name):
                text = script.read_text(encoding="ascii").lower()
                self.assertNotIn("c:\\jarvis", text)

    def test_no_script_hardcodes_a_user_name(self):
        for script in self.scripts():
            with self.subTest(script=script.name):
                text = script.read_text(encoding="ascii").lower()
                self.assertNotIn("\\users\\private-user", text)
                self.assertNotIn("c:/users/", text)

    def test_the_desktop_is_asked_for_not_assumed(self):
        text = (LAUNCHER / "install-shortcut.ps1").read_text(encoding="ascii")
        self.assertIn("GetFolderPath('Desktop')", text)


class TestManualStartOnly(unittest.TestCase):
    """V1 does not start with Windows, and the launcher must not quietly
    reintroduce that."""

    def test_the_launcher_registers_nothing(self):
        for script in sorted(LAUNCHER.glob("*.ps1")):
            with self.subTest(script=script.name):
                text = script.read_text(encoding="ascii")
                for forbidden in ("Register-ScheduledTask", "New-ScheduledTask",
                                  "schtasks", "CurrentVersion\\Run"):
                    self.assertNotIn(forbidden, text,
                                     f"{script.name}: otomatik baslatma eklenmis")

    def test_the_shortcut_says_it_does_not_autostart(self):
        text = (LAUNCHER / "install-shortcut.ps1").read_text(encoding="ascii")
        self.assertIn("kendiliginden baslamaz", text)


class TestPythonResolution(unittest.TestCase):
    def test_an_explicit_interpreter_still_wins(self):
        text = (LAUNCHER / "jarvis.ps1").read_text(encoding="ascii")
        self.assertIn("JARVIS_PYTHON", text)

    def test_a_zero_byte_store_alias_is_not_accepted(self):
        """The Store alias is a 0-byte file; running it adds a packaged shim and
        an AppX container to the process chain."""
        text = (LAUNCHER / "jarvis.ps1").read_text(encoding="ascii")
        self.assertIn("Length -gt 0", text)

    def test_a_missing_project_is_explained_rather_than_traced(self):
        text = (LAUNCHER / "jarvis.ps1").read_text(encoding="ascii")
        self.assertIn("run.py", text)
        self.assertIn("install-shortcut.ps1", text)


if __name__ == "__main__":
    unittest.main()
