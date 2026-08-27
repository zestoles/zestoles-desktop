from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from jarvis import tools
from jarvis.tools import MEDIUM, Workspace
from jarvis.tools.windows import PROTECTED_PROCESSES, resolve_application


class WindowsToolRegistrationTest(unittest.TestCase):
    def test_desktop_tools_are_registered(self):
        for name in ("system.time", "system.processes", "app.open", "app.close",
                     "clipboard.read", "clipboard.write", "fs.trash"):
            self.assertIn(name, tools.names())

    def test_side_effects_need_confirmation(self):
        for name in ("app.open", "app.close", "clipboard.write", "fs.trash"):
            self.assertEqual(tools.get(name).risk, MEDIUM)

    def test_app_alias_is_resolved_without_a_shell(self):
        with patch("jarvis.tools.windows.shutil.which", return_value="C:/Windows/notepad.exe"):
            self.assertEqual(resolve_application("Not Defteri"), "C:/Windows/notepad.exe")

    def test_open_uses_argument_vector_and_no_shell(self):
        workspace = Workspace(Path.cwd())
        proc = Mock(pid=42)
        with patch("jarvis.tools.windows.resolve_application", return_value="app.exe"), \
             patch("jarvis.tools.windows.subprocess.Popen", return_value=proc) as popen:
            result = tools.run("app.open", workspace=workspace, confirmed=True,
                               application="uygulama", arguments=["--safe"])
        self.assertTrue(result.ok)
        self.assertEqual(popen.call_args.args[0], ["app.exe", "--safe"])
        self.assertFalse(popen.call_args.kwargs["shell"])

    def test_protected_processes_are_not_closeable(self):
        workspace = Workspace(Path.cwd())
        name = next(iter(PROTECTED_PROCESSES))
        result = tools.run("app.close", workspace=workspace, confirmed=True,
                           process=name)
        self.assertFalse(result.ok)
        self.assertIn("korunan", result.error)


if __name__ == "__main__":
    unittest.main()
