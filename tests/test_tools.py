"""The tools JARVIS runs on the real machine, and what they refuse.

This layer exists because V1 needs "create test.txt on my desktop" to actually
create the file — something `agents/permissions.py` deliberately forbids, since
that gate guards *autonomous* work. These tests pin both halves: the tools do
real work, and the separation from the agent gate is intact.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import tools  # noqa: E402
from jarvis.tools import (  # noqa: E402
    HIGH,
    LOW,
    MEDIUM,
    ToolError,
    Workspace,
    WorkspaceViolation,
    classify_command,
    refuses,
)


class ToolCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "alan")
        self.outside = Path(self._tmp.name).resolve() / "disarida.txt"
        self.outside.write_text("DOKUNULMADI\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def run_tool(self, name, *, confirmed=False, **kwargs):
        return tools.run(name, workspace=self.workspace, confirmed=confirmed, **kwargs)

    def write(self, relative, content="icerik\n"):
        return self.run_tool("fs.write", path=relative, content=content, confirmed=True)


class TestWorkspaceContainment(ToolCase):
    def test_a_relative_path_lands_inside(self):
        self.assertEqual(self.workspace.resolve("a/b.txt").parent.parent,
                         self.workspace.root)

    def test_traversal_is_refused(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve("../disarida.txt")

    def test_a_deep_traversal_is_refused(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve("a/b/../../../disarida.txt")

    def test_an_absolute_path_outside_is_refused(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve(str(self.outside))

    def test_an_absolute_path_inside_is_allowed(self):
        inside = self.workspace.root / "var.txt"
        self.assertEqual(self.workspace.resolve(str(inside)), inside)

    def test_a_device_name_is_refused(self):
        for bad in ("NUL", "con.txt", "alt/COM3.dat", "LPT1"):
            with self.subTest(path=bad), self.assertRaises(WorkspaceViolation):
                self.workspace.resolve(bad)

    def test_a_nul_byte_is_refused(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve("a\x00b.txt")

    def test_an_empty_path_is_refused(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve("   ")


class TestRiskGate(ToolCase):
    """Asking about everything trains people to click yes. Asking about the
    right things is the whole point of the tier."""

    def test_a_read_runs_without_being_asked_about(self):
        self.write("not.txt", "merhaba\n")
        result = self.run_tool("fs.read", path="not.txt")
        self.assertTrue(result.ok, result.error)
        self.assertFalse(result.needs_confirmation)
        self.assertEqual(result.output, "merhaba\n")

    def test_a_write_without_confirmation_does_not_happen(self):
        result = self.run_tool("fs.write", path="yeni.txt", content="x")
        self.assertFalse(result.ok)
        self.assertTrue(result.needs_confirmation)
        self.assertEqual(result.risk, MEDIUM)
        self.assertFalse((self.workspace.root / "yeni.txt").exists(),
                         "onay alinmadan dosya olusturulmus")

    def test_the_same_write_happens_once_confirmed(self):
        result = self.run_tool("fs.write", path="yeni.txt", content="x", confirmed=True)
        self.assertTrue(result.ok, result.error)
        self.assertEqual((self.workspace.root / "yeni.txt").read_text(encoding="utf-8"), "x")

    def test_a_refusal_carries_what_was_asked_for(self):
        """The UI has to be able to show the user what it is confirming."""
        result = self.run_tool("fs.write", path="yeni.txt", content="x")
        self.assertEqual(result.detail["path"], "yeni.txt")

    def test_an_unknown_tool_fails_without_raising(self):
        result = self.run_tool("yok.boyle.bir.arac")
        self.assertFalse(result.ok)
        self.assertIn("bilinmeyen araç", result.error)

    def test_every_registered_tool_declares_a_known_risk(self):
        for entry in tools.REGISTRY.values():
            with self.subTest(tool=entry.name):
                self.assertIn(entry.risk, tools.RISKS)
                self.assertTrue(entry.summary, f"{entry.name} özet taşımıyor")

    def test_reads_are_low_and_writes_are_not(self):
        self.assertEqual(tools.get("fs.read").risk, LOW)
        self.assertEqual(tools.get("fs.list").risk, LOW)
        self.assertEqual(tools.get("fs.write").risk, MEDIUM)
        self.assertEqual(tools.get("fs.move").risk, MEDIUM)


class TestFilesystemTools(ToolCase):
    def test_write_then_read_round_trips(self):
        self.write("klasor/dosya.txt", "içerik satırı\n")
        result = self.run_tool("fs.read", path="klasor/dosya.txt")
        self.assertEqual(result.output, "içerik satırı\n")

    def test_listing_shows_what_is_there(self):
        self.write("a.txt")
        self.write("b.txt")
        result = self.run_tool("fs.list", path=".")
        self.assertIn("a.txt", result.output)
        self.assertIn("b.txt", result.output)

    def test_search_finds_by_pattern(self):
        self.write("derin/alt/rapor.pdf.txt")
        self.write("baska.md")
        result = self.run_tool("fs.search", pattern="*.md")
        self.assertIn("baska.md", result.output)
        self.assertNotIn("rapor", result.output)

    def test_reading_something_that_is_not_there_is_a_clear_error(self):
        result = self.run_tool("fs.read", path="yok.txt")
        self.assertFalse(result.ok)
        self.assertIn("bulunamadı", result.error)

    def test_reading_a_directory_is_a_clear_error(self):
        self.run_tool("fs.mkdir", path="bir_klasor", confirmed=True)
        result = self.run_tool("fs.read", path="bir_klasor")
        self.assertFalse(result.ok)
        self.assertIn("dosya değil", result.error)

    def test_move_does_not_overwrite_silently(self):
        self.write("bir.txt", "1")
        self.write("iki.txt", "2")
        result = self.run_tool("fs.move", source="bir.txt", destination="iki.txt",
                               confirmed=True)
        self.assertFalse(result.ok)
        self.assertIn("zaten var", result.error)
        self.assertEqual((self.workspace.root / "iki.txt").read_text(encoding="utf-8"), "2")

    def test_a_write_outside_the_workspace_touches_nothing(self):
        result = self.run_tool("fs.write", path=str(self.outside), content="EZILDI",
                               confirmed=True)
        self.assertFalse(result.ok)
        self.assertEqual(self.outside.read_text(encoding="utf-8"), "DOKUNULMADI\n")

    def test_append_adds_rather_than_replaces(self):
        self.write("log.txt", "bir\n")
        self.run_tool("fs.write", path="log.txt", content="iki\n", append=True,
                      confirmed=True)
        self.assertEqual(self.run_tool("fs.read", path="log.txt").output, "bir\niki\n")


class TestCommandClassification(unittest.TestCase):
    """Pure, so a command's risk can be shown before anything runs."""

    def test_ordinary_commands_are_low(self):
        for command in ("python --version", "git status", "dir", "echo merhaba"):
            with self.subTest(command=command):
                self.assertEqual(classify_command(command), LOW)
                self.assertEqual(refuses(command), "")

    def test_changing_commands_need_confirmation(self):
        for command in ("del rapor.txt", "pip install requests",
                        "git push origin master", "taskkill /pid 100"):
            with self.subTest(command=command):
                self.assertEqual(classify_command(command), MEDIUM)

    def test_catastrophic_commands_are_refused_not_offered(self):
        for command in ("format c:", "diskpart", "shutdown /s /t 0",
                        "reg delete HKLM\\Software\\X", "vssadmin delete shadows",
                        "Set-MpPreference -DisableRealtimeMonitoring $true",
                        "rm -rf /", "rm -r -f /"):
            with self.subTest(command=command):
                self.assertEqual(classify_command(command), HIGH)
                self.assertTrue(refuses(command), f"{command} reddedilmeliydi")


class TestShellTool(ToolCase):
    def test_a_command_runs_and_reports_its_output(self):
        result = self.run_tool("shell.run",
                               command=f'"{sys.executable}" -c "print(7*6)"',
                               confirmed=True)
        self.assertTrue(result.ok, result.error)
        self.assertIn("42", result.output)
        self.assertEqual(result.detail["exit_code"], 0)

    def test_a_failing_command_is_reported_as_failing(self):
        result = self.run_tool("shell.run",
                               command=f'"{sys.executable}" -c "raise SystemExit(3)"',
                               confirmed=True)
        self.assertFalse(result.ok)
        self.assertEqual(result.detail["exit_code"], 3)

    def test_stderr_is_kept_separately(self):
        result = self.run_tool(
            "shell.run",
            command=f'"{sys.executable}" -c "import sys; sys.stderr.write(\'bozuk\')"',
            confirmed=True)
        self.assertIn("bozuk", result.detail["stderr"])

    def test_a_command_that_hangs_is_stopped(self):
        result = self.run_tool(
            "shell.run",
            command=f'"{sys.executable}" -c "import time; time.sleep(30)"',
            timeout_s=2, confirmed=True)
        self.assertFalse(result.ok)
        self.assertTrue(result.detail.get("timed_out"))
        self.assertIn("durduruldu", result.error)

    def test_a_timeout_leaves_no_grandchild_running(self):
        """`shell=True` means the child is a shell and the work is *its* child.
        Killing only the handle we hold leaves that grandchild alive after
        JARVIS is closed, which is the thing V1 promises does not happen.

        The grandchild would write a file after the timeout has passed. The
        file staying absent is the evidence it was stopped.
        """
        import time as clock

        canary = self.workspace.root / "hayalet.txt"
        (self.workspace.root / "torun.py").write_text(
            "import pathlib, time\n"
            "time.sleep(5)\n"
            f"pathlib.Path(r'{canary}').write_text('hayatta')\n",
            encoding="utf-8")
        (self.workspace.root / "ebeveyn.py").write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, 'torun.py'])\n"
            "time.sleep(30)\n",
            encoding="utf-8")

        result = self.run_tool("shell.run",
                               command=f'"{sys.executable}" ebeveyn.py',
                               timeout_s=2, confirmed=True)
        self.assertFalse(result.ok)
        self.assertTrue(result.detail.get("timed_out"), result.detail)

        clock.sleep(7)
        self.assertFalse(canary.exists(),
                         "zaman asimindan sonra torun surec yasamaya devam etmis")

    def test_a_catastrophic_command_is_refused_even_when_confirmed(self):
        """Confirmation is not a key that opens every door."""
        result = self.run_tool("shell.run", command="format c: /q", confirmed=True)
        self.assertFalse(result.ok)
        self.assertTrue(result.detail.get("refused"))
        self.assertEqual(result.risk, HIGH)

    def test_a_command_runs_inside_the_workspace(self):
        self.write("burada.txt")
        result = self.run_tool(
            "shell.run",
            command=f'"{sys.executable}" -c "import os; print(os.path.isfile(\'burada.txt\'))"',
            confirmed=True)
        self.assertIn("True", result.output)


class TestSeparationFromTheAgentGate(unittest.TestCase):
    """Two trust bases, two layers. A hole in one must not be a hole in both."""

    def test_the_agent_gate_still_refuses_shell_and_writes(self):
        from jarvis.agents.permissions import FS_WRITE, SHELL, Grant

        asked = frozenset({SHELL, FS_WRITE})
        self.assertFalse(Grant.build("gece", asked).capabilities)
        self.assertTrue(Grant.build("gece", asked, sandboxed=True).capabilities)

    def test_the_tool_layer_does_not_import_the_agent_grant_system(self):
        import inspect

        source = inspect.getsource(tools)
        self.assertNotIn("from ..agents", source)
        self.assertNotIn("permissions import", source)

    def test_self_modification_is_still_off(self):
        from jarvis.lab.promotion import ALLOW_SELF_MODIFICATION

        self.assertFalse(ALLOW_SELF_MODIFICATION)


class TestCatalogue(unittest.TestCase):
    def test_the_catalogue_lists_every_tool_with_its_risk(self):
        entries = tools.catalogue()
        self.assertEqual(len(entries), len(tools.REGISTRY))
        for entry in entries:
            self.assertIn(entry["risk"], tools.RISKS)
            self.assertTrue(entry["summary"])

    def test_the_expected_v1_tools_exist(self):
        expected = {"fs.list", "fs.read", "fs.search", "fs.stat", "fs.write",
                    "fs.mkdir", "fs.move", "fs.copy", "shell.run", "system.info"}
        self.assertLessEqual(expected, set(tools.names()))


if __name__ == "__main__":
    unittest.main()



class TestArgumentsAreCheckedBeforeAnythingRuns(ToolCase):
    """Found by running it, not by reading it.

    The model called `system.info` with an argument the tool does not take. The
    TypeError travelled past `run()` — which only caught ToolError, OSError and
    ValueError — through the assistant loop, and reached the interface as a
    spinner that never stopped. A tool call that does not fit its tool is a
    refusal with a sentence the model can act on.
    """

    def test_an_argument_the_tool_does_not_take_is_refused(self):
        result = self.run_tool("system.info", info=True)
        self.assertFalse(result.ok)
        self.assertIn("almıyor", result.error)
        self.assertIn("info", result.error)

    def test_the_refusal_says_what_the_tool_does_accept(self):
        result = self.run_tool("fs.read", yol="a.txt")
        self.assertFalse(result.ok)
        self.assertIn("path", result.error)

    def test_a_missing_required_argument_is_refused(self):
        result = self.run_tool("fs.read")
        self.assertFalse(result.ok)
        self.assertIn("eksik", result.error)

    def test_optional_arguments_may_be_left_out(self):
        self.write("v.txt", "x")
        self.assertTrue(self.run_tool("fs.list").ok)

    def test_a_bad_call_is_refused_rather_than_confirmed(self):
        """A malformed MEDIUM call must not be put to the user as a question."""
        result = self.run_tool("fs.write", path="a.txt", content="x", mode="append")
        self.assertFalse(result.ok)
        self.assertFalse(result.needs_confirmation)
        self.assertFalse((self.workspace.root / "a.txt").exists())

    def test_anything_a_tool_raises_becomes_a_failed_result(self):
        import jarvis.tools as tools_module

        def explode(*, workspace):
            raise RuntimeError("beklenmedik")

        tools_module.REGISTRY["test.patlar"] = tools_module.Tool(
            "test.patlar", tools_module.LOW, "patlar", explode)
        try:
            result = self.run_tool("test.patlar")
        finally:
            tools_module.REGISTRY.pop("test.patlar", None)
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.error)

    def test_check_arguments_is_pure(self):
        from jarvis.tools import check_arguments, get

        self.assertEqual(check_arguments(get("fs.list"), {"path": "."}), "")
        self.assertTrue(check_arguments(get("fs.list"), {"bilinmeyen": 1}))
