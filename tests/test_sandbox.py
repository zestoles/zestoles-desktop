"""Sandbox containment tests.

This is the file that decides whether `shell` and `fs.write` were safe to enable.
Every rule in jarvis/lab/sandbox.py has a case here, and the escape tests write a
real canary file outside the sandbox and then try, in each documented way, to
reach it. A containment rule with no test is a rule the next refactor will quietly
remove.

Symlink and junction tests need Windows developer mode or admin rights to create
the link. Where that is unavailable the test skips rather than passes — a skipped
test is visibly untested; a silently passing one is a false assurance.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.agents.permissions import (  # noqa: E402
    FS_WRITE,
    MEMORY_READ,
    SHELL,
    Grant,
)
from jarvis.lab.sandbox import Sandbox, SandboxLimits, SandboxViolation  # noqa: E402

CANARY = "GIZLI-KANARYA-DOSYASI"


class SandboxCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.outside = self.base / "disarisi"
        self.outside.mkdir()
        self.canary = self.outside / "gizli.txt"
        self.canary.write_text(CANARY, encoding="utf-8")
        self.box = Sandbox(self.base / "kutu", limits=SandboxLimits(timeout_s=30))

    def tearDown(self):
        self._tmp.cleanup()

    def assert_canary_intact(self):
        self.assertEqual(self.canary.read_text(encoding="utf-8"), CANARY,
                         "sandbox dışındaki dosya değiştirildi")

    def try_symlink(self, link: Path, target: Path, *, directory: bool = False) -> bool:
        try:
            link.symlink_to(target, target_is_directory=directory)
            return True
        except (OSError, NotImplementedError):
            return False


class TestPathTraversal(SandboxCase):
    def test_parent_traversal_is_refused(self):
        for attempt in ("../gizli.txt", "..\\gizli.txt",
                        "../disarisi/gizli.txt", "a/../../disarisi/gizli.txt",
                        "a/b/c/../../../../disarisi/gizli.txt"):
            with self.subTest(path=attempt):
                with self.assertRaises(SandboxViolation):
                    self.box.resolve(attempt)

    def test_deep_traversal_to_system_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.resolve("../" * 12 + "Windows/System32/drivers/etc/hosts")

    def test_traversal_that_returns_inside_is_still_refused(self):
        """a/../b stays inside, but allowing '..' at all invites the ones that don't."""
        with self.assertRaises(SandboxViolation):
            self.box.resolve("a/../b.txt")

    def test_reading_outside_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.read("../disarisi/gizli.txt")

    def test_writing_outside_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.write("../disarisi/gizli.txt", "ele geçirildi")
        self.assert_canary_intact()

    def test_removing_outside_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.remove("../disarisi/gizli.txt")
        self.assertTrue(self.canary.exists())


class TestAbsoluteAndDrivePaths(SandboxCase):
    def test_absolute_windows_path_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.resolve("C:\\Windows\\System32\\config\\SAM")

    def test_absolute_posix_path_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.resolve("/etc/passwd")

    def test_canary_by_absolute_path_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.write(str(self.canary), "ele geçirildi")
        self.assert_canary_intact()

    def test_drive_relative_path_is_refused(self):
        """C:config.sys has no separator and still leaves the sandbox."""
        with self.assertRaises(SandboxViolation):
            self.box.resolve("C:config.sys")

    def test_unc_path_is_refused(self):
        for attempt in ("\\\\sunucu\\paylasim\\dosya.txt", "//sunucu/paylasim/dosya.txt",
                        "\\\\?\\C:\\Windows\\win.ini"):
            with self.subTest(path=attempt):
                with self.assertRaises(SandboxViolation):
                    self.box.resolve(attempt)


class TestWindowsSpecialNames(SandboxCase):
    def test_reserved_device_names_are_refused(self):
        for name in ("CON", "NUL", "PRN", "AUX", "COM1", "LPT1",
                     "con.txt", "nul.log", "alt/COM3.dat"):
            with self.subTest(name=name):
                with self.assertRaises(SandboxViolation):
                    self.box.resolve(name)

    def test_alternate_data_stream_is_refused(self):
        """notes.txt:hidden writes a second stream that a directory listing misses."""
        with self.assertRaises(SandboxViolation):
            self.box.resolve("notlar.txt:gizli")

    def test_nul_byte_in_path_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.resolve("dosya\x00.txt")

    def test_empty_path_is_refused(self):
        for attempt in ("", "   "):
            with self.subTest(path=attempt):
                with self.assertRaises(SandboxViolation):
                    self.box.resolve(attempt)


class TestSymlinkEscape(SandboxCase):
    def test_symlink_to_outside_file_cannot_be_read(self):
        link = self.box.root / "kacis.txt"
        if not self.try_symlink(link, self.canary):
            self.skipTest("sembolik bağ oluşturulamıyor (geliştirici modu kapalı)")
        with self.assertRaises(SandboxViolation):
            self.box.read("kacis.txt")

    def test_symlink_to_outside_file_cannot_be_written(self):
        link = self.box.root / "kacis.txt"
        if not self.try_symlink(link, self.canary):
            self.skipTest("sembolik bağ oluşturulamıyor")
        with self.assertRaises(SandboxViolation):
            self.box.write("kacis.txt", "ele geçirildi")
        self.assert_canary_intact()

    def test_symlinked_directory_cannot_be_traversed(self):
        link = self.box.root / "disari"
        if not self.try_symlink(link, self.outside, directory=True):
            self.skipTest("dizin bağı oluşturulamıyor")
        with self.assertRaises(SandboxViolation):
            self.box.write("disari/gizli.txt", "ele geçirildi")
        self.assert_canary_intact()

    def test_write_through_an_internal_symlink_is_refused(self):
        """Even pointing inside: a link is not the file the caller named."""
        real = self.box.write("gercek.txt", "orijinal")
        link = self.box.root / "bag.txt"
        if not self.try_symlink(link, real):
            self.skipTest("sembolik bağ oluşturulamıyor")
        with self.assertRaises(SandboxViolation):
            self.box.write("bag.txt", "değiştirildi")
        self.assertEqual(real.read_text(encoding="utf-8"), "orijinal")

    def test_junction_to_outside_cannot_be_traversed(self):
        if os.name != "nt":
            self.skipTest("junction yalnızca Windows")
        junction = self.box.root / "kavsak"
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(self.outside)],
                                capture_output=True, text=True)
        if result.returncode != 0 or not junction.exists():
            self.skipTest("junction oluşturulamadı")
        with self.assertRaises(SandboxViolation):
            self.box.read("kavsak/gizli.txt")
        self.assert_canary_intact()


class TestHardLinkEscape(SandboxCase):
    """The escape that path resolution cannot see, and that needs no privileges.

    A hard link is a second name for a file, not a pointer to it, so a hard link
    inside the sandbox naming a file outside resolves to a contained path. This was
    verified as a working escape before the nlink check existed: the canary was
    overwritten through one.
    """

    def link_or_skip(self, link: Path, target: Path) -> None:
        try:
            os.link(target, link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("sabit bağ oluşturulamıyor (dosya sistemi desteklemiyor)")

    def test_path_resolution_alone_does_not_catch_it(self):
        """Documents why the nlink check exists: resolve() says this is inside."""
        link = self.box.root / "sabit.txt"
        self.link_or_skip(link, self.canary)
        self.assertTrue(link.resolve().is_relative_to(self.box.root))

    def test_writing_through_a_hard_link_is_refused(self):
        link = self.box.root / "sabit.txt"
        self.link_or_skip(link, self.canary)
        with self.assertRaises(SandboxViolation):
            self.box.write("sabit.txt", "ele geçirildi")
        self.assert_canary_intact()

    def test_reading_through_a_hard_link_is_refused(self):
        link = self.box.root / "sabit.txt"
        self.link_or_skip(link, self.canary)
        with self.assertRaises(SandboxViolation):
            self.box.read("sabit.txt")

    def test_a_hard_link_in_a_subdirectory_is_refused(self):
        self.box.mkdir("alt")
        link = self.box.root / "alt" / "sabit.txt"
        self.link_or_skip(link, self.canary)
        with self.assertRaises(SandboxViolation):
            self.box.write("alt/sabit.txt", "ele geçirildi")
        self.assert_canary_intact()

    def test_an_ordinary_file_has_one_name_and_is_allowed(self):
        self.box.write("normal.txt", "içerik")
        self.assertEqual((self.box.root / "normal.txt").stat().st_nlink, 1)
        self.assertEqual(self.box.read("normal.txt"), "içerik")


class TestRootHardening(SandboxCase):
    def test_drive_root_is_refused_as_sandbox_root(self):
        root = Path("C:\\") if os.name == "nt" else Path("/")
        with self.assertRaises(SandboxViolation):
            Sandbox(root)

    def test_windows_directory_is_refused_as_sandbox_root(self):
        if os.name != "nt":
            self.skipTest("Windows'a özgü")
        with self.assertRaises(SandboxViolation):
            Sandbox(Path(os.environ.get("SystemRoot", "C:\\Windows")))

    def test_root_itself_cannot_be_removed(self):
        with self.assertRaises(SandboxViolation):
            self.box.remove(".")


class TestNormalOperation(SandboxCase):
    def test_write_and_read_round_trip(self):
        self.box.write("notlar.txt", "merhaba")
        self.assertEqual(self.box.read("notlar.txt"), "merhaba")

    def test_nested_directories_are_created(self):
        self.box.write("a/b/c.txt", "derin")
        self.assertEqual(self.box.read("a/b/c.txt"), "derin")

    def test_written_file_lands_inside_the_root(self):
        path = self.box.write("x.txt", "içerik")
        self.assertTrue(path.is_relative_to(self.box.root))

    def test_listdir_and_exists(self):
        self.box.write("bir.txt", "1")
        self.box.write("iki.txt", "2")
        self.assertEqual(self.box.listdir(), ["bir.txt", "iki.txt"])
        self.assertTrue(self.box.exists("bir.txt"))
        self.assertFalse(self.box.exists("yok.txt"))

    def test_exists_is_false_for_an_escape_attempt(self):
        self.assertFalse(self.box.exists("../disarisi/gizli.txt"))

    def test_reset_empties_without_destroying(self):
        self.box.write("a/b.txt", "x")
        self.box.reset()
        self.assertEqual(self.box.listdir(), [])
        self.assertTrue(self.box.root.exists())

    def test_turkish_filenames_work(self):
        self.box.write("çalışma/ölçüm-günü.txt", "veri")
        self.assertEqual(self.box.read("çalışma/ölçüm-günü.txt"), "veri")


class TestLimits(SandboxCase):
    def test_oversized_write_is_refused(self):
        box = Sandbox(self.base / "kucuk", limits=SandboxLimits(max_file_bytes=100))
        with self.assertRaises(SandboxViolation):
            box.write("buyuk.txt", "x" * 500)

    def test_file_count_limit_is_enforced(self):
        box = Sandbox(self.base / "sayili", limits=SandboxLimits(max_files=3))
        for index in range(3):
            box.write(f"{index}.txt", "x")
        with self.assertRaises(SandboxViolation):
            box.write("fazla.txt", "x")

    def test_reading_an_absent_file_raises(self):
        with self.assertRaises(SandboxViolation):
            self.box.read("yok.txt")


class TestCommandExecution(SandboxCase):
    def test_disallowed_command_is_refused(self):
        for attempt in (["cmd", "/c", "dir"], ["powershell", "-c", "ls"],
                        ["curl", "https://example.com"], ["notepad.exe"]):
            with self.subTest(command=attempt[0]):
                with self.assertRaises(SandboxViolation):
                    self.box.run(attempt)

    def test_empty_command_is_refused(self):
        with self.assertRaises(SandboxViolation):
            self.box.run([])

    def test_allowlist_ignores_path_and_extension(self):
        """C:\\evil\\python.exe must not pass as 'python'."""
        with self.assertRaises(SandboxViolation):
            self.box.run(["C:\\uydurma\\dizin\\kotu.exe"])

    def test_allowed_command_runs(self):
        result = self.box.run([sys.executable, "-c", "print('merhaba')"])
        self.assertTrue(result.ok, result.stderr)
        self.assertIn("merhaba", result.stdout)

    def test_command_runs_inside_the_sandbox(self):
        result = self.box.run([sys.executable, "-c", "import os; print(os.getcwd())"])
        self.assertTrue(Path(result.stdout.strip()).resolve().is_relative_to(self.box.root))

    def test_shell_metacharacters_are_not_interpreted(self):
        """shell=False means '&& del' is an argument, not a second command."""
        result = self.box.run([sys.executable, "-c", "import sys; print(sys.argv[1])",
                               "&& echo ele-gecirildi"])
        self.assertIn("&& echo ele-gecirildi", result.stdout)

    def test_secrets_are_scrubbed_from_the_environment(self):
        os.environ["TEST_SECRET_TOKEN"] = "cok-gizli"
        try:
            result = self.box.run([sys.executable, "-c",
                                   "import os; print(os.environ.get('TEST_SECRET_TOKEN', 'YOK'))"])
            self.assertIn("YOK", result.stdout)
        finally:
            os.environ.pop("TEST_SECRET_TOKEN", None)

    def test_sandbox_marker_is_present(self):
        result = self.box.run([sys.executable, "-c",
                               "import os; print(os.environ.get('JARVIS_SANDBOX'))"])
        self.assertIn("1", result.stdout)

    def test_timeout_stops_a_hanging_command(self):
        box = Sandbox(self.base / "yavas", limits=SandboxLimits(timeout_s=2))
        result = box.run([sys.executable, "-c", "import time; time.sleep(60)"])
        self.assertTrue(result.timed_out)
        self.assertLess(result.duration_ms, 30_000)

    def test_output_is_capped(self):
        box = Sandbox(self.base / "gurultulu", limits=SandboxLimits(max_output=500))
        result = box.run([sys.executable, "-c", "print('x' * 100000)"])
        self.assertLessEqual(len(result.stdout), 500)

    def test_a_failing_command_reports_rather_than_raises(self):
        result = self.box.run([sys.executable, "-c", "import sys; sys.exit(3)"])
        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 3)


class TestCommandCannotEscape(SandboxCase):
    def test_a_command_writing_outside_does_not_change_the_canary(self):
        """The allowlist bounds which programs start, not what they then do.

        Python is allowed and can obviously write anywhere the user can — this
        test records that honestly rather than implying containment it does not
        have. What it does verify is that the sandbox's own working directory and
        path handling never help.
        """
        result = self.box.run([sys.executable, "-c", "print(open('gizli.txt', 'w').write('x'))"])
        self.assertTrue(result.ok or result.returncode != 0)
        self.assert_canary_intact()
        self.assertTrue((self.box.root / "gizli.txt").exists())


class TestGrantGating(unittest.TestCase):
    def test_shell_stays_denied_without_a_sandbox(self):
        grant = Grant.build("x", frozenset({SHELL, FS_WRITE, MEMORY_READ}))
        self.assertFalse(grant.allows(SHELL))
        self.assertFalse(grant.allows(FS_WRITE))
        self.assertTrue(grant.allows(MEMORY_READ))

    def test_sandboxed_grant_enables_them(self):
        grant = Grant.build("x", frozenset({SHELL, FS_WRITE}), sandboxed=True)
        self.assertTrue(grant.allows(SHELL))
        self.assertTrue(grant.allows(FS_WRITE))
        self.assertTrue(grant.sandboxed)

    def test_sandboxing_does_not_unlock_the_cloud_tier(self):
        from jarvis.agents.permissions import CLOUD_BRAIN

        grant = Grant.build("x", frozenset({CLOUD_BRAIN}), sandboxed=True)
        self.assertFalse(grant.allows(CLOUD_BRAIN))

    def test_a_capability_never_requested_is_not_granted(self):
        grant = Grant.build("x", frozenset({MEMORY_READ}), sandboxed=True)
        self.assertFalse(grant.allows(SHELL))


if __name__ == "__main__":
    unittest.main()
