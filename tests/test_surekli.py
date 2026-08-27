"""7/24 omurgasi: --surekli bayragi ve terk izleyicisinin kapatilmasi.

Surekli modun tek farki yasam dongusundedir: son sekme kapandiginda OrphanWatch
hic kurulmaz, dolayisiyla surec kendini kapatmaz. Kapat dugmesi hala calisir --
o bilinçli bir eylemdir. Burada denetlenen sey bu kapinin kaynagin kendisinde
gorunur olmasidir; bir bayragin iki katmani ayristirmasini sozu degil, yapisal
baglantiyi dogrular.
"""

import inspect
import unittest
from pathlib import Path

from jarvis.cli.app import _parser
from jarvis.cli.interface import run_interface

ROOT = Path(__file__).resolve().parents[1]


class SurekliFlagTest(unittest.TestCase):
    def test_flag_defaults_to_off(self):
        args = _parser().parse_args([])
        self.assertFalse(args.surekli)

    def test_flag_turns_on(self):
        self.assertTrue(_parser().parse_args(["--surekli"]).surekli)

    def test_watchdog_only_when_not_surekli(self):
        source = inspect.getsource(run_interface)
        self.assertIn("None if surekli else _watch_for_abandonment", source)

    def test_launcher_passes_the_mode_through(self):
        text = (ROOT / "tools" / "launcher" / "jarvis.ps1").read_text(
            encoding="ascii")
        self.assertIn("param(", text)
        self.assertIn("[switch]$Surekli", text)
        self.assertIn("'--surekli'", text)

    def test_autostart_script_targets_continuous_mode(self):
        script = (ROOT / "tools" / "autostart" / "register-surekli.ps1")
        text = script.read_text(encoding="ascii")   # ascii disi karakter = hata
        self.assertIn("-Surekli", text)
        self.assertIn("JARVIS Surekli", text)


if __name__ == "__main__":
    unittest.main()
