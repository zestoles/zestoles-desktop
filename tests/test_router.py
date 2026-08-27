"""Routing tests.

Every misroute costs something real: sending small talk to Claude burns the
subscription allowance, and sending a design decision to qwen3 produces a
confident, shallow answer. These cases are the record of which is which.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.brain.router import CLOUD, LOCAL, Router, fold  # noqa: E402

SHOULD_BE_LOCAL = [
    "Selam Jarvis, naber?",
    "günaydın",
    "iyi geceler",
    "teşekkürler",
    "tamam",
    "saat kaç",
    "nasılsın",
    "evet devam et",
]

SHOULD_BE_CLOUD = [
    "Roblox'ta bir tycoon oyununda oyuncu verilerini kaydetmek icin DataStore mu "
    "ProfileService mi kullanmaliyim? Kisa gerekce ver.",
    "gece calisma sistemini nasil kuracagiz, planla ve riskleri de yaz",
    "kod calismiyor, hata veriyor, sebebini bulur musun",
    "jarvis/brain/router.py dosyasindaki skorlamayi refactor et",
    "ProfileService yerine kendi datastore wrapper'imi yazsam ne kaybederim",
    "hangisi daha iyi: her istekte yeni baglanti mi, connection pool mu",
    "bu mimariyi olceklenebilir hale getirmek icin ne onerirsin, avantaj ve "
    "dezavantajlariyla birlikte anlat",
]


class TestFold(unittest.TestCase):
    def test_strips_turkish_diacritics(self):
        self.assertEqual(fold("Gerekçe NASIL Değişir"), "gerekce nasil degisir")

    def test_dotted_capital_i(self):
        self.assertEqual(fold("İyi"), "iyi")


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.router = Router()

    def test_small_talk_stays_local(self):
        for message in SHOULD_BE_LOCAL:
            with self.subTest(message=message):
                decision = self.router.decide(message)
                self.assertEqual(decision.tier, LOCAL, decision.reason)

    def test_real_work_goes_to_cloud(self):
        for message in SHOULD_BE_CLOUD:
            with self.subTest(message=message):
                decision = self.router.decide(message)
                self.assertEqual(decision.tier, CLOUD, decision.reason)

    def test_mimari_is_not_a_question_particle(self):
        """'mimari' contains 'mi' — it must not read as a comparison question."""
        decision = self.router.decide("mimari")
        self.assertNotIn("karşılaştırmalı", decision.reason)

    def test_forced_modes_bypass_scoring(self):
        self.assertEqual(Router(mode=LOCAL).decide("kodu refactor et").tier, LOCAL)
        self.assertEqual(Router(mode=CLOUD).decide("selam").tier, CLOUD)


if __name__ == "__main__":
    unittest.main()
