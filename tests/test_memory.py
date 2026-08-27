"""Memory tests.

The provenance gate is the important one. On the first real run the local model
invented a Roblox service called AsyncResultStorage, described it confidently, and
the distiller filed it in the vault as knowledge. Left alone, that note would have
been recalled in later sessions and repeated with growing confidence — the exact
mechanism by which a self-learning system gets worse instead of better.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.memory.distill import (  # noqa: E402
    SELF_SOURCED,
    SUMMARY_SOURCED,
    UNVERIFIED_SOURCES,
    USER_SOURCED,
    Fact,
    accept,
    tidy_title,
)
from jarvis.memory.store import Store, chunk_markdown  # noqa: E402
from jarvis.memory.vault import Vault, parse_frontmatter, render  # noqa: E402
from jarvis.text import slugify  # noqa: E402


class TestProvenanceGate(unittest.TestCase):
    def test_user_stated_fact_is_kept(self):
        fact = Fact("proje", "Steal Speed", "Roblox tycoon oyunu.", [], USER_SOURCED)
        ok, why = accept(fact)
        self.assertTrue(ok, why)

    def test_jarvis_claim_is_refused(self):
        """The AsyncResultStorage case, as a test."""
        fact = Fact(
            "bilgi",
            "AsyncResultStorage Alternatifi",
            "ProfileService yerine AsyncResultStorage kullanmak yazmaları serialize eder.",
            [],
            SELF_SOURCED,
        )
        ok, why = accept(fact)
        self.assertFalse(ok)
        self.assertIn("doğrulanmadan", why)

    def test_jarvis_claim_refused_for_every_kind(self):
        for kind in ("kisi", "proje", "deneyim", "bilgi"):
            with self.subTest(kind=kind):
                ok, _ = accept(Fact(kind, "başlık", "içerik", [], SELF_SOURCED))
                self.assertFalse(ok)

    def test_empty_content_is_refused(self):
        ok, _ = accept(Fact("proje", "Başlık", "   ", [], USER_SOURCED))
        self.assertFalse(ok)

    def test_unknown_kind_is_refused(self):
        ok, _ = accept(Fact("uydurma", "Başlık", "içerik", [], USER_SOURCED))
        self.assertFalse(ok)


class TestUnverifiedSources(unittest.TestCase):
    """Session summaries record what was said, not what is true.

    They bypass accept() so the daily log stays complete, which means the only
    thing stopping a summarised hallucination from being read back as fact is
    their provenance class.
    """

    def test_summaries_count_as_unverified(self):
        self.assertIn(SUMMARY_SOURCED, UNVERIFIED_SOURCES)

    def test_self_claims_count_as_unverified(self):
        self.assertIn(SELF_SOURCED, UNVERIFIED_SOURCES)

    def test_user_statements_are_not_flagged(self):
        self.assertNotIn(USER_SOURCED, UNVERIFIED_SOURCES)


class TestVault(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Vault(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_then_read_round_trips(self):
        self.vault.write("proje", "Steal Speed", "Roblox tycoon.", tags=["roblox"])
        note = self.vault.find("proje", "steal-speed")
        self.assertIsNotNone(note)
        self.assertEqual(note.title, "Steal Speed")
        self.assertEqual(note.tags, ["roblox"])
        self.assertEqual(note.source, "kullanici")

    def test_append_preserves_earlier_content(self):
        self.vault.write("proje", "Steal Speed", "İlk bilgi.")
        self.vault.append("proje", "Steal Speed", "Sonradan öğrenilen.")
        note = self.vault.find("proje", "steal-speed")
        self.assertIn("İlk bilgi.", note.body)
        self.assertIn("Sonradan öğrenilen.", note.body)

    def test_source_survives_a_round_trip(self):
        self.vault.write("bilgi", "Şüpheli", "içerik", source="jarvis")
        self.assertEqual(self.vault.find("bilgi", "supheli").source, "jarvis")

    def test_append_does_not_launder_an_unverified_source(self):
        """Appending must not quietly upgrade a summary note to user-sourced."""
        self.vault.write("gunluk", "Günlük", "ilk", source=SUMMARY_SOURCED)
        self.vault.append("gunluk", "Günlük", "ikinci")
        self.assertEqual(self.vault.find("gunluk", "gunluk").source, SUMMARY_SOURCED)

    def test_wikilinks_are_extracted(self):
        self.vault.write("proje", "A", "Bağlantı: [[steal-speed]] ve [[roblox|Roblox]].")
        note = self.vault.find("proje", "a")
        self.assertEqual(note.links, ["roblox", "steal-speed"])

    def test_frontmatter_parses_lists(self):
        meta, body = parse_frontmatter(render(self.vault.write("proje", "T", "gövde",
                                                               tags=["a", "b"])))
        self.assertEqual(meta["tags"], ["a", "b"])
        self.assertEqual(body.strip(), "gövde")


class TestChunking(unittest.TestCase):
    def test_short_note_is_one_chunk(self):
        self.assertEqual(len(chunk_markdown("Kısa bir not.")), 1)

    def test_headings_split_sections(self):
        text = "# Bir\n" + "a" * 300 + "\n\n# İki\n" + "b" * 300
        chunks = chunk_markdown(text)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("# Bir"))

    def test_long_section_is_broken_up(self):
        text = "# Başlık\n" + "\n\n".join("paragraf " * 40 for _ in range(6))
        self.assertGreater(len(chunk_markdown(text)), 1)

    def test_empty_input_yields_nothing(self):
        self.assertEqual(chunk_markdown("   "), [])


class TestIndexTransactions(unittest.TestCase):
    def test_embedding_usage_can_write_to_the_shared_database(self):
        """Regression: reindex used to deadlock on its own usage callback."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "memory.db"
            vault = Vault(root / "vault")
            vault.write("proje", "ZESTOLES", "Yerel sesli asistan.")

            class RecordingEmbedder:
                def embed(self, texts):
                    # Budget.record does the same thing: a separate connection
                    # writes to the database while the model call completes.
                    with closing(sqlite3.connect(db, timeout=0.1)) as conn, conn:
                        conn.execute("CREATE TABLE IF NOT EXISTS usage (n INTEGER)")
                        conn.execute("INSERT INTO usage VALUES (?)", (len(texts),))
                    return [[1.0, 0.0] for _ in texts]

            store = Store(db, vault, RecordingEmbedder())
            report = store.reindex()

            self.assertEqual(report["eklendi"], 1)
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)


class TestTitle(unittest.TestCase):
    def test_trailing_punctuation_is_dropped(self):
        self.assertEqual(tidy_title("İlk öncelik oyuncu sayısını artırmak."),
                         "İlk öncelik oyuncu sayısını artırmak")

    def test_long_title_is_cut_on_a_word_boundary(self):
        title = tidy_title("Roblox'ta " + "uzun " * 30 + "başlık")
        self.assertLessEqual(len(title), 60)
        self.assertFalse(title.endswith(" "))

    def test_whitespace_is_collapsed(self):
        self.assertEqual(tidy_title("  iki   boşluk  "), "iki boşluk")


class TestSlug(unittest.TestCase):
    def test_turkish_becomes_readable_ascii(self):
        self.assertEqual(slugify("Görev Kuyruğu ve Ölçüm"), "gorev-kuyrugu-ve-olcum")

    def test_never_returns_empty(self):
        self.assertTrue(slugify("!!!"))


if __name__ == "__main__":
    unittest.main()
