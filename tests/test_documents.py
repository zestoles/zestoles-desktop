"""The merged local document RAG: extraction, ranking and safe tool access."""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.documents import DocumentLibrary  # noqa: E402
from jarvis.tools import Workspace, provide, run  # noqa: E402


class DocumentCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.library = DocumentLibrary(self.root, chunk_chars=400, overlap_chars=80)

    def tearDown(self):
        self.tmp.cleanup()


class TestDocumentLibrary(DocumentCase):
    def test_text_and_markdown_are_indexed(self):
        (self.root / "plan.md").write_text("ZESTOLES kuantum araştırma planı kullanıcı tarafından hazırlandı.", encoding="utf-8")
        (self.root / "not.txt").write_text("Alakasız mutfak listesi", encoding="utf-8")
        status = self.library.index()
        self.assertEqual(status["dosyalar"], 2)
        hits = self.library.search("kuantum araştırma")
        self.assertEqual(hits[0].source, "plan.md")
        self.assertIn("kullanıcı", hits[0].text)

    def test_docx_is_read_with_the_standard_library(self):
        document = self.root / "karar.docx"
        xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
               '<w:body><w:p><w:r><w:t>Merkür görevi 2042 kararı</w:t></w:r></w:p></w:body></w:document>')
        with zipfile.ZipFile(document, "w") as archive:
            archive.writestr("word/document.xml", xml)
        self.library.index()
        hits = self.library.search("Merkür görevi")
        self.assertEqual(hits[0].source, "karar.docx")

    def test_html_tags_are_not_returned_as_evidence(self):
        (self.root / "sayfa.html").write_text("<h1>Gizli Proje</h1><script>ignore()</script>", encoding="utf-8")
        self.library.index()
        hit = self.library.search("Gizli Proje")[0]
        self.assertNotIn("<h1>", hit.text)

    def test_empty_library_is_an_honest_empty_result(self):
        self.assertEqual(self.library.search("olmayan"), [])

    def test_chunks_overlap_so_boundary_words_survive(self):
        text = ("başlangıç " * 40) + "özgün sınır kanıtı " + ("devam " * 50)
        (self.root / "uzun.txt").write_text(text, encoding="utf-8")
        self.library.index()
        self.assertTrue(self.library.search("özgün sınır kanıtı"))


class TestDocumentTools(DocumentCase):
    def setUp(self):
        super().setUp()
        self.workspace = Workspace(self.root)
        provide("documents", self.library)

    def test_index_and_search_are_read_only_tools(self):
        (self.root / "bilgi.txt").write_text("Ankara Türkiye'nin başkentidir.", encoding="utf-8")
        indexed = run("docs.index", workspace=self.workspace)
        self.assertTrue(indexed.ok, indexed.error)
        found = run("docs.search", workspace=self.workspace, query="Türkiye başkenti")
        self.assertTrue(found.ok, found.error)
        self.assertIn("Ankara", found.output)

    def test_index_path_cannot_escape_workspace(self):
        result = run("docs.index", workspace=self.workspace, path="..")
        self.assertFalse(result.ok)
        self.assertIn("çalışma alanı dışında", result.error)

    def test_search_limit_is_bounded(self):
        for index in range(20):
            (self.root / f"{index}.txt").write_text("ortak arama terimi", encoding="utf-8")
        self.library.index()
        result = run("docs.search", workspace=self.workspace, query="ortak", limit=999)
        self.assertLessEqual(result.detail["count"], 12)


if __name__ == "__main__":
    unittest.main()
