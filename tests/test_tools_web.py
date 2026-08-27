"""Web tools, and the defences they are not allowed to drop.

These wrap the S4 research pipeline instead of reimplementing it. That is the
whole point: a fresh "fetch a page" tool would quietly lose the prompt-injection
defence and the citation discipline that phase built, and nobody would notice
until a page talked the model into something.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import tools  # noqa: E402
from jarvis.tools import LOW, MEDIUM, Workspace  # noqa: E402
from jarvis.tools.web import ALLOWED_SCHEMES, check_url  # noqa: E402


class Hit:
    def __init__(self, title, url):
        self.title = title
        self.url = url
        self.domain = url.split("/")[2] if "//" in url else url


class Page:
    def __init__(self, url):
        self.url = url


class Report:
    def __init__(self, *, synthesis="", pages=None, failures=None,
                 claims=None, injection_sources=None):
        self.question = "soru"
        self.hits = []
        self.pages = pages or []
        self.failures = failures or []
        self.synthesis = synthesis
        self.claims = claims or []
        self.injection_sources = injection_sources or []
        self.duration_ms = 10


class StubResearch:
    def __init__(self, *, hits=None, report=None, raises=None):
        self._hits = hits if hits is not None else []
        self._report = report
        self._raises = raises
        self.asked: list[str] = []

    def search(self, query):
        self.asked.append(query)
        if self._raises:
            raise self._raises
        return self._hits

    def investigate(self, question, **_kwargs):
        self.asked.append(question)
        if self._raises:
            raise self._raises
        return self._report or Report()


class WebToolCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "alan")
        self._saved = dict(tools.SERVICES)
        tools.SERVICES.clear()

    def tearDown(self):
        tools.SERVICES.clear()
        tools.SERVICES.update(self._saved)
        self._tmp.cleanup()

    def run_tool(self, name, *, confirmed=False, **kwargs):
        return tools.run(name, workspace=self.workspace, confirmed=confirmed, **kwargs)


class TestUrlChecking(unittest.TestCase):
    """A browser launch is arbitrary local read or script execution if the
    scheme is not checked."""

    def test_http_and_https_are_allowed(self):
        self.assertEqual(check_url("https://example.com/a"), "")
        self.assertEqual(check_url("http://example.com"), "")

    def test_a_file_url_is_refused(self):
        self.assertIn("http", check_url("file:///C:/Windows/win.ini"))

    def test_a_javascript_url_is_refused(self):
        self.assertTrue(check_url("javascript:alert(1)"))

    def test_other_schemes_are_refused(self):
        for url in ("ftp://x/y", "data:text/html,<b>x", "mailto:a@b.c",
                    "vbscript:x", "\\\\sunucu\\pay"):
            with self.subTest(url=url):
                self.assertTrue(check_url(url), url)

    def test_an_empty_url_is_refused(self):
        self.assertTrue(check_url(""))
        self.assertTrue(check_url("   "))

    def test_a_url_without_a_host_is_refused(self):
        self.assertTrue(check_url("http:///yol"))

    def test_the_allowlist_is_exactly_http_and_https(self):
        self.assertEqual(ALLOWED_SCHEMES, frozenset({"http", "https"}))


class TestMissingResearchIsSaid(WebToolCase):
    """Research not coming up must be a refusal with a reason, not a crash."""

    def test_search_says_the_subsystem_is_missing(self):
        result = self.run_tool("web.search", query="python")
        self.assertFalse(result.ok)
        self.assertIn("araştırma altyapısı", result.error)

    def test_research_says_the_subsystem_is_missing(self):
        result = self.run_tool("web.research", question="python nedir")
        self.assertFalse(result.ok)
        self.assertIn("araştırma altyapısı", result.error)


class TestSearch(WebToolCase):
    def test_results_carry_their_urls(self):
        tools.provide("research", StubResearch(hits=[
            Hit("Python", "https://python.org"),
            Hit("Docs", "https://docs.python.org")]))
        result = self.run_tool("web.search", query="python")
        self.assertTrue(result.ok, result.error)
        self.assertIn("https://python.org", result.output)
        self.assertEqual(result.detail["count"], 2)

    def test_no_results_is_a_failure_not_an_empty_success(self):
        tools.provide("research", StubResearch(hits=[]))
        result = self.run_tool("web.search", query="asdfqwer")
        self.assertFalse(result.ok)
        self.assertIn("sonuç döndürmedi", result.error)

    def test_a_dead_provider_is_reported_not_raised(self):
        tools.provide("research", StubResearch(raises=OSError("ağ yok")))
        result = self.run_tool("web.search", query="python")
        self.assertFalse(result.ok)
        self.assertIn("arama yapılamadı", result.error)

    def test_an_empty_query_is_refused(self):
        tools.provide("research", StubResearch(hits=[Hit("x", "https://x")]))
        self.assertFalse(self.run_tool("web.search", query="  ").ok)

    def test_the_limit_is_honoured(self):
        tools.provide("research", StubResearch(
            hits=[Hit(f"h{i}", f"https://x/{i}") for i in range(20)]))
        result = self.run_tool("web.search", query="x", limit=3)
        self.assertEqual(len(result.detail["urls"]), 3)


class TestResearch(WebToolCase):
    def test_a_report_becomes_an_answer_with_its_sources(self):
        tools.provide("research", StubResearch(report=Report(
            synthesis="Python bir dildir.",
            pages=[Page("https://python.org"), Page("https://docs.python.org")])))
        result = self.run_tool("web.research", question="python nedir")
        self.assertTrue(result.ok, result.error)
        self.assertIn("Python bir dildir", result.output)
        self.assertIn("https://python.org", result.output)
        self.assertEqual(len(result.detail["sources"]), 2)

    def test_reaching_nothing_is_a_failure_with_the_reason(self):
        tools.provide("research", StubResearch(report=Report(
            failures=["hiçbir sağlayıcı yok", "ağ yok"])))
        result = self.run_tool("web.research", question="x")
        self.assertFalse(result.ok)
        self.assertIn("sağlayıcı", result.error)

    def test_an_injection_attempt_is_reported_to_the_user(self):
        """A page that tried to address the model is a fact about the source."""
        tools.provide("research", StubResearch(report=Report(
            synthesis="özet", pages=[Page("https://kotu.example")],
            injection_sources=["https://kotu.example"])))
        result = self.run_tool("web.research", question="x")
        self.assertTrue(result.ok)
        self.assertIn("enjeksiyon", result.output)
        self.assertEqual(result.detail["injection_sources"], 1)

    def test_a_pipeline_that_raises_is_reported(self):
        tools.provide("research", StubResearch(raises=RuntimeError("çöktü")))
        result = self.run_tool("web.research", question="x")
        self.assertFalse(result.ok)
        self.assertIn("tamamlanamadı", result.error)


class TestOpeningABrowser(WebToolCase):
    def test_opening_needs_confirmation(self):
        result = self.run_tool("web.open", url="https://example.com")
        self.assertTrue(result.needs_confirmation)
        self.assertEqual(result.risk, MEDIUM)

    def test_a_refused_scheme_never_reaches_the_browser(self):
        import jarvis.tools.web as web

        called = []
        original = web.webbrowser.open
        web.webbrowser.open = lambda url: called.append(url) or True
        try:
            result = self.run_tool("web.open", url="file:///C:/Windows/win.ini",
                                   confirmed=True)
        finally:
            web.webbrowser.open = original
        self.assertFalse(result.ok)
        self.assertEqual(called, [], "reddedilen adres tarayiciya ulasmis")

    def test_an_allowed_url_is_opened(self):
        import jarvis.tools.web as web

        called = []
        original = web.webbrowser.open
        web.webbrowser.open = lambda url: called.append(url) or True
        try:
            result = self.run_tool("web.open", url="https://example.com",
                                   confirmed=True)
        finally:
            web.webbrowser.open = original
        self.assertTrue(result.ok, result.error)
        self.assertEqual(called, ["https://example.com"])

    def test_no_browser_is_reported_not_raised(self):
        import jarvis.tools.web as web

        original = web.webbrowser.open
        web.webbrowser.open = lambda url: False
        try:
            result = self.run_tool("web.open", url="https://example.com",
                                   confirmed=True)
        finally:
            web.webbrowser.open = original
        self.assertFalse(result.ok)
        self.assertIn("tarayıcı açılamadı", result.error)


class TestRegistration(unittest.TestCase):
    def test_the_web_tools_are_in_the_catalogue(self):
        expected = {"web.search", "web.research", "web.open"}
        self.assertLessEqual(expected, set(tools.names()))

    def test_reading_is_low_and_launching_is_not(self):
        self.assertEqual(tools.get("web.search").risk, LOW)
        self.assertEqual(tools.get("web.research").risk, LOW)
        self.assertEqual(tools.get("web.open").risk, MEDIUM)

    def test_the_wrappers_do_not_reimplement_fetching(self):
        """If these ever grow their own HTTP client, the injection defence in
        research/extract.py stops covering what the assistant reads."""
        import inspect

        import jarvis.tools.web as web

        source = inspect.getsource(web)
        for forbidden in ("urllib.request", "requests.", "http.client"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
