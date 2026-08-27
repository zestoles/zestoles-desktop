"""Where search results come from.

SearXNG is the preferred provider: it fans one query out across dozens of engines,
runs on the machine, needs no API key and logs nothing outward. But it needs a
container running, and a system meant to research unattended at four in the morning
cannot be one Docker Desktop restart away from being blind.

So search is a set of providers rather than one. The direct providers below are not
a degraded fallback — they are the sources JARVIS actually needs (repositories,
papers, technical discussion, reference), reached through free documented APIs with
no key and no scraping. When SearXNG is up it adds breadth on top.

Every provider fails soft. A dead provider returns nothing and says so; it never
raises into the research loop.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Protocol

from ..text import fold
from .http import FetchError, domain_of, get_json

log = logging.getLogger("jarvis.research.providers")

_STOPWORDS = frozenset({
    "nedir", "nasil", "neden", "hangi", "ile", "ve", "veya", "bir", "bu", "su",
    "icin", "gibi", "daha", "cok", "az", "farki", "farkı", "arasindaki", "midir",
    "mi", "mu", "the", "a", "an", "of", "to", "for", "and", "or", "is", "are",
    "what", "how", "why", "which", "between", "difference", "vs",
    "olarak", "belirtilen", "resmi", "modelinin", "baglam", "uzunlugu",
    "arac", "kullanma", "destegi", "destek", "hakkinda", "ilgili",
})


def key_terms(query: str, *, limit: int = 4) -> list[str]:
    """The distinctive words of a question, longest first.

    Search APIs that AND their terms return nothing for a full natural-language
    question: "Roblox ProfileService session locking nedir ve DataStore ile farki
    nedir" matched zero repositories. Providers get the terms that carry the
    meaning, not the sentence around them.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for word in query.split():
        cleaned = "".join(ch for ch in word if ch.isalnum() or ch in "-_.")
        folded = fold(cleaned)
        if len(folded) < 3 or folded in _STOPWORDS or folded in seen:
            continue
        seen.add(folded)
        terms.append(cleaned)
    def specificity(term: str) -> tuple[int, int]:
        # Product/model identifiers carry more meaning than long grammar words.
        # `qwen3.5` must outrank `belirtilen`; otherwise a search for Qwen can
        # mechanically accept an Airbus page containing the generic long word.
        score = 0
        if any(ch.isdigit() for ch in term):
            score += 100
        if any(ch.isupper() for ch in term[1:]):
            score += 60
        if any(ch in "-_." for ch in term):
            score += 30
        return score, len(term)

    terms.sort(key=specificity, reverse=True)
    return terms[:limit]


def is_relevant(hit: SearchHit, terms: list[str]) -> bool:
    """Whether a result plausibly concerns the question that was asked.

    Matching any term is far too weak. Asked about ProfileService session locking,
    Wikipedia returned "Android 17", "PostgreSQL" and "Jakarta Enterprise Beans" —
    all of which genuinely contain the word "session", and none of which are about
    the question. Relevance therefore requires one of the two *most distinctive*
    terms, which are the ones a keyword index cannot match by accident.
    """
    if not terms:
        return True
    haystack = fold(f"{hit.title} {hit.snippet} {hit.url}")
    return any(fold(term) in haystack for term in terms[:2])


def narrowing_search(fetch, terms: list[str]) -> list[SearchHit]:
    """Try the fullest query first, then drop terms until something comes back.

    Repository and Q&A APIs AND their terms together, so a precise four-word query
    reliably matches nothing while its first two words match the right thing. Going
    wide only after going narrow keeps precision where precision is available.
    """
    for count in range(min(3, len(terms)), 0, -1):
        hits = fetch(terms[:count])
        if hits:
            return hits
    return []


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    provider: str
    domain: str = ""
    published: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain:
            self.domain = domain_of(self.url)


class Provider(Protocol):
    name: str

    def available(self) -> bool: ...

    def search(self, query: str, *, limit: int) -> list[SearchHit]: ...


class SearxProvider:
    """A local SearXNG instance, queried through its JSON API.

    Requires `search.formats: [json]` in the instance settings — the default
    configuration serves HTML only, and the JSON endpoint answers 403 until it is
    enabled. That is the first thing to check when this provider goes quiet.
    """

    name = "searxng"

    def __init__(self, base_url: str = "http://127.0.0.1:8888", timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            get_json(f"{self.base_url}/search?q=test&format=json",
                     timeout=5, allow_local=True)
            return True
        except FetchError:
            return False

    def search(self, query: str, *, limit: int = 8) -> list[SearchHit]:
        url = (f"{self.base_url}/search?q={urllib.parse.quote(query)}"
               f"&format=json&language=auto&safesearch=0")
        try:
            payload = get_json(url, timeout=self.timeout, allow_local=True)
        except FetchError as exc:
            log.info("searxng sorgusu başarısız: %s", exc)
            return []
        results = payload.get("results", []) if isinstance(payload, dict) else []
        hits = []
        for item in results[:limit]:
            link = str(item.get("url", ""))
            if not link:
                continue
            hits.append(SearchHit(
                title=str(item.get("title", ""))[:300],
                url=link,
                snippet=str(item.get("content", ""))[:800],
                provider=self.name,
                published=str(item.get("publishedDate") or ""),
                extra={"engine": item.get("engine", "")},
            ))
        return hits


class _DuckParser(HTMLParser):
    def __init__(self, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.rows: list[dict[str, str]] = []
        self.mode = ""
        self.href = ""
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        values = dict(attrs)
        classes = set(str(values.get("class", "")).split())
        if "result__a" in classes and len(self.rows) < self.limit:
            self.mode, self.href, self.parts = "title", values.get("href", ""), []
        elif "result__snippet" in classes and self.rows:
            self.mode, self.parts = "snippet", []

    def handle_data(self, data):
        if self.mode:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag != "a" or not self.mode:
            return
        text = " ".join("".join(self.parts).split())
        if self.mode == "title":
            self.rows.append({"title": text, "href": self.href, "snippet": ""})
        elif self.rows:
            self.rows[-1]["snippet"] = text
        self.mode, self.href, self.parts = "", "", []


def parse_duckduckgo(html: str, *, limit: int = 8) -> list[SearchHit]:
    """Parse DuckDuckGo's no-JavaScript result page into direct public URLs."""
    parser = _DuckParser(max(1, limit))
    parser.feed(html)
    hits: list[SearchHit] = []
    for row in parser.rows:
        href = str(row.get("href") or "")
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
        if not target.startswith(("http://", "https://")):
            continue
        hits.append(SearchHit(
            title=row.get("title", "")[:300], url=target,
            snippet=row.get("snippet", "")[:800], provider="duckduckgo"))
    return hits


class DuckDuckGoProvider:
    """Free general-web fallback using DuckDuckGo's documented HTML surface."""

    name = "duckduckgo"

    def available(self) -> bool:
        # Do not spend a network call on status; search itself fails soft.
        return True

    def search(self, query: str, *, limit: int = 8) -> list[SearchHit]:
        from .http import get

        url = ("https://html.duckduckgo.com/html/?q="
               f"{urllib.parse.quote(query)}&kl=tr-tr")
        try:
            response = get(url, timeout=20, max_bytes=750_000,
                           headers={"User-Agent": "Mozilla/5.0 ZESTOLES/0.2"})
        except FetchError as exc:
            log.info("duckduckgo sorgusu başarısız: %s", exc)
            return []
        terms = key_terms(query)
        return [hit for hit in parse_duckduckgo(response.text, limit=limit)
                if is_relevant(hit, terms)]


class GitHubProvider:
    """Repository search. Unauthenticated and rate limited to 10 searches a minute.

    The user's brief names GitHub first among research sources, and a repository's
    star count and last push are unusually honest quality signals compared with
    anything else on the open web.
    """

    name = "github"

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 6) -> list[SearchHit]:
        terms = key_terms(query, limit=3)
        if not terms:
            return []

        def fetch(subset: list[str]) -> list[SearchHit]:
            url = ("https://api.github.com/search/repositories?q="
                   f"{urllib.parse.quote(' '.join(subset))}"
                   f"&sort=stars&order=desc&per_page={limit}")
            try:
                payload = get_json(url, timeout=20,
                                   headers={"Accept": "application/vnd.github+json"})
            except FetchError as exc:
                log.info("github sorgusu başarısız: %s", exc)
                return []
            items = payload.get("items", []) if isinstance(payload, dict) else []
            return [
                SearchHit(
                    title=f"{item.get('full_name', '')} ({item.get('stargazers_count', 0)}★)",
                    url=str(item.get("html_url", "")),
                    snippet=str(item.get("description") or "")[:800],
                    provider=self.name,
                    published=str(item.get("pushed_at") or ""),
                    extra={"stars": item.get("stargazers_count", 0),
                           "language": item.get("language") or "",
                           "archived": bool(item.get("archived"))},
                )
                for item in items[:limit] if item.get("html_url")
            ]

        return narrowing_search(fetch, terms)


class HackerNewsProvider:
    """Technical discussion via the Algolia index. No key, generous limits."""

    name = "hackernews"

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 6) -> list[SearchHit]:
        terms = key_terms(query, limit=3)
        url = ("https://hn.algolia.com/api/v1/search?query="
               f"{urllib.parse.quote(' '.join(terms) or query)}&tags=story"
               f"&hitsPerPage={limit}")
        try:
            payload = get_json(url, timeout=15)
        except FetchError as exc:
            log.info("hackernews sorgusu başarısız: %s", exc)
            return []
        hits = []
        for item in (payload.get("hits", []) if isinstance(payload, dict) else [])[:limit]:
            link = item.get("url") or f"https://news.ycombinator.com/item?id={item.get('objectID')}"
            hits.append(SearchHit(
                title=str(item.get("title") or "")[:300],
                url=str(link),
                snippet=str(item.get("story_text") or "")[:600],
                provider=self.name,
                published=str(item.get("created_at") or ""),
                extra={"points": item.get("points", 0),
                       "comments": item.get("num_comments", 0)},
            ))
        return [hit for hit in hits if hit.title]


class WikipediaProvider:
    """Reference lookup. Turkish first, English when Turkish has nothing."""

    name = "wikipedia"

    def __init__(self, languages: tuple[str, ...] = ("tr", "en")) -> None:
        self.languages = languages

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 4) -> list[SearchHit]:
        terms = key_terms(query, limit=4)
        for language in self.languages:
            url = (f"https://{language}.wikipedia.org/w/api.php?action=query&list=search"
                   f"&srsearch={urllib.parse.quote(' '.join(terms) or query)}"
                   f"&format=json&srlimit={limit}")
            try:
                payload = get_json(url, timeout=15)
            except FetchError as exc:
                log.info("wikipedia (%s) sorgusu başarısız: %s", language, exc)
                continue
            results = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
            if not results:
                continue
            hits = [
                SearchHit(
                    title=str(item.get("title", "")),
                    url=f"https://{language}.wikipedia.org/wiki/"
                        f"{urllib.parse.quote(str(item.get('title', '')).replace(' ', '_'))}",
                    snippet=_strip_tags(str(item.get("snippet", "")))[:600],
                    provider=self.name,
                    extra={"language": language},
                )
                for item in results[:limit]
            ]
            # Wikipedia always returns its best match, even when its best match is
            # about something else entirely. Unrelated results are worse than none.
            relevant = [hit for hit in hits if is_relevant(hit, terms)]
            if relevant:
                return relevant
        return []


class StackExchangeProvider:
    """Question-and-answer sites. Free and keyless, throttled by IP.

    For a question with a right answer that somebody has already had to find, this
    is often the densest source on the open web — and unlike a blog post, wrong
    answers there have usually been argued with underneath.
    """

    name = "stackexchange"

    def __init__(self, site: str = "stackoverflow") -> None:
        self.site = site

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        terms = key_terms(query, limit=4)
        if not terms:
            return []

        def fetch(subset: list[str]) -> list[SearchHit]:
            # withbody returns the question text in the API response. Stack
            # Overflow answers 403 to a non-browser user agent, so scraping the
            # page fails — and asking the documented API is the right thing to do
            # regardless of whether scraping would have worked.
            url = ("https://api.stackexchange.com/2.3/search/advanced?order=desc"
                   f"&sort=relevance&q={urllib.parse.quote(' '.join(subset))}"
                   f"&site={self.site}&pagesize={limit}&filter=withbody")
            try:
                payload = get_json(url, timeout=20)
            except FetchError as exc:
                log.info("stackexchange sorgusu başarısız: %s", exc)
                return []
            items = payload.get("items", []) if isinstance(payload, dict) else []
            hits = [
                SearchHit(
                    title=str(item.get("title", ""))[:300],
                    url=str(item.get("link", "")),
                    snippet="",
                    provider=self.name,
                    extra={"score": item.get("score", 0),
                           "answered": bool(item.get("is_answered")),
                           "body": str(item.get("body", ""))},
                )
                for item in items[:limit] if item.get("link")
            ]
            return [hit for hit in hits if is_relevant(hit, terms)]

        return narrowing_search(fetch, terms)


class DiscourseProvider:
    """A Discourse forum's public search. Configured per host.

    Vendor developer forums run on Discourse far more often than not, and for a
    platform like Roblox the official forum carries answers that exist nowhere
    else — including from the people who built the thing being asked about.
    """

    name = "discourse"

    def __init__(self, host: str, label: str = "") -> None:
        self.host = host.rstrip("/")
        self.name = f"discourse:{label or domain_of(host)}"

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        terms = key_terms(query, limit=4)
        url = f"{self.host}/search.json?q={urllib.parse.quote(' '.join(terms) or query)}"
        try:
            payload = get_json(url, timeout=20)
        except FetchError as exc:
            log.info("%s sorgusu başarısız: %s", self.name, exc)
            return []
        topics = payload.get("topics", []) if isinstance(payload, dict) else []
        hits = []
        for topic in topics[:limit]:
            slug = topic.get("slug")
            topic_id = topic.get("id")
            if not slug or not topic_id:
                continue
            hits.append(SearchHit(
                title=str(topic.get("title", ""))[:300],
                url=f"{self.host}/t/{slug}/{topic_id}",
                snippet="",
                provider=self.name,
                published=str(topic.get("created_at") or ""),
                extra={"replies": topic.get("reply_count", 0)},
            ))
        return [hit for hit in hits if is_relevant(hit, terms)]


class ArxivProvider:
    """Papers, for the rare question where the answer is genuinely academic."""

    name = "arxiv"

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 4) -> list[SearchHit]:
        from .extract import parse_arxiv_feed
        from .http import get

        url = ("http://export.arxiv.org/api/query?search_query=all:"
               f"{urllib.parse.quote(query)}&max_results={limit}")
        try:
            response = get(url, timeout=20)
        except FetchError as exc:
            log.info("arxiv sorgusu başarısız: %s", exc)
            return []
        return parse_arxiv_feed(response.text, self.name)[:limit]


def _strip_tags(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text)


def build_providers(config) -> list[Provider]:
    """The provider set, in the order results should be preferred."""
    get = config.get if config is not None else (lambda _k, d=None: d)
    providers: list[Provider] = []

    if get("research.searxng.enabled", True):
        providers.append(SearxProvider(
            base_url=get("research.searxng.url", "http://127.0.0.1:8888"),
            timeout=get("research.searxng.timeout_s", 20),
        ))
    if get("research.duckduckgo.enabled", True):
        providers.append(DuckDuckGoProvider())
    if get("research.github.enabled", True):
        providers.append(GitHubProvider())
    if get("research.stackexchange.enabled", True):
        providers.append(StackExchangeProvider(
            site=get("research.stackexchange.site", "stackoverflow")))
    if get("research.hackernews.enabled", True):
        providers.append(HackerNewsProvider())
    for entry in get("research.forums", []) or []:
        host = entry.get("url") if isinstance(entry, dict) else str(entry)
        if host:
            providers.append(DiscourseProvider(
                host, label=(entry.get("label", "") if isinstance(entry, dict) else "")))
    if get("research.wikipedia.enabled", True):
        providers.append(WikipediaProvider())
    if get("research.arxiv.enabled", False):
        providers.append(ArxivProvider())
    return providers
