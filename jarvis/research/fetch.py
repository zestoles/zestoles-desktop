"""Fetching a page and getting readable, defused text out of it."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .extract import MAX_TEXT_CHARS, Extracted, html_to_text, neutralise
from .http import FetchError, get

log = logging.getLogger("jarvis.research.fetch")

_TEXTUAL = ("text/html", "text/plain", "application/xhtml", "application/json",
            "application/xml", "text/xml", "text/markdown")


def fetch_page(url: str, *, timeout: int = 15, max_chars: int = MAX_TEXT_CHARS) -> Extracted:
    """Fetch, extract, defuse. Raises FetchError; callers are expected to fail soft."""
    response = get(url, timeout=timeout)
    content_type = response.content_type.casefold()

    if content_type and not any(kind in content_type for kind in _TEXTUAL):
        raise FetchError(f"metin olmayan içerik ({content_type}): {url}")

    if "html" in content_type or "xml" in content_type:
        text, title = html_to_text(response.text)
    else:
        text, title = response.text.strip(), ""

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    # Defuse before anything else touches it, so no code path can accidentally
    # hand raw page text to a model.
    text, flags = neutralise(text)

    return Extracted(
        url=response.url,
        title=title[:300],
        text=text,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        injection_flags=flags,
        truncated=truncated,
    )


def from_body(url: str, title: str, markup: str, *, max_chars: int = MAX_TEXT_CHARS) -> Extracted:
    """Build a source from content an API already returned.

    Some sites answer 403 to anything that is not a browser, and their API is both
    the working path and the polite one. Content arriving this way is defused
    exactly like a fetched page — it is no less written by a stranger.
    """
    text, _ = html_to_text(markup)
    truncated = len(text) > max_chars
    text, flags = neutralise(text[:max_chars])
    return Extracted(
        url=url, title=title[:300], text=text,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        injection_flags=flags, truncated=truncated,
    )


def fetch_many(urls: list[str], *, timeout: int = 15, should_stop=lambda: False
               ) -> tuple[list[Extracted], list[str]]:
    """Fetch several pages, keeping what worked and reporting what did not.

    Fail-soft is the whole point: research over five sources where two time out is
    research over three sources, not a failed run.
    """
    wanted = [url for url in urls if not should_stop()]
    if not wanted:
        return [], []
    pages_by_index: dict[int, Extracted] = {}
    failures_by_index: dict[int, str] = {}

    def one(index: int, url: str):
        if should_stop():
            return index, None, None
        try:
            return index, fetch_page(url, timeout=timeout), None
        except FetchError as exc:
            log.info("kaynak alınamadı: %s", exc)
            return index, None, f"{url}: {exc}"
        except Exception as exc:  # noqa: BLE001 - a bad page must not end the run
            log.warning("kaynak işlenemedi %s: %s", url, exc)
            return index, None, f"{url}: {type(exc).__name__}"

    with ThreadPoolExecutor(max_workers=min(6, len(wanted)),
                            thread_name_prefix="zestoles-kaynak") as pool:
        futures = [pool.submit(one, index, url)
                   for index, url in enumerate(wanted)]
        for future in as_completed(futures):
            index, page, failure = future.result()
            if page is not None:
                pages_by_index[index] = page
            elif failure:
                failures_by_index[index] = failure

    # Stable ordering keeps prompts and test/evaluation results reproducible.
    return ([pages_by_index[i] for i in sorted(pages_by_index)],
            [failures_by_index[i] for i in sorted(failures_by_index)])
