"""Web tools, built on the research pipeline that already exists.

S4 built search across six providers, source quality scoring, extraction with
prompt-injection defence and cross-verification of claims against independent
sources. None of that is rewritten here. These are thin wrappers that make it
reachable from the assistant loop, because the alternative — a fresh "fetch a
page" tool — would quietly drop the injection defence and the citation
discipline that took a phase to build.

## Why a page is never handed over raw

Extracted page text is untrusted input. It has already been through
`research/extract.py`, which strips the constructions that try to address the
model as if they were instructions. A tool that returned raw HTML would put that
defence back on whoever remembered to ask for it.

## Opening a browser is a change, not a read

`web.open` launches something on the user's desktop, so it sits at MEDIUM and
goes through confirmation like any other change. Only http and https are
accepted: `file://` would turn a browser launch into an arbitrary local read,
and `javascript:` into script execution in whatever page is focused.
"""

from __future__ import annotations

import logging
import webbrowser
from urllib.parse import urlparse

from . import LOW, MEDIUM, SERVICES, ToolResult, Workspace, tool

log = logging.getLogger("jarvis.tools.web")

#: What a browser may be pointed at. Everything else is refused rather than
#: sanitised — there is no safe reading of `javascript:` here.
ALLOWED_SCHEMES = frozenset({"http", "https"})

MAX_SUMMARY_CHARS = 6000


def _research():
    """The research system, or None when it did not come up."""
    return SERVICES.get("research")


def check_url(url: str) -> str:
    """Empty when the URL may be opened, otherwise why it may not."""
    text = str(url or "").strip()
    if not text:
        return "adres boş"
    try:
        parsed = urlparse(text)
    except ValueError:
        return f"adres çözümlenemedi: {text[:80]}"
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return (f"yalnızca http ve https açılabilir "
                f"(verilen: {parsed.scheme or 'şema yok'})")
    if not parsed.netloc:
        return "adreste alan adı yok"
    return ""


@tool("web.search", risk=LOW, summary="Web'de arama yapar ve kaynakları listeler")
def _search(*, workspace: Workspace, query: str, limit: int = 8) -> ToolResult:
    research = _research()
    if research is None:
        return ToolResult(False, error="araştırma altyapısı bu oturumda kurulu değil")
    if not str(query or "").strip():
        return ToolResult(False, error="arama sorgusu boş")

    try:
        hits = research.search(query)
    except Exception as exc:  # noqa: BLE001 - a dead provider is not a crash
        log.info("arama başarısız: %s", exc)
        return ToolResult(False, error=f"arama yapılamadı: {exc}")

    if not hits:
        return ToolResult(False, error="hiçbir sağlayıcı sonuç döndürmedi",
                          detail={"query": query, "count": 0})

    shown = hits[: max(1, int(limit))]
    lines = [f"{i}. {hit.title}\n   {hit.url}"
             for i, hit in enumerate(shown, start=1)]
    return ToolResult(True, output="\n".join(lines),
                      detail={"query": query, "count": len(hits),
                              "urls": [hit.url for hit in shown]})


@tool("web.research", risk=LOW,
      summary="Bir konuyu araştırır, kaynakları karşılaştırır ve özetler")
def _research_tool(*, workspace: Workspace, question: str) -> ToolResult:
    research = _research()
    if research is None:
        return ToolResult(False, error="araştırma altyapısı bu oturumda kurulu değil")
    if not str(question or "").strip():
        return ToolResult(False, error="araştırma sorusu boş")

    try:
        report = research.investigate(question)
    except Exception as exc:  # noqa: BLE001 - research failing is an answer
        log.info("araştırma başarısız: %s", exc)
        return ToolResult(False, error=f"araştırma tamamlanamadı: {exc}")

    if not report.pages and not report.synthesis:
        reason = "; ".join(report.failures[:2]) or "hiçbir kaynağa ulaşılamadı"
        return ToolResult(False, error=f"araştırma sonuç vermedi: {reason}",
                          detail={"question": question,
                                  "failures": list(report.failures)})

    safe_answer = getattr(report, "answer", "") or report.synthesis
    body = [safe_answer.strip()[:MAX_SUMMARY_CHARS]]
    if report.pages:
        body.append("\nKaynaklar:")
        body.extend(f"  - {page.url}" for page in report.pages[:10])
    # Said rather than left implicit: a page that tried to address the model is
    # a fact about the source, and the user is the one who should hear it.
    if report.injection_sources:
        body.append(f"\nUyarı: {len(report.injection_sources)} kaynakta "
                    f"yönerge enjeksiyonu denemesi görüldü ve temizlendi.")

    return ToolResult(
        True, output="\n".join(body).strip(),
        detail={"question": question,
                "sources": [page.url for page in report.pages],
                "claims": len(report.claims),
                "injection_sources": len(report.injection_sources),
                "failures": list(report.failures)})


@tool("web.open", risk=MEDIUM, summary="Bir adresi tarayıcıda açar")
def _open(*, workspace: Workspace, url: str) -> ToolResult:
    problem = check_url(url)
    if problem:
        return ToolResult(False, error=problem, detail={"url": url, "refused": True})

    target = str(url).strip()
    try:
        opened = webbrowser.open(target)
    except Exception as exc:  # noqa: BLE001 - no browser is not a crash
        log.info("tarayıcı açılamadı: %s", exc)
        return ToolResult(False, error=f"tarayıcı açılamadı: {exc}",
                          detail={"url": target})
    if not opened:
        return ToolResult(False, error="tarayıcı açılamadı (varsayılan tarayıcı yok)",
                          detail={"url": target})
    return ToolResult(True, output=f"tarayıcıda açıldı: {target}",
                      detail={"url": target})


__all__ = ["check_url", "ALLOWED_SCHEMES"]
