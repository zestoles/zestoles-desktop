"""Local document library with lightweight BM25-style retrieval.

This is the useful part of the legacy project's RAG rebuilt behind the current
tool and runtime boundaries.  It has no model or network dependency: documents
are read locally, split with overlap, and ranked from term frequency and inverse
document frequency.  PDF support is optional; text, Markdown, CSV, JSON, HTML
and DOCX work with the standard library (DOCX is extracted from its XML).
"""

from __future__ import annotations

import html
import math
import re
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from defusedxml import ElementTree

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".log", ".csv", ".json", ".html", ".htm"})
SUPPORTED = TEXT_EXTENSIONS | {".pdf", ".docx"}
TOKEN = re.compile(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]{2,}")
TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class Chunk:
    source: str
    text: str
    terms: Counter[str]


@dataclass(frozen=True, slots=True)
class DocumentHit:
    source: str
    text: str
    score: float


class DocumentLibrary:
    def __init__(self, root: Path, *, chunk_chars: int = 1200,
                 overlap_chars: int = 220) -> None:
        self.root = Path(root).expanduser().resolve()
        self.chunk_chars = max(400, int(chunk_chars))
        self.overlap_chars = max(0, min(int(overlap_chars), self.chunk_chars // 2))
        self._chunks: list[Chunk] = []
        self._document_frequency: Counter[str] = Counter()
        self._indexed_at = 0.0
        self._files = 0
        self._errors: list[str] = []
        self._lock = threading.RLock()

    def index(self, folder: Path | None = None) -> dict[str, object]:
        target = Path(folder or self.root).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        errors: list[str] = []
        files = 0
        for path in sorted(target.rglob("*")):
            if not path.is_file() or path.name.startswith("_") or path.suffix.lower() not in SUPPORTED:
                continue
            try:
                text = self._read(path)
                if not text.strip():
                    continue
                files += 1
                label = str(path.relative_to(target))
                for part in self._split(text):
                    chunks.append(Chunk(label, part, Counter(self._tokens(part))))
            except Exception as exc:  # noqa: BLE001 - one bad file is not the library
                errors.append(f"{path.name}: {type(exc).__name__}")
        df: Counter[str] = Counter()
        for chunk in chunks:
            df.update(chunk.terms.keys())
        with self._lock:
            self.root = target
            self._chunks = chunks
            self._document_frequency = df
            self._files = files
            self._errors = errors[:30]
            self._indexed_at = time.time()
        return self.status()

    def search(self, query: str, *, limit: int = 5) -> list[DocumentHit]:
        if not str(query).strip():
            return []
        with self._lock:
            empty = not self._chunks
        if empty:
            self.index()
        wanted = list(dict.fromkeys(self._tokens(query)))
        if not wanted:
            return []
        with self._lock:
            chunks = list(self._chunks)
            df = Counter(self._document_frequency)
        total = len(chunks)
        scored: list[DocumentHit] = []
        for chunk in chunks:
            length = max(1, sum(chunk.terms.values()))
            score = 0.0
            for term in wanted:
                tf = chunk.terms.get(term, 0)
                if not tf:
                    continue
                inverse = math.log(1 + (total - df.get(term, 0) + .5) /
                                   (df.get(term, 0) + .5))
                score += inverse * ((tf * 2.2) / (tf + 1.2 + .75 * length / 180))
            if score > 0:
                scored.append(DocumentHit(chunk.source, chunk.text, round(score, 4)))
        scored.sort(key=lambda hit: (-hit.score, hit.source))
        return scored[:max(1, min(int(limit), 12))]

    def status(self) -> dict[str, object]:
        with self._lock:
            return {"klasor": str(self.root), "dosyalar": self._files,
                    "parcalar": len(self._chunks), "indekslendi": self._indexed_at,
                    "hatalar": list(self._errors), "desteklenen": sorted(SUPPORTED)}

    def _read(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            raw = path.read_text(encoding="utf-8", errors="replace")
            return html.unescape(TAG.sub(" ", raw)) if suffix in (".html", ".htm") else raw
        if suffix == ".docx":
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(xml)
            return " ".join(node.text or "" for node in root.iter()
                            if node.tag.endswith("}t"))
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF okumak için pypdf kurulu değil") from exc
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        return ""

    def _split(self, text: str) -> Iterable[str]:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return
        step = self.chunk_chars - self.overlap_chars
        for start in range(0, len(clean), step):
            part = clean[start:start + self.chunk_chars].strip()
            if part:
                yield part
            if start + self.chunk_chars >= len(clean):
                break

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [item.lower() for item in TOKEN.findall(str(text))]


__all__ = ["Chunk", "DocumentHit", "DocumentLibrary", "SUPPORTED"]
