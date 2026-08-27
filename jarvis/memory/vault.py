"""Long-term memory as plain markdown files.

Memory a person cannot read is memory a person cannot correct. Notes are markdown
with a small frontmatter block and [[wikilinks]], so the vault opens in Obsidian —
or Notepad — with JARVIS switched off entirely. Obsidian is a window onto this
folder, never a dependency of it. If the index in SQLite is ever lost it can be
rebuilt from these files; the reverse is not true, so the files are the truth.

Frontmatter is parsed by hand rather than with PyYAML: the schema is four known
keys, and the vault should not need a third-party parser to stay readable.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..text import slugify

log = logging.getLogger("jarvis.memory.vault")

# Folders carry meaning; a note's kind is its directory.
KINDS = {
    "kisi": "kullanıcı hakkında kalıcı bilgiler",
    "proje": "aktif ve geçmiş projeler",
    "deneyim": "ne işe yaradı, ne yaramadı",
    "bilgi": "araştırmadan süzülen bilgi",
    "gunluk": "oturum özetleri",
}

_FRONT = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


@dataclass(slots=True)
class Note:
    path: Path
    kind: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    updated: datetime = field(default_factory=datetime.now)
    links: list[str] = field(default_factory=list)
    # Where the claim came from. Written into the file so it survives the index
    # being rebuilt, and so a human reading the vault can see it.
    source: str = "kullanici"

    @property
    def slug(self) -> str:
        return self.path.stem

    @property
    def text(self) -> str:
        """Title and body together — what gets embedded and searched."""
        return f"{self.title}\n\n{self.body}".strip()


def parse_frontmatter(raw: str) -> tuple[dict[str, str | list[str]], str]:
    match = _FRONT.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str | list[str]] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value
    return meta, raw[match.end():]


def render(note: Note) -> str:
    tags = "[" + ", ".join(note.tags) + "]"
    return (
        "---\n"
        f"title: {note.title}\n"
        f"kind: {note.kind}\n"
        f"kaynak: {note.source}\n"
        f"tags: {tags}\n"
        f"updated: {note.updated:%Y-%m-%d %H:%M}\n"
        "---\n\n"
        f"{note.body.strip()}\n"
    )


def extract_links(body: str) -> list[str]:
    return sorted({m.group(1).strip() for m in _WIKILINK.finditer(body)})


class Vault:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for kind in KINDS:
            (self.root / kind).mkdir(parents=True, exist_ok=True)
        self._ensure_home()

    # ------------------------------------------------------------------ read
    def path_for(self, kind: str, slug: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"bilinmeyen not türü: {kind}")
        return self.root / kind / f"{slug}.md"

    def read(self, path: Path) -> Note | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("not okunamadı %s: %s", path, exc)
            return None
        meta, body = parse_frontmatter(raw)
        tags = meta.get("tags", [])
        updated = meta.get("updated", "")
        try:
            when = datetime.strptime(str(updated), "%Y-%m-%d %H:%M")
        except ValueError:
            when = datetime.fromtimestamp(path.stat().st_mtime)
        return Note(
            path=path,
            kind=str(meta.get("kind") or path.parent.name),
            title=str(meta.get("title") or path.stem.replace("-", " ")),
            body=body.strip(),
            tags=tags if isinstance(tags, list) else [str(tags)],
            updated=when,
            links=extract_links(body),
            source=str(meta.get("kaynak") or "kullanici"),
        )

    def all_notes(self) -> Iterator[Note]:
        for kind in KINDS:
            for path in sorted((self.root / kind).glob("*.md")):
                note = self.read(path)
                if note is not None:
                    yield note

    def find(self, kind: str, slug: str) -> Note | None:
        path = self.path_for(kind, slug)
        return self.read(path) if path.exists() else None

    # ----------------------------------------------------------------- write
    def write(
        self,
        kind: str,
        title: str,
        body: str,
        *,
        slug: str | None = None,
        tags: list[str] | None = None,
        source: str = "kullanici",
    ) -> Note:
        slug = slug or slugify(title)
        note = Note(
            path=self.path_for(kind, slug),
            kind=kind,
            title=title,
            body=body.strip(),
            tags=tags or [],
            updated=datetime.now(),
            links=extract_links(body),
            source=source,
        )
        note.path.write_text(render(note), encoding="utf-8")
        return note

    def append(
        self,
        kind: str,
        title: str,
        addition: str,
        *,
        slug: str | None = None,
        source: str = "kullanici",
    ) -> Note:
        """Add to a note without losing what is already there.

        Long-term memory grows by accretion; overwriting a note would silently
        discard everything learned before this moment.
        """
        slug = slug or slugify(title)
        existing = self.find(kind, slug)
        if existing is None:
            return self.write(kind, title, addition, slug=slug, source=source)
        stamp = datetime.now().strftime("%Y-%m-%d")
        body = f"{existing.body}\n\n<!-- {stamp} -->\n{addition.strip()}"
        return self.write(kind, existing.title, body, slug=slug,
                          tags=existing.tags, source=existing.source)

    def _ensure_home(self) -> None:
        home = self.root / "00-JARVIS.md"
        if home.exists():
            return
        lines = [
            "---", "title: ZESTOLES hafızası", "kind: bilgi", "tags: [sistem]",
            f"updated: {datetime.now():%Y-%m-%d %H:%M}", "---", "",
            "# ZESTOLES hafızası", "",
            "Bu klasör ZESTOLES'in uzun süreli hafızasıdır. Dosyalar düz markdown —",
            "istersen Obsidian'da aç, istersen Not Defteri'nde. Elle düzelttiğin",
            "her şey geçerlidir; ZESTOLES bir sonraki taramada senin sürümünü okur.",
            "",
            "## Klasörler", "",
        ]
        lines += [f"- `{kind}/` — {desc}" for kind, desc in KINDS.items()]
        lines += [
            "",
            "## Silmek",
            "",
            "Bir dosyayı silmek onu unutturur. Yanlış öğrenilmiş bir şey görürsen",
            "dosyayı düzelt veya sil — ZESTOLES'e ayrıca söylemen gerekmez.",
            "",
        ]
        home.write_text("\n".join(lines), encoding="utf-8")
