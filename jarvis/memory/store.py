"""The searchable index over the vault, plus the short-term conversation log.

The vault holds the truth; this database is a derived index that can always be
rebuilt from it. Notes are chunked and embedded here so recall is semantic rather
than literal — "dün konuştuğumuz oyun projesi" has to find a note titled
"Roblox tycoon", and keyword search alone never will.

Retrieval is hybrid. Vector search understands meaning but drifts on proper nouns
and identifiers; keyword search nails "ProfileService" and misses paraphrase.
Reciprocal rank fusion combines the two rankings without needing their scores to
be on a comparable scale.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .embed import Embedder, cosine_ranking, pack
from .vault import Note, Vault

log = logging.getLogger("jarvis.memory.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    started  REAL NOT NULL,
    ended    REAL,
    title    TEXT,
    summary  TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts         REAL NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, ts);

CREATE TABLE IF NOT EXISTS notes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    path    TEXT UNIQUE NOT NULL,
    kind    TEXT NOT NULL,
    title   TEXT NOT NULL,
    updated REAL NOT NULL,
    hash    TEXT NOT NULL,
    source  TEXT NOT NULL DEFAULT 'kullanici'
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id   INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    ord       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61'
);
"""

TARGET_CHUNK = 700
MIN_CHUNK = 120


@dataclass(slots=True)
class Hit:
    note_title: str
    note_kind: str
    note_path: str
    text: str
    score: float
    source: str = "kullanici"


def chunk_markdown(text: str) -> list[str]:
    """Split on headings first, then on paragraphs, keeping pieces near TARGET_CHUNK.

    Headings are kept with the text beneath them so a chunk retains its subject.
    """
    sections = re.split(r"\n(?=#{1,6}\s)", text.strip())
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= TARGET_CHUNK:
            chunks.append(section)
            continue
        heading = section.splitlines()[0] if section.startswith("#") else ""
        buffer = ""
        for paragraph in re.split(r"\n\s*\n", section):
            if len(buffer) + len(paragraph) + 2 <= TARGET_CHUNK:
                buffer = f"{buffer}\n\n{paragraph}".strip()
                continue
            if buffer:
                chunks.append(buffer)
            buffer = f"{heading}\n{paragraph}".strip() if heading else paragraph.strip()
        if buffer:
            chunks.append(buffer)
    return [c for c in chunks if len(c) >= MIN_CHUNK] or ([text.strip()] if text.strip() else [])


class Store:
    def __init__(self, db_path: Path, vault: Vault, embedder: Embedder) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vault = vault
        self.embedder = embedder
        self.fts = False
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
            try:
                conn.executescript(FTS_SCHEMA)
                self.fts = True
            except sqlite3.OperationalError as exc:
                log.warning("FTS5 yok, anahtar kelime aramasi LIKE ile yapilacak: %s", exc)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive column migrations for databases created by an earlier build."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
        if "source" not in columns:
            conn.execute(
                "ALTER TABLE notes ADD COLUMN source TEXT NOT NULL DEFAULT 'kullanici'"
            )
            log.info("notes tablosuna source sutunu eklendi")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------------------------------------------------- short term
    def open_session(self, title: str = "") -> int:
        with closing(self._conn()) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO sessions (started, title) VALUES (?,?)", (time.time(), title)
            )
            return int(cursor.lastrowid)

    def close_session(self, session_id: int, summary: str = "") -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE sessions SET ended=?, summary=? WHERE id=?",
                (time.time(), summary, session_id),
            )

    def add_message(self, session_id: int, role: str, content: str) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO messages (session_id, ts, role, content) VALUES (?,?,?,?)",
                (session_id, time.time(), role, content),
            )

    def session_messages(self, session_id: int, limit: int = 200) -> list[dict[str, str]]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY ts LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def recent_sessions(self, limit: int = 5) -> list[sqlite3.Row]:
        with closing(self._conn()) as conn:
            return conn.execute(
                "SELECT id, started, ended, title, summary FROM sessions"
                " WHERE summary IS NOT NULL AND summary != ''"
                " ORDER BY started DESC LIMIT ?",
                (limit,),
            ).fetchall()

    # ------------------------------------------------------------ indexing
    def reindex(self, *, force: bool = False) -> dict[str, int]:
        """Bring the index in line with the vault. Cheap when nothing changed."""
        seen: set[str] = set()
        added = updated = unchanged = 0

        for note in self.vault.all_notes():
            key = str(note.path)
            seen.add(key)
            digest = hashlib.sha256(note.text.encode("utf-8")).hexdigest()
            # Never hold a SQLite write transaction while Ollama is running.
            # The embedder records its own usage through Budget, which may share
            # this database.  Doing that from inside this transaction makes the
            # process wait on its own lock once per chunk (15 seconds each).
            with closing(self._conn()) as conn:
                row = conn.execute("SELECT id, hash FROM notes WHERE path=?", (key,)).fetchone()
                if row and row["hash"] == digest and not force:
                    unchanged += 1
                    continue

            pieces, vectors = self._prepare_chunks(note)

            # Re-read after the model call: another ZESTOLES process may have
            # indexed the same note while embedding was being computed.
            with closing(self._conn()) as conn, conn:
                row = conn.execute("SELECT id, hash FROM notes WHERE path=?", (key,)).fetchone()
                if row and row["hash"] == digest and not force:
                    unchanged += 1
                    continue
                if row:
                    note_id = int(row["id"])
                    conn.execute(
                        "UPDATE notes SET kind=?, title=?, updated=?, hash=?, source=? WHERE id=?",
                        (note.kind, note.title, note.updated.timestamp(), digest,
                         note.source, note_id),
                    )
                    self._drop_chunks(conn, note_id)
                    updated += 1
                else:
                    cursor = conn.execute(
                        "INSERT INTO notes (path, kind, title, updated, hash, source)"
                        " VALUES (?,?,?,?,?,?)",
                        (key, note.kind, note.title, note.updated.timestamp(), digest,
                         note.source),
                    )
                    note_id = int(cursor.lastrowid)
                    added += 1
                self._write_chunks(conn, note_id, pieces, vectors)

        with closing(self._conn()) as conn, conn:
            rows = conn.execute("SELECT id, path FROM notes").fetchall()
            removed = 0
            for row in rows:
                if row["path"] not in seen:
                    self._drop_chunks(conn, int(row["id"]))
                    conn.execute("DELETE FROM notes WHERE id=?", (row["id"],))
                    removed += 1

        return {"eklendi": added, "guncellendi": updated,
                "degismedi": unchanged, "silindi": removed}

    def _drop_chunks(self, conn: sqlite3.Connection, note_id: int) -> None:
        if self.fts:
            ids = conn.execute("SELECT id, text FROM chunks WHERE note_id=?", (note_id,)).fetchall()
            for row in ids:
                conn.execute(
                    "INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)",
                    (row["id"], row["text"]),
                )
        conn.execute("DELETE FROM chunks WHERE note_id=?", (note_id,))

    def _prepare_chunks(self, note: Note) -> tuple[list[str], list[list[float]]]:
        """Embed a note with no index transaction open.

        Embedding can take seconds and its usage callback writes to the shared
        ledger.  Both are reasons to keep this phase outside the short atomic
        replacement of a note's chunks.
        """
        pieces = chunk_markdown(note.text)
        if not pieces:
            return [], []
        try:
            vectors = self.embedder.embed(pieces)
        except OSError as exc:
            log.warning("embedding alinamadi (%s) — not metinsel olarak indekslendi", exc)
            vectors = []
        return pieces, vectors

    def _write_chunks(self, conn: sqlite3.Connection, note_id: int,
                      pieces: list[str], vectors: list[list[float]]) -> None:
        for index, piece in enumerate(pieces):
            blob = pack(vectors[index]) if index < len(vectors) else None
            cursor = conn.execute(
                "INSERT INTO chunks (note_id, ord, text, embedding) VALUES (?,?,?,?)",
                (note_id, index, piece, blob),
            )
            if self.fts:
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?,?)",
                    (int(cursor.lastrowid), piece),
                )

    # -------------------------------------------------------------- recall
    def search(self, query: str, *, limit: int = 5) -> list[Hit]:
        vector_ranked = self._vector_search(query, limit * 4)
        keyword_ranked = self._keyword_search(query, limit * 4)
        fused = _fuse(vector_ranked, keyword_ranked)
        if not fused:
            return []

        ids = [chunk_id for chunk_id, _ in fused[:limit]]
        placeholders = ",".join("?" * len(ids))
        with closing(self._conn()) as conn:
            rows = conn.execute(
                f"SELECT c.id, c.text, n.title, n.kind, n.path, n.source FROM chunks c"
                f" JOIN notes n ON n.id = c.note_id WHERE c.id IN ({placeholders})",
                ids,
            ).fetchall()

        by_id = {int(r["id"]): r for r in rows}
        hits = []
        for chunk_id, score in fused[:limit]:
            row = by_id.get(chunk_id)
            if row:
                hits.append(Hit(row["title"], row["kind"], row["path"], row["text"],
                                score, row["source"]))
        return hits

    def _vector_search(self, query: str, limit: int) -> list[int]:
        try:
            vector = self.embedder.embed_one(query)
        except OSError as exc:
            log.debug("vector search skipped: %s", exc)
            return []
        if not vector:
            return []
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        if not rows:
            return []
        scores = cosine_ranking(vector, [r["embedding"] for r in rows])
        ordered = sorted(zip((int(r["id"]) for r in rows), scores), key=lambda p: -p[1])
        return [chunk_id for chunk_id, score in ordered[:limit] if score > 0.2]

    def _keyword_search(self, query: str, limit: int) -> list[int]:
        terms = [t for t in re.findall(r"\w{3,}", query) if t]
        if not terms:
            return []
        with closing(self._conn()) as conn:
            if self.fts:
                expression = " OR ".join(terms)
                try:
                    rows = conn.execute(
                        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
                        " ORDER BY rank LIMIT ?",
                        (expression, limit),
                    ).fetchall()
                    return [int(r["rowid"]) for r in rows]
                except sqlite3.OperationalError as exc:
                    log.debug("fts query failed: %s", exc)
            clause = " OR ".join("text LIKE ?" for _ in terms)
            rows = conn.execute(
                f"SELECT id FROM chunks WHERE {clause} LIMIT ?",
                [*(f"%{t}%" for t in terms), limit],
            ).fetchall()
        return [int(r["id"]) for r in rows]

    def stats(self) -> dict[str, int]:
        with closing(self._conn()) as conn:
            notes = conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"]
            chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            embedded = conn.execute(
                "SELECT COUNT(*) c FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()["c"]
            sessions = conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"]
            messages = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        return {"notlar": notes, "parcalar": chunks, "vektorlu": embedded,
                "oturumlar": sessions, "mesajlar": messages}


def _fuse(*rankings: list[int], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: position in each list matters, raw scores do not."""
    totals: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking):
            totals[chunk_id] = totals.get(chunk_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(totals.items(), key=lambda pair: -pair[1])
