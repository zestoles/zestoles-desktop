"""Memory, assembled: three layers behind one object.

  short term   the running conversation, logged as it happens
  long term    who the user is, what the projects are — markdown notes in the vault
  experience   what was tried, whether it worked, and why

Callers use `context_for()` before answering and `remember()` after; the layering
is an implementation detail. Nothing here raises on failure: a system whose memory
is offline should answer with a worse memory, not refuse to talk.
"""

from __future__ import annotations

import logging
from datetime import datetime

from ..brain.local import LocalBrain
from ..config import Config
from .distill import (
    SUMMARY_SOURCED,
    UNVERIFIED_SOURCES,
    VERIFIED_SOURCED,
    Fact,
    distill,
    summarise,
)
from .embed import Embedder
from .store import Hit, Store
from .vault import Vault

log = logging.getLogger("jarvis.memory")

_KIND_LABEL = {
    "kisi": "kullanıcı",
    "proje": "proje",
    "deneyim": "deneyim",
    "bilgi": "bilgi",
    "gunluk": "günlük",
}


class Memory:
    def __init__(self, config: Config, local: LocalBrain) -> None:
        self.config = config
        self.local = local
        self.vault = Vault(config.path("paths.vault", "vault"))
        self.embedder = Embedder(
            host=config.get("local.host"),
            model=config.get("memory.embed_model", "bge-m3"),
        )
        self.store = Store(config.path("paths.db", "data/jarvis.db"), self.vault, self.embedder)
        self.recall_limit = config.get("memory.recall_limit", 5)
        self.session_id: int | None = None
        self.enabled = bool(config.get("memory.enabled", True))

    # ------------------------------------------------------------- session
    def start_session(self, title: str = "") -> int:
        self.session_id = self.store.open_session(title)
        return self.session_id

    def remember(self, role: str, content: str) -> None:
        if not self.enabled or self.session_id is None or not content.strip():
            return
        try:
            self.store.add_message(self.session_id, role, content)
        except Exception as exc:  # noqa: BLE001 - memory must never break the reply
            log.warning("mesaj kaydedilemedi: %s", exc)

    def end_session(self) -> dict[str, object]:
        """Summarise, extract durable facts, write them to the vault, reindex."""
        result: dict[str, object] = {"ozet": "", "notlar": [], "reddedilen": [], "indeks": {}}
        if not self.enabled or self.session_id is None:
            return result

        messages = self.store.session_messages(self.session_id)
        if len(messages) < 2:
            self.store.close_session(self.session_id)
            return result

        summary = summarise(self.local, messages)
        self.store.close_session(self.session_id, summary)
        result["ozet"] = summary

        today = datetime.now().strftime("%Y-%m-%d")
        facts, refused = distill(self.local, messages, today=today)
        written = []
        for fact in facts:
            try:
                note = self.vault.append(
                    fact.kind, fact.title, fact.content, slug=fact.slug, source=fact.source
                )
                written.append(note.title)
            except (OSError, ValueError) as exc:
                log.warning("not yazılamadı (%s): %s", fact.title, exc)
        result["notlar"] = written
        result["reddedilen"] = refused

        if summary:
            self.vault.append(
                "gunluk", f"{today} oturum günlüğü", f"- {summary}", slug=today,
                source=SUMMARY_SOURCED,
            )

        result["indeks"] = self.reindex()
        self.session_id = None
        return result

    # -------------------------------------------------------------- recall
    def search(self, query: str, limit: int | None = None) -> list[Hit]:
        if not self.enabled:
            return []
        try:
            return self.store.search(query, limit=limit or self.recall_limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("hafıza aranamadı: %s", exc)
            return []

    def context_for(self, query: str) -> str:
        """A block of remembered material to place in front of the model.

        Labelled as memory rather than fact: it may be stale or simply wrong, and
        the model should prefer what the user says now over what it recorded then.
        """
        hits = self.search(query)
        if not hits:
            return ""
        lines = [
            "## Hafızandan (geçmiş oturumlardan hatırladıkların)",
            "",
            "Bunlar daha önce kaydettiğin notlar. Eskimiş veya yanlış olabilirler;",
            "kullanıcının şimdi söylediği her zaman önceliklidir. Bir notu",
            "kullanırsan bunu doğal biçimde belli et, alıntı yapar gibi değil.",
            "",
            "«doğrulanmadı» etiketli bir not, senin daha önce söylediğinin kaydıdır —",
            "bir kaynağa dayanmıyor. Onu gerçek diye tekrarlama; gerekiyorsa",
            "doğrulanmadığını söyle.",
            "",
            "«kaynaklarla doğrulandı» etiketli bir not, birbirinden bağımsız",
            "kaynaklarla karşılaştırılmış bilgidir. Ona güvenebilirsin, ama",
            "kaynaklar da yanılabilir; not içindeki bağlantılar kontrol edilebilir.",
            "",
        ]
        for hit in hits:
            label = _KIND_LABEL.get(hit.note_kind, hit.note_kind)
            if hit.source in UNVERIFIED_SOURCES:
                mark = "  «doğrulanmadı»"
            elif hit.source == VERIFIED_SOURCED:
                mark = "  «kaynaklarla doğrulandı»"
            else:
                mark = ""
            lines.append(f"### {hit.note_title}  ({label}){mark}")
            lines.append(hit.text.strip())
            lines.append("")
        return "\n".join(lines).strip()

    def recent_summaries(self, limit: int = 3) -> list[str]:
        try:
            rows = self.store.recent_sessions(limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("geçmiş oturumlar okunamadı: %s", exc)
            return []
        out = []
        for row in rows:
            when = datetime.fromtimestamp(row["started"]).strftime("%d.%m %H:%M")
            out.append(f"{when} — {row['summary']}")
        return out

    # ------------------------------------------------------------ indexing
    def reindex(self, *, force: bool = False) -> dict[str, int]:
        try:
            return self.store.reindex(force=force)
        except Exception as exc:  # noqa: BLE001
            log.warning("indeks güncellenemedi: %s", exc)
            return {}

    def stats(self) -> dict[str, object]:
        try:
            data: dict[str, object] = dict(self.store.stats())
        except Exception as exc:  # noqa: BLE001
            log.warning("istatistik alınamadı: %s", exc)
            data = {}
        data["kasa"] = str(self.vault.root)
        data["embed_model"] = self.embedder.model
        return data


__all__ = ["Memory", "Vault", "Store", "Embedder", "Fact", "Hit", "distill", "summarise"]
