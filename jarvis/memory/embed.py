"""Embeddings from the local Ollama server.

bge-m3 is multilingual and handles Turkish properly, which most English-first
embedding models do not — "görev kuyruğu" and "task queue" should land near each
other, and a model trained only on English will not put them there.

Vectors are stored as raw float32 bytes rather than JSON: a 1024-dimension vector
is 4 KB packed and roughly 20 KB as text, and the database holds one per chunk.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from array import array
from collections.abc import Callable

log = logging.getLogger("jarvis.memory.embed")

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected but not required
    np = None


def pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack(blob: bytes) -> list[float]:
    vector = array("f")
    vector.frombytes(blob)
    return list(vector)


class Embedder:
    def __init__(self, *, host: str, model: str = "bge-m3", timeout_s: int = 120,
                 usage: Callable[..., None] | None = None) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._dim: int | None = None
        #: Same ledger as the chat client. Embedding is a model call and costs
        #: real seconds — a reindex that took 2.6s of GPU time used to appear
        #: nowhere at all.
        self.usage = usage

    @property
    def dim(self) -> int | None:
        return self._dim

    def available(self) -> bool:
        try:
            self.embed(["ping"])
            return True
        except OSError as exc:
            log.debug("embedder unavailable: %s", exc)
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            self._record(started, ok=False, error=str(exc))
            raise
        vectors = data.get("embeddings") or []
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
        self._record(started, ok=True,
                     input_tokens=int(data.get("prompt_eval_count") or 0))
        return vectors

    def _record(self, started: float, *, ok: bool, input_tokens: int = 0,
                error: str | None = None) -> None:
        if self.usage is None:
            return
        try:
            self.usage(tier="local", model=self.model, purpose="embedding",
                       input_tokens=input_tokens,
                       duration_ms=int((time.monotonic() - started) * 1000),
                       ok=ok, error=error)
        except Exception as exc:  # noqa: BLE001 - accounting must not break a reindex
            log.debug("embedding kullanımı kaydedilemedi: %s", exc)

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed([text])
        return vectors[0] if vectors else []

    def unload(self) -> bool:
        """Release the embedding model instead of waiting for Ollama's timer."""
        payload = json.dumps({"model": self.model, "prompt": "",
                              "keep_alive": 0, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_s, 20)):
                return True
        except OSError as exc:
            log.debug("embedding modeli bellekten bırakılamadı: %s", exc)
            return False


def cosine_ranking(query: list[float], blobs: list[bytes]) -> list[float]:
    """Similarity of one query vector against many stored vectors.

    Uses numpy when present; the pure-Python path is correct but noticeably slower
    once the vault passes a few thousand chunks.
    """
    if not query or not blobs:
        return [0.0] * len(blobs)

    if np is not None:
        matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), -1)
        vector = np.asarray(query, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(vector))
        norms[norms == 0] = 1.0
        return (matrix @ vector / norms).tolist()

    scores = []
    q_norm = sum(v * v for v in query) ** 0.5 or 1.0
    for blob in blobs:
        stored = unpack(blob)
        dot = sum(a * b for a, b in zip(query, stored))
        norm = (sum(v * v for v in stored) ** 0.5 or 1.0) * q_norm
        scores.append(dot / norm)
    return scores
