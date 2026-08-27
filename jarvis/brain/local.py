"""Tier 1 brain: the local model served by Ollama.

Free, unlimited, offline, and always the fallback when the metered tier is
unavailable or exhausted. Weaker at planning and code than Tier 2 — the router
decides when that matters.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

log = logging.getLogger("jarvis.brain.local")

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


def strip_thinking(text: str) -> str:
    """qwen3 and similar models emit <think>…</think>. The user never sees it."""
    return _THINK_BLOCK.sub("", text).strip()


class ThinkFilter:
    """Removes <think>…</think> from a token stream without buffering the whole reply.

    Holds back only enough trailing text to recognise a tag split across chunks.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        self._buf += piece
        out: list[str] = []
        while self._buf:
            if self._inside:
                idx = self._buf.find(CLOSE_TAG)
                if idx == -1:
                    # Discard thinking content, keep a possible partial closing tag.
                    self._buf = self._buf[-(len(CLOSE_TAG) - 1) :]
                    break
                self._buf = self._buf[idx + len(CLOSE_TAG) :]
                self._inside = False
                continue

            idx = self._buf.find(OPEN_TAG)
            if idx == -1:
                hold = len(OPEN_TAG) - 1
                if len(self._buf) > hold:
                    out.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                break
            out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(OPEN_TAG) :]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        if self._inside:
            self._buf = ""
            return ""
        rest, self._buf = self._buf, ""
        return rest


class LocalBrain:
    def __init__(
        self,
        *,
        host: str,
        model: str,
        timeout_s: int = 300,
        keep_alive: str = "30m",
        think: bool = False,
        temperature: float = 0.7,
        context_window: int = 8192,
        usage: Callable[..., None] | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.think = think
        self.temperature = temperature
        self.context_window = max(2048, int(context_window))
        #: Called after every completed call, with the keywords the ledger takes.
        #:
        #: It lives here rather than at the call sites because the alternative was
        #: measured and found wanting. Accounting used to sit in `Brain.ask()`,
        #: but the improvement engine, the agents and the research pipeline all
        #: reach for `chat()` directly — so a soak that ran the local model twice
        #: reported zero calls, and "yerel model kullanımı" was unmeasurable for
        #: exactly the work nobody was watching.
        self.usage = usage

    # ------------------------------------------------------------------ state
    def available(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except OSError as exc:
            log.debug("ollama unreachable: %s", exc)
            return False

    def models(self) -> list[str]:
        data = self._get("/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]

    def has_model(self) -> bool:
        try:
            names = self.models()
        except OSError:
            return False
        return any(n == self.model or n.startswith(self.model.split(":")[0]) for n in names)

    def warm(self) -> None:
        """Load the model into VRAM so the first real question is not slow."""
        try:
            self._post("/api/chat", self._payload([{"role": "user", "content": "ok"}], stream=False))
        except OSError as exc:
            log.debug("warmup failed: %s", exc)

    def unload(self, model: str | None = None) -> bool:
        """Ask Ollama to release this model from VRAM immediately."""
        try:
            self._post("/api/generate", {
                "model": model or self.model, "prompt": "", "keep_alive": 0,
                "stream": False,
            })
            return True
        except OSError as exc:
            log.debug("model bellekten bırakılamadı: %s", exc)
            return False

    # ------------------------------------------------------------------- chat
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        think: bool | None = None,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        purpose: str = "yerel",
    ) -> str:
        """Answer once. Pass `schema` to constrain the reply to a JSON shape.

        `model` overrides the configured one for this call. Loading a second model
        evicts the first on a single-GPU machine, so callers that switch should
        switch rarely and deliberately.

        `purpose` is what the call gets recorded as. Every path through here is
        counted, including the ones that never touch `Brain.ask()`.
        """
        payload = self._payload(messages, think, temperature, stream=False, model=model)
        if schema is not None:
            payload["format"] = schema
        started = time.monotonic()
        try:
            data = self._post("/api/chat", payload)
        except OSError as exc:
            self._record(purpose, model or self.model, started, ok=False, error=str(exc))
            raise
        # Ollama reports what it actually processed; nothing here is an estimate.
        self._record(purpose, data.get("model") or model or self.model, started, ok=True,
                     input_tokens=int(data.get("prompt_eval_count") or 0),
                     output_tokens=int(data.get("eval_count") or 0))
        return strip_thinking(data.get("message", {}).get("content", ""))

    def _record(self, purpose: str, model: str, started: float, *, ok: bool,
                input_tokens: int = 0, output_tokens: int = 0,
                error: str | None = None) -> None:
        if self.usage is None:
            return
        try:
            self.usage(tier="local", model=model, purpose=purpose,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       duration_ms=int((time.monotonic() - started) * 1000),
                       ok=ok, error=error)
        except Exception as exc:  # noqa: BLE001 - accounting must not break the call
            log.debug("kullanım kaydedilemedi: %s", exc)

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        think: bool | None = None,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        purpose: str = "yerel-akis",
        record: bool = True,
    ) -> Iterator[str]:
        """Answer incrementally. Pass `schema` to constrain the JSON shape.

        Used by the voice path: a reply can start being spoken while the model
        is still generating it. The final chunk carries the real token counts
        and they are booked here -- so a caller that reaches this method
        directly (assistant streaming) finally lands in the ledger.

        `record=False` is for the one wrapper that books the same call itself
        (Brain.stream, with its own purpose label): one call, one row, and the
        pair cannot fork because the guard test pins the flag at the call site.
        """
        payload = self._payload(messages, think, temperature, stream=True,
                                model=model)
        if schema is not None:
            payload["format"] = schema
        request = self._request("/api/chat", payload)
        started = time.monotonic()
        filt = ThinkFilter()
        prompt_tokens = eval_tokens = 0
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for raw in response:
                    line = raw.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise OSError(f"ollama: {chunk['error']}")
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        visible = filt.feed(piece)
                        if visible:
                            yield visible
                    if chunk.get("done"):
                        prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
                        eval_tokens = int(chunk.get("eval_count") or 0)
                        break
            tail = filt.flush()
            if tail:
                yield tail
        except OSError as exc:
            if record:
                self._record(purpose, model or self.model, started, ok=False,
                             error=str(exc))
            raise
        if record:
            self._record(purpose, model or self.model, started, ok=True,
                         input_tokens=prompt_tokens, output_tokens=eval_tokens)

    # ---------------------------------------------------------------- plumbing
    def _payload(
        self,
        messages: list[dict[str, str]],
        think: bool | None = None,
        temperature: float | None = None,
        *,
        stream: bool,
        model: str | None = None,
    ) -> dict[str, Any]:
        return {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "think": self.think if think is None else think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.context_window,
            },
        }

    def _request(self, route: str, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self.host + route,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        with urllib.request.urlopen(self._request(route, payload), timeout=self.timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, route: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.host + route, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
