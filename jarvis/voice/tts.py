"""Turning an answer into speech, one sentence at a time.

Piper runs locally and fast -- measured on this machine at RTF 0.02, meaning a
three-second sentence takes about eighty milliseconds to produce. That number is
what makes the design here possible: there is no reason to synthesise a long
answer as one block and make the user wait for the whole thing.

## Why sentences

A long reply is split and spoken sentence by sentence, for three reasons that
all matter more than they sound:

- **the first word arrives sooner.** The user hears an answer beginning while
  the rest is still being made.
- **an interruption can land.** Speech that was queued as one four-minute blob
  cannot be stopped cleanly halfway; a queue of short pieces can be dropped
  between them, which is what barge-in needs.
- **it sounds less like reading.** Piper puts a natural boundary at the end of
  each utterance, so sentence-sized pieces carry sentence-shaped prosody.

Splitting is deliberately conservative. Turkish abbreviations ("vb.", "Dr.")
and decimals ("3.14.6") end in a full stop that does not end a sentence, and
cutting there produces the stutter people notice immediately in synthetic
speech.

## What it does not do

No rate control, no pitch shifting, no "emotion" parameters. Piper's voice is
what it is; dressing it up with prosody tricks would be inventing expression
JARVIS does not have.
"""

from __future__ import annotations

import io
import html
import logging
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("jarvis.voice.tts")

#: A blank line is a pause whatever the punctuation says.
_PARAGRAPH = re.compile(r"\n\s*\n")

#: Sentence ends, unless the thing before the stop says otherwise.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n{2,}")

#: A full stop after one of these is not the end of anything.
_ABBREVIATIONS = frozenset({
    "vb", "vs", "dr", "prof", "doç", "av", "sn", "bkz", "örn", "yy", "no",
    "mr", "mrs", "st", "etc", "ör", "bk", "çev", "haz", "yay",
})

#: Below this a "sentence" is punctuation or an artefact; glue it to the next.
MIN_PIECE = 3

#: Piper is fast but not free, and one utterance this long is a paragraph the
#: user cannot interrupt cleanly.
MAX_PIECE = 400

_FENCED_CODE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|mailto:)[^)]+\)")
_BARE_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$", re.MULTILINE)
_LINE_MARKER = re.compile(r"(?m)^\s{0,3}(?:#{1,6}|>|[-*+] |\d+[.)] )\s*")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")


def prepare_for_speech(text: str) -> str:
    """Turn screen-oriented prose into something a person would actually say.

    The visual answer remains untouched.  Only the TTS copy loses Markdown,
    raw URLs and long code blocks; otherwise a useful answer can turn into a
    minute of punctuation and protocol names when read aloud.
    """
    body = html.unescape(str(text or ""))
    body = _FENCED_CODE.sub(" Kod bloğunu ekranda gösteriyorum. ", body)
    body = _MARKDOWN_LINK.sub(r"\1", body)
    body = _BARE_URL.sub(" bağlantı ", body)
    body = _TABLE_RULE.sub(" ", body)
    body = _LINE_MARKER.sub("", body)
    body = body.replace("`", "").replace("**", "").replace("__", "")
    body = body.replace("ZESTOLES", "Zestoles")
    body = _EMOJI.sub(" ", body)
    # Keep paragraph pauses; sentence splitting intentionally uses them even
    # when the author omitted final punctuation.
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" *\n *", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


@dataclass(slots=True)
class Speech:
    """One synthesised piece: the audio, and what it cost to make."""

    wav: bytes
    text: str
    seconds: float
    generated_s: float
    sample_rate: int

    @property
    def real_time_factor(self) -> float:
        return self.generated_s / self.seconds if self.seconds else 0.0


def split_sentences(text: str) -> list[str]:
    """Break an answer where a speaker would pause. Pure, so it can be tested.

    Conservative on purpose: a wrong split is audible immediately -- "Python
    üç nokta on dört" read as three sentences is the artefact everyone
    recognises as a machine reading text.
    """
    raw = prepare_for_speech(text)
    if not raw:
        return []
    # Paragraphs first: collapsing whitespace before looking for the break
    # destroys the very thing being looked for. Each paragraph is tidied on its
    # own afterwards.
    paragraphs = [" ".join(p.split()) for p in _PARAGRAPH.split(raw)]
    body = "\n\n".join(p for p in paragraphs if p)
    if not body:
        return []

    pieces: list[str] = []
    for candidate in _BOUNDARY.split(body):
        candidate = candidate.strip()
        if not candidate:
            continue
        if pieces and _continues(pieces[-1], candidate):
            pieces[-1] = f"{pieces[-1]} {candidate}"
        else:
            pieces.append(candidate)

    out: list[str] = []
    for piece in pieces:
        if out and len(piece) < MIN_PIECE:
            out[-1] = f"{out[-1]} {piece}"
            continue
        out.extend(_chunk(piece) if len(piece) > MAX_PIECE else [piece])
    return out


def _continues(previous: str, following: str) -> bool:
    """Whether a split was really the middle of something."""
    tail = previous.rstrip(".").rsplit(" ", 1)[-1].lower()
    if tail in _ABBREVIATIONS:
        return True
    # A number on both sides of the stop: a version, a date, a decimal.
    if previous[-2:-1].isdigit() and following[:1].isdigit():
        return True
    # A single capital before the stop is an initial, not an ending.
    return len(tail) == 1 and tail.isalpha()


def _chunk(piece: str) -> list[str]:
    """Cut an over-long sentence at commas, then at spaces if it has to."""
    out: list[str] = []
    rest = piece
    while len(rest) > MAX_PIECE:
        window = rest[:MAX_PIECE]
        cut = max(window.rfind(", "), window.rfind("; "), window.rfind(" ve "))
        if cut < MAX_PIECE // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = MAX_PIECE
        out.append(rest[:cut].strip(" ,;"))
        rest = rest[cut:].strip(" ,;")
    if rest:
        out.append(rest)
    return out


class Speaker:
    """Piper, loaded once and used from whichever thread asks.

    onnxruntime sessions are not documented as thread-safe for concurrent
    `run` calls, and the interface can plausibly ask for two pieces at once, so
    synthesis is serialised. At RTF 0.02 the lock is never the thing anyone is
    waiting for.
    """

    def __init__(self, model_path: Path) -> None:
        from piper import PiperVoice

        self.model_path = Path(model_path)
        started = time.perf_counter()
        self._voice = PiperVoice.load(self.model_path, use_cuda=False)
        self.load_seconds = time.perf_counter() - started
        self.sample_rate = int(self._voice.config.sample_rate)
        self._lock = threading.Lock()
        log.info("ses modeli yüklendi: %s (%.2f sn, %s Hz)",
                 self.model_path.stem, self.load_seconds, self.sample_rate)

    def say(self, text: str) -> Speech | None:
        """Synthesise one piece. None when there is nothing to say."""
        body = prepare_for_speech(text)
        if not body:
            return None
        buffer = io.BytesIO()
        started = time.perf_counter()
        with self._lock:
            with wave.open(buffer, "wb") as wav:
                self._voice.synthesize_wav(body, wav)
        generated = time.perf_counter() - started
        data = buffer.getvalue()
        return Speech(wav=data, text=body, seconds=_duration(data),
                      generated_s=generated, sample_rate=self.sample_rate)

    def pieces(self, text: str) -> list[str]:
        return split_sentences(text)


def _duration(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except (wave.Error, EOFError):
        return 0.0


__all__ = ["Speaker", "Speech", "split_sentences", "prepare_for_speech",
           "MIN_PIECE", "MAX_PIECE"]
