"""Turning what was said into text, on this machine.

faster-whisper runs the Whisper model through ctranslate2, which means a real
GPU path without pulling in torch. On the card here that matters: the turbo
model transcribes a few seconds of speech in a fraction of a second, and the
gap between someone finishing a sentence and JARVIS knowing what it was is the
single biggest contributor to a voice assistant feeling slow.

## What arrives here

One segment of speech, already endpointed by the browser: the page watches the
microphone, decides where the user started and stopped, and posts the piece.
The decision about *when* someone finished talking is not made here -- it is
made where the audio is, because a round trip per frame would add latency to
the one thing that must never feel late.

So this module does one job: audio in, text out, with the confidence numbers
that let the caller decide whether to trust it.

## Refusing to guess

Whisper will happily produce fluent text from silence, from a cough, from a fan.
That is its worst failure mode in an assistant, because a hallucinated sentence
becomes a request JARVIS then acts on. Three cheap defences:

- **an energy floor.** A segment that is essentially silence never reaches the
  model at all.
- **`no_speech_prob`.** The model's own estimate that a segment contains no
  speech; above the threshold the transcript is discarded.
- **average log-probability.** Confident nonsense is rare, unconfident nonsense
  is common, and a very low mean logprob is the usual signature.

Every one of those returns an empty transcript with a reason, so the interface
can say "seni duyamadım" rather than acting on noise.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("jarvis.voice.stt")

#: Whisper expects 16 kHz mono.
TARGET_RATE = 16000

#: Below this peak amplitude (of full scale) a segment is silence, and asking a
#: transformer what silence says is how assistants end up acting on nothing.
SILENCE_PEAK = 0.012

#: The model's own "there is no speech here" probability, above which its text
#: is thrown away.
NO_SPEECH_MAX = 0.6

#: Mean log-probability below this is the usual signature of a confident
#: hallucination over noise.
LOGPROB_MIN = -1.1

#: Shorter than this is a click, a breath, or a door.
MIN_SECONDS = 0.25


@dataclass(slots=True)
class Transcript:
    """What was heard, and how much to trust it."""

    text: str = ""
    language: str = ""
    seconds: float = 0.0
    elapsed_s: float = 0.0
    no_speech: float = 0.0
    logprob: float = 0.0
    refused: str = ""
    segments: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.refused

    def as_dict(self) -> dict[str, object]:
        return {
            "metin": self.text,
            "dil": self.language,
            "saniye": round(self.seconds, 2),
            "sure": round(self.elapsed_s, 3),
            "konusma_yok": round(self.no_speech, 3),
            "guven": round(self.logprob, 3),
            "reddedildi": self.refused,
        }


class Listener:
    """The recogniser. Loads on construction, transcribes one segment at a time."""

    def __init__(self, *, model_dir: Path, model_name: str = "large-v3-turbo",
                 device: str = "auto", language: str = "tr") -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.language = language or None
        self._model_dir = Path(model_dir)
        chosen, compute = _pick_device(device)
        started = time.perf_counter()
        try:
            self._model = WhisperModel(model_name, device=chosen,
                                       compute_type=compute,
                                       download_root=str(model_dir))
        except Exception as exc:  # noqa: BLE001 - a missing GPU is not fatal
            if chosen == "cuda":
                log.warning("GPU'da yüklenemedi (%s), CPU'ya düşülüyor", exc)
                chosen, compute = "cpu", "int8"
                self._model = WhisperModel(model_name, device=chosen,
                                           compute_type=compute,
                                           download_root=str(model_dir))
            else:
                raise
        self.device = chosen
        self.compute_type = compute
        self.load_seconds = time.perf_counter() - started
        self._lock = threading.Lock()
        log.info("konuşma tanıma yüklendi: %s · %s/%s · %.1f sn",
                 model_name, chosen, compute, self.load_seconds)

    def _run(self, samples):
        segments, info = self._model.transcribe(
            samples, language=self.language, beam_size=1,
            vad_filter=True, condition_on_previous_text=False)
        return list(segments), info

    def _to_cpu(self) -> bool:
        """Rebuild the model on CPU. False when even that fails."""
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(self.model_name, device="cpu",
                                       compute_type="int8",
                                       download_root=str(self._model_dir))
        except Exception as exc:  # noqa: BLE001 - out of fallbacks
            log.warning("CPU'ya da geçilemedi: %s", exc)
            return False
        self.device, self.compute_type = "cpu", "int8"
        return True

    def hear(self, wav_bytes: bytes) -> Transcript:
        """Transcribe one segment of WAV audio."""
        started = time.perf_counter()
        try:
            samples, rate = _decode(wav_bytes)
        except (wave.Error, EOFError, ValueError) as exc:
            return Transcript(refused=f"ses çözülemedi: {exc}")

        seconds = len(samples) / float(rate or TARGET_RATE)
        if seconds < MIN_SECONDS:
            return Transcript(seconds=seconds, refused="çok kısa")

        peak = float(abs(samples).max()) if samples.size else 0.0
        if peak < SILENCE_PEAK:
            return Transcript(seconds=seconds, refused="sessiz")

        if rate != TARGET_RATE:
            samples = _resample(samples, rate, TARGET_RATE)

        with self._lock:
            try:
                collected, info = self._run(samples)
            except RuntimeError as exc:
                if self.device != "cuda":
                    return Transcript(seconds=seconds,
                                      refused=f"çözümlenemedi: {exc}")
                # The GPU libraries are only loaded when work is done, so a
                # broken CUDA install passes construction and fails here. Move
                # to CPU for good rather than rediscovering this every turn.
                log.warning("GPU'da çözümlenemedi (%s) — CPU'ya geçiliyor", exc)
                if not self._to_cpu():
                    return Transcript(seconds=seconds,
                                      refused=f"çözümlenemedi: {exc}")
                collected, info = self._run(samples)

        elapsed = time.perf_counter() - started
        text = " ".join(s.text.strip() for s in collected).strip()
        no_speech = max((s.no_speech_prob for s in collected), default=0.0)
        logprob = (sum(s.avg_logprob for s in collected) / len(collected)
                   if collected else 0.0)

        result = Transcript(
            text=text, language=getattr(info, "language", "") or "",
            seconds=seconds, elapsed_s=elapsed, no_speech=no_speech,
            logprob=logprob, segments=[s.text.strip() for s in collected])

        if not text:
            result.refused = "konuşma bulunamadı"
        elif no_speech > NO_SPEECH_MAX:
            result.text, result.refused = "", "konuşma değil"
        elif logprob < LOGPROB_MIN:
            result.text, result.refused = "", "anlaşılmadı"
        return result


# ------------------------------------------------------------------ helpers
def _register_cuda_libraries() -> int:
    """Put pip's CUDA DLLs on the search path. Returns how many were added.

    `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` install into the `nvidia`
    package and nothing on Windows looks there. Without this the model loads,
    claims CUDA, and dies on the first encode -- so the failure arrives at the
    user's first sentence rather than at startup.
    """
    try:
        import nvidia
    except ImportError:
        return 0

    directories = [entry for root in getattr(nvidia, "__path__", [])
                   for entry in Path(root).glob("*/bin") if entry.is_dir()]
    if not directories:
        return 0

    # Two mechanisms, because they cover different loaders. add_dll_directory
    # governs what Python loads; ctranslate2's extension resolves its own
    # dependencies through the ordinary Windows search, which reads PATH.
    # Measured: with only the first, the model loads and then fails on the first
    # encode with "cublas64_12.dll is not found".
    if hasattr(os, "add_dll_directory"):
        for entry in directories:
            try:
                os.add_dll_directory(str(entry))
            except OSError as exc:  # noqa: PERF203 - one bad directory is not fatal
                log.debug("DLL dizini eklenemedi (%s): %s", entry, exc)

    current = os.environ.get("PATH", "")
    missing = [str(d) for d in directories if str(d) not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing) + os.pathsep + current
    log.debug("%s CUDA kütüphane dizini kaydedildi", len(directories))
    return len(directories)


def _pick_device(preference: str) -> tuple[str, str]:
    """Choose where to run. 'auto' takes the GPU when ctranslate2 sees one."""
    choice = (preference or "auto").lower()
    if choice == "cpu":
        return "cpu", "int8"
    _register_cuda_libraries()
    if choice == "cuda":
        return "cuda", "float16"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as exc:  # noqa: BLE001 - no CUDA is an answer, not an error
        log.debug("CUDA sorgulanamadı: %s", exc)
    return "cpu", "int8"


def _decode(wav_bytes: bytes):
    """WAV bytes to mono float32 in [-1, 1], plus its sample rate."""
    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"desteklenmeyen örnek genişliği: {width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _resample(samples, source_rate: int, target_rate: int):
    """Linear resampling. Whisper is unbothered by it and it needs no scipy."""
    import numpy as np

    if source_rate == target_rate or samples.size == 0:
        return samples
    count = int(round(len(samples) * target_rate / float(source_rate)))
    if count <= 0:
        return samples[:0]
    source_index = np.linspace(0, len(samples) - 1, num=count, dtype=np.float64)
    return np.interp(source_index, np.arange(len(samples)), samples).astype("float32")


__all__ = ["Listener", "Transcript", "TARGET_RATE", "SILENCE_PEAK",
           "NO_SPEECH_MAX", "LOGPROB_MIN", "MIN_SECONDS"]
