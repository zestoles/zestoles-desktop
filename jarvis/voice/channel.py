"""The request channel for speech: audio in, an answer and audio back.

The page owns the microphone and the clock. It decides when the user started
speaking, when they stopped, and when to stop JARVIS mid-sentence -- all three
have to happen in milliseconds, and a round trip per decision would put the
network in the middle of the one thing that must never feel late.

So this side is deliberately request-shaped. One finished segment of speech
arrives; it is transcribed, answered, and spoken. Everything real-time lives in
the browser.

## Why the answer comes back in pieces

A reply is synthesised sentence by sentence and returned as a list of clips.
The page plays them in order and can drop the queue the instant the user starts
talking. One long clip could not be stopped cleanly, and stopping cleanly is
what makes an interruption feel like a conversation rather than a bug.

## The acknowledgements are made ahead of time

"bir saniye" has to arrive while the user is waiting, so it cannot be
synthesised after the wait has started. The short phrases are rendered once, in
the background, the first time voice is used, and served from memory afterwards
-- about thirty kilobytes for the lot.

## What this does not decide

Nothing here relaxes a permission. A spoken request goes through the same
`Assistant`, the same risk tiers and the same confirmation as a typed one, and
a MEDIUM tool called by voice still stops and waits for a person to agree.
Speech is an input method, not a credential.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backchannel import Backchannel, looks_like_question
from .tts import Speech

log = logging.getLogger("jarvis.voice.channel")

#: Longer than this and it is not a spoken request any more.
MAX_AUDIO_BYTES = 12 * 1024 * 1024


@dataclass(slots=True)
class Clip:
    """One piece of speech, ready for the page to play."""

    text: str
    wav_base64: str
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {"metin": self.text, "ses": self.wav_base64,
                "saniye": round(self.seconds, 2)}


@dataclass(slots=True)
class Timing:
    """Where the time went, so latency is measured rather than estimated."""

    stt_s: float = 0.0
    think_s: float = 0.0
    tts_s: float = 0.0
    first_audio_s: float = 0.0
    total_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"stt": round(self.stt_s, 3), "dusunme": round(self.think_s, 3),
                "tts": round(self.tts_s, 3),
                "ilk_ses": round(self.first_audio_s, 3),
                "toplam": round(self.total_s, 3)}


@dataclass
class VoiceChannel:
    """Wires the voice engines to the assistant service.

    `publish`, verildiğinde cevabın *ilk cümlesi* yanıtla döner, kalanı arka
    planda sentezlenip olay yoluyla akıtılır -- kullanıcı ilk sözcüğü bir
    robotun tüm cevabı hazırlamasını beklemeden duyar. None ise eski davranış:
    her parça hazır olmadan yanıt çıkmaz. Akış, olay yolu olmayan kurulumda
    bilinçli olarak devre dışıdır; sessizlik yerine yapısal kapı.
    """

    voice: Any
    service: Any
    backchannel: Backchannel = field(default_factory=Backchannel)
    publish: Any = None
    _cues: dict[str, Clip] = field(default_factory=dict, init=False)
    _cue_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _prepared: bool = field(default=False, init=False)

    # --------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        state = self.voice.status()
        state["ipuclari"] = sorted(self._cues)
        return state

    def shutdown(self) -> None:
        """Release an already loaded TTS sidecar without loading it on exit."""
        speaker = getattr(self.voice, "_speaker", None)
        close = getattr(speaker, "shutdown", None)
        if callable(close):
            close()

    # ---------------------------------------------------------------- speak
    def speak(self, text: str) -> dict[str, Any]:
        """Synthesise text the page already has. Used for replaying an answer."""
        speaker = self.voice.speaker()
        if speaker is None:
            return {"hata": self._why_no_voice("tts")}
        clips, seconds = self._render(speaker, text)
        return {"parcalar": [c.as_dict() for c in clips],
                "sure": round(seconds, 3)}

    # ----------------------------------------------------------------- hear
    def listen(self, audio_base64: str, *, spoken_seconds: float = 0.0,
               client_turn: str = ""
               ) -> dict[str, Any]:
        """One spoken turn: transcribe it, answer it, and speak the answer."""
        started = time.perf_counter()
        listener = self.voice.listener()
        if listener is None:
            return {"hata": self._why_no_voice("stt")}

        try:
            raw = base64.b64decode(audio_base64 or "", validate=False)
        except (ValueError, TypeError):
            return {"hata": "ses verisi çözülemedi"}
        if not raw:
            return {"hata": "ses verisi boş"}
        if len(raw) > MAX_AUDIO_BYTES:
            return {"hata": "ses kaydı çok uzun"}

        heard = listener.hear(raw)
        self._capture(heard, raw)
        timing = Timing(stt_s=heard.elapsed_s)
        self.backchannel.turn_finished()

        if not heard.ok:
            # Say why rather than answering something nobody said. "sessiz" and
            # "anlaşılmadı" need different reactions from the user.
            timing.total_s = time.perf_counter() - started
            return {"duyulmadi": True, "sebep": heard.refused or "anlaşılmadı",
                    "isitilen": heard.as_dict(), "sure": timing.as_dict()}

        # The browser creates the token before uploading.  That lets it reject
        # a late clip from the turn the user just interrupted, including the
        # awkward case where the clip arrives before the HTTP response.  Other
        # clients need not know about this and still receive a fresh token.
        requested_turn = re.sub(r"[^A-Za-z0-9_-]", "", client_turn or "")[:48]
        turn = requested_turn or uuid.uuid4().hex[:8]
        spoken: list[str] = []
        early_finished: list[float] = []
        early_jobs: list[tuple[str, concurrent.futures.Future]] = []
        early_pool = (concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="zestoles-erken-ses")
            if self.publish is not None else None)

        def render_early(text: str):
            clip = self._speak_early(text, turn)
            if clip is not None:
                early_finished.append(time.perf_counter())
            return clip

        def sink(text: str) -> None:
            # Synthesis no longer holds the model stream.  One worker preserves
            # sentence order while Ollama keeps producing the rest of the reply.
            if early_pool is not None:
                early_jobs.append((text, early_pool.submit(render_early, text)))

        think_started = time.perf_counter()
        request = {"op": "sor", "mesaj": heard.text, "cumle_akin": sink}
        answer = self.service.handle(request)
        # A new utterance may finish transcription while the cancelled turn is
        # releasing the shared conversation lock.  Briefly waiting here turns
        # that race into a natural interruption instead of asking the user to
        # repeat themselves.  No request is duplicated: a busy response means
        # `_ask` never started.
        retry_until = time.monotonic() + 2.0
        while (str(answer.get("hata", "")).casefold() == "meşgul"
               and time.monotonic() < retry_until):
            time.sleep(0.05)
            answer = self.service.handle(request)
        timing.think_s = time.perf_counter() - think_started
        if early_pool is not None:
            early_pool.shutdown(wait=True)
            for text, future in early_jobs:
                try:
                    if future.result() is not None:
                        spoken.append(text)
                except Exception as exc:  # pragma: no cover - worker is guarded
                    log.warning("erken ses işi tamamlanamadı: %s", exc)
        if early_finished:
            timing.first_audio_s = early_finished[0] - started

        reply = str(answer.get("cevap") or "")
        if answer.get("hata") and not reply:
            # Sessiz hata, boşluğa konuşmaktır: meşgul ya da başarısız bir
            # tur en azından duyulabilir olmalı.
            reply = ("Şu an başka bir işle meşgulüm; bitince tekrar söyler misin?"
                     if "meşgul" in str(answer.get("hata"))
                     else "Bir sorun oldu, bunu şimdi yapamıyorum.")
        if answer.get("durum") == "onay_bekliyor":
            pending = answer.get("bekleyen") or {}
            reply = (f"{pending.get('arac', 'bir işlem')} için onayın gerekiyor. "
                     "Onaylıyor musun?")
        # Model konuşurken söylenmiş cümleler `spoken`'da. Cevap içinde
        # sırayla yerleri bulunur; ARADA KALAN bölümler (sentezi patlayan,
        # yayınlanamayan) toplanıp normal yoldan söylenmek üzere sıraya girer
        # -- çünkü ses kuyruğu zaten erken cümlelerle başlıyor, eklenenler
        # kronolojik sırada devam eder. Cevapta hiçbir söylenen bulunamazsa
        # güvenli taraf: her şeyi baştan söylemek.
        remainder_parts: list[str] = []
        cursor = 0
        streaming = bool(spoken)
        for said in spoken:
            at = reply.find(said, cursor)
            if at < 0:
                remainder_parts = [reply]
                streaming = False
                break
            gap = reply[cursor:at].strip()
            if gap:
                remainder_parts.append(gap)
            cursor = at + len(said)
        else:
            tail = reply[cursor:].strip()
            if tail:
                remainder_parts.append(tail)
        remainder_text = " ".join(remainder_parts).strip()

        clips: list[Clip] = []
        if remainder_text:
            speaker = self.voice.speaker()
            if speaker is not None:
                tts_started = time.perf_counter()
                for piece in speaker.pieces(remainder_text):
                    try:
                        clip = self._one(speaker, piece)
                    except Exception as exc:  # noqa: BLE001 - tek parça turi düşürmez
                        log.warning("parça sentezlenemedi (%s): %s", piece[:40], exc)
                        continue
                    if clip is not None:
                        clips.append(clip)
                timing.tts_s = time.perf_counter() - tts_started

        timing.total_s = time.perf_counter() - started
        log.info("sesli tur: %.2f sn (stt %.2f · dusunme %.2f · tts %.2f · ilk ses %.2f) · duyulan=%r",
                 timing.total_s, timing.stt_s, timing.think_s, timing.tts_s,
                 timing.first_audio_s,
                 (heard.text or heard.refused)[:80])
        return {
            "isitilen": heard.as_dict(),
            "metin": heard.text,
            "cevap": reply,
            "parcalar": [c.as_dict() for c in clips],
            "tur": turn,
            "akis": streaming,
            "sure": timing.as_dict(),
            **{k: v for k, v in answer.items() if k != "cevap"},
        }

    def _capture(self, heard, raw: bytes) -> None:
        """Gerçek mikrofonun kanıtını diske bırak: son 10 segment.

        "Ne dediğimi anlamıyor" şikâyeti üç farklı katmandan gelebilir:
        tarayıcının uç noktalaması cümleyi yarıda kesmiş olabilir, mikrofon
        gürültülü olabilir, ya da Whisper'ın güven eşikleri doğru metni
        reddetmiş olabilir. Bunları ayırt etmenin tek yolu, gelen sesi ve
        çözümleyicinin kararını yan yana görmektir -- bu dosyalar odur.
        """
        try:
            base = (Path(__file__).resolve().parents[2] / "data" / "ses"
                    / "kayit")
            base.mkdir(parents=True, exist_ok=True)
            wavs = sorted(base.glob("*.wav"))
            for old in wavs[:-9]:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
            stamp = time.strftime("%H%M%S")
            (base / f"{stamp}.wav").write_bytes(raw)
            (base / f"{stamp}.json").write_text(
                json.dumps(heard.as_dict(), ensure_ascii=False),
                encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - kanıt, turi durdurmaz
            log.debug("ses kaydı saklanamadı: %s", exc)

    def _speak_early(self, text: str, turn: str) -> Clip | None:
        """Model yazarken bir cümleyi sentezleyip olay yoluyla yolla.

        Bu çağrı turun içinden gelir ve onu bekletmemeli: patlayan sentez ya da
        yayın turu düşürmez, yalnız o cümle sessiz kalır.
        """
        if self.publish is None:
            return None
        speaker = self.voice.speaker()
        if speaker is None:
            return None
        try:
            clip = self._one(speaker, text)
            if clip is None:
                return None
            self.publish({"tur": turn, "parcalar": [clip.as_dict()]})
        except Exception as exc:  # noqa: BLE001 - erken cümle turi durdurmaz
            log.warning("erken cümle gönderilemedi: %s", exc)
            return None
        return clip

    # ------------------------------------------------------------ internals
    def _one(self, speaker, piece: str) -> Clip | None:
        """Tek bir parçayı sentezle; boş üretim kaybolmaz."""
        speech = speaker.say(piece)
        if speech is None:
            return None
        return Clip(text=piece,
                    wav_base64=base64.b64encode(speech.wav).decode("ascii"),
                    seconds=speech.seconds)

    # --------------------------------------------------------- backchannel
    def acknowledge(self, *, speech_seconds: float, now: float | None = None,
                    confidence: float = 1.0, partial: str = "",
                    thinking: bool = False) -> dict[str, Any]:
        """Say a short thing, or say nothing and why.

        Two different moments, two different vocabularies: `thinking` is the
        pause after the user finished, while the answer is being made.
        "bakıyorum" while they are still talking would be answering a question
        they have not asked.
        """
        moment = now if now is not None else time.monotonic()
        if thinking:
            decision = self.backchannel.while_thinking(
                expected_wait_s=max(speech_seconds, 2.0), now=moment)
        else:
            decision = self.backchannel.while_listening(
                speech_seconds=speech_seconds, now=moment,
                confidence=confidence,
                looks_like_question=looks_like_question(partial))
        if not decision.speak:
            return {"sessiz": True, "sebep": decision.reason}
        clip = self._cue(decision.phrase)
        if clip is None:
            return {"sessiz": True, "sebep": "ses hazır değil"}
        return {"parcalar": [clip.as_dict()], "tur": decision.kind}

    def prepare(self) -> dict[str, Any]:
        """Render the short phrases once, so they are instant when wanted."""
        speaker = self.voice.speaker()
        if speaker is None:
            return {"hata": self._why_no_voice("tts")}
        started = time.perf_counter()
        made = 0
        for phrase in self.backchannel.phrases:
            if self._cue(phrase) is not None:
                made += 1
        self._prepared = True
        return {"hazir": made, "sure": round(time.perf_counter() - started, 2)}

    # ------------------------------------------------------------ internals
    def _render(self, speaker, text: str) -> tuple[list[Clip], float]:
        clips: list[Clip] = []
        total = 0.0
        for piece in speaker.pieces(text):
            speech = speaker.say(piece)
            if speech is None:
                continue
            clips.append(Clip(text=piece,
                              wav_base64=base64.b64encode(speech.wav).decode("ascii"),
                              seconds=speech.seconds))
            total += speech.seconds
        return clips, total

    def _cue(self, phrase: str) -> Clip | None:
        with self._cue_lock:
            if phrase in self._cues:
                return self._cues[phrase]
        speaker = self.voice.speaker()
        if speaker is None:
            return None
        speech = speaker.say(phrase)
        if speech is None:
            return None
        clip = Clip(text=phrase,
                    wav_base64=base64.b64encode(speech.wav).decode("ascii"),
                    seconds=speech.seconds)
        with self._cue_lock:
            self._cues[phrase] = clip
        return clip

    def _why_no_voice(self, half: str) -> str:
        state = self.voice.status()
        entry = state.get(half) or {}
        return (entry.get("sebep") or state.get(f"{half}_hata")
                or "ses katmanı kullanılamıyor")


__all__ = ["VoiceChannel", "Clip", "Timing", "MAX_AUDIO_BYTES"]
