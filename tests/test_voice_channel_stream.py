"""Akışkan sesli cevap: model yazarken konuşmaya başlamak.

Model yok, ağ yok. Sahte servis, turun İÇİNDEN cumle_akin'i çağırarak gerçek
asistanın erken cümle akışını taklit eder; sahte hoparlör anında sentezler.
Denetlenen sözleşme: erken cümleler olay yoluyla tur anahtarıyla gider, yanıt
yalnızca kalanı taşır, önek uyuşmazsa güvenli taraf (tamamını normal yoldan
söylemek) seçilir.
"""

import base64
import unittest

from jarvis.voice.channel import VoiceChannel
from jarvis.voice.stt import Transcript
from jarvis.voice.tts import Speech, split_sentences


def _wav() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class _FakeSpeaker:
    def __init__(self) -> None:
        self.boom_on: str | None = None   # bir kez patlar: geçici arıza taklidi
        self.count = 0

    def pieces(self, text: str) -> list[str]:
        return split_sentences(text)

    def say(self, text: str) -> Speech | None:
        if self.boom_on and text.startswith(self.boom_on):
            self.boom_on = None
            raise RuntimeError("patla")
        self.count += 1
        return Speech(wav=_wav(), text=text, seconds=0.1,
                      generated_s=0.001, sample_rate=16000)


class _FakeListener:
    def hear(self, raw: bytes) -> Transcript:
        return Transcript(text="merhaba", language="tr", seconds=1.0)


class _FakeVoice:
    def __init__(self) -> None:
        self.speaker_instance = _FakeSpeaker()

    def listener(self):
        return _FakeListener()

    def speaker(self):
        return self.speaker_instance

    def status(self) -> dict:
        return {}


class _StreamingService:
    """Turun içinden erken cümleleri akıtan sahte servis."""

    def __init__(self, reply: str, early: list[str]) -> None:
        self.reply = reply
        self.early = early

    def handle(self, request: dict) -> dict:
        sink = request.get("cumle_akin")
        if sink:
            for sentence in self.early:
                sink(sentence)
        return {"durum": "ok", "cevap": self.reply}


class _ReleasingService(_StreamingService):
    """A cancelled turn still owns the lock for two short polls."""

    def __init__(self) -> None:
        super().__init__("Yeni sözünü duydum.", early=[])
        self.calls = 0

    def handle(self, request: dict) -> dict:
        self.calls += 1
        if self.calls < 3:
            return {"hata": "meşgul", "durum": "calisiyor"}
        return super().handle(request)


AUDIO = base64.b64encode(b"sinyal").decode("ascii")


class EarlySentenceTest(unittest.TestCase):
    def _channel(self, voice, service, publish):
        return VoiceChannel(voice=voice, service=service, publish=publish)

    def test_early_clips_travel_via_events_rest_in_response(self):
        seen: list[dict] = []
        channel = self._channel(
            _FakeVoice(),
            _StreamingService("Bir. İki. Üç.", early=["Bir.", "İki."]),
            publish=lambda data: seen.append(data))

        result = channel.listen(AUDIO)

        self.assertTrue(result["akis"])
        self.assertTrue(result["tur"])
        turns = {payload["tur"] for payload in seen}
        self.assertEqual(turns, {result["tur"]})
        spoken = [c["metin"] for payload in seen for c in payload["parcalar"]]
        self.assertEqual(spoken, ["Bir.", "İki."])
        remainder = [c["metin"] for c in result["parcalar"]]
        self.assertEqual(remainder, ["Üç."])

    def test_prefix_mismatch_falls_back_to_full_answer(self):
        seen: list[dict] = []
        channel = self._channel(
            _FakeVoice(),
            _StreamingService("Gerçek cevap. Bitti.", early=["Sahte"]),
            publish=lambda data: seen.append(data))

        result = channel.listen(AUDIO)

        texts = [c["metin"] for c in result["parcalar"]]
        self.assertEqual(texts, ["Gerçek cevap.", "Bitti."])
        self.assertFalse(result["akis"])   # güvenli taraf: akış sayılmaz

    def test_without_publish_everything_rides_the_response(self):
        channel = self._channel(
            _FakeVoice(), _StreamingService("Bir. İki.", early=["Bir."]),
            publish=None)

        result = channel.listen(AUDIO)

        texts = [c["metin"] for c in result["parcalar"]]
        self.assertEqual(texts, ["Bir.", "İki."])
        self.assertFalse(result["akis"])

    def test_early_synthesis_failure_does_not_break_the_turn(self):
        seen: list[dict] = []
        voice = _FakeVoice()
        voice.speaker_instance.boom_on = "İki"
        channel = self._channel(
            voice,
            _StreamingService("Bir. İki. Üç.", early=["Bir.", "İki.", "Üç."]),
            publish=lambda data: seen.append(data))

        result = channel.listen(AUDIO)

        spoken = [c["metin"] for payload in seen for c in payload["parcalar"]]
        self.assertEqual(spoken, ["Bir.", "Üç."])   # patlayan cümle atlandı
        remainder = [c["metin"] for c in result["parcalar"]]
        self.assertEqual(remainder, ["İki."])       # aradaki boşluk kaybolmaz
        self.assertTrue(result["akis"])

    def test_each_turn_gets_a_fresh_token(self):
        channel = self._channel(_FakeVoice(),
                                _StreamingService("Selam.", early=[]),
                                publish=lambda data: None)
        first = channel.listen(AUDIO)["tur"]
        second = channel.listen(AUDIO)["tur"]
        self.assertNotEqual(first, second)

    def test_browser_turn_token_follows_early_clips(self):
        seen: list[dict] = []
        channel = self._channel(
            _FakeVoice(), _StreamingService("Selam.", early=["Selam."]),
            publish=seen.append)

        result = channel.listen(AUDIO, client_turn="voice_42-okay")

        self.assertEqual(result["tur"], "voice_42-okay")
        self.assertEqual(seen[0]["tur"], "voice_42-okay")

    def test_unsafe_browser_turn_token_is_not_echoed(self):
        channel = self._channel(_FakeVoice(), _StreamingService("Selam.", early=[]),
                                publish=lambda data: None)
        result = channel.listen(AUDIO, client_turn="<script>/voice 42")
        self.assertEqual(result["tur"], "scriptvoice42")

    def test_a_releasing_cancelled_turn_is_retried_without_repeating_speech(self):
        service = _ReleasingService()
        channel = self._channel(_FakeVoice(), service, publish=lambda data: None)

        result = channel.listen(AUDIO)

        self.assertEqual(service.calls, 3)
        self.assertEqual(result["cevap"], "Yeni sözünü duydum.")
        self.assertNotIn("meşgul", result["cevap"])


if __name__ == "__main__":
    unittest.main()
