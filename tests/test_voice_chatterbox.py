"""Yan surec istemcisinin saf parcalari ve motor secimi.

Bu testler model yuklemez ve surec baslatmaz: chatterbox'in agirligi ilk
kullanimdadur, burada denetlenen sey kapinin arkasindaki mantiktir -- WAV
suresi okumasi, port tutma, motor seciminin dosya kosullarina bagliligi.
"""

import io
import socket
import unittest
import wave
from pathlib import Path

from jarvis.voice import VoiceSystem
from jarvis.voice.chatterbox import ChatterboxSpeaker, free_port, wav_seconds
from tools.ses.chatterbox_server import speech_token_limit


class _StubConfig:
    """Dotted-get'i taklit eden en kucuk config; gercek Config'e gerek yok."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key: str, default=None):
        node = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _silence_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class WavSecondsTest(unittest.TestCase):
    def test_duration_matches_frames(self):
        self.assertAlmostEqual(wav_seconds(_silence_wav(0.5)), 0.5, places=2)

    def test_garbage_is_zero_not_crash(self):
        self.assertEqual(wav_seconds(b"bu bir wav degil"), 0.0)


class FreePortTest(unittest.TestCase):
    def test_returned_port_is_bindable(self):
        port = free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))  # birakilan port yeniden tutulabilmeli


class SpeechTokenLimitTest(unittest.TestCase):
    def test_short_reply_cannot_wander_to_the_upstream_thousand_token_cap(self):
        self.assertLess(speech_token_limit("Tabii, hemen bakıyorum."), 160)

    def test_long_reply_keeps_a_generous_budget(self):
        self.assertGreater(speech_token_limit("uzun metin " * 30), 900)

    def test_limit_never_exceeds_the_model_ceiling(self):
        self.assertEqual(speech_token_limit("x" * 2000), 1000)


class ChatterboxSpeakerConstructionTest(unittest.TestCase):
    def test_construction_spawns_nothing(self):
        speaker = ChatterboxSpeaker(
            python_exe="yok/python.exe", server_script="yok/sunucu.py",
            reference="yok/ref.wav")
        self.assertIsNone(speaker._proc)  # tembel: surec ilk say()'da
        # Tek harfli sonlar bas harf sayilir ve birlestirilir (bolenin bilincli
        # davranisi); gercek cumle sonlari bolunur.
        self.assertEqual(speaker.pieces("A. B."), ["A. B."])
        self.assertEqual(speaker.pieces("Merhaba! Nasilsin?"),
                         ["Merhaba!", "Nasilsin?"])
        command = speaker.command(8123)
        self.assertIn("--model-version", command)
        self.assertEqual(command[command.index("--model-version") + 1], "v3")
        self.assertIn("--cfg-weight", command)


class EngineSelectionTest(unittest.TestCase):
    def _system(self, tts: dict) -> VoiceSystem:
        return VoiceSystem(_StubConfig({"voice": {"tts": tts}}))

    def test_engine_defaults_to_piper(self):
        self.assertEqual(VoiceSystem(None)._tts_engine(), "piper")

    def test_chatterbox_without_files_reports_which_one(self):
        system = self._system({"engine": "chatterbox",
                               "python": "yok/python.exe",
                               "server": "yok/sunucu.py",
                               "reference": "yok/ref.wav"})
        cap = system.tts_capability()
        self.assertFalse(cap.ready)
        self.assertIn("chatterbox", cap.reason)

    def test_chatterbox_with_dummy_files_is_ready(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("python.exe", "sunucu.py", "ref.wav"):
                (root / name).write_bytes(b"x")
            system = self._system({
                "engine": "chatterbox",
                "python": str(root / "python.exe"),
                "server": str(root / "sunucu.py"),
                "reference": str(root / "ref.wav")})
            cap = system.tts_capability()
            self.assertTrue(cap.ready)
            self.assertEqual(cap.detail.get("motor"), "chatterbox")

    def test_missing_environment_blocks_speaker_honestly(self):
        system = self._system({"engine": "chatterbox",
                               "python": "yok/python.exe",
                               "server": "yok/sunucu.py",
                               "reference": "yok/ref.wav"})
        self.assertIsNone(system.speaker())
        self.assertTrue(system.status()["tts_hata"])

    def test_anchor_makes_relative_paths_absolute(self):
        anchored = VoiceSystem._anchor("bir/sey.txt")
        self.assertTrue(anchored.is_absolute())


if __name__ == "__main__":
    unittest.main()
