"""Speech: the parts that decide when JARVIS talks, and the parts that need a model.

Split deliberately. Everything about *conversation behaviour* -- when a short
"hı hı" belongs, where a sentence ends, when a segment is too quiet to be
speech -- is pure and runs everywhere, in milliseconds, on every commit. The
engines are exercised separately and skip when the models are not installed,
because a test that quietly passes without the thing it is testing is worse
than one that says it did not run.

The engine tests use real audio: Piper says a Turkish sentence, Whisper is asked
what it heard. That is a real signal through a real model. It is not a person
speaking into a microphone, and no test in this file claims otherwise.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import io
import sys
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.voice import VoiceSystem  # noqa: E402
from jarvis.voice.backchannel import (  # noqa: E402
    LISTENING,
    THINKING,
    Backchannel,
    looks_like_question,
)
from jarvis.voice.tts import (  # noqa: E402
    MAX_PIECE,
    prepare_for_speech,
    split_sentences,
)

VOICE = VoiceSystem()
HAS_TTS = VOICE.tts_capability().ready
HAS_STT = VOICE.stt_capability().ready


# ----------------------------------------------------------------- splitting
class TestSentenceSplitting(unittest.TestCase):
    """Where a speaker would pause -- and where they would not."""

    def test_nothing_from_nothing(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   \n  "), [])

    def test_one_sentence_stays_one(self):
        self.assertEqual(split_sentences("Tabii."), ["Tabii."])

    def test_sentences_are_separated(self):
        self.assertEqual(
            split_sentences("Dosyayı okudum. İçinde üç satır var."),
            ["Dosyayı okudum.", "İçinde üç satır var."])

    def test_a_version_number_is_not_three_sentences(self):
        """The artefact everyone recognises as a machine reading text."""
        self.assertEqual(split_sentences("Python 3.14.6 kurulu."),
                         ["Python 3.14.6 kurulu."])

    def test_an_abbreviation_does_not_end_a_sentence(self):
        self.assertEqual(len(split_sentences("Dr. Ahmet geldi.")), 1)
        self.assertEqual(len(split_sentences("Kalemler, defterler vb. şeyler.")), 1)

    def test_an_initial_does_not_end_a_sentence(self):
        self.assertEqual(len(split_sentences("M. Kemal geldi.")), 1)

    def test_questions_and_exclamations_split(self):
        self.assertEqual(len(split_sentences("Oldu mu? Sanırım oldu! Evet.")), 3)

    def test_a_paragraph_break_splits(self):
        self.assertEqual(len(split_sentences("Birinci\n\nİkinci")), 2)

    def test_an_enormous_sentence_is_cut_into_speakable_pieces(self):
        """One utterance this long cannot be interrupted cleanly."""
        text = ", ".join(f"madde {i}" for i in range(200)) + "."
        pieces = split_sentences(text)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(len(piece), MAX_PIECE + 10, piece)

    def test_no_piece_is_only_punctuation(self):
        for piece in split_sentences("Evet. . Hayır."):
            self.assertTrue(any(c.isalnum() for c in piece), piece)

    def test_nothing_is_lost(self):
        text = "Birinci cümle. İkinci cümle! Üçüncü cümle?"
        joined = " ".join(split_sentences(text))
        for word in ("Birinci", "İkinci", "Üçüncü"):
            self.assertIn(word, joined)

    def test_markdown_is_not_read_as_punctuation(self):
        spoken = prepare_for_speech(
            "## Sonuç\n- **Hazır.** [Ayrıntılar](https://example.com/test)")
        self.assertEqual(" ".join(spoken.split()), "Sonuç Hazır. Ayrıntılar")

    def test_code_blocks_are_announced_instead_of_recited(self):
        spoken = prepare_for_speech("Şunu kullan:\n```python\nprint('uzun kod')\n```")
        self.assertEqual(" ".join(spoken.split()),
                         "Şunu kullan: Kod bloğunu ekranda gösteriyorum.")

    def test_public_name_is_spoken_as_a_word(self):
        self.assertEqual(prepare_for_speech("Ben ZESTOLES'im."), "Ben Zestoles'im.")


# -------------------------------------------------------------- backchannel
class TestBackchannelWhileListening(unittest.TestCase):
    """Saying "hı hı" is listening. Saying it constantly is not."""

    def setUp(self):
        self.bc = Backchannel()

    def test_a_short_sentence_gets_nothing(self):
        """Interjecting into four words is interrupting, not listening."""
        decision = self.bc.while_listening(speech_seconds=1.5, now=100.0)
        self.assertFalse(decision.speak)

    def test_a_long_turn_gets_an_acknowledgement(self):
        decision = self.bc.while_listening(speech_seconds=8.0, now=100.0)
        self.assertTrue(decision.speak)
        self.assertIn(decision.phrase, LISTENING)

    def test_a_question_is_never_talked_over(self):
        """The rudest thing this module could do."""
        decision = self.bc.while_listening(speech_seconds=20.0, now=100.0,
                                           looks_like_question=True)
        self.assertFalse(decision.speak)
        self.assertIn("soru", decision.reason)

    def test_uncertain_speech_gets_nothing(self):
        decision = self.bc.while_listening(speech_seconds=20.0, now=100.0,
                                           confidence=0.2)
        self.assertFalse(decision.speak)

    def test_the_cooldown_holds(self):
        first = self.bc.while_listening(speech_seconds=8.0, now=100.0)
        self.assertTrue(first.speak)
        second = self.bc.while_listening(speech_seconds=12.0, now=101.0)
        self.assertFalse(second.speak, "arka arkaya iki kez konuşmamalı")

    def test_it_speaks_again_after_the_cooldown(self):
        self.bc.while_listening(speech_seconds=8.0, now=100.0)
        later = self.bc.while_listening(speech_seconds=20.0, now=130.0)
        self.assertTrue(later.speak)

    def test_it_never_repeats_itself_back_to_back(self):
        said = []
        now = 100.0
        for _ in range(6):
            decision = self.bc.while_listening(speech_seconds=30.0, now=now)
            if decision.speak:
                said.append(decision.phrase)
            now += 40.0
            self.bc.turn_finished()
        self.assertGreater(len(said), 2)
        for earlier, later in zip(said, said[1:]):
            self.assertNotEqual(earlier, later)

    def test_one_turn_has_a_ceiling(self):
        """However long they talk, four is a parody."""
        spoken = 0
        now = 100.0
        for _ in range(12):
            if self.bc.while_listening(speech_seconds=90.0, now=now).speak:
                spoken += 1
            now += 30.0
        self.assertLessEqual(spoken, self.bc.max_per_turn)

    def test_a_new_turn_resets_the_allowance(self):
        now = 100.0
        for _ in range(5):
            self.bc.while_listening(speech_seconds=90.0, now=now)
            now += 30.0
        self.bc.turn_finished()
        self.assertTrue(self.bc.while_listening(speech_seconds=90.0, now=now + 60).speak)

    def test_level_zero_switches_it_off(self):
        quiet = Backchannel(level=0)
        self.assertFalse(quiet.while_listening(speech_seconds=60.0, now=1.0).speak)

    def test_a_lower_level_raises_the_bar_rather_than_changing_the_words(self):
        shy = Backchannel(level=0.5)
        self.assertFalse(shy.while_listening(speech_seconds=6.0, now=1.0).speak)
        self.assertTrue(shy.while_listening(speech_seconds=30.0, now=1.0).speak)


class TestBackchannelWhileThinking(unittest.TestCase):
    def setUp(self):
        self.bc = Backchannel()

    def test_a_short_wait_is_not_filled(self):
        """Filling a half-second gap draws attention to a pause nobody felt."""
        self.assertFalse(self.bc.while_thinking(expected_wait_s=0.4, now=10.0).speak)

    def test_a_real_wait_is_acknowledged(self):
        decision = self.bc.while_thinking(expected_wait_s=4.0, now=10.0)
        self.assertTrue(decision.speak)
        self.assertIn(decision.phrase, THINKING)

    def test_it_does_not_say_done_before_running_something(self):
        for _ in range(4):
            decision = self.bc.while_thinking(expected_wait_s=4.0, now=100.0,
                                              used_tool=True)
            if decision.speak:
                self.assertNotEqual(decision.phrase, "tamam")

    def test_listening_and_thinking_use_different_words(self):
        """"bakıyorum" while someone is still talking answers a question they
        have not asked."""
        self.assertFalse(set(LISTENING) & set(THINKING))


class TestQuestionDetection(unittest.TestCase):
    def test_a_question_mark_is_a_question(self):
        self.assertTrue(looks_like_question("Bu ne?"))

    def test_turkish_question_particles(self):
        for text in ("Bunu yapar mısın", "Geldi mi", "Olur mu", "Biliyor musun"):
            with self.subTest(text=text):
                self.assertTrue(looks_like_question(text), text)

    def test_question_words_at_the_start(self):
        for text in ("Neden böyle oldu", "Nasıl yapacağız", "Kaç tane var"):
            with self.subTest(text=text):
                self.assertTrue(looks_like_question(text), text)

    def test_a_statement_is_not_a_question(self):
        for text in ("Bugün biraz yoruldum", "Dosyayı okudum", "Tamam olur"):
            with self.subTest(text=text):
                self.assertFalse(looks_like_question(text), text)

    def test_empty_is_not_a_question(self):
        self.assertFalse(looks_like_question(""))


# ------------------------------------------------------------- availability
class TestOptionalByConstruction(unittest.TestCase):
    """Voice must never be the reason JARVIS will not open."""

    def test_the_system_builds_without_touching_a_model(self):
        system = VoiceSystem()
        self.assertIsInstance(system.status(), dict)

    def test_it_reports_why_rather_than_only_that(self):
        for half in ("stt", "tts"):
            cap = VoiceSystem().capabilities()[half]
            if not cap.ready:
                self.assertTrue(cap.reason, half)

    def test_a_missing_model_is_a_reason_not_an_exception(self):
        class Elsewhere:
            def get(self, key, default=None):
                if key.endswith(".dir"):
                    return r"C:\yok\boyle\bir\yer"
                return default

        system = VoiceSystem(Elsewhere())
        self.assertFalse(system.available)
        self.assertIsNone(system.listener())
        self.assertIsNone(system.speaker())


# -------------------------------------------------------------- the engines
@unittest.skipUnless(HAS_TTS, "Türkçe ses modeli yok")
class TestTheVoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.speaker = VoiceSystem().speaker()

    def test_it_produces_playable_audio(self):
        speech = self.speaker.say("Merhaba.")
        self.assertIsNotNone(speech)
        with wave.open(io.BytesIO(speech.wav), "rb") as wav:
            self.assertGreater(wav.getnframes(), 0)
            self.assertEqual(wav.getnchannels(), 1)

    def test_the_audio_is_not_silence(self):
        speech = self.speaker.say("Merhaba, sesli test.")
        with wave.open(io.BytesIO(speech.wav), "rb") as wav:
            raw = wav.readframes(wav.getnframes())
        peak = max(abs(int.from_bytes(raw[i:i + 2], "little", signed=True))
                   for i in range(0, len(raw) - 1, 2))
        self.assertGreater(peak, 1000, "çıktı neredeyse sessiz")

    def test_nothing_to_say_produces_nothing(self):
        self.assertIsNone(self.speaker.say("   "))

    def test_it_is_faster_than_real_time(self):
        """Everything about the design assumes this. Measure it, do not assume."""
        self.speaker.say("Isınma.")
        speech = self.speaker.say("Bu cümle gerçek zamandan hızlı üretilmeli.")
        self.assertLess(speech.real_time_factor, 1.0,
                        f"RTF {speech.real_time_factor:.2f}")

    def test_turkish_characters_survive(self):
        speech = self.speaker.say("Çağrı, şüphe, ığdır, öğün, üzgün.")
        self.assertIsNotNone(speech)
        self.assertGreater(speech.seconds, 0.5)


@unittest.skipUnless(HAS_STT and HAS_TTS, "ses modelleri yok")
class TestHearing(unittest.TestCase):
    """Real audio through the real recogniser -- synthetic speech, not a person."""

    @classmethod
    def setUpClass(cls):
        system = VoiceSystem()
        cls.speaker = system.speaker()
        cls.listener = system.listener()
        cls.speaker.say("Isınma.")

    def spoken(self, text: str) -> bytes:
        return self.speaker.say(text).wav

    def test_it_hears_a_sentence_it_was_given(self):
        heard = self.listener.hear(self.spoken("Merhaba JARVIS."))
        self.assertTrue(heard.ok, heard.refused)
        self.assertIn("merhaba", heard.text.lower())

    def test_silence_is_refused_rather_than_invented(self):
        """Whisper will produce fluent text from nothing; that text becomes a
        request JARVIS acts on."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000 * 2)
        heard = self.listener.hear(buffer.getvalue())
        self.assertFalse(heard.ok)
        self.assertEqual(heard.text, "")

    def test_a_click_is_too_short_to_be_speech(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x10\x20" * 800)
        heard = self.listener.hear(buffer.getvalue())
        self.assertFalse(heard.ok)

    def test_broken_audio_is_refused_not_raised(self):
        heard = self.listener.hear(b"bu bir wav dosyasi degil")
        self.assertFalse(heard.ok)
        self.assertTrue(heard.refused)


if __name__ == "__main__":
    unittest.main()
