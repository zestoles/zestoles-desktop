"""The desktop interface: the page, and the wiring that serves it.

What these defend is mostly negative — that the interface does not show what is
not there. §47 of the V1 spec is blunt about it: no placeholder buttons, no fake
progress, no control for a feature that does not exist. A microphone button that
does nothing is worse than no microphone button, because it is a promise.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import inspect
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.bus.server import TOKEN_MARKER  # noqa: E402
from jarvis.bus.types import (  # noqa: E402
    ASSISTANT_TURN_FINISHED,
    ASSISTANT_TURN_STARTED,
    TOOL_FINISHED,
    TOOL_STARTED,
    translate,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "ui" / "jarvis.html"


def _keyframes(css: str) -> list[tuple[str, str]]:
    """Every @keyframes body, found by counting braces rather than guessing."""
    out = []
    for match in re.finditer(r"@keyframes\s+([\w-]+)\s*\{", css):
        depth, i = 1, match.end()
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        out.append((match.group(1), css[match.end():i - 1]))
    return out


class TestThePageExists(unittest.TestCase):
    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")

    def test_the_file_is_there(self):
        self.assertTrue(PAGE.is_file())

    def test_it_is_a_single_file_with_no_external_fetches(self):
        """Loopback, offline, no build step. A CDN reference would break the
        interface on the machine this is written for the moment it is offline."""
        for forbidden in ("http://cdn", "https://cdn", "unpkg.com", "jsdelivr",
                          "googleapis.com", "<script src=", "<link rel=\"stylesheet\""):
            self.assertNotIn(forbidden, self.text, forbidden)

    def test_it_carries_the_token_marker(self):
        self.assertIn(TOKEN_MARKER.decode("ascii"), self.text)

    def test_it_is_not_the_developer_dashboard(self):
        other = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertNotEqual(self.text, other)


class TestSpeaking(unittest.TestCase):
    """JARVIS speaks with a local model now, not the browser's voice.

    `speechSynthesis` was replaced because it could not provide the same natural
    Turkish voice on every machine. Audio now arrives from the server as a queue
    of clips, which can still be stopped by the explicit stop control.

    These pin what that has to keep doing, and the absence it must maintain: a
    microphone is not a licence to send audio anywhere.
    """

    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")

    def test_the_page_plays_audio_the_server_made(self):
        self.assertIn('op: "seslendir"', self.text)
        self.assertIn("decodeAudioData", self.text)

    def test_speech_arrives_as_a_queue_of_pieces(self):
        """One long clip cannot be stopped cleanly; a queue can."""
        self.assertIn("playback.queue", self.text)
        self.assertIn("playNext", self.text)

    def test_stopping_empties_the_queue_and_the_current_clip(self):
        stop = self.text.split("function stopSpeaking", 1)[1][:400]
        self.assertIn("queue.length = 0", stop)
        self.assertIn("stop()", stop)

    def test_speech_during_a_response_is_discarded(self):
        """Half duplex is deliberate: room echo and a second user utterance
        must never overlap the answer already being produced."""
        audio = self.text.split("function onAudio", 1)[1][:700]
        self.assertIn("playback.playing || busy", audio)
        self.assertIn("capturing = false", audio)
        self.assertIn("frames = []", audio)
        self.assertIn("return", audio)
        self.assertNotIn("stopSpeaking()", audio)

    def test_barge_in_has_no_hidden_threshold_or_cancel_request(self):
        audio = self.text.split("function onAudio", 1)[1][:2200]
        self.assertNotIn("bargeMs", self.text)
        self.assertNotIn('post({ op: "iptal" })', audio)

    def test_late_audio_from_an_interrupted_turn_is_rejected(self):
        enqueue = self.text.split("function enqueue", 1)[1][:900]
        self.assertIn("activeVoiceTurn", enqueue)
        self.assertIn("turn !== activeVoiceTurn", enqueue)

    def test_visible_entity_uses_real_output_audio_energy(self):
        play = self.text.split("function playNext", 1)[1][:1300]
        wave = self.text.split("function drawWave", 1)[1][:1500]
        self.assertIn("outputAnalyser", play)
        self.assertIn("getByteTimeDomainData", wave)
        self.assertNotIn("Math.sin(performance.now()", wave)
        self.assertIn("entityReactor.style.transform", self.text)


class TestInteractionModes(unittest.TestCase):
    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")

    def test_text_composer_is_an_explicit_mode(self):
        self.assertIn("body.text-session .composer", self.text)
        self.assertIn('setInputMode("text")', self.text)
        self.assertIn('classList.toggle("text-session"', self.text)

    def test_voice_mode_centres_the_entity_and_removes_dashboard_noise(self):
        self.assertIn("body.voice-session .nav-rail", self.text)
        self.assertIn("body.voice-session .telemetry", self.text)
        self.assertIn("body.voice-session #thread", self.text)
        self.assertIn('setInputMode("voice")', self.text)

    def test_entity_exposes_real_operating_states(self):
        self.assertIn('id="entity-reactor"', self.text)
        self.assertIn('id="entity-state"', self.text)
        for state in ("dinliyor", "dusunuyor", "konusuyor", "onay", "hata"):
            self.assertRegex(self.text, rf'body\[data-mode="{state}"\]')

    def test_stopping_the_turn_stops_the_voice(self):
        stop = self.text.split("stopBtn.onclick", 1)[1][:400]
        self.assertIn("stopSpeaking", stop, stop)

    def test_closing_stops_the_voice(self):
        closing = self.text.split("function closeJarvis", 1)[1][:400]
        self.assertIn("stopSpeaking", closing)

    def test_speaking_is_a_visible_state(self):
        self.assertIn("konusuyor", self.text)
        self.assertIn('setMode("konusuyor")', self.text)

    def test_the_speaking_state_is_cleared_when_the_voice_stops(self):
        stop = self.text.split("function stopSpeaking", 1)[1][:500]
        self.assertIn("konusuyor", stop, stop)

    def test_only_the_reply_is_spoken(self):
        """Tool cards and notes are for reading. A page that read the event
        stream aloud would announce failures before the turn that owns them has
        decided what they were."""
        spoken = [line for line in self.text.splitlines()
                  if "speakAnswer(" in line and "function speakAnswer" not in line]
        self.assertTrue(spoken, "hic seslendirme cagrisi yok")
        for line in spoken:
            self.assertNotIn("toolCard", line, line)
            self.assertNotIn("note(", line, line)

    def test_the_choice_is_remembered(self):
        self.assertIn("localStorage", self.text)

    def test_it_is_still_not_a_cloud_microphone(self):
        for forbidden in ("webkitSpeechRecognition", "SpeechRecognition("):
            self.assertNotIn(forbidden.lower(), self.text.lower(), forbidden)


class TestTheScriptActuallyRuns(unittest.TestCase):
    """Every other test here reads the page as text, which cannot see a syntax
    error or a loop that never ends. Node is not a dependency of JARVIS and is
    not required to run it -- so where it exists these run, and where it does
    not they skip rather than pretending the check happened.
    """

    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        cls.script = re.search(r"<script>(.*)</script>",
                               PAGE.read_text(encoding="utf-8"), re.S).group(1)

    def run_node(self, source, timeout=30):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parca.js"
            path.write_text(source, encoding="utf-8")
            return subprocess.run([self.node, str(path)], capture_output=True,
                                  text=True, timeout=timeout)

    @unittest.skipUnless(shutil.which("node"), "node yok")
    def test_the_page_script_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sayfa.js"
            path.write_text(self.script, encoding="utf-8")
            done = subprocess.run([self.node, "--check", str(path)],
                                  capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-1200:])

    @unittest.skipUnless(shutil.which("node"), "node yok")
    def test_the_recording_encoder_produces_a_real_wav(self):
        """`encodeWav` is the piece of page code that would silently corrupt
        every recording if it were wrong: the server would receive plausible
        bytes and the recogniser would hear noise. Run it and read the header."""
        start = self.script.index("function encodeWav(")
        end = self.script.index("// -------------------------------------------------------------- waveform")
        probe = self.script[start:end] + r"""
function frames(seconds, rate, freq) {
  const total = Math.floor(seconds * rate);
  const out = new Float32Array(total);
  for (let i = 0; i < total; i++) out[i] = Math.sin(2 * Math.PI * freq * i / rate) * 0.5;
  return [out];
}
global.btoa = (s) => Buffer.from(s, "binary").toString("base64");
const made = encodeWav(frames(1.0, 48000, 440), 48000, 16000);
const raw = Buffer.from(made.base64, "base64");
console.log(JSON.stringify({
  riff: raw.toString("ascii", 0, 4),
  wave: raw.toString("ascii", 8, 12),
  channels: raw.readUInt16LE(22),
  rate: raw.readUInt32LE(24),
  bits: raw.readUInt16LE(34),
  dataBytes: raw.readUInt32LE(40),
  totalBytes: raw.length,
  seconds: made.seconds,
  empty: encodeWav([], 48000, 16000)
}));
"""
        done = self.run_node(probe)
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        got = json.loads(done.stdout.strip())
        self.assertEqual(got["riff"], "RIFF")
        self.assertEqual(got["wave"], "WAVE")
        self.assertEqual(got["channels"], 1)
        self.assertEqual(got["rate"], 16000, "Whisper 16 kHz bekliyor")
        self.assertEqual(got["bits"], 16)
        self.assertEqual(got["dataBytes"], got["totalBytes"] - 44)
        self.assertAlmostEqual(got["seconds"], 1.0, places=2)
        self.assertIsNone(got["empty"], "bos kayit null donmeli")

    @unittest.skipUnless(shutil.which("node"), "node yok")
    def test_the_level_meter_is_a_real_rms(self):
        """Endpointing is decided from this number; a wrong one either cuts
        people off or never stops recording."""
        start = self.script.index("function rms(")
        end = self.script.index("function onAudio(")
        probe = self.script[start:end] + r"""
const silence = new Float32Array(1024);
const loud = new Float32Array(1024).fill(0.5);
const half = new Float32Array(1024);
for (let i = 0; i < half.length; i++) half[i] = i % 2 ? 0.25 : -0.25;
console.log(JSON.stringify([rms(silence), rms(loud), rms(half)]));
"""
        done = self.run_node(probe)
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        quiet, loud, half = json.loads(done.stdout.strip())
        self.assertEqual(quiet, 0.0)
        self.assertAlmostEqual(loud, 0.5, places=3)
        self.assertAlmostEqual(half, 0.25, places=3)


class TestNothingIsPromisedThatDoesNotExist(unittest.TestCase):
    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")

    def test_no_placeholders_are_left_for_the_user_to_find(self):
        # The input hint attribute and its CSS pseudo-element are both spelled
        # "placeholder" and both belong here. What must not appear is unfinished
        # work presented to the user as though it were a feature.
        visible = self.text.lower()
        for legitimate in ('placeholder="', "::placeholder"):
            visible = visible.replace(legitimate, "")
        for forbidden in ("todo", "fixme", "coming soon", "yakında",
                          "lorem ipsum", "placeholder", "geçici", "henüz yok"):
            self.assertNotIn(forbidden, visible, forbidden)

    def test_the_microphone_is_local(self):
        """There is a microphone now. What must never appear is a *cloud*
        recogniser: `webkitSpeechRecognition` uploads audio to Google, which is
        the one thing a local-first assistant cannot do with a microphone."""
        for forbidden in ("webkitspeechrecognition", "speechrecognition(",
                          "new speechrecognition", "azure", "deepgram",
                          "assemblyai", "openai.com"):
            self.assertNotIn(forbidden, self.text.lower(), forbidden)

    def test_the_microphone_button_is_hidden_until_the_backend_can_hear(self):
        """A microphone that cannot reach a recogniser is a dead control."""
        self.assertRegex(self.text, r"micBtn\.hidden\s*=\s*!voiceReady")

    def test_audio_only_goes_to_this_machine(self):
        for forbidden in ('fetch("http', "fetch('http", "xmlhttprequest"):
            self.assertNotIn(forbidden, self.text.lower(), forbidden)

    def test_the_settings_screen_settles_something(self):
        """The older form of this test was "there is no settings screen", on the
        grounds that an empty one is a promise. There is one now, so the promise
        has to be kept instead: it reads real settings and writes them back."""
        self.assertIn('op: "ayarlar"', self.text)
        self.assertIn('op: "ayar_kaydet"', self.text)

    def test_the_page_does_not_invent_settings_of_its_own(self):
        """Rows are rendered from what the server sends. A control the page made
        up would be one with no bounds, no validation and no test behind it --
        and the allow-list in jarvis/settings.py would not know about it."""
        rendering = self.text.split("function renderAyarlar", 1)[1][:1400]
        self.assertIn("data.ayarlar", rendering)
        for invented in ("orphan_grace", "history_turns", "max_steps"):
            self.assertNotIn(invented, self.text, invented)

    def test_no_dangerous_switch_is_reachable_from_the_page(self):
        for forbidden in ("self_modification", "sandbox", "promotion",
                          "permissions", "confirmed: true", "confirmed:true"):
            self.assertNotIn(forbidden, self.text.lower(), forbidden)

    def test_every_button_is_wired(self):
        """Each id used as a control has a handler somewhere in the script."""
        ids = set(re.findall(r'<button[^>]*id="([^"]+)"', self.text))
        self.assertTrue(ids, "hic buton bulunamadi")
        for name in ids:
            with self.subTest(button=name):
                self.assertRegex(
                    self.text, rf'{re.escape(name)}\w*\.onclick|{re.escape(name)}\w*\.addEventListener',
                    f"{name} butonunun isleyicisi yok")


class TestProgressIsNotInvented(unittest.TestCase):
    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")

    def test_tool_cards_come_from_the_event_stream(self):
        """`toolCard` may only be reached from the socket handler. A card built
        from the reply would be the interface asserting something it did not
        witness."""
        creation = [line for line in self.text.splitlines() if "toolCard(" in line
                    and "function toolCard" not in line]
        self.assertTrue(creation)
        for line in creation:
            self.assertIn("live =", line, line)

    def test_the_reported_failure_count_comes_from_the_backend(self):
        self.assertIn("result.basarisiz", self.text)

    def test_there_is_no_percentage_bar_to_make_up(self):
        for forbidden in ("progress", "%\"", "width: \" +"):
            self.assertNotIn(forbidden, self.text, forbidden)


class TestItDoesNotBurnTheMachine(unittest.TestCase):
    """§39: the interface must not keep the CPU busy to look alive."""

    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")
        # Comments explain why these are absent, so a literal search would find
        # the very words it is checking for. Strip them first.
        self.code = re.sub(r"/\*.*?\*/", "", self.text, flags=re.S)
        self.code = re.sub(r"^\s*//.*$", "", self.code, flags=re.M)

    def test_the_frame_loop_stops_when_nothing_is_happening(self):
        """A waveform needs animation frames; an idle page must not spend them.
        The old rule banned requestAnimationFrame outright, which would have
        banned the waveform rather than the waste."""
        loop = self.text.split("function startWave", 1)[1][:600]
        self.assertIn("requestAnimationFrame", loop)
        self.assertIn("if (!recording && !playback.playing)", loop)
        self.assertIn("waveTimer = null", loop)

    def test_no_polling_interval(self):
        self.assertNotIn("setInterval", self.code)

    def test_the_only_timers_are_reconnect_and_a_state_reset(self):
        timers = re.findall(r"setTimeout\(", self.text)
        self.assertLessEqual(len(timers), 3, "beklenenden fazla zamanlayici")

    def test_animation_moves_only_transform_and_opacity(self):
        """Anything else forces layout or paint on every frame.

        Braces are counted rather than pattern-matched: a keyframes block holds
        nested rules, so its first closing brace is not its end.
        """
        blocks = _keyframes(self.code)
        self.assertTrue(blocks, "hic @keyframes bulunamadi")
        for name, block in blocks:
            properties = set(re.findall(r"([a-z-]+)\s*:", block))
            with self.subTest(keyframes=name):
                self.assertLessEqual(properties, {"transform", "opacity"},
                                     f"{name}: {sorted(properties)}")


class TestTheStatesAreReal(unittest.TestCase):
    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")

    def test_the_states_the_backend_can_report_are_all_handled(self):
        for state in ("hazir", "dusunuyor", "calisiyor", "onay", "hata",
                      "cevrimdisi", "baslatiliyor"):
            with self.subTest(state=state):
                self.assertIn(state, self.text)

    def test_confirmation_asks_before_it_acts(self):
        self.assertIn('"op": "onay"', self.text.replace("op: ", '"op": '))

    def test_the_page_can_cancel(self):
        self.assertIn("iptal", self.text)

    def test_requests_carry_the_token(self):
        self.assertIn("X-Jarvis-Token", self.text)


class TestTheAssistantEventsReachTheWire(unittest.TestCase):
    """A page cannot show activity the bus refuses to carry. Unmapped pairs are
    dropped by design, so the mapping is the thing to pin."""

    def test_tool_events_are_translated(self):
        self.assertEqual(translate("assistant", "tool.start"), TOOL_STARTED)
        self.assertEqual(translate("assistant", "tool.done"), TOOL_FINISHED)

    def test_turn_events_are_translated(self):
        self.assertEqual(translate("assistant", "turn.start"), ASSISTANT_TURN_STARTED)
        self.assertEqual(translate("assistant", "turn.done"), ASSISTANT_TURN_FINISHED)

    def test_a_failed_tool_is_carried_as_an_error(self):
        self.assertEqual(translate("assistant", "tool.failed"), "error")

    def test_a_confirmation_is_carried_as_waiting_for_the_user(self):
        self.assertEqual(translate("assistant", "tool.confirm"), "waiting_for_user")


class TestTheLauncherOpensTheInterface(unittest.TestCase):
    def test_the_flag_exists(self):
        from jarvis.cli import app

        self.assertIn("--arayuz", inspect.getsource(app._parser))

    def test_the_desktop_launcher_uses_it(self):
        text = (ROOT / "tools" / "launcher" / "jarvis.ps1").read_text(encoding="ascii")
        # Varsayılan dal --arayuz'dur; -Surekli ayrı bir anahtardır.
        self.assertIn("if ($Surekli)", text)
        self.assertIn("'--arayuz'", text)

    def test_a_missing_page_is_reported_rather_than_crashing(self):
        from jarvis.cli.interface import run_interface

        class Config:
            def path(self, dotted, default=""):
                return Path("bu-dosya-yok-12345.html")

            def get(self, dotted, default=None):
                return default

        self.assertEqual(run_interface(object(), Config()), 1)


class TestShutdownGivesEverythingBack(unittest.TestCase):
    def test_the_exit_path_releases_the_lock_last(self):
        """The lock is what stops a second copy, so it is the one release that
        must not be skipped by an earlier failure."""
        from jarvis.cli import interface

        source = inspect.getsource(interface._shut_down)
        self.assertLess(source.index("server.stop"), source.index("lock.release"))
        self.assertIn("finally", inspect.getsource(interface.run_interface))

    def test_each_shutdown_step_is_guarded_on_its_own(self):
        from jarvis.cli import interface

        source = inspect.getsource(interface._shut_down)
        self.assertGreaterEqual(source.count("except Exception"), 2)


if __name__ == "__main__":
    unittest.main()
