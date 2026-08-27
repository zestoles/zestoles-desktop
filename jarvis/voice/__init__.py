"""Speech: hearing and being heard, entirely on this machine.

This is the first part of JARVIS that is not standard-library-only, and that is
a deliberate exception rather than a change of mind. Nothing in the standard
library can turn sound into Turkish text or Turkish text into sound; the choice
was between a local model and sending the user's microphone to somebody else's
server, which is the one thing this project is built not to do.

So the dependencies are real, and they are contained:

    faster-whisper   speech to text  (ctranslate2 runtime -- no torch)
    piper-tts        text to speech  (onnxruntime -- no torch)

## Optional by construction

Everything here answers `available()` before it does anything. When the packages
are missing, or a model has not been downloaded, or the GPU has gone away,
`VoiceSystem` comes up unavailable and says why -- and JARVIS keeps working as
it did before, in text. A voice layer that could stop the assistant from opening
would be a worse assistant than one that cannot speak.

That is also why models load lazily. Importing this module costs nothing; the
first transcription pays for the model. Startup stays as fast as it was for
someone who never turns the microphone on.

## Where the pieces live

    stt.py          faster-whisper, one segment of audio in, text out
    tts.py          piper, text in, WAV out, split into sentences
    turn.py         when the user has finished speaking (pure)
    backchannel.py  when a short "hı hı" is warranted (pure)

The two pure modules carry the conversation behaviour and have no dependency on
either engine, so the parts that decide *when JARVIS talks* are testable without
a model, a microphone, or a GPU.

## What is not here

Voice is not authentication. Nothing in this package identifies a speaker, and
no risk gate consults it: a spoken instruction goes through exactly the same
confirmation the typed one does. Recognising a voice would be a way to guess who
is asking, and a guess is not a permission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.voice")

#: Where downloaded models live. Kept inside the project rather than in a user
#: cache so that "what does JARVIS have on disk" is answerable by looking.
MODEL_ROOT = Path(__file__).resolve().parents[2] / "data" / "ses"

PIPER_DIR = MODEL_ROOT / "piper"
WHISPER_DIR = MODEL_ROOT / "whisper"

DEFAULT_VOICE = "tr_TR-dfki-medium"
DEFAULT_MODEL = "large-v3-turbo"


@dataclass(slots=True)
class Capability:
    """Whether one half of the voice layer can run, and why not when it cannot.

    The reason matters more than the flag: "paket kurulu değil" and "model
    indirilmemiş" need different answers from the user, and an interface that
    only knows "voice is off" cannot give either.
    """

    name: str
    ready: bool = False
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ad": self.name, "hazir": self.ready, "sebep": self.reason,
                **self.detail}


class VoiceSystem:
    """Both engines, assembled lazily and never fatally.

    Construction touches no model and imports no optional package: it records
    what is on disk and answers questions. The first real call is what pays.
    """

    def __init__(self, config=None) -> None:
        self.config = config
        self._listener = None
        self._speaker = None
        self._listener_failed = ""
        self._speaker_failed = ""

    # ------------------------------------------------------------ discovery
    def capabilities(self) -> dict[str, Capability]:
        return {"stt": self.stt_capability(), "tts": self.tts_capability()}

    def stt_capability(self) -> Capability:
        cap = Capability("stt")
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            cap.reason = "faster-whisper kurulu değil"
            return cap
        model_dir = self._whisper_dir()
        if not model_dir.exists() or not any(model_dir.iterdir()):
            cap.reason = "konuşma tanıma modeli indirilmemiş"
            cap.detail["dizin"] = str(model_dir)
            return cap
        cap.ready = True
        cap.detail["model"] = self._whisper_name()
        return cap

    def tts_capability(self) -> Capability:
        if self._tts_engine() == "chatterbox":
            return self._chatterbox_capability()
        return self._piper_capability()

    def _piper_capability(self) -> Capability:
        cap = Capability("tts")
        try:
            import piper  # noqa: F401
        except ImportError:
            cap.reason = "piper-tts kurulu değil"
            return cap
        model = self._piper_model()
        if not model.is_file():
            cap.reason = "Türkçe ses modeli indirilmemiş"
            cap.detail["dosya"] = str(model)
            return cap
        cap.ready = True
        cap.detail["ses"] = model.stem
        cap.detail["motor"] = "piper"
        return cap

    def _chatterbox_capability(self) -> Capability:
        """Yan surecin uc giris kosulu: ortam, betik, referans ses.

        Model dosyalari burada sorgulanmaz -- onlar ilk kullanida indirmeyle
        gelir; eksikligi kapasite degil, ilk sentezin suresi anlatir.
        """
        cap = Capability("tts")
        python_exe, script, reference = self._chatterbox_paths()
        for label, path in (("ortam", python_exe), ("betik", script),
                            ("referans", reference)):
            if not path.is_file():
                cap.reason = f"chatterbox {label} dosyası yok"
                cap.detail["dosya"] = str(path)
                return cap
        cap.ready = True
        cap.detail["motor"] = "chatterbox"
        cap.detail["referans"] = reference.name
        return cap

    @property
    def available(self) -> bool:
        """Voice is usable when JARVIS can both hear and answer."""
        caps = self.capabilities()
        return caps["stt"].ready and caps["tts"].ready

    # -------------------------------------------------------------- engines
    def listener(self):
        """The speech recogniser, or None. Loads the model on first use."""
        if self._listener is not None or self._listener_failed:
            return self._listener
        cap = self.stt_capability()
        if not cap.ready:
            self._listener_failed = cap.reason
            return None
        try:
            from .stt import Listener

            self._listener = Listener(
                model_dir=self._whisper_dir(), model_name=self._whisper_name(),
                device=self._get("voice.stt.device", "auto"),
                language=self._get("voice.language", "tr"))
        except Exception as exc:  # noqa: BLE001 - voice must not break JARVIS
            self._listener_failed = f"{type(exc).__name__}: {exc}"
            log.warning("konuşma tanıma kurulamadı: %s", exc)
            return None
        return self._listener

    def speaker(self):
        """The voice, or None. Loads the model on first use."""
        if self._speaker is not None or self._speaker_failed:
            return self._speaker
        cap = self.tts_capability()
        if not cap.ready:
            self._speaker_failed = cap.reason
            return None
        try:
            if cap.detail.get("motor") == "chatterbox":
                from .chatterbox import ChatterboxSpeaker

                python_exe, script, reference = self._chatterbox_paths()
                self._speaker = ChatterboxSpeaker(
                    python_exe=python_exe, server_script=script,
                    reference=reference,
                    exaggeration=float(self._get("voice.tts.exaggeration", 0.5)),
                    cfg_weight=float(self._get("voice.tts.cfg_weight", 0.3)),
                    temperature=float(self._get("voice.tts.temperature", 0.35)),
                    model_version=str(self._get("voice.tts.model_version", "v3")),
                    timeout_s=float(self._get("voice.tts.timeout_s", 600)))
            else:
                from .tts import Speaker

                self._speaker = Speaker(self._piper_model())
        except Exception as exc:  # noqa: BLE001
            self._speaker_failed = f"{type(exc).__name__}: {exc}"
            log.warning("ses sentezi kurulamadı: %s", exc)
            return None
        return self._speaker

    # -------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        caps = self.capabilities()
        return {
            "kullanilabilir": caps["stt"].ready and caps["tts"].ready,
            "stt": caps["stt"].as_dict(),
            "tts": caps["tts"].as_dict(),
            "stt_hata": self._listener_failed,
            "tts_hata": self._speaker_failed,
        }

    # ------------------------------------------------------------ internals
    def _get(self, key: str, default: Any) -> Any:
        if self.config is None:
            return default
        value = self.config.get(key, default)
        return default if value in (None, "") else value

    def _whisper_name(self) -> str:
        return str(self._get("voice.stt.model", DEFAULT_MODEL))

    def _whisper_dir(self) -> Path:
        return Path(self._get("voice.stt.dir", str(WHISPER_DIR)))

    def _piper_model(self) -> Path:
        name = str(self._get("voice.tts.voice", DEFAULT_VOICE))
        return Path(self._get("voice.tts.dir", str(PIPER_DIR))) / f"{name}.onnx"

    # ------------------------------------------------- chatterbox (yan surec)
    def _tts_engine(self) -> str:
        return str(self._get("voice.tts.engine", "piper"))

    @staticmethod
    def _anchor(path: str | Path) -> Path:
        """Göreli yolu proje köküne sabitler; mutlak yol aynen kalır."""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return MODEL_ROOT.parents[1] / candidate

    def _chatterbox_paths(self) -> tuple[Path, Path, Path]:
        python_exe = self._anchor(
            self._get("voice.tts.python",
                      "data/ses/chatterbox-v3-env/Scripts/python.exe"))
        script = self._anchor(
            self._get("voice.tts.server",
                      "tools/ses/chatterbox_server.py"))
        reference = self._anchor(
            self._get("voice.tts.reference",
                      "data/ses/referans/varsayilan.wav"))
        return python_exe, script, reference


__all__ = ["VoiceSystem", "Capability", "MODEL_ROOT", "PIPER_DIR", "WHISPER_DIR",
           "DEFAULT_VOICE", "DEFAULT_MODEL"]
