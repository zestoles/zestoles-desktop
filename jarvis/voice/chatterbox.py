"""Chatterbox sesi icin yan surec istemcisi.

`tools/ses/chatterbox_server.py`'i kendi ortaminda baslatir, hazir olmasini
bekler ve `Speaker` ile ayni yuzeyi sunar: `say()` bir parca metni alip WAV
doner, `pieces()` cevabi insan duraklarina boler. Kanal katmani hangi motorun
arkasinda oldugunu bilmek zorunda degil.

Neden yan surec: chatterbox'in bagimlilik zinciri (torch, transformers 5.x)
ana projenin yorumlayicisini kirletmemeli; ayri ortam ayni zamanda VRAM'i tek
bir surecte toplar -- model bosken kapatilirsa kart tamamen serbest kalir.

Dusuklukler dogru yone: surec olursa `say()` RuntimeError firlatir, VoiceSystem
bunu durust bir sebep olarak kaydeder; sessizce Piper'a donulmaz cunku kullanici
hangi motoru istedigini soylemistir, onu susturmak onu kandirmaktir.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import queue
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
import io
from pathlib import Path

from .tts import Speech, prepare_for_speech, split_sentences

log = logging.getLogger("jarvis.voice.chatterbox")

#: Windows'ta arka pencere acilmaz.
_CREATE_NO_WINDOW = 0x08000000

#: Ilk baslama suresi siniri. Model zaten diskteyse ~45 sn; ilk indirmede
#: gigabaytlarca inebilir, o yuzden cömektir.
DEFAULT_START_TIMEOUT_S = 600.0


def wav_seconds(data: bytes) -> float:
    """WAV baytlarinin saniye uzunlugu; okunamazsa 0.0."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except (wave.Error, EOFError):
        return 0.0


def free_port() -> int:
    """Kisa sureligine bir port tutup birakir. Yaris kosu loopback'te kabul."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ChatterboxSpeaker:
    """Yan sureci yoneten ses. Model ilk kullanimda yuklenir."""

    def __init__(self, *, python_exe: str | Path, server_script: str | Path,
                 reference: str | Path, exaggeration: float = 0.5,
                 cfg_weight: float = 0.3,
                 temperature: float = 0.35,
                 model_version: str = "v3",
                 timeout_s: float = DEFAULT_START_TIMEOUT_S) -> None:
        self.python_exe = Path(python_exe)
        self.server_script = Path(server_script)
        self.reference = Path(reference)
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.model_version = model_version
        self.timeout_s = timeout_s
        self.load_seconds = 0.0
        self.sample_rate = 24000
        self._proc: subprocess.Popen | None = None
        self._port = 0
        self._lock = threading.Lock()
        self._closed = False
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        atexit.register(self.shutdown)

    # ---------------------------------------------------------------- surface
    def pieces(self, text: str) -> list[str]:
        """Cevabi insan duraklarina boler -- Piper ile ayni bolen."""
        return split_sentences(text)

    def say(self, text: str) -> Speech | None:
        """Bir parcayi sentezle. Metin bos ise None."""
        body = prepare_for_speech(text)
        if not body:
            return None
        with self._lock:
            if self._proc is None and not self._closed:
                self._start()
            self._wait_healthy()
            payload = json.dumps({
                "metin": body,
                "abartma": self.exaggeration,
                "akiskanlik": self.cfg_weight,
                "sicaklik": self.temperature,
            }).encode("utf-8")
            started = time.perf_counter()
            answer = self._request(payload)
            generated = time.perf_counter() - started
            data = base64.b64decode(answer["ses"])
        self.sample_rate = int(answer.get("hz") or self.sample_rate)
        return Speech(wav=data, text=body,
                      seconds=wav_seconds(data) or float(answer.get("saniye") or 0),
                      generated_s=generated, sample_rate=self.sample_rate)

    def status(self) -> dict:
        return {"hazir": self._proc is not None and self._proc.poll() is None}

    def shutdown(self) -> None:
        """Sureci kibarca bitir. Kapanista ve nesne toplanirken cagrilir."""
        proc, self._proc = self._proc, None
        self._closed = True
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except OSError as exc:
                log.debug("yan surec kapatilamadi: %s", exc)

    # ------------------------------------------------------------- internals
    def _start(self) -> None:
        """Yan sureci baslat ve dinlemeye girmesini dogrula.

        Model yuklenmesini BEKLEMEZ: sunucu soketi modelden once acar, boylece
        bu cagri saniyeler icinde doner; agir yukleme ilk sentezde saglik
        yoklamasiyla beklenir (`_wait_healthy`).
        """
        if not self.python_exe.is_file():
            raise RuntimeError(f"ses ortami yok: {self.python_exe}")
        if not self.server_script.is_file():
            raise RuntimeError(f"sunucu betigi yok: {self.server_script}")
        if not self.reference.is_file():
            raise RuntimeError(f"referans ses yok: {self.reference}")

        self._port = free_port()
        command = self.command(self._port)
        hf_dir = self.server_script.parents[2] / "data" / "ses" / "hf"
        env = {**os.environ}
        if hf_dir.is_dir():
            env.setdefault("HF_HOME", str(hf_dir))
        log.info("chatterbox yan sureci basliyor: port %d", self._port)
        self._proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW, env=env, text=True,
            encoding="ascii", errors="replace")

        # Tek okuyucu thread: stdout'un tek tuketicisi odur. Her satir kuyruga
        # girer; ana thread zamani kendisi kontrol eder, takilan surec butu bir
        # baslatmayi kilitlemez ve satir kaybolmaz.
        def pump() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for raw in self._proc.stdout:
                self._lines.put(raw.rstrip("\n"))
            self._lines.put(None)

        threading.Thread(target=pump, name="chatterbox-stdout",
                         daemon=True).start()

        deadline = time.perf_counter() + 30.0
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.shutdown()
                raise RuntimeError("chatterbox portu acilmadi")
            line = self._readline(timeout=min(remaining, 5.0))
            if line is None:
                code = self._proc.poll()
                if code is not None:
                    self.shutdown()
                    raise RuntimeError(f"chatterbox erken sonlandi ({code})")
                continue
            if line.startswith("PORT"):
                break
        log.info("chatterbox yan sureci dinliyor: port %d", self._port)

    def command(self, port: int) -> list[str]:
        """Sidecar command, exposed as data so version/config wiring is tested."""
        return [str(self.python_exe), str(self.server_script),
                "--port", str(port), "--ref", str(self.reference),
                "--exaggeration", str(self.exaggeration),
                "--cfg-weight", str(self.cfg_weight),
                "--temperature", str(self.temperature),
                "--model-version", self.model_version]

    def _wait_healthy(self, *, timeout: float | None = None) -> None:
        """Model yuklenene ve isinma bitene dek bekle.

        Ilk sentez burada agirlik tasir: diskteki model ~45 sn'de hazir olur,
        ilk indirmede bu dakikalar surebilir. Sunucu bu surede /saglik'a
        {"hazir": false} doner; baglanti reddedilmez.
        """
        deadline = time.perf_counter() + (timeout or self.timeout_s)
        url = f"http://127.0.0.1:{self._port}/saglik"
        while True:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if json.loads(response.read().decode("utf-8")).get("hazir"):
                        return
            except (urllib.error.URLError, ConnectionError, OSError,
                    json.JSONDecodeError):
                pass  # soket ya da model daha hazir degil
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"chatterbox yukleme sirasinda sonlandi ({self._proc.poll()})")
            if time.perf_counter() >= deadline:
                raise RuntimeError("chatterbox zaman asimiyla hazir olmadi")
            time.sleep(1.0)

    def _readline(self, timeout: float) -> str | None:
        """Kuyruktaki bir sonraki satir; surede gelmezse None.

        Satirlari tek bir pump thread'i kuyruga yazar (bkz. `_start`); burada
        yalnizca zaman kontrollu beklenir.
        """
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def _request(self, payload: bytes, *, retry: bool = True) -> dict:
        url = f"http://127.0.0.1:{self._port}/konus"
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                reason = json.loads(detail).get("hata") or detail[:200]
            except json.JSONDecodeError:
                reason = detail[:200]
            raise RuntimeError(f"chatterbox reddetti: {reason}") from exc
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            if retry and self._proc is not None:
                log.warning("chatterbox baglantisi koptu (%s), yeniden basliyor",
                            exc)
                self.shutdown()
                self._closed = False
                self._start()
                return self._request(payload, retry=False)
            raise RuntimeError(f"chatterbox'a ulasilamadi: {exc}") from exc

    def __del__(self):  # son temizlik; shutdown zaten guard'li
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["ChatterboxSpeaker", "wav_seconds", "free_port"]
