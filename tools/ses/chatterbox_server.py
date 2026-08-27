r"""Chatterbox TTS sidecar: localhost HTTP server, metin alir, WAV dondurur.

Bu betik JARVIS'in kendi yorumlayicisinda DEGIL, ses ortaminda calisir:

    data\ses\chatterbox-env\Scripts\python.exe tools\ses\chatterbox_server.py ...

Cunku chatterbox'in bagimlilik zinciri (torch, transformers 5.x) ana projenin
ortasina tasinmaz; ayri bir cevre hem bagimliligi izole eder hem VRAM'i tek bir
surecte toplar. Ana taraf bu sureci bastan acar, hazir olmasini bekler ve
kapanista oldurur.

Akis sirasi bilerek boyledir:
    1. port baglanir ve servis arka planda dinlemeye baslar -- boylece ana tar
       /saglik'a vurup "yukleniyor mu?" diye sorabilir, baglanti reddedilmez
    2. model yuklenir
    3. kisa bir isinma sentezi yapilir -- CUDA derleme maliyeti ilk kullanici
       cumlesine degil, acilisa odetilmis olur
    4. HAZIR basilir

Stdout protokolu (ASCII, satir satiri):
    PORT <n>   dinlemeye baslandi
    HAZIR      model yuklu, isinma tamam

Konsol dersleri burada da gecerli: cp1254 konsola ASCII disi glif basmak
UnicodeEncodeError firlatir, o yuzden her cikti ASCII'dir.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Model yuklenirken /saglik'un donecegi durum. Handler ile main ayni sozlugu
#: gorur; bayrak main thread'de ters cevrilir.
STATE = {"hazir": False}


def speech_token_limit(text: str) -> int:
    """Bound stochastic run-away audio without clipping ordinary speech.

    Chatterbox emits roughly 25 acoustic tokens per second.  The upstream V3
    API hard-codes 1000 tokens for every utterance, so a three-word reply can
    occasionally wander for many seconds before EOS.  Turkish prose normally
    needs far less than three tokens per character; the fixed margin keeps
    pauses and slow delivery intact while still stopping pathological tails.
    """
    return max(96, min(1000, int(len(" ".join(text.split())) * 3.0 + 48)))


def wav_bytes(samples, rate: int) -> bytes:
    """float samples -> 16-bit mono WAV."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for s in samples:
            q = int(max(-1.0, min(1.0, float(s))) * 32767)
            frames += q.to_bytes(2, "little", signed=True)
        w.writeframes(bytes(frames))
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # erisim kaydi gurultu; sessiz
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200, {"hazir": STATE["hazir"]})

    def do_POST(self):
        if not STATE["hazir"]:
            self._json(503, {"hata": "model yukleniyor"})
            return
        if self.path != "/konus":
            self._json(404, {"hata": "bilinmeyen yol"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"hata": "istek okunamadi"})
            return
        text = str(req.get("metin") or "").strip()
        if not text:
            self._json(400, {"hata": "bos metin"})
            return
        try:
            data, dur, gen = HANDLER_SYNTH(
                text,
                float(req.get("abartma", ARGS.exaggeration)),
                float(req.get("akiskanlik", ARGS.cfg_weight)),
                float(req.get("sicaklik", ARGS.temperature)),
            )
        except Exception as exc:  # noqa: BLE001 - hata istemciye soylenebilir olmali
            self._json(500, {"hata": f"{type(exc).__name__}: {exc}"})
            return
        self._json(200, {
            "ses": base64.b64encode(data).decode("ascii"),
            "saniye": round(dur, 3),
            "uretim": round(gen, 3),
            "hz": HANDLER_RATE,
        })


#: Main thread model kurulumunda doldurur; handler sadece okur.
HANDLER_SYNTH = None
HANDLER_RATE = 24000
ARGS = argparse.Namespace(port=0, ref="", exaggeration=0.5, cfg_weight=0.5,
                          hf_dir="")


def main() -> int:
    global HANDLER_SYNTH, HANDLER_RATE, ARGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    #: Yüksek sıcaklık cümleden cümleye tını kaydırır -- tek bir cevabın
    #: parçaları farklı insanlar gibi duyulur. Tutarlılık buradan başlar.
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--model-version", default="v3")
    parser.add_argument("--hf-dir", default="")
    args = parser.parse_args()
    ARGS = args

    if args.hf_dir:
        import os
        os.environ.setdefault("HF_HOME", args.hf_dir)

    # 1. Once dinle: baglanti reddi yerine "yukleniyor" cevabi verebilmek icin
    # soket, modelden once yasar.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.25},
                     name="chatterbox-http", daemon=True).start()
    print(f"PORT {args.port}", flush=True)

    # 2-3. Model ve isinma. tqdm/torch gurultusu stderr'e gider; ana taraf onu
    # DEVNULL yapar, buradaki stdout satirlari temiz kalir.
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    t0 = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(
        device="cuda", t3_model=args.model_version)
    model.prepare_conditionals(args.ref)
    rate = int(model.sr)
    lock = threading.Lock()

    # V3's public generate() currently hard-codes max_new_tokens=1000.  Wrap
    # the bound method locally instead of editing the pinned dependency in its
    # virtual environment, so reinstalling is deterministic.
    original_inference = model.t3.inference
    token_budget = {"value": 1000}

    def bounded_inference(*positional, **keywords):
        requested = int(keywords.get("max_new_tokens", 1000))
        keywords["max_new_tokens"] = min(requested, token_budget["value"])
        return original_inference(*positional, **keywords)

    model.t3.inference = bounded_inference

    def synth(text: str, exaggeration: float, cfg_weight: float,
              temperature: float):
        with lock:  # generate bagimsiz degil; cagrilar siraya girer
            t1 = time.perf_counter()
            token_budget["value"] = speech_token_limit(text)
            try:
                wav = model.generate(text, language_id="tr",
                                     exaggeration=exaggeration, cfg_weight=cfg_weight,
                                     temperature=temperature)
            finally:
                token_budget["value"] = 1000
            gen = time.perf_counter() - t1
            samples = wav.squeeze(0).cpu().numpy()
            return wav_bytes(samples, rate), len(samples) / float(rate), gen

    HANDLER_SYNTH = synth
    HANDLER_RATE = rate
    synth("Evet, dinliyorum.", args.exaggeration, args.cfg_weight,
          args.temperature)

    # 4. Isinma bitti; artik gercek istekler anlik derleme odemeden girer.
    STATE["hazir"] = True
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Hazir: {time.perf_counter() - t0:.1f} sn, tepe VRAM {peak:.2f} GB",
          flush=True)
    print("HAZIR", flush=True)

    try:
        while True:  # main thread parkta; servis daemon thread'de
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
