"""A small, dependency-free and owner-paired Telegram bridge.

The legacy project made the first person who messaged the bot its owner and kept
the bot token in source.  This bridge deliberately does neither.  A new token is
stored below the ignored ``data/secrets`` directory and a Telegram account must
prove possession of a short-lived code displayed in the local HUD.

Telegram is only a transport.  Ordinary text and confirmations go through the
same :class:`AssistantService` as the local interface, so remote access cannot
bypass tool risk levels or create a second, less safe assistant.
"""

from __future__ import annotations

import base64
import ctypes
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("jarvis.integrations.telegram")

TOKEN_ENV = "ZESTOLES_TELEGRAM_TOKEN"
PAIR_TTL_S = 10 * 60


class TelegramError(RuntimeError):
    """A redacted Telegram API or configuration failure."""


class _DataBlob(ctypes.Structure):
    """Windows DATA_BLOB used by CryptProtectData/CryptUnprotectData."""

    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _dpapi_protect(value: str) -> str:
    """Encrypt a token for the current Windows account."""
    if os.name != "nt":
        raise TelegramError("Telegram anahtarı yalnız Windows DPAPI ile saklanabilir")
    raw = str(value).encode("utf-8")
    source_buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(
        len(raw), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    protected = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob), ctypes.c_wchar_p,
        ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not crypt32.CryptProtectData(
            ctypes.byref(source), "ZESTOLES Telegram token", None, None, None,
            0x01, ctypes.byref(protected)):
        raise TelegramError(f"Telegram anahtarı şifrelenemedi (Windows {ctypes.get_last_error()})")
    try:
        encrypted = ctypes.string_at(protected.pbData, protected.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        kernel32.LocalFree(ctypes.cast(protected.pbData, ctypes.c_void_p))


def _dpapi_unprotect(value: str) -> str:
    """Decrypt a token that belongs to the current Windows account."""
    if os.name != "nt":
        raise TelegramError("Telegram anahtarı yalnız Windows DPAPI ile açılabilir")
    try:
        raw = base64.b64decode(str(value), validate=True)
    except (ValueError, TypeError) as exc:
        raise TelegramError("şifreli Telegram anahtarı bozuk") from exc
    source_buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(
        len(raw), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    clear = _DataBlob()
    description = ctypes.c_wchar_p()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob), ctypes.POINTER(ctypes.c_wchar_p),
        ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not crypt32.CryptUnprotectData(
            ctypes.byref(source), ctypes.byref(description), None, None, None,
            0x01, ctypes.byref(clear)):
        raise TelegramError(f"Telegram anahtarı açılamadı (Windows {ctypes.get_last_error()})")
    try:
        return ctypes.string_at(clear.pbData, clear.cbData).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TelegramError("şifreli Telegram anahtarı geçersiz") from exc
    finally:
        kernel32.LocalFree(ctypes.cast(clear.pbData, ctypes.c_void_p))
        if description:
            kernel32.LocalFree(ctypes.cast(description, ctypes.c_void_p))


class TelegramGateway:
    def __init__(self, service, *, secret_file: Path, api: Callable | None = None,
                 poll_timeout_s: int = 20) -> None:
        self.service = service
        self.secret_file = Path(secret_file)
        self._api_override = api
        self.poll_timeout_s = max(1, int(poll_timeout_s))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._offset = 0
        self._pair_code = ""
        self._pair_deadline = 0.0
        self._messages: deque[dict[str, Any]] = deque(maxlen=100)
        self._last_error = ""
        self._bot_name = ""

    # --------------------------------------------------------- configuration
    def _stored(self) -> dict[str, Any]:
        if not self.secret_file.is_file():
            return {}
        try:
            data = json.loads(self.secret_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            protected = str(data.get("token_dpapi", "")).strip()
            if protected:
                data["token"] = _dpapi_unprotect(protected)
            return data
        except TelegramError as exc:
            log.warning("Telegram anahtarı açılamadı: %s", exc)
            return {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _token(self) -> str:
        return os.environ.get(TOKEN_ENV, "").strip() or str(
            self._stored().get("token", "")).strip()

    def configure(self, token: str) -> dict[str, Any]:
        candidate = str(token).strip()
        if not candidate or ":" not in candidate or len(candidate) < 20:
            return {"ok": False, "hata": "geçerli bir Telegram bot anahtarı girin"}
        try:
            identity = self._call("getMe", token=candidate)
        except TelegramError as exc:
            return {"ok": False, "hata": str(exc)}
        username = str(identity.get("username", ""))
        stored = self._stored()
        stored.update({"token": candidate, "bot_username": username})
        self._write_secret(stored)
        self._bot_name = username
        self._last_error = ""
        self._record("system", f"@{username or 'bot'} güvenle bağlandı")
        return {"ok": True, "bot": username, "durum": self.status()}

    def disconnect(self) -> dict[str, Any]:
        self.stop()
        try:
            self.secret_file.unlink(missing_ok=True)
        except OSError as exc:
            return {"ok": False, "hata": f"bağlantı bilgisi silinemedi: {exc}"}
        with self._lock:
            self._bot_name = ""
            self._pair_code = ""
            self._pair_deadline = 0
        self._record("system", "Telegram bağlantısı kaldırıldı")
        return {"ok": True, "durum": self.status()}

    def _write_secret(self, data: dict[str, Any]) -> None:
        self.secret_file.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(data)
        token = str(payload.pop("token", "")).strip()
        payload.pop("token_dpapi", None)
        if token:
            payload["token_dpapi"] = _dpapi_protect(token)
        tmp = self.secret_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.secret_file)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> dict[str, Any]:
        if not self._token():
            return {"ok": False, "hata": "önce yeni bot anahtarını kaydedin"}
        if self.running:
            return {"ok": True, "durum": self.status()}
        try:
            identity = self._call("getMe")
        except TelegramError as exc:
            return {"ok": False, "hata": str(exc)}
        self._bot_name = str(identity.get("username", ""))
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="zestoles-telegram",
                                        daemon=True)
        self._thread.start()
        self._record("system", "Telegram dinleyicisi açıldı")
        return {"ok": True, "durum": self.status()}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(self.poll_timeout_s + 2, 25))
        self._thread = None
        return {"ok": True, "durum": self.status()}

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive()
                    and not self._stop.is_set())

    # --------------------------------------------------------------- pairing
    def new_pair_code(self) -> dict[str, Any]:
        if not self._token():
            return {"ok": False, "hata": "önce bot anahtarını kaydedin"}
        with self._lock:
            self._pair_code = f"{secrets.randbelow(1_000_000):06d}"
            self._pair_deadline = time.time() + PAIR_TTL_S
            code = self._pair_code
        return {"ok": True, "kod": code, "gecerli_s": PAIR_TTL_S,
                "komut": f"/pair {code}"}

    def _owner(self) -> int | None:
        try:
            value = self._stored().get("owner_chat_id")
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _accept_pair(self, chat_id: int, text: str) -> bool:
        parts = text.strip().split(maxsplit=1)
        supplied = parts[1].strip() if len(parts) == 2 else ""
        with self._lock:
            valid = bool(self._pair_code and time.time() <= self._pair_deadline
                         and secrets.compare_digest(supplied, self._pair_code))
            if valid:
                self._pair_code = ""
                self._pair_deadline = 0
        if not valid:
            self._send(chat_id, "Eşleştirme kodu geçersiz veya süresi dolmuş.")
            return False
        stored = self._stored()
        stored["owner_chat_id"] = int(chat_id)
        self._write_secret(stored)
        self._record("system", "Telegram sahibi yerel kodla doğrulandı")
        self._send(chat_id, "ZESTOLES eşleştirildi. /yardim ile başlayabilirsin.")
        return True

    # -------------------------------------------------------------- transport
    def _call(self, method: str, params: dict[str, Any] | None = None, *,
              token: str | None = None) -> Any:
        chosen = token or self._token()
        if not chosen:
            raise TelegramError("Telegram yapılandırılmadı")
        if self._api_override is not None:
            result = self._api_override(method, params or {}, chosen)
            if isinstance(result, dict) and result.get("ok") is False:
                raise TelegramError(str(result.get("description", "Telegram isteği reddedildi")))
            return result.get("result", result) if isinstance(result, dict) else result
        url = f"https://api.telegram.org/bot{chosen}/{method}"
        body = urllib.parse.urlencode(params or {}).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST",
                                         headers={"Content-Type":
                                                  "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(request, timeout=self.poll_timeout_s + 5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            # Never include the URL: it contains the secret token.
            raise TelegramError(f"Telegram erişilemedi: {type(exc).__name__}") from None
        if not payload.get("ok"):
            raise TelegramError(str(payload.get("description", "Telegram isteği reddedildi")))
        return payload.get("result")

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                updates = self._call("getUpdates", {
                    "offset": self._offset,
                    "timeout": self.poll_timeout_s,
                    "allowed_updates": json.dumps(["message"]),
                }) or []
                self._last_error = ""
                for update in updates:
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    self._handle_update(update)
            except TelegramError as exc:
                self._last_error = str(exc)
                log.warning("Telegram dinleyicisi: %s", exc)
                self._stop.wait(3)
            except Exception as exc:  # noqa: BLE001 - bridge must stay isolated
                self._last_error = f"beklenmeyen hata: {type(exc).__name__}"
                log.exception("Telegram güncellemesi işlenemedi")
                self._stop.wait(3)

    def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        text = str(message.get("text", "")).strip()
        if chat_id is None or not text:
            return
        chat_id = int(chat_id)
        self._record("in", text)
        if text.lower().startswith("/pair"):
            self._accept_pair(chat_id, text)
            return
        if self._owner() != chat_id:
            self._send(chat_id, "Bu ZESTOLES özel. Yerel HUD'dan eşleştirme kodu oluştur.")
            return
        self._route_owner(chat_id, text)

    def _route_owner(self, chat_id: int, text: str) -> None:
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command in ("/start", "/yardim"):
            self._send(chat_id, "Komutlar: /durum, /gorevler, /hatirlaticilar, /iptal, /evet, /hayir. "
                       "Diğer mesajlar doğrudan ZESTOLES'e gider.")
            return
        operations = {
            "/durum": {"op": "durum"},
            "/gorevler": {"op": "gorevler"},
            "/hatirlaticilar": {"op": "hatirlaticilar"},
            "/iptal": {"op": "iptal"},
            "/evet": {"op": "onay", "evet": True},
            "/hayir": {"op": "onay", "evet": False},
        }
        payload = operations.get(command, {"op": "sor", "mesaj": text})
        try:
            result = self.service.handle(payload)
        except Exception as exc:  # noqa: BLE001
            log.exception("Telegram isteği çekirdeğe aktarılamadı")
            self._send(chat_id, f"İstek tamamlanamadı: {type(exc).__name__}")
            return
        self._send(chat_id, self._render(result))

    @staticmethod
    def _render(result: dict[str, Any]) -> str:
        if result.get("bekleyen"):
            item = result["bekleyen"]
            return (f"Onay gerekiyor: {item.get('arac', 'işlem')} "
                    f"({item.get('risk', 'risk')}). /evet veya /hayir")
        if result.get("hata"):
            return f"Hata: {result['hata']}"
        if result.get("cevap"):
            return str(result["cevap"])
        if "gorevler" in result:
            tasks = result.get("gorevler") or []
            if not tasks:
                return "Görev kuyruğu boş."
            return "\n".join(f"• {t.get('baslik', t.get('id', 'görev'))}: "
                             f"{t.get('durum', '—')}" for t in tasks[:20])
        if "hatirlaticilar" in result:
            items = result.get("hatirlaticilar") or []
            if not items:
                return "Bekleyen hatırlatıcı yok."
            return "\n".join(f"• #{item.get('id')} · {item.get('zaman')} · "
                             f"{item.get('metin')}" for item in items[:20])
        return json.dumps({k: v for k, v in result.items() if k != "gecmis"},
                          ensure_ascii=False, default=str)[:3500]

    def _send(self, chat_id: int, text: str) -> None:
        message = str(text).strip() or "—"
        for start in range(0, len(message), 3900):
            self._call("sendMessage", {"chat_id": chat_id,
                                        "text": message[start:start + 3900]})
        self._record("out", message)

    def _record(self, direction: str, text: str) -> None:
        with self._lock:
            self._messages.append({"yon": direction, "metin": str(text)[:600],
                                   "zaman": time.time()})

    def status(self) -> dict[str, Any]:
        stored = self._stored()
        paired = self._owner() is not None
        return {
            "yapilandirildi": bool(self._token()),
            "calisiyor": self.running,
            "eslesti": paired,
            "bot": self._bot_name or str(stored.get("bot_username", "")),
            "hata": self._last_error,
            "mesajlar": list(self._messages),
        }


__all__ = ["TelegramError", "TelegramGateway", "TOKEN_ENV"]
