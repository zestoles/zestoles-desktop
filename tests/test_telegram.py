"""Secure Telegram transport: secret handling, pairing and common-core routing."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.integrations.telegram import TelegramGateway  # noqa: E402


# Built at runtime so generic secret scanners do not mistake this fixture for a
# live Telegram credential in the repository.
TOKEN = "123456789:" + "A" * 35


class FakeService:
    def __init__(self):
        self.calls = []

    def handle(self, payload):
        self.calls.append(payload)
        if payload.get("op") == "sor":
            return {"durum": "hazir", "cevap": "çekirdek yanıtı", "basarili": True}
        if payload.get("op") == "gorevler":
            return {"gorevler": []}
        return {"durum": "hazir"}


class FakeTelegram:
    def __init__(self):
        self.calls = []
        self.valid_token = TOKEN

    def __call__(self, method, params, token):
        self.calls.append((method, dict(params), token))
        if token != self.valid_token:
            return {"ok": False, "description": "Unauthorized"}
        if method == "getMe":
            return {"ok": True, "result": {"username": "zestoles_test_bot"}}
        if method == "getUpdates":
            return {"ok": True, "result": []}
        return {"ok": True, "result": {"message_id": len(self.calls)}}

    @property
    def sent(self):
        return [params["text"] for method, params, _ in self.calls
                if method == "sendMessage"]


class TelegramCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.secret = Path(self.tmp.name) / "secrets" / "telegram.json"
        self.service = FakeService()
        self.api = FakeTelegram()
        self.gateway = TelegramGateway(self.service, secret_file=self.secret,
                                       api=self.api, poll_timeout_s=1)

    def tearDown(self):
        self.gateway.stop()
        self.tmp.cleanup()

    def configure(self):
        result = self.gateway.configure(TOKEN)
        self.assertTrue(result["ok"], result)

    def pair(self, chat_id=42):
        result = self.gateway.new_pair_code()
        self.assertTrue(result["ok"])
        self.gateway._handle_update({  # noqa: SLF001 - transport boundary test
            "update_id": 1, "message": {"chat": {"id": chat_id},
                                         "text": result["komut"]}})


class TestTelegramConfiguration(TelegramCase):
    def test_invalid_token_is_not_written(self):
        result = self.gateway.configure("not-a-token")
        self.assertFalse(result["ok"])
        self.assertFalse(self.secret.exists())

    def test_token_is_validated_and_secret_is_not_returned(self):
        result = self.gateway.configure(TOKEN)
        self.assertTrue(result["ok"])
        self.assertTrue(self.secret.is_file())
        stored_text = self.secret.read_text(encoding="utf-8")
        self.assertNotIn(TOKEN, stored_text)
        self.assertIn("token_dpapi", stored_text)
        self.assertNotIn(TOKEN, json.dumps(result, ensure_ascii=False))
        status = self.gateway.status()
        self.assertNotIn("token", status)
        self.assertEqual(status["bot"], "zestoles_test_bot")

    def test_environment_token_works_without_secret_file(self):
        with patch.dict(os.environ, {"ZESTOLES_TELEGRAM_TOKEN": TOKEN}):
            self.assertTrue(self.gateway.status()["yapilandirildi"])

    def test_api_error_never_exposes_token(self):
        result = self.gateway.configure("999999999:" + "B" * 35)
        self.assertFalse(result["ok"])
        self.assertNotIn("999999999", result["hata"])

    def test_disconnect_removes_secret(self):
        self.configure()
        result = self.gateway.disconnect()
        self.assertTrue(result["ok"])
        self.assertFalse(self.secret.exists())


class TestTelegramPairing(TelegramCase):
    def test_first_message_does_not_claim_ownership(self):
        self.configure()
        self.gateway._handle_update({  # noqa: SLF001
            "update_id": 1, "message": {"chat": {"id": 77}, "text": "merhaba"}})
        self.assertFalse(self.gateway.status()["eslesti"])
        self.assertFalse(self.service.calls)
        self.assertIn("özel", self.api.sent[-1])

    def test_pair_code_claims_owner(self):
        self.configure()
        self.pair(77)
        self.assertTrue(self.gateway.status()["eslesti"])
        self.assertEqual(json.loads(self.secret.read_text(encoding="utf-8"))[
            "owner_chat_id"], 77)

    def test_wrong_pair_code_is_rejected(self):
        self.configure()
        self.gateway.new_pair_code()
        self.gateway._handle_update({  # noqa: SLF001
            "update_id": 1, "message": {"chat": {"id": 77},
                                         "text": "/pair 000000"}})
        self.assertFalse(self.gateway.status()["eslesti"])
        self.assertIn("geçersiz", self.api.sent[-1])

    def test_non_owner_stays_denied_after_pairing(self):
        self.configure()
        self.pair(42)
        self.gateway._handle_update({  # noqa: SLF001
            "update_id": 2, "message": {"chat": {"id": 99}, "text": "dosya sil"}})
        self.assertFalse(self.service.calls)


class TestTelegramRouting(TelegramCase):
    def setUp(self):
        super().setUp()
        self.configure()
        self.pair(42)
        self.api.calls.clear()

    def update(self, text):
        self.gateway._handle_update({  # noqa: SLF001
            "update_id": 2, "message": {"chat": {"id": 42}, "text": text}})

    def test_text_uses_the_same_assistant_service(self):
        self.update("Bugünkü planı hazırla")
        self.assertEqual(self.service.calls[-1],
                         {"op": "sor", "mesaj": "Bugünkü planı hazırla"})
        self.assertIn("çekirdek yanıtı", self.api.sent[-1])

    def test_confirmation_commands_use_the_same_gate(self):
        self.update("/evet")
        self.assertEqual(self.service.calls[-1], {"op": "onay", "evet": True})
        self.update("/hayir")
        self.assertEqual(self.service.calls[-1], {"op": "onay", "evet": False})

    def test_task_command_is_routed(self):
        self.update("/gorevler")
        self.assertEqual(self.service.calls[-1], {"op": "gorevler"})
        self.assertIn("boş", self.api.sent[-1])

    def test_status_never_contains_owner_or_token(self):
        rendered = json.dumps(self.gateway.status(), ensure_ascii=False)
        self.assertNotIn(TOKEN, rendered)
        self.assertNotIn("owner_chat_id", rendered)


if __name__ == "__main__":
    unittest.main()
