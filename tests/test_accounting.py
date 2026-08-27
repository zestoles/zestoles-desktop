"""Every model call is counted, including the ones nobody is watching.

The S9b soak measured the hole this closes: Ollama's own log recorded a
`POST /api/chat` (11.6s) and a `POST /api/embed` (2.6s) during a night run, and
the `llm_calls` ledger recorded neither. Accounting lived in `Brain.ask()`, and
the improvement engine, the agents and the research pipeline all reach for
`LocalBrain.chat()` directly — so the work that ran unattended was exactly the
work that went unrecorded.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.brain.local import LocalBrain  # noqa: E402
from jarvis.memory.embed import Embedder  # noqa: E402


class Ledger:
    def __init__(self):
        self.rows = []

    def __call__(self, **fields):
        self.rows.append(fields)


class TestLocalChatAccounting(unittest.TestCase):
    def brain(self, ledger, reply=None, error=None):
        client = LocalBrain(host="http://127.0.0.1:11434", model="qwen3.5:9b",
                            usage=ledger)

        def fake_post(path, payload):
            if error is not None:
                raise error
            return reply

        client._post = fake_post  # noqa: SLF001 - the HTTP layer is not the subject
        return client

    def test_a_completed_call_is_recorded(self):
        ledger = Ledger()
        client = self.brain(ledger, reply={
            "message": {"content": "cevap"}, "model": "qwen3.5:9b",
            "prompt_eval_count": 120, "eval_count": 34})
        self.assertEqual(client.chat([{"role": "user", "content": "selam"}]), "cevap")

        self.assertEqual(len(ledger.rows), 1)
        row = ledger.rows[0]
        self.assertEqual(row["tier"], "local")
        self.assertEqual(row["model"], "qwen3.5:9b")
        self.assertTrue(row["ok"])

    def test_token_counts_come_from_the_server_not_from_a_guess(self):
        ledger = Ledger()
        client = self.brain(ledger, reply={
            "message": {"content": "x"}, "prompt_eval_count": 627, "eval_count": 704})
        client.chat([], purpose="deney-plani")
        self.assertEqual(ledger.rows[0]["input_tokens"], 627)
        self.assertEqual(ledger.rows[0]["output_tokens"], 704)

    def test_the_purpose_travels_with_the_call(self):
        ledger = Ledger()
        client = self.brain(ledger, reply={"message": {"content": "x"}})
        client.chat([], purpose="deney-plani")
        self.assertEqual(ledger.rows[0]["purpose"], "deney-plani")

    def test_an_unrecorded_purpose_still_has_one(self):
        ledger = Ledger()
        self.brain(ledger, reply={"message": {"content": "x"}}).chat([])
        self.assertTrue(ledger.rows[0]["purpose"])

    def test_a_failed_call_is_recorded_and_still_raises(self):
        """A model that is down is a fact about the night, not an absence of one."""
        ledger = Ledger()
        client = self.brain(ledger, error=OSError("ollama yok"))
        with self.assertRaises(OSError):
            client.chat([])
        self.assertEqual(len(ledger.rows), 1)
        self.assertFalse(ledger.rows[0]["ok"])
        self.assertIn("ollama", ledger.rows[0]["error"])

    def test_a_broken_ledger_cannot_break_a_call(self):
        def explode(**_fields):
            raise RuntimeError("defter bozuk")

        client = self.brain(explode, reply={"message": {"content": "cevap"}})
        self.assertEqual(client.chat([]), "cevap")

    def test_no_ledger_is_not_an_error(self):
        client = LocalBrain(host="http://127.0.0.1:11434", model="m")
        client._post = lambda path, payload: {"message": {"content": "x"}}  # noqa: SLF001
        self.assertEqual(client.chat([]), "x")

    def test_duration_is_measured(self):
        ledger = Ledger()
        self.brain(ledger, reply={"message": {"content": "x"}}).chat([])
        self.assertGreaterEqual(ledger.rows[0]["duration_ms"], 0)


class TestEmbeddingAccounting(unittest.TestCase):
    def test_an_embedding_run_is_recorded(self):
        ledger = Ledger()
        embedder = Embedder(host="http://127.0.0.1:11434", model="bge-m3",
                            usage=ledger)

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"embeddings": [[0.1, 0.2]]}'

        import jarvis.memory.embed as module

        original = module.urllib.request.urlopen
        module.urllib.request.urlopen = lambda *a, **k: Response()
        try:
            self.assertEqual(embedder.embed(["merhaba"]), [[0.1, 0.2]])
        finally:
            module.urllib.request.urlopen = original

        self.assertEqual(len(ledger.rows), 1)
        self.assertEqual(ledger.rows[0]["purpose"], "embedding")
        self.assertEqual(ledger.rows[0]["tier"], "local")

    def test_nothing_to_embed_is_not_a_call(self):
        ledger = Ledger()
        embedder = Embedder(host="http://x", model="bge-m3", usage=ledger)
        self.assertEqual(embedder.embed([]), [])
        self.assertEqual(ledger.rows, [])


class TestNoDoubleCounting(unittest.TestCase):
    """One call, one row. The ledger moved down a layer; it did not fork."""

    def test_the_brain_no_longer_records_local_calls_itself(self):
        from jarvis.brain import Brain

        source = inspect.getsource(Brain._ask_local)
        self.assertNotIn("budget.record", source)

    def test_the_brain_hands_its_purpose_to_the_client(self):
        from jarvis.brain import Brain

        self.assertIn("purpose=purpose", inspect.getsource(Brain._ask_local))

    def test_streaming_still_books_its_own_call(self):
        """`stream()` does not go through `chat()`, so it keeps its own record."""
        from jarvis.brain import Brain

        self.assertIn("budget.record", inspect.getsource(Brain.stream))

    def test_the_client_does_not_also_record_a_stream(self):
        """One call, one row. The wrapper books the call with its purpose
        label ("chat"), so it must tell LocalBrain.stream to stand down --
        pinned at the call site, because this is the pair that could fork."""
        from jarvis.brain import Brain

        self.assertIn("record=False", inspect.getsource(Brain.stream))


class TestFailedStreamIsRecorded(unittest.TestCase):
    """A stream that died is a fact about the run.

    `chat()` books its failures at the client — pinned above by
    test_a_failed_call_is_recorded_and_still_raises. `stream()` books its own,
    and used to book only the successful ones: the except branch returned before
    reaching budget.record, so a local model that went down mid-conversation
    left the ledger looking like nothing had been asked.
    """

    def brain(self, ledger, exploding: bool):
        from jarvis.brain import Brain

        brain = Brain.__new__(Brain)
        brain.local = _StubLocal(exploding)
        brain.budget = _StubBudget(ledger)
        brain.router = _StubRouter()
        brain.memory = None
        brain.cloud_enabled = False
        brain.core = ""
        brain.user_name = ""
        brain.last = None
        brain.last_decision = None
        return brain

    def drain(self, brain):
        return "".join(brain.stream([{"role": "user", "content": "selam"}]))

    def test_a_stream_that_fails_is_recorded_as_a_failure(self):
        ledger = Ledger()
        text = self.drain(self.brain(ledger, exploding=True))
        self.assertIn("yanıt vermedi", text)
        self.assertEqual(len(ledger.rows), 1, ledger.rows)
        self.assertFalse(ledger.rows[0]["ok"])
        self.assertIn("ollama", ledger.rows[0]["error"])
        self.assertEqual(ledger.rows[0]["tier"], "local")

    def test_a_stream_that_works_is_recorded_once_as_a_success(self):
        ledger = Ledger()
        self.assertEqual(self.drain(self.brain(ledger, exploding=False)), "merhaba")
        self.assertEqual(len(ledger.rows), 1, ledger.rows)
        self.assertTrue(ledger.rows[0]["ok"])


class _StubLocal:
    model = "qwen3.5:9b"

    def __init__(self, exploding: bool):
        self.exploding = exploding

    def stream(self, _payload, *, record: bool = True):
        # Gerçek istemci record=False gelince kaydetmeyi bırakır; burada defter
        # zaten sahte, yalnızca imzanın güncel sözleşmeye uyması yeter.
        if self.exploding:
            raise OSError("ollama yok")
        yield "merhaba"


class _StubBudget:
    def __init__(self, ledger):
        self.record = ledger


class _StubRouter:
    def decide(self, _text):
        return type("D", (), {"tier": "local", "reason": "yerel"})()


if __name__ == "__main__":
    unittest.main()
