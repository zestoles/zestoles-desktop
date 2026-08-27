"""Keeping the conversation inside what the model can actually read.

The rolling window was a message count: keep the last N. That works until
somebody pastes a file into the chat, and then twelve "messages" is a context
the model silently truncates from the front -- losing the system prompt, which
is where every rule about not inventing tool success lives.

So the budget is characters, and the pruning is deterministic:

- **whole turns**. Dropping a question and keeping its answer leaves the model
  reading a reply to something it cannot see, which is worse than dropping both.
- **the newest turns always survive**, budget or not. A context that dropped the
  thing just said would make JARVIS answer the previous question.
- **what was dropped is admitted**, in a system line. Never as the user: the
  user did not say it, and a note about missing context filed as their words is
  the same attribution bug the memory layer already had once.

Nothing is summarised by a model here. Dropped turns are already written through
to memory, so the durable copy lives in the layer built for it, and a turn does
not pay for a second model call to compress what it just said.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.assistant.context import (  # noqa: E402
    DEFAULT_BUDGET_CHARS,
    KEEP_RECENT_TURNS,
    NOTICE_ROLE,
    prune,
)


def turns(count, *, size=10):
    out = []
    for i in range(count):
        out.append({"role": "user", "content": f"s{i} " + "x" * size})
        out.append({"role": "assistant", "content": f"c{i} " + "y" * size})
    return out


def contents(history):
    return " ".join(m["content"] for m in history)


class TestNothingToDo(unittest.TestCase):
    def test_an_empty_history_stays_empty(self):
        kept, dropped = prune([], budget_chars=1000)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 0)

    def test_a_short_conversation_is_untouched(self):
        history = turns(3)
        kept, dropped = prune(history, budget_chars=100000)
        self.assertEqual(kept, history)
        self.assertEqual(dropped, 0)

    def test_it_does_not_mutate_what_it_was_given(self):
        history = turns(20, size=500)
        before = len(history)
        prune(history, budget_chars=1000)
        self.assertEqual(len(history), before)


class TestPruning(unittest.TestCase):
    def test_the_oldest_turns_go_first(self):
        history = turns(10, size=200)
        kept, dropped = prune(history, budget_chars=2000)
        self.assertGreater(dropped, 0)
        self.assertNotIn("s0", contents(kept))
        self.assertIn("s9", contents(kept))

    def test_it_stays_inside_the_budget(self):
        history = turns(40, size=300)
        kept, _dropped = prune(history, budget_chars=4000)
        body = sum(len(m["content"]) for m in kept if m["role"] != NOTICE_ROLE)
        self.assertLessEqual(body, 4000)

    def test_turns_are_never_split(self):
        """A reply whose question was dropped is a reply to nothing."""
        history = turns(30, size=200)
        kept, _dropped = prune(history, budget_chars=1500)
        conversation = [m for m in kept if m["role"] != NOTICE_ROLE]
        self.assertEqual(len(conversation) % 2, 0, conversation)
        for i in range(0, len(conversation), 2):
            self.assertEqual(conversation[i]["role"], "user")
            self.assertEqual(conversation[i + 1]["role"], "assistant")

    def test_order_is_preserved(self):
        history = turns(20, size=100)
        kept, _dropped = prune(history, budget_chars=2000)
        conversation = [m for m in kept if m["role"] != NOTICE_ROLE]
        seen = [m["content"].split()[0] for m in conversation if m["role"] == "user"]
        self.assertEqual(seen, sorted(seen, key=lambda s: int(s[1:])))

    def test_the_newest_turn_survives_a_budget_that_cannot_hold_it(self):
        """Dropping what was just said would answer the previous question."""
        history = turns(5, size=5000)
        kept, _dropped = prune(history, budget_chars=100)
        conversation = [m for m in kept if m["role"] != NOTICE_ROLE]
        self.assertTrue(conversation, "son tur her hâlükârda kalmalı")
        self.assertIn("s4", contents(conversation))

    def test_at_least_the_promised_number_of_recent_turns_survives(self):
        history = turns(20, size=4000)
        kept, _dropped = prune(history, budget_chars=10)
        conversation = [m for m in kept if m["role"] != NOTICE_ROLE]
        self.assertGreaterEqual(len(conversation), KEEP_RECENT_TURNS * 2)

    def test_a_single_enormous_message_is_trimmed_rather_than_dropped(self):
        history = [{"role": "user", "content": "A" * 50000},
                   {"role": "assistant", "content": "tamam"}]
        kept, _dropped = prune(history, budget_chars=2000)
        body = contents(kept)
        self.assertIn("tamam", body)
        self.assertLess(len(body), 12000)

    def test_a_trimmed_message_says_it_was_trimmed(self):
        history = [{"role": "user", "content": "A" * 50000},
                   {"role": "assistant", "content": "tamam"}]
        kept, _dropped = prune(history, budget_chars=2000)
        user_line = next(m for m in kept if m["role"] == "user")
        self.assertIn("kısaltıldı", user_line["content"])


class TestTheNotice(unittest.TestCase):
    def test_dropping_anything_is_admitted(self):
        history = turns(20, size=300)
        kept, dropped = prune(history, budget_chars=1500)
        self.assertGreater(dropped, 0)
        self.assertEqual(kept[0]["role"], NOTICE_ROLE)
        self.assertIn(str(dropped), kept[0]["content"])

    def test_the_notice_is_not_attributed_to_the_user(self):
        """The user did not say the context was trimmed. Filing it as their
        words is the attribution bug the memory layer already had once."""
        history = turns(20, size=300)
        kept, _dropped = prune(history, budget_chars=1500)
        self.assertNotEqual(NOTICE_ROLE, "user")
        self.assertNotEqual(NOTICE_ROLE, "assistant")

    def test_nothing_is_said_when_nothing_was_dropped(self):
        kept, dropped = prune(turns(2), budget_chars=100000)
        self.assertEqual(dropped, 0)
        self.assertFalse(any(m["role"] == NOTICE_ROLE for m in kept))

    def test_the_notice_points_at_memory_rather_than_pretending_to_summarise(self):
        history = turns(20, size=300)
        kept, _dropped = prune(history, budget_chars=1500)
        self.assertIn("hafıza", kept[0]["content"].lower())


class TestTheDefaults(unittest.TestCase):
    def test_the_budget_is_big_enough_for_a_real_conversation(self):
        self.assertGreaterEqual(DEFAULT_BUDGET_CHARS, 8000)

    def test_the_floor_keeps_more_than_the_last_exchange(self):
        self.assertGreaterEqual(KEEP_RECENT_TURNS, 2)


class TestTheServiceUsesIt(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile

        from jarvis.assistant import REPLY, Assistant
        from jarvis.assistant.service import AssistantService
        from jarvis.tools import Workspace

        self._tmp = tempfile.TemporaryDirectory()

        class Brain:
            def __init__(self):
                self.local = self
                self.seen = []

            def chat(self, messages, **_kwargs):
                self.seen.append(list(messages))
                return json.dumps({"action": REPLY, "message": "tamam"})

        self.brain = Brain()
        self.service = AssistantService(
            Assistant(self.brain, Workspace(Path(self._tmp.name) / "alan")))

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_huge_conversation_does_not_grow_the_prompt_without_bound(self):
        self.service.history = [
            {"role": "user", "content": "B" * 4000} for _ in range(40)]
        self.service.handle({"op": "sor", "mesaj": "peki"})
        sent = self.brain.seen[-1]
        size = sum(len(m["content"]) for m in sent)
        self.assertLess(size, 60000, f"istem {size} karakter")

    def test_the_system_prompt_is_still_the_first_thing_the_model_sees(self):
        """Everything about not inventing tool success is in it."""
        self.service.history = [
            {"role": "user", "content": "B" * 4000} for _ in range(40)]
        self.service.handle({"op": "sor", "mesaj": "peki"})
        sent = self.brain.seen[-1]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("ZESTOLES", sent[0]["content"])


if __name__ == "__main__":
    unittest.main()
