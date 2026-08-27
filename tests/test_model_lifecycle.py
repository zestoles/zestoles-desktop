from __future__ import annotations

import unittest
from types import SimpleNamespace

from jarvis.runtime import Runtime
from jarvis.state import SharedState


class Releasable:
    def __init__(self) -> None:
        self.calls = 0

    def unload(self) -> None:
        self.calls += 1


class Voice:
    def __init__(self) -> None:
        self.calls = 0

    def shutdown(self) -> None:
        self.calls += 1


class ModelLifecycleTest(unittest.TestCase):
    def test_runtime_shutdown_releases_every_loaded_model(self):
        chat, memory, embed, voice = Releasable(), Releasable(), Releasable(), Voice()
        runtime = Runtime(config=None, state=SharedState(),
                          brain=SimpleNamespace(local=chat),
                          memory=SimpleNamespace(local=memory, embedder=embed),
                          voice=voice)
        runtime.shutdown()
        self.assertEqual((chat.calls, memory.calls, embed.calls, voice.calls),
                         (1, 1, 1, 1))

    def test_shared_client_is_released_once(self):
        shared = Releasable()
        runtime = Runtime(config=None, state=SharedState(),
                          brain=SimpleNamespace(local=shared),
                          memory=SimpleNamespace(local=shared, embedder=None))
        runtime.shutdown()
        self.assertEqual(shared.calls, 1)


if __name__ == "__main__":
    unittest.main()
