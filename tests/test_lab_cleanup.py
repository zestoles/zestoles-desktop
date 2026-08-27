"""Lab temizlik rutininin kablolaması.

İki fonksiyon S5'ten beri duruyordu ve hiçbir şey çağırmıyordu -- büyüyen iki
alanın koruyucu kodu, kimsenin çalıştırmadığı koddur. Burada denetlenen şey
bağlantının kendisidir: kayıt, rutinin kuyruğa girmesi ve payload'ın geçişi.
Disk davranışının kendisi Lab'ın kendi testlerinde kalır.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any

from jarvis.autonomy import Routine, _configured_routines
from jarvis.autonomy import runners as runners_registry
from jarvis.lab import register_runner


class _FakePromoter:
    def __init__(self) -> None:
        self.kept: int | None = None

    def prune_snapshots(self, *, keep: int = 10) -> int:
        self.kept = keep
        return 3


class _FakeLab:
    def __init__(self) -> None:
        self.promoter = _FakePromoter()
        self.kept: int | None = None

    def cleanup(self, *, keep: int = 5) -> int:
        self.kept = keep
        return 2


@dataclass
class _FakeTask:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeContext:
    task: _FakeTask
    stopped: bool = False

    def should_stop(self) -> bool:
        return self.stopped


class LabCleanupWiringTest(unittest.TestCase):
    def test_registration_and_payload_passthrough(self):
        lab = _FakeLab()
        name = "lab.cleanup"
        self.assertNotIn(name, runners_registry.REGISTRY)
        register_runner(lab)
        try:
            runner = runners_registry.REGISTRY[name]
            ctx = _FakeContext(_FakeTask(payload={
                "keep_sandboxes": 7, "keep_snapshots": 11}))
            summary = runner(ctx)
            self.assertEqual(lab.kept, 7)
            self.assertEqual(lab.promoter.kept, 11)
            self.assertIn("2", summary) and self.assertIn("3", summary)
        finally:
            del runners_registry.REGISTRY[name]

    def test_routine_exists_with_configured_keeps(self):
        class _Cfg:
            def get(self, key, default=None):
                return {"lab.keep_sandboxes": 4, "lab.keep_snapshots": 6}.get(
                    key, default)

        routines = {r.kind: r for r in _configured_routines(_Cfg())}
        routine = routines.get("lab.cleanup")
        self.assertIsInstance(routine, Routine)
        self.assertEqual(routine.payload["keep_sandboxes"], 4)
        self.assertEqual(routine.payload["keep_snapshots"], 6)

    def test_stop_requested_means_no_deletion(self):
        lab = _FakeLab()
        register_runner(lab)
        try:
            runner = runners_registry.REGISTRY["lab.cleanup"]
            ctx = _FakeContext(_FakeTask(), stopped=True)
            self.assertEqual(runner(ctx), "durdurma istendi")
            self.assertIsNone(lab.kept)
        finally:
            del runners_registry.REGISTRY["lab.cleanup"]


if __name__ == "__main__":
    unittest.main()
