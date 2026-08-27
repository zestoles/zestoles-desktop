"""JARVIS LAB — safe experimentation, measurement, and controlled promotion.

The whole point of S5 is the sentence "only verified changes get promoted", and
that sentence needs four things to be true at once: an experiment can act without
touching the running system, its effect can be measured against what came before,
the decision to keep it is structural rather than persuasive, and installing it can
be undone.

    sandbox      nothing outside it can be reached, so discarding it is a complete
                 rollback of everything written to disk
    registry     what it was for, where it started, what it touched, what happened,
                 and a state machine whose only route to PROMOTED runs through the gate
    benchmark    a comparison against a baseline taken from the same code before the
                 change, with deleted and skipped tests counted as regressions
    promotion    snapshot, journal, atomic per-file writes, rollback on any failure

Self-modification stays off. JARVIS can build a candidate change to its own source,
measure it, and record that it passed — it cannot install it. Every check that
depends on that reads ALLOW_SELF_MODIFICATION, which is False, and turning it on is
a deliberate act rather than a configuration accident.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from .benchmark import BenchmarkResult, Comparison, compare, parse_unittest, run_suite
from .promotion import (
    ALLOW_SELF_MODIFICATION,
    Decision,
    PromotionRefused,
    PromotionResult,
    Promoter,
    Snapshot,
    evaluate,
    recover_interrupted,
)
from .registry import (
    CANDIDATE,
    DISCARDED,
    EXPERIMENT,
    FAILED,
    PASSED,
    PROMOTED,
    Experiment,
    ExperimentRegistry,
    TransitionRefused,
    current_commit,
)
from .provenance import Provenance, ProvenanceRefused
from .sandbox import (
    CommandResult,
    Sandbox,
    SandboxLimits,
    SandboxViolation,
)

log = logging.getLogger("jarvis.lab")


@dataclass(slots=True)
class ExperimentSession:
    """One experiment, from sandbox to verdict.

    Held by the caller rather than driven by the lab, because what an experiment
    *does* differs every time; what happens around it must not.
    """

    lab: Lab
    experiment: Experiment
    sandbox: Sandbox
    baseline: BenchmarkResult | None = None
    candidate: BenchmarkResult | None = None
    comparison: Comparison | None = None
    provenance: Provenance | None = None

    @property
    def id(self) -> str:
        return self.experiment.id

    @property
    def state(self) -> str:
        return self.lab.registry.get(self.id).state

    def record_baseline_source(self, files: dict[str, str], *,
                               test_target: str = "") -> Provenance:
        """Keep the bytes the baseline number will come from.

        Called before the baseline is measured, because the candidate is written
        over the same paths afterwards: once that happens the sandbox no longer
        holds the code the first measurement described.
        """
        self.provenance = Provenance.record(
            self.lab.provenance_dir, self.id, baseline=files, test_target=test_target)
        return self.provenance

    def record_candidate_source(self, files: dict[str, str]) -> None:
        if self.provenance is None:
            return
        self.provenance.add_candidate(files)

    def measure_baseline(self, **kwargs) -> BenchmarkResult:
        """Measure before the change. Everything later is relative to this."""
        self.baseline = run_suite(self.sandbox, **kwargs)
        self.lab.registry.record_measurements(self.id, baseline=self.baseline.as_dict())
        return self.baseline

    def measure_candidate(self, **kwargs) -> BenchmarkResult:
        self.candidate = run_suite(self.sandbox, **kwargs)
        self.lab.registry.record_measurements(self.id, result=self.candidate.as_dict())
        return self.candidate

    def settle(self) -> Comparison:
        """Compare and move the experiment to PASSED or FAILED.

        This is the only place PASSED is set, and it is set by arithmetic.
        """
        if self.candidate is None:
            raise PromotionRefused("aday ölçümü alınmadı")
        self.comparison = compare(self.baseline or BenchmarkResult(), self.candidate)
        self.lab.registry.record_measurements(self.id, comparison=self.comparison.as_dict())

        target = PASSED if self.comparison.acceptable else FAILED
        self.lab.registry.transition(self.id, target, reason=self.comparison.summary())
        self.lab.emit(
            "experiment.settled" if target == PASSED else "experiment.failed",
            f"Deney {self.id[:8]}: {self.comparison.summary()}",
            level="success" if target == PASSED else "warn", experiment=self.id)
        return self.comparison

    def promote(self, files: list[str], *, target_root: Path | None = None,
                allow_self_modification: bool = ALLOW_SELF_MODIFICATION) -> PromotionResult:
        if self.comparison is None:
            raise PromotionRefused("önce settle() çağrılmalı")
        self.lab.registry.record_files(self.id, files)
        return self.lab.promoter.promote(
            self.lab.registry.get(self.id), self.comparison,
            sandbox=self.sandbox, target_root=target_root or self.lab.promotion_target,
            files=files, allow_self_modification=allow_self_modification,
            provenance=self.provenance)

    def would_promote(self, files: list[str], *, target_root: Path | None = None) -> Decision:
        """The gate's verdict without acting on it. Useful for reporting."""
        return evaluate(
            self.lab.registry.get(self.id), self.comparison,
            target_root=target_root or self.lab.promotion_target,
            changed_files=files, project_root=self.lab.project_root)

    def discard(self, reason: str = "elle atıldı") -> None:
        experiment = self.lab.registry.get(self.id)
        if experiment.state not in (FAILED, DISCARDED, PROMOTED):
            self.lab.registry.transition(self.id, FAILED, reason=reason)
        if self.lab.registry.get(self.id).state == FAILED:
            self.lab.registry.transition(self.id, DISCARDED, reason=reason)
        self.sandbox.dispose()
        self.lab.emit("experiment.discarded", f"Deney {self.id[:8]} atıldı: {reason}",
                      experiment=self.id)


class Lab:
    def __init__(self, config: Config, *, events=None) -> None:
        self.config = config
        self.events = events
        self.project_root = config.root
        self.root = config.path("paths.lab", "data/lab")
        self.root.mkdir(parents=True, exist_ok=True)

        self.sandboxes_dir = self.root / "sandboxes"
        self.snapshots_dir = self.root / "snapshots"
        self.journal_dir = self.root / "journal"
        #: Outside the sandboxes on purpose — discard() disposes those, and a
        #: failed experiment is the one whose inputs someone wants to read later.
        self.provenance_dir = self.root / "provenance"
        self.promotion_target = config.path("lab.promotion_target", "data/lab/promoted")
        for directory in (self.sandboxes_dir, self.snapshots_dir, self.journal_dir,
                          self.provenance_dir, self.promotion_target):
            directory.mkdir(parents=True, exist_ok=True)

        self.limits = SandboxLimits(
            timeout_s=int(config.get("lab.timeout_s", 120)),
            max_output=int(config.get("lab.max_output", 200_000)),
            max_file_bytes=int(config.get("lab.max_file_bytes", 5_000_000)),
            max_files=int(config.get("lab.max_files", 2_000)),
            allowed_commands=tuple(config.get(
                "lab.allowed_commands", ["python", "python3", "py", "git", "pip"])),
            allow_network=bool(config.get("lab.allow_network", False)),
        )

        self.registry = ExperimentRegistry(config.path("paths.db", "data/jarvis.db"))
        self.promoter = Promoter(
            self.registry, snapshots_dir=self.snapshots_dir, journal_dir=self.journal_dir,
            project_root=self.project_root, events=events)

        # A journal on disk means a promotion died between its first write and its
        # last. That is the one state the system must not simply carry on from.
        self.recovered = recover_interrupted(self.journal_dir, events=events)

    def emit(self, kind: str, message: str, level: str = "info", **data) -> None:
        if self.events is not None:
            self.events.publish("lab", kind, message, level=level, data=data)

    # ------------------------------------------------------------- sandboxes
    def create_sandbox(self, name: str = "") -> Sandbox:
        from ..text import slugify

        label = slugify(name) if name else "deney"
        directory = self.sandboxes_dir / f"{label}-{uuid.uuid4().hex[:8]}"
        sandbox = Sandbox(directory, limits=self.limits)
        log.info("sandbox açıldı: %s", sandbox.root)
        return sandbox

    def list_sandboxes(self) -> list[Path]:
        return sorted(entry for entry in self.sandboxes_dir.iterdir() if entry.is_dir())

    def cleanup(self, *, keep: int = 5) -> int:
        existing = sorted(self.list_sandboxes(), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for directory in existing[keep:]:
            try:
                Sandbox(directory, limits=self.limits).dispose()
                removed += 1
            except (OSError, SandboxViolation) as exc:
                log.warning("sandbox silinemedi %s: %s", directory, exc)
        return removed

    # ------------------------------------------------------------ experiments
    def experiment(self, purpose: str, *, model: str = "") -> ExperimentSession:
        sandbox = self.create_sandbox(purpose[:30])
        record = self.registry.open(
            purpose, base_commit=current_commit(self.project_root),
            model=model, sandbox_path=str(sandbox.root))
        self.emit("experiment.opened", f"Deney açıldı: {purpose[:80]}",
                  experiment=record.id)
        return ExperimentSession(self, record, sandbox)

    def rollback(self, snapshot_id: str) -> list[str]:
        return self.promoter.rollback(snapshot_id)

    def status(self) -> dict[str, Any]:
        return {
            "kok": str(self.root),
            "hedef": str(self.promotion_target),
            "sandbox_sayisi": len(self.list_sandboxes()),
            "deneyler": self.registry.counts(),
            "anlik_goruntuler": len(self.promoter.list_snapshots()),
            "kurtarilan": len(self.recovered),
            "kendi_kodunu_degistirme": ALLOW_SELF_MODIFICATION,
            "izinli_komutlar": list(self.limits.allowed_commands),
            "ag": self.limits.allow_network,
        }


RUNNER_NAME = "lab.cleanup"


def register_runner(lab: Lab) -> None:
    """Expose sandbox/snapshot cleanup as an S2 routine.

    Both functions existed since S5 and neither was ever called: the only two
    growth spots in data/lab were guarded by code nobody invoked. Registered
    from here so the dependency still points one way -- lab knows about
    autonomy, autonomy knows nothing about lab.
    """
    from ..autonomy import runners as registry

    def _cleanup(ctx) -> str:
        if ctx.should_stop():
            return "durdurma istendi"
        payload = ctx.task.payload or {}
        sandboxes = lab.cleanup(keep=int(payload.get("keep_sandboxes", 5)))
        snapshots = lab.promoter.prune_snapshots(keep=int(payload.get("keep_snapshots", 10)))
        return f"sandbox {sandboxes} silindi · snapshot {snapshots} temizlendi"

    registry.REGISTRY[RUNNER_NAME] = _cleanup
    log.debug("lab temizlik çalıştırıcısı kaydedildi")


__all__ = [
    "Lab", "ExperimentSession", "Sandbox", "SandboxLimits", "SandboxViolation",
    "CommandResult", "ExperimentRegistry", "Experiment", "TransitionRefused",
    "BenchmarkResult", "Comparison", "compare", "run_suite", "parse_unittest",
    "Promoter", "PromotionResult", "PromotionRefused", "Decision", "evaluate",
    "Snapshot", "recover_interrupted", "ALLOW_SELF_MODIFICATION",
    "EXPERIMENT", "PASSED", "CANDIDATE", "PROMOTED", "FAILED", "DISCARDED",
]
