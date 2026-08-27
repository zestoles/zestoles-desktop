"""Deciding whether an experiment may become real, and undoing it when it should not.

Three separate things live here, in the order they matter:

## The gate

`evaluate()` is a pure function over recorded facts: the experiment's state, the
benchmark comparison, and the list of files it wants to change. It returns a list
of checks with pass/fail and a verdict. No model is consulted and no free text is
read, because the failure this exists to prevent is exactly a persuasive-sounding
argument that a change is fine.

Every check must pass. There is no weighting, no majority and no override.

## Snapshot and rollback

Before anything is written, the current contents of every affected file are copied
and hashed. Files that do not exist yet are recorded as absent, so rolling back
deletes them rather than leaving debris that looks like it was always there.

## Atomicity

Per-file writes are atomic — content goes to a temporary file and is moved into
place with `os.replace`. Across several files it cannot be, so a journal is written
before the first change and removed after the last. A journal found at startup
means a promotion was interrupted, and the recorded snapshot is restored before
anything else runs. The system is therefore either fully on the old version or
fully on the new one, never halfway.

## Self-modification is off

`ALLOW_SELF_MODIFICATION` is False and every check that depends on it refuses.
JARVIS may build a candidate change to its own source, measure it, and record that
it passed — it may not install it. That switch is a separate decision with its own
consequences, and this phase does not make it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark import Comparison
from .registry import CANDIDATE, FAILED, PASSED, PROMOTED, Experiment, ExperimentRegistry
from .sandbox import Sandbox, SandboxViolation

log = logging.getLogger("jarvis.lab.promotion")

#: The hard stop for this phase. Flipping it is not a configuration change; the
#: checks that read it exist so that turning it on is a visible, deliberate act.
ALLOW_SELF_MODIFICATION = False

JOURNAL_SUFFIX = ".promotion.json"


class PromotionRefused(RuntimeError):
    pass


@dataclass(slots=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class Decision:
    checks: list[Check] = field(default_factory=list)

    @property
    def promotable(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def blocking(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "promotable": self.promotable,
            "checks": [{"ad": c.name, "gecti": c.passed, "detay": c.detail}
                       for c in self.checks],
            "blocking": self.blocking,
        }

    def summary(self) -> str:
        if self.promotable:
            return f"{len(self.checks)}/{len(self.checks)} kontrol geçti — aday"
        return f"reddedildi: {self.blocking[0]}"


def protected_roots(project_root: Path) -> tuple[Path, ...]:
    """Places a promotion may never write while self-modification is off."""
    root = Path(project_root).resolve()
    return tuple(p for p in (
        root / "jarvis", root / "tests", root / "persona",
        root / "run.py", root / "config.json",
    ))


def evaluate(
    experiment: Experiment,
    comparison: Comparison | None,
    *,
    target_root: Path,
    changed_files: list[str],
    project_root: Path,
    allow_self_modification: bool = ALLOW_SELF_MODIFICATION,
    provenance=None,
) -> Decision:
    """Whether this experiment may be promoted. Arithmetic and paths only.

    `provenance` is optional and backward compatible: when it is None the gate
    behaves exactly as before, because callers outside the improvement engine
    (S5 by hand, the tests) never recorded one. When it is supplied it must
    verify — a recorded baseline that no longer matches its hashes means the
    measurement this decision rests on can no longer be checked.
    """
    decision = Decision()

    decision.checks.append(Check(
        "durum", experiment.state == PASSED,
        f"deney durumu {experiment.state}, PASSED olmalı"))

    if comparison is None:
        decision.checks.append(Check("ölçüm", False, "karşılaştırma yok"))
        return decision

    decision.checks.append(Check(
        "aday testleri", comparison.candidate.passed,
        comparison.candidate.summary()))
    # The planner already tells the model "the tests must pass against BOTH
    # baseline_code and candidate_code ... a test that only passes on the
    # candidate is a broken experiment, not a successful one" — and until this
    # check existed, nothing enforced it. A baseline that fails its own test
    # makes every "not worse than baseline" comparison vacuously true, which is
    # the cheapest way past this gate: write a broken before-version and any
    # after-version clears it. Measured on the 17.08 autonomous run: experiment
    # cb36e50d was promoted on a baseline with errors=1, exit_code=1.
    #
    # Comparison.acceptable deliberately still calls that case acceptable. It
    # answers "is the candidate worse", which is a different question from "may
    # this be installed", and callers depend on the first one.
    decision.checks.append(Check(
        "baseline geçerli",
        comparison.baseline.ran and comparison.baseline.passed,
        comparison.baseline.summary()))
    decision.checks.append(Check(
        "regresyon yok", not comparison.regressions,
        "; ".join(comparison.regressions) or "regresyon bulunmadı"))
    decision.checks.append(Check(
        "kapsam korundu",
        comparison.candidate.effective >= comparison.baseline.effective,
        f"etkin test {comparison.baseline.effective} → {comparison.candidate.effective}"))
    decision.checks.append(Check(
        "değişiklik var", bool(changed_files),
        f"{len(changed_files)} dosya"))

    if provenance is not None:
        problems = provenance.verify()
        decision.checks.append(Check(
            "kaynak kaydı bütün", not problems,
            "; ".join(problems[:3]) or provenance.summary()))

    resolved_target = Path(target_root).resolve()
    protected = protected_roots(project_root)

    inside_protected = [
        str(p) for p in protected
        if resolved_target == p or resolved_target.is_relative_to(p)
    ]
    decision.checks.append(Check(
        "hedef korumalı değil",
        allow_self_modification or not inside_protected,
        f"hedef korumalı alanda: {', '.join(inside_protected)}"
        if inside_protected else f"hedef {resolved_target}"))

    offending: list[str] = []
    for relative in changed_files:
        try:
            candidate_path = (resolved_target / relative).resolve()
        except (OSError, ValueError):
            offending.append(relative)
            continue
        if not candidate_path.is_relative_to(resolved_target):
            offending.append(relative)
            continue
        if not allow_self_modification and any(
                candidate_path == p or candidate_path.is_relative_to(p) for p in protected):
            offending.append(relative)

    decision.checks.append(Check(
        "dosyalar hedef içinde", not offending,
        f"hedef dışına veya korumalı alana yazacak: {', '.join(offending[:3])}"
        if offending else f"{len(changed_files)} dosya hedef içinde"))

    return decision


# --------------------------------------------------------------------- snapshot
@dataclass(slots=True)
class SnapshotEntry:
    relative: str
    existed: bool
    sha256: str = ""
    size: int = 0


@dataclass(slots=True)
class Snapshot:
    id: str
    target_root: str
    store: str
    entries: list[SnapshotEntry] = field(default_factory=list)
    created: float = 0.0

    @classmethod
    def create(cls, target_root: Path, relatives: list[str], store_dir: Path) -> Snapshot:
        snapshot_id = uuid.uuid4().hex[:12]
        store = Path(store_dir) / snapshot_id
        (store / "files").mkdir(parents=True, exist_ok=True)
        target_root = Path(target_root).resolve()

        entries: list[SnapshotEntry] = []
        for relative in sorted(set(relatives)):
            source = target_root / relative
            if source.is_file():
                data = source.read_bytes()
                copy = store / "files" / relative
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(data)
                entries.append(SnapshotEntry(
                    relative, True, hashlib.sha256(data).hexdigest(), len(data)))
            else:
                # Recorded as absent so a rollback deletes it rather than leaving
                # a new file behind looking like it was always there.
                entries.append(SnapshotEntry(relative, False))

        snapshot = cls(snapshot_id, str(target_root), str(store), entries, time.time())
        (store / "manifest.json").write_text(
            json.dumps(snapshot.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("anlık görüntü alındı: %s (%s dosya)", snapshot_id, len(entries))
        return snapshot

    @classmethod
    def load(cls, store: Path) -> Snapshot:
        data = json.loads((Path(store) / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            id=data["id"], target_root=data["target_root"], store=str(store),
            entries=[SnapshotEntry(**entry) for entry in data["entries"]],
            created=data.get("created", 0.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "target_root": self.target_root, "created": self.created,
            "entries": [{"relative": e.relative, "existed": e.existed,
                         "sha256": e.sha256, "size": e.size} for e in self.entries],
        }

    def restore(self) -> list[str]:
        """Put the target back exactly as it was. Returns what changed."""
        target = Path(self.target_root)
        restored: list[str] = []
        for entry in self.entries:
            destination = target / entry.relative
            if entry.existed:
                source = Path(self.store) / "files" / entry.relative
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != entry.sha256:
                    raise PromotionRefused(
                        f"anlık görüntü bozulmuş: {entry.relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(destination, data)
                restored.append(entry.relative)
            elif destination.exists():
                destination.unlink()
                restored.append(entry.relative)
        log.info("anlık görüntü geri yüklendi: %s (%s dosya)", self.id, len(restored))
        return restored

    def verify(self) -> bool:
        for entry in self.entries:
            if not entry.existed:
                continue
            source = Path(self.store) / "files" / entry.relative
            if not source.is_file():
                return False
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry.sha256:
                return False
        return True


def _atomic_write(destination: Path, data: bytes) -> None:
    """Write via a temporary file and a rename. Atomic for one file on one volume."""
    temporary = destination.with_name(destination.name + f".tmp-{uuid.uuid4().hex[:8]}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------- journal
def _journal_path(journal_dir: Path, experiment_id: str) -> Path:
    return Path(journal_dir) / f"{experiment_id}{JOURNAL_SUFFIX}"


def write_journal(journal_dir: Path, experiment_id: str, snapshot: Snapshot,
                  files: list[str]) -> Path:
    Path(journal_dir).mkdir(parents=True, exist_ok=True)
    path = _journal_path(journal_dir, experiment_id)
    _atomic_write(path, json.dumps({
        "experiment": experiment_id,
        "snapshot_store": snapshot.store,
        "target_root": snapshot.target_root,
        "files": files,
        "started": time.time(),
    }, ensure_ascii=False, indent=2).encode("utf-8"))
    return path


def recover_interrupted(journal_dir: Path, *, events=None) -> list[dict[str, Any]]:
    """Roll back any promotion that never finished.

    Called at startup. A journal on disk means the process died between the first
    write and the last, which is the one state the system must never simply carry
    on from.
    """
    directory = Path(journal_dir)
    if not directory.exists():
        return []

    recovered: list[dict[str, Any]] = []
    for journal in sorted(directory.glob(f"*{JOURNAL_SUFFIX}")):
        try:
            data = json.loads(journal.read_text(encoding="utf-8"))
            snapshot = Snapshot.load(Path(data["snapshot_store"]))
            restored = snapshot.restore()
            journal.unlink(missing_ok=True)
        except (OSError, KeyError, json.JSONDecodeError, PromotionRefused) as exc:
            log.error("yarım kalan terfi geri alınamadı (%s): %s", journal.name, exc)
            recovered.append({"journal": journal.name, "ok": False, "error": str(exc)})
            continue

        log.warning("yarım kalan terfi geri alındı: %s (%s dosya)",
                    data.get("experiment", "?"), len(restored))
        recovered.append({"journal": journal.name, "ok": True,
                          "experiment": data.get("experiment"), "restored": restored})
        if events is not None:
            events.publish("lab", "recover",
                           f"Yarım kalan terfi geri alındı: {len(restored)} dosya",
                           level="warn", data={"experiment": data.get("experiment")})
    return recovered


# -------------------------------------------------------------------- promoting
@dataclass(slots=True)
class PromotionResult:
    ok: bool
    experiment_id: str
    applied: list[str] = field(default_factory=list)
    snapshot_id: str = ""
    rolled_back: bool = False
    error: str = ""

    def summary(self) -> str:
        if self.ok:
            return f"{len(self.applied)} dosya terfi etti (geri dönüş: {self.snapshot_id})"
        return f"terfi başarısız{' — geri alındı' if self.rolled_back else ''}: {self.error}"


class Promoter:
    def __init__(self, registry: ExperimentRegistry, *, snapshots_dir: Path,
                 journal_dir: Path, project_root: Path, events=None) -> None:
        self.registry = registry
        self.snapshots_dir = Path(snapshots_dir)
        self.journal_dir = Path(journal_dir)
        self.project_root = Path(project_root)
        self.events = events
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(parents=True, exist_ok=True)

    def _emit(self, kind: str, message: str, level: str = "info", **data) -> None:
        if self.events is not None:
            self.events.publish("lab", kind, message, level=level, data=data)

    def _mark_failed(self, experiment_id: str, reason: str) -> None:
        """Record a failure, tolerating an experiment that already failed.

        Refusing a promotion must never itself raise: an already-FAILED experiment
        being offered again is exactly the case this gate exists to turn away
        quietly, and a bookkeeping exception there would hide the refusal behind a
        stack trace.
        """
        from .registry import TRANSITIONS

        experiment = self.registry.get(experiment_id)
        if experiment is None:
            return
        if FAILED in TRANSITIONS.get(experiment.state, frozenset()):
            self.registry.transition(experiment_id, FAILED, reason=reason)

    def promote(
        self,
        experiment: Experiment,
        comparison: Comparison,
        *,
        sandbox: Sandbox,
        target_root: Path,
        files: list[str],
        allow_self_modification: bool = ALLOW_SELF_MODIFICATION,
        provenance=None,
    ) -> PromotionResult:
        """Apply an experiment's files to the target, or leave nothing behind."""
        decision = evaluate(
            experiment, comparison, target_root=target_root, changed_files=files,
            project_root=self.project_root,
            allow_self_modification=allow_self_modification,
            provenance=provenance)

        if not decision.promotable:
            self.registry.record_measurements(experiment.id, comparison=comparison.as_dict())
            self._mark_failed(experiment.id, decision.summary())
            self._emit("promotion.refused", f"Terfi reddedildi: {decision.blocking[0]}",
                       level="warn", experiment=experiment.id)
            return PromotionResult(False, experiment.id, error=decision.summary())

        # Read everything out of the sandbox first. A read that fails now costs
        # nothing; one that fails halfway through writing costs a broken target.
        try:
            payload = {relative: sandbox.read(relative) for relative in files}
        except (SandboxViolation, OSError) as exc:
            self._mark_failed(experiment.id, f"okuma hatası: {exc}")
            return PromotionResult(False, experiment.id, error=str(exc))

        experiment = self.registry.transition(
            experiment.id, CANDIDATE, reason=decision.summary())

        target = Path(target_root).resolve()
        target.mkdir(parents=True, exist_ok=True)
        snapshot = Snapshot.create(target, files, self.snapshots_dir)
        journal = write_journal(self.journal_dir, experiment.id, snapshot, files)

        applied: list[str] = []
        try:
            for relative, content in payload.items():
                destination = target / relative
                if not destination.resolve().is_relative_to(target):
                    raise PromotionRefused(f"hedef dışına yazma girişimi: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(destination, content.encode("utf-8"))
                applied.append(relative)
        except Exception as exc:  # noqa: BLE001 - any failure means undo everything
            log.error("terfi sırasında hata, geri alınıyor: %s", exc)
            rolled_back = True
            try:
                snapshot.restore()
            except PromotionRefused as restore_error:
                rolled_back = False
                log.critical("geri alma da başarısız: %s", restore_error)
            journal.unlink(missing_ok=True)
            self._mark_failed(experiment.id, f"uygulama hatası: {exc}")
            self._emit("promotion.failed", f"Terfi başarısız, geri alındı: {exc}",
                       level="error", experiment=experiment.id)
            return PromotionResult(False, experiment.id, applied=[], rolled_back=rolled_back,
                                   snapshot_id=snapshot.id, error=str(exc))

        journal.unlink(missing_ok=True)
        self.registry.record_measurements(experiment.id, comparison=comparison.as_dict())
        self.registry.record_files(experiment.id, files)
        self.registry.transition(experiment.id, PROMOTED,
                                 reason=f"anlık görüntü {snapshot.id}")
        self._emit("promotion.done", f"{len(applied)} dosya terfi etti",
                   level="success", experiment=experiment.id, snapshot=snapshot.id)
        return PromotionResult(True, experiment.id, applied=applied, snapshot_id=snapshot.id)

    def rollback(self, snapshot_id: str) -> list[str]:
        """Undo a completed promotion by restoring its snapshot."""
        store = self.snapshots_dir / snapshot_id
        if not store.is_dir():
            raise PromotionRefused(f"anlık görüntü yok: {snapshot_id}")
        snapshot = Snapshot.load(store)
        if not snapshot.verify():
            raise PromotionRefused(f"anlık görüntü doğrulanamadı: {snapshot_id}")
        restored = snapshot.restore()
        self._emit("rollback", f"Geri alındı: {len(restored)} dosya",
                   level="warn", snapshot=snapshot_id)
        return restored

    def list_snapshots(self) -> list[str]:
        return sorted(entry.name for entry in self.snapshots_dir.iterdir() if entry.is_dir())

    def prune_snapshots(self, *, keep: int = 10) -> int:
        stores = sorted(
            (entry for entry in self.snapshots_dir.iterdir() if entry.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for store in stores[keep:]:
            shutil.rmtree(store, ignore_errors=True)
            removed += 1
        return removed
