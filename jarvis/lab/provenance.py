"""What an experiment actually measured, kept so the answer survives the run.

The registry already says why this has to exist: "An experiment that cannot be
reconstructed afterwards is not an experiment, it is a change of unknown origin."
Until this module, that promise was only half kept. The registry stored what the
measurements *said* — counts, durations, a comparison — and which paths were
touched, but never the bytes. The sandbox held one version at a time: the
candidate is written over the baseline before the second measurement, so by the
time an experiment finished, the code its baseline number came from was gone.

Measured on the 17.08 autonomous run: four experiments promoted with candidates
between 1.6x and 1.9x faster than their baselines. The ordering-bias explanation
was tested and came back at 2.2%, far too small — and the remaining explanation,
that the baseline really was slower, could not be checked, because no copy of the
baseline existed. A number nobody can go back to is not evidence.

## Shape

    data/lab/provenance/<experiment id>/
        manifest.json          hashes, sizes, when, what the target was
        baseline/<relative>    exact bytes the baseline number came from
        candidate/<relative>   exact bytes the candidate number came from

Kept outside the sandbox on purpose. `ExperimentSession.discard()` disposes the
sandbox, and a failed experiment is exactly the one whose inputs someone will
want to read later.

## What it is not

Not a security boundary. It records what was measured so a past decision can be
re-examined; it does not stop anyone with write access from editing both a file
and its recorded hash. What it does catch is the ordinary failure — an artifact
that changed, was truncated, or went missing between the run and the question.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.lab.provenance")

MANIFEST = "manifest.json"
BASELINE = "baseline"
CANDIDATE = "candidate"


class ProvenanceRefused(RuntimeError):
    """A recorded path that will not be written, or a manifest that will not load."""


@dataclass(slots=True)
class FileRecord:
    relative: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size": self.size}


def digest(text: str) -> tuple[str, int]:
    data = text.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _safe_target(root: Path, relative: str) -> Path:
    """Where `relative` may be written, or an error.

    The planner builds these paths rather than taking them from the model, but
    this module is downstream of that and does not get to assume it. Same rule as
    everywhere else: resolve, then require the result to still be inside.
    """
    if not relative or relative.strip() != relative:
        raise ProvenanceRefused(f"kullanılamaz yol: {relative!r}")
    root = Path(root).resolve()
    try:
        target = (root / relative).resolve()
    except (OSError, ValueError) as exc:
        raise ProvenanceRefused(f"yol çözümlenemedi: {relative!r}") from exc
    if target == root or not target.is_relative_to(root):
        raise ProvenanceRefused(f"kayıt dizini dışına yazma girişimi: {relative!r}")
    return target


@dataclass(slots=True)
class Provenance:
    """The recorded inputs of one experiment."""

    store: Path
    experiment_id: str
    baseline: dict[str, FileRecord] = field(default_factory=dict)
    candidate: dict[str, FileRecord] = field(default_factory=dict)
    test_target: str = ""
    created: float = 0.0

    # -------------------------------------------------------------- writing
    @classmethod
    def record(
        cls,
        root: Path,
        experiment_id: str,
        *,
        baseline: dict[str, str],
        candidate: dict[str, str] | None = None,
        test_target: str = "",
    ) -> Provenance:
        """Copy the measured sources and write the manifest beside them."""
        store = Path(root) / experiment_id
        provenance = cls(store=store, experiment_id=experiment_id,
                         test_target=test_target, created=time.time())
        provenance._write_side(BASELINE, baseline, provenance.baseline)
        if candidate:
            provenance._write_side(CANDIDATE, candidate, provenance.candidate)
        provenance._write_manifest()
        log.info("deney kaynağı kaydedildi: %s (%s baseline, %s aday dosya)",
                 experiment_id[:8], len(provenance.baseline), len(provenance.candidate))
        return provenance

    def add_candidate(self, candidate: dict[str, str]) -> None:
        """Record the candidate side after the baseline has been measured."""
        self._write_side(CANDIDATE, candidate, self.candidate)
        self._write_manifest()

    def _write_side(self, side: str, files: dict[str, str],
                    into: dict[str, FileRecord]) -> None:
        base = self.store / side
        base.mkdir(parents=True, exist_ok=True)
        for relative, content in files.items():
            target = _safe_target(base, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            sha, size = digest(content)
            into[relative] = FileRecord(relative, sha, size)

    def _write_manifest(self) -> None:
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / MANIFEST).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # -------------------------------------------------------------- reading
    @classmethod
    def load(cls, root: Path, experiment_id: str) -> Provenance:
        store = Path(root) / experiment_id
        path = store / MANIFEST
        if not path.is_file():
            raise ProvenanceRefused(f"kayıt yok: {experiment_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProvenanceRefused(f"kayıt okunamadı: {exc}") from exc

        def side(name: str) -> dict[str, FileRecord]:
            out: dict[str, FileRecord] = {}
            for relative, entry in (data.get(name) or {}).items():
                try:
                    out[relative] = FileRecord(relative, str(entry["sha256"]),
                                               int(entry["size"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ProvenanceRefused(
                        f"bozuk kayıt girdisi ({name}/{relative}): {exc}") from exc
            return out

        return cls(store=store, experiment_id=str(data.get("experiment", experiment_id)),
                   baseline=side(BASELINE), candidate=side(CANDIDATE),
                   test_target=str(data.get("test_target", "")),
                   created=float(data.get("created", 0.0)))

    def read(self, side: str, relative: str) -> str:
        """The exact bytes that were measured, as text."""
        return _safe_target(self.store / side, relative).read_text(encoding="utf-8")

    def verify(self) -> list[str]:
        """Everything that no longer matches what was recorded. Empty means intact."""
        problems: list[str] = []
        if not self.baseline:
            problems.append("baseline kaydı boş")
        for side, records in ((BASELINE, self.baseline), (CANDIDATE, self.candidate)):
            for relative, record in sorted(records.items()):
                try:
                    path = _safe_target(self.store / side, relative)
                except ProvenanceRefused as exc:
                    problems.append(str(exc))
                    continue
                if not path.is_file():
                    problems.append(f"{side}/{relative}: dosya yok")
                    continue
                content = path.read_text(encoding="utf-8")
                sha, size = digest(content)
                if sha != record.sha256:
                    problems.append(f"{side}/{relative}: içerik değişmiş")
                elif size != record.size:
                    problems.append(f"{side}/{relative}: boyut tutmuyor")
        return problems

    @property
    def intact(self) -> bool:
        return not self.verify()

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment_id,
            "created": self.created,
            "test_target": self.test_target,
            BASELINE: {rel: rec.as_dict() for rel, rec in sorted(self.baseline.items())},
            CANDIDATE: {rel: rec.as_dict() for rel, rec in sorted(self.candidate.items())},
        }

    def summary(self) -> str:
        problems = self.verify()
        state = "bütün" if not problems else f"{len(problems)} sorun"
        return (f"{self.experiment_id[:8]} · {len(self.baseline)} baseline "
                f"· {len(self.candidate)} aday · {state}")


__all__ = ["Provenance", "ProvenanceRefused", "FileRecord", "digest",
           "MANIFEST", "BASELINE", "CANDIDATE"]
