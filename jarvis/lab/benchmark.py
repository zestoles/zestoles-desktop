"""Measuring a change, and deciding whether it is actually better.

"The tests passed" is not evidence that a change is an improvement. It is not even
evidence that the change did no harm: the cheapest way to make a suite pass is to
delete the tests that fail, and the second cheapest is to make them skip. Both
produce a green run and both are worse than the code they replaced.

So a benchmark is a comparison against a baseline taken from the same code before
the change, and regressions are defined broadly enough to catch the tricks:

  a test that used to pass and now fails
  fewer tests running than before — deleted, renamed away, or silently skipped
  a new error where there was none
  a run that got materially slower

The comparison is arithmetic on counts. Nothing here asks a model whether the
change seems good, because the whole point of this module is to be the thing a
persuasive-sounding answer cannot talk its way past.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .sandbox import CommandResult, Sandbox

log = logging.getLogger("jarvis.lab.benchmark")

#: How much slower a run may get before it counts as a performance regression.
#: Wall-clock timing on a desktop is noisy — a threshold below about 1.5 would
#: fail runs for reasons that have nothing to do with the change.
SLOWDOWN_LIMIT = 1.5

_SUMMARY = re.compile(r"^Ran (\d+) tests? in ([\d.]+)s", re.MULTILINE)
_FAILURES = re.compile(r"failures=(\d+)")
_ERRORS = re.compile(r"errors=(\d+)")
_SKIPPED = re.compile(r"skipped=(\d+)")


@dataclass(slots=True)
class BenchmarkResult:
    ran: bool = False
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    duration_ms: int = 0
    exit_code: int = 0
    timed_out: bool = False
    output_tail: str = ""

    @property
    def passed(self) -> bool:
        return self.ran and not self.timed_out and self.failures == 0 and self.errors == 0

    @property
    def effective(self) -> int:
        """Tests that actually asserted something. Skips do not count as coverage."""
        return max(0, self.tests - self.skipped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran, "tests": self.tests, "failures": self.failures,
            "errors": self.errors, "skipped": self.skipped, "effective": self.effective,
            "duration_ms": self.duration_ms, "exit_code": self.exit_code,
            "timed_out": self.timed_out, "passed": self.passed,
        }

    def summary(self) -> str:
        if not self.ran:
            return "ölçüm alınamadı"
        state = "geçti" if self.passed else f"{self.failures} hata, {self.errors} error"
        return (f"{self.tests} test ({self.effective} etkin) · {state} "
                f"· {self.duration_ms} ms")


def parse_unittest(result: CommandResult) -> BenchmarkResult:
    """Read a unittest run. Its summary goes to stderr, so both streams are scanned."""
    text = f"{result.stdout}\n{result.stderr}"
    summary = _SUMMARY.search(text)
    if summary is None:
        return BenchmarkResult(
            ran=False, exit_code=result.returncode, timed_out=result.timed_out,
            duration_ms=result.duration_ms, output_tail=text[-1500:])

    failures = _FAILURES.search(text)
    errors = _ERRORS.search(text)
    skipped = _SKIPPED.search(text)
    return BenchmarkResult(
        ran=True,
        tests=int(summary.group(1)),
        failures=int(failures.group(1)) if failures else 0,
        errors=int(errors.group(1)) if errors else 0,
        skipped=int(skipped.group(1)) if skipped else 0,
        duration_ms=result.duration_ms,
        exit_code=result.returncode,
        timed_out=result.timed_out,
        output_tail=text[-1500:],
    )


def run_suite(sandbox: Sandbox, *, python: str = "python", target: str = "tests",
              timeout: int | None = None) -> BenchmarkResult:
    """Run a unittest suite inside the sandbox and measure it."""
    result = sandbox.run([python, "-m", "unittest", "discover", "-s", target],
                         timeout=timeout)
    measured = parse_unittest(result)
    log.info("benchmark: %s", measured.summary())
    return measured


@dataclass(slots=True)
class Comparison:
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        """No regressions and the candidate is green. Both, not either."""
        return not self.regressions and self.candidate.passed

    @property
    def speed_ratio(self) -> float:
        if not self.baseline.duration_ms:
            return 1.0
        return self.candidate.duration_ms / self.baseline.duration_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "regressions": self.regressions,
            "improvements": self.improvements,
            "notes": self.notes,
            "speed_ratio": round(self.speed_ratio, 3),
            "acceptable": self.acceptable,
        }

    def summary(self) -> str:
        if self.regressions:
            return f"regresyon: {self.regressions[0]}"
        if not self.candidate.passed:
            return f"aday geçmedi: {self.candidate.summary()}"
        gain = " · ".join(self.improvements) if self.improvements else "değişiklik yok"
        return f"kabul edilebilir ({gain})"


def compare(baseline: BenchmarkResult, candidate: BenchmarkResult, *,
            slowdown_limit: float = SLOWDOWN_LIMIT) -> Comparison:
    result = Comparison(baseline=baseline, candidate=candidate)

    if not candidate.ran:
        result.regressions.append("aday ölçümü alınamadı")
        return result
    if candidate.timed_out:
        result.regressions.append("aday zaman aşımına uğradı")
    if not baseline.ran:
        result.notes.append("baseline ölçümü yok — karşılaştırma yapılamadı")
        if not candidate.passed:
            result.regressions.append("aday testleri geçmiyor")
        return result

    if candidate.failures > baseline.failures:
        result.regressions.append(
            f"başarısız test arttı ({baseline.failures} → {candidate.failures})")
    if candidate.errors > baseline.errors:
        result.regressions.append(
            f"hata arttı ({baseline.errors} → {candidate.errors})")

    # Deleting or skipping tests is the cheapest way to turn a suite green, so a
    # drop in effective coverage counts as a regression even when everything passes.
    if candidate.effective < baseline.effective:
        result.regressions.append(
            f"etkin test sayısı düştü ({baseline.effective} → {candidate.effective})")
    if candidate.skipped > baseline.skipped:
        result.notes.append(
            f"atlanan test arttı ({baseline.skipped} → {candidate.skipped})")

    ratio = result.speed_ratio
    if baseline.duration_ms > 250 and ratio > slowdown_limit:
        result.regressions.append(f"belirgin yavaşlama ({ratio:.2f}×)")
    elif baseline.duration_ms > 250 and ratio < 0.8:
        result.improvements.append(f"hızlandı ({ratio:.2f}×)")

    if candidate.failures < baseline.failures:
        result.improvements.append(
            f"başarısız test azaldı ({baseline.failures} → {candidate.failures})")
    if candidate.effective > baseline.effective:
        result.improvements.append(
            f"etkin test sayısı arttı ({baseline.effective} → {candidate.effective})")

    return result
