"""Turning a hypothesis into an experiment the lab can actually run.

This is the piece S6 left out. The engine could observe, detect a gap, propose a
hypothesis and score it — and then stopped with "deney planlayıcı verilmedi",
because nothing turned an idea into files a sandbox could measure. S6 was tested
with plans fed in by hand.

## What an experiment is here

Not a change to JARVIS. Self-modification is off and stays off, so the loop can
never test itself by editing itself. What it can do is test the *idea* behind a
hypothesis in miniature: a small self-contained module, a test suite that pins
the behaviour, and a second version of the module that claims to be better. The
lab measures both and the arithmetic decides.

That is a real experiment with a real answer, and it is honest about its scope:
it says "this approach beats that approach on these tests, on this machine", not
"JARVIS is now better".

## The model is not trusted

Everything below exists because a language model wrote the input. The defence is
in layers, and none of them is "ask the model to behave":

**The model never supplies a path.** It supplies a module *name*, matched against
`^[a-z][a-z0-9_]{2,30}$`. The planner builds the paths. The entire traversal
class — `..`, absolute paths, drive-relative, UNC, ADS, device names — cannot be
expressed in the output format, so it cannot be attempted.

**The schema constrains the shape, the validator constrains the content.** The
schema is enforced by the model server and is therefore only as trustworthy as
the server; the validator is a pure function here and assumes the schema bought
nothing.

**The code is parsed before it is run.** Anything that does not parse is a
rejected plan rather than a failed experiment: cheaper, and it keeps "the model
wrote nonsense" separate from "the idea was wrong". Imports are an allowlist, and
a short list of names that make a sandbox pointless is refused outright.

**The sandbox is the same one S5 built.** Every write goes through
`Sandbox.write`, which resolves paths, refuses links, hard links, device names
and alternate data streams. The planner adds no way around it.

## Honest limits

The AST allowlist raises the cost of an escape; it does not make one impossible.
`Sandbox` is a workspace boundary, not a security boundary — a permitted command
can do whatever that program can do — and that is written down in the S5 notes
for the same reason it is written down here. What this module guarantees is that
generated code cannot *name* a path outside the sandbox and cannot import the
obvious tools for leaving it. A determined bypass by a capable adversarial model
is not in scope, and pretending otherwise would be the kind of claim this project
exists to avoid.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .engine import ImprovementPlan

log = logging.getLogger("jarvis.improve.planner")

#: The model names a module; the planner builds every path from it.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,30}$")

#: Reserved because the generated module is imported by the generated test.
RESERVED_NAMES = frozenset({
    "unittest", "test", "tests", "json", "os", "sys", "time", "re", "math",
    "typing", "random", "string", "abc", "io", "code", "types", "copy",
})

#: What generated code may import. Everything an experiment needs to compute and
#: assert, and nothing that reaches the filesystem, the network or the process.
ALLOWED_IMPORTS = frozenset({
    "unittest", "math", "time", "random", "string", "re", "json", "decimal",
    "fractions", "statistics", "collections", "collections.abc", "itertools",
    "functools", "operator", "heapq", "bisect", "array", "dataclasses",
    "typing", "enum", "copy", "textwrap", "difflib", "hashlib", "unicodedata",
})

#: Names that would make the sandbox beside the point.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview", "exit", "quit", "help",
})

#: Dunder attributes an experiment has a reason to touch. Everything else is the
#: first step of an escape (`__class__.__mro__`, `__globals__`, `__subclasses__`).
ALLOWED_DUNDERS = frozenset({
    "__init__", "__name__", "__main__", "__repr__", "__str__", "__eq__",
    "__hash__", "__lt__", "__le__", "__gt__", "__ge__", "__len__", "__iter__",
    "__next__", "__contains__", "__enter__", "__exit__", "__call__", "__doc__",
    "__post_init__", "__all__", "__add__", "__radd__", "__bool__", "__getitem__",
    "__setitem__", "__dict__",
})

MAX_FILE_BYTES = 20_000
MAX_TOTAL_BYTES = 50_000

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "module_name": {"type": "string"},
        "summary": {"type": "string"},
        "baseline_code": {"type": "string"},
        "candidate_code": {"type": "string"},
        "test_code": {"type": "string"},
    },
    "required": ["module_name", "baseline_code", "candidate_code", "test_code"],
}

PLANNER_SYSTEM = """\
Reason in English. Write code only.

You design a miniature experiment that compares two implementations of the same
idea. It runs alone in an empty sandbox with nothing but the Python standard
library, so it must be complete and self-contained.

You produce four things:

  module_name    a lowercase identifier, letters digits underscore, e.g. "kuyruk"
  baseline_code  a module implementing the straightforward version
  candidate_code the same module rewritten the way the hypothesis suggests
  test_code      a unittest file that imports the module and pins its behaviour

Rules that are enforced by code, not by trust. Breaking one means your plan is
thrown away without being run:

  Import only from: unittest, math, time, random, string, re, json, decimal,
  fractions, statistics, collections, itertools, functools, operator, heapq,
  bisect, array, dataclasses, typing, enum, copy, textwrap, difflib, hashlib,
  unicodedata. Nothing else. No os, no sys, no subprocess, no pathlib, no
  urllib.

  Never use open, eval, exec, compile, __import__, globals, locals, or input.
  Never touch dunder attributes like __class__ or __subclasses__.

  No file paths, no network, no reading anything from disk. Generate whatever
  data you need inside the test.

  test_code must import the module by its name and must contain a
  unittest.TestCase with at least two test methods and real assertions.

  The tests must pass against BOTH baseline_code and candidate_code. They pin
  behaviour, they do not describe the improvement. A test that only passes on
  the candidate is a broken experiment, not a successful one.

  baseline_code and candidate_code must define the same public names with the
  same call signatures, and must differ from each other.

Keep every file under 200 lines. Write no explanation outside the code.
"""


@dataclass(slots=True)
class PlanReview:
    """The verdict on one generated plan, and why."""

    plan: ImprovementPlan | None = None
    problems: list[str] = field(default_factory=list)
    module_name: str = ""
    summary: str = ""

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.problems

    def report(self) -> str:
        if self.ok:
            return f"plan kabul edildi: {self.module_name}"
        return "plan reddedildi: " + " · ".join(self.problems[:3])


# --------------------------------------------------------------- static checks
def _import_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        # A relative import has no module to allowlist and nothing to reach for.
        return [node.module or ""] if node.level == 0 else ["."]
    return []


def inspect_code(source: str, label: str, *,
                 extra_allowed: frozenset[str] = frozenset()) -> list[str]:
    """Everything wrong with one generated file. Pure; no execution, no I/O.

    `extra_allowed` exists for exactly one case: the test file has to import the
    module under test, whose name is not known until the plan is read. It is a
    single name the validator itself computed, never one the model supplied.
    """
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{label}: sözdizimi hatası satır {exc.lineno}: {exc.msg}"]

    allowed = ALLOWED_IMPORTS | extra_allowed
    for node in ast.walk(tree):
        for name in _import_names(node):
            root = name.split(".")[0]
            if name == ".":
                problems.append(f"{label}: göreli import yasak")
            elif name not in allowed and root not in allowed:
                problems.append(f"{label}: izinsiz import: {name}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"{label}: yasaklı isim: {node.id}")
        if isinstance(node, ast.Attribute):
            attribute = node.attr
            if (attribute.startswith("__") and attribute.endswith("__")
                    and attribute not in ALLOWED_DUNDERS):
                problems.append(f"{label}: yasaklı dunder erişimi: {attribute}")
            if attribute in FORBIDDEN_NAMES:
                problems.append(f"{label}: yasaklı çağrı: .{attribute}")
    return problems


def inspect_test(source: str, module_name: str) -> list[str]:
    """A test file has to actually test something, and test the right module."""
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # already reported by inspect_code

    imported = set()
    for node in ast.walk(tree):
        for name in _import_names(node):
            imported.add(name.split(".")[0])
    if module_name not in imported:
        problems.append(f"test dosyası {module_name} modülünü import etmiyor")

    cases = [n for n in ast.walk(tree)
             if isinstance(n, ast.ClassDef)
             and any(_base_name(b).endswith("TestCase") for b in n.bases)]
    if not cases:
        problems.append("test dosyasında unittest.TestCase yok")

    methods = [n.name for case in cases for n in case.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name.startswith("test")]
    if len(methods) < 2:
        problems.append(f"en az iki test metodu gerekiyor (bulunan: {len(methods)})")

    asserts = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr.startswith("assert")]
    if not asserts:
        problems.append("testlerde hiç assert yok")
    return problems


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def public_names(source: str) -> set[str]:
    """Top-level functions and classes a test could call."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")}


def validate(payload: dict[str, Any], hypothesis_id: str) -> PlanReview:
    """Turn an untrusted reply into a plan, or into the reasons it is not one.

    Pure function on purpose: it is the one thing standing between a model's
    output and a subprocess, and it must not depend on anything that could be
    talked into changing its mind.
    """
    review = PlanReview()

    # Not lowercased for the model: the test file has to import this exact name,
    # and silently normalising it here would turn a mismatch the model created
    # into an import error inside the sandbox twenty seconds later.
    name = str(payload.get("module_name", "")).strip()
    if not NAME_PATTERN.match(name):
        review.problems.append(f"modül adı kurala uymuyor: {name[:40]!r}")
        return review
    if name in RESERVED_NAMES:
        review.problems.append(f"modül adı ayrılmış: {name}")
        return review
    review.module_name = name
    review.summary = str(payload.get("summary", ""))[:200]

    baseline = str(payload.get("baseline_code", ""))
    candidate = str(payload.get("candidate_code", ""))
    test = str(payload.get("test_code", ""))

    for label, source in (("baseline", baseline), ("candidate", candidate),
                          ("test", test)):
        if not source.strip():
            review.problems.append(f"{label}: boş")
        elif len(source.encode("utf-8")) > MAX_FILE_BYTES:
            review.problems.append(f"{label}: {MAX_FILE_BYTES} bayt sınırını aşıyor")
    total = sum(len(s.encode("utf-8")) for s in (baseline, candidate, test))
    if total > MAX_TOTAL_BYTES:
        review.problems.append(f"toplam boyut sınırı aşıldı ({total})")
    if review.problems:
        return review

    for label, source in (("baseline", baseline), ("candidate", candidate)):
        review.problems.extend(inspect_code(source, label))
    review.problems.extend(
        inspect_code(test, "test", extra_allowed=frozenset({name})))
    review.problems.extend(inspect_test(test, name))

    if baseline.strip() == candidate.strip():
        review.problems.append("aday ile baseline aynı — ölçülecek bir fark yok")

    # A candidate that dropped a public name cannot be measured by the same
    # tests, and the comparison would silently be about something else.
    missing = public_names(baseline) - public_names(candidate)
    if missing:
        review.problems.append(
            f"aday baseline'daki isimleri kaybetmiş: {', '.join(sorted(missing))}")

    if review.problems:
        return review

    # Paths are built here, never taken from the model. `discover -s tests` puts
    # that directory on sys.path, which is why the module lives beside its test.
    module_path = f"tests/{name}.py"
    test_path = f"tests/test_{name}.py"
    review.plan = ImprovementPlan(
        hypothesis_id=hypothesis_id,
        setup_files={module_path: baseline, test_path: test},
        changed_files={module_path: candidate},
        promote=[module_path],
        test_target="tests",
    )
    return review


class ExperimentPlanner:
    """Asks the model for an experiment and refuses most of what comes back."""

    def __init__(self, brain, *, model: str = "", events=None,
                 temperature: float = 0.2) -> None:
        self.brain = brain
        self.model = model
        self.events = events
        self.temperature = temperature
        self.last_review: PlanReview | None = None

    def emit(self, kind: str, message: str, level: str = "info", **data) -> None:
        if self.events is not None:
            self.events.publish("improve", kind, message, level=level, data=data)

    def __call__(self, hypothesis, gap=None) -> ImprovementPlan | None:
        """The planner protocol `engine.cycle` expects."""
        instruction = self._instruction(hypothesis, gap)
        try:
            raw = self.brain.local.chat(
                [{"role": "system", "content": PLANNER_SYSTEM},
                 {"role": "user", "content": instruction}],
                schema=PLAN_SCHEMA, temperature=self.temperature,
                model=self.model or None, purpose="deney-plani")
        except OSError as exc:
            log.warning("deney planı üretilemedi: %s", exc)
            self.emit("plan.failed", f"Plan üretilemedi: {exc}", level="warn")
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("plan üretici çözümlenemeyen cevap verdi")
            self.emit("plan.rejected", "Plan çözümlenemedi: geçersiz JSON", level="warn")
            self.last_review = PlanReview(problems=["geçersiz JSON"])
            return None
        if not isinstance(payload, dict):
            self.last_review = PlanReview(problems=["cevap bir nesne değil"])
            self.emit("plan.rejected", self.last_review.report(), level="warn")
            return None

        review = validate(payload, getattr(hypothesis, "id", ""))
        self.last_review = review
        if not review.ok:
            log.info("plan reddedildi: %s", review.problems)
            self.emit("plan.rejected", review.report(), level="warn",
                      problems=review.problems[:5])
            return None

        self.emit("plan.ready",
                  f"Deney planı hazır: {review.module_name} "
                  f"({len(review.plan.setup_files)} dosya)",
                  module=review.module_name)
        return review.plan

    def _instruction(self, hypothesis, gap) -> str:
        title = getattr(hypothesis, "title", "")
        statement = getattr(hypothesis, "statement", "")
        gap_title = getattr(gap, "title", "") if gap is not None else ""
        return (
            f"[Hipotez]\n{title}\n\n"
            f"[İddia]\n{statement}\n\n"
            f"[Tespit edilen eksik]\n{gap_title or 'belirtilmemiş'}\n\n"
            "Bu hipotezi küçük ölçekte sınayan bir deney tasarla. Baseline "
            "hipotezin iyileştirmeyi önerdiği düz yaklaşımı, aday ise hipotezin "
            "önerdiği yaklaşımı uygulasın. Testler ikisinde de geçmeli."
        )


__all__ = [
    "ExperimentPlanner", "PlanReview", "validate", "inspect_code", "inspect_test",
    "public_names", "PLAN_SCHEMA", "PLANNER_SYSTEM", "NAME_PATTERN",
    "ALLOWED_IMPORTS", "FORBIDDEN_NAMES", "ALLOWED_DUNDERS", "MAX_FILE_BYTES",
]
