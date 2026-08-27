"""The conversation loop and its shutdown."""

from __future__ import annotations

from ..assistant import TOOL_ROLE
from ..brain import CLOUD, LOCAL
from ..identity import PRODUCT_SLUG
from ..state import IDLE, LISTENING, THINKING
from .commands import handle_command
from .session import Session
from .theme import BANNER, BLUE, BOLD, CYAN, DIM, RESET, YELLOW, tier_badge


def wind_down(session: Session) -> None:
    """Close the session: summarise it and keep whatever was worth keeping.

    Runs on the local model, so it costs nothing and works offline. Kept quiet
    unless it actually learned something.
    """
    runtime = session.runtime
    if runtime.core is not None and runtime.core.scheduler.running:
        runtime.core.stop(timeout=15)

    if session.memory is None:
        print(f"\n{DIM}kapatılıyor{RESET}")
        return

    print(f"\n{DIM}oturum hafızaya işleniyor…{RESET}", end="", flush=True)
    try:
        result = session.memory.end_session()
    except Exception as exc:  # noqa: BLE001 - shutdown must not fail loudly
        print(f"\r{YELLOW}hafıza yazılamadı: {exc}{RESET}\n")
        return

    print("\r" + " " * 40 + "\r", end="")
    notes = result.get("notlar") or []
    if notes:
        print(f"{DIM}hafızaya eklendi: {', '.join(notes)}{RESET}")

    # Shown, not hidden: a refused fact is usually JARVIS's own unverified claim,
    # and knowing what it nearly believed is more useful than a clean log.
    refused = result.get("reddedilen") or []
    if refused:
        print(f"{DIM}{YELLOW}hafızaya alınmadı ({len(refused)}):{RESET}")
        for line in refused[:4]:
            print(f"{DIM}  · {line}{RESET}")

    summary = result.get("ozet")
    if summary:
        print(f"{DIM}{summary}{RESET}")
    print(f"{DIM}kapatılıyor{RESET}")


class ToolActivity:
    """Prints what a tool is doing, and passes the event on to the real log.

    The UI has to show ground truth, so this sits on the same events the
    assistant already publishes rather than inventing a second channel that
    could drift out of step with it.
    """

    #: ASCII on purpose. A Windows console runs on the ANSI codepage — cp1254
    #: on a Turkish system — and printing a check mark there raises
    #: UnicodeEncodeError, which would take down the tool display on the machine
    #: this is written for. The HTML interface is UTF-8 and uses real glyphs.
    SHOWN = {
        "tool.start": (">", DIM),
        "tool.done": ("+", DIM),
        "tool.failed": ("!", YELLOW),
        "tool.denied": ("!", YELLOW),
        "decision.rejected": ("-", DIM),
        "turn.stalled": ("!", YELLOW),
    }

    def __init__(self, downstream=None) -> None:
        self.downstream = downstream

    def publish(self, source, kind, message, level="info", data=None):
        mark = self.SHOWN.get(kind)
        if mark is not None:
            glyph, colour = mark
            print(f"  {colour}{glyph} {message}{RESET}", flush=True)
        if self.downstream is not None:
            try:
                self.downstream.publish(source, kind, message, level=level, data=data)
            except Exception:  # noqa: BLE001 - display must not depend on the log
                pass


def ask_permission(tool_name: str, risk: str, arguments: dict) -> bool:
    """Ask before anything above a read. Default is no."""
    detail = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in arguments.items())
    print(f"\n  {YELLOW}{tool_name}{RESET} {DIM}({risk}){RESET}  {detail}")
    try:
        reply = input(f"  {BOLD}onaylıyor musun?{RESET} [e/H] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return reply in ("e", "evet", "y", "yes")


def _run_with_tools(session: Session, line: str) -> bool:
    """Answer through the tool loop. False when the loop is unavailable."""
    assistant = session.assistant(
        approve=ask_permission, events=ToolActivity(session.runtime.events))
    if assistant is None:
        return False

    session.state.set_activity(THINKING, "araçlar")
    try:
        turn = assistant.run(line, history=session.history[:-1])
    except KeyboardInterrupt:
        print(f"\n{DIM}kesildi{RESET}\n")
        session.state.set_activity(IDLE)
        return True

    if turn.pending is not None:
        print(f"{DIM}onay verilmedi, işlem yapılmadı{RESET}\n")
    if turn.stopped:
        print(f"{YELLOW}{turn.stopped}{RESET}")

    reply = turn.reply or "(cevap üretilemedi)"
    print(f"{BOLD}{BLUE}{PRODUCT_SLUG}{RESET} › {reply}\n")
    # What ran, then what was said about it -- the same record and the same order
    # the interface writes. What JARVIS remembers must not depend on which door
    # the work came through.
    record = turn.tool_record()
    if record:
        session.add(TOOL_ROLE, record)
    if turn.reply:
        session.add("assistant", turn.reply)

    # Said plainly rather than left to the prose: the model may claim success
    # over a failed tool, and the recorded results are what actually happened.
    if turn.failures:
        names = ", ".join(step.tool for step in turn.failures)
        print(f"{DIM}{YELLOW}not: {len(turn.failures)} araç çağrısı başarısız "
              f"oldu ({names}){RESET}\n")
    session.state.set_activity(IDLE)
    return True


def run_repl(session: Session) -> int:
    brain = session.brain
    print(BANNER)
    status = brain.status()
    if not status["local_up"]:
        print(f"{YELLOW}Ollama çalışmıyor — yerel katman devre dışı.{RESET}")
    elif not status["local_model_present"]:
        print(f"{YELLOW}Model {status['local_model']} bulunamadı: "
              f"ollama pull {status['local_model']}{RESET}")
    if not status["cloud_up"]:
        print(f"{YELLOW}claude CLI bulunamadı — Claude katmanı devre dışı.{RESET}")
    for warning in session.runtime.warnings:
        print(f"{YELLOW}{warning}{RESET}")
    print()

    while True:
        session.state.set_activity(LISTENING, "girdi bekleniyor")
        try:
            line = input(f"{BOLD}{CYAN}sen{RESET} › ").strip()
        except (EOFError, KeyboardInterrupt):
            wind_down(session)
            return 0

        if not line:
            continue

        forced: str | None = None
        if line.startswith("/"):
            try:
                handled, message, forced = handle_command(line, session)
            except SystemExit:
                wind_down(session)
                return 0
            if handled and message is None:
                continue
            line = message or ""
            if not line:
                continue

        session.add("user", line)

        # Tools first: the loop decides for itself whether the request needs
        # one, and answers directly when it does not. `/claude` and friends set
        # `forced`, which is a request for a specific tier rather than for work.
        if forced is None and _run_with_tools(session, line):
            continue

        tier, reason = brain.plan(line, forced=forced)
        session.state.set_activity(THINKING, f"{tier} katmanı")
        print(f"{tier_badge(tier, brain.cloud.model if tier == CLOUD else brain.local.model)}"
              f" {DIM}{'düşünüyor…' if tier == CLOUD else ''}{RESET}", flush=True)

        print(f"{BOLD}{BLUE}{PRODUCT_SLUG}{RESET} › ", end="", flush=True)
        collected: list[str] = []
        try:
            for piece in brain.stream(session.history, forced=forced):
                collected.append(piece)
                print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print(f"\n{DIM}kesildi{RESET}\n")
            session.state.set_activity(IDLE)
            continue
        print("\n")

        answer = brain.last
        text = "".join(collected).strip()
        if text:
            session.add("assistant", text)
        if answer and answer.degraded:
            print(f"{DIM}{YELLOW}not: Claude katmanı uygun değildi ({answer.degraded_reason}), "
                  f"yerel model cevapladı{RESET}\n")
        elif answer and answer.tier == LOCAL and brain.last_decision \
                and brain.last_decision.near_miss:
            print(f"{DIM}daha derin bir cevap istersen: /claude <soru>{RESET}\n")
