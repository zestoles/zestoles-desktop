"""Terminal interface.

Was one 600-line module until S7A. Split because the websocket layer is about to
become a second consumer of the same subsystems, and a front end that grew inside
the CLI would have had to be untangled from it first.

    theme      colours, banner, help text — imported by everything, imports nothing back
    views      read-only renderings of subsystem state
    actions    long-running operations with live event output
    session    the conversation buffer
    commands   slash-command dispatch
    repl       the conversation loop
    daemon     headless autonomous mode
    app        argument parsing and dispatch

Assembly is not here. It moved to `jarvis.runtime`, which both this package and the
websocket layer use, so neither owns the startup order.
"""

from __future__ import annotations

from .actions import run_orchestration, run_research
from .app import main
from .commands import handle_command
from .daemon import run_daemon
from .repl import run_repl, wind_down
from .session import Session
from .theme import BANNER, HELP, enable_ansi, format_event, tier_badge
from .views import (
    print_autonomy,
    print_events,
    print_memory,
    print_runtime,
    print_skills,
    print_status,
    print_tasks,
)

__all__ = [
    "main", "Session", "run_repl", "wind_down", "run_daemon",
    "handle_command", "run_orchestration", "run_research",
    "print_status", "print_memory", "print_autonomy", "print_tasks",
    "print_events", "print_skills", "print_runtime",
    "format_event", "tier_badge", "enable_ansi", "BANNER", "HELP",
]
