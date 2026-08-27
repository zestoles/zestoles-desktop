"""Slash commands.

One function, one long chain of prefixes. Deliberately not a dispatch table: every
branch returns the same three-part answer and reading them in order is how you find
out what the terminal can do.

The return is `(handled, message_to_send, forced_tier)`. A command that only prints
returns `(True, None, None)`; `/yerel` and `/claude` hand their text back to the
REPL with a tier attached, which is how "say this, but to that model" works without
a second code path.
"""

from __future__ import annotations

import os

from ..autonomy import Priority
from ..brain import CLOUD, LOCAL
from .actions import run_orchestration
from .theme import DIM, HELP, RESET, YELLOW
from .views import (
    print_autonomy,
    print_events,
    print_memory,
    print_skills,
    print_status,
    print_tasks,
)


def handle_command(line: str, session) -> tuple[bool, str | None, str | None]:
    """Returns (handled, message_to_send, forced_tier)."""
    parts = line.split(maxsplit=1)
    command = parts[0].casefold()
    rest = parts[1].strip() if len(parts) > 1 else ""
    brain = session.brain

    if command in ("/yardim", "/help", "/?"):
        print(HELP)
        return True, None, None

    if command in ("/durum", "/status"):
        print_status(brain)
        return True, None, None

    if command in ("/temizle", "/clear"):
        session.clear()
        print(f"{DIM}geçmiş sıfırlandı{RESET}\n")
        return True, None, None

    if command in ("/kisilik", "/persona"):
        path = brain.config.path("paths.persona", "persona/core.md")
        print(f"{DIM}{path}{RESET}\n")
        return True, None, None

    if command in ("/hafiza", "/memory"):
        print_memory(session.memory, rest)
        return True, None, None

    if command in ("/otonom", "/autonomy"):
        core = session.core
        if core is None:
            print(f"{YELLOW}otonomi devre dışı{RESET}\n")
        elif rest in ("dur", "duraklat", "pause"):
            core.pause()
            print(f"{DIM}otonom çalışma duraklatıldı{RESET}\n")
        elif rest in ("devam", "resume"):
            core.resume()
            print(f"{DIM}otonom çalışma sürüyor{RESET}\n")
        elif rest in ("baslat", "start"):
            core.start()
            print(f"{DIM}otonom döngü başlatıldı{RESET}\n")
        elif rest in ("kapat", "stop"):
            print(f"{DIM}duruluyor…{RESET}")
            print(f"{DIM}{'durdu' if core.stop() else 'bir görev hâlâ çalışıyor'}{RESET}\n")
        else:
            print_autonomy(core)
        return True, None, None

    if command in ("/gorev", "/task"):
        core = session.core
        if core is None:
            print(f"{YELLOW}otonomi devre dışı{RESET}\n")
        elif rest:
            kind, _, title = rest.partition(" ")
            try:
                task_id = core.submit(kind, title.strip() or kind, priority=Priority.NORMAL)
            except ValueError as exc:
                print(f"{YELLOW}{exc}{RESET}\n")
            else:
                print(f"{DIM}#{task_id} kuyruğa alındı{RESET}\n")
        else:
            print_tasks(core)
        return True, None, None

    if command in ("/olaylar", "/events"):
        print_events(session.core)
        return True, None, None

    if command in ("/ajan", "/agent"):
        if rest:
            run_orchestration(session.agents, rest)
        else:
            print(f"{DIM}kullanım: /ajan <hedef>{RESET}\n")
        return True, None, None

    if command in ("/beceri", "/skills"):
        print_skills(session.agents)
        return True, None, None

    if command in ("/kasa", "/vault"):
        path = brain.config.path("paths.vault", "vault")
        print(f"{DIM}{path}{RESET}")
        if os.name == "nt":
            os.startfile(path)  # noqa: S606 - opening the user's own memory folder
        return True, None, None

    if command == "/mod":
        if rest in ("auto", "local", "cloud"):
            brain.router.mode = rest
            print(f"{DIM}yönlendirme modu: {rest}{RESET}\n")
        else:
            print(f"{DIM}mevcut mod: {brain.router.mode} · seçenekler: auto, local, cloud{RESET}\n")
        return True, None, None

    if command in ("/yerel", "/local"):
        return (True, rest, LOCAL) if rest else (True, None, None)

    if command == "/claude":
        return (True, rest, CLOUD) if rest else (True, None, None)

    if command in ("/cikis", "/exit", "/quit", "/q"):
        raise SystemExit(0)

    print(f"{YELLOW}bilinmeyen komut: {command}  ({DIM}/yardim{RESET}{YELLOW}){RESET}\n")
    return True, None, None
