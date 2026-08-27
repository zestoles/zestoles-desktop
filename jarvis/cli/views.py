"""Read-only renderings of subsystem state.

Every function here takes a subsystem (or None) and prints. None is a normal
argument, not an error: a JARVIS running without memory or without autonomy is a
degraded JARVIS, not a broken one, and the terminal says which part is absent
rather than crashing on it.
"""

from __future__ import annotations

from .theme import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW, format_event


def print_memory(memory, query: str) -> None:
    if memory is None:
        print(f"{YELLOW}hafıza devre dışı{RESET}\n")
        return

    if query:
        hits = memory.search(query, limit=5)
        if not hits:
            print(f"{DIM}'{query}' için hafızada bir şey yok{RESET}\n")
            return
        print(f"\n{BOLD}Hafızada '{query}'{RESET}")
        for hit in hits:
            print(f"\n  {CYAN}{hit.note_title}{RESET} {DIM}({hit.note_kind}){RESET}")
            for line in hit.text.strip().splitlines()[:4]:
                print(f"    {line}")
        print()
        return

    stats = memory.stats()
    print(f"\n{BOLD}Hafıza{RESET}")
    print(f"  {stats.get('notlar', 0)} not · {stats.get('parcalar', 0)} parça · "
          f"{stats.get('vektorlu', 0)} vektörlü")
    print(f"  {stats.get('oturumlar', 0)} oturum · {stats.get('mesajlar', 0)} mesaj")
    print(f"  {DIM}kasa: {stats.get('kasa')}{RESET}")
    print(f"  {DIM}embedding: {stats.get('embed_model')}{RESET}")
    recent = memory.recent_summaries(3)
    if recent:
        print(f"\n{BOLD}Son oturumlar{RESET}")
        for line in recent:
            print(f"  {DIM}{line}{RESET}")
    print()


def print_autonomy(core) -> None:
    if core is None:
        print(f"{YELLOW}otonomi devre dışı{RESET}\n")
        return
    status = core.status()
    snapshot = core.snapshot()
    running = status["calisiyor"]
    state = f"{GREEN}çalışıyor{RESET}" if running else f"{DIM}durdu{RESET}"
    if status["duraklatildi"]:
        state = f"{YELLOW}duraklatıldı{RESET}"

    print(f"\n{BOLD}Otonomi{RESET}")
    print(f"  döngü : {state}")
    print(f"  mod   : {status['mod']}  {DIM}({status['gerekce']}){RESET}")
    print(f"  makine: {DIM}{snapshot.summary()}{RESET}")
    if status["aktif_gorev"]:
        print(f"  aktif : {CYAN}{status['aktif_gorev']}{RESET}")
    counts = status["kuyruk"] or {}
    if counts:
        print("  kuyruk: " + " · ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    else:
        print(f"  kuyruk: {DIM}boş{RESET}")
    print(f"  {DIM}bu oturumda {status['tamamlanan']} tamamlandı, "
          f"{status['basarisiz']} başarısız{RESET}\n")


def print_tasks(core) -> None:
    if core is None:
        print(f"{YELLOW}otonomi devre dışı{RESET}\n")
        return
    tasks = core.queue.list(limit=15)
    if not tasks:
        print(f"{DIM}kuyruk boş{RESET}\n")
        return
    print(f"\n{BOLD}Görevler{RESET}")
    for task in tasks:
        colour = {"running": CYAN, "done": GREEN, "quarantined": RED,
                  "failed": YELLOW}.get(task.state, DIM)
        print(f"  {DIM}#{task.id:<4}{RESET} {colour}{task.state:<12}{RESET} "
              f"{task.title}  {DIM}({task.kind}, p{task.priority}){RESET}")
        if task.error and task.state in ("quarantined", "failed"):
            print(f"        {RED}{task.error[:120]}{RESET}")
    print()


def print_events(core, limit: int = 20) -> None:
    if core is None:
        print(f"{YELLOW}otonomi devre dışı{RESET}\n")
        return
    events = core.events.since(86400, limit=limit)
    if not events:
        print(f"{DIM}kayıtlı olay yok{RESET}\n")
        return
    print(f"\n{BOLD}Son olaylar{RESET}")
    for event in reversed(events):
        print("  " + format_event(event))
    print()


def print_skills(agents) -> None:
    if agents is None:
        print(f"{YELLOW}ajan sistemi açık değil{RESET}\n")
        return
    skills = agents.skills.list()
    if not skills:
        print(f"{DIM}henüz öğrenilmiş beceri yok — doğrulanmış çok adımlı bir koşu "
              f"sonrası kaydedilir{RESET}\n")
        return
    print(f"\n{BOLD}Beceriler{RESET}")
    for skill in skills:
        mark = f"{DIM}(emekli){RESET} " if skill.retired else ""
        print(f"  {mark}{CYAN}{skill.name}{RESET} — {skill.summary()}")
    print()


def print_status(brain) -> None:
    status = brain.status()
    usage = status["usage"]
    verdict = status["budget_verdict"]

    def mark(ok: bool) -> str:
        return f"{GREEN}●{RESET}" if ok else f"{RED}●{RESET}"

    print(f"\n{BOLD}Sistem{RESET}")
    print(f"  {mark(bool(status['local_up']))} yerel model    {status['local_model']}"
          f"{'' if status['local_model_present'] else f'  {YELLOW}(model indirilmemiş){RESET}'}")
    if status.get("cloud_enabled", True):
        print(f"  {mark(bool(status['cloud_up']))} bulut katmanı  {status['cloud_model']}")
    else:
        print(f"  {DIM}○ bulut katmanı  kapalı — tamamen ücretsiz yerel mod{RESET}")
    print(f"  {DIM}yönlendirme modu: {status['router_mode']}{RESET}")

    print(f"\n{BOLD}Kota{RESET}")
    print(f"  son 1 saat : {usage['cloud_hour']}/{usage['limit_hour']} claude çağrısı")
    print(f"  son 24 saat: {usage['cloud_day']}/{usage['limit_day']} claude · "
          f"{usage['local_day']} yerel · {usage['cloud_day_tokens']:,} token")
    print(f"  gece limiti: {usage['limit_night']} çağrı")
    print(f"  {DIM}{verdict.reason}{RESET}")

    last = brain.last
    if last:
        print(f"\n{BOLD}Son yanıt{RESET}")
        print(f"  katman: {last.tier} ({last.model}) · {last.duration_ms} ms")
        print(f"  {DIM}gerekçe: {last.reason}{RESET}")
        if last.degraded:
            print(f"  {YELLOW}not: Claude katmanı istenmişti ama kullanılamadı{RESET}")
    print()


def print_runtime(runtime) -> None:
    """The composed view: one call, everything that is up and what is not."""
    print_status(runtime.brain)
    print_memory(runtime.memory, "")
    print_autonomy(runtime.core)
    if runtime.warnings:
        print(f"{BOLD}Uyarılar{RESET}")
        for warning in runtime.warnings:
            print(f"  {YELLOW}· {warning}{RESET}")
        print()
