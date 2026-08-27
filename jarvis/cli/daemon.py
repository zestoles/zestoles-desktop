"""Headless autonomous mode: the loop and its log, no conversation."""

from __future__ import annotations

import time

from .instance import InstanceLock
from .theme import BANNER, DIM, RESET, YELLOW, format_event


def run_daemon(runtime) -> int:
    """Foreground autonomous mode.

    This is what a scheduled task or a startup entry runs. Events stream to the
    terminal so an unattended run is still observable, and Ctrl+C shuts the loop
    down cleanly rather than abandoning a task mid-write.

    Only one of these may run at a time — see `instance.InstanceLock`.
    """
    core = runtime.core
    if core is None:
        print(f"{YELLOW}otonomi açılamadı{RESET}")
        return 1

    lock = InstanceLock(runtime.config.path("paths.daemon_lock", "data/daemon.lock"))
    if not lock.acquire():
        print(f"{YELLOW}otonom döngü zaten çalışıyor (PID {lock.holder}) — "
              f"ikinci bir kopya başlatılmadı{RESET}")
        return 1

    print(BANNER)
    print(f"{DIM}otonom mod · PID {lock.holder} · Ctrl+C ile durdurulur{RESET}\n")

    core.events.subscribe(lambda event: print(format_event(event), flush=True))
    core.start()

    if not core.scheduler.running:
        print(f"{YELLOW}döngü başlamadı — yapılandırmada otonomi kapalı olabilir{RESET}")
        lock.release()
        return 1

    try:
        while core.scheduler.running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"\n{DIM}durduruluyor…{RESET}")
    finally:
        stopped = core.stop(timeout=60)
        if not stopped:
            print(f"{YELLOW}bir görev zamanında bitmedi; bir sonraki başlangıçta "
                  f"kuyruğa geri konacak{RESET}")
        lock.release()
    return 0
