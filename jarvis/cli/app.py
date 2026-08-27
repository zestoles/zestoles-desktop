"""Argument parsing and dispatch.

Thin on purpose. Assembly lives in `jarvis.runtime`, rendering in `views`, work in
`actions`; this file decides which of them the arguments asked for and gets out of
the way.
"""

from __future__ import annotations

import argparse
import sys

from ..config import Config
from ..identity import PRODUCT_NAME, PRODUCT_SLUG
from ..logging_setup import setup as setup_logging
from ..runtime import build
from .actions import run_orchestration, run_research
from .daemon import run_daemon
from .instance import InstanceLock
from .interface import run_interface
from .repl import run_repl
from .session import Session
from .theme import DIM, RESET, YELLOW, enable_ansi
from .views import (
    print_autonomy,
    print_events,
    print_memory,
    print_runtime,
    print_skills,
    print_tasks,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PRODUCT_SLUG,
                                     description=f"{PRODUCT_NAME} çekirdeği")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="konsola hata ayıklama günlüğü")
    parser.add_argument("--durum", action="store_true", help="durumu yazdır ve çık")
    parser.add_argument("-m", "--mesaj", help="tek mesaj sor, cevabı yazdır ve çık")
    parser.add_argument("--katman", choices=["local", "cloud"], help="katmanı zorla")
    parser.add_argument("--hafiza", nargs="?", const="", help="hafızayı ara veya durumunu bas")
    parser.add_argument("--yenile", action="store_true", help="hafıza indeksini yeniden kur")
    parser.add_argument("--unutkan", action="store_true", help="bu oturumu hafızaya yazma")
    parser.add_argument("--otonom", action="store_true",
                        help="sohbet olmadan otonom döngüyü çalıştır")
    parser.add_argument("--gorevler", action="store_true", help="görev kuyruğunu bas ve çık")
    parser.add_argument("--olaylar", action="store_true", help="son olayları bas ve çık")
    parser.add_argument("--ajan", help="ajan ekibini bu hedefe koş ve çık")
    parser.add_argument("--beceriler", action="store_true", help="öğrenilmiş becerileri bas")
    parser.add_argument("--arastir", help="web'de araştır, çapraz doğrula ve çık")
    parser.add_argument("--arayuz", action="store_true",
                        help="masaüstü arayüzünü aç (V1 varsayılanı)")
    parser.add_argument("--surekli", action="store_true",
                        help="7/24 omurgası: arayüzü sunar ama sekme kapansa da "
                             "yaşar — kapatmak için arayüzdeki Kapat")
    parser.add_argument("--yayin", action="store_true",
                        help="canlı olay yayınını aç (websocket)")
    parser.add_argument("--port", type=int, help="yayın portu (varsayılan 8797)")
    return parser


def _start_bus(runtime, port: int | None):
    """Open the live stream. Failure costs visibility, not function."""
    from ..bus import build as build_bus

    bus, server = build_bus(runtime, port=port)
    if server is None:
        print(f"{YELLOW}yayın açılamadı — port meşgul olabilir{RESET}", file=sys.stderr)
    else:
        print(f"{DIM}canlı yayın: ws://{server.address[0]}:{server.address[1]}{RESET}")
    return bus, server


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    enable_ansi()
    setup_logging(verbose=args.verbose)
    config = Config.load()

    runtime = build(config, with_memory=not args.unutkan)
    for warning in runtime.warnings:
        print(warning, file=sys.stderr)

    if args.arastir:
        run_research(runtime.research, args.arastir)
        return 0

    if args.ajan:
        run_orchestration(runtime.agents, args.ajan)
        return 0

    if args.beceriler:
        print_skills(runtime.agents)
        return 0

    if args.otonom:
        server = _start_bus(runtime, args.port)[1] if args.yayin else None
        try:
            return run_daemon(runtime)
        finally:
            if server is not None:
                server.stop()

    if args.gorevler:
        print_tasks(runtime.core)
        return 0

    if args.olaylar:
        print_events(runtime.core, limit=40)
        return 0

    if args.yenile:
        if runtime.memory is None:
            print("hafıza devre dışı", file=sys.stderr)
            return 1
        print(f"{DIM}indeks kuruluyor…{RESET}")
        report = runtime.memory.reindex(force=True)
        print(" · ".join(f"{k}: {v}" for k, v in report.items()) or "değişiklik yok")
        return 0

    if args.hafiza is not None:
        print_memory(runtime.memory, args.hafiza)
        return 0

    if args.durum:
        print_runtime(runtime)
        return 0

    if args.mesaj:
        answer = runtime.brain.ask([{"role": "user", "content": args.mesaj}],
                                   forced=args.katman)
        if answer.error:
            print(f"hata: {answer.error}", file=sys.stderr)
            return 1
        print(answer.text)
        return 0

    if runtime.memory is not None:
        runtime.memory.start_session()
        runtime.memory.reindex()

    if args.arayuz or args.surekli:
        return run_interface(runtime, config, port=args.port,
                             surekli=args.surekli)

    # Autonomy runs alongside the conversation. The policy keeps it out of the way:
    # while the user is typing the machine reads as ACTIVE, so only requested work
    # directly is admitted and routine upkeep waits for him to step away.
    #
    # Unless a headless loop already has the lock. Since autostart there is always
    # one, and two schedulers would spend one nightly budget twice. The
    # conversation still queues work — the running daemon picks it up from the
    # shared queue within a tick.
    lock = InstanceLock(config.path("paths.daemon_lock", "data/daemon.lock"))
    if lock.acquire():
        runtime.start_autonomy()
    else:
        print(f"{DIM}otonom döngü başka bir süreçte çalışıyor (PID {lock.holder}) — "
              f"görevler oraya kuyruklanır{RESET}")
    server = _start_bus(runtime, args.port)[1] if args.yayin else None

    session = Session(runtime, history_turns=config.get("chat.history_turns", 12))
    try:
        return run_repl(session)
    finally:
        if server is not None:
            server.stop()
        lock.release()
