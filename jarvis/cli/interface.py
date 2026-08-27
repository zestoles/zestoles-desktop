"""The desktop interface: one window, one conversation, one clean exit.

Assembly only. The page is `ui/jarvis.html`, the answers come from
`AssistantService`, the live activity comes from the event bus — this file wires
those three together, opens a browser at them, and waits.

## Why the browser and not a window of our own

A real window needs a webview dependency, and this system is stdlib-only by
choice: no package index, no build step, nothing to break on a machine that has
not been prepared. The page is served from the same loopback socket the event
stream already uses, so there is one port, one origin and one thing to shut
down. Swapping the shell for a native window later changes this file and
nothing else.

## Shutting down

The point of V1 is that closing JARVIS closes JARVIS. So the exit path releases
the lock, stops the socket, stops the scheduler and processes the session into
memory — in that order, and in a `finally`, so a crash takes the same path a
Ctrl+C does.
"""

from __future__ import annotations

import logging
import json
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

from ..assistant.context import DEFAULT_BUDGET_CHARS
from ..assistant.service import AssistantService
from ..identity import PRODUCT_NAME
from .instance import InstanceLock
from .lifecycle import DEFAULT_GRACE_S, OrphanWatch
from .session import Session
from .theme import BOLD, CYAN, DIM, RESET, YELLOW

log = logging.getLogger("jarvis.cli.interface")


def run_interface(runtime, config, *, port: int | None = None,
                  open_browser: bool = True,
                  surekli: bool = False) -> int:
    """Serve the interface until interrupted. Returns a process exit code.

    `surekli`, 7/24 omurgasıdır: her şey aynıdır ama son sekme kapandığında
    süreç kendini kapatmaz -- V1'in "pencereyi kapatan JARVIS'i de kapatır"
    sözü yalnızca manuel mod için geçerlidir. Sürekli modda tek çıkış yolu
    arayüzdeki Kapat düğmesidir; o bilinçli bir eylemdir ve saygı duyulur.
    """
    from ..bus import build as build_bus

    page = config.path("paths.ui_assistant", "ui/jarvis.html")
    if not Path(page).is_file():
        print(f"{YELLOW}arayüz dosyası bulunamadı: {page}{RESET}")
        return 1

    session = Session(runtime, history_turns=config.get("chat.history_turns", 12))
    assistant = session.assistant()
    if assistant is None:
        print(f"{YELLOW}araç katmanı kurulamadı — arayüz açılmıyor{RESET}")
        for warning in runtime.warnings:
            print(f"{YELLOW}{warning}{RESET}")
        return 1

    lock = InstanceLock(config.path("paths.daemon_lock", "data/daemon.lock"))
    holding = lock.acquire()
    if holding:
        runtime.start_autonomy()
    else:
        print(f"{DIM}otonom döngü başka bir süreçte (PID {lock.holder}){RESET}")

    # `session.add` is what writes a turn through to memory. The terminal
    # already goes through it; wiring it here is what stops the interface
    # from holding a conversation and forgetting all of it.
    # Created before the service so the page can be handed a way to stop the
    # process: closing JARVIS is something the user asks for through the same
    # request channel as everything else.
    stop = threading.Event()
    service = build_service(runtime, assistant, remember=session.add,
                            queued=holding, shutdown=stop.set)

    bus, server = build_bus(runtime, port=port, ui_file=page,
                            request_handler=service.handle)
    if server is None and port is None:
        # A fixed port is convenient for bookmarks, but another local tool may
        # already own it (on this machine Roblox MCP owns 8787).  Manual mode
        # opens the actual URL itself, so an OS-selected loopback port is a
        # better fallback than refusing to start.  An explicit --port remains
        # strict because callers choosing it may depend on that exact address.
        log.warning("varsayılan arayüz portu meşgul — boş port seçiliyor")
        bus, server = build_bus(runtime, port=0, ui_file=page,
                                request_handler=service.handle)
    if server is None:
        print(f"{YELLOW}arayüz açılamadı — port meşgul olabilir{RESET}")
        if holding:
            lock.release()
        return 1

    # Isınma arka planda: ollama keep_alive süresi dolduysa ilk soru, modelin
    # diskten yüklenmesini bekler -- ölçüldü: 62 sn sessizlik, kullanıcı bu
    # sırada konuşmaya devam edip kilitteki "meşgul" ile karşılaşıyor. Yükleme
    # bedelini açılışa ödetiyoruz; sayfa açılırken model zaten hazırdır.
    threading.Thread(target=_warm_brain, args=(runtime,),
                     name="jarvis-isinma", daemon=True).start()

    host, bound = server.address
    url = f"http://{host}:{bound}/"
    control_file = _publish_control(config, server, url)
    print(f"{BOLD}{CYAN}{PRODUCT_NAME}{RESET} {DIM}·{RESET} {url}")
    _report_model(runtime)
    if surekli:
        print(f"{DIM}sürekli mod — sekmeler kapansa da yaşar; "
              f"kapatmak için arayüzdeki Kapat{RESET}\n")
    else:
        print(f"{DIM}kapatmak için {PRODUCT_NAME} penceresindeki Kapat düğmesi{RESET}\n")

    if open_browser:
        try:
            _open_window(url)
        except Exception as exc:  # noqa: BLE001 - no browser is not a failure
            log.info("tarayıcı açılamadı: %s", exc)
            print(f"{YELLOW}tarayıcı açılamadı, adresi elle açın: {url}{RESET}")

    # Sürekli modda terk izleyicisi hiç kurulmaz: kimse bakmasa da sistem
    # çalışmaya devam eder -- bu modun varlık sebebi tam olarak budur.
    watchdog = None if surekli else _watch_for_abandonment(server, stop, config)
    try:
        while not stop.is_set():
            stop.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{DIM}kapatılıyor…{RESET}")
        stop.set()
        if watchdog is not None:
            watchdog.join(timeout=8)
        _clear_control(control_file, server.token)
        _shut_down(runtime, server, lock if holding else None)
    return 0


def _open_window(url: str) -> bool:
    """Open a clean app window on Windows, falling back to the default browser.

    Edge's app mode is still the installed browser engine, so it adds no runtime
    dependency, but removes tabs/address chrome and makes ZESTOLES feel and
    behave like one desktop product.  No shell is involved and the URL is the
    already-bound loopback address.
    """
    candidates = [shutil.which("msedge")]
    for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.append(str(Path(base) / "Microsoft" / "Edge" /
                                  "Application" / "msedge.exe"))
    edge = next((item for item in candidates if item and Path(item).is_file()), None)
    if edge:
        subprocess.Popen(  # noqa: S603 - fixed executable, loopback URL
            [edge, f"--app={url}", "--start-maximized"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    return bool(webbrowser.open(url))


def _publish_control(config, server, url: str) -> Path:
    """Publish the per-run local control address for the Windows tray.

    The request token already protects every mutating endpoint.  The tray runs
    as the same Windows user and needs a way to request the exact same graceful
    shutdown as the HUD.  The file is runtime state, ignored by Git, replaced
    atomically, and removed only by the run that owns its token.
    """
    path = config.path("paths.control", "data/control.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"url": url, "token": server.token, "pid": os.getpid(),
               "started": time.time()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding="ascii")
    os.replace(tmp, path)
    return path


def _clear_control(path: Path, token: str) -> None:
    """Remove only our own control record; never erase a newer process's."""
    try:
        data = json.loads(path.read_text(encoding="ascii"))
        if data.get("token") == token:
            path.unlink(missing_ok=True)
    except (OSError, json.JSONDecodeError, AttributeError):
        pass


def _warm_brain(runtime) -> None:
    """Yerel modeli VRAM'e yükler; başarısızlık sessizdir, soğuk da çalışır."""
    local = getattr(getattr(runtime, "brain", None), "local", None)
    if local is None:
        return
    try:
        local.warm()
        log.info("yerel model ısıtıldı")
    except Exception as exc:  # noqa: BLE001 - ısınma zorunlu değil
        log.debug("model ısıtılamadı: %s", exc)


def _watch_for_abandonment(server, stop: threading.Event, config):
    """Close JARVIS once the last window has been gone long enough.

    The Kapat button covers the deliberate exit. This covers the ordinary one:
    people close the browser tab, which sends nothing, and without this the
    process would stay up as a daemon nobody asked for -- exactly what V1 is
    meant not to be.

    Returns the thread, or None when the watch is switched off.
    """
    grace = float(config.get("ui.orphan_grace_s", DEFAULT_GRACE_S))
    if grace <= 0:
        return None
    watch = OrphanWatch(grace_s=grace)

    def loop() -> None:
        while not stop.wait(5.0):
            try:
                clients = int(server.status().get("istemci", 0))
            except Exception as exc:  # noqa: BLE001 - a bad reading is not a reason to quit
                log.debug("istemci sayısı okunamadı: %s", exc)
                continue
            if watch.observe(clients, now=time.monotonic()):
                print(f"\n{DIM}pencere kapandı — {PRODUCT_NAME} kapanıyor{RESET}")
                stop.set()
                return

    thread = threading.Thread(target=loop, name="jarvis-kapanis", daemon=True)
    thread.start()
    return thread


def build_service(runtime, assistant, *, remember, queued: bool,
                  shutdown=None) -> AssistantService:
    """Wire the service, and open the queue door only when this process runs the loop.

    A queued request is run by the scheduler through this same service -- same
    lock, same conversation -- so the two have to be in one process. A second
    window does not hold the autonomy lock and its scheduler is not running, so
    work left there would wait for the process that does own the loop, and that
    process has no assistant to run it: three failures and a quarantined task.
    Saying the door is shut is the honest version of that.
    """
    core = runtime.core if queued else None
    config = getattr(runtime, "config", None)
    budget = DEFAULT_BUDGET_CHARS
    if config is not None:
        budget = int(config.get("chat.context_chars", DEFAULT_BUDGET_CHARS))
    service = AssistantService(
        assistant, remember=remember,
        queue=core.queue if core is not None else None,
        nudge=core.scheduler.nudge if core is not None else None,
        shutdown=shutdown, config=config, budget_chars=budget)
    service.voice = _build_voice(config, service, runtime=runtime)
    runtime.voice = service.voice
    service.telegram = _build_telegram(config, service)
    runtime.telegram = service.telegram
    service.reminders = getattr(runtime, "reminders", None)
    if core is not None:
        core.scheduler.assistant = service
    return service


def _build_telegram(config, service):
    """Build the optional bridge without starting network activity."""
    if config is None:
        return None
    try:
        from ..integrations import TelegramGateway
        secret = config.root / "data" / "secrets" / "telegram.json"
        return TelegramGateway(service, secret_file=secret)
    except Exception as exc:  # noqa: BLE001 - text UI remains usable
        log.warning("Telegram katmanı kurulamadı: %s", exc)
        return None


def _build_voice(config, service, runtime=None):
    """The speech channel, or None when this machine cannot do speech.

    Nothing here loads a model: `VoiceSystem` only looks at what is installed,
    and the engines load on first use. So a machine without the packages pays
    nothing for the check, and JARVIS opens exactly as fast as it did before
    voice existed -- which is the whole reason the layer is optional.

    `publish`, olay yolunu kanala açan tek köprüdür: akışkan cevabın kalan
    parçaları buradan tarayıcıya ulaşır. Otonomi kapalıysa köprü kurulmaz ve
    kanal tüm cevabı sentezleyip bekler -- eski, yavaş ama doğru davranış.
    """
    try:
        from ..voice import VoiceSystem
        from ..voice.backchannel import Backchannel
        from ..voice.channel import VoiceChannel
    except ImportError as exc:  # noqa: BLE001 - voice is optional by design
        log.info("ses katmanı yok: %s", exc)
        return None

    try:
        system = VoiceSystem(config)
        level = 1.0
        if config is not None:
            level = float(config.get("voice.backchannel", 1.0))
        return VoiceChannel(voice=system, service=service,
                            backchannel=Backchannel(level=level),
                            publish=_voice_publisher(runtime))
    except Exception as exc:  # noqa: BLE001 - never fatal
        log.warning("ses katmanı kurulamadı: %s", exc)
        return None


def _voice_publisher(runtime):
    """Ses parçalarını olay kaydına taşıyan kapalı bir işlev; yol yoksa None."""
    events = getattr(getattr(runtime, "core", None), "events", None)
    if events is None:
        return None

    def publish(data: dict) -> None:
        events.publish("ses", "parcalar", "ses akisi", data=data)

    return publish


def _report_model(runtime) -> None:
    """Say what is actually available rather than letting the page find out."""
    try:
        status = runtime.brain.status()
    except Exception as exc:  # noqa: BLE001 - a status call must not stop startup
        log.debug("durum okunamadı: %s", exc)
        return
    if not status.get("local_up"):
        print(f"{YELLOW}Ollama çalışmıyor — model yanıt vermeyecek.{RESET}")
    elif not status.get("local_model_present"):
        print(f"{YELLOW}Model bulunamadı: ollama pull {status.get('local_model')}{RESET}")
    else:
        print(f"{DIM}model: {status.get('local_model')}{RESET}")


def _shut_down(runtime, server, lock) -> None:
    """Give everything back, in the order that cannot leave a half-open system.

    Each step is guarded on its own: a failure in one must not skip the rest,
    because the one that matters most — releasing the lock — is last.
    """
    for label, action in (
        ("yayın", lambda: server.stop()),
        ("zamanlayıcı", lambda: runtime.shutdown()),
        ("hafıza", lambda: _close_memory(runtime)),
    ):
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail loudly
            log.warning("kapanışta hata (%s): %s", label, exc)
    if lock is not None:
        try:
            lock.release()
        except Exception as exc:  # noqa: BLE001
            log.warning("kilit bırakılamadı: %s", exc)


def _close_memory(runtime) -> None:
    if runtime.memory is None:
        return
    result = runtime.memory.end_session()
    notes = result.get("notlar") or []
    if notes:
        print(f"{DIM}hafızaya eklendi: {', '.join(notes)}{RESET}")


__all__ = ["run_interface"]
