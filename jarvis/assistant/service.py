"""What the interface talks to. One request in, one answer out.

`Assistant.run` is synchronous and knows nothing about clients. This wraps it
for something that does: a page that sends a message, waits for a reply, and may
have to answer a question in the middle of the turn.

## Confirmation, without a deadlock

The loop asks `approve(...)` before anything above a read, and the answer lives
in a browser. Blocking a turn on a callback that needs a second HTTP request to
resolve would hold the only lock the service has while waiting for the thing
that needs it — so the loop is given no callback at all. It stops instead, hands
back the pending call, and this service keeps it. The page shows what is being
asked; a later `onay` runs it and the conversation continues from the result.

That is why `Assistant` returns `pending` rather than raising: the caller that
has to ask is the one holding the connection.

## One turn at a time

A second request arriving mid-turn is told the assistant is busy rather than
queued. Two turns sharing one history and one workspace would interleave tool
calls, and the honest answer to "do two things at once" here is no.

`iptal` is deliberately answered before that lock is taken. Cancelling is the one
thing that must work *while* a turn is running, and a cancel that waited its turn
would arrive after the work it meant to stop.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .. import settings
from ..autonomy.tasks import State
from . import TOOL_ROLE, Step, Turn
from .background import KIND, enqueue
from .context import DEFAULT_BUDGET_CHARS, prune

log = logging.getLogger("jarvis.assistant.service")

ASK = "sor"
CONFIRM = "onay"
CANCEL = "iptal"
STATUS = "durum"
QUEUE = "kuyruk"
TASKS = "gorevler"
TASK_CANCEL = "gorev_iptal"
SHUTDOWN = "kapat"
SETTINGS = "ayarlar"
SETTING_SAVE = "ayar_kaydet"
VOICE_STATUS = "ses_durum"
VOICE_HEAR = "dinle"
VOICE_SPEAK = "seslendir"
VOICE_ACK = "onay_sesi"
VOICE_PREPARE = "ses_hazirla"
TELEGRAM_STATUS = "telegram_durum"
TELEGRAM_CONFIGURE = "telegram_ayarla"
TELEGRAM_START = "telegram_baslat"
TELEGRAM_STOP = "telegram_durdur"
TELEGRAM_PAIR = "telegram_eslestir"
TELEGRAM_DISCONNECT = "telegram_baglantiyi_kes"
REMINDERS = "hatirlaticilar"

#: Roles the model is given back. Everything else is record, not dialogue.
CONVERSATION_ROLES = frozenset({"user", "assistant"})

IDLE = "hazir"
WORKING = "calisiyor"
WAITING = "onay_bekliyor"
CLOSING = "kapaniyor"


@dataclass(slots=True)
class PendingCall:
    step: Step
    request: str
    history: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arac": self.step.tool,
            "risk": self.step.result.risk,
            "argumanlar": {k: str(v)[:400] for k, v in self.step.arguments.items()},
            "aciklama": self.step.result.error,
        }


def turn_as_dict(turn: Turn) -> dict[str, Any]:
    """A turn as the interface should read it.

    `basarili` comes from the recorded results, never from the reply, which is
    the same rule the terminal follows: a model may write "done" over a failure
    and the interface must not repeat it.
    """
    return {
        "cevap": turn.reply,
        "basarili": turn.succeeded,
        "iptal": turn.cancelled,
        "durduruldu": turn.stopped,
        "adimlar": [
            {"arac": step.tool, "ok": step.ok,
             "ozet": (step.result.output if step.ok else step.result.error)[:600]}
            for step in turn.steps
        ],
        "basarisiz": [step.tool for step in turn.failures],
    }


def task_as_dict(task) -> dict[str, Any]:
    """A queued request as the page should read it.

    `sonuc` is whatever the runner recorded, which is built from the real steps
    -- the same rule `turn_as_dict` follows, one layer further out.
    """
    return {
        "id": task.id,
        "baslik": task.title,
        "durum": task.state,
        "sonuc": task.result or "",
        "hata": task.error or "",
        "deneme": task.attempts,
        "olusturuldu": task.created,
    }


class AssistantService:
    def __init__(self, assistant, *, history_limit: int = 24,
                 remember=None, queue=None, nudge=None, shutdown=None,
                 config=None, budget_chars: int = DEFAULT_BUDGET_CHARS,
                 voice=None, telegram=None, reminders=None) -> None:
        self.assistant = assistant
        #: The task queue, when this process has one. Present so a
        #: request can be left for later instead of held on a socket.
        self.queue = queue
        #: Wakes the scheduler. Without it a queued request waits out a
        #: tick doing nothing, which reads as "JARVIS ignored me".
        self.nudge = nudge
        #: The voice channel, when this process has one. None means JARVIS
        #: works in text, which it always must be able to do.
        self.voice = voice
        #: Remote transport. It is attached after construction because it
        #: routes messages back through this same service.
        self.telegram = telegram
        self.reminders = reminders
        #: The live configuration, when this front end may edit it. Only the
        #: keys in `jarvis.settings.EDITABLE` are reachable through it.
        self.config = config
        #: Stops the process. None in a front end that does not own the
        #: lifetime -- the terminal REPL, or a second window.
        self.shutdown = shutdown
        #: Called with (role, content) for every turn that happened. The rolling
        #: window below is what the model sees next; this is what survives the
        #: session. Without it the interface would hold a conversation and forget
        #: all of it — the terminal writes through `Session.add`, and a second
        #: front end that quietly did not would be the kind of difference nobody
        #: notices until the next session comes up blank.
        self.remember = remember
        self.history: list[dict[str, str]] = []
        self.history_limit = max(2, history_limit)
        #: Characters of conversation the model is given. The message count
        #: above is a coarse cap; this is the one that matters, because one
        #: pasted file is worth a hundred ordinary messages.
        self.budget_chars = max(1000, int(budget_chars))
        self.pending: PendingCall | None = None
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        #: Bu turun son sınırı. Bir dil modeli çağrısının ortası kesilemez ama
        #: adımlar arasında bakılır: takılan bir döngü, dakikalarca sessiz
        #: kalmak yerine dürüstçe "duramadım" diye biter.
        self._deadline: float | None = None
        # The loop asks this between steps; `iptal` sets the event, the
        # deadline trips on a stuck loop.
        self.assistant.should_stop = self._should_stop
        # Deliberately none: see the module note on why the loop must not block
        # on an answer that arrives over a different connection.
        self.assistant.approve = None

    # ------------------------------------------------------------- dispatch
    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        op = str(payload.get("op", "")).strip().lower()

        # Before the lock, on purpose.
        if op == CANCEL:
            self._cancel.set()
            return {"durum": IDLE, "iptal": True}
        if op == STATUS:
            return self.status()

        # Also before the lock. Leaving work for later is most useful exactly
        # when a turn is already running, and reading the queue must not have to
        # wait for the turn it is being consulted about.
        if op == QUEUE:
            return self._queue_work(str(payload.get("mesaj", "")))
        if op == TASKS:
            return self._tasks()
        if op == TASK_CANCEL:
            return self._cancel_task(payload.get("id"))
        # Before the lock, for the same reason cancel is: an instruction to stop
        # that queues behind the work it means to stop arrives too late to be one.
        if op == SHUTDOWN:
            return self._shutdown()
        # Settings are read and written outside the turn lock: looking at them
        # while JARVIS is thinking is reasonable, and neither path touches the
        # conversation.
        if op == SETTINGS:
            return self._settings()
        if op == SETTING_SAVE:
            return self._save_setting(payload.get("anahtar"), payload.get("deger"))

        # Telegram lifecycle never takes the conversation lock. Messages coming
        # from Telegram do; configuration and pairing do not.
        if op == TELEGRAM_STATUS:
            return self._telegram("status")
        if op == TELEGRAM_CONFIGURE:
            return self._telegram("configure", str(payload.get("anahtar", "")))
        if op == TELEGRAM_START:
            return self._telegram("start")
        if op == TELEGRAM_STOP:
            return self._telegram("stop")
        if op == TELEGRAM_PAIR:
            return self._telegram("new_pair_code")
        if op == TELEGRAM_DISCONNECT:
            return self._telegram("disconnect")
        if op == REMINDERS:
            if self.reminders is None:
                return {"hata": "hatırlatma servisi kapalı"}
            return {"hatirlaticilar": self.reminders.list(),
                    "durum": self.reminders.status()}

        # Voice. Status, acknowledgements and preparation stay outside the turn
        # lock -- they answer while a turn is running, which is exactly when the
        # page needs them. `dinle` does take the lock, because it *is* a turn.
        if op == VOICE_STATUS:
            return self._voice_status()
        if op == VOICE_ACK:
            return self._voice_ack(payload)
        if op == VOICE_PREPARE:
            return self._voice_call("prepare")
        if op == VOICE_SPEAK:
            return self._voice_call("speak", str(payload.get("metin", "")))
        if op == VOICE_HEAR:
            return self._voice_hear(payload)

        if not self._lock.acquire(blocking=False):
            return {"hata": "meşgul", "durum": WORKING}
        try:
            if op == ASK:
                return self._ask(str(payload.get("mesaj", "")),
                                 sentence_sink=payload.get("cumle_akin"))
            if op == CONFIRM:
                return self._confirm(bool(payload.get("evet")))
            return {"hata": f"bilinmeyen işlem: {op[:40]}"}
        finally:
            self._lock.release()

    def _should_stop(self) -> bool:
        if self._cancel.is_set():
            return True
        deadline = self._deadline
        return bool(deadline is not None and time.monotonic() > deadline)

    def _turn_timeout_s(self, config) -> float:
        if config is None:
            return 60.0
        try:
            return max(5.0, float(config.get("assistant.turn_timeout_s", 60)))
        except (TypeError, ValueError):
            return 60.0

    def status(self) -> dict[str, Any]:
        if self.pending is not None:
            return {"durum": WAITING, "bekleyen": self.pending.as_dict()}
        return {"durum": WORKING if self._lock.locked() else IDLE,
                "gecmis": len(self.history)}

    # ---------------------------------------------------------------- work
    def _ask(self, message: str, sentence_sink=None) -> dict[str, Any]:
        text = message.strip()
        if not text:
            return {"hata": "boş mesaj"}
        if self.pending is not None:
            return {"hata": "önce bekleyen onayı cevaplayın", "durum": WAITING,
                    "bekleyen": self.pending.as_dict()}

        self._cancel.clear()
        self._deadline = time.monotonic() + self._turn_timeout_s(self.config)
        history = self._context()
        try:
            turn = self.assistant.run(text, history=history,
                                      on_sentence=sentence_sink)
        finally:
            self._deadline = None
        self._remember("user", text)

        self._record_tools(turn)

        if turn.pending is not None:
            self.pending = PendingCall(turn.pending, text, history)
            return {"durum": WAITING, "bekleyen": self.pending.as_dict(),
                    **turn_as_dict(turn)}

        if turn.reply:
            self._remember("assistant", turn.reply)
        return {"durum": IDLE, **turn_as_dict(turn)}

    def _confirm(self, approved: bool) -> dict[str, Any]:
        waiting, self.pending = self.pending, None
        if waiting is None:
            return {"hata": "bekleyen bir onay yok", "durum": IDLE}

        if not approved:
            self._remember("user", waiting.request)
            reply = f"{waiting.step.tool} onaylanmadı, işlem yapılmadı."
            self._remember("assistant", reply)
            return {"durum": IDLE, "cevap": reply, "basarili": False,
                    "adimlar": [], "basarisiz": [waiting.step.tool],
                    "reddedildi": True}

        self._cancel.clear()
        done = self.assistant.confirm(waiting.step)

        # The tool ran; the model has not seen its result yet. Continue the
        # conversation from that observation so the answer describes what
        # actually happened rather than what was about to.
        follow_up = list(waiting.history)
        follow_up.append({"role": "user", "content": waiting.request})
        follow_up.append({"role": "assistant",
                          "content": f"[{done.tool}] çalıştırıldı"})
        turn = self.assistant.run(f"[gözlem] {done.as_observation()}\n\n"
                                  "Bu sonuca göre kullanıcıya kısa bir cevap ver.",
                                  history=follow_up)
        turn.steps.insert(0, done)

        self._remember("user", waiting.request)
        self._record_tools(turn)
        if turn.reply:
            self._remember("assistant", turn.reply)
        return {"durum": IDLE, **turn_as_dict(turn)}

    # -------------------------------------------------------------- voice
    def _voice_status(self) -> dict[str, Any]:
        if self.voice is None:
            return {"kullanilabilir": False, "sebep": "ses katmani kapali"}
        try:
            return self.voice.status()
        except Exception as exc:  # noqa: BLE001 - voice must not break a reply
            log.warning("ses durumu alinamadi: %s", exc)
            return {"kullanilabilir": False, "sebep": str(exc)}

    def _voice_hear(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A spoken turn. Takes the same lock a typed one does.

        Speech is an input method, not a permission: this goes through the same
        `Assistant`, the same risk tiers and the same confirmation card. A
        MEDIUM tool asked for by voice still stops and waits for a person.
        """
        if self.voice is None:
            return {"hata": "ses katmani kapali"}
        try:
            return self.voice.listen(str(payload.get("ses", "")),
                                     spoken_seconds=float(payload.get("saniye", 0) or 0),
                                     client_turn=str(payload.get("tur", "")))
        except Exception as exc:  # noqa: BLE001
            log.warning("sesli tur basarisiz: %s", exc)
            return {"hata": f"ses islenemedi: {exc}"}

    def _voice_ack(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.voice is None:
            return {"sessiz": True, "sebep": "ses katmani kapali"}
        try:
            return self.voice.acknowledge(
                speech_seconds=float(payload.get("saniye", 0) or 0),
                confidence=float(payload.get("guven", 1) or 1),
                partial=str(payload.get("metin", "")),
                thinking=bool(payload.get("dusunme")))
        except Exception as exc:  # noqa: BLE001
            log.debug("onay sesi uretilemedi: %s", exc)
            return {"sessiz": True, "sebep": str(exc)}

    def _voice_call(self, method: str, *args) -> dict[str, Any]:
        if self.voice is None:
            return {"hata": "ses katmani kapali"}
        try:
            return getattr(self.voice, method)(*args)
        except Exception as exc:  # noqa: BLE001
            log.warning("ses cagrisi basarisiz (%s): %s", method, exc)
            return {"hata": str(exc)}

    def _telegram(self, method: str, *args) -> dict[str, Any]:
        if self.telegram is None:
            return {"hata": "Telegram katmanı kapalı"}
        try:
            return getattr(self.telegram, method)(*args)
        except Exception as exc:  # noqa: BLE001 - integration is optional
            log.warning("Telegram işlemi başarısız (%s): %s", method, exc)
            return {"hata": str(exc)}

    # ----------------------------------------------------------- settings
    def _settings(self) -> dict[str, Any]:
        if self.config is None:
            return {"hata": "ayarlar bu surecte acik degil"}
        return {"ayarlar": settings.read_all(self.config),
                "bilgi": settings.describe(self.config)}

    def _save_setting(self, key: Any, value: Any) -> dict[str, Any]:
        """Change one setting, or say why it was refused.

        The allow-list lives in `jarvis.settings`; anything not on it is not a
        setting as far as this channel is concerned. That check is deliberately
        not repeated here -- one gate, in one place, with its own tests.
        """
        if self.config is None:
            return {"hata": "ayarlar bu surecte acik degil"}
        ok, why = settings.write_one(self.config, str(key or ""), value)
        if not ok:
            return {"hata": why or "ayar kaydedilemedi"}
        applied = self._apply_live(str(key))
        return {"kaydedildi": True, "anahtar": key,
                "ayarlar": settings.read_all(self.config),
                "etkin": applied}

    def _apply_live(self, key: str) -> bool:
        """Make a saved setting take effect now where that is possible.

        Two of them are read once at startup by something this service owns, so
        they can be updated in place. The rest are read where they are used and
        need nothing; the page is told which is which rather than implying that
        every change is instant.
        """
        if key == "chat.history_turns":
            self.history_limit = max(2, int(self.config.get(key, 12)) * 2)
            if len(self.history) > self.history_limit:
                self.history = self.history[-self.history_limit:]
            return True
        if key == "assistant.max_steps":
            self.assistant.max_steps = max(1, int(self.config.get(key, 8)))
            return True
        return False

    # ----------------------------------------------------------- shutdown
    def _shutdown(self) -> dict[str, Any]:
        """Close JARVIS, after answering the request that asked for it.

        The reply travels over the socket this is about to close, so the order
        matters: hand back an answer, then stop. A running turn is cancelled
        first rather than left to finish into a process that is going away --
        and cancelling is honest, because `Turn.stopped` means the work did not
        complete and nothing will report that it did.
        """
        if self.shutdown is None:
            return {"hata": "kapatma bu surecte desteklenmiyor"}
        self._cancel.set()
        # Long enough for the answer to reach the page, short enough that the
        # user does not wonder whether the button worked.
        threading.Timer(0.4, self._stop_now).start()
        return {"durum": CLOSING, "kapaniyor": True}

    def _stop_now(self) -> None:
        try:
            self.shutdown()
        except Exception as exc:  # noqa: BLE001 - nothing left to fail into
            log.warning("kapatma başarısız: %s", exc)

    # -------------------------------------------------------------- queue
    def _queue_work(self, message: str) -> dict[str, Any]:
        if self.queue is None:
            return {"hata": "gorev kuyrugu bu surecte acik degil"}
        try:
            task_id = enqueue(self.queue, message)
        except ValueError as exc:
            return {"hata": str(exc)}
        if task_id is None:
            return {"hata": "ayni is zaten kuyrukta"}
        if self.nudge is not None:
            try:
                self.nudge()
            except Exception as exc:  # noqa: BLE001 - a sleepy scheduler is not a failure
                log.debug("zamanlayici uyandirilamadi: %s", exc)
        return {**self.status(), "gorev": task_id}

    def _tasks(self, limit: int = 12) -> dict[str, Any]:
        """The user's own queued work, newest first.

        Filtered to `assistant.ask` on purpose: the queue also carries routines
        and improvement work, and answering "what did I ask for" with the
        machine's own housekeeping is how a status list stops being read. The
        developer panel already shows everything.
        """
        if self.queue is None:
            return {"hata": "gorev kuyrugu bu surecte acik degil"}
        mine = [task for task in self.queue.list(limit=max(limit * 5, 60))
                if task.kind == KIND][:limit]
        return {**self.status(), "gorevler": [task_as_dict(task) for task in mine]}

    def _cancel_task(self, raw_id: Any) -> dict[str, Any]:
        if self.queue is None:
            return {"hata": "gorev kuyrugu bu surecte acik degil"}
        try:
            task_id = int(raw_id)
        except (TypeError, ValueError):
            return {"hata": "gorev numarasi gecersiz"}
        task = self.queue.get(task_id)
        if task is None:
            return {"hata": "boyle bir gorev yok"}
        if not self.queue.cancel(task_id):
            return {"hata": f"gorev zaten bitmis ({task.state})"}
        # Only when it was actually running. A queued item the user changed
        # their mind about is not a reason to stop the conversation they are
        # having right now.
        if task.state == State.RUNNING:
            self._cancel.set()
        return {**self.status(), "gorev": task_id, "iptal": True}

    # ------------------------------------------------------------- queued
    def run_queued(self, message: str, *, should_stop=None) -> Turn:
        """Run one turn on behalf of a queued task.

        Raises rather than returning a failed turn when it cannot start at all,
        because the caller is the task queue: an exception is a retry with
        backoff, and "the user was mid-conversation" is exactly the kind of
        failure that should be tried again rather than recorded as an answer.

        The live front door and this one share `_lock`, so the "one turn at a
        time" rule holds across both. They also share the conversation: a
        background turn the user asked for is still their conversation, and a
        second history would let JARVIS answer the same question twice without
        noticing.
        """
        text = message.strip()
        if not text:
            raise ValueError("bos gorev mesaji")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("asistan mesgul - gorev sonra tekrar denenecek")
        try:
            if self.pending is not None:
                raise RuntimeError("bekleyen onay var - gorev sonra tekrar denenecek")

            self._cancel.clear()
            self._deadline = time.monotonic() + self._turn_timeout_s(self.config)
            # Two ways to stop: the user pressing cancel, and the scheduler
            # shutting down. A queued turn must answer both.
            self.assistant.should_stop = (
                self._should_stop if should_stop is None
                else lambda: self._should_stop() or bool(should_stop()))
            try:
                turn = self.assistant.run(text, history=self._context())
            finally:
                self._deadline = None
                self.assistant.should_stop = self._should_stop

            self._remember("user", text)
            self._record_tools(turn)
            if turn.pending is None and turn.reply:
                self._remember("assistant", turn.reply)
            # Deliberately not stored as `self.pending`: nobody is holding a
            # connection to answer it, and taking the live confirmation slot
            # would leave the page asking about work it never requested.
            return turn
        finally:
            self._lock.release()

    def _record_tools(self, turn: Turn) -> None:
        """Write what actually ran, before whatever was said about it.

        Order matters on recall: the measurement, then the prose. A session that
        kept only the prose kept the one part of the turn that can be wrong.
        """
        record = turn.tool_record()
        if record:
            self._remember(TOOL_ROLE, record)

    def _context(self) -> list[dict[str, str]]:
        """What the model is given as the conversation so far.

        Pruned here rather than at write time: the full window stays available
        for anything else that reads it, and the model gets the part that fits.

        What JARVIS has been told about the user goes in front, as a system
        line. Never as the user's words -- a stored preference repeated back as
        something they just said is the attribution bug this project has
        already had once, in the memory layer.
        """
        history, dropped = prune(list(self.history), budget_chars=self.budget_chars)
        if dropped:
            log.info("bağlam budandı: %s mesaj", dropped)
        known = self._profile_summary()
        if known:
            history.insert(0, {"role": "system", "content": known})
        return history

    def _profile_summary(self) -> str:
        try:
            from ..tools import hafiza

            profile = hafiza._PROFILE  # noqa: SLF001 - module-level registry
            return profile.summary() if profile is not None else ""
        except Exception as exc:  # noqa: BLE001 - context must not break a turn
            log.debug("profil özeti alınamadı: %s", exc)
            return ""

    def _remember(self, role: str, content: str) -> None:
        # Only conversation goes back to the model. The tool record is written
        # for memory, and `chat` takes system/user/assistant -- handing the local
        # model a fourth role is asking it to parse something nobody promised it.
        if role in CONVERSATION_ROLES:
            self.history.append({"role": role, "content": content})
            if len(self.history) > self.history_limit:
                self.history = self.history[-self.history_limit:]
        if self.remember is None:
            return
        try:
            self.remember(role, content)
        except Exception as exc:  # noqa: BLE001 - memory must not cost a reply
            log.warning("hafızaya yazılamadı: %s", exc)

    def reset(self) -> None:
        self.history.clear()
        self.pending = None
        self._cancel.clear()


__all__ = ["AssistantService", "PendingCall", "turn_as_dict", "task_as_dict",
           "ASK", "CONFIRM", "CANCEL", "STATUS", "QUEUE", "TASKS",
           "TASK_CANCEL", "SHUTDOWN", "SETTINGS", "SETTING_SAVE",
           "VOICE_STATUS", "VOICE_HEAR", "VOICE_SPEAK", "VOICE_ACK",
           "VOICE_PREPARE", "IDLE", "WORKING", "WAITING", "CLOSING"]
