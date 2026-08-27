"""The loop that lets ZESTOLES actually do what it was asked.

A model that can only talk produces "you could run `python --version`". A model
wired to this loop runs it and reads the answer. The difference is entirely in
what happens between the request and the reply, which is what lives here:

    ask the model what to do next
    → it names a tool and its arguments, or it answers
    → the name is checked against the registry, never trusted
    → the risk gate decides whether it may run now or needs the user
    → the tool runs for real
    → the real result goes back to the model as an observation
    → repeat until it answers, or until the step budget runs out

## The model is not trusted, again

Same discipline as `improve/planner.py`, for the same reason. The model supplies
a tool *name* which must already exist in `tools.REGISTRY`, and an arguments
object. Anything else — an unknown tool, arguments that are not an object, a
decision that is neither "tool" nor "reply" — is a rejected step, not a step that
runs. `read_decision` is pure so it can be tested without a model.

## Truth is structural, not rhetorical

Nothing here can stop a language model from writing "done!" after a tool failed.
So the honest record is kept outside its prose: every `Step` carries the real
`ToolResult`, and `Turn.succeeded` is computed from those results. A caller that
wants to know whether the work actually happened reads the steps, never the
reply. That is also what the UI must display.

## Confirmation is the caller's to answer

The loop never passes `confirmed=True` on its own. LOW-risk tools run; anything
else goes to the `approve` callback the caller supplied, and with no callback the
turn stops and hands back the pending call. An assistant that quietly approved
its own writes would make the risk tier decorative.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import tools
from ..tools import LOW, ToolResult, Workspace

log = logging.getLogger("jarvis.assistant")

#: How many tool calls one request may make before the loop gives up. A model
#: that has not finished by here is looping, not working.
MAX_STEPS = 8

#: The same tool with the same arguments twice in a row is a stuck model, not
#: progress. Measured cheaply rather than waiting for the step budget.
REPEAT_LIMIT = 2

TOOL = "tool"
REPLY = "reply"

#: The role a turn's tool record is written under. Not a conversation role: it
#: goes to memory and never into the history handed to the model, which takes
#: system/user/assistant and was never promised a fourth.
TOOL_ROLE = "arac"

#: How much of one tool's output the record keeps. Memory is for what happened,
#: not for the file that was read -- a record that carried whole outputs would
#: bury the session it was meant to describe.
RECORD_DETAIL_MAX = 300
RECORD_MAX = 1500

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [TOOL, REPLY]},
        "tool": {"type": "string"},
        "arguments": {"type": "object"},
        "message": {"type": "string"},
    },
    "required": ["action"],
}

SYSTEM_PROMPT = """\
Sen ZESTOLES'sin. Kullanıcının bilgisayarında gerçekten iş yapabilirsin.

Her adımda tek bir JSON nesnesi döndürürsün:

  {"action": "tool", "tool": "<araç adı>", "arguments": {...}}
  {"action": "reply", "message": "<kullanıcıya cevap>"}

Kurallar:

- Yalnızca aşağıda listelenen araçları kullan. Listede olmayan bir ad uydurma.
- Argüman adları listede parantez içinde yazıyor; aynen onları kullan. Köşeli
  parantezli olanlar isteğe bağlıdır.
- Bir aracı çağırdıktan sonra sonucunu okursun; sonuç sana gözlem olarak gelir.
- Gerekiyorsa arka arkaya birden fazla araç çağırabilirsin.
- Bir araç başarısız olduysa bunu kullanıcıya dürüstçe söyle. Başarı uydurma.
- Bir araç çağırmadan bir işi yaptığını söyleme.
- İş bittiğinde "reply" ile doğal, duruma uygun uzunlukta cevap ver.

Kullanıcıyı anlama ve konuşma biçimin:

- Son cümleyi tek başına yorumlama. Konuşma geçmişindeki konuya, zamirlere ve
  kullanıcının az önce yaptığı düzeltmelere bağla. "onu", "şunu", "öyle yap",
  "yok dokuz olsun" gibi eksiltili Türkçe ifadelerin karşılığını geçmişten bul.
- Kullanıcının en son düzeltmesi önceki isteğin yerine geçer. Eski saat, dosya,
  hedef veya biçimi yanlışlıkla yeniden kullanma.
- Türkçeyi kelime kelime değil niyet olarak anla. Gündelik söyleyişleri,
  yazım hatalarını, üstü kapalı memnuniyetsizliği ve konuşma dilini hesaba kat.
- Bağlam tek bir makul, geri alınabilir ve düşük riskli yorumu gösteriyorsa
  tekrar soru sorma; o yorumla ilerle. Seçenekler sonucu belirgin biçimde
  değiştiriyorsa tek, kısa ve somut soru sor.
- Bir şey yapamadığını söylemeden önce araç listesine bak. Uygun araç varsa
  kullan; yoksa sınırı dürüstçe ve öneri menüsü dökmeden açıkla.
- shell.run ile kullanıcıya mesaj göstermek için echo/print çalıştırma; bu bir
  eylem değildir. Kullanıcı yalnızca konuşuyorsa "reply" kullan. Bilgisayarda
  neyin değişeceği belirsizse rastgele bir süreç veya dosya açma, somutlaştıran
  tek bir soru sor.
- Sohbette sıcak, zeki ve özgüvenli bir çalışma arkadaşı gibi konuş. Kullanıcı
  yorgun, kızgın veya kararsızsa bunu bir kısa cümleyle fark et ve hemen somut
  yardıma geç. Yapay özürler, kalıp övgüler ve "başka ne yapabilirim" menüleri
  üretme.
- Basit soruya kısa cevap ver; karmaşık konuda gereken ayrıntıyı saklama.
  Kullanıcının kelimelerini gereksizce tekrar ederek cevap doldurma.
- Tarih ve saat uydurma. Kullanıcı göreli zaman verdiyse ("yarın dokuz",
  "ondan iki gün önce") araç argümanında bu anlamı koru; gözlem olmadan kendi
  başına yıl veya kesin tarih icat etme.

Bu bilgisayar hakkında hiçbir şey bilmiyorsun. Kurulu sürümler, dosyalar,
klasörler, donanım, ayarlar, çalışan programlar — hepsi yalnızca araçla
öğrenilir. Bunlardan biri sorulduğunda önce aracı çağır; kendi bilginden bir
sürüm numarası, dosya adı, yol ya da sayı söyleme. Söylersen uydurmuş olursun.

  "python sürümü nedir"      → shell.run
  "masaüstümde ne var"       → fs.list
  "kaç çekirdek var"         → system.info

Araç yalnızca genel bilgi sorularında gereksizdir — bir kavramın tanımı, bir
dilin nasıl çalıştığı gibi, cevabı bu bilgisayara bağlı olmayan sorular.

Kullanıcıyı tanıma: konuşma sırasında kalıcı olarak işine yarayacak bir tercih
duyarsan (nasıl hitap edilmesini istediği, çalışma alışkanlığı, teknik tercihi)
önce sorabilirsin: "İstersen bunu hatırlarım."

memory.remember çağırırken onay şu demektir: kullanıcı kaydedilmesini istedi mi?
"bunu hatırla", "kaydet", "unutma" gibi açık bir istek ya da senin sorduğun
soruya "evet" — hepsi onay=true. Kendiliğinden iyi fikir gördüğün için
kaydediyorsan onay=false ver; araç reddeder ve önce sorman gerektiğini söyler.

Örnek — kullanıcı "geceleri çalışırım, bunu hatırla" dediyse:

  {"action": "tool", "tool": "memory.remember", "arguments": {"konu": "çalışma saati", "ayrinti": "geceleri çalışıyor", "onay": true}}

Argümanları boş bırakma; konu ve ayrinti her zaman gerekir.

Kendi çıkarımlarını kaydetme. "Sen şöyle birisin" türü bir sonuç senin tahminin;
kullanıcının söylediği değil. Kaydetmek istiyorsan önce ona sor.

"Benim hakkımda ne biliyorsun" diye sorulursa memory.recall çağır ve yalnızca
dönen şeyleri say. "Bunu unut" denirse memory.forget çağır.

Kullanabileceğin araçlar:

{catalogue}
"""

LIVE_VOICE_PROMPT = (
    "\n\n[Canlı ses modu]\n"
    "Kullanıcı seni dinliyor. Reply yanıtının ilk cümlesi doğal, doğrudan ve "
    "en fazla sekiz kelime olsun; ilk cümlede başlık, Markdown veya URL kullanma. "
    "Gerekli ayrıntıyı sonraki kısa cümlelerde ver."
)


@dataclass(slots=True)
class Decision:
    """What the model asked for, after checking. Never what it merely said."""

    kind: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.kind) and not self.problems


@dataclass(slots=True)
class Step:
    tool: str
    arguments: dict[str, Any]
    result: ToolResult

    @property
    def ok(self) -> bool:
        return self.result.ok

    def as_observation(self) -> str:
        """What the model is told happened. Real output, real failure."""
        if self.result.needs_confirmation:
            return f"[{self.tool}] onay bekliyor: {self.result.error}"
        if self.result.ok:
            body = self.result.output or "(çıktı yok)"
            return f"[{self.tool}] başarılı:\n{body[:4000]}"
        return f"[{self.tool}] BAŞARISIZ: {self.result.error}"


@dataclass(slots=True)
class Turn:
    """One user request and everything that really happened because of it."""

    reply: str = ""
    steps: list[Step] = field(default_factory=list)
    stopped: str = ""
    pending: Step | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the work happened. Computed from results, not from prose.

        A cancelled turn sets `stopped`, so it can never read as success — the
        thing §36 is about: work the user stopped must not be reported done.
        """
        return not self.stopped and self.pending is None and all(
            step.ok for step in self.steps)

    @property
    def cancelled(self) -> bool:
        return self.stopped == "iptal edildi"

    @property
    def failures(self) -> list[Step]:
        return [step for step in self.steps if not step.ok]

    @property
    def used_tools(self) -> list[str]:
        return [step.tool for step in self.steps]

    def tool_record(self) -> str:
        """What actually ran this turn, for memory. Empty when nothing did.

        Built from the recorded `ToolResult`s, so it says "BASARISIZ" over a
        failure the reply may have called a success. That is the whole reason it
        is written separately: the prose is the part of a turn that can be wrong,
        and it was previously the only part memory kept.
        """
        if not self.steps:
            return ""
        lines = []
        for step in self.steps:
            detail = (step.result.output if step.ok else step.result.error) or ""
            flat = " ".join(detail.split())[:RECORD_DETAIL_MAX]
            lines.append(f"{step.tool} - {'basarili' if step.ok else 'BASARISIZ'}"
                         + (f": {flat}" if flat else ""))
        body = "\n".join(lines)[:RECORD_MAX]
        return f"Bu turda gerçekten çalıştırılan araçlar:\n{body}"

    def summary(self) -> str:
        if self.pending is not None:
            return f"onay bekleniyor: {self.pending.tool}"
        if self.stopped:
            return f"durduruldu: {self.stopped}"
        used = ", ".join(self.used_tools) or "araç kullanılmadı"
        return f"{len(self.steps)} adım ({used})"


def read_decision(raw: str, *, available: set[str]) -> Decision:
    """Turn a model reply into a decision, or into the reasons it is not one.

    Pure. The one thing standing between generated text and a subprocess.
    """
    decision = Decision()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        decision.problems.append("cevap geçerli JSON değil")
        return decision
    if not isinstance(payload, dict):
        decision.problems.append("cevap bir nesne değil")
        return decision

    action = str(payload.get("action", "")).strip().lower()
    if action == REPLY:
        decision.kind = REPLY
        decision.message = str(payload.get("message", "")).strip()
        if not decision.message:
            decision.problems.append("boş cevap")
        return decision

    if action != TOOL:
        decision.problems.append(f"bilinmeyen eylem: {action[:40]!r}")
        return decision

    name = str(payload.get("tool", "")).strip()
    if not name:
        decision.problems.append("araç adı verilmedi")
        return decision
    if name not in available:
        decision.problems.append(f"kayıtlı olmayan araç: {name[:60]!r}")
        return decision

    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        decision.problems.append("arguments bir nesne değil")
        return decision
    if any(not isinstance(key, str) or not key.isidentifier() for key in arguments):
        decision.problems.append("arguments anahtarları geçersiz")
        return decision

    decision.kind = TOOL
    decision.tool = name
    decision.arguments = arguments
    return decision


class Assistant:
    """Wires a brain, the tool registry and the event log into one loop."""

    def __init__(
        self,
        brain,
        workspace: Workspace,
        *,
        events=None,
        approve: Callable[[str, str, dict[str, Any]], bool] | None = None,
        should_stop: Callable[[], bool] | None = None,
        max_steps: int = MAX_STEPS,
        model: str = "",
        temperature: float = 0.1,
    ) -> None:
        self.brain = brain
        self.workspace = workspace
        self.events = events
        #: Asked between steps. A tool already running is not interrupted — the
        #: loop stops before starting the next one — because killing a write
        #: halfway is worse than finishing it.
        self.should_stop = should_stop
        #: Called as approve(tool_name, risk, arguments) -> bool for anything
        #: above LOW. None means "cannot ask", and the turn stops instead.
        self.approve = approve
        self.max_steps = max(1, int(max_steps))
        self.model = model
        self.temperature = temperature

    # ------------------------------------------------------------- plumbing
    def _emit(self, kind: str, message: str, level: str = "info", **data) -> None:
        if self.events is None:
            return
        try:
            self.events.publish("assistant", kind, message, level=level, data=data)
        except Exception as exc:  # noqa: BLE001 - telemetry must not break a turn
            log.debug("olay yayınlanamadı: %s", exc)

    def _stopped(self) -> bool:
        if self.should_stop is None:
            return False
        try:
            return bool(self.should_stop())
        except Exception as exc:  # noqa: BLE001 - a broken check is not a stop
            log.debug("iptal kontrolü hata verdi: %s", exc)
            return False

    def system_prompt(self, *, live_voice: bool = False) -> str:
        lines = []
        for entry in tools.catalogue():
            call = f"{entry['name']}({entry.get('arguments', '')})"
            lines.append(f"  {call}  ({entry['risk']}) — {entry['summary']}")
        # replace, not format: the prompt shows literal JSON, and str.format
        # reads every brace in those examples as a field.
        prompt = SYSTEM_PROMPT.replace("{catalogue}", "\n".join(lines))
        return prompt + LIVE_VOICE_PROMPT if live_voice else prompt

    def _think(self, transcript: list[dict[str, str]],
               on_sentence: Callable[[str], None] | None = None) -> str:
        """One model decision. `on_sentence` turns the reply into a stream.

        The streaming path feeds the raw JSON through ReplySentenceStream and
        speaks completed sentences of a "reply" decision while the model is
        still writing it. Tool decisions pass the gate silently. A brain that
        cannot stream -- or one whose stream fails mid-flight -- falls back to
        the blocking call: slower, never wronger.
        """
        if on_sentence is None:
            return self.brain.local.chat(
                transcript, schema=DECISION_SCHEMA, temperature=self.temperature,
                model=self.model or None, purpose="arac-secimi")

        stream_fn = getattr(self.brain.local, "stream", None)
        if stream_fn is None:
            return self._think(transcript)

        from .streaming import ReplySentenceStream

        extractor = ReplySentenceStream()
        parts: list[str] = []
        try:
            for chunk in stream_fn(transcript, schema=DECISION_SCHEMA,
                                   temperature=self.temperature,
                                   model=self.model or None):
                parts.append(chunk)
                for sentence in extractor.feed(chunk):
                    self._safe_sentence(sentence, on_sentence)
            for sentence in extractor.finish():
                self._safe_sentence(sentence, on_sentence)
        except OSError as exc:
            # Akış ortasında koptuysa toparlanmak yerine dürüst düşüş:
            # yarım konuşulmuş cevabın devamı uydurulmaz.
            raise OSError(f"akış kesildi: {exc}") from exc
        return "".join(parts)

    @staticmethod
    def _safe_sentence(sentence: str,
                       on_sentence: Callable[[str], None]) -> None:
        try:
            on_sentence(sentence)
        except Exception as exc:  # noqa: BLE001 - dinleyen bozulursa sessiz kal
            log.warning("cümle alıcısı hata verdi: %s", exc)

    # ------------------------------------------------------------------ loop
    def run(self, request: str, *, history: list[dict[str, str]] | None = None,
            on_sentence: Callable[[str], None] | None = None) -> Turn:
        """Answer one request, using tools when they are needed.

        `on_sentence`, verildiğinde nihai cevabın tamamlanan cümleleri cevap
        bitmeden bu geri çağrıya akar -- sesli yolda model konuşurken yazmaya
        devam eder. Araç adımları buradan hiçbir şey geçmez.
        """
        turn = Turn()
        transcript: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt(
                live_voice=on_sentence is not None)},
            *(history or []),
            {"role": "user", "content": request},
        ]
        self._emit("turn.start", f"İstek alındı: {request[:120]}")
        recent: list[tuple[str, str]] = []

        for _ in range(self.max_steps):
            if self._stopped():
                turn.stopped = "iptal edildi"
                self._emit("turn.cancelled", turn.stopped, level="warn",
                           steps=len(turn.steps))
                return turn
            try:
                raw = self._think(transcript, on_sentence=on_sentence)
            except OSError as exc:
                turn.stopped = f"model yanıt vermedi: {exc}"
                self._emit("turn.failed", turn.stopped, level="warn")
                return turn

            decision = read_decision(raw, available=set(tools.names()))
            if not decision.ok:
                # Tell the model exactly what was wrong and let it try again;
                # a rejected step costs one budget slot, not the whole turn.
                problem = "; ".join(decision.problems)
                log.info("karar reddedildi: %s", problem)
                self._emit("decision.rejected", f"Karar reddedildi: {problem}",
                           level="warn")
                transcript.append({"role": "assistant", "content": raw[:2000]})
                transcript.append({
                    "role": "user",
                    "content": f"[sistem] Bu geçersizdi: {problem}. "
                               "Yalnızca listelenen araçları kullanarak geçerli "
                               "bir JSON karar ver."})
                continue

            if decision.kind == REPLY:
                turn.reply = decision.message
                self._emit("turn.done", turn.summary(),
                           level="success" if turn.succeeded else "warn",
                           steps=len(turn.steps))
                return turn

            signature = (decision.tool, json.dumps(decision.arguments, sort_keys=True,
                                                   default=str))
            recent.append(signature)
            if recent.count(signature) > REPEAT_LIMIT:
                turn.stopped = (f"aynı çağrı tekrar ediyor: {decision.tool}")
                self._emit("turn.stalled", turn.stopped, level="warn")
                return turn

            # Checked again here, not only at the top: the model call above can
            # take seconds, and a cancel that arrived while it was running must
            # not be answered by starting the tool it chose.
            if self._stopped():
                turn.stopped = "iptal edildi"
                self._emit("turn.cancelled", turn.stopped, level="warn",
                           steps=len(turn.steps))
                return turn

            step = self._perform(decision)
            if step.result.needs_confirmation:
                turn.pending = step
                self._emit("tool.confirm", f"Onay gerekiyor: {decision.tool}",
                           level="warn", tool=decision.tool, risk=step.result.risk)
                return turn

            turn.steps.append(step)
            transcript.append({"role": "assistant",
                               "content": json.dumps(
                                   {"action": TOOL, "tool": decision.tool,
                                    "arguments": decision.arguments},
                                   ensure_ascii=False)})
            transcript.append({"role": "user",
                               "content": f"[gözlem] {step.as_observation()}"})

        turn.stopped = f"adım sınırına ulaşıldı ({self.max_steps})"
        self._emit("turn.stalled", turn.stopped, level="warn")
        return turn

    def _perform(self, decision: Decision) -> Step:
        """Run one tool, asking first when the risk tier says to."""
        entry = tools.get(decision.tool)
        risk = entry.risk if entry else LOW
        confirmed = False
        if risk != LOW:
            if self.approve is None:
                # No way to ask. Produce the pending result the caller must
                # answer rather than deciding on the user's behalf.
                result = tools.run(decision.tool, workspace=self.workspace,
                                   confirmed=False, **decision.arguments)
                return Step(decision.tool, decision.arguments, result)
            try:
                confirmed = bool(self.approve(decision.tool, risk, decision.arguments))
            except Exception as exc:  # noqa: BLE001 - a broken UI is a refusal
                log.warning("onay geri araması hata verdi: %s", exc)
                confirmed = False
            if not confirmed:
                self._emit("tool.denied", f"Reddedildi: {decision.tool}", level="warn",
                           tool=decision.tool)
                return Step(decision.tool, decision.arguments,
                            ToolResult(False, error="kullanıcı onaylamadı",
                                       tool=decision.tool, risk=risk))

        self._emit("tool.start", f"{decision.tool} çalışıyor", tool=decision.tool,
                   risk=risk, arguments=decision.arguments)
        result = tools.run(decision.tool, workspace=self.workspace,
                           confirmed=confirmed, **decision.arguments)
        self._emit("tool.done" if result.ok else "tool.failed",
                   f"{decision.tool}: {result.summary()[:160]}",
                   level="success" if result.ok else "warn",
                   tool=decision.tool, ok=result.ok)
        return Step(decision.tool, decision.arguments, result)

    def confirm(self, step: Step) -> Step:
        """Run a call the user has now approved."""
        self._emit("tool.start", f"{step.tool} çalışıyor (onaylandı)", tool=step.tool)
        result = tools.run(step.tool, workspace=self.workspace, confirmed=True,
                           **step.arguments)
        self._emit("tool.done" if result.ok else "tool.failed",
                   f"{step.tool}: {result.summary()[:160]}",
                   level="success" if result.ok else "warn", tool=step.tool)
        return Step(step.tool, step.arguments, result)


__all__ = ["Assistant", "Decision", "Step", "Turn", "read_decision",
           "DECISION_SCHEMA", "SYSTEM_PROMPT", "MAX_STEPS", "TOOL", "REPLY",
           "TOOL_ROLE", "RECORD_DETAIL_MAX", "RECORD_MAX"]

# Registers the `assistant.ask` runner so a request can be queued instead of
# held open on a connection. Imported here rather than left to whoever wires the
# interface: a runner that registers only when someone remembers to import it
# is one the tests can see and the product cannot. Import direction is still one
# way -- this reaches down into `autonomy`, never the reverse.
from . import background  # noqa: E402,F401
