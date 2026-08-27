"""The short noises a listener makes, and the rules that keep them bearable.

"hı hı" is what tells a speaker they are still being heard. Without it a voice
assistant feels like talking into a recording; with too much of it, it feels
like being humoured. The difference is entirely in the rules, so the rules live
here as a pure function of the conversation state -- no model, no audio, no
clock of its own -- and can be tested by asserting on what it decides.

## Two different moments

**Listening.** The user has been talking for a while and has not finished. A
short acknowledgement belongs here and nowhere else: it says "go on". This one
is driven by voice activity alone, because it has to happen while they are
still speaking, and waiting for a transcript would make it arrive after the
moment it was for.

**Thinking.** The user has finished, JARVIS has understood, and the answer is
not ready yet -- a model call, a tool run. "bir saniye" here is not filler; it
is the difference between a pause that reads as working and one that reads as
broken.

They use different phrases because they mean different things. Saying
"bakıyorum" while someone is still talking would be answering a question they
have not asked.

## The rules, and what each one is defending against

- **A floor on how long they have been speaking.** Interjecting into a
  four-word sentence is not listening, it is interrupting.
- **A cooldown.** The failure everyone has heard: "hı hı" every two seconds.
- **Never over a question.** A question is a turn ending on purpose; talking
  over its last three words is the rudest thing in this file.
- **Never the same phrase twice running.** Repetition is what makes it sound
  generated rather than heard.
- **A cap per turn.** However long someone talks, three acknowledgements is
  plenty and four is a parody.

Selection is round-robin rather than random: the same conversation produces the
same sequence, which makes it testable, and a listener cannot tell the
difference between "unpredictable" and "not repeating".
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Said while the user is still talking. Short, and none of them mean anything
#: about the content -- an acknowledgement that took a position would be an
#: answer to a sentence that has not finished.
LISTENING = ("hı hı", "evet", "anladım", "hı hı", "tabii", "devam et")

#: Said after the user has finished, while the answer is being made.
THINKING = ("bir saniye", "bakıyorum", "tamam", "hemen bakıyorum")

#: Talking over someone who has said four words is not listening.
MIN_SPEECH_S = 4.5

#: How long after one before another may follow, in the same turn.
COOLDOWN_S = 6.0

#: However long the turn, this many is plenty.
MAX_PER_TURN = 3

#: Below this the interface is not confident the user is really speaking, and a
#: noise-triggered "hı hı" is worse than silence.
MIN_CONFIDENCE = 0.6


@dataclass(slots=True)
class Decision:
    """Whether to make a sound, which one, and why not when not."""

    phrase: str = ""
    kind: str = ""
    reason: str = ""

    @property
    def speak(self) -> bool:
        return bool(self.phrase)


@dataclass(slots=True)
class Backchannel:
    """Conversation-aware acknowledgements. One instance per conversation."""

    min_speech_s: float = MIN_SPEECH_S
    cooldown_s: float = COOLDOWN_S
    max_per_turn: int = MAX_PER_TURN
    #: 0 switches it off entirely, for someone who finds it distracting.
    level: float = 1.0

    _turn_count: int = field(default=0, init=False)
    _last_at: float = field(default=0.0, init=False)
    _last_phrase: str = field(default="", init=False)
    _listen_index: int = field(default=0, init=False)
    _think_index: int = field(default=0, init=False)

    # ------------------------------------------------------------ listening
    def while_listening(self, *, speech_seconds: float, now: float,
                        confidence: float = 1.0,
                        looks_like_question: bool = False) -> Decision:
        """Should JARVIS say "hı hı" while the user is still talking?"""
        if self.level <= 0:
            return Decision(reason="kapalı")
        if looks_like_question:
            # A question is a turn ending on purpose. Do not talk over it.
            return Decision(reason="soru bölünmez")
        if confidence < MIN_CONFIDENCE:
            return Decision(reason="konuşma olduğundan emin değil")
        if speech_seconds < self._scaled(self.min_speech_s):
            return Decision(reason="henüz kısa")
        if self._turn_count >= self.max_per_turn:
            return Decision(reason="bu turda yeterince söylendi")
        if self._last_at and now - self._last_at < self._scaled(self.cooldown_s):
            return Decision(reason="çok yakın")

        phrase = self._next(LISTENING, "_listen_index")
        self._turn_count += 1
        self._last_at = now
        self._last_phrase = phrase
        return Decision(phrase=phrase, kind="dinleme")

    # ------------------------------------------------------------- thinking
    def while_thinking(self, *, expected_wait_s: float, now: float,
                       used_tool: bool = False) -> Decision:
        """Should JARVIS say "bir saniye" before the answer is ready?

        Only when the wait is long enough to be noticed. Filling a
        half-second gap draws attention to a pause nobody would have felt.
        """
        if self.level <= 0:
            return Decision(reason="kapalı")
        if expected_wait_s < self._scaled(1.2):
            return Decision(reason="bekleme kısa")
        if self._last_at and now - self._last_at < 2.0:
            return Decision(reason="az önce konuşuldu")
        phrase = self._next(THINKING, "_think_index")
        if used_tool and phrase == "tamam":
            # "tamam" before running something reads as "done", which it is not.
            phrase = "bakıyorum"
        self._last_at = now
        self._last_phrase = phrase
        return Decision(phrase=phrase, kind="düşünme")

    # ---------------------------------------------------------------- turns
    def turn_finished(self) -> None:
        """The user stopped talking. Allowances reset with the new turn."""
        self._turn_count = 0

    def reset(self) -> None:
        self._turn_count = 0
        self._last_at = 0.0
        self._last_phrase = ""

    @property
    def phrases(self) -> list[str]:
        """Everything that might be said, so the audio can be made in advance."""
        return sorted(set(LISTENING) | set(THINKING))

    # ------------------------------------------------------------ internals
    def _scaled(self, value: float) -> float:
        """A lower level means a higher bar, not a different vocabulary."""
        level = max(0.05, min(2.0, self.level))
        return value / level

    def _next(self, pool: tuple[str, ...], cursor_name: str) -> str:
        index = getattr(self, cursor_name)
        for step in range(len(pool)):
            candidate = pool[(index + step) % len(pool)]
            if candidate != self._last_phrase:
                setattr(self, cursor_name, (index + step + 1) % len(pool))
                return candidate
        setattr(self, cursor_name, (index + 1) % len(pool))
        return pool[index % len(pool)]


def looks_like_question(text: str) -> bool:
    """A cheap guess, used only to decide whether to keep quiet.

    Wrong in the safe direction on purpose: a statement mistaken for a question
    costs one missed "hı hı", while a question mistaken for a statement means
    talking over somebody.
    """
    body = str(text or "").strip().lower()
    if not body:
        return False
    if body.endswith("?"):
        return True
    tail = body.rstrip(".!… ").split()[-1:] or [""]
    markers = ("mi", "mı", "mu", "mü", "misin", "mısın", "musun", "müsün",
               "miyim", "mıyım", "muyum", "müyüm", "mı?", "değil")
    if any(tail[0].endswith(m) for m in markers):
        return True
    starters = ("ne ", "neden", "nasıl", "kim ", "nerede", "nereye", "hangi",
                "kaç ", "niye", "niçin", "ne zaman")
    return body.startswith(starters)


__all__ = ["Backchannel", "Decision", "looks_like_question", "LISTENING",
           "THINKING", "MIN_SPEECH_S", "COOLDOWN_S", "MAX_PER_TURN"]
