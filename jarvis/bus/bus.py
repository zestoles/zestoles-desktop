"""Fan-out from the event log to however many watchers happen to be connected.

## The rule that shapes everything here

A publisher must never wait for a subscriber. The event log is written from the
autonomy scheduler's thread, from an agent run, from a research fetch — and if any
of those could be slowed by a browser tab that stopped reading, the interface would
be able to stall the system it is supposed to be observing.

So every subscriber has a bounded queue and a full queue drops. Nothing blocks,
ever. What a slow client loses is counted, and it is told the number, which is a
far more useful thing than silently receiving a stream with holes in it.

## Ordering and gaps

Sequence numbers are assigned under one lock, monotonically, for the whole bus.
A client therefore knows three things from `seq` alone: the order events happened
in, whether it missed any, and whether the snapshot it holds is current.

A recent-history ring lets a reconnecting client ask for what it missed instead of
starting over. When the gap is wider than the ring, replay refuses and the client
takes a fresh snapshot — an honest restart beats a stream that silently skips.

## Where the coupling is not

Domain code publishes to the event log exactly as it did before S7B. It has no
idea a bus exists. Attaching one is a subscription made from the composition root,
and detaching it leaves every subsystem working unchanged.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..state import SharedState
from .types import (
    DROPPED,
    Envelope,
    SNAPSHOT,
    SYSTEM_STATE_CHANGED,
    payload_from_event,
    translate,
)

log = logging.getLogger("jarvis.bus")

DEFAULT_RING = 500
DEFAULT_QUEUE = 250


@dataclass(slots=True)
class Subscriber:
    """One watcher's view of the stream."""

    id: int
    queue: queue.Queue
    dropped: int = 0
    last_seq: int = 0
    created: float = field(default_factory=time.time)
    closed: bool = False

    def get(self, timeout: float | None = None) -> Envelope | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[Envelope]:
        out = []
        while True:
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                return out


class EventBus:
    def __init__(self, state: SharedState, *, ring_size: int = DEFAULT_RING,
                 queue_size: int = DEFAULT_QUEUE) -> None:
        self.state = state
        self.queue_size = queue_size
        self._lock = threading.RLock()
        self._seq = 0
        self._ring: deque[Envelope] = deque(maxlen=ring_size)
        self._subscribers: dict[int, Subscriber] = {}
        self._next_id = 0
        self._detach = None
        self._unwatch = None
        self.published = 0
        self.dropped_total = 0

    # ------------------------------------------------------------ attaching
    def attach(self, event_log, *, watch_state: bool = True):
        """Start translating a log's events onto the wire. Returns a detach callable."""
        self._detach = event_log.subscribe(self._on_event)
        if watch_state:
            self._unwatch = self.state.watch(self._on_state)
        log.info("olay veri yolu bağlandı")
        return self.detach

    def detach(self) -> None:
        for stop in (self._detach, self._unwatch):
            if stop is not None:
                try:
                    stop()
                except Exception as exc:  # noqa: BLE001
                    log.debug("abonelik kapatılamadı: %s", exc)
        self._detach = self._unwatch = None

    def _on_event(self, event) -> None:
        try:
            wire_type = translate(event.source, event.kind)
            if wire_type is None:
                return
            self.publish(wire_type, payload_from_event(event))
        except Exception as exc:  # noqa: BLE001 - never break the publisher
            log.debug("olay yayınlanamadı: %s", exc)

    def _on_state(self, section: str, data: dict[str, Any]) -> None:
        # Only the activity field is worth a wire event; the rest of a section is
        # available in the snapshot and would otherwise flood the stream.
        if section != "system" or "activity" not in data:
            return
        self.publish(SYSTEM_STATE_CHANGED, {
            "activity": data.get("activity"),
            "detail": data.get("activity_detail", ""),
        })

    # ----------------------------------------------------------- publishing
    def publish(self, wire_type: str, payload: dict[str, Any] | None = None) -> Envelope:
        with self._lock:
            self._seq += 1
            envelope = Envelope(self._seq, wire_type, time.time(), payload or {})
            self._ring.append(envelope)
            self.published += 1
            targets = list(self._subscribers.values())

        for subscriber in targets:
            self._deliver(subscriber, envelope)
        return envelope

    def _deliver(self, subscriber: Subscriber, envelope: Envelope) -> None:
        if subscriber.closed:
            return
        try:
            subscriber.queue.put_nowait(envelope)
            subscriber.last_seq = envelope.seq
        except queue.Full:
            # The whole point: a stalled reader loses messages, the writer does not
            # lose time. The count is what the client is told so it can recover.
            subscriber.dropped += 1
            self.dropped_total += 1

    # ---------------------------------------------------------- subscribing
    def subscribe(self, *, queue_size: int | None = None) -> Subscriber:
        with self._lock:
            self._next_id += 1
            subscriber = Subscriber(
                id=self._next_id,
                queue=queue.Queue(maxsize=queue_size or self.queue_size),
                last_seq=self._seq,
            )
            self._subscribers[subscriber.id] = subscriber
        log.debug("abone eklendi: %s", subscriber.id)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        subscriber.closed = True
        with self._lock:
            self._subscribers.pop(subscriber.id, None)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # -------------------------------------------------------------- replay
    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def replay_from(self, known_seq: int) -> list[Envelope] | None:
        """What a reconnecting client missed, or None when the gap is too wide.

        Returning None is the honest answer: a stream that silently skips is worse
        than one that admits the client should start over from a snapshot.
        """
        with self._lock:
            # A client claiming to be ahead of us has outlived a restart: our
            # sequence began again at zero and its number means nothing here.
            # Returning "nothing missed" would leave it frozen at a position that
            # will not come round again for hours.
            if known_seq > self._seq:
                return None
            if not self._ring:
                return [] if known_seq >= self._seq else None
            oldest = self._ring[0].seq
            if known_seq < oldest - 1:
                return None
            return [e for e in self._ring if e.seq > known_seq]

    def snapshot(self) -> Envelope:
        """The frame a client receives on connecting: everything, plus where it is."""
        with self._lock:
            seq = self._seq
        return Envelope(seq, SNAPSHOT, time.time(), {
            "state": self.state.snapshot(),
            "published": self.published,
            "subscribers": self.subscriber_count,
        })

    def dropped_notice(self, subscriber: Subscriber) -> Envelope | None:
        if not subscriber.dropped:
            return None
        return Envelope(self.seq, DROPPED, time.time(), {
            "count": subscriber.dropped,
            "advice": "yeniden anlık görüntü al",
        })

    def status(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "aboneler": self.subscriber_count,
            "yayinlanan": self.published,
            "dusen": self.dropped_total,
            "halka": len(self._ring),
            "bagli": self._detach is not None,
        }
