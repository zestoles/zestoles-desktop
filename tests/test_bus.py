"""Event bus and websocket transport.

The scenarios that decide whether this is safe to leave running: a client that
disconnects, one that reconnects, one that stops reading, several at once, a flood
of events, a malformed message, and no client at all. Every one of them must leave
the backend untouched — the interface is an observer, and an observer that can
affect what it observes is a defect.

The framing tests speak the protocol directly rather than through a browser,
because a masked-frame bug produces a connection that looks fine until something
real closes it with a protocol error.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.bus.bus import EventBus  # noqa: E402
from jarvis.bus.server import BusServer  # noqa: E402
from jarvis.bus.types import (  # noqa: E402
    AGENT_STARTED,
    ALL_TYPES,
    DROPPED,
    ERROR,
    Envelope,
    RESEARCH_FINISHED,
    SNAPSHOT,
    SYSTEM_STATE_CHANGED,
    TRANSLATION,
    payload_from_event,
    translate,
)
from jarvis.bus.websocket import (  # noqa: E402
    CLOSE_NORMAL,
    OP_CLOSE,
    OP_PING,
    OP_TEXT,
    MAX_PAYLOAD,
    WebSocketConnection,
    WebSocketError,
    accept_key,
    client_mask,
    decode_frame,
    encode_close,
    encode_frame,
    handshake_response,
    parse_request,
)
from jarvis.state import SharedState  # noqa: E402


@dataclass
class FakeEvent:
    source: str
    kind: str
    message: str = "mesaj"
    level: str = "info"
    ts: float = 1000.0
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class FakeLog:
    """Just enough EventLog to subscribe to."""

    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)

    def publish(self, event):
        for callback in list(self.subscribers):
            callback(event)


# ------------------------------------------------------------------ translation
class TestTranslation(unittest.TestCase):
    def test_known_pairs_translate(self):
        self.assertEqual(translate("agent", "run.start"), AGENT_STARTED)
        self.assertEqual(translate("research", "done"), RESEARCH_FINISHED)

    def test_unknown_pairs_stay_off_the_wire(self):
        """Silence beats an event the far end cannot interpret."""
        self.assertIsNone(translate("uydurma", "sey"))

    def test_every_mapped_type_is_a_declared_type(self):
        for pair, wire_type in TRANSLATION.items():
            with self.subTest(pair=pair):
                self.assertIn(wire_type, ALL_TYPES)

    def test_payload_keeps_structure_and_label(self):
        payload = payload_from_event(FakeEvent("agent", "run.start", "başladı",
                                               data={"run": "abc"}))
        self.assertEqual(payload["source"], "agent")
        self.assertEqual(payload["data"]["run"], "abc")
        self.assertEqual(payload["label"], "başladı")

    def test_envelope_serialises(self):
        parsed = json.loads(Envelope(7, AGENT_STARTED, 1.0, {"a": 1}).to_json())
        self.assertEqual(parsed["seq"], 7)
        self.assertEqual(parsed["type"], AGENT_STARTED)

    def test_unserialisable_payload_degrades_rather_than_raising(self):
        parsed = json.loads(Envelope(1, ERROR, 1.0, {"obj": object()}).to_json())
        self.assertIsInstance(parsed["payload"]["obj"], str)


# -------------------------------------------------------------------- the bus
class TestEventBus(unittest.TestCase):
    def setUp(self):
        self.state = SharedState()
        self.bus = EventBus(self.state, ring_size=10, queue_size=5)

    def test_sequence_numbers_are_monotonic(self):
        seqs = [self.bus.publish(AGENT_STARTED).seq for _ in range(5)]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), 5)

    def test_subscribers_receive_in_order(self):
        subscriber = self.bus.subscribe()
        for index in range(3):
            self.bus.publish(AGENT_STARTED, {"i": index})
        received = [e.payload["i"] for e in subscriber.drain()]
        self.assertEqual(received, [0, 1, 2])

    def test_multiple_subscribers_all_receive(self):
        a, b, c = (self.bus.subscribe() for _ in range(3))
        self.bus.publish(AGENT_STARTED)
        for subscriber in (a, b, c):
            self.assertEqual(len(subscriber.drain()), 1)

    def test_a_full_queue_drops_instead_of_blocking(self):
        """The whole point: a stalled reader must not be able to stall the system."""
        subscriber = self.bus.subscribe(queue_size=2)
        for _ in range(10):
            self.bus.publish(AGENT_STARTED)
        self.assertEqual(subscriber.dropped, 8)
        self.assertEqual(self.bus.dropped_total, 8)

    def test_a_slow_subscriber_does_not_slow_the_publisher(self):
        self.bus.subscribe(queue_size=1)
        started = time.monotonic()
        for _ in range(500):
            self.bus.publish(AGENT_STARTED)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_a_dropped_notice_is_offered(self):
        subscriber = self.bus.subscribe(queue_size=1)
        for _ in range(5):
            self.bus.publish(AGENT_STARTED)
        notice = self.bus.dropped_notice(subscriber)
        self.assertEqual(notice.type, DROPPED)
        self.assertEqual(notice.payload["count"], 4)

    def test_no_notice_when_nothing_was_dropped(self):
        self.assertIsNone(self.bus.dropped_notice(self.bus.subscribe()))

    def test_unsubscribing_stops_delivery(self):
        subscriber = self.bus.subscribe()
        self.bus.unsubscribe(subscriber)
        self.bus.publish(AGENT_STARTED)
        self.assertEqual(subscriber.drain(), [])

    def test_publishing_with_no_subscribers_is_fine(self):
        """UI closed is the normal case, not an edge case."""
        self.assertEqual(self.bus.publish(AGENT_STARTED).seq, 1)
        self.assertEqual(self.bus.subscriber_count, 0)

    def test_attaching_translates_log_events(self):
        log = FakeLog()
        self.bus.attach(log, watch_state=False)
        subscriber = self.bus.subscribe()
        log.publish(FakeEvent("agent", "run.start"))
        self.assertEqual(subscriber.drain()[0].type, AGENT_STARTED)

    def test_unmapped_log_events_are_not_forwarded(self):
        log = FakeLog()
        self.bus.attach(log, watch_state=False)
        subscriber = self.bus.subscribe()
        log.publish(FakeEvent("uydurma", "sey"))
        self.assertEqual(subscriber.drain(), [])

    def test_a_malformed_event_cannot_break_the_publisher(self):
        class Broken:
            source = "agent"
            kind = "run.start"

            @property
            def message(self):
                raise RuntimeError("bozuk")

        log = FakeLog()
        self.bus.attach(log, watch_state=False)
        log.publish(Broken())  # must not raise
        self.assertTrue(True)

    def test_detaching_stops_translation(self):
        log = FakeLog()
        detach = self.bus.attach(log, watch_state=False)
        detach()
        subscriber = self.bus.subscribe()
        log.publish(FakeEvent("agent", "run.start"))
        self.assertEqual(subscriber.drain(), [])

    def test_state_changes_become_wire_events(self):
        self.bus.attach(FakeLog())
        subscriber = self.bus.subscribe()
        self.state.set_activity("thinking", "planlıyor")
        types = [e.type for e in subscriber.drain()]
        self.assertIn(SYSTEM_STATE_CHANGED, types)

    def test_snapshot_carries_state_and_position(self):
        self.bus.publish(AGENT_STARTED)
        snapshot = self.bus.snapshot()
        self.assertEqual(snapshot.type, SNAPSHOT)
        self.assertEqual(snapshot.seq, 1)
        self.assertIn("sections", snapshot.payload["state"])


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus(SharedState(), ring_size=5)

    def test_replay_returns_what_was_missed(self):
        for _ in range(4):
            self.bus.publish(AGENT_STARTED)
        missed = self.bus.replay_from(2)
        self.assertEqual([e.seq for e in missed], [3, 4])

    def test_replay_from_current_returns_nothing(self):
        self.bus.publish(AGENT_STARTED)
        self.assertEqual(self.bus.replay_from(1), [])

    def test_a_gap_wider_than_history_refuses(self):
        """An honest restart beats a stream that silently skips."""
        for _ in range(12):
            self.bus.publish(AGENT_STARTED)
        self.assertIsNone(self.bus.replay_from(1))

    def test_replay_on_an_empty_bus(self):
        self.assertEqual(self.bus.replay_from(0), [])

    def test_a_client_ahead_of_us_survived_a_restart(self):
        """Its sequence number is from the previous process and means nothing here.
        Answering "nothing missed" would freeze it at a position that will not come
        round again for hours, with gap detection disabled the whole time."""
        for _ in range(3):
            self.bus.publish(AGENT_STARTED)
        self.assertIsNone(self.bus.replay_from(9999))

    def test_a_client_ahead_of_an_empty_bus_also_restarts(self):
        self.assertIsNone(EventBus(SharedState()).replay_from(500))


# --------------------------------------------------------------- ws framing
class TestHandshake(unittest.TestCase):
    def test_accept_key_matches_the_rfc_example(self):
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
                         "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_a_valid_upgrade_is_accepted(self):
        headers = parse_request(
            b"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n\r\n")
        self.assertTrue(handshake_response(headers).startswith(b"HTTP/1.1 101"))

    def test_a_plain_http_request_is_refused(self):
        headers = parse_request(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertTrue(handshake_response(headers).startswith(b"HTTP/1.1 400"))

    def test_the_wrong_protocol_version_is_refused(self):
        headers = parse_request(
            b"GET / HTTP/1.1\r\nUpgrade: websocket\r\n"
            b"Sec-WebSocket-Key: abc\r\nSec-WebSocket-Version: 8\r\n\r\n")
        self.assertTrue(handshake_response(headers).startswith(b"HTTP/1.1 400"))

    def test_garbage_is_not_a_request(self):
        self.assertEqual(parse_request(b"\x00\x01\x02"), {})


class TestFraming(unittest.TestCase):
    def test_close_releases_socket_even_after_connection_is_marked_dead(self):
        left, right = socket.socketpair()
        try:
            connection = WebSocketConnection(left, ("local", 0))
            connection.open = False
            connection.close()
            self.assertEqual(left.fileno(), -1)
        finally:
            right.close()

    def test_a_short_text_frame_round_trips(self):
        frame = encode_frame("merhaba".encode("utf-8"))
        decoded, rest = decode_frame(frame)
        self.assertEqual(decoded["payload"].decode(), "merhaba")
        self.assertEqual(decoded["opcode"], OP_TEXT)
        self.assertEqual(rest, b"")

    def test_server_frames_are_never_masked(self):
        """A masked server frame is a protocol error the browser will close on."""
        self.assertFalse(decode_frame(encode_frame(b"x"))[0]["masked"])

    def test_medium_and_large_lengths(self):
        for size in (200, 70000):
            with self.subTest(size=size):
                decoded, _ = decode_frame(encode_frame(b"a" * size))
                self.assertEqual(len(decoded["payload"]), size)

    def test_a_masked_client_frame_is_unmasked(self):
        decoded, _ = decode_frame(client_mask("selam".encode("utf-8")))
        self.assertTrue(decoded["masked"])
        self.assertEqual(decoded["payload"].decode(), "selam")

    def test_a_partial_buffer_asks_for_more(self):
        """Truncated reads are the normal case on a stream socket, not an error."""
        frame = encode_frame(b"uzunca bir mesaj")
        self.assertIsNone(decode_frame(frame[:3]))

    def test_two_frames_in_one_buffer(self):
        buffer = encode_frame(b"bir") + encode_frame(b"iki")
        first, rest = decode_frame(buffer)
        second, remaining = decode_frame(rest)
        self.assertEqual(first["payload"], b"bir")
        self.assertEqual(second["payload"], b"iki")
        self.assertEqual(remaining, b"")

    def test_an_oversized_frame_is_refused(self):
        header = bytes([0x81, 127]) + struct.pack("!Q", MAX_PAYLOAD + 1)
        with self.assertRaises(WebSocketError):
            decode_frame(header + b"\x00" * 8)

    def test_close_frame_carries_a_code(self):
        decoded, _ = decode_frame(encode_close(CLOSE_NORMAL, "bitti"))
        self.assertEqual(decoded["opcode"], OP_CLOSE)
        self.assertEqual(struct.unpack("!H", decoded["payload"][:2])[0], CLOSE_NORMAL)

    def test_ping_opcode_survives(self):
        self.assertEqual(decode_frame(encode_frame(b"", opcode=OP_PING))[0]["opcode"],
                         OP_PING)

    def test_empty_payload_is_valid(self):
        self.assertEqual(decode_frame(encode_frame(b""))[0]["payload"], b"")


# ------------------------------------------------------------------- server
#: Liveness deadlines, not performance assertions. Long enough that a busy
#: desktop cannot fail them and short enough that a genuinely dead socket does
#: not hold the suite. The flake these tests used to have was not slowness — see
#: the handshake note below — but a deadline tight enough to fire on a stalled
#: machine would have hidden that behind a second, wrong explanation.
WIRE_TIMEOUT_S = 20.0


class WebSocketClient:
    """A minimal client, so the tests exercise the real socket path."""

    def __init__(self, host, port, timeout=WIRE_TIMEOUT_S):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.sock.sendall(
            b"GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n\r\n")
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise AssertionError("el sıkışma tamamlanmadan bağlantı kapandı")
            raw += chunk
        head, _, rest = raw.partition(b"\r\n\r\n")
        if not head.startswith(b"HTTP/1.1 101"):
            raise AssertionError(f"el sıkışma başarısız: {head[:60]!r}")
        # Whatever came after the header is already websocket data. Discarding it
        # ate the snapshot whenever the 101 and the first frame landed in one TCP
        # read — which is exactly the flake S9 spent a night not explaining: the
        # client then read every later frame one slot early, so the test waited
        # for a message that had already been thrown away. Rare, timing
        # dependent, and always "mesaj gelmedi" rather than a wrong message.
        self._buffer = rest

    def send(self, payload: dict) -> None:
        self.sock.sendall(client_mask(json.dumps(payload).encode("utf-8")))

    def send_raw(self, data: bytes) -> None:
        self.sock.sendall(data)

    def try_receive(self, timeout=WIRE_TIMEOUT_S) -> dict | None:
        """Next text message, or None when none arrived before the deadline."""
        deadline = time.monotonic() + timeout
        while True:
            decoded = decode_frame(self._buffer)
            if decoded is not None:
                frame, self._buffer = decoded
                if frame["opcode"] == OP_TEXT:
                    return json.loads(frame["payload"].decode("utf-8"))
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.sock.settimeout(max(0.05, remaining))
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                return None
            self._buffer += chunk

    def receive(self, timeout=WIRE_TIMEOUT_S) -> dict:
        message = self.try_receive(timeout=timeout)
        if message is None:
            raise AssertionError(f"zaman aşımı ({timeout:.0f}s): mesaj gelmedi")
        return message

    def receive_until(self, wire_type: str, timeout=WIRE_TIMEOUT_S) -> dict:
        """Skip whatever else is on the wire and wait for one type.

        The loop runs to the outer deadline instead of ending at the first inner
        timeout. The previous version called receive(), which raised on its own
        timeout, so "nothing for a moment" ended the wait rather than continuing
        it — and the assertion said only "mesaj gelmedi", which is why the flake
        took a soak to explain.
        """
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            message = self.try_receive(timeout=remaining)
            if message is None:
                break
            if message.get("type") == wire_type:
                return message
            seen.append(str(message.get("type")))
        raise AssertionError(f"{wire_type} gelmedi (görülen: {seen or 'hiçbir şey'})")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class TestBusServer(unittest.TestCase):
    def setUp(self):
        self.state = SharedState()
        self.bus = EventBus(self.state, ring_size=50, queue_size=50)
        self.server = BusServer(self.bus, host="127.0.0.1", port=0)
        self.assertTrue(self.server.start(), "sunucu başlamadı")
        self.host, self.port = self.server.address
        self.clients: list[WebSocketClient] = []

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.server.stop()

    def connect(self) -> WebSocketClient:
        client = WebSocketClient(self.host, self.port)
        self.clients.append(client)
        return client

    def test_a_new_client_gets_a_snapshot_first(self):
        """State first, then changes to it. Never the other way round."""
        message = self.connect().receive()
        self.assertEqual(message["type"], SNAPSHOT)
        self.assertIn("sections", message["payload"]["state"])

    def test_events_stream_after_the_snapshot(self):
        client = self.connect()
        client.receive()  # snapshot
        self.bus.publish(AGENT_STARTED, {"label": "başladı"})
        message = client.receive_until(AGENT_STARTED)
        self.assertEqual(message["payload"]["label"], "başladı")

    def test_ordering_is_preserved_over_the_wire(self):
        client = self.connect()
        client.receive()
        for index in range(10):
            self.bus.publish(AGENT_STARTED, {"i": index})
        seen = [client.receive_until(AGENT_STARTED)["payload"]["i"] for _ in range(10)]
        self.assertEqual(seen, list(range(10)))

    def test_multiple_clients_all_receive(self):
        clients = [self.connect() for _ in range(3)]
        for client in clients:
            client.receive()
        self.bus.publish(AGENT_STARTED, {"x": 1})
        for client in clients:
            self.assertEqual(client.receive_until(AGENT_STARTED)["payload"]["x"], 1)

    def test_a_disconnect_does_not_stop_the_backend(self):
        client = self.connect()
        client.receive()
        client.close()
        time.sleep(0.4)
        self.bus.publish(AGENT_STARTED)
        self.assertTrue(self.server.running)
        self.assertGreaterEqual(self.bus.seq, 1)

    def test_reconnect_gets_a_fresh_snapshot(self):
        first = self.connect()
        first.receive()
        self.bus.publish(AGENT_STARTED)
        first.close()

        second = self.connect()
        message = second.receive()
        self.assertEqual(message["type"], SNAPSHOT)
        self.assertGreaterEqual(message["seq"], 1)

    def test_resync_returns_a_new_snapshot(self):
        client = self.connect()
        client.receive()
        client.send({"op": "resync"})
        self.assertEqual(client.receive_until(SNAPSHOT)["type"], SNAPSHOT)

    def test_replay_delivers_what_was_missed(self):
        client = self.connect()
        client.receive()
        for _ in range(3):
            self.bus.publish(AGENT_STARTED)
        for _ in range(3):
            client.receive_until(AGENT_STARTED)

        client.send({"op": "replay", "seq": 1})
        replayed = client.receive_until(AGENT_STARTED)
        self.assertGreaterEqual(replayed["seq"], 2)

    def test_replay_beyond_history_falls_back_to_a_snapshot(self):
        bus = EventBus(self.state, ring_size=2)
        server = BusServer(bus, host="127.0.0.1", port=0)
        self.assertTrue(server.start())
        try:
            client = WebSocketClient(*server.address)
            self.clients.append(client)
            client.receive()
            for _ in range(20):
                bus.publish(AGENT_STARTED)
            client.send({"op": "replay", "seq": 1})
            self.assertEqual(client.receive_until(SNAPSHOT)["type"], SNAPSHOT)
        finally:
            server.stop()

    def test_a_frame_pipelined_with_the_handshake_is_not_lost(self):
        """The upgrade request and the first frame may share one TCP segment.

        Nothing in TCP separates them, and the handshake read is what finds the
        end of the headers — so the tail of that read can already be a message.
        This is the S9 flake, from the other side: the test client used to drop
        the same bytes and then read every later frame one slot early.
        """
        sock = socket.create_connection((self.host, self.port), timeout=WIRE_TIMEOUT_S)
        self.addCleanup(sock.close)
        sock.sendall(
            b"GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n\r\n"
            + client_mask(json.dumps({"op": "resync"}).encode("utf-8")))

        client = WebSocketClient.__new__(WebSocketClient)
        client.sock = sock
        raw = b""
        while b"\r\n\r\n" not in raw:
            raw += sock.recv(4096)
        head, _, rest = raw.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 101"), head[:40])
        client._buffer = rest

        # Two snapshots: the one every client gets, and the one the pipelined
        # resync asked for. Losing the request would leave only the first.
        self.assertEqual(client.receive()["type"], SNAPSHOT)
        self.assertEqual(client.receive_until(SNAPSHOT)["type"], SNAPSHOT)

    def test_the_test_client_keeps_bytes_that_follow_the_handshake(self):
        """The helper is part of the apparatus; a bug in it reads as a bug here."""
        client = self.connect()
        self.assertEqual(client.receive()["type"], SNAPSHOT)
        self.bus.publish(AGENT_STARTED, {"i": 0})
        self.assertEqual(client.receive_until(AGENT_STARTED)["payload"]["i"], 0)

    def test_a_malformed_message_is_ignored(self):
        client = self.connect()
        client.receive()
        client.send_raw(client_mask(b"{bu json degil"))
        client.send({"op": "resync"})
        self.assertEqual(client.receive_until(SNAPSHOT)["type"], SNAPSHOT)

    def test_an_unknown_command_is_ignored(self):
        client = self.connect()
        client.receive()
        client.send({"op": "kendini-imha-et"})
        client.send({"op": "resync"})
        self.assertEqual(client.receive_until(SNAPSHOT)["type"], SNAPSHOT)

    def test_a_plain_http_request_is_answered_without_killing_the_server(self):
        """Since S8 the same port also serves the page, so a plain GET is an HTTP
        request rather than a failed handshake. With no UI file configured it is
        a 404 — what must not happen is the socket dying over it."""
        raw = socket.create_connection((self.host, self.port), timeout=5)
        raw.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        response = raw.recv(4096)
        raw.close()
        self.assertTrue(response.startswith(b"HTTP/1.1 4"), response[:40])
        self.assertTrue(self.server.running)
        self.connect().receive()

    def _count_arrivals(self, client, expected: int,
                        timeout: float = WIRE_TIMEOUT_S) -> int:
        """How many of the burst actually arrived, within one honest deadline.

        Counts to the outer deadline rather than giving up at the first quiet
        second: a client that is merely being fed slowly is not a client that
        lost anything, and that is the distinction these tests are about.
        """
        seen = 0
        deadline = time.monotonic() + timeout
        while seen < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            message = client.try_receive(timeout=remaining)
            if message is None:
                break
            if message.get("type") == AGENT_STARTED:
                seen += 1
        return seen

    def test_a_burst_within_the_backlog_arrives_whole(self):
        """No loss while the client stays inside its allowance."""
        bus = EventBus(self.state, ring_size=1000, queue_size=1000)
        server = BusServer(bus, host="127.0.0.1", port=0)
        self.assertTrue(server.start())
        try:
            client = WebSocketClient(*server.address)
            self.clients.append(client)
            client.receive()
            for index in range(200):
                bus.publish(AGENT_STARTED, {"i": index})
            self.assertEqual(self._count_arrivals(client, 200), 200)
        finally:
            server.stop()

    def test_a_burst_beyond_the_backlog_drops_and_says_so(self):
        """Loss is the designed outcome for a reader that cannot keep up — but a
        silent stream with holes in it is not. The client is told the number."""
        bus = EventBus(self.state, ring_size=1000, queue_size=10)
        server = BusServer(bus, host="127.0.0.1", port=0)
        self.assertTrue(server.start())
        try:
            client = WebSocketClient(*server.address)
            self.clients.append(client)
            client.receive()
            for index in range(400):
                bus.publish(AGENT_STARTED, {"i": index})
            self.assertGreater(bus.dropped_total, 0)
            notice = client.receive_until(DROPPED, timeout=8)
            self.assertGreater(notice["payload"]["count"], 0)
        finally:
            server.stop()

    def test_the_server_survives_being_stopped_and_started(self):
        self.server.stop()
        self.assertFalse(self.server.running)
        self.assertTrue(self.server.start())
        self.host, self.port = self.server.address
        self.assertEqual(self.connect().receive()["type"], SNAPSHOT)

    def test_status_reports_connections(self):
        self.connect().receive()
        time.sleep(0.2)
        self.assertGreaterEqual(self.server.status()["kabul"], 1)


class TestServerFailSoft(unittest.TestCase):
    def test_a_busy_port_returns_false_rather_than_raising(self):
        """Losing the interface costs visibility, not function."""
        bus = EventBus(SharedState())
        first = BusServer(bus, host="127.0.0.1", port=0)
        self.assertTrue(first.start())
        host, port = first.address
        second = BusServer(EventBus(SharedState()), host=host, port=port)
        try:
            self.assertFalse(second.start())
        finally:
            first.stop()
            second.stop()

    def test_stopping_a_server_that_never_started_is_safe(self):
        BusServer(EventBus(SharedState()), port=0).stop()

    def test_the_backend_runs_with_no_server_at_all(self):
        bus = EventBus(SharedState())
        log = FakeLog()
        bus.attach(log, watch_state=False)
        for _ in range(50):
            log.publish(FakeEvent("agent", "run.start"))
        self.assertEqual(bus.published, 50)
        self.assertEqual(bus.subscriber_count, 0)


class TestConcurrency(unittest.TestCase):
    def test_publishing_from_many_threads_keeps_sequence_unique(self):
        bus = EventBus(SharedState(), ring_size=2000)
        seqs: list[int] = []
        lock = threading.Lock()

        def publisher():
            for _ in range(100):
                envelope = bus.publish(AGENT_STARTED)
                with lock:
                    seqs.append(envelope.seq)

        threads = [threading.Thread(target=publisher) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(seqs), 400)
        self.assertEqual(len(set(seqs)), 400)

    def test_subscribing_while_publishing_does_not_deadlock(self):
        bus = EventBus(SharedState())
        stop = threading.Event()

        def publisher():
            while not stop.is_set():
                bus.publish(AGENT_STARTED)

        thread = threading.Thread(target=publisher, daemon=True)
        thread.start()
        try:
            for _ in range(20):
                subscriber = bus.subscribe()
                bus.unsubscribe(subscriber)
        finally:
            stop.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
