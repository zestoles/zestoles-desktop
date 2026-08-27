"""The control centre: the page itself, and the server that hands it out.

A UI cannot be unit-tested the way a function can, so these check the properties
that actually decide whether it keeps working: that it has no dependencies to rot,
that it speaks the protocol the backend actually implements, that it says so when
it is disconnected, and that it invents nothing.

The last one gets its own test. A dashboard that fills in plausible numbers while
offline is worse than a blank one — it is confidently wrong about a machine you
are not watching any more.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import re
import socket
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jarvis.bus.bus import EventBus  # noqa: E402
from jarvis.bus.server import BusServer  # noqa: E402
from jarvis.bus.telemetry import TelemetryPump  # noqa: E402
from jarvis.bus.types import EVENT_TYPES, SNAPSHOT, SYSTEM_STATE_CHANGED  # noqa: E402
from jarvis.state import ACTIVITIES, SharedState  # noqa: E402

UI_PATH = ROOT / "ui" / "index.html"


def http_get(host, port, path="/", timeout=5.0) -> bytes:
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
                     f"Connection: close\r\n\r\n".encode("ascii"))
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        sock.close()


class TestUiFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = UI_PATH.read_text(encoding="utf-8")

    def test_the_file_exists(self):
        self.assertTrue(UI_PATH.is_file(), f"{UI_PATH} yok")

    def test_it_has_no_external_dependencies(self):
        """No CDN, no font host, no framework. A dependency is a page that stops
        opening in two years, and this one lives on a second monitor for a long time."""
        self.assertNotRegex(self.text, r"<script[^>]+src=")
        self.assertNotRegex(self.text, r"<link[^>]+href=\s*[\"']https?:")
        self.assertNotIn("cdn.", self.text)
        self.assertNotIn("googleapis", self.text)
        self.assertNotIn("unpkg", self.text)

    def test_the_only_external_url_is_the_websocket(self):
        urls = re.findall(r"https?://[^\s\"'<>)]+", self.text)
        remote = [u for u in urls if "127.0.0.1" not in u and "localhost" not in u]
        self.assertEqual(remote, [], f"dışarı bağlantı var: {remote}")

    def test_every_panel_the_brief_asked_for_is_present(self):
        for element_id in ("t-state", "t-mode", "t-link", "t-seq", "t-uptime",
                           "stream", "c-activity", "c-detail",
                           "r-now", "r-queue", "r-agent-model",
                           "m-cpu", "m-ram", "m-gpu", "m-vram",
                           "graph", "r-notes"):
            with self.subTest(element=element_id):
                self.assertIn(f'id="{element_id}"', self.text)

    def test_every_backend_activity_has_a_visual_state(self):
        """An activity the page cannot render shows as a hang."""
        for activity in ACTIVITIES:
            with self.subTest(activity=activity):
                self.assertIn(f'data-activity="{activity}"', self.text)

    def test_every_wire_event_type_is_labelled(self):
        for wire_type in EVENT_TYPES:
            with self.subTest(type=wire_type):
                self.assertIn(wire_type, self.text)

    def test_it_speaks_the_protocol_it_was_given(self):
        for operation in ('"resync"', '"replay"', '"snapshot"', '"dropped"',
                          '"heartbeat"'):
            with self.subTest(op=operation):
                self.assertIn(operation, self.text)

    def test_it_tracks_sequence_numbers(self):
        self.assertIn("lastSeq", self.text)
        self.assertIn("seq: state.lastSeq", self.text)

    def test_it_reconnects_with_backoff(self):
        self.assertIn("BACKOFF_MIN", self.text)
        self.assertIn("BACKOFF_MAX", self.text)
        self.assertIn("Math.pow(2", self.text)

    def test_a_snapshot_resets_the_sequence_downwards(self):
        """Found live: after a backend restart the server's sequence begins again
        at zero. A client that only moves its counter forward sits at the old
        number for hours, with gap detection silently disabled the whole time."""
        self.assertIn('envelope.type === "snapshot" || envelope.seq > state.lastSeq',
                      self.text)

    def test_resync_requests_are_throttled(self):
        """Asking for a snapshot per event would flood a busy minute."""
        self.assertIn("requestResync", self.text)
        self.assertIn("state.lastResync", self.text)

    def test_it_announces_being_offline(self):
        self.assertIn("ZESTOLES OFFLINE", self.text)
        self.assertIn("goOffline", self.text)

    def test_it_does_not_invent_data(self):
        """No random values, no synthetic node counts, no demo mode."""
        self.assertNotIn("Math.random", self.text)
        self.assertNotIn("demoData", self.text)
        self.assertNotIn("fakeData", self.text)
        self.assertNotIn("mockData", self.text)

    def test_the_graph_refuses_to_draw_without_real_notes(self):
        self.assertIn("notes < 3", self.text)
        self.assertIn("yeterli not yok", self.text)

    def test_it_respects_reduced_motion(self):
        self.assertIn("prefers-reduced-motion", self.text)

    def test_the_dom_is_bounded(self):
        """An always-on page must not grow a DOM node per event forever."""
        self.assertIn("MAX_ROWS", self.text)
        self.assertIn("removeChild(stream.lastChild)", self.text)


class TestUiServing(unittest.TestCase):
    def setUp(self):
        self.state = SharedState()
        self.bus = EventBus(self.state)
        self.server = BusServer(self.bus, host="127.0.0.1", port=0, ui_file=UI_PATH)
        self.assertTrue(self.server.start())
        self.host, self.port = self.server.address

    def tearDown(self):
        self.server.stop()

    def test_the_root_serves_the_page(self):
        response = http_get(self.host, self.port, "/")
        self.assertTrue(response.startswith(b"HTTP/1.1 200"))
        self.assertIn(b"text/html", response)
        self.assertIn(b"ZESTOLES", response)

    def test_index_html_serves_the_same_page(self):
        self.assertTrue(http_get(self.host, self.port, "/index.html")
                        .startswith(b"HTTP/1.1 200"))

    def test_query_strings_are_tolerated(self):
        self.assertTrue(http_get(self.host, self.port, "/?v=2")
                        .startswith(b"HTTP/1.1 200"))

    def test_the_zestoles_mark_is_served_from_the_fixed_asset_route(self):
        response = http_get(self.host, self.port, "/assets/zestoles-mark.png")
        self.assertTrue(response.startswith(b"HTTP/1.1 200"))
        self.assertIn(b"Content-Type: image/png", response)
        self.assertIn(b"\x89PNG\r\n\x1a\n", response)

    def test_any_other_path_is_refused(self):
        for path in ("/secrets", "/../config.json", "/vault/private.md",
                     "/assets/../config.json", "/assets/unknown.png"):
            with self.subTest(path=path):
                self.assertTrue(http_get(self.host, self.port, path)
                                .startswith(b"HTTP/1.1 404"))

    def test_serving_a_page_does_not_stop_the_socket(self):
        """One port, two protocols: the page must not disturb the stream."""
        http_get(self.host, self.port, "/")
        self.assertTrue(self.server.running)
        from test_bus import WebSocketClient

        client = WebSocketClient(self.host, self.port)
        try:
            self.assertEqual(client.receive()["type"], SNAPSHOT)
        finally:
            client.close()

    def test_a_server_without_a_ui_file_returns_404(self):
        bare = BusServer(EventBus(SharedState()), host="127.0.0.1", port=0)
        self.assertTrue(bare.start())
        try:
            host, port = bare.address
            self.assertTrue(http_get(host, port, "/").startswith(b"HTTP/1.1 404"))
        finally:
            bare.stop()


class FakeRuntime:
    def __init__(self, state, *, explode=False):
        self.state = state
        self.explode = explode
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1
        if self.explode:
            raise RuntimeError("yenileme patladı")
        self.state.update("system", resources={"cpu": 12.0, "ram": 30.0},
                          uptime_s=42.0)


class TestTelemetryPump(unittest.TestCase):
    def setUp(self):
        self.state = SharedState()
        self.bus = EventBus(self.state)
        self.runtime = FakeRuntime(self.state)
        self.pump = TelemetryPump(self.runtime, self.bus, interval_s=1.0)

    def test_it_does_nothing_when_nobody_is_watching(self):
        """A headless overnight run must not pay for an interface nobody opened."""
        self.assertFalse(self.pump.tick())
        self.assertEqual(self.runtime.refreshed, 0)
        self.assertEqual(self.pump.skipped, 1)

    def test_it_publishes_when_a_client_is_connected(self):
        subscriber = self.bus.subscribe()
        self.assertTrue(self.pump.tick())
        types = [e.type for e in subscriber.drain()]
        self.assertIn(SYSTEM_STATE_CHANGED, types)

    def test_the_published_frame_carries_resources(self):
        subscriber = self.bus.subscribe()
        self.pump.tick()
        frame = [e for e in subscriber.drain() if e.payload.get("telemetry")][0]
        self.assertEqual(frame.payload["resources"]["cpu"], 12.0)

    def test_a_failing_refresh_is_skipped_not_fatal(self):
        self.bus.subscribe()
        pump = TelemetryPump(FakeRuntime(self.state, explode=True), self.bus)
        self.assertFalse(pump.tick())
        self.assertEqual(pump.skipped, 1)

    def test_start_and_stop_are_clean(self):
        self.pump.start()
        self.assertTrue(self.pump.running)
        self.pump.stop()
        self.assertFalse(self.pump.running)

    def test_stopping_one_that_never_started_is_safe(self):
        TelemetryPump(self.runtime, self.bus).stop()


class TestBackendIndependence(unittest.TestCase):
    def test_the_bus_keeps_working_with_no_ui_and_no_server(self):
        """The interface is an observer. Nothing about it may be load-bearing."""
        bus = EventBus(SharedState())
        for _ in range(100):
            bus.publish(SYSTEM_STATE_CHANGED, {"activity": "idle"})
        self.assertEqual(bus.published, 100)
        self.assertEqual(bus.subscriber_count, 0)

    def test_a_client_that_vanishes_leaves_the_server_up(self):
        bus = EventBus(SharedState())
        server = BusServer(bus, host="127.0.0.1", port=0, ui_file=UI_PATH)
        self.assertTrue(server.start())
        try:
            from test_bus import WebSocketClient

            client = WebSocketClient(*server.address)
            client.receive()
            client.sock.close()          # vanish without a close frame
            time.sleep(0.4)
            bus.publish(SYSTEM_STATE_CHANGED, {"activity": "thinking"})
            self.assertTrue(server.running)
            survivor = WebSocketClient(*server.address, timeout=15)
            try:
                self.assertEqual(survivor.receive(timeout=15)["type"], SNAPSHOT)
            finally:
                survivor.close()
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
