"""The one route that reaches JARVIS from the page.

Everything else on this port travels outward: the page, then a stream of events.
This is the only thing coming back in, so it is the only thing that can be used
to make JARVIS do something — which is why the token, the size limit and the
"handler absent means the route does not exist" default are all tested here
rather than assumed.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.bus.bus import EventBus  # noqa: E402
from jarvis.state import SharedState  # noqa: E402
from jarvis.bus.server import (  # noqa: E402
    MAX_REQUEST_BYTES,
    REQUEST_PATH,
    TOKEN_MARKER,
    BusServer,
)
from jarvis.bus.websocket import parse_request  # noqa: E402


def raw_request(address, method, path, *, headers=None, body=b"", split=False):
    """Speak HTTP by hand. `split` sends the body in a second packet, which is
    the case that broke the handshake once."""
    supplied = headers or {}
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\n"
    for name, value in supplied.items():
        head += f"{name}: {value}\r\n"
    # Only when the caller did not set one: two Content-Length headers make the
    # server read whichever it saw last, which is a test artefact rather than
    # anything about the code under test.
    if not any(name.lower() == "content-length" for name in supplied):
        head += f"Content-Length: {len(body)}\r\n"
    head += "\r\n"

    sock = socket.create_connection(address, timeout=10)
    try:
        if split or not body:
            sock.sendall(head.encode("ascii"))
            if body:
                sock.sendall(body)
        else:
            sock.sendall(head.encode("ascii") + body)
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        sock.close()


def status_of(response: bytes) -> int:
    return int(response.split(b" ")[1])


def json_of(response: bytes):
    _, _, body = response.partition(b"\r\n\r\n")
    return json.loads(body.decode("utf-8"))


class RequestCase(unittest.TestCase):
    handler = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.page = Path(self._tmp.name) / "index.html"
        self.page.write_text(
            "<html><body>belirtec: " + TOKEN_MARKER.decode("ascii") + "</body></html>",
            encoding="utf-8")
        self.seen = []
        self.server = BusServer(
            EventBus(SharedState()), host="127.0.0.1", port=0, ui_file=self.page,
            request_handler=self.make_handler())
        self.assertTrue(self.server.start(), "sunucu baslamadi")
        self.address = self.server.address

    def tearDown(self):
        self.server.stop()
        self._tmp.cleanup()

    def make_handler(self):
        def handler(payload):
            self.seen.append(payload)
            return {"cevap": f"alindi: {payload.get('mesaj', '')}"}

        return handler

    def post(self, payload, *, token=None, **kwargs):
        body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) \
            else payload
        headers = {"X-Jarvis-Token": self.server.token if token is None else token}
        return raw_request(self.address, "POST", REQUEST_PATH,
                           headers=headers, body=body, **kwargs)


class TestTheRouteWorks(RequestCase):
    def test_a_request_reaches_the_handler_and_the_answer_comes_back(self):
        response = self.post({"mesaj": "merhaba"})
        self.assertEqual(status_of(response), 200)
        self.assertEqual(json_of(response)["cevap"], "alindi: merhaba")
        self.assertEqual(self.seen, [{"mesaj": "merhaba"}])

    def test_a_body_that_arrives_after_the_headers_is_not_lost(self):
        """Nothing in TCP separates the head from the body. The read that finds
        the end of the headers may hold none of it, or all of it."""
        response = self.post({"mesaj": "ayri paket"}, split=True)
        self.assertEqual(status_of(response), 200)
        self.assertEqual(self.seen, [{"mesaj": "ayri paket"}])

    def test_the_response_is_json(self):
        response = self.post({"mesaj": "x"})
        self.assertIn(b"application/json", response.split(b"\r\n\r\n")[0])

    def test_the_counter_moves(self):
        self.post({"mesaj": "bir"})
        self.post({"mesaj": "iki"})
        self.assertEqual(self.server.requests, 2)


class TestTheGuard(RequestCase):
    def test_a_request_without_a_token_is_refused(self):
        response = raw_request(self.address, "POST", REQUEST_PATH,
                               body=b'{"mesaj": "x"}')
        self.assertEqual(status_of(response), 403)
        self.assertEqual(self.seen, [], "belirtecsiz istek isleyiciye ulasmis")

    def test_a_wrong_token_is_refused(self):
        response = self.post({"mesaj": "x"}, token="yanlis")
        self.assertEqual(status_of(response), 403)
        self.assertEqual(self.seen, [])
        self.assertEqual(self.server.refused_requests, 1)

    def test_the_token_is_not_guessable_in_length(self):
        self.assertGreaterEqual(len(self.server.token), 24)

    def test_each_server_mints_its_own(self):
        other = BusServer(EventBus(SharedState()), host="127.0.0.1", port=0)
        self.assertNotEqual(self.server.token, other.token)

    def test_the_page_is_handed_the_real_token(self):
        response = raw_request(self.address, "GET", "/")
        self.assertEqual(status_of(response), 200)
        self.assertIn(self.server.token.encode("ascii"), response)
        self.assertNotIn(TOKEN_MARKER, response)


class TestBadInput(RequestCase):
    def test_a_body_that_is_not_json_is_refused(self):
        response = self.post(b"bu json degil")
        self.assertEqual(status_of(response), 400)
        self.assertEqual(self.seen, [])

    def test_a_body_that_is_not_an_object_is_refused(self):
        response = self.post(b'["liste"]')
        self.assertEqual(status_of(response), 400)
        self.assertEqual(self.seen, [])

    def test_an_oversized_body_is_refused_without_being_buffered(self):
        response = raw_request(
            self.address, "POST", REQUEST_PATH,
            headers={"X-Jarvis-Token": self.server.token,
                     "Content-Length": str(MAX_REQUEST_BYTES + 1)},
            body=b"")
        self.assertEqual(status_of(response), 413)
        self.assertEqual(self.seen, [])

    def test_an_unknown_path_is_not_the_route(self):
        response = raw_request(self.address, "POST", "/baska",
                               headers={"X-Jarvis-Token": self.server.token},
                               body=b"{}")
        self.assertEqual(status_of(response), 404)


class TestAHandlerThatBreaks(RequestCase):
    def make_handler(self):
        def handler(payload):
            self.seen.append(payload)
            raise RuntimeError("isleyici coktu")

        return handler

    def test_the_failure_is_reported_and_the_server_survives(self):
        response = self.post({"mesaj": "x"})
        self.assertEqual(status_of(response), 500)
        self.assertIn("isleyici coktu", json_of(response)["hata"])
        self.assertTrue(self.server.running)

    def test_the_next_request_is_still_served(self):
        self.post({"mesaj": "bir"})
        self.post({"mesaj": "iki"})
        self.assertEqual(len(self.seen), 2)


class TestWithoutAHandlerTheRouteDoesNotExist(unittest.TestCase):
    """A telemetry-only run must stay an observer."""

    def setUp(self):
        self.server = BusServer(EventBus(SharedState()), host="127.0.0.1", port=0)
        self.assertTrue(self.server.start())

    def tearDown(self):
        self.server.stop()

    def test_posting_gets_a_404(self):
        response = raw_request(self.server.address, "POST", REQUEST_PATH,
                               headers={"X-Jarvis-Token": self.server.token},
                               body=b"{}")
        self.assertEqual(status_of(response), 404)


class TestTheOldBehaviourIsIntact(unittest.TestCase):
    def test_a_get_is_still_a_get(self):
        headers = parse_request(b"GET /ws HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertEqual(headers["__method__"], "GET")
        self.assertEqual(headers["__path__"], "/ws")

    def test_a_post_now_parses_instead_of_vanishing(self):
        headers = parse_request(b"POST /istek HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertEqual(headers["__method__"], "POST")

    def test_garbage_is_still_not_a_request(self):
        self.assertEqual(parse_request(b"\x00\x01\x02"), {})

    def test_a_request_line_without_a_version_is_not_a_request(self):
        self.assertEqual(parse_request(b"GET /\r\n\r\n"), {})

    def test_a_page_without_the_marker_is_served_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "index.html"
            page.write_text("<html>eski panel</html>", encoding="utf-8")
            server = BusServer(EventBus(SharedState()), host="127.0.0.1", port=0, ui_file=page)
            self.assertTrue(server.start())
            try:
                response = raw_request(server.address, "GET", "/")
                self.assertIn(b"eski panel", response)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()



class TestThePanelHasItsOwnAddress(unittest.TestCase):
    """The main screen must never be the diagnostics screen. Two files, two
    routes, and nothing that turns a path into a directory walk."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.main = base / "jarvis.html"
        self.main.write_text("<html>ASISTAN</html>", encoding="utf-8")
        self.panel = base / "index.html"
        self.panel.write_text("<html>PANEL</html>", encoding="utf-8")
        self.server = BusServer(EventBus(SharedState()), host="127.0.0.1", port=0,
                                ui_file=self.main, panel_file=self.panel)
        self.assertTrue(self.server.start())

    def tearDown(self):
        self.server.stop()
        self._tmp.cleanup()

    def get(self, path):
        return raw_request(self.server.address, "GET", path)

    def test_the_root_is_the_assistant(self):
        self.assertIn(b"ASISTAN", self.get("/"))

    def test_the_panel_is_somewhere_else(self):
        self.assertIn(b"PANEL", self.get("/panel"))

    def test_a_trailing_slash_still_finds_the_panel(self):
        self.assertIn(b"PANEL", self.get("/panel/"))

    def test_a_query_string_does_not_hide_the_route(self):
        self.assertIn(b"PANEL", self.get("/panel?x=1"))

    def test_no_other_path_is_served(self):
        for path in ("/gizli", "/../config.json", "/ui/index.html",
                     "/panel/../../config.json", "/data/jarvis.db"):
            with self.subTest(path=path):
                self.assertEqual(status_of(self.get(path)), 404)

    def test_a_missing_panel_is_a_404_not_a_crash(self):
        server = BusServer(EventBus(SharedState()), host="127.0.0.1", port=0,
                           ui_file=self.main)
        self.assertTrue(server.start())
        try:
            self.assertEqual(status_of(raw_request(server.address, "GET", "/panel")), 404)
        finally:
            server.stop()
