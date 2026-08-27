"""The websocket server: one thread accepting, one thread per client.

## What a client sees

On connecting it receives a `snapshot` frame carrying the full runtime state and
the sequence number that snapshot is current as of. Everything after that is a
stream of typed events with increasing `seq`. That ordering is the contract: state
first, then changes to it, never the other way round.

A client that fell behind and lost messages receives a `dropped` frame telling it
how many, and may then ask for a fresh `snapshot`. A client that reconnects can ask
to replay from a sequence number it already has, and gets either the missed events
or — when the gap is wider than the ring — a snapshot instead.

    → {"op": "resync"}            fresh snapshot
    → {"op": "replay", "seq": N}  what was missed since N, or a snapshot
    → {"op": "ping"}              a pong, for liveness checks from the page

## What the backend sees

By default, nothing. The server holds a subscriber queue and a socket; it never
calls into a domain module and nothing calls into it. Killing every client, or
never starting the server at all, changes nothing about what JARVIS does — which
is the property that makes it safe to leave running.

A caller that wants the page to *ask* for something supplies `request_handler`,
and then one POST route exists. The server still knows nothing about what the
handler does: it decodes JSON, hands it over, and encodes whatever comes back.
Without a handler the route is a 404 and the observer property above is exactly
as it was.

That route needs a guard the event stream never did. Both are loopback-only, but
events travel outward and a request runs something, and every process on the
machine can reach loopback. So a per-run token is minted at construction and
substituted into the page as it is served; a POST without it is refused. This is
a same-machine guard, not remote authentication; see the limits documented in
docs/SECURITY-MODEL.md.

## Binding

Loopback only, by default and by intent. This socket carries live runtime state and
accepts commands; exposing it to the network would be handing out a window into the
machine with no authentication behind it.
"""

from __future__ import annotations

import json
import logging
import secrets
import socket
import threading
import time
from pathlib import Path

#: The one route that reaches JARVIS from the page.
REQUEST_PATH = "/istek"
#: A request body larger than this is refused rather than buffered.
MAX_REQUEST_BYTES = 256 * 1024
#: Replaced with the run's token as the page is served.
TOKEN_MARKER = b"__JARVIS_TOKEN__"
#: Where the developer dashboard lives when one is supplied. Kept off "/" so
#: the main screen is never the diagnostics screen: the V1 spec is explicit
#: that those are two different audiences.
PANEL_PATH = "/panel"
# Public interface assets are an explicit allow-list.  Keeping this table fixed
# preserves the original no-directory-walk boundary while allowing the product
# mark to live outside the HTML document.
STATIC_ASSETS = {
    "/assets/zestoles-mark.png": ("assets/zestoles-mark.png", "image/png"),
}

from .bus import EventBus, Subscriber  # noqa: E402
from .types import HEARTBEAT, Envelope
from .websocket import (
    ConnectionClosed,
    WebSocketConnection,
    WebSocketError,
    handshake_response,
    is_upgrade,
    read_request_and_rest,
)


def _close_quietly(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass

log = logging.getLogger("jarvis.bus.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8797
HEARTBEAT_S = 20.0
#: How long a client thread waits on its queue before checking whether it should
#: stop. Short enough to shut down promptly, long enough not to spin.
POLL_S = 0.5


class ClientHandler:
    def __init__(self, connection: WebSocketConnection, bus: EventBus,
                 subscriber: Subscriber, stop: threading.Event) -> None:
        self.connection = connection
        self.bus = bus
        self.subscriber = subscriber
        self.stop = stop
        self.sent = 0

    def run(self) -> None:
        try:
            self._send(self.bus.snapshot())
            reader = threading.Thread(target=self._read_loop, daemon=True,
                                      name="jarvis-ws-read")
            reader.start()
            self._write_loop()
        except ConnectionClosed as exc:
            log.debug("istemci ayrıldı: %s", exc)
        except Exception as exc:  # noqa: BLE001 - one client cannot take the server
            log.warning("istemci hatası: %s", exc)
        finally:
            self.bus.unsubscribe(self.subscriber)
            self.connection.close()

    def _send(self, envelope: Envelope) -> None:
        self.connection.send_text(envelope.to_json())
        self.sent += 1

    def _write_loop(self) -> None:
        last_beat = time.monotonic()
        while not self.stop.is_set() and self.connection.open:
            envelope = self.subscriber.get(timeout=POLL_S)
            if envelope is not None:
                self._send(envelope)
                # Drain what else is waiting before going back to the timed wait.
                # Without this a burst is delivered at one message per poll and
                # the queue fills behind a reader that is not actually slow.
                for extra in self.subscriber.drain():
                    self._send(extra)
                notice = self.bus.dropped_notice(self.subscriber)
                if notice is not None:
                    self.subscriber.dropped = 0
                    self._send(notice)
                continue

            # Idle. A heartbeat proves the connection to a client that would
            # otherwise have no way to tell a quiet system from a dead socket.
            if time.monotonic() - last_beat >= HEARTBEAT_S:
                last_beat = time.monotonic()
                self._send(Envelope(self.bus.seq, HEARTBEAT, time.time(),
                                    {"activity": self.bus.state.activity}))

    def _read_loop(self) -> None:
        while not self.stop.is_set() and self.connection.open:
            try:
                message = self.connection.receive(timeout=POLL_S)
            except ConnectionClosed:
                return
            except Exception as exc:  # noqa: BLE001
                log.debug("istemciden okunamadı: %s", exc)
                return
            if message:
                self._handle(message)

    def _handle(self, message: str) -> None:
        """Client commands. Anything unrecognised is ignored, not fatal."""
        try:
            request = json.loads(message)
            operation = str(request.get("op", "")).casefold()
        except (json.JSONDecodeError, AttributeError):
            log.debug("bozuk istemci mesajı yok sayıldı")
            return

        try:
            if operation == "resync":
                self._send(self.bus.snapshot())
            elif operation == "replay":
                self._replay(int(request.get("seq", 0)))
            elif operation == "ping":
                self._send(Envelope(self.bus.seq, HEARTBEAT, time.time(), {"pong": True}))
        except (ConnectionClosed, ValueError, TypeError) as exc:
            log.debug("istemci komutu işlenemedi: %s", exc)

    def _replay(self, known_seq: int) -> None:
        missed = self.bus.replay_from(known_seq)
        if missed is None:
            # The gap is wider than history. A snapshot is the honest answer; a
            # stream with an invisible hole in it is not.
            self._send(self.bus.snapshot())
            return
        for envelope in missed:
            self._send(envelope)


class BusServer:
    def __init__(self, bus: EventBus, *, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT, ui_file: Path | None = None,
                 panel_file: Path | None = None, request_handler=None) -> None:
        self.bus = bus
        self.host = host
        self.port = port
        self.ui_file = Path(ui_file) if ui_file else None
        self.panel_file = Path(panel_file) if panel_file else None
        #: Called with the decoded JSON body of a POST to REQUEST_PATH and
        #: expected to return something JSON-serialisable. None means the
        #: endpoint does not exist, which is what a telemetry-only run wants:
        #: the interface stays an observer unless something opts it in.
        self.request_handler = request_handler
        #: Minted per run and handed to the page that is served. The stream is
        #: loopback-only and always was, but it used to carry events outward and
        #: nothing inward. An endpoint that runs tools is a different thing to
        #: leave open to every process on the machine, so a caller has to show
        #: it read the page it claims to be.
        self.token = secrets.token_urlsafe(24)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: list[threading.Thread] = []
        self.accepted = 0
        self.rejected = 0
        self.served = 0
        self.requests = 0
        self.refused_requests = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def address(self) -> tuple[str, int]:
        if self._socket is None:
            return (self.host, self.port)
        return self._socket.getsockname()[:2]

    def start(self) -> bool:
        """Bind and begin accepting. Returns False rather than raising on failure.

        A port already in use must not stop JARVIS from running: the interface is
        an observer, and losing it costs visibility, not function.
        """
        if self.running:
            return True
        server: socket.socket | None = None
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # SO_REUSEADDR means opposite things on the two platforms. On Windows
            # it permits binding a port another socket is already listening on,
            # so two servers would silently split the incoming connections
            # between them. SO_EXCLUSIVEADDRUSE is the Windows way to say what
            # SO_REUSEADDR says on POSIX.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(8)
            server.settimeout(POLL_S)
        except OSError as exc:
            if server is not None:
                _close_quietly(server)
            log.warning("websocket sunucusu başlatılamadı (%s:%s): %s",
                        self.host, self.port, exc)
            return False

        self._socket = server
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True,
                                        name="jarvis-ws-accept")
        self._thread.start()
        log.info("websocket sunucusu dinliyor: ws://%s:%s", *self.address)
        return True

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

        # Client threads wake within one poll and exit on the stop flag. Waiting
        # for them means a stopped server has actually released its sockets — a
        # stop() that leaves threads draining in the background makes the next
        # start() race against the previous run.
        deadline = time.monotonic() + timeout
        for thread in list(self._clients):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        self._clients = [t for t in self._clients if t.is_alive()]
        if self._clients:
            log.debug("%s istemci thread'i hâlâ kapanıyor", len(self._clients))
        log.info("websocket sunucusu durdu")

    def _accept_loop(self) -> None:
        while not self._stop.is_set() and self._socket is not None:
            try:
                client, address = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return  # socket closed by stop()

            thread = threading.Thread(target=self._serve, args=(client, address),
                                      daemon=True, name="jarvis-ws-client")
            thread.start()
            self._clients = [t for t in self._clients if t.is_alive()]
            self._clients.append(thread)

    def _serve(self, client: socket.socket, address) -> None:
        try:
            headers, pipelined = read_request_and_rest(client)
        except (WebSocketError, OSError) as exc:
            self.rejected += 1
            log.debug("istek okunamadı (%s): %s", address, exc)
            _close_quietly(client)
            return

        # One port, two protocols. The page and the socket it opens are then the
        # same origin, which removes a whole class of "works in one browser"
        # problems for nothing more than a branch here.
        if not is_upgrade(headers):
            if headers.get("__method__") == "POST":
                self._serve_request(client, headers, pipelined)
            else:
                self._serve_static(client, headers.get("__path__", "/"))
            return

        try:
            response = handshake_response(headers)
            client.sendall(response)
            if not response.startswith(b"HTTP/1.1 101"):
                raise WebSocketError("geçersiz upgrade")
        except (WebSocketError, OSError) as exc:
            self.rejected += 1
            log.debug("el sıkışma reddedildi (%s): %s", address, exc)
            _close_quietly(client)
            return

        self.accepted += 1
        connection = WebSocketConnection(client, address, buffered=pipelined)
        subscriber = self.bus.subscribe()
        ClientHandler(connection, self.bus, subscriber, self._stop).run()

    def _serve_request(self, client: socket.socket, headers: dict[str, str],
                       already: bytes) -> None:
        """Answer one POST. The only way anything reaches JARVIS from the page."""
        try:
            if self.request_handler is None or \
                    headers.get("__path__", "").split("?")[0] != REQUEST_PATH:
                self._send_json(client, 404, {"hata": "yok"})
                return
            if not secrets.compare_digest(headers.get("x-jarvis-token", ""),
                                          self.token):
                self.refused_requests += 1
                log.warning("belirteçsiz istek reddedildi")
                self._send_json(client, 403, {"hata": "belirteç geçersiz"})
                return

            body = self._read_body(client, headers, already)
            if body is None:
                self._send_json(client, 413, {"hata": "gövde çok büyük"})
                return
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(client, 400, {"hata": f"gövde okunamadı: {exc}"})
                return
            if not isinstance(payload, dict):
                self._send_json(client, 400, {"hata": "gövde bir nesne değil"})
                return

            self.requests += 1
            try:
                answer = self.request_handler(payload)
            except Exception as exc:  # noqa: BLE001 - a handler must not kill the server
                log.exception("istek işleyici hata verdi")
                self._send_json(client, 500, {"hata": f"{type(exc).__name__}: {exc}"})
                return
            self._send_json(client, 200, answer if isinstance(answer, dict) else
                            {"sonuc": answer})
        except OSError as exc:
            log.debug("istek yanıtlanamadı: %s", exc)
        finally:
            _close_quietly(client)

    def _read_body(self, client: socket.socket, headers: dict[str, str],
                   already: bytes) -> bytes | None:
        """The rest of the body, or None when it is larger than allowed.

        `already` matters for the same reason it does in the handshake: nothing
        in TCP separates the headers from the body that follows them, so the
        read that found the end of the head may already hold part of it.
        """
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            length = 0
        if length > MAX_REQUEST_BYTES:
            return None
        body = bytearray(already[:length])
        while len(body) < length:
            chunk = client.recv(min(4096, length - len(body)))
            if not chunk:
                break
            body.extend(chunk)
        return bytes(body)

    def _send_json(self, client: socket.socket, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        reason = {200: "OK", 400: "Bad Request", 403: "Forbidden",
                  404: "Not Found", 413: "Payload Too Large",
                  500: "Internal Server Error"}.get(status, "OK")
        client.sendall(
            f"HTTP/1.1 {status} {reason}\r\n".encode("ascii")
            + b"Content-Type: application/json; charset=utf-8\r\n"
            + b"Cache-Control: no-store\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n" + body)

    def _static_for(self, path: str) -> tuple[Path, str, bool] | None:
        """Which fixed file answers this path, its type, and token policy.

        A fixed table rather than a lookup on disk: no route here can be turned
        into a directory walk by a crafted path.
        """
        route = path.split("?")[0].rstrip("/") or "/"
        if route in ("/", "/index.html"):
            return ((self.ui_file, "text/html; charset=utf-8", True)
                    if self.ui_file is not None else None)
        if route == PANEL_PATH:
            return ((self.panel_file, "text/html; charset=utf-8", True)
                    if self.panel_file is not None else None)
        asset = STATIC_ASSETS.get(route)
        anchor = self.ui_file or self.panel_file
        if asset is not None and anchor is not None:
            relative, content_type = asset
            return anchor.parent / relative, content_type, False
        return None

    def _serve_static(self, client: socket.socket, path: str) -> None:
        """Hand out a page. Read-only, fixed routes, no directory traversal."""
        self.served += 1
        try:
            static = self._static_for(path)
            if static is None or not static[0].is_file():
                client.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                               b"Connection: close\r\n\r\n")
                return
            page, content_type, inject_token = static
            body = page.read_bytes()
            # The page cannot be shipped with the token, so it carries a marker
            # and gets the real one as it is handed over. A page without the
            # marker is served unchanged, which is how the existing dashboard
            # keeps working without knowing any of this exists.
            if inject_token:
                body = body.replace(TOKEN_MARKER, self.token.encode("ascii"))
            client.sendall(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Type: {content_type}\r\n".encode("ascii")
                + b"Cache-Control: no-store\r\n"
                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n" + body)
        except OSError as exc:
            log.debug("arayüz sunulamadı: %s", exc)
        finally:
            _close_quietly(client)

    def status(self) -> dict[str, object]:
        return {
            "calisiyor": self.running,
            "adres": f"ws://{self.address[0]}:{self.address[1]}",
            "kabul": self.accepted,
            "red": self.rejected,
            "sayfa": self.served,
            "arayuz": str(self.ui_file) if self.ui_file else None,
            "istemci": sum(1 for t in self._clients if t.is_alive()),
        }
