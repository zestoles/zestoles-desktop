"""A small RFC 6455 server, implemented rather than depended on.

## Why not a library

The codebase is threaded end to end — the scheduler, agent runs, research fetches
all run on plain threads and the event log calls subscribers synchronously. The
usual websocket libraries are asyncio, and bridging the two would mean an event
loop in a second thread plus a queue in each direction, which is more moving parts
than the protocol subset actually needs.

What is needed is small: accept an upgrade, send text frames, read the occasional
text frame back, answer pings, close cleanly. That is implemented here and tested
directly, because the framing is pure functions over bytes.

## What is supported, and what is not

Supported: RFC 6455 version 13, text frames, continuation frames when reading,
ping/pong, close with status code, payloads up to 64 bits of length.

Not supported: extensions (no permessage-deflate), subprotocol negotiation, binary
frames beyond passing them through as bytes, TLS. The server binds to loopback and
that is where it is meant to stay — a websocket reachable from the network is a
door into this machine's runtime state.

## Framing details that bite

A client MUST mask its frames and a server MUST NOT mask its own. Getting that
backwards produces a connection that appears to work until the browser closes it
with a protocol error. Control frames may arrive interleaved between fragments of
a message, so reading a message means looping past them rather than assuming the
next frame continues the previous one.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import select
import socket
import struct
import threading
import time

log = logging.getLogger("jarvis.bus.websocket")

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

CLOSE_NORMAL = 1000
CLOSE_GOING_AWAY = 1001
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_TOO_BIG = 1009

#: Refuse anything larger rather than allocating for it. A client is expected to
#: send short control messages; a 100 MB frame is a mistake or an attack.
MAX_PAYLOAD = 1_000_000


class WebSocketError(RuntimeError):
    pass


class ConnectionClosed(WebSocketError):
    pass


# ------------------------------------------------------------------- handshake
def accept_key(client_key: str) -> str:
    """The value RFC 6455 requires in Sec-WebSocket-Accept."""
    # SHA-1 is mandated by RFC 6455 for this non-security handshake transform;
    # it is not used for signatures, passwords, or integrity decisions.
    digest = hashlib.sha1(  # nosec B324
        f"{client_key.strip()}{GUID}".encode("utf-8"), usedforsecurity=False
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def parse_request(raw: bytes) -> dict[str, str]:
    """Headers from an HTTP request, lowercased. Empty when it is not one.

    The method is kept rather than assumed. This used to accept only GET, so
    every other verb parsed as nothing and fell through to "serve the page" — a
    POST got HTML back instead of an answer. What now tells a socket that spoke
    noise apart from one that spoke HTTP is the request line's shape:
    METHOD SP TARGET SP HTTP/x.y — three parts, version last.
    """
    try:
        text = raw.decode("latin-1")
    except UnicodeDecodeError:
        return {}
    lines = text.split("\r\n")
    if not lines:
        return {}
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].upper().startswith("HTTP/"):
        return {}
    headers: dict[str, str] = {"__method__": parts[0].upper(),
                               "__path__": parts[1] or "/"}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        name, _, value = line.partition(":")
        headers[name.strip().casefold()] = value.strip()
    return headers


def handshake_response(headers: dict[str, str]) -> bytes:
    """The 101 response, or a 400 when the request is not a valid upgrade."""
    key = headers.get("sec-websocket-key", "")
    upgrade = headers.get("upgrade", "").casefold()
    version = headers.get("sec-websocket-version", "")

    if not key or upgrade != "websocket" or version != "13":
        return (b"HTTP/1.1 400 Bad Request\r\n"
                b"Content-Length: 0\r\n"
                b"Connection: close\r\n\r\n")

    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n"
    ).encode("ascii")


# ---------------------------------------------------------------------- frames
def encode_frame(payload: bytes, *, opcode: int = OP_TEXT, fin: bool = True) -> bytes:
    """A server frame. Never masked — a masked server frame is a protocol error."""
    header = bytearray()
    header.append((0x80 if fin else 0x00) | (opcode & 0x0F))

    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < (1 << 16):
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def encode_close(code: int = CLOSE_NORMAL, reason: str = "") -> bytes:
    return encode_frame(struct.pack("!H", code) + reason.encode("utf-8"),
                        opcode=OP_CLOSE)


def decode_frame(data: bytes) -> tuple[dict, bytes] | None:
    """Decode one frame from the front of `data`.

    Returns (frame, remaining) or None when more bytes are needed. Never raises on
    a truncated buffer: a partial read is the normal case on a stream socket.
    """
    if len(data) < 2:
        return None

    first, second = data[0], data[1]
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    offset = 2

    if length == 126:
        if len(data) < offset + 2:
            return None
        length = struct.unpack("!H", data[offset:offset + 2])[0]
        offset += 2
    elif length == 127:
        if len(data) < offset + 8:
            return None
        length = struct.unpack("!Q", data[offset:offset + 8])[0]
        offset += 8

    if length > MAX_PAYLOAD:
        raise WebSocketError(f"çerçeve fazla büyük: {length} bayt")

    mask_key = b""
    if masked:
        if len(data) < offset + 4:
            return None
        mask_key = data[offset:offset + 4]
        offset += 4

    if len(data) < offset + length:
        return None

    payload = bytearray(data[offset:offset + length])
    if masked:
        for index in range(length):
            payload[index] ^= mask_key[index % 4]

    frame = {"fin": fin, "opcode": opcode, "payload": bytes(payload), "masked": masked}
    return frame, data[offset + length:]


# ------------------------------------------------------------------ connection
#: A send that has not completed in this long means the peer is gone or wedged.
#: Generous on purpose — see the note in WebSocketConnection about why it must
#: never be short.
SEND_TIMEOUT_S = 30.0


class WebSocketConnection:
    """One client. Sends are serialised; reads happen on the owning thread.

    ## The socket timeout is set once and never changed

    A socket's timeout is a property of the socket, not of the call — so a reader
    thread that lowers it to poll briefly also lowers it for the writer thread
    sharing that socket. This was a real fault: read polling set the timeout to
    half a second, and under load a `sendall` that took longer raised
    `socket.timeout`, which is an `OSError`, which the send path correctly read as
    a dead connection and tore the client down.

    Reads therefore wait on `select` and leave the timeout alone. The socket keeps
    one generous send timeout for its whole life.
    """

    def __init__(self, sock: socket.socket, address, *, buffered: bytes = b"") -> None:
        self.sock = sock
        self.address = address
        self.open = True
        self._send_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        # Bytes the handshake read past the end of the request head. TCP does not
        # promise the upgrade request and the first frame arrive in separate
        # reads, and a client that pipelines them would otherwise have its first
        # message silently dropped.
        self._buffer = bytes(buffered)
        self.sock.settimeout(SEND_TIMEOUT_S)

    # -------------------------------------------------------------- sending
    def send_text(self, text: str) -> None:
        self._send(encode_frame(text.encode("utf-8"), opcode=OP_TEXT))

    def send_ping(self, payload: bytes = b"") -> None:
        self._send(encode_frame(payload, opcode=OP_PING))

    def _send(self, frame: bytes) -> None:
        if not self.open:
            raise ConnectionClosed("bağlantı kapalı")
        with self._send_lock:
            try:
                self.sock.sendall(frame)
            except OSError as exc:
                self.open = False
                raise ConnectionClosed(str(exc)) from exc

    def close(self, code: int = CLOSE_NORMAL, reason: str = "") -> None:
        # A failed send/read marks ``open`` false before the owner reaches its
        # finally block.  Returning early in that state leaked the underlying
        # socket; long GUI sessions and the test suite both exposed it as a
        # ResourceWarning.  Closed and protocol-open are different facts.
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            was_open = self.open
            self.open = False
            if was_open:
                try:
                    with self._send_lock:
                        self.sock.sendall(encode_close(code, reason))
                except OSError:
                    pass
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()

    # -------------------------------------------------------------- reading
    def receive(self, timeout: float | None = None) -> str | None:
        """Next text message, or None on timeout. Answers control frames inline.

        Waits with `select` rather than a socket timeout: the socket is shared
        with the writer thread and changing its timeout here would change the
        writer's too.
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        message = bytearray()
        while self.open:
            decoded = decode_frame(self._buffer)
            if decoded is None:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                else:
                    remaining = None
                try:
                    ready, _, _ = select.select([self.sock], [], [], remaining)
                except (OSError, ValueError) as exc:
                    self.open = False
                    raise ConnectionClosed(str(exc)) from exc
                if not ready:
                    return None
                try:
                    chunk = self.sock.recv(65536)
                except OSError as exc:
                    self.open = False
                    raise ConnectionClosed(str(exc)) from exc
                if not chunk:
                    self.open = False
                    raise ConnectionClosed("istemci bağlantıyı kapattı")
                self._buffer += chunk
                continue

            frame, self._buffer = decoded
            opcode = frame["opcode"]

            if opcode == OP_CLOSE:
                self.open = False
                raise ConnectionClosed("istemci kapanış çerçevesi gönderdi")
            if opcode == OP_PING:
                self._send(encode_frame(frame["payload"], opcode=OP_PONG))
                continue
            if opcode == OP_PONG:
                continue

            message.extend(frame["payload"])
            if frame["fin"]:
                return message.decode("utf-8", errors="replace")
        return None


def read_request_and_rest(sock: socket.socket, *, timeout: float = 10.0
                          ) -> tuple[dict[str, str], bytes]:
    """Read one HTTP request head, and hand back whatever came after it.

    The remainder is the point. Nothing in TCP separates the upgrade request from
    the frame a client sends immediately after it, so the read that finds the end
    of the headers may already hold the first message. Throwing that tail away
    loses the message with nothing to show for it — no error, no log line, just a
    client whose first command never happened.
    """
    sock.settimeout(timeout)
    raw = b""
    while b"\r\n\r\n" not in raw:
        try:
            chunk = sock.recv(4096)
        except (OSError, socket.timeout) as exc:
            raise WebSocketError(f"istek okunamadı: {exc}") from exc
        if not chunk:
            raise WebSocketError("istek tamamlanmadan bağlantı kapandı")
        raw += chunk
        if len(raw) > 16384:
            raise WebSocketError("istek başlıkları fazla büyük")
    head, _, rest = raw.partition(b"\r\n\r\n")
    return parse_request(head + b"\r\n\r\n"), rest


def read_request(sock: socket.socket, *, timeout: float = 10.0) -> dict[str, str]:
    """Read one HTTP request head. Separate from the handshake so the same port
    can answer a plain GET with a page instead of refusing it."""
    return read_request_and_rest(sock, timeout=timeout)[0]


def is_upgrade(headers: dict[str, str]) -> bool:
    return (headers.get("upgrade", "").casefold() == "websocket"
            and bool(headers.get("sec-websocket-key")))


def perform_handshake(sock: socket.socket, *, timeout: float = 10.0) -> dict[str, str]:
    """Read the request and answer it. Raises when it was not a valid upgrade."""
    headers = read_request(sock, timeout=timeout)
    response = handshake_response(headers)
    sock.sendall(response)
    if not response.startswith(b"HTTP/1.1 101"):
        raise WebSocketError("geçersiz upgrade isteği")
    return headers


def client_mask(payload: bytes, key: bytes | None = None) -> bytes:
    """Build a masked client frame. Used by the tests to speak the protocol."""
    key = key or os.urandom(4)
    masked = bytearray(payload)
    for index in range(len(masked)):
        masked[index] ^= key[index % 4]

    header = bytearray([0x80 | OP_TEXT])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < (1 << 16):
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + key + bytes(masked)
