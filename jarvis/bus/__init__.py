"""Runtime event bus and websocket transport.

The layer that makes JARVIS observable without making it dependent on being
observed. Domain code publishes to the event log exactly as it did before this
package existed; the bus subscribes, translates into a closed set of typed
messages, and fans them out to whoever is connected.

    domain → EventLog → EventBus → websocket → UI
                     ↘ SharedState ↗

Nothing in that chain points backwards. Removing the bus removes visibility and
nothing else, which is the property that lets it be started, stopped, or never
started at all without a thought about the rest of the system.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .bus import DEFAULT_QUEUE, DEFAULT_RING, EventBus, Subscriber
from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PANEL_PATH,
    BusServer,
    ClientHandler,
)
from .telemetry import DEFAULT_INTERVAL_S, TelemetryPump
from .types import (
    ALL_TYPES,
    EVENT_TYPES,
    PROTOCOL_TYPES,
    Envelope,
    TRANSLATION,
    payload_from_event,
    translate,
)
from .websocket import (
    ConnectionClosed,
    WebSocketConnection,
    WebSocketError,
    accept_key,
    decode_frame,
    encode_frame,
    handshake_response,
    parse_request,
)

log = logging.getLogger("jarvis.bus")


def _panel(config):
    """The developer dashboard, when it is on disk.

    Always the same file. The assistant page may take over "/", but diagnostics
    keep their own address so the main screen never becomes one.
    """
    panel = config.path("paths.ui", "ui/index.html")
    return panel if panel.is_file() else None


def build(runtime, *, host: str | None = None, port: int | None = None,
          start: bool = True, with_telemetry: bool = True,
          ui_file=None, request_handler=None
          ) -> tuple[EventBus, BusServer | None]:
    """Attach a bus to a runtime and, optionally, open the socket.

    Returns the bus even when the server could not start: a bus with no transport
    is still useful to anything in-process, and a busy port is not a reason to
    refuse to run.
    """
    config = runtime.config
    bus = EventBus(
        runtime.state,
        ring_size=int(config.get("bus.ring_size", DEFAULT_RING)),
        queue_size=int(config.get("bus.queue_size", DEFAULT_QUEUE)),
    )
    if runtime.events is not None:
        bus.attach(runtime.events)
    else:
        log.info("olay kaydı yok — veri yolu yalnızca doğrudan yayınları taşıyacak")

    if not start or not config.get("bus.enabled", True):
        return bus, None

    # Two pages, one server. The dashboard is the default because a
    # telemetry-only run is what most of this system's history assumed; the
    # assistant interface is opted into by whoever also supplies a handler for
    # it, which keeps "can be watched" and "can be asked" separate decisions.
    page = Path(ui_file) if ui_file else config.path("paths.ui", "ui/index.html")
    server = BusServer(
        bus,
        host=host or config.get("bus.host", DEFAULT_HOST),
        port=int(port if port is not None else config.get("bus.port", DEFAULT_PORT)),
        ui_file=page if page.is_file() else None,
        panel_file=_panel(config),
        request_handler=request_handler,
    )
    if not server.start():
        return bus, None

    if with_telemetry:
        pump = TelemetryPump(
            runtime, bus,
            interval_s=float(config.get("bus.telemetry_s", DEFAULT_INTERVAL_S)))
        pump.start()
        server.telemetry = pump
    return bus, server


__all__ = [
    "EventBus", "Subscriber", "BusServer", "ClientHandler", "build",
    "TelemetryPump", "DEFAULT_INTERVAL_S",
    "Envelope", "translate", "payload_from_event", "TRANSLATION",
    "EVENT_TYPES", "PROTOCOL_TYPES", "ALL_TYPES",
    "WebSocketConnection", "WebSocketError", "ConnectionClosed",
    "encode_frame", "decode_frame", "accept_key", "handshake_response",
    "parse_request", "DEFAULT_HOST", "DEFAULT_PORT", "DEFAULT_RING", "DEFAULT_QUEUE",
]
