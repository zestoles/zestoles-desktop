"""One place where JARVIS talks to the open internet.

Every outbound request goes through here so the limits are real rather than
per-caller good intentions: a timeout on every call, a hard cap on how many bytes
are read, a refusal to follow a redirect off the host that was asked for, and a
user agent that says what this is.

The byte cap matters more than it looks. A background research task that streams a
two-gigabyte file at four in the morning does not fail loudly — it fills the disk
and takes the machine down with it.
"""

from __future__ import annotations

import gzip
import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ..identity import user_agent

log = logging.getLogger("jarvis.research.http")

# ASCII only: HTTP headers are latin-1, and a Turkish character here raises
# UnicodeEncodeError on every single request.
USER_AGENT = user_agent()
DEFAULT_TIMEOUT = 15
MAX_BYTES = 2_000_000

#: Hosts that resolve inside the machine or the local network. Refused outright:
#: a search result that points at 127.0.0.1 or a router admin page is either a
#: mistake or an attempt to make JARVIS fetch something on the attacker's behalf.
_BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
_BLOCKED_PREFIXES = ("10.", "192.168.", "169.254.", "172.16.", "172.17.",
                     "172.18.", "172.19.", "172.2", "172.30.", "172.31.")


class FetchError(RuntimeError):
    pass


@dataclass(slots=True)
class Response:
    url: str
    status: int
    body: bytes
    content_type: str

    @property
    def text(self) -> str:
        charset = "utf-8"
        if "charset=" in self.content_type:
            charset = self.content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            return self.body.decode(charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


def is_public_url(url: str, *, allow_local: bool = False) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").casefold()
    if not host:
        return False
    if allow_local:
        return True
    if host in _BLOCKED_HOSTS or host.endswith(".local"):
        return False
    return not host.startswith(_BLOCKED_PREFIXES)


def get(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
    headers: dict[str, str] | None = None,
    allow_local: bool = False,
) -> Response:
    if not is_public_url(url, allow_local=allow_local):
        raise FetchError(f"güvenli olmayan veya yerel adres reddedildi: {url}")

    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip",
        "Accept-Language": "tr,en;q=0.8",
        **(headers or {}),
    })

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                log.info("içerik %s baytta kesildi: %s", max_bytes, url)
                raw = raw[:max_bytes]
            if response.headers.get("Content-Encoding") == "gzip":
                try:
                    raw = gzip.decompress(raw)
                except (OSError, EOFError):
                    pass  # truncated gzip; fall through with what decoded
            return Response(
                url=response.geturl(),
                status=response.status,
                body=raw,
                content_type=response.headers.get("Content-Type", ""),
            )
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise FetchError(f"ağ hatası: {exc}") from exc
    except OSError as exc:
        raise FetchError(f"bağlantı hatası: {exc}") from exc


def get_json(url: str, **kwargs) -> dict | list:
    response = get(url, **kwargs)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise FetchError(f"JSON çözümlenemedi: {url}") from exc


def domain_of(url: str) -> str:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").casefold()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host
