"""Turkish-aware text helpers shared across the system."""

from __future__ import annotations

import re
import unicodedata

_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def fold(text: str) -> str:
    """Lowercase and strip Turkish diacritics so matching tolerates real typing.

    Dotted capital İ is replaced before casefold(), which would otherwise expand
    it to "i" plus a combining dot.
    """
    return (
        text.replace("İ", "i").replace("I", "i").casefold().replace("̇", "").translate(_FOLD)
    )


def slugify(text: str, *, max_length: int = 60) -> str:
    """Filename-safe slug that keeps Turkish words readable: 'Görev Kuyruğu' -> 'gorev-kuyrugu'."""
    folded = fold(text)
    ascii_only = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return (slug[:max_length].rstrip("-") or "not")


def first_line(text: str, *, limit: int = 80) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit] + ("…" if len(line) > limit else "")
