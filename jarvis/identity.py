"""Public product identity shared by prompts, launchers and interfaces.

The Python package keeps its historical ``jarvis`` name so existing memory,
imports and user data remain compatible.  Everything a person sees is sourced
from here and presents the product as ZESTOLES.
"""

from __future__ import annotations

PRODUCT_NAME = "ZESTOLES"
PRODUCT_SLUG = "zestoles"
LEGACY_NAME = "JARVIS"
VERSION = "0.2.0"


def user_agent() -> str:
    """Return an ASCII-only user agent suitable for HTTP headers."""
    return f"{PRODUCT_NAME}/{VERSION} (personal research assistant; +local)"
