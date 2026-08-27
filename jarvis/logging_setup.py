"""Logging: everything to a rotating file, only warnings to the console.

The console belongs to the conversation. Diagnostics go to logs/jarvis.log.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import ROOT

_CONFIGURED = False


def setup(*, verbose: bool = False, log_dir: Path | None = None) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("jarvis")
    if _CONFIGURED:
        return logger

    directory = log_dir or (ROOT / "logs")
    directory.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = RotatingFileHandler(
        directory / "jarvis.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s | %(message)s")
    )
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

    _CONFIGURED = True
    return logger
