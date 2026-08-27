"""The settings a person may change, and the reason that is a short list.

A settings screen is a write path into the configuration of a program that runs
shell commands on the user's machine. So the interesting part of this module is
not the reading or the saving -- it is `EDITABLE`, which is an allow-list rather
than a deny-list.

That direction matters. A deny-list has to anticipate every dangerous key that
will ever exist; an allow-list makes a new key unreachable until somebody adds
it here on purpose, with its own bounds and its own test. The security-shaped
settings are not on it and are not configuration at all: self-modification is a
constant in `lab/promotion.py`, the sandbox grants are a frozenset in
`agents/permissions.py`, and the tool risk tiers live in the registry. None of
them can be reached from here, and a test asserts that they never appear.

## What "saved" means

Only the key that changed is written. Saving the whole resolved configuration
would freeze today's defaults into the user's file, and a later improvement to
any default would silently never reach them. The file is written whole through a
temporary file and replaced atomically, so an interrupted save cannot leave a
config that JARVIS then refuses to start from.

A file that is already corrupt is moved aside rather than overwritten in place.
The contents were unreadable anyway; keeping a copy costs nothing and losing
someone's settings without a trace is the kind of small betrayal that makes a
program hard to trust.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CONFIG_PATH, Config

log = logging.getLogger("jarvis.settings")


@dataclass(frozen=True, slots=True)
class Rule:
    """One setting a person may change, and the range that keeps it sane."""

    label: str
    kind: type
    low: float | None = None
    high: float | None = None
    hint: str = ""

    def parse(self, raw: Any) -> tuple[Any | None, str]:
        """Returns (value, problem). The form sends strings; that is fine."""
        if self.kind is int:
            try:
                value: Any = int(str(raw).strip())
            except (TypeError, ValueError):
                return None, "sayı olmalı"
        elif self.kind is bool:
            value = str(raw).strip().lower() in ("1", "true", "evet", "acik", "açık")
        else:
            value = str(raw).strip()
            if not value:
                return None, "boş olamaz"
        if self.low is not None and value < self.low:
            return None, f"en az {self.low:g} olmalı"
        if self.high is not None and value > self.high:
            return None, f"en fazla {self.high:g} olmalı"
        return value, ""


#: Everything a person may change from the settings screen. Nothing else is
#: reachable -- see the module note on why this is an allow-list.
EDITABLE: dict[str, Rule] = {
    "ui.orphan_grace_s": Rule(
        label="Pencere kapanınca kapanma süresi (sn)", kind=int, low=0, high=3600,
        hint="Son pencere kapandıktan bu kadar sonra JARVIS kendini kapatır. "
             "0 = kapanma, JARVIS açık kalır."),
    "chat.history_turns": Rule(
        label="Hatırlanan konuşma turu", kind=int, low=2, high=64,
        hint="Modele geri verilen son tur sayısı. Büyütmek bağlamı artırır, "
             "her isteği yavaşlatır."),
    "chat.context_chars": Rule(
        label="Modele verilen bağlam (karakter)", kind=int, low=2000, high=60000,
        hint="Konuşmanın modele gönderilen kısmının üst sınırı. Büyütmek daha "
             "çok şey hatırlatır, her isteği yavaşlatır."),
    "assistant.max_steps": Rule(
        label="Bir istekte en fazla araç adımı", kind=int, low=1, high=16,
        hint="Bir istek için arka arkaya kaç araç çağrılabilir."),
}


def read_all(config: Config) -> list[dict[str, Any]]:
    """Every editable setting with its current value, for the settings screen."""
    rows = []
    for key, rule in EDITABLE.items():
        rows.append({
            "anahtar": key,
            "etiket": rule.label,
            "deger": config.get(key, _default_for(key)),
            "aciklama": rule.hint,
            "en_az": rule.low,
            "en_fazla": rule.high,
        })
    return rows


def describe(config: Config) -> list[dict[str, str]]:
    """Facts about this installation. Worth seeing, not worth editing here.

    Deliberately carries no `anahtar`: a row without a key cannot be sent back
    as a change, so the screen cannot grow an edit box for the model host by
    accident.
    """
    return [
        {"etiket": "Yerel model", "deger": str(config.get("local.model", "—"))},
        {"etiket": "Model adresi", "deger": str(config.get("local.host", "—"))},
        {"etiket": "Çalışma alanı", "deger": str(config.get("assistant.workspace")
                                                or "kullanıcı ev dizini")},
        {"etiket": "Günlükler", "deger": str(config.path("paths.logs", "logs"))},
        {"etiket": "Kendi kendini değiştirme", "deger": "kapalı"},
        {"etiket": "Ağ", "deger": "yalnızca bu bilgisayar (127.0.0.1)"},
    ]


def write_one(config: Config, key: str, raw: Any, *,
              path: Path | None = None) -> tuple[bool, str]:
    """Change one setting and save it. Returns (ok, reason it was refused).

    The live `config` is only updated once the value has been accepted and the
    file has been written, so a refused change leaves the running system exactly
    as it was.
    """
    rule = EDITABLE.get(key)
    if rule is None:
        # Not a deny-list decision: the key simply is not offered.
        return False, "bu ayar değiştirilemez"
    value, problem = rule.parse(raw)
    if problem:
        return False, problem

    path = path or CONFIG_PATH
    stored = _read_file(path)
    node = stored
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            return False, "ayar dosyasında beklenmeyen yapı"
    node[parts[-1]] = value

    try:
        _write_file(path, stored)
    except OSError as exc:
        log.warning("ayar kaydedilemedi: %s", exc)
        return False, f"kaydedilemedi: {exc}"

    _apply(config, parts, value)
    log.info("ayar değişti: %s = %r", key, value)
    return True, ""


# ------------------------------------------------------------------ internals
def _default_for(key: str) -> Any:
    node: Any = Config.load.__globals__["DEFAULTS"]
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return copy.deepcopy(node)


def _read_file(path: Path) -> dict[str, Any]:
    """What is actually on disk, or an empty document when it cannot be read.

    A corrupt file is moved aside rather than silently overwritten: its contents
    were already unreachable, and keeping a copy costs nothing.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        backup = path.with_suffix(".json.bozuk")
        log.warning("ayar dosyası okunamadı (%s), %s olarak saklandı", exc, backup.name)
        try:
            path.replace(backup)
        except OSError:
            pass
        return {}
    return data if isinstance(data, dict) else {}


def _write_file(path: Path, data: dict[str, Any]) -> None:
    """Write whole, then replace. An interrupted save must not leave half a config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _apply(config: Config, parts: list[str], value: Any) -> None:
    """Update the running config so the change takes effect without a restart."""
    node = config._data  # noqa: SLF001 - this module is the config's write path
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


__all__ = ["EDITABLE", "Rule", "read_all", "describe", "write_one"]
