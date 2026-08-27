"""Configuration: built-in defaults layered under config.json on disk.

Defaults live here so ZESTOLES still starts if config.json is missing or partial.
Anything the user edits in config.json wins.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

log = logging.getLogger("jarvis.config")

DEFAULTS: dict[str, Any] = {
    "user": {"name": "Kullanıcı"},
    "language": {"speak": "tr", "think": "en"},
    "local": {
        "enabled": True,
        "host": "http://127.0.0.1:11434",
        "model": "qwen3.5:9b",
        # Loaded on demand for work the fast model cannot carry (S3 agents).
        "model_heavy": "qwen3:14b",
        "timeout_s": 300,
        "keep_alive": "30m",
        "think": False,
        "temperature": 0.7,
        "context_window": 16384,
    },
    "cloud": {
        "enabled": False,
        "cli": "claude",
        "model": "sonnet",
        "model_light": "haiku",
        "workdir": "data/cleanroom",
        "timeout_s": 300,
    },
    "budget": {
        "cloud_calls_per_hour": 0,
        "cloud_calls_per_day": 0,
        "cloud_calls_per_night": 0,
        "cloud_at_night": False,
        "night_hours": [1, 8],
    },
    "router": {
        # "local" keeps Claude out of the loop unless /claude asks for it by hand.
        # The system is designed to be complete without it; Claude is a consultant,
        # not the brain. Switch to "auto" only when there is allowance to spare.
        "mode": "local",
        "cloud_min_chars": 240,
        "cloud_score_threshold": 3,
    },
    # `context_chars` is the real limit on what the model reads: the turn
    # count above cannot see that one pasted file outweighs fifty replies.
    "chat": {"history_turns": 24, "context_chars": 32000},
    # The tool-using loop. `max_steps` is how many tools one request may call
    # before the loop gives up on it; `workspace` empty means the user's home
    # directory, which is what the tool layer scopes itself to. A turn that
    # cannot finish inside `turn_timeout_s` stops honestly instead of holding
    # the conversation hostage.
    "assistant": {"max_steps": 8, "workspace": "", "model": "",
                  "turn_timeout_s": 60},
    # The desktop shell. `orphan_grace_s` is how long JARVIS keeps running with
    # nobody attached before closing itself: long enough that a page reload never
    # reaches it, short enough that a closed tab does not leave a process behind.
    # Zero switches it off, for someone who wants it to outlive the page.
    "ui": {"orphan_grace_s": 120},
    # Speech. Both models are local; nothing here reaches the network. An empty
    # `stt.dir`/`tts.dir` means the default under data/ses. `backchannel` scales
    # how readily JARVIS says "hı hı" -- 0 switches it off.
    "voice": {
        "language": "tr",
        "backchannel": 1.0,
        "stt": {"model": "large-v3-turbo", "device": "auto", "dir": ""},
        # `engine` "piper" (anlik, robotik) ya da "chatterbox" (dogal, yan
        # surec; ilk yukleme ~45 sn). Chatterbox referansi klonlanmis bir
        # sestir: `data/ses/referans/` altindaki dosya degistirilerek ses
        # karakteri degistirilir.
        "tts": {"voice": "tr_TR-dfki-medium", "dir": "", "engine": "chatterbox",
                "exaggeration": 0.5, "cfg_weight": 0.3, "temperature": 0.35,
                "model_version": "v3", "reference": "", "python": "",
                "server": "", "timeout_s": 600},
    },
    "bus": {
        "enabled": True,
        # Loopback by intent: this socket carries live runtime state and accepts
        # commands, with nothing authenticating behind it.
        "host": "127.0.0.1",
        "port": 8797,
        # Recent history a reconnecting client can replay from before it has to
        # take a fresh snapshot instead.
        "ring_size": 500,
        # Per-client backlog. A reader that falls further behind than this loses
        # messages — the publisher is never made to wait for it.
        "queue_size": 250,
        # How often resources are re-measured while somebody is watching. The
        # pump does nothing at all when no client is connected.
        "telemetry_s": 3.0,
    },
    "improve": {
        "enabled": True,
        # Below this composite score an opportunity is shelved rather than tried.
        "minimum_score": 0.45,
        # Ceilings on self-improvement activity, counted from what actually ran so
        # a crash loop cannot reset them.
        "daily": {"arastirma": 6, "hipotez": 8, "deney": 4},
        # Tighter while nobody is awake to intervene.
        "nightly": {"arastirma": 4, "hipotez": 5, "deney": 3},
    },
    "lab": {
        # Commands a sandboxed experiment may start. Anything not listed is
        # refused — the allowlist is the boundary, not the caller's good sense.
        "allowed_commands": ["python", "python3", "py", "git", "pip"],
        "timeout_s": 120,
        "max_output": 200000,
        "max_file_bytes": 5000000,
        "max_files": 2000,
        # Off by default. An experiment that needs the network is an experiment
        # that can send whatever it read somewhere.
        "allow_network": False,
        "keep_sandboxes": 5,
        "keep_snapshots": 10,
        # Where a promoted experiment's files land. Deliberately NOT the jarvis
        # package: self-modification is off, and the promotion gate refuses any
        # target inside the source tree.
        "promotion_target": "data/lab/promoted",
    },
    "research": {
        "enabled": True,
        # Sources actually read per investigation. Each one costs a fetch and a
        # model call for the support judgement, so this is the main cost dial.
        "max_sources": 5,
        "per_provider": 5,
        "fetch_timeout_s": 15,
        "context_chars": 30000,
        # Independent publishers required before a claim may become knowledge.
        "min_independent": 2,
        "write_knowledge": True,
        "searxng": {
            "enabled": True,
            "url": "http://127.0.0.1:8888",
            "timeout_s": 20,
        },
        "duckduckgo": {"enabled": True},
        "github": {"enabled": True},
        "stackexchange": {"enabled": True, "site": "stackoverflow"},
        "hackernews": {"enabled": True},
        "wikipedia": {"enabled": True},
        "arxiv": {"enabled": False},
        # Vendor developer forums. Discourse exposes a public search endpoint, and
        # for a platform like Roblox the official forum holds answers that exist
        # nowhere else.
        "forums": [
            {"url": "https://devforum.roblox.com", "label": "roblox"},
        ],
    },
    "agents": {
        "enabled": True,
        "max_steps": 5,
        "total_timeout_s": 900,
        "save_skills": True,
        # One model per run: loading a second evicts the first on a single GPU and
        # costs ~70s, measured. "fast" or "heavy".
        "model": "fast",
        # Same model by default. Setting this to "heavy" gives genuinely
        # independent verification but pays the swap on every run — worth it at
        # night, not during a conversation.
        "verify_model": "fast",
    },
    "autonomy": {
        "enabled": True,
        # Seconds without keyboard or mouse input before the machine counts as free.
        "idle_after_s": 300,
        # Load from anything else on the machine that makes background work rude.
        "cpu_ceiling": 65.0,
        "gpu_ceiling": 55.0,
        "ram_ceiling": 88.0,
        "idle_concurrency": 1,
        "night_concurrency": 2,
        "tick_s": 5.0,
    },
    "memory": {
        "enabled": True,
        # Distillation and summarising run on the fast local model, always —
        # they happen after every session, including while the user is asleep.
        "model": "qwen3.5:9b",
        "embed_model": "bge-m3",
        "recall_limit": 5,
    },
    "paths": {
        "db": "data/jarvis.db",
        "persona": "persona/core.md",
        "vault": "vault",
        "lab": "data/lab",
        "ui": "ui/index.html",
        "logs": "logs",
        "documents": "data/belgeler",
        "control": "data/control.json",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Dotted-path access over the merged configuration tree."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or CONFIG_PATH
        data = copy.deepcopy(DEFAULTS)
        if path.exists():
            try:
                data = _merge(data, json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                # Defaults are a working JARVIS; a parse error here used to be a
                # JARVIS that would not open at all. Now that the settings screen
                # writes this file, an interrupted save must cost the settings
                # rather than the application. The file is left alone -- the next
                # save moves it aside and says so.
                log.warning("ayar dosyası okunamadı (%s) — varsayılanlar kullanılıyor",
                            exc)
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str, default: str = "") -> Path:
        """Resolve a configured path, relative entries anchored at the JARVIS root."""
        raw = Path(self.get(dotted, default))
        return raw if raw.is_absolute() else ROOT / raw

    @property
    def root(self) -> Path:
        return ROOT

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)
