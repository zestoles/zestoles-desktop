"""Persistent reminders with Turkish time parsing and Windows notifications."""

from __future__ import annotations

import datetime as dt
import html
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

log = logging.getLogger("jarvis.reminders")


def parse_when(expression: str, *, now: dt.datetime | None = None) -> dt.datetime | None:
    text = str(expression).strip().lower()
    current = now or dt.datetime.now()
    relative = re.search(r"(\d+)\s*(dakika|dk|saat|gün|gun)\s*sonra", text)
    if relative:
        amount, unit = int(relative.group(1)), relative.group(2)
        if unit in ("dakika", "dk"):
            return current + dt.timedelta(minutes=amount)
        if unit == "saat":
            return current + dt.timedelta(hours=amount)
        return current + dt.timedelta(days=amount)

    for pattern, date_first in (
        (r"(\d{4})-(\d{1,2})-(\d{1,2})[ t](\d{1,2}):(\d{2})", True),
        (r"(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})", False),
    ):
        match = re.search(pattern, text)
        if match:
            values = list(map(int, match.groups()))
            year, month, day, hour, minute = (values if date_first else
                                               (values[2], values[1], values[0], values[3], values[4]))
            try:
                return dt.datetime(year, month, day, hour, minute)
            except ValueError:
                return None

    clock = re.search(r"\b(\d{1,2})(?::|\.)(\d{2})\b", text)
    if clock:
        hour, minute = int(clock.group(1)), int(clock.group(2))
        if hour > 23 or minute > 59:
            return None
        target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if "yarın" in text or "yarin" in text:
            target += dt.timedelta(days=1)
        elif target <= current:
            target += dt.timedelta(days=1)
        return target
    return None


class ReminderService:
    def __init__(self, database: Path, *, notifier: Callable[[str], None] | None = None,
                 poll_s: float = 5.0) -> None:
        self.database = Path(database)
        self.notifier = notifier or windows_toast
        self.poll_s = max(.1, float(poll_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self):
        database = self._connect()
        try:
            yield database
            database.commit()
        except Exception:
            database.rollback()
            raise
        finally:
            database.close()

    def _ensure_schema(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                due REAL NOT NULL,
                text TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                created REAL NOT NULL,
                fired REAL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS reminders_due ON reminders(state, due)")

    def add(self, when: str | dt.datetime, text: str) -> dict[str, object]:
        message = str(text).strip()
        if not message:
            raise ValueError("hatırlatma metni boş")
        due = when if isinstance(when, dt.datetime) else parse_when(when)
        if due is None:
            raise ValueError("zaman anlaşılamadı; örnek: '30 dakika sonra' veya 'yarın 09:30'")
        if due <= dt.datetime.now() - dt.timedelta(seconds=1):
            raise ValueError("hatırlatma zamanı geçmişte")
        with self._lock, self._session() as db:
            cursor = db.execute("INSERT INTO reminders(due,text,state,created) VALUES(?,?,'pending',?)",
                                (due.timestamp(), message, time.time()))
            item_id = int(cursor.lastrowid)
        return {"id": item_id, "zaman": due.isoformat(timespec="minutes"), "metin": message,
                "durum": "pending"}

    def list(self, *, include_done: bool = False) -> list[dict[str, object]]:
        where = "" if include_done else "WHERE state='pending'"
        with self._lock, self._session() as db:
            rows = db.execute(f"SELECT id,due,text,state,created,fired FROM reminders {where} "
                              "ORDER BY due ASC LIMIT 200").fetchall()
        return [{"id": int(row["id"]), "zaman": dt.datetime.fromtimestamp(row["due"]).isoformat(timespec="minutes"),
                 "metin": row["text"], "durum": row["state"]} for row in rows]

    def cancel(self, reminder_id: int) -> bool:
        with self._lock, self._session() as db:
            cursor = db.execute("UPDATE reminders SET state='cancelled' "
                                "WHERE id=? AND state='pending'", (int(reminder_id),))
            return cursor.rowcount > 0

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="zestoles-reminders", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=min(self.poll_s + 2, 10))
        self._thread = None

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def tick(self, *, now: float | None = None) -> int:
        moment = time.time() if now is None else float(now)
        with self._lock, self._session() as db:
            rows = db.execute("SELECT id,text FROM reminders WHERE state='pending' AND due<=? "
                              "ORDER BY due", (moment,)).fetchall()
            ids = [int(row["id"]) for row in rows]
            if ids:
                marks = ",".join("?" for _ in ids)
                db.execute(f"UPDATE reminders SET state='fired', fired=? WHERE id IN ({marks})",
                           (moment, *ids))
        for row in rows:
            try:
                self.notifier(str(row["text"]))
            except Exception as exc:  # noqa: BLE001
                log.warning("hatırlatma bildirimi gösterilemedi: %s", exc)
        return len(rows)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                log.warning("hatırlatma döngüsü: %s", exc)

    def status(self) -> dict[str, object]:
        pending = self.list()
        return {"calisiyor": self.running, "bekleyen": len(pending),
                "siradaki": pending[0] if pending else None}


def windows_toast(text: str) -> None:
    """Show a toast without putting user text inside PowerShell source."""
    if os.name != "nt":
        return
    script = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$nodes = $xml.GetElementsByTagName('text')
$null = $nodes.Item(0).AppendChild($xml.CreateTextNode('ZESTOLES Hatırlatma'))
$null = $nodes.Item(1).AppendChild($xml.CreateTextNode($env:ZESTOLES_REMINDER_TEXT))
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('ZESTOLES').Show($toast)
"""
    environment = dict(os.environ)
    environment["ZESTOLES_REMINDER_TEXT"] = html.unescape(str(text))[:500]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                     env=environment, creationflags=flags, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


__all__ = ["ReminderService", "parse_when", "windows_toast"]
