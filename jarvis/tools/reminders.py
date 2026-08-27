"""Reminder tools routed through the common risk gate."""

from __future__ import annotations

from . import LOW, MEDIUM, SERVICES, ToolError, ToolResult, Workspace, tool


def _service():
    service = SERVICES.get("reminders")
    if service is None:
        raise ToolError("hatırlatma servisi açık değil")
    return service


@tool("reminder.add", risk=MEDIUM, summary="Kalıcı bir Windows hatırlatıcısı kurar")
def _add(*, workspace: Workspace, when: str, text: str) -> ToolResult:
    try:
        item = _service().add(when, text)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    return ToolResult(True, output=f"#{item['id']} · {item['zaman']} · {item['metin']}", detail=item)


@tool("reminder.list", risk=LOW, summary="Bekleyen hatırlatıcıları listeler")
def _list(*, workspace: Workspace) -> ToolResult:
    items = _service().list()
    if not items:
        return ToolResult(True, output="Bekleyen hatırlatıcı yok.", detail={"count": 0})
    lines = [f"#{item['id']} · {item['zaman']} · {item['metin']}" for item in items]
    return ToolResult(True, output="\n".join(lines), detail={"count": len(items)})


@tool("reminder.cancel", risk=MEDIUM, summary="Bir hatırlatıcıyı iptal eder")
def _cancel(*, workspace: Workspace, reminder_id: int) -> ToolResult:
    if not _service().cancel(reminder_id):
        raise ToolError(f"bekleyen hatırlatıcı bulunamadı: #{reminder_id}")
    return ToolResult(True, output=f"#{reminder_id} iptal edildi")
