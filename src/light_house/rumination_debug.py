"""Compact metadata-only rumination debug log (Proposal 3 phase 5).

No context markdown, conversation text, or private reflection bodies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from light_house.config import Settings
from light_house.dev_log import read_dev_log_tail
from light_house.inner_life_trace import RuminationTraceEntry

_DEBUG_LOCK = threading.Lock()
_BLOCK_SEP = "--- "
_DEBUG_FILE_DATE: date | None = None


@dataclass
class RuminationDebugEntry:
    agent_id: str
    thread_id: str
    wake_kind: str | None
    task_hint_label: str
    tool_rounds_cap: int | None
    tool_rounds_used: int
    tools_called: list[str] = field(default_factory=list)
    first_response_mode: str = "unknown"
    persisted: bool = False
    generated_text_chars: int = 0
    summary_line: str = ""
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _debug_path(settings: Settings) -> Path:
    return settings.rumination_debug_log_path.resolve()


def _rotate_if_new_day(path: Path, current_date: date | None) -> date:
    today = date.today()
    if current_date is None or current_date == today:
        return today
    path.write_text("", encoding="utf-8")
    return today


def _format_block(entry: RuminationDebugEntry) -> str:
    ts = entry.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wake = entry.wake_kind or "scheduled"
    cap = entry.tool_rounds_cap if entry.tool_rounds_cap is not None else "default"
    tools = ", ".join(entry.tools_called) if entry.tools_called else "(none)"
    persisted = (
        f"yes ({entry.generated_text_chars} chars)"
        if entry.persisted
        else "no"
    )
    summary = (entry.summary_line or "(no summary)").strip()
    if len(summary) > 240:
        summary = summary[:237] + "..."
    err = entry.error or "(none)"
    lines = [
        f"{_BLOCK_SEP}{ts} | agent={entry.agent_id} | thread={entry.thread_id} | wake={wake}",
        f"event_type: {entry.task_hint_label}",
        f"tool_rounds: {entry.tool_rounds_used} / cap {cap}",
        f"first_response: {entry.first_response_mode}",
        f"tools_called: {tools}",
        f"persisted: {persisted}",
        f"summary: {summary}",
        f"error: {err}",
        "",
    ]
    return "\n".join(lines)


def append_rumination_debug(settings: Settings, entry: RuminationDebugEntry) -> None:
    if not settings.rumination_debug_log_enabled:
        return
    path = _debug_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    block = _format_block(entry)
    global _DEBUG_FILE_DATE
    with _DEBUG_LOCK:
        _DEBUG_FILE_DATE = _rotate_if_new_day(path, _DEBUG_FILE_DATE)
        with path.open("a", encoding="utf-8") as f:
            f.write(block)


def append_rumination_debug_from_trace(
    settings: Settings,
    entry: RuminationTraceEntry,
    *,
    summary_line: str = "",
) -> None:
    append_rumination_debug(
        settings,
        RuminationDebugEntry(
            agent_id=entry.agent_id,
            thread_id=entry.thread_id,
            wake_kind=entry.wake_kind,
            task_hint_label=entry.task_hint_label,
            tool_rounds_cap=entry.tool_rounds_cap,
            tool_rounds_used=entry.tool_rounds_used,
            tools_called=list(entry.tools_called),
            first_response_mode=entry.first_response_mode,
            persisted=entry.persisted,
            generated_text_chars=entry.generated_text_chars,
            summary_line=summary_line.strip(),
            error=entry.error,
            timestamp=entry.timestamp,
        ),
    )


def read_rumination_debug_tail(path: Path, *, max_lines: int) -> tuple[str, int, bool]:
    return read_dev_log_tail(path, max_lines=max_lines)
