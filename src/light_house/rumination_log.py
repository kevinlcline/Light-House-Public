"""Portable NDJSON rumination activity log (metadata only, Proposal 3 phase 3)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from light_house.config import Settings
from light_house.inner_life_trace import RuminationTraceEntry, task_hint_label
from light_house.lights.registry import known_light_ids

_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class RuminationLogRecord:
    """One rumination run — public ledger fields only (no private body text)."""

    agent_id: str
    thread_id: str
    wake_kind: str | None
    summary_line: str
    tool_names: list[str]
    tool_rounds_used: int
    first_response_mode: str
    persisted: bool
    persisted_chars: int
    error: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    def to_json_line(self) -> str:
        wake = self.wake_kind or "scheduled"
        payload = {
            "run_id": self.run_id,
            "ts": self.ts,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "wake_kind": wake,
            "task_hint": task_hint_label(self.wake_kind),
            "summary_line": self.summary_line,
            "tool_names": self.tool_names,
            "tool_rounds_used": self.tool_rounds_used,
            "first_response_mode": self.first_response_mode,
            "persisted": self.persisted,
            "persisted_chars": self.persisted_chars,
            "error": self.error,
        }
        return json.dumps(payload, ensure_ascii=False)


def _log_path(settings: Settings) -> Path:
    return settings.rumination_log_path.resolve()


def append_rumination_log(settings: Settings, record: RuminationLogRecord) -> None:
    if not settings.rumination_log_enabled:
        return
    path = _log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.to_json_line() + "\n"
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def append_rumination_log_from_trace(
    settings: Settings,
    entry: RuminationTraceEntry,
    *,
    summary_line: str = "",
) -> None:
    summary = summary_line.strip()
    if not summary and entry.error:
        summary = f"(error: {entry.error[:120]})"
    if not summary:
        summary = "(no summary)"
    append_rumination_log(
        settings,
        RuminationLogRecord(
            agent_id=entry.agent_id,
            thread_id=entry.thread_id,
            wake_kind=entry.wake_kind,
            summary_line=summary[:500],
            tool_names=list(entry.tools_called),
            tool_rounds_used=entry.tool_rounds_used,
            first_response_mode=entry.first_response_mode,
            persisted=entry.persisted,
            persisted_chars=entry.generated_text_chars,
            error=entry.error,
        ),
    )


def read_rumination_log_entries(
    path: Path,
    *,
    max_lines: int,
    agent_id: str | None = None,
) -> tuple[list[dict], bool]:
    """Return newest entries first (up to max_lines), optionally filtered by agent_id."""
    if not path.is_file():
        return [], False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], False
    if not lines:
        return [], False
    truncated = len(lines) > max_lines
    tail = lines[-max_lines:] if truncated else lines
    entries: list[dict] = []
    for line in reversed(tail):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if agent_id and row.get("agent_id") != agent_id:
            continue
        entries.append(row)
    return entries, truncated


def format_rumination_log_entries(entries: list[dict], *, truncated: bool) -> str:
    if not entries:
        return "Rumination log is empty."
    lines: list[str] = []
    if truncated:
        lines.append("(log tail truncated — showing recent entries only)")
    for row in entries:
        wake = row.get("wake_kind", "?")
        agent = row.get("agent_id", "?")
        tools = row.get("tool_names") or []
        tool_str = ", ".join(tools) if tools else "(none)"
        persisted = "yes" if row.get("persisted") else "no"
        chars = row.get("persisted_chars", 0)
        summary = row.get("summary_line", "")
        err = row.get("error")
        ts = row.get("ts")
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts)) if isinstance(ts, (int, float)) else "?"
        line = (
            f"[{ts_str}] {agent} · wake={wake} · tools={tool_str} · "
            f"rounds={row.get('tool_rounds_used', 0)} · persisted={persisted} ({chars} chars)\n"
            f"  {summary}"
        )
        if err:
            line += f"\n  error: {err}"
        lines.append(line)
    return "\n\n".join(lines)


def read_rumination_log_for_agent(
    settings: Settings,
    *,
    requesting_agent_id: str,
    tail: int,
    filter_agent_id: str | None = None,
) -> str:
    if not settings.rumination_log_enabled:
        return "read_rumination_log is disabled (RUMINATION_LOG_ENABLED=false)."
    max_lines = min(max(1, tail), settings.rumination_log_max_read_lines)
    path = _log_path(settings)
    agent_filter = filter_agent_id.strip().lower() if filter_agent_id else None
    if agent_filter and agent_filter not in known_light_ids(settings):
        known = ", ".join(sorted(known_light_ids(settings)))
        return f"read_rumination_log failed: filter_agent_id must be one of {known}, or omitted."
    entries, truncated = read_rumination_log_entries(
        path,
        max_lines=max_lines,
        agent_id=agent_filter,
    )
    header = f"Rumination activity log (requested by {requesting_agent_id})"
    body = format_rumination_log_entries(entries, truncated=truncated)
    return f"{header}\n\n{body}"
