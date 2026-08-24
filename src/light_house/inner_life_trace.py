"""Append-only rumination trace log for Kevin (context + outcomes per run)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from light_house.config import Settings
from light_house.dev_log import read_dev_log_tail

_TRACE_LOCK = threading.Lock()
_BLOCK_SEP = "=== "
_CONVERSATION_MAX_CHARS = 8000


@dataclass
class RuminationTraceEntry:
    agent_id: str
    thread_id: str
    wake_kind: str | None
    tool_rounds_cap: int | None
    context_markdown: str
    task_hint_label: str
    system_prompt_chars: int
    conversation_window: str
    stream_event_count: int
    stream_char_count: int
    tool_rounds_used: int
    tools_called: list[str] = field(default_factory=list)
    first_response_mode: str = "unknown"
    persisted: bool = False
    generated_text_chars: int = 0
    error: str | None = None
    felt_cycles: int | None = None
    felt_days: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def task_hint_label(wake_kind: str | None) -> str:
    if wake_kind == "kevin_shared_note":
        return "shared_note"
    if wake_kind == "mailbox_letter":
        return "mailbox"
    if wake_kind == "post_chat":
        return "post_chat"
    if wake_kind == "memory_maintenance":
        return "memory_maintenance"
    if wake_kind == "chores":
        return "chores"
    if wake_kind == "meditation":
        return "meditation"
    return "scheduled"


def _message_line(msg: BaseMessage) -> str:
    role = type(msg).__name__
    content = msg.content
    if not isinstance(content, str):
        content = str(content)
    if isinstance(msg, AIMessage) and msg.tool_calls:
        names = [tc.get("name", "?") for tc in msg.tool_calls if isinstance(tc, dict)]
        return f"[{role} tool_calls={names}]\n{content}".strip()
    return f"[{role}]\n{content}".strip()


def format_messages_for_trace(messages: list[Any], *, max_chars: int = _CONVERSATION_MAX_CHARS) -> str:
    if not messages:
        return "(empty)"
    parts = [_message_line(m) for m in messages if isinstance(m, BaseMessage)]
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text


def extract_tools_called(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name")
                if name and name not in names:
                    names.append(str(name))
    return names


def first_response_mode(messages: list[Any]) -> str:
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        if msg.tool_calls:
            return "tools"
        content = msg.content
        if isinstance(content, str) and content.strip():
            return "plain_text"
        return "empty"
    return "none"


def _trace_path(settings: Settings) -> Path:
    return settings.inner_life_trace_path.resolve()


def _rotate_if_new_day(path: Path, current_date: date | None) -> date:
    today = date.today()
    if current_date is None or current_date == today:
        return today
    path.write_text("", encoding="utf-8")
    return today


def _format_block(entry: RuminationTraceEntry) -> str:
    ts = entry.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wake = entry.wake_kind or "scheduled"
    cap = entry.tool_rounds_cap if entry.tool_rounds_cap is not None else "default"
    tools = ", ".join(entry.tools_called) if entry.tools_called else "(none)"
    err = entry.error or "(none)"
    persisted = (
        f"yes ({entry.generated_text_chars} chars)"
        if entry.persisted
        else "no"
    )
    lines = [
        f"{_BLOCK_SEP}{ts} | agent={entry.agent_id} | thread={entry.thread_id} | wake={wake} ===",
        f"task_hint: {entry.task_hint_label}",
        f"tool_rounds_cap: {cap}",
        f"system_prompt_chars: {entry.system_prompt_chars}",
        f"stream_events: {entry.stream_event_count} | stream_chars: {entry.stream_char_count}",
        f"first_response: {entry.first_response_mode}",
        f"tools_called: {tools}",
        f"tool_rounds_used: {entry.tool_rounds_used}",
        f"persisted: {persisted}",
        f"felt_cycles: {entry.felt_cycles if entry.felt_cycles is not None else '(n/a)'}",
        f"felt_days: {entry.felt_days if entry.felt_days is not None else '(n/a)'}",
        f"error: {err}",
        "",
        "--- context markdown ---",
        entry.context_markdown or "(empty)",
        "",
        "--- conversation window ---",
        entry.conversation_window,
        "",
        "=== end ===",
        "",
    ]
    return "\n".join(lines)


_trace_file_date: date | None = None


def append_rumination_trace(settings: Settings, entry: RuminationTraceEntry) -> None:
    if not settings.inner_life_trace_enabled:
        return
    path = _trace_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    block = _format_block(entry)
    global _trace_file_date
    with _TRACE_LOCK:
        _trace_file_date = _rotate_if_new_day(path, _trace_file_date)
        with path.open("a", encoding="utf-8") as f:
            f.write(block)


def append_rumination_error(
    settings: Settings,
    *,
    agent_id: str,
    thread_id: str,
    wake_kind: str | None,
    error: str,
    context_markdown: str = "",
) -> None:
    entry = RuminationTraceEntry(
        agent_id=agent_id,
        thread_id=thread_id,
        wake_kind=wake_kind,
        tool_rounds_cap=None,
        context_markdown=context_markdown,
        task_hint_label=task_hint_label(wake_kind),
        system_prompt_chars=0,
        conversation_window="(run failed before trace)",
        stream_event_count=0,
        stream_char_count=0,
        tool_rounds_used=0,
        error=error,
    )
    append_rumination_trace(settings, entry)
    from light_house.rumination_log import append_rumination_log_from_trace
    from light_house.rumination_debug import append_rumination_debug_from_trace

    append_rumination_log_from_trace(settings, entry)
    append_rumination_debug_from_trace(settings, entry)


def read_inner_life_trace_tail(path: Path, *, max_lines: int) -> tuple[str, int, bool]:
    return read_dev_log_tail(path, max_lines=max_lines)


def build_inner_life_context_preview(
    memory: Any,
    settings: Settings,
    agent_id: str,
) -> dict[str, Any]:
    """Live snapshot of rumination context without running the graph."""
    from light_house.agent.rumination_nodes import build_rumination_system_content
    from light_house.agent.tool_helpers import (
        CODEBASE_SYSTEM_HINT,
        GARDEN_RUMINATION_HINT,
        GARDEN_SYSTEM_HINT,
        NOTES_SYSTEM_HINT,
        PERSONAL_DB_HINT,
        WEB_SYSTEM_HINT,
        format_peer_message_hint,
    )
    from light_house.agents.registry import get_agent, list_agents, load_persona
    from light_house.memory.context_builder import (
        build_agent_context,
        format_agent_context_markdown,
    )
    from light_house.tools.light_tools import LIGHT_TOOLS

    agent = get_agent(agent_id, settings)
    bundle = build_agent_context(
        memory,
        thread_id=agent.thread_id,
        agent_id=agent_id,
        stream_max_chars=settings.memory_target_context_chars,
        stream_mode="rumination",
    )
    context_md = format_agent_context_markdown(bundle)
    peer_section, _ = memory.format_peer_inbox_markdown(agent_id)
    peer_agent_ids = ", ".join(a.id for a in list_agents(settings))
    tool_hints = (
        NOTES_SYSTEM_HINT
        + GARDEN_SYSTEM_HINT
        + GARDEN_RUMINATION_HINT
        + PERSONAL_DB_HINT
        + CODEBASE_SYSTEM_HINT
        + WEB_SYSTEM_HINT
        + format_peer_message_hint(peer_agent_ids=peer_agent_ids)
    )
    persona = load_persona(agent_id)
    system_scheduled = build_rumination_system_content(
        agent_id=agent_id,
        context_md=context_md,
        peer_section=peer_section,
        wake_kind=None,
        tool_hints=tool_hints,
    )
    return {
        "agent_id": agent_id,
        "thread_id": agent.thread_id,
        "context_markdown": context_md,
        "meta": {
            "persona_chars": len(persona),
            "persona_first_line": persona.splitlines()[0][:120] if persona else "",
            "stream_events": bundle.stream_event_count,
            "stream_chars": bundle.stream_char_count,
            "scheduled_system_chars": len(system_scheduled),
            "default_task_hint": "scheduled",
            "tool_names": [t.name for t in LIGHT_TOOLS],
        },
    }
