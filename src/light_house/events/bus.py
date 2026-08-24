"""In-process event bus: NDJSON log + forward to existing wake handlers."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from light_house.config import Settings
from light_house.events.subscriptions import agent_accepts_event
from light_house.events.types import EventType, LightHouseEvent
from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

_LOG_LOCK = threading.Lock()
_settings: Settings | None = None
_memory: MemoryService | None = None
_rumination_graph: Any = None
_app_loop: asyncio.AbstractEventLoop | None = None
_queue: asyncio.Queue[LightHouseEvent] | None = None


def register_event_bus(
    *,
    settings: Settings,
    memory: MemoryService,
    rumination_graph: Any,
    app_loop: asyncio.AbstractEventLoop,
) -> None:
    """Wire bus dependencies (called once at startup)."""
    global _settings, _memory, _rumination_graph, _app_loop, _queue
    _settings = settings
    _memory = memory
    _rumination_graph = rumination_graph
    _app_loop = app_loop
    _queue = asyncio.Queue()


async def start_event_bus(*, cancel_event: asyncio.Event) -> None:
    """Run the async dispatcher until cancel_event is set."""
    if _settings is None or not _settings.event_bus_enabled:
        logger.info("Event bus disabled (EVENT_BUS_ENABLED=false)")
        return
    log_path = _settings.event_bus_log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Event bus started (log=%s)", log_path)
    while not cancel_event.is_set():
        if _queue is None:
            await asyncio.sleep(0.5)
            continue
        try:
            event = await asyncio.wait_for(_queue.get(), timeout=1.0)
        except TimeoutError:
            continue
        try:
            await _dispatch_event(event)
        except Exception:
            logger.exception(
                "Event dispatch failed type=%s id=%s",
                event.event_type.value,
                event.event_id,
            )


def publish(event: LightHouseEvent) -> None:
    """Publish an event: log + queue when enabled, else dispatch immediately."""
    settings = _settings
    if settings is None:
        logger.warning("Event dropped (bus not registered): %s", event.event_type.value)
        return

    if settings.event_bus_enabled:
        _append_log(settings, event)
        loop = _app_loop
        queue = _queue
        if loop is None or queue is None:
            logger.warning("Event logged but not queued (no app loop): %s", event.event_id)
            return

        def _enqueue() -> None:
            assert queue is not None
            queue.put_nowait(event)

        loop.call_soon_threadsafe(_enqueue)
        logger.debug(
            "Event queued type=%s agent=%s thread=%s id=%s",
            event.event_type.value,
            event.agent_id,
            event.thread_id,
            event.event_id,
        )
        return

    _submit_dispatch(event)


def _submit_dispatch(event: LightHouseEvent) -> None:
    """Schedule dispatch on the app loop (bus disabled — direct forward)."""
    loop = _app_loop
    if loop is None or not loop.is_running():
        logger.debug(
            "Event bus passthrough skipped (no loop): %s",
            event.event_type.value,
        )
        return
    asyncio.run_coroutine_threadsafe(_dispatch_event(event), loop)


def _append_log(settings: Settings, event: LightHouseEvent) -> None:
    path = settings.event_bus_log_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event.to_log_record(), ensure_ascii=False) + "\n"
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def read_event_log_tail(path: Path, *, max_lines: int) -> tuple[str, int, bool]:
    """Return tail of NDJSON event log (newest lines)."""
    if not path.is_file():
        return "", 0, False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", 0, False
    if not lines:
        return "", 0, False
    truncated = len(lines) > max_lines
    tail = lines[-max_lines:] if truncated else lines
    return "\n".join(tail) + ("\n" if tail else ""), len(tail), truncated


def format_event_log_display(raw_content: str, *, truncated: bool) -> str:
    """Human-readable lines from NDJSON event log tail (no private data)."""
    if not raw_content.strip():
        return "(event log empty)"
    lines: list[str] = []
    if truncated:
        lines.append("(event log tail truncated — showing recent entries only)")
    for line in raw_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            lines.append(line)
            continue
        if not isinstance(row, dict):
            continue
        ts = row.get("ts")
        ts_str = (
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))
            if isinstance(ts, (int, float))
            else "?"
        )
        event_type = row.get("event_type", "?")
        agent = row.get("agent_id") or "-"
        thread = row.get("thread_id") or "-"
        payload = row.get("payload")
        payload_brief = ""
        if isinstance(payload, dict) and payload:
            parts = []
            for key, value in list(payload.items())[:4]:
                text = str(value)
                if len(text) > 60:
                    text = text[:57] + "..."
                parts.append(f"{key}={text}")
            payload_brief = " · " + ", ".join(parts)
        lines.append(
            f"[{ts_str}] {event_type} · agent={agent} · thread={thread}{payload_brief}"
        )
    return "\n".join(lines)


async def _dispatch_event(event: LightHouseEvent) -> None:
    """Route event to the existing wake implementation (no behavior change in phase 1)."""
    match event.event_type:
        case EventType.CHAT_RECEIVED:
            await _handle_chat_received(event)
        case EventType.PEER_MESSAGE:
            await _handle_peer_message(event)
        case EventType.SHARED_NOTE_SAVED:
            await _handle_shared_note_saved(event)
        case EventType.SCHEDULED_WAKE:
            await _handle_scheduled_wake(event)
        case EventType.MAINTENANCE_WAKE:
            await _handle_maintenance_wake(event)
        case EventType.TOOL_COMPLETED | EventType.REPORT_READY:
            logger.debug(
                "Event type %s logged only (no handler yet) id=%s",
                event.event_type.value,
                event.event_id,
            )
        case _:
            logger.warning("Unknown event type: %s", event.event_type)


async def _handle_chat_received(event: LightHouseEvent) -> None:
    from light_house.agent.post_chat_wake import wake_agent_after_chat

    if not event.agent_id or not event.thread_id:
        logger.warning("chat_received missing agent_id/thread_id")
        return
    if _settings and not agent_accepts_event(_settings, event.agent_id, EventType.CHAT_RECEIVED):
        logger.info(
            "Wake skipped (subscription off) agent=%s event=post_chat",
            event.agent_id,
        )
        return
    await wake_agent_after_chat(agent_id=event.agent_id, thread_id=event.thread_id)


async def _handle_peer_message(event: LightHouseEvent) -> None:
    from light_house.agent.peer_chat_wake import wake_agent_for_peer_message

    payload = event.payload
    to_agent_id = event.agent_id or payload.get("to_agent_id")
    from_agent_id = payload.get("from_agent_id")
    message_id = payload.get("message_id")
    if not to_agent_id or not from_agent_id or not message_id:
        logger.warning("peer_message event missing fields: %s", event.event_id)
        return
    if _settings and not agent_accepts_event(_settings, str(to_agent_id), EventType.PEER_MESSAGE):
        logger.info(
            "Wake skipped (subscription off) agent=%s event=peer_message",
            to_agent_id,
        )
        return
    await wake_agent_for_peer_message(
        to_agent_id=str(to_agent_id),
        from_agent_id=str(from_agent_id),
        message_id=str(message_id),
    )


async def _handle_shared_note_saved(event: LightHouseEvent) -> None:
    from light_house.agent.rumination_wake import wake_agents_for_shared_note

    path = event.payload.get("path")
    if not isinstance(path, str) or not path.strip():
        logger.warning("shared_note_saved missing path")
        return
    await wake_agents_for_shared_note(path=path)


async def _handle_scheduled_wake(event: LightHouseEvent) -> None:
    from light_house.agent.inner_life_scheduler import trigger_scheduled_rumination

    if _settings is None or _memory is None or _rumination_graph is None:
        logger.warning("scheduled_wake skipped (bus not fully registered)")
        return
    if not event.agent_id or not event.thread_id:
        logger.warning("scheduled_wake missing agent_id/thread_id")
        return
    if _settings and not agent_accepts_event(_settings, event.agent_id, EventType.SCHEDULED_WAKE):
        logger.info(
            "Wake skipped (subscription off) agent=%s event=scheduled_rumination",
            event.agent_id,
        )
        return
    await trigger_scheduled_rumination(
        settings=_settings,
        memory=_memory,
        rumination_graph=_rumination_graph,
        thread_id=event.thread_id,
        agent_id=event.agent_id,
    )


async def _handle_maintenance_wake(event: LightHouseEvent) -> None:
    from light_house.agent.maintenance_wake import trigger_maintenance_rumination
    from light_house.personal.maintenance_time import mark_maintenance_slot

    if _settings is None or _rumination_graph is None:
        logger.warning("maintenance_wake skipped (bus not fully registered)")
        return
    if not event.agent_id or not event.thread_id:
        logger.warning("maintenance_wake missing agent_id/thread_id")
        return
    if _settings and not agent_accepts_event(_settings, event.agent_id, EventType.MAINTENANCE_WAKE):
        logger.info(
            "Wake skipped (subscription off) agent=%s event=memory_maintenance",
            event.agent_id,
        )
        return
    await trigger_maintenance_rumination(
        settings=_settings,
        rumination_graph=_rumination_graph,
        thread_id=event.thread_id,
        agent_id=event.agent_id,
    )
    mark_maintenance_slot(_settings, event.agent_id)
