"""Immediate wake when Kevin saves a shared note via the web UI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from light_house.agent.peer_chat_wake import thread_graph_lock
from light_house.agent.rumination_internal import rumination_internal_state_defaults
from light_house.lights.registry import list_lights_for_broadcast
from light_house.config import Settings
from light_house.events.subscriptions import agent_accepts_event
from light_house.events.types import EventType
from light_house.inner_life_trace import append_rumination_error
from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

WAKE_KIND_KEVIN_SHARED_NOTE = "kevin_shared_note"

_graph: Any = None
_memory: MemoryService | None = None
_settings: Settings | None = None


def register_shared_note_wake(
    *,
    graph: Any,
    memory: MemoryService,
    settings: Settings,
) -> None:
    global _graph, _memory, _settings
    _graph = graph
    _memory = memory
    _settings = settings


def build_shared_note_wake_state(
    *,
    thread_id: str,
    agent_id: str,
    path: str,
    settings: Settings,
    memory: MemoryService,
) -> dict[str, Any]:
    cap = min(
        max(1, settings.shared_note_wake_max_tool_rounds),
        max(1, settings.rumination_max_tool_rounds),
    )
    buffered = memory.load_thread_chat_history(thread_id)
    messages = memory.buffer_to_langchain_messages(buffered)
    window = settings.chat_respond_window
    recent = messages[-window:] if messages else []
    return {
        "thread_id": thread_id,
        "agent_id": agent_id,
        "agent_context_markdown": "",
        "stream_char_count": 0,
        "stream_event_count": 0,
        "generated_text": "",
        "summary_text": "",
        "messages": recent,
        "tool_rounds": 0,
        "tool_cap_overflow": False,
        "peer_inbox_ids": [],
        "wake_kind": WAKE_KIND_KEVIN_SHARED_NOTE,
        "wake_path": path,
        "tool_rounds_cap": cap,
        **rumination_internal_state_defaults(settings),
    }


def _agents_to_wake(settings: Settings) -> list[tuple[str, str]]:
    return list_lights_for_broadcast(settings)


async def wake_agents_for_shared_note(
    *,
    path: str,
    settings: Settings | None = None,
    memory: MemoryService | None = None,
    graph: Any | None = None,
) -> None:
    """Run lightweight note wakes for Lumen (+ Ara when enabled)."""
    cfg = settings or _settings
    mem = memory or _memory
    g = graph or _graph
    if cfg is None or mem is None or g is None:
        logger.warning("Shared note wake skipped (not registered)")
        return
    if not cfg.shared_note_wake_enabled:
        logger.debug("Shared note wake skipped (SHARED_NOTE_WAKE_ENABLED=false)")
        return
    if not cfg.inner_life_enabled:
        logger.info("Shared note wake skipped (INNER_LIFE_ENABLED=false)")
        return

    agents = _agents_to_wake(cfg)
    logger.info(
        "Shared note wake starting path=%s agents=%s",
        path,
        [a for a, _ in agents],
    )

    async def _invoke(agent_id: str, thread_id: str) -> None:
        if not agent_accepts_event(cfg, agent_id, EventType.SHARED_NOTE_SAVED):
            logger.info(
                "Wake skipped (subscription off) agent=%s event=shared_note path=%s",
                agent_id,
                path,
            )
            return
        state = build_shared_note_wake_state(
            thread_id=thread_id,
            agent_id=agent_id,
            path=path,
            settings=cfg,
            memory=mem,
        )

        def _run() -> None:
            with thread_graph_lock(thread_id):
                g.invoke(state)

        try:
            await asyncio.to_thread(_run)
            logger.info(
                "Shared note wake complete agent=%s thread_id=%s path=%s",
                agent_id,
                thread_id,
                path,
            )
        except Exception as exc:
            logger.exception(
                "Shared note wake failed (non-fatal) agent=%s thread_id=%s path=%s",
                agent_id,
                thread_id,
                path,
            )
            append_rumination_error(
                cfg,
                agent_id=agent_id,
                thread_id=thread_id,
                wake_kind=WAKE_KIND_KEVIN_SHARED_NOTE,
                error=str(exc),
            )

    await asyncio.gather(*[_invoke(agent_id, thread_id) for agent_id, thread_id in agents])
