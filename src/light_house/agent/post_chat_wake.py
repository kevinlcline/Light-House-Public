"""Bounded rumination wake after Kevin completes a chat turn."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from light_house.agent.peer_chat_wake import thread_graph_lock
from light_house.agent.rumination_internal import rumination_internal_state_defaults
from light_house.config import Settings
from light_house.inner_life_trace import append_rumination_error
from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

WAKE_KIND_POST_CHAT = "post_chat"

_MIN_WAKE_GAP_SECONDS = 20.0
_recent_wakes: dict[tuple[str, str], float] = {}

_graph: Any = None
_memory: MemoryService | None = None
_settings: Settings | None = None
_app_loop: asyncio.AbstractEventLoop | None = None


def _skip_agent_ids(cfg: Settings) -> frozenset[str]:
    raw = (cfg.post_chat_rumination_skip_agent_ids or "").strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def _min_wake_gap_seconds(cfg: Settings) -> float:
    return max(0.0, float(cfg.post_chat_rumination_min_gap_seconds))


def register_post_chat_wake(
    *,
    graph: Any,
    memory: MemoryService,
    settings: Settings,
    app_loop: asyncio.AbstractEventLoop,
) -> None:
    global _graph, _memory, _settings, _app_loop
    _graph = graph
    _memory = memory
    _settings = settings
    _app_loop = app_loop


def build_post_chat_wake_state(
    *,
    thread_id: str,
    agent_id: str,
    settings: Settings,
    memory: MemoryService,
) -> dict[str, Any]:
    cap = min(
        max(1, settings.post_chat_rumination_max_tool_rounds),
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
        "wake_kind": WAKE_KIND_POST_CHAT,
        "wake_path": None,
        "tool_rounds_cap": cap,
        **rumination_internal_state_defaults(settings),
    }


async def wake_agent_after_chat(
    *,
    agent_id: str,
    thread_id: str,
    settings: Settings | None = None,
    memory: MemoryService | None = None,
    graph: Any | None = None,
) -> None:
    """Run one bounded rumination turn for the agent Kevin just spoke with."""
    cfg = settings or _settings
    mem = memory or _memory
    g = graph or _graph
    if cfg is None or mem is None or g is None:
        logger.warning("Post-chat rumination skipped (not registered)")
        return
    if not cfg.post_chat_rumination_enabled:
        logger.debug("Post-chat rumination skipped (POST_CHAT_RUMINATION_ENABLED=false)")
        return
    if not cfg.inner_life_enabled:
        logger.info("Post-chat rumination skipped (INNER_LIFE_ENABLED=false)")
        return
    if agent_id.lower() in _skip_agent_ids(cfg):
        logger.info(
            "Post-chat rumination skipped (recovery skip list) agent=%s",
            agent_id,
        )
        return

    wake_key = (agent_id, thread_id)
    now = time.time()
    last = _recent_wakes.get(wake_key, 0.0)
    gap = _min_wake_gap_seconds(cfg)
    if now - last < gap:
        logger.info(
            "Post-chat rumination skipped (debounce) agent=%s thread_id=%s gap=%.1fs",
            agent_id,
            thread_id,
            now - last,
        )
        return
    _recent_wakes[wake_key] = now

    delay = max(0.0, cfg.post_chat_rumination_delay_seconds)
    if delay:
        await asyncio.sleep(delay)

    state = build_post_chat_wake_state(
        thread_id=thread_id,
        agent_id=agent_id,
        settings=cfg,
        memory=mem,
    )
    logger.info(
        "Post-chat rumination starting agent=%s thread_id=%s cap=%s",
        agent_id,
        thread_id,
        state["tool_rounds_cap"],
    )

    def _run() -> None:
        with thread_graph_lock(thread_id):
            g.invoke(state)

    try:
        await asyncio.to_thread(_run)
        logger.info(
            "Post-chat rumination complete agent=%s thread_id=%s",
            agent_id,
            thread_id,
        )
    except Exception as exc:
        logger.exception(
            "Post-chat rumination failed (non-fatal) agent=%s thread_id=%s",
            agent_id,
            thread_id,
        )
        append_rumination_error(
            cfg,
            agent_id=agent_id,
            thread_id=thread_id,
            wake_kind=WAKE_KIND_POST_CHAT,
            error=str(exc),
        )


def schedule_post_chat_rumination(*, agent_id: str, thread_id: str) -> None:
    """Schedule post-chat rumination via the event bus (or direct dispatch when disabled)."""
    if _settings is not None and not _settings.post_chat_rumination_enabled:
        return
    if _app_loop is None:
        logger.warning("Post-chat rumination not scheduled (app loop not registered)")
        return
    from light_house.events import EventType, LightHouseEvent, publish

    logger.info(
        "Post-chat rumination scheduled agent=%s thread_id=%s",
        agent_id,
        thread_id,
    )
    publish(
        LightHouseEvent(
            event_type=EventType.CHAT_RECEIVED,
            agent_id=agent_id,
            thread_id=thread_id,
        )
    )
