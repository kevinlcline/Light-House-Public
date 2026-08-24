"""Immediate main-graph wake when one agent messages another."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any

from light_house.agent.peer_wake_context import PeerWakeContext, clear_peer_wake_context, set_peer_wake_context
from light_house.agents.registry import get_agent
from light_house.config import Settings
from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

WAKE_KIND_PEER_MESSAGE = "peer_message"

_MIN_WAKE_GAP_SECONDS = 20.0
_recent_wakes: dict[tuple[str, str], float] = {}

_graph: Any = None
_memory: MemoryService | None = None
_settings: Settings | None = None
_app_loop: asyncio.AbstractEventLoop | None = None

_thread_locks: dict[str, threading.Lock] = {}
_lock_guard = threading.Lock()


def register_peer_chat_wake(
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


@contextmanager
def thread_graph_lock(thread_id: str):
    """Serialize graph invokes per thread (Kevin chat and peer wake)."""
    with _lock_guard:
        lock = _thread_locks.setdefault(thread_id, threading.Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def build_peer_chat_wake_state(
    *,
    thread_id: str,
    agent_id: str,
    from_agent_id: str,
    message_id: str,
    settings: Settings,
    memory: MemoryService,
) -> dict[str, Any]:
    cap = min(
        max(1, settings.peer_chat_wake_max_tool_rounds),
        max(1, settings.chat_max_tool_rounds),
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
        "retrieved_memories": [],
        "messages": recent,
        "tool_rounds": 0,
        "tool_cap_overflow": False,
        "peer_inbox_ids": [],
        "wake_kind": WAKE_KIND_PEER_MESSAGE,
        "wake_from_agent_id": from_agent_id,
        "peer_message_id": message_id,
        "tool_rounds_cap": cap,
    }


async def wake_agent_for_peer_message(
    *,
    to_agent_id: str,
    from_agent_id: str,
    message_id: str,
) -> None:
    """Run main chat graph for the receiver after a peer message."""
    if _graph is None or _memory is None or _settings is None:
        logger.warning("Peer chat wake skipped (not registered)")
        return
    if not _settings.peer_chat_wake_enabled:
        logger.debug("Peer chat wake skipped (PEER_CHAT_WAKE_ENABLED=false)")
        return

    wake_key = (to_agent_id, from_agent_id)
    now = time.time()
    last = _recent_wakes.get(wake_key, 0.0)
    if now - last < _MIN_WAKE_GAP_SECONDS:
        logger.info(
            "Peer chat wake skipped (debounce) from=%s to=%s gap=%.1fs",
            from_agent_id,
            to_agent_id,
            now - last,
        )
        return
    _recent_wakes[wake_key] = now

    receiver = get_agent(to_agent_id, _settings)
    thread_id = receiver.thread_id
    state = build_peer_chat_wake_state(
        thread_id=thread_id,
        agent_id=to_agent_id,
        from_agent_id=from_agent_id,
        message_id=message_id,
        settings=_settings,
        memory=_memory,
    )
    ctx = PeerWakeContext(
        from_agent_id=from_agent_id,
        to_agent_id=to_agent_id,
        message_id=message_id,
        receiver_thread_id=thread_id,
    )
    logger.info(
        "Peer chat wake starting from=%s to=%s message_id=%s",
        from_agent_id,
        to_agent_id,
        message_id,
    )

    def _invoke() -> None:
        set_peer_wake_context(ctx)
        try:
            with thread_graph_lock(thread_id):
                _graph.invoke(state)
        finally:
            clear_peer_wake_context()

    try:
        await asyncio.to_thread(_invoke)
        logger.info(
            "Peer chat wake complete from=%s to=%s message_id=%s",
            from_agent_id,
            to_agent_id,
            message_id,
        )
    except Exception:
        logger.exception(
            "Peer chat wake failed (non-fatal) from=%s to=%s message_id=%s",
            from_agent_id,
            to_agent_id,
            message_id,
        )


def schedule_peer_chat_wake(
    *,
    to_agent_id: str,
    from_agent_id: str,
    message_id: str,
) -> bool:
    """
    Schedule peer wake via the event bus.

    Returns False when peer chat is off or the pair's dialogue budget is exhausted
    (message may still have been delivered — caller decides).
    """
    if _app_loop is None:
        logger.warning("Peer chat wake not scheduled (app loop not registered)")
        return False
    if _settings is not None and not _settings.peer_chat_wake_enabled:
        return False
    from light_house.agent.peer_dialogue import begin_peer_wake_turn
    from light_house.events import EventType, LightHouseEvent, publish

    if _settings is not None:
        gate = begin_peer_wake_turn(from_agent_id, to_agent_id, settings=_settings)
        if not gate.allow_wake:
            logger.info(
                "Peer chat wake skipped (dialogue %s) from=%s to=%s turns=%d/%d",
                gate.reason,
                from_agent_id,
                to_agent_id,
                gate.turns,
                gate.max_turns,
            )
            return False

    publish(
        LightHouseEvent(
            event_type=EventType.PEER_MESSAGE,
            agent_id=to_agent_id,
            payload={
                "from_agent_id": from_agent_id,
                "message_id": message_id,
            },
        )
    )
    return True
