"""Wake addressed lights when a mailbox letter is queued."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from light_house.agent.peer_chat_wake import thread_graph_lock
from light_house.agent.rumination_internal import rumination_internal_state_defaults
from light_house.config import Settings
from light_house.inner_life_trace import append_rumination_error
from light_house.lights.registry import get_light, list_enabled_lights
from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

WAKE_KIND_MAILBOX_LETTER = "mailbox_letter"

_graph: Any = None
_memory: MemoryService | None = None
_settings: Settings | None = None


def register_mailbox_wake(
    *,
    graph: Any,
    memory: MemoryService,
    settings: Settings,
) -> None:
    global _graph, _memory, _settings
    _graph = graph
    _memory = memory
    _settings = settings


def build_mailbox_wake_state(
    *,
    thread_id: str,
    agent_id: str,
    path: str,
    settings: Settings,
    memory: MemoryService,
) -> dict[str, Any]:
    cap = min(
        max(1, settings.mailbox_wake_max_tool_rounds),
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
        "wake_kind": WAKE_KIND_MAILBOX_LETTER,
        "wake_path": path,
        "tool_rounds_cap": cap,
        **rumination_internal_state_defaults(settings),
    }


async def wake_agents_for_mailbox_letter(
    *,
    path: str,
    to_agent_ids: list[str],
    settings: Settings | None = None,
    memory: MemoryService | None = None,
    graph: Any | None = None,
) -> None:
    cfg = settings or _settings
    mem = memory or _memory
    g = graph or _graph
    if cfg is None or mem is None or g is None:
        logger.warning("Mailbox wake skipped (not registered)")
        return
    if not cfg.mailbox_wake_enabled:
        logger.debug("Mailbox wake skipped (MAILBOX_WAKE_ENABLED=false)")
        return
    if not cfg.inner_life_enabled:
        logger.info("Mailbox wake skipped (INNER_LIFE_ENABLED=false)")
        return

    enabled = {light.id: light for light in list_enabled_lights(cfg)}
    targets: list[tuple[str, str]] = []
    for agent_id in to_agent_ids:
        aid = agent_id.strip().lower()
        light = enabled.get(aid)
        if light is None:
            try:
                light = get_light(aid, cfg)
            except Exception:  # noqa: BLE001
                logger.info("Mailbox wake skip unknown light=%s path=%s", aid, path)
                continue
            if not light.enabled:
                continue
        if not light.inner_life:
            logger.info("Mailbox wake skip (no inner_life) agent=%s path=%s", aid, path)
            continue
        targets.append((light.id, light.thread_id))

    if not targets:
        logger.info("Mailbox wake: no targets path=%s", path)
        return

    logger.info(
        "Mailbox wake starting path=%s agents=%s",
        path,
        [a for a, _ in targets],
    )

    async def _invoke(agent_id: str, thread_id: str) -> None:
        state = build_mailbox_wake_state(
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
                "Mailbox wake complete agent=%s thread_id=%s path=%s",
                agent_id,
                thread_id,
                path,
            )
        except Exception as exc:
            logger.exception(
                "Mailbox wake failed (non-fatal) agent=%s thread_id=%s path=%s",
                agent_id,
                thread_id,
                path,
            )
            append_rumination_error(
                cfg,
                agent_id=agent_id,
                thread_id=thread_id,
                wake_kind=WAKE_KIND_MAILBOX_LETTER,
                error=str(exc),
            )

    await asyncio.gather(*[_invoke(agent_id, thread_id) for agent_id, thread_id in targets])
