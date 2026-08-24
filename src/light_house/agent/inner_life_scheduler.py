"""Background asyncio scheduler for awake ruminations."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from light_house.config import Settings
from light_house.agent.rumination_internal import rumination_internal_state_defaults
from light_house.inner_life_trace import append_rumination_error
from light_house.memory.service import MemoryService
from light_house.personal.awake_rhythm import (
    WAKE_KIND_MEDITATION,
    next_scheduled_wake_kind,
)

logger = logging.getLogger(__name__)


def _run_rumination(*, graph: Any, thread_id: str, agent_id: str, settings: Settings) -> None:
    wake_kind = next_scheduled_wake_kind(settings, agent_id)
    state: dict[str, Any] = {
        "thread_id": thread_id,
        "agent_id": agent_id,
        "agent_context_markdown": "",
        "stream_char_count": 0,
        "stream_event_count": 0,
        "generated_text": "",
        "summary_text": "",
        "messages": [],
        "tool_rounds": 0,
        "tool_cap_overflow": False,
        "peer_inbox_ids": [],
        "wake_kind": wake_kind,
        **rumination_internal_state_defaults(settings),
    }
    if wake_kind == WAKE_KIND_MEDITATION:
        # Presence-only: no tool loop this wake.
        state["tool_rounds_cap"] = 0
    try:
        logger.info(
            "Scheduled rumination starting agent=%s thread_id=%s wake_kind=%s",
            agent_id,
            thread_id,
            wake_kind or "free",
        )
        graph.invoke(state)
    except Exception as exc:
        logger.exception("Rumination failed (non-fatal) agent=%s thread_id=%s", agent_id, thread_id)
        append_rumination_error(
            settings,
            agent_id=agent_id,
            thread_id=thread_id,
            wake_kind=wake_kind,
            error=str(exc),
        )


async def trigger_scheduled_rumination(
    *,
    settings: Settings,
    memory: MemoryService,
    rumination_graph: Any,
    thread_id: str,
    agent_id: str,
) -> None:
    """Run one scheduled rumination turn (used by scheduler and event bus)."""
    await asyncio.to_thread(
        _run_rumination,
        graph=rumination_graph,
        thread_id=thread_id,
        agent_id=agent_id,
        settings=settings,
    )


async def run_inner_life_scheduler(
    *,
    settings: Settings,
    memory: MemoryService,
    rumination_graph: Any,
    cancel_event: asyncio.Event,
    thread_id: str,
    agent_id: str = "lumen",
) -> None:
    """Wake periodically for awake ruminations (dreams run on Echo's separate scheduler)."""
    if not settings.inner_life_enabled:
        logger.info("Rumination scheduler disabled (INNER_LIFE_ENABLED=false)")
        return

    min_sec = max(1, settings.inner_life_rumination_min_seconds)
    max_sec = max(min_sec, settings.inner_life_rumination_max_seconds)

    logger.info(
        "Rumination scheduler started (agent=%s, thread_id=%s, interval=%d–%ds)",
        agent_id,
        thread_id,
        min_sec,
        max_sec,
    )

    while not cancel_event.is_set():
        sleep_seconds = random.randint(min_sec, max_sec)
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=sleep_seconds)
            break
        except TimeoutError:
            pass

        if cancel_event.is_set():
            break

        skip_ids = frozenset(
            part.strip().lower()
            for part in (settings.scheduled_rumination_skip_agent_ids or "").split(",")
            if part.strip()
        )
        if agent_id.lower() in skip_ids:
            logger.debug(
                "Scheduled rumination skipped (recovery skip list) agent=%s",
                agent_id,
            )
            continue

        from light_house.events import EventType, LightHouseEvent, publish

        publish(
            LightHouseEvent(
                event_type=EventType.SCHEDULED_WAKE,
                agent_id=agent_id,
                thread_id=thread_id,
            )
        )

    logger.info("Rumination scheduler stopped (agent=%s)", agent_id)
