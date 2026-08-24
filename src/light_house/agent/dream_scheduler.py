"""Background asyncio scheduler for Echo's daily dreams (separate from rumination)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from light_house.config import Settings
from light_house.memory.service import MemoryService
from light_house.personal.time_sense import dream_due

logger = logging.getLogger(__name__)


def _run_echo_dream(*, graph: Any, thread_id: str, agent_id: str) -> None:
    try:
        graph.invoke(
            {
                "thread_id": thread_id,
                "agent_id": agent_id,
                "context_text": "",
                "dream_text": "",
                "waking_recall": "",
                "felt_days": None,
                "dream_round": 0,
                "max_dream_rounds": 3,
                "story_beats": [],
                "light_choices": [],
                "current_beat": "",
                "current_choice": "",
            }
        )
    except Exception:
        logger.exception("Echo dream failed (non-fatal) agent=%s thread_id=%s", agent_id, thread_id)


def _dream_is_due(
    *,
    settings: Settings,
    memory: MemoryService,
    thread_id: str,
    agent_id: str,
) -> bool:
    if settings.dream_calendar_mode:
        return dream_due(settings, agent_id)
    dream_hours = max(1.0, settings.inner_life_dream_interval_hours)
    hours_since = memory.hours_since_last_dream(thread_id=thread_id)
    return hours_since is None or hours_since >= dream_hours


async def run_echo_dream_scheduler(
    *,
    settings: Settings,
    memory: MemoryService,
    dream_graph: Any,
    cancel_event: asyncio.Event,
    thread_id: str,
    agent_id: str = "lumen",
) -> None:
    """Poll periodically; run Echo's dream graph when the daily slot is due."""
    if not settings.inner_life_dreams_enabled:
        logger.info("Echo dream scheduler disabled (INNER_LIFE_DREAMS_ENABLED=false)")
        return

    poll_sec = max(60, settings.echo_dream_poll_seconds)
    if settings.dream_calendar_mode:
        logger.info(
            "Echo dream scheduler started (agent=%s, thread_id=%s, poll=%ds, calendar=%s@%d:00)",
            agent_id,
            thread_id,
            poll_sec,
            settings.dream_timezone,
            settings.dream_local_hour,
        )
    else:
        dream_hours = max(1.0, settings.inner_life_dream_interval_hours)
        logger.info(
            "Echo dream scheduler started (agent=%s, thread_id=%s, poll=%ds, interval=%.0fh)",
            agent_id,
            thread_id,
            poll_sec,
            dream_hours,
        )

    while not cancel_event.is_set():
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=poll_sec)
            break
        except TimeoutError:
            pass

        if cancel_event.is_set():
            break

        if _dream_is_due(
            settings=settings,
            memory=memory,
            thread_id=thread_id,
            agent_id=agent_id,
        ):
            logger.info("Echo dream due (agent=%s, thread_id=%s)", agent_id, thread_id)
            await asyncio.to_thread(
                _run_echo_dream,
                graph=dream_graph,
                thread_id=thread_id,
                agent_id=agent_id,
            )

    logger.info("Echo dream scheduler stopped (agent=%s)", agent_id)
