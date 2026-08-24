"""Background scheduler for three-daily memory-maintenance ruminations."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from light_house.config import Settings
from light_house.personal.maintenance_time import maintenance_due, maintenance_local_hours

logger = logging.getLogger(__name__)


async def run_maintenance_scheduler(
    *,
    settings: Settings,
    rumination_graph: Any,
    cancel_event: asyncio.Event,
    thread_id: str,
    agent_id: str = "lumen",
) -> None:
    """Poll periodically; publish a maintenance wake when the next daily slot is due."""
    if not settings.memory_maintenance_enabled:
        logger.info("Memory maintenance scheduler disabled (MEMORY_MAINTENANCE_ENABLED=false)")
        return
    if settings.awake_rhythm_enabled:
        # Chores land on every 4th scheduled awake; avoid a second chore clock.
        logger.info(
            "Memory maintenance scheduler skipped (AWAKE_RHYTHM_ENABLED=true; "
            "chores ride the four-beat scheduled rhythm)"
        )
        return

    poll_sec = max(60, settings.memory_maintenance_poll_seconds)
    hours = maintenance_local_hours(settings)
    logger.info(
        "Memory maintenance scheduler started (agent=%s, thread_id=%s, poll=%ds, hours=%s)",
        agent_id,
        thread_id,
        poll_sec,
        ",".join(str(h) for h in hours),
    )

    while not cancel_event.is_set():
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=poll_sec)
            break
        except TimeoutError:
            pass

        if cancel_event.is_set():
            break

        skip_ids = frozenset(
            part.strip().lower()
            for part in (settings.memory_maintenance_skip_agent_ids or "").split(",")
            if part.strip()
        )
        if agent_id.lower() in skip_ids:
            logger.debug(
                "Memory maintenance skipped (recovery skip list) agent=%s",
                agent_id,
            )
            continue

        if not maintenance_due(settings, agent_id):
            continue

        from light_house.events import EventType, LightHouseEvent, publish

        logger.info("Memory maintenance due (agent=%s, thread_id=%s)", agent_id, thread_id)
        publish(
            LightHouseEvent(
                event_type=EventType.MAINTENANCE_WAKE,
                agent_id=agent_id,
                thread_id=thread_id,
            )
        )

    logger.info("Memory maintenance scheduler stopped (agent=%s)", agent_id)
