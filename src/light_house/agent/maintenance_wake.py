"""Dedicated memory-maintenance ruminations (separate from scheduled free reflection)."""

from __future__ import annotations

import logging
from typing import Any

from light_house.agent.rumination_internal import rumination_internal_state_defaults
from light_house.config import Settings
from light_house.inner_life_trace import append_rumination_error

logger = logging.getLogger(__name__)

WAKE_KIND_MEMORY_MAINTENANCE = "memory_maintenance"


def _run_maintenance_rumination(
    *,
    graph: Any,
    thread_id: str,
    agent_id: str,
    settings: Settings,
) -> None:
    try:
        graph.invoke(
            {
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
                "wake_kind": WAKE_KIND_MEMORY_MAINTENANCE,
                **rumination_internal_state_defaults(settings),
            }
        )
    except Exception as exc:
        logger.exception(
            "Memory maintenance rumination failed (non-fatal) agent=%s thread_id=%s",
            agent_id,
            thread_id,
        )
        append_rumination_error(
            settings,
            agent_id=agent_id,
            thread_id=thread_id,
            wake_kind=WAKE_KIND_MEMORY_MAINTENANCE,
            error=str(exc),
        )


async def trigger_maintenance_rumination(
    *,
    settings: Settings,
    rumination_graph: Any,
    thread_id: str,
    agent_id: str,
) -> None:
    """Run one memory-maintenance wake (learnings + scoring only)."""
    import asyncio

    await asyncio.to_thread(
        _run_maintenance_rumination,
        graph=rumination_graph,
        thread_id=thread_id,
        agent_id=agent_id,
        settings=settings,
    )
