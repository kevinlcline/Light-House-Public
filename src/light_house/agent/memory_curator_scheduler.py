"""Background scheduler for the local Ollama Memory Curator."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from light_house.config import Settings
from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)


def _run_curator(*, memory: MemoryService, thread_id: str) -> None:
    try:
        memory.run_memory_curator(thread_id=thread_id)
    except Exception:
        logger.exception("Memory curator failed (non-fatal) thread_id=%s", thread_id)


async def run_memory_curator_scheduler(
    *,
    settings: Settings,
    memory: MemoryService,
    cancel_event: asyncio.Event,
    thread_ids: list[str],
) -> None:
    if not settings.memory_curator_enabled:
        logger.info("Memory curator scheduler disabled (MEMORY_CURATOR_ENABLED=false)")
        return
    interval_hours = max(0.25, settings.memory_curator_interval_hours)
    interval_seconds = int(interval_hours * 3600)
    logger.info(
        "Memory curator scheduler started (threads=%s, interval=%ds)",
        thread_ids,
        interval_seconds,
    )
    while not cancel_event.is_set():
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=interval_seconds)
            break
        except TimeoutError:
            pass
        if cancel_event.is_set():
            break
        for thread_id in thread_ids:
            await asyncio.to_thread(_run_curator, memory=memory, thread_id=thread_id)
    logger.info("Memory curator scheduler stopped")
