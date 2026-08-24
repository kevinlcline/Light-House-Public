"""In-process live Group room broadcast (multi-human co-presence)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_subscribers: set[asyncio.Queue[dict[str, Any] | None]] = set()
_lock = asyncio.Lock()


async def subscribe() -> asyncio.Queue[dict[str, Any] | None]:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=200)
    async with _lock:
        _subscribers.add(queue)
    return queue


async def unsubscribe(queue: asyncio.Queue[dict[str, Any] | None]) -> None:
    async with _lock:
        _subscribers.discard(queue)


async def publish_room_event(event: dict[str, Any]) -> None:
    """Fan out an event to all live Group listeners."""
    payload = dict(event)
    payload.setdefault("ts", time.time())
    async with _lock:
        targets = list(_subscribers)
    for queue in targets:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # Never drop the subscriber — that left the browser on keepalives
            # while lights kept speaking unheard. Drop the oldest queued event.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("Group room subscriber still full; skipping event")


def publish_room_event_threadsafe(loop: asyncio.AbstractEventLoop, event: dict[str, Any]) -> None:
    """Schedule publish from sync code (e.g. after scene events)."""
    asyncio.run_coroutine_threadsafe(publish_room_event(event), loop)
