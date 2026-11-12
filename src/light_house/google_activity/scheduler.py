"""Timer that writes the Google arrival log. It never starts a light."""

from __future__ import annotations

import asyncio
import logging

from light_house.config import Settings
from light_house.google_activity.poll import poll_google_activity

logger = logging.getLogger(__name__)


async def run_google_activity_scheduler(
    *,
    settings: Settings,
    cancel_event: asyncio.Event,
) -> None:
    interval = max(30, int(settings.google_activity_poll_seconds))
    logger.info(
        "Google activity log started (every %ss). Arrivals are logged. Lights are not started.",
        interval,
    )
    while not cancel_event.is_set():
        try:
            written = await asyncio.to_thread(poll_google_activity, settings)
            if written:
                logger.info("Google activity log appended %s line(s)", written)
        except Exception:
            logger.exception("Google activity poll failed")
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=interval)
        except TimeoutError:
            continue
    logger.info("Google activity log stopped")
