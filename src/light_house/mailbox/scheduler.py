"""Poll mailbox .queue and wake addressed lights."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from light_house.config import Settings
from light_house.mailbox.letters import (
    list_pending_notifies,
    mark_notify_done,
    resolve_notes_root,
)
from light_house.mailbox.wake import wake_agents_for_mailbox_letter

logger = logging.getLogger(__name__)


async def process_mailbox_queue(*, settings: Settings) -> int:
    """Process all pending notify files. Returns count processed."""
    notes_root = resolve_notes_root(settings)
    pending = list_pending_notifies(notes_root)
    processed = 0
    for notify_path in pending:
        try:
            raw = json.loads(notify_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Bad mailbox notify file %s: %s", notify_path, exc)
            mark_notify_done(notes_root, notify_path)
            continue
        path = str(raw.get("path") or "").strip()
        to_raw = raw.get("to") or []
        if isinstance(to_raw, str):
            to_ids = [to_raw]
        elif isinstance(to_raw, list):
            to_ids = [str(x).strip() for x in to_raw if str(x).strip()]
        else:
            to_ids = []
        if not path or not to_ids:
            logger.warning("Mailbox notify missing path/to: %s", notify_path.name)
            mark_notify_done(notes_root, notify_path)
            continue
        await wake_agents_for_mailbox_letter(path=path, to_agent_ids=to_ids, settings=settings)
        mark_notify_done(notes_root, notify_path)
        processed += 1
    return processed


async def run_mailbox_scheduler(
    *,
    settings: Settings,
    cancel_event: asyncio.Event,
) -> None:
    if not settings.mailbox_wake_enabled:
        logger.info("Mailbox scheduler disabled (MAILBOX_WAKE_ENABLED=false)")
        return
    poll_sec = max(5, int(settings.mailbox_poll_seconds))
    notes_root = resolve_notes_root(settings)
    logger.info(
        "Mailbox scheduler started poll=%ds queue=%s",
        poll_sec,
        notes_root / "shared" / "mailbox" / ".queue",
    )
    while not cancel_event.is_set():
        try:
            n = await process_mailbox_queue(settings=settings)
            if n:
                logger.info("Mailbox scheduler processed %d notify(s)", n)
        except Exception:  # noqa: BLE001
            logger.exception("Mailbox scheduler tick failed (non-fatal)")
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=poll_sec)
        except asyncio.TimeoutError:
            continue
    logger.info("Mailbox scheduler stopped")


def ensure_mailbox_dirs(settings: Settings) -> Path:
    """Create mailbox folder skeleton; return shared mailbox path."""
    from light_house.mailbox.letters import (
        done_dir,
        queue_dir,
        shared_mailbox_dir,
    )

    notes_root = resolve_notes_root(settings)
    root = shared_mailbox_dir(notes_root)
    (root / "from_reed").mkdir(parents=True, exist_ok=True)
    (root / "to_reed").mkdir(parents=True, exist_ok=True)
    queue_dir(notes_root).mkdir(parents=True, exist_ok=True)
    done_dir(notes_root).mkdir(parents=True, exist_ok=True)
    return root
