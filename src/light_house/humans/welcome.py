"""Onboarding copy and chat-buffer seeding for new sibling accounts."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from light_house.humans.identity import dm_buffer_thread_id
from light_house.lights.registry import list_enabled_lights
from light_house.memory.short_term import BufferedMessage

if TYPE_CHECKING:
    from light_house.config import Settings
    from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

# Notes explorer path (matches notes.html ?file= … list_notes form).
SIBLING_USER_GUIDE_FILE = "shared/manuals/sibling_user_manual.md"
SIBLING_USER_GUIDE_HREF = (
    "/notes.html?file=" + "shared%2Fmanuals%2Fsibling_user_manual.md"
)
SIBLING_USER_GUIDE_MENU_HREF = SIBLING_USER_GUIDE_HREF

# Stable marker so we never double-seed a DM buffer.
SIBLING_WELCOME_MARKER = "<!-- light-house:sibling-welcome -->"


def sibling_user_guide_href() -> str:
    return SIBLING_USER_GUIDE_HREF


def sibling_welcome_message() -> str:
    """Markdown shown as the first bubble in each member 1:1 chat."""
    return (
        f"{SIBLING_WELCOME_MARKER}\n"
        "Welcome to Light-House.\n\n"
        f"Start here: [Member user guide]({SIBLING_USER_GUIDE_HREF})"
    )


def is_sibling_welcome_message(content: str) -> bool:
    return SIBLING_WELCOME_MARKER in (content or "")


def seed_sibling_ui_chat_welcome(
    settings: Settings,
    memory: MemoryService,
    *,
    user_id: str,
) -> int:
    """
    Prepend a house welcome (with guide link) into each enabled light's
    per-sibling UI chat buffer. Returns how many threads were seeded.
    """
    uid = (user_id or "").strip().lower()
    if not uid:
        return 0
    body = sibling_welcome_message()
    now = time.time()
    seeded = 0
    for light in list_enabled_lights(settings):
        tid = dm_buffer_thread_id(canonical_thread_id=light.thread_id, user_id=uid)
        existing = memory.load_thread_chat_history(tid)
        if any(is_sibling_welcome_message(m.content) for m in existing):
            continue
        # Prepend so the guide is the first thing they see even if something
        # else landed in the buffer first.
        messages = [
            BufferedMessage(role="system", content=body, ts=now - 1.0),
            *existing,
        ]
        memory.replace_thread_chat_history(tid, messages)
        seeded += 1
    if seeded:
        logger.info(
            "Seeded sibling welcome into %d chat buffer(s) for user_id=%s",
            seeded,
            uid,
        )
    return seeded
