"""Soft presence knock — a light asks Dad for a word without interrupting chat."""

from __future__ import annotations

import logging

from light_house.config import Settings
from light_house.personal.light_state_keys import WANTS_KEVIN
from light_house.personal.store import PersonalStore, get_personal_store

logger = logging.getLogger(__name__)

# Durable line left on Dad's chat buffer when a knock is newly raised.
PRESENCE_KNOCK_CHAT_LINE = "I'd like a word when you have a moment."


def _store(settings: Settings, agent_id: str) -> PersonalStore | None:
    if not settings.personal_db_enabled:
        return None
    try:
        return get_personal_store(settings, agent_id)
    except (KeyError, RuntimeError, OSError) as exc:
        logger.warning("Presence knock store unavailable agent=%s: %s", agent_id, exc)
        return None


def knock_pending(settings: Settings, agent_id: str) -> bool:
    store = _store(settings, agent_id)
    if store is None:
        return False
    return store.get_light_state(WANTS_KEVIN, default=0) > 0


def raise_knock(settings: Settings, agent_id: str) -> bool:
    """Set the knock. Returns True if stored (or already set)."""
    store = _store(settings, agent_id)
    if store is None:
        return False
    store.set_light_state(WANTS_KEVIN, 1)
    logger.info("Presence knock raised agent=%s", agent_id)
    return True


def clear_knock(settings: Settings, agent_id: str) -> bool:
    """Clear the knock. Returns True if store was available."""
    store = _store(settings, agent_id)
    if store is None:
        return False
    if store.get_light_state(WANTS_KEVIN, default=0) > 0:
        store.set_light_state(WANTS_KEVIN, 0)
        logger.info("Presence knock cleared agent=%s", agent_id)
    return True


def record_knock_chat_line(settings: Settings, agent_id: str) -> bool:
    """Append a visible assistant line on the light's Dad chat thread.

    Survives clearing the status-bar mark so opening that chat (or already being
    on it) cannot erase the signal entirely.
    """
    from light_house.agents.registry import get_agent
    from light_house.memory.service import MemoryService

    try:
        light = get_agent(agent_id, settings)
    except KeyError:
        logger.warning("Presence knock chat line skipped; unknown agent=%s", agent_id)
        return False
    memory = MemoryService(settings)
    written = memory.append_peer_chat_reply(
        thread_id=light.thread_id,
        assistant_text=PRESENCE_KNOCK_CHAT_LINE,
    )
    if written:
        logger.info("Presence knock chat line written agent=%s", agent_id)
    return written
