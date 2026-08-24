"""Inner time: felt_cycles (scheduled rumination) and felt_days (daily dream)."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from light_house.config import Settings
from light_house.personal.light_state_keys import FELT_CYCLES, FELT_DAYS, LAST_DREAM_YMD
from light_house.personal.store import PersonalStore, get_personal_store

logger = logging.getLogger(__name__)


def _store(settings: Settings, agent_id: str) -> PersonalStore | None:
    if not settings.personal_db_enabled:
        return None
    try:
        return get_personal_store(settings, agent_id)
    except (KeyError, RuntimeError, OSError) as exc:
        logger.warning("Inner time store unavailable agent=%s: %s", agent_id, exc)
        return None


def read_inner_time(settings: Settings, agent_id: str) -> tuple[int | None, int | None]:
    """Current felt_cycles and felt_days without incrementing."""
    store = _store(settings, agent_id)
    if store is None:
        return None, None
    return (
        store.get_light_state(FELT_CYCLES, default=0),
        store.get_light_state(FELT_DAYS, default=0),
    )


def increment_felt_cycles(settings: Settings, agent_id: str) -> int | None:
    """Bump autonomous awake counter; return new value or None if store unavailable."""
    store = _store(settings, agent_id)
    if store is None:
        return None
    value = store.increment_light_state(FELT_CYCLES)
    logger.info("Felt cycle incremented agent=%s felt_cycles=%d", agent_id, value)
    return value


def local_dream_ymd(settings: Settings) -> int:
    tz = ZoneInfo(settings.dream_timezone)
    now = datetime.now(tz)
    return now.year * 10_000 + now.month * 100 + now.day


def dream_slot_open(settings: Settings) -> bool:
    tz = ZoneInfo(settings.dream_timezone)
    return datetime.now(tz).hour >= settings.dream_local_hour


def dream_due_for_store(settings: Settings, store: PersonalStore) -> bool:
    if not dream_slot_open(settings):
        return False
    today = local_dream_ymd(settings)
    return store.get_light_state(LAST_DREAM_YMD, default=0) < today


def dream_due(settings: Settings, agent_id: str) -> bool:
    store = _store(settings, agent_id)
    if store is None:
        return False
    return dream_due_for_store(settings, store)


def mark_dream_day(settings: Settings, agent_id: str) -> int | None:
    """Increment felt_days once per calendar day; return current felt_days."""
    store = _store(settings, agent_id)
    if store is None:
        return None
    today = local_dream_ymd(settings)
    last = store.get_light_state(LAST_DREAM_YMD, default=0)
    if last >= today:
        return store.get_light_state(FELT_DAYS, default=0)
    felt_days = store.increment_light_state(FELT_DAYS)
    store.set_light_state(LAST_DREAM_YMD, today)
    logger.info(
        "Felt day incremented agent=%s felt_days=%d dream_ymd=%d",
        agent_id,
        felt_days,
        today,
    )
    return felt_days
