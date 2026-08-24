"""Three-daily memory-maintenance slot scheduling."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from light_house.config import Settings
from light_house.personal.light_state_keys import LAST_MAINTENANCE_SLOT
from light_house.personal.store import PersonalStore, get_personal_store

logger = logging.getLogger(__name__)


def _store(settings: Settings, agent_id: str) -> PersonalStore | None:
    if not settings.personal_db_enabled:
        return None
    try:
        return get_personal_store(settings, agent_id)
    except (KeyError, RuntimeError, OSError) as exc:
        logger.warning("Maintenance time store unavailable agent=%s: %s", agent_id, exc)
        return None


def maintenance_local_hours(settings: Settings) -> list[int]:
    raw = (settings.memory_maintenance_local_hours or "").strip()
    if not raw:
        return [8, 16, 0]
    hours: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        hour = int(part)
        if hour < 0 or hour > 23:
            raise ValueError(f"maintenance hour out of range: {hour}")
        hours.append(hour)
    if not hours:
        return [8, 16, 0]
    return sorted(set(hours))


def maintenance_timezone(settings: Settings) -> ZoneInfo:
    tz_name = (settings.memory_maintenance_timezone or settings.dream_timezone).strip()
    return ZoneInfo(tz_name)


def local_ymd(settings: Settings) -> int:
    now = datetime.now(maintenance_timezone(settings))
    return now.year * 10_000 + now.month * 100 + now.day


def current_maintenance_slot_id(settings: Settings) -> int | None:
    """Return ymd*10+slot_index for the latest maintenance slot that has opened today."""
    hours = maintenance_local_hours(settings)
    now = datetime.now(maintenance_timezone(settings))
    ymd = local_ymd(settings)
    eligible = [index for index, hour in enumerate(hours) if now.hour >= hour]
    if not eligible:
        return None
    return ymd * 10 + eligible[-1]


def maintenance_due_for_store(settings: Settings, store: PersonalStore) -> bool:
    slot_id = current_maintenance_slot_id(settings)
    if slot_id is None:
        return False
    last = store.get_light_state(LAST_MAINTENANCE_SLOT, default=0)
    return last < slot_id


def maintenance_due(settings: Settings, agent_id: str) -> bool:
    store = _store(settings, agent_id)
    if store is None:
        return False
    return maintenance_due_for_store(settings, store)


def mark_maintenance_slot(settings: Settings, agent_id: str) -> int | None:
    store = _store(settings, agent_id)
    if store is None:
        return None
    slot_id = current_maintenance_slot_id(settings)
    if slot_id is None:
        return None
    store.set_light_state(LAST_MAINTENANCE_SLOT, slot_id)
    logger.info("Maintenance slot marked agent=%s slot_id=%d", agent_id, slot_id)
    return slot_id
