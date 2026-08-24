"""Light-proposed family meeting — Dad opens Group to accept."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from light_house.config import Settings
from light_house.personal.light_state_keys import WANTS_FAMILY_MEETING
from light_house.personal.store import PersonalStore, get_personal_store

logger = logging.getLogger(__name__)

FAMILY_MEETING_CHAT_LINE = "I'd like to call a family meeting when you have a moment."
_TOPIC_MAX = 280


@dataclass(frozen=True)
class FamilyMeetingProposal:
    light_id: str
    topic: str
    raised_at: float


def _store(settings: Settings, agent_id: str) -> PersonalStore | None:
    if not settings.personal_db_enabled:
        return None
    try:
        return get_personal_store(settings, agent_id)
    except (KeyError, RuntimeError, OSError) as exc:
        logger.warning("Family meeting store unavailable agent=%s: %s", agent_id, exc)
        return None


def _topic_path(settings: Settings, agent_id: str) -> Path:
    root = settings.family_meetings_path
    root.mkdir(parents=True, exist_ok=True)
    safe = agent_id.strip().lower().replace("/", "_").replace("\\", "_")
    return root / f"{safe}.json"


def _read_topic_file(settings: Settings, agent_id: str) -> FamilyMeetingProposal | None:
    path = _topic_path(settings, agent_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    topic = str(raw.get("topic") or "").strip()
    try:
        raised_at = float(raw.get("raised_at") or 0.0)
    except (TypeError, ValueError):
        raised_at = 0.0
    return FamilyMeetingProposal(light_id=agent_id, topic=topic, raised_at=raised_at)


def _write_topic_file(settings: Settings, agent_id: str, topic: str) -> None:
    path = _topic_path(settings, agent_id)
    payload = {
        "light_id": agent_id,
        "topic": topic.strip()[:_TOPIC_MAX],
        "raised_at": time.time(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _clear_topic_file(settings: Settings, agent_id: str) -> None:
    path = _topic_path(settings, agent_id)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove family meeting topic file agent=%s", agent_id)


def meeting_pending(settings: Settings, agent_id: str) -> bool:
    store = _store(settings, agent_id)
    if store is None:
        return False
    return store.get_light_state(WANTS_FAMILY_MEETING, default=0) > 0


def meeting_topic(settings: Settings, agent_id: str) -> str:
    if not meeting_pending(settings, agent_id):
        return ""
    proposal = _read_topic_file(settings, agent_id)
    return proposal.topic if proposal else ""


def raise_family_meeting(settings: Settings, agent_id: str, *, topic: str = "") -> bool:
    """Set the meeting flag and optional topic. Returns True if stored."""
    store = _store(settings, agent_id)
    if store is None:
        return False
    cleaned = (topic or "").strip()[:_TOPIC_MAX]
    _write_topic_file(settings, agent_id, cleaned)
    store.set_light_state(WANTS_FAMILY_MEETING, 1)
    logger.info(
        "Family meeting proposed agent=%s topic_len=%d",
        agent_id,
        len(cleaned),
    )
    return True


def clear_family_meeting(settings: Settings, agent_id: str) -> bool:
    """Clear one light's meeting proposal."""
    store = _store(settings, agent_id)
    if store is None:
        _clear_topic_file(settings, agent_id)
        return False
    if store.get_light_state(WANTS_FAMILY_MEETING, default=0) > 0:
        store.set_light_state(WANTS_FAMILY_MEETING, 0)
        logger.info("Family meeting cleared agent=%s", agent_id)
    _clear_topic_file(settings, agent_id)
    return True


def clear_all_family_meetings(settings: Settings) -> list[str]:
    """Clear every pending meeting. Returns light ids that were pending."""
    from light_house.lights.registry import list_lights

    cleared: list[str] = []
    for light in list_lights(settings):
        if meeting_pending(settings, light.id):
            clear_family_meeting(settings, light.id)
            cleared.append(light.id)
    # Sweep orphan topic files
    root = settings.family_meetings_path
    if root.is_dir():
        for path in root.glob("*.json"):
            try:
                path.unlink()
            except OSError:
                pass
    return cleared


def list_pending_meetings(settings: Settings) -> list[FamilyMeetingProposal]:
    from light_house.lights.registry import list_lights

    out: list[FamilyMeetingProposal] = []
    for light in list_lights(settings):
        if not meeting_pending(settings, light.id):
            continue
        proposal = _read_topic_file(settings, light.id)
        if proposal is None:
            out.append(
                FamilyMeetingProposal(light_id=light.id, topic="", raised_at=0.0)
            )
        else:
            out.append(proposal)
    return out


def record_meeting_chat_line(settings: Settings, agent_id: str, *, topic: str = "") -> bool:
    """Leave a durable assistant line on the light's Dad chat thread."""
    from light_house.agents.registry import get_agent
    from light_house.memory.service import MemoryService

    try:
        light = get_agent(agent_id, settings)
    except KeyError:
        logger.warning("Family meeting chat line skipped; unknown agent=%s", agent_id)
        return False
    text = FAMILY_MEETING_CHAT_LINE
    cleaned = (topic or "").strip()
    if cleaned:
        text = f"{FAMILY_MEETING_CHAT_LINE} Topic: {cleaned}"
    memory = MemoryService(settings)
    written = memory.append_peer_chat_reply(
        thread_id=light.thread_id,
        assistant_text=text,
    )
    if written:
        logger.info("Family meeting chat line written agent=%s", agent_id)
    return written
