"""Event subscription keys and event-type mapping (Proposal 3 phase 2)."""

from __future__ import annotations

from light_house.events.subscription_keys import DEFAULT_EVENT_SUBSCRIPTIONS
from light_house.events.types import EventType
from light_house.personal.store import get_personal_store

# Persisted keys in each agent's personal DB.
SUBSCRIPTION_POST_CHAT = "post_chat"
SUBSCRIPTION_SCHEDULED_RUMINATION = "scheduled_rumination"
SUBSCRIPTION_MEMORY_MAINTENANCE = "memory_maintenance"
SUBSCRIPTION_PEER_MESSAGE = "peer_message"
SUBSCRIPTION_SHARED_NOTE = "shared_note"

EVENT_TYPE_SUBSCRIPTION: dict[EventType, str] = {
    EventType.CHAT_RECEIVED: SUBSCRIPTION_POST_CHAT,
    EventType.SCHEDULED_WAKE: SUBSCRIPTION_SCHEDULED_RUMINATION,
    EventType.MAINTENANCE_WAKE: SUBSCRIPTION_MEMORY_MAINTENANCE,
    EventType.PEER_MESSAGE: SUBSCRIPTION_PEER_MESSAGE,
    EventType.SHARED_NOTE_SAVED: SUBSCRIPTION_SHARED_NOTE,
}


def subscription_key_for_event(event_type: EventType) -> str | None:
    return EVENT_TYPE_SUBSCRIPTION.get(event_type)


def agent_accepts_event(settings: Settings, agent_id: str, event_type: EventType) -> bool:
    """Return False when subscriptions are enabled and the agent opted out."""
    if not settings.event_subscriptions_enabled:
        return True
    if not settings.personal_db_enabled:
        return True
    key = subscription_key_for_event(event_type)
    if key is None:
        return True
    try:
        store = get_personal_store(settings, agent_id)
        return store.is_event_subscribed(key)
    except Exception:
        return True


def format_event_subscriptions_context(settings: Settings, agent_id: str) -> str:
    """Read-only markdown block for rumination/chat context."""
    if not settings.event_subscriptions_enabled or not settings.personal_db_enabled:
        return ""
    try:
        store = get_personal_store(settings, agent_id)
        return store.format_event_subscriptions_section()
    except Exception:
        return ""
