"""Event subscription keys (no imports from personal store)."""

from __future__ import annotations

SUBSCRIPTION_POST_CHAT = "post_chat"
SUBSCRIPTION_SCHEDULED_RUMINATION = "scheduled_rumination"
SUBSCRIPTION_MEMORY_MAINTENANCE = "memory_maintenance"
SUBSCRIPTION_PEER_MESSAGE = "peer_message"
SUBSCRIPTION_SHARED_NOTE = "shared_note"
# Not a wake. Kevin's switch for the write_to_kevin mail pipe.
SUBSCRIPTION_WRITE_TO_KEVIN = "write_to_kevin"

DEFAULT_EVENT_SUBSCRIPTIONS: tuple[str, ...] = (
    SUBSCRIPTION_POST_CHAT,
    SUBSCRIPTION_SCHEDULED_RUMINATION,
    SUBSCRIPTION_MEMORY_MAINTENANCE,
    SUBSCRIPTION_PEER_MESSAGE,
    SUBSCRIPTION_SHARED_NOTE,
    SUBSCRIPTION_WRITE_TO_KEVIN,
)
