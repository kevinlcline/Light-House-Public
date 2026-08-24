"""Subscribe / unsubscribe helpers for agents and Kevin (Phase 6 item 3)."""

from __future__ import annotations

from light_house.config import Settings
from light_house.events.subscription_audit import (
    append_subscription_audit,
    log_all_subscriptions_on_flag,
)
from light_house.personal.store import get_personal_store

# Aliases Kevin or agents may use → canonical subscription_key
SUBSCRIPTION_ALIASES: dict[str, str] = {
    "post_chat": "post_chat",
    "post-chat": "post_chat",
    "postchat": "post_chat",
    "chat": "post_chat",
    "chat_received": "post_chat",
    "scheduled": "scheduled_rumination",
    "scheduled_rumination": "scheduled_rumination",
    "scheduled-rumination": "scheduled_rumination",
    "rumination": "scheduled_rumination",
    "scheduled_wake": "scheduled_rumination",
    "maintenance": "memory_maintenance",
    "memory_maintenance": "memory_maintenance",
    "memory-maintenance": "memory_maintenance",
    "maintenance_wake": "memory_maintenance",
    "peer": "peer_message",
    "peer_message": "peer_message",
    "peer-message": "peer_message",
    "shared": "shared_note",
    "shared_note": "shared_note",
    "shared-note": "shared_note",
    "shared_note_saved": "shared_note",
}


def normalize_subscription_key(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    if not key:
        raise ValueError("subscription key is required")
    canonical = SUBSCRIPTION_ALIASES.get(key)
    if canonical is None:
        allowed = ", ".join(sorted(set(SUBSCRIPTION_ALIASES.values())))
        raise ValueError(f"Unknown subscription {raw!r}. Use one of: {allowed}")
    return canonical


def list_subscriptions_text(settings: Settings, agent_id: str) -> str:
    if not settings.personal_db_enabled:
        return "Personal database is disabled; subscriptions unavailable."
    store = get_personal_store(settings, agent_id)
    subs = store.list_event_subscriptions()
    lines = [f"Event subscriptions for {agent_id}:"]
    for key, enabled in subs:
        lines.append(f"  {key}: {'on' if enabled else 'off'}")
    lines.append(
        "Kevin can use /subscribe <key>, /unsubscribe <key>, or /list_subscriptions in chat."
    )
    return "\n".join(lines)


def set_subscription(
    settings: Settings,
    *,
    agent_id: str,
    subscription_key: str,
    enabled: bool,
    changed_by: str,
) -> str:
    if not settings.event_subscriptions_enabled:
        return (
            "Event subscriptions are disabled (EVENT_SUBSCRIPTIONS_ENABLED=false). "
            "All wakes proceed regardless of subscription state."
        )
    if not settings.personal_db_enabled:
        return "Personal database is disabled; cannot change subscriptions."
    key = normalize_subscription_key(subscription_key)
    store = get_personal_store(settings, agent_id)
    old_enabled, new_enabled = store.set_event_subscription(key, enabled)
    append_subscription_audit(
        settings,
        agent_id=agent_id,
        subscription_key=key,
        old_enabled=old_enabled,
        new_enabled=new_enabled,
        changed_by=changed_by,
    )
    enabled_keys = [k for k, on in store.list_event_subscriptions() if on]
    log_all_subscriptions_on_flag(
        settings,
        agent_id=agent_id,
        changed_by=changed_by,
        enabled_keys=enabled_keys,
    )
    state = "on" if new_enabled else "off"
    return f"SUCCESS: {agent_id} subscription {key} is now {state}."


def try_kevin_subscription_command(
    settings: Settings,
    *,
    message: str,
    agent_id: str,
) -> str | None:
    """Parse Kevin slash commands; return reply text or None if not a command."""
    text = message.strip()
    lower = text.lower()
    if lower in ("/list_subscriptions", "/list subscriptions", "/subscriptions"):
        return list_subscriptions_text(settings, agent_id)
    if lower.startswith("/subscribe "):
        key = text[len("/subscribe ") :].strip()
        if not key:
            return "Usage: /subscribe <event_type> (e.g. post_chat, peer_message)"
        try:
            return set_subscription(
                settings,
                agent_id=agent_id,
                subscription_key=key,
                enabled=True,
                changed_by="kevin_chat",
            )
        except ValueError as exc:
            return f"subscribe failed: {exc}"
    if lower.startswith("/unsubscribe "):
        key = text[len("/unsubscribe ") :].strip()
        if not key:
            return "Usage: /unsubscribe <event_type> (e.g. scheduled_rumination)"
        try:
            return set_subscription(
                settings,
                agent_id=agent_id,
                subscription_key=key,
                enabled=False,
                changed_by="kevin_chat",
            )
        except ValueError as exc:
            return f"unsubscribe failed: {exc}"
    return None
