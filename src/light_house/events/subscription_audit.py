"""Append-only audit log for event subscription changes (Phase 6 item 3)."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from light_house.config import Settings
from light_house.events.subscription_keys import DEFAULT_EVENT_SUBSCRIPTIONS

logger = logging.getLogger(__name__)

_AUDIT_LOCK = threading.Lock()


def _audit_path(settings: Settings) -> Path:
    return settings.subscription_audit_log_path.resolve()


def append_subscription_audit(
    settings: Settings,
    *,
    agent_id: str,
    subscription_key: str,
    old_enabled: bool,
    new_enabled: bool,
    changed_by: str,
) -> None:
    if not settings.subscription_audit_enabled:
        return
    path = _audit_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "agent_id": agent_id,
        "subscription_key": subscription_key,
        "old_enabled": old_enabled,
        "new_enabled": new_enabled,
        "changed_by": changed_by,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _AUDIT_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    logger.info(
        "Subscription change agent=%s key=%s %s→%s by=%s",
        agent_id,
        subscription_key,
        old_enabled,
        new_enabled,
        changed_by,
    )


def log_all_subscriptions_on_flag(
    settings: Settings,
    *,
    agent_id: str,
    changed_by: str,
    enabled_keys: list[str],
) -> None:
    """Flag when every wake type is subscribed (audit only, never blocks)."""
    if not settings.subscription_audit_enabled:
        return
    if set(enabled_keys) != set(DEFAULT_EVENT_SUBSCRIPTIONS):
        return
    path = _audit_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "agent_id": agent_id,
        "event": "all_subscriptions_on",
        "changed_by": changed_by,
        "note": "All wake subscriptions are on — review if unintended.",
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _AUDIT_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
