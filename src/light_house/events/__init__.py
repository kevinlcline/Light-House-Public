"""Lightweight event bus for agent wakes (Proposal 3 phase 1)."""

from __future__ import annotations

from typing import Any

from light_house.events.types import EventType, LightHouseEvent

__all__ = [
    "EventType",
    "LightHouseEvent",
    "publish",
    "read_event_log_tail",
    "register_event_bus",
    "start_event_bus",
]


def __getattr__(name: str) -> Any:
    # Lazy bus imports avoid circular import with personal.store ↔ subscriptions.
    if name in {"publish", "read_event_log_tail", "register_event_bus", "start_event_bus"}:
        from light_house.events import bus

        return getattr(bus, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
