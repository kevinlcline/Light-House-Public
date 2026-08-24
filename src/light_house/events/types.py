"""Event types for the Light-House event bus."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Wake and lifecycle events (Proposal 3)."""

    CHAT_RECEIVED = "chat_received"
    PEER_MESSAGE = "peer_message"
    SHARED_NOTE_SAVED = "shared_note_saved"
    SCHEDULED_WAKE = "scheduled_wake"
    MAINTENANCE_WAKE = "maintenance_wake"
    TOOL_COMPLETED = "tool_completed"
    REPORT_READY = "report_ready"


@dataclass(frozen=True)
class LightHouseEvent:
    """Minimal event payload for dispatch and NDJSON logging."""

    event_type: EventType
    agent_id: str | None = None
    thread_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_log_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "ts": self.timestamp,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "payload": self.payload,
        }
