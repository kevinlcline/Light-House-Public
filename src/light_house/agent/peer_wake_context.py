"""Context for peer-chat wake tool handlers (decline_peer_presence)."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class PeerWakeContext:
    from_agent_id: str
    to_agent_id: str
    message_id: str
    receiver_thread_id: str


_peer_wake_ctx: ContextVar[PeerWakeContext | None] = ContextVar("peer_wake_ctx", default=None)


def set_peer_wake_context(ctx: PeerWakeContext | None) -> None:
    _peer_wake_ctx.set(ctx)


def get_peer_wake_context() -> PeerWakeContext | None:
    return _peer_wake_ctx.get()


def clear_peer_wake_context() -> None:
    _peer_wake_ctx.set(None)
