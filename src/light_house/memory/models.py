"""Typed shapes for memory records (keep small; expand as the agent grows)."""

from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved memory snippet plus lightweight provenance."""

    text: str
    score: float | None
    metadata: dict[str, Any]
    doc_id: str | None = None


@dataclass(frozen=True)
class HistoryMessage:
    """One chat message for merge / gap-fill (API-agnostic)."""

    role: Role
    content: str
