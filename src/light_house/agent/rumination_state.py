"""LangGraph state for background rumination loop."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RuminationState(TypedDict):
    thread_id: str
    agent_id: str
    agent_context_markdown: str
    stream_char_count: int
    stream_event_count: int
    generated_text: str
    summary_text: str
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    tool_cap_overflow: bool
    peer_inbox_ids: list[str]
    wake_kind: str | None
    wake_path: str | None
    tool_rounds_cap: int | None
    internal_step: int
    internal_steps_cap: int | None
    internal_halt: bool
    internal_chars_used: int
    felt_cycles: int | None
