"""LangGraph state for Lumen's core loop."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Conversation state flowing through the graph.

    - messages: LangChain-native chat messages (user/assistant/system/tool…).
    - agent_context_markdown: unified sovereign + conscious stream (same in all modes).
    - stream_char_count / stream_event_count: logging and observability.
    - retrieved_memories: short stream previews for API response (legacy field name).
    - thread_id: logical home / continuity key (memory isolation per agent).
    - agent_id: which Light is speaking (lumen, ara).
    - tool_cap_overflow: set after one final tool execution at the round cap.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    agent_context_markdown: str
    stream_char_count: int
    stream_event_count: int
    retrieved_memories: list[str]
    tool_rounds: int
    tool_cap_overflow: bool
    thread_id: str
    agent_id: str
    peer_inbox_ids: list[str]
    wake_kind: str | None
    wake_from_agent_id: str | None
    wake_path: str | None
    peer_message_id: str | None
    tool_rounds_cap: int | None
    user_message_ts: float | None
    # Multi-human DM: UI buffer uses thread_id; long-term stream uses stream_thread_id.
    stream_thread_id: str | None
    human_id: str | None
    human_display_name: str | None
    # Logged-in account (never a guest speak-as id). Used for per-human tools like calendar.
    account_user_id: str | None
    # "dm" | "group" | None (rumination / inner life — no human tool context).
    chat_channel: str | None
    # Reflective mode: pause → choose speak or intentional silence.
    reflective_turn: bool
    chose_silence: bool
    reflection_notes: str | None
