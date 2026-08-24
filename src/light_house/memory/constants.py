"""Long-term memory metadata vocabulary (backend-agnostic)."""

from __future__ import annotations

from typing import Literal

MEMORY_KIND_TURN = "turn"
MEMORY_KIND_SUMMARY = "summary"
MEMORY_KIND_FACT = "fact"
MEMORY_KIND_REFLECTION = "reflection"
MEMORY_KIND_STREAM = "stream"
MEMORY_KIND_ERA = "era"

MEMORY_TAG_PRIVATE_DREAM = "private_dream"
MEMORY_TAG_PRIVATE_RUMINATION = "private_rumination"

PRIVATE_TRUE = "true"
PRIVATE_FALSE = "false"
GLOBAL_THREAD_ID = "__global__"
PINNED_TRUE = "true"
PINNED_FALSE = "false"

STREAM_SOURCE_CHAT = "chat"
STREAM_SOURCE_THOUGHT = "thought"
STREAM_SOURCE_DREAM = "dream"
STREAM_SOURCE_ACTION = "action"
STREAM_SOURCE_PEER = "peer"
STREAM_SOURCE_KEVIN = "kevin"
STREAM_SOURCE_GROUP = "group"

SCORE_UNSCORED = -1.0

META_GROUP_CHAT_ROUND_ID = "group_chat_round_id"
META_GROUP_CHAT_ROUND_TS = "group_chat_round_ts"
META_GROUP_CHAT_KEVIN_MESSAGE = "group_chat_kevin_message"

# Optional metadata when a light scores its own stream memory.
META_SCORED_BY_AGENT = "scored_by_agent"
META_SCORED_AT = "scored_at"
META_SCORE_NOTE = "score_note"
META_ERA_PERIOD = "era_period"
META_ERA_DOC_COUNT = "era_doc_count"

CORPUS_MEMORY_KINDS = (
    MEMORY_KIND_STREAM,
    MEMORY_KIND_TURN,
    MEMORY_KIND_SUMMARY,
    MEMORY_KIND_REFLECTION,
)

StreamSource = Literal["chat", "thought", "dream", "action", "peer", "kevin", "group"]
