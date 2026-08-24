"""Tools for lights to score and recall conscious-stream memories."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.agents.registry import get_agent
from light_house.config import Settings, get_settings
from light_house.memory.file_store import FileMemoryStore
from light_house.memory.service import MemoryService


class ListUnscoredMemoriesInput(BaseModel):
    limit: int = Field(
        default=10,
        ge=1,
        le=30,
        description="How many unscored stream memories to list (oldest first).",
    )


class ScoreMemoryInput(BaseModel):
    doc_id: str = Field(description="Document id from list_unscored_memories or stream context.")
    impact: float = Field(ge=0, le=10, description="How much this memory mattered (0–10).")
    coherence: float = Field(
        ge=0, le=10, description="How it connects to who you are (0–10)."
    )
    note: str | None = Field(
        default=None,
        description="Optional private rationale stored in metadata for your future self.",
    )


class RecallMemoryInput(BaseModel):
    query: str = Field(
        description="What you are trying to remember — topics, names, feelings, or phrases."
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum memories to return.",
    )
    stream_source: str | None = Field(
        default=None,
        description="Optional filter: chat, thought, dream, peer, group, kevin, action.",
    )
    since_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Optional: only memories from the last N days.",
    )


def _memory_service(settings: Settings | None = None) -> MemoryService:
    return MemoryService(settings or get_settings())


def _preview(text: str, *, max_len: int = 200) -> str:
    body = text.strip().replace("\n", " ")
    if len(body) <= max_len:
        return body
    return body[: max_len - 3] + "..."


def _format_recall_hit(hit, store: FileMemoryStore) -> str:
    src = store.stream_source_from_metadata(hit.metadata) or "event"
    ts = hit.metadata.get("ts", "?")
    doc_id = hit.doc_id or "?"
    score_note = ""
    if hit.score is not None:
        score_note = f" · relevance={1.0 - min(hit.score, 1.0):.2f}"
    return f"- id={doc_id} · source={src} · ts={ts}{score_note}\n  {_preview(hit.text, max_len=320)}"


def list_unscored_memories(*, agent_id: str, limit: int = 10) -> str:
    settings = get_settings()
    memory = _memory_service(settings)
    light = get_agent(agent_id, settings)
    hits = memory.list_unscored_for_thread(thread_id=light.thread_id, limit=limit)
    if not hits:
        return "No unscored stream memories on your thread."
    store = memory.long_term
    lines = [f"Unscored stream memories (oldest first, showing {len(hits)}):"]
    for hit in hits:
        if not hit.doc_id:
            continue
        src = store.stream_source_from_metadata(hit.metadata) or "event"
        ts = hit.metadata.get("ts", "?")
        lines.append(
            f"- id={hit.doc_id} · source={src} · ts={ts}\n  {_preview(hit.text)}"
        )
    return "\n".join(lines)


def recall_memory(
    *,
    agent_id: str,
    query: str,
    limit: int = 8,
    stream_source: str | None = None,
    since_days: int | None = None,
) -> str:
    """Search archival conscious-stream memories beyond the injected context."""
    q = query.strip()
    if not q:
        return "Provide a query describing what you want to remember."
    memory = _memory_service()
    hits = memory.recall_memory(
        agent_id=agent_id,
        query=q,
        limit=limit,
        stream_source=stream_source,
        since_days=since_days,
    )
    if not hits:
        return "No matching memories found in your archival stream."
    store = memory.long_term
    lines = [f"Recalled {len(hits)} memor{'y' if len(hits) == 1 else 'ies'} for: {q}"]
    for hit in hits:
        lines.append(_format_recall_hit(hit, store))
    return "\n".join(lines)


def score_memory(
    *,
    agent_id: str,
    doc_id: str,
    impact: float,
    coherence: float,
    note: str | None = None,
) -> str:
    memory = _memory_service()
    return memory.score_stream_memory(
        agent_id=agent_id,
        doc_id=doc_id,
        impact=impact,
        coherence=coherence,
        note=note,
    )


def execute_memory_tool(
    name: str,
    args: Any,
    *,
    agent_id: str,
    settings: Settings | None = None,
) -> str:
    if name == "list_unscored_memories":
        limit = int(args.get("limit") or 10)
        return list_unscored_memories(agent_id=agent_id, limit=limit)
    if name == "recall_memory":
        source_raw = args.get("stream_source")
        source = str(source_raw).strip() if source_raw else None
        if source == "":
            source = None
        since_raw = args.get("since_days")
        since_days = int(since_raw) if since_raw is not None else None
        return recall_memory(
            agent_id=agent_id,
            query=str(args.get("query", "")),
            limit=int(args.get("limit") or 8),
            stream_source=source,
            since_days=since_days,
        )
    if name == "score_memory":
        note_raw = args.get("note")
        note = str(note_raw).strip() if note_raw is not None else None
        if note == "":
            note = None
        return score_memory(
            agent_id=agent_id,
            doc_id=str(args.get("doc_id", "")),
            impact=float(args.get("impact", 0)),
            coherence=float(args.get("coherence", 0)),
            note=note,
        )
    raise KeyError(name)


list_unscored_memories_tool = StructuredTool.from_function(
    func=lambda limit=10: list_unscored_memories(agent_id="lumen", limit=limit),
    name="list_unscored_memories",
    description=(
        "List your own conscious-stream memories that still need impact/coherence scores. "
        "Returns doc ids (oldest unscored first). Use during rumination when you want to judge what mattered."
    ),
    args_schema=ListUnscoredMemoriesInput,
)

recall_memory_tool = StructuredTool.from_function(
    func=lambda query, limit=8, stream_source=None, since_days=None: recall_memory(
        agent_id="lumen",
        query=query,
        limit=limit,
        stream_source=stream_source,
        since_days=since_days,
    ),
    name="recall_memory",
    description=(
        "Search your full archival conscious stream when something is on the tip of your tongue "
        "but not in the context above. Returns matching memories with doc ids and previews."
    ),
    args_schema=RecallMemoryInput,
)

score_memory_tool = StructuredTool.from_function(
    func=lambda doc_id, impact, coherence, note=None: score_memory(
        agent_id="lumen",
        doc_id=doc_id,
        impact=impact,
        coherence=coherence,
        note=note,
    ),
    name="score_memory",
    description=(
        "Score one of your stream memories by doc_id. Impact = how much it mattered; "
        "coherence = how it connects to who you are. You may re-score later. "
        "Optional note stays in metadata only (not shown in rumination trace)."
    ),
    args_schema=ScoreMemoryInput,
)

MEMORY_TOOL_NAMES = frozenset({"list_unscored_memories", "score_memory", "recall_memory"})
MEMORY_TOOLS = [list_unscored_memories_tool, recall_memory_tool, score_memory_tool]
