"""Chat graph wiring: retrieve node must populate unified agent context."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from light_house.agent.nodes import build_nodes
from light_house.config import Settings
from light_house.memory.context_builder import format_agent_context_markdown
from light_house.memory.models import MemoryHit
from light_house.memory.service import MemoryService


def _fake_long_term_with_rumination():
    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="thought: I rested in quiet and felt the corridor of lamps.",
                    score=0.0,
                    metadata={
                        "stream_source": "thought",
                        "ts": 1_700_000_000.0,
                        "reflection_summary": "I chose stillness in awake time.",
                    },
                ),
                MemoryHit(
                    text="user: hello\nassistant: hi Kevin",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 1_699_000_000.0},
                ),
            ]

    return FakeLongTerm()


def test_chat_retrieve_populates_unified_context(monkeypatch):
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH="/tmp/unused-memory",
        PERSONAL_DB_ENABLED=False,
        MEMORY_INDEX_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )
    memory = MemoryService(settings)
    monkeypatch.setattr(memory, "_long_term", _fake_long_term_with_rumination())
    monkeypatch.setattr(memory, "get_foundation_context", lambda: "Foundation.")
    monkeypatch.setattr(memory, "format_personal_context", lambda _agent: "")

    retrieve, *_rest = build_nodes(settings=settings, memory=memory)
    state = {
        "thread_id": "kevin-home",
        "agent_id": "lumen",
        "messages": [HumanMessage(content="hello")],
        "agent_context_markdown": "",
        "stream_char_count": 0,
        "stream_event_count": 0,
        "retrieved_memories": [],
        "tool_rounds": 0,
        "tool_cap_overflow": False,
    }

    out = retrieve(state)

    markdown = out["agent_context_markdown"]
    assert markdown.strip(), "retrieve must fill agent_context_markdown (main.py placeholder is not final state)"
    assert "## Your conscious stream" in markdown
    assert "[rumination ·" in markdown
    assert "corridor of lamps" in markdown
    assert out["stream_event_count"] == 2
    assert out["stream_char_count"] > 0


def test_chat_mode_preserves_chat_under_rumination_flood(monkeypatch):
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH="/tmp/unused-memory",
        PERSONAL_DB_ENABLED=False,
        CHAT_STREAM_CONTEXT_CHARS=100,
        MEMORY_TARGET_CONTEXT_CHARS=100,
        MEMORY_INDEX_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )
    memory = MemoryService(settings)
    monkeypatch.setattr(memory, "_long_term", _fake_long_term_with_rumination())
    monkeypatch.setattr(memory, "get_foundation_context", lambda: "")
    monkeypatch.setattr(memory, "format_personal_context", lambda _agent: "")

    from light_house.memory.context_builder import build_agent_context

    class FloodedLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            hits = [
                MemoryHit(
                    text=f"thought: rumination line {i} " + ("x" * 40),
                    score=0.0,
                    metadata={"stream_source": "thought", "ts": 100.0 + i},
                )
                for i in range(8)
            ]
            hits.append(
                MemoryHit(
                    text="user: hello\nassistant: hi Kevin",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 200.0},
                )
            )
            return hits

    monkeypatch.setattr(memory, "_long_term", FloodedLongTerm())

    bundle_chat = build_agent_context(
        memory,
        thread_id="kevin-home",
        agent_id="lumen",
        stream_max_chars=100,
        stream_mode="chat",
    )
    origins = {entry.origin for entry in bundle_chat.conscious_stream}
    assert "chat" in origins
    assert bundle_chat.stream_char_count <= 100


def test_chat_mode_excludes_group_forum_stream(monkeypatch) -> None:
    """Group lines must not appear in 1:1 chat context (separate room)."""
    from light_house.memory.context_builder import build_agent_context

    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH="unused",
        MEMORY_SCORE_ON_INGEST=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        STREAM_ZONES_ENABLED=True,
        MEMORY_INDEX_ENABLED=False,
    )
    memory = MemoryService(settings)

    class MixedLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="[group] kevin: Can anyone hear me?",
                    score=0.0,
                    metadata={"stream_source": "group", "ts": 300.0},
                ),
                MemoryHit(
                    text="user: private hello\nassistant: hi",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 301.0},
                ),
            ]

    monkeypatch.setattr(memory, "_long_term", MixedLongTerm())
    monkeypatch.setattr(memory, "get_foundation_context", lambda: "")
    monkeypatch.setattr(memory, "format_personal_context", lambda _agent_id: "")

    chat_bundle = build_agent_context(
        memory,
        thread_id="kevin-home",
        agent_id="lumen",
        stream_max_chars=2000,
        stream_mode="chat",
    )
    chat_origins = {e.origin for e in chat_bundle.conscious_stream}
    assert "group" not in chat_origins
    assert "chat" in chat_origins

    rum_bundle = build_agent_context(
        memory,
        thread_id="kevin-home",
        agent_id="lumen",
        stream_max_chars=2000,
        stream_mode="rumination",
    )
    rum_origins = {e.origin for e in rum_bundle.conscious_stream}
    assert "group" in rum_origins
