"""Unified context builder tests."""

from __future__ import annotations

from light_house.config import Settings
from light_house.memory.context_builder import (
    AgentContextBundle,
    StreamEntry,
    _cap_stream_entries_zoned,
    build_agent_context,
    format_agent_context_markdown,
    format_chat_thread_markdown,
    format_kevin_messages_timeline,
    format_stream_entry,
    stream_zone_for_origin,
    stream_zone_limits,
)
from light_house.memory.models import MemoryHit
from light_house.memory.service import MemoryService
from light_house.memory.short_term import BufferedMessage


def test_format_stream_entry_includes_origin_tag():
    entry = StreamEntry(origin="rumination", text="I rested in quiet.", ts=1_700_000_000.0)
    formatted = format_stream_entry(entry, timezone_name="UTC")
    assert formatted.startswith("[rumination ·")
    assert "I rested in quiet." in formatted


def test_format_agent_context_stream_is_chronological():
    bundle = AgentContextBundle(
        foundation="",
        personal_knowledge="",
        event_subscriptions="",
        pinned_facts=[],
        conscious_stream=[
            StreamEntry(origin="chat", text="newer", ts=200.0),
            StreamEntry(origin="chat", text="older", ts=100.0),
        ],
        stream_char_count=10,
        display_timezone="UTC",
    )
    markdown = format_agent_context_markdown(bundle)
    assert markdown.index("older") < markdown.index("newer")


def test_format_chat_thread_markdown_is_chronological():
    buffered = [
        BufferedMessage(role="user", content="first", ts=100.0),
        BufferedMessage(role="assistant", content="reply one", ts=101.0),
        BufferedMessage(role="user", content="second", ts=200.0),
        BufferedMessage(role="assistant", content="reply two", ts=201.0),
    ]
    markdown = format_chat_thread_markdown(
        buffered,
        assistant_name="Lumen",
        timezone_name="UTC",
    )
    assert "first" in markdown
    assert "second" in markdown
    assert markdown.index("first") < markdown.index("second")


def test_format_chat_thread_markdown_respects_max_messages():
    buffered = [
        BufferedMessage(role="user", content=f"msg-{i}", ts=float(i))
        for i in range(10)
    ]
    markdown = format_chat_thread_markdown(
        buffered,
        assistant_name="Lumen",
        timezone_name="UTC",
        max_messages=3,
    )
    assert "msg-7" in markdown
    assert "msg-8" in markdown
    assert "msg-9" in markdown
    assert "msg-0" not in markdown
    assert "msg-6" not in markdown


def test_format_kevin_messages_timeline_numbers_and_orders():
    buffered = [
        BufferedMessage(role="user", content="first", ts=100.0),
        BufferedMessage(role="assistant", content="reply", ts=101.0),
        BufferedMessage(role="user", content="second", ts=200.0),
    ]
    markdown = format_kevin_messages_timeline(buffered, timezone_name="UTC")
    assert "## Kevin's messages this session" in markdown
    assert "1." in markdown and "2." in markdown
    assert markdown.index("first") < markdown.index("second")
    assert "UTC" in markdown


def test_format_kevin_messages_timeline_splits_sessions_by_gap():
    buffered = [
        BufferedMessage(role="user", content="yesterday hello", ts=100.0),
        BufferedMessage(role="assistant", content="hi", ts=101.0),
        BufferedMessage(role="user", content="Thank you Lumen", ts=10_000.0),
        BufferedMessage(role="assistant", content="You're welcome", ts=10_001.0),
        BufferedMessage(role="user", content="About timestamps…", ts=10_900.0),
    ]
    markdown = format_kevin_messages_timeline(
        buffered,
        timezone_name="UTC",
        session_gap_seconds=7200,
    )
    session_part = markdown.split("## Kevin's messages in buffer")[0]
    assert "Thank you Lumen" in session_part
    assert session_part.index("Thank you Lumen") < session_part.index("About timestamps")
    assert "yesterday hello" not in session_part


def test_build_agent_context_includes_origin_tags(monkeypatch):
    monkeypatch.setenv("MEMORY_INDEX_ENABLED", "false")
    settings = Settings(
        memory_store_path="/tmp/unused-memory",
        personal_db_enabled=False,
        memory_index_enabled=False,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="thought: Private awake prose here.",
                    score=0.0,
                    metadata={
                        "stream_source": "thought",
                        "ts": 1_700_000_000.0,
                        "reflection_summary": "I chose stillness.",
                    },
                ),
                MemoryHit(
                    text="user: hi\nassistant: hello",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 1_699_000_000.0},
                ),
            ]

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")

    bundle = build_agent_context(service, thread_id="kevin-home", agent_id="lumen")
    markdown = format_agent_context_markdown(bundle)

    assert "## Your conscious stream" in markdown
    assert "[rumination ·" in markdown
    assert "[chat ·" in markdown
    assert "Private awake prose" in markdown


def test_chat_and_rumination_markdown_identical(monkeypatch):
    monkeypatch.setenv("MEMORY_INDEX_ENABLED", "false")
    settings = Settings(
        memory_store_path="/tmp/unused-memory",
        personal_db_enabled=False,
        memory_index_enabled=False,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return [MemoryHit(text="Kevin is real.", score=0.0, metadata={"ts": 1.0})]

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="thought: Inner life.",
                    score=0.0,
                    metadata={"stream_source": "thought", "ts": 2.0},
                )
            ]

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "Foundation text.")
    monkeypatch.setattr(service, "format_personal_context", lambda _agent: "Personal line.")

    bundle_a = build_agent_context(service, thread_id="kevin-home", agent_id="lumen")
    bundle_b = build_agent_context(service, thread_id="kevin-home", agent_id="lumen")
    assert format_agent_context_markdown(bundle_a) == format_agent_context_markdown(bundle_b)


def test_build_agent_context_caps_stream_to_target(monkeypatch):
    monkeypatch.setenv("MEMORY_TARGET_CONTEXT_CHARS", "30")
    monkeypatch.setenv("MEMORY_INDEX_ENABLED", "false")
    settings = Settings(
        memory_store_path="/tmp/unused-memory",
        personal_db_enabled=False,
        memory_index_enabled=False,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="newest entry is longer than the cap allows",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 3.0},
                ),
                MemoryHit(
                    text="older",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 2.0},
                ),
                MemoryHit(
                    text="oldest",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 1.0},
                ),
            ]

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")

    bundle = build_agent_context(service, thread_id="kevin-home", agent_id="lumen")

    assert bundle.stream_char_count <= 30
    assert len(bundle.conscious_stream) == 1
    assert "newest entry" in bundle.conscious_stream[0].text


def test_build_agent_context_excludes_stream_origins(monkeypatch):
    monkeypatch.setenv("MEMORY_INDEX_ENABLED", "false")
    settings = Settings(
        memory_store_path="/tmp/unused-memory",
        personal_db_enabled=False,
        memory_index_enabled=False,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="dream: A lamp in the hall.",
                    score=0.0,
                    metadata={"stream_source": "dream", "ts": 3.0},
                ),
                MemoryHit(
                    text="thought: Quiet evening.",
                    score=0.0,
                    metadata={"stream_source": "thought", "ts": 2.0},
                ),
                MemoryHit(
                    text="user: hi\nassistant: hello",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 1.0},
                ),
            ]

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")

    bundle = build_agent_context(
        service,
        thread_id="kevin-home",
        agent_id="lumen",
        exclude_stream_origins=frozenset({"dream"}),
    )
    origins = {e.origin for e in bundle.conscious_stream}
    assert "dream" not in origins
    assert "rumination" in origins
    assert "chat" in origins


def test_stream_zone_for_origin_maps_families():
    assert stream_zone_for_origin("chat") == "chat"
    assert stream_zone_for_origin("kevin") == "chat"
    assert stream_zone_for_origin("rumination") == "rumination"
    assert stream_zone_for_origin("dream") == "dream"
    assert stream_zone_for_origin("peer") == "other"


def test_zoned_cap_reserves_chat_minimum_when_rumination_is_newer():
    settings = Settings(
        memory_store_path="/tmp/unused-memory",
        personal_db_enabled=False,
        memory_index_enabled=False,
    )
    limits = stream_zone_limits(settings, "chat", 100)
    entries = [
        StreamEntry(origin="rumination", text="r" * 60, ts=3.0),
        StreamEntry(origin="rumination", text="r" * 60, ts=2.0),
        StreamEntry(origin="chat", text="c" * 30, ts=1.0),
    ]
    capped = _cap_stream_entries_zoned(entries, max_chars=100, limits=limits)
    origins = {entry.origin for entry in capped}
    assert "chat" in origins
    assert sum(len(entry.text) for entry in capped) <= 100


def test_rumination_mode_favors_rumination_zone():
    settings = Settings(
        memory_store_path="/tmp/unused-memory",
        personal_db_enabled=False,
        memory_index_enabled=False,
    )
    limits = stream_zone_limits(settings, "rumination", 100)
    entries = [
        StreamEntry(origin="chat", text="c" * 80, ts=3.0),
        StreamEntry(origin="rumination", text="r" * 40, ts=2.0),
        StreamEntry(origin="rumination", text="r" * 40, ts=1.0),
    ]
    capped = _cap_stream_entries_zoned(entries, max_chars=100, limits=limits)
    rum_chars = sum(len(entry.text) for entry in capped if entry.origin == "rumination")
    chat_chars = sum(len(entry.text) for entry in capped if entry.origin == "chat")
    assert rum_chars >= chat_chars
    assert sum(len(entry.text) for entry in capped) <= 100


def test_build_agent_context_with_zones_disabled_uses_legacy_cap(monkeypatch):
    monkeypatch.setenv("MEMORY_INDEX_ENABLED", "false")
    settings = Settings(
        memory_store_path="/tmp/unused-memory",
        personal_db_enabled=False,
        stream_zones_enabled=False,
        memory_index_enabled=False,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="thought: " + ("inner " * 20),
                    score=0.0,
                    metadata={"stream_source": "thought", "ts": 2.0},
                ),
                MemoryHit(
                    text="user: hi\nassistant: hello",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 1.0},
                ),
            ]

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")

    bundle = build_agent_context(
        service,
        thread_id="kevin-home",
        agent_id="lumen",
        stream_max_chars=30,
        stream_mode="chat",
    )
    assert bundle.stream_char_count <= 30
    assert bundle.conscious_stream[0].origin == "rumination"
