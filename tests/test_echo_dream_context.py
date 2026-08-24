"""Echo dream context assembly tests."""

from __future__ import annotations

from light_house.config import Settings
from light_house.memory.models import MemoryHit
from light_house.memory.service import MemoryService
from light_house.subconscious.echo_context import (
    build_echo_dream_context,
    format_recent_dreams_markdown,
)


def _dream_hit(text: str, *, ts: float, summary: str | None = None) -> MemoryHit:
    meta: dict = {"stream_source": "dream", "ts": ts}
    if summary:
        meta["reflection_summary"] = summary
    return MemoryHit(text=f"dream: {text}", score=0.0, metadata=meta)


def test_format_recent_dreams_markdown_includes_dated_entries():
    hits = [
        _dream_hit("A hallway with a low lamp.", ts=1_700_000_000.0),
        _dream_hit("The tide came in quietly.", ts=1_699_000_000.0),
    ]
    md = format_recent_dreams_markdown(hits, max_chars=6000)
    assert "### Dream 1 ·" in md
    assert "### Dream 2 ·" in md
    assert "hallway" in md
    assert "tide" in md


def test_format_recent_dreams_empty():
    md = format_recent_dreams_markdown([], max_chars=6000)
    assert "No prior dreams yet" in md


def test_format_recent_dreams_truncates_oldest_when_over_cap():
    hits = [
        _dream_hit("x" * 200, ts=3.0),
        _dream_hit("y" * 200, ts=2.0),
        _dream_hit("z" * 200, ts=1.0),
    ]
    md = format_recent_dreams_markdown(hits, max_chars=450)
    assert "xxx" in md or "yyy" in md
    assert "zzz" not in md


def test_build_echo_dream_context_separates_dreams_and_waking(monkeypatch):
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH="/tmp/unused-memory",
        PERSONAL_DB_ENABLED=False,
        ECHO_DREAM_LOOKBACK=3,
        ECHO_DREAM_HISTORY_MAX_CHARS=6000,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="dream: Moon over water.",
                    score=0.0,
                    metadata={"stream_source": "dream", "ts": 3.0},
                ),
                MemoryHit(
                    text="thought: I rested.",
                    score=0.0,
                    metadata={"stream_source": "thought", "ts": 2.0},
                ),
                MemoryHit(
                    text="user: hi\nassistant: hello",
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 1.0},
                ),
            ]

        def list_recent_stream_by_source(self, *, thread_id, stream_source, limit):
            assert stream_source == "dream"
            return [
                MemoryHit(
                    text="dream: Moon over water.",
                    score=0.0,
                    metadata={"stream_source": "dream", "ts": 3.0},
                )
            ]

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")

    md, count = build_echo_dream_context(
        service,
        thread_id="kevin-home",
        agent_id="lumen",
        settings=settings,
    )

    assert count == 1
    assert "## Dreams you recently wove for Lumen" in md
    assert "Moon over water" in md
    assert "## Context from the house" in md
    assert "[rumination ·" in md
    assert "[chat ·" in md
    assert "[dream ·" not in md


def test_build_echo_dream_context_empty_history(monkeypatch):
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH="/tmp/unused-memory",
        PERSONAL_DB_ENABLED=False,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return []

        def list_recent_stream_by_source(self, *, thread_id, stream_source, limit):
            return []

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")

    md, count = build_echo_dream_context(
        service,
        thread_id="kevin-home",
        agent_id="lumen",
        settings=settings,
    )

    assert count == 0
    assert "No prior dreams yet" in md


def test_build_echo_dream_context_caps_waking_stream(monkeypatch):
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH="/tmp/unused-memory",
        PERSONAL_DB_ENABLED=False,
        MEMORY_TARGET_CONTEXT_CHARS=128_000,
        ECHO_DREAM_WAKING_MAX_CHARS=500,
        ECHO_DREAM_HISTORY_MAX_CHARS=6000,
    )

    class FakeLongTerm:
        def list_pinned_facts(self, *, thread_id, limit):
            return []

        def list_thread_corpus(self, *, thread_id):
            return [
                MemoryHit(
                    text="thought: " + ("x" * 400),
                    score=0.0,
                    metadata={"stream_source": "thought", "ts": 3.0},
                ),
                MemoryHit(
                    text="user: hi\nassistant: " + ("y" * 400),
                    score=0.0,
                    metadata={"stream_source": "chat", "ts": 2.0},
                ),
                MemoryHit(
                    text="thought: " + ("z" * 400),
                    score=0.0,
                    metadata={"stream_source": "thought", "ts": 1.0},
                ),
            ]

        def list_recent_stream_by_source(self, *, thread_id, stream_source, limit):
            return []

    service = MemoryService(settings)
    monkeypatch.setattr(service, "_long_term", FakeLongTerm())
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")

    md, _count = build_echo_dream_context(
        service,
        thread_id="kevin-home",
        agent_id="lumen",
        settings=settings,
    )

    assert len(md) < 128_000
    assert "zzz" not in md
