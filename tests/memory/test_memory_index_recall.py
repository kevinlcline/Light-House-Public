"""Memory index, FTS recall, and era summary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.memory.constants import MEMORY_KIND_ERA, STREAM_SOURCE_CHAT, STREAM_SOURCE_THOUGHT
from light_house.memory.context_builder import build_agent_context, format_agent_context_markdown
from light_house.memory.curator import _era_period_label
from light_house.memory.file_store import FileMemoryStore
from light_house.memory.index_builder import build_memory_index, format_memory_index_markdown
from light_house.memory.models import MemoryHit
from light_house.memory.service import MemoryService
from light_house.tools.memory_tools import recall_memory


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


def test_fts_search_finds_matching_stream(store: FileMemoryStore) -> None:
    store.remember_stream_event(
        thread_id="t1",
        text="We fixed the deploy pipeline on a rainy Tuesday.",
        stream_source=STREAM_SOURCE_CHAT,
    )
    store.remember_stream_event(
        thread_id="t1",
        text="Quiet rumination about solitude and light.",
        stream_source=STREAM_SOURCE_THOUGHT,
    )
    hits = store.search_stream_corpus("deploy pipeline", thread_id="t1", k=5)
    assert hits
    assert any("deploy" in hit.text.lower() for hit in hits)


def test_era_summary_listed_separately_from_corpus(store: FileMemoryStore) -> None:
    store.add_era_summary(
        thread_id="t1",
        period_label="2026-03",
        summary_text="A month of careful building.",
        doc_count=12,
    )
    store.remember_stream_event(
        thread_id="t1",
        text="chat line",
        stream_source=STREAM_SOURCE_CHAT,
    )
    corpus = store.list_thread_corpus(thread_id="t1")
    eras = store.list_era_summaries(thread_id="t1")
    assert len(corpus) == 1
    assert len(eras) == 1
    assert eras[0].metadata.get("memory_kind") == MEMORY_KIND_ERA


def test_memory_index_markdown_mentions_archival_awareness(store: FileMemoryStore) -> None:
    store.remember_stream_event(
        thread_id="t1",
        text="hello world",
        stream_source=STREAM_SOURCE_CHAT,
    )
    settings = Settings(memory_store_path=store._store_path, personal_db_enabled=False)
    index = build_memory_index(
        store,
        settings=settings,
        thread_id="t1",
        agent_id="lumen",
        in_prompt_count=1,
        in_prompt_chars=11,
    )
    md = format_memory_index_markdown(index, timezone_name="UTC")
    assert "Memory index" in md
    assert "recall_memory" in md
    assert "Stream corpus" in md


def test_build_agent_context_includes_memory_index(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        memory_store_path=tmp_path / "memory",
        personal_db_enabled=False,
        memory_index_enabled=True,
    )
    service = MemoryService(settings)
    service.remember_stream_event(
        thread_id="kevin-home",
        text="user: hi\nassistant: hello",
        stream_source=STREAM_SOURCE_CHAT,
    )
    monkeypatch.setattr(service, "get_foundation_context", lambda: "")
    bundle = build_agent_context(service, thread_id="kevin-home", agent_id="lumen")
    markdown = format_agent_context_markdown(bundle)
    assert "## Memory index" in markdown
    assert "## Your conscious stream" in markdown
    assert markdown.index("Memory index") < markdown.index("conscious stream")


def test_recall_memory_tool_returns_matches(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        memory_store_path=tmp_path / "memory",
        memory_recall_semantic_enabled=False,
    )
    service = MemoryService(settings)
    service.remember_stream_event(
        thread_id="kevin-home",
        text="The lighthouse journal upgrade was discussed at length.",
        stream_source=STREAM_SOURCE_CHAT,
    )

    def _service(_settings=None):
        return service

    monkeypatch.setattr("light_house.tools.memory_tools._memory_service", _service)
    monkeypatch.setattr(
        "light_house.tools.memory_tools.get_agent",
        lambda agent_id, _settings: type("L", (), {"thread_id": "kevin-home"})(),
    )
    result = recall_memory(agent_id="lumen", query="journal upgrade")
    assert "Recalled" in result
    assert "journal" in result.lower()


def test_era_period_label_single_month() -> None:
    hits = [
        MemoryHit(text="a", score=None, metadata={"ts": 1_700_000_000.0}),
        MemoryHit(text="b", score=None, metadata={"ts": 1_700_100_000.0}),
    ]
    label, start, end = _era_period_label(hits)
    assert "–" not in label
    assert start is not None and end is not None
