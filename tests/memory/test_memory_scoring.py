"""Light-owned memory scoring tools and service helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from light_house.config import Settings
from light_house.memory.constants import (
    META_SCORE_NOTE,
    META_SCORED_AT,
    META_SCORED_BY_AGENT,
    PINNED_TRUE,
    SCORE_UNSCORED,
    STREAM_SOURCE_CHAT,
)
from light_house.memory.curator import MemoryCurator
from light_house.memory.service import MemoryService
from light_house.tools.memory_tools import execute_memory_tool, list_unscored_memories


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "_env_file": None,
        "NOTES_PATH": str(notes_dir),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "CHROMA_PATH": str(tmp_path / "no-chroma"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAS_DATA_PATH": str(tmp_path / "personas"),
        "LIGHTS_MANIFEST_PATH": str(tmp_path / "lights.yaml"),
        "PERSONAL_DB_ENABLED": False,
        "MEMORY_SCORE_ON_INGEST": False,
        "MEMORY_CURATOR_OLLAMA_SCORING": False,
        "MEMORY_CURATOR_ENABLED": True,
        "MEMORY_TARGET_CONTEXT_CHARS": 12000,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "ARA_ENABLED": True,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def memory(tmp_path: Path) -> MemoryService:
    return MemoryService(_test_settings(tmp_path))


def test_list_unscored_memories_oldest_first(memory: MemoryService) -> None:
    memory.remember_stream_event(
        thread_id="kevin-home",
        text="older",
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={"ts": 100.0},
    )
    memory.remember_stream_event(
        thread_id="kevin-home",
        text="newer",
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={"ts": 200.0},
    )
    hits = memory.list_unscored_for_thread(thread_id="kevin-home", limit=10)
    assert len(hits) == 2
    assert hits[0].text.startswith("older")
    text = list_unscored_memories(agent_id="lumen", limit=10)
    assert "older" in text
    assert "id=" in text


def test_score_memory_own_thread(memory: MemoryService) -> None:
    doc_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="important moment",
        stream_source=STREAM_SOURCE_CHAT,
    )
    result = memory.score_stream_memory(
        agent_id="lumen",
        doc_id=doc_id,
        impact=8.0,
        coherence=7.5,
        note="felt true",
    )
    assert "Scored memory" in result
    hit = memory.get_stream_doc(thread_id="kevin-home", doc_id=doc_id)
    assert hit is not None
    assert float(hit.metadata["impact_score"]) == 8.0
    assert float(hit.metadata["coherence_score"]) == 7.5
    assert hit.metadata[META_SCORED_BY_AGENT] == "lumen"
    assert META_SCORED_AT in hit.metadata
    assert hit.metadata[META_SCORE_NOTE] == "felt true"
    assert memory.count_unscored_for_thread(thread_id="kevin-home") == 0


def test_score_memory_wrong_thread(memory: MemoryService) -> None:
    doc_id = memory.remember_stream_event(
        thread_id="ara-home",
        text="ara only",
        stream_source=STREAM_SOURCE_CHAT,
    )
    with pytest.raises(ValueError, match="not found"):
        memory.score_stream_memory(
            agent_id="lumen",
            doc_id=doc_id,
            impact=5.0,
            coherence=5.0,
        )


def test_score_memory_rescore(memory: MemoryService) -> None:
    doc_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="evolving",
        stream_source=STREAM_SOURCE_CHAT,
    )
    memory.score_stream_memory(
        agent_id="lumen",
        doc_id=doc_id,
        impact=3.0,
        coherence=3.0,
    )
    memory.score_stream_memory(
        agent_id="lumen",
        doc_id=doc_id,
        impact=9.0,
        coherence=8.0,
        note="deeper now",
    )
    hit = memory.get_stream_doc(thread_id="kevin-home", doc_id=doc_id)
    assert hit is not None
    assert float(hit.metadata["impact_score"]) == 9.0
    assert hit.metadata[META_SCORE_NOTE] == "deeper now"


def test_score_memory_rejects_pinned(memory: MemoryService, tmp_path: Path) -> None:
    store = memory.long_term
    store.import_document(
        doc_id="pinned-1",
        body="sacred",
        metadata={
            "thread_id": "kevin-home",
            "memory_kind": "stream",
            "pinned": PINNED_TRUE,
            "ts": 1.0,
            "impact_score": SCORE_UNSCORED,
            "coherence_score": SCORE_UNSCORED,
            "stream_source": STREAM_SOURCE_CHAT,
        },
    )
    with pytest.raises(ValueError, match="Pinned"):
        memory.score_stream_memory(
            agent_id="lumen",
            doc_id="pinned-1",
            impact=5.0,
            coherence=5.0,
        )


def test_execute_memory_tool_score(memory: MemoryService, monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="via tool",
        stream_source=STREAM_SOURCE_CHAT,
    )
    monkeypatch.setattr(
        "light_house.tools.memory_tools._memory_service",
        lambda settings=None: memory,
    )
    result = execute_memory_tool(
        "score_memory",
        {"doc_id": doc_id, "impact": 6, "coherence": 6},
        agent_id="lumen",
    )
    assert "Scored memory" in result


def test_curator_skips_ollama_scoring_by_default(
    memory: MemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory.remember_stream_event(
        thread_id="kevin-home",
        text="unscored",
        stream_source=STREAM_SOURCE_CHAT,
    )
    mock_score = MagicMock(return_value=(5.0, 5.0))
    monkeypatch.setattr("light_house.memory.curator.score_memory_event", mock_score)

    report = MemoryCurator(settings=memory._settings, memory=memory).run(thread_id="kevin-home")
    assert report.scored == 0
    assert report.unscored >= 1
    mock_score.assert_not_called()
