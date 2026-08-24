"""Memory curator: LLM summarize with algorithmic fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from light_house.config import Settings
from light_house.memory.constants import (
    SCORE_UNSCORED,
    STREAM_SOURCE_CHAT,
    STREAM_SOURCE_GROUP,
    STREAM_SOURCE_THOUGHT,
)
from light_house.memory.curator import CuratorReport, MemoryCurator
from light_house.memory.service import MemoryService


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "_env_file": None,
        "NOTES_PATH": str(notes_dir),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "CHROMA_PATH": str(tmp_path / "no-chroma"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "MEMORY_TARGET_CONTEXT_CHARS": 500,
        "MEMORY_CURATOR_PRUNE_RATIO": 0.9,
        "MEMORY_CURATOR_BATCH_SIZE": 5,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "ARA_ENABLED": True,
        "MEMORY_LEARNINGS_SUMMARIZE_ENABLED": False,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def fail_curator_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        content = ""

    fake_model = MagicMock()
    fake_model.invoke.return_value = FakeResponse()
    monkeypatch.setattr(
        "light_house.memory.curator.build_curator_chat_model",
        lambda *args, **kwargs: fake_model,
    )


def test_curator_prunes_low_retention_over_budget(
    tmp_path: Path, fail_curator_llm: None
) -> None:
    memory = MemoryService(_settings(tmp_path))
    high_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="important " * 40,
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={
            "impact_score": 9.0,
            "coherence_score": 9.0,
            "ts": 1_700_000_000.0,
        },
    )
    low_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="forgettable " * 80,
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={
            "impact_score": 1.0,
            "coherence_score": 1.0,
            "ts": 1_600_000_000.0,
        },
    )

    report = MemoryCurator(settings=memory._settings, memory=memory).run(thread_id="kevin-home")
    assert report.corpus_chars_after <= int(500 * 0.9)
    assert memory._long_term._get_doc(high_id) is not None
    assert memory._long_term._get_doc(low_id) is None
    assert report.deleted >= 1
    assert any("algorithmic prune" in action for action in report.actions)


def test_curator_protects_group_stream_lines(
    tmp_path: Path, fail_curator_llm: None
) -> None:
    memory = MemoryService(_settings(tmp_path, MEMORY_TARGET_CONTEXT_CHARS=100))
    group_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="[group] kevin: Can anyone hear me? " + ("x" * 80),
        stream_source=STREAM_SOURCE_GROUP,
        extra_metadata={
            "impact_score": 1.0,
            "coherence_score": 1.0,
            "ts": 1_600_000_000.0,
        },
    )
    filler_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="forgettable chat " * 40,
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={
            "impact_score": 1.0,
            "coherence_score": 1.0,
            "ts": 1_600_000_000.0,
        },
    )
    MemoryCurator(settings=memory._settings, memory=memory).run(thread_id="kevin-home")
    assert memory._long_term._get_doc(group_id) is not None
    assert memory._long_term._get_doc(filler_id) is None


def test_curator_prunes_old_unscored_before_recent(
    tmp_path: Path, fail_curator_llm: None
) -> None:
    memory = MemoryService(_settings(tmp_path, MEMORY_TARGET_CONTEXT_CHARS=200))
    recent_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="recent unscored " * 10,
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={"ts": 1_750_000_000.0},
    )
    old_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="old unscored " * 30,
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={"ts": 1_600_000_000.0},
    )
    for hit in memory._long_term.list_thread_corpus(thread_id="kevin-home"):
        assert float(hit.metadata.get("impact_score", SCORE_UNSCORED)) < 0

    MemoryCurator(settings=memory._settings, memory=memory).run(thread_id="kevin-home")
    assert memory._long_term._get_doc(recent_id) is not None
    assert memory._long_term._get_doc(old_id) is None


def test_curator_skips_fresh_thoughts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_curator_llm: None
) -> None:
    import light_house.memory.curator as curator_mod

    memory = MemoryService(_settings(tmp_path, MEMORY_TARGET_CONTEXT_CHARS=100))
    thought_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="fresh inner thought " * 20,
        stream_source=STREAM_SOURCE_THOUGHT,
        extra_metadata={"ts": 1_750_000_000.0},
    )
    filler_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="filler " * 50,
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={"ts": 1_600_000_000.0, "impact_score": 1.0, "coherence_score": 1.0},
    )
    monkeypatch.setattr(curator_mod.time, "time", lambda: 1_750_000_100.0)

    MemoryCurator(settings=memory._settings, memory=memory).run(thread_id="kevin-home")
    assert memory._long_term._get_doc(thought_id) is not None
    assert memory._long_term._get_doc(filler_id) is None


def test_curator_llm_summarize_before_algorithmic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = MemoryService(_settings(tmp_path, MEMORY_TARGET_CONTEXT_CHARS=300))
    low_id = memory.remember_stream_event(
        thread_id="kevin-home",
        text="low value " * 60,
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={"impact_score": 1.0, "coherence_score": 1.0, "ts": 1.0},
    )

    class FakeResponse:
        content = (
            '{"summary":"A quiet evening together.","delete_ids":["'
            + low_id
            + '"],"fade_ids":[]}'
        )

    fake_model = MagicMock()
    fake_model.invoke.return_value = FakeResponse()
    monkeypatch.setattr(
        "light_house.memory.curator.build_curator_chat_model",
        lambda *args, **kwargs: fake_model,
    )

    report = MemoryCurator(settings=memory._settings, memory=memory).run(thread_id="kevin-home")
    assert memory._long_term._get_doc(low_id) is None
    assert any("condensed" in action for action in report.actions)
    assert not any("algorithmic prune" in action for action in report.actions)
    summaries = memory._long_term.list_recent_summaries(thread_id="kevin-home", limit=5)
    assert any("quiet evening" in hit.text for hit in summaries)
