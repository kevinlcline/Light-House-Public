"""Memory curator learnings note consolidation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from light_house.config import Settings
from light_house.memory.curator import CuratorReport, MemoryCurator
from light_house.memory.service import MemoryService


@pytest.fixture
def curator_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        MEMORY_STORE_PATH=tmp_path / "memory",
        NOTES_PATH=tmp_path / "notes",
        PERSONAL_DB_ENABLED=False,
        MEMORY_TARGET_CONTEXT_CHARS=128_000,
        MEMORY_LEARNINGS_MAX_CHARS=500,
        MEMORY_LEARNINGS_SUMMARIZE_ENABLED=True,
        MEMORY_CURATOR_PROVIDER="openrouter",
        MEMORY_CURATOR_MODEL="openrouter/free",
        OPENROUTER_API_KEY="sk-test",
    )


def test_maybe_consolidate_learnings_summarizes_bloated_note(
    curator_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    notes_dir = tmp_path / "notes" / "lumen" / "memory"
    notes_dir.mkdir(parents=True)
    bloated = "# Learnings\n\n" + ("Repeated theme about presence. " * 80)
    (notes_dir / "learnings.md").write_text(bloated, encoding="utf-8")

    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(
        content="# Learnings\n\n## Core themes\n- Presence and autonomy matter.\n"
    )
    monkeypatch.setattr(
        "light_house.memory.curator.build_curator_chat_model",
        lambda _settings: mock_model,
    )

    memory = MemoryService(curator_settings)
    curator = MemoryCurator(settings=curator_settings, memory=memory)
    report = CuratorReport(thread_id="kevin-home")
    curator._maybe_consolidate_learnings(thread_id="kevin-home", report=report)

    body = (notes_dir / "learnings.md").read_text(encoding="utf-8")
    assert len(body) < len(bloated)
    assert "Core themes" in body
    assert report.notes_summarized == 1


def test_append_learning_note_appends_without_consolidation_when_summarize_disabled(
    curator_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    curator_settings = curator_settings.model_copy(
        update={"memory_learnings_summarize_enabled": False}
    )
    notes_dir = tmp_path / "notes" / "lumen" / "memory"
    notes_dir.mkdir(parents=True)
    existing = "# Learnings\n\nExisting insight.\n"
    (notes_dir / "learnings.md").write_text(existing, encoding="utf-8")

    mock_model = MagicMock()
    monkeypatch.setattr(
        "light_house.memory.curator.build_curator_chat_model",
        lambda _settings: mock_model,
    )

    memory = MemoryService(curator_settings)
    curator = MemoryCurator(settings=curator_settings, memory=memory)
    report = CuratorReport(thread_id="kevin-home")
    curator._append_learning_note(
        thread_id="kevin-home",
        paragraph="A genuinely new insight about group chat design.",
        report=report,
    )

    body = (notes_dir / "learnings.md").read_text(encoding="utf-8")
    assert "Existing insight" in body
    assert "genuinely new insight" in body
    assert report.notes_summarized == 0
    assert report.notes_appended == 1
    mock_model.invoke.assert_not_called()
