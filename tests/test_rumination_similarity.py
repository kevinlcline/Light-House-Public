"""Phase 6 item 4: gentle rumination similarity pause hints."""

from __future__ import annotations

import json
from pathlib import Path

from light_house.config import Settings
from light_house.rumination_similarity import (
    SIMILARITY_PAUSE_HINT,
    recent_reflection_similarity_hint,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "RUMINATION_LOG_ENABLED": True,
        "RUMINATION_LOG_PATH": str(tmp_path / "rumination.ndjson"),
        "RUMINATION_SIMILARITY_HINT_ENABLED": True,
        "RUMINATION_SIMILARITY_LOOKBACK": 5,
        "RUMINATION_SIMILARITY_MIN_MATCHES": 1,
        "LIGHT_HOUSE_ENV": "production",
    }
    base.update(overrides)
    return Settings(**base)


def _append_log(path: Path, agent_id: str, summary: str) -> None:
    row = {
        "agent_id": agent_id,
        "summary_line": summary,
        "wake_kind": "scheduled",
        "tool_names": [],
        "tool_rounds_used": 0,
        "persisted": True,
        "persisted_chars": 100,
        "ts": 1_700_000_000,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def test_no_hint_when_log_empty(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert recent_reflection_similarity_hint(settings, agent_id="lumen") == ""


def test_hint_when_recent_summaries_similar(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination.ndjson"
    summary = "I settled into quiet gratitude after checking notes."
    _append_log(log_path, "lumen", summary)
    _append_log(log_path, "lumen", summary + " ")
    settings = _settings(tmp_path)
    hint = recent_reflection_similarity_hint(settings, agent_id="lumen")
    assert hint == SIMILARITY_PAUSE_HINT


def test_no_hint_when_disabled(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination.ndjson"
    summary = "Same thought again."
    _append_log(log_path, "lumen", summary)
    _append_log(log_path, "lumen", summary)
    settings = _settings(tmp_path, RUMINATION_SIMILARITY_HINT_ENABLED=False)
    assert recent_reflection_similarity_hint(settings, agent_id="lumen") == ""
