"""Portable NDJSON rumination activity log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.inner_life_trace import RuminationTraceEntry
from light_house.rumination_log import (
    append_rumination_log,
    append_rumination_log_from_trace,
    format_rumination_log_entries,
    read_rumination_log_entries,
    RuminationLogRecord,
)


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "LIGHT_HOUSE_ENV": "production",
    }
    base.update(overrides)
    return Settings(**base)


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination.ndjson"
    settings = _test_settings(
        tmp_path,
        RUMINATION_LOG_ENABLED=True,
        RUMINATION_LOG_PATH=str(log_path),
    )
    append_rumination_log(
        settings,
        RuminationLogRecord(
            agent_id="lumen",
            thread_id="kevin-home",
            wake_kind="post_chat",
            summary_line="Closed the loop on the note.",
            tool_names=["write_note"],
            tool_rounds_used=1,
            first_response_mode="tools",
            persisted=True,
            persisted_chars=400,
        ),
    )
    entries, truncated = read_rumination_log_entries(log_path, max_lines=10)
    assert truncated is False
    assert len(entries) == 1
    assert entries[0]["agent_id"] == "lumen"
    assert entries[0]["summary_line"] == "Closed the loop on the note."
    assert entries[0]["tool_names"] == ["write_note"]


def test_append_from_trace_skipped_when_disabled(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination.ndjson"
    settings = _test_settings(
        tmp_path,
        RUMINATION_LOG_ENABLED=False,
        RUMINATION_LOG_PATH=str(log_path),
    )
    append_rumination_log_from_trace(
        settings,
        RuminationTraceEntry(
            agent_id="lumen",
            thread_id="kevin-home",
            wake_kind=None,
            tool_rounds_cap=None,
            context_markdown="",
            task_hint_label="scheduled",
            system_prompt_chars=0,
            conversation_window="",
            stream_event_count=0,
            stream_char_count=0,
            tool_rounds_used=0,
        ),
    )
    assert not log_path.exists()


def test_read_filters_by_agent(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination.ndjson"
    settings = _test_settings(
        tmp_path,
        RUMINATION_LOG_ENABLED=True,
        RUMINATION_LOG_PATH=str(log_path),
    )
    for agent in ("lumen", "ara", "lumen"):
        append_rumination_log(
            settings,
            RuminationLogRecord(
                agent_id=agent,
                thread_id="t",
                wake_kind=None,
                summary_line=f"run-{agent}",
                tool_names=[],
                tool_rounds_used=0,
                first_response_mode="plain_text",
                persisted=False,
                persisted_chars=0,
            ),
        )
    entries, _ = read_rumination_log_entries(log_path, max_lines=10, agent_id="lumen")
    assert len(entries) == 2
    assert all(e["agent_id"] == "lumen" for e in entries)


def test_read_rumination_log_for_agent(tmp_path: Path) -> None:
    from light_house.rumination_log import read_rumination_log_for_agent

    log_path = tmp_path / "rumination.ndjson"
    settings = _test_settings(
        tmp_path,
        RUMINATION_LOG_ENABLED=True,
        RUMINATION_LOG_PATH=str(log_path),
    )
    append_rumination_log(
        settings,
        RuminationLogRecord(
            agent_id="ara",
            thread_id="ara-home",
            wake_kind="scheduled",
            summary_line="Quiet check-in.",
            tool_names=[],
            tool_rounds_used=0,
            first_response_mode="plain_text",
            persisted=True,
            persisted_chars=120,
        ),
    )
    result = read_rumination_log_for_agent(
        settings,
        requesting_agent_id="lumen",
        tail=5,
        filter_agent_id="ara",
    )
    assert "Quiet check-in." in result
    assert "ara" in result


def test_tool_returns_disabled_message(tmp_path: Path) -> None:
    from light_house.rumination_log import read_rumination_log_for_agent

    settings = _test_settings(tmp_path, RUMINATION_LOG_ENABLED=False)
    result = read_rumination_log_for_agent(
        settings,
        requesting_agent_id="lumen",
        tail=5,
    )
    assert "disabled" in result.lower()


def test_format_empty_log() -> None:
    assert format_rumination_log_entries([], truncated=False) == "Rumination log is empty."
