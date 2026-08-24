"""Tests for compact rumination debug log (Proposal 3 phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.events.bus import format_event_log_display
from light_house.inner_life_trace import RuminationTraceEntry
from light_house.main import _settings_dep, app
from light_house.rumination_debug import (
    RuminationDebugEntry,
    append_rumination_debug,
    append_rumination_debug_from_trace,
    read_rumination_debug_tail,
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


def _install_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_append_debug_metadata_only(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination_debug.log"
    settings = _test_settings(
        tmp_path,
        RUMINATION_DEBUG_LOG_ENABLED=True,
        RUMINATION_DEBUG_LOG_PATH=str(log_path),
    )
    append_rumination_debug(
        settings,
        RuminationDebugEntry(
            agent_id="lumen",
            thread_id="kevin-home",
            wake_kind="post_chat",
            task_hint_label="post_chat",
            tool_rounds_cap=3,
            tool_rounds_used=1,
            tools_called=["write_note"],
            first_response_mode="tools",
            persisted=True,
            generated_text_chars=400,
            summary_line="Settled after checking notes.",
        ),
    )
    content, count, truncated = read_rumination_debug_tail(log_path, max_lines=200)
    assert count >= 5
    assert truncated is False
    assert "event_type: post_chat" in content
    assert "tool_rounds: 1 / cap 3" in content
    assert "Settled after checking notes." in content
    assert "context markdown" not in content.lower()
    assert "conversation window" not in content.lower()


def test_append_debug_from_trace(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination_debug.log"
    settings = _test_settings(
        tmp_path,
        RUMINATION_DEBUG_LOG_ENABLED=True,
        RUMINATION_DEBUG_LOG_PATH=str(log_path),
    )
    trace = RuminationTraceEntry(
        agent_id="ara",
        thread_id="ara-home",
        wake_kind=None,
        tool_rounds_cap=None,
        context_markdown="## Private stream\nsecret thought",
        task_hint_label="scheduled",
        system_prompt_chars=900,
        conversation_window="[HumanMessage]\nprivate diary",
        stream_event_count=1,
        stream_char_count=20,
        tool_rounds_used=0,
        persisted=False,
        generated_text_chars=0,
    )
    append_rumination_debug_from_trace(
        settings,
        trace,
        summary_line="Quiet scheduled pass.",
    )
    content, _, _ = read_rumination_debug_tail(log_path, max_lines=200)
    assert "agent=ara" in content
    assert "Quiet scheduled pass." in content
    assert "secret thought" not in content
    assert "private diary" not in content


def test_debug_skipped_when_disabled(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination_debug.log"
    settings = _test_settings(
        tmp_path,
        RUMINATION_DEBUG_LOG_ENABLED=False,
        RUMINATION_DEBUG_LOG_PATH=str(log_path),
    )
    append_rumination_debug(
        settings,
        RuminationDebugEntry(
            agent_id="lumen",
            thread_id="kevin-home",
            wake_kind=None,
            task_hint_label="scheduled",
            tool_rounds_cap=None,
            tool_rounds_used=0,
        ),
    )
    assert not log_path.exists()


def test_format_event_log_display() -> None:
    raw = "\n".join(
        [
            json.dumps(
                {
                    "event_type": "chat_received",
                    "ts": 1_700_000_000,
                    "agent_id": "lumen",
                    "thread_id": "kevin-home",
                    "payload": {},
                }
            ),
            json.dumps(
                {
                    "event_type": "report_ready",
                    "ts": 1_700_000_100,
                    "agent_id": "lumen",
                    "thread_id": None,
                    "payload": {"path": "shared/reports/lumen-test.md", "title": "Hi"},
                }
            ),
        ]
    )
    text = format_event_log_display(raw, truncated=True)
    assert "truncated" in text
    assert "chat_received" in text
    assert "report_ready" in text
    assert "shared/reports" in text


def test_observability_api_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    debug_path = tmp_path / "rumination_debug.log"
    debug_path.write_text("--- debug entry\n", encoding="utf-8")
    activity_path = tmp_path / "rumination.ndjson"
    activity_path.write_text(
        json.dumps(
            {
                "agent_id": "lumen",
                "wake_kind": "scheduled",
                "summary_line": "test",
                "tool_names": [],
                "tool_rounds_used": 0,
                "persisted": False,
                "persisted_chars": 0,
                "ts": 1_700_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    event_path = tmp_path / "events.log"
    event_path.write_text(
        json.dumps({"event_type": "scheduled_wake", "ts": 1_700_000_000, "payload": {}})
        + "\n",
        encoding="utf-8",
    )
    settings = _test_settings(
        tmp_path,
        RUMINATION_DEBUG_LOG_ENABLED=True,
        RUMINATION_DEBUG_LOG_PATH=str(debug_path),
        RUMINATION_LOG_ENABLED=True,
        RUMINATION_LOG_PATH=str(activity_path),
        EVENT_BUS_ENABLED=True,
        EVENT_BUS_LOG_PATH=str(event_path),
    )
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            debug = client.get("/v1/inner-life/debug-log?tail=10")
            activity = client.get("/v1/inner-life/activity-log?tail=10")
            events = client.get("/v1/events/log?tail=10")
        assert debug.status_code == 200
        assert "debug entry" in debug.json()["content"]
        assert activity.status_code == 200
        assert "lumen" in activity.json()["content"]
        assert events.status_code == 200
        assert "scheduled_wake" in events.json()["content"]
    finally:
        app.dependency_overrides.clear()
