"""Inner life rumination trace log and preview API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from light_house.config import Settings
from light_house.inner_life_trace import (
    RuminationTraceEntry,
    append_rumination_trace,
    extract_tools_called,
    first_response_mode,
    format_messages_for_trace,
    read_inner_life_trace_tail,
    task_hint_label,
)
from light_house.main import _settings_dep, app
from light_house.memory.service import MemoryService


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": False,
        "LIGHT_HOUSE_ENV": "production",
    }
    base.update(overrides)
    return Settings(**base)


def _install_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_append_and_read_tail_roundtrip(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination-trace.log"
    settings = _test_settings(
        tmp_path,
        INNER_LIFE_TRACE_ENABLED=True,
        INNER_LIFE_TRACE_PATH=str(log_path),
    )
    entry = RuminationTraceEntry(
        agent_id="lumen",
        thread_id="kevin-home",
        wake_kind="post_chat",
        tool_rounds_cap=3,
        context_markdown="## Stream\n- hello",
        task_hint_label="post_chat",
        system_prompt_chars=1200,
        conversation_window="[HumanMessage]\nseed",
        stream_event_count=2,
        stream_char_count=40,
        tool_rounds_used=0,
        first_response_mode="plain_text",
        persisted=True,
        generated_text_chars=500,
    )
    append_rumination_trace(settings, entry)
    content, count, truncated = read_inner_life_trace_tail(log_path, max_lines=200)
    assert count >= 10
    assert truncated is False
    assert "agent=lumen" in content
    assert "wake=post_chat" in content
    assert "## Stream" in content
    assert "tool_rounds_used: 0" in content


def test_append_skipped_when_disabled(tmp_path: Path) -> None:
    log_path = tmp_path / "rumination-trace.log"
    settings = _test_settings(
        tmp_path,
        INNER_LIFE_TRACE_ENABLED=False,
        INNER_LIFE_TRACE_PATH=str(log_path),
    )
    append_rumination_trace(
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


def test_task_hint_label_and_message_helpers() -> None:
    assert task_hint_label("post_chat") == "post_chat"
    assert task_hint_label("kevin_shared_note") == "shared_note"
    assert task_hint_label(None) == "scheduled"

    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="thinking", tool_calls=[]),
    ]
    assert first_response_mode(msgs) == "plain_text"
    assert extract_tools_called(msgs) == []

    tool_msgs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "write_note", "args": {}, "id": "1"}],
        )
    ]
    assert first_response_mode(tool_msgs) == "tools"
    assert extract_tools_called(tool_msgs) == ["write_note"]
    assert "HumanMessage" in format_messages_for_trace(msgs)


def test_trace_api_returns_404_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(
        tmp_path,
        INNER_LIFE_TRACE_ENABLED=False,
        LIGHT_HOUSE_ENV="production",
    )
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            trace = client.get("/v1/inner-life/trace")
            preview = client.get("/v1/inner-life/context/preview?agent_id=lumen")
        assert trace.status_code == 404
        assert preview.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_trace_api_returns_tail_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "rumination-trace.log"
    log_path.write_text("block one\n=== end ===\n\nblock two\n", encoding="utf-8")
    settings = _test_settings(
        tmp_path,
        INNER_LIFE_TRACE_ENABLED=True,
        INNER_LIFE_TRACE_PATH=str(log_path),
        INNER_LIFE_TRACE_MAX_TAIL_LINES=2000,
    )
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.get("/v1/inner-life/trace?tail=2")
        assert response.status_code == 200
        data = response.json()
        assert data["lines"] == 2
        assert data["truncated"] is True
        assert "block two" in data["content"]
        assert response.headers.get("cache-control") == "no-store"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("agent_id", ["lumen", "ara"])
def test_preview_api_returns_markdown(
    tmp_path: Path, agent_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _test_settings(
        tmp_path,
        INNER_LIFE_TRACE_ENABLED=True,
        ARA_ENABLED=True,
    )
    memory = MemoryService(settings)
    _install_settings(settings, monkeypatch)
    try:
        from unittest.mock import patch

        with patch("light_house.main._memory", memory):
            with TestClient(app) as client:
                response = client.get(f"/v1/inner-life/context/preview?agent_id={agent_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == agent_id
        assert isinstance(data["context_markdown"], str)
        assert "meta" in data
        assert data["meta"]["stream_events"] >= 0
        assert isinstance(data["meta"]["tool_names"], list)
        assert len(data["meta"]["tool_names"]) > 0
        assert response.headers.get("cache-control") == "no-store"
    finally:
        app.dependency_overrides.clear()
