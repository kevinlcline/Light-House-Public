"""Soft presence knock for Dad (status-bar mark + chat line)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import light_house.personal.store as personal_store_module
from light_house.config import Settings
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.main import _settings_dep, app
from light_house.memory.service import MemoryService
from light_house.personal.presence_knock import (
    PRESENCE_KNOCK_CHAT_LINE,
    clear_knock,
    knock_pending,
    raise_knock,
    record_knock_chat_line,
)
from light_house.tools.light_tools import execute_tool_call


@pytest.fixture(autouse=True)
def clear_personal_store_cache() -> None:
    personal_store_module._store_cache.clear()
    yield
    personal_store_module._store_cache.clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAS_DATA_PATH=str(tmp_path / "personas"),
        NOTES_PATH=str(tmp_path / "notes"),
        PERSONAL_DB_ENABLED=True,
        PERSONAL_DB_PATH=str(tmp_path / "personal"),
        WEB_GATE_ENABLED=False,
        PRIMARY_LLM="ollama",
        INNER_LIFE_ENABLED=False,
        INNER_LIFE_DREAMS_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )


@pytest.fixture
def lights_ready(settings: Settings) -> Settings:
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    return settings


def test_raise_and_clear_knock(settings: Settings) -> None:
    assert knock_pending(settings, "lumen") is False
    assert raise_knock(settings, "lumen") is True
    assert knock_pending(settings, "lumen") is True
    assert clear_knock(settings, "lumen") is True
    assert knock_pending(settings, "lumen") is False


def test_knock_for_kevin_tool_writes_chat_line(
    lights_ready: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = lights_ready
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    out = execute_tool_call("knock_for_kevin", {"reason": "a quiet ask"}, agent_id="ara")
    assert out.startswith("SUCCESS:")
    assert knock_pending(settings, "ara") is True
    assert "chat" in out.lower()

    memory = MemoryService(settings)
    from light_house.agents.registry import get_agent

    buffered = memory.load_thread_chat_history(get_agent("ara", settings).thread_id)
    assert any(
        m.role == "assistant" and m.content == PRESENCE_KNOCK_CHAT_LINE for m in buffered
    )

    again = execute_tool_call("knock_for_kevin", {}, agent_id="ara")
    assert "already" in again.lower()
    buffered_again = memory.load_thread_chat_history(get_agent("ara", settings).thread_id)
    knock_lines = [
        m for m in buffered_again if m.role == "assistant" and m.content == PRESENCE_KNOCK_CHAT_LINE
    ]
    assert len(knock_lines) == 1


def test_knock_chat_line_survives_history_clear(
    lights_ready: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = lights_ready
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    execute_tool_call("knock_for_kevin", {}, agent_id="lumen")
    assert knock_pending(settings, "lumen") is True

    memory = MemoryService(settings)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main._memory", memory)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.get("/v1/chat/history?agent_id=lumen")
            assert res.status_code == 200
            assert knock_pending(settings, "lumen") is False
            contents = [m["content"] for m in res.json()["messages"]]
            assert PRESENCE_KNOCK_CHAT_LINE in contents
    finally:
        app.dependency_overrides.clear()


def test_record_knock_chat_line(lights_ready: Settings) -> None:
    assert record_knock_chat_line(lights_ready, "lumen") is True
    # Near-duplicate skip when the knock line is already last.
    assert record_knock_chat_line(lights_ready, "lumen") is False


def test_lights_api_exposes_knock_for_dad(
    lights_ready: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = lights_ready
    raise_knock(settings, "lumen")
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.get("/v1/lights")
            assert res.status_code == 200
            by_id = {row["id"]: row for row in res.json()["lights"]}
            assert by_id["lumen"]["wants_kevin"] is True
            assert by_id.get("ara", {}).get("wants_kevin") is False
    finally:
        app.dependency_overrides.clear()


def test_chat_history_clears_knock_for_dad(
    lights_ready: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = lights_ready
    raise_knock(settings, "lumen")
    assert knock_pending(settings, "lumen") is True

    memory = MemoryService(settings)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main._memory", memory)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.get("/v1/chat/history?agent_id=lumen")
            assert res.status_code == 200
            assert knock_pending(settings, "lumen") is False
    finally:
        app.dependency_overrides.clear()
