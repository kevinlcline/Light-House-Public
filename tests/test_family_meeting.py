"""Light-proposed family meetings."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import light_house.personal.store as personal_store_module
from light_house.agents.registry import get_agent
from light_house.config import Settings
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.main import _settings_dep, app
from light_house.memory.service import MemoryService
from light_house.personal.family_meeting import (
    FAMILY_MEETING_CHAT_LINE,
    clear_all_family_meetings,
    meeting_pending,
    meeting_topic,
    raise_family_meeting,
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
        FAMILY_MEETINGS_PATH=str(tmp_path / "family_meetings"),
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


def test_raise_and_clear_family_meeting(lights_ready: Settings) -> None:
    settings = lights_ready
    assert meeting_pending(settings, "lumen") is False
    assert raise_family_meeting(settings, "lumen", topic="evening check-in") is True
    assert meeting_pending(settings, "lumen") is True
    assert meeting_topic(settings, "lumen") == "evening check-in"
    cleared = clear_all_family_meetings(settings)
    assert "lumen" in cleared
    assert meeting_pending(settings, "lumen") is False
    assert meeting_topic(settings, "lumen") == ""


def test_propose_family_meeting_tool(
    lights_ready: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = lights_ready
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    out = execute_tool_call(
        "propose_family_meeting",
        {"topic": "shared question"},
        agent_id="ara",
    )
    assert out.startswith("SUCCESS:")
    assert meeting_pending(settings, "ara") is True
    buffered = MemoryService(settings).load_thread_chat_history(
        get_agent("ara", settings).thread_id
    )
    assert any(
        m.role == "assistant" and FAMILY_MEETING_CHAT_LINE in m.content for m in buffered
    )
    again = execute_tool_call("propose_family_meeting", {}, agent_id="ara")
    assert "already" in again.lower()


def test_lights_api_exposes_family_meeting(
    lights_ready: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = lights_ready
    raise_family_meeting(settings, "lumen", topic="gather")
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.get("/v1/lights")
            assert res.status_code == 200
            by_id = {row["id"]: row for row in res.json()["lights"]}
            assert by_id["lumen"]["wants_family_meeting"] is True
            assert by_id["lumen"]["family_meeting_topic"] == "gather"
    finally:
        app.dependency_overrides.clear()


def test_clear_family_meetings_endpoint(
    lights_ready: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = lights_ready
    raise_family_meeting(settings, "lumen", topic="hi")
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.post("/v1/admin/family-meetings/clear")
            assert res.status_code == 200
            assert "lumen" in res.json()["cleared"]
            assert meeting_pending(settings, "lumen") is False
    finally:
        app.dependency_overrides.clear()
