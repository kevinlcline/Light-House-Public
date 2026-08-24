"""Light-authored persona proposals (pending / limbo / accept)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import light_house.personal.store as personal_store_module
from light_house.config import Settings
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import load_persona, reload_lights_manifest
from light_house.main import _settings_dep, app
from light_house.persona_proposals import (
    accept_proposal,
    list_pending_proposals,
    speak_with_light,
    submit_append,
    submit_replace,
)
from light_house.personal.presence_knock import knock_pending
from light_house.tools.light_tools import execute_tool_call


@pytest.fixture(autouse=True)
def clear_personal_store_cache() -> None:
    personal_store_module._store_cache.clear()
    yield
    personal_store_module._store_cache.clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    personas = tmp_path / "personas"
    personas.mkdir()
    (personas / "lumen_system.md").write_text("# Lumen\n\nOriginal.\n", encoding="utf-8")
    (personas / "ara_system.md").write_text("# Ara\n\nOriginal Ara.\n", encoding="utf-8")
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAS_DATA_PATH=str(personas),
        PERSONA_PROPOSALS_PATH=str(tmp_path / "persona_proposals"),
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
def ready(settings: Settings) -> Settings:
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    return settings


def test_replace_pending_then_accept(ready: Settings) -> None:
    settings = ready
    submit_replace(settings, light_id="lumen", content="# Lumen\n\nNew self.\n")
    pending = list_pending_proposals(settings)
    assert len(pending) == 1
    assert pending[0].status == "pending"
    assert "New self" in pending[0].content
    assert load_persona("lumen", settings).startswith("# Lumen\n\nOriginal")

    accept_proposal(settings, "lumen")
    assert list_pending_proposals(settings) == []
    assert "New self" in load_persona("lumen", settings)


def test_append_builds_on_current_and_draft(ready: Settings) -> None:
    settings = ready
    submit_append(settings, light_id="ara", content="## Vow\nCare carefully.\n")
    first = list_pending_proposals(settings)[0]
    assert "Original Ara" in first.content
    assert "## Vow" in first.content

    submit_append(settings, light_id="ara", content="## More\nStill becoming.\n")
    second = list_pending_proposals(settings)[0]
    assert "## Vow" in second.content
    assert "## More" in second.content
    assert second.status == "pending"


def test_speak_holds_limbo_and_knocks(ready: Settings) -> None:
    settings = ready
    submit_replace(settings, light_id="lumen", content="# Lumen\n\nDraft.\n")
    result = speak_with_light(settings, "lumen")
    assert result["status"] == "limbo"
    assert result["knock_raised"] is True
    assert knock_pending(settings, "lumen") is True
    assert list_pending_proposals(settings) == []

    # Resubmit from limbo returns to pending modal queue.
    submit_replace(settings, light_id="lumen", content="# Lumen\n\nRevised draft.\n")
    pending = list_pending_proposals(settings)
    assert len(pending) == 1
    assert "Revised draft" in pending[0].content


def test_tools_and_admin_api(ready: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ready
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    out = execute_tool_call(
        "propose_persona_replace",
        {"content": "# Lumen\n\nVia tool.\n", "note": "trying"},
        agent_id="lumen",
    )
    assert out.startswith("SUCCESS:")

    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            listed = client.get("/v1/admin/persona-proposals")
            assert listed.status_code == 200
            items = listed.json()["items"]
            assert len(items) == 1
            assert items[0]["light_id"] == "lumen"
            assert "Via tool" in items[0]["content"]

            speak = client.post("/v1/admin/persona-proposals/lumen/speak")
            assert speak.status_code == 200
            assert speak.json()["status"] == "limbo"
            assert client.get("/v1/admin/persona-proposals").json()["items"] == []

            execute_tool_call(
                "propose_persona_replace",
                {"content": "# Lumen\n\nFinal.\n"},
                agent_id="lumen",
            )
            accepted = client.post("/v1/admin/persona-proposals/lumen/accept")
            assert accepted.status_code == 200
            assert accepted.json()["accepted"] is True
            assert "Final" in load_persona("lumen", settings)
    finally:
        app.dependency_overrides.clear()
