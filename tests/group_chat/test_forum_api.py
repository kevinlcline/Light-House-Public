"""HTTP smoke for open-forum queue join / utter."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.group_chat.queue_room import reset_group_forum_for_tests
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.main import _settings_dep, app
from light_house.web_gate import SESSION_COOKIE, session_cookie_header


def _settings(tmp_path: Path) -> Settings:
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
        GROUP_CHAT_ENABLED=True,
        GROUP_CHAT_HISTORY_PATH=str(tmp_path / "group_chat/rounds.ndjson"),
        WEB_GATE_ENABLED=True,
        WEB_GATE_PASSWORD="dad-house-code",
        WEB_GATE_SESSION_SECRET="test-session-secret",
        INNER_LIFE_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )


def _cookie(settings: Settings) -> dict[str, str]:
    header = session_cookie_header(settings, secure=False, user_id="kevin", role="dad")
    token = header.split(";", 1)[0].split("=", 1)[1]
    return {SESSION_COOKIE: token}


@pytest.fixture
def forum_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    with TestClient(app) as client:
        yield client, settings
    app.dependency_overrides.pop(_settings_dep, None)
    reset_group_forum_for_tests()


def test_queue_join_grants_floor(forum_client) -> None:
    client, settings = forum_client
    cookies = _cookie(settings)
    res = client.post(
        "/v1/group-chat/queue/join",
        json={"speaker_id": "kevin", "display_name": "Kevin"},
        cookies=cookies,
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["joined"] is True
    assert data["floor"]["speaker_id"] == "kevin"
    assert data["paused"] is False

    uttered = client.post(
        "/v1/group-chat/utter",
        json={
            "message": "Hello open forum",
            "speaker_id": "kevin",
            "display_name": "Kevin",
        },
        cookies=cookies,
    )
    assert uttered.status_code == 200, uttered.text
    body = uttered.json()
    assert body["paused"] is True
    assert any(u["text"] == "Hello open forum" for u in body["transcript"])
