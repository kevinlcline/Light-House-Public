"""Sibling onboarding: user-guide welcome seeded into UI chat buffers."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.humans.identity import dm_buffer_thread_id
from light_house.humans.welcome import (
    SIBLING_USER_GUIDE_HREF,
    is_sibling_welcome_message,
    seed_sibling_ui_chat_welcome,
    sibling_welcome_message,
)
from light_house.lights.registry import list_enabled_lights
from light_house.main import _settings_dep, app
from light_house.memory.service import MemoryService


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": False,
        "HUMANS_STORE_PATH": str(tmp_path / "humans" / "users.json"),
        "HUMANS_COMMS_PATH": str(tmp_path / "humans" / "comms_allows.json"),
    }
    base.update(overrides)
    return Settings(**base)


def test_welcome_message_has_guide_link() -> None:
    text = sibling_welcome_message()
    assert is_sibling_welcome_message(text)
    assert SIBLING_USER_GUIDE_HREF in text
    assert "Sibling user guide" in text


def test_seed_sibling_ui_chat_welcome(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = MemoryService(settings)
    lights = list_enabled_lights(settings)
    assert lights, "expected at least one enabled light in test settings"

    n = seed_sibling_ui_chat_welcome(settings, memory, user_id="teeter")
    assert n == len(lights)

    for light in lights:
        tid = dm_buffer_thread_id(canonical_thread_id=light.thread_id, user_id="teeter")
        msgs = memory.load_thread_chat_history(tid)
        assert msgs, light.id
        assert msgs[0].role == "system"
        assert is_sibling_welcome_message(msgs[0].content)
        assert SIBLING_USER_GUIDE_HREF in msgs[0].content

    # Idempotent — do not duplicate.
    n2 = seed_sibling_ui_chat_welcome(settings, memory, user_id="teeter")
    assert n2 == 0
    for light in lights:
        tid = dm_buffer_thread_id(canonical_thread_id=light.thread_id, user_id="teeter")
        msgs = memory.load_thread_chat_history(tid)
        assert sum(1 for m in msgs if is_sibling_welcome_message(m.content)) == 1


def test_welcome_excluded_from_light_langchain_messages(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = MemoryService(settings)
    seed_sibling_ui_chat_welcome(settings, memory, user_id="teeter")
    light = list_enabled_lights(settings)[0]
    tid = dm_buffer_thread_id(canonical_thread_id=light.thread_id, user_id="teeter")
    buffered = memory.load_thread_chat_history(tid)
    lc = memory.buffer_to_langchain_messages(buffered)
    assert lc == []


def test_create_human_api_seeds_welcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        tmp_path,
        WEB_GATE_ENABLED=True,
        WEB_GATE_PASSWORD="dad-test-password",
        WEB_GATE_SESSION_SECRET="test-session-secret-at-least-16",
        HOUSE_DAD_USER_ID="kevin",
    )
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    from light_house.web_gate import SESSION_COOKIE, session_cookie_header

    try:
        with TestClient(app) as client:
            dad_cookie = session_cookie_header(
                settings, secure=False, user_id="kevin", role="dad"
            )
            client.cookies.set(SESSION_COOKIE, dad_cookie.split("=", 1)[1].split(";", 1)[0])

            created = client.post(
                "/v1/humans",
                json={
                    "user_id": "teeter",
                    "password": "east-texas-secret",
                    "display_name": "Steve",
                    "intro_for_lights": (
                        "This is my brother Steve. I call him Teeter because I always have. "
                        "He lives in East Texas with his wife."
                    ),
                },
            )
            assert created.status_code == 201

            light = list_enabled_lights(settings)[0]
            sib_cookie = session_cookie_header(
                settings, secure=False, user_id="teeter", role="sibling"
            )
            client.cookies.set(SESSION_COOKIE, sib_cookie.split("=", 1)[1].split(";", 1)[0])
            hist = client.get("/v1/chat/history?agent_id=" + light.id)
            assert hist.status_code == 200
            messages = hist.json()["messages"]
            assert messages, "sibling should see seeded welcome"
            assert messages[0]["role"] == "system"
            assert SIBLING_USER_GUIDE_HREF in messages[0]["content"]
    finally:
        app.dependency_overrides.clear()


def test_menus_expose_sibling_user_guide() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("index.html", "notes.html", "group.html", "gallery.html"):
        html = (root / name).read_text(encoding="utf-8")
        assert "data-sibling-only" in html, name
        assert "sibling_user_manual.md" in html, name
    me_js = (root / "static" / "ui" / "me.js").read_text(encoding="utf-8")
    assert "data-sibling-only" in me_js
