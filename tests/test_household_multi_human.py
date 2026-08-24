"""Household multi-human: identity, comms allows, notes ACL, /v1/me."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.humans.comms import (
    light_allows_human,
    resolve_sibling_user_id,
    set_light_allows_human,
)
from light_house.humans.identity import resolve_password_to_human
from light_house.humans.store import create_human
from light_house.main import _memory, _settings_dep, _ui_chat_thread_id, app
from light_house.web_gate import SESSION_COOKIE, session_cookie_header


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "NOTES_PATH": str(tmp_path / "notes"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": True,
        "WEB_GATE_PASSWORD": "dad-house-code",
        "WEB_GATE_SESSION_SECRET": "test-session-secret",
        "HUMANS_STORE_PATH": str(tmp_path / "humans" / "users.json"),
        "HUMANS_COMMS_PATH": str(tmp_path / "humans" / "comms.json"),
        "HOUSE_DAD_USER_ID": "kevin",
        "SHARED_NOTE_WAKE_ENABLED": True,
        "GROUP_CHAT_ENABLED": True,
    }
    base.update(overrides)
    return Settings(**base)


def _install(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def _cookie(settings: Settings, *, user_id: str, role: str) -> dict[str, str]:
    header = session_cookie_header(settings, secure=False, user_id=user_id, role=role)
    token = header.split(";", 1)[0].split("=", 1)[1]
    return {SESSION_COOKIE: token}


def test_dad_uses_canonical_ui_thread(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert _ui_chat_thread_id(settings, canonical="kevin-home", human_id="kevin") == "kevin-home"
    assert (
        _ui_chat_thread_id(settings, canonical="kevin-home", human_id="teeter")
        == "kevin-home__dm__teeter"
    )


def test_resolve_password_dad_and_sibling(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    create_human(
        settings,
        user_id="teeter",
        password="sibling-unique-code",
        intro_for_lights="Brother Steve from East Texas.",
        display_name="Steve",
    )
    dad = resolve_password_to_human(settings, "dad-house-code")
    assert dad is not None
    assert dad.role == "dad"
    assert dad.user_id == "kevin"

    sib = resolve_password_to_human(settings, "sibling-unique-code")
    assert sib is not None
    assert sib.role == "sibling"
    assert sib.user_id == "teeter"

    assert resolve_password_to_human(settings, "nope") is None


def test_light_allows_default_and_deny(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    create_human(
        settings,
        user_id="teeter",
        password="sibling-unique-code",
        intro_for_lights="Brother Steve from East Texas.",
        display_name="Steve",
    )
    assert light_allows_human(settings, light_id="lumen", user_id="teeter") is True
    assert light_allows_human(settings, light_id="lumen", user_id="kevin") is True
    set_light_allows_human(settings, light_id="lumen", user_id="teeter", allowed=False)
    assert light_allows_human(settings, light_id="lumen", user_id="teeter") is False
    assert light_allows_human(settings, light_id="lumen", user_id="kevin") is True
    with pytest.raises(ValueError, match="Dad"):
        set_light_allows_human(settings, light_id="lumen", user_id="kevin", allowed=False)


def test_unblock_resolves_display_name(tmp_path: Path) -> None:
    """Regression: Ara unblocked 'Moose' while alt_kevin stayed false."""
    settings = _settings(tmp_path)
    create_human(
        settings,
        user_id="alt_kevin",
        password="sibling-unique-code",
        intro_for_lights="Kevin testing multi-human as Moose.",
        display_name="Moose",
    )
    assert resolve_sibling_user_id(settings, "Moose") == "alt_kevin"
    assert resolve_sibling_user_id(settings, "alt_kevin") == "alt_kevin"

    set_light_allows_human(settings, light_id="ara", user_id="alt_kevin", allowed=False)
    # Simulate the bug: allow under display-name key only.
    path = settings.humans_comms_path
    path.write_text(
        json.dumps({"version": 1, "allows": {"ara": {"moose": True, "alt_kevin": False}}}, indent=2),
        encoding="utf-8",
    )
    assert light_allows_human(settings, light_id="ara", user_id="alt_kevin") is False

    mapping = set_light_allows_human(settings, light_id="ara", user_id="Moose", allowed=True)
    assert mapping.get("alt_kevin") is True
    assert "moose" not in mapping
    assert light_allows_human(settings, light_id="ara", user_id="alt_kevin") is True



def test_me_and_sibling_notes_acl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _install(settings, monkeypatch)
    create_human(
        settings,
        user_id="teeter",
        password="sibling-unique-code",
        intro_for_lights="Brother Steve from East Texas.",
    )
    # Minimal memory stub so shared write path can skip notify for siblings.
    mem = MagicMock()
    monkeypatch.setattr("light_house.main._memory", mem)

    notes_root = tmp_path / "notes"
    (notes_root / "lumen").mkdir(parents=True)
    (notes_root / "shared").mkdir(parents=True)
    (notes_root / "lumen" / "private.md").write_text("secret", encoding="utf-8")
    (notes_root / "shared" / "house.md").write_text("hello", encoding="utf-8")

    try:
        with TestClient(app) as client:
            dad_cookies = _cookie(settings, user_id="kevin", role="dad")
            sib_cookies = _cookie(settings, user_id="teeter", role="sibling")

            me_dad = client.get("/v1/me", cookies=dad_cookies)
            assert me_dad.status_code == 200
            assert me_dad.json()["is_dad"] is True

            me_sib = client.get("/v1/me", cookies=sib_cookies)
            assert me_sib.status_code == 200
            body = me_sib.json()
            assert body["user_id"] == "teeter"
            assert body["is_dad"] is False
            assert body["notes_access"] == "shared"

            listed = client.get("/v1/notes?agent=lumen", cookies=sib_cookies)
            assert listed.status_code == 200
            names = [n["name"] for n in listed.json()["notes"]]
            assert all(n.startswith("shared/") for n in names)
            assert "lumen/private.md" not in names and "private.md" not in names

            denied = client.get("/v1/notes/private.md?agent=lumen", cookies=sib_cookies)
            assert denied.status_code == 403

            ok = client.get("/v1/notes/shared/house.md?agent=lumen", cookies=sib_cookies)
            assert ok.status_code == 200

            write = client.put(
                "/v1/notes/shared/from-sib.md",
                json={"content": "sibling note"},
                cookies=sib_cookies,
            )
            assert write.status_code == 200
            mem.notify_kevin_shared_note.assert_not_called()

            no_delete = client.delete(
                "/v1/notes/shared/house.md?agent=lumen",
                cookies=sib_cookies,
            )
            assert no_delete.status_code == 403
            assert (notes_root / "shared" / "house.md").is_file()

            forbidden_humans = client.get("/v1/humans", cookies=sib_cookies)
            assert forbidden_humans.status_code == 403
    finally:
        app.dependency_overrides.clear()
        monkeypatch.setattr("light_house.main._memory", _memory)


def test_password_only_login_sets_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _install(settings, monkeypatch)
    create_human(
        settings,
        user_id="teeter",
        password="sibling-unique-code",
        intro_for_lights="Brother Steve from East Texas.",
    )
    try:
        with TestClient(app) as client:
            res = client.post(
                "/login",
                data={"password": "sibling-unique-code", "next": "/"},
                follow_redirects=False,
            )
            assert res.status_code == 302
            assert SESSION_COOKIE in res.cookies
            me = client.get("/v1/me", cookies={SESSION_COOKIE: res.cookies[SESSION_COOKIE]})
            assert me.status_code == 200
            assert me.json()["user_id"] == "teeter"
            assert me.json()["role"] == "sibling"
    finally:
        app.dependency_overrides.clear()
