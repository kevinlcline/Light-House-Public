"""Human user store and setup API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.humans.comms import light_allows_human, set_light_allows_human
from light_house.humans.store import (
    DEFAULT_DAD_VOICE_ID,
    DEFAULT_SIBLING_VOICE_ID,
    HumansError,
    authenticate_human,
    create_human,
    delete_human,
    get_dad_voice_id,
    intro_text_for_lights,
    list_human_voices,
    list_humans,
    set_dad_voice_id,
    update_human,
    validate_user_id,
    verify_password,
    voice_id_for_human,
)
from light_house.main import _settings_dep, app


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


def _install(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_validate_user_id() -> None:
    assert validate_user_id("teeter") == "teeter"
    assert validate_user_id("Steve_1") == "steve_1"
    with pytest.raises(HumansError):
        validate_user_id("kevin")
    with pytest.raises(HumansError):
        validate_user_id("1bad")


def test_create_and_authenticate(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    intro = (
        "This is my brother Steve. I call him Teeter because I always have. "
        "He lives in East Texas with his wife. He never finished highschool, "
        "but I'm sure you will notice he is a sharp mind."
    )
    user = create_human(
        settings,
        user_id="teeter",
        password="east-texas-secret",
        intro_for_lights=intro,
        display_name="Steve",
    )
    assert user.user_id == "teeter"
    assert user.display_name == "Steve"
    assert user.notes_access == "shared"
    assert "East Texas" in user.intro_for_lights

    assert authenticate_human(settings, "teeter", "east-texas-secret") is not None
    assert authenticate_human(settings, "teeter", "wrong") is None
    assert intro_text_for_lights(settings, "teeter").startswith("This is my brother")

    listed = list_humans(settings)
    assert len(listed) == 1
    assert listed[0].user_id == "teeter"

    with pytest.raises(HumansError, match="already exists"):
        create_human(
            settings,
            user_id="teeter",
            password="another-password",
            intro_for_lights=intro,
        )


def test_update_and_delete_human(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    create_human(
        settings,
        user_id="teeter",
        password="east-texas-secret",
        intro_for_lights="Brother Steve from East Texas.",
        display_name="Steve",
    )
    updated = update_human(
        settings,
        user_id="teeter",
        display_name="Teeter",
        intro_for_lights="Updated intro for the lights about Teeter.",
        enabled=False,
    )
    assert updated.display_name == "Teeter"
    assert "Updated intro" in updated.intro_for_lights
    assert updated.enabled is False
    assert authenticate_human(settings, "teeter", "east-texas-secret") is None

    update_human(settings, user_id="teeter", enabled=True, password="brand-new-secret")
    assert authenticate_human(settings, "teeter", "east-texas-secret") is None
    assert authenticate_human(settings, "teeter", "brand-new-secret") is not None

    deleted = delete_human(settings, user_id="teeter")
    assert deleted.user_id == "teeter"
    assert list_humans(settings) == []
    with pytest.raises(HumansError, match="Unknown user"):
        delete_human(settings, user_id="teeter")


def test_password_not_stored_plaintext(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    create_human(
        settings,
        user_id="teeter",
        password="east-texas-secret",
        intro_for_lights="Brother Steve / Teeter from East Texas.",
    )
    raw = (tmp_path / "humans" / "users.json").read_text(encoding="utf-8")
    assert "east-texas-secret" not in raw
    assert "scrypt$" in raw
    assert verify_password("east-texas-secret", __import__("json").loads(raw)["users"]["teeter"]["password_hash"])


def test_humans_api_create_and_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _install(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            page = client.get("/user-setup.html")
            assert page.status_code == 200
            assert b"Introduce them to the lights" in page.content
            assert b"Existing siblings" in page.content

            empty = client.get("/v1/humans")
            assert empty.status_code == 200
            assert empty.json()["users"] == []

            bad = client.post(
                "/v1/humans",
                json={
                    "user_id": "teeter",
                    "password": "short",
                    "intro_for_lights": "Too short password should fail.",
                },
            )
            assert bad.status_code == 422

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
            body = created.json()
            assert body["user_id"] == "teeter"
            assert body["display_name"] == "Steve"
            assert body["voice_id"] == DEFAULT_SIBLING_VOICE_ID
            assert "password" not in body
            assert "password_hash" not in body

            listed = client.get("/v1/humans")
            assert listed.status_code == 200
            assert len(listed.json()["users"]) == 1

            one = client.get("/v1/humans/teeter")
            assert one.status_code == 200
            assert "East Texas" in one.json()["intro_for_lights"]

            patched = client.patch(
                "/v1/humans/teeter",
                json={
                    "display_name": "Teeter",
                    "intro_for_lights": "Teeter lives in East Texas.",
                },
            )
            assert patched.status_code == 200
            assert patched.json()["display_name"] == "Teeter"

            set_light_allows_human(settings, light_id="ara", user_id="teeter", allowed=False)
            assert light_allows_human(settings, light_id="ara", user_id="teeter") is False

            deleted = client.delete("/v1/humans/teeter")
            assert deleted.status_code == 200
            assert deleted.json()["user_id"] == "teeter"
            assert client.get("/v1/humans").json()["users"] == []
            assert light_allows_human(settings, light_id="ara", user_id="teeter") is True
    finally:
        app.dependency_overrides.clear()


def test_human_and_dad_voices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, HOUSE_DAD_USER_ID="kevin")
    assert get_dad_voice_id(settings) == DEFAULT_DAD_VOICE_ID
    assert set_dad_voice_id(settings, "am_fenrir") == "am_fenrir"
    assert voice_id_for_human(settings, "kevin") == "am_fenrir"

    user = create_human(
        settings,
        user_id="teeter",
        password="east-texas-secret",
        intro_for_lights="Brother Steve from East Texas.",
        display_name="Steve",
        voice_id="bm_george",
    )
    assert user.voice_id == "bm_george"
    updated = update_human(settings, user_id="teeter", voice_id="af_sky")
    assert updated.voice_id == "af_sky"
    assert voice_id_for_human(settings, "teeter") == "af_sky"

    voices = list_human_voices(settings)
    by_id = {row["user_id"]: row for row in voices}
    assert by_id["kevin"]["voice_id"] == "am_fenrir"
    assert by_id["kevin"]["is_dad"] is True
    assert by_id["teeter"]["voice_id"] == "af_sky"

    _install(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            me = client.get("/v1/me")
            assert me.status_code == 200
            assert me.json()["voice_id"] == "am_fenrir"

            mapped = client.get("/v1/humans/voices")
            assert mapped.status_code == 200
            ids = {row["user_id"] for row in mapped.json()["voices"]}
            assert "kevin" in ids
            assert "teeter" in ids

            dad_patch = client.patch("/v1/humans/dad", json={"voice_id": "am_adam"})
            assert dad_patch.status_code == 200
            assert dad_patch.json()["voice_id"] == "am_adam"

            sib_patch = client.patch(
                "/v1/humans/teeter",
                json={"voice_id": "am_liam"},
            )
            assert sib_patch.status_code == 200
            assert sib_patch.json()["voice_id"] == "am_liam"
    finally:
        app.dependency_overrides.clear()
