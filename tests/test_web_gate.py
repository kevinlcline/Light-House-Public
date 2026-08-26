"""Web gate: password login and protected routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.main import _settings_dep, app
from light_house.web_gate import SESSION_COOKIE, session_cookie_header

_LUMEN_ONLY_MANIFEST = """\
version: 1
primary_light_id: lumen
lights:
  - id: lumen
    display_name: Lumen
    thread_id: lumen-home
    enabled: true
    persona_file: lumen_system.md
    notes_dir: lumen
    inner_life: true
    dreams: true
    report_back: false
    voice_id: af_heart
"""


def _gate_settings(tmp_path: Path, **overrides: object) -> Settings:
    manifest = tmp_path / "lights.yaml"
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": True,
        "WEB_GATE_PASSWORD": "test-secret-pass",
        "WEB_GATE_SESSION_SECRET": "test-session-secret-key",
        "WEB_GATE_SESSION_DAYS": 1,
        "ARA_ENABLED": False,
        "PRIMARY_LLM": "openrouter",
        "OPENROUTER_API_KEY": "test-openrouter-key",
        "LIGHTS_MANIFEST_PATH": str(manifest),
    }
    base.update(overrides)
    settings = Settings(**base)
    if not Path(settings.lights_manifest_path).is_file():
        Path(settings.lights_manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(settings.lights_manifest_path).write_text(_LUMEN_ONLY_MANIFEST, encoding="utf-8")
    return settings


def _install_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.config.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_gate_disabled_allows_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _gate_settings(tmp_path, WEB_GATE_ENABLED=False)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.get("/v1/agents")
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_gate_enabled_blocks_api_without_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.get("/v1/agents")
        assert response.status_code == 401
        assert response.json()["detail"] == "Authentication required"
    finally:
        app.dependency_overrides.clear()


def test_gate_enabled_shows_landing_at_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.get("/")
        assert response.status_code == 200
        assert "Light-House" in response.text
        assert "Enter" in response.text
        assert "no-store" in response.headers.get("cache-control", "")
        head = client.head("/")
        assert head.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_gate_enabled_serves_robots_and_llms_without_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            robots = client.get("/robots.txt")
            assert robots.status_code == 200
            assert "Google-Extended" in robots.text
            assert "Allow: /" in robots.text
            assert "Disallow: /" not in robots.text
            assert "no-store" in robots.headers.get("cache-control", "")

            llms = client.get("/llms.txt")
            assert llms.status_code == 200
            assert "Light-House" in llms.text
            assert "free" in llms.text.lower()
    finally:
        app.dependency_overrides.clear()


def test_gate_enabled_redirects_protected_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.get("/notes.html", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"].startswith("/login?next=")
    finally:
        app.dependency_overrides.clear()


def test_login_wrong_password(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/login",
                data={"password": "wrong", "next": "/notes.html"},
                follow_redirects=False,
            )
        assert response.status_code == 302
        assert "error=1" in response.headers["location"]
        assert SESSION_COOKIE not in response.cookies
    finally:
        app.dependency_overrides.clear()


def test_login_success_grants_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            login = client.post(
                "/login",
                data={"password": "test-secret-pass", "next": "/"},
                follow_redirects=False,
            )
            assert login.status_code == 302
            assert login.headers["location"] == "/"
            assert SESSION_COOKIE in login.cookies
            assert "no-store" in login.headers.get("cache-control", "")

            authed = client.get("/v1/agents")
            assert authed.status_code == 200

            chat = client.get("/")
            assert chat.status_code == 200
            assert "no-store" in chat.headers.get("cache-control", "")
    finally:
        app.dependency_overrides.clear()


def test_logout_clears_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            client.post(
                "/login",
                data={"password": "test-secret-pass", "next": "/"},
                follow_redirects=True,
            )
            logout = client.post("/logout", follow_redirects=False)
            assert logout.status_code == 302

            blocked = client.get("/v1/agents")
            assert blocked.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_settings_reject_gate_without_password() -> None:
    with pytest.raises(ValueError, match="WEB_GATE_PASSWORD"):
        Settings(
            _env_file=None,
            WEB_GATE_ENABLED=True,
            WEB_GATE_SESSION_SECRET="secret",
        )


def test_settings_reject_gate_without_secret() -> None:
    with pytest.raises(ValueError, match="WEB_GATE_SESSION_SECRET"):
        Settings(
            _env_file=None,
            WEB_GATE_ENABLED=True,
            WEB_GATE_PASSWORD="pass",
        )


def test_session_cookie_header_format() -> None:
    settings = Settings(
        _env_file=None,
        WEB_GATE_ENABLED=True,
        WEB_GATE_PASSWORD="pass",
        WEB_GATE_SESSION_SECRET="secret",
        WEB_GATE_SESSION_DAYS=7,
    )
    header = session_cookie_header(settings, secure=True, user_id="kevin", role="dad")
    assert "HttpOnly" in header
    assert "Secure" in header
    assert f"{SESSION_COOKIE}=" in header
