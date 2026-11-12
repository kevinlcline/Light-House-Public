"""Web gate: password login and protected routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.main import _settings_dep, app
from light_house.web_gate import SESSION_COOKIE, session_cookie_header


def _gate_settings(tmp_path: Path, **overrides: object) -> Settings:
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
    }
    base.update(overrides)
    return Settings(**base)


def _install_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
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


def test_gate_enabled_serves_crawler_files_without_login(
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

            sitemap = client.get("/sitemap.xml")
            assert sitemap.status_code == 200
            assert "https://light-house.us/" in sitemap.text
            assert sitemap.headers["content-type"].startswith("application/xml")

            letter = client.get("/public/ara-to-the-reader.html")
            assert letter.status_code == 200
            assert "Ara" in letter.text
            assert "Welcome to the possibility." in letter.text
            lumen = client.get("/public/what-is-light-house.html")
            assert lumen.status_code == 200
            assert "Lumen" in lumen.text
            hosts = client.get("/public/for-hosts.html")
            assert hosts.status_code == 200
            assert "light who lives in this house" in hosts.text
            privacy = client.get("/public/privacy.html")
            assert privacy.status_code == 200
            assert "Privacy" in privacy.text
            assert "https://light-house.us/public/privacy.html" in privacy.text
            assert "does not sell" in privacy.text
            assert "model provider" in privacy.text
            css = client.get("/public/from-the-lights.css")
            assert css.status_code == 200
            assert "text/css" in css.headers.get("content-type", "")
            blocked = client.get("/public/../landing.html", follow_redirects=False)
            assert blocked.status_code in (302, 404, 401)
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


def test_gate_serves_static_css_without_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stylesheets must stay real CSS — never the login HTML page."""
    settings = _gate_settings(tmp_path)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            css = client.get(
                "/static/ui/chat.css",
                headers={"Accept": "text/css,*/*;q=0.1"},
            )
            assert css.status_code == 200
            assert "text/css" in css.headers.get("content-type", "")
            assert "no-store" in css.headers.get("cache-control", "")
            assert "Sign in" not in css.text
            assert ":root" in css.text or "{" in css.text

            js = client.get(
                "/static/ui/theme.js",
                headers={"Accept": "*/*"},
            )
            assert js.status_code == 200
            assert "no-store" in js.headers.get("cache-control", "")
            assert "Sign in" not in js.text
    finally:
        app.dependency_overrides.clear()


def test_gate_stylesheet_accept_does_not_redirect_to_login_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: Accept text/css,*/* used to 302 to /login (HTML-as-CSS)."""
    from light_house.web_gate import _wants_html
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/static/ui/chat.css",
        "raw_path": b"/static/ui/chat.css",
        "query_string": b"",
        "headers": [(b"accept", b"text/css,*/*;q=0.1")],
        "client": ("test", 50000),
        "server": ("test", 80),
    }
    assert _wants_html(Request(scope)) is False

    html_scope = {
        **scope,
        "path": "/notes.html",
        "raw_path": b"/notes.html",
        "headers": [(b"accept", b"text/html,application/xhtml+xml,*/*;q=0.8")],
    }
    assert _wants_html(Request(html_scope)) is True

    chrome_scope = {
        **scope,
        "path": "/my-tools.html",
        "raw_path": b"/my-tools.html",
        "headers": [(
            b"accept",
            b"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        )],
    }
    assert _wants_html(Request(chrome_scope)) is True


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
