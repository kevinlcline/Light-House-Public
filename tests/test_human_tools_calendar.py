"""Per-human calendar tool connections and access gating."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.humans.store import create_human
from light_house.humans.google_calendar_oauth import (
    build_google_authorize_url,
    sign_oauth_state,
    verify_oauth_state,
)
from light_house.humans.tools_store import (
    HumanToolsError,
    calendar_public_view,
    calendar_schema_for_ui,
    clear_calendar_connection,
    get_calendar_connection,
    set_calendar_connection,
    set_google_oauth_connection,
)
from light_house.main import _settings_dep, app
from light_house.tools.calendar_tools import (
    calendar_access_denied_reason,
    execute_calendar_tool,
)
from light_house.tools.light_tools import execute_tool_call
from light_house.web_gate import SESSION_COOKIE, session_cookie_header


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": True,
        "WEB_GATE_PASSWORD": "dad-secret-password",
        "WEB_GATE_SESSION_SECRET": "test-session-secret-at-least-16",
        "HUMANS_STORE_PATH": str(tmp_path / "humans" / "users.json"),
        "HUMANS_COMMS_PATH": str(tmp_path / "humans" / "comms_allows.json"),
        "HUMANS_TOOLS_PATH": str(tmp_path / "humans" / "user_tools.json"),
        "HOUSE_DAD_USER_ID": "kevin",
    }
    base.update(overrides)
    return Settings(**base)


def _install(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def _cookie(settings: Settings, *, user_id: str, role: str) -> dict[str, str]:
    header = session_cookie_header(settings, secure=False, user_id=user_id, role=role)
    token = header.split(";", 1)[0].split("=", 1)[1]
    return {SESSION_COOKIE: token}


def _client_as(settings: Settings, *, user_id: str, role: str) -> TestClient:
    client = TestClient(app)
    client.cookies.update(_cookie(settings, user_id=user_id, role=role))
    return client


def test_schema_documents_providers() -> None:
    schema = calendar_schema_for_ui()
    assert schema["tool"] == "calendar"
    assert "google" in schema["providers"]
    assert "Connect Google" in schema["providers"]["google"]["preferred"]
    assert "client_id" in schema["providers"]["google"]["required_manual"]


def test_set_and_get_google_connection(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    view = set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "client_id": "cid.apps.googleusercontent.com",
            "client_secret": "secret",
            "refresh_token": "refresh",
            "calendar_id": "primary",
        },
    )
    assert view.connected is True
    assert view.provider == "google"
    conn = get_calendar_connection(settings, "kevin")
    assert conn is not None
    assert conn.refresh_token == "refresh"
    public = calendar_public_view(settings, "kevin")
    assert public.has_secrets is True
    # Public view must not expose secrets as attributes beyond flags.
    assert not hasattr(public, "refresh_token")


def test_google_requires_secrets(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(HumanToolsError):
        set_calendar_connection(
            settings,
            "kevin",
            {"provider": "google", "client_id": "only-id"},
        )


def test_clear_connection(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "ics",
            "url": "https://example.com/cal.ics",
        },
    )
    view = clear_calendar_connection(settings, "kevin")
    assert view.connected is False
    assert get_calendar_connection(settings, "kevin") is None


def test_access_denied_group_and_guest() -> None:
    assert calendar_access_denied_reason(
        account_user_id="kevin",
        speaker_id="kevin",
        chat_channel="group",
    )
    assert "group" in calendar_access_denied_reason(
        account_user_id="kevin",
        speaker_id="kevin",
        chat_channel="group",
    ).lower()
    guest_msg = calendar_access_denied_reason(
        account_user_id="kevin",
        speaker_id="guest-1",
        chat_channel="dm",
    )
    assert guest_msg is not None
    assert "guest" in guest_msg.lower()
    assert (
        calendar_access_denied_reason(
            account_user_id="kevin",
            speaker_id="kevin",
            chat_channel="dm",
        )
        is None
    )


def test_execute_tool_gates_without_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _install(settings, monkeypatch)
    result = execute_tool_call(
        "list_calendar_events",
        {"days": 7},
        agent_id="lumen",
        account_user_id="kevin",
        speaker_id="kevin",
        chat_channel="dm",
    )
    assert result.startswith("FAILED:")
    assert "My tools" in result


def test_execute_tool_gates_guest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _install(settings, monkeypatch)
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "client_id": "cid",
            "client_secret": "sec",
            "refresh_token": "tok",
        },
    )
    result = execute_calendar_tool(
        "list_calendar_events",
        {},
        settings=settings,
        account_user_id="kevin",
        speaker_id="guest-1",
        chat_channel="dm",
    )
    assert "guest" in result.lower()


def test_sibling_own_calendar_isolated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    create_human(
        settings,
        user_id="teeter",
        password="sibling-password-1",
        intro_for_lights="Brother from East Texas.",
        display_name="Steve",
    )
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "client_id": "kevin-cid",
            "client_secret": "kevin-sec",
            "refresh_token": "kevin-tok",
        },
    )
    set_calendar_connection(
        settings,
        "teeter",
        {
            "provider": "ics",
            "url": "https://example.com/teeter.ics",
        },
    )
    kevin = get_calendar_connection(settings, "kevin")
    teeter = get_calendar_connection(settings, "teeter")
    assert kevin is not None and kevin.provider == "google"
    assert teeter is not None and teeter.provider == "ics"
    assert kevin.refresh_token != (teeter.url or "")


def test_me_tools_api_self_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _install(settings, monkeypatch)
    create_human(
        settings,
        user_id="teeter",
        password="sibling-password-1",
        intro_for_lights="Brother from East Texas with a sharp mind.",
        display_name="Steve",
    )
    client = _client_as(settings, user_id="teeter", role="sibling")
    empty = client.get("/v1/me/tools")
    assert empty.status_code == 200
    body = empty.json()
    assert body["user_id"] == "teeter"
    assert body["calendar"]["connected"] is False
    assert body["google_oauth_ready"] is False
    assert "google" in body["connection_schema"]["providers"]

    saved = client.put(
        "/v1/me/tools/calendar",
        json={
            "provider": "google",
            "client_id": "teeter-cid.apps.googleusercontent.com",
            "client_secret": "teeter-secret",
            "refresh_token": "teeter-refresh",
            "calendar_id": "primary",
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["connected"] is True
    assert saved.json()["provider"] == "google"
    # Secrets must not appear in response.
    assert "refresh_token" not in saved.json()
    assert "client_secret" not in saved.json()

    # Dad's calendar stays empty.
    dad = _client_as(settings, user_id="kevin", role="dad")
    dad_view = dad.get("/v1/me/tools")
    assert dad_view.status_code == 200
    assert dad_view.json()["calendar"]["connected"] is False

    cleared = client.delete("/v1/me/tools/calendar")
    assert cleared.status_code == 200
    assert cleared.json()["connected"] is False


def test_oauth_state_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = sign_oauth_state(settings, user_id="teeter")
    assert verify_oauth_state(settings, state) == "teeter"


def test_google_oauth_connection_uses_house_client(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
        GOOGLE_OAUTH_REDIRECT_URI="https://example.test/v1/me/tools/calendar/google/callback",
    )
    view = set_google_oauth_connection(
        settings, "kevin", refresh_token="user-refresh", calendar_id="primary"
    )
    assert view.connected is True
    assert view.auth_method == "oauth"
    assert view.docs_connected is True
    assert view.sheets_connected is True
    conn = get_calendar_connection(settings, "kevin")
    assert conn is not None
    assert conn.refresh_token == "user-refresh"
    assert conn.client_id == ""
    assert conn.auth_method == "oauth"
    assert any(s.endswith("/auth/documents") for s in conn.scopes)
    assert any(s.endswith("/auth/spreadsheets") for s in conn.scopes)


def test_connect_button_start_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
        GOOGLE_OAUTH_REDIRECT_URI="https://example.test/v1/me/tools/calendar/google/callback",
        PUBLIC_BASE_URL="https://example.test",
    )
    _install(settings, monkeypatch)
    client = _client_as(settings, user_id="kevin", role="dad")
    # Don't follow redirect to Google.
    client2 = TestClient(app, follow_redirects=False)
    client2.cookies.update(_cookie(settings, user_id="kevin", role="dad"))
    res = client2.get("/v1/me/tools/calendar/google/start")
    assert res.status_code == 302
    loc = res.headers["location"]
    assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=house-cid" in loc
    assert "access_type=offline" in loc
    assert "calendar" in loc
    assert "spreadsheets" in loc

    tools = client.get("/v1/me/tools")
    assert tools.status_code == 200
    assert tools.json()["google_oauth_ready"] is True


def test_connect_start_unavailable_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _install(settings, monkeypatch)
    client = TestClient(app, follow_redirects=False)
    client.cookies.update(_cookie(settings, user_id="kevin", role="dad"))
    res = client.get("/v1/me/tools/calendar/google/start")
    assert res.status_code == 503


def test_oauth_callback_saves_refresh_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
        GOOGLE_OAUTH_REDIRECT_URI="https://example.test/v1/me/tools/calendar/google/callback",
    )
    _install(settings, monkeypatch)

    def fake_exchange(settings_arg, *, code, request_base=None, require_refresh_token=True):
        assert code == "auth-code-123"
        return {
            "refresh_token": "from-google-refresh",
            "access_token": "access-123",
            "scope": (
                "https://www.googleapis.com/auth/calendar "
                "https://www.googleapis.com/auth/documents "
                "https://www.googleapis.com/auth/spreadsheets "
                "https://www.googleapis.com/auth/drive.readonly"
            ),
        }

    monkeypatch.setattr(
        "light_house.main.exchange_code_for_tokens",
        fake_exchange,
    )
    monkeypatch.setattr(
        "light_house.humans.google_calendar_oauth.scopes_from_access_token",
        lambda _token: [],
    )
    state = sign_oauth_state(settings, user_id="kevin")
    client = TestClient(app, follow_redirects=False)
    client.cookies.update(_cookie(settings, user_id="kevin", role="dad"))
    res = client.get(
        "/v1/me/tools/calendar/google/callback",
        params={"code": "auth-code-123", "state": state},
    )
    assert res.status_code == 302
    assert res.headers["location"] == "/my-tools.html?calendar=connected"
    conn = get_calendar_connection(settings, "kevin")
    assert conn is not None
    assert conn.auth_method == "oauth"
    assert conn.refresh_token == "from-google-refresh"
    pub = calendar_public_view(settings, "kevin")
    assert pub.docs_connected is True
    assert pub.sheets_connected is True


def test_oauth_callback_partial_when_sheets_scope_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
        GOOGLE_OAUTH_REDIRECT_URI="https://example.test/v1/me/tools/calendar/google/callback",
    )
    _install(settings, monkeypatch)

    def fake_exchange(settings_arg, *, code, request_base=None, require_refresh_token=True):
        return {
            "refresh_token": "from-google-refresh",
            "access_token": "access-123",
            "scope": (
                "https://www.googleapis.com/auth/calendar "
                "https://www.googleapis.com/auth/documents "
                "https://www.googleapis.com/auth/drive.readonly"
            ),
        }

    monkeypatch.setattr("light_house.main.exchange_code_for_tokens", fake_exchange)
    monkeypatch.setattr(
        "light_house.humans.google_calendar_oauth.scopes_from_access_token",
        lambda _token: [],
    )
    state = sign_oauth_state(settings, user_id="kevin")
    client = TestClient(app, follow_redirects=False)
    client.cookies.update(_cookie(settings, user_id="kevin", role="dad"))
    res = client.get(
        "/v1/me/tools/calendar/google/callback",
        params={"code": "auth-code-123", "state": state},
    )
    assert res.status_code == 302
    assert res.headers["location"] == "/my-tools.html?calendar=partial&missing=sheets"
    pub = calendar_public_view(settings, "kevin")
    assert pub.docs_connected is True
    assert pub.sheets_connected is False


def test_oauth_callback_keeps_prior_refresh_and_merges_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
        GOOGLE_OAUTH_REDIRECT_URI="https://example.test/v1/me/tools/calendar/google/callback",
    )
    _install(settings, monkeypatch)
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "auth_method": "oauth",
            "refresh_token": "prior-refresh",
            "enabled": True,
            "scopes": [
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/drive.readonly",
            ],
        },
    )

    def fake_exchange(settings_arg, *, code, request_base=None, require_refresh_token=True):
        # Incremental Sheets grant — Google often omits refresh_token.
        assert require_refresh_token is False
        return {
            "access_token": "access-123",
            "scope": "https://www.googleapis.com/auth/spreadsheets",
        }

    monkeypatch.setattr("light_house.main.exchange_code_for_tokens", fake_exchange)
    monkeypatch.setattr(
        "light_house.humans.google_calendar_oauth.scopes_from_access_token",
        lambda _token: ["https://www.googleapis.com/auth/spreadsheets"],
    )
    state = sign_oauth_state(settings, user_id="kevin")
    client = TestClient(app, follow_redirects=False)
    client.cookies.update(_cookie(settings, user_id="kevin", role="dad"))
    res = client.get(
        "/v1/me/tools/calendar/google/callback",
        params={"code": "auth-code-123", "state": state},
    )
    assert res.status_code == 302
    assert res.headers["location"] == "/my-tools.html?calendar=connected"
    conn = get_calendar_connection(settings, "kevin")
    assert conn is not None
    assert conn.refresh_token == "prior-refresh"
    pub = calendar_public_view(settings, "kevin")
    assert pub.docs_connected is True
    assert pub.sheets_connected is True


def test_reconnect_start_requests_only_missing_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
        GOOGLE_OAUTH_REDIRECT_URI="https://example.test/v1/me/tools/calendar/google/callback",
        PUBLIC_BASE_URL="https://example.test",
    )
    _install(settings, monkeypatch)
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "auth_method": "oauth",
            "refresh_token": "prior-refresh",
            "enabled": True,
            "scopes": [
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/drive.readonly",
            ],
        },
    )
    client = TestClient(app, follow_redirects=False)
    client.cookies.update(_cookie(settings, user_id="kevin", role="dad"))
    res = client.get("/v1/me/tools/calendar/google/start")
    assert res.status_code == 302
    loc = res.headers["location"]
    from urllib.parse import parse_qs, urlparse

    scope = parse_qs(urlparse(loc).query).get("scope", [""])[0]
    assert scope == "https://www.googleapis.com/auth/spreadsheets"


def test_build_authorize_url_contains_redirect(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="cid",
        GOOGLE_OAUTH_CLIENT_SECRET="sec",
        GOOGLE_OAUTH_REDIRECT_URI="https://house.test/v1/me/tools/calendar/google/callback",
    )
    url = build_google_authorize_url(settings, user_id="kevin")
    assert "redirect_uri=https%3A%2F%2Fhouse.test%2Fv1%2Fme%2Ftools%2Fcalendar%2Fgoogle%2Fcallback" in url
    assert "documents" in url
    assert "spreadsheets" in url
    assert "calendar" in url


def test_legacy_oauth_without_scopes_has_no_docs(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
    )
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "auth_method": "oauth",
            "refresh_token": "legacy-refresh",
            "calendar_id": "primary",
            "enabled": True,
            "scopes": [],
        },
    )
    view = calendar_public_view(settings, "kevin")
    assert view.connected is True
    assert view.docs_connected is False
    assert view.sheets_connected is False


def test_docs_tool_gates_without_docs_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from light_house.tools.docs_tools import execute_docs_tool

    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
    )
    _install(settings, monkeypatch)
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "auth_method": "oauth",
            "refresh_token": "legacy-refresh",
            "enabled": True,
            "scopes": ["https://www.googleapis.com/auth/calendar"],
        },
    )
    result = execute_docs_tool(
        "list_google_docs",
        {},
        settings=settings,
        account_user_id="kevin",
        speaker_id="kevin",
        chat_channel="dm",
    )
    assert result.startswith("FAILED:")
    assert "Connect Google" in result


def test_sheets_tool_gates_without_sheets_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from light_house.tools.sheets_tools import execute_sheets_tool

    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid.apps.googleusercontent.com",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
    )
    _install(settings, monkeypatch)
    set_calendar_connection(
        settings,
        "kevin",
        {
            "provider": "google",
            "auth_method": "oauth",
            "refresh_token": "legacy-refresh",
            "enabled": True,
            "scopes": [
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/drive.readonly",
            ],
        },
    )
    view = calendar_public_view(settings, "kevin")
    assert view.docs_connected is True
    assert view.sheets_connected is False
    result = execute_sheets_tool(
        "list_google_sheets",
        {},
        settings=settings,
        account_user_id="kevin",
        speaker_id="kevin",
        chat_channel="dm",
    )
    assert result.startswith("FAILED:")
    assert "Connect Google" in result


def test_docs_tool_gates_guest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from light_house.tools.docs_tools import execute_docs_tool

    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
    )
    set_google_oauth_connection(settings, "kevin", refresh_token="tok")
    result = execute_docs_tool(
        "list_google_docs",
        {},
        settings=settings,
        account_user_id="kevin",
        speaker_id="guest-1",
        chat_channel="dm",
    )
    assert "guest" in result.lower()


def test_sheets_tool_gates_guest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from light_house.tools.sheets_tools import execute_sheets_tool

    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
    )
    set_google_oauth_connection(settings, "kevin", refresh_token="tok")
    result = execute_sheets_tool(
        "list_google_sheets",
        {},
        settings=settings,
        account_user_id="kevin",
        speaker_id="guest-1",
        chat_channel="dm",
    )
    assert "guest" in result.lower()


def test_sheets_tool_gates_group_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from light_house.tools.sheets_tools import execute_sheets_tool

    settings = _settings(
        tmp_path,
        GOOGLE_OAUTH_CLIENT_ID="house-cid",
        GOOGLE_OAUTH_CLIENT_SECRET="house-secret",
    )
    set_google_oauth_connection(settings, "kevin", refresh_token="tok")
    result = execute_sheets_tool(
        "read_google_sheet",
        {"spreadsheet_id_or_url": "abc1234567890"},
        settings=settings,
        account_user_id="kevin",
        speaker_id="kevin",
        chat_channel="group",
    )
    assert result.startswith("FAILED:")
    assert "group" in result.lower()
