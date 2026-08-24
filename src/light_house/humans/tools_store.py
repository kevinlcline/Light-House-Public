"""Per-human tool connections (calendars first) — self-serve, not admin-managed."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from light_house.config import Settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

CalendarProvider = Literal["google", "caldav", "ics"]
CALENDAR_PROVIDERS = frozenset({"google", "caldav", "ics"})

# Secrets never leave the API in clear text once stored.
_SECRET_KEYS = frozenset({"client_secret", "refresh_token", "password"})


class HumanToolsError(ValueError):
    """Invalid human tool connection."""


@dataclass(frozen=True)
class CalendarConnectionPublic:
    """Safe view for UI / lights (no secret values)."""

    connected: bool
    provider: str | None
    calendar_id: str | None
    enabled: bool
    has_secrets: bool
    updated_at: float | None
    auth_method: str | None = None  # "oauth" | "manual" | None
    docs_connected: bool = False
    sheets_connected: bool = False
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalendarConnection:
    """Full calendar / Google credentials for tool execution."""

    provider: CalendarProvider
    enabled: bool
    client_id: str
    client_secret: str
    refresh_token: str
    calendar_id: str
    url: str
    username: str
    password: str
    updated_at: float
    auth_method: str = "manual"  # "oauth" | "manual"
    scopes: tuple[str, ...] = ()


def _store_path(settings: Settings) -> Path:
    return settings.humans_tools_path.resolve()


def _empty_store() -> dict[str, Any]:
    return {"version": 1, "users": {}}


def _load(settings: Settings) -> dict[str, Any]:
    path = _store_path(settings)
    if not path.is_file():
        return _empty_store()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read human tools store %s: %s", path, exc)
        return _empty_store()
    if not isinstance(raw, dict):
        return _empty_store()
    users = raw.get("users")
    if not isinstance(users, dict):
        raw["users"] = {}
    raw.setdefault("version", 1)
    return raw


def _save(settings: Settings, data: dict[str, Any]) -> None:
    path = _store_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normalize_user_id(user_id: str) -> str:
    cleaned = (user_id or "").strip().lower()
    if not cleaned:
        raise HumanToolsError("user_id is required")
    return cleaned


def _parse_calendar(raw: dict[str, Any] | None) -> CalendarConnection | None:
    if not isinstance(raw, dict):
        return None
    provider = str(raw.get("provider") or "").strip().lower()
    if provider not in CALENDAR_PROVIDERS:
        return None
    auth_method = str(raw.get("auth_method") or "manual").strip().lower() or "manual"
    if auth_method not in {"oauth", "manual"}:
        auth_method = "manual"
    scopes_raw = raw.get("scopes")
    scopes: tuple[str, ...] = ()
    if isinstance(scopes_raw, list):
        scopes = tuple(sorted({str(s).strip() for s in scopes_raw if str(s).strip()}))
    elif isinstance(scopes_raw, str) and scopes_raw.strip():
        scopes = tuple(sorted({s.strip() for s in scopes_raw.split() if s.strip()}))
    return CalendarConnection(
        provider=provider,  # type: ignore[arg-type]
        enabled=bool(raw.get("enabled", True)),
        client_id=str(raw.get("client_id") or "").strip(),
        client_secret=str(raw.get("client_secret") or "").strip(),
        refresh_token=str(raw.get("refresh_token") or "").strip(),
        calendar_id=str(raw.get("calendar_id") or "primary").strip() or "primary",
        url=str(raw.get("url") or "").strip(),
        username=str(raw.get("username") or "").strip(),
        password=str(raw.get("password") or "").strip(),
        updated_at=float(raw.get("updated_at") or 0.0),
        auth_method=auth_method,
        scopes=scopes,
    )


def get_calendar_connection(settings: Settings, user_id: str) -> CalendarConnection | None:
    uid = _normalize_user_id(user_id)
    with _LOCK:
        data = _load(settings)
        user = data["users"].get(uid)
        if not isinstance(user, dict):
            return None
        return _parse_calendar(user.get("calendar"))


def _google_has_usable_secrets(settings: Settings, conn: CalendarConnection) -> bool:
    if not conn.refresh_token:
        return False
    if conn.auth_method == "oauth":
        from light_house.humans.google_calendar_oauth import google_oauth_configured

        return google_oauth_configured(settings)
    return bool(conn.client_id and conn.client_secret and conn.refresh_token)


def resolve_google_client_credentials(
    settings: Settings,
    conn: CalendarConnection,
) -> tuple[str, str, str]:
    """Return (client_id, client_secret, refresh_token) for Google API calls."""
    if not conn.refresh_token:
        raise HumanToolsError("Google calendar missing refresh_token")
    if conn.auth_method == "oauth" or not (conn.client_id and conn.client_secret):
        from light_house.humans.google_calendar_oauth import google_oauth_configured

        if not google_oauth_configured(settings):
            raise HumanToolsError(
                "Google OAuth house credentials missing "
                "(GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET)"
            )
        return (
            settings.google_oauth_client_id.strip(),  # type: ignore[union-attr]
            settings.google_oauth_client_secret.strip(),  # type: ignore[union-attr]
            conn.refresh_token,
        )
    return conn.client_id, conn.client_secret, conn.refresh_token


def calendar_public_view(settings: Settings, user_id: str) -> CalendarConnectionPublic:
    conn = get_calendar_connection(settings, user_id)
    if conn is None:
        return CalendarConnectionPublic(
            connected=False,
            provider=None,
            calendar_id=None,
            enabled=False,
            has_secrets=False,
            updated_at=None,
            auth_method=None,
        )
    has_secrets = False
    docs_connected = False
    sheets_connected = False
    if conn.provider == "google":
        has_secrets = _google_has_usable_secrets(settings, conn)
        from light_house.humans.google_calendar_oauth import (
            scope_grants_docs,
            scope_grants_sheets,
        )

        docs_connected = bool(
            has_secrets and conn.enabled and scope_grants_docs(list(conn.scopes))
        )
        sheets_connected = bool(
            has_secrets and conn.enabled and scope_grants_sheets(list(conn.scopes))
        )
    elif conn.provider == "caldav":
        has_secrets = bool(conn.url and conn.username and conn.password)
    elif conn.provider == "ics":
        has_secrets = bool(conn.url)
    return CalendarConnectionPublic(
        connected=has_secrets and conn.enabled,
        provider=conn.provider,
        calendar_id=conn.calendar_id if conn.provider != "ics" else None,
        enabled=conn.enabled,
        has_secrets=has_secrets,
        updated_at=conn.updated_at or None,
        auth_method=conn.auth_method,
        docs_connected=docs_connected,
        sheets_connected=sheets_connected,
        scopes=conn.scopes,
    )


def _validate_calendar_payload(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider") or "").strip().lower()
    if provider not in CALENDAR_PROVIDERS:
        raise HumanToolsError(
            "provider must be one of: google, caldav, ics"
        )
    enabled = bool(payload.get("enabled", True))
    out: dict[str, Any] = {
        "provider": provider,
        "enabled": enabled,
        "updated_at": time.time(),
    }
    if provider == "google":
        auth_method = str(payload.get("auth_method") or "manual").strip().lower() or "manual"
        if auth_method not in {"oauth", "manual"}:
            raise HumanToolsError("auth_method must be oauth or manual")
        client_id = str(payload.get("client_id") or "").strip()
        client_secret = str(payload.get("client_secret") or "").strip()
        refresh_token = str(payload.get("refresh_token") or "").strip()
        calendar_id = str(payload.get("calendar_id") or "primary").strip() or "primary"
        if not refresh_token:
            raise HumanToolsError("google calendar requires refresh_token")
        if auth_method == "manual" and (not client_id or not client_secret):
            raise HumanToolsError(
                "manual google calendar requires client_id, client_secret, and refresh_token "
                "(or use Connect Google)"
            )
        scopes_raw = payload.get("scopes")
        scopes: list[str] = []
        if isinstance(scopes_raw, list):
            scopes = sorted({str(s).strip() for s in scopes_raw if str(s).strip()})
        elif isinstance(scopes_raw, str) and scopes_raw.strip():
            scopes = sorted({s.strip() for s in scopes_raw.split() if s.strip()})
        out.update(
            {
                "auth_method": auth_method,
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "calendar_id": calendar_id,
                "scopes": scopes,
            }
        )
    elif provider == "caldav":
        url = str(payload.get("url") or "").strip()
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "").strip()
        calendar_id = str(payload.get("calendar_id") or "").strip()
        if not url or not username or not password:
            raise HumanToolsError("caldav requires url, username, and password")
        out.update(
            {
                "url": url,
                "username": username,
                "password": password,
                "calendar_id": calendar_id or "default",
            }
        )
    else:  # ics
        url = str(payload.get("url") or "").strip()
        if not url:
            raise HumanToolsError("ics requires url (secret or public iCal feed)")
        out["url"] = url
    return out


def set_calendar_connection(
    settings: Settings,
    user_id: str,
    payload: dict[str, Any],
) -> CalendarConnectionPublic:
    uid = _normalize_user_id(user_id)
    cleaned = _validate_calendar_payload(payload)
    with _LOCK:
        data = _load(settings)
        users = data.setdefault("users", {})
        user = users.get(uid)
        if not isinstance(user, dict):
            user = {}
        user["calendar"] = cleaned
        users[uid] = user
        _save(settings, data)
    logger.info("Saved calendar connection for user=%s provider=%s", uid, cleaned["provider"])
    return calendar_public_view(settings, uid)


def set_google_oauth_connection(
    settings: Settings,
    user_id: str,
    *,
    refresh_token: str,
    calendar_id: str = "primary",
    scopes: list[str] | None = None,
) -> CalendarConnectionPublic:
    """Store a one-click OAuth connection (house client id/secret used at runtime)."""
    from light_house.humans.google_calendar_oauth import GOOGLE_CONNECT_SCOPES

    scope_list = scopes if scopes is not None else sorted(GOOGLE_CONNECT_SCOPES.split())
    return set_calendar_connection(
        settings,
        user_id,
        {
            "provider": "google",
            "auth_method": "oauth",
            "refresh_token": refresh_token,
            "calendar_id": calendar_id or "primary",
            "enabled": True,
            "scopes": scope_list,
        },
    )


def clear_calendar_connection(settings: Settings, user_id: str) -> CalendarConnectionPublic:
    uid = _normalize_user_id(user_id)
    with _LOCK:
        data = _load(settings)
        users = data.setdefault("users", {})
        user = users.get(uid)
        if isinstance(user, dict) and "calendar" in user:
            del user["calendar"]
            if not user:
                del users[uid]
            else:
                users[uid] = user
            _save(settings, data)
    logger.info("Cleared calendar connection for user=%s", uid)
    return calendar_public_view(settings, uid)


def mask_secret(value: str | None) -> str:
    """Return a short mask for UI confirmation (never the real secret)."""
    if not value:
        return ""
    if len(value) <= 8:
        return "••••••••"
    return value[:4] + "…" + value[-4:]


def calendar_schema_for_ui() -> dict[str, Any]:
    """Document fields users fill on their My tools page."""
    return {
        "tool": "calendar",
        "providers": {
            "google": {
                "preferred": "Click Connect Google (Calendar + Docs + Sheets; no tokens to paste).",
                "required_manual": ["provider", "client_id", "client_secret", "refresh_token"],
                "optional": ["calendar_id", "enabled", "scopes"],
                "defaults": {"calendar_id": "primary", "enabled": True},
                "notes": (
                    "Normal path: Connect Google button (Calendar + Docs). "
                    "Advanced/manual: paste your own OAuth client credentials."
                ),
                "example": {
                    "provider": "google",
                    "auth_method": "oauth",
                    "refresh_token": "(stored automatically by Connect)",
                    "calendar_id": "primary",
                    "enabled": True,
                },
            },
            "caldav": {
                "required": ["provider", "url", "username", "password"],
                "optional": ["calendar_id", "enabled"],
                "defaults": {"enabled": True},
                "notes": "CalDAV URL from Fastmail, Nextcloud, iCloud, etc.",
                "example": {
                    "provider": "caldav",
                    "url": "https://caldav.example.com/dav/calendars/user/home/",
                    "username": "you@example.com",
                    "password": "app-password",
                    "calendar_id": "home",
                    "enabled": True,
                },
            },
            "ics": {
                "required": ["provider", "url"],
                "optional": ["enabled"],
                "defaults": {"enabled": True},
                "notes": "Read-only iCal (.ics) feed URL. Cannot create events.",
                "example": {
                    "provider": "ics",
                    "url": "https://calendar.google.com/calendar/ical/…/basic.ics",
                    "enabled": True,
                },
            },
        },
        "secret_keys": sorted(_SECRET_KEYS),
    }
