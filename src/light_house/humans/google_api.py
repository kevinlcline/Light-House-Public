"""Shared Google HTTP helpers for the signed-in human's connection."""

from __future__ import annotations

import httpx

from light_house.config import Settings
from light_house.humans.tools_store import CalendarConnection, resolve_google_client_credentials

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def google_access_token(
    settings: Settings,
    conn: CalendarConnection,
    *,
    client: httpx.Client | None = None,
) -> str:
    client_id, client_secret, refresh_token = resolve_google_client_credentials(
        settings, conn
    )
    own = client is None
    http = client or httpx.Client(timeout=20.0)
    try:
        resp = http.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    finally:
        if own:
            http.close()
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google token refresh failed ({resp.status_code}): {resp.text[:300]}"
        )
    token = resp.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Google token refresh returned no access_token")
    return token


def resolve_google_user_id(
    settings: Settings,
    *,
    account_user_id: str | None,
    speaker_id: str | None,
    chat_channel: str | None,
    quiet_host: bool,
) -> tuple[str | None, str | None]:
    """Return (user_id, error).

    Chat uses the signed-in human. A quiet sit, with no one signed in on the
    thread, may read the host's Google log and mail. It does not wake anyone.
    """
    # Imported here so tools/__init__ can finish loading this module.
    from light_house.tools.calendar_tools import calendar_access_denied_reason

    denied = calendar_access_denied_reason(
        account_user_id=account_user_id,
        speaker_id=speaker_id,
        chat_channel=chat_channel,
    )
    if denied is None:
        return (account_user_id or "").strip(), None
    if not quiet_host:
        return None, denied
    channel = (chat_channel or "dm").strip().lower()
    if channel == "group":
        return None, denied
    if "guest" in denied.lower() or "visitor" in denied.lower():
        return None, denied
    if (account_user_id or "").strip():
        return None, denied
    host = (settings.house_dad_user_id or "").strip()
    if not host:
        return None, denied
    return host, None
