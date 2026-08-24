"""House-level Google OAuth for one-click Calendar + Docs + Sheets (stupid-simple UX)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from light_house.config import Settings

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"
GOOGLE_DOCUMENTS_SCOPE = "https://www.googleapis.com/auth/documents"
GOOGLE_SPREADSHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
GOOGLE_DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
# One Connect button grants Calendar + Docs + Sheets (+ Drive read for listing).
GOOGLE_CONNECT_SCOPES = " ".join(
    (
        GOOGLE_CALENDAR_SCOPE,
        GOOGLE_DOCUMENTS_SCOPE,
        GOOGLE_SPREADSHEETS_SCOPE,
        GOOGLE_DRIVE_READONLY_SCOPE,
    )
)
CALLBACK_PATH = "/v1/me/tools/calendar/google/callback"
STATE_TTL_SEC = 600


class GoogleOAuthError(ValueError):
    """Google OAuth configuration or exchange failure."""


def google_oauth_configured(settings: Settings) -> bool:
    return bool(
        (settings.google_oauth_client_id or "").strip()
        and (settings.google_oauth_client_secret or "").strip()
    )


def google_oauth_redirect_uri(settings: Settings, *, request_base: str | None = None) -> str:
    """
    Exact redirect URI registered in Google Cloud.

    Prefer GOOGLE_OAUTH_REDIRECT_URI, else PUBLIC_BASE_URL + callback path,
    else request_base + callback path.
    """
    explicit = (settings.google_oauth_redirect_uri or "").strip()
    if explicit:
        return explicit.rstrip("/")
    public = (settings.public_base_url or "").strip().rstrip("/")
    if public:
        return f"{public}{CALLBACK_PATH}"
    base = (request_base or "").strip().rstrip("/")
    if base:
        return f"{base}{CALLBACK_PATH}"
    raise GoogleOAuthError(
        "Set GOOGLE_OAUTH_REDIRECT_URI (or PUBLIC_BASE_URL) so Google can "
        f"return to {CALLBACK_PATH}"
    )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def sign_oauth_state(settings: Settings, *, user_id: str) -> str:
    secret = (settings.web_gate_session_secret or settings.google_oauth_client_secret or "").strip()
    if not secret:
        raise GoogleOAuthError("WEB_GATE_SESSION_SECRET (or Google client secret) required for OAuth state")
    payload = {
        "uid": user_id.strip().lower(),
        "exp": int(time.time()) + STATE_TTL_SEC,
        "n": secrets.token_hex(8),
    }
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64url(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_oauth_state(settings: Settings, state: str) -> str:
    """Return user_id from a valid state token."""
    secret = (settings.web_gate_session_secret or settings.google_oauth_client_secret or "").strip()
    if not secret:
        raise GoogleOAuthError("OAuth state secret missing")
    try:
        body, sig = (state or "").split(".", 1)
    except ValueError as exc:
        raise GoogleOAuthError("Invalid OAuth state") from exc
    expected = _b64url(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise GoogleOAuthError("Invalid OAuth state signature")
    try:
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise GoogleOAuthError("Invalid OAuth state payload") from exc
    if not isinstance(payload, dict):
        raise GoogleOAuthError("Invalid OAuth state payload")
    exp = int(payload.get("exp") or 0)
    if exp < int(time.time()):
        raise GoogleOAuthError("OAuth state expired — try Connect again")
    uid = str(payload.get("uid") or "").strip().lower()
    if not uid:
        raise GoogleOAuthError("OAuth state missing user")
    return uid


def scopes_needed_for_connect(existing_scopes: list[str] | None) -> list[str]:
    """
    Scopes to request on Connect / Reconnect.

    When Calendar/Docs already work but Sheets is missing, request only the
    missing pieces so Google's granular consent screen highlights Sheets.
    """
    have = {s.strip() for s in (existing_scopes or []) if str(s).strip()}
    wanted = list(GOOGLE_CONNECT_SCOPES.split())
    if not have:
        return wanted
    missing = [s for s in wanted if s not in have]
    # Already have everything — full re-consent refresh.
    return missing or wanted


def build_google_authorize_url(
    settings: Settings,
    *,
    user_id: str,
    request_base: str | None = None,
    scopes: list[str] | None = None,
) -> str:
    if not google_oauth_configured(settings):
        raise GoogleOAuthError(
            "Google connect is not set up yet. "
            "Dad needs GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in the environment."
        )
    redirect_uri = google_oauth_redirect_uri(settings, request_base=request_base)
    state = sign_oauth_state(settings, user_id=user_id)
    scope_list = [s.strip() for s in (scopes or []) if str(s).strip()]
    scope_value = " ".join(scope_list) if scope_list else GOOGLE_CONNECT_SCOPES
    params = {
        "client_id": settings.google_oauth_client_id.strip(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope_value,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_tokens(
    settings: Settings,
    *,
    code: str,
    request_base: str | None = None,
    require_refresh_token: bool = True,
) -> dict[str, Any]:
    if not google_oauth_configured(settings):
        raise GoogleOAuthError("Google OAuth is not configured")
    redirect_uri = google_oauth_redirect_uri(settings, request_base=request_base)
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_oauth_client_id.strip(),
                "client_secret": settings.google_oauth_client_secret.strip(),
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if resp.status_code >= 400:
        logger.warning("Google token exchange failed: %s %s", resp.status_code, resp.text[:400])
        raise GoogleOAuthError(
            f"Google token exchange failed ({resp.status_code}). "
            "Try Connect again; if it keeps failing, Dad should check the redirect URI matches Google Cloud."
        )
    data = resp.json()
    if not isinstance(data, dict):
        raise GoogleOAuthError("Google token exchange returned unexpected data")
    refresh = str(data.get("refresh_token") or "").strip()
    if require_refresh_token and not refresh:
        # Google may omit refresh_token on re-consent without prompt=consent; we force consent.
        raise GoogleOAuthError(
            "Google did not return a refresh token. Disconnect in Google Account permissions "
            "for this app, then click Connect Google again."
        )
    return data


def parse_granted_scopes(token_payload: dict[str, Any]) -> list[str]:
    """Normalize scope string from Google token response into a sorted list."""
    raw = str(token_payload.get("scope") or "").strip()
    if not raw:
        return []
    return sorted({s.strip() for s in raw.split() if s.strip()})


def scopes_from_access_token(access_token: str) -> list[str]:
    """Ask Google which scopes an access token actually carries (authoritative)."""
    token = (access_token or "").strip()
    if not token:
        return []
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"access_token": token},
            )
    except httpx.HTTPError as exc:
        logger.warning("Google tokeninfo request failed: %s", exc)
        return []
    if resp.status_code >= 400:
        logger.warning(
            "Google tokeninfo failed: %s %s", resp.status_code, resp.text[:300]
        )
        return []
    data = resp.json()
    if not isinstance(data, dict):
        return []
    return parse_granted_scopes(data)


def merge_oauth_scopes(*scope_lists: list[str] | tuple[str, ...] | None) -> list[str]:
    merged: set[str] = set()
    for scopes in scope_lists:
        if not scopes:
            continue
        for scope in scopes:
            text = str(scope).strip()
            if text:
                merged.add(text)
    return sorted(merged)


def resolve_granted_scopes(
    token_payload: dict[str, Any],
    *,
    prior_scopes: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """
    Figure out which scopes to persist after OAuth.

    Prefer tokeninfo (what the access token can actually do), merge with the
    token response and any previously stored scopes (include_granted_scopes).
    """
    from_response = parse_granted_scopes(token_payload)
    access = str(token_payload.get("access_token") or "").strip()
    from_tokeninfo = scopes_from_access_token(access) if access else []
    merged = merge_oauth_scopes(from_response, from_tokeninfo, prior_scopes)
    if not merged:
        # Last resort — assume full connect set only when Google returned nothing.
        logger.warning("Google OAuth returned no scopes; defaulting to full connect set")
        return sorted(GOOGLE_CONNECT_SCOPES.split())
    logger.info(
        "Google OAuth scopes response=%s tokeninfo=%s prior=%s merged=%s",
        from_response,
        from_tokeninfo,
        list(prior_scopes or ()),
        merged,
    )
    return merged


def missing_connect_scope_labels(scopes: list[str] | None) -> list[str]:
    """Short labels for UI when expected Connect scopes are absent."""
    have = {s.strip() for s in (scopes or []) if str(s).strip()}
    missing: list[str] = []
    if GOOGLE_CALENDAR_SCOPE not in have and not any(
        s.endswith("/auth/calendar") or s.endswith("/auth/calendar.events") for s in have
    ):
        missing.append("calendar")
    if GOOGLE_DOCUMENTS_SCOPE not in have:
        missing.append("docs")
    if GOOGLE_SPREADSHEETS_SCOPE not in have:
        missing.append("sheets")
    return missing


def scope_grants_docs(scopes: list[str] | None) -> bool:
    if not scopes:
        return False
    return GOOGLE_DOCUMENTS_SCOPE in scopes


def scope_grants_sheets(scopes: list[str] | None) -> bool:
    if not scopes:
        return False
    return GOOGLE_SPREADSHEETS_SCOPE in scopes or any(
        s.endswith("/auth/spreadsheets") for s in scopes
    )


def scope_grants_calendar(scopes: list[str] | None) -> bool:
    if not scopes:
        # Legacy connections before scopes were stored — calendar-only.
        return True
    return GOOGLE_CALENDAR_SCOPE in scopes or any(
        s.endswith("/auth/calendar") or s.endswith("/auth/calendar.events") for s in scopes
    )


def request_base_from_headers(
    *,
    scheme: str | None,
    host: str | None,
    forwarded_proto: str | None = None,
    forwarded_host: str | None = None,
) -> str | None:
    proto = (forwarded_proto or scheme or "https").split(",")[0].strip() or "https"
    hostname = (forwarded_host or host or "").split(",")[0].strip()
    if not hostname:
        return None
    return f"{proto}://{hostname}"
