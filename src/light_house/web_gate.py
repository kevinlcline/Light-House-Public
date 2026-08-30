"""Password gate for public Light-House deployments."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from pathlib import Path
from urllib.parse import quote

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response

from light_house.config import Settings, get_settings

logger = logging.getLogger(__name__)

SESSION_COOKIE = "lh_session"

NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}

# Public marketing / crawler paths (no session). House UI stays gated.
_PUBLIC_GET_PATHS = frozenset(
    {
        "/robots.txt",
        "/llms.txt",
        "/sitemap.xml",
        "/favicon.ico",
        "/favicon.png",
        "/apple-touch-icon.png",
    }
)
_PUBLIC_DIR_PREFIX = "/public/"


def safe_public_file(repo_root: Path, url_path: str) -> Path | None:
    """Resolve a /public/… URL to a file under repo_root/public, or None."""
    if not url_path.startswith(_PUBLIC_DIR_PREFIX):
        return None
    rel = url_path[len(_PUBLIC_DIR_PREFIX) :]
    if not rel or rel.startswith("/") or ".." in rel.split("/"):
        return None
    root = (repo_root / "public").resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if target.is_file():
        return target
    return None


def _public_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".xml":
        return "application/xml; charset=utf-8"
    if suffix in {".txt", ".md"}:
        return "text/plain; charset=utf-8"
    return "text/html; charset=utf-8"



def apply_no_store_headers(response: Response) -> Response:
    for key, value in NO_STORE_HEADERS.items():
        response.headers[key] = value
    return response


def sanitize_next(next_path: str | None) -> str:
    """Allow only same-site relative redirects after login."""
    if not next_path or not next_path.startswith("/") or next_path.startswith("//"):
        return "/"
    return next_path


def request_is_secure(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    forwarded = request.headers.get("x-forwarded-proto", "")
    return forwarded.split(",")[0].strip().lower() == "https"


def _sign_payload(payload: dict[str, object], secret: str) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    sig = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(body + sig).decode().rstrip("=")


def _verify_token(token: str, secret: str) -> bool:
    return parse_session_payload(token, secret) is not None


def parse_session_payload(token: str | None, secret: str) -> dict[str, object] | None:
    """Return session payload if token is valid and unexpired."""
    if not token or not secret:
        return None
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode())
    except (ValueError, UnicodeEncodeError):
        return None
    if len(raw) <= 32:
        return None
    body, sig = raw[:-32], raw[-32:]
    expected = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp < int(time.time()):
        return None
    return payload


def is_authenticated(request: Request, settings: Settings) -> bool:
    if not settings.web_gate_enabled:
        return True
    token = request.cookies.get(SESSION_COOKIE)
    secret = settings.web_gate_session_secret
    if not token or not secret:
        return False
    return _verify_token(token, secret)


def check_password(settings: Settings, password: str) -> bool:
    expected = settings.web_gate_password or ""
    if not expected:
        return False
    return secrets.compare_digest(password, expected)


def session_cookie_header(
    settings: Settings,
    *,
    secure: bool,
    user_id: str,
    role: str,
) -> str:
    exp = int(time.time()) + settings.web_gate_session_days * 86400
    token = _sign_payload(
        {
            "exp": exp,
            "user_id": user_id.strip().lower(),
            "role": role.strip().lower(),
        },
        settings.web_gate_session_secret or "",
    )
    parts = [
        f"{SESSION_COOKIE}={token}",
        f"Max-Age={settings.web_gate_session_days * 86400}",
        "Path=/",
        "HttpOnly",
        "SameSite=Lax",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def clear_session_cookie_header(*, secure: bool) -> str:
    parts = [f"{SESSION_COOKIE}=", "Max-Age=0", "Path=/", "HttpOnly", "SameSite=Lax"]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept or "*/*" in accept or not accept


def _login_redirect(path: str) -> RedirectResponse:
    return apply_no_store_headers(
        RedirectResponse(f"/login?next={quote(path)}", status_code=302)
    )


class WebGateMiddleware(BaseHTTPMiddleware):
    """Require a signed session cookie when WEB_GATE_ENABLED is true."""

    def __init__(self, app, *, repo_root: Path) -> None:
        super().__init__(app)
        self.repo_root = repo_root

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.web_gate_enabled:
            return await call_next(request)

        path = request.url.path
        method = request.method

        if path == "/health" and method == "GET":
            return await call_next(request)

        if path == "/login" and method in ("GET", "POST"):
            return await call_next(request)

        if path == "/logout" and method == "POST":
            return await call_next(request)

        # Public homepage notify signup (no session required).
        if path == "/v1/public/notify" and method == "POST":
            return await call_next(request)

        # robots.txt / llms.txt / favicons — crawlers (Gemini, Googlebot, …) need these
        # without hitting the password gate. Serve text files here with no-store so
        # Cloudflare cannot keep a stale copy that still looks like an AI opt-out.
        if path in _PUBLIC_GET_PATHS and method in ("GET", "HEAD"):
            if path in ("/robots.txt", "/llms.txt", "/sitemap.xml"):
                target = self.repo_root / path.lstrip("/")
                if target.is_file():
                    return FileResponse(
                        target,
                        media_type=(
                            "application/xml; charset=utf-8"
                            if path == "/sitemap.xml"
                            else "text/plain; charset=utf-8"
                        ),
                        headers={
                            **NO_STORE_HEADERS,
                            "X-Robots-Tag": "all",
                        },
                    )
                return Response("Not found", status_code=404)
            return await call_next(request)

        if method in ("GET", "HEAD"):
            public_file = safe_public_file(self.repo_root, path)
            if public_file is not None:
                return FileResponse(
                    public_file,
                    media_type=_public_media_type(public_file),
                    headers={
                        **NO_STORE_HEADERS,
                        "X-Robots-Tag": "all",
                    },
                )

        authed = is_authenticated(request, settings)

        # Public landing story at "/" for guests and fetchers (GET + HEAD).
        if path == "/" and method in ("GET", "HEAD"):
            if authed:
                response = await call_next(request)
                return apply_no_store_headers(response)
            landing = self.repo_root / "landing.html"
            if landing.is_file():
                return FileResponse(landing, headers=NO_STORE_HEADERS)
            return Response("Not found", status_code=404)

        if authed:
            return await call_next(request)

        if path.startswith("/v1/"):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)

        if _wants_html(request):
            return _login_redirect(path)
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
