"""Read-only HTTP fetch for main agents (Lumen, Ara)."""

from __future__ import annotations

import ipaddress
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_BLOCKED_SCHEMES = frozenset({"http", "https"})
_USER_AGENT = "Light-House-Agent/1.0 (read-only fetch)"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _validate_fetch_url(url: str) -> str:
    raw = url.strip()
    if not raw:
        raise ValueError("URL is required")
    if "\x00" in raw:
        raise ValueError("Invalid URL")
    parsed = urlparse(raw)
    if parsed.scheme not in _BLOCKED_SCHEMES:
        raise ValueError("Only http and https URLs are allowed")
    if not parsed.netloc:
        raise ValueError("URL must include a host")
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a host")
    lowered = host.lower()
    if lowered in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"):
        raise ValueError("Fetching local addresses is not allowed")
    if lowered.endswith(".local") or lowered.endswith(".internal"):
        raise ValueError("Fetching internal hostnames is not allowed")
    try:
        addr = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        ):
            raise ValueError("Fetching private or reserved addresses is not allowed")
    return raw


def _html_to_text(body: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(body)
        parser.close()
        text = parser.get_text()
        if text:
            return text
    except Exception:
        logger.debug("HTML parse failed; falling back to raw strip", exc_info=True)
    return re.sub(r"<[^>]+>", " ", body)


def fetch_url_text(
    url: str,
    *,
    timeout_seconds: float = 15.0,
    max_chars: int = 12_000,
    max_bytes: int = 512_000,
) -> str:
    """
    Fetch a public http(s) URL and return readable text (HTML stripped when possible).

    Raises ValueError for blocked URLs; returns error strings for HTTP failures when called from tools.
    """
    safe_url = _validate_fetch_url(url)
    timeout = httpx.Timeout(max(1.0, timeout_seconds))

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        response = client.get(safe_url)
        response.raise_for_status()
        _validate_fetch_url(str(response.url))

    content_type = (response.headers.get("content-type") or "").lower()
    raw = response.content[: max(1, max_bytes)]

    if "html" in content_type or raw.lstrip()[:1] in (b"<",):
        try:
            body = raw.decode(response.encoding or "utf-8", errors="replace")
            text = _html_to_text(body)
        except Exception:
            text = raw.decode("utf-8", errors="replace")
    elif "json" in content_type or "text" in content_type or "xml" in content_type:
        text = raw.decode(response.encoding or "utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported content type: {content_type or 'unknown'}")

    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n… [truncated]"
    return text
