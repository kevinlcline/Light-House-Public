"""Read-only web search for main agents (Lumen, Ara)."""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = "Light-House-Agent/1.0 (read-only search)"
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_MAX_QUERY_LEN = 500


def _resolve_ddg_href(href: str) -> str:
    if "uddg=" not in href:
        return href
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    uddg = qs.get("uddg")
    if uddg:
        return unquote(uddg[0])
    return href


class _DDGResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str, str]] = []
        self._cur_url = ""
        self._cur_title = ""
        self._capture: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {k: (v or "") for k, v in attrs}
        cls = attrs_d.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._cur_url = attrs_d.get("href", "")
            self._capture = "title"
            self._buf = []
        elif tag == "a" and "result__snippet" in cls:
            self._capture = "snippet"
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture:
            return
        text = "".join(self._buf).strip()
        if self._capture == "title":
            self._cur_title = text
            self._capture = None
        elif self._capture == "snippet":
            url = _resolve_ddg_href(self._cur_url)
            if self._cur_title and url:
                self.results.append((self._cur_title, url, text))
            self._cur_url = ""
            self._cur_title = ""
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buf.append(data)


def _normalize_query(query: str) -> str:
    q = query.strip()
    if not q:
        raise ValueError("Search query is required")
    if len(q) > _MAX_QUERY_LEN:
        q = q[:_MAX_QUERY_LEN]
    return q


def _format_results(query: str, hits: list[tuple[str, str, str]]) -> str:
    if not hits:
        return f'No results for "{query}". Try different keywords or fetch_url on a known site.'
    lines = [f'Search results for "{query}":', ""]
    for i, (title, url, snippet) in enumerate(hits, start=1):
        lines.append(f"{i}. {title}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    lines.append("Use fetch_url on a link when you need the full page text.")
    return "\n".join(lines).rstrip()


def _search_ddg(
    query: str,
    *,
    max_results: int,
    timeout_seconds: float,
) -> list[tuple[str, str, str]]:
    timeout = httpx.Timeout(max(1.0, timeout_seconds))
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        response = client.post(_DDG_HTML_URL, data={"q": query})
        response.raise_for_status()
        html = response.text

    parser = _DDGResultParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        logger.debug("DDG HTML parse failed; trying regex fallback", exc_info=True)
        return _search_ddg_regex_fallback(html, max_results=max_results)

    return parser.results[: max(1, max_results)]


def _search_ddg_regex_fallback(html: str, *, max_results: int) -> list[tuple[str, str, str]]:
    titles: list[tuple[str, str]] = []
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
        html,
        re.I,
    ):
        titles.append((_resolve_ddg_href(m.group(1)), m.group(2).strip()))

    snippets = [m.group(1).strip() for m in re.finditer(r'class="result__snippet"[^>]*>([^<]+)</a>', html, re.I)]

    hits: list[tuple[str, str, str]] = []
    for i, (url, title) in enumerate(titles[:max_results]):
        snippet = snippets[i] if i < len(snippets) else ""
        hits.append((title, url, snippet))
    return hits


def _search_brave(
    query: str,
    *,
    api_key: str,
    max_results: int,
    timeout_seconds: float,
) -> list[tuple[str, str, str]]:
    timeout = httpx.Timeout(max(1.0, timeout_seconds))
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key.strip(),
    }
    params = {"q": query, "count": max(1, min(max_results, 20))}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        response = client.get(_BRAVE_SEARCH_URL, params=params)
        response.raise_for_status()
        data = response.json()

    hits: list[tuple[str, str, str]] = []
    for item in (data.get("web") or {}).get("results") or []:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("description") or "").strip()
        if title and url:
            hits.append((title, url, snippet))
    return hits[: max(1, max_results)]


def search_web_text(
    query: str,
    *,
    max_results: int = 5,
    timeout_seconds: float = 15.0,
    brave_api_key: str | None = None,
) -> str:
    """
    Run a web search and return a formatted string for the agent.

    Uses Brave Search API when brave_api_key is set; otherwise DuckDuckGo HTML.
    """
    q = _normalize_query(query)
    limit = max(1, min(max_results, 20))

    try:
        if brave_api_key and brave_api_key.strip():
            hits = _search_brave(
                q,
                api_key=brave_api_key,
                max_results=limit,
                timeout_seconds=timeout_seconds,
            )
        else:
            hits = _search_ddg(q, max_results=limit, timeout_seconds=timeout_seconds)
    except httpx.HTTPStatusError as exc:
        return f"Search failed: HTTP {exc.response.status_code}"
    except httpx.RequestError as exc:
        return f"Search failed: {exc}"

    return _format_results(q, hits)
