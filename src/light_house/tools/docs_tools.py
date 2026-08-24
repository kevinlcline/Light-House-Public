"""Per-human Google Docs tools — private 1:1 only, same Google connect as calendar."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.config import Settings
from light_house.humans.google_calendar_oauth import scope_grants_docs
from light_house.humans.tools_store import (
    calendar_public_view,
    get_calendar_connection,
    resolve_google_client_credentials,
)
from light_house.tools.calendar_tools import calendar_access_denied_reason

logger = logging.getLogger(__name__)

DOCS_TOOL_NAMES = frozenset(
    {
        "list_google_docs",
        "read_google_doc",
        "create_google_doc",
        "append_google_doc",
    }
)

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_DOCS_API = "https://docs.googleapis.com/v1/documents"
_DRIVE_FILES_API = "https://www.googleapis.com/drive/v3/files"
_DOC_ID_RE = re.compile(
    r"(?:https?://)?(?:docs\.google\.com/document/d/)?([a-zA-Z0-9_-]{10,})"
)

_UNAVAILABLE_NOT_CONNECTED = (
    "FAILED: This human has not connected Google Docs yet. "
    "Tell them (briefly) to open **My tools** and click **Connect Google** "
    "(or Reconnect if Calendar was connected earlier without Docs) — "
    "you cannot connect it for them."
)

_UNAVAILABLE_DISABLED = (
    "FAILED: This human's Google connection is disabled. "
    "They can re-enable it on their **My tools** page."
)

_MAX_READ_CHARS = 12000


class ListGoogleDocsArgs(BaseModel):
    query: str = Field(
        default="",
        max_length=200,
        description="Optional name search (matches document title). Empty = recent docs.",
    )
    max_docs: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Max documents to return (1–30). Default 10.",
    )


class ReadGoogleDocArgs(BaseModel):
    doc_id_or_url: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Google Doc id or full docs.google.com URL.",
    )
    max_chars: int = Field(
        default=8000,
        ge=500,
        le=_MAX_READ_CHARS,
        description="Max characters of plain text to return.",
    )


class CreateGoogleDocArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Document title.")
    content: str = Field(
        default="",
        max_length=50000,
        description="Optional initial plain-text body.",
    )


class AppendGoogleDocArgs(BaseModel):
    doc_id_or_url: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Google Doc id or full docs.google.com URL.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Plain text to append at the end of the document.",
    )


def _tool_failed(msg: str) -> str:
    if msg.startswith("FAILED:"):
        return msg
    return f"FAILED: {msg}"


def _extract_doc_id(doc_id_or_url: str) -> str:
    raw = (doc_id_or_url or "").strip()
    if not raw:
        raise ValueError("doc_id_or_url is required")
    match = _DOC_ID_RE.search(raw)
    if not match:
        raise ValueError("Could not parse a Google Doc id from that value")
    return match.group(1)


def _google_access_token(settings: Settings, conn: Any) -> str:
    client_id, client_secret, refresh_token = resolve_google_client_credentials(
        settings, conn
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google token refresh failed ({resp.status_code}): {resp.text[:300]}"
        )
    token = resp.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Google token refresh returned no access_token")
    return token


def _require_docs_connection(settings: Settings, *, account_user_id: str) -> Any | str:
    view = calendar_public_view(settings, account_user_id)
    if not view.has_secrets or view.provider != "google":
        return _UNAVAILABLE_NOT_CONNECTED
    if not view.enabled:
        return _UNAVAILABLE_DISABLED
    if not view.docs_connected:
        return _UNAVAILABLE_NOT_CONNECTED
    conn = get_calendar_connection(settings, account_user_id)
    if conn is None or not scope_grants_docs(list(conn.scopes)):
        return _UNAVAILABLE_NOT_CONNECTED
    return conn


def _docs_plain_text(document: dict[str, Any]) -> str:
    body = document.get("body") or {}
    content = body.get("content") or []
    chunks: list[str] = []

    def walk(elements: list[Any]) -> None:
        for el in elements:
            if not isinstance(el, dict):
                continue
            paragraph = el.get("paragraph")
            if isinstance(paragraph, dict):
                parts: list[str] = []
                for pe in paragraph.get("elements") or []:
                    if not isinstance(pe, dict):
                        continue
                    tr = pe.get("textRun")
                    if isinstance(tr, dict) and isinstance(tr.get("content"), str):
                        parts.append(tr["content"])
                chunks.append("".join(parts))
            table = el.get("table")
            if isinstance(table, dict):
                for row in table.get("tableRows") or []:
                    if not isinstance(row, dict):
                        continue
                    for cell in row.get("tableCells") or []:
                        if isinstance(cell, dict):
                            walk(cell.get("content") or [])

    walk(content)
    text = "".join(chunks)
    # Docs API often ends paragraphs with \n already.
    return text.replace("\r\n", "\n")


def _end_index(document: dict[str, Any]) -> int:
    body = document.get("body") or {}
    content = body.get("content") or []
    end = 1
    for el in content:
        if isinstance(el, dict) and isinstance(el.get("endIndex"), int):
            end = max(end, el["endIndex"])
    # Insert before the final newline of the doc body.
    return max(1, end - 1)


def _list_docs(settings: Settings, conn: Any, *, query: str, max_docs: int) -> str:
    token = _google_access_token(settings, conn)
    q_parts = ["mimeType='application/vnd.google-apps.document'", "trashed=false"]
    q = (query or "").strip().replace("'", "\\'")
    if q:
        q_parts.append(f"name contains '{q}'")
    params = {
        "q": " and ".join(q_parts),
        "pageSize": max_docs,
        "orderBy": "modifiedTime desc",
        "fields": "files(id,name,modifiedTime,webViewLink)",
    }
    with httpx.Client(timeout=25.0) as client:
        resp = client.get(
            _DRIVE_FILES_API,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google Drive list failed ({resp.status_code}): {resp.text[:300]}"
        )
    files = resp.json().get("files") or []
    if not files:
        label = f" matching `{query.strip()}`" if query.strip() else ""
        return f"SUCCESS: No Google Docs found{label}."
    lines = [f"SUCCESS: Google Docs ({len(files)}):"]
    for f in files:
        name = str(f.get("name") or "(untitled)")
        doc_id = str(f.get("id") or "")
        modified = str(f.get("modifiedTime") or "")
        link = str(f.get("webViewLink") or f"https://docs.google.com/document/d/{doc_id}/edit")
        lines.append(f"- {name} — id=`{doc_id}` modified={modified}\n  {link}")
    return "\n".join(lines)


def _read_doc(settings: Settings, conn: Any, *, doc_id_or_url: str, max_chars: int) -> str:
    doc_id = _extract_doc_id(doc_id_or_url)
    token = _google_access_token(settings, conn)
    with httpx.Client(timeout=25.0) as client:
        resp = client.get(
            f"{_DOCS_API}/{quote(doc_id, safe='')}",
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google Docs read failed ({resp.status_code}): {resp.text[:300]}"
        )
    document = resp.json()
    title = str(document.get("title") or "(untitled)")
    text = _docs_plain_text(document)
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    note = " (truncated)" if truncated else ""
    link = f"https://docs.google.com/document/d/{doc_id}/edit"
    return (
        f"SUCCESS: Google Doc `{title}` id=`{doc_id}`{note}\n"
        f"Link: {link}\n\n{text or '(empty document)'}"
    )


def _create_doc(settings: Settings, conn: Any, *, title: str, content: str) -> str:
    token = _google_access_token(settings, conn)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=25.0) as client:
        resp = client.post(_DOCS_API, headers=headers, json={"title": title.strip()})
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Google Docs create failed ({resp.status_code}): {resp.text[:300]}"
            )
        document = resp.json()
        doc_id = str(document.get("documentId") or "")
        if not doc_id:
            raise RuntimeError("Google Docs create returned no documentId")
        body = (content or "").strip()
        if body:
            # Refresh to get endIndex, then insert at start (index 1).
            got = client.get(
                f"{_DOCS_API}/{quote(doc_id, safe='')}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if got.status_code >= 400:
                raise RuntimeError(
                    f"Google Docs create ok but read-back failed ({got.status_code})"
                )
            insert_text = body if body.endswith("\n") else body + "\n"
            upd = client.post(
                f"{_DOCS_API}/{quote(doc_id, safe='')}:batchUpdate",
                headers=headers,
                json={
                    "requests": [
                        {
                            "insertText": {
                                "location": {"index": 1},
                                "text": insert_text,
                            }
                        }
                    ]
                },
            )
            if upd.status_code >= 400:
                raise RuntimeError(
                    f"Doc created but writing body failed ({upd.status_code}): "
                    f"{upd.text[:300]}"
                )
    link = f"https://docs.google.com/document/d/{doc_id}/edit"
    return f"SUCCESS: Created Google Doc `{title.strip()}` id=`{doc_id}`\nLink: {link}"


def _append_doc(settings: Settings, conn: Any, *, doc_id_or_url: str, content: str) -> str:
    doc_id = _extract_doc_id(doc_id_or_url)
    token = _google_access_token(settings, conn)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=25.0) as client:
        got = client.get(
            f"{_DOCS_API}/{quote(doc_id, safe='')}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if got.status_code >= 400:
            raise RuntimeError(
                f"Google Docs read failed ({got.status_code}): {got.text[:300]}"
            )
        document = got.json()
        title = str(document.get("title") or "(untitled)")
        index = _end_index(document)
        text = content if content.endswith("\n") else content + "\n"
        # Prefer a leading newline so appends don't glue to prior text.
        if index > 1 and not text.startswith("\n"):
            text = "\n" + text
        upd = client.post(
            f"{_DOCS_API}/{quote(doc_id, safe='')}:batchUpdate",
            headers=headers,
            json={
                "requests": [
                    {"insertText": {"location": {"index": index}, "text": text}}
                ]
            },
        )
        if upd.status_code >= 400:
            raise RuntimeError(
                f"Google Docs append failed ({upd.status_code}): {upd.text[:300]}"
            )
    link = f"https://docs.google.com/document/d/{doc_id}/edit"
    return (
        f"SUCCESS: Appended {len(content)} chars to `{title}` id=`{doc_id}`\n"
        f"Link: {link}"
    )


def execute_docs_tool(
    name: str,
    args: dict[str, Any],
    *,
    settings: Settings,
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
) -> str:
    denied = calendar_access_denied_reason(
        account_user_id=account_user_id,
        speaker_id=speaker_id,
        chat_channel=chat_channel,
    )
    if denied:
        # Reuse calendar gating copy, but swap the product word for clarity.
        return denied.replace("Calendar tools", "Google Docs tools").replace(
            "calendar access", "Google Docs access"
        ).replace("calendar", "Google Docs")
    assert account_user_id is not None
    conn_or_err = _require_docs_connection(settings, account_user_id=account_user_id)
    if isinstance(conn_or_err, str):
        return conn_or_err
    conn = conn_or_err
    try:
        if name == "list_google_docs":
            return _list_docs(
                settings,
                conn,
                query=str(args.get("query") or ""),
                max_docs=max(1, min(30, int(args.get("max_docs") or 10))),
            )
        if name == "read_google_doc":
            return _read_doc(
                settings,
                conn,
                doc_id_or_url=str(args.get("doc_id_or_url") or ""),
                max_chars=max(500, min(_MAX_READ_CHARS, int(args.get("max_chars") or 8000))),
            )
        if name == "create_google_doc":
            return _create_doc(
                settings,
                conn,
                title=str(args.get("title") or ""),
                content=str(args.get("content") or ""),
            )
        if name == "append_google_doc":
            return _append_doc(
                settings,
                conn,
                doc_id_or_url=str(args.get("doc_id_or_url") or ""),
                content=str(args.get("content") or ""),
            )
        return _tool_failed(f"unknown docs tool: {name}")
    except Exception as exc:  # noqa: BLE001 - tool boundary
        logger.warning("Docs tool %s failed for user=%s: %s", name, account_user_id, exc)
        return _tool_failed(f"{name} — {exc}")


def build_docs_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="list_google_docs",
            description=(
                "List the signed-in human's Google Docs (private 1:1 only; not group; "
                "not guests). Optional title query."
            ),
            func=lambda query="", max_docs=10: (
                "FAILED: list_google_docs must run through the agent tool runner"
            ),
            args_schema=ListGoogleDocsArgs,
        ),
        StructuredTool.from_function(
            name="read_google_doc",
            description=(
                "Read plain text from one Google Doc by id or URL "
                "(private 1:1 only; not group; not guests)."
            ),
            func=lambda doc_id_or_url, max_chars=8000: (
                "FAILED: read_google_doc must run through the agent tool runner"
            ),
            args_schema=ReadGoogleDocArgs,
        ),
        StructuredTool.from_function(
            name="create_google_doc",
            description=(
                "Create a Google Doc with a title and optional body "
                "(private 1:1 only; not group; not guests)."
            ),
            func=lambda title, content="": (
                "FAILED: create_google_doc must run through the agent tool runner"
            ),
            args_schema=CreateGoogleDocArgs,
        ),
        StructuredTool.from_function(
            name="append_google_doc",
            description=(
                "Append plain text to the end of a Google Doc "
                "(private 1:1 only; not group; not guests)."
            ),
            func=lambda doc_id_or_url, content: (
                "FAILED: append_google_doc must run through the agent tool runner"
            ),
            args_schema=AppendGoogleDocArgs,
        ),
    ]
