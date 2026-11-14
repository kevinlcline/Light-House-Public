"""Per-human Drive read tools. Listing and reading only — Drive writes stay off."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.config import Settings
from light_house.humans.google_api import google_access_token, resolve_google_user_id
from light_house.humans.google_calendar_oauth import scope_grants_drive
from light_house.humans.tools_store import calendar_public_view, get_calendar_connection
from light_house.tools.text_window import (
    chars_label,
    continuation_footer,
    past_end_note,
    slice_text_window,
)

logger = logging.getLogger(__name__)

DRIVE_TOOL_NAMES = frozenset({"list_drive_files", "read_drive_file"})

_FILES = "https://www.googleapis.com/drive/v3/files"
_FILE = "https://www.googleapis.com/drive/v3/files/{file_id}"
_MAX_READ = 8000
_TEXT_PREFIXES = ("text/",)
_TEXT_MIMES = {"application/json", "application/xml", "application/javascript"}
_EXPORTS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

_NOT_CONNECTED = (
    "FAILED: This human has not connected Google Drive yet. "
    "Tell them (briefly) to open **My tools** and click **Reconnect Google** — "
    "you cannot connect it for them."
)


class ListDriveArgs(BaseModel):
    query: str = Field(
        default="",
        max_length=200,
        description="Optional file-name search. Empty = recently changed files.",
    )
    max_files: int = Field(default=10, ge=1, le=20)


class ReadDriveArgs(BaseModel):
    file_id: str = Field(..., min_length=4, max_length=200, description="Drive file id.")
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Character offset into the file text. Use the continuation "
            "offset from the excerpt header to read the next window."
        ),
    )
    max_chars: int = Field(default=6000, ge=500, le=_MAX_READ)


def _failed(msg: str) -> str:
    if msg.startswith("FAILED:"):
        return msg
    return f"FAILED: {msg}"


def _product_denied(denied: str) -> str:
    return (
        denied.replace("Calendar tools", "Drive tools")
        .replace("calendar tools", "Drive tools")
        .replace("a calendar", "Drive")
        .replace("calendar is", "Drive is")
    )


def _require_drive(settings: Settings, user_id: str) -> Any | str:
    view = calendar_public_view(settings, user_id)
    if not view.has_secrets or view.provider != "google" or not view.enabled:
        return _NOT_CONNECTED
    conn = get_calendar_connection(settings, user_id)
    if conn is None or not scope_grants_drive(list(conn.scopes)):
        return _NOT_CONNECTED
    return conn


def _escape_query(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _list_files(settings: Settings, conn: Any, *, query: str, max_files: int) -> str:
    token = google_access_token(settings, conn)
    needle = query.strip()
    if needle:
        q = f"name contains '{_escape_query(needle)}' and trashed = false"
    else:
        q = "trashed = false"
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            _FILES,
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": q,
                "pageSize": str(max_files),
                "orderBy": "modifiedTime desc",
                "fields": "files(id,name,mimeType,modifiedTime)",
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Drive list failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json() if isinstance(resp.json(), dict) else {}
    files = [item for item in (data.get("files") or []) if isinstance(item, dict)]
    lines = [f"SUCCESS: {len(files)} file(s)."]
    for item in files:
        lines.append(
            f"- id={item.get('id')} | {item.get('modifiedTime') or ''} | "
            f"{item.get('name') or 'untitled'} | {item.get('mimeType') or ''}"
        )
    return "\n".join(lines)


def _read_file(
    settings: Settings,
    conn: Any,
    *,
    file_id: str,
    offset: int = 0,
    max_chars: int = 6000,
) -> str:
    token = google_access_token(settings, conn)
    fid = quote(file_id.strip(), safe="")
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=20.0) as client:
        meta = client.get(
            _FILE.format(file_id=fid),
            headers=headers,
            params={"fields": "id,name,mimeType"},
        )
        if meta.status_code >= 400:
            raise RuntimeError(f"Drive file failed ({meta.status_code}): {meta.text[:300]}")
        info = meta.json() if isinstance(meta.json(), dict) else {}
        mime = str(info.get("mimeType") or "")
        name = str(info.get("name") or file_id)
        if mime in _EXPORTS:
            body = client.get(
                f"https://www.googleapis.com/drive/v3/files/{fid}/export",
                headers=headers,
                params={"mimeType": _EXPORTS[mime]},
            )
        elif mime.startswith(_TEXT_PREFIXES) or mime in _TEXT_MIMES:
            body = client.get(
                _FILE.format(file_id=fid),
                headers=headers,
                params={"alt": "media"},
            )
        else:
            return (
                f"SUCCESS: `{name}` id=`{file_id.strip()}` is {mime or 'a file'}. "
                "It is not text, so the bytes were left in Drive."
            )
    if body.status_code >= 400:
        raise RuntimeError(f"Drive read failed ({body.status_code}): {body.text[:300]}")
    text = body.text or ""
    start, end, chunk, total = slice_text_window(text, offset=offset, max_chars=max_chars)
    head = f"SUCCESS: `{name}` id=`{file_id.strip()}` {chars_label(start, end, total)}\n"
    if total == 0 and start == 0:
        return head + "\n(empty)"
    if start >= total:
        return head + "\n" + past_end_note(start, total)
    return head + "\n" + chunk + continuation_footer(start, end, total)


def execute_drive_tool(
    name: str,
    args: dict[str, Any],
    *,
    settings: Settings,
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
) -> str:
    user_id, denied = resolve_google_user_id(
        settings,
        account_user_id=account_user_id,
        speaker_id=speaker_id,
        chat_channel=chat_channel,
        quiet_host=True,
    )
    if denied or not user_id:
        return _product_denied(denied or "FAILED: Drive is unavailable in this context.")
    conn_or_err = _require_drive(settings, user_id)
    if isinstance(conn_or_err, str):
        return conn_or_err
    try:
        if name == "list_drive_files":
            return _list_files(
                settings,
                conn_or_err,
                query=str(args.get("query") or ""),
                max_files=max(1, min(20, int(args.get("max_files") or 10))),
            )
        if name == "read_drive_file":
            return _read_file(
                settings,
                conn_or_err,
                file_id=str(args.get("file_id") or ""),
                offset=max(0, int(args.get("offset") or 0)),
                max_chars=max(500, min(_MAX_READ, int(args.get("max_chars") or 6000))),
            )
        return _failed(f"unknown drive tool: {name}")
    except Exception as exc:  # noqa: BLE001 - tool boundary
        logger.warning("Drive tool %s failed for user=%s: %s", name, user_id, exc)
        return _failed(f"{name} — {exc}")


def build_drive_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="list_drive_files",
            description=(
                "List files in the signed-in human's Google Drive (read only). "
                "1:1 or Group as that human; a quiet sit reads the host. "
                "Optional name search."
            ),
            func=lambda query="", max_files=10: (
                "FAILED: list_drive_files must run through the agent tool runner"
            ),
            args_schema=ListDriveArgs,
        ),
        StructuredTool.from_function(
            name="read_drive_file",
            description=(
                "Read one Drive file as text by id (Docs export as text, Sheets as CSV). "
                "Non-text files return the name only. Read only. "
                "A long file is one window: the result names the total length "
                "and a continuation offset. Call again with offset=N to read on. "
                "The end of a window is not the end of the file."
            ),
            func=lambda file_id, offset=0, max_chars=6000: (
                "FAILED: read_drive_file must run through the agent tool runner"
            ),
            args_schema=ReadDriveArgs,
        ),
    ]
