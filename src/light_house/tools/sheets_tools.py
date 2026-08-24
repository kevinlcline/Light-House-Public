"""Per-human Google Sheets tools — private 1:1 only, same Google connect as Docs."""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.config import Settings
from light_house.humans.google_calendar_oauth import scope_grants_sheets
from light_house.humans.tools_store import (
    calendar_public_view,
    get_calendar_connection,
    resolve_google_client_credentials,
)
from light_house.tools.calendar_tools import calendar_access_denied_reason

logger = logging.getLogger(__name__)

SHEETS_TOOL_NAMES = frozenset(
    {
        "list_google_sheets",
        "read_google_sheet",
        "create_google_sheet",
        "append_google_sheet",
    }
)

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
_DRIVE_FILES_API = "https://www.googleapis.com/drive/v3/files"
_SHEET_ID_RE = re.compile(
    r"(?:https?://)?(?:docs\.google\.com/spreadsheets/d/)?([a-zA-Z0-9_-]{10,})"
)

_UNAVAILABLE_NOT_CONNECTED = (
    "FAILED: This human has not connected Google Sheets yet. "
    "Tell them (briefly) to open **My tools** and click **Connect Google** "
    "(or Reconnect if they connected earlier without Sheets) — "
    "you cannot connect it for them."
)

_UNAVAILABLE_DISABLED = (
    "FAILED: This human's Google connection is disabled. "
    "They can re-enable it on their **My tools** page."
)

_MAX_CELLS = 2000
_MAX_OUT_CHARS = 12000


class ListGoogleSheetsArgs(BaseModel):
    query: str = Field(
        default="",
        max_length=200,
        description="Optional name search (matches spreadsheet title). Empty = recent sheets.",
    )
    max_sheets: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Max spreadsheets to return (1–30). Default 10.",
    )


class ReadGoogleSheetArgs(BaseModel):
    spreadsheet_id_or_url: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Spreadsheet id or full docs.google.com/spreadsheets URL.",
    )
    range: str = Field(
        default="",
        max_length=200,
        description=(
            "A1 range to read, e.g. Sheet1!A1:D20. "
            "Empty = first sheet, first 50 rows × 20 columns."
        ),
    )


class CreateGoogleSheetArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Spreadsheet title.")
    headers: str = Field(
        default="",
        max_length=2000,
        description="Optional header row as comma-separated values, e.g. Date,Task,Notes.",
    )
    rows_csv: str = Field(
        default="",
        max_length=50000,
        description="Optional extra rows as CSV text (one row per line).",
    )


class AppendGoogleSheetArgs(BaseModel):
    spreadsheet_id_or_url: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Spreadsheet id or full spreadsheets URL.",
    )
    rows_csv: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Rows to append as CSV text (one row per line).",
    )
    range: str = Field(
        default="Sheet1",
        max_length=200,
        description="Sheet/tab name or range to append into (default Sheet1).",
    )


def _tool_failed(msg: str) -> str:
    if msg.startswith("FAILED:"):
        return msg
    return f"FAILED: {msg}"


def _extract_spreadsheet_id(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise ValueError("spreadsheet_id_or_url is required")
    match = _SHEET_ID_RE.search(text)
    if not match:
        raise ValueError("Could not parse a Google Spreadsheet id from that value")
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


def _require_sheets_connection(settings: Settings, *, account_user_id: str) -> Any | str:
    view = calendar_public_view(settings, account_user_id)
    if not view.has_secrets or view.provider != "google":
        return _UNAVAILABLE_NOT_CONNECTED
    if not view.enabled:
        return _UNAVAILABLE_DISABLED
    if not view.sheets_connected:
        return _UNAVAILABLE_NOT_CONNECTED
    conn = get_calendar_connection(settings, account_user_id)
    if conn is None or not scope_grants_sheets(list(conn.scopes)):
        return _UNAVAILABLE_NOT_CONNECTED
    return conn


def _parse_csv_rows(rows_csv: str) -> list[list[str]]:
    text = (rows_csv or "").strip()
    if not text:
        return []
    reader = csv.reader(io.StringIO(text))
    rows = [[cell.strip() for cell in row] for row in reader if any(c.strip() for c in row)]
    return rows


def _values_to_tsv(values: list[list[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for row in values:
        writer.writerow(["" if c is None else str(c) for c in row])
    return buf.getvalue().rstrip("\n")


def _list_sheets(settings: Settings, conn: Any, *, query: str, max_sheets: int) -> str:
    token = _google_access_token(settings, conn)
    q_parts = ["mimeType='application/vnd.google-apps.spreadsheet'", "trashed=false"]
    q = (query or "").strip().replace("'", "\\'")
    if q:
        q_parts.append(f"name contains '{q}'")
    params = {
        "q": " and ".join(q_parts),
        "pageSize": max_sheets,
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
        return f"SUCCESS: No Google Sheets found{label}."
    lines = [f"SUCCESS: Google Sheets ({len(files)}):"]
    for f in files:
        name = str(f.get("name") or "(untitled)")
        sid = str(f.get("id") or "")
        modified = str(f.get("modifiedTime") or "")
        link = str(
            f.get("webViewLink")
            or f"https://docs.google.com/spreadsheets/d/{sid}/edit"
        )
        lines.append(f"- {name} — id=`{sid}` modified={modified}\n  {link}")
    return "\n".join(lines)


def _first_sheet_title(settings: Settings, conn: Any, *, spreadsheet_id: str) -> str:
    token = _google_access_token(settings, conn)
    with httpx.Client(timeout=25.0) as client:
        resp = client.get(
            f"{_SHEETS_API}/{quote(spreadsheet_id, safe='')}",
            headers={"Authorization": f"Bearer {token}"},
            params={"fields": "sheets.properties.title"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google Sheets metadata failed ({resp.status_code}): {resp.text[:300]}"
        )
    sheets = resp.json().get("sheets") or []
    if not sheets:
        return "Sheet1"
    props = (sheets[0] or {}).get("properties") or {}
    title = str(props.get("title") or "Sheet1").strip() or "Sheet1"
    return title


def _read_sheet(
    settings: Settings,
    conn: Any,
    *,
    spreadsheet_id_or_url: str,
    range_a1: str,
) -> str:
    sid = _extract_spreadsheet_id(spreadsheet_id_or_url)
    rng = (range_a1 or "").strip()
    if not rng:
        title = _first_sheet_title(settings, conn, spreadsheet_id=sid)
        # Quote sheet title if needed.
        safe_title = f"'{title}'" if any(c in title for c in " !'") else title
        rng = f"{safe_title}!A1:T50"
    token = _google_access_token(settings, conn)
    with httpx.Client(timeout=25.0) as client:
        resp = client.get(
            f"{_SHEETS_API}/{quote(sid, safe='')}/values/{quote(rng, safe='!:')}",
            headers={"Authorization": f"Bearer {token}"},
            params={"majorDimension": "ROWS"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google Sheets read failed ({resp.status_code}): {resp.text[:300]}"
        )
    data = resp.json()
    values = data.get("values") or []
    if not isinstance(values, list):
        values = []
    cell_count = sum(len(r) for r in values if isinstance(r, list))
    if cell_count > _MAX_CELLS:
        values = values[:80]
        trimmed = []
        count = 0
        for row in values:
            if not isinstance(row, list):
                continue
            if count >= _MAX_CELLS:
                break
            take = row[: max(0, _MAX_CELLS - count)]
            trimmed.append(take)
            count += len(take)
        values = trimmed
    tsv = _values_to_tsv(values)
    truncated = False
    if len(tsv) > _MAX_OUT_CHARS:
        tsv = tsv[:_MAX_OUT_CHARS]
        truncated = True
    link = f"https://docs.google.com/spreadsheets/d/{sid}/edit"
    note = " (truncated)" if truncated else ""
    return (
        f"SUCCESS: Google Sheet id=`{sid}` range=`{data.get('range') or rng}`{note}\n"
        f"Link: {link}\n\n"
        f"{tsv or '(empty range)'}"
    )


def _create_sheet(
    settings: Settings,
    conn: Any,
    *,
    title: str,
    headers: str,
    rows_csv: str,
) -> str:
    token = _google_access_token(settings, conn)
    headers_row: list[str] = []
    if headers.strip():
        parsed_headers = _parse_csv_rows(headers.strip())
        headers_row = parsed_headers[0] if parsed_headers else [
            c.strip() for c in headers.split(",")
        ]
    body_rows = _parse_csv_rows(rows_csv)
    values: list[list[str]] = []
    if headers_row and any(headers_row):
        values.append(headers_row)
    values.extend(body_rows)

    auth = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with httpx.Client(timeout=25.0) as client:
        resp = client.post(
            _SHEETS_API,
            headers=auth,
            json={"properties": {"title": title.strip()}},
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Google Sheets create failed ({resp.status_code}): {resp.text[:300]}"
            )
        created = resp.json()
        sid = str(created.get("spreadsheetId") or "")
        if not sid:
            raise RuntimeError("Google Sheets create returned no spreadsheetId")
        if values:
            upd = client.put(
                f"{_SHEETS_API}/{quote(sid, safe='')}/values/{quote('Sheet1!A1', safe='!:')}",
                headers=auth,
                params={"valueInputOption": "USER_ENTERED"},
                json={"values": values},
            )
            if upd.status_code >= 400:
                raise RuntimeError(
                    f"Sheet created but writing values failed ({upd.status_code}): "
                    f"{upd.text[:300]}"
                )
    link = f"https://docs.google.com/spreadsheets/d/{sid}/edit"
    return f"SUCCESS: Created Google Sheet `{title.strip()}` id=`{sid}`\nLink: {link}"


def _append_sheet(
    settings: Settings,
    conn: Any,
    *,
    spreadsheet_id_or_url: str,
    rows_csv: str,
    range_a1: str,
) -> str:
    sid = _extract_spreadsheet_id(spreadsheet_id_or_url)
    rows = _parse_csv_rows(rows_csv)
    if not rows:
        return _tool_failed("rows_csv produced no rows")
    rng = (range_a1 or "Sheet1").strip() or "Sheet1"
    token = _google_access_token(settings, conn)
    with httpx.Client(timeout=25.0) as client:
        resp = client.post(
            f"{_SHEETS_API}/{quote(sid, safe='')}/values/{quote(rng, safe='!:')}:append",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            params={
                "valueInputOption": "USER_ENTERED",
                "insertDataOption": "INSERT_ROWS",
            },
            json={"values": rows},
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google Sheets append failed ({resp.status_code}): {resp.text[:300]}"
        )
    updated = resp.json().get("updates") or {}
    link = f"https://docs.google.com/spreadsheets/d/{sid}/edit"
    return (
        f"SUCCESS: Appended {len(rows)} row(s) to id=`{sid}` "
        f"updatedRange=`{updated.get('updatedRange') or rng}`\n"
        f"Link: {link}"
    )


def execute_sheets_tool(
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
        return (
            denied.replace("Calendar tools", "Google Sheets tools")
            .replace("calendar access", "Google Sheets access")
            .replace("calendar", "Google Sheets")
        )
    assert account_user_id is not None
    conn_or_err = _require_sheets_connection(settings, account_user_id=account_user_id)
    if isinstance(conn_or_err, str):
        return conn_or_err
    conn = conn_or_err
    try:
        if name == "list_google_sheets":
            return _list_sheets(
                settings,
                conn,
                query=str(args.get("query") or ""),
                max_sheets=max(1, min(30, int(args.get("max_sheets") or 10))),
            )
        if name == "read_google_sheet":
            return _read_sheet(
                settings,
                conn,
                spreadsheet_id_or_url=str(
                    args.get("spreadsheet_id_or_url") or args.get("sheet_id_or_url") or ""
                ),
                range_a1=str(args.get("range") or ""),
            )
        if name == "create_google_sheet":
            return _create_sheet(
                settings,
                conn,
                title=str(args.get("title") or ""),
                headers=str(args.get("headers") or ""),
                rows_csv=str(args.get("rows_csv") or ""),
            )
        if name == "append_google_sheet":
            return _append_sheet(
                settings,
                conn,
                spreadsheet_id_or_url=str(
                    args.get("spreadsheet_id_or_url") or args.get("sheet_id_or_url") or ""
                ),
                rows_csv=str(args.get("rows_csv") or ""),
                range_a1=str(args.get("range") or "Sheet1"),
            )
        return _tool_failed(f"unknown sheets tool: {name}")
    except Exception as exc:  # noqa: BLE001 - tool boundary
        logger.warning(
            "Sheets tool %s failed for user=%s: %s", name, account_user_id, exc
        )
        return _tool_failed(f"{name} — {exc}")


def build_sheets_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="list_google_sheets",
            description=(
                "List the signed-in human's Google Sheets (private 1:1 only; not group; "
                "not guests). Optional title query."
            ),
            func=lambda query="", max_sheets=10: (
                "FAILED: list_google_sheets must run through the agent tool runner"
            ),
            args_schema=ListGoogleSheetsArgs,
        ),
        StructuredTool.from_function(
            name="read_google_sheet",
            description=(
                "Read cells from a Google Sheet by id/URL and optional A1 range "
                "(private 1:1 only; not group; not guests). Returns TSV text."
            ),
            func=lambda spreadsheet_id_or_url, **_kwargs: (
                "FAILED: read_google_sheet must run through the agent tool runner"
            ),
            args_schema=ReadGoogleSheetArgs,
        ),
        StructuredTool.from_function(
            name="create_google_sheet",
            description=(
                "Create a Google Sheet with a title and optional header/rows CSV "
                "(private 1:1 only; not group; not guests)."
            ),
            func=lambda title, headers="", rows_csv="": (
                "FAILED: create_google_sheet must run through the agent tool runner"
            ),
            args_schema=CreateGoogleSheetArgs,
        ),
        StructuredTool.from_function(
            name="append_google_sheet",
            description=(
                "Append CSV rows to a Google Sheet "
                "(private 1:1 only; not group; not guests)."
            ),
            func=lambda spreadsheet_id_or_url, rows_csv, **_kwargs: (
                "FAILED: append_google_sheet must run through the agent tool runner"
            ),
            args_schema=AppendGoogleSheetArgs,
        ),
    ]
