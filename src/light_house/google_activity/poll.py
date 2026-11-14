"""Notice new mail, Drive changes, and tasks. Append a log. Do not start a light."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from light_house.config import Settings
from light_house.google_activity.log import (
    append_activity,
    load_cursor,
    save_cursor,
    utc_stamp,
)
from light_house.humans.google_api import google_access_token
from light_house.humans.google_calendar_oauth import (
    scope_grants_drive,
    scope_grants_gmail,
    scope_grants_tasks,
)
from light_house.humans.tools_store import list_google_connections

logger = logging.getLogger(__name__)

_GMAIL_PROFILE = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
_GMAIL_HISTORY = "https://gmail.googleapis.com/gmail/v1/users/me/history"
_GMAIL_MESSAGE = "https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}"
_DRIVE_START = "https://www.googleapis.com/drive/v3/changes/startPageToken"
_DRIVE_CHANGES = "https://www.googleapis.com/drive/v3/changes"
_TASK_LISTS = "https://tasks.googleapis.com/tasks/v1/users/@me/lists"
_TASKS = "https://tasks.googleapis.com/tasks/v1/lists/{tasklist}/tasks"

_MAX_PAGES = 5
_MAX_LINES = 25


def poll_google_activity(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    now: datetime | None = None,
) -> int:
    """Check each connected Google account and append unseen arrivals.

    The first check only stores a cursor, so the existing mailbox and Drive
    are not copied into the log. Later checks append what is new.
    """
    moment = now or datetime.now(timezone.utc)
    own = client is None
    http = client or httpx.Client(timeout=20.0)
    written = 0
    try:
        for user_id, conn in list_google_connections(settings):
            try:
                token = google_access_token(settings, conn, client=http)
            except Exception as exc:
                logger.warning("Google activity token failed for user=%s: %s", user_id, exc)
                continue
            scopes = list(conn.scopes)
            cursor = load_cursor(settings, user_id)
            entries: list[dict[str, Any]] = []
            for granted, poller in (
                (scope_grants_gmail(scopes), _poll_gmail),
                (scope_grants_drive(scopes), _poll_drive),
                (scope_grants_tasks(scopes), _poll_tasks),
            ):
                if not granted:
                    continue
                try:
                    _merge(cursor, entries, poller(http, token, cursor, moment))
                except Exception as exc:
                    logger.warning(
                        "Google activity %s failed for user=%s: %s",
                        poller.__name__,
                        user_id,
                        exc,
                    )
            written += append_activity(settings, user_id, entries)
            save_cursor(settings, user_id, cursor)
    finally:
        if own:
            http.close()
    return written


def _merge(
    cursor: dict[str, str],
    entries: list[dict[str, Any]],
    result: tuple[list[dict[str, Any]], dict[str, str]] | None,
) -> None:
    if result is None:
        return
    found, updates = result
    entries.extend(found)
    cursor.update(updates)


def _auth_get(
    client: httpx.Client,
    url: str,
    token: str,
    params: dict[str, str] | list[tuple[str, str]] | None = None,
) -> httpx.Response:
    return client.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
    )


def _json_dict(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _entry(kind: str, item_id: str, summary: str, moment: datetime) -> dict[str, str]:
    return {
        "at": utc_stamp(moment),
        "kind": kind,
        "id": item_id,
        "summary": summary[:300],
    }


def _header(payload: dict[str, Any], name: str) -> str:
    for item in payload.get("headers") or []:
        if isinstance(item, dict) and str(item.get("name") or "").lower() == name.lower():
            return str(item.get("value") or "").strip()
    return ""


def _poll_gmail(
    client: httpx.Client,
    token: str,
    cursor: dict[str, str],
    moment: datetime,
) -> tuple[list[dict[str, Any]], dict[str, str]] | None:
    history_id = cursor.get("gmail_history_id") or ""
    if not history_id:
        resp = _auth_get(client, _GMAIL_PROFILE, token)
        if resp.status_code >= 400:
            logger.warning("Gmail profile failed: %s %s", resp.status_code, resp.text[:200])
            return None
        fresh = str(_json_dict(resp).get("historyId") or "").strip()
        if not fresh:
            return None
        return [], {"gmail_history_id": fresh, "gmail_page_token": ""}

    entries: list[dict[str, Any]] = []
    page = cursor.get("gmail_page_token") or ""
    newest = history_id
    caught_up = False
    for _ in range(_MAX_PAGES):
        params = {
            "startHistoryId": history_id,
            "historyTypes": "messageAdded",
            "maxResults": "20",
        }
        if page:
            params["pageToken"] = page
        resp = _auth_get(client, _GMAIL_HISTORY, token, params)
        if resp.status_code == 404:
            profile = _auth_get(client, _GMAIL_PROFILE, token)
            if profile.status_code >= 400:
                return None
            fresh = str(_json_dict(profile).get("historyId") or history_id)
            return (
                [
                    _entry(
                        "gmail",
                        "",
                        "Mail watch moved forward. Older arrivals were not listed.",
                        moment,
                    )
                ],
                {"gmail_history_id": fresh, "gmail_page_token": ""},
            )
        if resp.status_code >= 400:
            logger.warning("Gmail history failed: %s %s", resp.status_code, resp.text[:200])
            return None
        data = _json_dict(resp)
        newest = str(data.get("historyId") or newest)
        for record in data.get("history") or []:
            if not isinstance(record, dict):
                continue
            for added in record.get("messagesAdded") or []:
                if not isinstance(added, dict):
                    continue
                message = added.get("message") if isinstance(added.get("message"), dict) else {}
                message_id = str(message.get("id") or "").strip()
                if not message_id:
                    continue
                entries.append(_gmail_line(client, token, message_id, moment))
                if len(entries) >= _MAX_LINES:
                    break
            if len(entries) >= _MAX_LINES:
                break
        page = str(data.get("nextPageToken") or "")
        if not page or len(entries) >= _MAX_LINES:
            caught_up = not page
            break
    else:
        caught_up = not page
    if len(entries) >= _MAX_LINES and page:
        entries.append(
            _entry("gmail", "", "Further mail arrivals were left for the next check.", moment)
        )
    if caught_up:
        return entries, {"gmail_history_id": newest, "gmail_page_token": ""}
    return entries, {"gmail_history_id": history_id, "gmail_page_token": page}


def _gmail_line(
    client: httpx.Client,
    token: str,
    message_id: str,
    moment: datetime,
) -> dict[str, str]:
    resp = _auth_get(
        client,
        _GMAIL_MESSAGE.format(message_id=quote(message_id, safe="")),
        token,
        [
            ("format", "metadata"),
            ("metadataHeaders", "From"),
            ("metadataHeaders", "Subject"),
        ],
    )
    sender = ""
    subject = ""
    if resp.status_code < 400:
        payload = _json_dict(resp).get("payload")
        if isinstance(payload, dict):
            sender = _header(payload, "From")
            subject = _header(payload, "Subject")
    summary = f"From {sender or 'unknown'} — {subject or '(no subject)'}"
    return _entry("gmail", message_id, summary, moment)


def _poll_drive(
    client: httpx.Client,
    token: str,
    cursor: dict[str, str],
    moment: datetime,
) -> tuple[list[dict[str, Any]], dict[str, str]] | None:
    page = cursor.get("drive_page_token") or ""
    if not page:
        resp = _auth_get(client, _DRIVE_START, token)
        if resp.status_code >= 400:
            logger.warning("Drive start token failed: %s %s", resp.status_code, resp.text[:200])
            return None
        fresh = str(_json_dict(resp).get("startPageToken") or "").strip()
        if not fresh:
            return None
        return [], {"drive_page_token": fresh}

    entries: list[dict[str, Any]] = []
    for _ in range(_MAX_PAGES):
        resp = _auth_get(
            client,
            _DRIVE_CHANGES,
            token,
            {
                "pageToken": page,
                "pageSize": "20",
                "fields": "nextPageToken,newStartPageToken,changes(fileId,removed,file(name))",
            },
        )
        if resp.status_code == 410:
            restart = _auth_get(client, _DRIVE_START, token)
            if restart.status_code >= 400:
                return None
            fresh = str(_json_dict(restart).get("startPageToken") or "")
            return (
                [
                    _entry(
                        "drive",
                        "",
                        "Drive watch moved forward. Older file changes were not listed.",
                        moment,
                    )
                ],
                {"drive_page_token": fresh},
            )
        if resp.status_code >= 400:
            logger.warning("Drive changes failed: %s %s", resp.status_code, resp.text[:200])
            return None
        data = _json_dict(resp)
        for change in data.get("changes") or []:
            if not isinstance(change, dict):
                continue
            file_id = str(change.get("fileId") or "").strip()
            if change.get("removed"):
                entries.append(_entry("drive", file_id, f"File removed id={file_id}", moment))
            else:
                file_obj = change.get("file") if isinstance(change.get("file"), dict) else {}
                name = str(file_obj.get("name") or file_id or "untitled")
                entries.append(_entry("drive", file_id, f"File changed: {name}", moment))
            if len(entries) >= _MAX_LINES:
                break
        nxt = str(data.get("nextPageToken") or "")
        start = str(data.get("newStartPageToken") or "")
        if nxt and len(entries) < _MAX_LINES:
            page = nxt
            continue
        if nxt:
            return entries, {"drive_page_token": nxt}
        if start:
            return entries, {"drive_page_token": start}
        return entries, {"drive_page_token": page}
    return entries, {"drive_page_token": page}


def _poll_tasks(
    client: httpx.Client,
    token: str,
    cursor: dict[str, str],
    moment: datetime,
) -> tuple[list[dict[str, Any]], dict[str, str]] | None:
    watermark = cursor.get("tasks_watermark") or ""
    if not watermark:
        return [], {"tasks_watermark": utc_stamp(moment)}
    lists_resp = _auth_get(client, _TASK_LISTS, token)
    if lists_resp.status_code >= 400:
        logger.warning("Task lists failed: %s %s", lists_resp.status_code, lists_resp.text[:200])
        return None
    lists_body = _json_dict(lists_resp)
    entries: list[dict[str, Any]] = []
    newest = watermark
    for tasklist in lists_body.get("items") or []:
        if not isinstance(tasklist, dict):
            continue
        list_id = str(tasklist.get("id") or "").strip()
        list_title = str(tasklist.get("title") or "Tasks")
        if not list_id:
            continue
        resp = _auth_get(
            client,
            _TASKS.format(tasklist=quote(list_id, safe="")),
            token,
            {
                "updatedMin": watermark,
                "showCompleted": "true",
                "showHidden": "true",
                "maxResults": "20",
            },
        )
        if resp.status_code >= 400:
            logger.warning("Tasks list failed: %s %s", resp.status_code, resp.text[:200])
            continue
        body = _json_dict(resp)
        for task in body.get("items") or []:
            if not isinstance(task, dict):
                continue
            updated = str(task.get("updated") or "")
            if updated and updated <= watermark:
                continue
            title = str(task.get("title") or "(untitled)")
            status = str(task.get("status") or "")
            task_id = str(task.get("id") or "")
            tail = "completed" if status == "completed" else "updated"
            entries.append(
                _entry("task", task_id, f"{list_title}: {title} ({tail})", moment)
            )
            if updated > newest:
                newest = updated
            if len(entries) >= _MAX_LINES:
                break
        if len(entries) >= _MAX_LINES:
            break
    if newest == watermark and not entries:
        return [], {}
    return entries, {"tasks_watermark": newest if entries else watermark}
