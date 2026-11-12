"""Per-human Google Tasks. Same Connect Google sign-in as the calendar."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.config import Settings
from light_house.humans.google_api import google_access_token, resolve_google_user_id
from light_house.humans.google_calendar_oauth import (
    scope_grants_tasks,
    scope_grants_tasks_write,
)
from light_house.humans.tools_store import calendar_public_view, get_calendar_connection

logger = logging.getLogger(__name__)

TASKS_TOOL_NAMES = frozenset(
    {
        "list_google_tasklists",
        "list_google_tasks",
        "create_google_task",
        "complete_google_task",
    }
)

_LISTS = "https://tasks.googleapis.com/tasks/v1/users/@me/lists"
_TASKS = "https://tasks.googleapis.com/tasks/v1/lists/{tasklist}/tasks"
_TASK = "https://tasks.googleapis.com/tasks/v1/lists/{tasklist}/tasks/{task}"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_NOT_CONNECTED = (
    "FAILED: This human has not connected Google Tasks yet. "
    "Tell them (briefly) to open **My tools** and click **Reconnect Google** — "
    "you cannot connect it for them."
)
_NOT_WRITE = (
    "FAILED: Google Tasks is connected for reading, not for changes. "
    "They can **Reconnect Google** on **My tools** and leave Tasks checked."
)


class ListTasklistsArgs(BaseModel):
    max_lists: int = Field(default=10, ge=1, le=20)


class ListTasksArgs(BaseModel):
    tasklist_id: str = Field(
        default="@default",
        max_length=200,
        description="Task list id from list_google_tasklists, or @default.",
    )
    show_completed: bool = Field(default=False)
    max_tasks: int = Field(default=15, ge=1, le=30)


class CreateTaskArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    notes: str = Field(default="", max_length=4000)
    due: str = Field(
        default="",
        max_length=40,
        description="Optional due date YYYY-MM-DD.",
    )
    tasklist_id: str = Field(default="@default", max_length=200)


class CompleteTaskArgs(BaseModel):
    task_id: str = Field(..., min_length=2, max_length=200)
    tasklist_id: str = Field(default="@default", max_length=200)


def _failed(msg: str) -> str:
    if msg.startswith("FAILED:"):
        return msg
    return f"FAILED: {msg}"


def _product_denied(denied: str) -> str:
    return (
        denied.replace("Calendar tools", "Tasks tools")
        .replace("calendar tools", "Tasks tools")
        .replace("a calendar", "Tasks")
        .replace("calendar is", "Tasks is")
    )


def _require_tasks(settings: Settings, user_id: str, *, write: bool) -> Any | str:
    view = calendar_public_view(settings, user_id)
    if not view.has_secrets or view.provider != "google" or not view.enabled:
        return _NOT_CONNECTED
    conn = get_calendar_connection(settings, user_id)
    scopes = list(conn.scopes) if conn is not None else []
    if conn is None or not scope_grants_tasks(scopes):
        return _NOT_CONNECTED
    if write and not scope_grants_tasks_write(scopes):
        return _NOT_WRITE
    return conn


def _due_rfc3339(due: str) -> str | None:
    text = due.strip()
    if not text:
        return None
    if _DATE_RE.match(text):
        return f"{text}T00:00:00.000Z"
    if "T" in text:
        return text
    raise ValueError("due must be YYYY-MM-DD")


def _list_lists(settings: Settings, conn: Any, *, max_lists: int) -> str:
    token = google_access_token(settings, conn)
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(_LISTS, headers={"Authorization": f"Bearer {token}"})
    if resp.status_code >= 400:
        raise RuntimeError(f"Task lists failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json() if isinstance(resp.json(), dict) else {}
    items = [item for item in (data.get("items") or []) if isinstance(item, dict)][:max_lists]
    lines = [f"SUCCESS: {len(items)} task list(s)."]
    for item in items:
        lines.append(f"- id={item.get('id')} | {item.get('title') or 'Tasks'}")
    return "\n".join(lines)


def _list_tasks(
    settings: Settings,
    conn: Any,
    *,
    tasklist_id: str,
    show_completed: bool,
    max_tasks: int,
) -> str:
    token = google_access_token(settings, conn)
    url = _TASKS.format(tasklist=quote(tasklist_id.strip() or "@default", safe="@"))
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params={
                "maxResults": str(max_tasks),
                "showCompleted": "true" if show_completed else "false",
                "showHidden": "false",
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Tasks failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json() if isinstance(resp.json(), dict) else {}
    items = [item for item in (data.get("items") or []) if isinstance(item, dict)]
    lines = [f"SUCCESS: {len(items)} task(s) on `{tasklist_id}`."]
    for item in items:
        status = item.get("status") or "needsAction"
        due = item.get("due") or ""
        lines.append(
            f"- id={item.get('id')} | {status} | {due} | {item.get('title') or '(untitled)'}"
        )
    return "\n".join(lines)


def _create_task(
    settings: Settings,
    conn: Any,
    *,
    tasklist_id: str,
    title: str,
    notes: str,
    due: str,
) -> str:
    token = google_access_token(settings, conn)
    url = _TASKS.format(tasklist=quote(tasklist_id.strip() or "@default", safe="@"))
    payload: dict[str, str] = {"title": title.strip()}
    if notes.strip():
        payload["notes"] = notes
    due_value = _due_rfc3339(due)
    if due_value:
        payload["due"] = due_value
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Create task failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json() if isinstance(resp.json(), dict) else {}
    return f"SUCCESS: Task `{data.get('title') or title}` id=`{data.get('id') or '?'}`"


def _complete_task(settings: Settings, conn: Any, *, tasklist_id: str, task_id: str) -> str:
    token = google_access_token(settings, conn)
    url = _TASK.format(
        tasklist=quote(tasklist_id.strip() or "@default", safe="@"),
        task=quote(task_id.strip(), safe=""),
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.patch(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"status": "completed"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Complete task failed ({resp.status_code}): {resp.text[:300]}")
    return f"SUCCESS: Completed task id=`{task_id.strip()}`"


def execute_tasks_tool(
    name: str,
    args: dict[str, Any],
    *,
    settings: Settings,
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
) -> str:
    writing = name in {"create_google_task", "complete_google_task"}
    user_id, denied = resolve_google_user_id(
        settings,
        account_user_id=account_user_id,
        speaker_id=speaker_id,
        chat_channel=chat_channel,
        quiet_host=not writing,
    )
    if denied or not user_id:
        return _product_denied(denied or "FAILED: Tasks are unavailable in this context.")
    conn_or_err = _require_tasks(settings, user_id, write=writing)
    if isinstance(conn_or_err, str):
        return conn_or_err
    try:
        if name == "list_google_tasklists":
            return _list_lists(
                settings,
                conn_or_err,
                max_lists=max(1, min(20, int(args.get("max_lists") or 10))),
            )
        if name == "list_google_tasks":
            return _list_tasks(
                settings,
                conn_or_err,
                tasklist_id=str(args.get("tasklist_id") or "@default"),
                show_completed=bool(args.get("show_completed")),
                max_tasks=max(1, min(30, int(args.get("max_tasks") or 15))),
            )
        if name == "create_google_task":
            return _create_task(
                settings,
                conn_or_err,
                tasklist_id=str(args.get("tasklist_id") or "@default"),
                title=str(args.get("title") or ""),
                notes=str(args.get("notes") or ""),
                due=str(args.get("due") or ""),
            )
        if name == "complete_google_task":
            return _complete_task(
                settings,
                conn_or_err,
                tasklist_id=str(args.get("tasklist_id") or "@default"),
                task_id=str(args.get("task_id") or ""),
            )
        return _failed(f"unknown tasks tool: {name}")
    except Exception as exc:  # noqa: BLE001 - tool boundary
        logger.warning("Tasks tool %s failed for user=%s: %s", name, user_id, exc)
        return _failed(f"{name} — {exc}")


def build_tasks_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="list_google_tasklists",
            description=(
                "List the signed-in human's Google Task lists. "
                "1:1 or Group as that human; a quiet sit reads the host."
            ),
            func=lambda max_lists=10: (
                "FAILED: list_google_tasklists must run through the agent tool runner"
            ),
            args_schema=ListTasklistsArgs,
        ),
        StructuredTool.from_function(
            name="list_google_tasks",
            description=(
                "List tasks on one Google Task list (default @default). "
                "1:1 or Group as that human; a quiet sit reads the host."
            ),
            func=lambda tasklist_id="@default", show_completed=False, max_tasks=15: (
                "FAILED: list_google_tasks must run through the agent tool runner"
            ),
            args_schema=ListTasksArgs,
        ),
        StructuredTool.from_function(
            name="create_google_task",
            description=(
                "Add a task on the signed-in human's Google Tasks. "
                "Optional due date YYYY-MM-DD. On a quiet sit this waits."
            ),
            func=lambda title, notes="", due="", tasklist_id="@default": (
                "FAILED: create_google_task must run through the agent tool runner"
            ),
            args_schema=CreateTaskArgs,
        ),
        StructuredTool.from_function(
            name="complete_google_task",
            description=(
                "Mark one Google task completed. On a quiet sit this waits."
            ),
            func=lambda task_id, tasklist_id="@default": (
                "FAILED: complete_google_task must run through the agent tool runner"
            ),
            args_schema=CompleteTaskArgs,
        ),
    ]
