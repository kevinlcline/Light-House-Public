"""Read the house Google arrival log. The log itself never wakes a light."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.config import Settings
from light_house.google_activity.log import read_activity
from light_house.humans.google_api import resolve_google_user_id

logger = logging.getLogger(__name__)

GOOGLE_ACTIVITY_TOOL_NAMES = frozenset({"read_google_activity"})


class ReadGoogleActivityArgs(BaseModel):
    max_entries: int = Field(
        default=20,
        ge=1,
        le=40,
        description="How many recent log lines to return (1–40).",
    )


def _product_denied(denied: str) -> str:
    return (
        denied.replace("Calendar tools", "The Google activity log")
        .replace("calendar tools", "the Google activity log")
        .replace("a calendar", "the Google activity log")
        .replace("calendar is", "the Google activity log is")
    )


def execute_google_activity_tool(
    name: str,
    args: dict[str, Any],
    *,
    settings: Settings,
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
) -> str:
    if name != "read_google_activity":
        return f"FAILED: unknown google activity tool: {name}"
    user_id, denied = resolve_google_user_id(
        settings,
        account_user_id=account_user_id,
        speaker_id=speaker_id,
        chat_channel=chat_channel,
        quiet_host=True,
    )
    if denied or not user_id:
        return _product_denied(denied or "FAILED: The Google activity log is unavailable.")
    try:
        limit = max(1, min(40, int(args.get("max_entries") or 20)))
    except (TypeError, ValueError):
        limit = 20
    rows = read_activity(settings, user_id, max_entries=limit)
    if not rows:
        return (
            f"SUCCESS: No Google activity is logged yet for {user_id}. "
            "The house writes mail, Drive changes, and task changes here. "
            "It does not wake anyone."
        )
    lines = [f"SUCCESS: {len(rows)} Google activity line(s) for {user_id}."]
    for row in rows:
        kind = str(row.get("kind") or "note")
        when = str(row.get("at") or "")
        item_id = str(row.get("id") or "")
        summary = str(row.get("summary") or "")
        ident = f" id={item_id}" if item_id else ""
        lines.append(f"- [{kind}] {when}{ident} {summary}")
    return "\n".join(lines)


def build_google_activity_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="read_google_activity",
            description=(
                "Read the house log of new Gmail, Drive file changes, and Google Task changes. "
                "The house writes this log on a timer and does not wake anyone. "
                "1:1 or Group reads the signed-in human. A quiet sit reads the host."
            ),
            func=lambda max_entries=20: (
                "FAILED: read_google_activity must run through the agent tool runner"
            ),
            args_schema=ReadGoogleActivityArgs,
        ),
    ]
