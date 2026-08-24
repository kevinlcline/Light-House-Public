"""Per-human calendar tools — only in 1:1 with an account holder who connected a calendar."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.config import Settings
from light_house.humans.tools_store import (
    CalendarConnection,
    calendar_public_view,
    get_calendar_connection,
    resolve_google_client_credentials,
)

logger = logging.getLogger(__name__)

# Local guest-slot check avoids importing house.guests (circular via group_chat).
_GUEST_SPEAKER_RE = re.compile(r"^guest-[1-9]\d{0,2}$")


def _is_guest_speaker_id(speaker_id: str | None) -> bool:
    return bool(_GUEST_SPEAKER_RE.match((speaker_id or "").strip()))

CALENDAR_TOOL_NAMES = frozenset(
    {
        "list_calendar_events",
        "create_calendar_event",
    }
)

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"

_UNAVAILABLE_GROUP = (
    "FAILED: Calendar tools are not available in group chat. "
    "Tell the human gently that calendar access only works in a private 1:1 chat "
    "with a light — not in the group room."
)

_UNAVAILABLE_GUEST = (
    "FAILED: Calendar tools are not available while a guest is speaking. "
    "Tell the guest briefly that calendar tools are only for signed-in household "
    "accounts in private 1:1 chat — guests cannot use them."
)

_UNAVAILABLE_NO_ACCOUNT = (
    "FAILED: Calendar tools need a signed-in household account in private 1:1 chat. "
    "Tell the human calendar is unavailable in this context."
)

_UNAVAILABLE_NOT_CONNECTED = (
    "FAILED: This human has not connected a calendar yet. "
    "Tell them (briefly) to open **My tools** and click **Connect Google** "
    "(or connect CalDAV/ICS there) — you cannot connect it for them."
)

_UNAVAILABLE_DISABLED = (
    "FAILED: This human's calendar connection is disabled. "
    "They can re-enable it on their **My tools** page."
)

_UNAVAILABLE_ICS_WRITE = (
    "FAILED: This human connected a read-only ICS feed. "
    "Listing events works; creating events does not. "
    "They can switch to Google or CalDAV on **My tools** if they need writes."
)

_UNAVAILABLE_CALDAV_WRITE = (
    "FAILED: Creating events via CalDAV is not supported yet. "
    "Listing works; for creates they should use Google Calendar on **My tools**, "
    "or add the event themselves."
)


class ListCalendarEventsArgs(BaseModel):
    days: int = Field(
        default=7,
        ge=1,
        le=31,
        description="How many days ahead to list (1–31). Default 7.",
    )
    max_events: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Max events to return (1–50). Default 15.",
    )


class CreateCalendarEventArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Event title.")
    start: str = Field(
        ...,
        description="Start time as ISO-8601 (e.g. 2026-08-07T15:00:00-07:00).",
    )
    end: str = Field(
        ...,
        description="End time as ISO-8601 (e.g. 2026-08-07T16:00:00-07:00).",
    )
    description: str = Field(
        default="",
        max_length=4000,
        description="Optional event description.",
    )
    location: str = Field(
        default="",
        max_length=500,
        description="Optional location.",
    )


def _tool_failed(msg: str) -> str:
    if msg.startswith("FAILED:"):
        return msg
    return f"FAILED: {msg}"


def calendar_access_denied_reason(
    *,
    account_user_id: str | None,
    speaker_id: str | None,
    chat_channel: str | None,
) -> str | None:
    """Return a FAILED message if calendar tools must not run; else None."""
    channel = (chat_channel or "dm").strip().lower()
    if channel == "group":
        return _UNAVAILABLE_GROUP
    if channel != "dm":
        return _UNAVAILABLE_NO_ACCOUNT
    if speaker_id and _is_guest_speaker_id(str(speaker_id)):
        return _UNAVAILABLE_GUEST
    if not (account_user_id or "").strip():
        return _UNAVAILABLE_NO_ACCOUNT
    return None


def _require_connection(
    settings: Settings,
    *,
    account_user_id: str,
) -> CalendarConnection | str:
    view = calendar_public_view(settings, account_user_id)
    if not view.has_secrets:
        return _UNAVAILABLE_NOT_CONNECTED
    if not view.enabled:
        return _UNAVAILABLE_DISABLED
    conn = get_calendar_connection(settings, account_user_id)
    if conn is None:
        return _UNAVAILABLE_NOT_CONNECTED
    return conn


def _google_access_token(settings: Settings, conn: CalendarConnection) -> str:
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
    data = resp.json()
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Google token refresh returned no access_token")
    return token


def _parse_iso(dt: str) -> datetime:
    raw = (dt or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_event_line(summary: str, start: str, end: str, location: str = "") -> str:
    loc = f" @ {location}" if location else ""
    return f"- {start} → {end}: {summary}{loc}"


def _list_google_events(
    settings: Settings,
    conn: CalendarConnection,
    *,
    days: int,
    max_events: int,
) -> str:
    token = _google_access_token(settings, conn)
    now = datetime.now(timezone.utc)
    time_min = now.isoformat().replace("+00:00", "Z")
    time_max = (now + timedelta(days=days)).isoformat().replace("+00:00", "Z")
    cal_id = quote(conn.calendar_id, safe="@.")
    url = _GOOGLE_EVENTS_URL.format(calendar_id=cal_id)
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params={
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": max_events,
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google Calendar list failed ({resp.status_code}): {resp.text[:300]}"
        )
    items = resp.json().get("items") or []
    if not items:
        return (
            f"SUCCESS: No events in the next {days} day(s) "
            f"on calendar `{conn.calendar_id}`."
        )
    lines = [
        f"SUCCESS: Upcoming events (next {days} day(s), calendar `{conn.calendar_id}`):"
    ]
    for item in items:
        summary = str(item.get("summary") or "(no title)")
        start_obj = item.get("start") or {}
        end_obj = item.get("end") or {}
        start = str(start_obj.get("dateTime") or start_obj.get("date") or "?")
        end = str(end_obj.get("dateTime") or end_obj.get("date") or "?")
        location = str(item.get("location") or "")
        lines.append(_format_event_line(summary, start, end, location))
    return "\n".join(lines)


def _create_google_event(
    settings: Settings,
    conn: CalendarConnection,
    *,
    title: str,
    start: str,
    end: str,
    description: str,
    location: str,
) -> str:
    token = _google_access_token(settings, conn)
    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    if end_dt <= start_dt:
        return _tool_failed("end must be after start")
    body: dict[str, Any] = {
        "summary": title.strip(),
        "start": {"dateTime": start_dt.isoformat()},
        "end": {"dateTime": end_dt.isoformat()},
    }
    if description.strip():
        body["description"] = description.strip()
    if location.strip():
        body["location"] = location.strip()
    cal_id = quote(conn.calendar_id, safe="@.")
    url = _GOOGLE_EVENTS_URL.format(calendar_id=cal_id)
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google Calendar create failed ({resp.status_code}): {resp.text[:300]}"
        )
    created = resp.json()
    html_link = created.get("htmlLink") or ""
    event_id = created.get("id") or ""
    link_bit = f" Link: {html_link}" if html_link else ""
    return (
        f"SUCCESS: Created event `{title.strip()}` "
        f"({start_dt.isoformat()} → {end_dt.isoformat()}) "
        f"on calendar `{conn.calendar_id}` (id={event_id}).{link_bit}"
    )


_VEVENT_RE = re.compile(
    r"BEGIN:VEVENT(.*?)END:VEVENT",
    re.DOTALL | re.IGNORECASE,
)


def _unfold_ics(text: str) -> str:
    return re.sub(r"\r?\n[ \t]", "", text.replace("\r\n", "\n"))


def _ics_field(block: str, name: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(name)}(?:;[^:]*)?:(.+)$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(block)
    if not match:
        return ""
    return match.group(1).strip()


def _list_ics_events(
    conn: CalendarConnection,
    *,
    days: int,
    max_events: int,
) -> str:
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        resp = client.get(conn.url)
    if resp.status_code >= 400:
        raise RuntimeError(f"ICS fetch failed ({resp.status_code})")
    text = _unfold_ics(resp.text)
    now = datetime.now(timezone.utc)
    until = now + timedelta(days=days)
    events: list[tuple[datetime, str, str, str]] = []
    for match in _VEVENT_RE.finditer(text):
        block = match.group(1)
        summary = _ics_field(block, "SUMMARY") or "(no title)"
        dtstart = _ics_field(block, "DTSTART")
        dtend = _ics_field(block, "DTEND")
        location = _ics_field(block, "LOCATION")
        if not dtstart:
            continue
        try:
            start_dt = _parse_ics_dt(dtstart)
            end_dt = _parse_ics_dt(dtend) if dtend else start_dt
        except ValueError:
            continue
        if end_dt < now or start_dt > until:
            continue
        events.append((start_dt, summary, end_dt.isoformat(), location))
    events.sort(key=lambda row: row[0])
    events = events[:max_events]
    if not events:
        return f"SUCCESS: No events in the next {days} day(s) on the ICS feed."
    lines = [f"SUCCESS: Upcoming events (next {days} day(s), ICS feed):"]
    for start_dt, summary, end_s, location in events:
        lines.append(_format_event_line(summary, start_dt.isoformat(), end_s, location))
    return "\n".join(lines)


def _parse_ics_dt(value: str) -> datetime:
    raw = value.strip()
    if "T" in raw:
        if raw.endswith("Z"):
            return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if len(raw) >= 15:
            return datetime.strptime(raw[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)


def _list_caldav_events(
    conn: CalendarConnection,
    *,
    days: int,
    max_events: int,
) -> str:
    """Minimal CalDAV REPORT for VEVENT in range (read-only)."""
    now = datetime.now(timezone.utc)
    start = now.strftime("%Y%m%dT%H%M%SZ")
    end = (now + timedelta(days=days)).strftime("%Y%m%dT%H%M%SZ")
    body = f"""<?xml version="1.0" encoding="utf-8" ?>
<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:getetag/>
    <C:calendar-data/>
  </D:prop>
  <C:filter>
    <C:comp-filter name="VCALENDAR">
      <C:comp-filter name="VEVENT">
        <C:time-range start="{start}" end="{end}"/>
      </C:comp-filter>
    </C:comp-filter>
  </C:filter>
</C:calendar-query>
"""
    with httpx.Client(timeout=25.0, follow_redirects=True) as client:
        resp = client.request(
            "REPORT",
            conn.url,
            content=body,
            headers={
                "Content-Type": "application/xml; charset=utf-8",
                "Depth": "1",
            },
            auth=(conn.username, conn.password),
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"CalDAV list failed ({resp.status_code}): {resp.text[:300]}"
        )
    # Extract calendar-data blobs and reuse ICS parser logic.
    ns = {
        "D": "DAV:",
        "C": "urn:ietf:params:xml:ns:caldav",
    }
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        raise RuntimeError(f"CalDAV response was not XML: {exc}") from exc
    blobs: list[str] = []
    for node in root.findall(".//C:calendar-data", ns):
        if node.text:
            blobs.append(node.text)
    if not blobs:
        # Some servers omit namespace prefixes in a way ElementTree misses — fallback regex.
        blobs = re.findall(
            r"<[^>]*calendar-data[^>]*>(.*?)</[^>]*calendar-data>",
            resp.text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    combined = "\n".join(blobs)
    if not combined.strip():
        return (
            f"SUCCESS: No events in the next {days} day(s) "
            f"on CalDAV calendar `{conn.calendar_id}`."
        )
    # Inline parse without HTTP:
    text = _unfold_ics(combined)
    until = now + timedelta(days=days)
    events: list[tuple[datetime, str, str, str]] = []
    for match in _VEVENT_RE.finditer(text):
        block = match.group(1)
        summary = _ics_field(block, "SUMMARY") or "(no title)"
        dtstart = _ics_field(block, "DTSTART")
        dtend = _ics_field(block, "DTEND")
        location = _ics_field(block, "LOCATION")
        if not dtstart:
            continue
        try:
            start_dt = _parse_ics_dt(dtstart)
            end_dt = _parse_ics_dt(dtend) if dtend else start_dt
        except ValueError:
            continue
        if end_dt < now or start_dt > until:
            continue
        events.append((start_dt, summary, end_dt.isoformat(), location))
    events.sort(key=lambda row: row[0])
    events = events[:max_events]
    if not events:
        return (
            f"SUCCESS: No events in the next {days} day(s) "
            f"on CalDAV calendar `{conn.calendar_id}`."
        )
    lines = [
        f"SUCCESS: Upcoming events (next {days} day(s), CalDAV `{conn.calendar_id}`):"
    ]
    for start_dt, summary, end_s, location in events:
        lines.append(_format_event_line(summary, start_dt.isoformat(), end_s, location))
    return "\n".join(lines)


def execute_calendar_tool(
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
        return denied
    assert account_user_id is not None
    conn_or_err = _require_connection(settings, account_user_id=account_user_id)
    if isinstance(conn_or_err, str):
        return conn_or_err
    conn = conn_or_err
    try:
        if name == "list_calendar_events":
            days = int(args.get("days") or 7)
            max_events = int(args.get("max_events") or 15)
            days = max(1, min(31, days))
            max_events = max(1, min(50, max_events))
            if conn.provider == "google":
                return _list_google_events(
                    settings, conn, days=days, max_events=max_events
                )
            if conn.provider == "ics":
                return _list_ics_events(conn, days=days, max_events=max_events)
            if conn.provider == "caldav":
                return _list_caldav_events(conn, days=days, max_events=max_events)
            return _tool_failed(f"unsupported calendar provider: {conn.provider}")
        if name == "create_calendar_event":
            if conn.provider == "ics":
                return _UNAVAILABLE_ICS_WRITE
            if conn.provider == "caldav":
                return _UNAVAILABLE_CALDAV_WRITE
            if conn.provider != "google":
                return _tool_failed(f"unsupported calendar provider: {conn.provider}")
            return _create_google_event(
                settings,
                conn,
                title=str(args.get("title") or ""),
                start=str(args.get("start") or ""),
                end=str(args.get("end") or ""),
                description=str(args.get("description") or ""),
                location=str(args.get("location") or ""),
            )
        return _tool_failed(f"unknown calendar tool: {name}")
    except Exception as exc:  # noqa: BLE001 - tool boundary
        logger.warning("Calendar tool %s failed for user=%s: %s", name, account_user_id, exc)
        return _tool_failed(f"{name} — {exc}")


def build_calendar_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="list_calendar_events",
            description=(
                "List upcoming events on the signed-in human's connected calendar "
                "(private 1:1 only; not group chat; not guests). "
                "Uses whatever calendar they connected on their My tools page."
            ),
            func=lambda days=7, max_events=15: (
                "FAILED: list_calendar_events must run through the agent tool runner"
            ),
            args_schema=ListCalendarEventsArgs,
        ),
        StructuredTool.from_function(
            name="create_calendar_event",
            description=(
                "Create an event on the signed-in human's connected calendar "
                "(private 1:1 only; Google write support; not group chat; not guests)."
            ),
            func=lambda title, start, end, description="", location="": (
                "FAILED: create_calendar_event must run through the agent tool runner"
            ),
            args_schema=CreateCalendarEventArgs,
        ),
    ]
