"""A note a light leaves itself, handed once on a later free wake.

Not the in-progress list. That glance has no day and stays until it is dropped.
A promise stays quiet until the day the light named. The first free wake on or
after that day is handed it once. A late wake says how many days late.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from light_house.config import Settings
from light_house.personal.age import humanize_age
from light_house.personal.in_progress import is_free_awake

MAX_WAITING = 20
MAX_NOTE_LEN = 280
_DAY_COUNT = re.compile(r"(?:in\s+)?(\d+)\s*(?:days?|d)?\Z")


@dataclass(frozen=True)
class Promise:
    id: int
    note: str
    due_ymd: str
    created_at: float
    updated_at: float
    handed_at: float | None


def normalize_promise_note(note: str) -> tuple[str, str]:
    display = " ".join((note or "").split())
    if not display:
        raise ValueError("Leave the words — a promise needs a note.")
    if len(display) > MAX_NOTE_LEN:
        raise ValueError(f"Note is too long (max {MAX_NOTE_LEN} characters).")
    return display, display.casefold()


def local_today(timezone_name: str, *, now: datetime | None = None) -> date:
    tz = ZoneInfo(timezone_name)
    current = now.astimezone(tz) if now is not None else datetime.now(tz)
    return current.date()


def require_due_ymd(due_ymd: str) -> str:
    try:
        return date.fromisoformat(due_ymd).isoformat()
    except ValueError as exc:
        raise ValueError("The day has to be YYYY-MM-DD.") from exc


def parse_due_day(raw: str, *, today: date) -> date:
    """A calendar day, or a count of days from today on the house clock."""
    text = " ".join((raw or "").split()).casefold()
    if not text:
        raise ValueError("Name a day — a promise needs a day.")
    if text in ("today", "0"):
        return today
    counted = _DAY_COUNT.fullmatch(text)
    if counted:
        days = int(counted.group(1))
        if days > 3650:
            raise ValueError("That day is too far out.")
        return today + timedelta(days=days)
    normalized = text.replace("/", "-")
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "Name a day as YYYY-MM-DD, today, or a count of days (3, or in 3 days)."
        ) from exc


def days_late(due_ymd: str, today: date) -> int:
    due = date.fromisoformat(due_ymd)
    return (today - due).days


def _written_age(created_at: float, *, now: float) -> str:
    age = humanize_age(created_at, now=now)
    if age == "0m":
        return "just now"
    return f"{age} ago"


def format_promise_line(item: Promise, *, today: date, now: float) -> str:
    written = _written_age(item.created_at, now=now)
    late = days_late(item.due_ymd, today)
    if late >= 1:
        unit = "day" if late == 1 else "days"
        when = f"This was due {late} {unit} ago."
    else:
        when = "Due today."
    return f"- {item.note} — written {written}. {when}"


def format_handed_promises(items: list[Promise], *, today: date, now: float) -> str:
    """The block a free wake is handed. Empty when nothing is due."""
    if not items:
        return ""
    title = "## A promise you left" if len(items) == 1 else "## Promises you left"
    lines = [
        title,
        "Handed once, on this free awake. "
        "Let it go, or name another day, with **promise**.",
    ]
    lines.extend(format_promise_line(item, today=today, now=now) for item in items)
    return "\n".join(lines)


def format_waiting_promises(items: list[Promise]) -> str:
    if not items:
        return (
            "No promises waiting. "
            "One already handed stays quiet unless you leave the same words with another day."
        )
    lines = ["Waiting:"]
    for item in items:
        lines.append(f"- #{item.id} — {item.note} — due {item.due_ymd}")
    lines.append(
        "Drop one, or move the day, before it is handed. "
        "Kevin does not write these."
    )
    return "\n".join(lines)


def match_promise(query: str, items: list[Promise]) -> Promise | None:
    text = " ".join((query or "").split())
    if not text:
        raise ValueError("Name the promise — its id, or the words you left.")
    token = text[1:].strip() if text.startswith("#") else text
    if token.isdigit():
        found = [item for item in items if item.id == int(token)]
        return found[0] if found else None

    key = text.casefold()
    exact = [item for item in items if item.note.casefold() == key]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        listed = ", ".join(f"#{item.id}" for item in exact)
        raise ValueError(f"More than one promise has those words: {listed}. Give the id.")

    prefixed = [item for item in items if item.note.casefold().startswith(key)]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        listed = ", ".join(item.note for item in prefixed)
        raise ValueError(
            f"More than one promise starts that way: {listed}. Give more of the words, or the id."
        )
    return None


def leave_reply(item: Promise, *, created: bool, today: date) -> str:
    due = date.fromisoformat(item.due_ymd)
    late = (today - due).days
    if created:
        head = f"Left for {item.due_ymd}: {item.note}."
    else:
        head = f"Named another day, {item.due_ymd}: {item.note}."
    if late > 0:
        tail = "That day has passed. The next free awake hands it and says how late."
    elif late == 0:
        tail = "The next free awake hands it once."
    else:
        tail = "It stays quiet until that day. The first free awake on or after it hands this once."
    return (
        f"{head} {tail} "
        "Drop it or move the day with promise. Kevin does not write these."
    )


def hand_due_promises_for_agent(
    settings: Settings,
    agent_id: str,
    *,
    wake_kind: str | None,
    today: date | None = None,
    now: float | None = None,
) -> str:
    """Put due promises on a free wake, once. Other wakes leave them waiting."""
    if not is_free_awake(wake_kind):
        return ""
    if not settings.personal_db_enabled:
        return ""
    try:
        from light_house.personal.store import get_personal_store

        clock = time.time() if now is None else now
        if today is None:
            tz_name = getattr(settings, "dream_timezone", None) or "America/Los_Angeles"
            today = local_today(tz_name)
        store = get_personal_store(settings, agent_id)
        due = store.list_due_promises(today.isoformat())
        if not due:
            return ""
        text = format_handed_promises(due, today=today, now=clock)
        store.mark_promises_handed([item.id for item in due], clock)
        return text
    except Exception:
        return ""
