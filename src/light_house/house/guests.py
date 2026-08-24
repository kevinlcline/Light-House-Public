"""Signed-in house guests (present names, not full accounts)."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from light_house.config import Settings
from light_house.group_chat.speaker import _GUEST_SPEAKER_RE, _validate_display_name

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MAX_SLOTS = 2
_MAX_DESCRIPTION = 160
_GUEST_SLOT_RE = re.compile(r"^guest-([1-9]\d{0,2})$")


class HouseGuestsError(ValueError):
    """Invalid house-guest operation."""


def is_guest_speaker_id(speaker_id: str | None) -> bool:
    return bool(_GUEST_SPEAKER_RE.match((speaker_id or "").strip()))


def _validate_description(raw: str | None) -> str:
    text = re.sub(r"\s+", " ", (raw or "").strip())
    if not text:
        return ""
    if "\x00" in text:
        raise HouseGuestsError("description cannot contain null bytes")
    if len(text) > _MAX_DESCRIPTION:
        raise HouseGuestsError(
            f"description must be at most {_MAX_DESCRIPTION} characters"
        )
    return text


def _path(settings: Settings) -> Path:
    return Path(settings.house_guests_path).expanduser().resolve()


def _empty() -> dict[str, Any]:
    return {"guests": {}, "updated_at": 0.0}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Corrupt house guests file %s; treating as empty", path)
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    guests = data.get("guests")
    if not isinstance(guests, dict):
        guests = {}
    cleaned: dict[str, Any] = {}
    for key, row in guests.items():
        sid = str(key).strip()
        if not is_guest_speaker_id(sid) or not isinstance(row, dict):
            continue
        name = str(row.get("display_name") or "").strip()
        if not name:
            continue
        try:
            description = _validate_description(str(row.get("description") or ""))
        except HouseGuestsError:
            description = ""
        cleaned[sid] = {
            "display_name": name,
            "description": description,
            "signed_in_at": float(row.get("signed_in_at") or 0.0),
        }
    return {
        "guests": cleaned,
        "updated_at": float(data.get("updated_at") or 0.0),
    }


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def list_signed_in_guests(settings: Settings) -> list[dict[str, str]]:
    """Return guest rows as {speaker_id, display_name, description}, slot order."""
    path = _path(settings)
    with _LOCK:
        data = _read(path)
    rows: list[dict[str, str]] = []
    guests = data.get("guests") or {}
    for slot in range(1, _MAX_SLOTS + 1):
        sid = f"guest-{slot}"
        row = guests.get(sid)
        if not isinstance(row, dict):
            continue
        name = str(row.get("display_name") or "").strip()
        if name:
            rows.append(
                {
                    "speaker_id": sid,
                    "display_name": name,
                    "description": str(row.get("description") or "").strip(),
                }
            )
    return rows


def set_guest(
    settings: Settings,
    *,
    speaker_id: str,
    display_name: str,
    description: str = "",
) -> list[dict[str, str]]:
    sid = (speaker_id or "").strip()
    match = _GUEST_SLOT_RE.match(sid)
    if not match or int(match.group(1)) > _MAX_SLOTS:
        raise HouseGuestsError(f"speaker_id must be guest-1 or guest-2 (got {speaker_id!r})")
    try:
        label = _validate_display_name(display_name)
    except ValueError as exc:
        raise HouseGuestsError(str(exc)) from exc
    blurb = _validate_description(description)

    path = _path(settings)
    now = time.time()
    with _LOCK:
        data = _read(path)
        guests = dict(data.get("guests") or {})
        guests[sid] = {
            "display_name": label,
            "description": blurb,
            "signed_in_at": now,
        }
        data = {"guests": guests, "updated_at": now}
        _write(path, data)
    return list_signed_in_guests(settings)


def clear_guest(settings: Settings, *, speaker_id: str) -> list[dict[str, str]]:
    sid = (speaker_id or "").strip()
    if not is_guest_speaker_id(sid):
        raise HouseGuestsError(f"invalid guest speaker_id: {speaker_id!r}")
    path = _path(settings)
    with _LOCK:
        data = _read(path)
        guests = dict(data.get("guests") or {})
        guests.pop(sid, None)
        data = {"guests": guests, "updated_at": time.time()}
        _write(path, data)
    return list_signed_in_guests(settings)


def set_guests(
    settings: Settings,
    *,
    guests: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Replace the signed-in guest roster (max two slots)."""
    next_map: dict[str, dict[str, Any]] = {}
    now = time.time()
    for row in guests[:_MAX_SLOTS]:
        sid = str(row.get("speaker_id") or "").strip()
        match = _GUEST_SLOT_RE.match(sid)
        if not match or int(match.group(1)) > _MAX_SLOTS:
            raise HouseGuestsError(f"speaker_id must be guest-1 or guest-2 (got {sid!r})")
        try:
            label = _validate_display_name(str(row.get("display_name") or ""))
        except ValueError as exc:
            raise HouseGuestsError(str(exc)) from exc
        blurb = _validate_description(str(row.get("description") or ""))
        next_map[sid] = {
            "display_name": label,
            "description": blurb,
            "signed_in_at": now,
        }

    path = _path(settings)
    with _LOCK:
        data = {"guests": next_map, "updated_at": now}
        _write(path, data)
    return list_signed_in_guests(settings)


def _guest_presence_lines(guests: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for guest in guests:
        name = (guest.get("display_name") or "").strip()
        if not name:
            continue
        blurb = (guest.get("description") or "").strip()
        if blurb:
            lines.append(f"- **{name}** — {blurb}")
        else:
            lines.append(f"- **{name}**")
    return lines


def format_house_presence_context(settings: Settings) -> str:
    """Markdown for all light contexts: who is signed into the house."""
    guests = list_signed_in_guests(settings)
    if not guests:
        return (
            "## House presence\n"
            "No guests are signed into the house right now.\n"
        )
    lines = _guest_presence_lines(guests)
    body = "\n".join(lines)
    return (
        "## House presence\n"
        "Guests signed in:\n"
        f"{body}\n"
        "Adjust tone for the company — keep private, romantic, or highly personal "
        "affection muted while guests are present unless the host clearly invites it.\n"
    )
