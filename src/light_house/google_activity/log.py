"""Append-only arrival log. One file per human. No wakes."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from light_house.config import Settings

_LOCK = threading.Lock()


def _safe_user_id(user_id: str) -> str:
    uid = (user_id or "").strip().lower()
    if not uid or "/" in uid or "\\" in uid or ".." in uid:
        raise ValueError("user_id is not a safe log name")
    return uid


def activity_path(settings: Settings, user_id: str) -> Path:
    return settings.google_activity_dir / f"{_safe_user_id(user_id)}.jsonl"


def cursor_path(settings: Settings, user_id: str) -> Path:
    return settings.google_activity_dir / f"{_safe_user_id(user_id)}.cursor.json"


def load_cursor(settings: Settings, user_id: str) -> dict[str, str]:
    path = cursor_path(settings, user_id)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            out[key] = value
    return out


def save_cursor(settings: Settings, user_id: str, cursor: dict[str, str]) -> None:
    path = cursor_path(settings, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {k: v for k, v in cursor.items() if isinstance(v, str)}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_activity(settings: Settings, user_id: str, entries: list[dict[str, Any]]) -> int:
    if not entries:
        return 0
    path = activity_path(settings, user_id)
    lines = [json.dumps(entry, separators=(",", ":"), sort_keys=True) for entry in entries]
    blob = "\n".join(lines) + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(blob)
    return len(entries)


def read_activity(settings: Settings, user_id: str, *, max_entries: int = 20) -> list[dict[str, Any]]:
    path = activity_path(settings, user_id)
    if not path.is_file():
        return []
    limit = max(1, min(40, int(max_entries)))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            rows.append(raw)
    return rows[-limit:]


def outbound_note_path(settings: Settings) -> Path:
    """One house log of notes written to the host. Not a wake."""
    return settings.google_activity_dir / "write-to-kevin.jsonl"


def append_outbound_note(settings: Settings, entry: dict[str, Any]) -> None:
    path = outbound_note_path(settings)
    line = json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def utc_stamp(moment: datetime | None = None) -> str:
    when = moment or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
