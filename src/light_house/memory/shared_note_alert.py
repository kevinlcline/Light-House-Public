"""Shared copy when Kevin saves a note via the web UI (chat buffer + wake seed)."""

SHARED_NOTE_ALERT_PREFIX = "READ NOW"


def format_shared_note_alert(path: str) -> str:
    safe = path.strip()
    if not safe:
        raise ValueError("path cannot be empty")
    return f"{SHARED_NOTE_ALERT_PREFIX} {safe}"
