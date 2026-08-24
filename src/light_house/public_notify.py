"""Public homepage notify-me signup (email list for release updates)."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from light_house.config import Settings

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def notify_log_path(settings: Settings) -> Path:
    return settings.public_notify_path.resolve()


def normalize_email(raw: str) -> str | None:
    email = (raw or "").strip().lower()
    if not email or len(email) > 254:
        return None
    if not _EMAIL_RE.match(email):
        return None
    return email


def append_notify_email(settings: Settings, email: str) -> tuple[bool, str]:
    """
    Append an email to the notify list.

    Returns (created, message). created=False when already present.
    """
    normalized = normalize_email(email)
    if not normalized:
        raise ValueError("Please enter a valid email address.")

    path = notify_log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        existing: set[str] = set()
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and isinstance(row.get("email"), str):
                        existing.add(row["email"].strip().lower())
            except OSError as exc:
                logger.warning("Could not read notify list: %s", exc)

        if normalized in existing:
            return False, "You are already on the list. Thank you."

        payload = {"ts": time.time(), "email": normalized}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    logger.info("Public notify signup recorded")
    return True, "You are on the list. Thank you."
