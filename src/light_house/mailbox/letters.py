"""Mailbox letter format, paths, and notify queue (filesystem)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from light_house.config import Settings, get_settings
from light_house.lights.registry import list_enabled_lights

logger = logging.getLogger(__name__)

MAILBOX_DIR_NAME = "mailbox"
QUEUE_DIR_NAME = ".queue"
DONE_DIR_NAME = ".done"
REED_ID = "reed"
ALL_TOKEN = "all"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_SAFE_SLUG = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True)
class Letter:
    from_id: str
    to_ids: tuple[str, ...]
    subject: str
    body: str
    created_at: str
    path: str | None = None  # relative to notes root, posix


def resolve_notes_root(settings: Settings | None = None) -> Path:
    cfg = settings or get_settings()
    return Path(cfg.notes_path).expanduser().resolve()


def shared_mailbox_dir(notes_root: Path) -> Path:
    return notes_root / "shared" / MAILBOX_DIR_NAME


def private_mailbox_dir(notes_root: Path, light_id: str) -> Path:
    return notes_root / light_id.strip().lower() / MAILBOX_DIR_NAME


def queue_dir(notes_root: Path) -> Path:
    return shared_mailbox_dir(notes_root) / QUEUE_DIR_NAME


def done_dir(notes_root: Path) -> Path:
    return shared_mailbox_dir(notes_root) / DONE_DIR_NAME


def _slug(text: str, *, max_len: int = 48) -> str:
    raw = (text or "letter").strip().lower()
    slug = _SAFE_SLUG.sub("-", raw).strip("-._") or "letter"
    return slug[:max_len]


def _parse_to_field(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parts = [p.strip().lower() for p in re.split(r"[,;\s]+", text) if p.strip()]
    return parts


def _parse_frontmatter(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip("\"'")
        if key:
            out[key] = value
    return out


def parse_letter(text: str, *, path: str | None = None) -> Letter:
    normalized = text if text.endswith("\n") else f"{text}\n"
    match = _FRONTMATTER_RE.match(normalized)
    if not match:
        # Bare body — treat as letter to reed from unknown
        return Letter(
            from_id="unknown",
            to_ids=(REED_ID,),
            subject="(no subject)",
            body=text.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
            path=path,
        )
    meta = _parse_frontmatter(match.group(1))
    body = match.group(2).strip()
    from_id = (meta.get("from") or "unknown").strip().lower()
    to_raw = meta.get("to") or meta.get("addressed") or ""
    to_ids = tuple(_parse_to_field(to_raw)) or (REED_ID,)
    subject = (meta.get("subject") or "(no subject)").strip()
    created = (meta.get("created") or meta.get("date") or "").strip()
    if not created:
        created = datetime.now(timezone.utc).isoformat()
    return Letter(
        from_id=from_id,
        to_ids=to_ids,
        subject=subject,
        body=body,
        created_at=created,
        path=path,
    )


def render_letter(letter: Letter) -> str:
    to_line = ", ".join(letter.to_ids)
    return (
        "---\n"
        f"from: {letter.from_id}\n"
        f"to: {to_line}\n"
        f"subject: {letter.subject}\n"
        f"created: {letter.created_at}\n"
        "---\n\n"
        f"{letter.body.rstrip()}\n"
    )


def expand_recipients(
    to_ids: list[str] | tuple[str, ...],
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Expand `all` to enabled light ids; keep reed; de-dupe."""
    cfg = settings or get_settings()
    enabled = {light.id for light in list_enabled_lights(cfg)}
    out: list[str] = []
    for raw in to_ids:
        token = raw.strip().lower()
        if not token:
            continue
        if token == ALL_TOKEN:
            for light_id in sorted(enabled):
                if light_id not in out:
                    out.append(light_id)
            continue
        if token not in out:
            out.append(token)
    return out


def light_recipients(
    to_ids: list[str] | tuple[str, ...],
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Recipients that are enabled lights (excludes reed and unknown humans)."""
    cfg = settings or get_settings()
    enabled = {light.id for light in list_enabled_lights(cfg)}
    return [r for r in expand_recipients(to_ids, settings=cfg) if r in enabled]


def write_letter(
    *,
    from_id: str,
    to_ids: list[str],
    subject: str,
    body: str,
    settings: Settings | None = None,
    private: bool = False,
    filename: str | None = None,
) -> Letter:
    """
    Write a mailbox letter and return it (with path).

    - private=True and exactly one light recipient → notes/{light}/mailbox/
    - otherwise → notes/shared/mailbox/ (to_reed/ when only reed is addressed)
    """
    cfg = settings or get_settings()
    notes_root = resolve_notes_root(cfg)
    created = datetime.now(timezone.utc).isoformat()
    recipients = expand_recipients(to_ids, settings=cfg)
    if not recipients:
        raise ValueError("Letter needs at least one recipient in `to`")
    letter = Letter(
        from_id=from_id.strip().lower() or "unknown",
        to_ids=tuple(recipients),
        subject=(subject or "(no subject)").strip(),
        body=(body or "").strip(),
        created_at=created,
    )
    lights = light_recipients(recipients, settings=cfg)
    only_reed = recipients == [REED_ID]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base_name = filename or f"{stamp}_{_slug(letter.subject)}.md"
    if not base_name.endswith((".md", ".markdown", ".txt")):
        base_name = f"{base_name}.md"

    if private and len(lights) == 1 and not only_reed:
        dest_dir = private_mailbox_dir(notes_root, lights[0])
        rel = f"{lights[0]}/{MAILBOX_DIR_NAME}/{base_name}"
    elif only_reed:
        dest_dir = shared_mailbox_dir(notes_root) / "to_reed"
        rel = f"shared/{MAILBOX_DIR_NAME}/to_reed/{base_name}"
    else:
        dest_dir = shared_mailbox_dir(notes_root) / "from_reed" if letter.from_id == REED_ID else shared_mailbox_dir(notes_root)
        if letter.from_id == REED_ID:
            rel = f"shared/{MAILBOX_DIR_NAME}/from_reed/{base_name}"
        else:
            rel = f"shared/{MAILBOX_DIR_NAME}/{base_name}"

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / base_name
    if path.exists():
        stem = path.stem
        path = dest_dir / f"{stem}_{uuid.uuid4().hex[:6]}{path.suffix}"
        rel = str(path.relative_to(notes_root)).replace("\\", "/")

    path.write_text(render_letter(letter), encoding="utf-8")
    written = Letter(
        from_id=letter.from_id,
        to_ids=letter.to_ids,
        subject=letter.subject,
        body=letter.body,
        created_at=letter.created_at,
        path=rel,
    )
    logger.info(
        "Mailbox letter written from=%s to=%s path=%s",
        written.from_id,
        ",".join(written.to_ids),
        written.path,
    )
    return written


def queue_notify(
    letter: Letter,
    *,
    settings: Settings | None = None,
) -> Path | None:
    """
    Queue a notify job so the house wakes addressed lights.

    Returns queue file path, or None if no lights to wake (e.g. to: reed only).
    """
    cfg = settings or get_settings()
    notes_root = resolve_notes_root(cfg)
    lights = light_recipients(list(letter.to_ids), settings=cfg)
    if not lights:
        logger.info("Mailbox notify skipped (no light recipients) path=%s", letter.path)
        return None
    if not letter.path:
        raise ValueError("Letter path required to queue notify")

    qdir = queue_dir(notes_root)
    qdir.mkdir(parents=True, exist_ok=True)
    notify_id = uuid.uuid4().hex
    payload = {
        "id": notify_id,
        "path": letter.path,
        "from": letter.from_id,
        "to": lights,
        "subject": letter.subject,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
    out = qdir / f"{notify_id}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Mailbox notify queued id=%s path=%s to=%s",
        notify_id,
        letter.path,
        ",".join(lights),
    )
    return out


def list_pending_notifies(notes_root: Path) -> list[Path]:
    qdir = queue_dir(notes_root)
    if not qdir.is_dir():
        return []
    return sorted(qdir.glob("*.json"))


def mark_notify_done(notes_root: Path, notify_path: Path) -> None:
    done = done_dir(notes_root)
    done.mkdir(parents=True, exist_ok=True)
    dest = done / notify_path.name
    try:
        notify_path.replace(dest)
    except OSError:
        # Cross-device fallback
        dest.write_text(notify_path.read_text(encoding="utf-8"), encoding="utf-8")
        notify_path.unlink(missing_ok=True)


def list_letters_for(
    recipient_id: str,
    *,
    settings: Settings | None = None,
    limit: int = 50,
) -> list[Letter]:
    """List recent letters addressed to recipient_id (reed or a light)."""
    cfg = settings or get_settings()
    notes_root = resolve_notes_root(cfg)
    rid = recipient_id.strip().lower()
    candidates: list[Path] = []
    shared = shared_mailbox_dir(notes_root)
    if shared.is_dir():
        candidates.extend(p for p in shared.rglob("*") if p.is_file() and p.suffix in {".md", ".markdown", ".txt"})
    if rid != REED_ID:
        priv = private_mailbox_dir(notes_root, rid)
        if priv.is_dir():
            candidates.extend(
                p for p in priv.rglob("*") if p.is_file() and p.suffix in {".md", ".markdown", ".txt"}
            )

    letters: list[Letter] = []
    for path in candidates:
        # Skip queue/done internals
        parts = set(path.parts)
        if QUEUE_DIR_NAME in parts or DONE_DIR_NAME in parts:
            continue
        try:
            rel = str(path.relative_to(notes_root)).replace("\\", "/")
            letter = parse_letter(path.read_text(encoding="utf-8"), path=rel)
        except (OSError, UnicodeError):
            continue
        expanded = expand_recipients(list(letter.to_ids), settings=cfg)
        if rid in expanded or (rid != REED_ID and ALL_TOKEN in letter.to_ids):
            letters.append(letter)
    letters.sort(key=lambda L: L.created_at, reverse=True)
    return letters[: max(1, limit)]
