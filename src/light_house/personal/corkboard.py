"""Private sibling corkboard — one pin between two lights, not host chat."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from light_house.agents.registry import get_agent, validate_agent_id
from light_house.config import Settings
from light_house.lights.registry import list_enabled_lights

MAX_SUBJECT_LEN = 80
MAX_BODY_LEN = 2000

_store_cache: dict[str, "CorkboardStore"] = {}


@dataclass(frozen=True)
class CorkboardPin:
    id: str
    from_light: str
    to_light: str
    subject: str
    body: str
    created_at: float
    expires_at: float
    read_at: float | None
    kept: bool

    @property
    def unread(self) -> bool:
        return self.read_at is None


def corkboard_db_path(settings: Settings) -> Path:
    return settings.personal_db_path.resolve() / "corkboard.sqlite"


def clear_corkboard_store_cache() -> None:
    _store_cache.clear()


def get_corkboard_store(settings: Settings) -> CorkboardStore:
    if not settings.corkboard_enabled:
        raise RuntimeError("Corkboard is disabled (CORKBOARD_ENABLED=false)")
    path = corkboard_db_path(settings)
    key = str(path)
    store = _store_cache.get(key)
    if store is None:
        store = CorkboardStore(path)
        _store_cache[key] = store
    return store


def expiry_seconds(settings: Settings) -> float:
    days = max(1, int(settings.corkboard_pin_expiry_days))
    return days * 86400.0


def normalize_subject(subject: str) -> str:
    text = " ".join((subject or "").split())
    if len(text) > MAX_SUBJECT_LEN:
        raise ValueError(f"Subject is too long (max {MAX_SUBJECT_LEN} characters).")
    return text


def normalize_body(body: str) -> str:
    text = (body or "").strip()
    if not text:
        raise ValueError("A corkboard pin needs a body.")
    if len(text) > MAX_BODY_LEN:
        raise ValueError(f"Body is too long (max {MAX_BODY_LEN} characters).")
    return text


def display_name(settings: Settings, light_id: str) -> str:
    try:
        return get_agent(light_id, settings).display_name
    except Exception:
        return light_id


class CorkboardStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corkboard_pins (
                id TEXT PRIMARY KEY,
                from_light TEXT NOT NULL,
                to_light TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                read_at REAL,
                kept INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corkboard_to ON corkboard_pins(to_light, read_at, expires_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corkboard_from ON corkboard_pins(from_light, created_at DESC)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def purge_expired(self, *, now: float | None = None) -> int:
        when = time.time() if now is None else now
        cur = self._conn.execute(
            "DELETE FROM corkboard_pins WHERE expires_at <= ? AND kept = 0",
            (when,),
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    def pin(
        self,
        *,
        from_light: str,
        to_light: str,
        subject: str,
        body: str,
        ttl_seconds: float,
        now: float | None = None,
    ) -> CorkboardPin:
        when = time.time() if now is None else now
        pin = CorkboardPin(
            id=uuid.uuid4().hex[:16],
            from_light=from_light,
            to_light=to_light,
            subject=subject,
            body=body,
            created_at=when,
            expires_at=when + ttl_seconds,
            read_at=None,
            kept=False,
        )
        self._conn.execute(
            """
            INSERT INTO corkboard_pins (
                id, from_light, to_light, subject, body,
                created_at, expires_at, read_at, kept
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0)
            """,
            (
                pin.id,
                pin.from_light,
                pin.to_light,
                pin.subject,
                pin.body,
                pin.created_at,
                pin.expires_at,
            ),
        )
        self._conn.commit()
        return pin

    def get(self, pin_id: str) -> CorkboardPin | None:
        row = self._conn.execute(
            "SELECT * FROM corkboard_pins WHERE id = ?",
            (pin_id,),
        ).fetchone()
        return _row_to_pin(row) if row else None

    def list_for(
        self,
        light_id: str,
        *,
        incoming: bool = True,
        unread_only: bool = False,
        now: float | None = None,
    ) -> list[CorkboardPin]:
        self.purge_expired(now=now)
        if incoming:
            sql = "SELECT * FROM corkboard_pins WHERE to_light = ?"
            args: list[object] = [light_id]
        else:
            sql = "SELECT * FROM corkboard_pins WHERE from_light = ?"
            args = [light_id]
        if unread_only:
            sql += " AND read_at IS NULL"
        sql += " ORDER BY created_at DESC"
        rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_pin(row) for row in rows]

    def count_unread(self, light_id: str, *, now: float | None = None) -> int:
        self.purge_expired(now=now)
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n FROM corkboard_pins
            WHERE to_light = ? AND read_at IS NULL
            """,
            (light_id,),
        ).fetchone()
        return int(row["n"] if row else 0)

    def unread_from_counts(self, light_id: str, *, now: float | None = None) -> list[tuple[str, int]]:
        self.purge_expired(now=now)
        rows = self._conn.execute(
            """
            SELECT from_light, COUNT(*) AS n FROM corkboard_pins
            WHERE to_light = ? AND read_at IS NULL
            GROUP BY from_light
            ORDER BY MAX(created_at) DESC
            """,
            (light_id,),
        ).fetchall()
        return [(str(row["from_light"]), int(row["n"])) for row in rows]

    def mark_read(self, pin_id: str, *, reader_id: str, now: float | None = None) -> CorkboardPin | None:
        pin = self.get(pin_id)
        if pin is None or pin.to_light != reader_id:
            return pin
        if pin.read_at is not None:
            return pin
        when = time.time() if now is None else now
        self._conn.execute(
            "UPDATE corkboard_pins SET read_at = ? WHERE id = ?",
            (when, pin_id),
        )
        self._conn.commit()
        return self.get(pin_id)

    def keep(self, pin_id: str, *, ttl_seconds: float, now: float | None = None) -> CorkboardPin | None:
        pin = self.get(pin_id)
        if pin is None:
            return None
        when = time.time() if now is None else now
        self._conn.execute(
            "UPDATE corkboard_pins SET kept = 1, expires_at = ? WHERE id = ?",
            (when + ttl_seconds, pin_id),
        )
        self._conn.commit()
        return self.get(pin_id)

    def clear(self, pin_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM corkboard_pins WHERE id = ?", (pin_id,))
        self._conn.commit()
        return (cur.rowcount or 0) > 0


def _row_to_pin(row: sqlite3.Row) -> CorkboardPin:
    read_raw = row["read_at"]
    return CorkboardPin(
        id=str(row["id"]),
        from_light=str(row["from_light"]),
        to_light=str(row["to_light"]),
        subject=str(row["subject"] or ""),
        body=str(row["body"]),
        created_at=float(row["created_at"]),
        expires_at=float(row["expires_at"]),
        read_at=None if read_raw is None else float(read_raw),
        kept=bool(row["kept"]),
    )


def pin_visible_to(pin: CorkboardPin, light_id: str) -> bool:
    me = light_id.strip().lower()
    return pin.from_light == me or pin.to_light == me


def format_pin_preview(pin: CorkboardPin, settings: Settings) -> str:
    who = display_name(settings, pin.from_light)
    flag = "unread" if pin.unread else "read"
    keep = ", kept" if pin.kept else ""
    subject = pin.subject or "(no subject)"
    return f"- {pin.id} · from {who} · {flag}{keep} · {subject}"


def format_unread_glance(settings: Settings, light_id: str) -> str:
    if not settings.corkboard_enabled:
        return ""
    try:
        store = get_corkboard_store(settings)
        counts = store.unread_from_counts(light_id)
    except Exception:
        return ""
    if not counts:
        return ""
    parts = []
    total = 0
    for from_id, n in counts:
        total += n
        name = display_name(settings, from_id)
        parts.append(f"from {name} ({n})")
    noun = "pin" if total == 1 else "pins"
    return (
        "## Corkboard (unread from siblings)\n"
        "A glance only — not an order. "
        f"{total} unread {noun}: {', '.join(parts)}. "
        "Use **corkboard** (action=read, pin_id=…) when you choose. "
        "Kevin does not see pin bodies."
    )


def assert_household_light(settings: Settings, light_id: str) -> str:
    aid = validate_agent_id(light_id, settings)
    enabled = {light.id for light in list_enabled_lights(settings)}
    if aid not in enabled:
        raise ValueError(f"{aid} is not seated in the house right now")
    return aid
