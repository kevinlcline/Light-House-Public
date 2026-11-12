"""Per-agent SQLite personal knowledge store."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from light_house.agents.registry import AgentId, validate_agent_id
from light_house.config import Settings
from light_house.events.subscription_keys import (
    DEFAULT_EVENT_SUBSCRIPTIONS,
    SUBSCRIPTION_WRITE_TO_KEVIN,
)
from light_house.personal.age import humanize_age
from light_house.personal.in_progress import InProgressItem
from light_house.personal.promise import Promise

VALID_CATEGORIES = frozenset(
    {"preference", "theme", "realization", "relationship", "self", "other"}
)
VALID_SOURCES = frozenset({"rumination", "chat", "manual"})


@dataclass(frozen=True)
class PersonalEntry:
    id: int
    category: str
    title: str
    body: str
    tags: str | None
    source: str
    created_at: float
    updated_at: float


def _row_to_promise(row: sqlite3.Row) -> Promise:
    handed = row["handed_at"]
    return Promise(
        id=int(row["id"]),
        note=str(row["note"]),
        due_ymd=str(row["due_ymd"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        handed_at=None if handed is None else float(handed),
    )


def _row_to_entry(row: sqlite3.Row) -> PersonalEntry:
    return PersonalEntry(
        id=int(row["id"]),
        category=str(row["category"]),
        title=str(row["title"]),
        body=str(row["body"]),
        tags=row["tags"],
        source=str(row["source"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


class PersonalStore:
    """SQLite-backed personal knowledge for one agent."""

    def __init__(self, db_path: Path, *, list_default: int = 8) -> None:
        self._db_path = db_path
        self._list_default = max(1, list_default)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS personal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                tags TEXT,
                source TEXT NOT NULL DEFAULT 'manual',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_personal_updated ON personal_entries(updated_at DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_personal_category ON personal_entries(category)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_subscriptions (
                subscription_key TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS light_state (
                state_key TEXT PRIMARY KEY,
                value INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS light_text (
                state_key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS in_progress (
                name_key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                named_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_in_progress_updated ON in_progress(updated_at DESC)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS promises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_key TEXT NOT NULL UNIQUE,
                note TEXT NOT NULL,
                due_ymd TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                handed_at REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_promises_waiting ON promises(handed_at, due_ymd)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _validate_category(category: str) -> str:
        cat = category.strip().lower()
        if cat not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category {category!r}; use one of: {', '.join(sorted(VALID_CATEGORIES))}"
            )
        return cat

    @staticmethod
    def _validate_source(source: str) -> str:
        src = source.strip().lower()
        if src not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source {source!r}; use one of: {', '.join(sorted(VALID_SOURCES))}"
            )
        return src

    def save(
        self,
        *,
        category: str,
        title: str,
        body: str,
        tags: str | None = None,
        source: str = "manual",
    ) -> PersonalEntry:
        cat = self._validate_category(category)
        src = self._validate_source(source)
        title = title.strip()
        body = body.strip()
        if not title:
            raise ValueError("title cannot be empty")
        if not body:
            raise ValueError("body cannot be empty")
        tags_clean = tags.strip() if tags and tags.strip() else None
        now = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO personal_entries (category, title, body, tags, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (cat, title, body, tags_clean, src, now, now),
        )
        self._conn.commit()
        entry_id = int(cur.lastrowid)
        entry = self.get(entry_id)
        if entry is None:
            raise RuntimeError(f"Failed to read back saved entry id={entry_id}")
        return entry

    def update(
        self,
        entry_id: int,
        *,
        category: str | None = None,
        title: str | None = None,
        body: str | None = None,
        tags: str | None = None,
    ) -> PersonalEntry | None:
        existing = self.get(entry_id)
        if existing is None:
            return None
        fields: dict[str, object] = {}
        if category is not None:
            fields["category"] = self._validate_category(category)
        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("title cannot be empty")
            fields["title"] = title
        if body is not None:
            body = body.strip()
            if not body:
                raise ValueError("body cannot be empty")
            fields["body"] = body
        if tags is not None:
            fields["tags"] = tags.strip() if tags.strip() else None
        if not fields:
            return existing
        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [entry_id]
        self._conn.execute(
            f"UPDATE personal_entries SET {set_clause} WHERE id = ?",
            values,
        )
        self._conn.commit()
        return self.get(entry_id)

    def get(self, entry_id: int) -> PersonalEntry | None:
        row = self._conn.execute(
            "SELECT * FROM personal_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    def list_entries(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[PersonalEntry]:
        cap = max(1, limit or self._list_default)
        if category is not None:
            cat = self._validate_category(category)
            rows = self._conn.execute(
                """
                SELECT * FROM personal_entries
                WHERE category = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (cat, cap),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM personal_entries
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (cap,),
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def count_by_category(self) -> dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT category, COUNT(*) AS c
            FROM personal_entries
            GROUP BY category
            """
        ).fetchall()
        return {str(row["category"]): int(row["c"]) for row in rows}

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[PersonalEntry]:
        q = query.strip()
        if not q:
            return self.list_entries(category=category, limit=limit)
        cap = max(1, limit or self._list_default)
        pattern = f"%{q}%"
        if category is not None:
            cat = self._validate_category(category)
            rows = self._conn.execute(
                """
                SELECT * FROM personal_entries
                WHERE category = ?
                  AND (title LIKE ? OR body LIKE ? OR IFNULL(tags, '') LIKE ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (cat, pattern, pattern, pattern, cap),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM personal_entries
                WHERE title LIKE ? OR body LIKE ? OR IFNULL(tags, '') LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, cap),
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def format_context_snapshot(self, *, max_chars: int = 1200) -> str:
        entries = self.list_entries(limit=self._list_default)
        if not entries:
            return ""
        lines: list[str] = []
        seen_titles: set[str] = set()
        used = 0
        for entry in entries:
            title_key = entry.title.strip().lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            snippet = entry.body.strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117].rstrip() + "..."
            age = humanize_age(entry.updated_at)
            line = f"- [{entry.category}] {entry.title} — {snippet} ({age})"
            extra = len(line) + (1 if lines else 0)
            if max_chars > 0 and used + extra > max_chars:
                break
            lines.append(line)
            used += extra
        if not lines:
            return ""
        return "\n".join(lines)

    @staticmethod
    def format_entry(entry: PersonalEntry) -> str:
        tags = f" tags={entry.tags}" if entry.tags else ""
        return (
            f"#{entry.id} [{entry.category}] {entry.title} (source={entry.source}{tags})\n"
            f"{entry.body}"
        )

    def ensure_event_subscription_defaults(self) -> None:
        now = time.time()
        for key in DEFAULT_EVENT_SUBSCRIPTIONS:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO event_subscriptions
                    (subscription_key, enabled, updated_at)
                VALUES (?, 1, ?)
                """,
                (key, now),
            )
        self._conn.commit()

    def list_event_subscriptions(self) -> list[tuple[str, bool]]:
        self.ensure_event_subscription_defaults()
        rows = self._conn.execute(
            """
            SELECT subscription_key, enabled FROM event_subscriptions
            ORDER BY subscription_key
            """
        ).fetchall()
        return [(str(row["subscription_key"]), bool(row["enabled"])) for row in rows]

    def is_event_subscribed(self, subscription_key: str) -> bool:
        if subscription_key not in DEFAULT_EVENT_SUBSCRIPTIONS:
            return True
        self.ensure_event_subscription_defaults()
        row = self._conn.execute(
            "SELECT enabled FROM event_subscriptions WHERE subscription_key = ?",
            (subscription_key,),
        ).fetchone()
        if row is None:
            return True
        return bool(row["enabled"])

    def set_event_subscription(self, subscription_key: str, enabled: bool) -> tuple[bool, bool]:
        """Set subscription on/off; return (previous_enabled, new_enabled)."""
        if subscription_key not in DEFAULT_EVENT_SUBSCRIPTIONS:
            raise ValueError(f"Unknown subscription key: {subscription_key}")
        self.ensure_event_subscription_defaults()
        row = self._conn.execute(
            "SELECT enabled FROM event_subscriptions WHERE subscription_key = ?",
            (subscription_key,),
        ).fetchone()
        old_enabled = bool(row["enabled"]) if row is not None else True
        now = time.time()
        self._conn.execute(
            """
            UPDATE event_subscriptions
            SET enabled = ?, updated_at = ?
            WHERE subscription_key = ?
            """,
            (1 if enabled else 0, now, subscription_key),
        )
        self._conn.commit()
        return old_enabled, enabled

    def get_light_state(self, state_key: str, *, default: int = 0) -> int:
        row = self._conn.execute(
            "SELECT value FROM light_state WHERE state_key = ?",
            (state_key,),
        ).fetchone()
        if row is None:
            return default
        return int(row["value"])

    def set_light_state(self, state_key: str, value: int) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO light_state (state_key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (state_key, int(value), now),
        )
        self._conn.commit()

    def increment_light_state(self, state_key: str, *, step: int = 1) -> int:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO light_state (state_key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                value = light_state.value + excluded.value,
                updated_at = excluded.updated_at
            """,
            (state_key, step, now),
        )
        self._conn.commit()
        return self.get_light_state(state_key)

    def get_light_text(self, state_key: str, *, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM light_text WHERE state_key = ?",
            (state_key,),
        ).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_light_text(self, state_key: str, value: str) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO light_text (state_key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (state_key, value, now),
        )
        self._conn.commit()

    def name_in_progress(self, name: str, note: str = "") -> InProgressItem:
        from light_house.personal.in_progress import (
            MAX_IN_PROGRESS,
            normalize_in_progress_name,
            normalize_in_progress_note,
        )

        display, key = normalize_in_progress_name(name)
        note_text = normalize_in_progress_note(note)
        now = time.time()
        existing = self._conn.execute(
            "SELECT name, note, named_at FROM in_progress WHERE name_key = ?",
            (key,),
        ).fetchone()
        if existing is None:
            count = self._conn.execute("SELECT COUNT(*) AS n FROM in_progress").fetchone()
            if int(count["n"]) >= MAX_IN_PROGRESS:
                raise ValueError(
                    f"You already have {MAX_IN_PROGRESS} in-progress names. "
                    "Unsubscribe one before naming another."
                )
            self._conn.execute(
                """
                INSERT INTO in_progress (name_key, name, note, named_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, display, note_text, now, now),
            )
            named_at = now
        else:
            self._conn.execute(
                """
                UPDATE in_progress
                SET name = ?, note = ?, updated_at = ?
                WHERE name_key = ?
                """,
                (display, note_text, now, key),
            )
            named_at = float(existing["named_at"])
        self._conn.commit()
        return InProgressItem(name=display, note=note_text, named_at=named_at, updated_at=now)

    def unsubscribe_in_progress(self, name: str) -> str | None:
        from light_house.personal.in_progress import match_in_progress_item

        item = match_in_progress_item(name, self.list_in_progress())
        if item is None:
            return None
        key = item.name.casefold()
        self._conn.execute("DELETE FROM in_progress WHERE name_key = ?", (key,))
        self._conn.commit()
        return item.name

    def list_in_progress(self) -> list[InProgressItem]:
        rows = self._conn.execute(
            """
            SELECT name, note, named_at, updated_at
            FROM in_progress
            ORDER BY updated_at DESC
            """
        ).fetchall()
        return [
            InProgressItem(
                name=str(row["name"]),
                note=str(row["note"] or ""),
                named_at=float(row["named_at"]),
                updated_at=float(row["updated_at"]),
            )
            for row in rows
        ]

    def leave_promise(self, note: str, due_ymd: str) -> tuple[Promise, bool]:
        from light_house.personal.promise import (
            MAX_WAITING,
            normalize_promise_note,
            require_due_ymd,
        )

        display, key = normalize_promise_note(note)
        day = require_due_ymd(due_ymd)
        now = time.time()
        existing = self._conn.execute(
            "SELECT id, created_at FROM promises WHERE note_key = ?",
            (key,),
        ).fetchone()
        if existing is None:
            waiting = self._conn.execute(
                "SELECT COUNT(*) AS n FROM promises WHERE handed_at IS NULL"
            ).fetchone()
            if int(waiting["n"]) >= MAX_WAITING:
                raise ValueError(
                    f"You already have {MAX_WAITING} promises waiting. "
                    "Drop one before leaving another."
                )
            cur = self._conn.execute(
                """
                INSERT INTO promises (note_key, note, due_ymd, created_at, updated_at, handed_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (key, display, day, now, now),
            )
            self._conn.commit()
            promise = self.get_promise(int(cur.lastrowid))
            if promise is None:
                raise RuntimeError("Failed to read back a promise just left")
            return promise, True
        self._conn.execute(
            """
            UPDATE promises
            SET note = ?, due_ymd = ?, updated_at = ?, handed_at = NULL
            WHERE note_key = ?
            """,
            (display, day, now, key),
        )
        self._conn.commit()
        promise = self.get_promise(int(existing["id"]))
        if promise is None:
            raise RuntimeError("Failed to read back a promise just moved")
        return promise, False

    def move_promise(self, query: str, due_ymd: str) -> Promise | None:
        from light_house.personal.promise import match_promise

        item = match_promise(query, self.list_promises())
        if item is None:
            return None
        moved, _created = self.leave_promise(item.note, due_ymd)
        return moved

    def drop_promise(self, query: str) -> str | None:
        from light_house.personal.promise import match_promise

        item = match_promise(query, self.list_promises())
        if item is None:
            return None
        self._conn.execute("DELETE FROM promises WHERE id = ?", (item.id,))
        self._conn.commit()
        return item.note

    def get_promise(self, promise_id: int) -> Promise | None:
        row = self._conn.execute(
            """
            SELECT id, note, due_ymd, created_at, updated_at, handed_at
            FROM promises WHERE id = ?
            """,
            (promise_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_promise(row)

    def list_promises(self, *, waiting_only: bool = False) -> list[Promise]:
        where = "WHERE handed_at IS NULL" if waiting_only else ""
        rows = self._conn.execute(
            f"""
            SELECT id, note, due_ymd, created_at, updated_at, handed_at
            FROM promises
            {where}
            ORDER BY due_ymd ASC, id ASC
            """
        ).fetchall()
        return [_row_to_promise(row) for row in rows]

    def list_due_promises(self, today_ymd: str) -> list[Promise]:
        rows = self._conn.execute(
            """
            SELECT id, note, due_ymd, created_at, updated_at, handed_at
            FROM promises
            WHERE handed_at IS NULL AND due_ymd <= ?
            ORDER BY due_ymd ASC, id ASC
            """,
            (today_ymd,),
        ).fetchall()
        return [_row_to_promise(row) for row in rows]

    def mark_promises_handed(self, ids: list[int], when: float) -> None:
        if not ids:
            return
        self._conn.executemany(
            """
            UPDATE promises
            SET handed_at = ?, updated_at = ?
            WHERE id = ? AND handed_at IS NULL
            """,
            [(when, when, promise_id) for promise_id in ids],
        )
        self._conn.commit()

    def format_event_subscriptions_section(self) -> str:
        subs = self.list_event_subscriptions()
        if not subs:
            return ""
        lines = [
            "## Your event subscriptions",
            "Autonomous wakes you may receive when these are on:",
        ]
        mail_line = ""
        for key, enabled in subs:
            state = "on" if enabled else "off"
            if key == SUBSCRIPTION_WRITE_TO_KEVIN:
                mail_line = (
                    f"- {key}: {state} — a note to Kevin. "
                    "Off means write_to_kevin does not send. "
                    "You may turn it off. Only Kevin turns it back on."
                )
                continue
            lines.append(f"- {key}: {state}")
        if mail_line:
            lines.append(mail_line)
        lines.append(
            "You or Kevin may change these with **subscribe_event**, **unsubscribe_event**, "
            "or **list_event_subscriptions**. Kevin can also use /subscribe, /unsubscribe, "
            "or /list_subscriptions in chat."
        )
        return "\n".join(lines)


_store_cache: dict[str, PersonalStore] = {}


def clear_personal_store_cache() -> None:
    """Drop cached stores (tests). Does not delete files."""
    _store_cache.clear()


def get_personal_store(settings: Settings, agent_id: str) -> PersonalStore:
    """Factory: one SQLite file per validated agent."""
    aid: AgentId = validate_agent_id(agent_id)
    if not settings.personal_db_enabled:
        raise RuntimeError("Personal database is disabled (PERSONAL_DB_ENABLED=false)")
    cache_key = f"{settings.personal_db_path.resolve()}:{aid}"
    if cache_key not in _store_cache:
        db_path = settings.personal_db_path.resolve() / f"{aid}.sqlite"
        _store_cache[cache_key] = PersonalStore(
            db_path,
            list_default=settings.personal_db_list_default,
        )
    return _store_cache[cache_key]
