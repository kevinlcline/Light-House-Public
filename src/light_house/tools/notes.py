"""Safe read/write/delete access for agent notes (private + shared folders)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from light_house.lights.registry import known_light_ids

_SHARED_DIR_NAME = "shared"
_PENDING_DELETES_FILE = ".pending_deletes.json"

_ALLOWED_SUFFIXES = {".txt", ".md", ".markdown"}
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]*$")
_MAX_PATH_LEN = 512
_WRITING_PREFIX = "writing/"
_WRITING_HISTORY_PREFIX = "writing/_history/"


@dataclass(frozen=True)
class NoteInfo:
    name: str
    size_bytes: int
    modified_at: str


@dataclass(frozen=True)
class NoteDeleteOutcome:
    deleted: bool
    message: str
    pending: bool = False


class SharedDeleteRegistry:
    """Tracks dual-agent delete votes for shared notes."""

    def __init__(self, shared_dir: Path) -> None:
        self._shared_dir = shared_dir.resolve()
        self._path = self._shared_dir / _PENDING_DELETES_FILE

    def _load(self) -> dict[str, dict[str, object]]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save(self, data: dict[str, dict[str, object]]) -> None:
        self._shared_dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_vote(
        self, inner_path: str, agent_id: str, *, required_voters: frozenset[str]
    ) -> tuple[list[str], bool]:
        """Add light vote; return current voters and whether all required voters agreed."""
        if agent_id not in required_voters:
            raise ValueError(f"Invalid light for shared delete vote: {agent_id}")
        data = self._load()
        entry = data.get(inner_path)
        if not isinstance(entry, dict):
            entry = {}
        votes_raw = entry.get("votes")
        votes = [v for v in votes_raw if isinstance(v, str)] if isinstance(votes_raw, list) else []
        if agent_id not in votes:
            votes.append(agent_id)
        entry["votes"] = votes
        entry["updated_at"] = time.time()
        data[inner_path] = entry
        self._save(data)
        all_voted = required_voters.issubset(set(votes))
        return votes, all_voted

    def clear(self, inner_path: str) -> None:
        data = self._load()
        if inner_path not in data:
            return
        del data[inner_path]
        self._save(data)

    def get_votes(self, inner_path: str) -> list[str]:
        entry = self._load().get(inner_path)
        if not isinstance(entry, dict):
            return []
        votes_raw = entry.get("votes")
        if not isinstance(votes_raw, list):
            return []
        return [v for v in votes_raw if isinstance(v, str)]


class NoteWriter:
    """
    Read/write/delete access under a single notes directory.

    No execute, no paths outside ``notes_dir``.
    Supports nested subfolders (e.g. ``journal/may.md``).
    """

    def __init__(self, notes_dir: Path, *, max_chars_per_write: int = 32_000) -> None:
        self._notes_dir = notes_dir.resolve()
        self._max_chars = max(1, max_chars_per_write)
        self._last_archived_rel: str | None = None
        self._notes_dir.mkdir(parents=True, exist_ok=True)

    @property
    def notes_dir(self) -> Path:
        return self._notes_dir

    @staticmethod
    def _split_relative(relative: str) -> list[str]:
        if "\x00" in relative:
            raise ValueError("Invalid path")
        raw = relative.strip().replace("\\", "/")
        if raw.startswith("/"):
            raise ValueError("Invalid path")
        parts = [p for p in raw.split("/") if p]
        if not parts or ".." in parts or "." in parts:
            raise ValueError("Invalid path")
        return parts

    @staticmethod
    def _validate_segment(part: str) -> None:
        if not _SAFE_NAME.match(part):
            raise ValueError(
                "Path segments may only contain letters, numbers, dots, hyphens, and underscores"
            )

    @staticmethod
    def sanitize_path(relative_path: str) -> str:
        """Sanitize a relative note file path (e.g. ``journal/may.md``)."""
        parts = NoteWriter._split_relative(relative_path)
        for part in parts[:-1]:
            NoteWriter._validate_segment(part)
        name = parts[-1]
        NoteWriter._validate_segment(name)
        suffix = Path(name).suffix.lower()
        if not suffix:
            name = f"{name}.txt"
            parts[-1] = name
        elif suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"Allowed extensions: {', '.join(sorted(_ALLOWED_SUFFIXES))}")
        if len(name) > 128:
            raise ValueError("Filename too long (max 128 characters)")
        result = "/".join(parts)
        if len(result) > _MAX_PATH_LEN:
            raise ValueError(f"Path too long (max {_MAX_PATH_LEN} characters)")
        return result

    @staticmethod
    def sanitize_dir_path(relative_dir: str) -> str:
        """Sanitize a relative directory path under notes/ (e.g. ``journal`` or ``research/ideas``)."""
        parts = NoteWriter._split_relative(relative_dir)
        for part in parts:
            NoteWriter._validate_segment(part)
        result = "/".join(parts)
        if len(result) > _MAX_PATH_LEN:
            raise ValueError(f"Path too long (max {_MAX_PATH_LEN} characters)")
        return result

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Basename-only sanitizer (backward compatible with flat API callers)."""
        name = filename.strip().replace("\\", "/").split("/")[-1]
        return NoteWriter.sanitize_path(name)

    def _resolve(self, relative_path: str) -> Path:
        safe = self.sanitize_path(relative_path)
        target = (self._notes_dir / safe).resolve()
        if not str(target).startswith(str(self._notes_dir)):
            raise ValueError("Path must stay inside the notes folder")
        return target

    def _resolve_dir(self, relative_dir: str) -> Path:
        safe = self.sanitize_dir_path(relative_dir)
        target = (self._notes_dir / safe).resolve()
        if not str(target).startswith(str(self._notes_dir)):
            raise ValueError("Path must stay inside the notes folder")
        return target

    def mkdir(self, relative_dir: str) -> Path:
        """Create a subdirectory under notes/ (idempotent)."""
        target = self._resolve_dir(relative_dir)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write(self, filename: str, content: str) -> Path:
        """Create or replace a note file."""
        self._last_archived_rel = None
        text = self._validate_content(content)
        safe = self.sanitize_path(filename)
        path = self._resolve(filename)
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if existing.strip() != text.strip():
                self._last_archived_rel = self._archive_writing_draft(safe, existing)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _archive_writing_draft(self, safe_rel: str, existing_body: str) -> str | None:
        """Copy prior writing/ draft to writing/_history/ before overwrite."""
        if not safe_rel.startswith(_WRITING_PREFIX):
            return None
        if safe_rel.startswith(_WRITING_HISTORY_PREFIX):
            return None
        if not existing_body.strip():
            return None

        rel_under = safe_rel[len(_WRITING_PREFIX) :]
        parts = rel_under.split("/")
        file_name = parts[-1]
        stem = Path(file_name).stem
        suffix = Path(file_name).suffix or ".md"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if len(parts) > 1:
            subdir = "/".join(parts[:-1])
            history_rel = f"{_WRITING_HISTORY_PREFIX}{subdir}/{stem}-{ts}{suffix}"
        else:
            history_rel = f"{_WRITING_HISTORY_PREFIX}{stem}-{ts}{suffix}"

        history_path = self._resolve(history_rel)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(existing_body, encoding="utf-8")
        return history_rel

    def append(self, filename: str, content: str) -> Path:
        """Append to a note (creates the file if it does not exist)."""
        text = self._validate_content(content)
        path = self._resolve(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > 0:
            existing = path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            text = existing + text
        path.write_text(text, encoding="utf-8")
        return path

    def list_notes(self) -> list[NoteInfo]:
        """List all note files under notes/ recursively (newest first)."""
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        rows: list[NoteInfo] = []
        seen: set[str] = set()
        for suffix in _ALLOWED_SUFFIXES:
            for path in self._notes_dir.rglob(f"*{suffix}"):
                if not path.is_file():
                    continue
                rel = path.relative_to(self._notes_dir).as_posix()
                if rel in seen:
                    continue
                try:
                    self.sanitize_path(rel)
                except ValueError:
                    continue
                seen.add(rel)
                stat = path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                rows.append(
                    NoteInfo(
                        name=rel,
                        size_bytes=stat.st_size,
                        modified_at=mtime,
                    )
                )
        rows.sort(key=lambda n: n.modified_at, reverse=True)
        return rows

    def read(
        self,
        filename: str,
        *,
        offset: int = 0,
        max_chars: int | None = None,
    ) -> str:
        """Read a note file (same path rules as write).

        Large notes return a character window with a continuation hint rather than failing.
        """
        path = self._resolve(filename)
        if not path.is_file():
            raise FileNotFoundError(filename)
        size_bytes = path.stat().st_size
        # Absolute load ceiling (bytes) — prevents reading multi-megabyte accidents whole.
        load_ceiling = max(self._max_chars * 8, 2_000_000)
        if size_bytes > load_ceiling:
            raise ValueError(
                f"Note too large to load ({size_bytes} bytes; max {load_ceiling})"
            )
        text = path.read_text(encoding="utf-8")
        total = len(text)
        start = max(0, int(offset or 0))
        window = self._max_chars if max_chars is None else int(max_chars)
        window = max(1, min(window, self._max_chars))
        if start >= total:
            return f"[Note empty at offset {start}; total {total} characters]\n"
        chunk = text[start : start + window]
        end = start + len(chunk)
        if start == 0 and end >= total:
            return chunk
        header = (
            f"[Note excerpt characters {start}-{end} of {total}; "
            f"call read_note again with offset={end} to continue]\n\n"
        )
        return header + chunk

    def delete(self, filename: str) -> None:
        """Delete a note file (must exist; not a directory)."""
        path = self._resolve(filename)
        if not path.is_file():
            raise FileNotFoundError(filename)
        path.unlink()

    def _validate_content(self, content: str) -> str:
        if not isinstance(content, str):
            content = str(content)
        text = content.strip()
        if not text:
            raise ValueError("Content cannot be empty")
        if len(text) > self._max_chars:
            raise ValueError(f"Content too long (max {self._max_chars} characters)")
        if "\x00" in text:
            raise ValueError("Content cannot contain NUL bytes")
        return text


class AgentNoteWriter:
    """
    Notes access for a main agent: one private folder (``notes/lumen`` or ``notes/ara``)
    plus ``notes/shared`` for cross-agent files. Echo does not get note tools.

    Paths without a prefix resolve under the agent's private folder (e.g. ``journal/may.md``).
    Paths starting with ``shared/`` resolve under ``notes/shared/`` (e.g. ``shared/household.md``).
    """

    def __init__(
        self,
        notes_base: Path,
        private_relative: str,
        *,
        max_chars_per_write: int = 32_000,
        delete_voters: frozenset[str] | None = None,
    ) -> None:
        from light_house.config import get_settings

        self._delete_voters = delete_voters or known_light_ids(get_settings())
        self._notes_base = notes_base.resolve()
        self._private_relative = private_relative.strip().replace("\\", "/").strip("/")
        if not self._private_relative or "/" in self._private_relative:
            raise ValueError("Invalid private notes folder name")
        NoteWriter._validate_segment(self._private_relative)
        self._private_dir = (self._notes_base / self._private_relative).resolve()
        self._shared_dir = (self._notes_base / _SHARED_DIR_NAME).resolve()
        self._max_chars = max(1, max_chars_per_write)
        self._private = NoteWriter(self._private_dir, max_chars_per_write=max_chars_per_write)
        self._shared = NoteWriter(self._shared_dir, max_chars_per_write=max_chars_per_write)
        self._private_dir.mkdir(parents=True, exist_ok=True)
        self._shared_dir.mkdir(parents=True, exist_ok=True)

    @property
    def notes_base(self) -> Path:
        return self._notes_base

    @property
    def notes_dir(self) -> Path:
        """Private notes directory (backward compatible)."""
        return self._private_dir

    @property
    def private_dir(self) -> Path:
        return self._private_dir

    @property
    def shared_dir(self) -> Path:
        return self._shared_dir

    @property
    def private_relative(self) -> str:
        return self._private_relative

    @property
    def last_archived_draft(self) -> str | None:
        """Relative private path of draft archived on the last write (writing/ only)."""
        return self._private._last_archived_rel

    def _canonicalize_path_input(self, relative: str) -> str:
        """Normalize model-supplied paths (absolute paths, notes/ara/… prefixes, etc.)."""
        raw = relative.strip().replace("\\", "/")
        if not raw:
            raise ValueError("Invalid path")

        lower = raw.lower()
        shared_anchor = f"/notes/{_SHARED_DIR_NAME}/"
        private_anchor = f"/notes/{self._private_relative}/"
        notes_anchor = "/notes/"

        if shared_anchor in lower:
            idx = lower.index(shared_anchor)
            raw = f"{_SHARED_DIR_NAME}/" + raw[idx + len(shared_anchor) :]
        elif private_anchor in lower:
            idx = lower.index(private_anchor)
            raw = raw[idx + len(private_anchor) :]
        elif notes_anchor in lower:
            idx = lower.index(notes_anchor)
            tail = raw[idx + len(notes_anchor) :]
            if tail.startswith(f"{self._private_relative}/"):
                raw = tail[len(self._private_relative) + 1 :]
            else:
                raw = tail

        for prefix in (
            f"notes/{self._private_relative}/",
            f"notes/{_SHARED_DIR_NAME}/",
        ):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :]

        if raw.startswith("notes/"):
            tail = raw[len("notes/") :]
            if tail.startswith(f"{self._private_relative}/"):
                raw = tail[len(self._private_relative) + 1 :]
            else:
                raw = tail

        return raw

    def _split_agent_path(self, relative: str) -> tuple[bool, str]:
        relative = self._canonicalize_path_input(relative)
        parts = NoteWriter._split_relative(relative)
        if parts[0] == _SHARED_DIR_NAME:
            inner = "/".join(parts[1:])
            return True, inner
        return False, relative

    def _normalize_private_path(self, relative: str) -> str:
        """Strip agent prefix (``lumen/…`` or ``ara/…``) when present for private resolution."""
        parts = NoteWriter._split_relative(relative)
        if parts[0] == self._private_relative:
            inner = "/".join(parts[1:])
            if not inner:
                raise ValueError("Invalid path")
            return inner
        return relative

    def _private_path_was_prefixed(self, relative: str) -> bool:
        parts = NoteWriter._split_relative(relative)
        return bool(parts and parts[0] == self._private_relative)

    def sanitize_path(self, relative_path: str) -> str:
        relative_path = self._canonicalize_path_input(relative_path)
        is_shared, inner = self._split_agent_path(relative_path)
        if is_shared:
            if not inner:
                raise ValueError("Shared note path must include a filename, e.g. shared/ideas.md")
            safe_inner = self._private.sanitize_path(inner)
            return f"{_SHARED_DIR_NAME}/{safe_inner}"
        prefixed = self._private_path_was_prefixed(relative_path)
        private_rel = self._normalize_private_path(relative_path)
        safe = self._private.sanitize_path(private_rel)
        if prefixed:
            return f"{self._private_relative}/{safe}"
        return safe

    def sanitize_dir_path(self, relative_dir: str) -> str:
        relative_dir = self._canonicalize_path_input(relative_dir)
        is_shared, inner = self._split_agent_path(relative_dir)
        if is_shared:
            if not inner:
                return _SHARED_DIR_NAME
            safe_inner = self._private.sanitize_dir_path(inner)
            return f"{_SHARED_DIR_NAME}/{safe_inner}"
        prefixed = self._private_path_was_prefixed(relative_dir)
        private_rel = self._normalize_private_path(relative_dir)
        safe = self._private.sanitize_dir_path(private_rel)
        if prefixed:
            return f"{self._private_relative}/{safe}"
        return safe

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        return NoteWriter.sanitize_filename(filename)

    def mkdir(self, relative_dir: str) -> Path:
        relative_dir = self._canonicalize_path_input(relative_dir)
        is_shared, inner = self._split_agent_path(relative_dir)
        if is_shared:
            if not inner:
                self._shared_dir.mkdir(parents=True, exist_ok=True)
                return self._shared_dir
            return self._shared.mkdir(inner)
        return self._private.mkdir(self._normalize_private_path(relative_dir))

    def write(self, filename: str, content: str) -> Path:
        filename = self._canonicalize_path_input(filename)
        is_shared, inner = self._split_agent_path(filename)
        if is_shared:
            return self._shared.write(inner, content)
        return self._private.write(self._normalize_private_path(filename), content)

    def append(self, filename: str, content: str) -> Path:
        filename = self._canonicalize_path_input(filename)
        is_shared, inner = self._split_agent_path(filename)
        if is_shared:
            return self._shared.append(inner, content)
        return self._private.append(self._normalize_private_path(filename), content)

    def read(
        self,
        filename: str,
        *,
        offset: int = 0,
        max_chars: int | None = None,
    ) -> str:
        filename = self._canonicalize_path_input(filename)
        is_shared, inner = self._split_agent_path(filename)
        if is_shared:
            return self._shared.read(inner, offset=offset, max_chars=max_chars)
        return self._private.read(
            self._normalize_private_path(filename),
            offset=offset,
            max_chars=max_chars,
        )

    def list_notes(self) -> list[NoteInfo]:
        rows: list[NoteInfo] = []
        for info in self._private.list_notes():
            rows.append(
                NoteInfo(
                    name=f"{self._private_relative}/{info.name}",
                    size_bytes=info.size_bytes,
                    modified_at=info.modified_at,
                )
            )
        for info in self._shared.list_notes():
            rows.append(
                NoteInfo(
                    name=f"{_SHARED_DIR_NAME}/{info.name}",
                    size_bytes=info.size_bytes,
                    modified_at=info.modified_at,
                )
            )
        rows.sort(key=lambda n: n.modified_at, reverse=True)
        return rows

    def display_path(self, path: Path) -> str:
        resolved = path.resolve()
        if str(resolved).startswith(str(self._shared_dir)):
            rel = resolved.relative_to(self._shared_dir).as_posix()
            return f"{_SHARED_DIR_NAME}/{rel}"
        rel = resolved.relative_to(self._private_dir).as_posix()
        return f"{self._private_relative}/{rel}"

    def delete(self, agent_id: str, filename: str) -> NoteDeleteOutcome:
        """
        Delete a note. Private notes delete immediately.

        Shared notes require every enabled light to call delete on the same path.
        """
        filename = self._canonicalize_path_input(filename)
        is_shared, inner = self._split_agent_path(filename)
        if is_shared:
            return self._delete_shared(agent_id, inner, filename)
        return self._delete_private(filename)

    def delete_for_operator(self, filename: str) -> NoteDeleteOutcome:
        """Delete immediately from Kevin's notes UI (no dual-agent consent)."""
        filename = self._canonicalize_path_input(filename)
        is_shared, inner = self._split_agent_path(filename)
        if is_shared:
            if not inner:
                raise ValueError("Shared note path must include a filename, e.g. shared/ideas.md")
            safe_inner = self._shared.sanitize_path(inner)
            display = f"{_SHARED_DIR_NAME}/{safe_inner}"
            path = (self._shared_dir / safe_inner).resolve()
            if not path.is_file():
                raise FileNotFoundError(display)
            self._shared.delete(safe_inner)
            SharedDeleteRegistry(self._shared_dir).clear(safe_inner)
            return NoteDeleteOutcome(deleted=True, message=f"Deleted {display}")
        return self._delete_private(filename)

    def _delete_private(self, filename: str) -> NoteDeleteOutcome:
        private_rel = self._normalize_private_path(filename)
        display = self.sanitize_path(filename)
        self._private.delete(private_rel)
        return NoteDeleteOutcome(deleted=True, message=f"Deleted note {display}")

    def _delete_shared(self, agent_id: str, inner: str, filename: str) -> NoteDeleteOutcome:
        if not inner:
            raise ValueError("Shared note path must include a filename, e.g. shared/ideas.md")
        safe_inner = self._shared.sanitize_path(inner)
        display = f"{_SHARED_DIR_NAME}/{safe_inner}"
        path = (self._shared_dir / safe_inner).resolve()
        if not path.is_file():
            raise FileNotFoundError(display)

        registry = SharedDeleteRegistry(self._shared_dir)
        already_voted = agent_id in registry.get_votes(safe_inner)
        votes, all_voted = registry.record_vote(safe_inner, agent_id, required_voters=self._delete_voters)

        if not all_voted:
            missing = sorted(self._delete_voters - set(votes))
            awaiting = missing[0] if missing else agent_id
            if already_voted:
                prefix = f"Your delete vote for {display} was already recorded."
            else:
                prefix = f"Delete vote recorded for {display}."
            return NoteDeleteOutcome(
                deleted=False,
                pending=True,
                message=(
                    f"{prefix} The file is still on disk — NOT DELETED yet. "
                    f"Awaiting {awaiting} to call delete_shared on the same path."
                ),
            )

        self._shared.delete(safe_inner)
        registry.clear(safe_inner)
        voters = ", ".join(sorted(set(votes)))
        return NoteDeleteOutcome(
            deleted=True,
            message=f"Deleted shared note {display} (all required lights agreed: {voters})",
        )
