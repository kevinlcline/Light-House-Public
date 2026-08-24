"""Read-only access to the Light-House project tree for main agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "htmlcov",
        "dist",
        "build",
        ".cursor",
    }
)
_BLOCKED_FILE_NAMES = frozenset({".env"})
_BLOCKED_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".sqlite",
        ".sqlite3",
        ".bin",
        ".pem",
        ".key",
        ".woff",
        ".woff2",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".zip",
        ".tar",
        ".gz",
    }
)
_MAX_LIST_ENTRIES = 500
_DEFAULT_LIST_ENTRIES = 200


@dataclass(frozen=True)
class CodebaseEntry:
    path: str
    kind: str
    size_bytes: int | None = None


class CodebaseReader:
    """
    Read-only traversal under the Light-House repo root.

    Blocks secrets, venv, git internals, and large/binary artifacts.
    """

    def __init__(self, root: Path, *, max_chars_per_read: int = 32_000) -> None:
        self._root = root.resolve()
        self._max_chars = max(1, max_chars_per_read)
        self._max_read_bytes = self._max_chars * 4

    @staticmethod
    def default_repo_root() -> Path:
        """Repository root (``Light-House/``)."""
        return Path(__file__).resolve().parent.parent.parent.parent

    @property
    def root(self) -> Path:
        return self._root

    def _split_relative(self, relative: str) -> list[str]:
        if "\x00" in relative:
            raise ValueError("Invalid path")
        raw = relative.strip().replace("\\", "/")
        if raw.startswith("/"):
            raise ValueError("Path must be relative to the project root")
        parts = [p for p in raw.split("/") if p]
        if ".." in parts or "." in parts:
            raise ValueError("Invalid path")
        return parts

    def resolve_dir(self, relative_path: str = "") -> Path:
        parts = self._split_relative(relative_path)
        target = (self._root / Path(*parts)).resolve() if parts else self._root.resolve()
        if not str(target).startswith(str(self._root)):
            raise ValueError("Path must stay inside the Light-House project")
        if not target.is_dir():
            raise FileNotFoundError(relative_path or ".")
        return target

    def resolve_file(self, relative_path: str) -> Path:
        if not relative_path.strip():
            raise ValueError("Path required")
        parts = self._split_relative(relative_path)
        target = (self._root / Path(*parts)).resolve()
        if not str(target).startswith(str(self._root)):
            raise ValueError("Path must stay inside the Light-House project")
        if not target.is_file():
            raise FileNotFoundError(relative_path)
        self._check_readable_file(target)
        return target

    def _check_readable_file(self, path: Path) -> None:
        name = path.name
        if name in _BLOCKED_FILE_NAMES:
            raise ValueError(f"Reading {name} is not allowed")
        if name.startswith(".env."):
            raise ValueError("Reading env secret files is not allowed")
        suffix = path.suffix.lower()
        if suffix in _BLOCKED_SUFFIXES:
            raise ValueError(f"Reading {suffix} files is not allowed")
        size = path.stat().st_size
        if size > self._max_read_bytes:
            raise ValueError(
                f"File too large to read ({size} bytes; max {self._max_read_bytes})"
            )

    def list_directory(
        self,
        relative_path: str = "",
        *,
        max_entries: int = _DEFAULT_LIST_ENTRIES,
    ) -> list[CodebaseEntry]:
        """List files and directories under a project path (non-recursive)."""
        directory = self.resolve_dir(relative_path)
        cap = max(1, min(max_entries, _MAX_LIST_ENTRIES))
        prefix = Path(relative_path.strip("/")) if relative_path.strip() else Path()
        rows: list[CodebaseEntry] = []

        for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.is_dir() and child.name in _SKIP_DIR_NAMES:
                continue
            if child.is_file() and child.name in _BLOCKED_FILE_NAMES:
                continue
            rel = (prefix / child.name).as_posix() if prefix.parts else child.name
            if child.is_dir():
                rows.append(CodebaseEntry(path=rel, kind="dir"))
            else:
                suffix = child.suffix.lower()
                if suffix in _BLOCKED_SUFFIXES:
                    continue
                rows.append(
                    CodebaseEntry(path=rel, kind="file", size_bytes=child.stat().st_size)
                )
            if len(rows) >= cap:
                break
        return rows

    def read_file(self, relative_path: str) -> str:
        path = self.resolve_file(relative_path)
        text = path.read_text(encoding="utf-8")
        if len(text) > self._max_chars:
            return text[: self._max_chars] + "\n… [truncated]"
        return text

    def format_listing(self, entries: list[CodebaseEntry], *, base_path: str) -> str:
        if not entries:
            label = base_path or "project root"
            return f"No listable entries under {label}."
        lines: list[str] = []
        for entry in entries:
            if entry.kind == "dir":
                lines.append(f"[dir]  {entry.path}/")
            else:
                size = entry.size_bytes or 0
                lines.append(f"[file] {entry.path} ({size} bytes)")
        header = f"Light-House codebase listing: {base_path or '.'}"
        return header + "\n" + "\n".join(lines)
