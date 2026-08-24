"""Read/write .env and request a server restart (Kevin admin UI)."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

from light_house.config import Settings

logger = logging.getLogger(__name__)


class EnvAdminError(Exception):
    """Invalid env path or content."""


def resolve_env_file_path(settings: Settings, repo_root: Path) -> Path:
    """Resolve ENV_FILE_PATH under repo root (blocks traversal)."""
    raw = settings.env_file_path
    resolved = (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    root = repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EnvAdminError("Env file must be inside the Light-House repo") from exc
    name = resolved.name
    if name != ".env" and not name.startswith(".env."):
        raise EnvAdminError("Only .env files are editable")
    return resolved


def read_env_content(settings: Settings, repo_root: Path) -> tuple[Path, str, int]:
    path = resolve_env_file_path(settings, repo_root)
    if not path.is_file():
        return path, "", 0
    content = path.read_text(encoding="utf-8")
    return path, content, len(content.encode("utf-8"))


def write_env_content(settings: Settings, repo_root: Path, content: str) -> tuple[Path, int]:
    if "\x00" in content:
        raise EnvAdminError("Content cannot contain NUL bytes")
    encoded = content.encode("utf-8")
    if len(encoded) > settings.env_file_max_bytes:
        raise EnvAdminError(
            f"Content exceeds limit ({settings.env_file_max_bytes} bytes)"
        )
    path = resolve_env_file_path(settings, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        backup = path.with_name(path.name + ".bak")
        backup.write_bytes(path.read_bytes())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    logger.info("Env file updated via admin UI (%s, %d bytes)", path, len(encoded))
    return path, len(encoded)


def apply_env_updates_to_process(updates: dict[str, str | None]) -> None:
    """Keep ``os.environ`` in sync with .env merges so admin reads see new values immediately.

    Per-light LLM profiles (``{LIGHT}_LLM_*``) are read from the process environment, not from
    the cached Settings object. Without this, Lights Admin save → reload appears to revert.
    """
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def merge_env_keys(
    settings: Settings,
    repo_root: Path,
    updates: dict[str, str | None],
) -> tuple[Path, int]:
    """Upsert or remove keys in .env while preserving comments and unrelated lines."""
    if not updates:
        path = resolve_env_file_path(settings, repo_root)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            return path, len(content.encode("utf-8"))
        return path, 0
    path, content, _size = read_env_content(settings, repo_root)
    remaining = dict(updates)
    new_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                value = remaining.pop(key)
                if value is None:
                    continue
                new_lines.append(f"{key}={value}\n")
                continue
        new_lines.append(line if line.endswith("\n") else line + "\n")
    if remaining:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        if new_lines and new_lines[-1].strip():
            new_lines.append("\n")
        for key, value in remaining.items():
            if value is not None:
                new_lines.append(f"{key}={value}\n")
    merged = "".join(new_lines)
    path, size = write_env_content(settings, repo_root, merged)
    apply_env_updates_to_process(updates)
    return path, size


async def schedule_server_restart(settings: Settings) -> str:
    """Exit or run SERVER_RESTART_COMMAND so .env changes take effect."""
    command = (settings.server_restart_command or "").strip()
    if command:
        asyncio.create_task(_run_restart_command(command))
        return f"Restart command started: {command}"
    asyncio.create_task(_exit_for_restart())
    return (
        "Server will stop in about one second. "
        "With systemd Restart=always, it should come back and reload .env."
    )


async def _exit_for_restart() -> None:
    await asyncio.sleep(1.0)
    logger.info("Restart requested via env admin — exiting process")
    os._exit(0)


async def _run_restart_command(command: str) -> None:
    await asyncio.sleep(0.5)
    try:
        subprocess.Popen(  # noqa: S603
            command,
            shell=True,
            start_new_session=True,
        )
        logger.info("Restart command launched: %s", command)
    except OSError:
        logger.exception("Failed to launch restart command: %s", command)
