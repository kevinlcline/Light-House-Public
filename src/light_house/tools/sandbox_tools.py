"""Code sandbox tools for lights (private workshop + shared playpen)."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from light_house.config import Settings
from light_house.lights.registry import known_light_ids

PLAYPEN_NAME = "sandbox"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def shared_root() -> Path:
    """Repo-root ``shared/`` (sandboxes live under ``shared/workspaces/``)."""
    return repo_root() / "shared"

SANDBOX_TOOL_NAMES = frozenset(
    {
        "sandbox_list",
        "sandbox_read",
        "sandbox_write",
        "sandbox_append",
        "sandbox_mkdir",
        "sandbox_delete",
        "sandbox_run",
    }
)

_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]*$")
_MAX_PATH_LEN = 512
_MAX_LIST_ENTRIES = 200
_RUN_TIMEOUT_SECS = 30
_RUN_OUTPUT_CHARS = 8_000
_ALLOWED_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".md",
        ".txt",
        ".html",
        ".css",
        ".toml",
        ".yaml",
        ".yml",
        ".csv",
        ".ini",
        ".cfg",
        ".xml",
        ".svg",
        ".sh",
        ".sql",
        ".r",
        ".rb",
        ".go",
        ".rs",
    }
)
_RUN_BINS = frozenset({"python", "python3"})
_SECRET_ENV_RE = re.compile(
    r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|OPENROUTER|OPENAI|ANTHROPIC)",
    re.I,
)


class SandboxError(ValueError):
    """User-facing sandbox path or policy error."""


@dataclass(frozen=True)
class SandboxTarget:
    """Resolved sandbox root for a tool call."""

    kind: Literal["own", "playpen", "peer"]
    light_id: str | None
    root: Path
    label: str


def workspaces_root() -> Path:
    return (shared_root() / "workspaces").resolve()


def ensure_workspace_layout(*, light_ids: frozenset[str] | None = None) -> Path:
    root = workspaces_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / PLAYPEN_NAME).mkdir(parents=True, exist_ok=True)
    ids = light_ids if light_ids is not None else known_light_ids()
    for light_id in sorted(ids):
        (root / light_id).mkdir(parents=True, exist_ok=True)
    return root


def _split_relative(relative: str) -> list[str]:
    if "\x00" in relative:
        raise SandboxError("Invalid path")
    raw = relative.strip().replace("\\", "/")
    if not raw or raw == ".":
        return []
    if raw.startswith("/"):
        raise SandboxError("Use a relative path inside the sandbox (no leading /)")
    parts = [p for p in raw.split("/") if p and p != "."]
    if not parts:
        return []
    if ".." in parts:
        raise SandboxError("Path must stay inside the sandbox (no ..)")
    return parts


def _validate_segment(part: str) -> None:
    if not _SAFE_NAME.match(part):
        raise SandboxError(
            "Path segments may only contain letters, numbers, dots, hyphens, and underscores"
        )


def sanitize_dir_path(relative_dir: str) -> str:
    parts = _split_relative(relative_dir)
    if not parts:
        raise SandboxError("Directory path is required")
    for part in parts:
        _validate_segment(part)
    result = "/".join(parts)
    if len(result) > _MAX_PATH_LEN:
        raise SandboxError(f"Path too long (max {_MAX_PATH_LEN} characters)")
    return result


def sanitize_file_path(relative_path: str) -> str:
    parts = _split_relative(relative_path)
    if not parts:
        raise SandboxError("File path is required")
    for part in parts[:-1]:
        _validate_segment(part)
    name = parts[-1]
    _validate_segment(name)
    suffix = Path(name).suffix.lower()
    if not suffix:
        raise SandboxError(
            f"Filename needs an extension. Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
        )
    if suffix not in _ALLOWED_SUFFIXES:
        raise SandboxError(
            f"Extension {suffix!r} not allowed. Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
        )
    if len(name) > 128:
        raise SandboxError("Filename too long (max 128 characters)")
    result = "/".join(parts)
    if len(result) > _MAX_PATH_LEN:
        raise SandboxError(f"Path too long (max {_MAX_PATH_LEN} characters)")
    return result


def _normalize_space(space: str | None, *, agent_id: str) -> str:
    raw = (space or "own").strip().lower()
    if not raw or raw in {"own", "me", "self", agent_id}:
        return "own"
    if raw in {"playpen", PLAYPEN_NAME, "shared_playpen", "joint"}:
        return "playpen"
    return raw


def resolve_target(
    *,
    agent_id: str,
    space: str | None = None,
    write: bool = False,
    run: bool = False,
) -> SandboxTarget:
    ensure_workspace_layout()
    agent = agent_id.strip().lower()
    if not agent:
        raise SandboxError("light_id is required")
    kind = _normalize_space(space, agent_id=agent)
    root_base = workspaces_root()

    if kind == "own":
        root = (root_base / agent).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return SandboxTarget(kind="own", light_id=agent, root=root, label=f"own ({agent})")

    if kind == "playpen":
        root = (root_base / PLAYPEN_NAME).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return SandboxTarget(kind="playpen", light_id=None, root=root, label="playpen")

    # Peer sandbox (read-only)
    known = known_light_ids()
    if kind not in known and not (root_base / kind).is_dir():
        raise SandboxError(
            f"Unknown sandbox space {kind!r}. Use own, playpen, or a light id."
        )
    if write or run:
        raise SandboxError(
            f"Cannot write/run in another light's sandbox ({kind}). "
            "Use space=own or space=playpen for writes/runs; peer space is read-only."
        )
    root = (root_base / kind).resolve()
    if not root.is_dir():
        raise SandboxError(f"Peer sandbox not found: {kind}")
    return SandboxTarget(kind="peer", light_id=kind, root=root, label=f"peer ({kind})")


def _resolve_under(root: Path, relative: str, *, as_dir: bool = False) -> Path:
    safe = sanitize_dir_path(relative) if as_dir else sanitize_file_path(relative)
    target = (root / safe).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SandboxError("Path must stay inside the sandbox") from exc
    return target


def _validate_content(content: str, *, max_chars: int) -> str:
    if not isinstance(content, str):
        content = str(content)
    if "\x00" in content:
        raise SandboxError("Content cannot contain NUL bytes")
    if len(content) > max_chars:
        raise SandboxError(f"Content too long (max {max_chars} characters)")
    return content


def _success(message: str) -> str:
    return f"SUCCESS: {message}"


def _failed(message: str) -> str:
    return f"FAILED: {message}"


def sandbox_list(
    *,
    agent_id: str,
    path: str = "",
    space: str | None = None,
) -> str:
    target = resolve_target(agent_id=agent_id, space=space, write=False)
    rel = path.strip()
    if rel:
        folder = _resolve_under(target.root, rel, as_dir=True)
    else:
        folder = target.root
    if not folder.exists():
        return _failed(f"sandbox_list — path not found in {target.label}: {rel or '.'}")
    if not folder.is_dir():
        return _failed(f"sandbox_list — not a directory: {rel}")

    entries: list[str] = []
    try:
        children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return _failed(f"sandbox_list — {exc}")

    for child in children:
        if len(entries) >= _MAX_LIST_ENTRIES:
            entries.append("… (truncated)")
            break
        name = child.name
        if child.is_dir():
            entries.append(f"{name}/")
        else:
            entries.append(name)

    header = f"Sandbox {target.label}"
    if rel:
        header += f" / {sanitize_dir_path(rel)}"
    if not entries:
        hint = (
            " (empty — create files with sandbox_write, e.g. "
            "path=README.md or experiments/hello.py; playpen uses space=playpen)"
        )
        return _success(f"{header}{hint}\n(no entries)")
    return _success(f"{header}\n" + "\n".join(entries))


def sandbox_read(
    *,
    agent_id: str,
    path: str,
    space: str | None = None,
    max_chars: int = 32_000,
) -> str:
    target = resolve_target(agent_id=agent_id, space=space, write=False)
    file_path = _resolve_under(target.root, path, as_dir=False)
    if not file_path.is_file():
        return _failed(f"sandbox_read — file not found in {target.label}: {path}")
    size = file_path.stat().st_size
    if size > max_chars * 4:
        return _failed(f"sandbox_read — file too large (max ~{max_chars} characters)")
    text = file_path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… [truncated]"
    return _success(f"sandbox_read ({target.label}) {sanitize_file_path(path)}\n\n{text}")


def sandbox_write(
    *,
    agent_id: str,
    path: str,
    content: str,
    space: str | None = None,
    max_chars: int = 32_000,
) -> str:
    target = resolve_target(agent_id=agent_id, space=space, write=True)
    text = _validate_content(content, max_chars=max_chars)
    if not text.strip():
        return _failed("sandbox_write — content cannot be empty")
    file_path = _resolve_under(target.root, path, as_dir=False)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
    rel = sanitize_file_path(path)
    return _success(f"sandbox_write ({target.label}) wrote {rel} ({len(text)} chars)")


def sandbox_append(
    *,
    agent_id: str,
    path: str,
    content: str,
    space: str | None = None,
    max_chars: int = 32_000,
) -> str:
    target = resolve_target(agent_id=agent_id, space=space, write=True)
    text = _validate_content(content, max_chars=max_chars)
    if not text:
        return _failed("sandbox_append — content cannot be empty")
    file_path = _resolve_under(target.root, path, as_dir=False)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if file_path.is_file():
        existing = file_path.read_text(encoding="utf-8")
        if len(existing) + len(text) > max_chars:
            return _failed(f"sandbox_append — result would exceed {max_chars} characters")
        if existing and not existing.endswith("\n"):
            existing += "\n"
    combined = existing + text
    file_path.write_text(combined, encoding="utf-8")
    return _success(
        f"sandbox_append ({target.label}) updated {sanitize_file_path(path)} "
        f"({len(combined)} chars total)"
    )


def sandbox_mkdir(
    *,
    agent_id: str,
    path: str,
    space: str | None = None,
) -> str:
    target = resolve_target(agent_id=agent_id, space=space, write=True)
    folder = _resolve_under(target.root, path, as_dir=True)
    folder.mkdir(parents=True, exist_ok=True)
    return _success(f"sandbox_mkdir ({target.label}) {sanitize_dir_path(path)}")


def sandbox_delete(
    *,
    agent_id: str,
    path: str,
    space: str | None = None,
) -> str:
    target = resolve_target(agent_id=agent_id, space=space, write=True)
    # Try file first, then empty directory.
    try:
        file_path = _resolve_under(target.root, path, as_dir=False)
        if file_path.is_file():
            file_path.unlink()
            return _success(f"sandbox_delete ({target.label}) deleted file {sanitize_file_path(path)}")
    except SandboxError:
        pass

    folder = _resolve_under(target.root, path, as_dir=True)
    if not folder.exists():
        return _failed(f"sandbox_delete — not found in {target.label}: {path}")
    if not folder.is_dir():
        return _failed(f"sandbox_delete — not a file or directory: {path}")
    try:
        next(folder.iterdir())
        return _failed(
            "sandbox_delete — directory is not empty "
            "(delete files first, or delete only empty dirs)"
        )
    except StopIteration:
        pass
    folder.rmdir()
    return _success(f"sandbox_delete ({target.label}) removed empty dir {sanitize_dir_path(path)}")


def _blocked_run_tokens(command: str) -> str | None:
    for token in (";", "&&", "||", "|", "`", "$(", ">", "<", "\n"):
        if token in command:
            return f"Unsupported shell operator in command: {token!r}"
    lowered = command.lower().strip()
    if lowered.startswith("sudo"):
        return "sudo is not allowed"
    return None


def _scrubbed_env() -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in os.environ.items():
        if _SECRET_ENV_RE.search(key):
            continue
        cleaned[key] = value
    # Keep a minimal useful PATH/HOME/LANG if present.
    return cleaned


def sandbox_run(
    *,
    agent_id: str,
    command: str,
    space: str | None = None,
    cwd: str = "",
    timeout_secs: int = _RUN_TIMEOUT_SECS,
) -> str:
    target = resolve_target(agent_id=agent_id, space=space, write=True, run=True)

    cmd = (command or "").strip()
    if not cmd:
        return _failed("sandbox_run — command is required")
    blocked = _blocked_run_tokens(cmd)
    if blocked:
        return _failed(f"sandbox_run — {blocked}")

    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        return _failed(f"sandbox_run — {exc}")
    if not argv:
        return _failed("sandbox_run — empty command")
    bin_name = Path(argv[0]).name
    if bin_name not in _RUN_BINS:
        return _failed(
            f"sandbox_run — only {', '.join(sorted(_RUN_BINS))} allowed for now "
            f"(got {bin_name!r}). Node comes later."
        )

    if cwd.strip():
        work = _resolve_under(target.root, cwd, as_dir=True)
        if not work.is_dir():
            return _failed(f"sandbox_run — cwd not found: {cwd}")
    else:
        work = target.root

    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_secs)),
            check=False,
            env=_scrubbed_env(),
        )
    except subprocess.TimeoutExpired:
        return _failed(f"sandbox_run — timed out after {timeout_secs}s")
    except OSError as exc:
        return _failed(f"sandbox_run — {exc}")

    elapsed = time.monotonic() - started
    out = (completed.stdout or "") + (completed.stderr or "")
    out = out.strip() or "(no output)"
    if len(out) > _RUN_OUTPUT_CHARS:
        out = out[: _RUN_OUTPUT_CHARS - 20] + "\n… [truncated]"
    status = "ok" if completed.returncode == 0 else "error"
    return _success(
        f"sandbox_run ({target.label}) {status} exit={completed.returncode} "
        f"in {elapsed:.2f}s\n$ {cmd}\n{out}"
    )


def execute_sandbox_tool(
    name: str,
    args: dict[str, Any],
    *,
    agent_id: str,
    settings: Settings,
) -> str:
    space = args.get("space")
    space_s = str(space).strip() if space is not None else None
    path = str(args.get("path") or args.get("filename") or "")
    max_write = settings.notes_max_chars_per_write
    max_read = settings.codebase_max_chars_per_read

    try:
        if name == "sandbox_list":
            return sandbox_list(agent_id=agent_id, path=path, space=space_s)
        if name == "sandbox_read":
            return sandbox_read(
                agent_id=agent_id,
                path=path,
                space=space_s,
                max_chars=max_read,
            )
        if name == "sandbox_write":
            return sandbox_write(
                agent_id=agent_id,
                path=path,
                content=str(args.get("content") or ""),
                space=space_s,
                max_chars=max_write,
            )
        if name == "sandbox_append":
            return sandbox_append(
                agent_id=agent_id,
                path=path,
                content=str(args.get("content") or ""),
                space=space_s,
                max_chars=max_write,
            )
        if name == "sandbox_mkdir":
            return sandbox_mkdir(agent_id=agent_id, path=path, space=space_s)
        if name == "sandbox_delete":
            return sandbox_delete(agent_id=agent_id, path=path, space=space_s)
        if name == "sandbox_run":
            cwd = str(args.get("cwd") or "")
            timeout = args.get("timeout_secs") or _RUN_TIMEOUT_SECS
            try:
                timeout_i = int(timeout)
            except (TypeError, ValueError):
                timeout_i = _RUN_TIMEOUT_SECS
            return sandbox_run(
                agent_id=agent_id,
                command=str(args.get("command") or ""),
                space=space_s,
                cwd=cwd,
                timeout_secs=min(max(timeout_i, 1), 120),
            )
    except SandboxError as exc:
        return _failed(f"{name} — {exc}")
    raise KeyError(name)
