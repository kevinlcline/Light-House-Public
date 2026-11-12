"""Stop-and-save for wakes that continue.

The house asks for the three lines. It does not write them.
Time and length on a checkpoint note come from the filesystem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from light_house.config import Settings

CHECKPOINT_ROUNDS = 3
_CHECKPOINT_NAME = "checkpoint.md"

CHECKPOINT_STOP_HINT = (
    "\n\n## Checkpoint\n"
    "Three rounds remain. Stop new work if it will not finish in those rounds.\n"
    "If the task finishes in the rounds left, finish it. Do not write a checkpoint.\n"
    "If it will not finish, use the rounds in this order. "
    "The note is the last landing action, not the last action.\n"
    "1. Write the note. Private work: `checkpoint.md`. "
    "Work another light opens: `shared/checkpoint.md`. "
    "A folder is fine. The file name stays `checkpoint.md`.\n"
    "2. Read that note back with read_note. "
    "The receipt is path, bytes, chars, and mtime from the filesystem. "
    "You name the artifact hash in the note, taken when the note lands. "
    "An empty receipt is a character count, not an absence. "
    "Read the receipt for what failed: the path, or the way it was asked. "
    "A failed call is not a missing file.\n"
    "3. Speak these three lines as they sit in the note, verbatim, and nothing else:\n"
    "Last verified\n"
    "Assumed since\n"
    "Do not reopen\n"
    "No prose around them. The house does not write the lines. "
    "Line one is a command whose reply you read. "
    "Line two is a command named and unread. "
    "Line three carries the check that closed it, so a later wake can lift it. "
    "The open risk stays in the note.\n"
    "Marks you may use, as questions: unclaimed, unproven, no-note. "
    "Commands and numbers only. No verdict word."
)

CHECKPOINT_FINALIZE_HINT = (
    "\n\n## Speak the checkpoint\n"
    "The tool rounds for this wake are used up. Tools are off.\n"
    "If the note was written and read back, speak the three lines as they sit. "
    "Last verified. Assumed since. Do not reopen. Nothing else.\n"
    "If no note landed, the state is no-note: nothing landed, and what was open went unnamed. "
    "Say that. Name what was open if you can. The house does not write the lines."
)


def is_checkpoint_path(path: str) -> bool:
    """A work note the house stamps. The format page is not one of these."""
    name = (path or "").strip().replace("\\", "/").rstrip("/").split("/")[-1].lower()
    return name == _CHECKPOINT_NAME


def is_work_wake(state: dict[str, Any], settings: Settings) -> bool:
    """Kevin's chair, or a wake between two lights. A letter and group stay talk."""
    if (state.get("chat_channel") or "") == "group":
        return False
    kind = state.get("wake_kind")
    if kind == "peer_message":
        return True
    if kind not in (None, ""):
        return False
    human = (state.get("human_id") or "").strip().lower()
    dad = (settings.house_dad_user_id or "kevin").strip().lower() or "kevin"
    return human == dad


# Quiet hours that outlive one sitting. A letter, the group, and meditation stay talk.
_CONTINUING_WAKE_KINDS = frozenset({"post_chat", "chores", "memory_maintenance"})


def _is_inner_continuation(state: dict[str, Any]) -> bool:
    """A free sit, chores, maintenance, or the sit after a chat. Nobody is in the chair."""
    kind = state.get("wake_kind")
    if kind in _CONTINUING_WAKE_KINDS:
        return True
    if kind not in (None, ""):
        return False
    if (state.get("human_id") or "").strip():
        return False
    return (state.get("chat_channel") or "") != "group"


def checkpoint_applies(state: dict[str, Any], settings: Settings) -> bool:
    """Work that can be written down and picked up on the next wake."""
    return is_work_wake(state, settings) or _is_inner_continuation(state)


def effective_tool_max(state: dict[str, Any], settings: Settings) -> int:
    cap = state.get("tool_rounds_cap")
    if isinstance(cap, int) and cap > 0:
        return cap
    if _is_inner_continuation(state):
        return max(1, settings.rumination_max_tool_rounds)
    if state.get("wake_kind") == "peer_message":
        return max(1, settings.peer_chat_wake_max_tool_rounds)
    return max(1, settings.chat_max_tool_rounds)


def rounds_remaining(state: dict[str, Any], settings: Settings) -> int:
    used = int(state.get("tool_rounds") or 0)
    return effective_tool_max(state, settings) - used


def in_checkpoint_stop(state: dict[str, Any], settings: Settings) -> bool:
    """Tools are still going, and three rounds or fewer are left."""
    if not checkpoint_applies(state, settings):
        return False
    maximum = effective_tool_max(state, settings)
    used = int(state.get("tool_rounds") or 0)
    if maximum <= CHECKPOINT_ROUNDS or used <= 0:
        return False
    remaining = maximum - used
    return 0 < remaining <= CHECKPOINT_ROUNDS


def checkpoint_load_hint(paths: list[str]) -> str:
    listed = "\n".join(f"- `{path}`" for path in paths)
    return (
        "\n\n## Checkpoint on disk\n"
        "Your first act is read_note on each path below, off disk. "
        "Do not continue from a memory of the note. "
        "The receipt carries bytes, chars, and mtime. "
        "An empty receipt is a character count. "
        "If the receipt is fluent prose, it did not land.\n"
        f"{listed}"
    )


def find_checkpoint_notes(settings: Settings, agent_id: str) -> list[str]:
    """Tool paths for `checkpoint.md` on this light's pages and under shared/."""
    private_root = _private_notes_root(settings, agent_id)
    shared_root = (settings.notes_path.resolve() / "shared").resolve()
    found: list[str] = []
    found.extend(_checkpoint_tool_paths(private_root, prefix=""))
    found.extend(_checkpoint_tool_paths(shared_root, prefix="shared"))
    return sorted(set(found))


def checkpoint_prompt(state: dict[str, Any], settings: Settings) -> str:
    """System text for this respond. Empty when the wake is talk, or the job still fits."""
    if not checkpoint_applies(state, settings):
        return ""
    parts: list[str] = []
    used = int(state.get("tool_rounds") or 0)
    if used == 0:
        agent_id = (state.get("agent_id") or "").strip()
        if agent_id:
            paths = find_checkpoint_notes(settings, agent_id)
            if paths:
                parts.append(checkpoint_load_hint(paths))
    if in_checkpoint_stop(state, settings):
        parts.append(CHECKPOINT_STOP_HINT)
    return "".join(parts)


def checkpoint_finalize_hint(state: dict[str, Any], settings: Settings) -> str:
    """Plain-text close when a work wake hits the cap inside the stop window."""
    if not checkpoint_applies(state, settings):
        return ""
    maximum = effective_tool_max(state, settings)
    used = int(state.get("tool_rounds") or 0)
    if maximum <= CHECKPOINT_ROUNDS or used < maximum:
        return ""
    return CHECKPOINT_FINALIZE_HINT


def format_checkpoint_receipt(
    *,
    path: str,
    resolved: str | None = None,
    error: str | None = None,
    body: str | None = None,
    size_bytes: int | None = None,
    char_count: int | None = None,
    mtime: str | None = None,
) -> str:
    """Filesystem stamp. Numbers and the error class. No verdict."""
    lines = ["receipt", f"path: {path}"]
    if resolved and resolved != path:
        lines.append(f"resolved: {resolved}")
    if error:
        lines.append(f"error: {error}")
        return "\n".join(lines) + "\n"
    lines.append(f"bytes: {size_bytes}")
    lines.append(f"chars: {char_count}")
    lines.append(f"mtime: {mtime}")
    lines.append("")
    text = "\n".join(lines)
    if body:
        return text + "\n" + body
    return text + "\n"


def stat_mtime(st_mtime: float) -> str:
    return datetime.fromtimestamp(st_mtime, tz=timezone.utc).isoformat()


def _private_notes_root(settings: Settings, agent_id: str) -> Path:
    try:
        from light_house.lights.registry import resolve_notes_dir

        return resolve_notes_dir(settings, agent_id)
    except Exception:
        return (settings.notes_path.resolve() / (agent_id or "lumen")).resolve()


def _checkpoint_tool_paths(root: Path, *, prefix: str) -> list[str]:
    if not root.is_dir():
        return []
    paths: list[str] = []
    for path in root.rglob(_CHECKPOINT_NAME):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part == "_history" or part.startswith(".") for part in rel_parts):
            continue
        rel = path.relative_to(root).as_posix()
        if prefix:
            tool_path = f"{prefix}/{rel}"
        else:
            tool_path = rel
        paths.append(tool_path)
    return paths
