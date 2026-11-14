"""Detect markdown tables/lists that Studio shows as a stage modal."""

from __future__ import annotations

import re

# Header + separator row — same idea as Studio StageMarkdown.findTable.
_TABLE_SEP_RE = re.compile(
    r"(?m)^[ \t]*\|?(?:[ \t]*:?-{2,}:?[ \t]*\|){1,}[ \t]*:?-{2,}:?[ \t]*\|?[ \t]*$"
)
_LIST_ITEM_RE = re.compile(r"(?m)^[ \t]*(?:[-*+•]|\d+[\.)])\s+\S")


def utterance_holds_stage(text: str) -> bool:
    """True when the line should stay on stage until the host clears a modal."""
    raw = (text or "").strip()
    if not raw:
        return False
    if "|" in raw and _TABLE_SEP_RE.search(raw):
        return True
    return len(_LIST_ITEM_RE.findall(raw)) >= 2
