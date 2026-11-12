"""One slice of a long text, with a continuation a reader can follow.

The cap stays. The cap is not the end of the text.
"""

from __future__ import annotations


def slice_text_window(text: str, *, offset: int, max_chars: int) -> tuple[int, int, str, int]:
    """Return start, end, chunk, and total length.

    When start >= total the offset is past the end and chunk is empty.
    """
    body = text or ""
    total = len(body)
    start = max(0, int(offset or 0))
    window = max(1, int(max_chars))
    if start >= total:
        return start, total, "", total
    chunk = body[start : start + window]
    return start, start + len(chunk), chunk, total


def chars_label(start: int, end: int, total: int) -> str:
    """Where this window sits. An excerpt is anything other than the whole text."""
    if start >= total:
        return f"chars {start}-{total} of {total}"
    excerpt = " (excerpt)" if start > 0 or end < total else ""
    return f"chars {start}-{end} of {total}{excerpt}"


def continuation_footer(start: int, end: int, total: int) -> str:
    """The line a light already knows how to follow from notes and transcripts."""
    if end < total:
        return f"\n\n[excerpt {start}-{end} of {total}. Continue with offset={end}]"
    return ""


def past_end_note(start: int, total: int) -> str:
    return f"[Nothing remains at offset {start}; the text is {total} characters.]"
