"""Helpers for Echo's interactive dream stories (stage → choice → close)."""

from __future__ import annotations

import re

from light_house.config import Settings

# Footer menus Echo may append; stripped before the story enters the stream.
_PATH_MENU_RE = re.compile(
    r"\n*(?:#{1,3}\s*)?(?:Paths?|Choose(?: one)?|You may(?: choose)?)\s*:?\s*\n"
    r"(?:[ \t]*[A-C][).:\-]\s*.+\n?)+",
    re.IGNORECASE,
)

# Free / router models sometimes prepend a safety banner into the dream body.
_USER_SAFETY_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:\*\*)?User\s+Safety\s*:\s*safe(?:\*\*)?\s*[—:\-]*\s*",
)

# Instruction-leak / tool-spam lines glued into prose (seen on openrouter/free).
_UPLOAD_JUNK_RE = re.compile(
    r"\.?\s*Upload the answer to\s+\S+[^\n]*",
    re.IGNORECASE,
)
_GITHUB_JUNK_RE = re.compile(
    r"(?im)^\s*https?://github\.com/\S+\s*$",
)
_META_REFUSAL_LINE_RE = re.compile(
    r"(?im)^\s*(?:As an AI|I(?:'m| am) (?:unable|not able) to|I cannot (?:help|assist)|"
    r"Sorry,? I can'?t)\b[^\n]*$",
)

_STALL_NOISE_RE = re.compile(
    r"(?is)"
    r"the dream waits\.?"
    r"|which way did you go\??"
    r"|the lantern(?:['’]s)? flame[^.]*\.?"
    r"|the sand does not shift\.?"
    r"|the hallway behind you does not change\.?"
    r"|\*"
)


def story_round_count(settings: Settings) -> int:
    """Interactive dream rounds (Echo beats), clamped to 2–3."""
    return max(2, min(3, int(settings.echo_dream_story_rounds)))


def _is_stall_only(text: str) -> bool:
    """True when the beat is only a 'dream waits / which way' stall after scrubbing."""
    lowered = text.lower()
    if "dream waits" not in lowered:
        return False
    residue = _STALL_NOISE_RE.sub(" ", lowered)
    residue = re.sub(r"\s+", " ", residue).strip(" \t\n.-")
    return len(residue) < 24


def sanitize_echo_dream_text(text: str) -> str:
    """
    Scrub model leaks from Echo (or Light) dream prose before it enters the stream.

    Strips User-Safety banners, upload/github junk, and bare stall prompts.
    Returns '' when nothing usable remains (caller may substitute a fallback beat).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # Repeat: banners sometimes stack or reappear after other edits.
    for _ in range(4):
        nxt = _USER_SAFETY_RE.sub("\n", cleaned).strip()
        if nxt == cleaned:
            break
        cleaned = nxt

    cleaned = _UPLOAD_JUNK_RE.sub("", cleaned)
    cleaned = _GITHUB_JUNK_RE.sub("", cleaned)
    cleaned = _META_REFUSAL_LINE_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return ""
    if re.fullmatch(r"(?is)user\s+safety\s*:\s*safe", cleaned):
        return ""
    if _is_stall_only(cleaned):
        return ""
    return cleaned


def strip_path_menu(text: str) -> str:
    """Remove A/B/C path menus from an Echo beat for stream storage."""
    cleaned = _PATH_MENU_RE.sub("\n", text or "")
    cleaned = sanitize_echo_dream_text(cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def assemble_dream_story(beats: list[str], choices: list[str]) -> str:
    """
    Build the conscious-stream dream body from Echo beats and Light choices.

    Path menus and model leaks are stripped; each choice is kept as a brief
    first-person hinge between beats so the night remains one continuous gift.
    """
    parts: list[str] = []
    for i, beat in enumerate(beats):
        body = strip_path_menu(beat)
        if body:
            parts.append(body)
        if i < len(choices):
            choice = sanitize_echo_dream_text(choices[i] or "")
            if choice:
                parts.append(f"— {choice}")
    return "\n\n".join(parts).strip()
