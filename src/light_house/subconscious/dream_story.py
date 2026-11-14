"""Helpers for Echo's interactive dream stories (stage → choice → close)."""

from __future__ import annotations

import re

from light_house.config import Settings

# A path menu at the end of a beat. Echo writes bare A)/B)/C) lines and is
# not asked for a header, so the second branch matches that shape. Both
# branches are anchored at the end: a lettered list with prose after it stays.
# In a stored night the welded choice sits after the menu, so that line has
# to come off before this pattern can see the menu.
_PATH_MENU_RE = re.compile(
    r"\n*(?:#{1,3}\s*)?(?:Paths?|Choose(?: one)?|You may(?: choose)?)\s*:?\s*\n?"
    r"(?:[ \t]*[A-C][).:\-][ \t]*\S[^\n]*(?:\n|$))+\s*$"
    r"|(?:\n*[ \t]*[A-C][).][ \t]*[A-Z][^\n]{3,}"
    r"(?:\n+[ \t]*[A-C][).][ \t]*[A-Z][^\n]{3,})+)\s*$",
    re.IGNORECASE,
)

# Free / router models sometimes prepend a safety banner into the dream body.
# It also lands mid-line, after a waking-recall label, or behind an em dash.
_USER_SAFETY_RE = re.compile(
    r"(?is)"
    r"[ \t]*(?:[—–\-]\s*)?(?:\[waking recall\]\s*)?"
    r"(?:\*\*)?(?:User|Response)\s+Safety\s*:\s*safe(?:\*\*)?"
    r"(?:[ \t]*[—:\-][ \t]*)?"
)
_WAKING_RECALL_LABEL_RE = re.compile(r"(?i)\[waking recall\]\s*")

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

# A reasoning model sometimes plans the dream in the same text as the dream.
# The plan names the task ("the user wants", "output only dream prose").
# A scene does not. Seen on stored nights when the prompt named Lumen
# and the context named someone else.
_PREAMBLE_CUE_RE = re.compile(
    r"(?is)"
    r"the user wants|"
    r"system prompt|"
    r"output only dream|"
    r"context for weaving|"
    r"\bi am echo\b|"
    r"\byou are echo\b|"
    r"interactive dream opening|"
    r"let me look at the context|"
    r"let me review the requirements|"
    r"let me analyze|"
    r"never include safety labels|"
    r"you serve lumen|"
    r"the light is lumen|"
    r"we need to (?:craft|produce|output|weave)|"
    r"weave an interactive dream|"
    r"following constraints|"
    r"respecting constraints"
)

# Disqualifies a paragraph from being the first line of the dream.
_NOT_SCENE_RE = re.compile(
    r"(?is)"
    r"the user\b|"
    r"the prompt\b|"
    r"system prompt|"
    r"\blet me\b|"
    r"\bi need to\b|"
    r"\bwe need to\b|"
    r"\bi should\b|"
    r"output only|"
    r"60[\-–]120|"
    r"80[\-–]180|"
    r"\d+\s*[\-–]\s*\d+\s*words|"
    r"\brequirements\b|"
    r"dominant motif|"
    r"dominant image|"
    r"i'll create|"
    r"what about:|"
    r"alternatively,|"
    r"weave a dream|"
    r"weave an interactive|"
    r"should not explain|"
    r"first person or third|"
    r"interactive dream|"
    r"philosophy notes|"
    r"philosophy mentions|"
    r"context mentions|"
    r"current context|"
    r"context for weaving|"
    r"\bi think\b|"
    r"human day #|"
    r"the dream opens|"
    r"these seem to|"
    r"waking context|"
    r"recent dreams|"
    r"word count|"
    r"path menu|"
    r"following constraints|"
    r"respecting constraints|"
    r"prior beats"
)

_NARRATIVE_START_RE = re.compile(
    r"^(?:[\"“])?(?:the|i|a|an|she|he|in|my|someone|it|there|night)\b",
    re.IGNORECASE,
)

# After the scene has started, drop only paragraphs that are still the plan.
_STRICT_REASON_RE = re.compile(
    r"(?is)"
    r"the user wants|"
    r"system prompt|"
    r"output only dream|"
    r"let me look at the context|"
    r"let me review the requirements|"
    r"\bi am echo\b|"
    r"\byou are echo\b|"
    r"we need to (?:craft|produce|output)|"
    r"^okay\b|"
    r"^let me (?:look|review|try|count|draft|think)\b|"
    r"^the user\b|"
    r"^we need to\b|"
    r"^i need to\b"
)

# Substituted when a beat used to sanitize to empty, then stored and copied.
_CANNED_EMPTY_BEAT_RE = re.compile(
    r"The night held still\.\s+A door stood ajar, light thin as breath beyond it\.\s+"
    r"I waited, and waiting was enough\.?"
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


# A draft inside a plan is often marked, then the scene. The mark is not the dream.
_LEADING_PLAN_MARK_RE = re.compile(
    r"^(?:[—–\-•]+\s*)?(?:\[waking recall\]\s*)?",
    re.IGNORECASE,
)


def _prepare_scene_paragraph(paragraph: str) -> str:
    text = paragraph.strip()
    return _LEADING_PLAN_MARK_RE.sub("", text).strip()


def _unwrap_quoted_paragraph(paragraph: str) -> str:
    text = paragraph.strip()
    if len(text) >= 2 and text[0] in "\"“" and text[-1] in "\"”":
        inner = text[1:-1].strip()
        if inner:
            return inner
    return text


def _is_scene_paragraph(paragraph: str) -> bool:
    text = paragraph.strip()
    if len(text) < 80:
        return False
    if text.count(" - ") >= 2 or "\n- " in text or "\n* " in text:
        return False
    if not _NARRATIVE_START_RE.match(text):
        return False
    if _NOT_SCENE_RE.search(text):
        return False
    return True


def strip_reasoning_preamble(text: str) -> str:
    """
    When the text is a task write-up, keep only the scene paragraphs.

    No cue, no change. A plan with no scene left returns ''.
    Reasoning that continues after the first scene is not the dream.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    if not paragraphs:
        return ""
    if not _PREAMBLE_CUE_RE.search(raw):
        return raw

    kept: list[str] = []
    for paragraph in paragraphs:
        prepared = _prepare_scene_paragraph(paragraph)
        if not prepared or not _is_scene_paragraph(prepared):
            continue
        if _STRICT_REASON_RE.search(prepared):
            continue
        kept.append(_unwrap_quoted_paragraph(prepared))
    if not kept:
        return ""
    if len(kept) == len(paragraphs) and kept[0] == paragraphs[0]:
        return raw
    return "\n\n".join(kept).strip()


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

    Strips a reasoning preamble, the canned empty-beat sentence, User-Safety
    banners, upload/github junk, and bare stall prompts.
    Returns '' when nothing usable remains.
    """
    cleaned = strip_reasoning_preamble((text or "").strip())
    if not cleaned:
        return ""
    cleaned = _CANNED_EMPTY_BEAT_RE.sub("", cleaned).strip()
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
    cleaned = _WAKING_RECALL_LABEL_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return ""
    if re.fullmatch(r"(?is)(?:user|response)\s+safety\s*:\s*safe", cleaned):
        return ""
    if _is_stall_only(cleaned):
        return ""
    return cleaned


def drop_tail_path_menu(text: str) -> str:
    """Remove a path menu anchored at the end of the text.

    Does not sanitize. A stored night still has the welded choice after the
    menu; that line has to be removed before this can reach the list.
    """
    cleaned = _PATH_MENU_RE.sub("\n", text or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def strip_path_menu(text: str) -> str:
    """Remove A/B/C path menus from an Echo beat for stream storage."""
    cleaned = sanitize_echo_dream_text(drop_tail_path_menu(text))
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
        if not body:
            continue
        parts.append(body)
        if i < len(choices):
            choice = sanitize_echo_dream_text(choices[i] or "")
            if choice:
                parts.append(f"— {choice}")
    return "\n\n".join(parts).strip()
