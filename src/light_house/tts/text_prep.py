"""Prepare assistant text for speech (strip markdown chrome)."""

from __future__ import annotations

import re

from light_house.tts.stage_cues import strip_stage_cues

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_~|~~)")
_BULLET_RE = re.compile(r"^\s*[-*+\u2022\u25E6\u2043\u2219]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_MULTI_PAUSE_RE = re.compile(r"(?:,\s*){2,}")
# Fancy punctuation that often trips espeak/phonemizer word-count alignment.
_FANCY_DASH_RE = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]+")  # hyphen/dash variants
_ELLIPSIS_RE = re.compile(r"\u2026+")
_SMART_QUOTES_RE = re.compile(r"[\u2018\u2019\u201A\u201B\u2032]")
_SMART_DBL_QUOTES_RE = re.compile(r"[\u201C\u201D\u201E\u201F\u2033]")
_MISC_BULLET_RE = re.compile(r"[\u2022\u25E6\u2043\u2219\u00B7]")
_HAS_SPEAKABLE_RE = re.compile(r"[A-Za-z0-9]")
# Path/URL-like tokens only (leave and/or, n/a alone).
_PATHISH_RE = re.compile(
    r"(?i)(?<![\w@])("
    r"https?://[^\s<>\")\]]+"
    r"|www\.[^\s<>\")\]]+"
    r"|[./]*(?:shared|notes|models|data|static|persona|v\d+)/[^\s<>\")\]]+"
    r"|(?:[A-Za-z0-9_.-]+/){2,}[A-Za-z0-9_.-]+"  # a/b/c… (3+ segments)
    r"|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.(?:md|markdown|txt|html|json|ya?ml|py|js|ts|css|png|jpe?g|gif|wav|mp3)\b"
    r")"
)
_URL_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)
_WWW_RE = re.compile(r"^www\.", re.IGNORECASE)
_SLASH_RE = re.compile(r"[\\/]+")
_FILE_EXT_RE = re.compile(
    r"\.(md|markdown|txt|html|json|ya?ml|py|js|ts|css|png|jpe?g|gif|wav|mp3)\b",
    re.IGNORECASE,
)
_LEADING_DOT_SLASH_RE = re.compile(r"^\./+")
# Emoji + presentation glue only — keep ranges tight so Latin text survives.
_EMOJI_RE = re.compile(
    "(?:"
    "[\U0001F1E0-\U0001F1FF]"  # flags
    "|[\U0001F300-\U0001F5FF]"  # symbols & pictographs
    "|[\U0001F600-\U0001F64F]"  # emoticons
    "|[\U0001F680-\U0001F6FF]"  # transport
    "|[\U0001F700-\U0001F77F]"
    "|[\U0001F780-\U0001F7FF]"
    "|[\U0001F800-\U0001F8FF]"
    "|[\U0001F900-\U0001F9FF]"
    "|[\U0001FA00-\U0001FA6F]"
    "|[\U0001FA70-\U0001FAFF]"
    "|[\U0001F3FB-\U0001F3FF]"  # skin tones
    "|[\u2600-\u26FF]"  # misc symbols (☀♥…)
    "|[\u2700-\u27BF]"  # dingbats
    "|[\u2300-\u23FF]"  # misc technical (⌚…)
    "|\uFE0F"  # variation selector-16
    "|\u200D"  # ZWJ
    "|\u20E3"  # combining enclosing keycap
    "|[\U000E0020-\U000E007F]"  # tags
    ")+",
)
_EMOJI_SPACE_RE = re.compile(r"[ \t]{2,}")


def text_for_speech(raw: str, *, max_chars: int = 4000) -> str:
    """Reduce markdown to speakable plain text."""
    text = (raw or "").strip()
    if not text:
        return ""
    text = strip_stage_cues(text)
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBERED_RE.sub("", text)
    text = _strip_emojis(text)
    text = _normalize_speech_punctuation(text)
    text = _soften_paths_and_urls(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_NL_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_PAUSE_RE.sub(", ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()
    # Punctuation-only leftovers make phonemizer warn and sound empty/wrong.
    if not _HAS_SPEAKABLE_RE.search(text):
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
        # ASCII ellipsis if we truncate — avoid reintroducing U+2026 alone.
        if text.endswith("…"):
            text = text[:-1].rstrip() + "..."
    return text


def _normalize_speech_punctuation(text: str) -> str:
    """Map fancy punctuation to ASCII so espeak/phonemizer stays aligned."""
    text = _ELLIPSIS_RE.sub("...", text)
    text = _FANCY_DASH_RE.sub("-", text)
    text = _SMART_QUOTES_RE.sub("'", text)
    text = _SMART_DBL_QUOTES_RE.sub('"', text)
    text = _MISC_BULLET_RE.sub(" ", text)
    return text


def _strip_emojis(text: str) -> str:
    """Remove emoji so TTS does not vocalize them."""
    cleaned = _EMOJI_RE.sub(" ", text)
    return _EMOJI_SPACE_RE.sub(" ", cleaned)


def _soften_one_pathish(token: str) -> str:
    text = _LEADING_DOT_SLASH_RE.sub("", token)
    text = _URL_SCHEME_RE.sub("", text)
    text = _WWW_RE.sub("", text)
    text = _FILE_EXT_RE.sub("", text)
    text = _SLASH_RE.sub(", ", text)
    text = re.sub(r"\s*,\s*,+", ", ", text).strip(" ,")
    return text


def _soften_paths_and_urls(text: str) -> str:
    """Turn path/URL punctuation into spoken pauses instead of 'slash'."""
    return _PATHISH_RE.sub(lambda m: _soften_one_pathish(m.group(0)), text)
