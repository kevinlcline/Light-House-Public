"""Speech chunking for pipelined TTS (mirrors static/ui/tts.js)."""

from __future__ import annotations

import re

_ABBREV_RE = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K)\.",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r'[^.!?]+(?:[.!?]+["\u201d\u2019]*)?|[^.!?]+$')

MAX_CHUNK_CHARS = 280
# Keep in sync with static/ui/tts.js — short clips stall on CPU synth of N+1.
MIN_CHUNK_CHARS = 72


def split_speech_chunks(
    text: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Split assistant text into sentence-ish chunks for sequential TTS."""
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return []

    def _protect(match: re.Match[str]) -> str:
        return match.group(0).replace(".", "\u2024")

    protected = _ABBREV_RE.sub(_protect, raw)
    rough: list[str] = []
    for para in re.split(r"\n{2,}", protected):
        line = re.sub(r"\s*\n\s*", " ", para).strip()
        if not line:
            continue
        pieces = _SENTENCE_RE.findall(line) or [line]
        for piece in pieces:
            cleaned = piece.replace("\u2024", ".").strip()
            if cleaned:
                rough.append(cleaned)

    merged: list[str] = []
    for piece in rough:
        if merged and (len(merged[-1]) < min_chars or len(piece) < min_chars):
            merged[-1] = f"{merged[-1]} {piece}".strip()
        else:
            merged.append(piece)

    chunks: list[str] = []
    for piece in merged:
        if len(piece) <= max_chars:
            chunks.append(piece)
            continue
        rest = piece
        while len(rest) > max_chars:
            cut = rest.rfind(" ", 0, max_chars)
            if cut < max_chars * 0.5:
                cut = max_chars
            chunks.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            chunks.append(rest)
    return [c for c in chunks if c]
