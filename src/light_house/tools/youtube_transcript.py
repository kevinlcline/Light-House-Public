"""Fetch YouTube captions/transcripts for lights (no YouTube API key)."""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
)

logger = logging.getLogger(__name__)

# YouTube video ids are 11 chars from this alphabet.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }
)


def extract_video_id(url_or_id: str) -> str:
    """
    Parse a YouTube video id from a URL or bare id.

    Accepts watch, youtu.be, shorts, embed, live, and raw 11-char ids.
    """
    raw = (url_or_id or "").strip()
    if not raw:
        raise ValueError("A YouTube URL or video id is required")
    if _VIDEO_ID_RE.fullmatch(raw):
        return raw

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS and not host.endswith(".youtube.com"):
        # Bare path-ish strings sometimes arrive without scheme.
        if "youtube.com" in raw or "youtu.be" in raw:
            if "://" not in raw:
                return extract_video_id("https://" + raw.lstrip("/"))
        raise ValueError(
            "Not a YouTube URL or video id. "
            "Pass a watch/youtu.be/shorts link or an 11-character video id."
        )

    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.strip("/").split("/", 1)[0]
        if _VIDEO_ID_RE.fullmatch(candidate):
            return candidate
        raise ValueError(f"Could not parse video id from {raw!r}")

    path = parsed.path or ""
    for prefix in ("/shorts/", "/embed/", "/live/", "/v/"):
        if path.startswith(prefix):
            candidate = path[len(prefix) :].split("/", 1)[0]
            if _VIDEO_ID_RE.fullmatch(candidate):
                return candidate

    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        candidate = qs["v"][0].strip()
        if _VIDEO_ID_RE.fullmatch(candidate):
            return candidate

    raise ValueError(f"Could not parse video id from {raw!r}")


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _snippets_to_text(
    snippets: list[dict],
    *,
    timestamp_every_seconds: float = 60.0,
) -> str:
    parts: list[str] = []
    next_mark = 0.0
    for item in snippets:
        text = str(item.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        start = float(item.get("start") or 0.0)
        if start >= next_mark:
            parts.append(f"\n[{_format_timestamp(start)}] {text}")
            next_mark = start + max(15.0, timestamp_every_seconds)
        else:
            parts.append(text)
    body = " ".join(parts)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _fetch_transcript(video_id: str, languages: tuple[str, ...]):
    api = YouTubeTranscriptApi()
    preferred = tuple(languages) if languages else ("en",)
    try:
        return api.fetch(video_id, languages=preferred)
    except NoTranscriptFound:
        transcript_list = api.list(video_id)
        try:
            return transcript_list.find_transcript(list(preferred)).fetch()
        except Exception:
            pass
        try:
            return transcript_list.find_generated_transcript(list(preferred)).fetch()
        except Exception:
            pass
        # Any available track (manual first via iteration order of the list).
        for transcript in transcript_list:
            try:
                return transcript.fetch()
            except Exception:
                continue
        raise


def fetch_youtube_transcript(
    url_or_id: str,
    *,
    languages: tuple[str, ...] = ("en",),
    max_chars: int = 12_000,
    timestamp_every_seconds: float = 60.0,
) -> str:
    """
    Return a readable YouTube transcript for lights.

    Uses public captions (manual or auto-generated). No YouTube Data API key.
    """
    video_id = extract_video_id(url_or_id)
    fetched = _fetch_transcript(video_id, languages)
    snippets = fetched.to_raw_data()
    body = _snippets_to_text(
        snippets,
        timestamp_every_seconds=timestamp_every_seconds,
    )
    if not body:
        raise ValueError(f"Transcript for {video_id} was empty")

    language = getattr(fetched, "language_code", None) or getattr(fetched, "language", "") or "?"
    header = (
        f"YouTube transcript for https://www.youtube.com/watch?v={video_id}\n"
        f"Language: {language}\n"
    )
    text = header + "\n" + body
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n… [truncated]"
    return text


def youtube_transcript_error_message(exc: BaseException) -> str:
    """Map library/network failures to short tool-facing messages."""
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, TranscriptsDisabled):
        return "Captions are disabled for this video."
    if isinstance(exc, NoTranscriptFound):
        return "No transcript/captions were found for this video."
    if isinstance(exc, AgeRestricted):
        return "This video is age-restricted; captions could not be fetched."
    if isinstance(exc, (VideoUnavailable, VideoUnplayable, InvalidVideoId)):
        return "This YouTube video is unavailable or the id is invalid."
    if isinstance(exc, (IpBlocked, RequestBlocked)):
        return "YouTube blocked the transcript request from this server. Try again later."
    if isinstance(exc, CouldNotRetrieveTranscript):
        return f"Could not retrieve transcript: {exc}"
    logger.warning("Unexpected YouTube transcript error: %s", exc, exc_info=True)
    return f"Could not fetch YouTube transcript: {exc}"
