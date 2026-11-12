"""Fetch YouTube captions/transcripts for lights (no YouTube API key)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
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
_TS_IN_TEXT = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\]")
_HARD_WINDOW_CAP = 32_000
DEFAULT_WINDOW_CHARS = 12_000
DEFAULT_CACHE_DIR = Path("./data/youtube_transcripts")


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


def parse_timestamp(raw: str | int | float | None) -> float:
    """Parse `1:05:00`, `12:00`, or a seconds number into seconds."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        if raw < 0:
            raise ValueError("start must be zero or later")
        return float(raw)
    text = str(raw).strip()
    if not text:
        return 0.0
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
        raise ValueError(
            "start must be seconds or a timestamp like 12:00 or 1:05:00"
        )
    nums = [float(p) for p in parts]
    if len(nums) == 2:
        minutes, secs = nums
        hours = 0.0
    else:
        hours, minutes, secs = nums
    return hours * 3600.0 + minutes * 60.0 + secs


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


def _caption_end_seconds(snippets: list[dict]) -> float:
    if not snippets:
        return 0.0
    last = snippets[-1]
    return float(last.get("start") or 0.0) + float(last.get("duration") or 0.0)


def _cache_path(cache_dir: Path, video_id: str) -> Path:
    return cache_dir / f"{video_id}.json"


def _load_cached(cache_dir: Path | None, video_id: str) -> dict | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, video_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read YouTube transcript cache %s: %s", path, exc)
        return None
    snippets = data.get("snippets")
    if not isinstance(snippets, list) or not snippets:
        return None
    return data


def _save_cache(cache_dir: Path | None, video_id: str, language: str, snippets: list[dict]) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, video_id)
        path.write_text(
            json.dumps(
                {"video_id": video_id, "language": language, "snippets": snippets},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write YouTube transcript cache: %s", exc)


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


def load_youtube_snippets(
    url_or_id: str,
    *,
    languages: tuple[str, ...] = ("en",),
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> tuple[str, str, list[dict]]:
    """Return (video_id, language, snippets), using the on-disk cache when present."""
    video_id = extract_video_id(url_or_id)
    if not refresh:
        cached = _load_cached(cache_dir, video_id)
        if cached is not None:
            language = str(cached.get("language") or "?")
            return video_id, language, list(cached["snippets"])
    fetched = _fetch_transcript(video_id, languages)
    snippets = fetched.to_raw_data()
    language = getattr(fetched, "language_code", None) or getattr(fetched, "language", "") or "?"
    if not snippets:
        raise ValueError(f"Transcript for {video_id} was empty")
    _save_cache(cache_dir, video_id, str(language), snippets)
    return video_id, str(language), snippets


def format_full_transcript(
    video_id: str,
    language: str,
    snippets: list[dict],
    *,
    timestamp_every_seconds: float = 60.0,
) -> str:
    """Full caption text with a short header — no character window."""
    body = _snippets_to_text(snippets, timestamp_every_seconds=timestamp_every_seconds)
    if not body:
        raise ValueError(f"Transcript for {video_id} was empty")
    end = _format_timestamp(_caption_end_seconds(snippets))
    header = (
        f"# YouTube transcript\n\n"
        f"Source: https://www.youtube.com/watch?v={video_id}\n"
        f"Language: {language}\n"
        f"Captions cover: 0:00–{end}\n"
    )
    return header + "\n" + body + "\n"


def fetch_youtube_transcript(
    url_or_id: str,
    *,
    languages: tuple[str, ...] = ("en",),
    max_chars: int = DEFAULT_WINDOW_CHARS,
    timestamp_every_seconds: float = 60.0,
    start: str | int | float | None = None,
    offset: int = 0,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> str:
    """
    Return a readable YouTube transcript window for lights.

    Uses public captions (manual or auto-generated). No YouTube Data API key.
    Long videos are paged: pass ``start`` (timestamp) or ``offset`` (characters)
    to continue. ``max_chars <= 0`` returns the full remaining text.
    """
    video_id, language, snippets = load_youtube_snippets(
        url_or_id,
        languages=languages,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    start_seconds = parse_timestamp(start)
    selected = [
        item
        for item in snippets
        if float(item.get("start") or 0.0) >= start_seconds
    ]
    if not selected:
        end = _format_timestamp(_caption_end_seconds(snippets))
        return (
            f"YouTube transcript for https://www.youtube.com/watch?v={video_id}\n"
            f"Language: {language}\n"
            f"Captions cover 0:00–{end}. Nothing remains at start={_format_timestamp(start_seconds)}."
        )
    body = _snippets_to_text(
        selected,
        timestamp_every_seconds=timestamp_every_seconds,
    )
    if not body:
        raise ValueError(f"Transcript for {video_id} was empty")

    start_char = max(0, int(offset or 0))
    unlimited = max_chars is None or int(max_chars) <= 0
    window = len(body) if unlimited else max(1, min(int(max_chars), _HARD_WINDOW_CAP))
    total = len(body)
    if start_char >= total:
        end = _format_timestamp(_caption_end_seconds(snippets))
        return (
            f"YouTube transcript for https://www.youtube.com/watch?v={video_id}\n"
            f"Language: {language}\n"
            f"[Transcript empty at offset {start_char}; "
            f"{total} characters remain from {_format_timestamp(start_seconds)}; "
            f"full captions cover 0:00–{end}]\n"
        )
    chunk = body[start_char : start_char + window]
    end_char = start_char + len(chunk)
    full_end = _format_timestamp(_caption_end_seconds(snippets))
    marks = _TS_IN_TEXT.findall(chunk)
    window_from = marks[0] if marks else _format_timestamp(start_seconds)
    window_to = marks[-1] if marks else window_from
    header = (
        f"YouTube transcript for https://www.youtube.com/watch?v={video_id}\n"
        f"Language: {language}\n"
        f"Captions cover 0:00–{full_end} ({total} characters from "
        f"{_format_timestamp(start_seconds)}).\n"
    )
    if start_char == 0 and end_char >= total:
        return header + "\n" + body
    header += (
        f"This window: characters {start_char}-{end_char} of {total} "
        f"(about {window_from}–{window_to}).\n"
        f"To continue: youtube_transcript with offset={end_char}"
        + (f' or start="{window_to}"' if window_to else "")
        + ".\n"
    )
    text = header + "\n" + chunk
    if end_char < total:
        text = text.rstrip() + "\n… [truncated]"
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


def list_youtube_cues(
    url_or_id: str,
    *,
    languages: tuple[str, ...] = ("en",),
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> dict:
    """Timed caption cues for the Studio clip picker (host, not a light tool)."""
    video_id, language, snippets = load_youtube_snippets(
        url_or_id,
        languages=languages,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    cues: list[dict] = []
    for item in snippets:
        text = str(item.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        cues.append(
            {
                "start": float(item.get("start") or 0.0),
                "duration": float(item.get("duration") or 0.0),
                "text": text,
            }
        )
    if not cues:
        raise ValueError(f"Transcript for {video_id} was empty")
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "language": language,
        "duration_seconds": _caption_end_seconds(snippets),
        "cues": cues,
    }


def excerpt_youtube_span(
    snippets: list[dict],
    *,
    start_seconds: float,
    end_seconds: float,
    max_chars: int = 8_000,
) -> str:
    """Plain caption text for [start, end) on the source clock."""
    start_at = max(0.0, float(start_seconds))
    end_at = max(start_at, float(end_seconds))
    selected = [
        item
        for item in snippets
        if start_at <= float(item.get("start") or 0.0) < end_at
    ]
    if not selected:
        return ""
    body = _snippets_to_text(selected)
    window = max(1, int(max_chars))
    if len(body) > window:
        return body[: window - 20].rstrip() + "\n… [excerpt capped]"
    return body


def dump_youtube_transcript(
    url_or_id: str,
    dest: Path,
    *,
    languages: tuple[str, ...] = ("en",),
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> Path:
    """Write the full caption file (no 12k window) for host-side sharing."""
    video_id, language, snippets = load_youtube_snippets(
        url_or_id,
        languages=languages,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    text = format_full_transcript(video_id, language, snippets)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m light_house.tools.youtube_transcript",
        description=(
            "Fetch public YouTube captions. Print a window, or dump the full "
            "transcript into a note the lights can read_note."
        ),
    )
    parser.add_argument("url", help="YouTube watch/youtu.be/shorts URL or 11-character id")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the entire transcript here (e.g. notes/shared/youtube/video-id.md)",
    )
    parser.add_argument("--start", default=None, help="Window start: 12:00, 1:05:00, or seconds")
    parser.add_argument("--offset", type=int, default=0, help="Character offset into the remaining text")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_WINDOW_CHARS,
        help="Window size when printing (ignored with --out)",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore the local caption cache")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Caption cache directory",
    )
    args = parser.parse_args(argv)
    try:
        if args.out is not None:
            path = dump_youtube_transcript(
                args.url,
                args.out,
                cache_dir=args.cache_dir,
                refresh=args.refresh,
            )
            print(f"Wrote {path} ({path.stat().st_size} bytes)")
            return 0
        text = fetch_youtube_transcript(
            args.url,
            start=args.start,
            offset=args.offset,
            max_chars=args.max_chars,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return 0
    except Exception as exc:
        print(youtube_transcript_error_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
