"""Studio 'now on the stage' clip — picture on the phone, words on the sitting."""

from __future__ import annotations

from typing import Any

from light_house.config import Settings
from light_house.tools.youtube_transcript import (
    excerpt_youtube_span,
    list_youtube_cues,
    load_youtube_snippets,
    parse_timestamp,
    youtube_transcript_error_message,
)

SHOW_TEXT_CAP = 8_000


def load_cues_for_studio(settings: Settings, url_or_id: str, *, refresh: bool = False) -> dict[str, Any]:
    if not settings.web_access_enabled:
        raise PermissionError("Web access is disabled (WEB_ACCESS_ENABLED=false).")
    return list_youtube_cues(
        url_or_id,
        cache_dir=settings.youtube_transcript_cache_path,
        refresh=refresh,
    )


def build_stage_show(
    settings: Settings,
    *,
    url_or_id: str,
    start: str | int | float,
    end: str | int | float,
    title: str = "",
) -> dict[str, Any]:
    if not settings.web_access_enabled:
        raise PermissionError("Web access is disabled (WEB_ACCESS_ENABLED=false).")
    start_seconds = parse_timestamp(start)
    end_seconds = parse_timestamp(end)
    if end_seconds <= start_seconds:
        raise ValueError("Out must be after In.")
    video_id, language, snippets = load_youtube_snippets(
        url_or_id,
        cache_dir=settings.youtube_transcript_cache_path,
    )
    text = excerpt_youtube_span(
        snippets,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        max_chars=SHOW_TEXT_CAP,
    )
    if not text:
        raise ValueError("No captions in that span.")
    label = (title or "").strip() or f"{_fmt(start_seconds)}–{_fmt(end_seconds)}"
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": label,
        "language": language,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "start_label": _fmt(start_seconds),
        "end_label": _fmt(end_seconds),
        "text": text,
    }


def format_stage_show_block(clip: dict[str, Any] | None) -> str:
    if not clip:
        return ""
    title = str(clip.get("title") or "Clip").strip() or "Clip"
    start = str(clip.get("start_label") or _fmt(float(clip.get("start_seconds") or 0)))
    end = str(clip.get("end_label") or _fmt(float(clip.get("end_seconds") or 0)))
    body = str(clip.get("text") or "").strip()
    if not body:
        return ""
    return (
        f"## Now on the stage\n"
        f"Clip: {title}  ({start}–{end})\n"
        f"{body}\n"
    )


def studio_show_error(exc: BaseException) -> str:
    if isinstance(exc, PermissionError):
        return str(exc)
    return youtube_transcript_error_message(exc)


def _fmt(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
