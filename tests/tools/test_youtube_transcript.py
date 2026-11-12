"""Tests for YouTube transcript tool helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from light_house.agent.tool_helpers import _summarize_tool_call
from light_house.config import Settings
from light_house.tools.lumen_tools import execute_tool_call
from light_house.tools.light_tools import LIGHT_TOOLS, _youtube_transcript_for
from light_house.tools.youtube_transcript import (
    dump_youtube_transcript,
    excerpt_youtube_span,
    extract_video_id,
    fetch_youtube_transcript,
    list_youtube_cues,
    parse_timestamp,
    youtube_transcript_error_message,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("jNQXAC9IVRw", "jNQXAC9IVRw"),
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "jNQXAC9IVRw"),
        ("https://youtu.be/jNQXAC9IVRw", "jNQXAC9IVRw"),
        ("https://www.youtube.com/shorts/jNQXAC9IVRw", "jNQXAC9IVRw"),
        ("https://www.youtube.com/embed/jNQXAC9IVRw", "jNQXAC9IVRw"),
        ("https://www.youtube.com/live/jNQXAC9IVRw", "jNQXAC9IVRw"),
        ("youtube.com/watch?v=jNQXAC9IVRw&t=12s", "jNQXAC9IVRw"),
    ],
)
def test_extract_video_id_accepts_common_forms(raw: str, expected: str):
    assert extract_video_id(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "nope", "https://example.com/watch?v=jNQXAC9IVRw", "too-short", "this-id-is-way-too-long"],
)
def test_extract_video_id_rejects_bad_input(raw: str):
    with pytest.raises(ValueError):
        extract_video_id(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 0.0),
        ("", 0.0),
        (12, 12.0),
        ("90", 90.0),
        ("12:00", 720.0),
        ("1:05:00", 3900.0),
        ("1:02:03", 3723.0),
    ],
)
def test_parse_timestamp(raw, expected):
    assert parse_timestamp(raw) == expected


def test_parse_timestamp_rejects_junk():
    with pytest.raises(ValueError):
        parse_timestamp("noon")


def _fetched(snippets: list[dict]):
    return SimpleNamespace(
        language_code="en",
        to_raw_data=lambda: snippets,
    )


def _long_snippets() -> list[dict]:
    return [
        {"text": "Hello there", "start": 0.0, "duration": 1.0},
        {"text": "General\nKenobi", "start": 1.5, "duration": 1.0},
        {"text": "Later line about the middle of the talk", "start": 65.0, "duration": 2.0},
        {"text": "Hour two begins with a new argument", "start": 3600.0, "duration": 3.0},
    ]


def test_fetch_youtube_transcript_formats_and_windows():
    snippets = _long_snippets()
    fetched = _fetched(snippets)
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        text = fetch_youtube_transcript(
            "jNQXAC9IVRw",
            max_chars=500,
            cache_dir=None,
        )
    assert "watch?v=jNQXAC9IVRw" in text
    assert "Language: en" in text
    assert "Hello there" in text
    assert "General Kenobi" in text
    assert "[1:05]" in text
    assert "Hour two" in text
    assert "1:00:00" in text or "1:00:03" in text

    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        truncated = fetch_youtube_transcript(
            "jNQXAC9IVRw",
            max_chars=40,
            cache_dir=None,
        )
    assert "This window:" in truncated
    assert "offset=" in truncated
    assert truncated.rstrip().endswith("… [truncated]")
    assert "Hour two" not in truncated


def test_fetch_youtube_transcript_pages_by_start_and_offset():
    fetched = _fetched(_long_snippets())
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        later = fetch_youtube_transcript(
            "jNQXAC9IVRw",
            start="1:00:00",
            max_chars=500,
            cache_dir=None,
        )
    assert "Hour two" in later
    assert "Hello there" not in later

    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        first = fetch_youtube_transcript(
            "jNQXAC9IVRw",
            max_chars=40,
            cache_dir=None,
        )
    assert "offset=" in first
    offset = int(first.split("offset=")[1].split()[0].rstrip("."))
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        second = fetch_youtube_transcript(
            "jNQXAC9IVRw",
            offset=offset,
            max_chars=400,
            cache_dir=None,
        )
    assert "Hour two" in second


def test_cache_avoids_a_second_youtube_hit(tmp_path):
    fetched = _fetched(_long_snippets())
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ) as mocked:
        first = fetch_youtube_transcript("jNQXAC9IVRw", max_chars=500, cache_dir=tmp_path)
        second = fetch_youtube_transcript("jNQXAC9IVRw", max_chars=500, cache_dir=tmp_path)
    assert mocked.call_count == 1
    assert "Hello there" in first
    assert "Hello there" in second
    assert (tmp_path / "jNQXAC9IVRw.json").is_file()


def test_dump_writes_full_captions(tmp_path):
    dest = tmp_path / "shared" / "youtube" / "jNQXAC9IVRw.md"
    fetched = _fetched(_long_snippets())
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        dump_youtube_transcript("jNQXAC9IVRw", dest, cache_dir=None)
    text = dest.read_text(encoding="utf-8")
    assert text.startswith("# YouTube transcript")
    assert "Hour two" in text
    assert "truncated" not in text


def test_youtube_transcript_tool_disabled_by_web_access():
    settings = Settings(_env_file=None, WEB_ACCESS_ENABLED=False)
    assert "disabled" in _youtube_transcript_for(settings, "jNQXAC9IVRw").lower()


def test_execute_tool_call_youtube_transcript():
    settings = Settings(_env_file=None, WEB_ACCESS_ENABLED=True, YOUTUBE_TRANSCRIPT_MAX_CHARS=500)
    fake = "YouTube transcript for https://www.youtube.com/watch?v=jNQXAC9IVRw\n\nHi"
    with (
        patch("light_house.tools.light_tools.get_settings", return_value=settings),
        patch(
            "light_house.tools.light_tools.fetch_youtube_transcript",
            return_value=fake,
        ) as mocked,
    ):
        result = execute_tool_call(
            "youtube_transcript",
            {"url_or_id": "https://youtu.be/jNQXAC9IVRw", "start": "12:00", "offset": 20},
            agent_id="lumen",
        )
    assert result == fake
    mocked.assert_called_once()
    assert mocked.call_args.args[0] == "https://youtu.be/jNQXAC9IVRw"
    assert mocked.call_args.kwargs["start"] == "12:00"
    assert mocked.call_args.kwargs["offset"] == 20
    assert mocked.call_args.kwargs["max_chars"] == 500


def test_execute_tool_call_accepts_url_alias():
    settings = Settings(_env_file=None, WEB_ACCESS_ENABLED=True)
    with (
        patch("light_house.tools.light_tools.get_settings", return_value=settings),
        patch(
            "light_house.tools.light_tools.fetch_youtube_transcript",
            return_value="ok",
        ) as mocked,
    ):
        result = execute_tool_call(
            "youtube_transcript",
            {"url": "jNQXAC9IVRw"},
            agent_id="lumen",
        )
    assert result == "ok"
    assert mocked.call_args.args[0] == "jNQXAC9IVRw"


def test_error_message_for_transcripts_disabled():
    from youtube_transcript_api import TranscriptsDisabled

    msg = youtube_transcript_error_message(TranscriptsDisabled("jNQXAC9IVRw"))
    assert "disabled" in msg.lower()


def test_tool_is_registered():
    names = {t.name for t in LIGHT_TOOLS}
    assert "youtube_transcript" in names


def test_list_cues_and_excerpt_span():
    fetched = _fetched(_long_snippets())
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        payload = list_youtube_cues("jNQXAC9IVRw", cache_dir=None)
    assert payload["video_id"] == "jNQXAC9IVRw"
    assert payload["cues"][0]["text"] == "Hello there"
    assert payload["duration_seconds"] >= 3600
    excerpt = excerpt_youtube_span(
        payload["cues"],
        start_seconds=3600,
        end_seconds=3700,
    )
    assert "Hour two" in excerpt
    assert "Hello there" not in excerpt


def test_summarize_youtube_transcript_call():
    summary = _summarize_tool_call(
        "youtube_transcript",
        {"url_or_id": "https://youtu.be/jNQXAC9IVRw", "start": "12:00"},
    )
    assert summary.startswith("youtube_transcript(")
    assert "jNQXAC9IVRw" in summary
    assert "start=12:00" in summary
