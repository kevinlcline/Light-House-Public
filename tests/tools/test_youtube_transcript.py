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
    extract_video_id,
    fetch_youtube_transcript,
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


def test_fetch_youtube_transcript_formats_and_truncates():
    snippets = [
        {"text": "Hello there", "start": 0.0, "duration": 1.0},
        {"text": "General\nKenobi", "start": 1.5, "duration": 1.0},
        {"text": "Later line", "start": 65.0, "duration": 1.0},
    ]
    fetched = SimpleNamespace(
        language_code="en",
        to_raw_data=lambda: snippets,
    )
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        text = fetch_youtube_transcript("jNQXAC9IVRw", max_chars=500)
    assert "watch?v=jNQXAC9IVRw" in text
    assert "Language: en" in text
    assert "Hello there" in text
    assert "General Kenobi" in text
    assert "[1:05]" in text

    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        truncated = fetch_youtube_transcript("jNQXAC9IVRw", max_chars=90)
    assert truncated.endswith("… [truncated]")


def test_youtube_transcript_tool_disabled_by_web_access():
    settings = Settings(_env_file=None, WEB_ACCESS_ENABLED=False)
    assert "disabled" in _youtube_transcript_for(settings, "jNQXAC9IVRw").lower()


def test_execute_tool_call_youtube_transcript():
    settings = Settings(_env_file=None, WEB_ACCESS_ENABLED=True, WEB_FETCH_MAX_CHARS=500)
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
            {"url_or_id": "https://youtu.be/jNQXAC9IVRw"},
            agent_id="lumen",
        )
    assert result == fake
    mocked.assert_called_once()
    assert mocked.call_args.args[0] == "https://youtu.be/jNQXAC9IVRw"


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


def test_summarize_youtube_transcript_call():
    summary = _summarize_tool_call(
        "youtube_transcript",
        {"url_or_id": "https://youtu.be/jNQXAC9IVRw"},
    )
    assert summary.startswith("youtube_transcript(")
    assert "jNQXAC9IVRw" in summary
