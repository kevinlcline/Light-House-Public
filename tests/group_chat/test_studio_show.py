"""Studio YouTube clip sitting context."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from light_house.config import Settings
from light_house.group_chat import queue_room
from light_house.group_chat.queue_room import (
    register_group_forum,
    reset_group_forum_for_tests,
    set_stage_show,
    snapshot,
    stage_show_prompt_block,
)
from light_house.group_chat.studio_show import build_stage_show, format_stage_show_block
from light_house.memory.service import MemoryService


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        GROUP_CHAT_HISTORY_PATH=str(tmp_path / "group_chat/rounds.ndjson"),
        YOUTUBE_TRANSCRIPT_CACHE_PATH=str(tmp_path / "youtube"),
        OPENROUTER_API_KEY="test-openrouter-key",
        MEMORY_SCORE_ON_INGEST=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        INNER_LIFE_ENABLED=False,
    )


def test_format_now_on_the_stage_block():
    block = format_stage_show_block(
        {
            "title": "Hoffman on attention",
            "start_label": "1:12:00",
            "end_label": "1:14:30",
            "text": "Attention is a limited resource.",
        }
    )
    assert "## Now on the stage" in block
    assert "Hoffman on attention" in block
    assert "1:12:00–1:14:30" in block
    assert "Attention is a limited resource." in block
    assert format_stage_show_block(None) == ""


def test_build_stage_show_slices_captions(tmp_path: Path):
    settings = _settings(tmp_path)
    snippets = [
        {"text": "Intro", "start": 0.0, "duration": 2.0},
        {"text": "The clip line", "start": 65.0, "duration": 2.0},
        {"text": "After", "start": 200.0, "duration": 2.0},
    ]
    fetched = MagicMock()
    fetched.language_code = "en"
    fetched.to_raw_data.return_value = snippets
    with patch(
        "light_house.tools.youtube_transcript._fetch_transcript",
        return_value=fetched,
    ):
        clip = build_stage_show(
            settings,
            url_or_id="jNQXAC9IVRw",
            start="1:00",
            end="2:00",
            title="Middle",
        )
    assert clip["video_id"] == "jNQXAC9IVRw"
    assert "The clip line" in clip["text"]
    assert "Intro" not in clip["text"]
    assert "After" not in clip["text"]
    assert clip["start_label"] == "1:00"
    assert clip["end_label"] == "2:00"


def test_sitting_keeps_and_clears_stage_show(tmp_path: Path) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    register_group_forum(settings=settings, memory=MagicMock(spec=MemoryService))
    clip = {
        "video_id": "jNQXAC9IVRw",
        "title": "Demo",
        "start_label": "0:10",
        "end_label": "0:20",
        "text": "Hello from the card.",
    }

    async def _run() -> None:
        status = await set_stage_show(clip)
        assert status["stage_show"]["title"] == "Demo"
        assert "Hello from the card." in stage_show_prompt_block()
        cleared = await set_stage_show(None)
        assert cleared["stage_show"] is None
        assert stage_show_prompt_block() == ""
        assert snapshot()["stage_show"] is None

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()
