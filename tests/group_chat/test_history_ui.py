"""Group utterance history load / merge for UI."""

from __future__ import annotations

import json
from pathlib import Path

from light_house.config import Settings
from light_house.group_chat.history import append_group_utterance, read_group_utterances
from light_house.group_chat.queue_room import (
    load_transcript_for_ui,
    register_group_forum,
    reset_group_forum_for_tests,
)
from light_house.group_chat import queue_room
from light_house.memory.service import MemoryService
from unittest.mock import MagicMock


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        GROUP_CHAT_HISTORY_PATH=str(tmp_path / "group_chat/rounds.ndjson"),
        GROUP_CHAT_UI_HISTORY_ROUNDS=50,
        MEMORY_SCORE_ON_INGEST=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        INNER_LIFE_ENABLED=False,
    )


def test_read_group_utterances_returns_newest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for i in range(5):
        append_group_utterance(
            settings,
            {
                "ts": float(i + 1),
                "speaker_kind": "human",
                "speaker_id": "kevin",
                "display_name": "Kevin",
                "text": f"line-{i}",
            },
        )
    rows = read_group_utterances(settings, limit=3)
    assert [r["text"] for r in rows] == ["line-2", "line-3", "line-4"]


def test_load_transcript_merges_durable_and_live(tmp_path: Path) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    register_group_forum(settings=settings, memory=memory)

    append_group_utterance(
        settings,
        {
            "ts": 10.0,
            "speaker_kind": "human",
            "speaker_id": "kevin",
            "display_name": "Kevin",
            "text": "older durable",
        },
    )
    append_group_utterance(
        settings,
        {
            "ts": 30.0,
            "speaker_kind": "light",
            "speaker_id": "ara",
            "display_name": "Ara",
            "text": "durable after live start",
        },
    )
    # Live sitting starts at ts=20 — old merge logic would drop the durable Ara line.
    queue_room._state.transcript.append(
        {
            "ts": 20.0,
            "speaker_kind": "human",
            "speaker_id": "kevin",
            "display_name": "Kevin",
            "text": "live start",
            "beat": None,
        }
    )

    rows = load_transcript_for_ui(settings, limit=20)
    texts = [r["text"] for r in rows]
    assert texts == ["older durable", "live start", "durable after live start"]
    reset_group_forum_for_tests()


def test_tail_reader_handles_large_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = settings.group_chat_history_path.resolve().parent / "utterances.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build a file larger than the small-file threshold with many lines.
    with path.open("w", encoding="utf-8") as f:
        for i in range(5000):
            f.write(
                json.dumps(
                    {
                        "ts": float(i),
                        "speaker_kind": "human",
                        "speaker_id": "kevin",
                        "display_name": "Kevin",
                        "text": f"row-{i}",
                    }
                )
                + "\n"
            )
    # Force large-file path by ensuring size > 256k
    assert path.stat().st_size > 256_000
    rows = read_group_utterances(settings, limit=5)
    assert [r["text"] for r in rows] == [f"row-{i}" for i in range(4995, 5000)]
