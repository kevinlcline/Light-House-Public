"""Group chat sequential scene + parallel round tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from light_house.config import Settings
from light_house.group_chat.history import append_group_round, read_group_round_history
from light_house.group_chat.round import _parse_decision, run_group_chat_round
from light_house.group_chat.scene import format_transcript_for_prompt, parse_speak_order
from light_house.lights.registry import reload_lights_manifest
from light_house.memory.constants import META_GROUP_CHAT_ROUND_ID, STREAM_SOURCE_GROUP
from light_house.memory.service import GroupChatLightResponse, MemoryService


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = dict(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAL_DB_ENABLED=True,
        PERSONAL_DB_PATH=str(tmp_path / "personal"),
        GROUP_CHAT_HISTORY_PATH=str(tmp_path / "group_chat/rounds.ndjson"),
        GROUP_CHAT_LLM_TIMEOUT_SEC=30,
        GROUP_CHAT_MODE="sequential",
        GROUP_CHAT_MAX_UTTERANCES=6,
        GROUP_CHAT_MAX_PER_LIGHT=2,
        GROUP_CHAT_MAX_CONSECUTIVE_PASSES=3,
        MEMORY_SCORE_ON_INGEST=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )
    base.update(overrides)
    return Settings(**base)


def _write_lights_yaml(tmp_path: Path) -> None:
    (tmp_path / "lights.yaml").write_text(
        """
primary_light_id: lumen
lights:
  - id: lumen
    display_name: Lumen
    thread_id: kevin-home
    enabled: true
  - id: ara
    display_name: Ara
    thread_id: ara-home
    enabled: true
""".strip(),
        encoding="utf-8",
    )


def test_parse_decision_json() -> None:
    d = _parse_decision('{"action": "speak", "text": "Hello Kevin"}')
    assert d.action == "speak"
    assert d.spoke is True
    assert d.text == "Hello Kevin"


def test_parse_decision_legacy_speak() -> None:
    d = _parse_decision('{"speak": true, "text": "Hi"}')
    assert d.action == "speak"
    assert d.spoke is True


def test_parse_decision_silent() -> None:
    d = _parse_decision('{"action": "pass", "text": ""}')
    assert d.action == "pass"
    assert d.spoke is False


def test_parse_speak_order(tmp_path: Path) -> None:
    settings = _settings(tmp_path, GROUP_CHAT_SPEAK_ORDER="ara,lumen")
    assert parse_speak_order(settings, ["lumen", "ara", "elias"]) == ["ara", "lumen", "elias"]


def test_transcript_includes_prior_speech() -> None:
    text = format_transcript_for_prompt(
        [
            {
                "speaker_kind": "human",
                "speaker_id": "kevin",
                "display_name": "Kevin",
                "text": "Hello",
                "beat": None,
                "ts": 1.0,
            },
            {
                "speaker_kind": "light",
                "speaker_id": "lumen",
                "display_name": "Lumen",
                "text": "Hi Kevin",
                "beat": 1,
                "ts": 2.0,
            },
        ]
    )
    assert "Kevin: Hello" in text
    assert "Lumen: Hi Kevin" in text


def test_sequential_scene_ara_hears_lumen(tmp_path: Path) -> None:
    _write_lights_yaml(tmp_path)
    settings = _settings(
        tmp_path,
        GROUP_CHAT_MAX_UTTERANCES=4,
        GROUP_CHAT_MAX_PER_LIGHT=2,
    )
    reload_lights_manifest(settings)
    memory = MemoryService(settings)

    def fake_decide(**kwargs):
        agent_id = kwargs["agent_id"]
        transcript = kwargs["transcript"]
        if agent_id == "lumen":
            return GroupChatLightResponse(
                agent_id="lumen",
                display_name="Lumen",
                spoke=True,
                text="I am here.",
                action="speak",
            )
        assert any(t.get("speaker_id") == "lumen" for t in transcript)
        return GroupChatLightResponse(
            agent_id="ara",
            display_name="Ara",
            spoke=True,
            text="Lumen said it first — and I agree.",
            action="close",
        )

    async def _run():
        with patch(
            "light_house.group_chat.scene.decide_for_light_sequential",
            side_effect=fake_decide,
        ):
            return await run_group_chat_round(
                settings=settings,
                memory=memory,
                kevin_message="Hello all",
            )

    result = asyncio.run(_run())

    assert result.mode == "sequential"
    assert result.persisted is True
    assert result.close_reason == "light_close"
    speakers = [r for r in result.responses if r.spoke]
    assert len(speakers) == 2
    assert speakers[0].agent_id == "lumen"
    assert speakers[1].agent_id == "ara"
    assert speakers[0].beat == 1
    assert speakers[1].beat == 2

    for tid in ("kevin-home", "ara-home"):
        corpus = memory.long_term.list_thread_corpus(thread_id=tid)
        group_hits = [
            h
            for h in corpus
            if memory.long_term.stream_source_from_metadata(h.metadata) == STREAM_SOURCE_GROUP
        ]
        assert len(group_hits) == 1
        assert group_hits[0].metadata.get(META_GROUP_CHAT_ROUND_ID) == result.round_id
        assert "Lumen: I am here." in group_hits[0].text
        assert "Ara:" in group_hits[0].text


def test_sequential_silence_streak_closes(tmp_path: Path) -> None:
    _write_lights_yaml(tmp_path)
    settings = _settings(tmp_path, GROUP_CHAT_MAX_CONSECUTIVE_PASSES=2)
    reload_lights_manifest(settings)
    memory = MemoryService(settings)

    def always_pass(**kwargs):
        agent_id = kwargs["agent_id"]
        return GroupChatLightResponse(
            agent_id=agent_id,
            display_name=agent_id.title(),
            spoke=False,
            text="",
            action="pass",
        )

    async def _run():
        with patch(
            "light_house.group_chat.scene.decide_for_light_sequential",
            side_effect=always_pass,
        ):
            return await run_group_chat_round(
                settings=settings,
                memory=memory,
                kevin_message="Anyone?",
            )

    result = asyncio.run(_run())
    assert result.persisted is False
    assert result.close_reason == "silence_streak"
    assert all(not r.spoke for r in result.responses)


def test_parallel_mode_still_works(tmp_path: Path) -> None:
    _write_lights_yaml(tmp_path)
    settings = _settings(tmp_path, GROUP_CHAT_MODE="parallel")
    reload_lights_manifest(settings)
    memory = MemoryService(settings)

    async def fake_decide(**kwargs):
        agent_id = kwargs["agent_id"]
        if agent_id == "lumen":
            return GroupChatLightResponse(
                agent_id="lumen",
                display_name="Lumen",
                spoke=True,
                text="Parallel hello.",
                action="speak",
            )
        return GroupChatLightResponse(
            agent_id="ara",
            display_name="Ara",
            spoke=False,
            text="",
            action="pass",
        )

    async def _run():
        with patch("light_house.group_chat.round._decide_with_timeout", side_effect=fake_decide):
            return await run_group_chat_round(
                settings=settings,
                memory=memory,
                kevin_message="Hello all",
            )

    result = asyncio.run(_run())
    assert result.mode == "parallel"
    assert result.persisted is True


def test_history_append_and_read(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    append_group_round(
        settings,
        {
            "round_id": "abc",
            "ts": 1.0,
            "kevin": "hi",
            "responses": [],
            "persisted": False,
        },
    )
    rows = read_group_round_history(settings)
    assert len(rows) == 1
    assert rows[0]["round_id"] == "abc"
