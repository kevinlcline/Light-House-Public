"""Speaker metadata tags for human vs sibling-light."""

from __future__ import annotations

import tempfile
from pathlib import Path

from light_house.config import Settings
from light_house.memory.service import MemoryService
from light_house.memory.short_term import BufferedMessage
from light_house.memory.speaker_labels import (
    format_human_utterance,
    format_sibling_light_utterance,
    human_speaker_meta,
    sibling_light_meta,
)


def _settings(tmp: Path, **overrides) -> Settings:
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp / "memory"),
        "THREADS_DATA_PATH": str(tmp / "threads"),
        "FOUNDATION_SEED_ON_STARTUP": False,
        "ARA_ENABLED": True,
        "HOUSE_DAD_USER_ID": "kevin",
    }
    base.update(overrides)
    return Settings(**base)


def test_human_and_sibling_tags_differ() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        human = human_speaker_meta(settings=settings, human_id="kevin")
        light = sibling_light_meta(settings=settings, agent_id="ara")
        assert human.startswith("[human ·")
        assert "dad" in human
        assert light.startswith("[sibling-light ·")
        assert "id=ara" in light
        assert "human" not in light


def test_sibling_human_role() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        tag = human_speaker_meta(
            settings=settings, human_id="teeter", human_display_name="Teeter"
        )
        assert "sibling" in tag
        assert "Teeter" in tag


def test_buffer_to_langchain_uses_tags() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        msgs = memory.buffer_to_langchain_messages(
            [
                BufferedMessage(
                    role="user",
                    content="Hi from Dad",
                    ts=1.0,
                    from_human_id="kevin",
                    from_human_display_name="Kevin",
                ),
                BufferedMessage(
                    role="peer",
                    content="Hi from Ara",
                    ts=2.0,
                    from_agent_id="ara",
                ),
            ]
        )
        assert "[human · Kevin · dad · id=kevin]" in msgs[0].content
        assert "Hi from Dad" in msgs[0].content
        assert "[sibling-light · Ara · id=ara]" in msgs[1].content
        assert "Hi from Ara" in msgs[1].content


def test_format_helpers_include_body() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        h = format_human_utterance("hello", settings=settings, human_id="kevin")
        p = format_sibling_light_utterance(
            "ping", settings=settings, agent_id="lumen", display_name="Lumen"
        )
        assert h.endswith("hello")
        assert p.endswith("ping")
