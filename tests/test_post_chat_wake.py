"""Post-chat → bounded rumination wake for the same agent."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from light_house.agent import post_chat_wake
from light_house.agent.post_chat_wake import (
    WAKE_KIND_POST_CHAT,
    build_post_chat_wake_state,
    register_post_chat_wake,
    schedule_post_chat_rumination,
    wake_agent_after_chat,
)
from light_house.config import Settings
from light_house.memory.service import MemoryService


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "_env_file": None,
        "NOTES_PATH": str(notes_dir),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "MEMORY_SCORE_ON_INGEST": False,
        "INNER_LIFE_ENABLED": True,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "POST_CHAT_RUMINATION_ENABLED": True,
        "POST_CHAT_RUMINATION_DELAY_SECONDS": 0,
    }
    base.update(overrides)
    return Settings(**base)


def test_build_post_chat_wake_state_includes_wake_kind_and_cap(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    memory = MemoryService(settings)
    memory.persist_exchange(
        thread_id="kevin-home",
        user_text="hello",
        assistant_text="hi there",
    )
    state = build_post_chat_wake_state(
        thread_id="kevin-home",
        agent_id="lumen",
        settings=settings,
        memory=memory,
    )
    assert state["wake_kind"] == WAKE_KIND_POST_CHAT
    assert state["wake_path"] is None
    assert state["tool_rounds_cap"] == settings.post_chat_rumination_max_tool_rounds
    assert len(state["messages"]) >= 2


def test_wake_agent_after_chat_invokes_one_agent(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    memory = MemoryService(settings)
    graph = MagicMock()
    graph.invoke = MagicMock()

    asyncio.run(
        wake_agent_after_chat(
            agent_id="lumen",
            thread_id="kevin-home",
            settings=settings,
            memory=memory,
            graph=graph,
        )
    )

    assert graph.invoke.call_count == 1
    state = graph.invoke.call_args.args[0]
    assert state["agent_id"] == "lumen"
    assert state["thread_id"] == "kevin-home"
    assert state["wake_kind"] == WAKE_KIND_POST_CHAT


def test_wake_agent_after_chat_skipped_when_disabled(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, POST_CHAT_RUMINATION_ENABLED=False)
    memory = MemoryService(settings)
    graph = MagicMock()

    asyncio.run(
        wake_agent_after_chat(
            agent_id="lumen",
            thread_id="kevin-home",
            settings=settings,
            memory=memory,
            graph=graph,
        )
    )

    graph.invoke.assert_not_called()


def test_wake_agent_after_chat_debounce(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    memory = MemoryService(settings)
    graph = MagicMock()
    graph.invoke = MagicMock()

    post_chat_wake._recent_wakes.clear()
    post_chat_wake._recent_wakes[("lumen", "kevin-home")] = time.time()

    asyncio.run(
        wake_agent_after_chat(
            agent_id="lumen",
            thread_id="kevin-home",
            settings=settings,
            memory=memory,
            graph=graph,
        )
    )

    graph.invoke.assert_not_called()


def test_schedule_post_chat_rumination_uses_event_bus(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    memory = MagicMock()
    graph = MagicMock()
    loop = asyncio.new_event_loop()
    try:
        register_post_chat_wake(
            graph=graph,
            memory=memory,
            settings=settings,
            app_loop=loop,
        )
        from light_house.events import EventType, register_event_bus

        register_event_bus(
            settings=settings,
            memory=memory,
            rumination_graph=graph,
            app_loop=loop,
        )
        with patch("light_house.events.publish") as publish_mock:
            schedule_post_chat_rumination(agent_id="ara", thread_id="ara-home")
            publish_mock.assert_called_once()
            event = publish_mock.call_args.args[0]
            assert event.event_type == EventType.CHAT_RECEIVED
            assert event.agent_id == "ara"
            assert event.thread_id == "ara-home"
    finally:
        loop.close()
        post_chat_wake._graph = None
        post_chat_wake._memory = None
        post_chat_wake._settings = None
        post_chat_wake._app_loop = None
