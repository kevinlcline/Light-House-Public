"""Event bus: publish, log, dispatch."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from light_house.config import Settings
from light_house.events.bus import (
    publish,
    read_event_log_tail,
    register_event_bus,
    start_event_bus,
)
from light_house.events.types import EventType, LightHouseEvent


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "LIGHT_HOUSE_ENV": "production",
    }
    base.update(overrides)
    return Settings(**base)


def test_publish_appends_log_when_enabled(tmp_path: Path) -> None:
    log_path = tmp_path / "events" / "event.log"
    settings = _test_settings(
        tmp_path,
        EVENT_BUS_ENABLED=True,
        EVENT_BUS_LOG_PATH=str(log_path),
    )
    loop = asyncio.new_event_loop()
    try:
        register_event_bus(
            settings=settings,
            memory=MagicMock(),
            rumination_graph=MagicMock(),
            app_loop=loop,
        )
        event = LightHouseEvent(
            event_type=EventType.CHAT_RECEIVED,
            agent_id="lumen",
            thread_id="kevin-home",
        )
        publish(event)
        assert log_path.is_file()
        row = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert row["event_type"] == "chat_received"
        assert row["agent_id"] == "lumen"
        assert row["thread_id"] == "kevin-home"
    finally:
        loop.close()


def test_read_event_log_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "event.log"
    log_path.write_text(
        "\n".join(
            json.dumps({"line": i})
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    content, count, truncated = read_event_log_tail(log_path, max_lines=2)
    assert count == 2
    assert truncated is True
    assert '"line": 3' in content
    assert '"line": 4' in content


def test_dispatcher_handles_chat_received(tmp_path: Path) -> None:
    async def run() -> None:
        settings = _test_settings(tmp_path, EVENT_BUS_ENABLED=True)
        loop = asyncio.get_running_loop()
        register_event_bus(
            settings=settings,
            memory=MagicMock(),
            rumination_graph=MagicMock(),
            app_loop=loop,
        )
        cancel = asyncio.Event()
        dispatch_task = asyncio.create_task(start_event_bus(cancel_event=cancel))

        with patch(
            "light_house.events.bus._handle_chat_received",
            new_callable=AsyncMock,
        ) as handle_mock:
            publish(
                LightHouseEvent(
                    event_type=EventType.CHAT_RECEIVED,
                    agent_id="lumen",
                    thread_id="kevin-home",
                )
            )
            for _ in range(50):
                if handle_mock.await_count:
                    break
                await asyncio.sleep(0.05)
            handle_mock.assert_awaited_once()

        cancel.set()
        await asyncio.wait_for(dispatch_task, timeout=2.0)

    asyncio.run(run())


def test_publish_passthrough_when_disabled(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, EVENT_BUS_ENABLED=False)
    loop = MagicMock()
    loop.is_running.return_value = True
    register_event_bus(
        settings=settings,
        memory=MagicMock(),
        rumination_graph=MagicMock(),
        app_loop=loop,
    )
    with patch("light_house.events.bus.asyncio.run_coroutine_threadsafe") as run_safe:
        publish(
            LightHouseEvent(
                event_type=EventType.PEER_MESSAGE,
                agent_id="ara",
                payload={"from_agent_id": "lumen", "message_id": "m1"},
            )
        )
        run_safe.assert_called_once()
