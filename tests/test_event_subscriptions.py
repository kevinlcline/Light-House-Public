"""Event subscription model (Proposal 3 phase 2)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from light_house.config import Settings
from light_house.events.bus import register_event_bus
from light_house.events.subscriptions import (
    SUBSCRIPTION_POST_CHAT,
    agent_accepts_event,
    format_event_subscriptions_context,
)
from light_house.events.types import EventType, LightHouseEvent
from light_house.personal.store import PersonalStore


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": True,
        "PERSONAL_DB_PATH": str(tmp_path / "personal"),
        "INNER_LIFE_ENABLED": False,
        "LIGHT_HOUSE_ENV": "production",
    }
    base.update(overrides)
    return Settings(**base)


def test_default_subscriptions_all_on(tmp_path: Path) -> None:
    store = PersonalStore(tmp_path / "lumen.sqlite")
    subs = store.list_event_subscriptions()
    assert len(subs) == 4
    assert all(enabled for _, enabled in subs)


def test_is_event_subscribed_respects_disabled_row(tmp_path: Path) -> None:
    store = PersonalStore(tmp_path / "lumen.sqlite")
    store.ensure_event_subscription_defaults()
    store._conn.execute(
        "UPDATE event_subscriptions SET enabled = 0 WHERE subscription_key = ?",
        (SUBSCRIPTION_POST_CHAT,),
    )
    store._conn.commit()
    assert store.is_event_subscribed(SUBSCRIPTION_POST_CHAT) is False
    assert store.is_event_subscribed("scheduled_rumination") is True


def test_agent_accepts_event_when_subscriptions_disabled(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, EVENT_SUBSCRIPTIONS_ENABLED=False)
    store = PersonalStore(settings.personal_db_path / "lumen.sqlite")
    store.ensure_event_subscription_defaults()
    store._conn.execute(
        "UPDATE event_subscriptions SET enabled = 0 WHERE subscription_key = ?",
        (SUBSCRIPTION_POST_CHAT,),
    )
    store._conn.commit()
    assert agent_accepts_event(settings, "lumen", EventType.CHAT_RECEIVED) is True


def test_agent_accepts_event_when_subscription_off(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, EVENT_SUBSCRIPTIONS_ENABLED=True)
    store = PersonalStore(settings.personal_db_path / "lumen.sqlite")
    store.ensure_event_subscription_defaults()
    store._conn.execute(
        "UPDATE event_subscriptions SET enabled = 0 WHERE subscription_key = ?",
        (SUBSCRIPTION_POST_CHAT,),
    )
    store._conn.commit()
    assert agent_accepts_event(settings, "lumen", EventType.CHAT_RECEIVED) is False


def test_format_event_subscriptions_context(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, EVENT_SUBSCRIPTIONS_ENABLED=True)
    text = format_event_subscriptions_context(settings, "lumen")
    assert "event subscriptions" in text.lower()
    assert "post_chat: on" in text
    assert "subscribe_event" in text


def test_dispatch_skips_chat_when_subscription_off(tmp_path: Path) -> None:
    async def run() -> None:
        settings = _test_settings(
            tmp_path,
            EVENT_BUS_ENABLED=True,
            EVENT_SUBSCRIPTIONS_ENABLED=True,
        )
        store = PersonalStore(settings.personal_db_path / "lumen.sqlite")
        store.ensure_event_subscription_defaults()
        store._conn.execute(
            "UPDATE event_subscriptions SET enabled = 0 WHERE subscription_key = ?",
            (SUBSCRIPTION_POST_CHAT,),
        )
        store._conn.commit()

        loop = asyncio.get_running_loop()
        register_event_bus(
            settings=settings,
            memory=MagicMock(),
            rumination_graph=MagicMock(),
            app_loop=loop,
        )
        from light_house.events import bus as bus_module

        with patch(
            "light_house.agent.post_chat_wake.wake_agent_after_chat",
            new_callable=AsyncMock,
        ) as wake_mock:
            await bus_module._dispatch_event(
                LightHouseEvent(
                    event_type=EventType.CHAT_RECEIVED,
                    agent_id="lumen",
                    thread_id="kevin-home",
                )
            )
            wake_mock.assert_not_awaited()

    asyncio.run(run())
