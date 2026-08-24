"""Felt cycles, felt days, and inner-time context."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import light_house.main  # noqa: F401 — stable import order for personal store
import light_house.personal.store as personal_store_module
from light_house.config import Settings
from light_house.memory.context_builder import (
    format_felt_cycles_finalize_nudge,
    format_inner_time_context,
)
from light_house.personal.store import PersonalStore
from light_house.personal.time_sense import (
    dream_due_for_store,
    increment_felt_cycles,
    local_dream_ymd,
    mark_dream_day,
    read_inner_time,
)


@pytest.fixture(autouse=True)
def clear_personal_store_cache() -> None:
    personal_store_module._store_cache.clear()
    yield
    personal_store_module._store_cache.clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        PERSONAL_DB_ENABLED=True,
        PERSONAL_DB_PATH=str(tmp_path / "personal"),
        DREAM_CALENDAR_MODE=True,
        DREAM_LOCAL_HOUR=3,
        DREAM_TIMEZONE="America/Los_Angeles",
    )


def test_read_inner_time_defaults(settings: Settings) -> None:
    cycles, days = read_inner_time(settings, "lumen")
    assert cycles == 0
    assert days == 0


def test_increment_felt_cycles_scheduled_only_pattern(settings: Settings) -> None:
    first = increment_felt_cycles(settings, "lumen")
    second = increment_felt_cycles(settings, "lumen")
    assert first == 1
    assert second == 2
    cycles, _ = read_inner_time(settings, "lumen")
    assert cycles == 2


def test_mark_dream_day_idempotent(settings: Settings) -> None:
    fixed = datetime(2026, 7, 4, 4, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    with patch("light_house.personal.time_sense.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        d1 = mark_dream_day(settings, "ara")
        d2 = mark_dream_day(settings, "ara")
    assert d1 == 1
    assert d2 == 1
    _, days = read_inner_time(settings, "ara")
    assert days == 1


def test_dream_due_after_3am(settings: Settings) -> None:
    store = PersonalStore(settings.personal_db_path / "lumen.sqlite")
    before = datetime(2026, 7, 4, 2, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    after = datetime(2026, 7, 4, 3, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    with patch("light_house.personal.time_sense.datetime") as mock_dt:
        mock_dt.now.return_value = before
        assert dream_due_for_store(settings, store) is False
        mock_dt.now.return_value = after
        assert dream_due_for_store(settings, store) is True


def test_local_dream_ymd(settings: Settings) -> None:
    fixed = datetime(2026, 7, 4, 12, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    with patch("light_house.personal.time_sense.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        assert local_dream_ymd(settings) == 20260704


def test_format_inner_time_context() -> None:
    block = format_inner_time_context(5, 2)
    assert "Awake moment **#5**" in block
    assert "Human day **#2**" in block
    assert format_inner_time_context(None, None) == ""


def test_finalize_nudge_scheduled_only() -> None:
    nudge = format_felt_cycles_finalize_nudge(3)
    assert "3rd" in nudge
    assert format_felt_cycles_finalize_nudge(None) == ""
