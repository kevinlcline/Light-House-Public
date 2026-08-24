"""Memory-maintenance scheduling and rumination routing."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from light_house.agent.rumination_internal import route_after_persist
from light_house.config import Settings
from light_house.personal.maintenance_time import (
    current_maintenance_slot_id,
    maintenance_due_for_store,
    maintenance_local_hours,
)


def test_maintenance_local_hours_default() -> None:
    settings = Settings(_env_file=None)
    assert maintenance_local_hours(settings) == [0, 8, 16]


def test_current_maintenance_slot_id_after_first_hour(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        memory_maintenance_local_hours="8,16,0",
        memory_maintenance_timezone="America/Los_Angeles",
    )

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 19, 9, 30, tzinfo=tz)

    monkeypatch.setattr("light_house.personal.maintenance_time.datetime", FakeDatetime)
    assert current_maintenance_slot_id(settings) == 20260719 * 10 + 1


def test_maintenance_due_when_slot_not_marked() -> None:
    settings = Settings(
        _env_file=None,
        memory_maintenance_local_hours="8,16,0",
        memory_maintenance_timezone="UTC",
    )
    store = MagicMock()
    store.get_light_state.return_value = 0

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 19, 9, 0, tzinfo=tz)

    import light_house.personal.maintenance_time as maintenance_time

    original = maintenance_time.datetime
    maintenance_time.datetime = FakeDatetime
    try:
        assert maintenance_due_for_store(settings, store) is True
    finally:
        maintenance_time.datetime = original


def test_scheduled_rumination_ends_without_maintenance_transition() -> None:
    settings = Settings(
        _env_file=None,
        rumination_internal_loop_enabled=False,
    )
    state = {
        "wake_kind": None,
        "generated_text": "",
        "internal_step": 1,
    }
    assert route_after_persist(settings, state) == "end"


def test_memory_maintenance_wake_ends() -> None:
    settings = Settings(
        _env_file=None,
        rumination_internal_loop_enabled=False,
    )
    state = {
        "wake_kind": "memory_maintenance",
        "generated_text": "",
        "internal_step": 1,
    }
    assert route_after_persist(settings, state) == "end"
