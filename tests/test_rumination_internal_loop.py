"""Phase 6 item 1: multi-step internal rumination loop."""

from __future__ import annotations

from light_house.agent.rumination_internal import (
    body_requests_halt,
    effective_internal_steps_cap,
    route_after_persist,
    rumination_internal_state_defaults,
    should_continue_internal_loop,
)
from light_house.config import Settings


def _settings(**overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": "/tmp/notes",
        "MEMORY_STORE_PATH": "/tmp/memory",
        "THREADS_DATA_PATH": "/tmp/threads",
        "LIGHT_HOUSE_ENV": "production",
        "RUMINATION_INTERNAL_LOOP_ENABLED": True,
        "RUMINATION_MAX_INTERNAL_STEPS": 3,
        "RUMINATION_MAX_INTERNAL_CHARS": 5000,
    }
    base.update(overrides)
    return Settings(**base)


def _state(**overrides: object) -> dict:
    base = {
        "thread_id": "kevin-home",
        "agent_id": "lumen",
        "generated_text": "Some reflection text.",
        "internal_step": 1,
        "internal_steps_cap": None,
        "internal_halt": False,
        "internal_chars_used": 100,
    }
    base.update(overrides)
    return base


def test_defaults_disabled_in_production() -> None:
    settings = _settings(RUMINATION_INTERNAL_LOOP_ENABLED=False)
    assert should_continue_internal_loop(settings, _state()) is False


def test_continue_when_under_cap() -> None:
    settings = _settings()
    assert should_continue_internal_loop(settings, _state()) is True
    assert route_after_persist(settings, _state()) == "begin_next_step"


def test_halt_at_max_steps() -> None:
    settings = _settings(RUMINATION_MAX_INTERNAL_STEPS=2)
    state = _state(internal_step=2)
    assert should_continue_internal_loop(settings, state) is False
    assert route_after_persist(settings, state) == "end"


def test_halt_marker_stops_loop() -> None:
    settings = _settings()
    state = _state(
        generated_text="I am done.\n[rest now]",
        internal_halt=True,
    )
    assert body_requests_halt("Closing thought [rest now]")
    assert should_continue_internal_loop(settings, state) is False


def test_halt_on_empty_body() -> None:
    settings = _settings()
    assert should_continue_internal_loop(settings, _state(generated_text="")) is False


def test_chars_cap_stops_loop() -> None:
    settings = _settings(RUMINATION_MAX_INTERNAL_CHARS=500)
    state = _state(internal_chars_used=500)
    assert should_continue_internal_loop(settings, state) is False


def test_per_wake_steps_cap_override() -> None:
    settings = _settings(RUMINATION_MAX_INTERNAL_STEPS=5)
    state = _state(internal_steps_cap=2, internal_step=2)
    assert effective_internal_steps_cap(settings, state) == 2
    assert should_continue_internal_loop(settings, state) is False


def test_internal_state_defaults() -> None:
    settings = _settings()
    defaults = rumination_internal_state_defaults(settings)
    assert defaults["internal_step"] == 1
    assert defaults["internal_halt"] is False
