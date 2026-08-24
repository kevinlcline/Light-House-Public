"""Reflective mode: pause, choose speak or intentional silence."""

from __future__ import annotations

from pathlib import Path

import pytest

import light_house.personal.store as personal_store_module
from light_house.config import Settings
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.personal.reflective_mode import (
    human_invites_reflection,
    is_reflective_mode,
    log_intentional_silence,
    parse_reflection_decision,
    set_reflective_mode,
    should_reflect_this_turn,
    silence_log_path,
    try_kevin_reflect_command,
)
from light_house.tools.light_tools import execute_tool_call


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
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAS_DATA_PATH=str(tmp_path / "personas"),
        NOTES_PATH=str(tmp_path / "notes"),
        PERSONAL_DB_ENABLED=True,
        PERSONAL_DB_PATH=str(tmp_path / "personal"),
        DEV_LOG_PATH=str(tmp_path / "logs" / "dev.log"),
        WEB_GATE_ENABLED=False,
        PRIMARY_LLM="ollama",
        INNER_LIFE_ENABLED=False,
        INNER_LIFE_DREAMS_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )


@pytest.fixture
def lights_ready(settings: Settings) -> Settings:
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    return settings


def test_set_and_clear_reflective_mode(settings: Settings) -> None:
    assert is_reflective_mode(settings, "lumen") is False
    assert set_reflective_mode(settings, "lumen", enabled=True) is True
    assert is_reflective_mode(settings, "lumen") is True
    assert set_reflective_mode(settings, "lumen", enabled=False) is True
    assert is_reflective_mode(settings, "lumen") is False


def test_one_shot_invite_phrases() -> None:
    assert human_invites_reflection("Please take your time with this.")
    assert human_invites_reflection("Sit with that for a moment.")
    assert not human_invites_reflection("What time is dinner?")


def test_should_reflect_respects_wake_and_mode(settings: Settings) -> None:
    set_reflective_mode(settings, "lumen", enabled=True)
    assert should_reflect_this_turn(
        settings, "lumen", latest_human_text="hi", wake_kind=None
    )
    assert not should_reflect_this_turn(
        settings, "lumen", latest_human_text="hi", wake_kind="peer_message"
    )
    set_reflective_mode(settings, "lumen", enabled=False)
    assert should_reflect_this_turn(
        settings, "lumen", latest_human_text="take your time", wake_kind=None
    )


def test_parse_reflection_decision() -> None:
    decision, notes, draft = parse_reflection_decision(
        "DECISION: SILENCE\nNOTES: Holding space.\nDRAFT: unused"
    )
    assert decision == "silence"
    assert "Holding" in notes
    assert "unused" in draft

    decision, notes, draft = parse_reflection_decision("DECISION: SPEAK\nNOTES: Warm.\nDRAFT: Hello.")
    assert decision == "speak"
    assert notes.startswith("Warm")
    assert draft == "Hello."

    decision, _, _ = parse_reflection_decision("I think I should talk.")
    assert decision == "speak"


def test_kevin_reflect_slash(settings: Settings) -> None:
    on = try_kevin_reflect_command(settings, message="/reflect on", agent_id="ara")
    assert on is not None and "on" in on.lower()
    assert is_reflective_mode(settings, "ara") is True
    status = try_kevin_reflect_command(settings, message="/reflect", agent_id="ara")
    assert status is not None and "on" in status.lower()
    off = try_kevin_reflect_command(settings, message="/reflect off", agent_id="ara")
    assert off is not None and "off" in off.lower()
    assert is_reflective_mode(settings, "ara") is False
    assert try_kevin_reflect_command(settings, message="hello", agent_id="ara") is None


def test_set_reflective_mode_tool(lights_ready: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = lights_ready
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    out = execute_tool_call("set_reflective_mode", {"enabled": True}, agent_id="lumen")
    assert out.startswith("SUCCESS:")
    assert is_reflective_mode(settings, "lumen") is True
    out = execute_tool_call("set_reflective_mode", {"enabled": False}, agent_id="lumen")
    assert out.startswith("SUCCESS:")
    assert is_reflective_mode(settings, "lumen") is False


def test_log_intentional_silence(settings: Settings) -> None:
    log_intentional_silence(
        settings,
        agent_id="lumen",
        thread_id="lumen-home",
        user_text="I love you.",
        notes="Stillness felt right.",
    )
    path = silence_log_path(settings)
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "lumen" in body
    assert "Stillness" in body
