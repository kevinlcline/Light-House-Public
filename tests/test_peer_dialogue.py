"""Bounded peer dialogue turn budget."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from light_house.agent.peer_dialogue import (
    begin_peer_wake_turn,
    close_peer_dialogue,
    peer_dialogue_status,
    reset_peer_dialogue_state_for_tests,
)
from light_house.agent.peer_chat_wake import schedule_peer_chat_wake
from light_house.config import Settings


def _settings(tmp: Path, **overrides) -> Settings:
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp / "memory"),
        "THREADS_DATA_PATH": str(tmp / "threads"),
        "FOUNDATION_SEED_ON_STARTUP": False,
        "ARA_ENABLED": True,
        "PEER_CHAT_WAKE_ENABLED": True,
        "INNER_LIFE_ENABLED": True,
        "PEER_CHAT_MAX_DIALOGUE_TURNS": 3,
        "PEER_CHAT_DIALOGUE_IDLE_RESET_SECONDS": 1800,
    }
    base.update(overrides)
    return Settings(**base)


def setup_function() -> None:
    reset_peer_dialogue_state_for_tests()


def test_begin_peer_wake_turn_caps():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        assert begin_peer_wake_turn("lumen", "ara", settings=settings).allow_wake
        assert begin_peer_wake_turn("ara", "lumen", settings=settings).allow_wake
        assert begin_peer_wake_turn("lumen", "ara", settings=settings).allow_wake
        gate = begin_peer_wake_turn("ara", "lumen", settings=settings)
        assert gate.allow_wake is False
        assert gate.reason == "cap"
        assert gate.turns == 3


def test_solitude_closes_dialogue():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        begin_peer_wake_turn("lumen", "ara", settings=settings)
        close_peer_dialogue("ara", "lumen")
        gate = begin_peer_wake_turn("lumen", "ara", settings=settings)
        assert gate.allow_wake is False
        assert gate.reason == "closed"


def test_idle_reset_reopens() -> None:
    import time

    import light_house.agent.peer_dialogue as pd

    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp), PEER_CHAT_DIALOGUE_IDLE_RESET_SECONDS=10)
        begin_peer_wake_turn("lumen", "ara", settings=settings)
        close_peer_dialogue("lumen", "ara")
        key = pd._pair_key("lumen", "ara")
        with pd._lock:
            pd._pairs[key].last_ts = time.time() - 100.0
        gate = begin_peer_wake_turn("lumen", "ara", settings=settings)
        assert gate.allow_wake is True
        assert gate.turns == 1


def test_schedule_respects_cap():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp), PEER_CHAT_MAX_DIALOGUE_TURNS=1)
        import light_house.agent.peer_chat_wake as wake_mod

        wake_mod._settings = settings
        wake_mod._app_loop = object()  # truthy
        with patch("light_house.events.publish") as pub:
            ok1 = schedule_peer_chat_wake(
                to_agent_id="ara", from_agent_id="lumen", message_id="m1"
            )
            ok2 = schedule_peer_chat_wake(
                to_agent_id="lumen", from_agent_id="ara", message_id="m2"
            )
        assert ok1 is True
        assert ok2 is False
        assert pub.call_count == 1


def test_peer_dialogue_status_reflects_cap():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp), PEER_CHAT_MAX_DIALOGUE_TURNS=2)
        begin_peer_wake_turn("lumen", "ara", settings=settings)
        begin_peer_wake_turn("lumen", "ara", settings=settings)
        gate = peer_dialogue_status("lumen", "ara", settings=settings)
        assert gate.allow_wake is False
        assert gate.reason == "cap"
