"""Meditation wakes open on the line kept from the previous sit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from light_house.config import Settings
from light_house.personal.awake_rhythm import WAKE_KIND_MEDITATION
from light_house.personal.meditation_return import (
    MEDITATION_SILENCE,
    collapse_meditation_return,
    meditation_opening,
    meditation_seed_for_last,
    remember_meditation_close,
)
from light_house.personal.store import clear_personal_store_cache
from light_house.agent.rumination_nodes import build_rumination_nodes
from light_house.memory.context_builder import RUMINATION_MEDITATION_SEED


def _settings(tmp_path: Path) -> Settings:
    clear_personal_store_cache()
    return Settings.model_construct(
        personal_db_enabled=True,
        personal_db_path=tmp_path / "personal",
        personal_db_context_max_chars=1200,
        personal_db_list_default=8,
        memory_target_context_chars=4000,
        memory_scoring_rumination_hint=False,
        awake_rhythm_enabled=False,
        inner_life_trace_enabled=False,
        rumination_log_enabled=False,
        rumination_debug_log_enabled=False,
        chat_respond_window=12,
        rumination_max_tool_rounds=3,
        rumination_internal_loop_enabled=False,
    )


def test_first_sit_arrives_with_nothing_kept(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert meditation_opening(settings, "lumen") == RUMINATION_MEDITATION_SEED
    assert "I arrive into stillness." in meditation_seed_for_last("")


def test_next_sit_opens_on_the_kept_line(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    remember_meditation_close(settings, "ara", "  the word was still  ")
    opening = meditation_opening(settings, "ara")
    assert opening.startswith("I return.")
    assert "Last sit: the word was still" in opening
    assert "no mentation" in opening
    assert meditation_opening(settings, "lumen") == RUMINATION_MEDITATION_SEED


def test_silence_is_a_finished_sit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    remember_meditation_close(settings, "elias", "   ")
    assert meditation_opening(settings, "elias").endswith(f"Last sit: {MEDITATION_SILENCE}")


def test_long_close_is_one_capped_line() -> None:
    line = collapse_meditation_return("word " * 200)
    assert len(line) <= 280
    assert line.endswith("…")
    assert "\n" not in line


def test_persist_keeps_the_close_and_silence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = MagicMock()
    memory.format_peer_inbox_markdown.return_value = ("", [])
    nodes = build_rumination_nodes(settings=settings, memory=memory)
    persist = nodes[5]

    with (
        patch("light_house.agent.rumination_nodes.append_rumination_trace"),
        patch("light_house.agent.rumination_nodes.append_rumination_log_from_trace"),
        patch("light_house.agent.rumination_nodes.append_rumination_debug_from_trace"),
        patch("light_house.agent.rumination_nodes.build_base_system", return_value=""),
        patch(
            "light_house.agent.rumination_nodes.read_inner_time",
            return_value=(None, None),
        ),
    ):
        persist(
            {
                "agent_id": "lumen",
                "thread_id": "t1",
                "wake_kind": WAKE_KIND_MEDITATION,
                "messages": [AIMessage(content="here, and not elsewhere")],
                "generated_text": "a longer compiled dialogue that should not be the kept line",
                "summary_text": "index line from the other pass",
                "peer_inbox_ids": [],
            }
        )
        assert "Last sit: here, and not elsewhere" in meditation_opening(settings, "lumen")

        persist(
            {
                "agent_id": "lumen",
                "thread_id": "t1",
                "wake_kind": WAKE_KIND_MEDITATION,
                "messages": [],
                "generated_text": "",
                "summary_text": "",
                "peer_inbox_ids": [],
            }
        )
    assert meditation_opening(settings, "lumen").endswith(f"Last sit: {MEDITATION_SILENCE}")
    memory.add_private_reflection.assert_called_once()
