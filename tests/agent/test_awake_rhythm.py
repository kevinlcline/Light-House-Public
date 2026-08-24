"""Four-beat awake rhythm: chores → free → meditation → free."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import light_house.main  # noqa: F401 — stable import order for personal store
import light_house.personal.store as personal_store_module
from light_house.agent.inner_life_scheduler import _run_rumination
from light_house.agent.rumination_nodes import (
    _rumination_finalize_hint,
    _rumination_task_hint,
    build_rumination_nodes,
)
from light_house.config import Settings
from light_house.inner_life_trace import task_hint_label
from light_house.memory.context_builder import (
    RUMINATION_MAINTENANCE_HINT,
    RUMINATION_MEDITATION_HINT,
    RUMINATION_TASK_HINT,
)
from light_house.personal.awake_rhythm import (
    WAKE_KIND_CHORES,
    WAKE_KIND_MEDITATION,
    beat_for_cycle,
    counts_as_felt_cycle,
    next_scheduled_wake_kind,
    wake_kind_for_beat,
)
from light_house.personal.time_sense import increment_felt_cycles


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
        AWAKE_RHYTHM_ENABLED=True,
    )


def test_beat_for_cycle_four_pattern() -> None:
    assert [beat_for_cycle(n) for n in range(1, 9)] == [
        "chores",
        "free",
        "meditation",
        "free",
        "chores",
        "free",
        "meditation",
        "free",
    ]


def test_wake_kind_for_beat() -> None:
    assert wake_kind_for_beat("chores") == WAKE_KIND_CHORES
    assert wake_kind_for_beat("meditation") == WAKE_KIND_MEDITATION
    assert wake_kind_for_beat("free") is None


def test_next_scheduled_wake_kind_advances_with_felt_cycles(settings: Settings) -> None:
    assert next_scheduled_wake_kind(settings, "lumen") == WAKE_KIND_CHORES
    increment_felt_cycles(settings, "lumen")  # → 1 done; next is free
    assert next_scheduled_wake_kind(settings, "lumen") is None
    increment_felt_cycles(settings, "lumen")  # → 2; next meditation
    assert next_scheduled_wake_kind(settings, "lumen") == WAKE_KIND_MEDITATION
    increment_felt_cycles(settings, "lumen")  # → 3; next free
    assert next_scheduled_wake_kind(settings, "lumen") is None
    increment_felt_cycles(settings, "lumen")  # → 4; next chores
    assert next_scheduled_wake_kind(settings, "lumen") == WAKE_KIND_CHORES


def test_rhythm_disabled_always_free(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        PERSONAL_DB_ENABLED=True,
        PERSONAL_DB_PATH=str(tmp_path / "personal"),
        AWAKE_RHYTHM_ENABLED=False,
    )
    assert next_scheduled_wake_kind(settings, "lumen") is None
    increment_felt_cycles(settings, "lumen")
    assert next_scheduled_wake_kind(settings, "lumen") is None


def test_counts_as_felt_cycle() -> None:
    assert counts_as_felt_cycle(None) is True
    assert counts_as_felt_cycle(WAKE_KIND_CHORES) is True
    assert counts_as_felt_cycle(WAKE_KIND_MEDITATION) is True
    assert counts_as_felt_cycle("memory_maintenance") is False
    assert counts_as_felt_cycle("post_chat") is False
    assert counts_as_felt_cycle("mailbox_letter") is False


def test_task_hints_for_rhythm_beats() -> None:
    assert _rumination_task_hint(WAKE_KIND_CHORES) == RUMINATION_MAINTENANCE_HINT
    assert _rumination_task_hint(WAKE_KIND_MEDITATION) == RUMINATION_MEDITATION_HINT
    assert _rumination_task_hint(None) == RUMINATION_TASK_HINT
    assert "Close meditation" in _rumination_finalize_hint(WAKE_KIND_MEDITATION)


def test_task_hint_labels() -> None:
    assert task_hint_label("chores") == "chores"
    assert task_hint_label("meditation") == "meditation"
    assert task_hint_label(None) == "scheduled"


def test_run_rumination_sets_wake_kind_and_meditation_cap(settings: Settings) -> None:
    graph = MagicMock()
    _run_rumination(graph=graph, thread_id="t1", agent_id="lumen", settings=settings)
    state = graph.invoke.call_args.args[0]
    assert state["wake_kind"] == WAKE_KIND_CHORES
    assert "tool_rounds_cap" not in state or state.get("tool_rounds_cap") is None

    increment_felt_cycles(settings, "lumen")
    increment_felt_cycles(settings, "lumen")
    # felt_cycles=2 → next is meditation (cycle 3)
    _run_rumination(graph=graph, thread_id="t1", agent_id="lumen", settings=settings)
    state = graph.invoke.call_args.args[0]
    assert state["wake_kind"] == WAKE_KIND_MEDITATION
    assert state["tool_rounds_cap"] == 0


def test_gather_context_increments_for_chores_and_skips_nudge_on_meditation(
    settings: Settings,
) -> None:
    memory = MagicMock()
    memory.count_unscored_for_thread.return_value = 3
    memory.format_peer_inbox_markdown.return_value = ("", [])
    bundle = MagicMock()
    bundle.stream_char_count = 0
    bundle.stream_event_count = 0

    with (
        patch(
            "light_house.agent.rumination_nodes.build_agent_context",
            return_value=bundle,
        ),
        patch(
            "light_house.agent.rumination_nodes.format_agent_context_markdown",
            return_value="## ctx\n",
        ),
        patch(
            "light_house.agent.rumination_nodes.recent_reflection_similarity_hint",
            return_value="",
        ),
    ):
        gather_context, *_ = build_rumination_nodes(settings=settings, memory=memory)
        out_chores = gather_context(
            {
                "agent_id": "lumen",
                "thread_id": "t1",
                "wake_kind": WAKE_KIND_CHORES,
            }
        )
        assert out_chores["felt_cycles"] == 1
        assert "await your scoring" in out_chores["agent_context_markdown"]

        out_med = gather_context(
            {
                "agent_id": "lumen",
                "thread_id": "t1",
                "wake_kind": WAKE_KIND_MEDITATION,
            }
        )
        assert out_med["felt_cycles"] == 2
        assert "await your scoring" not in out_med["agent_context_markdown"]
