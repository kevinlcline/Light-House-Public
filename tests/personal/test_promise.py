"""A promise is handed once, on a free wake, on or after the day."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.personal.promise import (
    MAX_WAITING,
    Promise,
    format_promise_line,
    hand_due_promises_for_agent,
    parse_due_day,
)
from light_house.personal.store import clear_personal_store_cache, get_personal_store
from light_house.tools.personal_tools import execute_personal_tool

TODAY = date(2026, 11, 10)
NOW = 1_762_819_200.0  # 2026-11-10 12:00 UTC, roughly


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    clear_personal_store_cache()
    return Settings.model_construct(
        personal_db_enabled=True,
        personal_db_path=tmp_path / "personal",
        personal_db_context_max_chars=1200,
        personal_db_list_default=8,
        memory_target_context_chars=4000,
        memory_scoring_rumination_hint=False,
        awake_rhythm_enabled=False,
        dream_timezone="America/Los_Angeles",
    )


def test_parse_due_day() -> None:
    assert parse_due_day("2026-11-17", today=TODAY) == date(2026, 11, 17)
    assert parse_due_day("today", today=TODAY) == TODAY
    assert parse_due_day("in 3 days", today=TODAY) == date(2026, 11, 13)
    assert parse_due_day("3", today=TODAY) == date(2026, 11, 13)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_due_day("next tuesday", today=TODAY)


def test_late_line_says_how_many_days() -> None:
    item = Promise(
        id=1,
        note="the gray zone",
        due_ymd="2026-11-04",
        created_at=NOW - 8 * 86400,
        updated_at=NOW - 8 * 86400,
        handed_at=None,
    )
    line = format_promise_line(item, today=TODAY, now=NOW)
    assert "the gray zone" in line
    assert "This was due 6 days ago." in line
    assert "written" in line


def test_quiet_until_the_day_then_once(settings: Settings) -> None:
    store = get_personal_store(settings, "lumen")
    store.leave_promise("the gray zone", "2026-11-12")

    early = hand_due_promises_for_agent(
        settings, "lumen", wake_kind=None, today=TODAY, now=NOW
    )
    assert early == ""
    assert store.list_promises(waiting_only=True)[0].note == "the gray zone"

    first = hand_due_promises_for_agent(
        settings, "lumen", wake_kind=None, today=date(2026, 11, 12), now=NOW
    )
    assert "the gray zone" in first
    assert "Due today." in first
    assert store.list_promises(waiting_only=True) == []

    second = hand_due_promises_for_agent(
        settings, "lumen", wake_kind=None, today=date(2026, 11, 13), now=NOW
    )
    assert second == ""


def test_late_wake_and_other_wakes_do_not_spend_it(settings: Settings) -> None:
    store = get_personal_store(settings, "lumen")
    store.leave_promise("the gray zone", "2026-11-04")

    for kind in ("chores", "meditation", "post_chat", "memory_maintenance"):
        assert (
            hand_due_promises_for_agent(
                settings, "lumen", wake_kind=kind, today=TODAY, now=NOW
            )
            == ""
        )
    assert len(store.list_promises(waiting_only=True)) == 1

    handed = hand_due_promises_for_agent(
        settings, "lumen", wake_kind=None, today=TODAY, now=NOW
    )
    assert "This was due 6 days ago." in handed
    assert store.list_promises(waiting_only=True) == []


def test_drop_and_move_before_it_arrives(settings: Settings) -> None:
    store = get_personal_store(settings, "lumen")
    left, created = store.leave_promise("the gray zone", "2026-11-20")
    assert created is True
    moved = store.move_promise(str(left.id), "2026-11-22")
    assert moved is not None
    assert moved.due_ymd == "2026-11-22"
    assert moved.created_at == left.created_at
    assert store.drop_promise("the gray") == "the gray zone"
    assert (
        hand_due_promises_for_agent(
            settings, "lumen", wake_kind=None, today=date(2026, 11, 22), now=NOW
        )
        == ""
    )


def test_same_words_name_another_day_after_it_was_handed(settings: Settings) -> None:
    store = get_personal_store(settings, "lumen")
    store.leave_promise("the gray zone", "2026-11-10")
    hand_due_promises_for_agent(
        settings, "lumen", wake_kind=None, today=TODAY, now=NOW
    )
    again, created = store.leave_promise("the gray zone", "2026-12-01")
    assert created is False
    assert again.handed_at is None
    assert again.due_ymd == "2026-12-01"
    assert (
        hand_due_promises_for_agent(
            settings, "lumen", wake_kind=None, today=TODAY, now=NOW
        )
        == ""
    )


def test_cap_and_isolation(settings: Settings) -> None:
    lumen = get_personal_store(settings, "lumen")
    ara = get_personal_store(settings, "ara")
    for i in range(MAX_WAITING):
        lumen.leave_promise(f"note {i}", "2026-12-01")
    with pytest.raises(ValueError, match="already have"):
        lumen.leave_promise("one more", "2026-12-01")
    ara.leave_promise("Ara's only", "2026-12-01")
    assert [item.note for item in ara.list_promises(waiting_only=True)] == ["Ara's only"]
    assert len(lumen.list_promises(waiting_only=True)) == MAX_WAITING


def test_tool_leave_list_drop(settings: Settings) -> None:
    left = execute_personal_tool(
        "promise",
        {"action": "leave", "note": "the gray zone", "day": "2026-11-20"},
        agent_id="lumen",
        settings=settings,
    )
    assert "Left for 2026-11-20: the gray zone." in left
    assert "Kevin does not write these." in left

    listed = execute_personal_tool(
        "promise",
        {"action": "list"},
        agent_id="lumen",
        settings=settings,
    )
    assert "the gray zone" in listed
    assert "2026-11-20" in listed

    dropped = execute_personal_tool(
        "promise",
        {"action": "drop", "name": "the gray"},
        agent_id="lumen",
        settings=settings,
    )
    assert "Dropped: the gray zone." in dropped


def test_promise_in_light_tools() -> None:
    from light_house.tools.approval_gate import HELD_TOOLS
    from light_house.tools.light_tools import LIGHT_TOOLS

    assert "promise" in {tool.name for tool in LIGHT_TOOLS}
    assert "promise" not in HELD_TOOLS


def test_gather_context_hands_only_on_free(settings: Settings) -> None:
    from unittest.mock import MagicMock, patch

    from light_house.agent.rumination_nodes import build_rumination_nodes
    from light_house.personal.promise import local_today

    today = local_today("America/Los_Angeles")
    get_personal_store(settings, "lumen").leave_promise("the gray zone", today.isoformat())
    memory = MagicMock()
    memory.count_unscored_for_thread.return_value = 0
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
        chores = gather_context(
            {"agent_id": "lumen", "thread_id": "t1", "wake_kind": "chores"}
        )
        free = gather_context(
            {"agent_id": "lumen", "thread_id": "t1", "wake_kind": None}
        )
        again = gather_context(
            {"agent_id": "lumen", "thread_id": "t1", "wake_kind": None}
        )

    assert "the gray zone" not in chores["agent_context_markdown"]
    assert "the gray zone" in free["agent_context_markdown"]
    assert "Due today." in free["agent_context_markdown"]
    assert "the gray zone" not in again["agent_context_markdown"]
