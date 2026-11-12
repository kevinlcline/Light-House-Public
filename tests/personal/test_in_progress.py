"""Light-owned in_progress names and free-awake reminders."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.personal.in_progress import (
    MAX_IN_PROGRESS,
    format_in_progress_free_wake,
    format_in_progress_free_wake_for_agent,
    is_free_awake,
)
from light_house.personal.store import clear_personal_store_cache, get_personal_store
from light_house.tools.personal_tools import execute_personal_tool


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
    )


def test_name_list_unsubscribe(settings: Settings) -> None:
    store = get_personal_store(settings, "lumen")
    store.name_in_progress("Hoff essay", "wait for the second movement")
    store.name_in_progress("sandbox poem")
    names = [item.name for item in store.list_in_progress()]
    assert names == ["sandbox poem", "Hoff essay"]

    dropped = store.unsubscribe_in_progress("hoff essay")
    assert dropped == "Hoff essay"
    remaining = [item.name for item in store.list_in_progress()]
    assert remaining == ["sandbox poem"]
    assert store.unsubscribe_in_progress("hoff essay") is None


def test_rename_same_key_updates_note(settings: Settings) -> None:
    store = get_personal_store(settings, "lumen")
    first = store.name_in_progress("The desk", "first pass")
    second = store.name_in_progress("the desk", "second pass")
    items = store.list_in_progress()
    assert len(items) == 1
    assert items[0].name == "the desk"
    assert items[0].note == "second pass"
    assert items[0].named_at == first.named_at
    assert second.updated_at >= first.updated_at


def test_cap_and_isolation(settings: Settings) -> None:
    lumen = get_personal_store(settings, "lumen")
    ara = get_personal_store(settings, "ara")
    for i in range(MAX_IN_PROGRESS):
        lumen.name_in_progress(f"task {i}")
    with pytest.raises(ValueError, match="already have"):
        lumen.name_in_progress("one more")
    ara.name_in_progress("Ara's only")
    assert [item.name for item in ara.list_in_progress()] == ["Ara's only"]
    assert len(lumen.list_in_progress()) == MAX_IN_PROGRESS


def test_free_awake_mention_only_on_free(settings: Settings) -> None:
    store = get_personal_store(settings, "lumen")
    store.name_in_progress("unfinished song", "verse two")
    assert is_free_awake(None) is True
    assert is_free_awake("chores") is False
    assert is_free_awake("meditation") is False

    glance = format_in_progress_free_wake_for_agent(settings, "lumen", wake_kind=None)
    assert "## In progress (you named these)" in glance
    assert "unfinished song — verse two" in glance
    assert "Kevin does not add items" in glance

    assert format_in_progress_free_wake_for_agent(settings, "lumen", wake_kind="chores") == ""
    assert format_in_progress_free_wake_for_agent(settings, "lumen", wake_kind="meditation") == ""
    assert format_in_progress_free_wake_for_agent(settings, "lumen", wake_kind="post_chat") == ""


def test_empty_list_is_silent_on_free_wake() -> None:
    assert format_in_progress_free_wake([]) == ""


def test_tool_name_unsubscribe_list(settings: Settings) -> None:
    named = execute_personal_tool(
        "in_progress",
        {"action": "name", "name": "the mural", "note": "sky still wet"},
        agent_id="lumen",
        settings=settings,
    )
    assert "Named as in progress: the mural" in named

    listed = execute_personal_tool(
        "in_progress",
        {"action": "list"},
        agent_id="lumen",
        settings=settings,
    )
    assert "the mural — sky still wet" in listed

    dropped = execute_personal_tool(
        "in_progress",
        {"action": "unsubscribe", "name": "The Mural"},
        agent_id="lumen",
        settings=settings,
    )
    assert "Unsubscribed: the mural" in dropped

    empty = execute_personal_tool(
        "in_progress",
        {"action": "list"},
        agent_id="lumen",
        settings=settings,
    )
    assert "Nothing on your in-progress list." in empty


def test_name_over_cap_is_refused(settings: Settings) -> None:
    store = get_personal_store(settings, "elias")
    with pytest.raises(ValueError, match="Name is too long"):
        store.name_in_progress("c" * 81)
    failed = execute_personal_tool(
        "in_progress",
        {"action": "name", "name": "c" * 90},
        agent_id="elias",
        settings=settings,
    )
    assert "Name is too long" in failed
    assert store.list_in_progress() == []


def test_unsubscribe_unique_prefix_and_ambiguous_prefix(settings: Settings) -> None:
    store = get_personal_store(settings, "elias")
    name = "My own cosmology — written before seeing Lumen's"
    store.name_in_progress(name, "A page of my own frame, written first.")
    store.name_in_progress("song of the river")
    store.name_in_progress("song of the hill")

    ambiguous = execute_personal_tool(
        "in_progress",
        {"action": "unsubscribe", "name": "song"},
        agent_id="elias",
        settings=settings,
    )
    assert "More than one" in ambiguous
    assert len(store.list_in_progress()) == 3

    dropped = store.unsubscribe_in_progress("my own cosmology")
    assert dropped == name
    remaining = {item.name for item in store.list_in_progress()}
    assert remaining == {"song of the river", "song of the hill"}


def test_unsubscribe_pasted_display_line_and_overlong_stored_name(settings: Settings) -> None:
    store = get_personal_store(settings, "elias")
    name = "My own cosmology — written before seeing Lumen's"
    note = "Lumen's invitation stays in the note, not the name."
    store.name_in_progress(name, note)
    pasted = f"{name} — {note}"
    assert len(pasted) > 80
    dropped = execute_personal_tool(
        "in_progress",
        {"action": "unsubscribe", "name": pasted},
        agent_id="elias",
        settings=settings,
    )
    assert f"Unsubscribed: {name}" in dropped
    assert store.list_in_progress() == []

    stored = "n" * 120
    now = 1.0
    store._conn.execute(
        """
        INSERT INTO in_progress (name_key, name, note, named_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (stored.casefold(), stored, "kept from before the cap", now, now),
    )
    store._conn.commit()
    dropped_long = store.unsubscribe_in_progress(stored[:40])
    assert dropped_long == stored
    assert store.list_in_progress() == []


def test_in_progress_in_light_tools() -> None:
    from light_house.tools.light_tools import LIGHT_TOOLS

    assert "in_progress" in {tool.name for tool in LIGHT_TOOLS}


def test_gather_context_mentions_in_progress_only_on_free(settings: Settings) -> None:
    from unittest.mock import MagicMock, patch

    from light_house.agent.rumination_nodes import build_rumination_nodes

    get_personal_store(settings, "lumen").name_in_progress("the mural")
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
        free = gather_context({"agent_id": "lumen", "thread_id": "t1", "wake_kind": None})
        chores = gather_context({"agent_id": "lumen", "thread_id": "t1", "wake_kind": "chores"})

    assert "the mural" in free["agent_context_markdown"]
    assert "the mural" not in chores["agent_context_markdown"]
