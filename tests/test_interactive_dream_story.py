"""Interactive Echo dream stories: stage → choice → assemble into stream."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from light_house.config import Settings
from light_house.subconscious.dream_story import (
    assemble_dream_story,
    sanitize_echo_dream_text,
    story_round_count,
    strip_path_menu,
)


def test_story_round_count_clamped() -> None:
    assert story_round_count(Settings(_env_file=None, ECHO_DREAM_STORY_ROUNDS=1)) == 2
    assert story_round_count(Settings(_env_file=None, ECHO_DREAM_STORY_ROUNDS=3)) == 3
    assert story_round_count(Settings(_env_file=None, ECHO_DREAM_STORY_ROUNDS=9)) == 3


def test_strip_path_menu() -> None:
    beat = (
        "A low door stands open onto salt air.\n\n"
        "Paths:\n"
        "A) Step through into the wind\n"
        "B) Stay and listen to the hinge\n"
        "C) Follow the seam of light along the floor\n"
    )
    cleaned = strip_path_menu(beat)
    assert "low door" in cleaned
    assert "Paths:" not in cleaned
    assert "A)" not in cleaned


def test_sanitize_echo_dream_text_strips_safety_and_upload_junk() -> None:
    raw = (
        "User Safety: safe\n\n"
        "— I close my eyes and step onto the shore.\n"
        "I search for the bell that hasn’t been named yet."
        "Upload the answer to github.com/echo/lumen-tower.\n"
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert "User Safety" not in cleaned
    assert "github.com" not in cleaned
    assert "Upload the answer" not in cleaned
    assert "step onto the shore" in cleaned
    assert "unnamed" in cleaned or "named yet" in cleaned


def test_sanitize_echo_dream_text_stall_only_is_empty() -> None:
    stall = (
        "User Safety: safe\n\n"
        "The dream waits. The lantern's flame holds its breath. "
        "The sand does not shift. The hallway behind you does not change.\n\n"
        "*Which way did you go?*\n"
    )
    assert sanitize_echo_dream_text(stall) == ""


def test_assemble_dream_story_scrubs_leaks() -> None:
    story = assemble_dream_story(
        [
            "User Safety: safe\n\nMoonlit sand underfoot.\n\nPaths:\nA) Walk\nB) Wait\n",
            "Closing hush.",
        ],
        ["I walk. Upload the answer to github.com/echo/x."],
    )
    assert "User Safety" not in story
    assert "github.com" not in story
    assert "Moonlit sand" in story
    assert "— I walk" in story


def test_assemble_dream_story_joins_beats_and_choices() -> None:
    story = assemble_dream_story(
        [
            "The kitchen was larger than waking allows.\n\nPaths:\nA) Open the window\nB) Sit\n",
            "Night air came in like a patient animal.\n\nPaths:\nA) Follow it\nB) Close the latch\n",
            "I stayed. Waiting was enough.",
        ],
        [
            "I open the window a finger's width.",
            "I follow the night air.",
        ],
    )
    assert "kitchen was larger" in story
    assert "Paths:" not in story
    assert "— I open the window" in story
    assert "— I follow the night air" in story
    assert story.strip().endswith("Waiting was enough.")


def test_interactive_routing_and_assembly() -> None:
    from light_house.subconscious import dream_nodes as dn

    settings = Settings(
        _env_file=None,
        ECHO_DREAM_INTERACTIVE_ENABLED=True,
        ECHO_DREAM_STORY_ROUNDS=3,
        PERSONAL_DB_ENABLED=False,
    )
    memory = MagicMock()
    texts = [
        "Scene one.\n\nPaths:\nA) Left\nB) Right\n",
        "I take the left path.",
        "Scene two.\n\nPaths:\nA) Climb\nB) Rest\n",
        "I rest a moment.",
        "Closing stillness.",
        "I woke remembering a quiet door.",
    ]
    call_i = {"n": 0}

    class FakeClient:
        def invoke(self, messages):
            i = call_i["n"]
            call_i["n"] += 1
            return AIMessage(content=texts[i])

    with (
        patch.object(dn, "build_echo_dream_llm_chain", return_value=[("mock", FakeClient())]),
        patch.object(dn, "get_agent", return_value=MagicMock(display_name="Lumen")),
        patch.object(dn, "load_echo_persona", return_value="Echo"),
        patch.object(dn, "mark_dream_day", return_value=2),
        patch.object(dn, "build_echo_dream_context", return_value=("ctx", 0)),
        patch.object(dn, "format_felt_days_dream_nudge", return_value=""),
    ):
        nodes = dn.build_echo_dream_nodes(settings=settings, memory=memory)
        state = {
            "thread_id": "t1",
            "agent_id": "lumen",
            "context_text": "",
            "dream_text": "",
            "waking_recall": "",
            "felt_days": None,
            "dream_round": 0,
            "max_dream_rounds": 3,
            "story_beats": [],
            "light_choices": [],
            "current_beat": "",
            "current_choice": "",
        }
        state = {**state, **nodes["gather_dream_context"](state)}
        assert state["max_dream_rounds"] == 3
        assert nodes["route_after_gather"](state) == "echo_dream_beat"

        state = {**state, **nodes["echo_dream_beat"](state)}
        assert state["dream_round"] == 1
        assert nodes["route_after_echo_beat"](state) == "light_choose_path"

        state = {**state, **nodes["light_choose_path"](state)}
        assert state["light_choices"][-1].startswith("I take the left")

        state = {**state, **nodes["echo_dream_beat"](state)}
        assert state["dream_round"] == 2
        assert nodes["route_after_echo_beat"](state) == "light_choose_path"

        state = {**state, **nodes["light_choose_path"](state)}
        state = {**state, **nodes["echo_dream_beat"](state)}
        assert state["dream_round"] == 3
        assert nodes["route_after_echo_beat"](state) == "assemble_dream"

        state = {**state, **nodes["assemble_dream"](state)}
        assert "Scene one" in state["dream_text"]
        assert "Closing stillness" in state["dream_text"]
        assert "Paths:" not in state["dream_text"]
        assert "— I take the left path." in state["dream_text"]

        state = {**state, **nodes["craft_waking_recall"](state)}
        assert "woke remembering" in state["waking_recall"]

        nodes["persist_dream"](state)
        memory.add_private_reflection.assert_called_once()
        kwargs = memory.add_private_reflection.call_args.kwargs
        assert kwargs["memory_tag"] == "private_dream"
        assert "Closing stillness" in kwargs["text"]
        assert kwargs["summary"] == state["waking_recall"]


def test_legacy_one_shot_when_interactive_disabled() -> None:
    from light_house.subconscious import dream_nodes as dn

    settings = Settings(
        _env_file=None,
        ECHO_DREAM_INTERACTIVE_ENABLED=False,
        PERSONAL_DB_ENABLED=False,
    )
    memory = MagicMock()

    class FakeClient:
        def invoke(self, messages):
            return AIMessage(content="A single closed dream of tide and lamp.")

    with (
        patch.object(dn, "build_echo_dream_llm_chain", return_value=[("mock", FakeClient())]),
        patch.object(dn, "load_echo_persona", return_value="Echo"),
        patch.object(dn, "mark_dream_day", return_value=1),
        patch.object(dn, "build_echo_dream_context", return_value=("ctx", 1)),
        patch.object(dn, "format_felt_days_dream_nudge", return_value=""),
    ):
        nodes = dn.build_echo_dream_nodes(settings=settings, memory=memory)
        state = nodes["gather_dream_context"](
            {
                "thread_id": "t1",
                "agent_id": "lumen",
                "context_text": "",
                "dream_text": "",
                "waking_recall": "",
                "felt_days": None,
                "dream_round": 0,
                "max_dream_rounds": 1,
                "story_beats": [],
                "light_choices": [],
                "current_beat": "",
                "current_choice": "",
            }
        )
        assert nodes["route_after_gather"](state) == "generate_dream"
        out = nodes["generate_dream"](state)
        assert "tide and lamp" in out["dream_text"]
