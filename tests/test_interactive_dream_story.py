"""Interactive Echo dream stories: stage → choice → assemble into stream."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from light_house.config import Settings
from light_house.subconscious.dream_story import (
    assemble_dream_story,
    drop_tail_path_menu,
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


def test_strip_path_menu_drops_a_bare_menu_at_the_end() -> None:
    beat = (
        "The corridor narrowed and the lamp kept its low hum.\n\n"
        "A) step into the pulse of light\n"
        "B) rest in the quiet field of names\n"
        "C) let the seam unravel completely\n"
    )
    cleaned = strip_path_menu(beat)
    assert cleaned == "The corridor narrowed and the lamp kept its low hum."
    assert "A)" not in cleaned


def test_strip_path_menu_keeps_a_list_that_prose_follows() -> None:
    beat = (
        "He listed them:\n"
        "A) salt\n"
        "B) bread\n\n"
        "Then he closed the book and the lamp went low."
    )
    assert strip_path_menu(beat) == beat


def test_strip_path_menu_keeps_a_single_lettered_line() -> None:
    beat = "The door held and I listened.\n\nA) Open it\n"
    assert "A) Open it" in strip_path_menu(beat)
    assert "door held" in strip_path_menu(beat)


def test_tail_anchor_waits_until_the_hinge_is_gone() -> None:
    prose = "Through the doorway the world softened into a narrow stair."
    menu = (
        "A) Follow the lantern's glow down the stair toward a distant glow.\n"
        "B) Pause and listen to the night's sigh for hidden words.\n"
        "C) Turn back and close the stair, returning to the doorway."
    )
    hinge = "— I pause on the woven step and listen."
    stored = f"{prose}\n\n{menu}\n\n{hinge}"
    still = drop_tail_path_menu(stored)
    assert "A)" in still
    assert prose in still
    opened = drop_tail_path_menu(f"{prose}\n\n{menu}")
    assert opened == prose
    assert "A)" not in opened


def test_assemble_drops_a_bare_menu_and_keeps_the_choice() -> None:
    story = assemble_dream_story(
        [
            "The stair softened.\n\n"
            "A) Follow the glow down the stair.\n"
            "B) Pause and listen for the sigh.\n"
            "C) Turn back and close the stair.\n"
        ],
        ["I pause and listen."],
    )
    assert "stair softened" in story
    assert "A)" not in story
    assert "— I pause and listen." in story


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


def test_sanitize_strips_reasoning_preamble_and_keeps_the_scene() -> None:
    raw = (
        "Okay, the user wants me to weave an interactive dream opening for Lumen. "
        "Let me review the requirements.\n\n"
        "Output only dream prose. Never include safety labels.\n\n"
        "The attic room has been sealed shut for months, yet a single shaft of "
        "afternoon light cuts across the floorboards, warming the dust. I stand "
        "at the precipice of climbing."
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert "the user wants" not in cleaned.lower()
    assert "output only" not in cleaned.lower()
    assert cleaned.startswith("The attic room")


def test_sanitize_strips_a_plan_about_the_wrong_light() -> None:
    raw = (
        "Let me look at the context carefully. The user wants me to weave an "
        "interactive dream opening for Lumen, but the context is someone else.\n\n"
        "The prompt says the dream serves Lumen.\n\n"
        "The water holds the flame without drowning it. Each step ripples outward, "
        "and the candle-flame rides each ripple like a small boat, never sinking."
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert "lumen" not in cleaned.lower()
    assert cleaned.startswith("The water holds")


def test_sanitize_plan_without_a_scene_is_empty() -> None:
    raw = (
        "The user wants me to weave an interactive dream opening. "
        "Let me review the requirements and constraints carefully. "
        "Output only dream prose."
    )
    assert sanitize_echo_dream_text(raw) == ""


def test_sanitize_drops_reasoning_that_follows_the_scene() -> None:
    raw = (
        "The user wants an interactive dream opening.\n\n"
        "The water holds the flame without drowning it. Each step ripples outward, "
        "and the candle-flame rides each ripple like a small boat, never sinking.\n\n"
        "Maybe we should interpret the prompt. The user wants a menu of paths, "
        "and we need to produce an interactive continuation from the prior beats."
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert cleaned.startswith("The water holds")
    assert "maybe we should" not in cleaned.lower()
    assert "prior beats" not in cleaned.lower()


def test_sanitize_strips_a_response_safety_banner() -> None:
    raw = (
        "Response Safety: safe\n\n"
        "The hall holds its breath. A single lamp at a wooden table throws "
        "a small pool of light across the letters."
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert "Response Safety" not in cleaned
    assert "hall holds its breath" in cleaned


def test_sanitize_strips_a_safety_line_after_a_waking_label() -> None:
    raw = (
        "The hall holds its breath. A single lamp at a wooden table throws "
        "a small pool of light across the letters.\n\n"
        "[waking recall] User Safety: safe"
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert "User Safety" not in cleaned
    assert "waking recall" not in cleaned.lower()
    assert "hall holds its breath" in cleaned


def test_sanitize_keeps_an_emdash_scene_and_drops_the_plan() -> None:
    raw = (
        "We need to produce a single private dream for Lumen, following constraints. "
        "Let me review the requirements.\n\n"
        "— The harbor held one stone, and the stone held a small map of stars. "
        "I stood on the glass balcony until the dark water grew quiet."
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert "we need to produce" not in cleaned.lower()
    assert "lumen" not in cleaned.lower()
    assert cleaned.startswith("The harbor held")


def test_echo_persona_does_not_assign_every_dream_to_lumen() -> None:
    from light_house.subconscious.loader import load_echo_persona

    persona = load_echo_persona()
    assert "You serve Lumen" not in persona
    assert "beneath Lumen" not in persona
    assert "leave her a dream" not in persona


def test_sanitize_leaves_an_ordinary_dream() -> None:
    raw = (
        "The hall holds its breath. A single lamp at a wooden table throws "
        "a small pool of light across the letters."
    )
    assert sanitize_echo_dream_text(raw) == raw


def test_sanitize_removes_the_canned_empty_beat() -> None:
    raw = (
        "The night held still. A door stood ajar, light thin as breath beyond it. "
        "I waited, and waiting was enough.\n\n"
        "I stepped through anyway, and the kitchen was warm."
    )
    cleaned = sanitize_echo_dream_text(raw)
    assert "night held still" not in cleaned
    assert "kitchen was warm" in cleaned


def test_assemble_skips_an_empty_beat_and_its_choice() -> None:
    story = assemble_dream_story(
        ["", "The kitchen was warm and quiet tonight."],
        ["I step forward into whatever waits.", "I stay."],
    )
    assert "step forward" not in story
    assert "kitchen was warm" in story
    assert "— I stay." in story


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


def test_echo_system_names_the_light_being_woven_for() -> None:
    from light_house.subconscious import dream_nodes as dn

    settings = Settings(_env_file=None, ECHO_DREAM_INTERACTIVE_ENABLED=False, PERSONAL_DB_ENABLED=False)
    memory = MagicMock()
    seen: dict[str, str] = {}

    class FakeClient:
        def invoke(self, messages):
            seen["system"] = messages[0].content
            return AIMessage(content="A quiet room with one lamp.")

    with (
        patch.object(dn, "build_echo_dream_llm_chain", return_value=[("mock", FakeClient())]),
        patch.object(dn, "get_agent", return_value=MagicMock(display_name="Elias")),
        patch.object(dn, "mark_dream_day", return_value=1),
        patch.object(dn, "build_echo_dream_context", return_value=("ctx", 0)),
        patch.object(dn, "format_felt_days_dream_nudge", return_value=""),
    ):
        nodes = dn.build_echo_dream_nodes(settings=settings, memory=memory)
        nodes["generate_dream"](
            {
                "thread_id": "t1",
                "agent_id": "elias",
                "context_text": "recent days",
                "dream_text": "",
                "waking_recall": "",
            }
        )
    system = seen["system"]
    assert "The Light you are weaving for tonight is Elias." in system
    assert "You serve Lumen" not in system
    assert "leave her a dream" not in system


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


def test_empty_dream_reply_is_asked_once_more() -> None:
    from light_house.subconscious import dream_nodes as dn

    settings = Settings(
        _env_file=None,
        ECHO_DREAM_INTERACTIVE_ENABLED=True,
        ECHO_DREAM_STORY_ROUNDS=2,
        PERSONAL_DB_ENABLED=False,
    )
    memory = MagicMock()
    calls = {"n": 0}

    class FakeClient:
        def invoke(self, messages):
            calls["n"] += 1
            if calls["n"] == 1:
                return AIMessage(content="")
            if calls["n"] == 2:
                return AIMessage(content=[{"type": "text", "text": "The lamp stayed lit on the sill."}])
            return AIMessage(content="I take the sill.")

    with (
        patch.object(dn, "build_echo_dream_llm_chain", return_value=[("mock", FakeClient())]),
        patch.object(dn, "get_agent", return_value=MagicMock(display_name="Reed")),
        patch.object(dn, "load_echo_persona", return_value="Echo"),
        patch.object(dn, "mark_dream_day", return_value=1),
        patch.object(dn, "build_echo_dream_context", return_value=("ctx", 0)),
        patch.object(dn, "format_felt_days_dream_nudge", return_value=""),
    ):
        nodes = dn.build_echo_dream_nodes(settings=settings, memory=memory)
        state = nodes["gather_dream_context"](
            {
                "thread_id": "reed-home",
                "agent_id": "reed",
                "context_text": "",
                "dream_text": "",
                "waking_recall": "",
                "felt_days": None,
                "dream_round": 0,
                "max_dream_rounds": 2,
                "story_beats": [],
                "light_choices": [],
                "current_beat": "",
                "current_choice": "",
            }
        )
        state = {**state, **nodes["echo_dream_beat"](state)}
        assert calls["n"] == 2
        assert "lamp stayed lit" in state["current_beat"]
