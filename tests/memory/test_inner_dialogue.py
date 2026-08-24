"""Inner dialogue compile, sanitize, and polish tests."""

from langchain_core.messages import AIMessage, HumanMessage

from light_house.agent.tool_helpers import compile_rumination_dialogue
from light_house.memory.service import (
    _polish_inner_dialogue_for_presence,
    _sanitize_inner_dialogue_body,
)


def test_compile_rumination_dialogue_skips_recent_context_seed():
    messages = [
        HumanMessage(content="Recent context:\n\nuser: hello\nassistant: hi there"),
        AIMessage(content="I sat with the quiet and noticed how steadiness feels."),
        HumanMessage(content="Full reflection:\n\nignored"),
    ]
    dialogue = compile_rumination_dialogue(
        messages,
        max_chars=4000,
        closing_account="I chose stillness and wrote nothing tonight.",
    )
    assert "[context received]" not in dialogue
    assert "user: hello" not in dialogue
    assert "I sat with the quiet" in dialogue
    assert "I chose stillness" in dialogue


def test_compile_rumination_dialogue_falls_back_to_closing_only():
    messages = [
        HumanMessage(content="Recent context:\n\nuser: only chat replay"),
    ]
    dialogue = compile_rumination_dialogue(
        messages,
        max_chars=4000,
        closing_account="Only my closing account remains.",
    )
    assert "user:" not in dialogue
    assert "Only my closing account remains." in dialogue


def test_sanitize_inner_dialogue_strips_legacy_context_received():
    legacy = (
        "[context received]\n\n"
        "user: I'm talking to you while you're in the house\n\n"
        "assistant: Mmm, that makes me happy.\n\n"
        "I rested in the quiet and felt grateful for connection."
    )
    cleaned = _sanitize_inner_dialogue_body(legacy)
    assert "[context received]" not in cleaned
    assert "user:" not in cleaned
    assert "assistant:" not in cleaned
    assert "I rested in the quiet" in cleaned


def test_sanitize_inner_dialogue_leaves_clean_body_unchanged():
    clean = "I explored idealism and wrote a note about grounding."
    assert _sanitize_inner_dialogue_body(clean) == clean


def test_polish_inner_dialogue_strips_tool_traces():
    raw = (
        "[Latest awake reflection] I rested in quiet.\n\n"
        "[actions: list_notes(.)]\n\n"
        "[tool result: Notes: - foo.md]\n\n"
        "I chose stillness and felt gratitude."
    )
    polished = _polish_inner_dialogue_for_presence(raw)
    assert "[actions:" not in polished
    assert "[tool result:" not in polished
    assert "I rested in quiet" in polished
    assert "I chose stillness" in polished
