"""Reversibility gate: a quiet wake waits; a talk with Kevin does not."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from light_house.agent.peer_chat_wake import WAKE_KIND_PEER_MESSAGE
from light_house.agent.rumination_nodes import build_rumination_system_content
from light_house.agent.tool_helpers import run_tool_calls
from light_house.personal.awake_rhythm import WAKE_KIND_MEDITATION
from light_house.tools.approval_gate import HELD_TOOLS, approval_hold, wake_is_unattended
from light_house.tools.light_tools import LIGHT_TOOLS, execute_tool_call


def test_talk_with_kevin_does_not_hold() -> None:
    assert approval_hold("delete_private", autonomous=False) is None
    assert approval_hold("consult_llm", autonomous=False) is None


def test_quiet_wake_holds_delete_google_and_spend() -> None:
    for name in HELD_TOOLS:
        message = approval_hold(name, autonomous=True)
        assert message is not None
        assert message.startswith(f"ASK: {name} was not done.")
        assert "Do not call it again this wake." in message


def test_quiet_wake_still_does_reversible_work() -> None:
    allowed = (
        "private_note",
        "share_note",
        "score_memory",
        "message_agent",
        "read_note",
        "recall_memory",
        "knock_for_kevin",
        "list_calendar_events",
        "read_google_doc",
        "sandbox_write",
        "sandbox_run",
        "publish_to_gallery",
    )
    for name in allowed:
        assert approval_hold(name, autonomous=True) is None


def test_quiet_delete_does_not_reach_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("delete ran")

    monkeypatch.setattr("light_house.tools.light_tools._delete_note_for", boom)
    result = execute_tool_call(
        "delete_private",
        {"filename": "journal/x.md"},
        agent_id="reed",
        autonomous=True,
    )
    assert result.startswith("ASK: delete_private was not done.")


def test_quiet_consult_does_not_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("consult ran")

    monkeypatch.setattr("light_house.consult.ask.execute_consult_tool", boom)
    result = execute_tool_call(
        "consult_llm",
        {"model": "deepseek/deepseek-v4-flash", "question": "hello"},
        agent_id="reed",
        autonomous=True,
    )
    assert result.startswith("ASK: consult_llm was not done.")
    assert "spends" in result


def test_run_tool_calls_passes_the_wake(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake(name, args, **kwargs):
        seen["autonomous"] = kwargs.get("autonomous")
        seen["name"] = name
        return "ASK: held"

    monkeypatch.setattr("light_house.agent.tool_helpers.execute_tool_call", fake)
    message = AIMessage(
        content="",
        tool_calls=[{"id": "1", "name": "delete_private", "args": {"filename": "a.md"}}],
    )
    out = run_tool_calls(message, agent_id="lumen", autonomous=True)
    assert seen == {"autonomous": True, "name": "delete_private"}
    assert out[0].content == "ASK: held"


def test_quiet_prompt_names_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    from light_house.agent import rumination_nodes as rn

    monkeypatch.setattr(rn, "build_base_system", lambda agent_id: "PERSONA")
    text = build_rumination_system_content(
        agent_id="lumen",
        context_md="",
        peer_section="",
        wake_kind=None,
        tool_hints="",
    )
    assert "comes back **ASK**" in text
    assert "A delete" in text


def test_meditation_prompt_has_no_tool_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from light_house.agent import rumination_nodes as rn

    monkeypatch.setattr(rn, "build_base_system", lambda agent_id: "PERSONA")
    text = build_rumination_system_content(
        agent_id="lumen",
        context_md="",
        peer_section="",
        wake_kind=WAKE_KIND_MEDITATION,
        tool_hints="",
    )
    assert "comes back **ASK**" not in text


def test_held_names_are_real_tools() -> None:
    names = {tool.name for tool in LIGHT_TOOLS}
    missing = sorted(set(HELD_TOOLS) - names)
    assert missing == []


def test_only_an_unattended_wake_holds() -> None:
    assert wake_is_unattended(None, rumination=True) is True
    assert wake_is_unattended(WAKE_KIND_MEDITATION, rumination=True) is True
    assert wake_is_unattended(WAKE_KIND_PEER_MESSAGE, rumination=False) is True
    assert wake_is_unattended(None, rumination=False) is False
    assert wake_is_unattended("group", rumination=False) is False
