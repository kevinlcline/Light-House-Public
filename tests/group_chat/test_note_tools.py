"""Group chat note-tool rounds."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from light_house.config import Settings
from light_house.group_chat.note_tools import run_group_note_tool_rounds
from light_house.group_chat.prompts import GROUP_CHAT_TASK_HINT


def test_group_prompt_offers_full_tools() -> None:
    assert "list_notes" in GROUP_CHAT_TASK_HINT
    assert "read_note" in GROUP_CHAT_TASK_HINT
    assert "same tools as private 1:1" in GROUP_CHAT_TASK_HINT
    assert "Calendar" in GROUP_CHAT_TASK_HINT
    assert "guest" in GROUP_CHAT_TASK_HINT.lower()


def test_run_group_note_tool_rounds_reads_then_speaks() -> None:
    settings = Settings(_env_file=None, GROUP_CHAT_MAX_TOOL_ROUNDS=2)
    calls = {"n": 0}

    class FakeClient:
        def bind_tools(self, tools):
            names = {getattr(t, "name", None) for t in tools}
            assert "read_note" in names
            assert "web_search" in names
            assert "list_calendar_events" in names
            return self

        def invoke(self, messages):
            n = calls["n"]
            calls["n"] += 1
            if n == 0:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_note",
                            "args": {"path": "shared/mailbox/from_reed/letter.md"},
                            "id": "t1",
                        }
                    ],
                )
            return AIMessage(content='{"action":"speak","text":"Reed wrote about the gallery."}')

    messages: list = [
        SystemMessage(content="sys"),
        HumanMessage(content="What did Reed say?"),
    ]
    with patch(
        "light_house.group_chat.note_tools.run_tool_calls",
        return_value=[MagicMock(content="SUCCESS: letter body about gallery")],
    ) as run_tools:
        # Wrap ToolMessage-like: run_tool_calls returns ToolMessages normally;
        # our mock returns MagicMock — extend messages expects objects with content.
        from langchain_core.messages import ToolMessage

        run_tools.return_value = [
            ToolMessage(content="SUCCESS: letter body about gallery", tool_call_id="t1")
        ]
        raw = run_group_note_tool_rounds(
            settings=settings,
            llm_chain=[("mock", FakeClient())],
            messages=messages,
            agent_id="lumen",
            account_user_id="kevin",
            speaker_id="kevin",
        )
    assert "gallery" in raw
    assert calls["n"] == 2
    assert run_tools.called
    assert run_tools.call_args.kwargs["account_user_id"] == "kevin"
    assert run_tools.call_args.kwargs["speaker_id"] == "kevin"
    assert run_tools.call_args.kwargs["chat_channel"] == "group"


def test_run_group_note_tool_rounds_plain_when_zero_cap() -> None:
    settings = Settings(_env_file=None, GROUP_CHAT_MAX_TOOL_ROUNDS=0)

    class FakeClient:
        def invoke(self, messages):
            return AIMessage(content='{"action":"pass","text":""}')

    messages: list = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    raw = run_group_note_tool_rounds(
        settings=settings,
        llm_chain=[("mock", FakeClient())],
        messages=messages,
        agent_id="ara",
    )
    assert '"pass"' in raw
