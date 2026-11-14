"""Textual DeepSeek tool calls must run, and must not be stored as speech."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from light_house.agent.tool_helpers import (
    ai_message_text,
    invoke_resilient_plain,
    invoke_with_tools,
    latest_assistant_text,
    normalize_model_message,
    prepare_plain_close,
)
from light_house.config import Settings

_BAR = "\uff5c"

def _dsml(body: str) -> str:
    return body.replace("｜", _BAR)


_LIST = _dsml(
    """<｜DSML｜ calls>
<｜DSML｜ invoke name="sandbox_list">
<｜DSML｜ parameter name="space" string="true">playpen</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>"""
)

_WRITE = _dsml(
    """Let me put the checker on disk.

<｜DSML｜ calls>
<｜DSML｜ invoke name="sandbox_write">
<｜DSML｜ parameter name="path" string="true">check_plasma.py</｜DSML｜ parameter>
<｜DSML｜ parameter name="content" string="true">print("ok")
</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>"""
)

_COMPACT = _dsml(
    """<｜DSML｜tool_calls>
<｜DSML｜invoke name="read_note">
<｜DSML｜parameter name="path" string="true">mailbox/from_reed/letter.md</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
)

_NUMBERS = _dsml(
    """<｜DSML｜ calls>
<｜DSML｜ invoke name="read_note">
<｜DSML｜ parameter name="path" string="true">memory/learnings.inbox.md</｜DSML｜ parameter>
<｜DSML｜ parameter name="offset" string="false">2400</｜DSML｜ parameter>
<｜DSML｜ parameter name="max_chars" string="false">3400</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>"""
)

_TWO = _dsml(
    """<｜DSML｜ calls>
<｜DSML｜ invoke name="list_personal">

</｜DSML｜ invoke>
<｜DSML｜ invoke name="read_note">
<｜DSML｜ parameter name="path" string="true">ara/writing/garden/seeds.md</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>"""
)


def test_doubled_bars_become_a_tool_call_and_not_speech() -> None:
    """A wake can emit ｜｜DSML｜｜. That call must run, and must not sit in the chair."""
    doubled = _dsml(
        """<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="sandbox_run">
<｜｜DSML｜｜ parameter name="command" string="true">python3 check_wiring.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="space" string="true">playpen</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>"""
    )
    message = normalize_model_message(AIMessage(content=doubled))
    assert message.tool_calls[0]["name"] == "sandbox_run"
    assert message.tool_calls[0]["args"] == {
        "command": "python3 check_wiring.py",
        "space": "playpen",
    }
    assert message.content == ""
    assert "DSML" not in ai_message_text(message)

    client = MagicMock()
    client.invoke.return_value = AIMessage(content=doubled)
    plain = invoke_resilient_plain([("mock", client)], [HumanMessage(content="hi")])
    assert plain.tool_calls == []
    assert plain.content == ""
    assert "DSML" not in (plain.content or "")


def test_spaced_call_becomes_a_tool_call_and_not_speech() -> None:
    message = normalize_model_message(AIMessage(content=_LIST))
    assert message.tool_calls[0]["name"] == "sandbox_list"
    assert message.tool_calls[0]["args"] == {"space": "playpen"}
    assert message.content == ""
    assert ai_message_text(message) == ""


def test_prose_stays_and_file_body_does_not() -> None:
    message = normalize_model_message(AIMessage(content=_WRITE))
    assert message.content == "Let me put the checker on disk."
    assert message.tool_calls[0]["name"] == "sandbox_write"
    args = message.tool_calls[0]["args"]
    assert args["path"] == "check_plasma.py"
    assert args["content"] == 'print("ok")\n'
    assert "print" not in message.content
    assert "DSML" not in message.content


def test_compact_form_and_numeric_params() -> None:
    compact = normalize_model_message(AIMessage(content=_COMPACT))
    assert compact.tool_calls[0]["args"] == {"path": "mailbox/from_reed/letter.md"}

    numbers = normalize_model_message(AIMessage(content=_NUMBERS))
    assert numbers.tool_calls[0]["args"]["offset"] == 2400
    assert numbers.tool_calls[0]["args"]["max_chars"] == 3400
    assert isinstance(numbers.tool_calls[0]["args"]["path"], str)


def test_two_invokes_and_an_unclosed_call_is_not_speech() -> None:
    message = normalize_model_message(AIMessage(content=_TWO))
    assert [tc["name"] for tc in message.tool_calls] == ["list_personal", "read_note"]
    assert message.tool_calls[0]["args"] == {}

    hanging = _dsml(
        'I was going to say it.\n\n<｜DSML｜ invoke name="sandbox_write">'
        '<｜DSML｜ parameter name="content" string="true">SECRET BODY'
    )
    message = normalize_model_message(AIMessage(content=hanging))
    assert message.tool_calls == []
    assert message.content == "I was going to say it."
    assert "SECRET" not in message.content


def test_plain_pass_strips_without_attaching_a_call() -> None:
    client = MagicMock()
    client.invoke.return_value = AIMessage(content=_LIST)
    out = invoke_resilient_plain([("mock", client)], [HumanMessage(content="hi")])
    assert out.tool_calls == []
    assert out.content == ""


def test_tool_pass_recovers_the_call() -> None:
    client = MagicMock()
    client.invoke.return_value = AIMessage(content=_LIST)
    out = invoke_with_tools(client, [])
    assert out.tool_calls[0]["name"] == "sandbox_list"
    assert "DSML" not in (out.content or "")


def test_existing_tool_call_is_not_duplicated() -> None:
    raw = AIMessage(
        content=_LIST,
        tool_calls=[
            {"name": "sandbox_list", "id": "already", "args": {"space": "playpen"}}
        ],
    )
    message = normalize_model_message(raw)
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["id"] == "already"


def test_sentence_token_and_tool_result_wrapper_leave_the_words() -> None:
    token = _dsml("<｜begin▁of▁sentence｜>\nI need to read the pin.")
    assert ai_message_text(AIMessage(content=token)) == "I need to read the pin."

    wrapped = "![179go3]</tool_result>\nYou're still here. Good."
    assert ai_message_text(AIMessage(content=wrapped)) == "You're still here. Good."

    headed = "### Tool Result\n\nThe note came back attached.\n\n" + _LIST
    text = ai_message_text(AIMessage(content=headed))
    assert text == "The note came back attached."
    assert "DSML" not in text
    assert "playpen" not in text

    echoed = (
        "<tool_result>seed_text: (empty — use seed_text parameter)\n\n"
        "I'll try the right parameter name."
    )
    assert ai_message_text(AIMessage(content=echoed)) == "I'll try the right parameter name."


def test_a_stray_marker_is_not_speech() -> None:
    assert ai_message_text(AIMessage(content=">")) == ""
    assert ai_message_text(AIMessage(content=">\n")) == ""
    marked = ">\n\n" + _LIST
    assert ai_message_text(AIMessage(content=marked)) == ""
    assert "DSML" not in ai_message_text(AIMessage(content=marked))
    assert ai_message_text(AIMessage(content="I'm here.")) == "I'm here."
    assert ai_message_text(AIMessage(content="😊")) == "😊"


def test_plain_close_drops_an_unrun_tool_call() -> None:
    hanging = AIMessage(
        content=">",
        tool_calls=[{"name": "web_search", "id": "1", "args": {"query": "monroe"}}],
    )
    closed = prepare_plain_close(
        [HumanMessage(content="What did he write?"), hanging]
    )
    assert len(closed) == 1
    assert isinstance(closed[0], HumanMessage)

    spoken = AIMessage(
        content="The page is there.",
        tool_calls=[{"name": "web_search", "id": "2", "args": {"query": "monroe"}}],
    )
    kept = prepare_plain_close([spoken])
    assert kept[0].content == "The page is there."
    assert not kept[0].tool_calls


def test_latest_speech_skips_a_tool_reach() -> None:
    messages = [
        AIMessage(
            content="Let me check.",
            tool_calls=[{"name": "read_note", "id": "1", "args": {"path": "a.md"}}],
        ),
        AIMessage(content="The page is there."),
    ]
    assert latest_assistant_text(messages) == "The page is there."
    assert latest_assistant_text(messages[:1]) is None


def test_mailbox_close_is_not_copied_into_chat() -> None:
    from light_house.agent.rumination_nodes import build_rumination_nodes
    from light_house.mailbox.wake import WAKE_KIND_MAILBOX_LETTER

    settings = Settings.model_construct(
        personal_db_enabled=False,
        memory_scoring_rumination_hint=False,
        inner_life_trace_enabled=False,
        rumination_log_enabled=False,
        rumination_debug_log_enabled=False,
        chat_respond_window=12,
        rumination_max_tool_rounds=3,
        rumination_internal_loop_enabled=False,
        report_back_enabled=False,
    )
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
                "wake_kind": WAKE_KIND_MAILBOX_LETTER,
                "messages": [AIMessage(content="I read the letter.")],
                "generated_text": "I read the letter.\n\n[rest now]",
                "summary_text": "read the letter",
                "peer_inbox_ids": [],
            }
        )
    memory.add_private_reflection.assert_called_once()
    memory.append_peer_chat_reply.assert_not_called()
