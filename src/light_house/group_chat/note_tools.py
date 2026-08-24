"""Bounded note-tool access for group chat turns."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from light_house.agent.tool_helpers import (
    ai_message_text,
    invoke_resilient_plain,
    last_message_has_tool_calls,
    run_tool_calls,
)
from light_house.config import Settings
from light_house.tools.light_tools import GROUP_CHAT_NOTE_TOOLS

logger = logging.getLogger(__name__)

_FINALIZE_HINT = (
    "Note tools are done for this beat (or unused). "
    "Reply now with **only** the JSON action object — no tools, no prose outside JSON."
)


def bind_group_note_tools(client: BaseChatModel) -> BaseChatModel:
    return client.bind_tools(GROUP_CHAT_NOTE_TOOLS)


def invoke_with_group_note_tools(
    llm_chain: list[tuple[str, BaseChatModel]],
    messages: list[BaseMessage],
) -> AIMessage:
    last_exc: Exception | None = None
    for tier_name, client in llm_chain:
        try:
            response = bind_group_note_tools(client).invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError("Model did not return an AIMessage")
            return response
        except Exception as exc:
            last_exc = exc
            logger.exception("Group note-tools LLM tier %s failed", tier_name)
            if tier_name != llm_chain[-1][0]:
                logger.warning("Trying next LLM tier after %s group-note failure", tier_name)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No LLM clients configured")


def run_group_note_tool_rounds(
    *,
    settings: Settings,
    llm_chain: list[tuple[str, BaseChatModel]],
    messages: list[BaseMessage],
    agent_id: str,
) -> str:
    """
    Allow a few note-tool rounds, then return the final plain-text decision body.

    ``messages`` is mutated in place (tool call + ToolMessage history).
    """
    max_rounds = max(0, int(settings.group_chat_max_tool_rounds))
    if max_rounds <= 0:
        response = invoke_resilient_plain(llm_chain, messages)
        messages.append(response)
        return ai_message_text(response)

    used_tools = False
    for round_i in range(max_rounds):
        response = invoke_with_group_note_tools(llm_chain, messages)
        messages.append(response)
        if not last_message_has_tool_calls(messages):
            text = ai_message_text(response)
            if text.strip():
                return text
            break
        used_tools = True
        tool_messages = run_tool_calls(
            response,
            agent_id=agent_id,
            chat_channel="group",
        )
        messages.extend(tool_messages)
        logger.info(
            "Group chat note tools agent=%s round=%d/%d calls=%d",
            agent_id,
            round_i + 1,
            max_rounds,
            len(tool_messages),
        )

    # Cap hit while still tool-calling, or empty non-tool reply — force JSON finalize.
    finalize_msgs = list(messages)
    if used_tools or last_message_has_tool_calls(finalize_msgs):
        finalize_msgs.append(HumanMessage(content=_FINALIZE_HINT))
    response = invoke_resilient_plain(llm_chain, finalize_msgs)
    messages.append(response)
    return ai_message_text(response)
