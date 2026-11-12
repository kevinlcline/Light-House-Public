"""A tool-cap close must come back as speech, not a stray marker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from light_house.agent.nodes import build_nodes
from light_house.config import Settings


def _profile() -> MagicMock:
    profile = MagicMock()
    profile.provider.value = "openrouter"
    profile.model = "m"
    profile.model_fallback = None
    profile.inner_life_model = None
    return profile


def _finalize(replies: list[str]):
    settings = Settings.model_construct(
        chat_respond_window=12,
        chat_max_tool_rounds=12,
        house_dad_user_id="kevin",
    )
    memory = MagicMock()
    memory.load_thread_chat_history.return_value = []
    nodes = build_nodes(settings=settings, memory=memory)
    finalize = nodes[4]
    pending = list(replies)

    def invoke(_chain, _messages):
        return AIMessage(content=pending.pop(0))

    with (
        patch("light_house.agent.nodes.lookup_agent_llm_profile", return_value=_profile()),
        patch("light_house.agent.nodes.build_agent_llm_chain", return_value=[("mock", MagicMock())]),
        patch("light_house.agent.nodes.build_base_system", return_value=""),
        patch("light_house.agent.nodes.invoke_resilient_plain", side_effect=invoke),
    ):
        return finalize(
            {
                "agent_id": "lumen",
                "thread_id": "kevin-home",
                "wake_kind": None,
                "messages": [
                    HumanMessage(content="What did he write?"),
                    AIMessage(
                        content=">",
                        tool_calls=[
                            {
                                "name": "web_search",
                                "id": "1",
                                "args": {"query": "monroe"},
                            }
                        ],
                    ),
                ],
                "agent_context_markdown": "",
            }
        )


def test_finalize_asks_again_when_the_close_is_only_a_marker() -> None:
    out = _finalize([">", "People could set a body down, and some of them got bored."])
    text = out["messages"][0].content
    assert "bored" in text
    assert text.strip() != ">"


def test_finalize_speaks_when_both_closes_are_empty() -> None:
    out = _finalize([">", ">"])
    text = out["messages"][0].content
    assert "Ask me once more" in text
    assert text.strip() != ">"
