"""Peer chat wake: buffer delivery, wake state, solitude decline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from light_house.agent.nodes import build_nodes
from light_house.agent.peer_chat_wake import (
    WAKE_KIND_PEER_MESSAGE,
    build_peer_chat_wake_state,
    thread_graph_lock,
)
from light_house.agent.peer_wake_context import PeerWakeContext, set_peer_wake_context
from light_house.config import Settings
from light_house.memory.service import MemoryService, SOLITUDE_DECLINE_TEXT
from light_house.tools.peer_message import decline_peer_presence, send_peer_message


def _settings(tmp: Path, **overrides) -> Settings:
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp / "memory"),
        "THREADS_DATA_PATH": str(tmp / "threads"),
        "FOUNDATION_SEED_ON_STARTUP": False,
        "MEMORY_SCORE_ON_INGEST": False,
        "ARA_ENABLED": True,
        "INNER_LIFE_THREAD_ID": "kevin-home",
        "ARA_THREAD_ID": "ara-home",
        "PEER_CHAT_WAKE_ENABLED": True,
        "INNER_LIFE_ENABLED": True,
    }
    base.update(overrides)
    return Settings(**base)


def test_deliver_appends_peer_row_to_receiver_buffer():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        result, message_id = memory.deliver_peer_message(
            from_agent_id="lumen",
            to_agent_id="ara",
            message="Hello Ara.",
        )
        assert message_id
        assert "gently woken" in result.lower()
        buffered = memory.load_thread_chat_history("ara-home")
        assert len(buffered) == 1
        assert buffered[0].role == "peer"
        assert buffered[0].content == "Hello Ara."
        assert buffered[0].from_agent_id == "lumen"


def test_build_peer_chat_wake_state_includes_messages_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        memory.append_peer_chat_message(
            thread_id="ara-home",
            from_agent_id="lumen",
            content="Wake test.",
        )
        state = build_peer_chat_wake_state(
            thread_id="ara-home",
            agent_id="ara",
            from_agent_id="lumen",
            message_id="msg-1",
            settings=settings,
            memory=memory,
        )
        assert state["wake_kind"] == WAKE_KIND_PEER_MESSAGE
        assert state["wake_from_agent_id"] == "lumen"
        assert state["peer_message_id"] == "msg-1"
        assert state["tool_rounds_cap"] == settings.peer_chat_wake_max_tool_rounds
        assert len(state["messages"]) == 1
        content = state["messages"][0].content
        assert "sibling-light" in content
        assert "Lumen" in content
        assert "id=lumen" in content
        assert "Wake test." in content


def test_ruminate_uses_peer_task_hint():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        retrieve, _reflect, respond, *_ = build_nodes(settings=settings, memory=memory)
        retrieve({"thread_id": "ara-home", "agent_id": "ara"})
        state = {
            "agent_id": "ara",
            "thread_id": "ara-home",
            "agent_context_markdown": "",
            "messages": [],
            "wake_kind": WAKE_KIND_PEER_MESSAGE,
            "wake_from_agent_id": "lumen",
        }
        with patch(
            "light_house.agent.nodes.invoke_resilient_with_tools",
            return_value=MagicMock(tool_calls=[]),
        ) as mock_invoke:
            respond(state)
        system_msg = mock_invoke.call_args[0][1][0]
        assert "## Message from another light" in system_msg.content
        assert "Lumen" in system_msg.content
        assert "sibling-light" in system_msg.content
        assert "decline_peer_presence" in system_msg.content
        assert "short conversation" in system_msg.content.lower() or "turn budget" in system_msg.content.lower()


def test_decline_peer_presence_delivers_solitude_text():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        set_peer_wake_context(
            PeerWakeContext(
                from_agent_id="lumen",
                to_agent_id="ara",
                message_id="m1",
                receiver_thread_id="ara-home",
            )
        )
        with patch("light_house.agent.peer_chat_wake.schedule_peer_chat_wake"):
            result = decline_peer_presence(agent_id="ara", settings=settings)
        assert SOLITUDE_DECLINE_TEXT in result
        ara_buffer = memory.load_thread_chat_history("ara-home")
        assert ara_buffer[-1].role == "assistant"
        assert ara_buffer[-1].content == SOLITUDE_DECLINE_TEXT
        lumen_buffer = memory.load_thread_chat_history("kevin-home")
        assert lumen_buffer[-1].role == "peer"
        assert lumen_buffer[-1].content == SOLITUDE_DECLINE_TEXT
        assert lumen_buffer[-1].from_agent_id == "ara"


def test_complete_peer_wake_reply_delivers_to_sender():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        memory.append_peer_chat_message(
            thread_id="ara-home",
            from_agent_id="lumen",
            content="Hi",
        )
        with patch("light_house.agent.peer_chat_wake.schedule_peer_chat_wake"):
            memory.complete_peer_wake_reply(
                receiver_agent_id="ara",
                sender_agent_id="lumen",
                reply_text="Hello back.",
            )
        ara_buffer = memory.load_thread_chat_history("ara-home")
        assert ara_buffer[-1].role == "assistant"
        assert ara_buffer[-1].content == "Hello back."
        lumen_buffer = memory.load_thread_chat_history("kevin-home")
        assert lumen_buffer[-1].role == "peer"
        assert lumen_buffer[-1].content == "Hello back."


def test_complete_peer_wake_reply_wakes_sender_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        with patch("light_house.agent.peer_chat_wake.schedule_peer_chat_wake") as sched:
            memory.complete_peer_wake_reply(
                receiver_agent_id="ara",
                sender_agent_id="lumen",
                reply_text="Hello back.",
            )
        sched.assert_called_once()
        assert sched.call_args.kwargs["to_agent_id"] == "lumen"
        assert sched.call_args.kwargs["from_agent_id"] == "ara"


def test_complete_peer_wake_reply_solitude_does_not_wake_sender():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        with patch("light_house.agent.peer_chat_wake.schedule_peer_chat_wake") as sched:
            memory.complete_peer_wake_reply(
                receiver_agent_id="ara",
                sender_agent_id="lumen",
                reply_text=SOLITUDE_DECLINE_TEXT,
            )
        sched.assert_not_called()


def test_complete_peer_wake_reply_wake_sender_false():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        with patch("light_house.agent.peer_chat_wake.schedule_peer_chat_wake") as sched:
            memory.complete_peer_wake_reply(
                receiver_agent_id="ara",
                sender_agent_id="lumen",
                reply_text="Hello back.",
                wake_sender=False,
            )
        sched.assert_not_called()


def test_thread_graph_lock_serializes():
    import threading
    import time

    order: list[str] = []

    def work(name: str, delay: float) -> None:
        with thread_graph_lock("test-thread"):
            order.append(f"{name}-start")
            time.sleep(delay)
            order.append(f"{name}-end")

    t1 = threading.Thread(target=work, args=("a", 0.05))
    t2 = threading.Thread(target=work, args=("b", 0.01))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(order) == 4
    assert order[1].endswith("-end") and order[3].endswith("-end")
    first, second = order[0].split("-")[0], order[2].split("-")[0]
    assert first == second or order.index(f"{first}-end") < order.index(f"{second}-start")


def test_send_peer_message_schedules_wake_when_enabled():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.peer_message.get_settings", return_value=settings):
            with patch(
                "light_house.agent.peer_chat_wake.schedule_peer_chat_wake",
                return_value=True,
            ) as sched:
                result = send_peer_message(
                    from_agent_id="lumen",
                    args={"to_agent_id": "ara", "message": "Ping"},
                )
        assert "gently woken" in result.lower()
        sched.assert_called_once()
        assert sched.call_args.kwargs["to_agent_id"] == "ara"
