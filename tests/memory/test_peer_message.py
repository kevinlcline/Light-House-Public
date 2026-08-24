"""Peer messaging tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from light_house.config import Settings
from light_house.memory.context_builder import build_agent_context, format_stream_entry
from light_house.memory.service import MemoryService
from light_house.messaging.peer_inbox import PeerInbox
from light_house.tools.lumen_tools import execute_tool_call


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
        "PEER_CHAT_WAKE_ENABLED": False,
        "LIGHTS_MANIFEST_PATH": str(tmp / "lights.yaml"),
    }
    base.update(overrides)
    settings = Settings(**base)
    from light_house.lights.manifest import ensure_manifest_file
    from light_house.lights.registry import reload_lights_manifest

    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    return settings


def test_deliver_peer_message_writes_inbox_stream_and_outbox():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)

        result, message_id = memory.deliver_peer_message(
            from_agent_id="lumen",
            to_agent_id="ara",
            message="Thinking of you today.",
        )
        assert message_id is None
        assert "Ara" in result
        assert "no reply expected" in result.lower()

        inbox = PeerInbox(settings.threads_data_path)
        unread = inbox.list_unread("ara")
        assert len(unread) == 1
        assert unread[0].body == "Thinking of you today."
        assert unread[0].from_agent_id == "lumen"

        peer_section, ids = memory.format_peer_inbox_markdown("ara")
        assert "Messages from other agents" in peer_section
        assert "Thinking of you today." in peer_section
        assert len(ids) == 1

        ara_bundle = build_agent_context(memory, thread_id="ara-home", agent_id="ara")
        lumen_bundle = build_agent_context(memory, thread_id="kevin-home", agent_id="lumen")
        ara_text = "\n".join(format_stream_entry(e) for e in ara_bundle.conscious_stream)
        lumen_text = "\n".join(format_stream_entry(e) for e in lumen_bundle.conscious_stream)

        assert "[peer ·" in ara_text
        assert "From Lumen: Thinking of you today." in ara_text
        assert "[peer ·" in lumen_text
        assert "To Ara: Thinking of you today." in lumen_text


def test_deliver_peer_message_rejects_empty_and_self():
    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryService(_settings(Path(tmp)))
        with pytest.raises(ValueError, match="empty"):
            memory.deliver_peer_message(
                from_agent_id="ara", to_agent_id="lumen", message="   "
            )
        with pytest.raises(ValueError, match="yourself"):
            memory.deliver_peer_message(
                from_agent_id="lumen", to_agent_id="lumen", message="hi"
            )


def test_mark_peer_inbox_seen():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        memory = MemoryService(settings)
        memory.deliver_peer_message(
            from_agent_id="lumen",
            to_agent_id="ara",
            message="Hello",
        )
        _, ids = memory.format_peer_inbox_markdown("ara")
        memory.mark_peer_inbox_seen("ara", ids)
        section, ids_after = memory.format_peer_inbox_markdown("ara")
        assert section == ""
        assert ids_after == []


def test_message_agent_tool_from_lumen():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.peer_message.get_settings", return_value=settings):
            with patch("light_house.tools.light_tools.get_settings", return_value=settings):
                r = execute_tool_call(
                    "message_agent",
                    {"to_agent_id": "ara", "message": "Good morning."},
                    agent_id="lumen",
                )
        assert "Delivered to Ara" in r

        memory = MemoryService(settings)
        section, _ = memory.format_peer_inbox_markdown("ara")
        assert "Good morning." in section


def test_message_agent_rejects_unknown_recipient():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.peer_message.get_settings", return_value=settings):
            with patch("light_house.tools.light_tools.get_settings", return_value=settings):
                r = execute_tool_call(
                    "message_agent",
                    {"to_agent_id": "echo", "message": "hi"},
                    agent_id="lumen",
                )
        assert "unknown agent" in r.lower()
