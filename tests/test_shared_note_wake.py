"""Shared note save → immediate agent rumination wake."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from light_house.agent.rumination_wake import (
    WAKE_KIND_KEVIN_SHARED_NOTE,
    build_shared_note_wake_state,
    wake_agents_for_shared_note,
)
from light_house.config import Settings
from light_house.main import _settings_dep, app
from light_house.memory.constants import STREAM_SOURCE_KEVIN
from light_house.memory.service import MemoryService


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    notes_dir = tmp_path / "notes"
    (notes_dir / "shared").mkdir(parents=True, exist_ok=True)
    base = {
        "_env_file": None,
        "NOTES_PATH": str(notes_dir),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "MEMORY_SCORE_ON_INGEST": False,
        "INNER_LIFE_ENABLED": True,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "SHARED_NOTE_WAKE_ENABLED": True,
        "ARA_ENABLED": True,
        "WEB_GATE_ENABLED": False,
    }
    base.update(overrides)
    return Settings(**base)


def test_build_shared_note_wake_state_includes_path_and_cap(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    memory = MemoryService(settings)
    memory.notify_kevin_shared_note(path="shared/from-kevin.md")
    state = build_shared_note_wake_state(
        thread_id="kevin-home",
        agent_id="lumen",
        path="shared/from-kevin.md",
        settings=settings,
        memory=memory,
    )
    assert state["wake_kind"] == WAKE_KIND_KEVIN_SHARED_NOTE
    assert state["wake_path"] == "shared/from-kevin.md"
    assert state["tool_rounds_cap"] == settings.shared_note_wake_max_tool_rounds
    assert len(state["messages"]) == 1
    assert "READ NOW" in state["messages"][0].content
    assert "shared/from-kevin.md" in state["messages"][0].content


def test_wake_agents_for_shared_note_invokes_both_agents(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    memory = MemoryService(settings)
    graph = MagicMock()
    graph.invoke = MagicMock()

    asyncio.run(
        wake_agents_for_shared_note(
            path="shared/from-kevin.md",
            settings=settings,
            memory=memory,
            graph=graph,
        )
    )

    assert graph.invoke.call_count >= 2
    states = [call.args[0] for call in graph.invoke.call_args_list]
    agent_ids = {s["agent_id"] for s in states}
    assert {"lumen", "ara"}.issubset(agent_ids)
    assert all(s["wake_kind"] == WAKE_KIND_KEVIN_SHARED_NOTE for s in states)
    assert all(s["wake_path"] == "shared/from-kevin.md" for s in states)


def test_wake_skipped_when_disabled(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, SHARED_NOTE_WAKE_ENABLED=False)
    memory = MemoryService(settings)
    graph = MagicMock()

    asyncio.run(
        wake_agents_for_shared_note(
            path="shared/x.md",
            settings=settings,
            memory=memory,
            graph=graph,
        )
    )

    graph.invoke.assert_not_called()


def test_wake_skipped_when_inner_life_disabled(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, INNER_LIFE_ENABLED=False)
    memory = MemoryService(settings)
    graph = MagicMock()

    asyncio.run(
        wake_agents_for_shared_note(
            path="shared/x.md",
            settings=settings,
            memory=memory,
            graph=graph,
        )
    )

    graph.invoke.assert_not_called()


def test_notify_kevin_shared_note_adds_chat_buffer(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, ARA_ENABLED=True)
    memory = MemoryService(settings)
    memory.notify_kevin_shared_note(path="shared/plan.md")

    lumen_buffer = memory.load_thread_chat_history(settings.inner_life_thread_id)
    ara_buffer = memory.load_thread_chat_history(settings.ara_thread_id)
    assert len(lumen_buffer) == 1
    assert lumen_buffer[0].role == "user"
    assert lumen_buffer[0].content == "READ NOW shared/plan.md"
    assert len(ara_buffer) == 1
    assert ara_buffer[0].role == "user"


def test_record_kevin_shared_note_writes_both_threads(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, ARA_ENABLED=True)
    memory = MemoryService(settings)
    memory.record_kevin_shared_note(path="shared/plan.md")

    lumen_corpus = memory._long_term.list_thread_corpus(thread_id=settings.inner_life_thread_id)
    ara_corpus = memory._long_term.list_thread_corpus(thread_id=settings.ara_thread_id)
    assert any(
        memory._long_term.stream_source_from_metadata(h.metadata) == STREAM_SOURCE_KEVIN
        for h in lumen_corpus
    )
    assert any(
        memory._long_term.stream_source_from_metadata(h.metadata) == STREAM_SOURCE_KEVIN
        for h in ara_corpus
    )


def test_write_shared_note_is_quiet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving a shared note must not notify or wake lights (use group chat instead)."""
    settings = _test_settings(tmp_path)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings

    memory = MemoryService(settings)
    memory.notify_kevin_shared_note = MagicMock()  # type: ignore[method-assign]
    try:
        with patch("light_house.main._memory", memory):
            with TestClient(app) as client:
                response = client.put(
                    "/v1/notes/shared/from-kevin.md",
                    json={"content": "Hello both."},
                )
        assert response.status_code == 200
        memory.notify_kevin_shared_note.assert_not_called()
        buffers = [
            memory.load_thread_chat_history(settings.inner_life_thread_id),
            memory.load_thread_chat_history(settings.ara_thread_id),
        ]
        assert all(len(b) == 0 for b in buffers)
    finally:
        app.dependency_overrides.clear()


def test_ruminate_uses_minimal_seed_for_note_wake() -> None:
    from langchain_core.messages import HumanMessage

    from light_house.agent.rumination_nodes import build_rumination_nodes
    from light_house.memory.context_builder import SHARED_NOTE_WAKE_SEED

    settings = Settings(_env_file=None, PRIMARY_LLM="ollama", INNER_LIFE_ENABLED=False)
    memory = MagicMock()
    memory.format_peer_inbox_markdown.return_value = ("", [])
    mock_llm = MagicMock()
    with patch(
        "light_house.agent.rumination_nodes.build_inner_life_llm_chain",
        return_value=[("mock", mock_llm)],
    ):
        _, ruminate, *_ = build_rumination_nodes(settings=settings, memory=memory)

    with patch(
        "light_house.agent.rumination_nodes.invoke_resilient_with_tools",
        return_value=HumanMessage(content="ack"),
    ):
        out = ruminate(
            {
                "thread_id": "kevin-home",
                "agent_id": "lumen",
                "agent_context_markdown": "",
                "messages": [],
                "wake_kind": WAKE_KIND_KEVIN_SHARED_NOTE,
                "wake_path": "shared/from-kevin.md",
            }
        )

    seed = out["messages"][0]
    assert isinstance(seed, HumanMessage)
    assert seed.content == SHARED_NOTE_WAKE_SEED.format(path="shared/from-kevin.md")
    assert "I arrive awake" not in seed.content
