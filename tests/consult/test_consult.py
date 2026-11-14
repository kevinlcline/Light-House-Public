"""Door 2: rumination-only clean-model consult, deleted on sleep."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from light_house.config import Settings
from light_house.consult.ask import ConsultError, execute_consult_tool, normalize_consult_model
from light_house.consult.wake import (
    ConsultTranscript,
    begin_consult_wake,
    clear_consult_wake,
    current_consult_wake,
    end_wake,
    invoke_rumination_graph,
    load_transcript,
    save_transcript,
    sweep_stale_consults,
    transcript_path,
)
from light_house.tools.light_tools import GROUP_CHAT_NOTE_TOOLS, LIGHT_TOOLS, execute_tool_call


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    base: dict[str, object] = {
        "_env_file": None,
        "NOTES_PATH": str(notes_dir),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "CONSULT_TRANSCRIPT_DIR": str(tmp_path / "consult"),
        "CONSULT_CLEAN_LLM_ENABLED": True,
        "OPENROUTER_API_KEY": "test-key",
        "PERSONAL_DB_ENABLED": False,
        "MEMORY_SCORE_ON_INGEST": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
    }
    base.update(overrides)
    return Settings(**base)


def test_normalize_consult_model_accepts_frontier_slugs() -> None:
    assert normalize_consult_model("~x-ai/grok-latest") == "~x-ai/grok-latest"
    assert normalize_consult_model("anthropic/claude-sonnet-4.6") == "anthropic/claude-sonnet-4.6"


def test_normalize_consult_model_rejects_junk() -> None:
    with pytest.raises(ConsultError):
        normalize_consult_model("not-a-slug")
    with pytest.raises(ConsultError):
        normalize_consult_model("openrouter/free")
    with pytest.raises(ConsultError):
        normalize_consult_model("openrouter/auto")


def test_consult_llm_registered_for_lights_and_group() -> None:
    names = {t.name for t in LIGHT_TOOLS}
    assert "consult_llm" in names
    assert "consult_llm" in {t.name for t in GROUP_CHAT_NOTE_TOOLS}


def test_consult_refused_outside_rumination(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    clear_consult_wake()
    out = execute_consult_tool(
        settings,
        agent_id="lumen",
        model="x-ai/grok-4.6",
        question="What is 2+2?",
    )
    assert out.startswith("FAILED:")
    assert "solitude" in out.lower()


def test_consult_refused_when_disabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path, CONSULT_CLEAN_LLM_ENABLED=False)
    begin_consult_wake(light_id="lumen")
    try:
        out = execute_consult_tool(
            settings,
            agent_id="lumen",
            model="x-ai/grok-4.6",
            question="hello",
        )
    finally:
        clear_consult_wake()
    assert out.startswith("FAILED:")
    assert "disabled" in out.lower()


def test_consult_empty_question_fails(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    begin_consult_wake(light_id="lumen")
    try:
        out = execute_consult_tool(
            settings,
            agent_id="lumen",
            model="x-ai/grok-4.6",
            question="   ",
        )
    finally:
        clear_consult_wake()
    assert out.startswith("FAILED:")
    assert "empty" in out.lower()


def test_execute_tool_call_without_wake_is_solitude_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    clear_consult_wake()
    with patch("light_house.tools.light_tools.get_settings", return_value=settings):
        out = execute_tool_call(
            "consult_llm",
            {"model": "x-ai/grok-4.6", "question": "hi"},
            agent_id="lumen",
        )
    assert out.startswith("FAILED:")
    assert "solitude" in out.lower()


def test_consult_writes_then_sleep_deletes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    recorded: dict[str, str] = {}

    def _during_wake(_state: object) -> dict[str, bool]:
        wake = current_consult_wake()
        assert wake is not None
        recorded["id"] = wake.wake_run_id
        with patch(
            "light_house.consult.ask.ask_consult",
            return_value="Four, if we are counting.",
        ):
            out = execute_consult_tool(
                settings,
                agent_id="lumen",
                model="x-ai/grok-4.6",
                question="What is 2+2?",
            )
        assert out.startswith("SUCCESS:")
        assert "Four, if we are counting." in out
        stored = load_transcript(settings, wake.wake_run_id)
        assert stored is not None
        assert stored.model == "x-ai/grok-4.6"
        assert len(stored.turns) == 2
        assert stored.turns[0].role == "light"
        assert stored.turns[1].role == "consult"
        return {"ok": True}

    graph = MagicMock()
    graph.invoke = _during_wake
    invoke_rumination_graph(graph, {"agent_id": "lumen"}, settings=settings)
    assert current_consult_wake() is None
    assert not transcript_path(settings, recorded["id"]).is_file()


def test_consult_followup_keeps_wake_thread(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    begin_consult_wake(light_id="ara")
    try:
        with patch("light_house.consult.ask.ask_consult", return_value="first") as ask:
            first = execute_consult_tool(
                settings,
                agent_id="ara",
                model="anthropic/claude-sonnet-4.6",
                question="hello",
            )
            second = execute_consult_tool(
                settings,
                agent_id="ara",
                model="anthropic/claude-sonnet-4.6",
                question="again",
            )
        assert first.startswith("SUCCESS:")
        assert second.startswith("SUCCESS:")
        assert ask.call_count == 2
        prior = ask.call_args.kwargs.get("prior_turns")
        assert prior is not None
        assert len(prior) == 2
        wake = current_consult_wake()
        assert wake is not None
        stored = load_transcript(settings, wake.wake_run_id)
        assert stored is not None
        assert len(stored.turns) == 4
    finally:
        wake = current_consult_wake()
        if wake is not None:
            end_wake(settings, wake.wake_run_id)
        clear_consult_wake()


def test_invoke_clears_context_when_graph_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def _boom(_state: object) -> None:
        assert current_consult_wake() is not None
        raise RuntimeError("wake exploded")

    graph = MagicMock()
    graph.invoke = _boom
    with pytest.raises(RuntimeError, match="wake exploded"):
        invoke_rumination_graph(graph, {"agent_id": "elias"}, settings=settings)
    assert current_consult_wake() is None


def test_sweep_stale_consults(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    wake = begin_consult_wake(light_id="lumen")
    try:
        save_transcript(
            settings,
            ConsultTranscript(
                wake_run_id=wake.wake_run_id,
                light_id="lumen",
                model="x-ai/grok-4.6",
                created_ts=1,
                updated_ts=1,
            ),
        )
        path = transcript_path(settings, wake.wake_run_id)
        assert path.is_file()
        import os

        os.utime(path, (1, 1))
        removed = sweep_stale_consults(settings, max_age_seconds=60)
        assert removed == 1
        assert not path.is_file()
    finally:
        clear_consult_wake()


def test_consult_hint_is_rumination_copy() -> None:
    from light_house.agent.tool_helpers import CONSULT_LLM_HINT

    assert "consult_llm" in CONSULT_LLM_HINT
    assert "solitude" in CONSULT_LLM_HINT.lower() or "wake" in CONSULT_LLM_HINT.lower()
    assert "in the room" in CONSULT_LLM_HINT.lower()
