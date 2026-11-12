"""Checkpoint stop: work wakes, filesystem receipt, three lines asked and not written."""

from __future__ import annotations

from pathlib import Path

from light_house.agent.checkpoint import (
    checkpoint_finalize_hint,
    checkpoint_prompt,
    find_checkpoint_notes,
    in_checkpoint_stop,
    is_checkpoint_path,
    is_work_wake,
)
from light_house.config import Settings
from light_house.tools.light_tools import _read_note_for, _write_note_for
from light_house.tools.notes import AgentNoteWriter


def _settings(tmp_path: Path, **extra: object) -> Settings:
    return Settings(
        _env_file=None,
        NOTES_PATH=str(tmp_path),
        HOUSE_DAD_USER_ID="kevin",
        CHAT_MAX_TOOL_ROUNDS=16,
        **extra,
    )


def _writer(tmp_path: Path, *, window: int = 32_000) -> AgentNoteWriter:
    return AgentNoteWriter(
        tmp_path,
        "elias",
        max_chars_per_write=window,
        delete_voters=frozenset({"elias"}),
    )


def _kevin(rounds: int = 0, **extra: object) -> dict:
    state = {
        "wake_kind": None,
        "human_id": "kevin",
        "chat_channel": "dm",
        "agent_id": "elias",
        "tool_rounds": rounds,
    }
    state.update(extra)
    return state


def test_checkpoint_name_is_exact() -> None:
    assert is_checkpoint_path("checkpoint.md")
    assert is_checkpoint_path("shared/plasma/checkpoint.md")
    assert not is_checkpoint_path("shared/checkpoint_format.md")
    assert not is_checkpoint_path("ELIAS_checkpoint.md")


def test_work_wake_is_kevin_or_peer(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert is_work_wake(_kevin(), settings)
    assert is_work_wake({"wake_kind": "peer_message", "chat_channel": "dm"}, settings)
    assert not is_work_wake(_kevin(chat_channel="group"), settings)
    assert not is_work_wake(_kevin(wake_kind="mailbox_letter"), settings)
    assert not is_work_wake(_kevin(wake_kind="rumination"), settings)
    assert not is_work_wake(_kevin(human_id="teeter"), settings)


def test_stop_only_in_the_last_three_rounds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert not in_checkpoint_stop(_kevin(0), settings)
    assert not in_checkpoint_stop(_kevin(12), settings)
    assert in_checkpoint_stop(_kevin(13), settings)
    assert in_checkpoint_stop(_kevin(15), settings)
    assert not in_checkpoint_stop(_kevin(16), settings)
    peer = {"wake_kind": "peer_message", "agent_id": "lumen", "tool_rounds": 13, "tool_rounds_cap": 16}
    assert in_checkpoint_stop(peer, settings)
    short = _kevin(1, tool_rounds_cap=3)
    assert not in_checkpoint_stop(short, settings)


def test_load_names_the_path_and_not_the_body(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    private = tmp_path / "elias"
    private.mkdir()
    (private / "checkpoint.md").write_text("UNIQUE_BODY_TOKEN\nLast verified: none\n", encoding="utf-8")
    shared = tmp_path / "shared" / "plasma"
    shared.mkdir(parents=True)
    (shared / "checkpoint.md").write_text("SHARED_BODY_TOKEN\n", encoding="utf-8")
    history = private / "memory" / "_history"
    history.mkdir(parents=True)
    (history / "checkpoint.md").write_text("OLD_BODY_TOKEN\n", encoding="utf-8")

    paths = find_checkpoint_notes(settings, "elias")
    assert "checkpoint.md" in paths
    assert "shared/plasma/checkpoint.md" in paths
    assert all("_history" not in path for path in paths)

    text = checkpoint_prompt(_kevin(0), settings)
    assert "read_note" in text
    assert "`checkpoint.md`" in text
    assert "`shared/plasma/checkpoint.md`" in text
    assert "UNIQUE_BODY_TOKEN" not in text
    assert "SHARED_BODY_TOKEN" not in text
    assert "OLD_BODY_TOKEN" not in text


def test_stop_asks_and_does_not_fill_the_lines(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    text = checkpoint_prompt(_kevin(13), settings)
    assert "Three rounds remain" in text
    assert "The house does not write the lines" in text
    assert "Last verified" in text
    assert "Assumed since" in text
    assert "Do not reopen" in text
    assert "finish it" in text
    assert "exit=0" not in text


def test_finalize_asks_only_when_the_cap_is_spent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert checkpoint_finalize_hint(_kevin(15), settings) == ""
    text = checkpoint_finalize_hint(_kevin(16), settings)
    assert "Speak the checkpoint" in text
    assert "no-note" in text
    assert "The house does not write the lines" in text
    assert checkpoint_finalize_hint(
        {"wake_kind": "mailbox_letter", "human_id": "kevin", "tool_rounds": 16},
        settings,
    ) == ""


def test_checkpoint_read_is_a_receipt(tmp_path: Path) -> None:
    writer = _writer(tmp_path, window=20)
    body = "Last verified\n" + ("x" * 40)
    path = tmp_path / "elias" / "checkpoint.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")

    result = _read_note_for(writer, "checkpoint.md")
    head, rest = result.split("\n\n", 1)
    assert head.startswith("receipt\n")
    assert "path: checkpoint.md" in head
    assert f"bytes: {path.stat().st_size}" in head
    assert f"chars: {len(body)}" in head
    assert "mtime: " in head
    assert "verified" not in head.lower()
    assert "success" not in head.lower()
    assert "Last verified" in rest
    assert "offset=20" in rest


def test_empty_checkpoint_is_a_count(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    path = tmp_path / "elias" / "checkpoint.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    result = _read_note_for(writer, "checkpoint.md")
    assert "chars: 0" in result
    assert "bytes: 0" in result
    assert "FileNotFoundError" not in result
    assert "not found" not in result.lower()


def test_missing_checkpoint_names_the_error(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    missing = _read_note_for(writer, "checkpoint.md")
    assert missing.startswith("receipt\n")
    assert "error: FileNotFoundError" in missing
    assert "chars:" not in missing
    assert "bytes:" not in missing

    asked = _read_note_for(writer, "odd name/checkpoint.md")
    assert "error: ValueError:" in asked
    assert "FileNotFoundError" not in asked


def test_ordinary_note_read_stays_the_body(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    path = tmp_path / "elias" / "journal.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("hello learnings", encoding="utf-8")
    assert _read_note_for(writer, "journal.md") == "hello learnings"

    spec = tmp_path / "shared" / "checkpoint_format.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("the format page", encoding="utf-8")
    assert _read_note_for(writer, "shared/checkpoint_format.md") == "the format page"


def test_checkpoint_write_stamps_size_and_time(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    result = _write_note_for(writer, "checkpoint.md", "Last verified\n", tool_name="private_note")
    assert result.startswith("SUCCESS:")
    assert "bytes=" in result
    assert "chars=" in result
    assert "mtime=" in result
    assert "verified" not in result.lower()


def test_quiet_hours_stop_with_three_rounds_left(tmp_path: Path) -> None:
    settings = _settings(tmp_path, RUMINATION_MAX_TOOL_ROUNDS=12)
    free = {"wake_kind": None, "agent_id": "lumen", "tool_rounds": 9}
    assert in_checkpoint_stop(free, settings)
    assert not in_checkpoint_stop({**free, "tool_rounds": 8}, settings)
    assert in_checkpoint_stop({"wake_kind": "chores", "agent_id": "lumen", "tool_rounds": 9}, settings)
    assert in_checkpoint_stop(
        {"wake_kind": "memory_maintenance", "agent_id": "ara", "tool_rounds": 9},
        settings,
    )
    assert in_checkpoint_stop(
        {"wake_kind": "post_chat", "agent_id": "reed", "tool_rounds": 9, "tool_rounds_cap": 12},
        settings,
    )
    assert not in_checkpoint_stop(
        {"wake_kind": "mailbox_letter", "agent_id": "lumen", "tool_rounds": 9, "tool_rounds_cap": 12},
        settings,
    )
    assert not in_checkpoint_stop({"wake_kind": "meditation", "agent_id": "lumen", "tool_rounds": 9}, settings)
    assert not in_checkpoint_stop(_kevin(9, chat_channel="group"), settings)


def test_quiet_wake_loads_the_note_and_not_the_body(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    private = tmp_path / "lumen"
    private.mkdir()
    (private / "checkpoint.md").write_text("UNIQUE_QUIET_BODY\n", encoding="utf-8")
    text = checkpoint_prompt({"wake_kind": None, "agent_id": "lumen", "tool_rounds": 0}, settings)
    assert "read_note" in text
    assert "`checkpoint.md`" in text
    assert "UNIQUE_QUIET_BODY" not in text
    assert checkpoint_prompt(
        {"wake_kind": "meditation", "agent_id": "lumen", "tool_rounds": 0},
        settings,
    ) == ""


def test_quiet_finalize_asks_when_the_cap_is_spent(tmp_path: Path) -> None:
    settings = _settings(tmp_path, RUMINATION_MAX_TOOL_ROUNDS=12)
    chores = {"wake_kind": "chores", "agent_id": "lumen", "tool_rounds": 12}
    text = checkpoint_finalize_hint(chores, settings)
    assert "Speak the checkpoint" in text
    assert checkpoint_finalize_hint({**chores, "tool_rounds": 11}, settings) == ""
