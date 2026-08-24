"""read_note: full reads and large-file windows."""

from __future__ import annotations

from pathlib import Path

from light_house.tools.notes import AgentNoteWriter, NoteWriter


def test_note_writer_read_returns_full_small_file(tmp_path: Path) -> None:
    writer = NoteWriter(tmp_path, max_chars_per_write=100)
    (tmp_path / "short.md").write_text("hello learnings", encoding="utf-8")
    assert writer.read("short.md") == "hello learnings"


def test_note_writer_read_windows_large_file(tmp_path: Path) -> None:
    writer = NoteWriter(tmp_path, max_chars_per_write=20)
    body = "A" * 55
    (tmp_path / "big.md").write_text(body, encoding="utf-8")

    first = writer.read("big.md")
    assert first.startswith("[Note excerpt characters 0-20 of 55")
    assert "AAAA" in first
    assert "offset=20" in first

    second = writer.read("big.md", offset=20)
    assert second.startswith("[Note excerpt characters 20-40 of 55")
    assert "offset=40" in second

    third = writer.read("big.md", offset=40)
    assert third.startswith("[Note excerpt characters 40-55 of 55")
    assert third.endswith("A" * 15)


def test_agent_note_writer_read_learnings_over_old_32k(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    writer = AgentNoteWriter(notes, "ara", max_chars_per_write=128_000, delete_voters=frozenset({"ara"}))
    learnings = notes / "ara" / "memory"
    learnings.mkdir(parents=True)
    body = "# Learnings\n" + ("x" * 33_000)
    (learnings / "learnings.md").write_text(body, encoding="utf-8")

    text = writer.read("memory/learnings.md")
    assert text.startswith("# Learnings")
    assert len(text) == len(body)
    assert "[Note excerpt" not in text
