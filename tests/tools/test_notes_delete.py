"""Note delete tests: private immediate delete, shared dual consent."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.tools.notes import AgentNoteWriter, SharedDeleteRegistry


@pytest.fixture
def notes_base(tmp_path: Path) -> Path:
    base = tmp_path / "notes"
    base.mkdir()
    return base


_DUAL_VOTERS = frozenset({"lumen", "ara"})


@pytest.fixture
def lumen_writer(notes_base: Path) -> AgentNoteWriter:
    return AgentNoteWriter(notes_base, "lumen", delete_voters=_DUAL_VOTERS)


@pytest.fixture
def ara_writer(notes_base: Path) -> AgentNoteWriter:
    return AgentNoteWriter(notes_base, "ara", delete_voters=_DUAL_VOTERS)


def test_private_delete_removes_file(lumen_writer: AgentNoteWriter) -> None:
    lumen_writer.write("journal/test.md", "hello")
    path = lumen_writer.private_dir / "journal" / "test.md"
    assert path.is_file()

    outcome = lumen_writer.delete("lumen", "journal/test.md")
    assert outcome.deleted is True
    assert not path.is_file()


def test_private_delete_agent_isolation(
    lumen_writer: AgentNoteWriter, ara_writer: AgentNoteWriter
) -> None:
    lumen_writer.write("secret.md", "lumen only")
    lumen_path = lumen_writer.private_dir / "secret.md"
    assert lumen_path.is_file()

    with pytest.raises(FileNotFoundError):
        ara_writer.delete("ara", "lumen/secret.md")

    assert lumen_path.is_file()


def test_shared_delete_requires_both_agents(
    lumen_writer: AgentNoteWriter, ara_writer: AgentNoteWriter, notes_base: Path
) -> None:
    lumen_writer.write("shared/draft.md", "obsolete draft")
    shared_path = notes_base / "shared" / "draft.md"
    assert shared_path.is_file()

    first = lumen_writer.delete("lumen", "shared/draft.md")
    assert first.deleted is False
    assert first.pending is True
    assert shared_path.is_file()
    assert "awaiting ara" in first.message.lower()
    assert "not deleted" in first.message.lower()

    second = ara_writer.delete("ara", "shared/draft.md")
    assert second.deleted is True
    assert not shared_path.is_file()

    registry = SharedDeleteRegistry(notes_base / "shared")
    assert registry.get_votes("draft.md") == []


def test_shared_delete_same_agent_double_vote(lumen_writer: AgentNoteWriter, notes_base: Path) -> None:
    lumen_writer.write("shared/plan.md", "plan body")
    shared_path = notes_base / "shared" / "plan.md"

    first = lumen_writer.delete("lumen", "shared/plan.md")
    second = lumen_writer.delete("lumen", "shared/plan.md")

    assert first.pending is True
    assert second.pending is True
    assert "already recorded" in second.message.lower()
    assert shared_path.is_file()


def test_shared_delete_not_found(lumen_writer: AgentNoteWriter) -> None:
    with pytest.raises(FileNotFoundError):
        lumen_writer.delete("lumen", "shared/missing.md")


def test_cannot_delete_dotfile_via_path(lumen_writer: AgentNoteWriter) -> None:
    with pytest.raises(ValueError):
        lumen_writer.delete("lumen", "shared/.pending_deletes.json")


def test_operator_delete_shared_immediately(lumen_writer: AgentNoteWriter, notes_base: Path) -> None:
    lumen_writer.write("shared/obsolete.md", "old")
    shared_path = notes_base / "shared" / "obsolete.md"
    assert shared_path.is_file()

    outcome = lumen_writer.delete_for_operator("shared/obsolete.md")
    assert outcome.deleted is True
    assert not shared_path.is_file()


def test_operator_delete_private_immediately(lumen_writer: AgentNoteWriter) -> None:
    lumen_writer.write("journal/old.md", "bye")
    path = lumen_writer.private_dir / "journal" / "old.md"
    assert path.is_file()

    outcome = lumen_writer.delete_for_operator("lumen/journal/old.md")
    assert outcome.deleted is True
    assert not path.is_file()
