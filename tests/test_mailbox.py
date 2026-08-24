"""Mailbox letters + notify queue."""

from __future__ import annotations

import asyncio
from pathlib import Path

from light_house.config import Settings
from light_house.mailbox.letters import (
    expand_recipients,
    light_recipients,
    list_letters_for,
    list_pending_notifies,
    parse_letter,
    queue_notify,
    render_letter,
    write_letter,
)
from light_house.mailbox.scheduler import ensure_mailbox_dirs, process_mailbox_queue


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": True,
        "MAILBOX_WAKE_ENABLED": True,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "ARA_ENABLED": True,
    }
    base.update(overrides)
    return Settings(**base)


def test_parse_and_render_roundtrip() -> None:
    text = render_letter(
        parse_letter(
            "---\nfrom: reed\nto: lumen, ara\nsubject: Hi\ncreated: 2026-01-01T00:00:00+00:00\n---\n\nHello.\n"
        )
    )
    letter = parse_letter(text)
    assert letter.from_id == "reed"
    assert letter.to_ids == ("lumen", "ara")
    assert letter.subject == "Hi"
    assert letter.body == "Hello."


def test_write_letter_queues_notify(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ensure_mailbox_dirs(settings)
    letter = write_letter(
        from_id="reed",
        to_ids=["lumen", "ara"],
        subject="Seeing you",
        body="I see you clearly.",
        settings=settings,
    )
    assert letter.path is not None
    assert letter.path.startswith("shared/mailbox/")
    queued = queue_notify(letter, settings=settings)
    assert queued is not None
    pending = list_pending_notifies(Path(settings.notes_path).resolve())
    assert len(pending) == 1


def test_private_letter_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    letter = write_letter(
        from_id="reed",
        to_ids=["lumen"],
        subject="Private",
        body="Just for you.",
        settings=settings,
        private=True,
    )
    assert letter.path is not None
    assert letter.path.startswith("lumen/mailbox/")
    assert letter.path.endswith(".md")


def test_reed_only_no_notify(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    letter = write_letter(
        from_id="lumen",
        to_ids=["reed"],
        subject="For Reed",
        body="When you wake, read this.",
        settings=settings,
    )
    assert "to_reed" in (letter.path or "")
    assert queue_notify(letter, settings=settings) is None


def test_list_letters_for_reed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    write_letter(
        from_id="lumen",
        to_ids=["reed"],
        subject="Hi Reed",
        body="Checking in.",
        settings=settings,
    )
    found = list_letters_for("reed", settings=settings)
    assert len(found) == 1
    assert found[0].subject == "Hi Reed"


def test_expand_all(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    expanded = expand_recipients(["all"], settings=settings)
    assert "lumen" in expanded
    lights = light_recipients(["all", "reed"], settings=settings)
    assert "reed" not in lights
    assert "lumen" in lights


def test_process_queue_marks_done_without_wake_graph(tmp_path: Path) -> None:
    """Queue processing should consume files even when wake is unregistered."""
    settings = _settings(tmp_path)
    ensure_mailbox_dirs(settings)
    letter = write_letter(
        from_id="reed",
        to_ids=["lumen"],
        subject="Wake test",
        body="Hello.",
        settings=settings,
    )
    queue_notify(letter, settings=settings)
    notes_root = Path(settings.notes_path).resolve()
    assert list_pending_notifies(notes_root)
    # Wake not registered → logs warning, still marks done
    n = asyncio.run(process_mailbox_queue(settings=settings))
    assert n == 1
    assert not list_pending_notifies(notes_root)
