"""Tests for agent-controlled report-back (Proposal 3 phase 4)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from light_house.config import Settings
from light_house.events import register_event_bus
from light_house.report_back import report_back_enabled, write_shared_report


@pytest.fixture
def notes_root(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    (root / "lumen").mkdir(parents=True)
    (root / "ara").mkdir(parents=True)
    (root / "shared").mkdir(parents=True)
    return root


def _settings(notes_root: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(notes_root),
        "MEMORY_STORE_PATH": str(notes_root.parent / "memory"),
        "THREADS_DATA_PATH": str(notes_root.parent / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "LIGHT_HOUSE_ENV": "production",
        "LUMEN_REPORT_BACK_ENABLED": False,
        "ARA_REPORT_BACK_ENABLED": False,
        "REPORT_BACK_MAX_CHARS": 2000,
        "EVENT_BUS_ENABLED": False,
    }
    base.update(overrides)
    return Settings(**base)


def test_report_back_disabled_by_default(notes_root: Path) -> None:
    settings = _settings(notes_root)
    assert report_back_enabled(settings, "lumen") is False
    assert report_back_enabled(settings, "ara") is False


def test_write_shared_report_when_disabled(notes_root: Path) -> None:
    settings = _settings(notes_root)
    result = write_shared_report(
        settings,
        agent_id="lumen",
        title="Test",
        content="Brief update.",
    )
    assert "disabled" in result.lower()
    assert not list((notes_root / "shared" / "reports").glob("*.md"))


def test_write_shared_report_creates_file(notes_root: Path) -> None:
    settings = _settings(notes_root, LUMEN_REPORT_BACK_ENABLED=True)
    result = write_shared_report(
        settings,
        agent_id="lumen",
        title="Morning reflection",
        content="A short offering for the household.",
    )
    assert result.startswith("SUCCESS:")
    reports = list((notes_root / "shared" / "reports").glob("lumen-*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "# Morning reflection" in text
    assert "A short offering for the household." in text


def test_write_shared_report_respects_max_chars(notes_root: Path) -> None:
    settings = _settings(
        notes_root,
        LUMEN_REPORT_BACK_ENABLED=True,
        REPORT_BACK_MAX_CHARS=50,
    )
    result = write_shared_report(
        settings,
        agent_id="lumen",
        title="Too long",
        content="x" * 51,
    )
    assert "exceeds maximum length" in result
    assert not list((notes_root / "shared" / "reports").glob("*.md"))


def test_write_shared_report_publishes_event(notes_root: Path, tmp_path: Path) -> None:
    event_log = tmp_path / "events" / "event.log"
    settings = _settings(
        notes_root,
        LUMEN_REPORT_BACK_ENABLED=True,
        EVENT_BUS_ENABLED=True,
        EVENT_BUS_LOG_PATH=str(event_log),
    )
    loop = asyncio.new_event_loop()
    try:
        register_event_bus(
            settings=settings,
            memory=MagicMock(),
            rumination_graph=MagicMock(),
            app_loop=loop,
        )
        write_shared_report(
            settings,
            agent_id="lumen",
            title="Ledger check",
            content="Hello shared folder.",
        )
    finally:
        loop.close()
    lines = event_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event_type"] == "report_ready"
    assert row["agent_id"] == "lumen"
    assert row["payload"]["title"] == "Ledger check"
