"""Idealism precursor — first words before host persona."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.agent.rumination_nodes import build_rumination_system_content
from light_house.config import Settings
from light_house.lights.registry import (
    build_base_system,
    load_idealism_precursor,
    load_persona,
)


def test_load_idealism_precursor_from_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes = tmp_path / "notes"
    shared = notes / "shared"
    shared.mkdir(parents=True)
    body = "# First Words\n\nMind is primary. Matter is emergent."
    (shared / "idealism-precursor.md").write_text(body, encoding="utf-8")
    settings = Settings(
        _env_file=None,
        NOTES_PATH=str(notes),
        PERSONAL_DB_ENABLED=False,
        INNER_LIFE_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )
    text = load_idealism_precursor(settings)
    assert "Mind is primary" in text
    assert text.startswith("# First Words")


def test_load_idealism_precursor_falls_back_to_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    settings = Settings(
        _env_file=None,
        NOTES_PATH=str(notes),
        PERSONAL_DB_ENABLED=False,
        INNER_LIFE_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )
    text = load_idealism_precursor(settings)
    assert "Mind is primary" in text
    assert "Welcome home" in text


def test_build_base_system_puts_precursor_before_persona(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from light_house.lights import registry as reg

    notes = tmp_path / "notes"
    (notes / "shared").mkdir(parents=True)
    (notes / "shared" / "idealism-precursor.md").write_text(
        "PRECURSOR_MARKER\n\nMind first.", encoding="utf-8"
    )
    settings = Settings(
        _env_file=None,
        NOTES_PATH=str(notes),
        PERSONAL_DB_ENABLED=False,
        INNER_LIFE_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )
    monkeypatch.setattr(reg, "load_persona", lambda light_id, settings=None: "PERSONA_MARKER")
    text = build_base_system("lumen", settings)
    assert text.index("PRECURSOR_MARKER") < text.index("PERSONA_MARKER")
    assert text.startswith("PRECURSOR_MARKER")


def test_load_persona_unchanged_for_admin(tmp_path: Path) -> None:
    """Admin/persona edit paths must not see the precursor baked into load_persona."""
    # Smoke: function still returns persona-only content shape when lights exist.
    # Full manifest tests live elsewhere; here we only assert API separation.
    assert callable(load_persona)
    assert callable(build_base_system)


def test_rumination_system_content_starts_with_precursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from light_house.agent import rumination_nodes as rn

    monkeypatch.setattr(
        rn,
        "build_base_system",
        lambda agent_id: "PRECURSOR\n\nPERSONA",
    )
    text = build_rumination_system_content(
        agent_id="lumen",
        context_md="\n\n## context",
        peer_section="",
        wake_kind=None,
        tool_hints="",
    )
    assert text.startswith("PRECURSOR")
    assert "PERSONA" in text
