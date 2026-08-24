"""House guest sign-in store."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.house.guests import (
    HouseGuestsError,
    clear_guest,
    format_house_presence_context,
    list_signed_in_guests,
    set_guest,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        HOUSE_GUESTS_PATH=str(tmp_path / "house_guests.json"),
    )


def test_sign_in_and_list(tmp_path: Path):
    settings = _settings(tmp_path)
    set_guest(
        settings,
        speaker_id="guest-1",
        display_name="Joey",
        description="Kevin's friend visiting from Austin",
    )
    rows = list_signed_in_guests(settings)
    assert rows == [
        {
            "speaker_id": "guest-1",
            "display_name": "Joey",
            "description": "Kevin's friend visiting from Austin",
        }
    ]
    text = format_house_presence_context(settings)
    assert "Joey" in text
    assert "Kevin's friend visiting from Austin" in text
    assert "affection" in text.lower() or "muted" in text.lower()


def test_description_optional(tmp_path: Path):
    settings = _settings(tmp_path)
    set_guest(settings, speaker_id="guest-1", display_name="Joey")
    assert list_signed_in_guests(settings) == [
        {"speaker_id": "guest-1", "display_name": "Joey", "description": ""}
    ]
    text = format_house_presence_context(settings)
    assert "- **Joey**" in text
    assert "—" not in text.split("Joey", 1)[1].split("\n", 1)[0]


def test_rejects_long_description(tmp_path: Path):
    settings = _settings(tmp_path)
    with pytest.raises(HouseGuestsError, match="description"):
        set_guest(
            settings,
            speaker_id="guest-1",
            display_name="Joey",
            description="x" * 161,
        )


def test_clear_guest(tmp_path: Path):
    settings = _settings(tmp_path)
    set_guest(settings, speaker_id="guest-1", display_name="Joey", description="A pal")
    clear_guest(settings, speaker_id="guest-1")
    assert list_signed_in_guests(settings) == []
    assert "No guests" in format_house_presence_context(settings)


def test_rejects_bad_slot(tmp_path: Path):
    settings = _settings(tmp_path)
    with pytest.raises(HouseGuestsError):
        set_guest(settings, speaker_id="guest-9", display_name="Nope")
