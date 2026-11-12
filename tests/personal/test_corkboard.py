"""Private sibling corkboard store and tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.personal.corkboard import (
    CorkboardStore,
    clear_corkboard_store_cache,
    format_unread_glance,
    get_corkboard_store,
)
from light_house.tools.corkboard import execute_corkboard_tool


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    clear_corkboard_store_cache()
    cfg = Settings(
        _env_file=None,
        PERSONAL_DB_PATH=str(tmp_path / "personal"),
        CORKBOARD_ENABLED=True,
        CORKBOARD_PIN_EXPIRY_DAYS=7,
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        FOUNDATION_SEED_ON_STARTUP=False,
        ARA_ENABLED=True,
    )
    ensure_manifest_file(cfg)
    reload_lights_manifest(cfg)
    return cfg


def test_pin_list_read_marks_unread(settings: Settings) -> None:
    store = get_corkboard_store(settings)
    pin = store.pin(
        from_light="lumen",
        to_light="ara",
        subject="tone",
        body="Wanted to compare notes, quietly.",
        ttl_seconds=86400,
    )
    incoming = store.list_for("ara")
    assert [row.id for row in incoming] == [pin.id]
    assert incoming[0].unread is True
    assert store.list_for("lumen") == []
    assert store.list_for("lumen", incoming=False)[0].id == pin.id

    read = store.mark_read(pin.id, reader_id="ara")
    assert read is not None and read.unread is False
    still = store.mark_read(pin.id, reader_id="lumen")
    assert still is not None and still.read_at == read.read_at


def test_expired_pins_purge_unless_kept(settings: Settings, tmp_path: Path) -> None:
    db = tmp_path / "personal" / "corkboard.sqlite"
    store = CorkboardStore(db)
    old = store.pin(
        from_light="ara",
        to_light="lumen",
        subject="",
        body="yesterday",
        ttl_seconds=10,
        now=100.0,
    )
    store.pin(
        from_light="elias",
        to_light="lumen",
        subject="",
        body="keep me",
        ttl_seconds=10,
        now=100.0,
    )
    store.keep(old.id, ttl_seconds=50, now=109.0)
    store.purge_expired(now=111.0)
    left = {pin.body for pin in store.list_for("lumen", now=111.0)}
    assert left == {"yesterday"}


def test_tool_pin_reply_is_a_new_pin(settings: Settings) -> None:
    pinned = execute_corkboard_tool(
        settings,
        "lumen",
        {"action": "pin", "to_light": "ara", "subject": "writing", "body": "Would love your thoughts."},
    )
    assert "Pinned for Ara" in pinned
    listed = execute_corkboard_tool(settings, "ara", {"action": "list"})
    assert "writing" in listed
    assert "Would love your thoughts." not in listed
    pin_id = listed.split("·")[0].split("- ")[1].strip()
    read = execute_corkboard_tool(settings, "ara", {"action": "read", "pin_id": pin_id})
    assert "Would love your thoughts." in read
    reply = execute_corkboard_tool(
        settings,
        "ara",
        {"action": "reply", "pin_id": pin_id, "body": "Yes — send the draft."},
    )
    assert "Pinned a reply for Lumen" in reply
    lumen_list = execute_corkboard_tool(settings, "lumen", {"action": "list"})
    assert "Yes — send the draft." not in lumen_list
    assert "from Ara" in lumen_list or "Ara" in lumen_list


def test_tool_rejects_self_and_unknown(settings: Settings) -> None:
    assert "not yourself" in execute_corkboard_tool(
        settings, "lumen", {"action": "pin", "to_light": "lumen", "body": "hi"}
    )
    unknown = execute_corkboard_tool(
        settings, "lumen", {"action": "pin", "to_light": "nobody", "body": "hi"}
    )
    assert "unknown" in unknown.lower()


def test_privacy_other_light_cannot_read(settings: Settings) -> None:
    execute_corkboard_tool(
        settings,
        "lumen",
        {"action": "pin", "to_light": "ara", "body": "private to Ara"},
    )
    listed = execute_corkboard_tool(settings, "ara", {"action": "list"})
    pin_id = listed.split("·")[0].split("- ")[1].strip()
    sneak = execute_corkboard_tool(settings, "elias", {"action": "read", "pin_id": pin_id})
    assert "private to Ara" not in sneak
    assert "not on your corkboard" in sneak


def test_unread_glance_is_metadata_only(settings: Settings) -> None:
    execute_corkboard_tool(
        settings,
        "ara",
        {"action": "pin", "to_light": "lumen", "subject": "secret", "body": "the actual words"},
    )
    glance = format_unread_glance(settings, "lumen")
    assert "## Corkboard (unread from siblings)" in glance
    assert "from Ara (1)" in glance
    assert "the actual words" not in glance
    assert format_unread_glance(settings, "ara") == ""
