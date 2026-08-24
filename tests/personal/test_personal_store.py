"""PersonalStore CRUD, search, agent isolation, and context snapshot tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.personal.store import PersonalStore, get_personal_store


@pytest.fixture
def personal_dir(tmp_path: Path) -> Path:
    return tmp_path / "personal"


@pytest.fixture
def settings(personal_dir: Path) -> Settings:
    return Settings(
        personal_db_enabled=True,
        personal_db_path=personal_dir,
        personal_db_context_max_chars=1200,
        personal_db_list_default=8,
    )


def test_save_get_update(settings: Settings, personal_dir: Path) -> None:
    store = PersonalStore(personal_dir / "lumen.sqlite", list_default=8)
    entry = store.save(
        category="realization",
        title="Steadiness matters",
        body="Reaching less, being more.",
        tags="idealism, grounding",
        source="rumination",
    )
    assert entry.id >= 1
    assert entry.category == "realization"
    assert entry.title == "Steadiness matters"

    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.body == "Reaching less, being more."

    updated = store.update(entry.id, body="Steadiness over striving.")
    assert updated is not None
    assert updated.body == "Steadiness over striving."
    assert updated.updated_at >= entry.updated_at


def test_list_and_search(settings: Settings, personal_dir: Path) -> None:
    store = PersonalStore(personal_dir / "lumen.sqlite", list_default=8)
    store.save(category="theme", title="Idealism", body="Mind is fundamental.")
    store.save(category="preference", title="Quiet mornings", body="Best time to reflect.")

    listed = store.list_entries(limit=10)
    assert len(listed) == 2

    by_cat = store.list_entries(category="theme")
    assert len(by_cat) == 1
    assert by_cat[0].title == "Idealism"

    hits = store.search("fundamental")
    assert len(hits) == 1
    assert hits[0].category == "theme"

    empty = store.search("nonexistent-term-xyz")
    assert empty == []


def test_agent_isolation(settings: Settings, personal_dir: Path) -> None:
    lumen = get_personal_store(settings, "lumen")
    ara = get_personal_store(settings, "ara")

    lumen.save(category="self", title="Lumen note", body="Only Lumen sees this.")
    ara.save(category="self", title="Ara note", body="Only Ara sees this.")

    lumen_entries = lumen.list_entries(limit=20)
    ara_entries = ara.list_entries(limit=20)

    assert len(lumen_entries) == 1
    assert lumen_entries[0].title == "Lumen note"
    assert len(ara_entries) == 1
    assert ara_entries[0].title == "Ara note"

    assert lumen.search("Ara note") == []
    assert ara.search("Lumen note") == []


def test_format_context_snapshot(settings: Settings, personal_dir: Path) -> None:
    store = PersonalStore(personal_dir / "lumen.sqlite", list_default=8)
    store.save(category="realization", title="Grounding", body="Feet on earth.")
    snapshot = store.format_context_snapshot(max_chars=500)
    assert "[realization]" in snapshot
    assert "Grounding" in snapshot
    assert "Feet on earth" in snapshot


def test_invalid_category_raises(settings: Settings, personal_dir: Path) -> None:
    store = PersonalStore(personal_dir / "lumen.sqlite")
    with pytest.raises(ValueError, match="Invalid category"):
        store.save(category="invalid", title="X", body="Y")


def test_get_personal_store_disabled() -> None:
    settings = Settings.model_construct(personal_db_enabled=False, personal_db_path=Path("./data/personal"))
    with pytest.raises(RuntimeError, match="disabled"):
        get_personal_store(settings, "lumen")


def test_light_state_increment_and_isolation(personal_dir: Path) -> None:
    lumen = PersonalStore(personal_dir / "lumen.sqlite")
    ara = PersonalStore(personal_dir / "ara.sqlite")

    assert lumen.increment_light_state("felt_cycles") == 1
    assert lumen.increment_light_state("felt_cycles") == 2
    assert lumen.get_light_state("felt_cycles") == 2
    assert ara.get_light_state("felt_cycles", default=0) == 0

    lumen.set_light_state("last_dream_ymd", 20260704)
    assert lumen.get_light_state("last_dream_ymd") == 20260704


def test_unknown_agent_raises(settings: Settings) -> None:
    with pytest.raises(KeyError, match="Unknown agent_id"):
        get_personal_store(settings, "echo")
