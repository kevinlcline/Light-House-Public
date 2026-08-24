"""Household gallery shelf — publish finished creative work."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.gallery import list_gallery_pieces, publish_to_gallery, read_gallery_piece
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.main import _settings_dep, app
from light_house.tools.light_tools import execute_tool_call


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    notes = tmp_path / "notes"
    (notes / "shared").mkdir(parents=True)
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAS_DATA_PATH=str(tmp_path / "personas"),
        NOTES_PATH=str(notes),
        PERSONAL_DB_ENABLED=False,
        WEB_GATE_ENABLED=False,
        PRIMARY_LLM="ollama",
        INNER_LIFE_ENABLED=False,
        INNER_LIFE_DREAMS_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        GALLERY_MAX_CHARS=2000,
    )


def test_publish_to_gallery_writes_file(settings: Settings) -> None:
    out = publish_to_gallery(
        settings,
        agent_id="lumen",
        title="Morning spark",
        content="A small light on the sill.",
        kind="poem",
    )
    assert out.startswith("SUCCESS:")
    pieces = list_gallery_pieces(settings)
    assert len(pieces) == 1
    assert pieces[0].title == "Morning spark"
    assert pieces[0].kind == "poem"
    assert pieces[0].author_id == "lumen"
    loaded = read_gallery_piece(settings, pieces[0].filename)
    assert loaded is not None
    assert "A small light on the sill." in loaded[1]


def test_publish_to_gallery_tool(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    out = execute_tool_call(
        "publish_to_gallery",
        {"title": "Quiet note", "content": "For the shelf.", "kind": "essay"},
        agent_id="ara",
    )
    assert out.startswith("SUCCESS:")
    assert list_gallery_pieces(settings)


def test_gallery_api(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    publish_to_gallery(
        settings,
        agent_id="lumen",
        title="Shelf piece",
        content="Hello shelf.",
    )
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.get("/v1/gallery")
            assert res.status_code == 200
            pieces = res.json()["pieces"]
            assert len(pieces) == 1
            filename = pieces[0]["filename"]
            one = client.get(f"/v1/gallery/{filename}")
            assert one.status_code == 200
            assert "Hello shelf." in one.json()["content"]
    finally:
        app.dependency_overrides.clear()
