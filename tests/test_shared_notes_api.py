"""Shared note write API (Kevin → shared/ for Lumen + Ara)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.main import _settings_dep, app


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    notes_dir = tmp_path / "notes"
    (notes_dir / "shared").mkdir(parents=True, exist_ok=True)
    base = {
        "_env_file": None,
        "NOTES_PATH": str(notes_dir),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": False,
    }
    base.update(overrides)
    return Settings(**base)


def _install(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_write_shared_note_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path)
    _install(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.put(
                "/v1/notes/shared/from-kevin.md",
                json={"content": "Hello Lumen and Ara."},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "shared/from-kevin.md"
        assert data["size_bytes"] > 0
        written = (tmp_path / "notes" / "shared" / "from-kevin.md").read_text(encoding="utf-8")
        assert written == "Hello Lumen and Ara."
    finally:
        app.dependency_overrides.clear()


def test_write_shared_note_replaces_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = tmp_path / "notes" / "shared" / "household.md"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text("old text", encoding="utf-8")
    settings = _test_settings(tmp_path)
    _install(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.put(
                "/v1/notes/shared/household.md",
                json={"content": "Updated for both agents."},
            )
        assert response.status_code == 200
        assert shared.read_text(encoding="utf-8") == "Updated for both agents."
    finally:
        app.dependency_overrides.clear()


def test_write_shared_note_rejects_empty_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path)
    _install(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.put(
                "/v1/notes/shared/x.md",
                json={"content": "   "},
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_write_shared_note_rejects_invalid_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path)
    _install(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.put(
                "/v1/notes/shared/invalid$.md",
                json={"content": "nope"},
            )
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_agents_can_read_written_shared_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path)
    _install(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            client.put(
                "/v1/notes/shared/plan.md",
                json={"content": "Shared plan body."},
            )
            lumen = client.get("/v1/notes/shared/plan.md?agent=lumen")
            ara = client.get("/v1/notes/shared/plan.md?agent=ara")
        assert lumen.status_code == 200
        assert ara.status_code == 200
        assert lumen.json()["content"] == "Shared plan body."
        assert ara.json()["content"] == "Shared plan body."
    finally:
        app.dependency_overrides.clear()


def test_delete_shared_note_via_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path)
    _install(settings, monkeypatch)
    shared = tmp_path / "notes" / "shared" / "trash.md"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text("remove me", encoding="utf-8")
    try:
        with TestClient(app) as client:
            response = client.delete("/v1/notes/shared/trash.md?agent=lumen")
        assert response.status_code == 200
        assert response.json()["name"] == "shared/trash.md"
        assert not shared.is_file()
    finally:
        app.dependency_overrides.clear()


def test_delete_private_note_via_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path)
    _install(settings, monkeypatch)
    private = tmp_path / "notes" / "lumen" / "draft.md"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("private", encoding="utf-8")
    try:
        with TestClient(app) as client:
            response = client.delete("/v1/notes/lumen/draft.md?agent=lumen")
        assert response.status_code == 200
        assert not private.is_file()
    finally:
        app.dependency_overrides.clear()
