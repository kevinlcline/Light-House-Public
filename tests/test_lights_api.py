"""Lights list API — cache headers and manifest reload."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.main import _settings_dep, app


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "LIGHTS_MANIFEST_PATH": str(tmp_path / "lights.yaml"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAS_DATA_PATH": str(tmp_path / "personas"),
        "NOTES_PATH": str(tmp_path / "notes"),
        "WEB_GATE_ENABLED": False,
        "PRIMARY_LLM": "ollama",
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
    }
    base.update(overrides)
    return Settings(**base)


def _install_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    monkeypatch.setattr("light_house.main.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_lights_list_no_store_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            res = client.get("/v1/lights")
            assert res.status_code == 200
            assert res.headers.get("cache-control") == "no-store"
            ids = [light["id"] for light in res.json()["lights"]]
            assert "elias" in ids
    finally:
        app.dependency_overrides.clear()
