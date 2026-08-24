"""Lights manifest and registry tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.lights.manifest import bootstrap_manifest_dict, ensure_manifest_file, load_manifest
from light_house.lights.registry import (
    get_light,
    list_enabled_lights,
    list_lights_for_broadcast,
    reload_lights_manifest,
    validate_light_id,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "LIGHTS_MANIFEST_PATH": str(tmp_path / "lights.yaml"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "INNER_LIFE_THREAD_ID": "kevin-home",
        "ARA_THREAD_ID": "ara-home",
        "ARA_ENABLED": True,
    }
    base.update(overrides)
    return Settings(**base)


def test_bootstrap_creates_manifest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = ensure_manifest_file(settings)
    assert path.is_file()
    manifest = load_manifest(settings)
    assert manifest.primary_light_id == "lumen"
    assert len(manifest.lights) == 3
    by_id = {light.id: light for light in manifest.lights}
    assert by_id["lumen"].voice_id == "af_sarah"
    assert by_id["ara"].voice_id == "af_bella"
    assert by_id["elias"].voice_id == "am_michael"


def test_list_enabled_respects_enabled_flag(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ARA_ENABLED=False)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    enabled = list_enabled_lights(settings)
    ids = {light.id for light in enabled}
    assert "lumen" in ids
    assert "ara" not in ids


def test_validate_unknown_light(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    with pytest.raises(KeyError):
        validate_light_id("nova", settings)


def test_broadcast_pairs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    pairs = list_lights_for_broadcast(settings)
    assert ("lumen", "kevin-home") in pairs
    assert ("ara", "ara-home") in pairs


def test_get_light_notes_dir(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    light = get_light("lumen", settings)
    assert light.notes_dir == "lumen"
    assert light.thread_id == "kevin-home"
