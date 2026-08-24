"""Lights admin: manifest CRUD, personas, LLM env merge."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.env_admin import merge_env_keys
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.lights_admin import (
    LightsAdminError,
    create_light,
    delete_light,
    read_persona_detail,
    update_light,
    write_persona_content,
)
from light_house.main import _settings_dep, app


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": False,
        "LIGHT_HOUSE_ENV": "production",
        "LIGHTS_MANIFEST_PATH": str(tmp_path / "lights.yaml"),
        "PERSONAS_DATA_PATH": str(tmp_path / "personas"),
        "INNER_LIFE_THREAD_ID": "kevin-home",
        "ARA_THREAD_ID": "ara-home",
        "ARA_ENABLED": True,
    }
    base.update(overrides)
    settings = Settings(**base)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    return settings


def _install_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_merge_env_keys_upserts_without_clobbering(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env_path = repo / ".env"
    env_path.write_text("# header\nFOO=bar\nBAZ=old\n", encoding="utf-8")
    settings = _test_settings(tmp_path, ENV_FILE_PATH=Path(".env"))
    path, _size = merge_env_keys(
        settings,
        repo,
        {"BAZ": "new", "NOVA_LLM_MODEL": "llama3.1:8b"},
    )
    assert path == env_path
    text = env_path.read_text(encoding="utf-8")
    assert "# header" in text
    assert "FOO=bar" in text
    assert "BAZ=new" in text
    assert "NOVA_LLM_MODEL=llama3.1:8b" in text
    assert "BAZ=old" not in text


def test_merge_env_keys_syncs_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin LLM save must update process env or the UI reload appears to revert."""
    import os

    from light_house.lights_admin import read_light_llm, write_light_llm

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text(
        "LUMEN_LLM_PROVIDER=openrouter\nLUMEN_LLM_MODEL=deepseek/deepseek-v4-pro\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "deepseek/deepseek-v4-pro")
    settings = _test_settings(tmp_path, ENV_FILE_PATH=Path(".env"), OPENROUTER_API_KEY="test-key")

    before = read_light_llm(settings, "lumen")
    assert before.model == "deepseek/deepseek-v4-pro"

    write_light_llm(
        settings,
        repo,
        "lumen",
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        inner_life_model="deepseek/deepseek-v4-flash",
    )

    assert os.environ.get("LUMEN_LLM_MODEL") == "deepseek/deepseek-v4-flash"
    after = read_light_llm(settings, "lumen")
    assert after.model == "deepseek/deepseek-v4-flash"
    assert "deepseek/deepseek-v4-flash" in (repo / ".env").read_text(encoding="utf-8")


def test_create_light_writes_manifest_persona_and_notes(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    result = create_light(
        settings,
        repo,
        light_id="nova",
        display_name="Nova",
        llm_provider="ollama",
        llm_model="llama3.1:8b",
        voice_id="am_fenrir",
    )
    assert result["light"]["id"] == "nova"
    assert result["light"]["voice_id"] == "am_fenrir"
    persona_path = tmp_path / "personas" / "nova_system.md"
    assert persona_path.is_file()
    notes_path = tmp_path / "notes" / "nova"
    assert notes_path.is_dir()
    manifest = (tmp_path / "lights.yaml").read_text(encoding="utf-8")
    assert "nova" in manifest
    assert "am_fenrir" in manifest


def test_update_light_primary_and_toggles(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    create_light(
        settings,
        repo,
        light_id="nova",
        display_name="Nova",
        llm_provider="ollama",
        llm_model="llama3.1:8b",
    )
    result = update_light(
        settings,
        "nova",
        display_name="Nova Prime",
        enabled=False,
        set_primary=True,
        voice_id="bf_emma",
    )
    assert result["light"]["display_name"] == "Nova Prime"
    assert result["light"]["enabled"] is False
    assert result["light"]["voice_id"] == "bf_emma"
    reload_lights_manifest(settings)
    from light_house.lights.registry import get_primary_light_id

    assert get_primary_light_id(settings) == "nova"


def test_delete_light_guards(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    with pytest.raises(LightsAdminError, match="primary"):
        delete_light(settings, "lumen")
    delete_light(settings, "ara")
    with pytest.raises(LightsAdminError, match="primary|last light"):
        delete_light(settings, "lumen")


def test_persona_copy_on_write_from_bundled(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    detail = read_persona_detail(settings, "lumen")
    assert detail["source"] in ("bundled", "data", "package")
    assert detail["content"]
    write_persona_content(settings, "lumen", detail["content"] + "\n\n## Admin edit\n")
    detail2 = read_persona_detail(settings, "lumen")
    assert detail2["source"] == "data"
    assert "Admin edit" in detail2["content"]


def test_admin_api_returns_404_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _test_settings(tmp_path, LIGHTS_ADMIN_ENABLED=False)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            res = client.get("/v1/admin/lights")
        assert res.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_admin_api_create_and_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    settings = _test_settings(tmp_path, LIGHTS_ADMIN_ENABLED=True)
    _install_settings(settings, monkeypatch)
    try:
        with patch("light_house.main._REPO_ROOT", repo):
            with TestClient(app) as client:
                post = client.post(
                    "/v1/admin/lights",
                    json={
                        "id": "ember",
                        "display_name": "Ember",
                        "llm_provider": "ollama",
                        "llm_model": "llama3.1:8b",
                    },
                )
                assert post.status_code == 200
                get_res = client.get("/v1/admin/lights")
                assert get_res.status_code == 200
                ids = [light["id"] for light in get_res.json()["lights"]]
                assert "ember" in ids
    finally:
        app.dependency_overrides.clear()
