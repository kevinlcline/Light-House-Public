"""Env admin: read/write .env and restart API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.env_admin import EnvAdminError, resolve_env_file_path, write_env_content
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
    }
    base.update(overrides)
    return Settings(**base)


def _install_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings


def test_resolve_env_file_path_blocks_traversal(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, ENV_FILE_PATH=Path("../outside.env"))
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(EnvAdminError):
        resolve_env_file_path(settings, repo)


def test_write_env_creates_backup(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env_path = repo / ".env"
    env_path.write_text("OLD=1\n", encoding="utf-8")
    settings = _test_settings(tmp_path, ENV_FILE_PATH=Path(".env"))
    path, size = write_env_content(settings, repo, "NEW=2\n")
    assert path == env_path
    assert size == len("NEW=2\n".encode())
    assert env_path.read_text(encoding="utf-8") == "NEW=2\n"
    assert (repo / ".env.bak").read_text(encoding="utf-8") == "OLD=1\n"


def test_env_api_returns_404_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path, ENV_EDITOR_ENABLED=False)
    _install_settings(settings, monkeypatch)
    try:
        with TestClient(app) as client:
            get_res = client.get("/v1/admin/env")
            put_res = client.put("/v1/admin/env", json={"content": "X=1\n"})
            restart_res = client.post("/v1/admin/restart")
        assert get_res.status_code == 404
        assert put_res.status_code == 404
        assert restart_res.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_env_api_read_write_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env_path = repo / ".env"
    env_path.write_text("FOO=bar\n", encoding="utf-8")
    settings = _test_settings(
        tmp_path,
        ENV_EDITOR_ENABLED=True,
        ENV_FILE_PATH=Path(".env"),
    )
    _install_settings(settings, monkeypatch)
    try:
        with patch("light_house.main._REPO_ROOT", repo):
            with TestClient(app) as client:
                get_res = client.get("/v1/admin/env")
                assert get_res.status_code == 200
                data = get_res.json()
                assert data["content"] == "FOO=bar\n"
                assert "FOO=bar" in data["content"]

                put_res = client.put("/v1/admin/env", json={"content": "FOO=baz\n# comment\n"})
                assert put_res.status_code == 200
                assert env_path.read_text(encoding="utf-8") == "FOO=baz\n# comment\n"
                assert get_res.headers.get("cache-control") == "no-store"
    finally:
        app.dependency_overrides.clear()


def test_restart_api_schedules_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path, ENV_EDITOR_ENABLED=True)
    _install_settings(settings, monkeypatch)
    scheduled: list[str] = []

    async def fake_schedule(s: Settings) -> str:
        scheduled.append("called")
        return "test restart message"

    monkeypatch.setattr("light_house.main.schedule_server_restart", fake_schedule)
    try:
        with TestClient(app) as client:
            res = client.post("/v1/admin/restart")
        assert res.status_code == 200
        assert res.json()["ok"] is True
        assert scheduled == ["called"]
    finally:
        app.dependency_overrides.clear()
