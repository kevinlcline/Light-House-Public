"""Public notify-me email signup on the landing page."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.main import _settings_dep, app
from light_house.public_notify import append_notify_email, normalize_email


def test_normalize_email() -> None:
    assert normalize_email("  Ada@Example.COM ") == "ada@example.com"
    assert normalize_email("not-an-email") is None
    assert normalize_email("") is None


def test_append_notify_email_dedupes(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        PUBLIC_NOTIFY_PATH=str(tmp_path / "notify.ndjson"),
        WEB_GATE_ENABLED=False,
        PERSONAL_DB_ENABLED=False,
        INNER_LIFE_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
    )
    created, msg = append_notify_email(settings, "friend@example.com")
    assert created is True
    assert "Thank you" in msg
    created2, msg2 = append_notify_email(settings, "FRIEND@example.com")
    assert created2 is False
    assert "already" in msg2.lower()
    lines = (tmp_path / "notify.ndjson").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_public_notify_endpoint_bypasses_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PUBLIC_NOTIFY_PATH=str(tmp_path / "notify.ndjson"),
        PERSONAL_DB_ENABLED=False,
        INNER_LIFE_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        WEB_GATE_ENABLED=True,
        WEB_GATE_PASSWORD="gate-pass",
        WEB_GATE_SESSION_SECRET="test-session-secret-key",
    )
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            blocked = client.get("/v1/agents")
            assert blocked.status_code == 401
            res = client.post("/v1/public/notify", json={"email": "visitor@example.com"})
            assert res.status_code == 200
            body = res.json()
            assert body["ok"] is True
            assert body["created"] is True
            bad = client.post("/v1/public/notify", json={"email": "nope"})
            assert bad.status_code == 422
    finally:
        app.dependency_overrides.clear()
