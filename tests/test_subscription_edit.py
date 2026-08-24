"""Phase 6 item 3: subscription editing and audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.events.subscription_edit import (
    normalize_subscription_key,
    set_subscription,
    try_kevin_subscription_command,
)
from light_house.main import _settings_dep, app
from light_house.personal.store import PersonalStore
from light_house.tools.personal_tools import execute_personal_tool


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": True,
        "PERSONAL_DB_PATH": str(tmp_path / "personal"),
        "EVENT_SUBSCRIPTIONS_ENABLED": True,
        "SUBSCRIPTION_AUDIT_ENABLED": True,
        "SUBSCRIPTION_AUDIT_LOG_PATH": str(tmp_path / "subscription_audit.ndjson"),
        "LIGHT_HOUSE_ENV": "production",
    }
    base.update(overrides)
    return Settings(**base)


def test_normalize_subscription_aliases() -> None:
    assert normalize_subscription_key("chat") == "post_chat"
    assert normalize_subscription_key("scheduled") == "scheduled_rumination"
    assert normalize_subscription_key("peer") == "peer_message"


def test_set_subscription_via_store(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    result = set_subscription(
        settings,
        agent_id="lumen",
        subscription_key="post_chat",
        enabled=False,
        changed_by="test",
    )
    assert "SUCCESS" in result
    store = PersonalStore(settings.personal_db_path / "lumen.sqlite")
    assert store.is_event_subscribed("post_chat") is False
    audit = (tmp_path / "subscription_audit.ndjson").read_text(encoding="utf-8")
    row = json.loads(audit.strip().splitlines()[-1])
    assert row["subscription_key"] == "post_chat"
    assert row["new_enabled"] is False


def test_kevin_slash_commands(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    listed = try_kevin_subscription_command(
        settings, message="/list_subscriptions", agent_id="lumen"
    )
    assert listed is not None
    assert "post_chat: on" in listed

    unsub = try_kevin_subscription_command(
        settings, message="/unsubscribe scheduled", agent_id="lumen"
    )
    assert unsub is not None
    assert "SUCCESS" in unsub

    assert try_kevin_subscription_command(
        settings, message="hello Kevin", agent_id="lumen"
    ) is None


def test_agent_tool_subscribe(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    result = execute_personal_tool(
        "unsubscribe_event",
        {"event_type": "peer_message"},
        agent_id="ara",
        settings=settings,
    )
    assert "SUCCESS" in result
    store = PersonalStore(settings.personal_db_path / "ara.sqlite")
    assert store.is_event_subscribed("peer_message") is False


def test_chat_api_subscription_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(tmp_path)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat",
                json={"message": "/list_subscriptions", "agent_id": "lumen"},
            )
        assert response.status_code == 200
        assert "post_chat" in response.json()["reply"]
    finally:
        app.dependency_overrides.clear()
