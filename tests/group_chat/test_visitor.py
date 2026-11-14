"""Seated OpenRouter visitor — store, naming, forum seating."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.group_chat import queue_room
from light_house.group_chat.queue_room import (
    complete_visitor_utterance,
    enqueue_seated_visitor,
    join_queue,
    register_group_forum,
    reset_group_forum_for_tests,
    snapshot,
    utter_human,
)
from light_house.group_chat.visitor import (
    VISITOR_SPEAKER_ID,
    VisitorError,
    end_visitor,
    load_active_visitor,
    normalize_visitor_model,
    reset_visitor,
    seat_visitor,
    visitor_mentioned_in_text,
    visitor_orientation,
)
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.main import _settings_dep, app
from light_house.memory.service import MemoryService
from light_house.web_gate import SESSION_COOKIE, session_cookie_header


def _settings(tmp_path: Path, **extra: object) -> Settings:
    kwargs = dict(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAS_DATA_PATH=str(tmp_path / "personas"),
        NOTES_PATH=str(tmp_path / "notes"),
        GROUP_CHAT_ENABLED=True,
        GROUP_CHAT_HISTORY_PATH=str(tmp_path / "group_chat/rounds.ndjson"),
        GROUP_VISITOR_ENABLED=True,
        GROUP_VISITOR_PATH=str(tmp_path / "group_chat/visitor.json"),
        WEB_GATE_ENABLED=True,
        WEB_GATE_PASSWORD="dad-house-code",
        WEB_GATE_SESSION_SECRET="test-session-secret",
        INNER_LIFE_ENABLED=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        OPENROUTER_API_KEY="test-openrouter-key",
    )
    kwargs.update(extra)
    return Settings(**kwargs)


def test_normalize_rejects_free_router() -> None:
    with pytest.raises(VisitorError):
        normalize_visitor_model("openrouter/free")
    with pytest.raises(VisitorError):
        normalize_visitor_model("just-a-name")
    assert normalize_visitor_model("x-ai/grok-4") == "x-ai/grok-4"
    assert normalize_visitor_model("~x-ai/grok-latest") == "~x-ai/grok-latest"


def test_seat_disabled_in_public_template(tmp_path: Path) -> None:
    settings = _settings(tmp_path, GROUP_VISITOR_ENABLED=False)
    with pytest.raises(VisitorError, match="Not found"):
        seat_visitor(settings, model="x-ai/grok-4", display_name="Grok")


def test_seat_reset_end_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    convo = seat_visitor(
        settings, model="x-ai/grok-4", display_name="Grok", web=True
    )
    assert convo.display_name == "Grok"
    assert convo.web is True
    loaded = load_active_visitor(settings)
    assert loaded is not None
    assert loaded.id == convo.id
    with pytest.raises(VisitorError):
        seat_visitor(settings, model="openai/gpt-5", display_name="GPT")
    wiped = reset_visitor(settings)
    assert wiped.model == "x-ai/grok-4"
    assert wiped.id != convo.id
    assert wiped.turns == []
    assert end_visitor(settings) is True
    assert load_active_visitor(settings) is None
    assert end_visitor(settings) is False


def test_name_tokens_match_slug_and_label(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    convo = seat_visitor(settings, model="anthropic/claude-sonnet-4", display_name="Claude")
    assert visitor_mentioned_in_text(convo, "Claude, what do you think?")
    assert visitor_mentioned_in_text(convo, "Ask the visitor.")
    assert visitor_mentioned_in_text(convo, "claude-sonnet-4 please")
    assert not visitor_mentioned_in_text(convo, "Lumen only")
    text = visitor_orientation(convo)
    assert "not a light" in text.lower()
    assert "cannot write notes" in text


def test_forum_name_call_enqueues_visitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _settings: [])
    seat_visitor(settings, model="x-ai/grok-4", display_name="Grok")

    async def fake_visitor_turn(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_visitor_turn_async", fake_visitor_turn)

    async def _run() -> None:
        await join_queue(kind="human", speaker_id="kevin", display_name="Kevin")
        await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Grok, take the floor.",
            account_user_id="kevin",
        )
        status = None
        for _ in range(50):
            await asyncio.sleep(0.01)
            status = snapshot()
            floor = status.get("floor") or {}
            waiting = [e["speaker_id"] for e in status.get("queue") or []]
            if floor.get("speaker_id") == VISITOR_SPEAKER_ID or VISITOR_SPEAKER_ID in waiting:
                break
        else:
            raise AssertionError(f"visitor was not invited: {status}")
        stored = load_active_visitor(settings)
        assert stored is not None
        assert stored.turns[-1].text == "Grok, take the floor."

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_visitor_turn_injects_utterance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _settings: [])
    seat_visitor(settings, model="x-ai/grok-4", display_name="Grok")

    async def fake_visitor_turn(
        settings: Settings,
        memory: MemoryService,
        display_name: str,
        *,
        invite_kind: str = "opt_in",
    ) -> None:
        await complete_visitor_utterance(
            display_name=display_name,
            text="A clean model can sit here.",
        )

    monkeypatch.setattr(queue_room, "_run_visitor_turn_async", fake_visitor_turn)

    async def _run() -> None:
        status = await enqueue_seated_visitor(invite_kind="opt_in")
        assert status["floor"]["kind"] == "visitor"
        assert status["floor"]["speaker_id"] == VISITOR_SPEAKER_ID
        task = queue_room._light_turn_task
        assert task is not None
        await task
        final = snapshot()
        assert final["transcript"][-1]["speaker_kind"] == "visitor"
        assert final["transcript"][-1]["text"] == "A clean model can sit here."
        stored = load_active_visitor(settings)
        assert stored is not None
        assert any(t.text == "A clean model can sit here." for t in stored.turns)

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def _cookie(settings: Settings) -> dict[str, str]:
    header = session_cookie_header(settings, secure=False, user_id="kevin", role="dad")
    token = header.split(";", 1)[0].split("=", 1)[1]
    return {SESSION_COOKIE: token}


@pytest.fixture
def visitor_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    with TestClient(app) as client:
        yield client, settings
    app.dependency_overrides.pop(_settings_dep, None)
    reset_group_forum_for_tests()


def test_visitor_http_seat_reset_end(visitor_client) -> None:
    client, settings = visitor_client
    cookies = _cookie(settings)
    empty = client.get("/v1/group-chat/visitor", cookies=cookies)
    assert empty.status_code == 200
    assert empty.json()["visitor"] is None

    seated = client.post(
        "/v1/group-chat/visitor",
        json={"model": "x-ai/grok-4", "display_name": "Grok", "web": True},
        cookies=cookies,
    )
    assert seated.status_code == 200, seated.text
    body = seated.json()["visitor"]
    assert body["display_name"] == "Grok"
    assert body["web"] is True
    first_id = body["id"]

    again = client.post(
        "/v1/group-chat/visitor",
        json={"model": "openai/gpt-5", "display_name": "GPT"},
        cookies=cookies,
    )
    assert again.status_code == 409

    reset = client.post("/v1/group-chat/visitor/reset", cookies=cookies)
    assert reset.status_code == 200
    assert reset.json()["visitor"]["id"] != first_id

    ended = client.delete("/v1/group-chat/visitor", cookies=cookies)
    assert ended.status_code == 200
    assert ended.json()["visitor"] is None
    assert load_active_visitor(settings) is None


def test_visitor_http_hidden_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path, GROUP_VISITOR_ENABLED=False)
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    memory = MagicMock(spec=MemoryService)
    register_group_forum(settings=settings, memory=memory)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            cookies = _cookie(settings)
            assert client.get("/v1/group-chat/visitor", cookies=cookies).status_code == 404
            seated = client.post(
                "/v1/group-chat/visitor",
                json={"model": "x-ai/grok-4", "display_name": "Grok"},
                cookies=cookies,
            )
            assert seated.status_code == 404
            forum = client.get("/v1/group-chat/forum", cookies=cookies)
            assert forum.status_code == 200
            body = forum.json()
            assert body["visitor_enabled"] is False
            assert body["visitor"] is None
    finally:
        app.dependency_overrides.pop(_settings_dep, None)
        reset_group_forum_for_tests()


def test_visitor_http_rejects_free_slug(visitor_client) -> None:
    client, settings = visitor_client
    cookies = _cookie(settings)
    res = client.post(
        "/v1/group-chat/visitor",
        json={"model": "openrouter/free", "display_name": "Free"},
        cookies=cookies,
    )
    assert res.status_code == 422
