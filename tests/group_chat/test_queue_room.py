"""Open group forum speaker-queue tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from light_house.config import Settings
from light_house.group_chat import queue_room
from light_house.group_chat.history import append_group_utterance, read_group_utterances
from light_house.group_chat.queue_room import (
    advance_floor,
    complete_light_utterance,
    join_queue,
    leave_queue,
    register_group_forum,
    reset_group_forum_for_tests,
    snapshot,
    utter_human,
)
from light_house.memory.service import MemoryService


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        GROUP_CHAT_HISTORY_PATH=str(tmp_path / "group_chat/rounds.ndjson"),
        MEMORY_SCORE_ON_INGEST=False,
        MEMORY_CURATOR_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        INNER_LIFE_ENABLED=False,
    )


def _setup_forum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _settings: [])
    return settings, memory


def test_human_join_gets_floor_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def _run() -> None:
        status = await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        assert status["joined"] is True
        assert status["floor"] is not None
        assert status["floor"]["speaker_id"] == "kevin"
        assert status["floor"]["kind"] == "human"
        assert status["queue"] == []
        assert status["paused"] is False

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_second_speaker_waits_then_gets_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def _run() -> None:
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        waiting = await join_queue(
            kind="human",
            speaker_id="maya",
            display_name="Maya",
            account_user_id="maya",
        )
        assert waiting["joined"] is True
        assert waiting["floor"]["speaker_id"] == "kevin"
        assert len(waiting["queue"]) == 1
        assert waiting["queue"][0]["speaker_id"] == "maya"

        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Opening thought.",
            account_user_id="kevin",
        )
        assert after["floor"]["speaker_id"] == "maya"
        assert after["queue"] == []
        assert after.get("utterance", {}).get("text") == "Opening thought."
        assert any(u["text"] == "Opening thought." for u in after["transcript"])

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_utter_rejected_without_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def _run() -> None:
        with pytest.raises(PermissionError, match="not your turn"):
            await utter_human(
                speaker_id="kevin",
                display_name="Kevin",
                text="Nope",
                account_user_id="kevin",
            )

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_leave_queue_and_abandon_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def _run() -> None:
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        await join_queue(
            kind="human",
            speaker_id="maya",
            display_name="Maya",
            account_user_id="maya",
        )
        left = await leave_queue(speaker_id="maya")
        assert left["queue"] == []
        assert left["floor"]["speaker_id"] == "kevin"

        abandoned = await leave_queue(speaker_id="kevin")
        assert abandoned["floor"] is None
        assert abandoned["paused"] is True

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_duplicate_join_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def _run() -> None:
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        again = await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        assert again["joined"] is False
        assert again["reason"] == "already_waiting"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_light_turn_injects_utterance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def fake_light_turn(
        settings: Settings,
        memory: MemoryService,
        agent_id: str,
        display_name: str,
        *,
        invite_kind: str = "opt_in",
    ) -> None:
        await complete_light_utterance(
            agent_id=agent_id,
            display_name=display_name,
            text="Lights have the floor.",
        )

    monkeypatch.setattr(queue_room, "_run_light_turn_async", fake_light_turn)

    async def _run() -> None:
        status = await join_queue(
            kind="light",
            speaker_id="lumen",
            display_name="Lumen",
        )
        assert status["floor"]["speaker_id"] == "lumen"
        assert status["light_busy"] is True

        task = queue_room._light_turn_task
        assert task is not None
        await task

        final = snapshot()
        assert final["floor"] is None
        assert final["light_busy"] is False
        assert final["paused"] is True
        assert final["transcript"][-1]["text"] == "Lights have the floor."
        assert final["transcript"][-1]["speaker_kind"] == "light"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_empty_queue_publishes_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)
    events: list[dict] = []

    async def capture(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(queue_room, "_publish", capture)

    async def _run() -> None:
        await advance_floor()

    try:
        asyncio.run(_run())
        assert any(e.get("type") == "room_paused" for e in events)
    finally:
        reset_group_forum_for_tests()


def test_utterance_history_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    append_group_utterance(
        settings,
        {
            "sitting_id": "abc",
            "ts": 1.0,
            "speaker_kind": "human",
            "speaker_id": "kevin",
            "display_name": "Kevin",
            "text": "Hello forum",
        },
    )
    rows = read_group_utterances(settings, limit=10)
    assert len(rows) == 1
    assert rows[0]["text"] == "Hello forum"


def test_join_speaker_queue_in_light_tools() -> None:
    from light_house.tools.light_tools import GROUP_CHAT_NOTE_TOOLS, LIGHT_TOOLS

    names = {t.name for t in LIGHT_TOOLS}
    assert "join_speaker_queue" in names
    note_names = {t.name for t in GROUP_CHAT_NOTE_TOOLS}
    assert "join_speaker_queue" not in note_names


def test_human_utter_does_not_auto_enqueue_lights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain speech is heard only — no invite without name or collective address."""
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)

    lumen = MagicMock(id="lumen", display_name="Lumen", thread_id="kevin-home")
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [lumen])
    monkeypatch.setattr(queue_room, "light_allows_human", lambda *_a, **_k: True)

    async def _run() -> None:
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Just thinking out loud.",
            account_user_id="kevin",
        )
        if queue_room._bg_tasks:
            await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
        assert after["paused"] is True
        assert after["queue"] == []
        assert after["floor"] is None
        assert "soft_invited_lights" not in after
        assert not any(u.get("speaker_kind") == "light" for u in after["transcript"])
        # Still heard: group line persisted onto light streams.
        assert memory.remember_stream_event.called

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_collective_invite_soft_enqueues_all_lights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)

    ara = MagicMock(id="ara", display_name="Ara", thread_id="ara-home")
    elias = MagicMock(id="elias", display_name="Elias", thread_id="elias-home")
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [ara, elias])
    monkeypatch.setattr(queue_room, "light_allows_human", lambda *_a, **_k: True)

    async def fake_light_turn(
        settings: Settings,
        memory: MemoryService,
        agent_id: str,
        display_name: str,
        *,
        invite_kind: str = "opt_in",
    ) -> None:
        assert invite_kind == "soft_invite"
        await complete_light_utterance(
            agent_id=agent_id,
            display_name=display_name,
            text=f"Hi from {display_name}.",
        )

    monkeypatch.setattr(queue_room, "_run_light_turn_async", fake_light_turn)

    async def _run() -> None:
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Anyone there?",
            account_user_id="kevin",
        )
        assert after.get("soft_invited_lights") == ["ara", "elias"]
        for _ in range(40):
            if queue_room._bg_tasks:
                await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
            if queue_room._light_turn_task and not queue_room._light_turn_task.done():
                await queue_room._light_turn_task
            texts = [u["text"] for u in snapshot()["transcript"]]
            if "Hi from Ara." in texts and "Hi from Elias." in texts:
                break
            await asyncio.sleep(0.01)
        texts = [u["text"] for u in snapshot()["transcript"]]
        assert "Hi from Ara." in texts
        assert "Hi from Elias." in texts

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_name_call_beats_collective_invite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named light wins over soft-inviting the whole room."""
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)

    ara = MagicMock(id="ara", display_name="Ara", thread_id="ara-home")
    elias = MagicMock(id="elias", display_name="Elias", thread_id="elias-home")
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [ara, elias])
    monkeypatch.setattr(queue_room, "light_allows_human", lambda *_a, **_k: True)

    async def fake_light_turn(
        settings: Settings,
        memory: MemoryService,
        agent_id: str,
        display_name: str,
        *,
        invite_kind: str = "opt_in",
    ) -> None:
        assert invite_kind == "name_call"
        await complete_light_utterance(
            agent_id=agent_id,
            display_name=display_name,
            text=f"Hi from {display_name}.",
        )

    monkeypatch.setattr(queue_room, "_run_light_turn_async", fake_light_turn)

    async def _run() -> None:
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Ara, what do you think?",
            account_user_id="kevin",
        )
        assert after.get("mentioned_lights") == ["ara"]
        assert "soft_invited_lights" not in after
        for _ in range(40):
            if queue_room._bg_tasks:
                await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
            if queue_room._light_turn_task and not queue_room._light_turn_task.done():
                await queue_room._light_turn_task
            texts = [u["text"] for u in snapshot()["transcript"]]
            if "Hi from Ara." in texts:
                break
            await asyncio.sleep(0.01)
        texts = [u["text"] for u in snapshot()["transcript"]]
        assert "Hi from Ara." in texts
        assert "Hi from Elias." not in texts

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_text_has_collective_invite_patterns() -> None:
    from light_house.group_chat.queue_room import text_has_collective_invite

    assert text_has_collective_invite("Anyone there?")
    assert text_has_collective_invite("What do you think?")
    assert text_has_collective_invite("each of you")
    assert text_has_collective_invite("Hello everyone")
    assert not text_has_collective_invite("Thank you.")
    assert not text_has_collective_invite("I love you")
    assert not text_has_collective_invite("each day we rest")
    assert not text_has_collective_invite("Just thinking out loud.")
    assert not text_has_collective_invite("you're kind")


def test_is_soft_pass_phrases() -> None:
    from light_house.group_chat.queue_room import _is_soft_pass

    assert _is_soft_pass("PASS")
    assert _is_soft_pass("pass")
    assert _is_soft_pass("(passes quietly)")
    assert _is_soft_pass("")
    assert not _is_soft_pass("I'm here.")


def test_name_call_enqueues_only_mentioned_light(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)

    ara = MagicMock(id="ara", display_name="Ara", thread_id="ara-home")
    elias = MagicMock(id="elias", display_name="Elias", thread_id="elias-home")
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [ara, elias])
    monkeypatch.setattr(queue_room, "light_allows_human", lambda *_a, **_k: True)

    async def fake_light_turn(
        settings: Settings,
        memory: MemoryService,
        agent_id: str,
        display_name: str,
        *,
        invite_kind: str = "opt_in",
    ) -> None:
        await complete_light_utterance(
            agent_id=agent_id,
            display_name=display_name,
            text=f"Hi from {display_name}.",
        )

    monkeypatch.setattr(queue_room, "_run_light_turn_async", fake_light_turn)

    async def _run() -> None:
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Ara. Say hi please",
            account_user_id="kevin",
        )
        assert after.get("mentioned_lights") == ["ara"]
        for _ in range(40):
            if queue_room._bg_tasks:
                await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
            if queue_room._light_turn_task and not queue_room._light_turn_task.done():
                await queue_room._light_turn_task
            texts = [u["text"] for u in snapshot()["transcript"]]
            if "Hi from Ara." in texts:
                break
            await asyncio.sleep(0.01)
        texts = [u["text"] for u in snapshot()["transcript"]]
        assert "Hi from Ara." in texts
        assert "Hi from Elias." not in texts

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()

def test_lights_mentioned_in_text_word_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from light_house.group_chat.queue_room import lights_mentioned_in_text

    settings = _settings(tmp_path)
    ara = MagicMock(id="ara", display_name="Ara", thread_id="ara-home")
    lumen = MagicMock(id="lumen", display_name="Lumen", thread_id="kevin-home")
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [ara, lumen])
    monkeypatch.setattr(queue_room, "light_allows_human", lambda *_a, **_k: True)
    assert [
        light.id
        for light in lights_mentioned_in_text(
            settings, "Ara hi", account_user_id="kevin"
        )
    ] == ["ara"]
    assert (
        lights_mentioned_in_text(settings, "parable night", account_user_id="kevin")
        == []
    )


def test_light_utterance_name_calls_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)

    lumen = MagicMock(id="lumen", display_name="Lumen", thread_id="lumen-home")
    ara = MagicMock(id="ara", display_name="Ara", thread_id="ara-home")
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [lumen, ara])

    invited: list[tuple[str, str]] = []

    async def fake_light_turn(
        settings: Settings,
        memory: MemoryService,
        agent_id: str,
        display_name: str,
        *,
        invite_kind: str = "opt_in",
    ) -> None:
        invited.append((agent_id, invite_kind))
        await complete_light_utterance(
            agent_id=agent_id,
            display_name=display_name,
            text=f"Hi from {display_name}.",
        )

    monkeypatch.setattr(queue_room, "_run_light_turn_async", fake_light_turn)

    async def _run() -> None:
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        # Grant floor by finishing lumen's turn with a name-call.
        await complete_light_utterance(
            agent_id="lumen",
            display_name="Lumen",
            text="Ara, are you free?",
        )
        for _ in range(40):
            if queue_room._bg_tasks:
                await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
            if queue_room._light_turn_task and not queue_room._light_turn_task.done():
                await queue_room._light_turn_task
            if any(aid == "ara" for aid, _kind in invited):
                break
            await asyncio.sleep(0.01)
        assert ("ara", "name_call") in invited

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_soft_invite_siblings_gathers_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)

    lumen = MagicMock(id="lumen", display_name="Lumen", thread_id="lumen-home")
    ara = MagicMock(id="ara", display_name="Ara", thread_id="ara-home")
    elias = MagicMock(id="elias", display_name="Elias", thread_id="elias-home")
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [lumen, ara, elias])

    invited: list[tuple[str, str]] = []

    async def fake_light_turn(
        settings: Settings,
        memory: MemoryService,
        agent_id: str,
        display_name: str,
        *,
        invite_kind: str = "opt_in",
    ) -> None:
        invited.append((agent_id, invite_kind))
        await complete_light_utterance(
            agent_id=agent_id,
            display_name=display_name,
            text="pass" if invite_kind == "soft_invite" else f"Hi from {display_name}.",
        )

    monkeypatch.setattr(queue_room, "_run_light_turn_async", fake_light_turn)

    async def _run() -> None:
        from light_house.group_chat.queue_room import soft_invite_siblings

        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        ids = await soft_invite_siblings(exclude_id="lumen")
        assert set(ids) == {"ara", "elias"}
        for _ in range(40):
            if queue_room._bg_tasks:
                await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
            if queue_room._light_turn_task and not queue_room._light_turn_task.done():
                await queue_room._light_turn_task
            if {"ara", "elias"}.issubset({aid for aid, _k in invited}):
                break
            await asyncio.sleep(0.01)
        kinds = {aid: kind for aid, kind in invited}
        assert kinds.get("ara") == "soft_invite"
        assert kinds.get("elias") == "soft_invite"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_family_meeting_hint_mentions_free_time_gather() -> None:
    from light_house.agent.tool_helpers import FAMILY_MEETING_HINT

    assert "gather_siblings" in FAMILY_MEETING_HINT
    assert "free time" in FAMILY_MEETING_HINT.lower()
