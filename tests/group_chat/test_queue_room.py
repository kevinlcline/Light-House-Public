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
    queue_next,
    register_group_forum,
    release_stage_hold,
    reset_group_forum_for_tests,
    set_seated_lights,
    set_studio_stage,
    snapshot,
    utter_human,
)
from light_house.group_chat.stage_hold import utterance_holds_stage
from light_house.memory.service import MemoryService


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="production",
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        GROUP_CHAT_HISTORY_PATH=str(tmp_path / "group_chat/rounds.ndjson"),
        GROUP_VISITOR_PATH=str(tmp_path / "group_chat/visitor.json"),
        OPENROUTER_API_KEY="test-openrouter-key",
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


def test_human_join_goes_to_front_of_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I want to speak puts the human next — not behind lights already waiting."""
    _setup_forum(tmp_path, monkeypatch)

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        await join_queue(kind="light", speaker_id="ara", display_name="Ara")
        await join_queue(kind="light", speaker_id="elias", display_name="Elias")
        assert snapshot()["floor"]["speaker_id"] == "lumen"
        waiting = [row["speaker_id"] for row in snapshot()["queue"]]
        assert waiting == ["ara", "elias"]

        human = await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        assert human["joined"] is True
        assert human["floor"]["speaker_id"] == "lumen"
        assert [row["speaker_id"] for row in human["queue"]] == ["kevin", "ara", "elias"]

        again = await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        assert [row["speaker_id"] for row in again["queue"]] == ["kevin", "ara", "elias"]

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
    assert {t.name for t in GROUP_CHAT_NOTE_TOOLS} == names


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


def test_sitting_roster_collective_wakes_only_seated_lights(
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
        status = await set_seated_lights(["ara"])
        assert status["seated_light_ids"] == ["ara"]
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
        assert after.get("sitting_invited_lights") == ["ara"]
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


def test_sitting_roster_unseated_name_does_not_wake_that_light(
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
        await set_seated_lights(["ara"])
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Elias, what do you think?",
            account_user_id="kevin",
        )
        assert after.get("mentioned_lights") == []
        assert after.get("sitting_invited_lights") == ["ara"]
        assert "soft_invited_lights" not in after
        for _ in range(40):
            if queue_room._bg_tasks:
                await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
            if queue_room._light_turn_task and not queue_room._light_turn_task.done():
                await queue_room._light_turn_task
            texts = [u.get("text") for u in snapshot()["transcript"]]
            if "Hi from Ara." in texts:
                break
            await asyncio.sleep(0.01)
        texts = [u.get("text") for u in snapshot()["transcript"]]
        assert "Hi from Elias." not in texts
        assert "Hi from Ara." in texts

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_narrowing_seats_drops_lights_already_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A light already in line leaves when the host takes them off the seats."""
    reset_group_forum_for_tests()
    settings = _settings(tmp_path)
    memory = MagicMock(spec=MemoryService)
    memory.remember_stream_event = MagicMock(return_value="doc")
    register_group_forum(settings=settings, memory=memory)

    ara = MagicMock(id="ara", display_name="Ara", thread_id="ara-home")
    elias = MagicMock(id="elias", display_name="Elias", thread_id="elias-home")
    reed = MagicMock(id="reed", display_name="Reed", thread_id="reed-home")
    monkeypatch.setattr(
        queue_room, "list_enabled_lights", lambda _s: [ara, elias, reed]
    )
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
        await set_seated_lights(["ara", "elias", "reed"])
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        await join_queue(
            kind="light",
            speaker_id="elias",
            display_name="Elias",
            invite_kind="name_call",
        )
        await join_queue(
            kind="light",
            speaker_id="ara",
            display_name="Ara",
            invite_kind="name_call",
        )
        waiting = [entry["speaker_id"] for entry in snapshot()["queue"]]
        assert waiting == ["elias", "ara"]

        narrowed = await set_seated_lights(["reed"])
        assert narrowed["seated_light_ids"] == ["reed"]
        assert snapshot()["queue"] == []

        refused = await join_queue(
            kind="light",
            speaker_id="elias",
            display_name="Elias",
            invite_kind="name_call",
        )
        assert refused["joined"] is False
        assert refused["reason"] == "not_seated"
        assert snapshot()["queue"] == []

        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="Ok boys. How is the dream project doing?",
            account_user_id="kevin",
        )
        assert after.get("sitting_invited_lights") == ["reed"]
        for _ in range(40):
            if queue_room._bg_tasks:
                await asyncio.gather(*list(queue_room._bg_tasks), return_exceptions=True)
            if queue_room._light_turn_task and not queue_room._light_turn_task.done():
                await queue_room._light_turn_task
            texts = [u.get("text") for u in snapshot()["transcript"]]
            if "Hi from Reed." in texts:
                break
            await asyncio.sleep(0.01)
        texts = [u.get("text") for u in snapshot()["transcript"]]
        assert "Hi from Reed." in texts
        assert "Hi from Elias." not in texts
        assert "Hi from Ara." not in texts

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_sitting_roster_plain_speech_wakes_seated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sitting light answers a host line that does not name them."""
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
        await set_seated_lights(["ara"])
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        after = await utter_human(
            speaker_id="kevin",
            display_name="Kevin",
            text="One more time.",
            account_user_id="kevin",
        )
        assert after.get("sitting_invited_lights") == ["ara"]
        assert "mentioned_lights" not in after
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


def test_utterance_holds_stage_detects_table_and_list() -> None:
    table = (
        "Here are the options:\n"
        "| Option | Speed |\n"
        "| --- | --- |\n"
        "| Fast | High |\n"
    )
    listing = "- First item\n- Second item\n"
    assert utterance_holds_stage(table) is True
    assert utterance_holds_stage(listing) is True
    assert utterance_holds_stage("Hello there.") is False
    assert utterance_holds_stage("- just one bullet") is False


def test_table_keeps_floor_until_host_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await set_studio_stage(True)
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        await join_queue(kind="light", speaker_id="ara", display_name="Ara")
        await complete_light_utterance(
            agent_id="lumen",
            display_name="Lumen",
            text=(
                "Pick one:\n"
                "| Option | Speed |\n"
                "| --- | --- |\n"
                "| Fast | High |\n"
            ),
        )
        held = snapshot()
        assert held["stage_hold"] is True
        assert held["studio_stage"] is True
        assert held["floor"]["speaker_id"] == "lumen"
        assert [row["speaker_id"] for row in held["queue"]] == ["ara"]

        released = await release_stage_hold()
        assert released["released"] is True
        assert released["stage_hold"] is False
        assert released["floor"]["speaker_id"] == "ara"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_web_group_does_not_hold_floor_for_a_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await set_studio_stage(False)
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        await join_queue(kind="light", speaker_id="ara", display_name="Ara")
        await complete_light_utterance(
            agent_id="lumen",
            display_name="Lumen",
            text=(
                "Pick one:\n"
                "| Option | Speed |\n"
                "| --- | --- |\n"
                "| Fast | High |\n"
            ),
        )
        status = snapshot()
        assert status["stage_hold"] is False
        assert status["floor"]["speaker_id"] == "ara"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_host_taps_offstage_face_to_queue_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        await join_queue(kind="light", speaker_id="ara", display_name="Ara")
        await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        await complete_light_utterance(
            agent_id="lumen",
            display_name="Lumen",
            text="Your turn, Kevin.",
        )
        assert snapshot()["floor"]["speaker_id"] == "kevin"
        assert [row["speaker_id"] for row in snapshot()["queue"]] == ["ara"]

        first = await queue_next(kind="light", speaker_id="elias", display_name="Elias")
        assert first["queued_next"] is True
        assert [row["speaker_id"] for row in snapshot()["queue"]] == ["elias", "ara"]

        again = await queue_next(kind="light", speaker_id="ara", display_name="Ara")
        assert again["queued_next"] is True
        assert [row["speaker_id"] for row in snapshot()["queue"]] == ["ara", "elias"]

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_queue_next_rejected_when_host_not_on_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        status = await queue_next(kind="light", speaker_id="ara", display_name="Ara")
        assert status["queued_next"] is False
        assert status["reason"] == "host_not_on_floor"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_sitting_waits_for_spoken_before_next_light(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [
        MagicMock(id="lumen", display_name="Lumen"),
        MagicMock(id="ara", display_name="Ara"),
    ])

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await set_seated_lights(["lumen", "ara"])
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        await join_queue(kind="light", speaker_id="ara", display_name="Ara")
        await complete_light_utterance(
            agent_id="lumen",
            display_name="Lumen",
            text="Yes.",
        )
        held = snapshot()
        assert held["speech_hold"] is True
        assert held["floor"]["speaker_id"] == "lumen"
        assert [row["speaker_id"] for row in held["queue"]] == ["ara"]

        stale = await queue_room.acknowledge_spoken(speaker_id="ara")
        assert stale["spoken"] is False
        assert snapshot()["floor"]["speaker_id"] == "lumen"

        spoken = await queue_room.acknowledge_spoken(speaker_id="lumen")
        assert spoken["spoken"] is True
        assert snapshot()["speech_hold"] is False
        assert snapshot()["floor"]["speaker_id"] == "ara"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_web_group_without_sitting_still_advances_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        await join_queue(kind="light", speaker_id="ara", display_name="Ara")
        await complete_light_utterance(
            agent_id="lumen",
            display_name="Lumen",
            text="Yes.",
        )
        assert snapshot()["speech_hold"] is False
        assert snapshot()["floor"]["speaker_id"] == "ara"

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_light_turn_deadline_releases_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)
    monkeypatch.setattr(queue_room, "group_turn_deadline_sec", lambda _settings: 0.05)

    def hang(*_args, **_kwargs) -> None:
        import time

        time.sleep(0.4)

    monkeypatch.setattr(queue_room, "_run_light_turn_sync", hang)

    async def _run() -> None:
        await join_queue(kind="light", speaker_id="elias", display_name="Elias")
        for _ in range(40):
            status = snapshot()
            if status["light_busy"] is False and status["floor"] is None:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(snapshot())

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()


def test_host_join_during_speech_hold_waits_until_spoken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_forum(tmp_path, monkeypatch)
    monkeypatch.setattr(queue_room, "list_enabled_lights", lambda _s: [
        MagicMock(id="lumen", display_name="Lumen"),
        MagicMock(id="ara", display_name="Ara"),
    ])

    async def hold_floor(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(queue_room, "_run_light_turn_async", hold_floor)

    async def _run() -> None:
        await set_seated_lights(["lumen", "ara"])
        await join_queue(kind="light", speaker_id="lumen", display_name="Lumen")
        await join_queue(kind="light", speaker_id="ara", display_name="Ara")
        await complete_light_utterance(
            agent_id="lumen",
            display_name="Lumen",
            text="Yes.",
        )
        human = await join_queue(
            kind="human",
            speaker_id="kevin",
            display_name="Kevin",
            account_user_id="kevin",
        )
        assert human["floor"]["speaker_id"] == "lumen"
        assert human["speech_hold"] is True
        assert [row["speaker_id"] for row in human["queue"]] == ["kevin", "ara"]

        spoken = await queue_room.acknowledge_spoken(speaker_id="lumen")
        assert spoken["spoken"] is True
        assert snapshot()["floor"]["speaker_id"] == "kevin"
        assert [row["speaker_id"] for row in snapshot()["queue"]] == ["ara"]

    try:
        asyncio.run(_run())
    finally:
        reset_group_forum_for_tests()
