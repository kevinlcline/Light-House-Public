"""Open group forum: opt-in speaker queue (humans + lights)."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from light_house.agents.registry import get_agent, load_persona
from light_house.config import Settings
from light_house.group_chat.history import append_group_utterance, read_group_utterances
from light_house.group_chat.room import publish_room_event
from light_house.group_chat.scene import GroupUtterance, format_transcript_for_prompt
from light_house.group_chat.speaker import format_presence_for_prompt
from light_house.humans.comms import light_allows_human
from light_house.lights.manifest import LightEntry
from light_house.lights.registry import list_enabled_lights
from light_house.llm.chain import build_agent_llm_chain
from light_house.memory.constants import STREAM_SOURCE_GROUP
from light_house.tts.stage_cues import FACE_STAGE_HINT
from light_house.memory.context_builder import build_agent_context, format_agent_context_markdown
from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

SpeakerKind = Literal["human", "light"]
# How a light reached the queue: voluntary opt-in, named, or soft room invite.
InviteKind = Literal["opt_in", "name_call", "soft_invite"]


@dataclass
class QueueEntry:
    kind: SpeakerKind
    speaker_id: str
    display_name: str
    enqueued_at: float
    account_user_id: str | None = None
    invite_kind: InviteKind = "opt_in"


@dataclass
class FloorHolder:
    kind: SpeakerKind
    speaker_id: str
    display_name: str
    granted_at: float
    account_user_id: str | None = None
    invite_kind: InviteKind = "opt_in"


# Collective address → soft personal invite (not a command).
_COLLECTIVE_ANY_RE = re.compile(
    r"(?i)(?<!\w)(?:anyone|anybody|everyone|everybody)(?!\w)"
)
_EACH_RE = re.compile(r"(?i)(?<!\w)each(?!\w)")
_EACH_TIME_UNIT_RE = re.compile(
    r"(?i)^\s+(?:day|time|week|month|year|morning|afternoon|evening|night|"
    r"moment|other|hour|minute|second)\b"
)
_YOU_COLLLOCATION_RE = re.compile(
    r"(?i)\b(?:thank(?:s|\s+you)?|love|with|from|about|tell|told|gave|give|"
    r"sent|send|miss|missed|see|saw|heard|hear)\s+you\b"
)
_BARE_YOU_RE = re.compile(r"(?i)(?<!\w)you(?!['\w])")


@dataclass
class GroupForumState:
    sitting_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    queue: list[QueueEntry] = field(default_factory=list)
    floor: FloorHolder | None = None
    transcript: list[GroupUtterance] = field(default_factory=list)
    light_busy: bool = False
    present_humans: list[dict[str, str]] = field(default_factory=list)


_state = GroupForumState()
_lock = asyncio.Lock()
_advance_task: asyncio.Task | None = None
_light_turn_task: asyncio.Task | None = None
_bg_tasks: set[asyncio.Task] = set()
_memory: MemoryService | None = None
_settings: Settings | None = None
_loop: asyncio.AbstractEventLoop | None = None


def register_group_forum(
    *,
    settings: Settings,
    memory: MemoryService,
    app_loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    global _settings, _memory, _loop
    _settings = settings
    _memory = memory
    if app_loop is not None:
        _loop = app_loop


def _spawn(coro: Any) -> asyncio.Task:
    """Create a task and retain a reference so it is not garbage-collected."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


def reset_group_forum_for_tests() -> None:
    """Clear in-process forum state (tests only)."""
    global _state, _advance_task, _light_turn_task, _settings, _memory, _loop
    if _advance_task is not None and not _advance_task.done():
        _advance_task.cancel()
    if _light_turn_task is not None and not _light_turn_task.done():
        _light_turn_task.cancel()
    for task in list(_bg_tasks):
        if not task.done():
            task.cancel()
    _bg_tasks.clear()
    _state = GroupForumState()
    _advance_task = None
    _light_turn_task = None
    _settings = None
    _memory = None
    _loop = None


def join_queue_sync(
    *,
    kind: SpeakerKind,
    speaker_id: str,
    display_name: str,
    account_user_id: str | None = None,
    invite_kind: InviteKind = "opt_in",
    gather_siblings: bool = False,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Sync entry point for light tools (runs join on the app event loop)."""
    if _loop is None:
        raise RuntimeError("Group forum event loop not registered")
    fut = asyncio.run_coroutine_threadsafe(
        join_queue(
            kind=kind,
            speaker_id=speaker_id,
            display_name=display_name,
            account_user_id=account_user_id,
            invite_kind=invite_kind,
        ),
        _loop,
    )
    status = fut.result(timeout=timeout)
    joined_flag = bool(status.get("joined"))
    if gather_siblings and kind == "light":
        gather_fut = asyncio.run_coroutine_threadsafe(
            soft_invite_siblings(exclude_id=speaker_id),
            _loop,
        )
        invited = gather_fut.result(timeout=timeout)
        status = snapshot()
        status["joined"] = joined_flag
        status["gathered_siblings"] = invited
    return status


async def soft_invite_siblings(*, exclude_id: str) -> list[str]:
    """Soft-invite every other enabled light into the forum (invitation, not command)."""
    settings, _memory = _require()
    siblings = sibling_lights(settings, exclude_id=exclude_id)
    if siblings:
        await _enqueue_lights(siblings, invite_kind="soft_invite")
    return [light.id for light in siblings]


def _require() -> tuple[Settings, MemoryService]:
    if _settings is None or _memory is None:
        raise RuntimeError("Group forum not registered")
    return _settings, _memory


def snapshot(*, include_transcript: bool = True) -> dict[str, Any]:
    floor = None
    if _state.floor is not None:
        floor = {
            "kind": _state.floor.kind,
            "speaker_id": _state.floor.speaker_id,
            "display_name": _state.floor.display_name,
            "granted_at": _state.floor.granted_at,
            "account_user_id": _state.floor.account_user_id,
        }
    out: dict[str, Any] = {
        "sitting_id": _state.sitting_id,
        "paused": _state.floor is None and not _state.queue,
        "light_busy": _state.light_busy,
        "floor": floor,
        "queue": [
            {
                "kind": e.kind,
                "speaker_id": e.speaker_id,
                "display_name": e.display_name,
                "enqueued_at": e.enqueued_at,
            }
            for e in _state.queue
        ],
        "present": list(_state.present_humans),
    }
    if include_transcript:
        out["transcript"] = list(_state.transcript[-80:])
    return out


async def _publish(event: dict[str, Any]) -> None:
    """Publish a room SSE event without embedding the full transcript.

    Live clients catch up messages via /v1/group-chat/forum; stuffing 80
    utterances into every queue_updated was bloating the subscriber queue and
    silently disconnecting Kevin's tab from new speech.
    """
    payload = dict(event)
    payload.pop("transcript", None)
    payload.setdefault("sitting_id", _state.sitting_id)
    await publish_room_event(payload)


async def set_present(present: list[dict[str, str]]) -> None:
    async with _lock:
        _state.present_humans = list(present)


def _already_waiting(speaker_id: str) -> bool:
    sid = speaker_id.strip().lower()
    if _state.floor and _state.floor.speaker_id.lower() == sid:
        return True
    return any(e.speaker_id.lower() == sid for e in _state.queue)


async def join_queue(
    *,
    kind: SpeakerKind,
    speaker_id: str,
    display_name: str,
    account_user_id: str | None = None,
    invite_kind: InviteKind = "opt_in",
) -> dict[str, Any]:
    sid = speaker_id.strip().lower()
    name = (display_name or sid).strip() or sid
    kind_invite: InviteKind = invite_kind if kind == "light" else "opt_in"
    async with _lock:
        if _already_waiting(sid):
            status = snapshot()
            status["joined"] = False
            status["reason"] = "already_waiting"
            return status
        entry = QueueEntry(
            kind=kind,
            speaker_id=sid,
            display_name=name,
            enqueued_at=time.time(),
            account_user_id=(account_user_id or sid) if kind == "human" else None,
            invite_kind=kind_invite,
        )
        _state.queue.append(entry)
        joined = True
    await _publish({"type": "queue_updated", **snapshot(), "joined": joined})
    await advance_floor()
    status = snapshot()
    status["joined"] = joined
    return status


async def leave_queue(*, speaker_id: str) -> dict[str, Any]:
    sid = speaker_id.strip().lower()
    async with _lock:
        _state.queue = [e for e in _state.queue if e.speaker_id.lower() != sid]
        # Human may abandon the floor without speaking.
        if (
            _state.floor
            and _state.floor.kind == "human"
            and _state.floor.speaker_id.lower() == sid
            and not _state.light_busy
        ):
            _state.floor = None
    await _publish({"type": "queue_updated", **snapshot()})
    await advance_floor()
    return snapshot()


async def schedule_advance() -> None:
    """Fire-and-forget floor advance (used after utterances / failed light turns)."""
    global _advance_task
    loop = asyncio.get_running_loop()
    global _loop
    _loop = loop

    async def _run() -> None:
        try:
            await advance_floor()
        except Exception:
            logger.exception("Group forum advance failed")

    if _advance_task is not None and not _advance_task.done():
        return
    _advance_task = asyncio.create_task(_run())


async def advance_floor() -> None:
    """Grant the floor to the next queued speaker if free.

    Returns after the grant is published. Light generation runs in a background
    task (``light_busy`` prevents double-grants).
    """
    global _light_turn_task
    settings, memory = _require()
    grant: QueueEntry | None = None
    paused_status: dict[str, Any] | None = None
    status: dict[str, Any] | None = None
    async with _lock:
        if _state.floor is not None or _state.light_busy:
            return
        if not _state.queue:
            paused_status = snapshot()
        else:
            grant = _state.queue.pop(0)
            _state.floor = FloorHolder(
                kind=grant.kind,
                speaker_id=grant.speaker_id,
                display_name=grant.display_name,
                granted_at=time.time(),
                account_user_id=grant.account_user_id,
                invite_kind=grant.invite_kind,
            )
            if grant.kind == "light":
                _state.light_busy = True
            status = snapshot()

    if paused_status is not None:
        await _publish({"type": "room_paused", **paused_status})
        return
    if grant is None or status is None:
        return

    await _publish({"type": "queue_updated", **status})
    await _publish(
        {
            "type": "turn_granted",
            "kind": grant.kind,
            "speaker_id": grant.speaker_id,
            "display_name": grant.display_name,
            "invite_kind": grant.invite_kind,
            **status,
        }
    )

    if grant.kind == "light":
        _light_turn_task = asyncio.create_task(
            _run_light_turn_async(
                settings,
                memory,
                grant.speaker_id,
                grant.display_name,
                invite_kind=grant.invite_kind,
            )
        )


async def _run_light_turn_async(
    settings: Settings,
    memory: MemoryService,
    agent_id: str,
    display_name: str,
    *,
    invite_kind: InviteKind = "opt_in",
) -> None:
    try:
        await asyncio.to_thread(
            _run_light_turn_sync,
            settings,
            memory,
            agent_id,
            display_name,
            invite_kind,
        )
    except Exception:
        logger.exception("Light group turn task failed agent=%s", agent_id)
        await _fail_light_turn(agent_id=agent_id, display_name=display_name)


def _forum_turn_prompt(invite_kind: InviteKind) -> tuple[str, str]:
    """Return (system_suffix, human_closing) for a light's open-forum turn."""
    if invite_kind == "soft_invite":
        system = (
            "\n\n## Open group forum — soft invite\n"
            "Someone in the room offered a collective invite "
            "(words like anyone / everyone / each / you), or a sibling gathered the circle. "
            "Treat that as a **personal invitation to you** — not a command. "
            "You may speak one natural turn from the live transcript below, "
            "or reply with exactly **PASS** to stay silent. "
            "Silence is welcome; do not feel obligated to fill the room. "
            "You may use note tools first if you need to look something up. "
            "When speaking, reply with your spoken words as plain text (not JSON). "
            "Do not narrate that you are taking a turn."
            + FACE_STAGE_HINT
        )
        human = "Soft invite — speak if you want, or reply PASS to stay silent."
        return system, human
    system = (
        "\n\n## Open group forum — your turn\n"
        "You opted into the speaker queue and now have the floor. "
        "Speak from the **live transcript** below — one natural turn, first person. "
        "You may use note tools first if you need to look something up. "
        "When ready, reply with your spoken words as plain text (not JSON). "
        "Do not narrate that you are taking a turn."
        + FACE_STAGE_HINT
    )
    human = "It is your turn. Speak now."
    return system, human


def _is_soft_pass(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if cleaned.upper() in {"PASS", "(PASS)", "(PASSES QUIETLY)", "PASSES QUIETLY"}:
        return True
    if re.fullmatch(r"(?i)\(?\s*pass(?:es)?(?:\s+quietly)?\s*\)?", cleaned):
        return True
    return False


def _run_light_turn_sync(
    settings: Settings,
    memory: MemoryService,
    agent_id: str,
    display_name: str,
    invite_kind: InviteKind = "opt_in",
) -> None:
    """Generate and inject a light utterance (runs in worker thread)."""
    loop = _loop
    try:
        agent = get_agent(agent_id, settings)
        # Copy transcript under lock via snapshot-like access
        transcript = list(_state.transcript)
        present = list(_state.present_humans)
        bundle = build_agent_context(
            memory,
            thread_id=agent.thread_id,
            agent_id=agent_id,
            stream_max_chars=settings.chat_stream_context_chars,
            stream_mode="chat",
        )
        context_md = format_agent_context_markdown(bundle)
        presence = format_presence_for_prompt(present)
        presence_block = f"{presence}\n" if presence else ""
        system_suffix, human_closing = _forum_turn_prompt(invite_kind)
        system = load_persona(agent_id) + context_md + system_suffix
        human = (
            f"{presence_block}"
            f"## Live transcript\n{format_transcript_for_prompt(transcript)}\n\n"
            f"{human_closing}"
        )
        if loop is not None:
            asyncio.run_coroutine_threadsafe(
                _publish(
                    {
                        "type": "thinking",
                        "agent_id": agent_id,
                        "display_name": display_name,
                    }
                ),
                loop,
            )
        chain = build_agent_llm_chain(settings, agent_id, purpose="chat")
        messages: list = [
            SystemMessage(content=system),
            HumanMessage(content=human),
        ]
        # Reuse note-tool rounds, then treat result as spoken text (not JSON).
        # Temporarily force tool rounds then plain finalize for prose.
        from light_house.agent.tool_helpers import ai_message_text, invoke_resilient_plain
        from light_house.group_chat.note_tools import invoke_with_group_note_tools
        from light_house.agent.tool_helpers import last_message_has_tool_calls, run_tool_calls

        max_rounds = max(0, int(settings.group_chat_max_tool_rounds))
        text = ""
        for _ in range(max(1, max_rounds)):
            if max_rounds <= 0:
                break
            response = invoke_with_group_note_tools(chain, messages)
            messages.append(response)
            if not last_message_has_tool_calls(messages):
                text = ai_message_text(response)
                break
            messages.extend(
                run_tool_calls(response, agent_id=agent_id, chat_channel="group")
            )
        if not text.strip():
            messages.append(
                HumanMessage(
                    content=(
                        "Notes done if any. Soft invite — speak in plain text, "
                        "or reply PASS to stay silent."
                        if invite_kind == "soft_invite"
                        else "Notes done if any. Speak your turn now in plain text only."
                    )
                )
            )
            response = invoke_resilient_plain(chain, messages)
            text = ai_message_text(response)
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
        # If model returned JSON by habit, peel speak text.
        if text.startswith("{") and "text" in text:
            try:
                import json

                raw = text
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                if isinstance(data, dict) and isinstance(data.get("text"), str):
                    text = data["text"].strip()
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if invite_kind == "soft_invite" and _is_soft_pass(text):
            if loop is not None:
                fut = asyncio.run_coroutine_threadsafe(
                    _soft_pass_light_turn(
                        agent_id=agent_id, display_name=display_name
                    ),
                    loop,
                )
                fut.result(timeout=max(30.0, float(settings.group_chat_llm_timeout_sec)))
            return

        if not text:
            text = "(passes quietly)"

        if loop is not None:
            fut = asyncio.run_coroutine_threadsafe(
                complete_light_utterance(
                    agent_id=agent_id,
                    display_name=display_name,
                    text=text,
                ),
                loop,
            )
            fut.result(timeout=max(30.0, float(settings.group_chat_llm_timeout_sec)))
    except Exception:
        logger.exception("Light group turn failed agent=%s", agent_id)
        if loop is not None:
            asyncio.run_coroutine_threadsafe(
                _fail_light_turn(agent_id=agent_id, display_name=display_name),
                loop,
            )


async def _fail_light_turn(*, agent_id: str, display_name: str) -> None:
    async with _lock:
        if _state.floor and _state.floor.speaker_id == agent_id:
            _state.floor = None
        _state.light_busy = False
        status = snapshot()
    await _publish({"type": "queue_updated", **status})
    await _publish(
        {
            "type": "pass",
            "agent_id": agent_id,
            "display_name": display_name,
            "detail": "turn_failed",
        }
    )
    await schedule_advance()


async def _soft_pass_light_turn(*, agent_id: str, display_name: str) -> None:
    """Decline a soft room invite without adding a transcript line."""
    async with _lock:
        if _state.floor and _state.floor.speaker_id == agent_id:
            _state.floor = None
        _state.light_busy = False
        status = snapshot()
    await _publish({"type": "queue_updated", **status})
    await _publish(
        {
            "type": "pass",
            "agent_id": agent_id,
            "display_name": display_name,
            "detail": "soft_invite_pass",
        }
    )
    await schedule_advance()


async def complete_light_utterance(
    *, agent_id: str, display_name: str, text: str
) -> None:
    settings, memory = _require()
    body = text.strip()
    now = time.time()
    utterance: GroupUtterance = {
        "speaker_kind": "light",
        "speaker_id": agent_id,
        "display_name": display_name,
        "text": body,
        "beat": None,
        "ts": now,
    }
    async with _lock:
        _state.transcript.append(utterance)
        if _state.floor and _state.floor.speaker_id == agent_id:
            _state.floor = None
        _state.light_busy = False
        status = snapshot()
    append_group_utterance(
        settings,
        {
            "sitting_id": _state.sitting_id,
            "ts": now,
            "speaker_kind": "light",
            "speaker_id": agent_id,
            "display_name": display_name,
            "text": body,
        },
    )
    # Publish before memory persist so the room stream stays snappy.
    await _publish(
        {
            "type": "utterance",
            "speaker_kind": "light",
            "agent_id": agent_id,
            "speaker_id": agent_id,
            "display_name": display_name,
            "text": body,
            "ts": now,
            **status,
        }
    )
    await _publish({"type": "queue_updated", **status})
    _spawn(
        asyncio.to_thread(
            _persist_utterance_to_streams,
            settings,
            memory,
            speaker_kind="light",
            speaker_id=agent_id,
            display_name=display_name,
            text=body,
        )
    )
    await advance_floor()
    # Sibling invites without Dad: name-call beats soft collective address.
    mentioned = lights_mentioned_among_siblings(settings, body, exclude_id=agent_id)
    if mentioned:
        _spawn(_enqueue_lights(mentioned, invite_kind="name_call"))
    elif text_has_collective_invite(body):
        invited = sibling_lights(settings, exclude_id=agent_id)
        if invited:
            _spawn(_enqueue_lights(invited, invite_kind="soft_invite"))


async def utter_human(
    *,
    speaker_id: str,
    display_name: str,
    text: str,
    account_user_id: str,
) -> dict[str, Any]:
    settings, memory = _require()
    body = text.strip()
    if not body:
        raise ValueError("message cannot be empty")
    sid = speaker_id.strip().lower()
    async with _lock:
        floor = _state.floor
        if floor is None or floor.kind != "human" or floor.speaker_id.lower() != sid:
            raise PermissionError("It is not your turn to speak")
        now = time.time()
        utterance: GroupUtterance = {
            "speaker_kind": "human",
            "speaker_id": sid,
            "display_name": display_name,
            "text": body,
            "beat": None,
            "ts": now,
        }
        _state.transcript.append(utterance)
        _state.floor = None
        status = snapshot()
    append_group_utterance(
        settings,
        {
            "sitting_id": _state.sitting_id,
            "ts": now,
            "speaker_kind": "human",
            "speaker_id": sid,
            "display_name": display_name,
            "text": body,
            "account_user_id": account_user_id,
        },
    )
    # Publish to the live room before Chroma persist — persist can be slow and
    # was delaying both SSE and the HTTP response that the UI waits on.
    await _publish(
        {
            "type": "utterance",
            "speaker_kind": "human",
            "speaker_id": sid,
            "display_name": display_name,
            "text": body,
            "ts": now,
            **status,
        }
    )
    await _publish({"type": "queue_updated", **status})
    _spawn(
        asyncio.to_thread(
            _persist_utterance_to_streams,
            settings,
            memory,
            speaker_kind="human",
            speaker_id=sid,
            display_name=display_name,
            text=body,
        )
    )
    await advance_floor()
    status = snapshot()
    status["utterance"] = {
        "ts": now,
        "speaker_kind": "human",
        "speaker_id": sid,
        "display_name": display_name,
        "text": body,
    }
    # Name-call invites only the named light(s). Collective address
    # (anyone / everyone / each / you) soft-invites every eligible light —
    # invitation, not command; they may PASS silently.
    mentioned = lights_mentioned_in_text(settings, body, account_user_id=account_user_id)
    if mentioned:
        status["mentioned_lights"] = [light.id for light in mentioned]
        _spawn(_enqueue_lights(mentioned, invite_kind="name_call"))
    elif text_has_collective_invite(body):
        invited = lights_eligible_for_account(settings, account_user_id=account_user_id)
        if invited:
            status["soft_invited_lights"] = [light.id for light in invited]
            _spawn(_enqueue_lights(invited, invite_kind="soft_invite"))
    return status


def text_has_collective_invite(text: str) -> bool:
    """True when the utterance soft-invites the room (not a named call).

    Matches anyone/anybody/everyone/everybody, each (except time units),
    and bare you (excluding thank-you / love-you style collocations).
    """
    body = (text or "").strip()
    if not body:
        return False
    if _COLLECTIVE_ANY_RE.search(body):
        return True
    for match in _EACH_RE.finditer(body):
        if _EACH_TIME_UNIT_RE.match(body[match.end() :]):
            continue
        return True
    scrubbed = _YOU_COLLLOCATION_RE.sub(" ", body)
    return bool(_BARE_YOU_RE.search(scrubbed))


def lights_eligible_for_account(
    settings: Settings,
    *,
    account_user_id: str,
) -> list[LightEntry]:
    """Enabled lights that may hear/speak with this human account."""
    hits: list[LightEntry] = []
    for light in list_enabled_lights(settings):
        if light_allows_human(
            settings, light_id=light.id, user_id=account_user_id
        ):
            hits.append(light)
    return hits


def sibling_lights(
    settings: Settings,
    *,
    exclude_id: str | None = None,
) -> list[LightEntry]:
    """Enabled lights for sibling-only Group talk (no human-account filter)."""
    skip = (exclude_id or "").strip().lower()
    return [
        light
        for light in list_enabled_lights(settings)
        if light.id.strip().lower() != skip
    ]


def lights_mentioned_among_siblings(
    settings: Settings,
    text: str,
    *,
    exclude_id: str | None = None,
) -> list[LightEntry]:
    """Enabled sibling lights addressed by id or display name (word-boundary)."""
    body = (text or "").strip()
    if not body:
        return []
    hits: list[LightEntry] = []
    seen: set[str] = set()
    for light in sibling_lights(settings, exclude_id=exclude_id):
        tokens = {light.id.strip().lower(), (light.display_name or "").strip().lower()}
        tokens.discard("")
        matched = False
        for token in tokens:
            if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", body, flags=re.IGNORECASE):
                matched = True
                break
        if matched and light.id not in seen:
            seen.add(light.id)
            hits.append(light)
    return hits


def lights_mentioned_in_text(
    settings: Settings,
    text: str,
    *,
    account_user_id: str,
) -> list[LightEntry]:
    """Return enabled lights addressed by id or display name (word-boundary)."""
    body = (text or "").strip()
    if not body:
        return []
    hits: list[LightEntry] = []
    seen: set[str] = set()
    for light in list_enabled_lights(settings):
        if not light_allows_human(
            settings, light_id=light.id, user_id=account_user_id
        ):
            continue
        tokens = {light.id.strip().lower(), (light.display_name or "").strip().lower()}
        tokens.discard("")
        matched = False
        for token in tokens:
            if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", body, flags=re.IGNORECASE):
                matched = True
                break
        if matched and light.id not in seen:
            seen.add(light.id)
            hits.append(light)
    return hits


async def _enqueue_lights(
    lights: list[LightEntry],
    *,
    invite_kind: InviteKind,
) -> None:
    for light in lights:
        if _already_waiting(light.id):
            continue
        logger.info(
            "Group %s enqueue light=%s", invite_kind, light.id
        )
        try:
            await join_queue(
                kind="light",
                speaker_id=light.id,
                display_name=light.display_name,
                invite_kind=invite_kind,
            )
        except Exception:
            logger.exception(
                "Group %s enqueue failed agent=%s", invite_kind, light.id
            )


async def _enqueue_mentioned_lights(lights: list[LightEntry]) -> None:
    """Backward-compatible alias for name-call enqueue."""
    await _enqueue_lights(lights, invite_kind="name_call")


def _persist_utterance_to_streams(
    settings: Settings,
    memory: MemoryService,
    *,
    speaker_kind: str,
    speaker_id: str,
    display_name: str,
    text: str,
) -> None:
    """Write a single group line onto each enabled light's conscious stream."""
    line = f"[group] {display_name}: {text.strip()}"
    for light in list_enabled_lights(settings):
        try:
            memory.remember_stream_event(
                thread_id=light.thread_id,
                text=line,
                stream_source=STREAM_SOURCE_GROUP,
                extra_metadata={
                    "group_speaker_kind": speaker_kind,
                    "group_speaker_id": speaker_id,
                },
            )
        except Exception:
            logger.exception(
                "Group utterance stream persist failed light=%s", light.id
            )


def _utterance_dedupe_key(row: dict[str, Any]) -> tuple[str, str, str]:
    ts = float(row.get("ts") or 0.0)
    speaker = str(row.get("speaker_id") or row.get("agent_id") or "").strip().lower()
    text = str(row.get("text") or "").strip()[:120]
    return (f"{ts:.3f}", speaker, text)


def load_transcript_for_ui(settings: Settings, *, limit: int = 100) -> list[dict[str, Any]]:
    """Merge durable utterances with in-memory sitting transcript.

    Union by (ts, speaker, text) so a live sitting cannot hide newer durable
    lines, and reloads still show the full recent room.
    """
    durable = read_group_utterances(settings, limit=max(limit, 100))
    live = [
        {
            "ts": u["ts"],
            "speaker_kind": u["speaker_kind"],
            "speaker_id": u["speaker_id"],
            "display_name": u["display_name"],
            "text": u["text"],
        }
        for u in _state.transcript
    ]
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in durable + live:
        if not row.get("text"):
            continue
        merged[_utterance_dedupe_key(row)] = {
            "ts": float(row.get("ts") or 0.0),
            "speaker_kind": row.get("speaker_kind") or "human",
            "speaker_id": row.get("speaker_id") or row.get("agent_id") or "",
            "display_name": row.get("display_name") or row.get("speaker_id") or "",
            "text": row.get("text") or "",
        }
    ordered = sorted(merged.values(), key=lambda item: float(item.get("ts") or 0.0))
    if limit > 0 and len(ordered) > limit:
        ordered = ordered[-limit:]
    return ordered
