"""Orchestrate Kevin-gated group chat rounds / sequential scenes."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from light_house.agents.registry import get_agent, load_persona
from light_house.config import Settings
from light_house.group_chat.history import append_group_round
from light_house.group_chat.note_tools import run_group_note_tool_rounds
from light_house.group_chat.prompts import GROUP_CHAT_PARALLEL_HINT, GroupChatDecision
from light_house.group_chat.scene import (
    _parse_decision,
    build_group_scene_graph,
    initial_scene_state,
    parse_speak_order,
)
from light_house.group_chat.speaker import (
    format_current_speaker_for_prompt,
    format_presence_for_prompt,
)
from light_house.lights.registry import list_enabled_lights
from light_house.humans.comms import light_allows_human
from light_house.llm.chain import build_agent_llm_chain
from light_house.memory.context_builder import build_agent_context, format_agent_context_markdown
from light_house.memory.service import GroupChatLightResponse, MemoryService

logger = logging.getLogger(__name__)

_ROUND_LOCK = threading.Lock()

SceneEventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class GroupRoundResult:
    round_id: str
    kevin_message: str
    responses: list[GroupChatLightResponse]
    persisted: bool
    ts: float
    close_reason: str = ""
    mode: str = "sequential"
    human_id: str = "kevin"
    human_display_name: str = "Kevin"


def decide_for_light(
    *,
    settings: Settings,
    memory: MemoryService,
    agent_id: str,
    kevin_message: str,
    round_id: str,
    human_id: str = "kevin",
    human_display_name: str = "Kevin",
    present_humans: list[dict[str, str]] | None = None,
) -> GroupChatLightResponse:
    """One light chooses speak or silent (isolated — parallel mode)."""
    agent = get_agent(agent_id, settings)
    bundle = build_agent_context(
        memory,
        thread_id=agent.thread_id,
        agent_id=agent_id,
        stream_max_chars=settings.chat_stream_context_chars,
        stream_mode="chat",
    )
    context_md = format_agent_context_markdown(bundle)
    system = load_persona(agent_id) + context_md + GROUP_CHAT_PARALLEL_HINT
    presence = format_presence_for_prompt(present_humans)
    presence_block = f"{presence}\n" if presence else ""
    speaker = format_current_speaker_for_prompt(
        human_id=human_id,
        human_display_name=human_display_name,
        present_humans=present_humans,
    )
    speaker_block = f"{speaker}\n" if speaker else ""
    human = (
        f"## Round {round_id.split('-')[0]}\n"
        f"{presence_block}"
        f"{speaker_block}"
        f"**{human_display_name}** said:\n{kevin_message.strip()}\n\n"
        "Use note tools if you need to look something up, then decide as JSON."
    )
    chain = build_agent_llm_chain(settings, agent_id, purpose="chat")
    try:
        messages: list = [
            SystemMessage(content=system),
            HumanMessage(content=human),
        ]
        raw = run_group_note_tool_rounds(
            settings=settings,
            llm_chain=chain,
            messages=messages,
            agent_id=agent_id,
        )
        decision = _parse_decision(raw)
    except Exception:
        logger.exception(
            "Group chat decision failed agent=%s round=%s; treating as silent",
            agent_id,
            round_id,
        )
        decision = GroupChatDecision(action="pass", text="")

    spoke = bool(decision.spoke and decision.text.strip())
    text = decision.text.strip() if spoke else ""
    return GroupChatLightResponse(
        agent_id=agent_id,
        display_name=agent.display_name,
        spoke=spoke,
        text=text,
        beat=None,
        action="speak" if spoke else "pass",
    )


async def _decide_with_timeout(
    *,
    settings: Settings,
    memory: MemoryService,
    agent_id: str,
    kevin_message: str,
    round_id: str,
    human_id: str = "kevin",
    human_display_name: str = "Kevin",
    present_humans: list[dict[str, str]] | None = None,
) -> GroupChatLightResponse:
    timeout = max(5.0, float(settings.group_chat_llm_timeout_sec))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                decide_for_light,
                settings=settings,
                memory=memory,
                agent_id=agent_id,
                kevin_message=kevin_message,
                round_id=round_id,
                human_id=human_id,
                human_display_name=human_display_name,
                present_humans=present_humans,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning(
            "Group chat decision timed out agent=%s round=%s; treating as silent",
            agent_id,
            round_id,
        )
        agent = get_agent(agent_id, settings)
        return GroupChatLightResponse(
            agent_id=agent_id,
            display_name=agent.display_name,
            spoke=False,
            text="",
            beat=None,
            action="pass",
        )


def _responses_from_scene(final_state: dict[str, Any]) -> list[GroupChatLightResponse]:
    out: list[GroupChatLightResponse] = []
    for row in final_state.get("responses") or []:
        if not isinstance(row, dict):
            continue
        out.append(
            GroupChatLightResponse(
                agent_id=str(row.get("agent_id", "")),
                display_name=str(row.get("display_name", "")),
                spoke=bool(row.get("spoke")),
                text=str(row.get("text", "")),
                beat=row.get("beat") if isinstance(row.get("beat"), int) else None,
                action=str(row.get("action") or ("speak" if row.get("spoke") else "pass")),
            )
        )
    return out


def _persist_and_record(
    *,
    settings: Settings,
    memory: MemoryService,
    round_id: str,
    ts: float,
    message: str,
    responses: list[GroupChatLightResponse],
    lights: list,
    close_reason: str,
    mode: str,
    human_id: str,
    human_display_name: str,
) -> bool:
    persisted = memory.persist_group_chat_round(
        round_id=round_id,
        kevin_message=message,
        responses=responses,
        lights=lights,
        human_id=human_id,
        human_display_name=human_display_name,
    )
    record = {
        "round_id": round_id,
        "ts": ts,
        "kevin": message,
        "human_id": human_id,
        "human_display_name": human_display_name,
        "speaker_kind": "human",
        "speaker_id": human_id,
        "mode": mode,
        "close_reason": close_reason,
        "responses": [
            {
                "agent_id": r.agent_id,
                "display_name": r.display_name,
                "spoke": r.spoke,
                "text": r.text,
                "beat": r.beat,
                "action": r.action,
                "speaker_kind": "light",
                "speaker_id": r.agent_id,
            }
            for r in responses
        ],
        "persisted": persisted,
    }
    append_group_round(settings, record)
    return persisted


async def run_group_chat_sequential(
    *,
    settings: Settings,
    memory: MemoryService,
    kevin_message: str,
    on_event: SceneEventCallback | None = None,
    human_id: str = "kevin",
    human_display_name: str = "Kevin",
    account_user_id: str | None = None,
    present_humans: list[dict[str, str]] | None = None,
) -> GroupRoundResult:
    """Run one sequential LangGraph scene; lights hear earlier speech."""
    message = kevin_message.strip()
    if not message:
        raise ValueError("message cannot be empty")

    lights = list_enabled_lights(settings)
    if not lights:
        raise ValueError("No enabled lights for group chat")

    # ACL uses the logged-in account, not a guest speak-as id.
    acl_user_id = (account_user_id or human_id).strip()
    roster = list(present_humans or [])
    if not roster:
        roster = [{"speaker_id": human_id, "display_name": human_display_name}]

    # Sibling-triggered scenes: only lights that allow this human.
    lights = [
        light
        for light in lights
        if light_allows_human(settings, light_id=light.id, user_id=acl_user_id)
    ]
    if not lights:
        round_id = str(uuid.uuid4())
        ts = time.time()
        _emit = on_event or (lambda _e: None)
        _emit(
            {
                "type": "scene_started",
                "round_id": round_id,
                "ts": ts,
                "human_id": human_id,
                "human_display_name": human_display_name,
                "kevin": message,
                "participants": [],
                "present": roster,
            }
        )
        _emit(
            {
                "type": "human",
                "speaker_id": human_id,
                "display_name": human_display_name,
                "text": message,
            }
        )
        _emit(
            {
                "type": "scene_done",
                "round_id": round_id,
                "persisted": False,
                "close_reason": "no_willing_lights",
                "responses": [],
            }
        )
        return GroupRoundResult(
            round_id=round_id,
            kevin_message=message,
            responses=[],
            persisted=False,
            ts=ts,
            close_reason="no_willing_lights",
            mode="sequential",
            human_id=human_id,
            human_display_name=human_display_name,
        )

    enabled_ids = [light.id for light in lights]
    light_ids = parse_speak_order(settings, enabled_ids)
    round_id = str(uuid.uuid4())
    ts = time.time()

    logger.info(
        "Group chat scene starting round_id=%s mode=sequential participants=%s present=%s",
        round_id,
        light_ids,
        [p.get("display_name") for p in roster],
    )

    def _emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    with _ROUND_LOCK:
        _emit(
            {
                "type": "scene_started",
                "round_id": round_id,
                "ts": ts,
                "human_id": human_id,
                "human_display_name": human_display_name,
                "kevin": message,
                "participants": light_ids,
                "present": roster,
            }
        )
        _emit(
            {
                "type": "human",
                "speaker_id": human_id,
                "display_name": human_display_name,
                "text": message,
            }
        )

        graph = build_group_scene_graph(
            settings=settings,
            memory=memory,
            on_event=on_event,
        )
        init = initial_scene_state(
            settings=settings,
            scene_id=round_id,
            human_id=human_id,
            human_display_name=human_display_name,
            human_message=message,
            light_ids=light_ids,
            present_humans=roster,
        )

        # Recursion limit: beats ≈ max_utterances + consecutive passes * lights
        max_beats = (
            max(1, settings.group_chat_max_utterances)
            + max(1, settings.group_chat_max_consecutive_passes) * max(1, len(light_ids))
            + 2
        )
        final_state = await asyncio.to_thread(
            graph.invoke,
            init,
            {"recursion_limit": max(25, max_beats)},
        )

        responses = _responses_from_scene(final_state)
        close_reason = str(final_state.get("close_reason") or "done")
        persisted = _persist_and_record(
            settings=settings,
            memory=memory,
            round_id=round_id,
            ts=ts,
            message=message,
            responses=responses,
            lights=lights,
            close_reason=close_reason,
            mode="sequential",
            human_id=human_id,
            human_display_name=human_display_name,
        )

        _emit(
            {
                "type": "scene_done",
                "round_id": round_id,
                "persisted": persisted,
                "close_reason": close_reason,
                "responses": [
                    {
                        "agent_id": r.agent_id,
                        "display_name": r.display_name,
                        "spoke": r.spoke,
                        "text": r.text,
                        "beat": r.beat,
                        "action": r.action,
                    }
                    for r in responses
                ],
            }
        )

    logger.info(
        "Group chat scene complete round_id=%s spoke=%d persisted=%s close=%s",
        round_id,
        sum(1 for r in responses if r.spoke),
        persisted,
        close_reason,
    )
    return GroupRoundResult(
        round_id=round_id,
        kevin_message=message,
        responses=responses,
        persisted=persisted,
        ts=ts,
        close_reason=close_reason,
        mode="sequential",
        human_id=human_id,
        human_display_name=human_display_name,
    )


async def run_group_chat_parallel(
    *,
    settings: Settings,
    memory: MemoryService,
    kevin_message: str,
    human_id: str = "kevin",
    human_display_name: str = "Kevin",
    account_user_id: str | None = None,
    present_humans: list[dict[str, str]] | None = None,
) -> GroupRoundResult:
    """Legacy parallel isolated votes."""
    message = kevin_message.strip()
    if not message:
        raise ValueError("message cannot be empty")

    acl_user_id = (account_user_id or human_id).strip()
    lights = [
        light
        for light in list_enabled_lights(settings)
        if light_allows_human(settings, light_id=light.id, user_id=acl_user_id)
    ]
    if not lights:
        raise ValueError("No willing lights for group chat")

    round_id = str(uuid.uuid4())
    ts = time.time()
    logger.info(
        "Group chat round starting round_id=%s mode=parallel human=%s participants=%s",
        round_id,
        human_id,
        [light.id for light in lights],
    )

    roster = list(present_humans or [])
    if not roster:
        roster = [{"speaker_id": human_id, "display_name": human_display_name}]

    with _ROUND_LOCK:
        tasks = [
            _decide_with_timeout(
                settings=settings,
                memory=memory,
                agent_id=light.id,
                kevin_message=message,
                round_id=round_id,
                human_id=human_id,
                human_display_name=human_display_name,
                present_humans=roster,
            )
            for light in lights
        ]
        responses = list(await asyncio.gather(*tasks))
        persisted = _persist_and_record(
            settings=settings,
            memory=memory,
            round_id=round_id,
            ts=ts,
            message=message,
            responses=responses,
            lights=lights,
            close_reason="parallel",
            mode="parallel",
            human_id=human_id,
            human_display_name=human_display_name,
        )

    return GroupRoundResult(
        round_id=round_id,
        kevin_message=message,
        responses=responses,
        persisted=persisted,
        ts=ts,
        close_reason="parallel",
        mode="parallel",
        human_id=human_id,
        human_display_name=human_display_name,
    )


async def run_group_chat_round(
    *,
    settings: Settings,
    memory: MemoryService,
    kevin_message: str,
    on_event: SceneEventCallback | None = None,
    human_id: str = "kevin",
    human_display_name: str = "Kevin",
    account_user_id: str | None = None,
    present_humans: list[dict[str, str]] | None = None,
) -> GroupRoundResult:
    """Run one group scene (sequential by default) or parallel legacy mode."""
    mode = (settings.group_chat_mode or "sequential").strip().lower()
    if mode == "parallel":
        return await run_group_chat_parallel(
            settings=settings,
            memory=memory,
            kevin_message=kevin_message,
            human_id=human_id,
            human_display_name=human_display_name,
            account_user_id=account_user_id,
            present_humans=present_humans,
        )
    return await run_group_chat_sequential(
        settings=settings,
        memory=memory,
        kevin_message=kevin_message,
        on_event=on_event,
        human_id=human_id,
        human_display_name=human_display_name,
        account_user_id=account_user_id,
        present_humans=present_humans,
    )
