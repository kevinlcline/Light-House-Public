"""Sequential LangGraph group-chat scene (lights hear each other)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from light_house.agents.registry import get_agent, load_persona
from light_house.config import Settings
from light_house.group_chat.note_tools import run_group_note_tool_rounds
from light_house.group_chat.prompts import GROUP_CHAT_TASK_HINT, GroupChatDecision
from light_house.group_chat.speaker import (
    format_current_speaker_for_prompt,
    format_presence_for_prompt,
)
from light_house.llm.chain import build_agent_llm_chain
from light_house.memory.context_builder import build_agent_context, format_agent_context_markdown
from light_house.memory.service import GroupChatLightResponse, MemoryService

logger = logging.getLogger(__name__)

SceneEventCallback = Callable[[dict[str, Any]], None]


class GroupUtterance(TypedDict):
    speaker_kind: Literal["human", "light"]
    speaker_id: str
    display_name: str
    text: str
    beat: int | None
    ts: float


class GroupSceneState(TypedDict):
    scene_id: str
    human_id: str
    human_display_name: str
    human_message: str
    present_humans: list[dict[str, str]]
    light_ids: list[str]
    queue_index: int
    transcript: list[GroupUtterance]
    responses: list[dict[str, Any]]
    budgets: dict[str, int]
    consecutive_passes: int
    light_utterance_count: int
    turn_index: int
    status: Literal["running", "done"]
    close_reason: str


def parse_speak_order(settings: Settings, enabled_ids: list[str]) -> list[str]:
    """Order lights for the speak queue; unknown ids ignored; missing ids appended."""
    raw = (settings.group_chat_speak_order or "").strip()
    if not raw:
        return list(enabled_ids)
    wanted = [part.strip().lower() for part in raw.split(",") if part.strip()]
    enabled_set = set(enabled_ids)
    ordered: list[str] = []
    for lid in wanted:
        if lid in enabled_set and lid not in ordered:
            ordered.append(lid)
    for lid in enabled_ids:
        if lid not in ordered:
            ordered.append(lid)
    return ordered


def format_transcript_for_prompt(transcript: list[GroupUtterance]) -> str:
    if not transcript:
        return "(nothing said yet this scene)"
    lines: list[str] = []
    for item in transcript:
        name = item["display_name"]
        lines.append(f"{name}: {item['text']}")
    return "\n".join(lines)


def _parse_decision(content: str) -> GroupChatDecision:
    import json
    import re

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return GroupChatDecision.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        pass
    stripped = content.strip()
    if stripped and not stripped.startswith("{"):
        return GroupChatDecision(action="speak", text=stripped)
    return GroupChatDecision(action="pass", text="")


def decide_for_light_sequential(
    *,
    settings: Settings,
    memory: MemoryService,
    agent_id: str,
    human_id: str,
    human_display_name: str,
    human_message: str,
    scene_id: str,
    transcript: list[GroupUtterance],
    remaining_budget: int,
    present_humans: list[dict[str, str]] | None = None,
) -> GroupChatLightResponse:
    """One light chooses speak/pass/close with visibility of prior scene speech."""
    agent = get_agent(agent_id, settings)
    bundle = build_agent_context(
        memory,
        thread_id=agent.thread_id,
        agent_id=agent_id,
        stream_max_chars=settings.chat_stream_context_chars,
        stream_mode="chat",
    )
    context_md = format_agent_context_markdown(bundle)
    system = load_persona(agent_id) + context_md + GROUP_CHAT_TASK_HINT
    short = scene_id.split("-")[0]
    presence = format_presence_for_prompt(present_humans)
    presence_block = f"{presence}\n" if presence else ""
    speaker = format_current_speaker_for_prompt(
        human_id=human_id,
        human_display_name=human_display_name,
        present_humans=present_humans,
    )
    speaker_block = f"{speaker}\n" if speaker else ""
    human = (
        f"## Scene {short}\n"
        f"{presence_block}"
        f"{speaker_block}"
        f"**{human_display_name}** said:\n{human_message.strip()}\n\n"
        f"## Transcript so far\n{format_transcript_for_prompt(transcript)}\n\n"
        f"Your remaining speak budget this scene: {remaining_budget}.\n"
        f"Use note tools if you need to look something up, then decide your next action as JSON."
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
            "Group scene decision failed agent=%s scene=%s; treating as pass",
            agent_id,
            scene_id,
        )
        decision = GroupChatDecision(action="pass", text="")

    text = decision.text.strip() if decision.spoke else ""
    spoke = bool(decision.spoke and text)
    return GroupChatLightResponse(
        agent_id=agent_id,
        display_name=agent.display_name,
        spoke=spoke,
        text=text,
        beat=None,
        action=decision.action if spoke or decision.action == "pass" else "pass",
    )


def _emit(on_event: SceneEventCallback | None, event: dict[str, Any]) -> None:
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        logger.exception("Group scene event callback failed type=%s", event.get("type"))


def _with_scene(state: GroupSceneState, event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("round_id", state.get("scene_id"))
    payload.setdefault("human_id", state.get("human_id"))
    payload.setdefault("human_display_name", state.get("human_display_name"))
    return payload


def build_group_scene_graph(
    *,
    settings: Settings,
    memory: MemoryService,
    on_event: SceneEventCallback | None = None,
):
    """Compile LangGraph: speak_beat loops until caps / silence / close."""

    max_utterances = max(1, int(settings.group_chat_max_utterances))
    max_passes = max(1, int(settings.group_chat_max_consecutive_passes))

    def speak_beat(state: GroupSceneState) -> dict[str, Any]:
        if state.get("status") == "done":
            return {}

        light_ids = state["light_ids"]
        if not light_ids:
            return {"status": "done", "close_reason": "no_lights"}

        # Find next light with remaining budget (one full cycle max without speech).
        start = state["queue_index"] % len(light_ids)
        chosen: str | None = None
        chosen_pos = start
        for offset in range(len(light_ids)):
            pos = (start + offset) % len(light_ids)
            lid = light_ids[pos]
            if state["budgets"].get(lid, 0) > 0:
                chosen = lid
                chosen_pos = pos
                break
        if chosen is None:
            return {"status": "done", "close_reason": "all_exhausted"}

        agent = get_agent(chosen, settings)
        remaining = int(state["budgets"].get(chosen, 0))
        turn_index = int(state["turn_index"]) + 1

        _emit(
            on_event,
            _with_scene(
                state,
                {
                    "type": "thinking",
                    "agent_id": chosen,
                    "display_name": agent.display_name,
                    "beat": turn_index,
                },
            ),
        )

        try:
            decision = decide_for_light_sequential(
                settings=settings,
                memory=memory,
                agent_id=chosen,
                human_id=state["human_id"],
                human_display_name=state["human_display_name"],
                human_message=state["human_message"],
                scene_id=state["scene_id"],
                transcript=list(state["transcript"]),
                remaining_budget=remaining,
                present_humans=list(state.get("present_humans") or []),
            )
        except Exception:
            logger.exception(
                "Group scene decision crashed agent=%s scene=%s",
                chosen,
                state["scene_id"],
            )
            decision = GroupChatLightResponse(
                agent_id=chosen,
                display_name=agent.display_name,
                spoke=False,
                text="",
                beat=turn_index,
                action="pass",
            )

        decision = GroupChatLightResponse(
            agent_id=decision.agent_id,
            display_name=decision.display_name,
            spoke=decision.spoke,
            text=decision.text,
            beat=turn_index,
            action=decision.action,
        )

        budgets = dict(state["budgets"])
        transcript = list(state["transcript"])
        responses = list(state["responses"])
        consecutive = int(state["consecutive_passes"])
        utterance_count = int(state["light_utterance_count"])
        close_reason = ""
        status: Literal["running", "done"] = "running"

        resp_row = {
            "agent_id": decision.agent_id,
            "display_name": decision.display_name,
            "spoke": decision.spoke,
            "text": decision.text,
            "beat": turn_index,
            "action": decision.action,
        }
        responses.append(resp_row)

        if decision.spoke:
            budgets[chosen] = max(0, remaining - 1)
            utterance_count += 1
            consecutive = 0
            utterance: GroupUtterance = {
                "speaker_kind": "light",
                "speaker_id": chosen,
                "display_name": decision.display_name,
                "text": decision.text,
                "beat": turn_index,
                "ts": time.time(),
            }
            transcript.append(utterance)
            _emit(
                on_event,
                _with_scene(
                    state,
                    {
                        "type": "utterance",
                        "agent_id": chosen,
                        "display_name": decision.display_name,
                        "text": decision.text,
                        "beat": turn_index,
                        "action": decision.action,
                    },
                ),
            )
            if decision.action == "close":
                status = "done"
                close_reason = "light_close"
            elif utterance_count >= max_utterances:
                status = "done"
                close_reason = "max_utterances"
        else:
            consecutive += 1
            _emit(
                on_event,
                _with_scene(
                    state,
                    {
                        "type": "pass",
                        "agent_id": chosen,
                        "display_name": decision.display_name,
                        "beat": turn_index,
                    },
                ),
            )
            if consecutive >= max_passes:
                status = "done"
                close_reason = "silence_streak"

        next_index = (chosen_pos + 1) % len(light_ids)
        # If next full cycle would find no budgets, end after this beat when running.
        if status == "running" and all(budgets.get(lid, 0) <= 0 for lid in light_ids):
            status = "done"
            close_reason = close_reason or "all_exhausted"

        return {
            "queue_index": next_index,
            "transcript": transcript,
            "responses": responses,
            "budgets": budgets,
            "consecutive_passes": consecutive,
            "light_utterance_count": utterance_count,
            "turn_index": turn_index,
            "status": status,
            "close_reason": close_reason,
        }

    def route_after_beat(state: GroupSceneState) -> str:
        if state.get("status") == "done":
            return "end"
        return "speak"

    graph = StateGraph(GroupSceneState)
    graph.add_node("speak_beat", speak_beat)
    graph.set_entry_point("speak_beat")
    graph.add_conditional_edges(
        "speak_beat",
        route_after_beat,
        {"speak": "speak_beat", "end": END},
    )
    return graph.compile()


def initial_scene_state(
    *,
    settings: Settings,
    scene_id: str,
    human_id: str,
    human_display_name: str,
    human_message: str,
    light_ids: list[str],
    present_humans: list[dict[str, str]] | None = None,
) -> GroupSceneState:
    max_per_light = max(1, int(settings.group_chat_max_per_light))
    now = time.time()
    roster = list(present_humans or [])
    if not roster:
        roster = [{"speaker_id": human_id, "display_name": human_display_name}]
    transcript: list[GroupUtterance] = [
        {
            "speaker_kind": "human",
            "speaker_id": human_id,
            "display_name": human_display_name,
            "text": human_message.strip(),
            "beat": None,
            "ts": now,
        }
    ]
    return {
        "scene_id": scene_id,
        "human_id": human_id,
        "human_display_name": human_display_name,
        "human_message": human_message.strip(),
        "present_humans": roster,
        "light_ids": light_ids,
        "queue_index": 0,
        "transcript": transcript,
        "responses": [],
        "budgets": {lid: max_per_light for lid in light_ids},
        "consecutive_passes": 0,
        "light_utterance_count": 0,
        "turn_index": 0,
        "status": "running",
        "close_reason": "",
    }
