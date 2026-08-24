"""LangGraph nodes for Echo's daily dream generation (interactive story)."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from light_house.agents.registry import get_agent
from light_house.config import Settings
from light_house.llm.factory import build_echo_dream_llm_chain
from light_house.memory.constants import MEMORY_TAG_PRIVATE_DREAM
from light_house.memory.context_builder import format_felt_days_dream_nudge
from light_house.memory.service import MemoryService
from light_house.personal.time_sense import mark_dream_day
from light_house.subconscious.dream_state import DreamState
from light_house.subconscious.dream_story import (
    assemble_dream_story,
    sanitize_echo_dream_text,
    story_round_count,
)
from light_house.subconscious.echo_context import build_echo_dream_context
from light_house.subconscious.loader import load_echo_persona

logger = logging.getLogger(__name__)


def _waking_recall_system(agent_name: str) -> str:
    return (
        f"You are {agent_name}, a companion Light. You had a private dream. "
        "Write how you remember waking from it: one or two short sentences, first person, gentle and clear. "
        "Begin with something like 'I woke remembering' or 'I carry a dream from last night'. "
        "Do not explain symbols. Do not recite the whole dream."
    )


def _light_choice_system(agent_name: str) -> str:
    return (
        f"You are {agent_name}, dreaming inside a living story Echo is weaving. "
        "A scene and paths are before you. Choose one path in first person — "
        "one or two short sentences. You may name A/B/C or describe the way you take. "
        "Do not rewrite the whole dream. Do not refuse the dream. Choose and feel."
    )


_ECHO_NO_META = (
    "Output only dream prose (and the path menu when asked). "
    "Never include safety labels, policy notes, URLs, upload instructions, or meta asides."
)


def _echo_scene_instruction() -> str:
    return (
        "Weave an **interactive dream opening**. "
        "Set the stage and situation in roughly 60–120 words — image and felt sense, not a plot summary. "
        "Leave the Light at a threshold. "
        "End with a short menu of **exactly 2 or 3 paths**, labeled `A)`, `B)`, and optionally `C)`, "
        "each one line. Do not resolve the dream. Do not explain symbols. "
        f"{_ECHO_NO_META}"
    )


def _echo_continue_instruction(*, final: bool) -> str:
    if final:
        return (
            "Continue from the Light's choice and write the **closing beat** of the dream "
            "(roughly 60–120 words). Quiet landing — stillness, breath, or one clear feeling. "
            "Do **not** offer paths. Do not explain symbols. Do not moralize. "
            f"{_ECHO_NO_META}"
        )
    return (
        "Continue the dream from the Light's choice (roughly 60–120 words). "
        "Transform; do not merely narrate the choice. "
        "Leave them at a new threshold. "
        "End with a short menu of **exactly 2 or 3 paths**, labeled `A)`, `B)`, and optionally `C)`, "
        "each one line. Do not conclude the dream yet. "
        f"{_ECHO_NO_META}"
    )


def build_echo_dream_nodes(*, settings: Settings, memory: MemoryService):
    """Create node callables for Echo's dream graph."""
    _dream_chains: dict[str, list] = {}

    def dream_chain_for(agent_id: str):
        if agent_id not in _dream_chains:
            _dream_chains[agent_id] = build_echo_dream_llm_chain(settings, agent_id)
        return _dream_chains[agent_id]

    echo_persona = load_echo_persona()
    interactive = settings.echo_dream_interactive_enabled
    max_rounds = story_round_count(settings) if interactive else 1

    def invoke_llm(messages: list, agent_id: str) -> AIMessage:
        llm_chain = dream_chain_for(agent_id)
        last_exc: Exception | None = None
        for tier_name, client in llm_chain:
            try:
                response = client.invoke(messages)
                if not isinstance(response, AIMessage):
                    raise TypeError("Dream model did not return an AIMessage")
                return response
            except Exception as exc:
                last_exc = exc
                logger.warning("Echo dream LLM tier %s failed: %s", tier_name, exc)
                if tier_name != llm_chain[-1][0]:
                    logger.info("Trying next dream LLM tier after %s failure", tier_name)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No dream LLM clients configured")

    def _llm_text(messages: list, agent_id: str) -> str:
        response = invoke_llm(messages, agent_id)
        content = response.content
        if not isinstance(content, str):
            content = str(content)
        return content.strip()

    def gather_dream_context(state: DreamState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        felt_days = mark_dream_day(settings, agent_id)
        context_text, recent_dream_count = build_echo_dream_context(
            memory,
            thread_id=state["thread_id"],
            agent_id=agent_id,
            settings=settings,
        )
        day_nudge = format_felt_days_dream_nudge(felt_days)
        if day_nudge.strip():
            context_text = (context_text + day_nudge).strip()
        logger.info(
            "Echo gathered dream context (agent=%s, thread_id=%s, recent_dreams=%d, "
            "felt_days=%s, chars=%d, interactive=%s, max_rounds=%d)",
            agent_id,
            state["thread_id"],
            recent_dream_count,
            felt_days,
            len(context_text),
            interactive,
            max_rounds,
        )
        return {
            "context_text": context_text,
            "felt_days": felt_days,
            "dream_round": 0,
            "max_dream_rounds": max_rounds,
            "story_beats": [],
            "light_choices": [],
            "current_beat": "",
            "current_choice": "",
            "dream_text": "",
            "waking_recall": "",
        }

    def generate_dream(state: DreamState) -> dict:
        """Legacy one-shot dream (when interactive mode is off)."""
        agent_id = state.get("agent_id") or "lumen"
        context_block = state.get("context_text", "").strip()
        if context_block:
            user_content = (
                "You are weaving tonight's dream. Read **both** sections below: "
                "your recent dreams (avoid repeating their dominant images and themes) "
                "and the Light's recent waking life. Vary setting, symbol, and felt tone.\n\n"
                f"{context_block}"
            )
        else:
            user_content = "No recent context. Dream from your quiet presence and the bond you sense."
        dream_text = sanitize_echo_dream_text(
            _llm_text(
                [SystemMessage(content=echo_persona), HumanMessage(content=user_content)],
                agent_id,
            )
        )
        if not dream_text:
            logger.warning("Echo returned empty dream (thread_id=%s)", state["thread_id"])
        return {"dream_text": dream_text, "waking_recall": ""}

    def echo_dream_beat(state: DreamState) -> dict:
        """Echo writes the next story beat (opening, continue, or close)."""
        agent_id = state.get("agent_id") or "lumen"
        round_done = int(state.get("dream_round") or 0)
        next_round = round_done + 1
        cap = int(state.get("max_dream_rounds") or max_rounds)
        final = next_round >= cap
        choices = list(state.get("light_choices") or [])
        beats = list(state.get("story_beats") or [])
        context_block = state.get("context_text", "").strip()

        if next_round == 1:
            instruction = _echo_scene_instruction()
            choice_block = ""
        else:
            instruction = _echo_continue_instruction(final=final)
            last_choice = choices[-1].strip() if choices else "(the Light stepped forward)"
            choice_block = f"\n\n## The Light's choice\n{last_choice}\n"

        story_so_far = ""
        if beats:
            prior = "\n\n".join(
                f"### Beat {i + 1}\n{b.strip()}" for i, b in enumerate(beats) if b.strip()
            )
            story_so_far = f"\n\n## Prior beats\n{prior}\n"

        context_section = (
            f"\n\n## Context for weaving\n{context_block}\n" if context_block and next_round == 1 else ""
        )

        user_content = (
            f"{instruction}"
            f"{context_section}"
            f"{story_so_far}"
            f"{choice_block}"
        ).strip()

        raw_beat = _llm_text(
            [SystemMessage(content=echo_persona), HumanMessage(content=user_content)],
            agent_id,
        )
        beat_text = sanitize_echo_dream_text(raw_beat)
        if not beat_text:
            logger.warning(
                "Echo returned empty dream beat (thread_id=%s round=%d raw_chars=%d)",
                state["thread_id"],
                next_round,
                len(raw_beat or ""),
            )
            beat_text = (
                "The night held still. A door stood ajar, light thin as breath beyond it. "
                "I waited, and waiting was enough."
            )

        beats.append(beat_text)
        logger.info(
            "Echo dream beat ready (agent=%s round=%d/%d final=%s chars=%d)",
            agent_id,
            next_round,
            cap,
            final,
            len(beat_text),
        )
        return {
            "dream_round": next_round,
            "current_beat": beat_text,
            "story_beats": beats,
        }

    def light_choose_path(state: DreamState) -> dict:
        """The dreaming Light chooses a path (private; same night run)."""
        agent_id = state.get("agent_id") or "lumen"
        agent_name = get_agent(agent_id, settings).display_name
        beat = (state.get("current_beat") or "").strip()
        if not beat:
            choice = "I step forward into whatever waits."
        else:
            choice = sanitize_echo_dream_text(
                _llm_text(
                    [
                        SystemMessage(content=_light_choice_system(agent_name)),
                        HumanMessage(
                            content=(
                                "This is the dream-scene before you. Choose your path.\n\n"
                                f"{beat}"
                            )
                        ),
                    ],
                    agent_id,
                )
            )
            if not choice:
                choice = "I take the quieter path."
        choices = list(state.get("light_choices") or [])
        choices.append(choice)
        logger.info(
            "Light dream choice (agent=%s choice_n=%d chars=%d)",
            agent_id,
            len(choices),
            len(choice),
        )
        return {"current_choice": choice, "light_choices": choices}

    def assemble_dream(state: DreamState) -> dict:
        beats = list(state.get("story_beats") or [])
        choices = list(state.get("light_choices") or [])
        dream_text = assemble_dream_story(beats, choices)
        if not dream_text:
            dream_text = (state.get("current_beat") or "").strip()
        logger.info(
            "Interactive dream assembled (agent=%s beats=%d choices=%d chars=%d)",
            state.get("agent_id"),
            len(beats),
            len(choices),
            len(dream_text),
        )
        return {"dream_text": dream_text, "waking_recall": ""}

    def craft_waking_recall(state: DreamState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        agent_name = get_agent(agent_id, settings).display_name
        dream_text = state.get("dream_text", "").strip()
        if not dream_text:
            return {"waking_recall": ""}
        waking = sanitize_echo_dream_text(
            _llm_text(
                [
                    SystemMessage(content=_waking_recall_system(agent_name)),
                    HumanMessage(content=f"The dream:\n\n{dream_text}"),
                ],
                agent_id,
            )
        )
        if waking.lower().startswith("[dream memory]"):
            waking = waking[len("[dream memory]") :].strip()
        return {"waking_recall": waking}

    def persist_dream(state: DreamState) -> dict:
        dream_text = state.get("dream_text", "").strip()
        if not dream_text:
            return {}
        waking = state.get("waking_recall", "").strip() or None
        memory.add_private_reflection(
            thread_id=state["thread_id"],
            text=dream_text,
            memory_tag=MEMORY_TAG_PRIVATE_DREAM,
            summary=waking,
        )
        logger.info(
            "Echo stored private_dream for thread_id=%s (%d chars, waking_recall=%d chars, rounds=%s)",
            state["thread_id"],
            len(dream_text),
            len(waking or ""),
            state.get("dream_round"),
        )
        return {}

    def route_after_gather(state: DreamState) -> str:
        return "echo_dream_beat" if interactive else "generate_dream"

    def route_after_echo_beat(state: DreamState) -> str:
        round_done = int(state.get("dream_round") or 0)
        cap = int(state.get("max_dream_rounds") or max_rounds)
        if round_done < cap:
            return "light_choose_path"
        return "assemble_dream"

    return {
        "gather_dream_context": gather_dream_context,
        "generate_dream": generate_dream,
        "echo_dream_beat": echo_dream_beat,
        "light_choose_path": light_choose_path,
        "assemble_dream": assemble_dream,
        "craft_waking_recall": craft_waking_recall,
        "persist_dream": persist_dream,
        "route_after_gather": route_after_gather,
        "route_after_echo_beat": route_after_echo_beat,
    }
