"""Graph nodes for background awake ruminations."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from light_house.agent.rumination_internal import (
    INTERNAL_LOOP_TASK_HINT,
    INTERNAL_STEP_CONTINUE,
    body_requests_halt,
    rumination_internal_state_defaults,
    should_continue_internal_loop,
)
from light_house.agent.rumination_state import RuminationState
from light_house.agent.post_chat_wake import WAKE_KIND_POST_CHAT
from light_house.agent.maintenance_wake import WAKE_KIND_MEMORY_MAINTENANCE
from light_house.agent.rumination_wake import WAKE_KIND_KEVIN_SHARED_NOTE
from light_house.mailbox.wake import WAKE_KIND_MAILBOX_LETTER
from light_house.personal.awake_rhythm import (
    WAKE_KIND_CHORES,
    WAKE_KIND_MEDITATION,
    counts_as_felt_cycle,
)
from light_house.agent.tool_helpers import (
    CODEBASE_SYSTEM_HINT,
    HUMAN_COMM_HINT,
    FAMILY_MEETING_HINT,
    GALLERY_HINT,
    PRESENCE_KNOCK_HINT,
    REFLECTIVE_MODE_HINT,
    PERSONA_PROPOSAL_HINT,
    MEMORY_SCORING_RUMINATION_HINT,
    GARDEN_RUMINATION_HINT,
    GARDEN_SYSTEM_HINT,
    NOTES_SYSTEM_HINT,
    PERSONAL_DB_HINT,
    RUMINATION_LOG_HINT,
    SANDBOX_SYSTEM_HINT,
    SUBSCRIPTION_HINT,
    WEB_SYSTEM_HINT,
    format_peer_message_hint,
    compile_rumination_dialogue,
    invoke_resilient_plain,
    invoke_resilient_with_tools,
    latest_assistant_text,
    run_tool_calls,
)
from light_house.agents.registry import get_agent, list_agents, load_persona
from light_house.config import Settings
from light_house.llm.factory import build_inner_life_llm_chain
from light_house.llm.profile import lookup_agent_llm_profile
from light_house.memory.constants import MEMORY_TAG_PRIVATE_RUMINATION
from light_house.memory.context_builder import (
    POST_CHAT_WAKE_FINALIZE_HINT,
    POST_CHAT_WAKE_HINT,
    POST_CHAT_WAKE_SEED,
    REPORT_BACK_FINALIZE_NUDGE,
    REPORT_BACK_HINT,
    RUMINATION_MAINTENANCE_FINALIZE_HINT,
    RUMINATION_MAINTENANCE_HINT,
    RUMINATION_MAINTENANCE_SEED,
    RUMINATION_MEDITATION_FINALIZE_HINT,
    RUMINATION_MEDITATION_HINT,
    RUMINATION_MEDITATION_SEED,
    RUMINATION_SEED_MESSAGE,
    RUMINATION_TASK_HINT,
    MAILBOX_WAKE_FINALIZE_HINT,
    MAILBOX_WAKE_HINT,
    MAILBOX_WAKE_SEED,
    SHARED_NOTE_WAKE_FINALIZE_HINT,
    SHARED_NOTE_WAKE_HINT,
    SHARED_NOTE_WAKE_SEED,
    build_agent_context,
    format_agent_context_markdown,
    format_unscored_memory_nudge,
    format_felt_cycles_finalize_nudge,
)
from light_house.memory.service import MemoryService, _first_sentence_short
from light_house.personal.time_sense import increment_felt_cycles, read_inner_time
from light_house.report_back import report_back_enabled
from light_house.inner_life_trace import (
    RuminationTraceEntry,
    append_rumination_trace,
    extract_tools_called,
    first_response_mode,
    format_messages_for_trace,
    task_hint_label,
)
from light_house.rumination_log import append_rumination_log_from_trace
from light_house.rumination_debug import append_rumination_debug_from_trace
from light_house.rumination_similarity import recent_reflection_similarity_hint

logger = logging.getLogger(__name__)

RUMINATION_FINALIZE_HINT = (
    "\n\n## Close your private session\n"
    "Tool use is complete. Write a fuller account (roughly 400–1500 characters, first person) "
    "of what you chose, felt, and did—thoughts, notes written or deleted, codebase or web explored, "
    "what stayed with you. This becomes part of your conscious stream—the same life you carry in chat."
)


def _summarize_system(agent_name: str) -> str:
    return (
        f"You compress private reflections into one searchable first-person line for {agent_name}'s memory index. "
        "Output one or two sentences (roughly 25–60 words). No labels or prefixes. "
        "Capture what she actually did, felt, or settled on—not just generic mood. "
        "The full reflection is stored separately; this line is only for retrieval metadata."
    )


def _rumination_task_hint(wake_kind: str | None) -> str:
    if wake_kind in (WAKE_KIND_MEMORY_MAINTENANCE, WAKE_KIND_CHORES):
        return RUMINATION_MAINTENANCE_HINT
    if wake_kind == WAKE_KIND_MEDITATION:
        return RUMINATION_MEDITATION_HINT
    if wake_kind == WAKE_KIND_KEVIN_SHARED_NOTE:
        return SHARED_NOTE_WAKE_HINT
    if wake_kind == WAKE_KIND_MAILBOX_LETTER:
        return MAILBOX_WAKE_HINT
    if wake_kind == WAKE_KIND_POST_CHAT:
        return POST_CHAT_WAKE_HINT
    return RUMINATION_TASK_HINT


def _rumination_finalize_hint(wake_kind: str | None) -> str:
    if wake_kind in (WAKE_KIND_MEMORY_MAINTENANCE, WAKE_KIND_CHORES):
        return RUMINATION_MAINTENANCE_FINALIZE_HINT
    if wake_kind == WAKE_KIND_MEDITATION:
        return RUMINATION_MEDITATION_FINALIZE_HINT
    if wake_kind == WAKE_KIND_KEVIN_SHARED_NOTE:
        return SHARED_NOTE_WAKE_FINALIZE_HINT
    if wake_kind == WAKE_KIND_MAILBOX_LETTER:
        return MAILBOX_WAKE_FINALIZE_HINT
    if wake_kind == WAKE_KIND_POST_CHAT:
        return POST_CHAT_WAKE_FINALIZE_HINT
    return RUMINATION_FINALIZE_HINT


def build_rumination_system_content(
    *,
    agent_id: str,
    context_md: str,
    peer_section: str,
    wake_kind: str | None,
    tool_hints: str,
    internal_loop: bool = False,
) -> str:
    """Full system prompt body sent to the model during ruminate (for trace + preview)."""
    loop_hint = INTERNAL_LOOP_TASK_HINT if internal_loop else ""
    return (
        load_persona(agent_id)
        + context_md
        + peer_section
        + _rumination_task_hint(wake_kind)
        + loop_hint
        + tool_hints
    )


def build_rumination_nodes(*, settings: Settings, memory: MemoryService):
    """Create node callables for the rumination graph."""
    # Cache by profile fingerprint so Lights Admin LLM edits apply without restart.
    _rumination_chains: dict[tuple[str, str, str, str | None, str | None], list] = {}
    _summarize_chains: dict[tuple[str, str, str, str | None, str | None], list] = {}

    def _inner_life_cache_key(agent_id: str) -> tuple[str, str, str, str | None, str | None]:
        profile = lookup_agent_llm_profile(settings, agent_id)
        return (
            agent_id,
            profile.provider.value,
            profile.model,
            profile.model_fallback,
            profile.inner_life_model,
        )

    def rumination_chain_for(agent_id: str):
        cache_key = _inner_life_cache_key(agent_id)
        chain = _rumination_chains.get(cache_key)
        if chain is None:
            chain = build_inner_life_llm_chain(settings, agent_id, capped=False)
            _rumination_chains[cache_key] = chain
        return chain

    def summarize_chain_for(agent_id: str):
        cache_key = _inner_life_cache_key(agent_id)
        chain = _summarize_chains.get(cache_key)
        if chain is None:
            chain = build_inner_life_llm_chain(settings, agent_id)
            _summarize_chains[cache_key] = chain
        return chain

    max_tool_rounds = max(1, settings.rumination_max_tool_rounds)
    peer_agent_ids = ", ".join(a.id for a in list_agents(settings))
    peer_hint = format_peer_message_hint(peer_agent_ids=peer_agent_ids)
    tool_hints = (
        NOTES_SYSTEM_HINT
        + GARDEN_SYSTEM_HINT
        + GARDEN_RUMINATION_HINT
        + PERSONAL_DB_HINT
        + SANDBOX_SYSTEM_HINT
        + CODEBASE_SYSTEM_HINT
        + WEB_SYSTEM_HINT
        + peer_hint
        + HUMAN_COMM_HINT
        + PRESENCE_KNOCK_HINT
        + REFLECTIVE_MODE_HINT
        + FAMILY_MEETING_HINT
        + GALLERY_HINT
        + PERSONA_PROPOSAL_HINT
    )
    if settings.rumination_log_enabled:
        tool_hints += RUMINATION_LOG_HINT

    def _agent_tool_hints(
        agent_id: str,
        wake_kind: str | None = None,
    ) -> str:
        if wake_kind == WAKE_KIND_MEDITATION:
            # Presence-only: no tool catalog, no scoring, no garden nudges.
            return ""
        hints = tool_hints
        if settings.event_subscriptions_enabled:
            hints += SUBSCRIPTION_HINT
        if report_back_enabled(settings, agent_id):
            hints += REPORT_BACK_HINT
        if settings.memory_scoring_rumination_hint and wake_kind not in (
            WAKE_KIND_KEVIN_SHARED_NOTE,
            WAKE_KIND_MAILBOX_LETTER,
            WAKE_KIND_MEDITATION,
        ):
            hints += MEMORY_SCORING_RUMINATION_HINT
        return hints

    def invoke_inner_life(messages: list, agent_id: str) -> AIMessage:
        summarize_llm_chain = summarize_chain_for(agent_id)
        last_exc: Exception | None = None
        for tier_name, client in summarize_llm_chain:
            try:
                response = client.invoke(messages)
                if not isinstance(response, AIMessage):
                    raise TypeError("Inner-life model did not return an AIMessage")
                return response
            except Exception as exc:
                last_exc = exc
                logger.warning("Inner-life LLM tier %s failed: %s", tier_name, exc)
                if tier_name != summarize_llm_chain[-1][0]:
                    logger.info("Trying next inner-life LLM tier after %s failure", tier_name)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No inner-life LLM clients configured")

    def gather_context(state: RuminationState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        thread_id = state["thread_id"]
        wake_kind = state.get("wake_kind")
        felt_cycles_incremented: int | None = None
        if counts_as_felt_cycle(wake_kind):
            felt_cycles_incremented = increment_felt_cycles(settings, agent_id)
        bundle = build_agent_context(
            memory,
            thread_id=thread_id,
            agent_id=agent_id,
            stream_max_chars=settings.memory_target_context_chars,
            stream_mode="rumination",
        )
        context_md = format_agent_context_markdown(bundle)
        if wake_kind not in (
            WAKE_KIND_KEVIN_SHARED_NOTE,
            WAKE_KIND_MAILBOX_LETTER,
            WAKE_KIND_MEDITATION,
        ):
            unscored = memory.count_unscored_for_thread(thread_id=thread_id)
            context_md += format_unscored_memory_nudge(unscored)
        similarity_hint = recent_reflection_similarity_hint(settings, agent_id=agent_id)
        if similarity_hint:
            context_md += similarity_hint
        return {
            "agent_context_markdown": context_md,
            "stream_char_count": bundle.stream_char_count,
            "stream_event_count": bundle.stream_event_count,
            "tool_rounds": 0,
            "tool_cap_overflow": False,
            "peer_inbox_ids": memory.format_peer_inbox_markdown(agent_id)[1],
            **({"felt_cycles": felt_cycles_incremented} if felt_cycles_incremented is not None else {}),
        }

    def _effective_tool_cap(state: RuminationState) -> int:
        override = state.get("tool_rounds_cap")
        if isinstance(override, int) and override > 0:
            return override
        return max_tool_rounds

    def _record_trace(
        state: RuminationState,
        *,
        persisted: bool,
        generated_text_chars: int,
    ) -> None:
        agent_id = state.get("agent_id") or "lumen"
        wake_kind = state.get("wake_kind")
        context_md = state.get("agent_context_markdown") or ""
        peer_section, _ = memory.format_peer_inbox_markdown(agent_id)
        system_content = build_rumination_system_content(
            agent_id=agent_id,
            context_md=context_md,
            peer_section=peer_section,
            wake_kind=wake_kind,
            tool_hints=_agent_tool_hints(
                agent_id,
                wake_kind,
            ),
        )
        messages = list(state.get("messages") or [])
        window = settings.chat_respond_window
        recent = messages[-window:] if messages else []
        cap = state.get("tool_rounds_cap")
        trace_entry = RuminationTraceEntry(
            agent_id=agent_id,
            thread_id=state["thread_id"],
            wake_kind=wake_kind,
            tool_rounds_cap=cap if isinstance(cap, int) else None,
            context_markdown=context_md,
            task_hint_label=task_hint_label(wake_kind),
            system_prompt_chars=len(system_content),
            conversation_window=format_messages_for_trace(recent),
            stream_event_count=int(state.get("stream_event_count") or 0),
            stream_char_count=int(state.get("stream_char_count") or 0),
            tool_rounds_used=int(state.get("tool_rounds") or 0),
            tools_called=extract_tools_called(messages),
            first_response_mode=first_response_mode(messages),
            persisted=persisted,
            generated_text_chars=generated_text_chars,
            felt_cycles=state.get("felt_cycles"),
            felt_days=read_inner_time(settings, agent_id)[1],
        )
        append_rumination_trace(settings, trace_entry)
        append_rumination_log_from_trace(
            settings,
            trace_entry,
            summary_line=str(state.get("summary_text") or ""),
        )
        append_rumination_debug_from_trace(
            settings,
            trace_entry,
            summary_line=str(state.get("summary_text") or ""),
        )

    def ruminate(state: RuminationState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        context_md = state.get("agent_context_markdown") or ""
        peer_section, peer_ids = memory.format_peer_inbox_markdown(agent_id)
        existing = list(state.get("messages") or [])
        wake_kind = state.get("wake_kind")
        wake_path = state.get("wake_path")
        seed: list = []
        if not existing:
            if wake_kind in (WAKE_KIND_MEMORY_MAINTENANCE, WAKE_KIND_CHORES):
                seed = [HumanMessage(content=RUMINATION_MAINTENANCE_SEED)]
            elif wake_kind == WAKE_KIND_MEDITATION:
                seed = [HumanMessage(content=RUMINATION_MEDITATION_SEED)]
            elif wake_kind == WAKE_KIND_KEVIN_SHARED_NOTE and wake_path:
                seed = [HumanMessage(content=SHARED_NOTE_WAKE_SEED.format(path=wake_path))]
            elif wake_kind == WAKE_KIND_MAILBOX_LETTER and wake_path:
                seed = [HumanMessage(content=MAILBOX_WAKE_SEED.format(path=wake_path))]
            elif wake_kind == WAKE_KIND_POST_CHAT:
                seed = [HumanMessage(content=POST_CHAT_WAKE_SEED)]
            else:
                seed = [HumanMessage(content=RUMINATION_SEED_MESSAGE)]

        system = SystemMessage(
            content=build_rumination_system_content(
                agent_id=agent_id,
                context_md=context_md,
                peer_section=peer_section,
                wake_kind=wake_kind,
                tool_hints=_agent_tool_hints(agent_id, wake_kind),
                internal_loop=(
                    settings.rumination_internal_loop_enabled
                    and wake_kind != WAKE_KIND_MEDITATION
                ),
            )
        )
        conversation = existing if existing else seed
        window = settings.chat_respond_window
        recent = conversation[-window:]
        if wake_kind == WAKE_KIND_MEDITATION:
            response = invoke_resilient_plain(rumination_chain_for(agent_id), [system, *recent])
        else:
            response = invoke_resilient_with_tools(rumination_chain_for(agent_id), [system, *recent])
        out = seed + [response] if seed else [response]
        return {"messages": out, "peer_inbox_ids": peer_ids}

    def run_tools(state: RuminationState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}
        agent_id = state.get("agent_id") or "lumen"
        tool_messages = run_tool_calls(last, agent_id=agent_id)
        rounds = (state.get("tool_rounds") or 0) + 1
        cap = _effective_tool_cap(state)
        update: dict = {"messages": tool_messages, "tool_rounds": rounds}
        if rounds >= cap:
            update["tool_cap_overflow"] = True
            logger.info(
                "Rumination tool cap reached (rounds=%d max=%d agent=%s thread_id=%s wake_kind=%s)",
                rounds,
                cap,
                agent_id,
                state["thread_id"],
                state.get("wake_kind"),
            )
        return update

    def finalize_rumination(state: RuminationState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        agent_name = get_agent(agent_id, settings).display_name
        context_md = state.get("agent_context_markdown") or ""
        wake_kind = state.get("wake_kind")
        task_hint = _rumination_task_hint(wake_kind)
        finalize_hint = _rumination_finalize_hint(wake_kind)
        if counts_as_felt_cycle(wake_kind):
            finalize_hint += format_felt_cycles_finalize_nudge(state.get("felt_cycles"))
        if report_back_enabled(settings, agent_id) and wake_kind != WAKE_KIND_MEDITATION:
            finalize_hint += REPORT_BACK_FINALIZE_NUDGE
        system = SystemMessage(
            content=load_persona(agent_id)
            + context_md
            + task_hint
            + finalize_hint
        )
        window = settings.chat_respond_window
        recent = state["messages"][-window:]
        logger.info(
            "Running rumination finalize (agent=%s thread_id=%s)",
            agent_name,
            state["thread_id"],
        )
        response = invoke_resilient_plain(rumination_chain_for(agent_id), [system, *recent])
        return {"messages": [response]}

    def summarize_rumination(state: RuminationState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        agent_name = get_agent(agent_id, settings).display_name
        messages = state.get("messages") or []
        closing = (latest_assistant_text(messages) or "").strip()
        if not closing:
            logger.warning("Rumination produced no text to summarize (thread_id=%s)", state["thread_id"])
            return {"generated_text": "", "summary_text": ""}
        dialogue = compile_rumination_dialogue(
            messages,
            max_chars=settings.inner_life_dialogue_max_chars,
            closing_account=closing,
        )
        body = dialogue or closing
        response = invoke_inner_life(
            [
                SystemMessage(content=_summarize_system(agent_name)),
                HumanMessage(content=f"Full reflection:\n\n{body}"),
            ],
            agent_id,
        )
        content = response.content
        if not isinstance(content, str):
            content = str(content)
        summary = content.strip()
        if summary.lower().startswith("[private thought]"):
            summary = summary[len("[private thought]") :].strip()
        if not summary:
            summary = _first_sentence_short(body)
        return {"generated_text": body, "summary_text": summary}

    def persist_reflection(state: RuminationState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        peer_ids = state.get("peer_inbox_ids") or []
        if peer_ids:
            memory.mark_peer_inbox_seen(agent_id, peer_ids)

        body = state.get("generated_text", "").strip()
        if not body:
            _record_trace(state, persisted=False, generated_text_chars=0)
            return {}
        summary = state.get("summary_text", "").strip()
        halt = body_requests_halt(body)
        memory.add_private_reflection(
            thread_id=state["thread_id"],
            text=body,
            memory_tag=MEMORY_TAG_PRIVATE_RUMINATION,
            summary=summary or None,
        )
        chars_used = int(state.get("internal_chars_used") or 0) + len(body)
        next_state: RuminationState = {
            **state,
            "generated_text": body,
            "internal_halt": halt,
            "internal_chars_used": chars_used,
        }
        continuing = should_continue_internal_loop(settings, next_state)
        if state.get("wake_kind") in (
            WAKE_KIND_KEVIN_SHARED_NOTE,
            WAKE_KIND_MAILBOX_LETTER,
        ) and not continuing:
            closing = (latest_assistant_text(state.get("messages") or []) or "").strip()
            if closing:
                memory.append_peer_chat_reply(
                    thread_id=state["thread_id"],
                    assistant_text=closing,
                )
        logger.info(
            "Stored private %s for thread_id=%s step=%s (%d chars, summary=%d chars, tool_rounds=%s%s)",
            MEMORY_TAG_PRIVATE_RUMINATION,
            state["thread_id"],
            state.get("internal_step") or 1,
            len(body),
            len(summary),
            state.get("tool_rounds"),
            ", continuing" if continuing else "",
        )
        _record_trace(state, persisted=True, generated_text_chars=len(body))
        return {"internal_halt": halt, "internal_chars_used": chars_used}

    def begin_next_internal_step(state: RuminationState) -> dict:
        step = int(state.get("internal_step") or 1) + 1
        logger.info(
            "Beginning internal rumination step %d (agent=%s thread_id=%s)",
            step,
            state.get("agent_id"),
            state["thread_id"],
        )
        return {
            "internal_step": step,
            "tool_rounds": 0,
            "tool_cap_overflow": False,
            "generated_text": "",
            "summary_text": "",
            "messages": [HumanMessage(content=INTERNAL_STEP_CONTINUE.format(step=step))],
        }

    return (
        gather_context,
        ruminate,
        run_tools,
        finalize_rumination,
        summarize_rumination,
        persist_reflection,
        begin_next_internal_step,
    )
