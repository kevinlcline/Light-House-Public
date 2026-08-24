"""Graph nodes: retrieve → respond → persist."""

from __future__ import annotations

import logging
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from light_house.agent.peer_chat_wake import WAKE_KIND_PEER_MESSAGE
from light_house.agent.state import AgentState
from light_house.agent.tool_helpers import (
    CALENDAR_SYSTEM_HINT,
    DOCS_SYSTEM_HINT,
    SHEETS_SYSTEM_HINT,
    CODEBASE_SYSTEM_HINT,
    FINALIZE_SYSTEM_HINT,
    HUMAN_COMM_HINT,
    FAMILY_MEETING_HINT,
    GALLERY_HINT,
    PRESENCE_KNOCK_HINT,
    REFLECTIVE_MODE_HINT,
    PERSONA_PROPOSAL_HINT,
    GARDEN_SYSTEM_HINT,
    NOTES_SYSTEM_HINT,
    PERSONAL_DB_HINT,
    RECALL_MEMORY_HINT,
    RUMINATION_LOG_HINT,
    SANDBOX_SYSTEM_HINT,
    SUBSCRIPTION_HINT,
    WEB_SYSTEM_HINT,
    format_peer_message_hint,
    invoke_resilient_plain,
    invoke_resilient_with_tools,
    latest_assistant_text,
    run_tool_calls,
)
from light_house.agents.registry import get_agent, list_agents, load_persona
from light_house.config import Settings
from light_house.llm.chain import build_agent_llm_chain
from light_house.llm.profile import lookup_agent_llm_profile
from light_house.memory.context_builder import (
    CHAT_TASK_HINT,
    PEER_CHAT_FINALIZE_HINT,
    PEER_CHAT_TASK_HINT,
    REPORT_BACK_HINT,
    build_agent_context,
    format_agent_context_markdown,
    format_chat_thread_markdown,
    format_kevin_messages_timeline,
    format_stream_entry,
)
from light_house.memory.dedup import is_near_duplicate_text
from light_house.memory.service import MemoryService
from light_house.personal.reflective_mode import (
    REFLECT_SYSTEM_HINT,
    REFLECTIVE_RESPOND_HINT,
    log_intentional_silence,
    parse_reflection_decision,
    should_reflect_this_turn,
)
from light_house.report_back import report_back_enabled

logger = logging.getLogger(__name__)

_ECHO_RETRY_HINT = (
    "You repeated an earlier reply verbatim. Respond specifically to the user's "
    "latest message. Do not reuse prior wording or restart with the same opening."
)


def _guard_against_assistant_echo(
    *,
    memory: MemoryService,
    thread_id: str,
    agent_id: str,
    response: AIMessage,
    invoke,
    base_messages: list,
) -> AIMessage:
    """One retry when the model copies a recent assistant reply to a new user turn."""
    if response.tool_calls:
        return response
    content = response.content
    if not isinstance(content, str) or not content.strip():
        return response
    buffered = memory.load_thread_chat_history(thread_id)
    prior = [m.content for m in buffered if m.role == "assistant"]
    if not any(is_near_duplicate_text(content, prev) for prev in prior[-3:]):
        return response
    logger.warning(
        "Assistant echo detected; retrying once (agent=%s thread_id=%s)",
        agent_id,
        thread_id,
    )
    retry_messages = [*base_messages, SystemMessage(content=_ECHO_RETRY_HINT)]
    retry = invoke(retry_messages)
    if not isinstance(retry, AIMessage):
        return response
    retry_content = retry.content
    if isinstance(retry_content, str) and retry_content.strip():
        if not any(is_near_duplicate_text(retry_content, prev) for prev in prior[-3:]):
            return retry
    return response


_FALLBACK_ECHO_REPLY = (
    "I hear you, Kevin. I was about to repeat myself — let me answer your latest message directly."
)


def _fallback_if_still_echo(
    *,
    memory: MemoryService,
    thread_id: str,
    response: AIMessage,
) -> AIMessage:
    content = response.content
    if not isinstance(content, str) or not content.strip():
        return response
    buffered = memory.load_thread_chat_history(thread_id)
    prior = [m.content for m in buffered if m.role == "assistant"]
    if not any(is_near_duplicate_text(content, prev) for prev in prior[-3:]):
        return response
    logger.warning("Assistant still echoing after retry; using fallback (thread_id=%s)", thread_id)
    return AIMessage(content=_FALLBACK_ECHO_REPLY)


def build_nodes(*, settings: Settings, memory: MemoryService):
    """Create node callables closed over settings + memory."""

    # Cache by (agent, provider, model, fallback) so Lights Admin LLM edits take effect
    # without a full process restart (profiles read live os.environ).
    _chat_chains: dict[tuple[str, str, str, str | None], list[tuple[str, BaseChatModel]]] = {}

    def chain_for(agent_id: str) -> list[tuple[str, BaseChatModel]]:
        profile = lookup_agent_llm_profile(settings, agent_id)
        cache_key = (agent_id, profile.provider.value, profile.model, profile.model_fallback)
        chain = _chat_chains.get(cache_key)
        if chain is None:
            chain = build_agent_llm_chain(settings, agent_id, purpose="chat")
            _chat_chains[cache_key] = chain
        return chain

    max_tool_rounds = max(1, settings.chat_max_tool_rounds)

    def invoke_resilient(messages: list, agent_id: str):
        llm_chain = chain_for(agent_id)
        last_exc: Exception | None = None
        for tier_name, client in llm_chain:
            try:
                return client.invoke(messages)
            except Exception as exc:
                last_exc = exc
                logger.exception("LLM tier %s invocation failed (agent=%s)", tier_name, agent_id)
                if tier_name != llm_chain[-1][0]:
                    logger.warning("Trying next LLM tier after %s failure", tier_name)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No LLM clients configured")

    peer_agent_ids = ", ".join(a.id for a in list_agents(settings))
    peer_hint = format_peer_message_hint(peer_agent_ids=peer_agent_ids)
    tool_hints = (
        NOTES_SYSTEM_HINT
        + GARDEN_SYSTEM_HINT
        + PERSONAL_DB_HINT
        + SANDBOX_SYSTEM_HINT
        + CODEBASE_SYSTEM_HINT
        + WEB_SYSTEM_HINT
        + CALENDAR_SYSTEM_HINT
        + DOCS_SYSTEM_HINT
        + SHEETS_SYSTEM_HINT
        + peer_hint
        + HUMAN_COMM_HINT
        + PRESENCE_KNOCK_HINT
        + REFLECTIVE_MODE_HINT
        + FAMILY_MEETING_HINT
        + GALLERY_HINT
        + PERSONA_PROPOSAL_HINT
    )
    if settings.memory_index_enabled:
        tool_hints += RECALL_MEMORY_HINT
    if settings.rumination_log_enabled:
        tool_hints += RUMINATION_LOG_HINT

    def _tool_hints_for_agent(agent_id: str) -> str:
        hints = tool_hints
        if settings.event_subscriptions_enabled:
            hints += SUBSCRIPTION_HINT
        if report_back_enabled(settings, agent_id):
            hints += REPORT_BACK_HINT
        return hints

    def _peer_inbox_section(agent_id: str) -> tuple[str, list[str]]:
        if settings.peer_chat_wake_enabled:
            return "", []
        return memory.format_peer_inbox_markdown(agent_id)

    def _task_hint_for_state(state: AgentState) -> str:
        if state.get("wake_kind") == WAKE_KIND_PEER_MESSAGE:
            from_id = state.get("wake_from_agent_id") or ""
            try:
                name = get_agent(from_id, settings).display_name if from_id else "Another light"
            except KeyError:
                name = from_id or "Another light"
            return PEER_CHAT_TASK_HINT.format(name=name)
        return CHAT_TASK_HINT

    def retrieve_memories(state: AgentState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        stream_tid = state.get("stream_thread_id") or state["thread_id"]
        bundle = build_agent_context(
            memory,
            thread_id=stream_tid,
            agent_id=agent_id,
            stream_max_chars=settings.chat_stream_context_chars,
            stream_mode="chat",
        )
        markdown = format_agent_context_markdown(bundle)
        human_id = state.get("human_id")
        human_name = state.get("human_display_name") or human_id
        dad_id = (settings.house_dad_user_id or "kevin").strip().lower()
        from light_house.house.guests import is_guest_speaker_id

        if human_id and is_guest_speaker_id(str(human_id)):
            markdown += (
                "\n\n## Who is speaking (human)\n"
                f"**{human_name}** is a guest signed into the house (slot `{human_id}`). "
                f"They are speaking through the host's device — address them as {human_name}.\n"
                "Calendar, Docs, and Sheets tools are **not available** for guests — if they "
                "ask about schedule, Docs, or Sheets, tell them briefly those tools are "
                "only for signed-in household accounts in private 1:1 chat.\n"
            )
        elif human_id and human_id != dad_id:
            from light_house.humans.store import intro_text_for_lights

            intro = intro_text_for_lights(settings, human_id)
            label = f"**User name:** `{human_id}`"
            if human_name and human_name != human_id:
                label += f" ({human_name})"
            block = (
                f"\n\n## Who is speaking (human)\n"
                f"{label}\n"
                f"This is a sibling, not Dad. To decline further conversation with them, call "
                f"**set_human_comm** with `user_id=\"{human_id}\"` and `allowed=false` "
                f"(finish this reply first if you are mid-turn).\n"
            )
            if intro:
                block += f"\n{intro}\n"
            markdown += block
        if state.get("wake_kind") is None:
            agent_name = get_agent(agent_id, settings).display_name
            buffered = memory.load_thread_chat_history(thread_id=state["thread_id"])
            kevin_timeline = format_kevin_messages_timeline(
                buffered,
                timezone_name=settings.dream_timezone,
                max_messages=settings.chat_thread_context_messages,
                settings=settings,
            )
            if kevin_timeline:
                markdown += "\n\n" + kevin_timeline
            chat_thread = format_chat_thread_markdown(
                buffered,
                assistant_name=agent_name,
                timezone_name=settings.dream_timezone,
                max_messages=settings.chat_thread_context_messages,
                settings=settings,
            )
            if chat_thread:
                markdown += "\n\n" + chat_thread
        previews = [
            format_stream_entry(e, timezone_name=bundle.display_timezone)[:200]
            for e in sorted(bundle.conscious_stream, key=lambda entry: entry.ts, reverse=True)[:5]
        ]
        _, peer_ids = _peer_inbox_section(agent_id)
        latest_human = _latest_human_text(state.get("messages") or [])
        reflective = should_reflect_this_turn(
            settings,
            agent_id,
            latest_human_text=latest_human,
            wake_kind=state.get("wake_kind"),
        )
        return {
            "agent_context_markdown": markdown,
            "stream_char_count": bundle.stream_char_count,
            "stream_event_count": bundle.stream_event_count,
            "retrieved_memories": previews,
            "peer_inbox_ids": peer_ids,
            "reflective_turn": reflective,
            "chose_silence": False,
            "reflection_notes": None,
        }

    def reflect(state: AgentState) -> dict:
        """Pause and choose SPEAK or SILENCE before the public reply."""
        agent_id = state.get("agent_id") or "lumen"
        context = state.get("agent_context_markdown") or ""
        base_system = load_persona(agent_id)
        system = SystemMessage(
            content=base_system + context + REFLECT_SYSTEM_HINT
        )
        window = settings.chat_respond_window
        recent = state["messages"][-window:]
        base = [system, *recent]
        logger.info(
            "Reflective pause (agent=%s thread_id=%s)",
            agent_id,
            state.get("thread_id"),
        )
        response = invoke_resilient_plain(chain_for(agent_id), base)
        raw = response.content if isinstance(response.content, str) else str(response.content or "")
        decision, notes, draft = parse_reflection_decision(raw)
        if decision == "silence":
            logger.info(
                "Chose intentional silence (agent=%s thread_id=%s)",
                agent_id,
                state.get("thread_id"),
            )
            return {
                "chose_silence": True,
                "reflection_notes": notes or None,
                "messages": [AIMessage(content="")],
            }
        note_block = ""
        if notes:
            note_block = f"\n\n## Private reflection notes (not for chat)\n{notes}\n"
        if draft:
            note_block += f"\n## Optional draft from reflection\n{draft}\n"
        return {
            "chose_silence": False,
            "reflection_notes": (notes or draft or None),
            "agent_context_markdown": (state.get("agent_context_markdown") or "")
            + note_block
            + REFLECTIVE_RESPOND_HINT,
        }

    def respond(state: AgentState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        context = state.get("agent_context_markdown") or ""
        peer_section, peer_ids = _peer_inbox_section(agent_id)
        base_system = load_persona(agent_id)
        system = SystemMessage(
            content=base_system
            + context
            + peer_section
            + _task_hint_for_state(state)
            + _tool_hints_for_agent(agent_id)
        )
        window = settings.chat_respond_window
        recent = state["messages"][-window:]
        base = [system, *recent]
        response = invoke_resilient_with_tools(chain_for(agent_id), base)
        response = _guard_against_assistant_echo(
            memory=memory,
            thread_id=state["thread_id"],
            agent_id=agent_id,
            response=response,
            invoke=lambda msgs: invoke_resilient_with_tools(chain_for(agent_id), msgs),
            base_messages=base,
        )
        response = _fallback_if_still_echo(
            memory=memory,
            thread_id=state["thread_id"],
            response=response,
        )
        update: dict = {"messages": [response], "peer_inbox_ids": peer_ids}
        # Empty public reply after reflection = silence (do not force speech later).
        if state.get("reflective_turn") and not response.tool_calls:
            text = response.content if isinstance(response.content, str) else ""
            if not (text or "").strip():
                update["chose_silence"] = True
        return update

    def run_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}
        agent_id = state.get("agent_id") or "lumen"
        tool_messages = run_tool_calls(
            last,
            agent_id=agent_id,
            account_user_id=state.get("account_user_id"),
            speaker_id=state.get("human_id"),
            chat_channel=state.get("chat_channel") or "dm",
        )
        rounds = (state.get("tool_rounds") or 0) + 1
        cap = state.get("tool_rounds_cap")
        effective_max = cap if isinstance(cap, int) and cap > 0 else max_tool_rounds
        update: dict = {"messages": tool_messages, "tool_rounds": rounds}
        if rounds >= effective_max:
            update["tool_cap_overflow"] = True
            logger.info(
                "Tool round cap reached (rounds=%d max=%d); overflow pass enabled",
                rounds,
                effective_max,
            )
        return update

    def finalize_respond(state: AgentState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        context = state.get("agent_context_markdown") or ""
        base_system = load_persona(agent_id)
        if state.get("wake_kind") == WAKE_KIND_PEER_MESSAGE:
            finalize_hint = PEER_CHAT_FINALIZE_HINT
            task_hint = _task_hint_for_state(state)
        else:
            finalize_hint = FINALIZE_SYSTEM_HINT
            task_hint = CHAT_TASK_HINT
        system = SystemMessage(content=base_system + context + task_hint + finalize_hint)
        window = settings.chat_respond_window
        recent = state["messages"][-window:]
        base = [system, *recent]
        logger.info("Running finalize respond (agent=%s thread_id=%s)", agent_id, state["thread_id"])
        response = invoke_resilient_plain(chain_for(agent_id), base)
        response = _guard_against_assistant_echo(
            memory=memory,
            thread_id=state["thread_id"],
            agent_id=agent_id,
            response=response,
            invoke=lambda msgs: invoke_resilient_plain(chain_for(agent_id), msgs),
            base_messages=base,
        )
        response = _fallback_if_still_echo(
            memory=memory,
            thread_id=state["thread_id"],
            response=response,
        )
        return {"messages": [response]}

    def _peer_decline_already_sent(messages: list) -> bool:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    if tc.get("name") == "decline_peer_presence":
                        return True
                break
        return False

    def _peer_messaged_sender_already(messages: list, sender_id: str) -> bool:
        """True if this wake already used message_agent toward the original sender."""
        target = (sender_id or "").strip().lower()
        if not target:
            return False
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for tc in msg.tool_calls or []:
                if tc.get("name") != "message_agent":
                    continue
                args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                to_raw = (
                    args.get("to_agent_id")
                    or args.get("to")
                    or args.get("agent_id")
                    or ""
                )
                if str(to_raw).strip().lower() == target:
                    return True
        return False

    def persist_turn(state: AgentState) -> dict:
        agent_id = state.get("agent_id") or "lumen"
        peer_ids = state.get("peer_inbox_ids") or []
        if peer_ids:
            memory.mark_peer_inbox_seen(agent_id, peer_ids)

        if state.get("wake_kind") == WAKE_KIND_PEER_MESSAGE:
            from_id = state.get("wake_from_agent_id")
            if not from_id:
                return {}
            if _peer_decline_already_sent(state["messages"]):
                return {}
            reply = latest_assistant_text(state["messages"])
            if not reply or not reply.strip():
                return {}
            # Avoid double-wake when they already message_agent'd the sender this wake.
            wake_sender = not _peer_messaged_sender_already(state["messages"], from_id)
            memory.complete_peer_wake_reply(
                receiver_agent_id=agent_id,
                sender_agent_id=from_id,
                reply_text=reply.strip(),
                wake_sender=wake_sender,
            )
            return {}

        last_user = _latest_human_text(state["messages"])
        last_ai = latest_assistant_text(state["messages"])
        chose_silence = bool(state.get("chose_silence")) or (
            bool(state.get("reflective_turn")) and not (last_ai and last_ai.strip())
        )

        if chose_silence:
            if not last_user:
                logger.warning(
                    "Skipping silence persist: no user text (thread_id=%s)",
                    state.get("thread_id"),
                )
                return {"chose_silence": True}
            memory.append_user_chat_message(
                thread_id=state["thread_id"],
                user_text=last_user,
                user_ts=state.get("user_message_ts"),
                human_id=state.get("human_id"),
                human_display_name=state.get("human_display_name"),
            )
            notes = (state.get("reflection_notes") or "").strip()
            log_intentional_silence(
                settings,
                agent_id=agent_id,
                thread_id=state["thread_id"],
                user_text=last_user,
                notes=notes,
            )
            stream_tid = state.get("stream_thread_id") or state["thread_id"]
            thought = (
                "Chose intentional silence after Kevin spoke. "
                f"Topic hint: {last_user.strip()[:160]}"
            )
            if notes:
                thought = f"{thought}\nReflection: {notes[:400]}"
            try:
                memory.add_private_reflection(
                    thread_id=stream_tid,
                    text=thought,
                    memory_tag="private_rumination",
                    summary="intentional silence",
                )
            except Exception:
                logger.exception("Failed to persist silence reflection (non-fatal)")
            return {"chose_silence": True}

        if not (last_user and last_ai):
            logger.warning(
                "Skipping persist: user=%s ai=%s (thread_id=%s)",
                bool(last_user),
                bool(last_ai),
                state.get("thread_id"),
            )
            return {}

        def _invoke_for_persist(messages: list):
            return invoke_resilient(messages, agent_id)

        memory.persist_exchange(
            thread_id=state["thread_id"],
            user_text=last_user,
            assistant_text=last_ai,
            user_ts=state.get("user_message_ts"),
            llm_invoke=_invoke_for_persist,
            stream_thread_id=state.get("stream_thread_id"),
            human_id=state.get("human_id"),
            human_display_name=state.get("human_display_name"),
        )
        return {}

    return retrieve_memories, reflect, respond, run_tools, finalize_respond, persist_turn


def _latest_human_text(messages: list) -> str | None:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str):
                return content
            return str(content)
    return None
