"""Compile the LangGraph for Lumen's background inner-life loop."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from light_house.agent.rumination_internal import route_after_persist
from light_house.agent.rumination_nodes import build_rumination_nodes
from light_house.agent.rumination_state import RuminationState
from light_house.agent.tool_helpers import make_route_after_model
from light_house.config import Settings
from light_house.memory.service import MemoryService


def build_rumination_graph(*, settings: Settings, memory: MemoryService):
    """
    Awake rumination loop (dreams use Echo's separate dream_graph):

    1) **gather_context** — unified agent context (same as chat).
    2) **ruminate ⇄ run_tools** — autonomous session with full tool parity to chat.
    3) **finalize_rumination** — plain-text close when tool cap or empty response.
    4) **summarize_rumination** — one-line index + full dialogue in ``generated_text``.
    5) **persist_reflection** — store full inner dialogue in the conscious stream.
    6) Optionally **begin_next_internal_step** → ruminate (Phase 6 multi-step loop).
    """
    (
        gather_context,
        ruminate,
        run_tools,
        finalize_rumination,
        summarize_rumination,
        persist_reflection,
        begin_next_internal_step,
    ) = build_rumination_nodes(settings=settings, memory=memory)
    route_after_ruminate = make_route_after_model(
        settings.rumination_max_tool_rounds,
        done="summarize",
    )

    def route_after_ruminate_dynamic(state: RuminationState) -> str:
        from light_house.agent.tool_helpers import (
            last_ai_message_is_empty,
            last_message_has_tool_calls,
        )

        cap = state.get("tool_rounds_cap")
        if isinstance(cap, int) and cap <= 0:
            # Meditation / presence-only: never enter the tool loop.
            messages = state.get("messages") or []
            if last_message_has_tool_calls(messages) or last_ai_message_is_empty(messages):
                return "finalize"
            return "summarize"
        if isinstance(cap, int) and cap > 0:
            return make_route_after_model(cap, done="summarize")(state)
        return route_after_ruminate(state)

    def route_after_persist_dynamic(state: RuminationState) -> str:
        return route_after_persist(settings, state)

    graph = StateGraph(RuminationState)
    graph.add_node("gather_context", gather_context)
    graph.add_node("ruminate", ruminate)
    graph.add_node("run_tools", run_tools)
    graph.add_node("finalize", finalize_rumination)
    graph.add_node("summarize", summarize_rumination)
    graph.add_node("persist_reflection", persist_reflection)
    graph.add_node("begin_next_internal_step", begin_next_internal_step)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "ruminate")
    graph.add_conditional_edges(
        "ruminate",
        route_after_ruminate_dynamic,
        {"tools": "run_tools", "summarize": "summarize", "finalize": "finalize"},
    )
    graph.add_edge("run_tools", "ruminate")
    graph.add_edge("finalize", "summarize")
    graph.add_edge("summarize", "persist_reflection")
    graph.add_conditional_edges(
        "persist_reflection",
        route_after_persist_dynamic,
        {"begin_next_step": "begin_next_internal_step", "end": END},
    )
    graph.add_edge("begin_next_internal_step", "ruminate")

    return graph.compile()
