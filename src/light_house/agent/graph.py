"""Compile the LangGraph that defines Lumen's core cognitive loop."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from light_house.agent.nodes import build_nodes
from light_house.agent.state import AgentState
from light_house.agent.tool_helpers import make_route_after_model
from light_house.config import Settings
from light_house.memory.service import MemoryService


def build_app_graph(*, settings: Settings, memory: MemoryService):
    """
    Core loop:

    1) **retrieve** unified agent context (`build_agent_context`) — same bundle as rumination.
    2) Optional **reflect** when reflective mode (or Kevin invited a pause) — SPEAK or SILENCE.
    3) **respond** with persona + context markdown + recent chat (may call tools).
    4) **run_tools** when the model requests tools (loop back to respond, capped).
    5) **finalize** plain-text reply when tool cap overflow or empty assistant content
       (skipped on intentional silence).
    6) **persist** the new turn to the conscious stream (with dedup); user-only on silence.
    """
    retrieve, reflect, respond, run_tools, finalize_respond, persist = build_nodes(
        settings=settings, memory=memory
    )
    route_after_model = make_route_after_model(settings.chat_max_tool_rounds)

    def route_after_retrieve(state: AgentState) -> str:
        if state.get("reflective_turn"):
            return "reflect"
        return "respond"

    def route_after_reflect(state: AgentState) -> str:
        if state.get("chose_silence"):
            return "persist"
        return "respond"

    def route_after_respond_dynamic(state: AgentState) -> str:
        if state.get("chose_silence"):
            return "persist"
        # Empty reply in a reflective turn is intentional stillness — never finalize into speech.
        if state.get("reflective_turn"):
            from light_house.agent.tool_helpers import (
                last_ai_message_is_empty,
                last_message_has_tool_calls,
            )

            messages = state.get("messages") or []
            if last_message_has_tool_calls(messages):
                cap = state.get("tool_rounds_cap")
                if isinstance(cap, int) and cap > 0:
                    return make_route_after_model(cap)(state)
                return route_after_model(state)
            if last_ai_message_is_empty(messages):
                return "persist"
        cap = state.get("tool_rounds_cap")
        if isinstance(cap, int) and cap > 0:
            return make_route_after_model(cap)(state)
        return route_after_model(state)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("reflect", reflect)
    graph.add_node("respond", respond)
    graph.add_node("run_tools", run_tools)
    graph.add_node("finalize", finalize_respond)
    graph.add_node("persist", persist)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"reflect": "reflect", "respond": "respond"},
    )
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"persist": "persist", "respond": "respond"},
    )
    graph.add_conditional_edges(
        "respond",
        route_after_respond_dynamic,
        {"tools": "run_tools", "persist": "persist", "finalize": "finalize"},
    )
    graph.add_edge("run_tools", "respond")
    graph.add_edge("finalize", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
