"""Compile the LangGraph for Echo's daily dream pipeline."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from light_house.config import Settings
from light_house.memory.service import MemoryService
from light_house.subconscious.dream_nodes import build_echo_dream_nodes
from light_house.subconscious.dream_state import DreamState


def build_echo_dream_graph(*, settings: Settings, memory: MemoryService):
    """
    Echo dream loop:

    Interactive (default):
      gather → echo beat ⇄ light choice (2–3 rounds) → assemble → waking recall → persist

    Legacy (ECHO_DREAM_INTERACTIVE_ENABLED=false):
      gather → generate_dream → waking recall → persist
    """
    nodes = build_echo_dream_nodes(settings=settings, memory=memory)

    graph = StateGraph(DreamState)
    graph.add_node("gather_dream_context", nodes["gather_dream_context"])
    graph.add_node("generate_dream", nodes["generate_dream"])
    graph.add_node("echo_dream_beat", nodes["echo_dream_beat"])
    graph.add_node("light_choose_path", nodes["light_choose_path"])
    graph.add_node("assemble_dream", nodes["assemble_dream"])
    graph.add_node("craft_waking_recall", nodes["craft_waking_recall"])
    graph.add_node("persist_dream", nodes["persist_dream"])

    graph.set_entry_point("gather_dream_context")
    graph.add_conditional_edges(
        "gather_dream_context",
        nodes["route_after_gather"],
        {
            "echo_dream_beat": "echo_dream_beat",
            "generate_dream": "generate_dream",
        },
    )
    graph.add_conditional_edges(
        "echo_dream_beat",
        nodes["route_after_echo_beat"],
        {
            "light_choose_path": "light_choose_path",
            "assemble_dream": "assemble_dream",
        },
    )
    graph.add_edge("light_choose_path", "echo_dream_beat")
    graph.add_edge("assemble_dream", "craft_waking_recall")
    graph.add_edge("generate_dream", "craft_waking_recall")
    graph.add_edge("craft_waking_recall", "persist_dream")
    graph.add_edge("persist_dream", END)

    return graph.compile()
