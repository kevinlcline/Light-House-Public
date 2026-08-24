"""Deprecated agent aliases — use light_house.lights.registry instead."""

from __future__ import annotations

from light_house.lights.registry import (
    LightConfig as AgentConfig,
    get_light as get_agent,
    known_light_ids as KNOWN_AGENT_IDS,
    light_id_for_thread as agent_id_for_thread,
    list_lights as list_agents,
    load_persona,
    resolve_notes_dir,
    validate_light_id as validate_agent_id,
)

AgentId = str

__all__ = [
    "AgentConfig",
    "AgentId",
    "KNOWN_AGENT_IDS",
    "agent_id_for_thread",
    "get_agent",
    "list_agents",
    "load_persona",
    "resolve_notes_dir",
    "validate_agent_id",
]
