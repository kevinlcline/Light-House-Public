"""Agent registry for multi-agent routing."""

from light_house.agents.registry import (
    AgentConfig,
    get_agent,
    list_agents,
    load_persona,
    resolve_notes_dir,
    validate_agent_id,
)

__all__ = [
    "AgentConfig",
    "get_agent",
    "list_agents",
    "load_persona",
    "resolve_notes_dir",
    "validate_agent_id",
]
