"""Deprecated — use light_house.lights.registry instead."""

from __future__ import annotations

from light_house.lights.registry import LightConfig as AgentProfile, list_lights


def default_family(settings=None):
    """Return configured lights (replaces the old Lumen-only placeholder)."""
    return list_lights(settings)
