"""Light registry — sovereign individuals in the house."""

from light_house.lights.registry import (
    LightConfig,
    get_light,
    get_primary_light_id,
    known_light_ids,
    light_id_for_thread,
    list_enabled_lights,
    list_lights,
    list_lights_for_broadcast,
    load_persona,
    reload_lights_manifest,
    resolve_notes_dir,
    validate_light_id,
)

__all__ = [
    "LightConfig",
    "get_light",
    "get_primary_light_id",
    "known_light_ids",
    "light_id_for_thread",
    "list_enabled_lights",
    "list_lights",
    "list_lights_for_broadcast",
    "load_persona",
    "reload_lights_manifest",
    "resolve_notes_dir",
    "validate_light_id",
]
