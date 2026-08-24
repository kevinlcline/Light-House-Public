"""Light lookup — backed by data/lights.yaml."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from light_house.config import Settings, get_settings
from light_house.lights.manifest import LightEntry, LightsManifest, load_manifest

LightConfig = LightEntry

_manifest: LightsManifest | None = None


def reload_lights_manifest(settings: Settings | None = None) -> LightsManifest:
    global _manifest
    cfg = settings or get_settings()
    _manifest = load_manifest(cfg)
    return _manifest


def _get_manifest(settings: Settings | None = None) -> LightsManifest:
    global _manifest
    if _manifest is None:
        reload_lights_manifest(settings)
    assert _manifest is not None
    return _manifest


def known_light_ids(settings: Settings | None = None) -> frozenset[str]:
    return frozenset(light.id for light in _get_manifest(settings).lights)


def validate_light_id(light_id: str, settings: Settings | None = None) -> str:
    if light_id not in known_light_ids(settings):
        raise KeyError(f"Unknown light_id: {light_id}")
    return light_id


def get_primary_light_id(settings: Settings | None = None) -> str:
    return _get_manifest(settings).primary_light_id


def get_light(light_id: str, settings: Settings | None = None) -> LightConfig:
    validate_light_id(light_id, settings)
    manifest = _get_manifest(settings)
    return manifest.by_id()[light_id]


def list_lights(settings: Settings | None = None) -> list[LightConfig]:
    return list(_get_manifest(settings).lights)


def list_enabled_lights(settings: Settings | None = None) -> list[LightConfig]:
    return [light for light in list_lights(settings) if light.enabled]


def list_lights_for_broadcast(settings: Settings | None = None) -> list[tuple[str, str]]:
    """(light_id, thread_id) pairs for shared-note and similar multi-light wakes."""
    return [(light.id, light.thread_id) for light in list_enabled_lights(settings)]


def light_id_for_thread(settings: Settings, thread_id: str) -> str:
    for light in list_lights(settings):
        if light.thread_id == thread_id:
            return light.id
    return get_primary_light_id(settings)


def bundled_persona_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "persona"


def personas_data_dir(settings: Settings) -> Path:
    return settings.personas_data_path.resolve()


def resolve_persona_file(settings: Settings, light: LightEntry) -> tuple[Path, str]:
    """Return (path, source) for the resolved persona file."""
    data_path = personas_data_dir(settings) / light.persona_file
    if data_path.is_file():
        return data_path, "data"
    bundled = bundled_persona_dir() / light.persona_file
    if bundled.is_file():
        return bundled, "bundled"
    try:
        text = resources.files("light_house.persona").joinpath(light.persona_file).read_text(
            encoding="utf-8"
        )
        if text.strip():
            return bundled, "package"
    except (TypeError, FileNotFoundError, OSError):
        pass
    raise FileNotFoundError(
        f"{light.persona_file} not found in data/personas, bundled persona, or package resources"
    )


def persona_write_path(settings: Settings, light: LightEntry) -> Path:
    """Path for admin UI writes (always under data/personas/)."""
    name = Path(light.persona_file).name
    if name != light.persona_file or ".." in light.persona_file:
        raise ValueError(f"Invalid persona_file {light.persona_file!r}")
    return personas_data_dir(settings) / name


def resolve_notes_dir(settings: Settings, light_id: str) -> Path:
    light = get_light(light_id, settings)
    return (settings.notes_path.resolve() / light.notes_dir).resolve()


def load_persona(light_id: str, settings: Settings | None = None) -> str:
    cfg = settings or get_settings()
    light = get_light(light_id, cfg)
    path, _source = resolve_persona_file(cfg, light)
    if _source == "package":
        text = resources.files("light_house.persona").joinpath(light.persona_file).read_text(
            encoding="utf-8"
        )
        return text.strip()
    return path.read_text(encoding="utf-8").strip()
